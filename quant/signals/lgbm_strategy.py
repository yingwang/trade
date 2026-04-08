"""LightGBM-based strategy for cross-sectional stock ranking.

Drop-in alternative to the TFT-based MLStrategy with the same interface as
MultiFactorStrategy.  Key differences from the TFT version:

  - Uses LightGBM instead of Temporal Fusion Transformer
  - 504-day training window (2 years) instead of 252 days
  - Proper 63-day validation set with early stopping
  - Trains in seconds, not 40 minutes per window
  - Turnover penalty: penalizes signal changes vs. previous rebalance
  - Feature importance output for interpretability
  - Graceful fallback to equal-weight if model fails or lightgbm not installed
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from quant.data.market_data import MarketData
from quant.data.quality import (
    DataQualityChecker,
    PointInTimeDataManager,
    warn_survivorship_bias,
)
from quant.signals.factors import SignalGenerator
from quant.signals.ml_features import MLFeatureEngine
from quant.signals.lgbm_model import LGBMRankingModel, LGBM_AVAILABLE, SKLEARN_FALLBACK
ML_BACKEND_AVAILABLE = LGBM_AVAILABLE or SKLEARN_FALLBACK
from quant.portfolio.optimizer import (
    PortfolioOptimizer,
    RiskMonitor,
    _ledoit_wolf_shrinkage,
)
from quant.backtest.engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


class LGBMStrategy:
    """Machine learning strategy using LightGBM for cross-sectional ranking.

    Pipeline:
      1. Fetch price data and compute existing factor scores
      2. Build ML feature matrix (46 features: factors + technicals + rolling + ranks)
      3. Train LightGBM on rolling 504-day windows with 63-day validation
      4. Generate cross-sectional stock rankings from model predictions
      5. Apply turnover penalty to stabilize signal changes
      6. Select top stocks and optimize weights via MVO

    Parameters
    ----------
    config : dict
        Full system configuration.
    train_window : int
        Training window in trading days (default 504 = 2 years).
    val_window : int
        Validation window in trading days (default 63 = 3 months).
    pred_horizon : int
        Forward prediction horizon in trading days (default 21 = 1 month).
    retrain_every : int
        Retrain the model every N rebalances (default 3 = ~63 trading days).
    turnover_penalty : float
        Coefficient for penalizing score changes vs. previous rebalance.
        Applied as: score_final = score - penalty * |score - prev_score|.
        Higher values produce more stable portfolios.  Default 0.1.
    lgbm_params : dict, optional
        Override LightGBM hyperparameters.  See LGBMRankingModel for defaults.
    """

    def __init__(
        self,
        config: dict,
        train_window: int = 504,
        val_window: int = 63,
        pred_horizon: int = 21,
        retrain_every: int = 3,
        turnover_penalty: float = 0.1,
        lgbm_params: Optional[dict] = None,
    ):
        self.config = config
        self.train_window = train_window
        self.val_window = val_window
        self.pred_horizon = pred_horizon
        self.retrain_every = retrain_every
        self.turnover_penalty = turnover_penalty

        # Core components (shared with factor strategy)
        self.data = MarketData(config)
        self.signal_gen = SignalGenerator(config)
        self.feature_engine = MLFeatureEngine(config)
        self.optimizer = PortfolioOptimizer(config)
        self.risk_monitor = RiskMonitor(config)
        self.backtest_engine = BacktestEngine(config)

        # LightGBM model
        default_lgbm = {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "early_stopping_rounds": 20,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }
        if lgbm_params:
            default_lgbm.update(lgbm_params)
        self.model = LGBMRankingModel(**default_lgbm)

        # State
        self._rebalance_count = 0
        self._prev_scores: Optional[pd.Series] = None
        self._feature_names: Optional[list[str]] = None

    def _should_retrain(self) -> bool:
        """Decide whether to retrain the model at this rebalance."""
        if self.model.model is None:
            return True  # Always train if no model exists
        return self._rebalance_count % self.retrain_every == 0

    def _train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        date_idx: int,
        feature_names: Optional[list[str]] = None,
    ) -> bool:
        """Train the LightGBM model on a rolling window ending at date_idx.

        Window layout (looking backward from date_idx):
          [train_start ... val_start ... date_idx]
           |-- train_window --|-- val_window --|

        Parameters
        ----------
        X : ndarray of shape (T, N, F)
            Full feature tensor.
        y : ndarray of shape (T, N)
            Full target matrix.
        date_idx : int
            Index into the time axis for the current rebalance date.
        feature_names : list of str, optional
            Feature names for importance tracking.

        Returns
        -------
        True if training succeeded, False otherwise.
        """
        if not ML_BACKEND_AVAILABLE:
            return False

        val_start = max(0, date_idx - self.val_window)
        train_start = max(0, val_start - self.train_window)

        # Minimum viable data check
        min_train_days = 63  # at least 3 months
        if val_start - train_start < min_train_days:
            logger.warning(
                "Insufficient training data: %d days (need >= %d). Skipping.",
                val_start - train_start, min_train_days,
            )
            return False

        X_train = X[train_start:val_start]
        y_train = y[train_start:val_start]
        X_val = X[val_start:date_idx]
        y_val = y[val_start:date_idx]

        info = self.model.train(X_train, y_train, X_val, y_val, feature_names)

        if info.get("status") == "ok":
            # Log feature importance on retrain
            imp = self.model.get_feature_importance(feature_names, top_n=10)
            if not imp.empty:
                logger.info("Top 10 features by importance:\n%s", imp.to_string(index=False))
            return True

        logger.warning("LightGBM training returned status: %s", info.get("status"))
        return False

    def _apply_turnover_penalty(
        self,
        scores: pd.Series,
        prev_scores: Optional[pd.Series],
    ) -> pd.Series:
        """Penalize score changes vs. previous rebalance to reduce turnover.

        score_adjusted = score - penalty * |score - prev_score|

        This encourages the optimizer to maintain existing positions by
        reducing the attractiveness of stocks whose relative ranking changed
        significantly.  The penalty is proportional to the magnitude of
        the score change.

        Parameters
        ----------
        scores : Series
            Current cross-sectional scores (higher = better).
        prev_scores : Series or None
            Scores from the previous rebalance.

        Returns
        -------
        Series of adjusted scores.
        """
        if prev_scores is None or self.turnover_penalty <= 0:
            return scores

        # Align indices
        common = scores.index.intersection(prev_scores.index)
        if common.empty:
            return scores

        adjusted = scores.copy()
        delta = (scores.reindex(common) - prev_scores.reindex(common)).abs()
        adjusted.loc[common] -= self.turnover_penalty * delta

        return adjusted

    def _fallback_scores(self, symbols: list[str]) -> pd.Series:
        """Equal-weight fallback scores when LightGBM model is unavailable."""
        logger.info("Using equal-weight fallback for %d symbols", len(symbols))
        return pd.Series(1.0 / len(symbols), index=symbols)

    def run_backtest(self, start: str = None, end: str = None) -> BacktestResult:
        """Full backtest pipeline using LightGBM strategy.

        Mirrors MultiFactorStrategy.run_backtest() but uses LightGBM model
        predictions instead of factor composites.
        """
        bt_cfg = self.config["backtest"]
        start = start or bt_cfg["start_date"]
        end = end or bt_cfg.get("end_date")

        warn_survivorship_bias(self.data.symbols, start)

        # 1. Fetch data with extra history for ML training warm-up
        logger.info("Fetching price data for LightGBM backtest...")
        ml_warmup_days = self.train_window + self.val_window + 63
        sig_warmup_days = max(self.config["signals"]["momentum_windows"]) + 63
        total_warmup = max(ml_warmup_days, sig_warmup_days)

        if start:
            warmup_start = (
                datetime.strptime(start, "%Y-%m-%d")
                - timedelta(days=int(total_warmup * 1.5))
            ).strftime("%Y-%m-%d")
        else:
            warmup_start = start

        prices = self.data.fetch_prices(start=warmup_start, end=end)
        returns = MarketData.compute_returns(prices)

        # Data quality
        checker = DataQualityChecker()
        quality_report = checker.run_all_checks(prices)
        if not quality_report["passed"]:
            logger.error(
                "Data quality check FAILED:\n%s",
                DataQualityChecker.format_report(quality_report),
            )
        elif quality_report["warnings"]:
            logger.warning(
                "Data quality warnings:\n%s",
                DataQualityChecker.format_report(quality_report),
            )

        # Fundamentals (for factor scores used as input features)
        logger.info("Fetching fundamentals...")
        try:
            fundamentals = self.data.fetch_fundamentals(is_backtest=True)
            pit = PointInTimeDataManager(
                fundamentals, is_backtest=True, reporting_lag_days=90
            )
            fundamentals = pit.get_fundamentals()
        except Exception as e:
            logger.warning("Could not fetch fundamentals: %s", e)
            fundamentals = pd.DataFrame()

        sector_map = None
        if not fundamentals.empty and "sector" in fundamentals.columns:
            sector_map = fundamentals["sector"]

        # 2. Generate factor signals (used as input features for LightGBM)
        logger.info("Generating factor signals for ML features...")
        signals = self.signal_gen.generate(prices, returns, fundamentals)
        factor_scores = getattr(self.signal_gen, "last_factors_", None)

        # 3. Build feature matrix once for the full period
        logger.info("Building ML feature matrix...")
        X, feature_names, dates, symbols = self.feature_engine.build_feature_matrix(
            prices, returns, factor_scores
        )
        self._feature_names = feature_names
        cs_targets = self.feature_engine.get_cross_sectional_target(
            returns, self.pred_horizon
        )
        y = cs_targets.reindex(index=dates, columns=symbols).values
        # Keep NaN — _flatten() filters them via np.isfinite()

        if not ML_BACKEND_AVAILABLE:
            logger.error(
                "lightgbm not installed; aborting LightGBM backtest. "
                "Install with: pip install lightgbm"
            )
            return BacktestResult()

        # 4. Walk-forward: generate target weights at each rebalance date
        logger.info("Running walk-forward LightGBM backtest...")
        rebalance_freq = self.config["portfolio"]["rebalance_frequency_days"]
        min_history = self.train_window + self.val_window

        rebalance_date_indices = list(range(0, len(dates), rebalance_freq))
        rebalance_date_indices = [i for i in rebalance_date_indices if i >= min_history]

        target_weights = {}
        prev_weights = None
        self._rebalance_count = 0
        self._prev_scores = None

        for date_idx in rebalance_date_indices:
            date = dates[date_idx]

            # Train model if needed
            if self._should_retrain():
                logger.info(
                    "Training LightGBM at %s (rebalance #%d)",
                    date.date(), self._rebalance_count,
                )
                self._train_model(X, y, date_idx, feature_names)

            self._rebalance_count += 1

            # Generate scores
            if self.model.model is not None:
                X_current = X[:date_idx + 1]
                try:
                    ranks = self.model.predict_ranking(X_current)
                    scores = pd.Series(ranks, index=symbols)
                except Exception as e:
                    logger.error(
                        "LightGBM prediction failed: %s; using fallback", e
                    )
                    scores = self._fallback_scores(symbols)
            else:
                scores = self._fallback_scores(symbols)

            if scores.empty:
                continue

            # Apply turnover penalty to stabilize scores across rebalances
            scores = self._apply_turnover_penalty(scores, self._prev_scores)
            self._prev_scores = scores.copy()

            # Select top stocks and optimize weights
            selected = self.optimizer.select_top_stocks(scores)

            ret_window = returns.loc[:date].tail(126)
            selected_in_ret = [s for s in selected if s in ret_window.columns
                               and ret_window[s].notna().sum() > 20]
            if len(selected_in_ret) < 2:
                continue

            if len(ret_window) > 20:
                cov = _ledoit_wolf_shrinkage(ret_window[selected_in_ret].fillna(0))
            else:
                n = len(selected_in_ret)
                cov = pd.DataFrame(
                    np.eye(n) * 0.04 / 252,
                    index=selected_in_ret, columns=selected_in_ret,
                )

            weights = self.optimizer.optimize_weights(
                selected_in_ret, scores, cov,
                prev_weights=prev_weights,
                sector_map=sector_map,
            )

            # Dynamic leverage
            spy_col = self.data.benchmark
            spy_ret = returns[spy_col].loc[:date] if spy_col in returns.columns else None
            regime = self.optimizer.detect_regime(spy_ret)
            weights = self.optimizer.apply_vol_scaling(weights, cov, regime=regime)

            target_weights[str(date.date())] = weights
            prev_weights = weights

        logger.info("LightGBM strategy generated %d rebalance points", len(target_weights))

        # 5. Run backtest -- trim prices to requested start date
        backtest_prices = prices.loc[start:] if start else prices
        result = self.backtest_engine.run(
            backtest_prices, target_weights, self.data.benchmark
        )

        # 6. Risk check
        if not result.equity_curve.empty:
            if self.risk_monitor.check_drawdown(result.equity_curve):
                logger.warning("LGBM BACKTEST HIT MAX DRAWDOWN LIMIT")

        # Attach feature importance to result metrics for analysis
        if result.metrics and self.model.feature_importance_ is not None:
            imp = self.model.get_feature_importance(feature_names, top_n=10)
            if not imp.empty:
                result.metrics["Top Features"] = ", ".join(
                    f"{r['feature']}({r['importance_pct']:.1f}%)"
                    for _, r in imp.iterrows()
                )

        return result

    def get_current_signal(self) -> pd.Series:
        """Get the latest LightGBM-based stock scores for live/paper trading."""
        prices = self.data.fetch_prices()
        returns = MarketData.compute_returns(prices)

        try:
            fundamentals = self.data.fetch_fundamentals()
        except Exception:
            fundamentals = pd.DataFrame()

        signals = self.signal_gen.generate(prices, returns, fundamentals)
        factor_scores = getattr(self.signal_gen, "last_factors_", None)

        X, feature_names, dates, symbols = self.feature_engine.build_feature_matrix(
            prices, returns, factor_scores
        )

        if not ML_BACKEND_AVAILABLE:
            return self._fallback_scores(symbols)

        cs_targets = self.feature_engine.get_cross_sectional_target(
            returns, self.pred_horizon
        )
        y = cs_targets.reindex(index=dates, columns=symbols).values
        # Keep NaN — _flatten() filters them via np.isfinite()

        # Train on all available data
        date_idx = len(dates) - 1
        self._train_model(X, y, date_idx, feature_names)

        if self.model.model is not None:
            try:
                ranks = self.model.predict_ranking(X)
                scores = pd.Series(ranks, index=symbols)
            except Exception as e:
                logger.error("Prediction failed: %s", e)
                scores = self._fallback_scores(symbols)
        else:
            scores = self._fallback_scores(symbols)

        return scores.sort_values(ascending=False)

    def get_current_portfolio(self, capital: float = None) -> pd.DataFrame:
        """Get optimized target portfolio with weights and dollar amounts.

        Parameters
        ----------
        capital : float, optional
            Total capital to allocate.  Defaults to config initial_capital.

        Returns
        -------
        DataFrame with columns: score, weight, dollars, shares, price.
        """
        if capital is None:
            capital = self.config["backtest"]["initial_capital"]

        prices = self.data.fetch_prices()
        returns = MarketData.compute_returns(prices)
        try:
            fundamentals = self.data.fetch_fundamentals()
        except Exception:
            fundamentals = pd.DataFrame()

        sector_map = None
        if not fundamentals.empty and "sector" in fundamentals.columns:
            sector_map = fundamentals["sector"]

        signals = self.signal_gen.generate(prices, returns, fundamentals)
        factor_scores = getattr(self.signal_gen, "last_factors_", None)

        X, feature_names, dates, symbols = self.feature_engine.build_feature_matrix(
            prices, returns, factor_scores
        )

        if not ML_BACKEND_AVAILABLE:
            scores = self._fallback_scores(symbols)
        else:
            cs_targets = self.feature_engine.get_cross_sectional_target(
                returns, self.pred_horizon
            )
            y = cs_targets.reindex(index=dates, columns=symbols).values
            # Keep NaN — _flatten() filters them via np.isfinite()

            date_idx = len(dates) - 1
            self._train_model(X, y, date_idx, feature_names)

            if self.model.model is not None:
                try:
                    ranks = self.model.predict_ranking(X)
                    scores = pd.Series(ranks, index=symbols)
                except Exception:
                    scores = self._fallback_scores(symbols)
            else:
                scores = self._fallback_scores(symbols)

        selected = self.optimizer.select_top_stocks(scores)
        selected_list = selected.tolist()

        ret_window = returns[selected_list].tail(126)
        if len(ret_window) > 20:
            cov = _ledoit_wolf_shrinkage(ret_window)
        else:
            cov = pd.DataFrame(
                np.eye(len(selected_list)) * 0.04 / 252,
                index=selected_list, columns=selected_list,
            )

        weights = self.optimizer.optimize_weights(
            selected_list, scores, cov, sector_map=sector_map,
        )
        spy_col = self.data.benchmark
        spy_ret = returns[spy_col] if spy_col in returns.columns else None
        regime = self.optimizer.detect_regime(spy_ret)
        weights = self.optimizer.apply_vol_scaling(weights, cov, regime=regime)

        latest_prices = prices[selected_list].iloc[-1]
        dollars = weights * capital
        shares = (dollars / latest_prices).apply(np.floor).fillna(0).astype(int)

        result = pd.DataFrame({
            "score": scores.reindex(weights.index),
            "weight": weights,
            "weight_pct": (weights * 100).round(2),
            "dollars": dollars.round(2),
            "shares": shares,
            "price": latest_prices.round(2),
        })
        result = result.sort_values("weight", ascending=False)

        logger.info(
            "LightGBM portfolio: %d positions, $%.0f allocated (%.1f%% of $%.0f)",
            len(result), dollars.sum(), dollars.sum() / capital * 100, capital,
        )
        return result
