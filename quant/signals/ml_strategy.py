"""ML-based strategy using the Temporal Fusion Transformer.

Mirrors the interface of MultiFactorStrategy but uses TFT model predictions
instead of hand-crafted factor composites for stock selection and scoring.

Key design decisions:
  - Rolling window: train on 252 days, validate on 63 days, predict 21 days ahead
  - Retrain every N rebalances (default: every 3 = ~63 trading days)
  - Graceful fallback: if model training fails or insufficient data, fall back
    to equal-weight across the universe
  - Uses the same portfolio optimizer and risk monitor as the factor strategy
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
from quant.portfolio.optimizer import (
    PortfolioOptimizer,
    RiskMonitor,
    _ledoit_wolf_shrinkage,
)
from quant.backtest.engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    from quant.signals.tft_model import TFTModelTrainer, TORCH_AVAILABLE
except ImportError:
    TORCH_AVAILABLE = False
    TFTModelTrainer = None


class MLStrategy:
    """Machine learning strategy using Temporal Fusion Transformer.

    Pipeline:
      1. Fetch price data and compute existing factor scores
      2. Build ML feature matrix (factors + technicals + rolling stats + ranks)
      3. Train TFT model on rolling windows
      4. Generate cross-sectional stock rankings from model predictions
      5. Select top stocks and optimize weights via MVO

    Parameters
    ----------
    config : dict
        Full system configuration.
    train_window : int
        Training window in trading days (default 252 = 1 year).
    val_window : int
        Validation window in trading days (default 63 = 3 months).
    pred_horizon : int
        Forward prediction horizon in trading days (default 21 = 1 month).
    retrain_every : int
        Retrain the model every N rebalances (default 3).
    hidden_dim : int
        TFT hidden dimension.
    seq_len : int
        Input sequence length for the TFT model.
    device : str
        PyTorch device ('cpu', 'cuda', 'mps', or 'auto').
    """

    def __init__(
        self,
        config: dict,
        train_window: int = 252,
        val_window: int = 63,
        pred_horizon: int = 21,
        retrain_every: int = 3,
        hidden_dim: int = 64,
        seq_len: int = 63,
        device: str = "auto",
    ):
        self.config = config
        self.train_window = train_window
        self.val_window = val_window
        self.pred_horizon = pred_horizon
        self.retrain_every = retrain_every
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.device = device

        # Core components (shared with factor strategy)
        self.data = MarketData(config)
        self.signal_gen = SignalGenerator(config)
        self.feature_engine = MLFeatureEngine(config)
        self.optimizer = PortfolioOptimizer(config)
        self.risk_monitor = RiskMonitor(config)
        self.backtest_engine = BacktestEngine(config)

        # ML model state
        self.trainer: Optional[TFTModelTrainer] = None
        self._rebalance_count = 0
        self._last_train_info: Optional[dict] = None

    def _ensure_trainer(self, num_features: int) -> bool:
        """Initialize or verify the TFT trainer.

        Returns True if trainer is ready, False if PyTorch is unavailable.
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available; ML strategy will use fallback scoring")
            return False

        if self.trainer is None or self.trainer.num_features != num_features:
            self.trainer = TFTModelTrainer(
                num_features=num_features,
                hidden_dim=self.hidden_dim,
                seq_len=self.seq_len,
                pred_horizon=self.pred_horizon,
                learning_rate=1e-3,
                batch_size=256,
                max_epochs=50,
                patience=5,
                device=self.device,
            )
        return True

    def _should_retrain(self) -> bool:
        """Decide whether to retrain the model at this rebalance."""
        if self.trainer is None or self.trainer.model is None:
            return True  # Always train if no model exists
        return self._rebalance_count % self.retrain_every == 0

    def _train_model(
        self,
        X: np.ndarray,
        targets: np.ndarray,
        train_end_idx: int,
    ) -> bool:
        """Train the TFT model on a rolling window ending at train_end_idx.

        Parameters
        ----------
        X : ndarray of shape (T, N, F)
        targets : ndarray of shape (T, N)
        train_end_idx : int
            Index into the time axis marking the end of training data.

        Returns
        -------
        True if training succeeded, False otherwise.
        """
        val_start = max(0, train_end_idx - self.val_window)
        train_start = max(0, val_start - self.train_window)

        # Minimum viable data check
        min_train_samples = self.seq_len + 20
        if val_start - train_start < min_train_samples:
            logger.warning(
                "Insufficient training data: %d days (need %d). Skipping training.",
                val_start - train_start, min_train_samples,
            )
            return False

        X_train = X[train_start:val_start]
        y_train = targets[train_start:val_start]
        X_val = X[val_start:train_end_idx]
        y_val = targets[val_start:train_end_idx]

        try:
            info = self.trainer.train(X_train, y_train, X_val, y_val)
            self._last_train_info = info
            if info.get("status") == "empty_dataset":
                logger.warning("Training produced empty dataset; model not updated")
                return False
            logger.info("Model training result: %s", {
                k: v for k, v in info.items() if k != "history"
            })
            return True
        except Exception as e:
            logger.error("Model training failed: %s", e, exc_info=True)
            return False

    def _generate_ml_scores(
        self,
        X: np.ndarray,
        symbols: list[str],
    ) -> pd.Series:
        """Generate stock scores from the trained TFT model.

        Returns a Series of scores indexed by symbol (higher = better).
        Falls back to equal scores if prediction fails.
        """
        try:
            ranks = self.trainer.predict_ranking(X)
            scores = pd.Series(ranks, index=symbols)
            # Log feature importance periodically
            if self._rebalance_count % self.retrain_every == 0:
                importance = self.trainer.get_feature_importance(
                    self.feature_engine.feature_names
                )
                if not importance.empty:
                    logger.info("Top 10 features:\n%s", importance.head(10))
            return scores
        except Exception as e:
            logger.error("ML prediction failed: %s; using equal-weight fallback", e)
            return pd.Series(1.0 / len(symbols), index=symbols)

    def _fallback_scores(self, symbols: list[str]) -> pd.Series:
        """Equal-weight fallback scores when ML model is unavailable."""
        logger.info("Using equal-weight fallback for %d symbols", len(symbols))
        return pd.Series(1.0 / len(symbols), index=symbols)

    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        date_idx: int,
        factor_scores: Optional[dict] = None,
    ) -> pd.Series:
        """Generate ML-based stock scores for a single rebalance date.

        Parameters
        ----------
        prices : DataFrame
            Full price history.
        returns : DataFrame
            Full return history.
        date_idx : int
            Index into the time axis for the current rebalance date.
        factor_scores : dict, optional
            Pre-computed factor DataFrames from SignalGenerator.

        Returns
        -------
        Series of stock scores indexed by symbol.
        """
        symbols = [c for c in prices.columns if c != self.data.benchmark]

        # Build feature matrix
        X, feature_names, dates, syms = self.feature_engine.build_feature_matrix(
            prices, returns, factor_scores
        )

        # Ensure trainer is ready
        if not self._ensure_trainer(len(feature_names)):
            return self._fallback_scores(symbols)

        # Build targets for training
        cs_targets = self.feature_engine.get_cross_sectional_target(
            returns, self.pred_horizon
        )
        y = cs_targets.reindex(index=dates, columns=syms).values
        y = np.nan_to_num(y, nan=0.5)  # Default to median rank for missing

        # Train if needed
        if self._should_retrain():
            logger.info("Retraining TFT model at rebalance #%d (date_idx=%d)",
                        self._rebalance_count, date_idx)
            trained = self._train_model(X, y, date_idx)
            if not trained and self.trainer.model is None:
                self._rebalance_count += 1
                return self._fallback_scores(symbols)

        self._rebalance_count += 1

        # Generate scores using data up to current date
        if self.trainer.model is None:
            return self._fallback_scores(symbols)

        X_current = X[:date_idx + 1]
        scores = self._generate_ml_scores(X_current, syms)
        return scores

    def run_backtest(self, start: str = None, end: str = None) -> BacktestResult:
        """Full backtest pipeline using ML strategy.

        Mirrors MultiFactorStrategy.run_backtest() but uses TFT model
        predictions instead of factor composites.
        """
        bt_cfg = self.config["backtest"]
        start = start or bt_cfg["start_date"]
        end = end or bt_cfg.get("end_date")

        warn_survivorship_bias(self.data.symbols, start)

        # 1. Fetch data with extra history for ML training warm-up
        logger.info("Fetching price data for ML backtest...")
        ml_warmup_days = self.train_window + self.val_window + self.seq_len + 63
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
            logger.error("Data quality check FAILED:\n%s",
                         DataQualityChecker.format_report(quality_report))

        # Fundamentals (for factor scores)
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

        # 2. Generate factor signals (used as input features for ML)
        logger.info("Generating factor signals for ML features...")
        signals = self.signal_gen.generate(prices, returns, fundamentals)
        factor_scores = getattr(self.signal_gen, "last_factors_", None)

        # 3. Build feature matrix once for the full period
        logger.info("Building ML feature matrix...")
        X, feature_names, dates, symbols = self.feature_engine.build_feature_matrix(
            prices, returns, factor_scores
        )
        cs_targets = self.feature_engine.get_cross_sectional_target(
            returns, self.pred_horizon
        )
        y = cs_targets.reindex(index=dates, columns=symbols).values
        y = np.nan_to_num(y, nan=0.5)

        # Ensure trainer
        if not self._ensure_trainer(len(feature_names)):
            logger.error("Cannot proceed without PyTorch; aborting ML backtest")
            return BacktestResult()

        # 4. Walk-forward: generate target weights at each rebalance date
        logger.info("Running walk-forward ML backtest...")
        rebalance_freq = self.config["portfolio"]["rebalance_frequency_days"]
        min_history = self.train_window + self.val_window + self.seq_len

        rebalance_date_indices = list(range(0, len(dates), rebalance_freq))
        rebalance_date_indices = [i for i in rebalance_date_indices if i >= min_history]

        target_weights = {}
        prev_weights = None
        self._rebalance_count = 0

        for date_idx in rebalance_date_indices:
            date = dates[date_idx]

            # Train model if needed
            if self._should_retrain():
                logger.info("Retraining at %s (rebalance #%d)",
                            date.date(), self._rebalance_count)
                self._train_model(X, y, date_idx)

            self._rebalance_count += 1

            # Generate scores
            if self.trainer.model is not None:
                X_current = X[:date_idx + 1]
                scores = self._generate_ml_scores(X_current, symbols)
            else:
                scores = self._fallback_scores(symbols)

            if scores.empty:
                continue

            # Select top stocks and optimize weights
            selected = self.optimizer.select_top_stocks(scores)

            ret_window = returns.loc[:date].tail(126)
            selected_in_ret = [s for s in selected if s in ret_window.columns]
            if len(selected_in_ret) < 2:
                continue

            if len(ret_window) > 20:
                cov = _ledoit_wolf_shrinkage(ret_window[selected_in_ret])
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

        logger.info("ML strategy generated %d rebalance points", len(target_weights))

        # 5. Run backtest
        backtest_prices = prices.loc[start:] if start else prices
        result = self.backtest_engine.run(
            backtest_prices, target_weights, self.data.benchmark
        )

        # 6. Risk check
        if not result.equity_curve.empty:
            if self.risk_monitor.check_drawdown(result.equity_curve):
                logger.warning("ML BACKTEST HIT MAX DRAWDOWN LIMIT")

        return result

    def get_current_signal(self) -> pd.Series:
        """Get the latest ML-based stock scores for live/paper trading."""
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

        if not self._ensure_trainer(len(feature_names)):
            return self._fallback_scores(symbols)

        cs_targets = self.feature_engine.get_cross_sectional_target(
            returns, self.pred_horizon
        )
        y = cs_targets.reindex(index=dates, columns=symbols).values
        y = np.nan_to_num(y, nan=0.5)

        # Train on all available data
        date_idx = len(dates) - 1
        self._train_model(X, y, date_idx)

        if self.trainer.model is not None:
            scores = self._generate_ml_scores(X, symbols)
        else:
            scores = self._fallback_scores(symbols)

        return scores.sort_values(ascending=False)
