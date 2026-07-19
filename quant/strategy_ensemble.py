"""Dual-strategy ensemble: combines factor + ML strategies with capital allocation.

Architecture:
  - Strategy A: multi-factor composite (hand-crafted alpha signals)
  - Strategy B: LightGBM cross-sectional ranking model
  - Signals are combined into a unified portfolio via weighted averaging
  - "Consensus boost": stocks selected by BOTH strategies receive extra weight
  - Shared risk limits (total drawdown, total exposure) apply to the combined portfolio

The ensemble is designed to be used with the existing BacktestEngine by producing
unified target_weights at each rebalance date.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from quant.data.market_data import MarketData
from quant.data.point_in_time import load_point_in_time_bundle
from quant.data.quality import (
    DataQualityChecker,
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
from quant.backtest.calendar import fixed_rebalance_dates

logger = logging.getLogger(__name__)


class StrategyEnsemble:
    """Combines two strategies with configurable capital allocation.

    Parameters
    ----------
    config : dict
        Full system configuration.
    strategy_a_weight : float
        Capital fraction for strategy A (default 0.5).
    strategy_b_weight : float
        Capital fraction for strategy B (default 0.5).
    consensus_boost : float
        Extra weight multiplier for stocks selected by both strategies.
        E.g., 1.3 means consensus stocks get 30% more weight.
    max_total_positions : int
        Maximum number of positions in the combined portfolio.
        Defaults to config max_positions * 1.5 (rounded).
    """

    def __init__(
        self,
        config: dict,
        strategy_a_weight: float = 0.5,
        strategy_b_weight: float = 0.5,
        consensus_boost: float = 1.3,
        max_total_positions: Optional[int] = None,
    ):
        if abs(strategy_a_weight + strategy_b_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Strategy weights must sum to 1.0, got "
                f"{strategy_a_weight} + {strategy_b_weight} = "
                f"{strategy_a_weight + strategy_b_weight}"
            )

        self.config = config
        self.weight_a = strategy_a_weight
        self.weight_b = strategy_b_weight
        self.consensus_boost = consensus_boost
        self.max_positions = max_total_positions or int(
            config["portfolio"]["max_positions"] * 1.5
        )

        # Shared components
        self.data = MarketData(config)
        self.signal_gen = SignalGenerator(config)
        self.feature_engine = MLFeatureEngine(config)
        self.optimizer = PortfolioOptimizer(config)
        self.risk_monitor = RiskMonitor(config)
        self.backtest_engine = BacktestEngine(config)
        self.pit_universe, self.delisting_returns = load_point_in_time_bundle(config)

        # Override max_positions for the ensemble's optimizer
        self.optimizer.max_positions = self.max_positions

        logger.info(
            "Ensemble initialized: A=%.0f%%, B=%.0f%%, consensus_boost=%.2f, "
            "max_positions=%d",
            self.weight_a * 100, self.weight_b * 100,
            self.consensus_boost, self.max_positions,
        )

    def _combine_scores(
        self,
        scores_a: pd.Series,
        scores_b: pd.Series,
    ) -> pd.Series:
        """Combine scores from two strategies with consensus boosting.

        Steps:
          1. Normalize each strategy's scores to [0, 1] range
          2. Weight by capital allocation
          3. Apply consensus boost to stocks in both strategies' top picks
          4. Return combined scores

        Parameters
        ----------
        scores_a : Series
            Scores from strategy A (factor-based).
        scores_b : Series
            Scores from strategy B (ML-based).

        Returns
        -------
        Series of combined scores indexed by symbol.
        """
        all_symbols = sorted(set(scores_a.index) | set(scores_b.index))

        usable_a = scores_a.replace([np.inf, -np.inf], np.nan).dropna()
        usable_b = scores_b.replace([np.inf, -np.inf], np.nan).dropna()
        active_a = usable_a.nunique() > 1
        active_b = usable_b.nunique() > 1

        # Normalize to [0, 1] via rank percentile
        rank_a = (
            scores_a.rank(pct=True).reindex(all_symbols).fillna(0.5)
            if active_a
            else pd.Series(0.5, index=all_symbols)
        )
        rank_b = (
            scores_b.rank(pct=True).reindex(all_symbols).fillna(0.5)
            if active_b
            else pd.Series(0.5, index=all_symbols)
        )

        # Weighted combination
        combined = self.weight_a * rank_a + self.weight_b * rank_b

        # Consensus boost: find stocks that both strategies rate highly
        consensus = set()
        if active_a and active_b:
            top_n = self.config["portfolio"]["max_positions"]
            top_a = set(usable_a.nlargest(top_n).index)
            top_b = set(usable_b.nlargest(top_n).index)
            consensus = top_a & top_b

        if consensus:
            logger.info(
                "Consensus stocks (%d): %s",
                len(consensus), sorted(consensus),
            )
            for sym in consensus:
                if sym in combined.index:
                    combined[sym] *= self.consensus_boost

        return combined.sort_values(ascending=False)

    def _get_warmup_days(self) -> int:
        """Calculate total warm-up days needed for both strategies."""
        # Factor strategy warm-up
        sig_warmup = max(self.config["signals"]["momentum_windows"]) + 63

        # LightGBM strategy warm-up (train_window + val_window + buffer)
        ml_warmup = 504 + 63 + 63

        return max(sig_warmup, ml_warmup)

    def run_backtest(self, start: str = None, end: str = None) -> BacktestResult:
        """Full ensemble backtest.

        Both strategies share the same data feed but generate independent
        signals.  The ensemble combines them at each rebalance date.
        """
        bt_cfg = self.config["backtest"]
        start = start or bt_cfg["start_date"]
        end = end or bt_cfg.get("end_date")

        if self.pit_universe is None:
            warn_survivorship_bias(self.data.symbols, start)
        else:
            self.pit_universe.members_as_of(start)

        # 1. Fetch data with warm-up for both strategies
        warmup_days = self._get_warmup_days()
        if start:
            warmup_start = (
                datetime.strptime(start, "%Y-%m-%d")
                - timedelta(days=int(warmup_days * 1.5))
            ).strftime("%Y-%m-%d")
        else:
            warmup_start = start

        logger.info("Fetching price data for ensemble backtest...")
        prices = self.data.fetch_prices(start=warmup_start, end=end)
        returns = MarketData.compute_returns(prices)

        # Data quality
        checker = DataQualityChecker()
        quality_report = checker.run_all_checks(prices)
        if not quality_report["passed"]:
            logger.error("Data quality check FAILED:\n%s",
                         DataQualityChecker.format_report(quality_report))

        fundamentals = pd.DataFrame()
        logger.warning(
            "Backtest fundamentals disabled because no point-in-time feed is "
            "configured"
        )

        sector_map = None
        if not fundamentals.empty and "sector" in fundamentals.columns:
            sector_map = fundamentals["sector"]

        eligibility = None
        if self.pit_universe is not None:
            eligibility = self.pit_universe.eligibility_mask(
                prices.index,
                [column for column in prices.columns if column != self.data.benchmark],
            )
        delisting_events = (
            self.delisting_returns.events
            if self.delisting_returns is not None
            else None
        )

        # 2. Strategy A: factor-based signals (pre-compute full signal matrix)
        logger.info("Computing factor signals (Strategy A)...")
        factor_signals = self.signal_gen.generate(
            prices,
            returns,
            fundamentals,
            eligibility_mask=eligibility,
        )
        factor_scores_dict = getattr(self.signal_gen, "last_factors_", None)

        # 3. Strategy B: LightGBM features + model setup
        logger.info("Building ML features (Strategy B: LightGBM)...")
        from quant.signals.lgbm_model import LGBMRankingModel, LGBM_AVAILABLE
        ml_available = LGBM_AVAILABLE

        X_ml = None
        y_ml = None
        ml_feature_names = None
        ml_dates = None
        ml_symbols = None
        lgbm_model = None

        if ml_available:
            try:
                X_ml, ml_feature_names, ml_dates, ml_symbols = (
                    self.feature_engine.build_feature_matrix(
                        prices,
                        returns,
                        factor_scores_dict,
                        eligibility_mask=eligibility,
                    )
                )
                cs_targets = self.feature_engine.get_cross_sectional_target(
                    returns,
                    21,  # pred_horizon
                    eligibility_mask=eligibility,
                    delisting_returns=delisting_events,
                )
                y_ml = cs_targets.reindex(index=ml_dates, columns=ml_symbols).values
                # Keep NaN — _flatten() filters them via np.isfinite()

                lgbm_model = LGBMRankingModel(
                    num_leaves=31,
                    learning_rate=0.05,
                    n_estimators=200,
                    early_stopping_rounds=20,
                )
            except Exception as e:
                logger.error("ML feature/model setup failed: %s", e, exc_info=True)
                ml_available = False

        # 4. Walk-forward ensemble
        logger.info("Running walk-forward ensemble backtest...")
        rebalance_freq = self.config["portfolio"]["rebalance_frequency_days"]
        symbols = [c for c in prices.columns if c != self.data.benchmark]

        # Determine rebalance dates
        min_history_factor = max(self.config["signals"]["momentum_windows"]) + 42
        min_history_ml = 504 + 63  # train + val
        min_history = max(min_history_factor, min_history_ml) if ml_available else min_history_factor

        not_before = prices.index[0] + pd.Timedelta(days=int(min_history * 1.5))
        rebalance_dates = fixed_rebalance_dates(
            prices.index,
            rebalance_freq,
            anchor=bt_cfg.get("rebalance_anchor_date", "2000-01-03"),
            not_before=not_before,
        )

        target_weights = {}
        prev_weights = None
        ml_rebalance_count = 0
        # Live jobs start in a fresh process and retrain every rebalance; use
        # the same cadence here so research and operations are comparable.
        ml_retrain_every = 1

        for date in rebalance_dates:
            # --- Strategy A: factor scores ---
            if date not in factor_signals.index:
                continue
            scores_a = factor_signals.loc[date].dropna()
            if scores_a.empty:
                continue

            # --- Strategy B: LightGBM scores ---
            if ml_available and lgbm_model is not None and ml_dates is not None:
                date_idx = None
                for i, d in enumerate(ml_dates):
                    if d >= date:
                        date_idx = i
                        break
                if date_idx is None:
                    date_idx = len(ml_dates) - 1

                # Retrain periodically (purged/embargoed split — see lgbm_model)
                if ml_rebalance_count % ml_retrain_every == 0:
                    from quant.signals.lgbm_model import purged_train_val_split
                    split = purged_train_val_split(
                        X_ml, y_ml, date_idx,
                        train_window=504, val_window=63, pred_horizon=21,
                    )
                    if split is not None:
                        try:
                            lgbm_model.train(*split, feature_names=ml_feature_names)
                        except Exception as e:
                            logger.warning("LightGBM training failed at %s: %s",
                                           date.date(), e)

                ml_rebalance_count += 1

                if lgbm_model.model is not None:
                    try:
                        X_current = X_ml[:date_idx + 1]
                        ranks = lgbm_model.predict_ranking(X_current)
                        scores_b = pd.Series(ranks, index=ml_symbols)
                    except Exception as e:
                        logger.warning("LightGBM prediction failed at %s: %s",
                                       date.date(), e)
                        scores_b = pd.Series(0.5, index=symbols)
                else:
                    scores_b = pd.Series(0.5, index=symbols)
            else:
                # LightGBM not available: strategy B provides neutral scores
                scores_b = pd.Series(0.5, index=symbols)

            # --- Combine scores ---
            combined = self._combine_scores(scores_a, scores_b)
            if self.pit_universe is not None:
                members = self.pit_universe.members_as_of(date)
                combined = combined[combined.index.isin(members)]
                if combined.empty:
                    continue

            # --- Select and optimize ---
            selected = self.optimizer.select_top_stocks(combined)

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
                selected_in_ret, combined, cov,
                prev_weights=prev_weights,
                sector_map=sector_map,
            )

            # Dynamic leverage
            spy_col = self.data.benchmark
            spy_ret = returns[spy_col].loc[:date] if spy_col in returns.columns else None
            regime = self.optimizer.detect_regime(spy_ret)
            weights = self.optimizer.apply_vol_scaling(
                weights,
                cov,
                regime=regime,
                sector_map=sector_map,
            )
            regime_cap = min(
                self.optimizer.regime_caps.get(regime, 1.0),
                self.optimizer.max_leverage,
            )
            weights = self.optimizer.enforce_turnover_cap(
                weights,
                prev_weights,
                gross_exposure_cap=regime_cap,
                sector_map=sector_map,
            )

            target_weights[str(date.date())] = weights
            prev_weights = weights

        logger.info("Ensemble generated %d rebalance points", len(target_weights))

        # 5. Run backtest
        backtest_prices = prices.loc[start:] if start else prices
        execution_prices = self.data.last_open_prices_
        if execution_prices is not None and start:
            execution_prices = execution_prices.loc[start:]
        result = self.backtest_engine.run(
            backtest_prices,
            target_weights,
            self.data.benchmark,
            execution_prices=execution_prices,
            delisting_returns=(
                self.delisting_returns.events
                if self.delisting_returns is not None else None
            ),
        )
        result.metrics["Point-in-Time Universe"] = self.pit_universe is not None
        result.metrics["Point-in-Time Fundamentals"] = False
        result.metrics["Survivorship Bias Warning"] = self.pit_universe is None

        # 6. Post-backtest risk check
        if not result.equity_curve.empty:
            if self.risk_monitor.check_drawdown(result.equity_curve):
                logger.warning("ENSEMBLE BACKTEST HIT MAX DRAWDOWN LIMIT")

        # Add ensemble-specific metrics
        if result.metrics:
            result.metrics["Strategy A Weight"] = self.weight_a
            result.metrics["Strategy B Weight"] = self.weight_b
            result.metrics["Consensus Boost"] = self.consensus_boost
            result.metrics["ML Available"] = ml_available

        return result

    def get_current_signal(self) -> pd.Series:
        """Get latest combined signal from both strategies."""
        prices = self.data.fetch_prices()
        returns = MarketData.compute_returns(prices)

        try:
            fundamentals = self.data.fetch_fundamentals()
        except Exception:
            fundamentals = pd.DataFrame()

        eligibility = None
        if self.pit_universe is not None:
            eligibility = self.pit_universe.eligibility_mask(
                prices.index,
                [column for column in prices.columns if column != self.data.benchmark],
            )
        delisting_events = (
            self.delisting_returns.events
            if self.delisting_returns is not None
            else None
        )

        # Strategy A: factor signals
        factor_signals = self.signal_gen.generate(
            prices,
            returns,
            fundamentals,
            eligibility_mask=eligibility,
        )
        scores_a = factor_signals.iloc[-1].dropna()

        # Strategy B: LightGBM signals (train + predict)
        symbols = [c for c in prices.columns if c != self.data.benchmark]
        scores_b = pd.Series(0.5, index=symbols)  # default neutral

        try:
            from quant.signals.lgbm_model import (
                LGBMRankingModel, LGBM_AVAILABLE, purged_train_val_split,
            )
            if LGBM_AVAILABLE:
                factor_scores_dict = getattr(self.signal_gen, "last_factors_", None)
                X, names, dates, syms = self.feature_engine.build_feature_matrix(
                    prices,
                    returns,
                    factor_scores_dict,
                    eligibility_mask=eligibility,
                )
                cs_targets = self.feature_engine.get_cross_sectional_target(
                    returns,
                    21,
                    eligibility_mask=eligibility,
                    delisting_returns=delisting_events,
                )
                y = cs_targets.reindex(index=dates, columns=syms).values
                # Keep NaN — _flatten() filters them via np.isfinite()

                lgbm_model = LGBMRankingModel(
                    num_leaves=31, learning_rate=0.05,
                    n_estimators=200, early_stopping_rounds=20,
                )
                split = purged_train_val_split(
                    X, y, len(dates) - 1,
                    train_window=504, val_window=63, pred_horizon=21,
                )
                if split is not None:
                    lgbm_model.train(*split, feature_names=names)
                    if lgbm_model.model is not None:
                        ranks = lgbm_model.predict_ranking(X)
                        scores_b = pd.Series(ranks, index=syms)
        except Exception as e:
            logger.warning("LightGBM signal generation failed: %s", e)

        combined = self._combine_scores(scores_a, scores_b)
        if self.pit_universe is not None:
            members = self.pit_universe.members_as_of(prices.index[-1])
            combined = combined[combined.index.isin(members)]
        return combined.sort_values(ascending=False)
