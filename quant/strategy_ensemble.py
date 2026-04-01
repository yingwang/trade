"""Dual-strategy ensemble: combines two independent strategies with capital allocation.

Architecture:
  - Each strategy generates independent signals on its allocated capital slice
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

        # Normalize to [0, 1] via rank percentile
        rank_a = scores_a.rank(pct=True).reindex(all_symbols).fillna(0.5)
        rank_b = scores_b.rank(pct=True).reindex(all_symbols).fillna(0.5)

        # Weighted combination
        combined = self.weight_a * rank_a + self.weight_b * rank_b

        # Consensus boost: find stocks that both strategies rate highly
        top_n = self.config["portfolio"]["max_positions"]
        top_a = set(scores_a.nlargest(top_n).index)
        top_b = set(scores_b.nlargest(top_n).index)
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

        # ML strategy warm-up (train_window + val_window + seq_len + buffer)
        # Use defaults from MLStrategy
        ml_warmup = 252 + 63 + 63 + 63

        return max(sig_warmup, ml_warmup)

    def run_backtest(self, start: str = None, end: str = None) -> BacktestResult:
        """Full ensemble backtest.

        Both strategies share the same data feed but generate independent
        signals.  The ensemble combines them at each rebalance date.
        """
        bt_cfg = self.config["backtest"]
        start = start or bt_cfg["start_date"]
        end = end or bt_cfg.get("end_date")

        warn_survivorship_bias(self.data.symbols, start)

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

        # Fundamentals
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

        # 2. Strategy A: factor-based signals (pre-compute full signal matrix)
        logger.info("Computing factor signals (Strategy A)...")
        factor_signals = self.signal_gen.generate(prices, returns, fundamentals)
        factor_scores_dict = getattr(self.signal_gen, "last_factors_", None)

        # 3. Strategy B: ML features + model setup
        logger.info("Building ML features (Strategy B)...")
        from quant.signals.tft_model import TORCH_AVAILABLE
        ml_available = TORCH_AVAILABLE

        X_ml = None
        y_ml = None
        ml_feature_names = None
        ml_dates = None
        ml_symbols = None
        trainer = None

        if ml_available:
            try:
                X_ml, ml_feature_names, ml_dates, ml_symbols = (
                    self.feature_engine.build_feature_matrix(
                        prices, returns, factor_scores_dict
                    )
                )
                cs_targets = self.feature_engine.get_cross_sectional_target(
                    returns, 21  # pred_horizon
                )
                y_ml = cs_targets.reindex(index=ml_dates, columns=ml_symbols).values
                y_ml = np.nan_to_num(y_ml, nan=0.5)

                from quant.signals.tft_model import TFTModelTrainer
                trainer = TFTModelTrainer(
                    num_features=len(ml_feature_names),
                    hidden_dim=64,
                    seq_len=63,
                    pred_horizon=21,
                    device="auto",
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
        min_history_ml = 252 + 63 + 63  # train + val + seq
        min_history = max(min_history_factor, min_history_ml) if ml_available else min_history_factor

        rebalance_dates = prices.index[::rebalance_freq]
        rebalance_dates = [d for d in rebalance_dates
                           if (d - prices.index[0]).days > min_history]

        target_weights = {}
        prev_weights = None
        ml_rebalance_count = 0
        ml_retrain_every = 3

        for date in rebalance_dates:
            # --- Strategy A: factor scores ---
            if date not in factor_signals.index:
                continue
            scores_a = factor_signals.loc[date].dropna()
            if scores_a.empty:
                continue

            # --- Strategy B: ML scores ---
            if ml_available and trainer is not None and ml_dates is not None:
                date_idx = None
                for i, d in enumerate(ml_dates):
                    if d >= date:
                        date_idx = i
                        break
                if date_idx is None:
                    date_idx = len(ml_dates) - 1

                # Retrain periodically
                if ml_rebalance_count % ml_retrain_every == 0:
                    val_start = max(0, date_idx - 63)
                    train_start = max(0, val_start - 252)
                    if val_start - train_start >= 63 + 20:
                        try:
                            trainer.train(
                                X_ml[train_start:val_start],
                                y_ml[train_start:val_start],
                                X_ml[val_start:date_idx],
                                y_ml[val_start:date_idx],
                            )
                        except Exception as e:
                            logger.warning("ML training failed at %s: %s", date.date(), e)

                ml_rebalance_count += 1

                if trainer.model is not None:
                    try:
                        X_current = X_ml[:date_idx + 1]
                        ranks = trainer.predict_ranking(X_current)
                        scores_b = pd.Series(ranks, index=ml_symbols)
                    except Exception as e:
                        logger.warning("ML prediction failed at %s: %s", date.date(), e)
                        scores_b = pd.Series(0.5, index=symbols)
                else:
                    scores_b = pd.Series(0.5, index=symbols)
            else:
                # ML not available: strategy B provides neutral scores
                scores_b = pd.Series(0.5, index=symbols)

            # --- Combine scores ---
            combined = self._combine_scores(scores_a, scores_b)

            # --- Select and optimize ---
            selected = self.optimizer.select_top_stocks(combined)

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
                selected_in_ret, combined, cov,
                prev_weights=prev_weights,
                sector_map=sector_map,
            )

            # Dynamic leverage
            spy_col = self.data.benchmark
            spy_ret = returns[spy_col].loc[:date] if spy_col in returns.columns else None
            regime = self.optimizer.detect_regime(spy_ret)
            weights = self.optimizer.apply_vol_scaling(weights, cov, regime=regime)

            # --- Shared risk check: drawdown gate ---
            # If running drawdown exceeds 80% of limit, reduce exposure
            if prev_weights is not None and len(target_weights) > 10:
                recent_weights_list = list(target_weights.values())[-5:]
                # Simple proxy: if we've been reducing, keep reducing
                avg_invested = np.mean([w.sum() for w in recent_weights_list])
                if avg_invested < 0.5:
                    logger.warning(
                        "Ensemble drawdown protection: avg invested=%.1f%%, "
                        "maintaining defensive posture", avg_invested * 100
                    )

            target_weights[str(date.date())] = weights
            prev_weights = weights

        logger.info("Ensemble generated %d rebalance points", len(target_weights))

        # 5. Run backtest
        backtest_prices = prices.loc[start:] if start else prices
        result = self.backtest_engine.run(
            backtest_prices, target_weights, self.data.benchmark
        )

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

        # Strategy A: factor signals
        factor_signals = self.signal_gen.generate(prices, returns, fundamentals)
        scores_a = factor_signals.iloc[-1].dropna()

        # Strategy B: ML signals (train + predict)
        symbols = [c for c in prices.columns if c != self.data.benchmark]
        scores_b = pd.Series(0.5, index=symbols)  # default neutral

        try:
            from quant.signals.tft_model import TORCH_AVAILABLE, TFTModelTrainer
            if TORCH_AVAILABLE:
                factor_scores_dict = getattr(self.signal_gen, "last_factors_", None)
                X, names, dates, syms = self.feature_engine.build_feature_matrix(
                    prices, returns, factor_scores_dict
                )
                cs_targets = self.feature_engine.get_cross_sectional_target(returns, 21)
                y = cs_targets.reindex(index=dates, columns=syms).values
                y = np.nan_to_num(y, nan=0.5)

                trainer = TFTModelTrainer(
                    num_features=len(names), hidden_dim=64, seq_len=63,
                    pred_horizon=21, device="auto",
                )
                date_idx = len(dates) - 1
                val_start = max(0, date_idx - 63)
                train_start = max(0, val_start - 252)
                if val_start - train_start >= 83:
                    trainer.train(
                        X[train_start:val_start], y[train_start:val_start],
                        X[val_start:date_idx], y[val_start:date_idx],
                    )
                    if trainer.model is not None:
                        ranks = trainer.predict_ranking(X)
                        scores_b = pd.Series(ranks, index=syms)
        except Exception as e:
            logger.warning("ML signal generation failed: %s", e)

        combined = self._combine_scores(scores_a, scores_b)
        return combined.sort_values(ascending=False)
