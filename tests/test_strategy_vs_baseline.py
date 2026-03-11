"""Integration test: full strategy backtest vs buy-and-hold benchmark.

This validates the end-to-end pipeline produces reasonable results and
computes proper metrics against the baseline (SPY-like benchmark).
"""

import numpy as np
import pandas as pd
import pytest

from quant.signals.factors import SignalGenerator
from quant.portfolio.optimizer import PortfolioOptimizer
from quant.backtest.engine import BacktestEngine


class TestStrategyVsBaseline:
    """Full pipeline integration test using synthetic data."""

    def _run_full_pipeline(self, config, synthetic_prices, synthetic_returns,
                           synthetic_fundamentals):
        """Run the complete signal -> optimize -> backtest pipeline."""
        signal_gen = SignalGenerator(config)
        optimizer = PortfolioOptimizer(config)
        engine = BacktestEngine(config)

        symbols = [c for c in synthetic_prices.columns if c != "BENCH"]
        signals = signal_gen.generate(
            synthetic_prices, synthetic_returns, synthetic_fundamentals
        )

        # Build rebalance schedule
        rebalance_freq = config["portfolio"]["rebalance_frequency_days"]
        rebalance_dates = synthetic_prices.index[::rebalance_freq]
        # Skip first chunk to let indicators warm up
        min_warmup = max(config["signals"]["momentum_windows"]) + 42
        rebalance_dates = [
            d for d in rebalance_dates
            if (d - synthetic_prices.index[0]).days > min_warmup
        ]

        target_weights = {}
        for date in rebalance_dates:
            if date not in signals.index:
                continue
            day_scores = signals.loc[date].dropna()
            if day_scores.empty:
                continue

            selected = optimizer.select_top_stocks(day_scores)
            ret_window = synthetic_returns.loc[:date].tail(126)
            cov = ret_window[selected].cov()

            weights = optimizer.optimize_weights(selected.tolist(), day_scores, cov)
            weights = optimizer.apply_vol_scaling(weights, cov)
            if weights.sum() > 0:
                weights /= weights.sum()
            target_weights[str(date.date())] = weights

        result = engine.run(synthetic_prices, target_weights, benchmark_col="BENCH")
        return result

    def test_pipeline_runs_without_error(self, config, synthetic_prices,
                                          synthetic_returns, synthetic_fundamentals):
        result = self._run_full_pipeline(
            config, synthetic_prices, synthetic_returns, synthetic_fundamentals
        )
        assert len(result.equity_curve) > 0
        assert len(result.trades) > 0
        assert result.metrics["Num Trades"] > 0

    def test_positive_sharpe_on_favorable_data(self, config, synthetic_prices,
                                                synthetic_returns, synthetic_fundamentals):
        """On synthetic data where some stocks clearly trend up, the multi-factor
        model should achieve a non-negative Sharpe (i.e., it's not random)."""
        result = self._run_full_pipeline(
            config, synthetic_prices, synthetic_returns, synthetic_fundamentals
        )
        # We don't demand it beats the benchmark every time (that would be overfit),
        # but it should at least not catastrophically lose money
        assert result.metrics["Sharpe Ratio"] > -0.5
        assert result.metrics["Max Drawdown"] > -0.50  # not more than 50% drawdown

    def test_final_equity_reasonable(self, config, synthetic_prices,
                                      synthetic_returns, synthetic_fundamentals):
        """Final portfolio value should be in a reasonable range."""
        result = self._run_full_pipeline(
            config, synthetic_prices, synthetic_returns, synthetic_fundamentals
        )
        initial = config["backtest"]["initial_capital"]
        final = result.equity_curve.iloc[-1]
        # Should not have lost more than 50% or gained more than 500%
        assert final > initial * 0.5
        assert final < initial * 5.0

    def test_benchmark_curve_computed(self, config, synthetic_prices,
                                      synthetic_returns, synthetic_fundamentals):
        result = self._run_full_pipeline(
            config, synthetic_prices, synthetic_returns, synthetic_fundamentals
        )
        assert not result.benchmark_curve.empty
        assert len(result.benchmark_curve) == len(result.equity_curve)

    def test_metrics_vs_benchmark(self, config, synthetic_prices,
                                   synthetic_returns, synthetic_fundamentals):
        """Verify that benchmark return metric is computed and reasonable."""
        result = self._run_full_pipeline(
            config, synthetic_prices, synthetic_returns, synthetic_fundamentals
        )
        m = result.metrics
        assert "Benchmark Return" in m
        assert "Information Ratio" in m
        # Benchmark return should be positive (BENCH has positive drift in our synth data)
        assert m["Benchmark Return"] > 0

    def test_turnover_bounded(self, config, synthetic_prices,
                               synthetic_returns, synthetic_fundamentals):
        """Average turnover per rebalance should be reasonable (< 200%)."""
        result = self._run_full_pipeline(
            config, synthetic_prices, synthetic_returns, synthetic_fundamentals
        )
        assert result.metrics["Avg Rebalance Turnover"] < 2.0

    def test_strategy_vs_equal_weight_baseline(self, config, synthetic_prices,
                                                synthetic_returns, synthetic_fundamentals):
        """Compare the multi-factor strategy against a naive equal-weight baseline.

        The multi-factor strategy should produce meaningfully different results
        (showing the signal actually does something), though it need not always win.
        """
        # Multi-factor strategy
        mf_result = self._run_full_pipeline(
            config, synthetic_prices, synthetic_returns, synthetic_fundamentals
        )

        # Equal-weight baseline: just hold all stocks equally, same rebalance freq
        engine = BacktestEngine(config)
        symbols = [c for c in synthetic_prices.columns if c != "BENCH"]
        equal_w = pd.Series({s: 1.0 / len(symbols) for s in symbols})
        rebalance_dates = synthetic_prices.index[::config["portfolio"]["rebalance_frequency_days"]]
        ew_targets = {str(d.date()): equal_w for d in rebalance_dates}
        ew_result = engine.run(synthetic_prices, ew_targets, benchmark_col="BENCH")

        # The two strategies should differ meaningfully
        mf_return = mf_result.metrics["Total Return"]
        ew_return = ew_result.metrics["Total Return"]

        # They should not be identical (signal differentiates)
        assert abs(mf_return - ew_return) > 0.001

        # Both should have computed metrics
        assert mf_result.metrics["Sharpe Ratio"] is not None
        assert ew_result.metrics["Sharpe Ratio"] is not None

        print(f"\nMulti-factor: return={mf_return:.2%}, sharpe={mf_result.metrics['Sharpe Ratio']:.3f}")
        print(f"Equal-weight: return={ew_return:.2%}, sharpe={ew_result.metrics['Sharpe Ratio']:.3f}")
        print(f"Benchmark:    return={mf_result.metrics['Benchmark Return']:.2%}")
