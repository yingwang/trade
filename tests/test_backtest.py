"""Tests for the backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import BacktestEngine
from quant.backtest.report import monthly_returns_table, risk_report


class TestBacktestEngine:
    def test_risk_free_rate_is_converted_to_daily(self, config):
        config = {
            **config,
            "backtest": {**config["backtest"], "risk_free_rate": 0.0525},
        }
        engine = BacktestEngine(config)
        assert engine.risk_free_rate == pytest.approx((1.0 + 0.0525) ** (1 / 252) - 1)

    def test_no_trades_preserves_capital(self, config, synthetic_prices):
        """With no rebalance targets, portfolio stays in cash."""
        engine = BacktestEngine(config)
        result = engine.run(synthetic_prices, {}, benchmark_col="BENCH")
        # Should be close to initial capital (all cash, no trades)
        assert abs(result.equity_curve.iloc[-1] - config["backtest"]["initial_capital"]) < 1.0

    def test_single_rebalance(self, config, synthetic_prices):
        engine = BacktestEngine(config)
        first_date = synthetic_prices.index[5]
        weights = pd.Series({"AAAA": 0.5, "BBBB": 0.5})
        targets = {str(first_date.date()): weights}

        result = engine.run(synthetic_prices, targets, benchmark_col="BENCH")
        # Should have traded (may also include stop-loss trades)
        rebalance_trades = [t for t in result.trades if t.get("type") != "stop_loss"]
        assert len(rebalance_trades) == 1
        # Portfolio value should differ from initial capital
        assert result.equity_curve.iloc[-1] != config["backtest"]["initial_capital"]

    def test_metrics_computed(self, config, synthetic_prices):
        engine = BacktestEngine(config)
        # Rebalance every 63 days into equal-weight top stocks
        dates = synthetic_prices.index[::63]
        weights = pd.Series({"AAAA": 0.25, "BBBB": 0.25, "CCCC": 0.25, "GGGG": 0.25})
        targets = {str(d.date()): weights for d in dates}

        result = engine.run(synthetic_prices, targets, benchmark_col="BENCH")
        m = result.metrics

        assert "Sharpe Ratio" in m
        assert "Max Drawdown" in m
        assert "CAGR" in m
        assert "Total Return" in m
        assert m["Num Trades"] > 0

    def test_benchmark_curve_handles_leading_nan(self, config, synthetic_prices):
        """A NaN benchmark price on the first day must not wipe out the curve."""
        prices = synthetic_prices.copy()
        prices.iloc[0, prices.columns.get_loc("BENCH")] = np.nan

        engine = BacktestEngine(config)
        result = engine.run(prices, {}, benchmark_col="BENCH")
        assert result.benchmark_curve.notna().any()
        # Normalized to initial capital at the first valid benchmark price
        first_valid = result.benchmark_curve.dropna().iloc[0]
        assert first_valid == pytest.approx(config["backtest"]["initial_capital"])

    def test_transaction_costs_reduce_returns(self, config, synthetic_prices):
        """Higher transaction costs should result in lower final equity."""
        # Low cost
        config_low = {**config, "portfolio": {**config["portfolio"], "transaction_cost_bps": 1}}
        config_low["backtest"] = {**config["backtest"], "slippage_bps": 1}

        # High cost
        config_high = {**config, "portfolio": {**config["portfolio"], "transaction_cost_bps": 100}}
        config_high["backtest"] = {**config["backtest"], "slippage_bps": 50}

        weights = pd.Series({"AAAA": 0.5, "BBBB": 0.5})
        dates = synthetic_prices.index[::21]
        targets = {str(d.date()): weights for d in dates}

        r_low = BacktestEngine(config_low).run(synthetic_prices, targets, "BENCH")
        r_high = BacktestEngine(config_high).run(synthetic_prices, targets, "BENCH")

        assert r_low.equity_curve.iloc[-1] > r_high.equity_curve.iloc[-1]

    def test_market_impact_uses_actual_share_volume(self, config):
        cfg = {
            **config,
            "portfolio": {**config["portfolio"], "transaction_cost_bps": 0},
            "backtest": {
                **config["backtest"],
                "slippage_bps": 0,
                "market_impact_coeff": 10,
            },
        }
        engine = BacktestEngine(cfg)
        quantities = pd.Series({"AAAA": 1_000.0})
        prices = pd.Series({"AAAA": 100.0})

        low_liquidity = engine._cost(
            100_000,
            1_000_000,
            quantities=quantities,
            prices=prices,
            volumes=pd.Series({"AAAA": 10_000.0}),
        )
        high_liquidity = engine._cost(
            100_000,
            1_000_000,
            quantities=quantities,
            prices=prices,
            volumes=pd.Series({"AAAA": 10_000_000.0}),
        )

        assert low_liquidity > high_liquidity

    def test_missing_volume_uses_turnover_fallback(self, config):
        engine = BacktestEngine(config)
        expected = engine._cost(100_000, 1_000_000)
        actual = engine._cost(
            100_000,
            1_000_000,
            quantities=pd.Series({"AAAA": 1_000.0}),
            prices=pd.Series({"AAAA": 100.0}),
            volumes=pd.Series({"AAAA": np.nan}),
        )

        assert actual == pytest.approx(expected)


class TestBacktestReport:
    def test_monthly_returns_table(self, synthetic_prices):
        eq = synthetic_prices["BENCH"]
        table = monthly_returns_table(eq)
        assert "Annual" in table.columns
        assert table.shape[0] >= 2  # at least 2 years

    def test_risk_report(self, synthetic_returns):
        report = risk_report(synthetic_returns["BENCH"])
        assert "VaR 95%" in report
        assert "CVaR 95%" in report
        assert report["VaR 95%"] < 0  # VaR should be negative



class TestTargetProvider:
    def test_provider_sees_actual_drifted_weights(self, config, synthetic_prices):
        """The engine hands the provider the book as it stands at that close,
        not the previous target: after drift the weights differ from the
        target that produced them, and the sum is the invested fraction."""
        engine = BacktestEngine(config)
        dates = synthetic_prices.index
        symbols = [c for c in synthetic_prices.columns if c != "BENCH"]
        first, second = dates[10], dates[40]
        seen = {}

        def provider(date, current_weights):
            seen[date] = current_weights.copy()
            return pd.Series({symbols[0]: 0.5, symbols[1]: 0.5})

        result = engine.run(
            synthetic_prices, {}, benchmark_col="BENCH",
            target_provider=provider, rebalance_dates=[first, second],
        )
        assert set(seen) == {first, second}
        assert seen[first].empty
        held = seen[second]
        # Only the two targeted names can be held (one may have been stopped out
        # by the synthetic path), and what is held has drifted off the 50/50 target.
        assert len(held) >= 1 and set(held.index) <= {symbols[0], symbols[1]}
        assert 0.3 < held.sum() < 1.0
        assert not np.allclose(held.reindex([symbols[0], symbols[1]]).fillna(0.0).values, 0.5)
        assert set(result.targets) == {first, second}
        assert not result.equity_curve.empty


class TestImpactModel:
    def test_impact_scales_with_volatility_and_participation(self, config):
        cfg = {
            **config,
            "portfolio": {**config["portfolio"], "transaction_cost_bps": 0},
            "backtest": {**config["backtest"], "slippage_bps": 0, "market_impact_coeff": 1,
                         "market_impact_sigma_coeff": 1.0},
        }
        engine = BacktestEngine(cfg)
        q = pd.Series({"AAAA": 1_000.0}); p = pd.Series({"AAAA": 100.0})
        calm = engine._cost(100_000, 1e6, quantities=q, prices=p,
                            volumes=pd.Series({"AAAA": 100_000.0}), sigmas=pd.Series({"AAAA": 0.01}))
        wild = engine._cost(100_000, 1e6, quantities=q, prices=p,
                            volumes=pd.Series({"AAAA": 100_000.0}), sigmas=pd.Series({"AAAA": 0.04}))
        assert calm == pytest.approx(100.0)
        assert wild == pytest.approx(400.0)
