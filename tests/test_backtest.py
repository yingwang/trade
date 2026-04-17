"""Tests for the backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import BacktestEngine, BacktestResult
from quant.backtest.report import monthly_returns_table, risk_report


class TestBacktestEngine:
    def test_risk_free_rate_is_converted_to_daily(self, config):
        config = {
            **config,
            "backtest": {**config["backtest"], "risk_free_rate": 0.0525},
        }
        engine = BacktestEngine(config)
        assert engine.risk_free_rate == pytest.approx(0.0525 / 252)

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
