"""Tests for execution safety checks, reconciliation, and TWAP splitting."""

import pytest
import pandas as pd

from quant.execution.broker import Order
from quant.execution.safety import (
    DailyTracker,
    ExecutionLogger,
    PostTradeReconciler,
    PreTradeCheck,
    SafetyConfig,
    TWAPSplitter,
)


class TestPreTradeCheck:
    def setup_method(self):
        self.config = SafetyConfig(
            max_single_order_value=50_000,
            max_single_order_shares=5_000,
            max_daily_trade_value=200_000,
            max_daily_loss=10_000,
            max_adv_fraction=0.01,
            min_price=1.0,
            max_position_pct_of_portfolio=0.15,
        )
        self.checker = PreTradeCheck(self.config)

    def test_order_within_limits_passes(self):
        order = Order(symbol="AAPL", side="buy", quantity=100, order_type="market")
        ok, reason = self.checker.validate(order, price=150.0, portfolio_value=1_000_000)
        assert ok
        assert reason == "passed"

    def test_order_exceeds_max_value(self):
        order = Order(symbol="AAPL", side="buy", quantity=500, order_type="market")
        ok, reason = self.checker.validate(order, price=150.0, portfolio_value=1_000_000)
        # 500 * 150 = 75_000 > 50_000
        assert not ok
        assert "exceeds max" in reason

    def test_order_exceeds_max_shares(self):
        order = Order(symbol="PENNY", side="buy", quantity=6000, order_type="market")
        ok, reason = self.checker.validate(order, price=5.0, portfolio_value=1_000_000)
        assert not ok
        assert "shares" in reason

    def test_penny_stock_rejected(self):
        order = Order(symbol="PENNY", side="buy", quantity=100, order_type="market")
        ok, reason = self.checker.validate(order, price=0.50, portfolio_value=1_000_000)
        assert not ok
        assert "below minimum" in reason

    def test_position_concentration_rejected(self):
        order = Order(symbol="AAPL", side="buy", quantity=2000, order_type="market")
        # 2000 * 100 = 200_000, which is 20% of 1M portfolio
        # But first check: 200_000 > max_single_order_value (50k), so it fails there
        # Use smaller position that passes value but fails concentration
        config = SafetyConfig(
            max_single_order_value=200_000,
            max_position_pct_of_portfolio=0.10,
        )
        checker = PreTradeCheck(config)
        order = Order(symbol="AAPL", side="buy", quantity=800, order_type="market")
        ok, reason = checker.validate(order, price=150.0, portfolio_value=1_000_000)
        # 800 * 150 = 120_000 = 12% > 10%
        assert not ok
        assert "portfolio" in reason

    def test_adv_check(self):
        order = Order(symbol="SMALL", side="buy", quantity=200, order_type="market")
        ok, reason = self.checker.validate(
            order, price=10.0, portfolio_value=1_000_000,
            avg_daily_volume=10_000,
        )
        # 200 / 10_000 = 2% > 1%
        assert not ok
        assert "ADV" in reason

    def test_daily_cumulative_limit(self):
        checker = PreTradeCheck(SafetyConfig(
            max_single_order_value=100_000,
            max_daily_trade_value=100_000,
        ))
        # First order: $45k
        order1 = Order(symbol="AAPL", side="buy", quantity=300, order_type="market")
        ok, _ = checker.validate(order1, price=150.0, portfolio_value=1_000_000)
        assert ok
        checker.record_fill(300 * 150.0)

        # Second order: $60k, cumulative would be $105k > $100k
        order2 = Order(symbol="MSFT", side="buy", quantity=200, order_type="market")
        ok, reason = checker.validate(order2, price=300.0, portfolio_value=1_000_000)
        assert not ok
        assert "Daily trade value" in reason

    def test_daily_loss_limit(self):
        checker = PreTradeCheck(SafetyConfig(max_daily_loss=5_000))
        checker.record_fill(50_000, realised_pnl=-3_000)
        ok, _ = checker.check_daily_loss_limit(unrealised_pnl=-3_000)
        # total = -3000 + -3000 = -6000 > -5000
        assert not ok


class TestPostTradeReconciler:
    def test_no_drift(self):
        rec = PostTradeReconciler()
        target = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        actual = pd.Series({"AAPL": 50.0, "MSFT": 50.0})
        prices = {"AAPL": 100.0, "MSFT": 100.0}
        portfolio_value = 10_000.0

        df = rec.reconcile(target, actual, prices, portfolio_value)
        assert all(df["drift_abs"] < 0.01)

    def test_detects_drift(self):
        rec = PostTradeReconciler()
        target = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        actual = pd.Series({"AAPL": 80.0, "MSFT": 20.0})
        prices = {"AAPL": 100.0, "MSFT": 100.0}
        portfolio_value = 10_000.0

        df = rec.reconcile(target, actual, prices, portfolio_value)
        aapl_drift = df.loc["AAPL", "drift"]
        assert aapl_drift > 0.2  # 80% actual vs 50% target

    def test_missing_position(self):
        rec = PostTradeReconciler()
        target = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        actual = pd.Series({"AAPL": 50.0})  # MSFT missing
        prices = {"AAPL": 100.0, "MSFT": 100.0}
        portfolio_value = 10_000.0

        df = rec.reconcile(target, actual, prices, portfolio_value)
        assert "MSFT" in df.index
        assert df.loc["MSFT", "actual_weight"] == 0.0


class TestTWAPSplitter:
    def test_small_order_not_split(self):
        splitter = TWAPSplitter(adv_threshold=0.01)
        order = Order(symbol="AAPL", side="buy", quantity=50, order_type="market")
        assert not splitter.should_split(50, 100_000)

    def test_large_order_split(self):
        splitter = TWAPSplitter(adv_threshold=0.01, n_slices=5)
        order = Order(symbol="AAPL", side="buy", quantity=2000, order_type="market")
        assert splitter.should_split(2000, 100_000)  # 2% > 1%

        slices = splitter.split_order(order, 100_000)
        assert len(slices) == 5
        total_qty = sum(child.quantity for child, _ in slices)
        assert total_qty == 2000

    def test_slice_delays_increase(self):
        splitter = TWAPSplitter(adv_threshold=0.01, n_slices=3, duration_minutes=30)
        order = Order(symbol="AAPL", side="buy", quantity=300, order_type="market")
        slices = splitter.split_order(order, 10_000)
        delays = [d for _, d in slices]
        assert delays[0] == 0
        assert delays[1] > 0
        assert delays[2] > delays[1]


class TestSafetyConfig:
    def test_from_config_defaults(self):
        config = {}
        sc = SafetyConfig.from_config(config)
        assert sc.max_single_order_value == 50_000
        assert sc.require_paper_mode is True

    def test_from_config_override(self):
        config = {
            "safety": {
                "max_single_order_value": 100_000,
                "require_paper_mode": False,
            }
        }
        sc = SafetyConfig.from_config(config)
        assert sc.max_single_order_value == 100_000
        assert sc.require_paper_mode is False
