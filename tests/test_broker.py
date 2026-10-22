"""Tests for paper broker and order management."""

import logging
from datetime import datetime
from types import SimpleNamespace
import pytest

from quant.execution.broker import PaperBroker, Order, generate_rebalance_orders
from quant.execution.alpaca_broker import NY_TZ, AlpacaBroker
import pandas as pd


class TestPaperBroker:
    @pytest.mark.parametrize("limit_price", [None, float("nan"), 0.0])
    def test_limit_order_requires_valid_limit_price(self, limit_price):
        with pytest.raises(ValueError, match="limit price"):
            Order(
                symbol="AAAA",
                side="buy",
                quantity=10,
                order_type="limit",
                limit_price=limit_price,
            )

    def test_buy_order(self):
        broker = PaperBroker(initial_capital=100_000)
        broker.update_prices({"AAAA": 50.0})
        order = Order(symbol="AAAA", side="buy", quantity=100, order_type="market")
        result = broker.submit_order(order)

        assert result.status == "filled"
        assert broker.get_positions()["AAAA"] == 100
        assert broker.get_cash() < 100_000

    def test_sell_order(self):
        broker = PaperBroker(initial_capital=100_000)
        broker.update_prices({"AAAA": 50.0})
        broker.submit_order(Order(symbol="AAAA", side="buy", quantity=100, order_type="market"))
        broker.submit_order(Order(symbol="AAAA", side="sell", quantity=50, order_type="market"))

        assert broker.get_positions()["AAAA"] == 50

    def test_insufficient_cash_rejected(self):
        broker = PaperBroker(initial_capital=1_000)
        broker.update_prices({"AAAA": 50.0})
        order = Order(symbol="AAAA", side="buy", quantity=1000, order_type="market")
        result = broker.submit_order(order)
        assert result.status == "rejected"

    def test_insufficient_shares_rejected(self):
        broker = PaperBroker(initial_capital=100_000)
        broker.update_prices({"AAAA": 50.0})
        order = Order(symbol="AAAA", side="sell", quantity=100, order_type="market")
        result = broker.submit_order(order)
        assert result.status == "rejected"

    def test_portfolio_value(self):
        broker = PaperBroker(initial_capital=100_000, slippage_bps=0, txn_cost_bps=0)
        broker.update_prices({"AAAA": 50.0})
        broker.submit_order(Order(symbol="AAAA", side="buy", quantity=100, order_type="market"))
        # 100 shares * $50 + remaining cash
        assert broker.get_portfolio_value() == pytest.approx(100_000, abs=1)

    def test_slippage_applied(self):
        broker = PaperBroker(initial_capital=100_000, slippage_bps=100)  # 1%
        broker.update_prices({"AAAA": 100.0})
        order = Order(symbol="AAAA", side="buy", quantity=10, order_type="market")
        result = broker.submit_order(order)
        assert result.filled_price > 100.0  # slippage increases buy price

    def test_limit_buy_not_filled_above_limit(self):
        broker = PaperBroker(initial_capital=100_000, slippage_bps=100)  # 1%
        broker.update_prices({"AAAA": 100.0})
        order = Order(
            symbol="AAAA",
            side="buy",
            quantity=10,
            order_type="limit",
            limit_price=100.50,
        )
        result = broker.submit_order(order)
        assert result.status == "unfilled"
        assert broker.get_positions().empty

    def test_limit_sell_not_filled_below_limit(self):
        broker = PaperBroker(initial_capital=100_000, slippage_bps=100)
        broker.update_prices({"AAAA": 100.0})
        broker.submit_order(Order(symbol="AAAA", side="buy", quantity=10, order_type="market"))
        order = Order(
            symbol="AAAA",
            side="sell",
            quantity=10,
            order_type="limit",
            limit_price=99.50,
        )
        result = broker.submit_order(order)
        assert result.status == "unfilled"
        assert broker.get_positions()["AAAA"] == 10


class TestRebalanceOrders:
    def test_generates_buys_and_sells(self):
        current = pd.Series({"AAAA": 100, "BBBB": 50})
        target = pd.Series({"AAAA": 0.3, "CCCC": 0.7})
        prices = {"AAAA": 50.0, "BBBB": 100.0, "CCCC": 25.0}
        orders = generate_rebalance_orders(current, target, 100_000, prices)

        sides = {o.symbol: o.side for o in orders}
        assert sides["BBBB"] == "sell"  # exit BBBB
        assert sides["CCCC"] == "buy"   # enter CCCC

    # Minimum trade size: max($300, 0.25% of equity) by default.  Share
    # rounding used to turn into one-share trims and $200 adds.

    def test_tiny_trim_is_skipped_and_logged(self, caplog):
        # 100 shares held at $50; the target rounds to 99 -> a $50 trim.
        current = pd.Series({"AAAA": 100})
        target = pd.Series({"AAAA": 0.0495})
        prices = {"AAAA": 50.0}
        with caplog.at_level(logging.INFO, logger="quant.execution.broker"):
            orders = generate_rebalance_orders(current, target, 100_000, prices)
        assert orders == []
        skipped = [r for r in caplog.records if "Skipping SELL AAAA" in r.getMessage()]
        assert len(skipped) == 1 and skipped[0].levelno == logging.INFO
        assert "$50" in skipped[0].getMessage()

    def test_full_exit_of_tiny_position_is_sent(self):
        # Two shares at $50 is a $100 position: below the floor, but a full
        # exit must always go through.
        current = pd.Series({"AAAA": 2})
        target = pd.Series({"BBBB": 0.5})
        prices = {"AAAA": 50.0, "BBBB": 100.0}
        orders = generate_rebalance_orders(current, target, 100_000, prices)
        by_symbol = {o.symbol: o for o in orders}
        assert by_symbol["AAAA"].side == "sell"
        assert by_symbol["AAAA"].quantity == 2
        assert by_symbol["BBBB"].side == "buy"

    def test_order_above_floor_is_sent(self):
        # 100 -> 108 shares at $50 is a $400 add, above the $300 floor.
        current = pd.Series({"AAAA": 100})
        target = pd.Series({"AAAA": 0.054})
        prices = {"AAAA": 50.0}
        orders = generate_rebalance_orders(current, target, 100_000, prices)
        assert [(o.side, o.quantity) for o in orders] == [("buy", 8)]

    def test_floor_scales_with_equity(self):
        # The same $400 trim (8 shares at $50) clears the $300 floor on a
        # $100k book but not the 0.25% = $2,500 floor on a $1M book.
        current = pd.Series({"AAAA": 1000})
        prices = {"AAAA": 50.0}
        small = generate_rebalance_orders(
            current, pd.Series({"AAAA": 0.496}), 100_000, prices
        )
        big = generate_rebalance_orders(
            current, pd.Series({"AAAA": 0.0496}), 1_000_000, prices
        )
        assert [(o.side, o.quantity) for o in small] == [("sell", 8)]
        assert big == []

    def test_floor_is_configurable_and_zero_disables_it(self):
        current = pd.Series({"AAAA": 100})
        target = pd.Series({"AAAA": 0.0495})  # $50 trim
        prices = {"AAAA": 50.0}
        off = generate_rebalance_orders(
            current, target, 100_000, prices,
            min_trade_notional_usd=0, min_trade_fraction=0,
        )
        assert [(o.side, o.quantity) for o in off] == [("sell", 1)]
        raised = generate_rebalance_orders(
            current, pd.Series({"AAAA": 0.054}), 100_000, prices,  # $400 add
            min_trade_notional_usd=500, min_trade_fraction=0,
        )
        assert raised == []


class TestAlpacaBroker:
    def _make_broker(self):
        broker = object.__new__(AlpacaBroker)

        class FakeAPI:
            def __init__(self):
                self.cancelled = []

            def submit_order(self, **kwargs):
                return SimpleNamespace(id="abc")

            def cancel_order(self, order_id):
                self.cancelled.append(order_id)

        class DummySafety:
            def record_submission(self):
                pass

            def record_rejection(self):
                pass

            def record_fill(self, *args, **kwargs):
                pass

        class DummyLog:
            def log_order_submitted(self, *args, **kwargs):
                pass

            def log_order_rejected(self, *args, **kwargs):
                pass

            def log_order_filled(self, *args, **kwargs):
                pass

        broker.api = FakeAPI()
        broker.safety = DummySafety()
        broker.exec_log = DummyLog()
        return broker

    def test_partial_fill_cancels_remaining_order(self):
        broker = self._make_broker()
        broker._wait_for_fill = lambda order_id, timeout=30: SimpleNamespace(
            id="abc",
            filled_qty="40",
            filled_avg_price="10",
            status="partially_filled",
        )

        order = Order(symbol="AAPL", side="buy", quantity=100, order_type="market")
        result = broker._execute_single(order, signal_price=10.0)

        assert result.status == "partial_fill"
        assert result.quantity == 40.0
        assert broker.api.cancelled == ["abc"]

    def test_portfolio_history_drops_nulls_and_uses_eastern_dates(self):
        broker = self._make_broker()

        class HistoryAPI:
            def __init__(self):
                self.requests = []

            def get_portfolio_history(self, request):
                self.requests.append(request)
                # The bar at 22:00 Eastern is 02:00 UTC the next day and must
                # still be dated the Eastern day; the null bar is dropped.
                stamps = [
                    int(datetime(2026, 6, 22, 0, 0, tzinfo=NY_TZ).timestamp()),
                    int(datetime(2026, 6, 23, 22, 0, tzinfo=NY_TZ).timestamp()),
                    int(datetime(2026, 6, 24, 0, 0, tzinfo=NY_TZ).timestamp()),
                ]
                return SimpleNamespace(
                    timestamp=stamps,
                    equity=[130_804.0, 129_000.25, None],
                    profit_loss=[0.0, 0.0, None],
                )

        broker.api = HistoryAPI()
        points = broker.get_portfolio_history()
        assert points == [("2026-06-22", 130_804.0), ("2026-06-23", 129_000.25)]
        request = broker.api.requests[0]
        assert request.period == "1A" and request.timeframe == "1D"

    def test_fill_during_cancel_uses_final_broker_quantity(self):
        broker = self._make_broker()

        class RaceAPI:
            def __init__(self):
                self.current = SimpleNamespace(
                    id="abc",
                    filled_qty="40",
                    filled_avg_price="10",
                    status="partially_filled",
                )

            def submit_order(self, **kwargs):
                return SimpleNamespace(id="abc")

            def cancel_order(self, order_id):
                # The remaining shares fill while Alpaca processes the cancel.
                self.current = SimpleNamespace(
                    id=order_id,
                    filled_qty="100",
                    filled_avg_price="10.05",
                    status="filled",
                )

            def get_order(self, order_id):
                return self.current

        broker.api = RaceAPI()
        broker._monotonic = lambda: 0.0
        broker._sleep = lambda _: None
        broker._wait_for_fill = lambda order_id, timeout=30: SimpleNamespace(
            id="abc",
            filled_qty="40",
            filled_avg_price="10",
            status="partially_filled",
        )

        order = Order(symbol="AAPL", side="buy", quantity=100, order_type="market")
        result = broker._execute_single(order, signal_price=10.0)

        assert result.status == "filled"
        assert result.filled_quantity == 100.0
        assert result.quantity == 100.0
        assert result.filled_price == pytest.approx(10.05)
