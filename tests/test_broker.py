"""Tests for paper broker and order management."""

from types import SimpleNamespace
import pytest

from quant.execution.broker import PaperBroker, Order, generate_rebalance_orders
from quant.execution.alpaca_broker import AlpacaBroker
import pandas as pd


class TestPaperBroker:
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
