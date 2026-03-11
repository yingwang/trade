"""Tests for paper broker and order management."""

import pytest

from quant.execution.broker import PaperBroker, Order, generate_rebalance_orders
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


class TestRebalanceOrders:
    def test_generates_buys_and_sells(self):
        current = pd.Series({"AAAA": 100, "BBBB": 50})
        target = pd.Series({"AAAA": 0.3, "CCCC": 0.7})
        prices = {"AAAA": 50.0, "BBBB": 100.0, "CCCC": 25.0}
        orders = generate_rebalance_orders(current, target, 100_000, prices)

        sides = {o.symbol: o.side for o in orders}
        assert sides["BBBB"] == "sell"  # exit BBBB
        assert sides["CCCC"] == "buy"   # enter CCCC
