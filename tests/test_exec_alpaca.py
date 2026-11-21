"""AlpacaBroker boundary: share-class spelling, the clock's close, today's
fills, and the runner's records reaching the split guard.  Offline: the
trading and data clients are in-memory doubles."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from quant.data.corporate_actions import StockSplit, UnresolvedCorporateActionError
from quant.execution.alpaca_broker import (
    AlpacaBroker,
    client_id_intent,
    from_broker_symbol,
    to_broker_symbol,
)
from quant.execution.broker import Order


class TradingClient:
    def __init__(self, positions=(), orders=(), clock=None):
        self.positions = list(positions)
        self.orders = list(orders)
        self.clock = clock
        self.order_requests = []

    def get_all_positions(self):
        return self.positions

    def get_orders(self, request=None, **kwargs):
        self.order_requests.append(request if request is not None else kwargs)
        return self.orders

    def get_clock(self):
        if isinstance(self.clock, Exception):
            raise self.clock
        return self.clock


class DataClient:
    def __init__(self, prices):
        self.prices = prices
        self.requested = []

    def get_stock_latest_trade(self, request):
        self.requested.append(list(request.symbol_or_symbols))
        return {s: SimpleNamespace(price=self.prices[s]) for s in request.symbol_or_symbols if s in self.prices}


def _broker(trading=None, data=None):
    broker = object.__new__(AlpacaBroker)
    broker.trading_client = trading or TradingClient()
    broker.api = broker.trading_client
    broker.data_client = data
    broker.split_lookup = None
    broker._recent_splits_cache = None
    broker.entry_record_lookup = None
    return broker


def _position(symbol, qty, avg, current):
    return SimpleNamespace(symbol=symbol, qty=str(qty), avg_entry_price=str(avg), current_price=str(current))


def test_share_classes_are_spelled_with_a_dot_only_at_the_broker():
    assert to_broker_symbol("BRK-B") == "BRK.B" and from_broker_symbol("BF.B") == "BF-B"
    assert to_broker_symbol("AAPL") == "AAPL" == from_broker_symbol("AAPL")
    trading = TradingClient(positions=[_position("BRK.B", 5, 480.0, 500.0), _position("AAPL", 3, 200.0, 210.0)])
    data = DataClient({"BRK.B": 500.0, "AAPL": 210.0})
    broker = _broker(trading, data)

    # Positions and their details come back in the internal spelling.
    assert broker.get_positions().to_dict() == {"BRK-B": 5.0, "AAPL": 3.0}
    assert [d["symbol"] for d in broker.get_position_details()] == ["BRK-B", "AAPL"]
    # Quotes are asked for, and read back, in Alpaca's spelling.
    assert broker.get_current_prices(["BRK-B", "AAPL"]) == {"BRK-B": 500.0, "AAPL": 210.0}
    assert data.requested == [["BRK.B"], ["AAPL"]]

    # The order request carries Alpaca's symbol; the idempotency key is still
    # the hash of the internal one, so existing keys do not change.
    order = Order(symbol="BRK-B", side="buy", quantity=2, order_type="market", batch="b1")
    order.client_order_id = AlpacaBroker._base_client_order_id(order)
    request = broker._build_order_request(order)
    assert request.symbol == "BRK.B"
    assert request.client_order_id == AlpacaBroker._base_client_order_id(
        Order(symbol="BRK-B", side="buy", quantity=2, order_type="market", batch="b1")
    )


def test_minutes_to_close_reads_the_clock_and_is_none_when_unknown():
    now = datetime(2026, 11, 20, 20, 52, tzinfo=timezone.utc)
    clock = SimpleNamespace(timestamp=now, is_open=True, next_close=now + timedelta(minutes=8))
    assert _broker(TradingClient(clock=clock)).minutes_to_close() == pytest.approx(8.0)
    no_close = SimpleNamespace(timestamp=now, is_open=True, next_close=None)
    assert _broker(TradingClient(clock=no_close)).minutes_to_close() is None
    assert _broker(TradingClient(clock=RuntimeError("clock down"))).minutes_to_close() is None


def test_todays_filled_client_ids_keeps_filled_rebalance_orders_only():
    orders = [
        SimpleNamespace(client_order_id="qts-reba-" + "a" * 24, filled_qty="5"),
        SimpleNamespace(client_order_id="qts-reba-" + "b" * 24, filled_qty="0"),       # cancelled unfilled
        SimpleNamespace(client_order_id="qts-stop-" + "c" * 24, filled_qty="10"),      # a stop, not a rebalance
        SimpleNamespace(client_order_id="manual-1", filled_qty="1"),
    ]
    trading = TradingClient(orders=orders)
    assert _broker(trading).todays_filled_client_ids() == ["qts-reba-" + "a" * 24]
    request = trading.order_requests[0]
    assert str(getattr(request.status, "value", request.status)) == "all"
    assert request.after.hour == 0 and str(request.after.tzinfo) == "America/New_York"
    # Retry and TWAP slice ids name the intent they serve by their first characters.
    base = "qts-reba-" + "a" * 24
    assert client_id_intent(base + "-r1-1234abcd") == client_id_intent(base) == base


def test_runner_records_reach_the_split_guard():
    # 2-for-1 on 2026-10-01; Alpaca adjusted the units; the runner bought at 100.
    split = StockSplit("XYZ", 2.0, datetime(2026, 10, 1).date(), datetime(2026, 10, 1).date())
    trading = TradingClient(positions=[_position("XYZ", 20, 50.0, 40.0)])
    broker = _broker(trading)
    broker.split_lookup = lambda symbols: {"XYZ": split}

    # Without the runner's record the price scale (50/40 = 1.25 < sqrt 2)
    # passes here, and nothing asks for a rescale.
    assert broker.assert_corporate_actions_reconciled() == {}
    records = {"entry_prices": {"XYZ": 100.0}, "entry_dates": {"XYZ": "2026-09-01"}, "split_reconciled": {}}
    broker.entry_record_lookup = lambda: records
    assert broker.assert_corporate_actions_reconciled() == {"XYZ": split}

    # The same record against units the broker never adjusted fails closed.
    stale = _broker(TradingClient(positions=[_position("XYZ", 10, 100.0, 50.0)]))
    stale.split_lookup = lambda symbols: {"XYZ": split}
    stale.entry_record_lookup = lambda: records
    with pytest.raises(UnresolvedCorporateActionError, match="XYZ"):
        stale.assert_corporate_actions_reconciled()
