"""End-to-end runs of ``paper_trade_common.run_main`` against an in-memory broker.

The other live-path tests call one helper at a time.  These drive the whole
entry point the way GitHub Actions does (``python paper_trade.py`` with no
flags): the config is loaded from a YAML file, the broker is constructed
where run_main constructs it, the state file is read and written on disk,
the lock is taken and released, the event log is appended, and the markers
the workflows grep run.log for are read back from stdout.  Three things are
replaced: the Alpaca client (an in-memory broker that fills at the quoted
price and keeps positions and cash between runs, with the real safety
layer), the strategy (a fixed target, so the runner and not the optimizer
is under test) and the clock.
"""

import copy
import functools
import json
import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import paper_trade_common as common
from paper_trade_common import TradeProfile
from quant.execution.alpaca_broker import AlpacaBroker
from quant.execution.safety import PreTradeCheck, SafetyConfig
from quant.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = ("rebalance.yml", "rebalance-lgbm.yml", "rebalance-trend.yml")

# A Monday afternoon, in session.  The anchored 21-business-day schedule
# from 2000-01-03 has 2026-09-23 as the last scheduled date on or before it
# and 2026-10-22 as the next, so a book that rebalanced on 09-23 is quiet
# through Wednesday; the tests compute the dates rather than hard-code them.
FROZEN = datetime(2026, 10, 19, 15, 30)
TODAY = FROZEN.strftime("%Y-%m-%d")
ANCHOR = "2000-01-03"
PRICES = {"AAAA": 100.0, "BBBB": 50.0, "CCCC": 20.0}
TARGET = pd.Series({"AAAA": 0.10, "BBBB": 0.10, "CCCC": 0.10})


@functools.lru_cache(maxsize=1)
def _marker_pattern() -> re.Pattern:
    """The regex the workflows grep run.log with, read from the workflows.

    A marker the Python prints is only useful if that grep matches it, so
    the test matches with the workflow's own pattern; the three books must
    carry the same one.
    """
    patterns = set()
    for name in WORKFLOWS:
        text = (REPO / ".github" / "workflows" / name).read_text()
        found = re.search(r"grep -oE '([^']+)' run\.log", text)
        assert found, f"{name} no longer greps run.log for the stand-down markers"
        patterns.add(found.group(1))
    assert len(patterns) == 1, f"the books grep different marker patterns: {patterns}"
    return re.compile(patterns.pop())


def _markers(stdout: str) -> list[str]:
    return sorted(set(_marker_pattern().findall(stdout)))


class FrozenDatetime(datetime):
    """``datetime`` whose ``now()`` is pinned; the runner imports the class."""

    frozen = FROZEN

    @classmethod
    def now(cls, tz=None):
        return cls.frozen if tz is None else cls.frozen.astimezone(tz)


class FakeBroker:
    """What run_main calls on AlpacaBroker, backed by a dict of positions.

    Fills are immediate at the quoted price.  The safety layer is the real
    PreTradeCheck, so the daily counters the runner persists and restores
    are the real ones and an order the real broker would refuse is refused
    here too.  ``fail_once`` maps a symbol to the status its next order
    comes back with instead of ``filled``; an order of unknown fate changes
    nothing in the book, as at the broker it may or may not have.
    """

    def __init__(self, prices, cash=100_000.0, positions=None, *,
                 market_open=True, history=()):
        self.prices = dict(prices)
        self.cash = float(cash)
        self.positions = {s: float(q) for s, q in (positions or {}).items()}
        self.market_open = market_open
        self.history = list(history)
        self.open_orders_after_sweep = 0
        self.daily_pnl = 0.0
        self.fail_once: dict[str, str] = {}
        self.safety = PreTradeCheck(SafetyConfig())
        self.submitted = []
        self.reconciled = []
        self.constructed_with = []
        self.corporate_action_checks = 0
        self.clock_checks = 0
        self._counter = 0

    # run_main builds the broker itself; this stands in for the class.
    def construct(self, **kwargs):
        self.constructed_with.append(kwargs)
        # A new process starts with fresh counters; the runner restores them.
        self.safety = PreTradeCheck(kwargs.get("safety_config") or SafetyConfig())
        return self

    def is_market_open(self):
        self.clock_checks += 1
        return self.market_open

    def assert_corporate_actions_reconciled(self):
        self.corporate_action_checks += 1

    def get_portfolio_history(self):
        return list(self.history)

    def sweep_open_orders(self):
        return self.open_orders_after_sweep

    def get_daily_pnl(self):
        return self.daily_pnl

    def get_positions(self):
        return pd.Series(self.positions, dtype=float)

    def get_cash(self):
        return self.cash

    def get_portfolio_value(self):
        return self.cash + sum(q * self.prices[s] for s, q in self.positions.items())

    def get_current_prices(self, symbols):
        return {s: self.prices[s] for s in symbols if s in self.prices}

    def reconcile(self, target_weights):
        self.reconciled.append(target_weights.copy())
        return pd.DataFrame()

    def submit_order(self, order, avg_daily_volume=None):
        self.submitted.append(order)
        price = self.prices[order.symbol]
        order.client_order_id = AlpacaBroker._base_client_order_id(order)
        held_value = self.positions.get(order.symbol, 0.0) * price
        ok, reason = self.safety.validate(
            order, price, self.get_portfolio_value(),
            {order.symbol: held_value}, avg_daily_volume,
        )
        if not ok:
            order.status = "rejected"
            order.reject_reason = reason
            self.safety.record_rejection()
            return order
        self.safety.record_submission()
        status = self.fail_once.pop(order.symbol, "filled")
        if status != "filled":
            order.status = status
            return order
        qty = float(order.quantity)
        if order.side == "sell":
            held = self.positions.get(order.symbol, 0.0)
            qty = min(qty, held)
            self.positions[order.symbol] = held - qty
            if self.positions[order.symbol] <= 0:
                del self.positions[order.symbol]
            self.cash += qty * price
        else:
            self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + qty
            self.cash -= qty * price
        self._counter += 1
        order.status = "filled"
        order.filled_quantity = qty
        order.filled_price = price
        order.order_id = f"fake-{self._counter}"
        self.safety.record_fill(qty * price, client_order_id=order.client_order_id)
        return order


class FakeStrategy:
    """A strategy with a fixed target: the runner is what is under test."""

    def __init__(self, config, target=TARGET, prices=PRICES):
        self.config = config
        self.optimizer = SimpleNamespace(stop_loss_pct=float(config["risk"]["stop_loss_pct"]))
        self.data = SimpleNamespace(
            fetch_adv=lambda symbols, window=30: {},
            fetch_recent_splits=lambda symbols, lookback_days=90: {},
        )
        self.target = target.astype(float)
        self.prices = dict(prices)
        self.calls = []

    def get_current_portfolio(self, capital=None, prev_weights=None, **context):
        self.calls.append({"capital": capital, "prev_weights": prev_weights, **context})
        latest = pd.Series(self.prices, dtype=float).reindex(self.target.index)
        dollars = self.target * capital
        return pd.DataFrame({
            "score": 1.0,
            "weight": self.target,
            "weight_pct": (self.target * 100).round(2),
            "dollars": dollars.round(2),
            "shares": (dollars / latest).apply(np.floor).astype(int),
            "price": latest.round(2),
        })


def _write_config(path: Path, config: dict) -> dict:
    """The conftest config as a YAML file run_main can load.

    The keys the validator insists on and the runner reads are added with
    their production meaning: the 40% turnover cap and the anchored
    rebalance calendar.
    """
    cfg = copy.deepcopy(config)
    cfg["portfolio"].update({
        "max_turnover_per_rebalance": 0.40, "alpha_scale": 0.05, "risk_aversion": 1.0,
    })
    cfg["backtest"].update({
        "market_impact_coeff": 0.0, "risk_free_rate": 0.0, "rebalance_anchor_date": ANCHOR,
    })
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return load_config(path)


def _latest_scheduled(now: datetime) -> pd.Timestamp:
    """The last date of the anchored 21-business-day schedule on or before ``now``."""
    return pd.bdate_range(ANCHOR, pd.Timestamp(now).normalize(), freq="21B")[-1]


def _quiet_state(now: datetime = FROZEN, **extra) -> dict:
    """A state whose last rebalance was the latest scheduled date: nothing is due."""
    last = _latest_scheduled(now).to_pydatetime().replace(hour=18)
    return {"last_rebalance": last.isoformat(), "trade_history": [], **extra}


class Harness:
    """One book's files under a temporary directory, and a way to run it."""

    def __init__(self, root: Path, monkeypatch, config: dict):
        self.root = root
        self.monkeypatch = monkeypatch
        self.config_path = root / "config.yaml"
        self.config = _write_config(self.config_path, config)
        self.state_file = root / "logs" / "state.json"
        self.lock_file = root / "logs" / "e2e.lock"
        self.events_log = root / "logs" / "events.jsonl"
        self.strategy = FakeStrategy(self.config)

    def profile(self, **overrides) -> TradeProfile:
        fields = dict(
            name="e2e",
            description="end-to-end paper trading",
            status_banner="E2E STATUS",
            portfolio_banner="E2E TARGET",
            state_file=self.state_file,
            lock_file=self.lock_file,
            log_prefix="e2e",
            strategy_factory=lambda cfg: self.strategy,
            api_key_env="E2E_API_KEY",
            secret_key_env="E2E_SECRET_KEY",
            config_file=str(self.config_path),
            events_log=str(self.events_log),
        )
        fields.update(overrides)
        return TradeProfile(**fields)

    def run(self, broker: FakeBroker, *, now: datetime = FROZEN, argv=(), profile=None):
        """One scheduled attempt: a fresh process at ``now`` with no flags."""
        FrozenDatetime.frozen = now
        self.monkeypatch.setattr(common, "datetime", FrozenDatetime)
        self.monkeypatch.setattr(common, "setup_logging", lambda prefix: None)
        self.monkeypatch.setattr("quant.execution.alpaca_broker.AlpacaBroker", broker.construct)
        self.monkeypatch.setattr(sys, "argv", ["paper_trade.py", *argv])
        common.run_main(profile or self.profile())

    def state(self) -> dict:
        return json.loads(self.state_file.read_text())

    def write_state(self, state: dict) -> dict:
        common.save_state(self.state_file, state)
        return self.state()

    def events(self) -> list[dict]:
        if not self.events_log.exists():
            return []
        return [json.loads(line) for line in self.events_log.read_text().splitlines() if line.strip()]


@pytest.fixture
def harness(tmp_path, monkeypatch, config):
    # run_main creates ./logs where it runs; keep that under tmp_path.
    monkeypatch.chdir(tmp_path)
    return Harness(tmp_path, monkeypatch, config)


def test_due_rebalance_from_an_empty_book(harness, capsys):
    broker = FakeBroker(PRICES, cash=100_000.0)
    harness.monkeypatch.setenv("E2E_API_KEY", "key-from-env")
    harness.monkeypatch.setenv("E2E_SECRET_KEY", "secret-from-env")

    harness.run(broker)

    # The broker was built where run_main builds it, from the profile's
    # environment names and the config's safety section.
    [built] = broker.constructed_with
    assert built["api_key"] == "key-from-env" and built["secret_key"] == "secret-from-env"
    assert built["paper"] is True
    assert built["events_log_path"] == str(harness.events_log)
    assert isinstance(built["safety_config"], SafetyConfig)
    assert broker.split_lookup is harness.strategy.data.fetch_recent_splits
    assert broker.corporate_action_checks == 1

    # The strategy saw the account's real equity and its real (empty) book.
    [call] = harness.strategy.calls
    assert call["capital"] == 100_000.0
    assert isinstance(call["prev_weights"], pd.Series) and call["prev_weights"].empty

    # Three buys, by symbol, sized from the target weights at the quotes,
    # all stamped with this attempt's batch id.
    assert [(o.side, o.symbol, o.quantity) for o in broker.submitted] == [
        ("buy", "AAAA", 100), ("buy", "BBBB", 200), ("buy", "CCCC", 500),
    ]
    assert {o.batch for o in broker.submitted} == {FROZEN.isoformat()}
    assert broker.positions == {"AAAA": 100.0, "BBBB": 200.0, "CCCC": 500.0}
    assert broker.cash == pytest.approx(70_000.0)
    [reconciled] = broker.reconciled
    assert set(reconciled.index) == set(PRICES)

    state = harness.state()
    assert state["schema_version"] == 2
    assert state["last_rebalance"] == FROZEN.isoformat()
    assert "pending_rebalance" not in state
    [batch] = state["trade_history"]
    assert batch["date"] == FROZEN.isoformat()
    assert batch["completed"] is True and batch["outcome"] == "all orders filled"
    assert [(t["symbol"], t["side"], t["qty"], t["requested_qty"], t["status"], t["price"])
            for t in batch["trades"]] == [
        ("AAAA", "buy", 100.0, 100.0, "filled", 100.0),
        ("BBBB", "buy", 200.0, 200.0, "filled", 50.0),
        ("CCCC", "buy", 500.0, 500.0, "filled", 20.0),
    ]
    assert [t["client_order_id"] for t in batch["trades"]] == [o.client_order_id for o in broker.submitted]
    assert all(t["time"] == FROZEN.isoformat() for t in batch["trades"])
    # Entry prices and dates back the stop-loss from here on.
    assert state["entry_prices"] == {"AAAA": 100.0, "BBBB": 50.0, "CCCC": 20.0}
    assert state["entry_dates"] == {s: TODAY for s in PRICES}
    # The drawdown gate's record starts today; the broker had no history to seed from.
    assert state["equity_history"] == {TODAY: 100_000.0}
    assert state["peak_equity"] == 100_000.0
    assert "equity_history_seeded_at" not in state
    # The day's safety counters are saved for a same-day re-run.
    assert state["daily_tracker"]["total_value_traded"] == pytest.approx(30_000.0)
    assert state["daily_tracker"]["orders_filled"] == 3
    assert sorted(state["daily_tracker"]["processed_client_order_ids"]) == sorted(
        o.client_order_id for o in broker.submitted
    )

    assert [e["event"] for e in harness.events()] == ["rebalance_start", "rebalance_complete"]
    assert not harness.lock_file.exists()
    out = capsys.readouterr().out
    assert "TRADES TO EXECUTE (3 orders)" in out
    assert "Results: 3 filled, 0 rejected, 0 not submitted, $30,000 traded" in out
    assert "E2E STATUS" in out
    assert _markers(out) == []


def test_quiet_day_records_an_equity_snapshot_and_nothing_else(harness, capsys):
    before = harness.write_state(_quiet_state())
    broker = FakeBroker(PRICES, cash=90_000.0, positions={"AAAA": 100})

    harness.run(broker)

    assert harness.strategy.calls == []
    assert broker.submitted == []
    assert broker.clock_checks == 1 and broker.corporate_action_checks == 1
    state = harness.state()
    assert state["last_rebalance"] == before["last_rebalance"]
    assert state["trade_history"] == []
    assert state["equity_history"] == {TODAY: 100_000.0}
    assert state["peak_equity"] == 100_000.0
    # Nothing traded, so the daily counters are not written (a quiet run
    # must not produce a noise commit from the workflow).
    assert "daily_tracker" not in state
    assert harness.events() == []
    out = capsys.readouterr().out
    assert "Not time to rebalance yet" in out
    assert "E2E STATUS" in out
    assert _markers(out) == []


def test_closed_market_escalates_only_on_the_fifth_attempt_of_the_day(harness, capsys, caplog):
    before = harness.write_state(_quiet_state(entry_prices={"AAAA": 100.0}))
    # AAAA is 20% under its entry: a stop the runner must not sell into a closed market.
    broker = FakeBroker({**PRICES, "AAAA": 80.0}, cash=92_000.0,
                        positions={"AAAA": 100}, market_open=False)
    caplog.set_level(logging.WARNING, logger="paper_trade_common")

    for attempt in range(1, 6):
        caplog.clear()
        harness.run(broker, now=FROZEN + timedelta(hours=attempt))
        out = capsys.readouterr().out
        state = harness.state()
        assert state["market_closed_attempts"] == {TODAY: attempt}
        assert "E2E STATUS" in out
        if attempt < 5:
            # One early or late arrival only warns: the other slots retry.
            # The workflow greps this text to annotate the run.
            assert _markers(out) == []
            assert "Market is closed" in caplog.text
        else:
            assert _markers(out) == ["FULL_DAY_MARKET_CLOSED"]
            assert "FULL_DAY_MARKET_CLOSED" in caplog.text

    # Nothing but the attempt counter changed: no equity snapshot, no
    # stop-loss sale, no target, no orders, no event.
    assert harness.strategy.calls == []
    assert broker.submitted == []
    assert broker.positions == {"AAAA": 100.0}
    assert {k: v for k, v in state.items() if k != "market_closed_attempts"} == before
    assert harness.events() == []
    assert not harness.lock_file.exists()


def test_stop_loss_sale_on_a_non_rebalance_day(harness, capsys, caplog):
    # AAAA was bought at 100 and quotes 80, a 20% loss against the 12% stop;
    # BBBB was bought at 50 and quotes 50.
    before = harness.write_state(_quiet_state(
        entry_prices={"AAAA": 100.0, "BBBB": 50.0},
        entry_dates={"AAAA": "2026-10-01", "BBBB": "2026-10-01"},
    ))
    broker = FakeBroker({**PRICES, "AAAA": 80.0}, cash=82_000.0,
                        positions={"AAAA": 100, "BBBB": 200})
    caplog.set_level(logging.INFO, logger="paper_trade_common")

    harness.run(broker)

    [order] = broker.submitted
    assert (order.symbol, order.side, order.quantity, order.purpose) == ("AAAA", "sell", 100.0, "stop_loss")
    assert order.signal_price == 80.0
    assert broker.positions == {"BBBB": 200.0}
    assert broker.cash == pytest.approx(90_000.0)
    # No rebalance in the same run as a stop: the next run recomputes from
    # the confirmed book.
    assert harness.strategy.calls == []
    assert "Skipping rebalance in the same run as a stop-loss trigger" in caplog.text
    assert "Stop-loss triggered for AAAA: -20.0% loss (entry=$100.00, now=$80.00)" in caplog.text

    state = harness.state()
    assert state["last_rebalance"] == before["last_rebalance"]
    [batch] = state["trade_history"]
    assert batch["stop_loss"] is True and batch["completed"] is True
    assert batch["date"] == FROZEN.isoformat()
    assert batch["trades"] == [{
        "time": FROZEN.isoformat(),
        "symbol": "AAAA",
        "side": "sell",
        "qty": 100.0,
        "requested_qty": 100.0,
        "status": "filled",
        "price": 80.0,
        "signal_price": 80.0,
        "order_id": "fake-1",
        "client_order_id": order.client_order_id,
        "purpose": "stop_loss",
    }]
    # The stop base is cleared for the sold name and kept for the other.
    assert state["entry_prices"] == {"BBBB": 50.0}
    assert state["stop_loss_attempts"] == {}
    assert state["daily_tracker"]["total_value_traded"] == pytest.approx(8_000.0)
    assert state["equity_history"] == {TODAY: 100_000.0}
    out = capsys.readouterr().out
    assert _markers(out) == []

    # The next session, still unscheduled: the name is gone from the book
    # and from the stop record, so nothing is sold twice.
    next_day = FROZEN + timedelta(days=1)
    assert _latest_scheduled(next_day) == _latest_scheduled(FROZEN)
    harness.run(broker, now=next_day)
    assert len(broker.submitted) == 1
    assert harness.strategy.calls == []
    state = harness.state()
    assert len(state["trade_history"]) == 1
    assert state["last_rebalance"] == before["last_rebalance"]
    assert state["equity_history"] == {TODAY: 100_000.0, next_day.strftime("%Y-%m-%d"): 100_000.0}


def test_same_day_rerun_is_idempotent(harness, caplog):
    broker = FakeBroker(PRICES, cash=100_000.0)
    harness.run(broker)
    first = harness.state()
    assert len(broker.submitted) == 3
    caplog.set_level(logging.INFO, logger="paper_trade_common")

    # A later slot the same day is a fresh process: it restores the day's
    # counters from the state file, finds the rebalance done and sends nothing.
    harness.run(broker, now=FROZEN + timedelta(hours=1))

    assert len(broker.submitted) == 3
    assert len(harness.strategy.calls) == 1
    assert "Restored daily safety counters: $30000 already traded today" in caplog.text
    assert broker.safety.daily.orders_filled == 3
    assert broker.safety.daily.total_value_traded == pytest.approx(30_000.0)
    # The one new key is the stop check's scratch record, created empty the
    # first time it has a position to judge; nothing else moved.
    assert harness.state() == {**first, "stop_loss_attempts": {}}
    assert [e["event"] for e in harness.events()] == ["rebalance_start", "rebalance_complete"]
    assert broker.positions == {"AAAA": 100.0, "BBBB": 200.0, "CCCC": 500.0}


def test_resumed_attempt_keeps_the_batch_id_of_the_attempt_it_continues(harness):
    broker = FakeBroker(PRICES, cash=100_000.0)
    broker.fail_once = {"CCCC": "unknown"}

    harness.run(broker)

    # Two fills and one order of unknown fate: the cycle stays open under
    # the batch id of this attempt, and the two fills are already stop bases.
    state = harness.state()
    assert state["last_rebalance"] is None
    pending = state["pending_rebalance"]
    assert pending["started_at"] == FROZEN.isoformat()
    assert pending["attempts"] == 1 and pending["status"] == "incomplete"
    assert [(t["symbol"], t["status"]) for t in pending["trades"]] == [
        ("AAAA", "filled"), ("BBBB", "filled"), ("CCCC", "unknown"),
    ]
    [first] = state["trade_history"]
    assert first["completed"] is False
    assert first["outcome"] == "1 order(s) uncertain; attempt 1 of 3"
    assert state["entry_prices"] == {"AAAA": 100.0, "BBBB": 50.0}
    first_cccc = broker.submitted[-1]
    assert first_cccc.status == "unknown"

    # The next slot resumes the same attempt: only the missing leg goes out,
    # under the first attempt's batch id and therefore the same idempotency
    # key, so at the real broker it would find its own earlier fill.
    later = FROZEN + timedelta(hours=1)
    harness.run(broker, now=later)

    assert [(o.symbol, o.side, o.quantity) for o in broker.submitted[3:]] == [("CCCC", "buy", 500)]
    resumed = broker.submitted[3]
    assert resumed.batch == FROZEN.isoformat() != later.isoformat()
    assert resumed.client_order_id == first_cccc.client_order_id
    assert resumed.status == "filled"
    state = harness.state()
    assert state["last_rebalance"] == later.isoformat()
    assert "pending_rebalance" not in state
    assert len(state["trade_history"]) == 2
    closed = state["trade_history"][-1]
    assert closed["completed"] is True and closed["outcome"] == "all orders filled"
    assert [(t["symbol"], t["status"]) for t in closed["trades"]] == [
        ("AAAA", "filled"), ("BBBB", "filled"), ("CCCC", "unknown"), ("CCCC", "filled"),
    ]
    assert state["entry_prices"] == {"AAAA": 100.0, "BBBB": 50.0, "CCCC": 20.0}
    assert broker.positions == {"AAAA": 100.0, "BBBB": 200.0, "CCCC": 500.0}

    # And a third slot has nothing left to do.
    harness.run(broker, now=later + timedelta(hours=1))
    assert len(broker.submitted) == 4
    assert harness.state() == state


def test_open_orders_stand_down_marker(harness, capsys):
    harness.write_state(_quiet_state(entry_prices={"AAAA": 100.0}))
    broker = FakeBroker({**PRICES, "AAAA": 80.0}, cash=92_000.0, positions={"AAAA": 100})
    broker.open_orders_after_sweep = 2

    harness.run(broker)

    out = capsys.readouterr().out
    assert _markers(out) == ["STAND_DOWN:open_orders"]
    # Not even the due stop-loss sale is sized on top of working orders.
    assert broker.submitted == []
    assert harness.strategy.calls == []
    state = harness.state()
    assert state["entry_prices"] == {"AAAA": 100.0}
    assert state["trade_history"] == []
    # The drawdown check precedes the sweep, so the day's equity is on record.
    assert state["equity_history"] == {TODAY: 100_000.0}
    assert not harness.lock_file.exists()


def test_drawdown_gate_marker_blocks_new_opens(harness, capsys):
    # The account peaked at 150k before the state file existed: at 100k it
    # is a third down, past the 20% limit, and the peak is seeded from the
    # broker's own history on this first run.
    broker = FakeBroker(PRICES, cash=100_000.0,
                        history=[("2026-06-22", 150_000.0), ("2026-07-01", 140_000.0)])

    harness.run(broker)

    out = capsys.readouterr().out
    assert _markers(out) == ["STAND_DOWN:drawdown_gate"]
    assert "MAX DRAWDOWN BREACHED (-33.3% vs limit 20.0%)" in out
    # A due rebalance from an empty book is all buys: every one is blocked.
    assert len(harness.strategy.calls) == 1
    assert broker.submitted == []
    assert broker.positions == {}
    state = harness.state()
    assert state["equity_history"] == {
        "2026-06-22": 150_000.0, "2026-07-01": 140_000.0, TODAY: 100_000.0,
    }
    assert state["peak_equity"] == 150_000.0
    assert state["equity_history_seeded_at"] == FROZEN.isoformat()
    # With no sell to send the cycle counts as done: the book waits for the
    # next scheduled date rather than re-optimizing daily under the gate.
    assert state["last_rebalance"] == FROZEN.isoformat()
    assert state["trade_history"] == []
    assert [e["event"] for e in harness.events()] == ["drawdown_breach"]


def test_failed_trigger_marker_treats_the_day_as_not_due(harness, capsys, caplog):
    def trigger(strategy, broker, state):
        raise RuntimeError("quote feed down")

    harness.write_state(_quiet_state())
    broker = FakeBroker(PRICES, cash=100_000.0)
    caplog.set_level(logging.ERROR, logger="paper_trade_common")

    harness.run(broker, profile=harness.profile(rebalance_trigger=trigger))

    out = capsys.readouterr().out
    assert _markers(out) == ["STAND_DOWN:trigger_failed"]
    assert "Not time to rebalance yet" in out
    assert "Rebalance trigger failed; treating as not due: quote feed down" in caplog.text
    assert harness.strategy.calls == []
    assert broker.submitted == []
