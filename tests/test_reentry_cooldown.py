"""Re-entry cooldown after a fixed stop, for the two factor books.

A name sold on its fixed stop is not bought back for
``risk.reentry_cooldown_sessions`` sessions: the backtest reads the engine's
record of executed stops, the live path the runner's ``state["stop_exits"]``,
and both go through ``active_cooldowns`` so they count sessions alike.
"""

import copy

import numpy as np
import pandas as pd

import paper_trade_common as common
from quant.backtest.calendar import active_cooldowns
from quant.strategy import MultiFactorStrategy


def _sessions(n=30):
    return pd.bdate_range("2024-01-01", periods=n)


def test_active_cooldowns_counts_sessions_from_the_stop():
    idx = _sessions()
    stopped = {"AAAA": idx[10]}
    # Blocked on the stop session and the five after it, free on the sixth.
    for k in range(10, 16):
        assert "AAAA" in active_cooldowns(idx, idx[k], stopped, 5)
    assert active_cooldowns(idx, idx[16], stopped, 5) == {}
    # Before the stop session it is not (yet) blocked by that record either way.
    assert "AAAA" in active_cooldowns(idx, idx[9], stopped, 5)
    # A record dated after the last bar (sold during a session whose bar is
    # not in yet) counts as the last session.
    assert "AAAA" in active_cooldowns(idx[:10], idx[9], {"AAAA": idx[12]}, 5)
    # Off when the cooldown is zero or there is nothing recorded.
    assert active_cooldowns(idx, idx[11], stopped, 0) == {}
    assert active_cooldowns(idx, idx[11], {}, 5) == {}
    # String dates (as the state file keeps them) work too.
    assert "AAAA" in active_cooldowns(idx, idx[12], {"AAAA": str(idx[10].date())}, 5)


def _crash_prices(synthetic_prices, symbol, day, drop=0.30):
    """The conftest frame with ``symbol`` falling ``drop`` at ``day`` and staying there."""
    prices = synthetic_prices.copy()
    prices.loc[prices.index >= day, symbol] *= (1.0 - drop)
    return prices


def _cooldown_config(config, sessions):
    cfg = copy.deepcopy(config)
    cfg["portfolio"]["rebalance_frequency_days"] = 5
    cfg["risk"]["stop_loss_pct"] = 0.12
    cfg["risk"]["reentry_cooldown_sessions"] = sessions
    cfg["portfolio"]["max_turnover_per_rebalance"] = 1.0
    return cfg


def _run(cfg, prices, monkeypatch):
    strategy = MultiFactorStrategy(cfg)
    monkeypatch.setattr(strategy.data, "fetch_prices", lambda start=None, end=None: prices)
    strategy.data.last_open_prices_ = None
    strategy.data.last_volumes_ = None
    result = strategy.run_backtest(start=str(prices.index[len(prices) // 2].date()))
    return strategy, result


def test_backtest_does_not_rebuy_a_stopped_name_inside_the_cooldown(config, synthetic_prices, monkeypatch):
    # Find a name the book holds, then crash it so the engine's stop sells it.
    probe_cfg = _cooldown_config(config, 0)
    _, probe = _run(probe_cfg, synthetic_prices, monkeypatch)
    first_targets = [t for _, t in sorted(probe.targets.items()) if len(t)]
    held_name = first_targets[0].sort_values(ascending=False).index[0]
    crash_day = sorted(probe.targets)[2] + pd.Timedelta(days=1)
    prices = _crash_prices(synthetic_prices, held_name, crash_day)

    strategy, result = _run(_cooldown_config(config, 5), prices, monkeypatch)
    stops = [t for t in result.trades if t["type"] == "stop_loss" and held_name in t["symbols"]]
    assert stops, "the crash should trip the fixed stop"
    stop_day = stops[0]["date"]
    assert strategy.backtest_engine.stop_exits.get(held_name) is not None
    idx = prices.index
    at = idx.get_loc(stop_day)
    inside = [d for d in result.targets if at <= idx.get_loc(d) <= at + 5]
    assert inside, "a rebalance should fall inside the cooldown window"
    for d in inside:
        assert result.targets[d].get(held_name, 0.0) == 0.0, f"rebought {held_name} on {d.date()}"

    # The same frame without the cooldown buys it straight back: the rule,
    # not the frame, is what keeps it out.
    _, free = _run(_cooldown_config(config, 0), prices, monkeypatch)
    assert any(free.targets[d].get(held_name, 0.0) > 0 for d in inside if d in free.targets)


def test_live_target_skips_names_in_cooldown(config, synthetic_prices, monkeypatch):
    cfg = _cooldown_config(config, 5)
    live = MultiFactorStrategy(cfg)
    monkeypatch.setattr("quant.strategy.enforce_live_data_quality", lambda p, benchmark=None, as_of=None: p)
    monkeypatch.setattr(live.data, "fetch_prices", lambda start=None, end=None: synthetic_prices)
    monkeypatch.setattr(live.data, "fetch_fundamentals", lambda **kwargs: pd.DataFrame())
    free = live.get_current_portfolio(capital=1_000_000.0)
    top = free["weight"].sort_values(ascending=False).index[0]

    last = synthetic_prices.index[-1]
    blocked = live.get_current_portfolio(capital=1_000_000.0, stop_exits={top: str(last.date())})
    assert top not in blocked.index or float(blocked.loc[top, "weight"]) == 0.0
    assert len(blocked) > 0

    # A stop older than the cooldown no longer blocks the name.
    old = str(synthetic_prices.index[-20].date())
    again = live.get_current_portfolio(capital=1_000_000.0, stop_exits={top: old})
    assert top in again.index


def test_runner_records_fixed_stops_and_reads_old_state():
    class Strategy:
        reentry_cooldown = 5
        last_prices_ = None

    state = {"trade_history": [
        {"date": "2026-10-22", "stop_loss": True, "trades": [
            {"symbol": "AMAT", "side": "sell", "status": "filled", "time": "2026-10-22T14:42:55"},
            {"symbol": "AMD", "side": "sell", "status": "filled", "time": "2026-10-22T14:42:57"},
        ]},
        {"date": "2026-10-01", "trades": [
            {"symbol": "MU", "side": "sell", "status": "filled", "time": "2026-10-01T15:00:00"},
        ]},
    ]}
    # A state from before the record: the flagged stop sells are read back,
    # an ordinary sell is not.
    assert common.stop_exit_record(state) == {"AMAT": "2026-10-22", "AMD": "2026-10-22"}
    assert common.stop_exit_context(state) == {"stop_exits": {"AMAT": "2026-10-22", "AMD": "2026-10-22"}}

    common.record_fixed_stop_exits(Strategy(), state, ["LRCX"])
    assert set(state["stop_exits"]) == {"AMAT", "AMD", "LRCX"}

    # With prices to count on, records past the window are dropped.
    idx = pd.bdate_range("2026-10-01", "2026-11-13")
    Strategy.last_prices_ = pd.DataFrame(np.ones((len(idx), 1)), index=idx, columns=["X"])
    state["stop_exits"] = {"OLD": "2026-10-01", "NEW": "2026-11-12"}
    common.record_fixed_stop_exits(Strategy(), state, [])
    assert set(state["stop_exits"]) == {"NEW"}


def test_factor_profiles_carry_the_cooldown_hooks():
    import paper_trade
    import paper_trade_lgbm

    for profile in (paper_trade.PROFILE, paper_trade_lgbm.PROFILE):
        assert profile.portfolio_context is common.stop_exit_context
        assert profile.record_stop_exits is common.record_fixed_stop_exits

    from quant.utils.config import load_config

    assert load_config("config.yaml")["risk"]["reentry_cooldown_sessions"] == 5
