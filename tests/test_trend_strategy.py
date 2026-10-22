"""Trend strategy: rules, maintenance trigger, and an offline backtest run."""

import numpy as np
import pandas as pd
import pytest

from quant.trend_strategy import TrendStrategy


def _trend_config(config):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    cfg["trend"] = {
        "positions": 3,
        "rank_buffer": 5,
        "skip_days": 21,
        "lookback_long": 252,
        "lookback_mid": 126,
        "lookback_short": 63,
        "blend": {"long": 0.4, "mid": 0.3, "short": 0.3},
        "vol_window": 60,
        "sma_long": 200,
        "sma_short": 50,
        "trailing_stop": 0.15,
        "trailing_window": 60,
        "benchmark_vol_window": 21,
        "benchmark_vol_high": 0.30,
        "breadth_min": 0.30,
        "regime_caps": {0: None, 1: 0.6, 2: 0.0},
        "deleverage_tolerance": 0.10,
        "reentry_cooldown_sessions": 5,
    }
    cfg["portfolio"] = dict(config["portfolio"], max_positions=3, max_position_weight=0.5,
                            target_volatility=0.35, rebalance_frequency_days=5,
                            max_turnover_per_rebalance=1.0)
    cfg["risk"] = dict(config["risk"], stop_loss_pct=0.15, max_sector_weight=1.0)
    cfg["leverage"] = dict(config["leverage"], max_leverage=1.9)
    cfg["safety"] = {"max_position_pct_of_portfolio": 0.6}
    return cfg


def _synthetic_prices(seed=7, days=900, drifts=None):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=days)
    symbols = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE",
               "FFFF", "GGGG", "HHHH", "IIII", "JJJJ"]
    drifts = drifts or {}
    frames = {}
    for i, sym in enumerate(symbols):
        # Clear uptrends with modest noise, so most names are eligible and
        # the tests exercise the rules rather than the dice.
        mu = drifts.get(sym, 0.0008 + 0.0003 * (i % 3))
        rets = rng.normal(mu, 0.010, size=days)
        frames[sym] = 100.0 * np.cumprod(1.0 + rets)
    frames["BENCH"] = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, size=days))
    return pd.DataFrame(frames, index=dates)


def _calm_prices(days=800, seed=3):
    """Ten steady uptrends with next to no noise, so that across a whole
    backtest only a move written into a name by hand can trip a stop."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=days)
    symbols = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE",
               "FFFF", "GGGG", "HHHH", "IIII", "JJJJ"]
    frames = {}
    for i, sym in enumerate(symbols):
        rets = rng.normal(0.0005 + 0.0001 * i, 0.002, size=days)
        frames[sym] = 100.0 * np.cumprod(1.0 + rets)
    frames["BENCH"] = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.002, size=days))
    return pd.DataFrame(frames, index=dates)


@pytest.fixture
def strategy(config):
    return TrendStrategy(_trend_config(config))


def test_scores_only_include_names_in_uptrend(strategy):
    prices = _synthetic_prices()
    # Send one name straight down for the whole history: never eligible.
    prices["JJJJ"] = np.linspace(200.0, 50.0, len(prices))
    pre = strategy.precompute(prices)
    scores = strategy.scores_on(pre, prices.index[-1])
    assert "JJJJ" not in scores.index
    assert len(scores) >= 3
    assert scores.is_monotonic_decreasing


def test_selection_keeps_holdings_inside_the_rank_buffer(strategy):
    scores = pd.Series({s: 10.0 - i for i, s in enumerate("ABCDEFGH")})
    held = pd.Series({"E": 0.3, "H": 0.3})  # ranks 5 and 8
    chosen = strategy.select(scores, held)
    assert len(chosen) == 3
    assert chosen[0] == "E"       # rank 5 <= buffer 5: kept, and listed first
    assert "H" not in chosen      # rank 8 > buffer: dropped
    assert chosen[1:] == ["A", "B"]  # remaining slots fill from the top


def test_weights_are_inverse_vol_and_capped(strategy):
    rng = np.random.default_rng(1)
    rets = pd.DataFrame({
        "LOW": rng.normal(0, 0.005, 200),
        "MID": rng.normal(0, 0.015, 200),
        "HIGH": rng.normal(0, 0.04, 200),
    })
    w = strategy.inverse_vol_weights(rets, ["LOW", "MID", "HIGH"])
    assert abs(w.sum() - 1.0) < 1e-9
    assert w["LOW"] > w["MID"] > w["HIGH"]
    assert (w <= strategy.max_weight + 1e-9).all()


def test_regime_cap_steps_down_with_each_light(strategy):
    assert strategy.regime_cap({}) == pytest.approx(1.9)
    assert strategy.regime_cap({"benchmark_below_sma": True}) == pytest.approx(0.6)
    assert strategy.regime_cap({"benchmark_below_sma": True, "breadth_weak": True}) == 0.0


def test_gross_target_respects_vol_target_and_caps(strategy):
    rng = np.random.default_rng(2)
    calm = pd.DataFrame({"A": rng.normal(0, 0.004, 120), "B": rng.normal(0, 0.004, 120)})
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert strategy.gross_target(calm, w, cap=1.9) == pytest.approx(1.9)
    wild = pd.DataFrame({"A": rng.normal(0, 0.05, 120), "B": rng.normal(0, 0.05, 120)})
    g = strategy.gross_target(wild, w, cap=1.9)
    assert 0.0 < g < 1.0
    assert strategy.gross_target(wild, w, cap=0.0) == 0.0


def test_maintenance_sells_a_holding_below_its_trailing_stop(strategy):
    prices = _synthetic_prices()
    prices["BBBB"] = np.linspace(100.0, 140.0, len(prices))  # steady, never near a stop
    # AAAA falls 25% over the last ten sessions from a flat level.
    prices["AAAA"] = 100.0
    prices.loc[prices.index[-10]:, "AAAA"] = np.linspace(100.0, 75.0, 10)
    pre = strategy.precompute(prices)
    held = pd.Series({"AAAA": 0.4, "BBBB": 0.4})
    target = strategy.maintenance_target(pre, prices.index[-1], held)
    assert target is not None
    assert "AAAA" not in target.index
    assert target["BBBB"] == pytest.approx(0.4)
    assert strategy.last_decision_["exits"] == ["AAAA"]


def test_trailing_stop_measures_from_entry_when_the_entry_date_is_known(strategy):
    prices = _synthetic_prices()
    n = len(prices)
    # AAAA slid from 200 to 140 over sixty sessions and has sat at 140 for the
    # last thirty: well under its 60-day high, but flat since the book bought it.
    aaaa = np.full(n, 200.0)
    aaaa[-90:-30] = np.linspace(200.0, 140.0, 60)
    aaaa[-30:] = 140.0
    prices["AAAA"] = aaaa
    # BBBB was bought thirty sessions ago at 100, ran to 130 and gave a fifth back.
    bbbb = np.full(n, 100.0)
    bbbb[-30:-15] = np.linspace(100.0, 130.0, 15)
    bbbb[-15:] = np.linspace(130.0, 104.0, 15)
    prices["BBBB"] = bbbb
    pre = strategy.precompute(prices)
    date = prices.index[-1]
    held = pd.Series({"AAAA": 0.4, "BBBB": 0.4})
    entries = {"AAAA": prices.index[-20], "BBBB": prices.index[-30]}
    # Without entry dates the old 60-day high rule ejects both.
    assert strategy.trailing_exits(pre, date, held) == ["AAAA", "BBBB"]
    # Measured from the peak since entry only BBBB has broken down.
    assert strategy.trailing_exits(pre, date, held, entries) == ["BBBB"]
    # An entry after the last close cannot be judged yet.
    assert strategy.trailing_exits(pre, date, held, {"AAAA": date + pd.Timedelta(days=1), "BBBB": date + pd.Timedelta(days=1)}) == []
    target = strategy.maintenance_target(pre, date, held, entries)
    assert list(target.index) == ["AAAA"]


def test_entry_dates_are_kept_with_entry_prices_and_backfilled():
    import paper_trade_common as common
    import paper_trade_trend as trend

    class Broker:
        def get_positions(self):
            return pd.Series({"AAAA": 10.0})

    state = {"entry_prices": {}, "entry_dates": {}}
    filled = [
        {"symbol": "AAAA", "side": "buy", "status": "filled", "price": 10.0, "time": "2026-09-17T18:22:24"},
        {"symbol": "BBBB", "side": "buy", "status": "filled", "price": 5.0, "time": "2026-09-17T18:22:25"},
    ]
    common.update_entry_prices(state, filled, Broker(), {})
    assert state["entry_dates"] == {"AAAA": "2026-09-17", "BBBB": "2026-09-17"}
    common.update_entry_prices(
        state, [{"symbol": "BBBB", "side": "sell", "status": "filled", "price": 5.0, "time": "2026-09-18T15:00:00"}],
        Broker(), {},
    )
    assert "BBBB" not in state["entry_prices"] and "BBBB" not in state["entry_dates"]

    # A state written before entry dates existed: the earliest buy fill stands in.
    legacy = {
        "entry_prices": {"AAAA": 10.0},
        "trade_history": [{"date": "2026-09-17T18:22:40", "trades": [
            {"symbol": "AAAA", "side": "buy", "status": "filled", "time": "2026-09-17T18:22:24"},
        ]}],
    }
    assert trend._entry_dates(legacy) == {"AAAA": "2026-09-17"}


def test_maintenance_is_quiet_when_nothing_changed(strategy):
    prices = _synthetic_prices()
    pre = strategy.precompute(prices)
    # Flat, calm, everything above its averages: no light, no stop.
    held = pd.Series({"AAAA": 0.5, "BBBB": 0.5})
    prices_flat = prices.copy()
    for col in prices_flat.columns:
        prices_flat[col] = np.linspace(100.0, 130.0, len(prices_flat))
    pre = strategy.precompute(prices_flat)
    assert strategy.maintenance_target(pre, prices_flat.index[-1], held) is None


def test_maintenance_deleverages_when_the_market_breaks(strategy):
    prices = _synthetic_prices()
    # Benchmark collapses below its 200-day average with a volatility spike:
    # two lights, cap zero, the whole book goes to cash.
    bench = prices["BENCH"].to_numpy(copy=True)
    bench[-30:] = bench[-31] * np.cumprod(1.0 + np.tile([-0.06, 0.04], 15))
    prices["BENCH"] = bench
    pre = strategy.precompute(prices)
    lights = strategy.regime_lights(pre, prices.index[-1])
    assert lights["benchmark_below_sma"] and lights["benchmark_vol_high"]
    held = pd.Series({"AAAA": 0.9, "BBBB": 0.9})
    target = strategy.maintenance_target(pre, prices.index[-1], held)
    assert target is not None and len(target) == 0


def test_backtest_runs_offline_within_leverage_limits(strategy, monkeypatch):
    prices = _synthetic_prices(days=800)
    monkeypatch.setattr(strategy.data, "fetch_prices", lambda start=None, end=None: prices)
    strategy.data.last_open_prices_ = None
    strategy.data.last_volumes_ = None
    result = strategy.run_backtest(start="2020-06-01", end="2022-01-31")
    assert not result.equity_curve.empty
    assert result.targets, "no scheduled targets were produced"
    for target in result.targets.values():
        assert float(target.sum()) <= 1.9 + 1e-6
        assert len(target) <= 3
    assert "Sharpe Ratio" in result.metrics


def test_live_portfolio_table_and_trigger(strategy, monkeypatch):
    prices = _synthetic_prices()
    prices["BBBB"] = np.linspace(100.0, 140.0, len(prices))
    prices["AAAA"] = 100.0
    prices.loc[prices.index[-10]:, "AAAA"] = np.linspace(100.0, 75.0, 10)
    monkeypatch.setattr("quant.trend_strategy.enforce_live_data_quality",
                        lambda p, benchmark=None, as_of=None: p)
    monkeypatch.setattr(strategy.data, "fetch_prices", lambda start=None, end=None: prices)
    held = pd.Series({"AAAA": 0.4, "BBBB": 0.4})
    reason = strategy.maintenance_needed(held)
    assert reason and "AAAA" in reason
    table = strategy.get_current_portfolio(capital=100_000.0, prev_weights=held)
    assert list(table.columns) == ["score", "weight", "weight_pct", "dollars", "shares", "price"]
    assert "AAAA" not in table.index
    assert table.loc["BBBB", "weight"] == pytest.approx(0.4)
    # The mode resets: the next call is a scheduled rebalance again.
    assert strategy.next_mode == "scheduled"
    scheduled = strategy.get_current_portfolio(capital=100_000.0, prev_weights=held)
    assert len(scheduled) <= 3
    assert float(scheduled["weight"].sum()) <= 1.9 + 1e-6


def test_shipped_config_validates_and_builds():
    """config_trend.yaml must pass the shared validator and construct the strategy."""
    from quant.utils.config import load_config
    cfg = load_config("config_trend.yaml")
    strat = TrendStrategy(cfg)
    assert strat.positions == cfg["portfolio"]["max_positions"]
    assert strat.max_leverage <= 2.0
    # The safety cap must leave room for the largest levered position.
    assert cfg["safety"]["max_position_pct_of_portfolio"] >= strat.max_weight * strat.max_leverage


def test_scheduled_target_passes_over_names_under_their_stop(strategy):
    prices = _synthetic_prices()
    n = len(prices)
    prices["BBBB"] = np.linspace(100.0, 140.0, n)  # steady, never near a stop
    # AAAA: by far the strongest trend in the frame, then a quarter lost from
    # its peak over the last ten sessions. It still ranks first on every horizon.
    aaaa = 100.0 * np.exp(0.008 * np.arange(n))
    aaaa[-10:] = np.linspace(aaaa[-11], aaaa[-11] * 0.75, 10)
    prices["AAAA"] = aaaa
    pre = strategy.precompute(prices)
    date = prices.index[-1]
    assert strategy.scores_on(pre, date).index[0] == "AAAA"

    # Held, with its peak since entry a third above the close: sold here, not
    # kept for the daily check to sell an hour later.
    held = pd.Series({"AAAA": 0.5, "BBBB": 0.5})
    target = strategy.scheduled_target(pre, date, held, entry_dates={"AAAA": prices.index[-30]})
    assert "AAAA" not in target.index
    assert strategy.last_decision_["exits"] == ["AAAA"]
    assert len(target) == 3
    # Not held: the ranking decides, the stop only ever judges a holding. A
    # fresh entry starts its own peak at this close, so the daily check cannot
    # sell it the same day; judging a candidate by its 60-day high would be a
    # new selection rule, not part of the churn fix.
    target = strategy.scheduled_target(pre, date, None)
    assert "AAAA" in target.index
    assert strategy.last_decision_["below_stop"] == []
    # Opened at this very close: no peak to have fallen from yet, so it stays.
    target = strategy.scheduled_target(pre, date, held, entry_dates={"AAAA": date})
    assert "AAAA" in target.index
    assert strategy.last_decision_["exits"] == []


def test_stopped_name_is_not_bought_back_inside_the_cooldown(strategy):
    prices = _synthetic_prices()
    idx = prices.index
    prices["AAAA"] = 100.0 * np.exp(0.004 * np.arange(len(idx)))  # ranked first, at its high
    pre = strategy.precompute(prices)
    date = idx[-1]
    assert strategy.scores_on(pre, date).index[0] == "AAAA"

    # Sold on a stop at the previous close: not bought back the next session...
    target = strategy.scheduled_target(pre, date, None, stop_exits={"AAAA": str(idx[-2].date())})
    assert "AAAA" not in target.index
    assert strategy.last_decision_["cooldown"] == ["AAAA"]
    # ...nor five sessions on, the last session of the cooldown...
    target = strategy.scheduled_target(pre, date, None, stop_exits={"AAAA": idx[-6]})
    assert "AAAA" not in target.index
    # ...but six sessions on it is a candidate again and its record has lapsed.
    target = strategy.scheduled_target(pre, date, None, stop_exits={"AAAA": idx[-7]})
    assert "AAAA" in target.index
    assert strategy.last_decision_["cooldown"] == []
    assert strategy.active_stop_exits(idx, date, {"AAAA": idx[-7], "BBBB": idx[-6]}) == {"BBBB": idx[-6]}


def test_backtest_and_live_path_agree_on_a_stopped_name(config, monkeypatch):
    from quant.backtest.calendar import fixed_rebalance_dates

    # No rank buffer here: the names bought in AAAA's absence would otherwise
    # keep their places, and the question is whether AAAA may come back.
    cfg = _trend_config(config)
    cfg["trend"]["rank_buffer"] = cfg["trend"]["positions"]
    strategy = TrendStrategy(cfg)
    prices = _calm_prices()
    idx = prices.index
    scheduled = fixed_rebalance_dates(idx, strategy.rebalance_freq, anchor=strategy.anchor)
    # AAAA leads the whole frame. Three sessions before a scheduled date it
    # closes 16% under its peak (the daily check sells it at the next open),
    # falls further, then recovers to 8% under the peak and stays there. At
    # that scheduled date it ranks first on every horizon and is no longer
    # under its stop: only the record of the stop keeps it out. One scheduled
    # date later the cooldown has run and it is bought back.
    rebalance = [d for d in scheduled if d <= idx[-40]][-1]
    stop_pos = idx.get_loc(rebalance) - 3
    aaaa = 100.0 * np.exp(0.004 * np.arange(len(idx)))
    peak = aaaa[stop_pos - 1]
    aaaa[stop_pos] = peak * 0.84
    aaaa[stop_pos + 1] = peak * 0.76
    aaaa[stop_pos + 2] = peak * 0.85
    aaaa[stop_pos + 3:] = peak * 0.92
    prices["AAAA"] = aaaa
    stop_day = idx[stop_pos]
    after = next(d for d in scheduled if d > rebalance)

    monkeypatch.setattr(strategy.data, "fetch_prices", lambda start=None, end=None: prices)
    strategy.data.last_open_prices_ = None
    strategy.data.last_volumes_ = None
    result = strategy.run_backtest(start="2020-06-01")
    targets = result.targets
    before = [d for d in scheduled if d < stop_day][-1]
    assert "AAAA" in targets[before].index
    assert "AAAA" not in targets[stop_day].index          # the trailing stop
    assert "AAAA" not in targets[rebalance].index         # the cooldown
    assert "AAAA" in targets[after].index                 # and back in

    def live_decision(date):
        """The live path at ``date``, handed the book, entry dates and stop
        record the backtest kept for itself up to that close."""
        rows = [r for r in result.positions_history if r["date"] <= date]
        held = (rows[-1]["holdings"] * prices.loc[date]).dropna() / rows[-1]["value"]
        held = held[held > 1e-6]
        entry_dates = {}
        for sym in held.index:
            for row in reversed(rows):
                if float(row["holdings"].get(sym, 0.0)) <= 0:
                    break
                entry_dates[sym] = row["date"]
        live = TrendStrategy(cfg)
        monkeypatch.setattr("quant.trend_strategy.enforce_live_data_quality",
                            lambda p, benchmark=None, as_of=None: p)
        monkeypatch.setattr(live.data, "fetch_prices", lambda start=None, end=None: prices.loc[:date])
        table = live.get_current_portfolio(
            capital=100_000.0, prev_weights=held,
            entry_dates=entry_dates, stop_exits={"AAAA": str(stop_day.date())},
        )
        return live, table

    live, table = live_decision(rebalance)
    assert live.last_scores_.index[0] == "AAAA"
    assert live.last_decision_["cooldown"] == ["AAAA"]
    assert live.last_decision_["exits"] == [] and live.last_decision_["below_stop"] == []
    expected = targets[rebalance]
    assert set(table.index) == set(expected.index)
    assert np.allclose(table["weight"].reindex(expected.index).to_numpy(), expected.to_numpy())

    live, table = live_decision(after)
    assert live.last_decision_["cooldown"] == []
    expected = targets[after]
    assert "AAAA" in table.index and set(table.index) == set(expected.index)
    assert np.allclose(table["weight"].reindex(expected.index).to_numpy(), expected.to_numpy())


def test_stop_exits_are_kept_in_state_and_handed_to_the_strategy(strategy):
    import paper_trade_trend as trend

    today = pd.Timestamp.today().normalize()
    today_str = str(today.date())
    # The runner's fixed stop fires before any prices are fetched: recorded
    # for today, nothing pruned.
    state = {"entry_prices": {"MU": 900.0}, "entry_dates": {"MU": "2026-09-21"}}
    strategy.last_prices_ = None
    strategy.last_decision_ = {}
    trend._record_stop_exits(strategy, state, ["AMD"])
    assert state["stop_exits"] == {"AMD": today_str}
    # A later decision that sold on the trailing stop adds its exits, dated by
    # the close that triggered them, and drops records past the cooldown.
    idx = pd.date_range(end=today, periods=20, freq="D")
    strategy.last_prices_ = pd.DataFrame({"X": 1.0}, index=idx)
    strategy.last_decision_ = {"mode": "scheduled", "date": today_str, "exits": ["MRVL"]}
    state["stop_exits"]["OLD"] = str(idx[0].date())
    state["stop_exits"]["INTC"] = str(idx[-4].date())
    trend._record_stop_exits(strategy, state, [])
    assert state["stop_exits"] == {"AMD": today_str, "INTC": str(idx[-4].date()), "MRVL": today_str}
    # Both records reach the scheduled rebalance through the profile.
    assert trend.PROFILE.portfolio_context(state) == {
        "entry_dates": {"MU": "2026-09-21"}, "stop_exits": state["stop_exits"],
    }
    assert trend.PROFILE.record_stop_exits is trend._record_stop_exits
    # A state from before the record existed reads the runner's flagged
    # stop-loss sells back from the fill history; other sells stay unknown.
    legacy = {"trade_history": [
        {"date": "2026-10-13T18:39:09", "completed": True, "trades": [
            {"symbol": "INTC", "side": "sell", "status": "filled", "time": "2026-10-13T18:38:58"},
        ]},
        {"date": "2026-10-20T15:00:00", "completed": True, "stop_loss": True, "trades": [
            {"symbol": "SNOW", "side": "sell", "status": "filled", "time": "2026-10-20T14:59:00"},
        ]},
    ]}
    assert trend._stop_exits(legacy) == {"SNOW": "2026-10-20"}


def test_reentry_cooldown_is_configured_and_validated():
    from quant.utils.config import load_config, validate_config
    cfg = load_config("config_trend.yaml")
    assert TrendStrategy(cfg).reentry_cooldown == 5
    for bad in (-1, 2.5, True, "5"):
        cfg["trend"]["reentry_cooldown_sessions"] = bad
        with pytest.raises(ValueError, match="reentry_cooldown_sessions"):
            validate_config(cfg)

