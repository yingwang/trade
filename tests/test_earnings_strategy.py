"""Earnings book: point-in-time universe, announcement signals, an offline backtest."""

import numpy as np
import pandas as pd
import pytest

from quant.data.earnings import (
    announcement_panels,
    as_of,
    current_members,
    members_since,
    membership_mask,
    normalize_earnings_frame,
)
from quant.earnings_strategy import EarningsStrategy

SYMBOLS = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG", "HHHH", "IIII", "JJJJ"]


def _config(config):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    cfg["earnings"] = {
        "positions": 3, "rank_buffer": 5, "vol_window": 60, "sma_long": 200, "sma_short": 50,
        "trailing_stop": 0.15, "trailing_window": 60, "benchmark_vol_window": 21,
        "benchmark_vol_high": 0.30, "breadth_min": 0.30, "regime_caps": {0: None, 1: 0.6, 2: 0.0},
        "deleverage_tolerance": 0.10, "reentry_cooldown_sessions": 5,
        "weights": {"sue": 1.0, "eps_growth": 1.0, "days_to_next": 1.0, "momentum": 1.0},
        "momentum_lookback": 252, "momentum_skip": 126, "min_parts": 2,
    }
    cfg["portfolio"] = dict(config["portfolio"], max_positions=3, max_position_weight=0.5,
                            target_volatility=0.25, rebalance_frequency_days=5,
                            max_turnover_per_rebalance=1.0)
    cfg["risk"] = dict(config["risk"], stop_loss_pct=0.15, max_sector_weight=1.0)
    cfg["leverage"] = dict(config["leverage"], max_leverage=1.5)
    cfg["safety"] = {"max_position_pct_of_portfolio": 0.8}
    return cfg


def _prices(seed=5, days=900):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=days)
    frames = {}
    for i, sym in enumerate(SYMBOLS):
        rets = rng.normal(0.0006 + 0.0002 * (i % 4), 0.012, size=days)
        frames[sym] = 100.0 * np.cumprod(1.0 + rets)
    frames["BENCH"] = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, size=days))
    return pd.DataFrame(frames, index=dates)


def _snapshots(dates):
    """Everyone is a member from the start; JJJJ leaves on the last day of
    2020 and IIII only joins then."""
    first = pd.DataFrame(True, index=[dates[0]], columns=SYMBOLS)
    first.loc[dates[0], "IIII"] = False
    second = pd.DataFrame(True, index=[pd.Timestamp("2020-12-31")], columns=SYMBOLS)
    second.loc[pd.Timestamp("2020-12-31"), "JJJJ"] = False
    return pd.concat([first, second])


def _events(dates, seed=11):
    """Quarterly announcements with a surprise that is persistently positive
    for the first names and negative for the last."""
    rng = np.random.default_rng(seed)
    out = {}
    quarters = pd.date_range("2018-02-01", "2022-11-01", freq="QS-FEB") + pd.Timedelta(days=3)
    for i, sym in enumerate(SYMBOLS):
        bias = 0.3 - 0.06 * i
        est = 1.0 + 0.05 * np.arange(len(quarters))
        act = est + bias + rng.normal(0, 0.05, size=len(quarters))
        out[sym] = pd.DataFrame({
            "estimate": est, "actual": act,
            "surprise_pct": (act - est) / est * 100, "after_close": True,
        }, index=quarters)
    return out


@pytest.fixture
def strategy(config, monkeypatch):
    s = EarningsStrategy(_config(config))
    prices = _prices()
    s._snapshots = _snapshots(prices.index)
    s._events = _events(prices.index)
    monkeypatch.setattr(s.data, "fetch_prices", lambda start=None, end=None: prices)
    s.data.last_open_prices_ = None
    s.data.last_volumes_ = None
    return s


def test_as_of_is_known_only_from_its_date():
    values = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-03-02", "2020-06-01"]))
    dates = pd.to_datetime(["2020-03-01", "2020-03-02", "2020-05-29", "2020-06-01", "2020-12-31"])
    out = as_of(values, pd.DatetimeIndex(dates))
    assert np.isnan(out.iloc[0]) and out.iloc[1] == 1.0 and out.iloc[2] == 1.0 and out.iloc[3] == 2.0 and out.iloc[4] == 2.0


def test_membership_helpers():
    dates = pd.bdate_range("2019-01-01", periods=900)
    snaps = _snapshots(dates)
    mask = membership_mask(snaps, dates, SYMBOLS)
    assert not mask.loc["2019-06-03", "IIII"] and mask.loc["2021-01-04", "IIII"]
    assert mask.loc["2019-06-03", "JJJJ"] and not mask.loc["2021-01-04", "JJJJ"]
    assert "JJJJ" not in current_members(snaps) and "IIII" in current_members(snaps)
    assert members_since(snaps, "2021-06-01", lead_days=400) == SYMBOLS
    assert "JJJJ" not in members_since(snaps, "2022-06-01", lead_days=400)


def test_normalize_earnings_frame_reads_yfinance_shape():
    idx = pd.to_datetime(["2024-01-25 16:05:00-05:00", "2024-04-25 07:00:00-04:00"], utc=True)
    ed = pd.DataFrame({"EPS Estimate": [1.0, 1.1], "Reported EPS": [1.2, np.nan], "Surprise(%)": [20.0, np.nan]}, index=idx)
    frame = normalize_earnings_frame(ed)
    assert list(frame.index) == [pd.Timestamp("2024-01-25"), pd.Timestamp("2024-04-25")]
    assert bool(frame["after_close"].iloc[0]) and not bool(frame["after_close"].iloc[1])
    assert frame["actual"].iloc[0] == 1.2 and np.isnan(frame["actual"].iloc[1])


def test_announcement_panels_are_point_in_time():
    dates = pd.bdate_range("2019-01-01", periods=900)
    events = _events(dates)
    closes = _prices()[SYMBOLS]
    panels = announcement_panels(events, dates, closes)
    ann = events["AAAA"].index
    a_date = ann[ann > pd.Timestamp("2020-06-01")][0]
    before = dates[dates < a_date][-1]
    # The surprise of that announcement is unknown the session before and known from the announcement date.
    assert panels["surprise_pct"].loc[before, "AAAA"] == events["AAAA"].loc[: before, "surprise_pct"].iloc[-1]
    assert panels["surprise_pct"].loc[a_date, "AAAA"] == events["AAAA"].loc[a_date, "surprise_pct"]
    assert panels["days_since"].loc[a_date, "AAAA"] == 0
    assert panels["days_to_next"].loc[before, "AAAA"] == (a_date - before).days
    # The persistent beater carries a positive SUE, the persistent misser a negative one.
    late = dates[-1]
    assert panels["sue_consensus"].loc[late, "AAAA"] > 0 > panels["sue_consensus"].loc[late, "JJJJ"]


def test_scores_rank_members_only_and_favour_beats(strategy):
    prices = _prices()
    pre = strategy.precompute(prices)
    date = pd.Timestamp("2020-06-15")
    scores = strategy.scores_on(pre, date)
    assert "IIII" not in scores.index           # not a member yet
    assert "JJJJ" in scores.index
    late = prices.index[-1]
    scores_late = strategy.scores_on(pre, late)
    assert "JJJJ" not in scores_late.index      # left the index
    assert "IIII" in scores_late.index
    assert abs(scores_late.mean()) < 0.5
    # On the surprise part alone, the persistent beaters sit above the missers
    # (the full composite also carries momentum, which these random prices
    # hand out by chance).
    strategy.weights = {"sue": 1.0}
    strategy.min_parts = 1
    by_sue = strategy.scores_on(pre, late)
    assert by_sue.index.get_loc("AAAA") < by_sue.index.get_loc("HHHH")
    assert set(by_sue.index[:3]) <= {"AAAA", "BBBB", "CCCC", "DDDD"}


def test_scheduled_target_is_a_sized_book(strategy):
    prices = _prices()
    pre = strategy.precompute(prices)
    target = strategy.scheduled_target(pre, prices.index[-1], None)
    assert 1 <= len(target) <= 3
    assert (target > 0).all()
    assert target.sum() <= 1.5 + 1e-9
    assert (target <= 0.5 * 1.5 + 1e-9).all()


def test_backtest_runs_offline_on_the_point_in_time_universe(strategy):
    result = strategy.run_backtest(start="2020-06-01")
    assert len(result.equity_curve) > 200
    assert result.metrics["Point-in-Time Universe"] is True
    assert result.equity_curve.iloc[-1] > 0
    held = set()
    for row in result.positions_history:
        held |= set(row["holdings"].index[row["holdings"] > 0])
    assert held and held <= set(SYMBOLS)
    # JJJJ left the index at the end of 2020: it is never bought afterwards.
    after = [row for row in result.positions_history if row["date"] > pd.Timestamp("2021-02-01")]
    assert all(row["holdings"].get("JJJJ", 0) == 0 for row in after)
