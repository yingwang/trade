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
