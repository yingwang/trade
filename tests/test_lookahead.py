"""No bar after the decision date reaches a score, a trend decision or the
engine's book.

A rolling factor that peeked one bar ahead would backtest beautifully and
trade blind.  The check is the plain one: whatever is computed at date T
from the whole frame must equal what is computed at T from the frame cut
off at T.  It is run for every factor the production config enables, for
the composite the strategy trades on, for the trend book's scores and
regime lights, and for the backtest engine, whose provider must see only
the book marked at the decision close, the same book whether or not later
bars exist in the frame.
"""

import copy
import functools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import BacktestEngine
from quant.data.market_data import MarketData
from quant.signals.factors import SignalGenerator
from quant.trend_strategy import TrendStrategy
from quant.utils.config import load_config

PRODUCTION_SIGNALS = load_config(Path(__file__).resolve().parents[1] / "config.yaml")["signals"]
ENABLED_FACTORS = sorted(
    name for name, weight in PRODUCTION_SIGNALS["factor_weights"].items() if weight > 0
)
SYMBOLS = [f"S{i:02d}" for i in range(12)]
DAYS = 420
# Where the frame is cut: every production window (252-day momentum with
# its 21-day skip, the 252-day high, the 200-day average) is warm at the
# first, and the last is the frame's final bar.
CUTS = (300, 361, DAYS - 1)


@functools.lru_cache(maxsize=1)
def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2023-01-02", periods=DAYS)
    data = {}
    for i, sym in enumerate(SYMBOLS):
        rets = rng.normal(0.0002 + 0.0003 * (i % 4), 0.010 + 0.004 * (i % 3), DAYS)
        data[sym] = 100.0 * np.cumprod(1.0 + rets)
    data["BENCH"] = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.009, DAYS))
    return pd.DataFrame(data, index=dates)


def _signal_config() -> dict:
    """What SignalGenerator reads: the production signal set over the test universe."""
    return {
        "signals": copy.deepcopy(PRODUCTION_SIGNALS),
        "universe": {"symbols": list(SYMBOLS), "benchmark": "BENCH"},
    }


@functools.lru_cache(maxsize=None)
def _signals(cut: int | None):
    """Composite and per-factor scores from the frame cut at ``cut`` (None: whole)."""
    prices = _frame()
    if cut is not None:
        prices = prices.loc[:prices.index[cut]]
    generator = SignalGenerator(_signal_config())
    composite = generator.generate(prices, MarketData.compute_returns(prices))
    return composite, generator.last_factors_


@pytest.mark.parametrize("factor", ENABLED_FACTORS)
def test_enabled_factor_score_at_t_ignores_later_bars(factor):
    _, whole = _signals(None)
    assert factor in whole, f"{factor} is enabled but was not computed"
    for cut in CUTS:
        date = _frame().index[cut]
        _, truncated = _signals(cut)
        at_t = whole[factor].loc[date]
        assert at_t.notna().all(), f"{factor} is not warm at {date.date()}"
        np.testing.assert_allclose(
            truncated[factor].loc[date].to_numpy(), at_t.to_numpy(), rtol=0, atol=1e-10,
            err_msg=f"{factor} at {date.date()} depends on bars after it",
        )


def test_composite_signal_at_t_ignores_later_bars():
    whole, factors = _signals(None)
    assert set(ENABLED_FACTORS) <= set(factors)
    for cut in CUTS:
        date = _frame().index[cut]
        truncated, _ = _signals(cut)
        assert truncated.index[-1] == date
        assert whole.loc[date].notna().all()
        np.testing.assert_allclose(
            truncated.loc[date].to_numpy(), whole.loc[date].to_numpy(), rtol=0, atol=1e-10,
            err_msg=f"the composite at {date.date()} depends on bars after it",
        )


def test_trend_scores_and_lights_at_t_ignore_later_bars(config):
    strategy = TrendStrategy(config)
    prices = _frame()
    whole = strategy.precompute(prices)
    held = pd.Series(0.1, index=SYMBOLS[:4])
    for cut in CUTS:
        date = prices.index[cut]
        truncated = strategy.precompute(prices.loc[:date])
        scores = strategy.scores_on(whole, date)
        assert not scores.empty
        pd.testing.assert_series_equal(
            strategy.scores_on(truncated, date), scores, check_exact=False, atol=1e-10,
        )
        lights, lights_truncated = strategy.regime_lights(whole, date), strategy.regime_lights(truncated, date)
        for key in ("benchmark_below_sma", "benchmark_vol_high", "breadth_weak"):
            assert lights_truncated[key] == lights[key]
        assert lights_truncated["breadth"] == pytest.approx(lights["breadth"])
        assert strategy.trailing_exits(truncated, date, held) == strategy.trailing_exits(whole, date, held)


def test_engine_hands_the_provider_only_the_book_at_the_decision_close(config):
    cfg = copy.deepcopy(config)
    cfg["portfolio"]["transaction_cost_bps"] = 0
    cfg["backtest"].update({"slippage_bps": 0, "market_impact_coeff": 0})
    cfg["risk"]["stop_loss_pct"] = 0.0
    cfg["leverage"]["margin_annual_rate"] = 0.0
    prices = _frame().iloc[:120]
    decisions = list(prices.index[10:110:7])
    cut = decisions[7]

    def run(frame):
        seen = []

        def provider(date, current_weights):
            seen.append((date, current_weights.copy()))
            # A different pair every time, so the book changes at every call.
            i = len(seen) % len(SYMBOLS)
            return pd.Series({SYMBOLS[i]: 0.3, SYMBOLS[(i + 1) % len(SYMBOLS)]: 0.3})

        result = BacktestEngine(cfg).run(
            frame, {}, benchmark_col="BENCH",
            target_provider=provider, rebalance_dates=decisions,
        )
        return seen, result

    seen_whole, whole = run(prices)
    seen_cut, truncated = run(prices.loc[:cut])

    # One call per decision date, in session order, handed the book marked
    # at that close: the holdings the engine records for the date, over
    # the value it records for the date.
    assert [d for d, _ in seen_whole] == decisions
    rows = {row["date"]: row for row in whole.positions_history}
    for date, weights in seen_whole:
        row = rows[date]
        book = (row["holdings"] * prices.loc[date, SYMBOLS]) / row["value"]
        book = book[book > 1e-12]
        pd.testing.assert_series_equal(
            weights.sort_index(), book.sort_index(), check_names=False, check_exact=True,
        )

    # Bars after a decision date change nothing the provider saw up to it,
    # nor the equity curve up to it.
    up_to_cut = [(d, w) for d, w in seen_whole if d <= cut]
    assert [d for d, _ in seen_cut] == [d for d, _ in up_to_cut]
    for (_, from_whole), (_, from_cut) in zip(up_to_cut, seen_cut):
        pd.testing.assert_series_equal(from_cut, from_whole, check_exact=True)
    pd.testing.assert_series_equal(
        truncated.equity_curve, whole.equity_curve.loc[:cut], check_exact=True,
    )

    # A target decided at a close executes at the next session, never on
    # the bar it was decided on.
    trade_dates = [t["date"] for t in whole.trades if t["type"] == "rebalance"]
    assert trade_dates
    assert not set(trade_dates) & set(decisions)
    previous_session = {prices.index[i]: prices.index[i - 1] for i in range(1, len(prices))}
    assert all(previous_session[d] in set(decisions) for d in trade_dates)
