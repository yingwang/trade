"""Empty targets in strategy backtests (review 2026-11-21, F4).

The engine reads None as "no decision" and an empty target as "hold cash".
The trend rules that mean cash (two regime lights, every holding stopped)
now reach the backtest; an empty target the rules did not mean (nothing
ranked: a data gap) is a stand-down on the live path and no decision in the
backtest.  The second test needs the F4 part of the patch to
quant/trend_strategy.py (``provide``).
"""

import numpy as np
import pandas as pd
import pytest

from quant.earnings_strategy import EarningsStrategy
from quant.trend_strategy import TrendStrategy
from tests.test_earnings_strategy import _config, _events, _prices, _snapshots
from tests.test_trend_strategy import _calm_prices, _trend_config

pytestmark = pytest.mark.filterwarnings("ignore:SURVIVORSHIP BIAS WARNING")


def test_two_regime_lights_send_the_trend_backtest_to_cash(config, monkeypatch):
    strategy = TrendStrategy(_trend_config(config))
    prices = _calm_prices()
    bench = prices["BENCH"].to_numpy(copy=True)
    # The benchmark breaks below its 200-day average with a volatility spike.
    bench[-40:] = bench[-41] * np.cumprod(1.0 + np.tile([-0.06, 0.04], 20))
    prices["BENCH"] = bench
    monkeypatch.setattr(strategy.data, "fetch_prices", lambda start=None, end=None: prices)
    strategy.data.last_open_prices_ = None
    strategy.data.last_volumes_ = None
    result = strategy.run_backtest(start="2020-06-01")

    rows = {row["date"]: row for row in result.positions_history}
    invested_before = rows[prices.index[-45]]["holdings"].gt(0).any()
    assert invested_before
    cash_targets = [d for d, t in result.targets.items() if len(t) == 0]
    assert cash_targets, "the two-light rule produced no cash target"
    last = result.positions_history[-1]
    assert (last["holdings"] == 0).all(), "the backtest stayed invested through two lights"


def test_an_unmeant_empty_earnings_target_leaves_the_book_alone(config, monkeypatch):
    cfg = _config(config)
    cfg["earnings"].update(trailing_stop=0.0, regime_caps={0: None, 1: None, 2: None})
    cfg["risk"]["stop_loss_pct"] = 0.0
    strategy = EarningsStrategy(cfg)
    prices = _prices()
    strategy._snapshots = _snapshots(prices.index)
    strategy._events = _events(prices.index)
    monkeypatch.setattr(strategy.data, "fetch_prices", lambda start=None, end=None: prices)
    strategy.data.last_open_prices_ = None
    strategy.data.last_volumes_ = None
    cutoff = pd.Timestamp("2021-06-01")
    real_scores = strategy.scores_on

    def scores_on(pre, date):
        # After the cutoff nothing ranks: a data gap, not a rule.
        return pd.Series(dtype=float) if date >= cutoff else real_scores(pre, date)

    monkeypatch.setattr(strategy, "scores_on", scores_on)
    result = strategy.run_backtest(start="2020-06-01")
    # A week after the cutoff the last ranked target has been traded.
    rows = [row for row in result.positions_history if row["date"] >= cutoff]
    held_after_cutoff = rows[5]["holdings"]
    assert held_after_cutoff.gt(0).any()
    last = result.positions_history[-1]["holdings"]
    assert last.gt(0).any(), "a data gap sold the whole backtest book"
    pd.testing.assert_series_equal(last, held_after_cutoff, check_names=False)
