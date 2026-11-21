"""The earnings backtest asks Yahoo for the names the live fetch did not
cover (review 2026-11-21, F3).

The dashboard computes today's target first, which fetches announcements for
today's members only, and then runs the backtest on the same strategy.  The
backtest used to reuse that cache as it stood, so every former member had no
announcements, failed min_parts, and the published curve ranked only today's
members across the whole history.

Needs the F3 patch to quant/earnings_strategy.py (EarningsStrategy._ensure_events).
"""

import pandas as pd
import pytest

from quant.earnings_strategy import EarningsStrategy
from tests.test_earnings_strategy import SYMBOLS, _config, _events, _prices, _snapshots


@pytest.fixture
def setup(config, monkeypatch):
    strategy = EarningsStrategy(_config(config))
    prices = _prices()
    strategy._snapshots = _snapshots(prices.index)
    calls = []
    every = _events(prices.index)

    def fetch(symbols, limit=100, pause=0.0):
        calls.append(sorted(symbols))
        # Yahoo has nothing for HHHH.
        return {s: every[s] for s in symbols if s in every and s != "HHHH"}

    monkeypatch.setattr("quant.earnings_strategy.fetch_earnings", fetch)
    monkeypatch.setattr(
        strategy.data, "fetch_prices",
        lambda start=None, end=None: prices[
            [c for c in prices.columns if c in strategy.data.symbols or c == "BENCH"]
        ],
    )
    monkeypatch.setattr("quant.trend_strategy.enforce_live_data_quality",
                        lambda p, benchmark=None, as_of=None: p)
    strategy.data.last_open_prices_ = None
    strategy.data.last_volumes_ = None
    strategy.min_coverage = 0.0   # the coverage gate is not under test here
    return strategy, calls


def test_backtest_after_the_live_target_fetches_only_the_former_members(setup):
    strategy, calls = setup
    strategy.get_current_portfolio(capital=100_000.0, prev_weights=None)
    current = sorted(s for s in SYMBOLS if s != "JJJJ")   # JJJJ left the index
    assert calls == [current]

    strategy.run_backtest(start="2020-06-01")
    assert calls[1:] == [["JJJJ"]], "the backtest reused the live cache as it stood"
    assert "JJJJ" in strategy._events

    # Nothing new on a second run; HHHH, which Yahoo had nothing for, is not asked again.
    strategy.run_backtest(start="2020-06-01")
    assert len(calls) == 2
    assert "HHHH" not in strategy._events
