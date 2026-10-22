"""The live target is the backtest target: same date, same book, same weights.

Each strategy has two code paths to a target.  The backtest's provider
closure (``compute_target`` in ``MultiFactorStrategy.run_backtest``,
``provide`` in ``TrendStrategy.run_backtest``) is called by the engine at
each decision close with the book as it stands; the live
``get_current_portfolio`` is called by the runner with the account's actual
weights.  They share every building block but are written out twice, so a
fix applied to one can miss the other, and a backtest that validated one
path says nothing about the other.

These tests run the real engine, keep the exact ``prev_weights`` the
provider was handed at a date and the target it returned, then hand the
same book to the live path on the frame truncated at that date and require
the same weights.

Two live-only inputs are held fixed so that the comparison is about the
shared pipeline: the quality gate is the identity (the synthetic frame is
clean), and the fundamentals snapshot is empty, as it always is in the
backtest.  The last test in the multi-factor group says what happens when
that snapshot is not empty.
"""

import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.backtest.calendar import fixed_rebalance_dates
from quant.strategy import MultiFactorStrategy
from quant.trend_strategy import TrendStrategy
from quant.utils.config import load_config
from tests.test_trend_strategy import _calm_prices, _trend_config

pytestmark = pytest.mark.filterwarnings("ignore:SURVIVORSHIP BIAS WARNING")

PRODUCTION_SIGNALS = load_config(Path(__file__).resolve().parents[1] / "config.yaml")["signals"]


def _record_provider(monkeypatch, engine):
    """Wrap ``engine.run`` so each provider call's exact inputs and output are kept."""
    seen = {}
    real_run = engine.run

    def run(*args, **kwargs):
        provider = kwargs["target_provider"]

        def recording(date, prev_weights):
            target = provider(date, prev_weights)
            seen[pd.Timestamp(date)] = (
                prev_weights.copy(), None if target is None else target.copy(),
            )
            return target

        return real_run(*args, **{**kwargs, "target_provider": recording})

    monkeypatch.setattr(engine, "run", run)
    return seen


def _assert_same_target(live: pd.Series, expected: pd.Series, atol: float = 1e-10):
    assert set(live.index) == set(expected.index), (
        f"live {sorted(live.index)} vs backtest {sorted(expected.index)}"
    )
    np.testing.assert_allclose(
        live.reindex(expected.index).to_numpy(), expected.to_numpy(), rtol=0, atol=atol,
    )


def _targets_agree(live: pd.Series, expected: pd.Series) -> bool:
    return set(live.index) == set(expected.index) and np.allclose(
        live.reindex(expected.index).to_numpy(), expected.to_numpy(), rtol=0, atol=1e-10,
    )


# ---------------------------------------------------------------------------
# Multi-factor
# ---------------------------------------------------------------------------

def _factor_config(config: dict) -> dict:
    """The conftest config with the production signal set and turnover cap."""
    cfg = copy.deepcopy(config)
    cfg["signals"] = copy.deepcopy(PRODUCTION_SIGNALS)
    cfg["portfolio"]["max_turnover_per_rebalance"] = 0.40
    return cfg


def _factor_backtest(cfg: dict, prices: pd.DataFrame, monkeypatch):
    """A real backtest over the frame; returns the result and what the provider saw."""
    strategy = MultiFactorStrategy(cfg)
    monkeypatch.setattr(strategy.data, "fetch_prices", lambda start=None, end=None: prices)
    strategy.data.last_open_prices_ = None
    strategy.data.last_volumes_ = None
    seen = _record_provider(monkeypatch, strategy.backtest_engine)
    result = strategy.run_backtest(start=str(prices.index[len(prices) // 2].date()))
    return result, seen


def _factor_live(cfg: dict, prices: pd.DataFrame, date, prev_weights, monkeypatch,
                 fundamentals: pd.DataFrame | None = None) -> pd.Series:
    """The live path at ``date``'s close, handed ``prev_weights`` as the runner would."""
    live = MultiFactorStrategy(cfg)
    monkeypatch.setattr("quant.strategy.enforce_live_data_quality",
                        lambda p, benchmark=None, as_of=None: p)
    monkeypatch.setattr(live.data, "fetch_prices", lambda start=None, end=None: prices.loc[:date])
    snapshot = pd.DataFrame() if fundamentals is None else fundamentals
    monkeypatch.setattr(live.data, "fetch_fundamentals", lambda **kwargs: snapshot)
    table = live.get_current_portfolio(capital=1_000_000.0, prev_weights=prev_weights)
    return table["weight"]


@pytest.fixture
def factor_run(config, synthetic_prices, monkeypatch):
    cfg = _factor_config(config)
    result, seen = _factor_backtest(cfg, synthetic_prices, monkeypatch)
    scheduled = [d for d in sorted(seen) if seen[d][1] is not None]
    assert len(scheduled) >= 4, "the frame is too short for a drifted book"
    return cfg, synthetic_prices, result, seen, scheduled


def test_multifactor_target_matches_backtest_for_the_same_drifted_book(factor_run, monkeypatch, caplog):
    cfg, prices, result, seen, scheduled = factor_run
    # The third scheduled date with holdings: the book has drifted off two
    # targets, so the turnover penalty, constraint and cap all have a
    # previous book to bind against.
    date = [d for d in scheduled if len(seen[d][0]) > 0][2]
    prev, expected = seen[date]
    assert expected.equals(result.targets[date])
    assert 0.3 < float(prev.sum()) < 2.0

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="quant.portfolio.optimizer"):
        live = _factor_live(cfg, prices, date, prev, monkeypatch)

    # The comparison is between two solved optimizations, not two fallbacks.
    assert "did not converge" not in caplog.text
    _assert_same_target(live, expected)


def test_multifactor_target_matches_backtest_for_an_empty_book(factor_run, monkeypatch):
    cfg, prices, result, seen, scheduled = factor_run
    date = scheduled[0]
    prev, expected = seen[date]
    assert prev.empty

    # The provider turns the engine's empty book into None before the
    # optimizer sees it; handed None, the live path agrees exactly.
    _assert_same_target(_factor_live(cfg, prices, date, None, monkeypatch), expected)

    # The runner, though, hands get_current_portfolio what
    # current_broker_weights returns for an empty account: an empty Series,
    # which get_current_portfolio passes on as is.  optimize_weights anchors
    # the turnover penalty at equal weights for None and at zero for an empty
    # Series, and with the 40% turnover constraint the zero anchor is
    # infeasible, so the first live rebalance of a book can fall back to
    # score-proportional weights where the backtest's first target did not.
    as_the_runner_passes_it = _factor_live(cfg, prices, date, pd.Series(dtype=float), monkeypatch)
    if not _targets_agree(as_the_runner_passes_it, expected):
        pytest.xfail(
            "an empty previous book reaches the optimizer as None from the "
            "backtest and as an empty Series from the runner, and the two "
            "anchors give different first targets: live "
            f"{as_the_runner_passes_it.round(4).to_dict()} vs backtest "
            f"{expected.round(4).to_dict()}"
        )


def test_live_target_does_not_depend_on_the_fundamentals_snapshot_sectors(factor_run, monkeypatch):
    """The signal the live book trades should not change with the yfinance
    ``.info`` snapshot: the backtest never has one, and the static sector
    table in config is documented as the one classification for both paths.

    ``SignalGenerator.generate`` reads its neutralization sector map from
    the snapshot's ``sector`` column, so a live run with a snapshot
    industry-neutralizes the factors and a backtest, or a live run whose
    fundamentals fetch failed, does not.
    """
    cfg, prices, result, seen, scheduled = factor_run
    assert cfg["signals"]["industry_neutral"] is True
    date = [d for d in scheduled if len(seen[d][0]) > 0][2]
    prev, expected = seen[date]
    symbols = [c for c in prices.columns if c != "BENCH"]
    # Two sectors of five, enough for within-sector z-scoring to apply.
    snapshot = pd.DataFrame(
        {"sector": ["Tech"] * 5 + ["Health"] * 5}, index=symbols,
    )

    without = _factor_live(cfg, prices, date, prev, monkeypatch)
    with_snapshot = _factor_live(cfg, prices, date, prev, monkeypatch, fundamentals=snapshot)

    _assert_same_target(without, expected)
    if not _targets_agree(with_snapshot, expected):
        pytest.xfail(
            "with a fundamentals snapshot the live path industry-neutralizes "
            "by the snapshot's sector column, which the backtest never does: "
            f"live {with_snapshot.round(4).to_dict()} vs backtest "
            f"{expected.round(4).to_dict()}"
        )


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

def test_trend_target_matches_backtest_on_a_plain_scheduled_day(config, monkeypatch):
    """A scheduled date with a held book and no stop in sight: the case
    ``test_backtest_and_live_path_agree_on_a_stopped_name`` does not cover."""
    cfg = _trend_config(config)
    strategy = TrendStrategy(cfg)
    prices = _calm_prices()
    monkeypatch.setattr(strategy.data, "fetch_prices", lambda start=None, end=None: prices)
    strategy.data.last_open_prices_ = None
    strategy.data.last_volumes_ = None
    seen = _record_provider(monkeypatch, strategy.backtest_engine)
    result = strategy.run_backtest(start="2020-06-01")

    scheduled = set(fixed_rebalance_dates(prices.index, strategy.rebalance_freq, anchor=strategy.anchor))
    # Nothing happened between scheduled dates on the calm frame: no
    # trailing exit, no de-risking, no fixed stop.
    assert all(seen[d][1] is None for d in seen if d not in scheduled)
    assert not any(t["type"] == "stop_loss" for t in result.trades)
    held_days = [d for d in sorted(seen) if d in scheduled and len(seen[d][0]) > 0]
    date = held_days[3]
    prev, expected = seen[date]
    assert expected.equals(result.targets[date])

    # The live state's entry dates: the session each holding first appeared.
    rows = [r for r in result.positions_history if r["date"] <= date]
    entry_dates = {}
    for sym in prev.index:
        for row in reversed(rows):
            if float(row["holdings"].get(sym, 0.0)) <= 0:
                break
            entry_dates[sym] = row["date"]
    assert set(entry_dates) == set(prev.index)

    live = TrendStrategy(cfg)
    monkeypatch.setattr("quant.trend_strategy.enforce_live_data_quality",
                        lambda p, benchmark=None, as_of=None: p)
    monkeypatch.setattr(live.data, "fetch_prices", lambda start=None, end=None: prices.loc[:date])
    table = live.get_current_portfolio(
        capital=100_000.0, prev_weights=prev, entry_dates=entry_dates, stop_exits={},
    )

    assert live.last_decision_["mode"] == "scheduled"
    assert live.last_decision_["exits"] == [] and live.last_decision_["cooldown"] == []
    _assert_same_target(table["weight"], expected, atol=1e-12)
