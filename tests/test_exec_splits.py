"""The split guard on small ratios, with and without the runner's own records.

The price-scale test alone (average entry over current quote) cannot tell a
position the broker adjusted for a 2-for-1 split from one it did not: with a
threshold of ratio / 3 every adjusted position looked stale and the whole
book stopped trading for the 90 days the run-time split lookup reaches back.
With the runner's entry price and entry date the decision does not depend on
the price at all.
"""

from datetime import date

import pytest

from quant.data.corporate_actions import (
    StockSplit,
    UnresolvedCorporateActionError,
    assert_corporate_actions_reconciled,
    classify_split_positions,
    presplit_threshold,
)

AS_OF = date(2026, 11, 20)
SPLIT_DAY = date(2026, 10, 1)
ENTRY = 100.0       # the runner's entry price, on the pre-split scale
QTY = 10.0          # shares held before the split


def _split(ratio):
    return {"XYZ": StockSplit("XYZ", ratio, SPLIT_DAY, SPLIT_DAY)}


def _adjusted(ratio, move):
    """The broker applied the split: units times the ratio, average entry
    divided by it; ``move`` is the price change since entry."""
    return [{
        "symbol": "XYZ", "qty": QTY * ratio,
        "avg_entry_price": ENTRY / ratio, "current_price": ENTRY / ratio * move,
    }]


def _stale(ratio):
    """The broker left the old units while the quote moved to the new scale."""
    return [{"symbol": "XYZ", "qty": QTY, "avg_entry_price": ENTRY, "current_price": ENTRY / ratio}]


def _records(**extra):
    return {"entry_prices": {"XYZ": ENTRY}, "entry_dates": {"XYZ": "2026-09-01"}, **extra}


RATIOS = [2.0, 1.5, 3.0, 4.0]


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("move", [1.2, 0.7])
def test_adjusted_position_with_a_record_passes_at_a_gain_and_at_a_loss(ratio, move):
    unresolved, rescale = classify_split_positions(
        _adjusted(ratio, move), as_of=AS_OF, extra_splits=_split(ratio), **_records(),
    )
    assert unresolved == []
    # Its stop-loss base is still on the old scale: the runner must rescale it.
    assert list(rescale) == ["XYZ"] and rescale["XYZ"].ratio == ratio
    assert assert_corporate_actions_reconciled(
        _adjusted(ratio, move), as_of=AS_OF, extra_splits=_split(ratio), **_records(),
    ) == rescale


@pytest.mark.parametrize("ratio", RATIOS)
def test_stale_position_with_a_record_fails_closed(ratio):
    with pytest.raises(UnresolvedCorporateActionError, match="XYZ"):
        assert_corporate_actions_reconciled(
            _stale(ratio), as_of=AS_OF, extra_splits=_split(ratio), **_records(),
        )


@pytest.mark.parametrize("ratio", RATIOS)
def test_position_opened_after_the_split_passes_without_a_rescale(ratio):
    bought_after = [{"symbol": "XYZ", "qty": 20.0, "avg_entry_price": 50.0, "current_price": 35.0}]
    records = {"entry_prices": {"XYZ": 50.0}, "entry_dates": {"XYZ": "2026-10-05"}}
    unresolved, rescale = classify_split_positions(
        bought_after, as_of=AS_OF, extra_splits=_split(ratio), **records,
    )
    assert unresolved == [] and rescale == {}


@pytest.mark.parametrize("ratio", RATIOS)
def test_a_rescale_already_applied_is_not_asked_for_again(ratio):
    # After the runner divided its entry by the ratio and recorded the split.
    records = {
        "entry_prices": {"XYZ": ENTRY / ratio},
        "entry_dates": {"XYZ": "2026-09-01"},
        "reconciled": {"XYZ": SPLIT_DAY.isoformat()},
    }
    unresolved, rescale = classify_split_positions(
        _adjusted(ratio, 0.7), as_of=AS_OF, extra_splits=_split(ratio), **records,
    )
    assert unresolved == [] and rescale == {}


@pytest.mark.parametrize("ratio", RATIOS)
def test_without_a_record_the_price_scale_test_uses_sqrt_ratio(ratio):
    split = _split(ratio)
    assert presplit_threshold(ratio) == pytest.approx(ratio ** 0.5)
    # An adjusted position at a gain passes, a stale one fails closed.
    assert classify_split_positions(_adjusted(ratio, 1.2), as_of=AS_OF, extra_splits=split) == ([], {})
    with pytest.raises(UnresolvedCorporateActionError):
        assert_corporate_actions_reconciled(_stale(ratio), as_of=AS_OF, extra_splits=split)
    # At a 30% loss the price scale cannot tell for the smallest ratios
    # (1/0.7 = 1.43 above sqrt(2) = 1.41): without a record it fails closed.
    unresolved, _ = classify_split_positions(_adjusted(ratio, 0.7), as_of=AS_OF, extra_splits=split)
    assert bool(unresolved) == (ratio < 2.1)


def test_large_ratios_keep_the_ratio_over_three_threshold():
    assert presplit_threshold(25.0) == pytest.approx(25.0 / 3)
    assert presplit_threshold(10.0) == pytest.approx(10.0 / 3)
