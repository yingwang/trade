"""The recorded pre-split sales behind the dashboards' split correction
(review 2026-11-21, F5): the BKNG credit must not depend on the 2026-04-28
order staying among the newest 500 closed orders Alpaca returns."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from site_common import (
    _split_history_adjustments,
    _split_sold_credit,
    _validated_presplit_sales,
)

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = yaml.safe_load((ROOT / "config.yaml").read_text())["dashboard"]


def _order(side, qty, price, filled_at):
    return SimpleNamespace(
        symbol="BKNG", side=side, filled_qty=str(qty), filled_avg_price=str(price),
        filled_at=filled_at, submitted_at=None,
    )


SALE = _order("sell", 3, 175.4, "2026-04-28T15:48:10-04:00")
LATER_BUY = _order("buy", 4, 190.0, "2026-06-02T11:00:00-04:00")
LATER_SELL = _order("sell", 4, 200.0, "2026-07-01T11:00:00-04:00")


@pytest.mark.parametrize("book", ["multifactor", "lgbm"])
def test_credit_is_the_same_with_and_without_the_order(book):
    ledger = DASHBOARD["split_presplit_sales"][book]
    with_order = _split_sold_credit([SALE, LATER_BUY, LATER_SELL], presplit_sales=ledger)
    without = _split_sold_credit([LATER_BUY, LATER_SELL], presplit_sales=ledger)
    assert with_order == pytest.approx(12_628.80)
    assert without == pytest.approx(with_order)
    # Without the record the credit lived only as long as the order did.
    assert _split_sold_credit([SALE]) == pytest.approx(12_628.80)
    assert _split_sold_credit([]) == 0.0


@pytest.mark.parametrize("book", ["multifactor", "lgbm"])
def test_history_adjustment_is_the_same_with_and_without_the_order(book, monkeypatch):
    prices = pd.DataFrame(
        {"Close": [176.0, 175.4, 177.0, 181.0]},
        index=pd.to_datetime(["2026-04-27", "2026-04-28", "2026-04-29", "2026-05-01"]),
    )
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: prices)
    history = [{"date": d, "equity": 1_000_000.0}
               for d in ("2026-04-27", "2026-04-28", "2026-04-29", "2026-05-01")]
    ledger = DASHBOARD["split_presplit_sales"][book]
    with_order = _split_history_adjustments(history, [], [SALE], presplit_sales=ledger)
    without = _split_history_adjustments(history, [], [], presplit_sales=ledger)
    assert with_order == pytest.approx(without)
    # Before the sale: the 3 stale shares are marked at the post-split price;
    # from the sale on: the missing proceeds.
    assert without["2026-04-27"] == pytest.approx(24 * 3 * 176.0)
    assert without["2026-04-28"] == pytest.approx(12_628.80)
    assert without["2026-05-01"] == pytest.approx(12_628.80)
    assert _split_history_adjustments(history, [], [SALE]) == pytest.approx(with_order)


def test_orders_after_the_record_are_still_matched():
    ledger = {"BKNG": [{"date": "2026-04-28", "quantity": 3, "price": 175.4}]}
    stale_sell = _order("sell", 2, 200.0, "2026-07-01T11:00:00-04:00")  # no buy to cover it
    assert _split_sold_credit([stale_sell], presplit_sales=ledger) == pytest.approx(
        12_628.80 + 24 * 2 * 200.0
    )
    # A recorded holding of post-split shares covers a later sale of them.
    seeded = {"BKNG": [{"date": "2026-04-28", "quantity": 3, "price": 175.4,
                        "post_split_shares_held": 2}]}
    assert _split_sold_credit([stale_sell], presplit_sales=seeded) == pytest.approx(12_628.80)


@pytest.mark.parametrize("bad, message", [
    ({"XYZ": [{"date": "2026-04-28", "quantity": 3, "price": 1.0}]}, "Unknown"),
    ({"BKNG": []}, "non-empty list"),
    ({"BKNG": [{"date": "28/04/2026", "quantity": 3, "price": 1.0}]}, "YYYY-MM-DD"),
    ({"BKNG": [{"date": "2026-04-01", "quantity": 3, "price": 1.0}]}, "precedes the split"),
    ({"BKNG": [{"date": "2026-04-28", "quantity": 0, "price": 1.0}]}, "positive"),
])
def test_bad_records_are_rejected(bad, message):
    with pytest.raises(ValueError, match=message):
        _validated_presplit_sales(bad)
