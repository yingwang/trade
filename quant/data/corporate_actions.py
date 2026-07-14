"""Known corporate actions that require fail-closed broker handling.

Alpaca paper accounts can occasionally leave a pre-split position in old share
units while market data has already moved to the post-split price scale.  That
state is unsafe for an automated rebalance: a perfectly ordinary target-dollar
calculation can turn into an order that is off by the split ratio.

The dashboard may annotate/correct such data for display, but the trading path
must never guess.  It raises until the paper account is repaired or reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping


@dataclass(frozen=True)
class StockSplit:
    symbol: str
    ratio: float
    effective_date: date
    first_adjusted_session: date


# Booking Holdings: 25-for-1 split effective 2026-04-02; split-adjusted
# trading began 2026-04-06.  Keep both dates because legal effectiveness and
# the first adjusted market bar are different facts.
KNOWN_STOCK_SPLITS: dict[str, StockSplit] = {
    "BKNG": StockSplit(
        symbol="BKNG",
        ratio=25.0,
        effective_date=date(2026, 4, 2),
        first_adjusted_session=date(2026, 4, 6),
    ),
}


class UnresolvedCorporateActionError(RuntimeError):
    """Raised when a broker position appears to use stale split units."""


def looks_presplit(
    avg_entry_price: float,
    current_price: float,
    ratio: float,
) -> bool:
    """Conservative price-scale check for an unreconciled split position.

    A genuine pre-split average entry is roughly ``ratio`` times the current
    post-split quote.  ``ratio / 3`` leaves a wide tolerance for market moves,
    while avoiding false positives for positions opened after the split.
    """

    if avg_entry_price <= 0 or current_price <= 0 or ratio <= 1:
        return False
    return avg_entry_price / current_price > ratio / 3.0


def unresolved_splits(
    positions: Iterable[Mapping[str, object]],
    *,
    as_of: date | None = None,
) -> list[StockSplit]:
    """Return known splits whose position data is still on the old scale."""

    today = as_of or date.today()
    unresolved: list[StockSplit] = []
    for raw in positions:
        symbol = str(raw.get("symbol", "")).upper()
        split = KNOWN_STOCK_SPLITS.get(symbol)
        if split is None or today < split.first_adjusted_session:
            continue
        try:
            qty = abs(float(raw.get("qty", 0) or 0))
            avg_entry = float(raw.get("avg_entry_price", 0) or 0)
            current = float(raw.get("current_price", 0) or 0)
        except (TypeError, ValueError):
            continue
        if qty > 0 and looks_presplit(avg_entry, current, split.ratio):
            unresolved.append(split)
    return unresolved


def assert_corporate_actions_reconciled(
    positions: Iterable[Mapping[str, object]],
    *,
    as_of: date | None = None,
) -> None:
    """Abort automated trading when a known split has stale broker units."""

    unresolved = unresolved_splits(positions, as_of=as_of)
    if not unresolved:
        return
    details = ", ".join(f"{s.symbol} {s.ratio:g}-for-1" for s in unresolved)
    raise UnresolvedCorporateActionError(
        "Unreconciled corporate action in paper account: "
        f"{details}. Repair or reset the affected paper position before "
        "running automated orders."
    )
