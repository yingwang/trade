"""Known corporate actions that require fail-closed broker handling.

Alpaca paper accounts can occasionally leave a pre-split position in old share
units while market data has already moved to the post-split price scale.  That
state is unsafe for an automated rebalance: a perfectly ordinary target-dollar
calculation can turn into an order that is off by the split ratio.

The dashboard may annotate/correct such data for display, but the trading path
must never guess.  It raises until the paper account is repaired or reset.
"""

from __future__ import annotations

import math
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


def presplit_threshold(ratio: float) -> float:
    """Average entry over current quote above which a position looks stale.

    A genuine pre-split average entry is roughly ``ratio`` times the current
    post-split quote.  For a large ratio ``ratio / 3`` leaves a wide tolerance
    for market moves.  Below 6 that threshold falls under the price an
    adjusted position can show after an ordinary move (a 2-for-1 split gave
    0.67, so every adjusted position looked stale), and the midpoint on a log
    scale, ``sqrt(ratio)``, is used instead.
    """

    return ratio / 3.0 if ratio >= 6 else math.sqrt(ratio)


def looks_presplit(
    avg_entry_price: float,
    current_price: float,
    ratio: float,
) -> bool:
    """Price-scale check for an unreconciled split position, used only when
    the runner has no record of its own for the position (see
    ``classify_split_positions``)."""

    if avg_entry_price <= 0 or current_price <= 0 or ratio <= 1:
        return False
    return avg_entry_price / current_price > presplit_threshold(ratio)


def _parse_day(value) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def classify_split_positions(
    positions: Iterable[Mapping[str, object]],
    *,
    as_of: date | None = None,
    extra_splits: Mapping[str, StockSplit] | None = None,
    entry_prices: Mapping[str, float] | None = None,
    entry_dates: Mapping[str, str] | None = None,
    reconciled: Mapping[str, str] | None = None,
) -> tuple[list[StockSplit], dict[str, StockSplit]]:
    """``(unresolved, rescale)`` for the held positions with a known split.

    With the runner's own record of a position the decision does not depend
    on the price: a position opened on or after the split's first adjusted
    session (``entry_dates``) is on the new scale, and one opened before it
    was adjusted by the broker exactly when the runner's entry price is more
    than ``sqrt(ratio)`` times the broker's average entry (adjusted units
    divide the broker's average by the ratio; the runner's price does not
    move).  Such a position is returned in ``rescale``: its stop-loss base is
    still on the old scale and must be divided by the ratio once, after which
    ``reconciled`` (symbol to the first adjusted session already applied)
    marks it done.  Without a record the price-scale test applies.
    """

    today = as_of or date.today()
    table: dict[str, StockSplit] = dict(KNOWN_STOCK_SPLITS)
    for symbol, split in (extra_splits or {}).items():
        table.setdefault(str(symbol).upper(), split)
    entry_prices = {str(k).upper(): v for k, v in (entry_prices or {}).items()}
    entry_dates = {str(k).upper(): v for k, v in (entry_dates or {}).items()}
    reconciled = {str(k).upper(): v for k, v in (reconciled or {}).items()}
    unresolved: list[StockSplit] = []
    rescale: dict[str, StockSplit] = {}
    for raw in positions:
        symbol = str(raw.get("symbol", "")).upper()
        split = table.get(symbol)
        if split is None or today < split.first_adjusted_session:
            continue
        try:
            qty = abs(float(raw.get("qty", 0) or 0))
        except (TypeError, ValueError):
            # A known split position whose units cannot even be parsed is not a
            # state in which automated quantity calculations are safe.
            unresolved.append(split)
            continue
        if not math.isfinite(qty):
            unresolved.append(split)
            continue
        if qty <= 0:
            continue
        try:
            avg_entry = float(raw.get("avg_entry_price", 0) or 0)
            current = float(raw.get("current_price", 0) or 0)
        except (TypeError, ValueError):
            unresolved.append(split)
            continue
        if (
            not math.isfinite(avg_entry)
            or not math.isfinite(current)
            or avg_entry <= 0
            or current <= 0
        ):
            unresolved.append(split)
            continue
        opened = _parse_day(entry_dates.get(symbol))
        if opened is not None and opened >= split.first_adjusted_session:
            continue
        if _parse_day(reconciled.get(symbol)) == split.first_adjusted_session:
            continue
        try:
            recorded = float(entry_prices.get(symbol) or 0)
        except (TypeError, ValueError):
            recorded = 0.0
        if math.isfinite(recorded) and recorded > 0:
            if recorded / avg_entry > math.sqrt(split.ratio):
                rescale[symbol] = split
            else:
                unresolved.append(split)
            continue
        if looks_presplit(avg_entry, current, split.ratio):
            unresolved.append(split)
    return unresolved, rescale


def unresolved_splits(
    positions: Iterable[Mapping[str, object]],
    *,
    as_of: date | None = None,
    extra_splits: Mapping[str, StockSplit] | None = None,
    entry_prices: Mapping[str, float] | None = None,
    entry_dates: Mapping[str, str] | None = None,
    reconciled: Mapping[str, str] | None = None,
) -> list[StockSplit]:
    """Return known splits whose position data is still on the old scale.

    ``extra_splits`` extends the static table with splits discovered at run
    time (see MarketData.fetch_recent_splits), so the guard covers every held
    symbol and not only the ones someone remembered to list here.  The
    runner's records refine the decision (see ``classify_split_positions``).
    """

    return classify_split_positions(
        positions, as_of=as_of, extra_splits=extra_splits,
        entry_prices=entry_prices, entry_dates=entry_dates, reconciled=reconciled,
    )[0]


def assert_corporate_actions_reconciled(
    positions: Iterable[Mapping[str, object]],
    *,
    as_of: date | None = None,
    extra_splits: Mapping[str, StockSplit] | None = None,
    entry_prices: Mapping[str, float] | None = None,
    entry_dates: Mapping[str, str] | None = None,
    reconciled: Mapping[str, str] | None = None,
) -> dict[str, StockSplit]:
    """Abort automated trading when a known split has stale broker units.

    Returns the positions the broker did adjust whose stop-loss base the
    runner still has to rescale (empty without the runner's records).
    """

    unresolved, rescale = classify_split_positions(
        positions, as_of=as_of, extra_splits=extra_splits,
        entry_prices=entry_prices, entry_dates=entry_dates, reconciled=reconciled,
    )
    if not unresolved:
        return rescale
    details = ", ".join(f"{s.symbol} {s.ratio:g}-for-1" for s in unresolved)
    raise UnresolvedCorporateActionError(
        "Unreconciled corporate action in paper account: "
        f"{details}. Repair or reset the affected paper position before "
        "running automated orders."
    )
