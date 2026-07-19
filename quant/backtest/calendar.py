"""Deterministic rebalance calendar helpers.

The calendar is anchored to a fixed business date instead of to the first row
of whichever backtest slice happens to be requested.  Consequently, 1-, 3-
and 5-year reports rebalance on the same overlapping sessions.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_REBALANCE_ANCHOR = "2000-01-03"


def fixed_rebalance_dates(
    sessions: pd.DatetimeIndex,
    frequency_days: int,
    *,
    anchor: str = DEFAULT_REBALANCE_ANCHOR,
    not_before: pd.Timestamp | str | None = None,
) -> list[pd.Timestamp]:
    """Map a fixed business-day schedule to available market sessions.

    A scheduled date that is an exchange holiday maps to the next available
    session.  Mapping uses the supplied session index, so weekends and actual
    gaps never become synthetic trading dates.
    """
    if frequency_days <= 0:
        raise ValueError("frequency_days must be positive")
    if len(sessions) == 0:
        return []

    idx = pd.DatetimeIndex(sessions).sort_values().unique()
    tz = idx.tz
    anchor_ts = pd.Timestamp(anchor)
    if tz is not None:
        anchor_ts = anchor_ts.tz_localize(tz)
    start = max(anchor_ts, idx[0])
    scheduled = pd.bdate_range(
        start=anchor_ts,
        end=idx[-1],
        freq=f"{frequency_days}B",
        tz=tz,
    )
    scheduled = scheduled[scheduled >= start]

    positions = idx.searchsorted(scheduled, side="left")
    mapped = []
    seen = set()
    minimum = pd.Timestamp(not_before) if not_before is not None else None
    if minimum is not None and tz is not None and minimum.tzinfo is None:
        minimum = minimum.tz_localize(tz)
    for pos in positions:
        if pos >= len(idx):
            continue
        session = idx[pos]
        if minimum is not None and session < minimum:
            continue
        if session not in seen:
            mapped.append(session)
            seen.add(session)
    return mapped
