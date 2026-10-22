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


def active_cooldowns(
    sessions: pd.DatetimeIndex,
    date,
    stop_exits: dict | None,
    cooldown_sessions: int,
) -> dict:
    """The stop exits still inside a re-entry cooldown at ``date``.

    ``stop_exits`` maps a symbol to the day it was sold on a stop.  A name is
    blocked from that session through the ``cooldown_sessions`` sessions
    after it, counted on ``sessions`` (the price index), so a holiday week
    counts the same on the backtest and the live path.  A record dated after
    the last session in the index (the live runner sells during a session
    whose bar is not in yet) counts as that last session.  What comes back is
    also what callers keep, so a record never outlives its window.
    """
    if not stop_exits or cooldown_sessions <= 0:
        return {}
    idx = pd.DatetimeIndex(sessions)
    today = int(idx.searchsorted(pd.Timestamp(date), side="right")) - 1
    if today < 0:
        return {}
    active: dict = {}
    for symbol, stopped in stop_exits.items():
        try:
            at = int(idx.searchsorted(pd.Timestamp(stopped), side="left"))
        except (TypeError, ValueError):
            continue
        if today - min(at, today) <= int(cooldown_sessions):
            active[symbol] = stopped
    return active

