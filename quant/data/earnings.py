"""Free point-in-time inputs for the earnings book.

Two sources, both free and both dated by when the information became public:

* Yahoo's earnings calendar through ``yfinance.Ticker.get_earnings_dates``:
  about a hundred announcements per name (25 years of quarters) with the
  consensus EPS estimate, the reported EPS and the surprise.  Each row is
  dated by its announcement, so a signal built from it is known at that
  close and not before.
* The S&P 500 membership history maintained in the public fja05680/sp500
  repository (one row per date the list changed), which gives a point-in-time
  universe instead of today's list.  Names that have since been delisted have
  no yfinance history and drop out, so the universe is still a survivors'
  universe; what it removes is the hindsight in picking today's members.
"""

from __future__ import annotations

import csv
import logging
import os
import time
import urllib.request

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SP500_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
DEFAULT_UNIVERSE_FILE = "data/universe/sp500_history.csv"


# ----------------------------------------------------------------------
# Universe
# ----------------------------------------------------------------------

def load_sp500_membership(path: str = DEFAULT_UNIVERSE_FILE, url: str = SP500_HISTORY_URL) -> pd.DataFrame:
    """Boolean frame (snapshot date x ticker) of S&P 500 membership.

    The file is ``date,tickers`` with the tickers as one comma-separated
    string; it is downloaded once when missing.  Tickers come back in
    yfinance spelling (``BRK.B`` becomes ``BRK-B``).
    """
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "yingwang-trade research"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as out:
            out.write(resp.read())
    snapshots: dict[pd.Timestamp, set[str]] = {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header[:2] != ["date", "tickers"]:
            raise ValueError(f"Unexpected membership file header: {header}")
        for row in reader:
            tickers = {t.strip().upper().replace(".", "-") for t in row[1].split(",") if t.strip()}
            snapshots[pd.Timestamp(row[0])] = tickers
    all_tickers = sorted(set().union(*snapshots.values()))
    frame = pd.DataFrame(False, index=sorted(snapshots), columns=all_tickers)
    for date, tickers in snapshots.items():
        frame.loc[date, list(tickers)] = True
    return frame


def membership_mask(snapshots: pd.DataFrame, dates: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """Membership on every date in ``dates``: the latest snapshot on or before it."""
    aligned = snapshots.reindex(columns=list(columns), fill_value=False)
    return aligned.reindex(pd.DatetimeIndex(dates), method="ffill").fillna(False).astype(bool)


def members_since(snapshots: pd.DataFrame, start, lead_days: int = 400) -> list[str]:
    """Every name that was a member at some point from ``lead_days`` before
    ``start``: the snapshots dated from then on, plus the one in force at that
    cutoff (the list may not have changed for months before it)."""
    cutoff = pd.Timestamp(start) - pd.Timedelta(days=lead_days)
    first = max(0, int(snapshots.index.searchsorted(cutoff, side="right")) - 1)
    since = snapshots.iloc[first:]
    return sorted(c for c in since.columns if since[c].any())


def current_members(snapshots: pd.DataFrame) -> list[str]:
    last = snapshots.iloc[-1]
    return sorted(last.index[last])


# ----------------------------------------------------------------------
# Announcements
# ----------------------------------------------------------------------

def fetch_earnings(symbols: list[str], limit: int = 100, pause: float = 0.0) -> dict[str, pd.DataFrame]:
    """One frame per symbol from Yahoo: index = announcement date (New York
    calendar day), columns estimate, actual, surprise_pct, after_close.

    A name whose request fails is left out and logged; the caller decides
    whether the coverage is good enough.
    """
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    t0 = time.time()
    failed: list[str] = []
    # Two passes: Yahoo drops a few requests in a run of hundreds, and a
    # name without its history would be ranked on momentum alone or left out,
    # which made otherwise identical backtests differ by about a point a year.
    for attempt in range(2):
        todo = list(symbols) if attempt == 0 else failed
        failed = []
        if attempt and todo:
            time.sleep(5.0)
            logger.info("earnings dates: retrying %d symbols", len(todo))
        for i, sym in enumerate(todo):
            try:
                ed = yf.Ticker(sym).get_earnings_dates(limit=limit)
            except Exception as exc:
                logger.warning("%s: earnings dates failed: %s", sym, exc)
                failed.append(sym)
                continue
            if ed is None or ed.empty:
                continue
            out[sym] = normalize_earnings_frame(ed)
            if pause:
                time.sleep(pause)
            if attempt == 0 and (i + 1) % 100 == 0:
                logger.info("earnings dates: %d/%d in %.0fs", i + 1, len(todo), time.time() - t0)
    logger.info("earnings dates for %d/%d symbols in %.0fs (%d still failed)",
                len(out), len(symbols), time.time() - t0, len(failed))
    return out


def normalize_earnings_frame(ed: pd.DataFrame) -> pd.DataFrame:
    """yfinance's frame (tz-aware index, 'EPS Estimate' / 'Reported EPS' /
    'Surprise(%)') to the plain shape the panels are built from."""
    ed = ed.rename(columns={"EPS Estimate": "estimate", "Reported EPS": "actual", "Surprise(%)": "surprise_pct"})
    idx = pd.DatetimeIndex(ed.index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York")
    frame = pd.DataFrame({
        "estimate": pd.to_numeric(ed.get("estimate"), errors="coerce").values,
        "actual": pd.to_numeric(ed.get("actual"), errors="coerce").values,
        "surprise_pct": pd.to_numeric(ed.get("surprise_pct"), errors="coerce").values,
        "after_close": (idx.hour >= 16).astype(bool),
    }, index=idx.tz_localize(None).normalize())
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def as_of(values: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Step function: the latest value whose date is on or before each date."""
    values = values.dropna().sort_index()
    if values.empty:
        return pd.Series(np.nan, index=dates)
    pos = values.index.searchsorted(dates, side="right") - 1
    out = pd.Series(values.values[np.clip(pos, 0, len(values) - 1)], index=dates, dtype=float)
    out[pos < 0] = np.nan
    return out


def announcement_panels(
    events: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    closes: pd.DataFrame,
    sue_lookback: int = 8,
    sue_clip: float = 10.0,
    next_known_within_days: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Daily panels (dates x symbols) built from the announcement history.

    Every panel holds, on each date, the value of the latest announcement on
    or before that date (an announcement after the close counts from that
    close; a pre-market one is therefore read a session late, which is the
    conservative reading for a close-to-close pipeline):

    * ``sue_consensus``: (reported - estimate) over the standard deviation of
      the previous ``sue_lookback`` such errors, clipped to ``sue_clip`` (a run
      of exact hits otherwise sends one name to infinity).
    * ``sue_seasonal``: the same with the seasonal random walk, reported EPS
      minus the EPS four quarters earlier.
    * ``surprise_pct``: Yahoo's own surprise.
    * ``eps_growth_yoy``: (reported - reported four quarters ago) / |the latter|.
    * ``n_beats_4``: beats in the last four announcements.
    * ``days_since``: sessions-agnostic calendar days since the last announcement.
    * ``days_to_next``: calendar days to the next announcement on Yahoo's
      calendar (future dates included), NaN past the last known one.  In a
      backtest that calendar is the realized one, so a date is "known" long
      before the company confirmed it; with ``next_known_within_days`` the
      realized date is used only once the announcement is that close (how
      far ahead dates are typically confirmed) and before that the value is
      the usual quarterly cadence, 91 days after the last announcement, the
      same rough expectation Yahoo's forward calendar shows before a date
      is confirmed (the live path only ever sees that calendar).
    """
    cols = list(closes.columns)
    names = ("surprise_pct", "sue_consensus", "sue_seasonal", "eps_growth_yoy", "n_beats_4", "days_since", "days_to_next")
    panels = {k: pd.DataFrame(np.nan, index=dates, columns=cols) for k in names}
    day = np.timedelta64(1, "D")
    for sym in cols:
        ev = events.get(sym)
        if ev is None or ev.empty:
            continue
        reported = ev.dropna(subset=["actual"])
        if reported.empty:
            continue
        err = reported["actual"] - reported["estimate"]
        err_std = err.rolling(sue_lookback, min_periods=6).std().shift(1)
        sue_c = (err / err_std.replace(0, np.nan)).clip(-sue_clip, sue_clip)
        seasonal = reported["actual"] - reported["actual"].shift(4)
        sue_s = (seasonal / seasonal.rolling(sue_lookback, min_periods=6).std().shift(1).replace(0, np.nan)).clip(-sue_clip, sue_clip)
        growth = seasonal / reported["actual"].shift(4).abs().replace(0, np.nan)
        beats = (err > 0).astype(float).rolling(4, min_periods=4).sum()

        panels["surprise_pct"][sym] = as_of(reported["surprise_pct"], dates)
        panels["sue_consensus"][sym] = as_of(sue_c, dates)
        panels["sue_seasonal"][sym] = as_of(sue_s, dates)
        panels["eps_growth_yoy"][sym] = as_of(growth, dates)
        panels["n_beats_4"][sym] = as_of(beats, dates)
        last = as_of(pd.Series(reported.index.values.astype("datetime64[D]").astype(float), index=reported.index), dates)
        panels["days_since"][sym] = dates.values.astype("datetime64[D]").astype(float) - last.values
        all_dates = pd.DatetimeIndex(ev.index)
        nxt_pos = all_dates.searchsorted(dates, side="right")
        nxt = all_dates.values[np.clip(nxt_pos, 0, len(all_dates) - 1)]
        days_to = ((nxt - dates.values) / day).astype(float)
        days_to[nxt_pos >= len(all_dates)] = np.nan
        if next_known_within_days is not None:
            expected = np.maximum(last.values + 91.0 - dates.values.astype("datetime64[D]").astype(float), 0.0)
            far = ~(days_to <= float(next_known_within_days))
            days_to = np.where(far, expected, days_to)
        panels["days_to_next"][sym] = days_to
    return panels
