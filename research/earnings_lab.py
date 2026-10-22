"""Earnings-announcement signals from Yahoo's free earnings history.

``yfinance.Ticker.get_earnings_dates(limit=100)`` returns about 25 years of
announcement timestamps with the consensus EPS estimate, the reported EPS and
the surprise.  That is point-in-time by construction (each row is dated by
its announcement), so it can carry the signals the price factors cannot:
standardized unexpected earnings and post-earnings-announcement drift,
year-on-year earnings growth, and distance to the next announcement.

Every signal is known at the close of the announcement date and traded at the
next open, one session late for pre-market announcements, which is the
conservative reading for a daily-close pipeline.

    python research/earnings_lab.py --universe static --horizons 5,21,63
"""
import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd

from quant.utils.config import load_config
from research.factor_lab import PERIODS, fetch_universe, forward_returns, summarize, summarize_h

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def fetch_earnings(symbols: list[str], limit: int = 100) -> dict[str, pd.DataFrame]:
    """One frame per symbol: index = announcement date (New York calendar
    day), columns estimate, actual, surprise_pct, after_close (bool)."""
    import yfinance as yf

    out = {}
    t0 = time.time()
    for i, sym in enumerate(symbols):
        try:
            ed = yf.Ticker(sym).get_earnings_dates(limit=limit)
        except Exception as exc:
            logging.warning("%s: earnings dates failed: %s", sym, exc)
            continue
        if ed is None or ed.empty:
            continue
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
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        out[sym] = frame
        if (i + 1) % 100 == 0:
            print(f"  earnings dates: {i+1}/{len(symbols)} in {time.time()-t0:.0f}s")
    print(f"earnings dates for {len(out)}/{len(symbols)} symbols in {time.time()-t0:.0f}s")
    return out


def _as_of(values: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Step function: the latest value whose date is on or before each date."""
    values = values.dropna().sort_index()
    if values.empty:
        return pd.Series(np.nan, index=dates)
    pos = values.index.searchsorted(dates, side="right") - 1
    out = pd.Series(values.values[np.clip(pos, 0, len(values) - 1)], index=dates, dtype=float)
    out[pos < 0] = np.nan
    return out


def build_earnings_panels(events: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, closes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Daily panels (dates x symbols) of announcement-based signals, each
    signed so that higher is expected to be better."""
    cols = list(closes.columns)
    panels = {k: pd.DataFrame(np.nan, index=dates, columns=cols) for k in (
        "surprise_pct", "sue_consensus", "surprise_over_price", "sue_seasonal", "eps_growth_yoy",
        "days_since", "days_to_next", "n_beats_4",
    )}
    for sym in cols:
        ev = events.get(sym)
        if ev is None or ev.empty:
            continue
        reported = ev.dropna(subset=["actual"])
        if reported.empty:
            continue
        err = reported["actual"] - reported["estimate"]
        # A run of exact hits makes the error std vanish; the ratio is
        # clipped so one such name cannot sit at infinity (ranks are unaffected).
        err_std = err.rolling(8, min_periods=6).std().shift(1)
        sue_c = (err / err_std.replace(0, np.nan)).clip(-10, 10)
        seasonal = reported["actual"] - reported["actual"].shift(4)
        sue_s = (seasonal / seasonal.rolling(8, min_periods=6).std().shift(1).replace(0, np.nan)).clip(-10, 10)
        growth = seasonal / reported["actual"].shift(4).abs().replace(0, np.nan)
        beats = (err > 0).astype(float).rolling(4, min_periods=4).sum()
        price_at = closes[sym].reindex(reported.index, method="ffill")
        over_price = err / price_at.replace(0, np.nan)

        panels["surprise_pct"][sym] = _as_of(reported["surprise_pct"], dates)
        panels["sue_consensus"][sym] = _as_of(sue_c, dates)
        panels["surprise_over_price"][sym] = _as_of(over_price, dates)
        panels["sue_seasonal"][sym] = _as_of(sue_s, dates)
        panels["eps_growth_yoy"][sym] = _as_of(growth, dates)
        panels["n_beats_4"][sym] = _as_of(beats, dates)
        last = _as_of(pd.Series(reported.index.values.astype("datetime64[ns]").astype("int64") / 86400e9, index=reported.index), dates)
        panels["days_since"][sym] = dates.values.astype("datetime64[ns]").astype("int64") / 86400e9 - last.values
        # Distance to the next scheduled announcement (future rows included):
        # the earnings announcement premium accrues to names about to report.
        all_dates = pd.DatetimeIndex(ev.index)
        nxt = all_dates.values[np.clip(all_dates.searchsorted(dates, side="right"), 0, len(all_dates) - 1)]
        days_to = (nxt - dates.values).astype("timedelta64[D]").astype(float)
        days_to[all_dates.searchsorted(dates, side="right") >= len(all_dates)] = np.nan
        panels["days_to_next"][sym] = days_to
    return panels


def earnings_factors(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Signed factor frames.  ``recent`` variants keep only names 1 to 45
    sessions past their announcement, the window where drift is documented."""
    since = panels["days_since"]
    recent = (since >= 0) & (since <= 63)
    factors = {
        "surprise_pct": panels["surprise_pct"],
        "sue_consensus": panels["sue_consensus"],
        "sue_consensus (recent 63d)": panels["sue_consensus"].where(recent),
        "surprise_over_price": panels["surprise_over_price"],
        "surprise_over_price (recent 63d)": panels["surprise_over_price"].where(recent),
        "sue_seasonal": panels["sue_seasonal"],
        "eps_growth_yoy": panels["eps_growth_yoy"],
        "n_beats_4": panels["n_beats_4"],
        "days_since (fresh first)": -since,
        "days_to_next (soonest first)": -panels["days_to_next"],
        "announcing within 10d": (panels["days_to_next"] <= 10).astype(float).where(panels["days_to_next"].notna()),
    }
    return factors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["static", "sp500"], default="static")
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--horizons", default="5,21,63")
    ap.add_argument("--top", default="12,25")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]
    top_ns = tuple(int(x) for x in args.top.split(","))

    config = load_config("config.yaml")
    closes, opens, volumes, bench, symbols, mask, sector_map = fetch_universe(config, args.universe, args.start, args.end)
    px = closes[symbols]
    op = opens[symbols]
    events = fetch_earnings(symbols)
    years = pd.Series({s: int(e.index.min().year) for s, e in events.items() if len(e)})
    print("first announcement year across symbols:", years.value_counts().sort_index().to_dict())

    panels = build_earnings_panels(events, px.index, px)
    coverage = panels["sue_consensus"].where(mask).notna().sum(axis=1) / mask.sum(axis=1)
    print("share of members with a consensus SUE, by year:",
          {str(y): round(float(v), 2) for y, v in coverage.groupby(coverage.index.year).mean().items() if y % 2 == 0})

    factors = earnings_factors(panels)
    for name in ("sue_consensus", "sue_seasonal"):
        f = factors[name].where(mask)
        print(f"{name}: mean {f.stack().mean():.2f} std {f.stack().std():.2f} share>0 {(f.stack() > 0).mean():.2f}")

    blocks = {}
    for h in horizons:
        fwd = forward_returns(op, px, h).where(mask)
        dates_h = px.index[::h]
        rows = {}
        for name, f in factors.items():
            f = f.reindex(index=px.index, columns=symbols).where(mask)
            if h == 21:
                r = summarize(f, fwd, dates_h, top_ns)
                rows[name] = {"ic": r["ic"], "t": r["t"], "hit": r["hit"], **{f"top{n}": r[f"top{n}_ann"] for n in top_ns},
                              **{f"ic_{k}": r[f"ic_{k}"] for k in PERIODS}}
            else:
                rows[name] = summarize_h(f, fwd, dates_h, h, top_ns[0])
        blocks[f"h{h}"] = pd.DataFrame(rows).T
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for h, block in blocks.items():
        print(f"\n== horizon {h}, universe={args.universe} (ic, t on non-overlapping dates, topN = annualized excess of the top N over the universe mean)\n")
        print(block.round(3).sort_values("t", ascending=False).to_string())
        block.to_csv(f"research/out/earnings_lab_{args.universe}_{h}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
