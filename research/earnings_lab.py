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

import pandas as pd

from quant.utils.config import load_config
from research.factor_lab import PERIODS, fetch_universe, forward_returns, summarize, summarize_h

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


from quant.data.earnings import announcement_panels as build_earnings_panels, fetch_earnings  # noqa: E402


def earnings_factors(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Signed factor frames.  ``recent`` variants keep only names 1 to 45
    sessions past their announcement, the window where drift is documented."""
    since = panels["days_since"]
    recent = (since >= 0) & (since <= 63)
    factors = {
        "surprise_pct": panels["surprise_pct"],
        "sue_consensus": panels["sue_consensus"],
        "sue_consensus (recent 63d)": panels["sue_consensus"].where(recent),
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
