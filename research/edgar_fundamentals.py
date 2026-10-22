"""Point-in-time fundamental factors from EDGAR, judged by the factor lab's IC tools.

Builds the EDGAR fundamentals panel for the research universe (every number
becomes known on the date its filing hit the SEC), prints how much of the
universe the panel covers year by year, turns the panel into the value,
profitability, accrual, issuance, growth and earnings-surprise factors, and
evaluates each the way research/factor_lab.py evaluates the price factors:
21-day forward IC with a t-statistic on a non-overlapping subsample, the
long-only top-N excess over the universe, three sub-periods, the sector
neutral version where the universe has a sector table, and the other
holding horizons.

Share counts and per-share amounts in filings are stated in the share basis
of the filing date, so the script also pulls the split history and the
dividend-unadjusted closes from yfinance: market capitalisation is then the
same-day close times the share count restated to today's basis.

Usage (on Actions, through research.yml; EDGAR_USER_AGENT must be set):

    python research/edgar_fundamentals.py --universe sp500 --start 2008-01-01 --horizons 21,63
"""
import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

from quant.data.edgar import EdgarClient, build_fundamentals_panel, fundamental_factors, resolve_cik
from quant.signals.factors import neutralize_by_sector
from quant.utils.config import load_config
from research import factor_lab as lab

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Tickers whose CIK in the SEC table belongs to a holding company created
# after the start of the sample, so that the earlier registrant's facts
# carry the history: ExxonMobil Holdings took over XOM in 2025 (Exxon Mobil
# Corp is CIK 34088), Alphabet's own facts begin with 2013 comparatives
# (Google Inc. is CIK 1288776).
PREDECESSOR_CIKS = {"XOM": [34088], "GOOGL": [1288776], "GOOG": [1288776]}
COVERAGE_FIELDS = ["net_income_ttm", "equity", "shares", "sue"]


def fetch_splits_and_raw_closes(symbols, start, end=None):
    """Split ratios (dates x symbols, NaN where none) and dividend-unadjusted closes.

    One bulk download with the corporate actions; yfinance's close is split
    adjusted but not dividend adjusted, which is what a market capitalisation
    wants.  A failed download leaves both None and the caller falls back to
    the adjusted closes with no split restatement, which is still usable for
    companies that never split.
    """
    try:
        data = yf.download(list(symbols), start=start, end=end, auto_adjust=False, actions=True,
                           progress=False, threads=True)
    except Exception as exc:
        print(f"split/raw-close download failed ({exc}); market caps use adjusted closes and unrestated shares")
        return None, None
    if data is None or data.empty:
        return None, None
    if not isinstance(data.columns, pd.MultiIndex):
        data.columns = pd.MultiIndex.from_product([data.columns, [list(symbols)[0]]])
    fields = data.columns.get_level_values(0)
    splits = data["Stock Splits"] if "Stock Splits" in fields else None
    closes = data["Close"] if "Close" in fields else None
    if splits is not None:
        splits = splits.where(splits > 0)
        if getattr(splits.index, "tz", None) is not None:
            splits.index = splits.index.tz_convert(None)
    if closes is not None and getattr(closes.index, "tz", None) is not None:
        closes.index = closes.index.tz_convert(None)
    return splits, closes


def coverage_by_year(panel: dict, mask: pd.DataFrame) -> pd.DataFrame:
    """Share of that day's members with a value, averaged within each year."""
    members = mask.sum(axis=1).replace(0, np.nan)
    table = {}
    for field in COVERAGE_FIELDS:
        have = (panel[field].reindex(index=mask.index, columns=mask.columns).notna() & mask).sum(axis=1)
        table[field] = (have / members).groupby(mask.index.year).mean()
    return pd.DataFrame(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--universe", choices=["static", "sp500"], default="static",
                    help="static: config.yaml's 100 names; sp500: point-in-time S&P 500 membership")
    ap.add_argument("--top", default="12,25", help="top-N sizes for the long-only excess columns")
    ap.add_argument("--horizons", default="21,63", help="forward horizons in sessions; 21 is always run")
    args = ap.parse_args()
    top_ns = tuple(int(x) for x in args.top.split(","))
    horizons = [int(x) for x in args.horizons.split(",") if x.strip() and int(x) != lab.HORIZON]

    t0 = time.time()
    config = load_config("config.yaml")
    closes, opens, volumes, bench, symbols, mask, sector_map = lab.fetch_universe(config, args.universe, args.start, args.end)
    px = closes[symbols]
    op = opens[symbols]
    print(f"prices {px.shape} {px.index[0].date()} -> {px.index[-1].date()} fetched in {time.time()-t0:.0f}s; "
          f"members per day (mean) {mask.sum(axis=1).mean():.0f}")

    splits, raw_closes = fetch_splits_and_raw_closes(symbols, args.start, args.end)
    if splits is not None:
        n_splits = int((splits.notna() & (splits != 1)).sum().sum())
        print(f"split history: {n_splits} splits across {int((splits.notna() & (splits != 1)).any().sum())} names")

    client = EdgarClient()
    ticker_map = client.ticker_map()
    unmapped = []
    for s in symbols:
        try:
            resolve_cik(s, ticker_map)
        except KeyError:
            unmapped.append(s)
    print(f"SEC ticker table: {len(ticker_map)} entries; {len(symbols) - len(unmapped)}/{len(symbols)} universe names mapped"
          + (f"; unmapped: {' '.join(unmapped[:30])}{' ...' if len(unmapped) > 30 else ''}" if unmapped else ""))

    t1 = time.time()
    panel = build_fundamentals_panel(symbols, px.index, client, ticker_map, splits=splits, predecessors=PREDECESSOR_CIKS)
    print(f"EDGAR panel built in {time.time()-t1:.0f}s")

    coverage = coverage_by_year(panel, mask)
    print("\nCoverage: share of the day's members with a value, by year\n")
    print(coverage.round(2).to_string())
    coverage.to_csv(f"research/out/edgar_coverage_{args.universe}.csv")

    if raw_closes is not None:
        price_for_cap = raw_closes.reindex(index=px.index, columns=symbols)
        # Where the raw download lacks a name, the adjusted close is the
        # only price there is.
        price_for_cap = price_for_cap.where(price_for_cap.notna(), px)
    else:
        price_for_cap = px
    factors = fundamental_factors(panel, price_for_cap)

    # Same evaluation as the factor lab: members only, next-open execution.
    fwd = lab.forward_returns(op, px, lab.HORIZON).where(mask)
    eval_dates = px.index[::lab.HORIZON]
    rows = []
    frames = {}
    for name, f in factors.items():
        f = f.reindex(index=px.index, columns=symbols).where(mask)
        frames[name] = f
        raw = lab.summarize(f, fwd, eval_dates, top_ns)
        row = {"factor": name, "coverage": f.notna().sum(axis=1).div(mask.sum(axis=1).replace(0, np.nan)).mean(),
               **{f"raw_{k}": v for k, v in raw.items()}}
        if sector_map is not None:
            neu = lab.summarize(neutralize_by_sector(f, sector_map), fwd, eval_dates, top_ns)
            row.update({f"neu_{k}": v for k, v in neu.items()})
        rows.append(row)
    table = pd.DataFrame(rows).set_index("factor")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    top_cols = [f"raw_top{n}_ann" for n in top_ns]
    cols = ["coverage", "raw_ic", "raw_t", "raw_hit", *top_cols, "raw_ic_10-15", "raw_ic_16-20", "raw_ic_21-26"]
    if sector_map is not None:
        cols += ["neu_ic", "neu_t", f"neu_top{top_ns[0]}_ann", "neu_ic_21-26"]
    cols += ["raw_rank_turnover"]
    print(f"\n{lab.HORIZON}-day forward IC, next-open execution, {len(eval_dates)} non-overlapping evaluation dates, "
          f"universe={args.universe}; topN_ann = annualized return of the factor's top N over the universe mean; "
          f"coverage = share of members scored"
          + ("; neu_* = after within-sector z-scoring by the static table" if sector_map is not None else "") + "\n")
    print(table[cols].round(3).sort_values("raw_t", ascending=False).to_string())
    print("\nIC from 2021 on (raw):\n")
    print(table["raw_ic_21-26"].round(3).sort_values(ascending=False).to_string())
    table.to_csv(f"research/out/edgar_fundamentals_{args.universe}.csv")

    # How the fundamental factors relate to each other (pooled, 2016 on).
    stack = pd.concat({name: f.loc["2016":].stack() for name, f in frames.items()}, axis=1)
    print("\nCross-sectional correlation of the fundamental factors (2016 on, pooled):\n")
    print(stack.corr().round(2).to_string())

    if horizons:
        blocks = {}
        for h in horizons:
            fwd_h = lab.forward_returns(op, px, h).where(mask)
            dates_h = px.index[::h]
            blocks[f"h{h}"] = pd.DataFrame({name: lab.summarize_h(f, fwd_h, dates_h, h, top_ns[0])
                                            for name, f in frames.items()}).T
        multi = pd.concat(blocks, axis=1)
        multi.index.name = "factor"
        print(f"\nOther horizons (sessions), same universe: ic, t on a non-overlapping subsample, "
              f"top{top_ns[0]} annualized excess, IC 2021 on\n")
        print(multi.round(3).to_string())
        multi.to_csv(f"research/out/edgar_fundamentals_{args.universe}_horizons.csv")
    print(f"\ndone in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
