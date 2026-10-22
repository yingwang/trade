"""Backtest the earnings book (quant/earnings_strategy.py) on Actions.

Prints the engine's metrics, the yearly returns against SPY and the README
windows, with optional parameter overrides so variants can be compared
without editing config_earnings.yaml:

    python research/earnings_book.py --start 2010-01-01
    python research/earnings_book.py --override '{"earnings": {"positions": 15}, "portfolio": {"target_volatility": 0.3}}'
"""
import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd

from quant.earnings_strategy import EarningsStrategy
from quant.utils.config import load_config

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def deep_update(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def window_stats(equity: pd.Series, bench: pd.Series, start, end, rf=0.04) -> dict:
    e = equity.loc[start:end]
    b = bench.loc[start:end]
    if len(e) < 50:
        return {}
    r = e.pct_change().dropna()
    years = len(r) / 252.0
    excess = r - ((1 + rf) ** (1 / 252) - 1)
    dd = (e / e.cummax() - 1).min()
    return {
        "total": e.iloc[-1] / e.iloc[0] - 1,
        "cagr": (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1,
        "sharpe": excess.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
        "maxdd": dd,
        "spy_total": b.iloc[-1] / b.iloc[0] - 1 if len(b) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_earnings.yaml")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-10-22")
    ap.add_argument("--override", default="", help="JSON merged over the config")
    args = ap.parse_args()

    config = load_config(args.config)
    if args.override:
        deep_update(config, json.loads(args.override))
        print("overrides:", args.override)
    t0 = time.time()
    strategy = EarningsStrategy(config)
    result = strategy.run_backtest(start=args.start, end=args.end)
    print(f"\nbacktest ran in {time.time()-t0:.0f}s\n")
    print(result.summary())

    equity = result.equity_curve
    bench = result.benchmark_curve
    yearly = pd.DataFrame({
        "book": equity.resample("YE").last().pct_change(),
        "SPY": bench.resample("YE").last().pct_change(),
    })
    first = pd.Series({"book": equity.resample("YE").last().iloc[0] / equity.iloc[0] - 1,
                       "SPY": bench.resample("YE").last().iloc[0] / bench.iloc[0] - 1})
    yearly.iloc[0] = first
    yearly.index = yearly.index.year
    print("\nYearly returns:\n")
    print((yearly * 100).round(1).to_string())

    end = pd.Timestamp(args.end)
    rows = {}
    for label, years in (("5y", 5), ("3y", 3), ("1y", 1)):
        rows[label] = window_stats(equity, bench, end - pd.DateOffset(years=years), end)
    rows["all"] = window_stats(equity, bench, equity.index[0], end)
    print("\nWindows:\n")
    print(pd.DataFrame(rows).T.round(3).to_string())

    trades = pd.DataFrame(result.trades)
    if not trades.empty:
        by_type = trades.groupby("type").agg(n=("date", "count"), turnover=("turnover", "mean"), cost=("cost", "sum"))
        print("\nTrades:\n")
        print(by_type.round(4).to_string())
    equity.rename("equity").to_frame().join(bench.rename("spy")).to_csv("research/out/earnings_book_equity.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
