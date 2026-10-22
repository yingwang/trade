"""What does plain beta earn, with and without leverage, over the README windows?

The books' backtests beat SPY by a wide margin, but they run 1.5x to 1.9x
gross on a hand-picked universe.  This prints, for SPY, QQQ, the equal-weighted
static universe and the equal-weighted point-in-time S&P 500 (survivors only),
the total return, CAGR, Sharpe (4% risk-free), max drawdown and the same at
1.5x and 1.8x daily-rebalanced leverage with 6% a year charged on the borrowed
part: the number a model has to beat before it has added anything.
"""
import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

from quant.utils.config import load_config
from research.factor_lab import fetch_universe


def levered(returns: pd.Series, lev: float, margin_rate: float = 0.06) -> pd.Series:
    borrow = max(0.0, lev - 1.0)
    return lev * returns - borrow * margin_rate / 252.0


def stats(returns: pd.Series, rf: float = 0.04) -> dict:
    r = returns.dropna()
    if len(r) < 50:
        return {}
    equity = (1 + r).cumprod()
    years = len(r) / 252.0
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1
    excess = r - ((1 + rf) ** (1 / 252) - 1)
    sharpe = excess.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    dd = (equity / equity.cummax() - 1).min()
    return {"total": total, "cagr": cagr, "sharpe": sharpe, "maxdd": dd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-10-22")
    args = ap.parse_args()
    end = pd.Timestamp(args.end)
    windows = {"5y": end - pd.DateOffset(years=5), "3y": end - pd.DateOffset(years=3), "1y": end - pd.DateOffset(years=1)}

    config = load_config("config.yaml")
    closes, opens, volumes, bench, symbols, mask, _ = fetch_universe(config, "static", "2007-01-01", args.end)
    static_ew = closes[symbols].pct_change(fill_method=None).mean(axis=1)
    spy = closes[bench].pct_change(fill_method=None)
    closes2, _, _, _, symbols2, mask2, _ = fetch_universe(config, "sp500", "2007-01-01", args.end)
    sp_ew = closes2[symbols2].pct_change(fill_method=None).where(mask2).mean(axis=1)
    extra = yf.download(["QQQ", "IWM", "MTUM", "QUAL", "SPMO"], start="2007-01-01", auto_adjust=True, progress=False)["Close"]
    series = {
        "SPY": spy,
        "QQQ (Nasdaq-100)": extra["QQQ"].pct_change(fill_method=None),
        "IWM (Russell 2000)": extra["IWM"].pct_change(fill_method=None),
        "MTUM (momentum ETF)": extra["MTUM"].pct_change(fill_method=None),
        "QUAL (quality ETF)": extra["QUAL"].pct_change(fill_method=None),
        "SPMO (S&P momentum ETF)": extra["SPMO"].pct_change(fill_method=None),
        "static 100 EW": static_ew,
        "S&P 500 PIT EW (survivors)": sp_ew,
    }
    rows = []
    for name, r in series.items():
        for lev in (1.0, 1.5, 1.8):
            for w, start in windows.items():
                s = stats(levered(r.loc[start:end], lev))
                if s:
                    rows.append({"series": name, "lev": lev, "window": w, **s})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for w in windows:
        sub = table[table["window"] == w].pivot(index="series", columns="lev", values=["total", "cagr", "sharpe", "maxdd"])
        print(f"\n== {w} window ({windows[w].date()} -> {end.date()}), columns = leverage; 6%/yr on the borrowed part\n")
        print((sub * 100).round(1).to_string() if False else sub.round(3).to_string())
    print("\nFor reference, README windows ending 2026-10-22: multi-factor 5y +151.8% (neutralized: +127.5%), LGBM +137.6%, trend +200.1%; SPY +82.4%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
