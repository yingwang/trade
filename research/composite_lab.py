"""Does a composite of the signals that showed consistent IC earn anything
as a long-only book on the point-in-time S&P 500?

Combines, as cross-sectional z-scores among that day's members, the earnings
signals (consensus SUE, year-on-year EPS growth, distance to the next
announcement) with intermediate momentum and five-day reversal, then (1)
reports the composite's IC at several horizons and (2) runs a plain portfolio
simulation: every ``--every`` sessions hold the top ``--n`` names (a holding
stays while it is still inside the top ``--buffer``), equal-weighted, traded
at the next open, 15 bps a side, against the universe's equal weight and SPY.
No leverage, no vol targeting: the question is whether the ranking itself is
worth anything before the portfolio machinery is put on top.

    python research/composite_lab.py --universe sp500 --start 2007-01-01
"""
import argparse
import logging
import sys

import numpy as np
import pandas as pd

from quant.utils.config import load_config
from research.earnings_lab import build_earnings_panels, earnings_factors, fetch_earnings
from research.factor_lab import (
    PERIODS,
    fetch_universe,
    forward_returns,
    plain_momentum,
    short_term_reversal_factor,
    summarize_h,
    zscore,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
COST_BPS = 15.0


def member_z(frame: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Z-score across that day's members, winsorized at three sigma."""
    f = frame.where(mask)
    z = zscore(f).clip(-3, 3)
    return zscore(z)


def simulate(score: pd.DataFrame, opens: pd.DataFrame, closes: pd.DataFrame, mask: pd.DataFrame,
             n: int, buffer: int, every: int, cost_bps: float = COST_BPS) -> tuple[pd.Series, pd.Series]:
    """Equal-weight top-n book, rebalanced every ``every`` sessions at the next
    open; returns the daily return series of the book and its turnover series."""
    exe = opens.combine_first(closes)
    exe_ret = exe.pct_change(fill_method=None)          # open-to-open returns
    dates = score.index
    held: set = set()
    weights = pd.Series(0.0, index=score.columns)
    book_ret = pd.Series(0.0, index=dates)
    turnover = pd.Series(np.nan, index=dates)
    decision_days = set(dates[::every])
    pending = None
    for i, date in enumerate(dates):
        if i > 0:
            # Yesterday's weights earn today's open-to-open return (an order
            # placed on yesterday's signal fills at today's open; one session
            # of drift between the signal and the fill is the price of
            # honesty, the same as the backtest engine's next-open rule).
            if pending is not None:
                cost = (pending - weights).abs().sum() * cost_bps / 10_000.0
                weights = pending
                pending = None
            else:
                cost = 0.0
            r = (weights * exe_ret.loc[date].reindex(weights.index).fillna(0.0)).sum() - cost
            book_ret.loc[date] = r
            # Drift the weights with the returns.
            grown = weights * (1 + exe_ret.loc[date].reindex(weights.index).fillna(0.0))
            total = grown.sum()
            weights = grown / total if total > 0 else weights
        if date in decision_days:
            s = score.loc[date].where(mask.loc[date]).dropna()
            if len(s) < n:
                continue
            ranked = s.sort_values(ascending=False)
            top = list(ranked.index[:n])
            keep = [h for h in held if h in ranked.index[:buffer]]
            new = keep + [t for t in top if t not in keep]
            new = new[:n]
            held = set(new)
            target = pd.Series(0.0, index=score.columns)
            target.loc[new] = 1.0 / len(new)
            turnover.loc[date] = (target - weights).abs().sum() / 2.0
            pending = target
    return book_ret, turnover


def stats(r: pd.Series, rf: float = 0.04) -> dict:
    r = r.dropna()
    equity = (1 + r).cumprod()
    years = len(r) / 252.0
    excess = r - ((1 + rf) ** (1 / 252) - 1)
    return {
        "total": equity.iloc[-1] - 1,
        "cagr": equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "sharpe": excess.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
        "maxdd": (equity / equity.cummax() - 1).min(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["static", "sp500"], default="sp500")
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--buffer", type=int, default=40)
    ap.add_argument("--every", type=int, default=10)
    args = ap.parse_args()

    config = load_config("config.yaml")
    closes, opens, volumes, bench, symbols, mask, _ = fetch_universe(config, args.universe, args.start, args.end)
    px, op = closes[symbols], opens[symbols]
    returns = px.pct_change(fill_method=None)
    events = fetch_earnings(symbols)
    panels = build_earnings_panels(events, px.index, px)
    ef = earnings_factors(panels)

    parts = {
        "sue_consensus": ef["sue_consensus"],
        "eps_growth_yoy": ef["eps_growth_yoy"],
        "days_to_next": ef["days_to_next (soonest first)"],
        "mom_12_6": plain_momentum(px, 252, 126) if False else zscore(px.shift(126).pct_change(126)),
        "reversal_5": short_term_reversal_factor(returns, 5),
    }
    z = {k: member_z(v.reindex(index=px.index, columns=symbols), mask) for k, v in parts.items()}
    composites = {
        "earnings only (sue+growth+next)": (z["sue_consensus"].fillna(0) + z["eps_growth_yoy"].fillna(0) + z["days_to_next"].fillna(0)) / 3,
        "earnings + mom_12_6": (z["sue_consensus"].fillna(0) + z["eps_growth_yoy"].fillna(0) + z["days_to_next"].fillna(0) + z["mom_12_6"].fillna(0)) / 4,
        "earnings + mom + reversal": (z["sue_consensus"].fillna(0) + z["eps_growth_yoy"].fillna(0) + z["days_to_next"].fillna(0) + z["mom_12_6"].fillna(0) + z["reversal_5"].fillna(0)) / 5,
        "sue_consensus alone": z["sue_consensus"],
        "mom_12_6 alone": z["mom_12_6"],
    }
    for k in composites:
        composites[k] = composites[k].where(mask)

    pd.set_option("display.width", 250)
    rows = {}
    for h in (5, 10, 21):
        fwd = forward_returns(op, px, h).where(mask)
        for name, c in composites.items():
            r = summarize_h(c, fwd, px.index[::h], h, args.n)
            rows[(name, f"h{h}")] = r
    table = pd.DataFrame(rows).T
    print(f"\nComposite IC on the {args.universe} universe (ic, t on non-overlapping dates, top{args.n} annualized excess, IC 2021 on):\n")
    print(table.round(3).to_string())

    bench_ret = closes[bench].pct_change(fill_method=None)
    ew = returns.where(mask).mean(axis=1)
    print(f"\nSimulation: top {args.n} equal weight, rebalanced every {args.every} sessions (holdings kept while inside the top {args.buffer}), "
          f"next-open fills, {COST_BPS:.0f} bps a side, no leverage.\n")
    out = []
    for name, c in composites.items():
        book, turnover = simulate(c, op, px, mask, args.n, args.buffer, args.every)
        for label, (a, b) in {**PERIODS, "5y": ("2021-10-22", "2026-10-22"), "all": (str(px.index[0].date()), "2026-12-31")}.items():
            s = stats(book.loc[a:b])
            s_ew = stats(ew.loc[a:b])
            s_spy = stats(bench_ret.loc[a:b])
            out.append({"composite": name, "window": label, "book_cagr": s["cagr"], "book_sharpe": s["sharpe"], "book_maxdd": s["maxdd"],
                        "ew_cagr": s_ew["cagr"], "spy_cagr": s_spy["cagr"], "turnover_per_rebal": turnover.loc[a:b].mean()})
    res = pd.DataFrame(out).set_index(["composite", "window"])
    print(res.round(3).to_string())
    res.to_csv(f"research/out/composite_lab_{args.universe}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
