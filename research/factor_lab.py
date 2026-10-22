"""Candidate alpha factors on the universe, judged by the project's own IC tools.

For every factor, existing and candidate, this prints the 21-day forward IC
(rank correlation of the close-T score with the next-open-to-open return over
the following 21 sessions), its t-statistic on a 21-day-spaced subsample so
overlapping windows do not inflate it, the hit rate, the annualized return of
the top-12 names over the universe mean (what a long-only top-12 book cares
about), the raw and the sector-neutral version side by side, three sub-periods
for stability, and the correlation of each candidate with the live composite.

Usage (on Actions, through research.yml):

    python research/factor_lab.py --start 2008-01-01
"""
import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd

from quant.data.market_data import MarketData
from quant.signals.factors import (
    SignalGenerator,
    high_proximity_factor,
    mean_reversion_factor,
    momentum_factor,
    neutralize_by_sector,
    short_term_reversal_factor,
    trend_factor,
    trend_persistence_factor,
    volatility_contraction_factor,
    volatility_factor,
)
from quant.utils.config import load_config, static_sector_map

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
HORIZON = 21
TOP_N = 12


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)


# ----------------------------------------------------------------------
# Candidate factors (all: higher score = expected to do better)
# ----------------------------------------------------------------------

def rolling_beta(returns: pd.DataFrame, market: pd.Series, window: int) -> pd.DataFrame:
    cov = returns.rolling(window).cov(market)
    var = market.rolling(window).var()
    return cov.div(var, axis=0)


def residual_returns(returns: pd.DataFrame, market: pd.Series, window: int = 252) -> pd.DataFrame:
    """Daily returns net of the trailing-window market beta (Blitz, Huij & Martens)."""
    beta = rolling_beta(returns, market, window)
    alpha = returns.rolling(window).mean() - beta.mul(market.rolling(window).mean(), axis=0)
    return returns - beta.mul(market, axis=0) - alpha


def residual_momentum_factor(resid: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    window = resid.shift(skip).rolling(lookback - skip)
    return zscore(window.sum() / window.std().replace(0, np.nan))


def plain_momentum(prices: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    return zscore(prices.shift(skip).pct_change(lookback - skip))


def sharpe_momentum_factor(returns: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    window = returns.shift(skip).rolling(lookback - skip)
    return zscore(window.mean() / window.std().replace(0, np.nan))


def momentum_consistency_factor(returns: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Frog-in-the-pan (Da, Gurun & Warachka): momentum that arrived in many
    small steps continues more than momentum from a few jumps.  Score is the
    past return times the share of up days minus down days, signed so that
    continuous winners rank first."""
    r = returns.shift(skip)
    pos = (r > 0).astype(float).rolling(lookback - skip).mean()
    neg = (r < 0).astype(float).rolling(lookback - skip).mean()
    pret = (1 + r).rolling(lookback - skip).apply(np.prod, raw=True) - 1
    info_discreteness = np.sign(pret) * (neg - pos)
    # Low discreteness (continuous) is good for winners, bad for losers.
    return zscore(-info_discreteness * np.sign(pret) * pret.abs().rank(axis=1, pct=True))


def idiosyncratic_vol_factor(resid: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    return -zscore(resid.rolling(window).std())


def max_return_factor(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    return -zscore(returns.rolling(window).max())


def skewness_factor(returns: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    return -zscore(returns.rolling(window).skew())


def beta_factor(returns: pd.DataFrame, market: pd.Series, window: int = 252) -> pd.DataFrame:
    return -zscore(rolling_beta(returns, market, window))


def overnight_momentum_factor(opens: pd.DataFrame, closes: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Lou, Polk & Skouras: overnight (close-to-open) returns persist."""
    overnight = np.log(opens / closes.shift(1))
    return zscore(overnight.rolling(window).sum())


def intraday_reversal_factor(opens: pd.DataFrame, closes: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """The intraday (open-to-close) leg reverses; signed so that recent intraday losers rank first."""
    intraday = np.log(closes / opens)
    return -zscore(intraday.rolling(window).sum())


def abnormal_volume_factor(volumes: pd.DataFrame, short: int = 5, long: int = 63) -> pd.DataFrame:
    """Gervais, Kaniel & Mingelgrin: unusually high recent volume precedes higher returns."""
    ratio = np.log(volumes.rolling(short).mean() / volumes.rolling(long).mean().replace(0, np.nan))
    return zscore(ratio)


def volume_trend_factor(volumes: pd.DataFrame, returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Return-signed volume: up-day volume share over the window."""
    up = (volumes * (returns > 0)).rolling(window).sum()
    total = volumes.rolling(window).sum().replace(0, np.nan)
    return zscore(up / total)


def amihud_factor(returns: pd.DataFrame, closes: pd.DataFrame, volumes: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    dollar = (closes * volumes).replace(0, np.nan)
    illiq = (returns.abs() / dollar).rolling(window).mean()
    return zscore(np.log(illiq.replace(0, np.nan)))


def turnover_factor(closes: pd.DataFrame, volumes: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Dollar turnover trend: today's 21-day dollar volume over the 252-day level, negative sign
    (rising turnover in a name tends to precede lower returns)."""
    dollar = closes * volumes
    ratio = np.log(dollar.rolling(21).mean() / dollar.rolling(252).mean().replace(0, np.nan))
    return -zscore(ratio)


def low_proximity_factor(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Distance above the 52-week low, inverse: names near their low rank first (contrarian)."""
    low = prices.rolling(window).min()
    return -zscore(prices / low)


def acceleration_factor(prices: pd.DataFrame) -> pd.DataFrame:
    """Three-month momentum minus the twelve-month trend: is the move speeding up?"""
    return zscore(prices.pct_change(63) - prices.shift(63).pct_change(189) / 3.0)


def sector_momentum_factor(prices: pd.DataFrame, sector_map: pd.Series) -> pd.DataFrame:
    """Each name gets its sector's average 6-1 momentum: a top-down tilt."""
    mom = prices.shift(21).pct_change(105)
    sectors = sector_map.reindex(mom.columns)
    out = pd.DataFrame(index=mom.index, columns=mom.columns, dtype=float)
    for sector, members in sectors.groupby(sectors).groups.items():
        cols = [c for c in members if c in mom.columns]
        out[cols] = np.repeat(mom[cols].mean(axis=1).values[:, None], len(cols), axis=1)
    return zscore(out)


def seasonality_factor(returns: pd.DataFrame, years: int = 5) -> pd.DataFrame:
    """Heston & Sadka: the same calendar month's return in past years."""
    monthly = (1 + returns).resample("ME").prod() - 1
    lagged = sum(monthly.shift(12 * k) for k in range(1, years + 1)) / years
    daily = lagged.reindex(returns.index, method="ffill")
    return zscore(daily)


# ----------------------------------------------------------------------
# Universes
# ----------------------------------------------------------------------

from quant.data.earnings import load_sp500_membership, membership_mask  # noqa: E402


def fetch_universe(config: dict, universe: str, start: str, end: str | None):
    """Closes, opens, volumes, benchmark series and a membership mask."""
    data = MarketData(config)
    bench = config["universe"]["benchmark"]
    if universe == "static":
        closes = data.fetch_prices(start=start, end=end)
        opens, volumes = data.last_open_prices_, data.last_volumes_
        symbols = [c for c in closes.columns if c != bench]
        mask = pd.DataFrame(True, index=closes.index, columns=symbols)
        return closes, opens, volumes, bench, symbols, mask, static_sector_map(config)

    snapshots = load_sp500_membership()
    since = snapshots.loc[pd.Timestamp(start) - pd.Timedelta(days=400):]
    members = sorted(c for c in since.columns if since[c].any())
    data.symbols = members
    closes = data.fetch_prices(start=start, end=end)
    opens, volumes = data.last_open_prices_, data.last_volumes_
    enough = closes.notna().sum() >= 252
    symbols = [c for c in closes.columns if c != bench and enough.get(c, False)]
    closes = closes[symbols + [bench]]
    opens = opens[symbols]
    volumes = volumes[symbols]
    mask = membership_mask(snapshots, closes.index, symbols)
    listed = closes[symbols].notna()
    full = membership_mask(snapshots, closes.index, list(snapshots.columns))
    print(f"S&P 500 point-in-time universe: {len(members)} tickers were members since {start}, "
          f"{len(symbols)} have a yfinance history; members with data on a given day: "
          + ", ".join(f"{y}: {int((mask & listed).loc[str(y)].sum(axis=1).mean()):.0f}/{int(full.loc[str(y)].sum(axis=1).mean()):.0f}"
                      for y in (2008, 2012, 2016, 2020, 2024) if str(y) in closes.index.year.astype(str)))
    return closes, opens, volumes, bench, symbols, mask & listed, None


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def forward_returns(opens: pd.DataFrame, closes: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return from the next session's open to the open ``horizon`` sessions after it."""
    exe = opens.combine_first(closes)
    return exe.shift(-(horizon + 1)) / exe.shift(-1) - 1.0


def daily_ic(factor: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    f = factor.rank(axis=1)
    r = fwd.rank(axis=1)
    both = f.notna() & r.notna()
    f = f.where(both)
    r = r.where(both)
    n = both.sum(axis=1)
    fm = f.sub(f.mean(axis=1), axis=0)
    rm = r.sub(r.mean(axis=1), axis=0)
    ic = (fm * rm).sum(axis=1) / np.sqrt((fm ** 2).sum(axis=1) * (rm ** 2).sum(axis=1))
    return ic.where(n >= 20)


def top_minus_universe(factor: pd.DataFrame, fwd: pd.DataFrame, top_n: int) -> pd.Series:
    rank = factor.rank(axis=1, ascending=False)
    top = fwd.where(rank <= top_n).mean(axis=1)
    return top - fwd.mean(axis=1)


def summarize(factor: pd.DataFrame, fwd: pd.DataFrame, dates: pd.DatetimeIndex, top_ns=(TOP_N,)) -> dict:
    ic = daily_ic(factor, fwd).dropna()
    sub = ic.reindex(dates).dropna()
    width = factor.notna().sum(axis=1).replace(0, np.nan)
    turnover = (factor.rank(axis=1).diff(HORIZON).abs().mean(axis=1) / width).mean()
    out = {
        "ic": ic.mean(),
        "icir": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
        "t": sub.mean() / sub.std() * np.sqrt(len(sub)) if len(sub) > 5 and sub.std() > 0 else np.nan,
        "hit": (ic > 0).mean(),
        "n": len(sub),
        "rank_turnover": turnover,
    }
    for top_n in top_ns:
        tmu = top_minus_universe(factor, fwd, top_n).reindex(dates).dropna()
        out[f"top{top_n}_ann"] = tmu.mean() * 252 / HORIZON
    for label, (a, b) in PERIODS.items():
        part = ic.loc[a:b]
        out[f"ic_{label}"] = part.mean() if len(part) > 50 else np.nan
    return out


def summarize_h(factor: pd.DataFrame, fwd: pd.DataFrame, dates: pd.DatetimeIndex, horizon: int, top_n: int) -> dict:
    ic = daily_ic(factor, fwd).dropna()
    sub = ic.reindex(dates).dropna()
    tmu = top_minus_universe(factor, fwd, top_n).reindex(dates).dropna()
    recent = ic.loc["2021-01-01":]
    return {
        "ic": ic.mean(),
        "t": sub.mean() / sub.std() * np.sqrt(len(sub)) if len(sub) > 5 and sub.std() > 0 else np.nan,
        f"top{top_n}": tmu.mean() * 252 / horizon,
        "ic_21on": recent.mean() if len(recent) > 50 else np.nan,
    }


PERIODS = {
    "10-15": ("2010-01-01", "2015-12-31"),
    "16-20": ("2016-01-01", "2020-12-31"),
    "21-26": ("2021-01-01", "2026-12-31"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--universe", choices=["static", "sp500"], default="static",
                    help="static: config.yaml's 100 names; sp500: point-in-time S&P 500 membership")
    ap.add_argument("--top", default="12,25", help="top-N sizes for the long-only excess columns")
    ap.add_argument("--horizons", default="", help="extra forward horizons in sessions, e.g. 5,10,63 (21 is always run)")
    args = ap.parse_args()
    top_ns = tuple(int(x) for x in args.top.split(","))
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    t0 = time.time()
    config = load_config("config.yaml")
    closes, opens, volumes, bench, symbols, mask, sector_map = fetch_universe(config, args.universe, args.start, args.end)
    px = closes[symbols]
    op = opens[symbols]
    vol = volumes[symbols].astype(float)
    returns = px.pct_change(fill_method=None)
    market = closes[bench].pct_change(fill_method=None)
    print(f"prices {px.shape} {px.index[0].date()} -> {px.index[-1].date()} fetched in {time.time()-t0:.0f}s; "
          f"members per day (mean) {mask.sum(axis=1).mean():.0f}")

    # Everything is judged among that day's members only: scores and forward
    # returns of non-members are blanked before ranking.
    fwd = forward_returns(op, px, HORIZON).where(mask)

    # Context: how the universe itself did against the benchmark, equal-weighted
    # and rebalanced every session, so a factor's top-N excess can be read
    # against the pool it picks from.
    ew = returns.where(mask).mean(axis=1)
    print("\nEqual-weighted universe vs benchmark, annualized (daily rebalanced, no costs):")
    for label, (a, b) in {**PERIODS, "5y": ("2021-10-22", "2026-10-22"), "3y": ("2023-10-22", "2026-10-22"), "1y": ("2025-10-22", "2026-10-22")}.items():
        e = ew.loc[a:b]; m = market.loc[a:b]
        if len(e) > 50:
            print(f"  {label}: universe EW {((1+e).prod()**(252/len(e))-1)*100:+.1f}%/yr   {bench} {((1+m).prod()**(252/len(m))-1)*100:+.1f}%/yr   "
                  f"names {int(mask.loc[a:b].any().sum())}")
    # Non-overlapping evaluation dates every 21 sessions.
    eval_dates = px.index[::HORIZON]

    resid = residual_returns(returns, market, 252)
    sig_cfg = config["signals"]
    factors = {
        # --- what the live book uses ---
        "momentum(42/126/252)": momentum_factor(px, sig_cfg["momentum_windows"]),
        "high_proximity": high_proximity_factor(px, 252),
        "short_term_reversal(5)": short_term_reversal_factor(returns, 5),
        "vol_contraction": volatility_contraction_factor(returns, 10, 63),
        "trend_persistence": trend_persistence_factor(px, returns, 21),
        # --- disabled or filter-only in the live book ---
        "low_volatility(63)": volatility_factor(returns, 63),
        "mean_reversion(20)": mean_reversion_factor(px, 20),
        "trend(50/200)": trend_factor(px, 50, 200),
        # --- momentum variants ---
        "mom_12_1": plain_momentum(px, 252, 21),
        "mom_6_1": plain_momentum(px, 126, 21),
        "mom_3_0": plain_momentum(px, 63, 0),
        "mom_12_6 (intermediate)": zscore(px.shift(126).pct_change(126)),
        "residual_mom_12_1": residual_momentum_factor(resid, 252, 21),
        "residual_mom_6_1": residual_momentum_factor(resid, 126, 21),
        "sharpe_mom_12_1": sharpe_momentum_factor(returns, 252, 21),
        "mom_consistency(frog)": momentum_consistency_factor(returns, 252, 21),
        "acceleration": acceleration_factor(px),
        **({"sector_mom_6_1": sector_momentum_factor(px, sector_map)} if sector_map is not None else {}),
        # --- risk-style ---
        "idio_vol(63) low": idiosyncratic_vol_factor(resid, 63),
        "max_ret(21) low": max_return_factor(returns, 21),
        "skew(126) low": skewness_factor(returns, 126),
        "beta(252) low": beta_factor(returns, market, 252),
        # --- open/close and volume (new data for the pipeline) ---
        "overnight_mom(63)": overnight_momentum_factor(op, px, 63),
        "overnight_mom(126)": overnight_momentum_factor(op, px, 126),
        "intraday_reversal(63)": intraday_reversal_factor(op, px, 63),
        "abnormal_volume(5/63)": abnormal_volume_factor(vol, 5, 63),
        "volume_trend(63)": volume_trend_factor(vol, returns, 63),
        "amihud_illiq(63)": amihud_factor(returns, px, vol, 63),
        "turnover_trend low": turnover_factor(px, vol),
        # --- contrarian / calendar ---
        "low_proximity(252)": low_proximity_factor(px, 252),
        "seasonality(5y)": seasonality_factor(returns, 5),
    }

    # The live composite, as SignalGenerator builds it (members only), raw and neutralized.
    gen = SignalGenerator(config)
    full_mask = mask.reindex(columns=closes.columns, fill_value=False)
    composite_raw = gen.generate(closes, closes.pct_change(fill_method=None).dropna(how="all"), pd.DataFrame(),
                                 eligibility_mask=full_mask, sector_map=None)
    composite_neutral = (
        gen.generate(closes, closes.pct_change(fill_method=None).dropna(how="all"), pd.DataFrame(),
                     eligibility_mask=full_mask, sector_map=sector_map)
        if sector_map is not None else None
    )
    reference = (composite_neutral if composite_neutral is not None else composite_raw).reindex(index=px.index, columns=symbols)

    rows = []
    for name, f in factors.items():
        f = f.reindex(index=px.index, columns=symbols).where(mask)
        raw = summarize(f, fwd, eval_dates, top_ns)
        row = {"factor": name, "corr_live": f.corrwith(reference, axis=1).mean(), **{f"raw_{k}": v for k, v in raw.items()}}
        if sector_map is not None:
            neu = summarize(neutralize_by_sector(f, sector_map), fwd, eval_dates, top_ns)
            row.update({f"neu_{k}": v for k, v in neu.items()})
        rows.append(row)
    composites = [("LIVE composite (raw)", composite_raw)]
    if composite_neutral is not None:
        composites.insert(0, ("LIVE composite (neutral)", composite_neutral))
    for name, comp in composites:
        c = comp.reindex(index=px.index, columns=symbols).where(mask)
        s = summarize(c, fwd, eval_dates, top_ns)
        rows.append({"factor": name, "corr_live": np.nan, **{f"raw_{k}": v for k, v in s.items()}})

    table = pd.DataFrame(rows).set_index("factor")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    top_cols = [f"raw_top{n}_ann" for n in top_ns]
    cols = ["raw_ic", "raw_t", "raw_hit", *top_cols, "raw_ic_10-15", "raw_ic_16-20", "raw_ic_21-26"]
    if sector_map is not None:
        cols += ["neu_ic", "neu_t", f"neu_top{top_ns[0]}_ann", "neu_ic_21-26"]
    cols += ["raw_rank_turnover", "corr_live"]
    print(f"\n21-day forward IC, next-open execution, {len(eval_dates)} non-overlapping evaluation dates, "
          f"universe={args.universe}; topN_ann = annualized return of the factor's top N over the universe mean"
          + ("; neu_* = after within-sector z-scoring by the static table" if sector_map is not None else "") + "\n")
    print(table[cols].round(3).sort_values("raw_t", ascending=False).to_string())

    # Correlations between the stronger candidates, to see what is really new.
    strong = table[table["raw_t"].abs() > 1.5].index.tolist()
    strong = [s for s in strong if s in factors]
    if len(strong) > 1:
        frames = {s: factors[s].reindex(index=px.index, columns=symbols).where(mask) for s in strong}
        stack = pd.concat({s: frames[s].loc["2016":].stack() for s in strong}, axis=1)
        print("\nCross-sectional correlation of the stronger factors (2016 on, pooled):\n")
        print(stack.corr().round(2).to_string())

    table.to_csv(f"research/out/factor_lab_{args.universe}.csv")

    # Other holding horizons: a factor can be dead at 21 sessions and alive at
    # 5 (reversal) or 63 (slow momentum).  Same universe mask, same names.
    if horizons:
        blocks = {}
        for h in horizons:
            fwd_h = forward_returns(op, px, h).where(mask)
            dates_h = px.index[::h]
            rows_h = {}
            for name, f in factors.items():
                f = f.reindex(index=px.index, columns=symbols).where(mask)
                r = summarize_h(f, fwd_h, dates_h, h, top_ns[0])
                rows_h[name] = r
            for name, comp in composites:
                c = comp.reindex(index=px.index, columns=symbols).where(mask)
                rows_h[name] = summarize_h(c, fwd_h, dates_h, h, top_ns[0])
            blocks[f"h{h}"] = pd.DataFrame(rows_h).T
        multi = pd.concat(blocks, axis=1)
        multi.index.name = "factor"
        print(f"\nOther horizons (sessions), same universe: ic, t on a non-overlapping subsample, top{top_ns[0]} annualized excess, IC 2021 on\n")
        print(multi.round(3).to_string())
        multi.to_csv(f"research/out/factor_lab_{args.universe}_horizons.csv")
    print(f"\ndone in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
