"""Factor quality analysis utilities.

Provides tools for evaluating alpha factor signals:
  - Information Coefficient (IC) and ICIR
  - Quantile return analysis
  - Factor turnover measurement
  - Factor decay (IC at various forward horizons)
  - Factor correlation and VIF analysis

These utilities require realized forward returns and are intended for
post-hoc analysis, not real-time signal generation.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# Information Coefficient
# ======================================================================

def compute_daily_ic(factor: pd.DataFrame, forward_returns: pd.DataFrame,
                     method: str = "spearman") -> pd.Series:
    """Compute daily cross-sectional IC (rank correlation) between
    factor scores and forward returns.

    Parameters
    ----------
    factor : DataFrame (dates x symbols)
        Factor z-scores.
    forward_returns : DataFrame (dates x symbols)
        Forward returns over the holding period (e.g., 21-day forward).
    method : str
        Correlation method: 'spearman' (rank IC, preferred) or 'pearson'.

    Returns
    -------
    Series of daily IC values, indexed by date.
    """
    common_dates = factor.index.intersection(forward_returns.index)
    ic_values = pd.Series(index=common_dates, dtype=float, name="IC")

    for date in common_dates:
        f = factor.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 10:
            ic_values[date] = np.nan
            continue
        if method == "spearman":
            ic_values[date] = f[common].rank().corr(r[common].rank())
        else:
            ic_values[date] = f[common].corr(r[common])

    return ic_values


def compute_icir(ic_series: pd.Series) -> float:
    """IC Information Ratio: mean(IC) / std(IC).

    A value > 0.4 is generally considered production-grade.
    """
    clean = ic_series.dropna()
    if len(clean) < 20 or clean.std() == 0:
        return 0.0
    return clean.mean() / clean.std()


def ic_summary(factor: pd.DataFrame, forward_returns: pd.DataFrame,
               method: str = "spearman") -> dict:
    """Compute IC summary statistics for a factor.

    Returns dict with keys: mean_ic, std_ic, icir, hit_rate, t_stat, n_obs.
    """
    ic = compute_daily_ic(factor, forward_returns, method)
    clean = ic.dropna()
    n = len(clean)
    if n < 20:
        return {"mean_ic": np.nan, "std_ic": np.nan, "icir": np.nan,
                "hit_rate": np.nan, "t_stat": np.nan, "n_obs": n}

    mean_ic = clean.mean()
    std_ic = clean.std()
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit_rate = (clean > 0).mean()
    t_stat = mean_ic / (std_ic / np.sqrt(n))

    return {
        "mean_ic": round(mean_ic, 4),
        "std_ic": round(std_ic, 4),
        "icir": round(icir, 4),
        "hit_rate": round(hit_rate, 4),
        "t_stat": round(t_stat, 2),
        "n_obs": n,
    }


# ======================================================================
# Quantile return analysis
# ======================================================================

def quantile_returns(factor: pd.DataFrame, forward_returns: pd.DataFrame,
                     n_quantiles: int = 5) -> pd.DataFrame:
    """Compute mean forward return by factor quantile.

    Parameters
    ----------
    factor : DataFrame (dates x symbols)
    forward_returns : DataFrame (dates x symbols)
    n_quantiles : int
        Number of quantiles (default 5 = quintiles).

    Returns
    -------
    DataFrame with columns = quantile labels (Q1 lowest to Q5 highest),
    index = dates, values = mean forward return of that quantile.
    """
    common_dates = factor.index.intersection(forward_returns.index)
    results = []

    for date in common_dates:
        f = factor.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < n_quantiles * 2:
            continue
        f_sorted = f[common]
        r_matched = r[common]
        try:
            quantile_labels = pd.qcut(f_sorted, n_quantiles,
                                       labels=[f"Q{i+1}" for i in range(n_quantiles)],
                                       duplicates="drop")
        except ValueError:
            continue
        row = r_matched.groupby(quantile_labels).mean()
        row.name = date
        results.append(row)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def long_short_return(quantile_df: pd.DataFrame) -> pd.Series:
    """Compute the long-short spread: Q5 (top) minus Q1 (bottom).

    This is the theoretical return of a dollar-neutral factor portfolio.
    """
    if quantile_df.empty:
        return pd.Series(dtype=float)
    top = quantile_df.columns[-1]
    bottom = quantile_df.columns[0]
    return (quantile_df[top] - quantile_df[bottom]).rename("long_short")


# ======================================================================
# Factor turnover
# ======================================================================

def factor_turnover(factor: pd.DataFrame) -> pd.Series:
    """Measure daily factor turnover as mean absolute rank change.

    High turnover implies higher transaction costs for a strategy
    that trades on the factor signal.

    Returns Series of daily turnover values.
    """
    ranks = factor.rank(axis=1, pct=True)
    rank_change = ranks.diff().abs()
    # Mean absolute rank change across stocks
    return rank_change.mean(axis=1).rename("turnover")


def portfolio_turnover(weights_history: dict[str, pd.Series]) -> pd.Series:
    """Compute turnover between consecutive rebalance dates.

    Parameters
    ----------
    weights_history : dict mapping date_str -> pd.Series of weights

    Returns
    -------
    Series of turnover values indexed by date.
    """
    dates = sorted(weights_history.keys())
    turnovers = {}
    for i in range(1, len(dates)):
        prev = weights_history[dates[i-1]]
        curr = weights_history[dates[i]]
        all_syms = prev.index.union(curr.index)
        prev_aligned = prev.reindex(all_syms, fill_value=0)
        curr_aligned = curr.reindex(all_syms, fill_value=0)
        turnovers[dates[i]] = (curr_aligned - prev_aligned).abs().sum() / 2
    return pd.Series(turnovers, name="portfolio_turnover")


# ======================================================================
# Factor decay analysis
# ======================================================================

def factor_decay(factor: pd.DataFrame, returns: pd.DataFrame,
                 horizons: list[int] = None) -> pd.DataFrame:
    """Compute IC at multiple forward return horizons to measure signal decay.

    Parameters
    ----------
    factor : DataFrame (dates x symbols)
    returns : DataFrame of daily simple returns (dates x symbols)
    horizons : list of forward return horizons in trading days

    Returns
    -------
    DataFrame with columns = horizon, rows = IC statistics.
    """
    if horizons is None:
        horizons = [1, 5, 10, 21, 42, 63]

    results = {}
    for h in horizons:
        fwd_ret = returns.rolling(h).sum().shift(-h)
        summary = ic_summary(factor, fwd_ret)
        results[h] = summary

    return pd.DataFrame(results).T.rename_axis("horizon_days")


# ======================================================================
# Factor correlation and multi-collinearity
# ======================================================================

def factor_correlation_matrix(factors: dict[str, pd.DataFrame],
                               method: str = "spearman") -> pd.DataFrame:
    """Compute average cross-sectional correlation between factor pairs.

    For each date, compute pairwise rank correlation between all factors.
    Return the time-series average correlation matrix.
    """
    names = list(factors.keys())
    n = len(names)

    # Sample dates (every 21st day to reduce computation)
    sample_dates = list(factors[names[0]].index[::21])
    corr_sum = pd.DataFrame(0.0, index=names, columns=names)
    count = 0

    for date in sample_dates:
        daily = pd.DataFrame()
        skip = False
        for name in names:
            if date not in factors[name].index:
                skip = True
                break
            daily[name] = factors[name].loc[date].dropna()
        if skip or len(daily) < 10:
            continue
        if method == "spearman":
            c = daily.rank().corr()
        else:
            c = daily.corr()
        corr_sum += c.reindex(index=names, columns=names).fillna(0)
        count += 1

    if count == 0:
        return pd.DataFrame(np.nan, index=names, columns=names)
    return (corr_sum / count).round(3)


def compute_vif(factors: dict[str, pd.DataFrame],
                sample_date: pd.Timestamp = None) -> pd.Series:
    """Compute Variance Inflation Factor for each factor on a given date.

    VIF_j = 1 / (1 - R^2_j), where R^2_j is from regressing factor j
    on all other factors. VIF > 5 suggests problematic collinearity.

    Parameters
    ----------
    factors : dict of DataFrames
    sample_date : date to compute VIF for (default: last available date)

    Returns
    -------
    Series of VIF values, indexed by factor name.
    """
    names = list(factors.keys())
    if sample_date is None:
        sample_date = factors[names[0]].index[-1]

    daily = pd.DataFrame()
    for name in names:
        if sample_date in factors[name].index:
            daily[name] = factors[name].loc[sample_date]
    daily = daily.dropna()

    if len(daily) < len(names) + 2:
        return pd.Series(np.nan, index=names, name="VIF")

    vif = {}
    for target in names:
        y = daily[target].values
        X = daily[[c for c in names if c != target]].values
        # Add intercept
        X = np.column_stack([np.ones(len(X)), X])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_hat = X @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif[target] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
        except np.linalg.LinAlgError:
            vif[target] = np.nan

    return pd.Series(vif, name="VIF").round(2)
