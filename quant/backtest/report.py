"""Backtest reporting and analytics."""

import numpy as np
import pandas as pd


def monthly_returns_table(equity_curve: pd.Series) -> pd.DataFrame:
    """Create a monthly returns pivot table (rows=years, cols=months)."""
    monthly = equity_curve.resample("ME").last().pct_change().dropna()
    table = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = table.pivot_table(values="return", index="year", columns="month", aggfunc="sum")
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    pivot = pivot.rename(columns=month_names)

    # Add annual return column
    annual = equity_curve.resample("YE").last().pct_change().dropna()
    annual_by_year = pd.Series(annual.values, index=annual.index.year)
    pivot["Annual"] = annual_by_year.reindex(pivot.index)
    return pivot


def risk_report(returns: pd.Series) -> dict:
    """Compute extended risk statistics."""
    ret = returns.dropna()
    if len(ret) < 2:
        return {}

    return {
        "Daily Mean": ret.mean(),
        "Daily Std": ret.std(),
        "Skewness": ret.skew(),
        "Kurtosis": ret.kurtosis(),
        "Best Day": ret.max(),
        "Worst Day": ret.min(),
        "VaR 95%": ret.quantile(0.05),
        "CVaR 95%": ret[ret <= ret.quantile(0.05)].mean(),
        "Positive Days %": (ret > 0).mean(),
        "Avg Win / Avg Loss": abs(ret[ret > 0].mean() / ret[ret < 0].mean()) if (ret < 0).any() else np.inf,
    }
