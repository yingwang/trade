"""Alpha factor / signal generation for medium-term US equity strategies.

Factors implemented:
  - Cross-sectional momentum (1m, 3m, 6m, 12m)
  - Mean reversion (Bollinger z-score)
  - Trend following (SMA crossover)
  - Volatility (realized vol ranking)
  - Quality / Value (from fundamentals)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# Factor standardization utilities
# ======================================================================

def winsorize_zscore(df: pd.DataFrame, clip_val: float = 3.0) -> pd.DataFrame:
    """Winsorize to [-clip_val, clip_val] then re-zscore cross-sectionally.

    Applied per-row (per-date) to ensure each factor has controlled scale
    before composite weighting.
    """
    # Clip extremes
    clipped = df.clip(lower=-clip_val, upper=clip_val, axis=1)
    # Re-zscore after clipping
    mean = clipped.mean(axis=1)
    std = clipped.std(axis=1).replace(0, 1)
    return clipped.sub(mean, axis=0).div(std, axis=0)


def neutralize_by_sector(factor_df: pd.DataFrame,
                         sector_map: pd.Series) -> pd.DataFrame:
    """Industry-neutral z-score: z-score within each sector, then combine.

    Parameters
    ----------
    factor_df : DataFrame (dates x symbols)
        Raw factor scores.
    sector_map : Series (symbol -> sector string)
        Sector assignment for each symbol.

    Returns
    -------
    DataFrame of industry-neutralized z-scores (same shape as input).
    """
    result = factor_df.copy()
    symbols = factor_df.columns
    sectors = sector_map.reindex(symbols).dropna()
    if sectors.empty:
        return factor_df

    unique_sectors = sectors.unique()
    for sector in unique_sectors:
        mask = sectors[sectors == sector].index.tolist()
        mask = [s for s in mask if s in symbols]
        if len(mask) < 2:
            continue
        sector_data = factor_df[mask]
        mean = sector_data.mean(axis=1)
        std = sector_data.std(axis=1).replace(0, 1)
        result[mask] = sector_data.sub(mean, axis=0).div(std, axis=0)

    return result


# ======================================================================
# Price-based factors
# ======================================================================

def momentum_factor(prices: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """Cross-sectional momentum: average of past-return z-scores over
    multiple lookback windows (skipping the most recent month to avoid
    short-term reversal).

    Returns a DataFrame of z-scored composite momentum, index=date, cols=symbols.
    """
    if windows is None:
        windows = [63, 126, 252]

    skip = 21  # skip most recent month (short-term reversal effect)
    scores = []
    for w in windows:
        ret = prices.shift(skip).pct_change(w)
        # cross-sectional z-score each day
        zs = ret.sub(ret.mean(axis=1), axis=0).div(ret.std(axis=1), axis=0)
        scores.append(zs)

    composite = pd.concat(scores).groupby(level=0).mean()
    return composite


def mean_reversion_factor(prices: pd.DataFrame, window: int = 20,
                          zscore_threshold: float = 2.0) -> pd.DataFrame:
    """Mean-reversion signal based on Bollinger z-score.

    Returns z-score of current price vs rolling mean.  Negative = oversold
    (buy signal in mean-reversion framework).
    """
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    zscore = (prices - rolling_mean) / rolling_std
    # Invert: oversold (negative z) -> positive signal
    return -zscore


def trend_factor(prices: pd.DataFrame, short_window: int = 50,
                 long_window: int = 200) -> pd.DataFrame:
    """Trend-following factor: SMA short / SMA long ratio, cross-sectionally z-scored.

    >1 means short MA is above long MA (uptrend).
    """
    sma_short = prices.rolling(short_window).mean()
    sma_long = prices.rolling(long_window).mean()
    ratio = sma_short / sma_long

    # cross-sectional z-score
    zs = ratio.sub(ratio.mean(axis=1), axis=0).div(ratio.std(axis=1), axis=0)
    return zs


def trend_filter(prices: pd.DataFrame, long_window: int = 200,
                 penalty: float = 0.5) -> pd.DataFrame:
    """Trend filter: penalize stocks trading below their 200-day SMA.

    Returns a DataFrame of multipliers (1.0 for uptrend, `penalty` for downtrend).
    Used as a post-composite filter rather than a scored factor.
    """
    sma_long = prices.rolling(long_window).mean()
    above = prices >= sma_long
    return above.astype(float).replace(0.0, penalty)


def blowoff_filter(prices: pd.DataFrame, window: int = 20,
                    zscore_limit: float = 3.0, penalty: float = 0.5) -> pd.DataFrame:
    """Penalize stocks with extreme short-term gains (blowoff top protection).

    If a stock's Bollinger z-score exceeds `zscore_limit`, its composite score
    is multiplied by `penalty`.  Prevents chasing parabolic moves.
    """
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    zscore = (prices - rolling_mean) / rolling_std
    overextended = zscore > zscore_limit
    return (~overextended).astype(float).replace(0.0, penalty)


def volatility_factor(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Realized volatility factor (low-vol anomaly: prefer lower vol).

    Returns negative z-scored vol so that lower vol stocks get higher scores.
    """
    vol = returns.rolling(window).std() * np.sqrt(252)
    zs = vol.sub(vol.mean(axis=1), axis=0).div(vol.std(axis=1), axis=0)
    return -zs  # invert: low vol -> high score


# ======================================================================
# Fundamental factors
# ======================================================================

def value_factor(fundamentals: pd.DataFrame) -> pd.Series:
    """Composite value score from fundamental ratios.

    Uses inverse of PE and PB (lower = cheaper = higher value score).
    """
    scores = pd.DataFrame(index=fundamentals.index)

    for col in ["trailingPE", "forwardPE", "priceToBook"]:
        vals = pd.to_numeric(fundamentals[col], errors="coerce")
        if vals.notna().sum() > 2:
            # Invert and z-score (lower PE/PB = higher value)
            inv = 1.0 / vals.replace(0, np.nan)
            scores[col] = (inv - inv.mean()) / inv.std()

    if scores.empty:
        return pd.Series(0.0, index=fundamentals.index, name="value")

    return scores.mean(axis=1).rename("value")


def quality_factor(fundamentals: pd.DataFrame) -> pd.Series:
    """Composite quality score from ROE, profit margins, earnings growth."""
    scores = pd.DataFrame(index=fundamentals.index)

    for col in ["returnOnEquity", "profitMargins", "earningsGrowth"]:
        vals = pd.to_numeric(fundamentals[col], errors="coerce")
        if vals.notna().sum() > 2:
            scores[col] = (vals - vals.mean()) / vals.std()

    if scores.empty:
        return pd.Series(0.0, index=fundamentals.index, name="quality")

    return scores.mean(axis=1).rename("quality")


# ======================================================================
# Composite signal
# ======================================================================

class SignalGenerator:
    """Combines multiple alpha factors into a composite signal."""

    def __init__(self, config: dict):
        sig_cfg = config["signals"]
        self.momentum_windows = sig_cfg["momentum_windows"]
        self.mr_window = sig_cfg["mean_reversion_window"]
        self.mr_threshold = sig_cfg["mean_reversion_zscore_threshold"]
        self.vol_window = sig_cfg["volatility_window"]
        self.sma_short = sig_cfg["sma_short"]
        self.sma_long = sig_cfg["sma_long"]
        self.benchmark = config["universe"]["benchmark"]
        self.industry_neutral = sig_cfg.get("industry_neutral", False)
        self.winsorize_clip = sig_cfg.get("winsorize_clip", 3.0)

        # Factor weights — configurable from config.yaml, with defaults
        default_weights = {
            "momentum": 0.30,
            "mean_reversion": 0.15,
            "trend": 0.20,
            "volatility": 0.10,
            "value": 0.15,
            "quality": 0.10,
        }
        self.weights = {**default_weights, **sig_cfg.get("factor_weights", {})}

    def generate(self, prices: pd.DataFrame, returns: pd.DataFrame,
                 fundamentals: pd.DataFrame = None) -> pd.DataFrame:
        """Produce a composite alpha score for each symbol on each date.

        Pipeline:
          1. Compute scored factors (momentum, volatility, value, quality, etc.)
          2. Build weighted composite from scored factors
          3. Apply post-composite filters:
             - Trend filter: penalize stocks below 200d SMA
             - Blowoff filter: penalize stocks with extreme short-term gains

        Returns DataFrame (dates x symbols) of composite z-scores.
        """
        symbols = [c for c in prices.columns if c != self.benchmark]
        px = prices[symbols]
        ret = returns[[c for c in symbols if c in returns.columns]]

        factors = {}

        # Price-based (time-series)
        factors["momentum"] = momentum_factor(px, self.momentum_windows)
        factors["mean_reversion"] = mean_reversion_factor(px, self.mr_window, self.mr_threshold)
        factors["trend"] = trend_factor(px, self.sma_short, self.sma_long)
        factors["volatility"] = volatility_factor(ret, self.vol_window)

        # Fundamental (cross-sectional, static per rebalance)
        if fundamentals is not None and not fundamentals.empty:
            val = value_factor(fundamentals)
            qual = quality_factor(fundamentals)

            # Broadcast static scores to every date
            for name, series in [("value", val), ("quality", qual)]:
                df = pd.DataFrame(
                    np.tile(series.values, (len(px), 1)),
                    index=px.index,
                    columns=series.index,
                )
                # Align columns
                df = df.reindex(columns=px.columns)
                factors[name] = df

        # --- Industry neutralization + winsorization ---
        sector_map = None
        if self.industry_neutral and fundamentals is not None and "sector" in fundamentals.columns:
            sector_map = fundamentals["sector"]
            logger.info("Applying industry-neutral z-scoring")

        for name in list(factors.keys()):
            f = factors[name].reindex(index=px.index, columns=px.columns)
            # Industry-neutral z-score (within-sector ranking)
            if sector_map is not None:
                f = neutralize_by_sector(f, sector_map)
            # Winsorize + re-zscore to control outliers
            f = winsorize_zscore(f, clip_val=self.winsorize_clip)
            factors[name] = f

        # Weighted composite (only factors with weight > 0 contribute)
        composite = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        total_weight = 0.0
        for name, weight in self.weights.items():
            if weight > 0 and name in factors:
                f = factors[name].reindex(index=px.index, columns=px.columns)
                composite += weight * f.fillna(0)
                total_weight += weight

        if total_weight > 0:
            composite /= total_weight

        # --- Post-composite filters ---
        # Trend filter: penalize stocks below 200d SMA (score *= 0.5)
        tf = trend_filter(px, long_window=self.sma_long)
        composite = composite * tf.reindex(index=composite.index, columns=composite.columns).fillna(1.0)

        # Blowoff filter: penalize stocks with extreme overbought z-score > threshold
        bf = blowoff_filter(px, window=self.mr_window, zscore_limit=self.mr_threshold)
        composite = composite * bf.reindex(index=composite.index, columns=composite.columns).fillna(1.0)

        logger.info("Generated composite signal: shape=%s", composite.shape)
        return composite
