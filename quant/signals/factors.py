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
                         sector_map: pd.Series,
                         min_sector_size: int = 5) -> pd.DataFrame:
    """Industry-neutral z-score: z-score within each sector, then combine.

    Parameters
    ----------
    factor_df : DataFrame (dates x symbols)
        Raw factor scores.
    sector_map : Series (symbol -> sector string)
        Sector assignment for each symbol.
    min_sector_size : int
        Minimum number of stocks in a sector for within-sector z-scoring.
        Sectors smaller than this fall back to cross-sectional z-scoring
        to avoid unstable estimates from tiny samples.

    Returns
    -------
    DataFrame of industry-neutralized z-scores (same shape as input).
    """
    result = factor_df.copy()
    symbols = factor_df.columns
    sectors = sector_map.reindex(symbols).dropna()
    if sectors.empty:
        return factor_df

    # Stocks without sector assignment: apply cross-sectional z-score
    unsectored = [s for s in symbols if s not in sectors.index]
    small_sector_stocks = []

    unique_sectors = sectors.unique()
    for sector in unique_sectors:
        mask = sectors[sectors == sector].index.tolist()
        mask = [s for s in mask if s in symbols]
        if len(mask) < min_sector_size:
            # Sector too small for reliable within-sector z-scoring;
            # these stocks will be z-scored cross-sectionally below.
            small_sector_stocks.extend(mask)
            logger.debug("Sector '%s' has only %d stocks (< %d), "
                         "falling back to cross-sectional z-score",
                         sector, len(mask), min_sector_size)
            continue
        sector_data = factor_df[mask]
        mean = sector_data.mean(axis=1)
        std = sector_data.std(axis=1).replace(0, 1)
        result[mask] = sector_data.sub(mean, axis=0).div(std, axis=0)

    # Cross-sectional z-score for unsectored + small-sector stocks
    fallback = unsectored + small_sector_stocks
    if fallback:
        fallback = [s for s in fallback if s in symbols]
        if len(fallback) >= 2:
            fb_data = factor_df[fallback]
            mean = fb_data.mean(axis=1)
            std = fb_data.std(axis=1).replace(0, 1)
            result[fallback] = fb_data.sub(mean, axis=0).div(std, axis=0)

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


def short_term_reversal_factor(returns: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Short-term reversal: stocks that dropped most in the last 5 days
    tend to bounce back. Buy oversold, sell overbought.

    Complementary to momentum — momentum skips the most recent month,
    this factor exploits the most recent week.

    Returns cross-sectional z-score (negative recent return → positive signal).
    """
    recent_ret = returns.rolling(window).sum()
    # Invert: biggest losers get highest score (expected to bounce)
    zs = recent_ret.sub(recent_ret.mean(axis=1), axis=0).div(
        recent_ret.std(axis=1), axis=0)
    return -zs


def volume_momentum_factor(prices: pd.DataFrame, returns: pd.DataFrame,
                           window: int = 21) -> pd.DataFrame:
    """Volume-weighted momentum: price moves on high volume are more meaningful.

    Stocks rising on increasing volume score higher than those rising on
    declining volume (which may be false breakouts).

    Uses price * volume correlation as a proxy when volume data is in the
    price DataFrame, otherwise falls back to return autocorrelation.
    """
    # Use absolute returns weighted by return sign as proxy
    # Positive: strong consistent moves. Negative: choppy/reverting.
    autocorr = returns.rolling(window).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0,
        raw=True
    )
    # Stocks with positive autocorrelation (trending) score higher
    zs = autocorr.sub(autocorr.mean(axis=1), axis=0).div(
        autocorr.std(axis=1), axis=0)
    return zs


def high_proximity_factor(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """52-week high proximity: stocks near their 52-week high tend to
    continue outperforming (George & Hwang, 2004).

    Ratio of current price to 52-week high, cross-sectionally z-scored.
    Closer to high = higher score.
    """
    rolling_high = prices.rolling(window).max()
    proximity = prices / rolling_high  # 0 to 1, where 1 = at 52-week high
    zs = proximity.sub(proximity.mean(axis=1), axis=0).div(
        proximity.std(axis=1), axis=0)
    return zs


def volatility_contraction_factor(returns: pd.DataFrame,
                                  short_window: int = 10,
                                  long_window: int = 63) -> pd.DataFrame:
    """Volatility contraction pattern: stocks whose recent volatility has
    contracted relative to their longer-term volatility are coiling for
    a potential breakout.

    Low short-term vol / long-term vol = contraction = positive signal.
    Combined with momentum direction, this captures breakout setups.
    """
    short_vol = returns.rolling(short_window).std()
    long_vol = returns.rolling(long_window).std()
    ratio = short_vol / long_vol.replace(0, np.nan)
    # Invert: lower ratio (more contraction) = higher score
    zs = ratio.sub(ratio.mean(axis=1), axis=0).div(
        ratio.std(axis=1), axis=0)
    return -zs


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
# Filter utility
# ======================================================================

def _apply_filter_safe(composite: pd.DataFrame,
                       filter_df: pd.DataFrame) -> pd.DataFrame:
    """Apply a multiplicative filter that correctly handles negative scores.

    For positive composite scores: score * filter (reduces magnitude when
    filter < 1, which is the standard penalty behavior).

    For negative composite scores: score * (2 - filter) (increases the
    magnitude of the penalty when filter < 1, preventing the bug where
    multiplying a negative score by 0.5 would halve the penalty instead
    of increasing it).

    When filter == 1.0 (no penalty), both branches are identity.
    """
    positive_mask = composite >= 0
    result = composite.copy()
    result[positive_mask] = composite[positive_mask] * filter_df[positive_mask]
    result[~positive_mask] = composite[~positive_mask] * (2.0 - filter_df[~positive_mask])
    return result


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

        # New price-based factors (no fundamental data needed)
        factors["short_term_reversal"] = short_term_reversal_factor(ret, window=5)
        factors["volume_momentum"] = volume_momentum_factor(px, ret, window=21)
        factors["high_proximity"] = high_proximity_factor(px, window=252)
        factors["vol_contraction"] = volatility_contraction_factor(ret, short_window=10, long_window=63)

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

        # Weighted composite — normalize per-cell by available factor weights
        # so that stocks with partial data (e.g., recently IPO'd, missing
        # longer momentum windows) are not penalized toward zero.
        composite = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        weight_available = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        for name, weight in self.weights.items():
            if weight > 0 and name in factors:
                f = factors[name].reindex(index=px.index, columns=px.columns)
                has_data = f.notna().astype(float)
                composite += weight * f.fillna(0)
                weight_available += weight * has_data

        # Normalize by actual available weight per cell (not global total)
        weight_available = weight_available.replace(0, np.nan)
        composite = composite / weight_available

        # --- Post-composite filters ---
        # Apply multiplicative filters safely: for positive scores, the
        # filter reduces magnitude (penalizes); for negative scores, the
        # filter increases magnitude (makes more negative = stronger penalty).
        # This prevents the bug where multiplying a negative score by 0.5
        # would incorrectly *improve* the score of a down-trending stock.

        # Trend filter: penalize stocks below 200d SMA (score *= 0.5)
        tf = trend_filter(px, long_window=self.sma_long)
        tf = tf.reindex(index=composite.index, columns=composite.columns).fillna(1.0)
        composite = _apply_filter_safe(composite, tf)

        # Blowoff filter: penalize stocks with extreme overbought z-score > threshold
        bf = blowoff_filter(px, window=self.mr_window, zscore_limit=self.mr_threshold)
        bf = bf.reindex(index=composite.index, columns=composite.columns).fillna(1.0)
        composite = _apply_filter_safe(composite, bf)

        # Expose per-factor scores for dashboard / analysis
        self.last_factors_ = dict(factors)

        logger.info("Generated composite signal: shape=%s", composite.shape)
        return composite
