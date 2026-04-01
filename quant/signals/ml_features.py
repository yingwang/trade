"""Feature engineering for the ML (TFT) strategy.

Builds a rich feature matrix from price/volume data and the existing factor
signals.  Output is a 3D tensor (dates x symbols x features) suitable for
the Temporal Fusion Transformer's rolling-window training scheme.

Feature groups:
  1. Existing 5 factor scores (momentum, high_proximity, reversal, vol_contraction, volume_momentum)
  2. Technical indicators: RSI(14), MACD, Bollinger Band width, ATR(14), OBV trend
  3. Rolling statistics: 5/10/21/63 day returns, volatility, volume ratios
  4. Cross-sectional rank features (percentile rank of each feature across stocks)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# Technical indicators
# ======================================================================

def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(prices: pd.DataFrame,
                 fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, pd.DataFrame]:
    """MACD line, signal line, and histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd_line": macd_line,
        "macd_signal": signal_line,
        "macd_histogram": histogram,
    }


def compute_bollinger_width(prices: pd.DataFrame, window: int = 20,
                            num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Band width: (upper - lower) / middle.

    Measures volatility expansion/contraction.
    """
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = (upper - lower) / middle.replace(0, np.nan)
    return width


def compute_atr(prices: pd.DataFrame, returns: pd.DataFrame,
                window: int = 14) -> pd.DataFrame:
    """Average True Range approximation from close prices and returns.

    Without high/low data, we approximate TR as the absolute daily return
    multiplied by the price level, then take a rolling mean.
    """
    # Approximate true range from absolute daily changes
    tr = prices.diff().abs()
    atr = tr.rolling(window).mean()
    # Normalize by price to make it cross-sectionally comparable
    atr_pct = atr / prices.replace(0, np.nan)
    return atr_pct


def compute_obv_trend(prices: pd.DataFrame, returns: pd.DataFrame,
                      window: int = 21) -> pd.DataFrame:
    """On-Balance Volume trend proxy.

    Without volume data, we use signed returns as a proxy: cumulate the
    sign of daily returns over a rolling window.  Positive trend means
    more up-days than down-days (a volume-independent proxy for OBV slope).
    """
    signed = returns.apply(np.sign)
    obv_proxy = signed.rolling(window).sum()
    return obv_proxy


# ======================================================================
# Rolling statistics
# ======================================================================

def compute_rolling_returns(prices: pd.DataFrame,
                            windows: list[int] = None) -> dict[str, pd.DataFrame]:
    """Rolling total returns over multiple lookback windows."""
    if windows is None:
        windows = [5, 10, 21, 63]
    result = {}
    for w in windows:
        result[f"ret_{w}d"] = prices.pct_change(w)
    return result


def compute_rolling_volatility(returns: pd.DataFrame,
                               windows: list[int] = None) -> dict[str, pd.DataFrame]:
    """Rolling realized volatility (annualized) over multiple windows."""
    if windows is None:
        windows = [5, 10, 21, 63]
    result = {}
    for w in windows:
        result[f"vol_{w}d"] = returns.rolling(w).std() * np.sqrt(252)
    return result


def compute_volume_ratios(returns: pd.DataFrame,
                          windows: list[int] = None) -> dict[str, pd.DataFrame]:
    """Volume ratio proxies using absolute returns as a volume proxy.

    Ratio of short-window average absolute return to longer-window average,
    capturing activity spikes relative to baseline.
    """
    if windows is None:
        windows = [5, 10, 21, 63]
    abs_ret = returns.abs()
    result = {}
    for i in range(len(windows) - 1):
        short_w = windows[i]
        long_w = windows[-1]
        short_avg = abs_ret.rolling(short_w).mean()
        long_avg = abs_ret.rolling(long_w).mean()
        result[f"vratio_{short_w}_{long_w}"] = short_avg / long_avg.replace(0, np.nan)
    return result


# ======================================================================
# Cross-sectional rank features
# ======================================================================

def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank across stocks for each date (0 to 1)."""
    return df.rank(axis=1, pct=True)


# ======================================================================
# Feature matrix builder
# ======================================================================

class MLFeatureEngine:
    """Builds the full feature matrix for the TFT model.

    Parameters
    ----------
    config : dict
        Full system config (used to read signal parameters).
    """

    # Canonical feature names (stable ordering for model input)
    FACTOR_FEATURES = [
        "momentum", "high_proximity", "short_term_reversal",
        "vol_contraction", "volume_momentum",
    ]
    TECHNICAL_FEATURES = [
        "rsi_14", "macd_line", "macd_signal", "macd_histogram",
        "bb_width", "atr_14", "obv_trend",
    ]
    ROLLING_RET_FEATURES = ["ret_5d", "ret_10d", "ret_21d", "ret_63d"]
    ROLLING_VOL_FEATURES = ["vol_5d", "vol_10d", "vol_21d", "vol_63d"]
    VOLUME_RATIO_FEATURES = ["vratio_5_63", "vratio_10_63", "vratio_21_63"]

    def __init__(self, config: dict):
        self.config = config
        self.benchmark = config["universe"]["benchmark"]

    @property
    def feature_names(self) -> list[str]:
        """All feature names in canonical order."""
        base = (
            self.FACTOR_FEATURES
            + self.TECHNICAL_FEATURES
            + self.ROLLING_RET_FEATURES
            + self.ROLLING_VOL_FEATURES
            + self.VOLUME_RATIO_FEATURES
        )
        # Rank versions of every feature
        ranked = [f"{f}_rank" for f in base]
        return base + ranked

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def build_features(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        factor_scores: Optional[dict[str, pd.DataFrame]] = None,
    ) -> dict[str, pd.DataFrame]:
        """Build all features for every (date, symbol) pair.

        Parameters
        ----------
        prices : DataFrame
            Adjusted close prices (dates x symbols, may include benchmark).
        returns : DataFrame
            Daily returns (dates x symbols).
        factor_scores : dict, optional
            Pre-computed factor DataFrames from SignalGenerator.last_factors_.
            Keys should match FACTOR_FEATURES names.

        Returns
        -------
        dict mapping feature_name -> DataFrame (dates x symbols).
        """
        symbols = [c for c in prices.columns if c != self.benchmark]
        px = prices[symbols]
        ret = returns[[c for c in symbols if c in returns.columns]]

        features: dict[str, pd.DataFrame] = {}

        # --- Group 1: Existing factor scores ---
        if factor_scores is not None:
            for name in self.FACTOR_FEATURES:
                if name in factor_scores:
                    f = factor_scores[name].reindex(index=px.index, columns=px.columns)
                    features[name] = f
                else:
                    logger.warning("Factor '%s' not found in factor_scores, filling zeros", name)
                    features[name] = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        else:
            # Compute minimal factor proxies if no pre-computed scores available
            logger.info("No pre-computed factor scores; using return-based proxies")
            from quant.signals.factors import (
                momentum_factor, high_proximity_factor, short_term_reversal_factor,
                volatility_contraction_factor, volume_momentum_factor,
            )
            sig_cfg = self.config["signals"]
            features["momentum"] = momentum_factor(px, sig_cfg["momentum_windows"])
            features["high_proximity"] = high_proximity_factor(px, window=252)
            features["short_term_reversal"] = short_term_reversal_factor(ret, window=5)
            features["vol_contraction"] = volatility_contraction_factor(ret)
            features["volume_momentum"] = volume_momentum_factor(px, ret, window=21)

        # --- Group 2: Technical indicators ---
        features["rsi_14"] = compute_rsi(px, window=14)

        macd = compute_macd(px)
        features["macd_line"] = macd["macd_line"]
        features["macd_signal"] = macd["macd_signal"]
        features["macd_histogram"] = macd["macd_histogram"]

        features["bb_width"] = compute_bollinger_width(px, window=20)
        features["atr_14"] = compute_atr(px, ret, window=14)
        features["obv_trend"] = compute_obv_trend(px, ret, window=21)

        # --- Group 3: Rolling statistics ---
        for name, df in compute_rolling_returns(px).items():
            features[name] = df

        for name, df in compute_rolling_volatility(ret).items():
            features[name] = df

        for name, df in compute_volume_ratios(ret).items():
            features[name] = df

        # --- Group 4: Cross-sectional rank features ---
        base_names = list(features.keys())
        for name in base_names:
            features[f"{name}_rank"] = cross_sectional_rank(features[name])

        logger.info(
            "Built %d features for %d symbols x %d dates",
            len(features), len(symbols), len(px),
        )
        return features

    def build_feature_matrix(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        factor_scores: Optional[dict[str, pd.DataFrame]] = None,
    ) -> tuple[np.ndarray, list[str], pd.DatetimeIndex, list[str]]:
        """Build a 3D numpy array (dates x symbols x features).

        Returns
        -------
        X : ndarray of shape (T, N, F)
            Feature tensor.  NaN values are replaced with 0.
        feature_names : list of str
        dates : DatetimeIndex
        symbols : list of str
        """
        features = self.build_features(prices, returns, factor_scores)
        symbols = [c for c in prices.columns if c != self.benchmark]
        dates = prices.index

        # Ensure consistent ordering
        ordered_names = [n for n in self.feature_names if n in features]
        T = len(dates)
        N = len(symbols)
        F = len(ordered_names)

        X = np.full((T, N, F), np.nan, dtype=np.float32)
        for fi, name in enumerate(ordered_names):
            df = features[name].reindex(index=dates, columns=symbols)
            X[:, :, fi] = df.values

        # Replace NaN with 0 (models expect clean input)
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            logger.info("Replacing %d NaN values (%.2f%%) with 0 in feature matrix",
                        nan_count, 100.0 * nan_count / X.size)
        X = np.nan_to_num(X, nan=0.0)

        return X, ordered_names, dates, symbols

    def get_target(
        self,
        returns: pd.DataFrame,
        horizon: int = 21,
    ) -> pd.DataFrame:
        """Compute forward returns as the prediction target.

        Parameters
        ----------
        returns : DataFrame
            Daily returns (dates x symbols).
        horizon : int
            Forward-looking horizon in trading days.

        Returns
        -------
        DataFrame of forward cumulative returns (dates x symbols).
        Values at date t represent cumulative return from t+1 to t+horizon.
        """
        symbols = [c for c in returns.columns if c != self.benchmark]
        ret = returns[symbols]
        # Forward cumulative return: product of (1+r) over next `horizon` days - 1
        fwd = (1 + ret).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1
        return fwd

    def get_cross_sectional_target(
        self,
        returns: pd.DataFrame,
        horizon: int = 21,
    ) -> pd.DataFrame:
        """Cross-sectional rank of forward returns (0 to 1).

        This is the primary target for the TFT model: predict relative
        stock performance rather than absolute returns.
        """
        fwd = self.get_target(returns, horizon)
        return fwd.rank(axis=1, pct=True)
