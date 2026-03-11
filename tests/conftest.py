"""Shared test fixtures — synthetic market data so tests run offline."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def config():
    """Minimal test config."""
    return {
        "universe": {
            "symbols": ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE",
                         "FFFF", "GGGG", "HHHH", "IIII", "JJJJ"],
            "benchmark": "BENCH",
        },
        "data": {"lookback_years": 2, "frequency": "1d"},
        "signals": {
            "momentum_windows": [63, 126, 252],
            "mean_reversion_window": 20,
            "mean_reversion_zscore_threshold": 3.0,
            "volatility_window": 63,
            "sma_short": 50,
            "sma_long": 200,
        },
        "portfolio": {
            "max_positions": 5,
            "max_position_weight": 0.30,
            "min_position_weight": 0.05,
            "target_volatility": 0.15,
            "rebalance_frequency_days": 21,
            "transaction_cost_bps": 10,
        },
        "risk": {
            "max_drawdown_limit": 0.20,
            "max_sector_weight": 0.30,
            "stop_loss_pct": 0.12,
        },
        "backtest": {
            "start_date": "2020-01-01",
            "end_date": "2023-01-01",
            "initial_capital": 1_000_000,
            "slippage_bps": 5,
        },
    }


@pytest.fixture
def synthetic_prices():
    """Generate 3 years of synthetic daily prices for 10 stocks + benchmark.

    Some stocks have upward drift (winners), some downward (losers),
    some mean-reverting, so the multi-factor model has something to work with.
    """
    np.random.seed(42)
    n_days = 756  # ~3 years of trading days
    dates = pd.bdate_range("2020-01-02", periods=n_days)

    symbols = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE",
               "FFFF", "GGGG", "HHHH", "IIII", "JJJJ", "BENCH"]

    # Different drift/vol profiles
    profiles = {
        "AAAA": (0.0008, 0.015),   # strong uptrend
        "BBBB": (0.0006, 0.018),   # moderate uptrend, higher vol
        "CCCC": (0.0003, 0.012),   # mild uptrend, low vol
        "DDDD": (-0.0002, 0.020),  # slight downtrend, high vol
        "EEEE": (0.0004, 0.014),   # moderate up
        "FFFF": (-0.0001, 0.025),  # flat, very high vol
        "GGGG": (0.0005, 0.013),   # good risk-adjusted
        "HHHH": (0.0001, 0.022),   # flat, high vol
        "IIII": (0.0007, 0.016),   # strong up
        "JJJJ": (-0.0003, 0.019),  # downtrend
        "BENCH": (0.0003, 0.012),  # benchmark: moderate
    }

    prices = {}
    for sym in symbols:
        drift, vol = profiles[sym]
        log_returns = np.random.normal(drift, vol, n_days)
        price_series = 100 * np.exp(np.cumsum(log_returns))
        prices[sym] = price_series

    return pd.DataFrame(prices, index=dates)


@pytest.fixture
def synthetic_returns(synthetic_prices):
    return synthetic_prices.pct_change().dropna()


@pytest.fixture
def synthetic_fundamentals():
    """Fake fundamental data for the test universe."""
    return pd.DataFrame({
        "trailingPE": [15, 20, 25, 30, 18, 22, 12, 35, 16, 28],
        "forwardPE": [13, 18, 22, 28, 16, 20, 10, 32, 14, 25],
        "priceToBook": [3, 5, 8, 2, 4, 6, 1.5, 10, 3.5, 7],
        "pegRatio": [1.2, 1.5, 2.0, 0.8, 1.3, 1.8, 0.9, 2.5, 1.1, 2.2],
        "dividendYield": [0.02, 0.01, 0.005, 0.03, 0.015, 0.008, 0.025, 0.0, 0.018, 0.004],
        "profitMargins": [0.20, 0.15, 0.10, 0.25, 0.18, 0.12, 0.30, 0.08, 0.22, 0.09],
        "returnOnEquity": [0.25, 0.18, 0.12, 0.30, 0.20, 0.14, 0.35, 0.08, 0.28, 0.10],
        "debtToEquity": [0.5, 0.8, 1.2, 0.3, 0.6, 1.0, 0.2, 1.5, 0.4, 1.3],
        "earningsGrowth": [0.15, 0.10, 0.05, 0.20, 0.12, 0.07, 0.25, 0.02, 0.18, 0.03],
        "revenueGrowth": [0.12, 0.08, 0.04, 0.15, 0.10, 0.06, 0.20, 0.01, 0.14, 0.02],
        "marketCap": [2e12, 1.5e12, 1e12, 800e9, 1.2e12, 600e9, 500e9, 300e9, 1.8e12, 400e9],
        "sector": ["Tech", "Tech", "Health", "Finance", "Tech",
                    "Consumer", "Industrial", "Health", "Tech", "Finance"],
    }, index=["AAAA", "BBBB", "CCCC", "DDDD", "EEEE",
              "FFFF", "GGGG", "HHHH", "IIII", "JJJJ"])
