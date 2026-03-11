"""Tests for alpha factor generation."""

import numpy as np
import pandas as pd
import pytest

from quant.signals.factors import (
    momentum_factor, mean_reversion_factor, trend_factor,
    volatility_factor, value_factor, quality_factor, SignalGenerator,
)


class TestMomentumFactor:
    def test_shape_matches_input(self, synthetic_prices):
        px = synthetic_prices.drop(columns=["BENCH"])
        result = momentum_factor(px, windows=[21, 63])
        assert result.shape[1] == px.shape[1]

    def test_cross_sectional_zscore_centered(self, synthetic_prices):
        px = synthetic_prices.drop(columns=["BENCH"])
        result = momentum_factor(px, windows=[63])
        # Each row should be roughly mean-zero cross-sectionally
        row_means = result.dropna(how="all").mean(axis=1).dropna()
        assert abs(row_means.mean()) < 0.5

    def test_uptrending_stocks_rank_higher(self, synthetic_prices):
        """Stocks with positive drift should have higher momentum on average."""
        px = synthetic_prices.drop(columns=["BENCH"])
        result = momentum_factor(px, windows=[63, 126])
        avg = result.mean()
        # AAAA (strongest uptrend) should rank near the top
        assert avg["AAAA"] > avg.median()


class TestMeanReversionFactor:
    def test_output_shape(self, synthetic_prices):
        px = synthetic_prices.drop(columns=["BENCH"])
        result = mean_reversion_factor(px, window=20)
        assert result.shape == px.shape

    def test_oversold_gives_positive_signal(self):
        """A stock far below its moving average should get a positive signal."""
        dates = pd.bdate_range("2020-01-01", periods=50)
        # Price drops sharply at end
        px = pd.DataFrame({"A": np.concatenate([np.full(40, 100), np.full(10, 80)])}, index=dates)
        result = mean_reversion_factor(px, window=20)
        assert result["A"].iloc[-1] > 0  # oversold = positive


class TestTrendFactor:
    def test_uptrend_positive(self):
        dates = pd.bdate_range("2020-01-01", periods=300)
        px = pd.DataFrame({
            "UP": np.linspace(50, 150, 300),
            "DOWN": np.linspace(150, 50, 300),
        }, index=dates)
        result = trend_factor(px, short_window=50, long_window=200)
        last = result.iloc[-1]
        assert last["UP"] > last["DOWN"]


class TestVolatilityFactor:
    def test_low_vol_preferred(self, synthetic_returns):
        ret = synthetic_returns.drop(columns=["BENCH"])
        result = volatility_factor(ret, window=63)
        avg = result.mean()
        # CCCC has lowest vol (0.012), should score highest on average
        assert avg["CCCC"] > avg.median()


class TestValueFactor:
    def test_cheap_stocks_score_higher(self, synthetic_fundamentals):
        result = value_factor(synthetic_fundamentals)
        # GGGG has PE=12, PB=1.5 (cheapest) -> should score highest
        assert result["GGGG"] == result.max()

    def test_returns_series(self, synthetic_fundamentals):
        result = value_factor(synthetic_fundamentals)
        assert isinstance(result, pd.Series)
        assert len(result) == 10


class TestQualityFactor:
    def test_high_quality_scores_higher(self, synthetic_fundamentals):
        result = quality_factor(synthetic_fundamentals)
        # GGGG has highest ROE (0.35), margins (0.30), growth (0.25)
        assert result["GGGG"] == result.max()


class TestSignalGenerator:
    def test_composite_signal_shape(self, config, synthetic_prices,
                                     synthetic_returns, synthetic_fundamentals):
        gen = SignalGenerator(config)
        result = gen.generate(synthetic_prices, synthetic_returns, synthetic_fundamentals)
        symbols = [c for c in synthetic_prices.columns if c != "BENCH"]
        assert set(result.columns) == set(symbols)
        assert len(result) > 0

    def test_composite_without_fundamentals(self, config, synthetic_prices, synthetic_returns):
        gen = SignalGenerator(config)
        result = gen.generate(synthetic_prices, synthetic_returns, fundamentals=None)
        assert not result.empty

    def test_signal_differentiates_stocks(self, config, synthetic_prices,
                                           synthetic_returns, synthetic_fundamentals):
        """Composite signal should not be identical across all stocks."""
        gen = SignalGenerator(config)
        result = gen.generate(synthetic_prices, synthetic_returns, synthetic_fundamentals)
        last_row = result.iloc[-1].dropna()
        assert last_row.std() > 0.01
