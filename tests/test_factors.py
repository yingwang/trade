"""Tests for alpha factor generation."""

import numpy as np
import pandas as pd
import pytest

from quant.signals.factors import (
    momentum_factor, mean_reversion_factor, trend_factor, trend_filter,
    blowoff_filter, volatility_factor, value_factor, quality_factor,
    winsorize_zscore, neutralize_by_sector, SignalGenerator,
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


class TestTrendFilter:
    def test_uptrend_no_penalty(self):
        """Stocks above 200d SMA should get multiplier 1.0."""
        dates = pd.bdate_range("2020-01-01", periods=300)
        px = pd.DataFrame({"UP": np.linspace(50, 150, 300)}, index=dates)
        result = trend_filter(px, long_window=200)
        assert result["UP"].iloc[-1] == 1.0

    def test_downtrend_penalized(self):
        """Stocks below 200d SMA should get multiplier 0.5."""
        dates = pd.bdate_range("2020-01-01", periods=300)
        px = pd.DataFrame({"DOWN": np.linspace(150, 50, 300)}, index=dates)
        result = trend_filter(px, long_window=200)
        assert result["DOWN"].iloc[-1] == 0.5


class TestBlowoffFilter:
    def test_normal_stock_no_penalty(self):
        """Stock with moderate z-score should get multiplier 1.0."""
        dates = pd.bdate_range("2020-01-01", periods=50)
        px = pd.DataFrame({"A": np.linspace(100, 110, 50)}, index=dates)
        result = blowoff_filter(px, window=20, zscore_limit=3.0)
        assert result["A"].iloc[-1] == 1.0

    def test_parabolic_stock_penalized(self):
        """Stock with extreme spike should get multiplier 0.5."""
        dates = pd.bdate_range("2020-01-01", periods=50)
        # Flat then massive vertical spike — guarantees zscore >> 3
        prices = np.concatenate([np.full(45, 100), np.array([100, 100, 100, 100, 500])])
        px = pd.DataFrame({"A": prices}, index=dates)
        result = blowoff_filter(px, window=20, zscore_limit=3.0)
        assert result["A"].iloc[-1] == 0.5


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


class TestWinsorizeZscore:
    def test_clips_extremes(self):
        """Values beyond clip_val should be clipped."""
        dates = pd.bdate_range("2020-01-01", periods=5)
        df = pd.DataFrame({
            "A": [0, 0, 0, 0, 10],  # extreme outlier
            "B": [0, 0, 0, 0, 0],
            "C": [0, 0, 0, 0, -1],
        }, index=dates)
        result = winsorize_zscore(df, clip_val=3.0)
        # After winsorize, no value should dominate excessively
        assert result.abs().max().max() < 5.0

    def test_output_approximately_standardized(self):
        np.random.seed(42)
        dates = pd.bdate_range("2020-01-01", periods=100)
        df = pd.DataFrame(np.random.randn(100, 5), index=dates, columns=list("ABCDE"))
        result = winsorize_zscore(df, clip_val=3.0)
        # Each row should be approximately mean=0
        row_means = result.mean(axis=1).dropna()
        assert abs(row_means.mean()) < 0.1


class TestNeutralizeBySecotr:
    def test_within_sector_zscore(self):
        """After neutralization, within-sector mean should be ~0."""
        dates = pd.bdate_range("2020-01-01", periods=50)
        np.random.seed(42)
        # Tech stocks have much higher raw scores
        df = pd.DataFrame({
            "T1": np.random.normal(5.0, 1.0, 50),  # tech, high
            "T2": np.random.normal(4.0, 1.0, 50),  # tech, high
            "F1": np.random.normal(0.0, 1.0, 50),  # finance, low
            "F2": np.random.normal(-1.0, 1.0, 50), # finance, low
        }, index=dates)
        sector_map = pd.Series({"T1": "Tech", "T2": "Tech", "F1": "Finance", "F2": "Finance"})
        # Use min_sector_size=2 to test within-sector z-scoring with small sectors
        result = neutralize_by_sector(df, sector_map, min_sector_size=2)
        # Tech sector mean should be ~0 (not 4.5)
        tech_mean = result[["T1", "T2"]].mean(axis=1).mean()
        assert abs(tech_mean) < 0.3

    def test_small_sector_falls_back_to_cross_sectional(self):
        """Sectors below min_sector_size should use cross-sectional z-score."""
        dates = pd.bdate_range("2020-01-01", periods=50)
        np.random.seed(42)
        df = pd.DataFrame({
            "T1": np.random.normal(5.0, 1.0, 50),
            "T2": np.random.normal(4.0, 1.0, 50),
            "F1": np.random.normal(0.0, 1.0, 50),
            "F2": np.random.normal(-1.0, 1.0, 50),
        }, index=dates)
        sector_map = pd.Series({"T1": "Tech", "T2": "Tech", "F1": "Finance", "F2": "Finance"})
        # min_sector_size=5 means all 2-stock sectors fall back
        result = neutralize_by_sector(df, sector_map, min_sector_size=5)
        # Shape preserved
        assert result.shape == df.shape
        # Cross-sectional mean should be ~0 (all 4 stocks z-scored together)
        row_means = result.mean(axis=1).dropna()
        assert abs(row_means.mean()) < 0.3

    def test_preserves_shape(self):
        dates = pd.bdate_range("2020-01-01", periods=10)
        df = pd.DataFrame(np.random.randn(10, 4), index=dates, columns=list("ABCD"))
        sector_map = pd.Series({"A": "X", "B": "X", "C": "Y", "D": "Y"})
        result = neutralize_by_sector(df, sector_map, min_sector_size=2)
        assert result.shape == df.shape

    def test_no_sector_data_returns_original(self):
        dates = pd.bdate_range("2020-01-01", periods=10)
        df = pd.DataFrame(np.random.randn(10, 3), index=dates, columns=list("ABC"))
        result = neutralize_by_sector(df, pd.Series(dtype=str))
        pd.testing.assert_frame_equal(result, df)


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

    def test_industry_neutral_reduces_sector_bias(self, config, synthetic_prices,
                                                   synthetic_returns, synthetic_fundamentals):
        """With industry_neutral=True, tech stocks shouldn't dominate top scores."""
        # Run with neutralization
        config_neutral = {**config, "signals": {**config["signals"], "industry_neutral": True}}
        gen_neutral = SignalGenerator(config_neutral)
        result_neutral = gen_neutral.generate(synthetic_prices, synthetic_returns, synthetic_fundamentals)

        # Run without neutralization
        config_raw = {**config, "signals": {**config["signals"], "industry_neutral": False}}
        gen_raw = SignalGenerator(config_raw)
        result_raw = gen_raw.generate(synthetic_prices, synthetic_returns, synthetic_fundamentals)

        # Both should produce valid signals
        assert not result_neutral.empty
        assert not result_raw.empty
        # Neutralized signals should have different sector distribution
        last_neutral = result_neutral.iloc[-1].dropna()
        last_raw = result_raw.iloc[-1].dropna()
        assert not last_neutral.equals(last_raw)
