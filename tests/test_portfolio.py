"""Tests for portfolio optimizer and risk management."""

import numpy as np
import pandas as pd
import pytest

from quant.portfolio.optimizer import (
    PortfolioOptimizer,
    RiskMonitor,
    _ledoit_wolf_shrinkage,
)


class TestPortfolioOptimizer:
    def test_select_top_stocks(self, config):
        opt = PortfolioOptimizer(config)
        scores = pd.Series({
            "A": 2.0, "B": 1.5, "C": 1.0, "D": 0.5, "E": 0.0,
            "F": -0.5, "G": -1.0,
        })
        selected = opt.select_top_stocks(scores)
        assert len(selected) == config["portfolio"]["max_positions"]  # 5
        assert selected[0] == "A"

    def test_weights_sum_to_one(self, config):
        opt = PortfolioOptimizer(config)
        selected = ["A", "B", "C", "D", "E"]
        scores = pd.Series({"A": 2.0, "B": 1.5, "C": 1.0, "D": 0.5, "E": 0.3})
        cov = pd.DataFrame(
            np.eye(5) * 0.04 / 252,
            index=selected, columns=selected,
        )
        weights = opt.optimize_weights(selected, scores, cov)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_weights_respect_bounds(self, config):
        opt = PortfolioOptimizer(config)
        selected = ["A", "B", "C", "D", "E"]
        scores = pd.Series({"A": 5.0, "B": 0.1, "C": 0.1, "D": 0.1, "E": 0.1})
        cov = pd.DataFrame(
            np.eye(5) * 0.04 / 252,
            index=selected, columns=selected,
        )
        weights = opt.optimize_weights(selected, scores, cov)
        assert weights.max() <= config["portfolio"]["max_position_weight"] + 1e-6
        assert weights.min() >= config["portfolio"]["min_position_weight"] - 1e-6

    def test_enforce_bounds_preserves_limits_after_normalization(self, config):
        opt = PortfolioOptimizer(config)
        raw = pd.Series({"A": 0.80, "B": 0.10, "C": 0.05, "D": 0.03, "E": 0.02})
        weights = opt._enforce_bounds(raw)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert weights.max() <= config["portfolio"]["max_position_weight"] + 1e-6
        assert weights.min() >= config["portfolio"]["min_position_weight"] - 1e-6

    def test_vol_scaling_reduces_weights(self, config):
        opt = PortfolioOptimizer(config)
        weights = pd.Series({"A": 0.5, "B": 0.5})
        # High-vol covariance matrix
        cov = pd.DataFrame(
            [[0.10, 0.02], [0.02, 0.10]],
            index=["A", "B"], columns=["A", "B"],
        )
        scaled = opt.apply_vol_scaling(weights, cov, regime="high_vol")
        assert scaled.sum() <= weights.sum()

    def test_vol_scaling_levers_up_in_low_vol(self, config):
        opt = PortfolioOptimizer(config)
        weights = pd.Series({"A": 0.5, "B": 0.5})
        # Very low vol covariance -> scale should exceed 1.0
        cov = pd.DataFrame(
            [[0.0001, 0.0], [0.0, 0.0001]],
            index=["A", "B"], columns=["A", "B"],
        )
        scaled = opt.apply_vol_scaling(weights, cov, regime="low_vol")
        assert scaled.sum() > 1.0  # leveraged
        assert scaled.sum() <= config["leverage"]["regime_leverage_caps"]["low_vol"] + 0.01

    def test_detect_regime_low_vol(self, config):
        np.random.seed(0)
        opt = PortfolioOptimizer(config)
        # Low-vol SPY returns: ~5% annualized
        spy_ret = pd.Series(np.random.normal(0.0003, 0.003, 200))
        assert opt.detect_regime(spy_ret) == "low_vol"

    def test_detect_regime_high_vol(self, config):
        np.random.seed(0)
        opt = PortfolioOptimizer(config)
        # High-vol SPY returns: ~30% annualized
        spy_ret = pd.Series(np.random.normal(-0.001, 0.02, 200))
        assert opt.detect_regime(spy_ret) == "high_vol"

    def test_detect_regime_normal(self, config):
        np.random.seed(0)
        opt = PortfolioOptimizer(config)
        # Normal SPY returns: ~15% annualized vol
        spy_ret = pd.Series(np.random.normal(0.0003, 0.009, 200))
        assert opt.detect_regime(spy_ret) == "normal"

    def test_regime_caps_leverage_in_stress(self, config):
        opt = PortfolioOptimizer(config)
        weights = pd.Series({"A": 0.5, "B": 0.5})
        # Very low portfolio vol -> would want to lever up
        cov = pd.DataFrame(
            [[0.0001, 0.0], [0.0, 0.0001]],
            index=["A", "B"], columns=["A", "B"],
        )
        # But high_vol regime caps at 0.7
        scaled = opt.apply_vol_scaling(weights, cov, regime="high_vol")
        assert scaled.sum() <= 0.7 + 0.01

    def test_stop_loss(self, config):
        opt = PortfolioOptimizer(config)
        weights = pd.Series({"A": 0.5, "B": 0.5})
        entry = pd.Series({"A": 100.0, "B": 100.0})
        current = pd.Series({"A": 85.0, "B": 100.0})  # A dropped 15% > 12% stop
        result = opt.check_stop_losses(weights, entry, current)
        assert result["A"] == 0.0
        assert result["B"] > 0.0


class TestTransactionCostPenalty:
    """Tests for the turnover penalty in the optimizer."""

    def test_turnover_penalty_reduces_deviation_from_prev(self, config):
        """With high turnover penalty, optimizer should stay closer to prev_weights."""
        config_high_tc = {**config}
        config_high_tc["portfolio"] = {**config["portfolio"], "turnover_penalty": 0.5}
        opt = PortfolioOptimizer(config_high_tc)

        selected = ["A", "B", "C", "D", "E"]
        scores = pd.Series({"A": 2.0, "B": 1.5, "C": 1.0, "D": 0.5, "E": 0.3})
        cov = pd.DataFrame(np.eye(5) * 0.04 / 252, index=selected, columns=selected)

        prev_w = pd.Series({"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2})

        # With high turnover penalty
        weights_high_tc = opt.optimize_weights(
            selected, scores, cov, prev_weights=prev_w
        )

        # Without turnover penalty
        config_no_tc = {**config}
        config_no_tc["portfolio"] = {**config["portfolio"], "turnover_penalty": 0.0}
        opt_no_tc = PortfolioOptimizer(config_no_tc)
        weights_no_tc = opt_no_tc.optimize_weights(
            selected, scores, cov, prev_weights=prev_w
        )

        # High TC should produce less turnover than no TC
        turnover_high = np.abs(weights_high_tc - prev_w).sum()
        turnover_no = np.abs(weights_no_tc - prev_w).sum()
        assert turnover_high <= turnover_no + 1e-6

    def test_optimizer_with_no_prev_weights(self, config):
        """Optimizer should work without prev_weights (defaults to equal weight)."""
        opt = PortfolioOptimizer(config)
        selected = ["A", "B", "C", "D", "E"]
        scores = pd.Series({"A": 2.0, "B": 1.5, "C": 1.0, "D": 0.5, "E": 0.3})
        cov = pd.DataFrame(np.eye(5) * 0.04 / 252, index=selected, columns=selected)

        weights = opt.optimize_weights(selected, scores, cov, prev_weights=None)
        assert abs(weights.sum() - 1.0) < 1e-6
        assert len(weights) == 5


class TestSectorConstraints:
    """Tests for sector constraint enforcement in optimizer."""

    def test_sector_constraint_limits_concentration(self, config):
        """Sector constraint should prevent over-concentration.

        Uses 4 sectors so the constraint (30% per sector) is feasible with sum=1.
        Without constraints, Tech (highest alpha) would dominate at ~60%.
        """
        opt = PortfolioOptimizer(config)
        selected = ["A", "B", "C", "D", "E"]
        # Strong alpha for Tech stocks
        scores = pd.Series({"A": 3.0, "B": 2.5, "C": 1.0, "D": 0.8, "E": 0.6})
        cov = pd.DataFrame(np.eye(5) * 0.04 / 252, index=selected, columns=selected)
        sector_map = pd.Series({
            "A": "Tech", "B": "Tech",
            "C": "Health", "D": "Finance", "E": "Energy"
        })

        weights = opt.optimize_weights(
            selected, scores, cov, sector_map=sector_map
        )
        tech_weight = weights[["A", "B"]].sum()
        # max_sector_weight in test config is 0.30
        assert tech_weight <= config["risk"]["max_sector_weight"] + 0.02  # small tolerance

    def test_optimizer_works_without_sector_map(self, config):
        """Optimizer should function normally when no sector_map is provided."""
        opt = PortfolioOptimizer(config)
        selected = ["A", "B", "C", "D", "E"]
        scores = pd.Series({"A": 2.0, "B": 1.5, "C": 1.0, "D": 0.5, "E": 0.3})
        cov = pd.DataFrame(np.eye(5) * 0.04 / 252, index=selected, columns=selected)

        weights = opt.optimize_weights(selected, scores, cov, sector_map=None)
        assert abs(weights.sum() - 1.0) < 1e-6


class TestLedoitWolfShrinkage:
    """Tests for the Ledoit-Wolf covariance estimator."""

    def test_shrinkage_produces_valid_covariance(self):
        """Shrinkage output should be symmetric, positive semi-definite."""
        np.random.seed(42)
        T, N = 126, 18
        returns = pd.DataFrame(
            np.random.normal(0, 0.02, (T, N)),
            columns=[f"S{i}" for i in range(N)],
        )
        cov = _ledoit_wolf_shrinkage(returns)

        # Symmetric
        assert np.allclose(cov.values, cov.values.T, atol=1e-10)

        # Positive semi-definite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert np.all(eigenvalues >= -1e-10)

    def test_shrinkage_reduces_condition_number(self):
        """Shrinkage should improve the condition number vs sample covariance."""
        np.random.seed(42)
        T, N = 60, 18  # Very low T/N ratio
        returns = pd.DataFrame(
            np.random.normal(0, 0.02, (T, N)),
            columns=[f"S{i}" for i in range(N)],
        )

        sample_cov = returns.cov()
        shrunk_cov = _ledoit_wolf_shrinkage(returns)

        cond_sample = np.linalg.cond(sample_cov.values)
        cond_shrunk = np.linalg.cond(shrunk_cov.values)

        # Shrinkage should reduce condition number
        assert cond_shrunk < cond_sample

    def test_shrinkage_correct_shape(self):
        """Output should have same shape and labels as input columns."""
        np.random.seed(42)
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        returns = pd.DataFrame(
            np.random.normal(0, 0.02, (100, 4)),
            columns=symbols,
        )
        cov = _ledoit_wolf_shrinkage(returns)
        assert cov.shape == (4, 4)
        assert list(cov.index) == symbols
        assert list(cov.columns) == symbols

    def test_compute_covariance_method_parameter(self, config):
        """PortfolioOptimizer.compute_covariance should respect method parameter."""
        opt = PortfolioOptimizer(config)
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.normal(0, 0.02, (200, 5)),
            columns=["A", "B", "C", "D", "E"],
        )

        cov_lw = opt.compute_covariance(returns, method="ledoit_wolf")
        cov_sample = opt.compute_covariance(returns, method="sample")

        # Both should be valid covariance matrices
        assert cov_lw.shape == (5, 5)
        assert cov_sample.shape == (5, 5)
        # But they should differ (shrinkage effect)
        assert not np.allclose(cov_lw.values, cov_sample.values)


class TestRiskMonitor:
    def test_drawdown_breach_detected(self, config):
        monitor = RiskMonitor(config)
        # Equity drops 25% from peak
        eq = pd.Series([100, 110, 105, 90, 82, 80])
        assert monitor.check_drawdown(eq) is True

    def test_no_breach(self, config):
        monitor = RiskMonitor(config)
        eq = pd.Series([100, 102, 104, 103, 105])
        assert monitor.check_drawdown(eq) is False

    def test_var_cvar_computation(self, config):
        """VaR and CVaR should be computed correctly."""
        monitor = RiskMonitor(config)
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0005, 0.01, 500))

        result = monitor.compute_var_cvar(returns, confidence=0.95)
        assert "var" in result
        assert "cvar" in result
        # VaR should be negative (loss)
        assert result["var"] < 0
        # CVaR should be more negative than VaR (worse tail)
        assert result["cvar"] <= result["var"]

    def test_var_cvar_insufficient_data(self, config):
        """VaR/CVaR should return NaN for insufficient data."""
        monitor = RiskMonitor(config)
        returns = pd.Series([0.01, -0.01])
        result = monitor.compute_var_cvar(returns)
        assert np.isnan(result["var"])

    def test_hhi_equal_weight(self, config):
        """HHI for equal-weight portfolio should be 1/N."""
        monitor = RiskMonitor(config)
        n = 10
        weights = pd.Series(np.full(n, 1.0 / n))
        hhi = monitor.compute_hhi(weights)
        assert abs(hhi - 1.0 / n) < 1e-10

    def test_hhi_concentrated(self, config):
        """HHI for concentrated portfolio should be high."""
        monitor = RiskMonitor(config)
        weights = pd.Series({"A": 0.9, "B": 0.05, "C": 0.05})
        hhi = monitor.compute_hhi(weights)
        assert hhi > 0.5  # Very concentrated

    def test_sector_concentration_check(self, config):
        """Sector concentration check should detect breaches."""
        monitor = RiskMonitor(config)
        weights = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        sector_map = pd.Series({"A": "Tech", "B": "Tech", "C": "Tech", "D": "Finance"})

        result = monitor.check_sector_concentration(weights, sector_map)
        # Tech = 0.75, exceeds max_sector_weight of 0.30
        assert "Tech" in result["breaches"]
        assert result["max_sector_weight"] > config["risk"]["max_sector_weight"]

    def test_sector_concentration_no_breach(self, config):
        """No breach when sectors are balanced."""
        monitor = RiskMonitor(config)
        weights = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        sector_map = pd.Series({
            "A": "Tech", "B": "Finance", "C": "Health", "D": "Energy"
        })

        result = monitor.check_sector_concentration(weights, sector_map)
        assert len(result["breaches"]) == 0

    def test_comprehensive_risk_report(self, config):
        """Risk report should include all key metrics."""
        monitor = RiskMonitor(config)
        np.random.seed(42)

        symbols = ["A", "B", "C"]
        weights = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})
        returns = pd.DataFrame(
            np.random.normal(0, 0.01, (252, 3)),
            columns=symbols,
        )
        equity = pd.Series(np.cumprod(1 + np.random.normal(0.0003, 0.01, 252)))
        sector_map = pd.Series({"A": "Tech", "B": "Tech", "C": "Finance"})

        report = monitor.compute_risk_report(
            weights, returns, equity_curve=equity, sector_map=sector_map
        )
        assert "var_95" in report
        assert "cvar_95" in report
        assert "hhi" in report
        assert "effective_n" in report
        assert "annualized_vol" in report
        assert "sector_weights" in report
        assert "max_drawdown" in report

    def test_factor_exposure_computation(self, config):
        """Factor exposure regression should produce valid betas."""
        np.random.seed(42)
        n_days = 252
        symbols = ["A", "B", "C"]
        weights = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})

        # Simulate asset returns with known factor exposure
        mkt = np.random.normal(0.0003, 0.01, n_days)
        smb = np.random.normal(0, 0.005, n_days)
        hml = np.random.normal(0, 0.005, n_days)

        dates = pd.bdate_range("2023-01-01", periods=n_days)
        returns = pd.DataFrame({
            "A": 1.2 * mkt + 0.5 * smb + np.random.normal(0, 0.005, n_days),
            "B": 0.8 * mkt - 0.3 * hml + np.random.normal(0, 0.005, n_days),
            "C": 1.0 * mkt + 0.2 * smb + np.random.normal(0, 0.005, n_days),
        }, index=dates)

        factor_returns = pd.DataFrame({
            "Mkt-RF": mkt,
            "SMB": smb,
            "HML": hml,
        }, index=dates)

        result = RiskMonitor.compute_factor_exposures(
            weights, returns, factor_returns, window=252
        )
        assert "betas" in result
        assert "Mkt-RF" in result["betas"]
        # Market beta should be close to weighted average (0.4*1.2 + 0.3*0.8 + 0.3*1.0 = 1.02)
        assert 0.7 < result["betas"]["Mkt-RF"] < 1.4
        assert result["r_squared"] > 0.3  # Should explain meaningful variance

    def test_factor_exposure_no_factor_data(self, config):
        """Factor exposure should handle missing factor data gracefully."""
        weights = pd.Series({"A": 0.5, "B": 0.5})
        returns = pd.DataFrame(
            np.random.normal(0, 0.01, (100, 2)),
            columns=["A", "B"],
        )
        result = RiskMonitor.compute_factor_exposures(weights, returns)
        assert result["betas"] == {}
        assert np.isnan(result["r_squared"])
