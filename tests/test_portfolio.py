"""Tests for portfolio optimizer and risk management."""

import numpy as np
import pandas as pd
import pytest

from quant.portfolio.optimizer import PortfolioOptimizer, RiskMonitor


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
        # Very low vol covariance → scale should exceed 1.0
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
        # Very low portfolio vol → would want to lever up
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
