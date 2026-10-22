"""Tests for portfolio optimizer and risk management."""

import logging

import numpy as np
import pandas as pd
import pytest

import quant.portfolio.optimizer as optimizer_module
from quant.portfolio.optimizer import (
    PortfolioOptimizer,
    RiskMonitor,
    SolveStatus,
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
        symbols = list("ABCDE")
        weights = pd.Series(0.2, index=symbols)
        # Very low vol covariance -> scale should exceed 1.0
        cov = pd.DataFrame(
            np.eye(len(symbols)) * 0.0001,
            index=symbols,
            columns=symbols,
        )
        scaled = opt.apply_vol_scaling(weights, cov, regime="low_vol")
        assert scaled.sum() > 1.0  # leveraged
        assert scaled.sum() <= config["leverage"]["regime_leverage_caps"]["low_vol"] + 0.01

    def test_vol_scaling_respects_safety_position_cap(self, config):
        """Leveraged weights must not exceed the pre-trade concentration limit,
        otherwise the broker rejects those orders and the portfolio drifts."""
        config = {**config, "safety": {"max_position_pct_of_portfolio": 0.20}}
        opt = PortfolioOptimizer(config)
        # One position whose levered weight breaches the cap, eight that don't
        syms = list("ABCDEFGHI")
        weights = pd.Series([0.20] + [0.10] * 8, index=syms)
        # Very low vol -> scale up to the low_vol leverage cap (1.5x)
        cov = pd.DataFrame(
            np.eye(9) * 0.0001 / 252,
            index=syms, columns=syms,
        )
        scaled = opt.apply_vol_scaling(weights, cov, regime="low_vol")
        assert scaled.sum() > 1.0  # still leveraged overall
        assert scaled.max() <= 0.20 + 1e-9  # but no single position above cap
        assert scaled["B"] == pytest.approx(0.15)  # uncapped positions levered

    def test_vol_scaling_no_safety_section_unchanged(self, config):
        """Without a safety section, scaling behaves exactly as before."""
        assert "safety" not in config
        opt = PortfolioOptimizer(config)
        weights = pd.Series({"A": 0.5, "B": 0.5})
        cov = pd.DataFrame(
            [[0.0001, 0.0], [0.0, 0.0001]],
            index=["A", "B"], columns=["A", "B"],
        )
        scaled = opt.apply_vol_scaling(weights, cov, regime="low_vol")
        assert scaled.max() > 0.5  # levered above original weight, uncapped

    def test_vol_scaling_does_not_recreate_sector_breach(self, config):
        opt = PortfolioOptimizer(config)
        symbols = list("ABCDE")
        weights = pd.Series(0.2, index=symbols)
        cov = pd.DataFrame(
            np.eye(len(symbols)) * 0.0001,
            index=symbols,
            columns=symbols,
        )
        sectors = pd.Series(
            {"A": "Tech", "B": "Tech", "C": "Health", "D": "Finance", "E": "Energy"}
        )

        scaled = opt.apply_vol_scaling(
            weights,
            cov,
            regime="low_vol",
            sector_map=sectors,
        )

        assert scaled[["A", "B"]].sum() <= config["risk"]["max_sector_weight"] + 1e-9
        assert scaled.max() <= config["portfolio"]["max_position_weight"] + 1e-9

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


class TestTurnoverCap:
    """enforce_turnover_cap: total two-sided turnover including exit legs
    and post-scaling leverage changes, measured on final weights."""

    def _opt(self, config, cap):
        cfg = {**config, "portfolio": {**config["portfolio"],
                                       "max_turnover_per_rebalance": cap}}
        return PortfolioOptimizer(cfg)

    @staticmethod
    def _turnover(new, prev):
        union = new.index.union(prev.index)
        return float((new.reindex(union).fillna(0)
                      - prev.reindex(union).fillna(0)).abs().sum())

    def test_under_cap_untouched(self, config):
        opt = self._opt(config, 0.40)
        prev = pd.Series({"A": 0.5, "B": 0.5})
        new = pd.Series({"A": 0.6, "B": 0.4})  # turnover 0.2 < cap
        result = opt.enforce_turnover_cap(new, prev)
        pd.testing.assert_series_equal(result, new)

    def test_exit_legs_counted_and_capped(self, config):
        """Full portfolio replacement (turnover 2.0) must be capped at 40%,
        which the old selected-only constraint could never see."""
        opt = self._opt(config, 0.40)
        prev = pd.Series({"A": 0.5, "B": 0.5})
        new = pd.Series({"C": 0.5, "D": 0.5})
        result = opt.enforce_turnover_cap(new, prev)
        assert self._turnover(result, prev) <= 0.40 + 1e-9
        # Exiting names are sold down gradually, entering names scaled in
        assert result["A"] == pytest.approx(0.4)
        assert result["C"] == pytest.approx(0.1)

    def test_leverage_change_counted(self, config):
        """Same names, leverage 1.0 -> 1.5: the vol-scaling turnover the
        in-optimizer constraint cannot see is capped here."""
        opt = self._opt(config, 0.40)
        prev = pd.Series({"A": 0.5, "B": 0.5})
        new = pd.Series({"A": 0.75, "B": 0.75})
        result = opt.enforce_turnover_cap(new, prev)
        assert self._turnover(result, prev) == pytest.approx(0.40)

    def _opt_share(self, config, cap, share):
        cfg = {**config, "portfolio": {**config["portfolio"],
                                       "max_turnover_per_rebalance": cap,
                                       "exit_turnover_share": share}}
        return PortfolioOptimizer(cfg)

    def test_exits_cannot_starve_entries(self, config):
        """A long tail of small holdings replaced by a few large ones. Liquidating
        the old names outright is worth doing, but when it could draw on the
        whole budget the entries got nothing and the book lost most of its
        exposure in one rebalance, then lost more at the next one.  The target
        re-levers by more than the cap, so the whole cap now goes to the
        entries, the book gains the cap in exposure and the old names wait for
        the next rebalance, whatever the exit share."""
        prev = pd.Series({f"OLD{i}": 0.035 for i in range(15)})   # 52.5% invested
        new = pd.Series({f"NEW{i}": 0.22 for i in range(5)})      # 110% target

        for share in (1.0, 0.6):
            result = self._opt_share(config, 0.40, share).enforce_turnover_cap(new, prev)
            assert self._turnover(result, prev) == pytest.approx(0.40)
            bought = float(result.reindex(new.index).fillna(0.0).sum())
            assert bought == pytest.approx(0.40)
            assert float(result.sum()) == pytest.approx(float(prev.sum()) + 0.40)
            assert prev.index.isin(result.index).all()

    def test_under_invested_book_closes_the_gap_first(self, config):
        """Shaped like the 22 October 2026 preview: seven small exits, the held
        names trimmed, five new entries, a 67% book asked for 77%.  Splitting
        the cap evenly sold 20% to buy 20% and left the book at 67%; now the
        buys exceed the sells by the gap and the book arrives at the target
        inside the same cap."""
        opt = self._opt_share(config, 0.40, 0.6)
        prev = pd.Series(
            {**{f"H{i}": 0.089 for i in range(6)},                    # 53.4%
             **{f"X{i}": 0.02 for i in range(7)}}                     # 14.0%
        )                                                             # 67.4% invested
        new = pd.Series(
            {**{f"H{i}": 0.06 for i in range(6)},                     # trimmed
             **{f"N{i}": 0.0824 for i in range(5)}}                   # new entries
        )                                                             # 77.2% target
        result = opt.enforce_turnover_cap(new, prev)
        assert self._turnover(result, prev) == pytest.approx(0.40)
        assert float(result.sum()) == pytest.approx(float(new.sum()))
        bought = float((result.reindex(new.index).fillna(0.0)
                        - prev.reindex(new.index).fillna(0.0)).clip(lower=0).sum())
        sold = float((prev.reindex(prev.index) - result.reindex(prev.index).fillna(0.0)).clip(lower=0).sum())
        assert bought - sold == pytest.approx(float(new.sum()) - float(prev.sum()))

    def test_capped_rebalance_is_not_a_net_sale(self, config):
        """Shaped like the 23 September 2026 preview: small exits sold in full,
        a heavy reshuffle among the held names and new entries, a 77% book asked
        to go to 98%.  The old single blend spent the budget on the exits first
        and moved everything else a fifth of the way, a net sale."""
        opt = self._opt_share(config, 0.40, 0.6)
        prev = pd.Series(
            {**{f"X{i}": 0.015 for i in range(8)},
             **{f"H{i}": 0.065 for i in range(10)}}
        )                                                             # 77% invested
        new = pd.Series(
            {**{f"H{i}": 0.02 for i in range(10)},                    # trimmed
             **{f"N{i}": 0.0975 for i in range(8)}}                   # new entries
        )                                                             # 98% target
        result = opt.enforce_turnover_cap(new, prev)
        assert self._turnover(result, prev) <= 0.40 + 1e-9
        gross = float(result.sum())
        assert gross >= float(prev.sum()) - 1e-9
        assert gross <= float(new.sum()) + 1e-9
        # Every leg moved toward its target, none past it.
        union = result.index.union(prev.index).union(new.index)
        r = result.reindex(union).fillna(0.0)
        p = prev.reindex(union).fillna(0.0)
        n = new.reindex(union).fillna(0.0)
        assert ((r - p) * (n - p) >= -1e-12).all()
        assert ((r - n) * (p - n) >= -1e-12).all()

    def test_target_that_delevers_may_sell_more_than_it_buys(self, config):
        """Selling beyond the buys is allowed exactly as far as the target's own
        gross falls, so a de-levering target still gets there."""
        opt = self._opt_share(config, 0.40, 0.6)
        prev = pd.Series({"A": 0.7, "B": 0.7})                       # 140%
        new = pd.Series({"A": 0.5, "C": 0.5})                        # 100%
        result = opt.enforce_turnover_cap(new, prev)
        assert self._turnover(result, prev) == pytest.approx(0.40)
        assert float(result.sum()) == pytest.approx(1.0)

    def test_exit_share_bounds_liquidation(self, config):
        """Liquidating exits stays inside its own share of the budget."""
        opt = self._opt_share(config, 0.40, 0.6)
        prev = pd.Series({f"OLD{i}": 0.035 for i in range(15)})
        new = pd.Series({f"NEW{i}": 0.22 for i in range(5)})
        result = opt.enforce_turnover_cap(new, prev)
        sold = float((result.reindex(prev.index).fillna(0.0) - prev).abs().sum())
        assert sold <= 0.40 * opt.exit_turnover_share + 1e-9

    def test_no_prev_weights_is_noop(self, config):
        opt = self._opt(config, 0.40)
        new = pd.Series({"A": 0.5, "B": 0.5})
        result = opt.enforce_turnover_cap(new, None)
        pd.testing.assert_series_equal(result, new)

    def test_cap_disabled_at_one(self, config):
        opt = self._opt(config, 1.0)
        prev = pd.Series({"A": 1.0})
        new = pd.Series({"B": 1.0})
        result = opt.enforce_turnover_cap(new, prev)
        pd.testing.assert_series_equal(result, new)


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

    def test_infeasible_single_sector_cap_leaves_cash(self, config):
        """Normalization must not recreate a documented sector-cap breach."""
        opt = PortfolioOptimizer(config)
        selected = ["A", "B", "C", "D", "E"]
        scores = pd.Series({symbol: 1.0 for symbol in selected})
        cov = pd.DataFrame(np.eye(5) * 0.04 / 252, index=selected, columns=selected)
        sector_map = pd.Series({symbol: "Tech" for symbol in selected})

        weights = opt.optimize_weights(
            selected, scores, cov, sector_map=sector_map
        )

        assert weights.sum() == pytest.approx(config["risk"]["max_sector_weight"])
        assert weights.sum() < 1.0

    def test_missing_sector_classifications_share_unknown_cap(self, config):
        opt = PortfolioOptimizer(config)
        selected = ["A", "B", "C", "D", "E"]
        scores = pd.Series({symbol: 1.0 for symbol in selected})
        cov = pd.DataFrame(np.eye(5) * 0.04 / 252, index=selected, columns=selected)
        sector_map = pd.Series({"A": "Tech"})

        weights = opt.optimize_weights(
            selected,
            scores,
            cov,
            sector_map=sector_map,
        )

        assert weights[["B", "C", "D", "E"]].sum() <= (
            config["risk"]["max_sector_weight"] + 1e-9
        )


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

    def test_sparse_returns_produce_finite_covariance(self):
        returns = pd.DataFrame(
            {
                "A": [0.01, np.nan, 0.02, np.inf],
                "B": [np.nan, -0.01, 0.01, 0.02],
                "C": [0.00, 0.01, np.nan, -0.01],
            }
        )

        cov = _ledoit_wolf_shrinkage(returns)

        assert cov.index.tolist() == ["A", "B", "C"]
        assert cov.columns.tolist() == ["A", "B", "C"]
        assert np.isfinite(cov.to_numpy()).all()
        assert np.allclose(cov.to_numpy(), cov.to_numpy().T, atol=1e-10)

        # Sparse-data fallback must also remain positive semi-definite.
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

    def test_missing_sector_names_share_unknown_bucket(self, config):
        monitor = RiskMonitor(config)
        weights = pd.Series({"A": 0.2, "B": 0.2, "C": 0.2})
        sector_map = pd.Series({"A": "Tech"})

        result = monitor.check_sector_concentration(weights, sector_map)

        assert result["sector_weights"]["__UNKNOWN__"] == pytest.approx(0.4)
        assert "__UNKNOWN__" in result["breaches"]

    def test_risk_report_drops_incomplete_rows_instead_of_zero_filling(self, config):
        monitor = RiskMonitor(config)
        weights = pd.Series({"A": 0.5, "B": 0.5})
        returns = pd.DataFrame(
            {
                "A": [0.10, 0.10, 0.10],
                "B": [0.10, np.nan, -0.10],
            }
        )

        report = monitor.compute_risk_report(weights, returns)

        expected = pd.Series([0.10, 0.0]).std() * np.sqrt(252)
        assert report["annualized_vol"] == pytest.approx(expected)

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


class TestLeveragedPreviousWeights:
    """The in-optimizer turnover penalty must compare composition, not gross."""

    def test_leveraged_prev_weights_do_not_break_optimization(self, config, caplog):
        """Previous book at 1.5x gross, three of its six names leaving.

        Written for the gross-gap rescaling fix, this case only ever exercised
        the one shape the old subset turnover constraint could still satisfy
        (half the book retained, so the forced buying stayed inside the 40%
        cap); it did not catch the live calls with seven new names out of
        twelve, which that constraint made infeasible.  It now also checks the
        solver's own report, not only the absence of a warning, and the
        seven-new-names shape has its own test in TestSolverConvergence.
        """
        config = dict(config)
        config["portfolio"] = dict(config["portfolio"], max_turnover_per_rebalance=0.40)
        opt = PortfolioOptimizer(config)
        selected = ["A", "B", "C", "D", "E", "F"]
        scores = pd.Series({s: v for s, v in zip(selected, [1.5, 1.2, 1.0, 0.8, 0.6, 0.4])})
        rng = np.random.default_rng(0)
        rets = pd.DataFrame(rng.normal(0, 0.01, (300, 6)), columns=selected)
        cov = opt.compute_covariance(rets, method="sample")
        # Previous book at 1.5x gross with three names that are not selected now.
        prev = pd.Series({"A": 0.3, "B": 0.3, "C": 0.3, "X": 0.2, "Y": 0.2, "Z": 0.2})
        with caplog.at_level(logging.WARNING):
            weights = opt.optimize_weights(selected, scores, cov, prev_weights=prev)
        assert "did not converge" not in caplog.text
        assert "score-proportional" not in caplog.text
        assert opt.last_solve_status.success
        assert not opt.last_solve_status.fallback
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)


class TestTurnoverCapExitStubs:
    """Exits the blend would leave below min_weight are sold in full, within budget."""

    def _opt(self, config):
        config = dict(config)
        config["portfolio"] = dict(
            config["portfolio"],
            max_turnover_per_rebalance=0.40,
            min_position_weight=0.03,
        )
        return PortfolioOptimizer(config)

    def test_small_exit_is_liquidated_and_cap_holds(self, config):
        opt = self._opt(config)
        prev = pd.Series({"A": 0.30, "B": 0.30, "C": 0.30, "D": 0.04})
        new = pd.Series({"A": 0.25, "B": 0.25, "E": 0.25, "F": 0.25})
        result = opt.enforce_turnover_cap(new, prev)
        # D: blending alone would leave 0.04 * (1 - lam) > 0; it is sold outright.
        assert "D" not in result.index
        union = result.index.union(prev.index)
        turnover = float((result.reindex(union).fillna(0) - prev.reindex(union).fillna(0)).abs().sum())
        assert turnover <= 0.40 + 1e-9

    def test_large_exit_still_blends(self, config):
        opt = self._opt(config)
        prev = pd.Series({"A": 0.50, "B": 0.50})
        new = pd.Series({"A": 0.50, "C": 0.50})
        result = opt.enforce_turnover_cap(new, prev)
        # B is worth 50%: it cannot be sold within a 40% budget, so it is only reduced.
        assert result["B"] > 0.03
        union = result.index.union(prev.index)
        turnover = float((result.reindex(union).fillna(0) - prev.reindex(union).fillna(0)).abs().sum())
        assert turnover <= 0.40 + 1e-9


class TestPositionCapMargin:
    def test_optimizer_cap_sits_below_safety_limit(self, config):
        config = dict(config)
        config["safety"] = {"max_position_pct_of_portfolio": 0.20}
        config["portfolio"] = dict(config["portfolio"], max_position_weight=0.25)
        opt = PortfolioOptimizer(config)
        assert opt._effective_position_cap() == pytest.approx(0.19)
        scaled = opt.apply_hard_exposure_limits(pd.Series({"A": 0.24, "B": 0.10}))
        assert scaled["A"] <= 0.19 + 1e-12


def _production_like(config):
    """The live multi-factor book's portfolio and risk limits on the test config."""
    cfg = dict(config)
    cfg["portfolio"] = dict(
        config["portfolio"],
        max_positions=12,
        max_position_weight=0.12,
        min_position_weight=0.03,
        max_turnover_per_rebalance=0.40,
        exit_turnover_share=0.6,
        alpha_scale=0.05,
        risk_aversion=1.0,
    )
    cfg["risk"] = dict(config["risk"], max_sector_weight=0.50)
    cfg["safety"] = {"max_position_pct_of_portfolio": 0.20}
    return cfg


def _twelve_names_seven_new(opt, seed=7):
    """Shaped like the 23 September 2026 live calls.

    Twelve names selected, seven of them new; the previous book held twelve
    names at 1.3x gross, seven of which are leaving.  The retained names
    carried 5/12 of the previous composition, so at least 58% of the subset
    had to be bought: the old in-optimizer turnover constraint (40% on the
    selected names alone) could not be satisfied and SLSQP exited with
    mode 8 on every such call.
    """
    rng = np.random.default_rng(seed)
    held = [f"H{i}" for i in range(5)]
    new = [f"N{i}" for i in range(7)]
    gone = [f"X{i}" for i in range(7)]
    selected = held + new
    sectors = pd.Series({
        **{s: "Technology" for s in held[:3] + new[:3]},
        **{s: "Healthcare" for s in held[3:] + new[3:5]},
        **{s: "Financial Services" for s in new[5:]},
        **{s: "Energy" for s in gone},
    })
    scores = pd.Series(rng.normal(0.0, 1.0, len(selected)), index=selected)
    rets = pd.DataFrame(
        rng.normal(0.0, 0.015, (126, len(selected))), columns=selected
    )
    cov = opt.compute_covariance(rets, method="ledoit_wolf")
    prev = pd.Series({s: 1.3 / 12 for s in held + gone})
    return selected, scores, cov, prev, sectors


class TestSolverConvergence:
    """The reworked objective is smooth and its constraints are feasible, so
    SLSQP converges instead of silently handing the book to the
    score-proportional fallback."""

    def test_twelve_names_seven_new_converges(self, config, caplog):
        opt = PortfolioOptimizer(_production_like(config))
        selected, scores, cov, prev, sectors = _twelve_names_seven_new(opt)
        assert opt.last_solve_status is None
        with caplog.at_level(logging.INFO, logger="quant.portfolio.optimizer"):
            weights = opt.optimize_weights(
                selected, scores, cov, prev_weights=prev, sector_map=sectors
            )
        status = opt.last_solve_status
        assert isinstance(status, SolveStatus)
        assert status.success and not status.fallback
        assert status.status == 0
        assert 0 < status.iterations <= 100
        assert status.n_assets == 12
        assert np.isfinite(status.objective)
        assert "MVO converged" in caplog.text
        assert "score-proportional" not in caplog.text
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        # The covariance now shapes the book: the solver's weights are not the
        # fallback's, which only ever saw the scores.
        fallback = opt._score_proportional(selected, scores, sectors)
        assert not np.allclose(
            weights.reindex(selected).to_numpy(),
            fallback.reindex(selected).to_numpy(),
            atol=1e-3,
        )

    def test_solution_respects_bounds_sum_and_sector_caps(self, config):
        opt = PortfolioOptimizer(_production_like(config))
        selected, scores, cov, prev, sectors = _twelve_names_seven_new(opt)
        # Make Technology the strongest sector so its 50% cap binds.
        tech = sectors.reindex(selected) == "Technology"
        scores = scores.copy()
        scores[tech[tech].index] += 3.0
        weights = opt.optimize_weights(
            selected, scores, cov, prev_weights=prev, sector_map=sectors
        )
        assert opt.last_solve_status.success
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert weights.min() >= 0.03 - 1e-6
        assert weights.max() <= 0.12 + 1e-6
        tech_weight = float(weights[tech[tech].index].sum())
        assert tech_weight <= 0.50 + 1e-6
        assert tech_weight == pytest.approx(0.50, abs=1e-3)
        for sector in sectors.unique():
            names = sectors[sectors == sector].index.intersection(weights.index)
            assert float(weights[names].sum()) <= 0.50 + 1e-6

    def test_turnover_penalty_still_anchors_on_the_previous_book(self, config):
        """The smooth penalty keeps the same role as the exact L1 term: a
        larger coefficient moves the solution closer to the previous
        composition, and both solve (no fallback masking the comparison)."""
        selected, scores, cov, prev, sectors = _twelve_names_seven_new(
            PortfolioOptimizer(_production_like(config))
        )
        turnover = {}
        for penalty in (0.0, 0.05):
            cfg = _production_like(config)
            cfg["portfolio"] = dict(cfg["portfolio"], turnover_penalty=penalty)
            opt = PortfolioOptimizer(cfg)
            weights = opt.optimize_weights(
                selected, scores, cov, prev_weights=prev, sector_map=sectors
            )
            assert opt.last_solve_status.success
            composition = prev.reindex(selected).fillna(0.0) / prev.sum()
            turnover[penalty] = float((weights.reindex(selected) - composition).abs().sum())
        assert turnover[0.05] < turnover[0.0]

    def test_min_weight_dropped_when_a_sector_cannot_carry_it(self, config):
        """Eight names in one sector capped at 30% cannot all hold 5%; the
        lower bound must give way or the program has no feasible point."""
        cfg = dict(config)
        cfg["portfolio"] = dict(config["portfolio"], max_positions=8)
        opt = PortfolioOptimizer(cfg)
        selected = [f"T{i}" for i in range(8)] + ["F0", "F1"]
        sectors = pd.Series({**{s: "Tech" for s in selected[:8]},
                             "F0": "Finance", "F1": "Finance"})
        scores = pd.Series(np.linspace(1.0, 0.0, len(selected)), index=selected)
        cov = pd.DataFrame(np.eye(len(selected)) * 0.04 / 252,
                           index=selected, columns=selected)
        weights = opt.optimize_weights(selected, scores, cov, sector_map=sectors)
        assert opt.last_solve_status.success
        assert float(weights[selected[:8]].sum()) <= 0.30 + 1e-6
        assert weights.sum() == pytest.approx(0.60, abs=1e-6)   # 30% Tech + 30% Finance


class TestRiskAversionShapesTheBook:
    """With the solver converging, the covariance term finally has an effect."""

    def _solve(self, config, risk_aversion):
        cfg = dict(config)
        cfg["portfolio"] = dict(
            config["portfolio"], risk_aversion=risk_aversion, turnover_penalty=0.0
        )
        opt = PortfolioOptimizer(cfg)
        selected = ["A", "B", "C", "D", "E"]
        scores = pd.Series({"A": 2.0, "B": 1.5, "C": 1.0, "D": 0.5, "E": 0.0})
        # The best-scored names are also the most volatile.
        vols = np.array([0.45, 0.40, 0.35, 0.15, 0.10])
        cov = pd.DataFrame(np.diag(vols ** 2) / 252.0, index=selected, columns=selected)
        weights = opt.optimize_weights(selected, scores, cov)
        assert opt.last_solve_status.success
        return weights.reindex(selected), cov.to_numpy() * 252.0

    def test_near_zero_risk_aversion_follows_the_score_ranking(self, config):
        weights, _ = self._solve(config, risk_aversion=1e-4)
        # Weights are ordered like the scores, as score-proportional weights are:
        # the best names fill up to the cap, the worst sit at the floor.
        assert np.all(np.diff(weights.to_numpy()) <= 1e-6)
        assert weights["A"] == pytest.approx(config["portfolio"]["max_position_weight"], abs=1e-4)
        assert weights["E"] == pytest.approx(config["portfolio"]["min_position_weight"], abs=1e-4)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_high_risk_aversion_diversifies(self, config):
        low, cov = self._solve(config, risk_aversion=1e-4)
        high, _ = self._solve(config, risk_aversion=50.0)
        var_low = float(low.to_numpy() @ cov @ low.to_numpy())
        var_high = float(high.to_numpy() @ cov @ high.to_numpy())
        assert var_high < var_low
        # The calmest name overtakes the top score once risk dominates.
        assert high["E"] > high["A"]
        assert high.sum() == pytest.approx(1.0, abs=1e-6)
        assert high.max() <= config["portfolio"]["max_position_weight"] + 1e-6
        assert high.min() >= config["portfolio"]["min_position_weight"] - 1e-6


class TestObjectiveGradient:
    def test_analytic_gradient_matches_finite_differences(self):
        from scipy.optimize import check_grad

        rng = np.random.default_rng(1)
        n = 8
        alpha = rng.normal(0.0, 0.05, n)
        cov = np.cov(rng.normal(0.0, 0.02, (126, n)), rowvar=False) * 252.0
        w_prev = np.full(n, 1.0 / n)
        objective, gradient = PortfolioOptimizer._objective_and_gradient(
            alpha, cov, w_prev, risk_aversion=1.0, turnover_penalty=0.005, eps=1e-3
        )
        for _ in range(5):
            w = np.abs(rng.normal(0.1, 0.05, n))
            assert check_grad(objective, gradient, w) < 1e-6
        # At the previous weights themselves, the kink of the exact L1 term.
        assert check_grad(objective, gradient, w_prev.copy()) < 1e-6
        assert np.all(np.isfinite(gradient(w_prev)))
        assert objective(w_prev) == pytest.approx(
            -alpha @ w_prev + w_prev @ cov @ w_prev
        )

    def test_soft_abs_approaches_the_l1_distance(self):
        n = 4
        alpha = np.zeros(n)
        cov = np.zeros((n, n))
        w_prev = np.zeros(n)
        objective, _ = PortfolioOptimizer._objective_and_gradient(
            alpha, cov, w_prev, risk_aversion=0.0, turnover_penalty=1.0, eps=1e-3
        )
        w = np.array([0.10, -0.05, 0.02, 0.0])
        l1 = float(np.abs(w).sum())
        assert l1 - n * 1e-3 <= objective(w) <= l1


class TestSolverFallback:
    def _case(self, config):
        opt = PortfolioOptimizer(config)
        selected = ["A", "B", "C", "D", "E"]
        scores = pd.Series({"A": 2.0, "B": 1.5, "C": 1.0, "D": 0.5, "E": 0.3})
        cov = pd.DataFrame(np.eye(5) * 0.04 / 252, index=selected, columns=selected)
        return opt, selected, scores, cov

    def test_fallback_when_the_solver_raises(self, config, monkeypatch, caplog):
        def exploding_minimize(*args, **kwargs):
            raise RuntimeError("solver exploded")

        monkeypatch.setattr(optimizer_module, "minimize", exploding_minimize)
        opt, selected, scores, cov = self._case(config)
        with caplog.at_level(logging.WARNING, logger="quant.portfolio.optimizer"):
            weights = opt.optimize_weights(selected, scores, cov)
        status = opt.last_solve_status
        assert status.fallback and not status.success
        assert status.status == -1
        assert "solver exploded" in status.message
        assert "score-proportional" in caplog.text
        expected = opt._finalize_weights(opt._score_proportional(selected, scores), None)
        pd.testing.assert_series_equal(weights, expected)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_fallback_when_the_solver_reports_no_convergence(self, config, monkeypatch, caplog):
        from scipy.optimize import OptimizeResult

        def stuck_minimize(fun, x0, *args, **kwargs):
            return OptimizeResult(
                x=np.asarray(x0, dtype=float), fun=fun(x0), success=False,
                status=9, message="Iteration limit reached", nit=500,
            )

        monkeypatch.setattr(optimizer_module, "minimize", stuck_minimize)
        opt, selected, scores, cov = self._case(config)
        with caplog.at_level(logging.WARNING, logger="quant.portfolio.optimizer"):
            weights = opt.optimize_weights(selected, scores, cov)
        status = opt.last_solve_status
        assert status.fallback and not status.success
        assert status.status == 9 and status.iterations == 500
        assert "did not converge" in caplog.text
        expected = opt._finalize_weights(opt._score_proportional(selected, scores), None)
        pd.testing.assert_series_equal(weights, expected)

    def test_status_is_serializable_for_dashboards(self, config):
        opt, selected, scores, cov = self._case(config)
        opt.optimize_weights(selected, scores, cov)
        as_dict = opt.last_solve_status.as_dict()
        assert as_dict["success"] is True
        assert set(as_dict) == {
            "success", "fallback", "status", "message",
            "iterations", "n_assets", "objective",
        }


class TestTurnoverPenaltyEpsilonConfig:
    def test_default_and_override(self, config):
        assert PortfolioOptimizer(config).turnover_penalty_epsilon == pytest.approx(0.001)
        cfg = dict(config)
        cfg["portfolio"] = dict(config["portfolio"], turnover_penalty_epsilon=0.005)
        assert PortfolioOptimizer(cfg).turnover_penalty_epsilon == pytest.approx(0.005)

    def test_validate_config_rejects_zero_epsilon(self):
        import copy

        from quant.utils.config import load_config, validate_config

        config = copy.deepcopy(load_config("config.yaml"))
        config["portfolio"]["turnover_penalty_epsilon"] = 0.002
        validate_config(config)
        config["portfolio"]["turnover_penalty_epsilon"] = 0.0
        with pytest.raises(ValueError, match="turnover_penalty_epsilon"):
            validate_config(config)
