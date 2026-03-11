"""Portfolio construction and risk management.

Implements a constrained mean-variance-style optimizer that converts alpha
scores into target portfolio weights, subject to risk limits.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Constructs target portfolios from alpha signals with risk constraints."""

    def __init__(self, config: dict):
        pcfg = config["portfolio"]
        rcfg = config["risk"]

        self.max_positions = pcfg["max_positions"]
        self.max_weight = pcfg["max_position_weight"]
        self.min_weight = pcfg["min_position_weight"]
        self.target_vol = pcfg["target_volatility"]
        self.rebalance_freq = pcfg["rebalance_frequency_days"]
        self.txn_cost_bps = pcfg["transaction_cost_bps"]

        self.max_drawdown = rcfg["max_drawdown_limit"]
        self.max_sector_weight = rcfg["max_sector_weight"]
        self.stop_loss_pct = rcfg["stop_loss_pct"]

        # Dynamic leverage / regime config (backward-compatible defaults)
        lcfg = config.get("leverage", {})
        self.max_leverage = lcfg.get("max_leverage", 1.0)
        self.spy_vol_window = lcfg.get("regime_spy_vol_window", 63)
        self.regime_thresholds = lcfg.get("regime_thresholds", {"low": 0.12, "high": 0.20})
        self.regime_caps = lcfg.get("regime_leverage_caps",
                                    {"low_vol": 1.0, "normal": 1.0, "high_vol": 0.7})

    def select_top_stocks(self, scores: pd.Series) -> pd.Index:
        """Pick the top N stocks by composite alpha score."""
        valid = scores.dropna().sort_values(ascending=False)
        return valid.head(self.max_positions).index

    def optimize_weights(self, selected: list[str], scores: pd.Series,
                         cov_matrix: pd.DataFrame) -> pd.Series:
        """Risk-parity-tilted optimization: maximize alpha-weighted return
        subject to position and volatility constraints.

        Falls back to score-proportional weights if optimization fails.
        """
        n = len(selected)
        if n == 0:
            return pd.Series(dtype=float)

        alpha = scores.reindex(selected).fillna(0).values
        cov = cov_matrix.reindex(index=selected, columns=selected).fillna(0).values

        # Objective: maximize alpha'w - lambda * w'Cov*w
        risk_aversion = 1.0

        def neg_utility(w):
            ret = alpha @ w
            risk = w @ cov @ w
            return -(ret - risk_aversion * risk)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # fully invested
        ]
        bounds = [(self.min_weight, self.max_weight)] * n

        w0 = np.full(n, 1.0 / n)

        try:
            result = minimize(neg_utility, w0, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"maxiter": 500, "ftol": 1e-10})
            if result.success:
                weights = pd.Series(result.x, index=selected)
            else:
                logger.warning("Optimization did not converge, using score-proportional weights")
                weights = self._score_proportional(selected, scores)
        except Exception as e:
            logger.warning("Optimization failed (%s), using score-proportional weights", e)
            weights = self._score_proportional(selected, scores)

        # Enforce constraints post-hoc
        weights = weights.clip(lower=self.min_weight, upper=self.max_weight)
        weights /= weights.sum()

        return weights

    def _score_proportional(self, selected: list[str], scores: pd.Series) -> pd.Series:
        """Fallback: weights proportional to alpha scores (shifted positive)."""
        s = scores.reindex(selected).fillna(0)
        s = s - s.min() + 1e-6
        w = s / s.sum()
        return w.clip(lower=self.min_weight, upper=self.max_weight).pipe(lambda x: x / x.sum())

    def detect_regime(self, spy_returns: pd.Series) -> str:
        """Detect market regime from SPY realized volatility.

        Returns 'low_vol', 'normal', or 'high_vol'.
        """
        if spy_returns is None or len(spy_returns) < self.spy_vol_window:
            return "normal"
        recent = spy_returns.tail(self.spy_vol_window)
        spy_vol = recent.std() * np.sqrt(252)
        if spy_vol < self.regime_thresholds["low"]:
            return "low_vol"
        elif spy_vol > self.regime_thresholds["high"]:
            return "high_vol"
        return "normal"

    def apply_vol_scaling(self, weights: pd.Series,
                          cov_matrix: pd.DataFrame,
                          regime: str = "normal") -> pd.Series:
        """Scale portfolio to target vol with dynamic leverage based on regime.

        In calm markets (low_vol regime), allows leveraging up via margin.
        In stressed markets (high_vol regime), forces de-leveraging.
        """
        selected = weights.index.tolist()
        cov = cov_matrix.reindex(index=selected, columns=selected).fillna(0).values
        w = weights.values

        leverage_cap = min(self.regime_caps.get(regime, 1.0), self.max_leverage)
        port_vol = np.sqrt(w @ cov @ w) * np.sqrt(252)
        if port_vol > 0:
            scale = self.target_vol / port_vol
            scale = min(scale, leverage_cap)
            weights = weights * scale
            logger.info("Vol scaling: port_vol=%.1f%%, regime=%s, scale=%.2f, "
                        "invested=%.1f%%", port_vol * 100, regime, scale,
                        weights.sum() * 100)

        return weights

    def check_stop_losses(self, current_weights: pd.Series,
                          entry_prices: pd.Series,
                          current_prices: pd.Series) -> pd.Series:
        """Zero out positions that have hit the stop-loss threshold."""
        if entry_prices.empty or current_prices.empty:
            return current_weights

        returns_since_entry = (current_prices / entry_prices) - 1.0
        stopped = returns_since_entry < -self.stop_loss_pct
        if stopped.any():
            logger.info("Stop-loss triggered for: %s", stopped[stopped].index.tolist())
            current_weights[stopped] = 0.0
            if current_weights.sum() > 0:
                current_weights /= current_weights.sum()

        return current_weights

    def compute_covariance(self, returns: pd.DataFrame, window: int = 126) -> pd.DataFrame:
        """Exponentially-weighted covariance matrix."""
        return returns.ewm(span=window).cov().iloc[-len(returns.columns):]


class RiskMonitor:
    """Monitors portfolio-level risk metrics in real time."""

    def __init__(self, config: dict):
        self.max_drawdown = config["risk"]["max_drawdown_limit"]

    def check_drawdown(self, equity_curve: pd.Series) -> bool:
        """Return True if max drawdown limit has been breached."""
        if equity_curve.empty:
            return False
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        current_dd = drawdown.iloc[-1]
        if current_dd < -self.max_drawdown:
            logger.warning("MAX DRAWDOWN BREACHED: %.2f%% (limit %.2f%%)",
                           current_dd * 100, self.max_drawdown * 100)
            return True
        return False
