"""Portfolio construction and risk management.

Implements a constrained mean-variance-style optimizer that converts alpha
scores into target portfolio weights, subject to risk limits.

Enhancements (Phase 2 Portfolio Audit):
- Ledoit-Wolf shrinkage covariance estimation
- Transaction cost penalty in optimization objective
- Sector constraint enforcement in optimizer
- Enhanced RiskMonitor with VaR/CVaR, HHI, factor exposure framework
- Regularization for optimizer robustness
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def _ledoit_wolf_shrinkage(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute Ledoit-Wolf shrinkage covariance estimator.

    Uses the analytical formula from Ledoit & Wolf (2004) "A well-conditioned
    estimator for large-dimensional covariance matrices" to shrink the sample
    covariance toward a structured target (scaled identity).

    This is the single highest-impact improvement for portfolio stability when
    T/N ratio is low (e.g., 126 observations / 18 assets = 7.0).

    Parameters
    ----------
    returns : DataFrame
        T x N matrix of asset returns.

    Returns
    -------
    DataFrame
        Shrinkage covariance matrix (N x N), with same index/columns as input.
    """
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(returns.dropna())
        cov_shrunk = lw.covariance_
        shrinkage_intensity = lw.shrinkage_
        logger.debug("Ledoit-Wolf shrinkage intensity: %.4f", shrinkage_intensity)
        return pd.DataFrame(cov_shrunk, index=returns.columns, columns=returns.columns)
    except ImportError:
        logger.warning("sklearn not available; falling back to analytical Ledoit-Wolf")

    # Analytical Ledoit-Wolf (no sklearn dependency)
    X = returns.dropna().values
    T, N = X.shape
    if T < 2:
        return returns.cov()

    # De-mean
    X = X - X.mean(axis=0)

    # Sample covariance (1/T, not 1/(T-1), per Ledoit-Wolf convention)
    sample_cov = (X.T @ X) / T

    # Target: scaled identity (average variance on diagonal)
    mu = np.trace(sample_cov) / N
    target = mu * np.eye(N)

    # Compute optimal shrinkage intensity
    # delta = ||sample_cov - target||^2 (Frobenius norm squared, scaled)
    delta = np.sum((sample_cov - target) ** 2) / N

    # Estimate squared Frobenius norm of the error
    X2 = X ** 2
    # sum of squared (x_i x_j - sigma_ij)^2
    phi = np.sum((X2.T @ X2) / T - sample_cov ** 2) / N

    # Shrinkage intensity: kappa = phi / delta, clamped to [0, 1]
    kappa = max(0.0, min(1.0, phi / (T * delta))) if delta > 0 else 1.0

    cov_shrunk = kappa * target + (1 - kappa) * sample_cov
    logger.debug("Analytical Ledoit-Wolf shrinkage intensity: %.4f", kappa)

    return pd.DataFrame(cov_shrunk, index=returns.columns, columns=returns.columns)


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

        # Turnover penalty coefficient for transaction cost awareness
        # Default: txn_cost in decimal * 5 (aggressive penalty to reduce churn)
        self.turnover_penalty = pcfg.get("turnover_penalty",
                                         self.txn_cost_bps / 10000 * 5)
        self.max_turnover = pcfg.get("max_turnover_per_rebalance", 1.0)

    def select_top_stocks(self, scores: pd.Series) -> pd.Index:
        """Pick the top N stocks by composite alpha score."""
        valid = scores.dropna().sort_values(ascending=False)
        return valid.head(self.max_positions).index

    def optimize_weights(
        self,
        selected: list[str],
        scores: pd.Series,
        cov_matrix: pd.DataFrame,
        prev_weights: Optional[pd.Series] = None,
        sector_map: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Mean-variance optimization with transaction cost penalty and
        sector constraints.

        Objective: max alpha'w - lambda * w'Sigma*w - gamma * |w - w_prev| * TC

        Falls back to score-proportional weights if optimization fails.

        Parameters
        ----------
        selected : list of str
            Symbols to include in portfolio.
        scores : Series
            Alpha scores for all symbols.
        cov_matrix : DataFrame
            Covariance matrix (daily returns).
        prev_weights : Series, optional
            Previous portfolio weights for turnover penalty.
        sector_map : Series, optional
            Mapping of symbol -> sector for sector constraints.
        """
        n = len(selected)
        if n == 0:
            return pd.Series(dtype=float)

        alpha = scores.reindex(selected).fillna(0).values
        cov = cov_matrix.reindex(index=selected, columns=selected).fillna(0).values

        # Regularize covariance: add small ridge to diagonal for numerical stability
        ridge = 1e-6 * np.trace(cov) / n if np.trace(cov) > 0 else 1e-8
        cov_reg = cov + ridge * np.eye(n)

        # Previous weights for turnover penalty
        if prev_weights is not None:
            w_prev = prev_weights.reindex(selected).fillna(0).values
        else:
            w_prev = np.full(n, 1.0 / n)

        # Objective: maximize alpha'w - lambda * w'Cov*w - gamma * |w - w_prev|
        risk_aversion = 1.0
        turnover_penalty = self.turnover_penalty

        def neg_utility(w):
            ret = alpha @ w
            risk = w @ cov_reg @ w
            turnover_cost = turnover_penalty * np.sum(np.abs(w - w_prev))
            return -(ret - risk_aversion * risk - turnover_cost)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # fully invested
        ]

        # Max turnover constraint
        if self.max_turnover < 1.0:
            constraints.append({
                "type": "ineq",
                "fun": lambda w: self.max_turnover - np.sum(np.abs(w - w_prev)),
            })

        # Sector constraints: max weight per sector
        if sector_map is not None:
            sectors = sector_map.reindex(selected).dropna()
            for sector_name in sectors.unique():
                sector_mask = (sectors == sector_name).reindex(selected).fillna(False).values
                if sector_mask.sum() > 0:
                    constraints.append({
                        "type": "ineq",
                        "fun": lambda w, m=sector_mask: self.max_sector_weight - np.sum(w[m]),
                    })

        bounds = [(self.min_weight, self.max_weight)] * n
        w0 = np.full(n, 1.0 / n)

        try:
            result = minimize(neg_utility, w0, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"maxiter": 1000, "ftol": 1e-10})
            if result.success:
                weights = pd.Series(result.x, index=selected)
            else:
                logger.warning("Optimization did not converge (status=%d: %s), "
                               "using score-proportional weights",
                               result.status, result.message)
                weights = self._score_proportional(selected, scores, sector_map)
        except Exception as e:
            logger.warning("Optimization failed (%s), using score-proportional weights", e)
            weights = self._score_proportional(selected, scores, sector_map)

        # Enforce constraints post-hoc
        weights = weights.clip(lower=self.min_weight, upper=self.max_weight)
        weights /= weights.sum()

        return weights

    def _score_proportional(self, selected: list[str], scores: pd.Series,
                             sector_map: Optional[pd.Series] = None) -> pd.Series:
        """Fallback: weights proportional to alpha scores (shifted positive).

        Also enforces sector constraints by iteratively capping sector weights
        and redistributing excess to other sectors.
        """
        s = scores.reindex(selected).fillna(0)
        s = s - s.min() + 1e-6
        w = s / s.sum()
        w = w.clip(lower=self.min_weight, upper=self.max_weight)
        w = w / w.sum()

        # Enforce sector constraints if sector_map is provided
        if sector_map is not None:
            sectors = sector_map.reindex(selected).dropna()
            for _ in range(10):  # iterate to convergence
                breached = False
                for sector_name in sectors.unique():
                    mask = sectors == sector_name
                    sector_w = w[mask].sum()
                    if sector_w > self.max_sector_weight + 1e-6:
                        # Scale down this sector's weights to the limit
                        scale = self.max_sector_weight / sector_w
                        w[mask] = w[mask] * scale
                        breached = True
                if not breached:
                    break
                # Renormalize to sum to 1
                w = w / w.sum()
            w = w.clip(lower=self.min_weight, upper=self.max_weight)
            w = w / w.sum()

        return w

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

    def compute_covariance(self, returns: pd.DataFrame, window: int = 126,
                           method: str = "ledoit_wolf") -> pd.DataFrame:
        """Compute covariance matrix with optional shrinkage.

        Parameters
        ----------
        returns : DataFrame
            Daily returns (dates x symbols).
        window : int
            Lookback window in trading days.
        method : str
            'ledoit_wolf' (default, recommended), 'ewm', or 'sample'.
        """
        ret_window = returns.tail(window).dropna(how="all")

        if method == "ledoit_wolf":
            return _ledoit_wolf_shrinkage(ret_window)
        elif method == "ewm":
            return ret_window.ewm(span=window).cov().iloc[-len(ret_window.columns):]
        else:
            return ret_window.cov()


class RiskMonitor:
    """Monitors portfolio-level risk metrics in real time.

    Enhanced with VaR/CVaR, HHI concentration, and factor exposure framework.
    """

    def __init__(self, config: dict):
        self.max_drawdown = config["risk"]["max_drawdown_limit"]
        self.max_sector_weight = config["risk"].get("max_sector_weight", 0.40)

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

    def compute_var_cvar(self, returns: pd.Series, confidence: float = 0.95) -> dict:
        """Compute Value-at-Risk and Conditional VaR (Expected Shortfall).

        Parameters
        ----------
        returns : Series
            Portfolio returns (daily).
        confidence : float
            Confidence level (default 0.95 for 95% VaR).

        Returns
        -------
        dict with keys 'var', 'cvar', 'confidence'.
        """
        ret = returns.dropna()
        if len(ret) < 20:
            return {"var": np.nan, "cvar": np.nan, "confidence": confidence}

        alpha = 1 - confidence
        var = ret.quantile(alpha)
        cvar = ret[ret <= var].mean()

        return {
            "var": var,
            "cvar": cvar,
            "confidence": confidence,
        }

    def compute_hhi(self, weights: pd.Series) -> float:
        """Compute Herfindahl-Hirschman Index for portfolio concentration.

        HHI ranges from 1/N (equal weight) to 1.0 (single position).
        For an 18-position portfolio, HHI = 0.056 is perfectly diversified.
        HHI > 0.15 suggests meaningful concentration.
        """
        w = weights.dropna()
        if w.empty:
            return 0.0
        return float((w ** 2).sum())

    def check_sector_concentration(
        self, weights: pd.Series, sector_map: pd.Series
    ) -> dict:
        """Check sector concentration against limits.

        Returns
        -------
        dict with 'sector_weights', 'breaches' (list of sectors over limit),
        and 'max_sector_weight'.
        """
        if sector_map is None or sector_map.empty:
            return {"sector_weights": {}, "breaches": [], "max_sector_weight": 0.0}

        sectors = sector_map.reindex(weights.index).dropna()
        sector_weights = {}
        for sector in sectors.unique():
            mask = sectors == sector
            sector_weights[sector] = float(weights[mask].sum())

        breaches = [s for s, w in sector_weights.items() if w > self.max_sector_weight]
        max_sw = max(sector_weights.values()) if sector_weights else 0.0

        if breaches:
            logger.warning("Sector concentration breach: %s (limit %.0f%%)",
                           {s: f"{sector_weights[s]:.1%}" for s in breaches},
                           self.max_sector_weight * 100)

        return {
            "sector_weights": sector_weights,
            "breaches": breaches,
            "max_sector_weight": max_sw,
        }

    def compute_risk_report(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        equity_curve: Optional[pd.Series] = None,
        sector_map: Optional[pd.Series] = None,
    ) -> dict:
        """Comprehensive risk report for a portfolio.

        Parameters
        ----------
        weights : Series
            Current portfolio weights.
        returns : DataFrame
            Historical returns (dates x symbols).
        equity_curve : Series, optional
            Portfolio equity curve for drawdown analysis.
        sector_map : Series, optional
            Symbol -> sector mapping.

        Returns
        -------
        dict with risk metrics.
        """
        report = {}

        # Portfolio returns
        port_returns = (returns[weights.index] * weights).sum(axis=1).dropna()

        # VaR / CVaR
        var_cvar = self.compute_var_cvar(port_returns)
        report["var_95"] = var_cvar["var"]
        report["cvar_95"] = var_cvar["cvar"]

        # Annualized vol
        report["annualized_vol"] = float(port_returns.std() * np.sqrt(252))

        # HHI concentration
        report["hhi"] = self.compute_hhi(weights)

        # Effective N (1/HHI = equivalent number of equal-weight positions)
        report["effective_n"] = 1.0 / report["hhi"] if report["hhi"] > 0 else 0

        # Sector concentration
        if sector_map is not None:
            sector_info = self.check_sector_concentration(weights, sector_map)
            report["sector_weights"] = sector_info["sector_weights"]
            report["sector_breaches"] = sector_info["breaches"]
            report["max_sector_weight"] = sector_info["max_sector_weight"]

        # Drawdown
        if equity_curve is not None and not equity_curve.empty:
            peak = equity_curve.cummax()
            dd = (equity_curve - peak) / peak
            report["current_drawdown"] = float(dd.iloc[-1])
            report["max_drawdown"] = float(dd.min())
            report["drawdown_breached"] = self.check_drawdown(equity_curve)

        return report

    @staticmethod
    def compute_factor_exposures(
        weights: pd.Series,
        returns: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame] = None,
        window: int = 126,
    ) -> dict:
        """Estimate portfolio factor exposures via regression.

        If factor_returns is provided (e.g., Fama-French 5 factors), computes
        betas via OLS regression of portfolio returns on factor returns.

        Parameters
        ----------
        weights : Series
            Portfolio weights.
        returns : DataFrame
            Asset returns.
        factor_returns : DataFrame, optional
            Factor return time series (dates x factors). Columns should be
            factor names like 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'.
        window : int
            Lookback window for regression.

        Returns
        -------
        dict with 'betas' (dict of factor -> beta), 'r_squared', 'residual_vol'.
        """
        port_ret = (returns[weights.index] * weights).sum(axis=1).dropna()

        if factor_returns is None or factor_returns.empty:
            return {"betas": {}, "r_squared": np.nan, "residual_vol": np.nan}

        # Align dates
        common = port_ret.index.intersection(factor_returns.index)
        if len(common) < 30:
            return {"betas": {}, "r_squared": np.nan, "residual_vol": np.nan}

        y = port_ret.loc[common].tail(window).values
        X = factor_returns.loc[common].tail(window).values
        n = len(y)
        if n < X.shape[1] + 1:
            return {"betas": {}, "r_squared": np.nan, "residual_vol": np.nan}

        # Add intercept
        X_aug = np.column_stack([np.ones(n), X])

        try:
            betas, residuals, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            y_hat = X_aug @ betas
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            beta_dict = {col: betas[i + 1] for i, col in enumerate(factor_returns.columns)}
            residual_vol = np.sqrt(ss_res / (n - X.shape[1] - 1)) * np.sqrt(252)

            return {
                "betas": beta_dict,
                "alpha": betas[0] * 252,  # annualized
                "r_squared": r_sq,
                "residual_vol": residual_vol,
            }
        except np.linalg.LinAlgError:
            return {"betas": {}, "r_squared": np.nan, "residual_vol": np.nan}
