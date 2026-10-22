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
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolveStatus:
    """Outcome of the last ``optimize_weights`` call.

    ``method`` is "proportional" when the weights are score-proportional by
    configuration (no solver call) and "mvo" when the mean-variance solver
    was asked for.  ``fallback`` is True only when the solver was asked for
    and its result could not be used, so score-proportional weights stood
    in; ``status`` and ``message`` are SLSQP's exit mode when the solver
    ran, and -1 with a reason otherwise.
    """

    method: str
    success: bool
    fallback: bool
    status: int
    message: str
    iterations: int
    n_assets: int
    objective: float

    def as_dict(self) -> dict:
        return asdict(self)


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
    columns = returns.columns
    if len(columns) == 0:
        return pd.DataFrame(index=columns, columns=columns, dtype=float)

    clean = returns.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    observations = clean.dropna(how="any")
    if len(observations) < 2:
        # A single missing quote should not make the whole risk calculation
        # crash. Mean imputation is conservative enough for this fallback and
        # keeps the estimator finite; explicit missing-data filtering happens
        # before symbols are selected by each strategy.
        observations = clean.dropna(how="all")
        observations = observations.fillna(clean.mean()).fillna(0.0)

    if len(observations) < 2:
        variances = clean.var(ddof=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame(
            np.diag(variances.reindex(columns).to_numpy(dtype=float)),
            index=columns,
            columns=columns,
        )

    try:
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(observations)
        cov_shrunk = lw.covariance_
        shrinkage_intensity = lw.shrinkage_
        logger.debug("Ledoit-Wolf shrinkage intensity: %.4f", shrinkage_intensity)
        return pd.DataFrame(cov_shrunk, index=columns, columns=columns)
    except ImportError:
        logger.warning("sklearn not available; falling back to analytical Ledoit-Wolf")
    except (ValueError, FloatingPointError) as exc:
        logger.warning(
            "sklearn Ledoit-Wolf rejected the return matrix (%s); "
            "using the analytical estimator",
            exc,
        )

    # Analytical Ledoit-Wolf (no sklearn dependency)
    X = observations.to_numpy(dtype=float)
    T, N = X.shape
    if T < 2:
        return clean.cov().fillna(0.0)

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

    return pd.DataFrame(cov_shrunk, index=columns, columns=columns)


class PortfolioOptimizer:
    """Constructs target portfolios from alpha signals with risk constraints."""

    # How target weights are built from the scores.  "proportional" takes
    # the score-proportional weights directly, which is what the live books
    # run; "mvo" runs the constrained mean-variance solver.  On the backtest
    # windows ending 2026-10-22 the solver came out below proportional
    # (multi-factor 5y +127.5% vs +151.8%, LGBM 5y +125.8% vs +137.6%), so
    # it is kept as a research option and the default stays proportional.
    ALLOCATIONS = ("proportional", "mvo")
    DEFAULT_ALLOCATION = "proportional"
    # Default width of the soft-absolute-value turnover penalty (weight units).
    TURNOVER_PENALTY_EPSILON = 0.001
    # SLSQP budget.  The problem is a strictly convex program in at most
    # max_positions variables with analytic gradients; it converges in a few
    # dozen iterations, so hitting this limit means something is wrong and
    # the fallback should take over.
    SOLVER_MAX_ITER = 500
    SOLVER_FTOL = 1e-10

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
        # Width (in weight units) of the smooth soft-absolute-value used for
        # the turnover penalty inside the optimizer: below this distance from
        # the previous weight the penalty is quadratic, above it linear.  The
        # exact |w - w_prev| is not differentiable at the previous weight,
        # which is where the solution tends to sit, and SLSQP could not find a
        # descent direction there (exit mode 8).  0.001 is 10 bps of NAV, an
        # order below the smallest trade the book places.
        self.turnover_penalty_epsilon = float(
            pcfg.get("turnover_penalty_epsilon", self.TURNOVER_PENALTY_EPSILON)
        )
        self.max_turnover = pcfg.get("max_turnover_per_rebalance", 1.0)
        # Outcome of the last optimize_weights call, for tests and dashboards.
        self.last_solve_status: Optional[SolveStatus] = None
        # Share of the turnover budget that liquidating exits may take. Selling an
        # exit stub outright is worth doing, but it must not crowd out the entry
        # legs: with the whole budget available to exits the book sells more than
        # it buys at every rebalance and bleeds exposure.
        self.exit_turnover_share = float(
            pcfg.get("exit_turnover_share", 0.6)
        )

        # Per-position hard cap from the safety config: leverage scaling must
        # not push any single position past what pre-trade checks allow,
        # otherwise those orders get rejected at the broker and the actual
        # portfolio silently drifts from the target.
        self.max_position_pct_safety = config.get("safety", {}).get(
            "max_position_pct_of_portfolio")
        # Scores are ordinal signals, not percentage returns.  Convert their
        # cross-sectional z-scores to a documented annual return scale before
        # comparing them with an annualized covariance matrix.  Previously an
        # alpha value around 1.0 dominated daily variance around 1e-4, making
        # the nominal mean-variance optimizer effectively a score sorter.
        self.alpha_scale = pcfg.get("alpha_scale", 0.05)
        self.risk_aversion = pcfg.get("risk_aversion", 1.0)
        # Allocation switch; alpha_scale, risk_aversion and the turnover
        # penalty only matter on the "mvo" path.
        raw_allocation = pcfg.get("allocation", self.DEFAULT_ALLOCATION)
        self.allocation = str(raw_allocation).strip().lower()
        if self.allocation not in self.ALLOCATIONS:
            raise ValueError(
                "portfolio.allocation must be one of "
                f"{list(self.ALLOCATIONS)}, got {raw_allocation!r}"
            )
        self._allocation_logged = False

    def select_top_stocks(self, scores: pd.Series) -> pd.Index:
        """Pick the top N stocks by composite alpha score."""
        valid = pd.to_numeric(scores, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna().sort_values(ascending=False)
        return valid.head(self.max_positions).index

    # The pre-trade check re-evaluates each order against equity and prices
    # fetched moments later than the target was computed; a target sitting
    # exactly on the safety limit is rejected by any intraday tick against it.
    SAFETY_CAP_MARGIN = 0.95

    def _effective_position_cap(self) -> float:
        """Hard per-position cap shared by optimization and post-processing.

        Sits just below the broker-side safety limit so an order sized to it
        survives the safety layer's own re-check with fresher prices.
        """
        cap = float(self.max_weight)
        if self.max_position_pct_safety is not None:
            cap = min(cap, float(self.max_position_pct_safety) * self.SAFETY_CAP_MARGIN)
        return max(0.0, cap)

    @staticmethod
    def _aligned_sectors(
        selected: list[str] | pd.Index,
        sector_map: pd.Series,
    ) -> pd.Series:
        """Align sectors and group missing classifications into one risk bucket."""
        sectors = sector_map.reindex(selected).astype("object")
        return sectors.where(sectors.notna(), "__UNKNOWN__")

    def _sector_gross_capacity(
        self,
        selected: list[str] | pd.Index,
        sector_map: Optional[pd.Series],
    ) -> float:
        """Maximum feasible gross weight under position and sector caps."""
        position_cap = self._effective_position_cap()
        if sector_map is None:
            return min(1.0, len(selected) * position_cap)
        sectors = self._aligned_sectors(selected, sector_map)
        capacity = 0.0
        for sector in sectors.unique():
            count = int((sectors == sector).sum())
            capacity += min(self.max_sector_weight, count * position_cap)
        return min(1.0, max(0.0, capacity))

    def optimize_weights(
        self,
        selected: list[str],
        scores: pd.Series,
        cov_matrix: pd.DataFrame,
        prev_weights: Optional[pd.Series] = None,
        sector_map: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Target weights for the selected names.

        With ``portfolio.allocation: proportional`` (the default, and what
        the live books run) the weights are proportional to the shifted
        scores, projected into the position bounds and sector caps, with no
        solver call: exactly the allocation the live books have always
        ended up with.  Vol scaling, the total turnover cap and the hard
        exposure limits are applied by the callers afterwards, identically
        for either allocation.

        With ``portfolio.allocation: mvo`` this is a mean-variance
        optimization with a smooth turnover penalty and sector constraints.
        It minimizes, over the selected names,

            f(w) = -alpha'w + lambda * w'Sigma*w
                   + gamma * sum_i ( sqrt((w_i - p_i)^2 + eps^2) - eps )

        subject to sum(w) = target gross, lb <= w_i <= ub and sector sums
        <= max_sector_weight, where alpha are the cross-sectional z-scores
        on the alpha_scale horizon, Sigma the annualized covariance, p the
        previous composition and gamma the turnover penalty.  The last term
        is a soft absolute value: it equals |w_i - p_i| - eps once a name
        has moved more than eps and is quadratic inside that band, so the
        objective is smooth everywhere and SLSQP, given the analytic
        gradient, converges in a few dozen iterations.

        The earlier formulation used the exact L1 distance and also imposed
        max_turnover as a constraint on these same selected names.  That
        constraint counted every entry at its full weight while the exits it
        pays for were outside the subset, so it was infeasible whenever the
        new names exceeded roughly the cap; combined with the kink of the L1
        term, every live call ended in SLSQP exit mode 8 or 9 and the book
        was silently built from score-proportional weights.  The real cap is
        applied afterwards by enforce_turnover_cap on the final weights, over
        the union of old and new holdings, so it is not needed here.

        Falls back to score-proportional weights if optimization fails; the
        outcome of every call is kept in ``last_solve_status``.

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
        target_gross = self._sector_gross_capacity(selected, sector_map)
        if target_gross <= 0:
            return pd.Series(dtype=float)

        if self.allocation == "proportional":
            if not self._allocation_logged:
                logger.info(
                    "Allocation is score-proportional by design "
                    "(portfolio.allocation=proportional): weights follow the "
                    "scores within the position and sector limits, then vol "
                    "scaling and the turnover cap apply; the mean-variance "
                    "solver is available with portfolio.allocation: mvo"
                )
                self._allocation_logged = True
            self.last_solve_status = SolveStatus(
                method="proportional", success=True, fallback=False, status=0,
                message="score-proportional allocation by design",
                iterations=0, n_assets=n, objective=float("nan"),
            )
            weights = self._score_proportional(selected, scores, sector_map)
            return self._finalize_weights(weights, sector_map)

        alpha = self._scaled_alpha(selected, scores)
        cov_reg = self._annualized_covariance(selected, cov_matrix)
        if cov_reg is None:
            self.last_solve_status = SolveStatus(
                method="mvo", success=False, fallback=True, status=-1,
                message="covariance matrix is not positive-definite",
                iterations=0, n_assets=n, objective=float("nan"),
            )
            logger.warning("Covariance matrix is not positive-definite, "
                           "using score-proportional weights")
            return self._finalize_weights(
                self._score_proportional(selected, scores, sector_map), sector_map
            )

        w_prev = self._previous_composition(selected, prev_weights, target_gross)
        lower_bound, upper_bound = self._position_bounds(selected, target_gross, sector_map)
        constraints = self._linear_constraints(selected, target_gross, sector_map)

        objective, gradient = self._objective_and_gradient(
            alpha, cov_reg, w_prev,
            risk_aversion=float(self.risk_aversion),
            turnover_penalty=float(self.turnover_penalty),
            eps=float(self.turnover_penalty_epsilon),
        )

        # Start from a point that already satisfies the bounds, the gross
        # target and the sector caps: the score-proportional weights.  The
        # program is strictly convex, so the optimum is unique and the start
        # only affects the iteration count.
        w0 = (
            self._score_proportional(selected, scores, sector_map)
            .reindex(selected).fillna(target_gross / n)
            .to_numpy(dtype=float)
            .clip(lower_bound, upper_bound)
        )

        try:
            result = minimize(
                objective, w0, jac=gradient, method="SLSQP",
                bounds=[(lower_bound, upper_bound)] * n,
                constraints=constraints,
                options={"maxiter": self.SOLVER_MAX_ITER, "ftol": self.SOLVER_FTOL},
            )
        except Exception as exc:  # any solver failure falls back
            self.last_solve_status = SolveStatus(
                method="mvo", success=False, fallback=True, status=-1,
                message=str(exc),
                iterations=0, n_assets=n, objective=float("nan"),
            )
            logger.warning("Optimization failed (%s), using score-proportional weights", exc)
            weights = self._score_proportional(selected, scores, sector_map)
            return self._finalize_weights(weights, sector_map)

        iterations = int(getattr(result, "nit", 0) or 0)
        solved = bool(result.success) and np.all(np.isfinite(result.x))
        if solved:
            weights = pd.Series(np.asarray(result.x, dtype=float), index=selected)
            self.last_solve_status = SolveStatus(
                method="mvo", success=True, fallback=False, status=int(result.status),
                message=str(result.message), iterations=iterations,
                n_assets=n, objective=float(result.fun),
            )
            logger.info(
                "MVO converged: %d names, SLSQP status %d (%s) in %d iterations, "
                "objective %.6f, composition change vs previous %.1f%%",
                n, int(result.status), result.message, iterations,
                float(result.fun), float(np.abs(result.x - w_prev).sum()) * 100,
            )
        else:
            self.last_solve_status = SolveStatus(
                method="mvo", success=False, fallback=True, status=int(result.status),
                message=str(result.message), iterations=iterations,
                n_assets=n, objective=float("nan"),
            )
            logger.warning(
                "Optimization did not converge (status=%d: %s, %d iterations), "
                "using score-proportional weights",
                int(result.status), result.message, iterations,
            )
            weights = self._score_proportional(selected, scores, sector_map)
        return self._finalize_weights(weights, sector_map)

    @staticmethod
    def _objective_and_gradient(
        alpha: np.ndarray,
        cov: np.ndarray,
        w_prev: np.ndarray,
        *,
        risk_aversion: float,
        turnover_penalty: float,
        eps: float,
    ):
        """The smooth objective and its analytic gradient, as closures.

            f(w) = -alpha'w + risk_aversion * w'cov w
                   + turnover_penalty * sum_i ( sqrt((w_i - p_i)^2 + eps^2) - eps )

            grad f(w) = -alpha + 2 * risk_aversion * cov w
                        + turnover_penalty * (w - p) / sqrt((w - p)^2 + eps^2)

        The soft absolute value is infinitely differentiable, with curvature
        at most turnover_penalty / eps, so the quasi-Newton model SLSQP builds
        stays well conditioned; at eps = 0 the term would be the exact L1
        distance whose kink at the previous weights defeated the line search.
        """
        cov = np.asarray(cov, dtype=float)
        cov_sym = 0.5 * (cov + cov.T)
        alpha = np.asarray(alpha, dtype=float)
        w_prev = np.asarray(w_prev, dtype=float)
        eps = max(float(eps), 1e-12)
        eps_sq = eps * eps
        risk_aversion = float(risk_aversion)
        turnover_penalty = float(turnover_penalty)

        def objective(w):
            delta = w - w_prev
            soft_abs = np.sqrt(delta * delta + eps_sq)
            return float(
                -alpha @ w
                + risk_aversion * (w @ cov_sym @ w)
                + turnover_penalty * np.sum(soft_abs - eps)
            )

        def gradient(w):
            delta = w - w_prev
            soft_abs = np.sqrt(delta * delta + eps_sq)
            return (
                -alpha
                + 2.0 * risk_aversion * (cov_sym @ w)
                + turnover_penalty * delta / soft_abs
            )

        return objective, gradient

    def _scaled_alpha(self, selected: list[str], scores: pd.Series) -> np.ndarray:
        """Cross-sectional z-scores of the selected names on the alpha_scale horizon."""
        raw_alpha = pd.to_numeric(
            scores.reindex(selected), errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
        alpha_std = float(np.std(raw_alpha))
        if alpha_std > 1e-12:
            alpha = (raw_alpha - float(np.mean(raw_alpha))) / alpha_std
        else:
            alpha = np.zeros_like(raw_alpha)
        return alpha * float(self.alpha_scale)

    @staticmethod
    def _annualized_covariance(
        selected: list[str], cov_matrix: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Annualized, ridge-regularized covariance; None when not positive-definite."""
        n = len(selected)
        # Input covariance is daily.  Keep alpha and risk on the same annual
        # horizon so both terms materially influence the solution.
        cov = (
            cov_matrix.reindex(index=selected, columns=selected)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .to_numpy(dtype=float)
            * 252.0
        )
        # Regularize covariance: add small ridge to diagonal for numerical stability
        ridge = 1e-6 * np.trace(cov) / n if np.trace(cov) > 0 else 1e-8
        cov_reg = cov + ridge * np.eye(n)
        eigenvalues = np.linalg.eigvalsh(cov_reg)
        if np.any(eigenvalues <= 0) or np.any(np.isnan(eigenvalues)):
            logger.warning("Covariance matrix is not positive-definite "
                           "(min eigenvalue=%.2e)", float(np.min(eigenvalues)))
            return None
        return cov_reg

    @staticmethod
    def _previous_composition(
        selected: list[str],
        prev_weights: Optional[pd.Series],
        target_gross: float,
    ) -> np.ndarray:
        """Previous weights of the selected names, rescaled to the target gross.

        The candidate weights sum to target_gross (1.0 in the usual case)
        while the previous weights carry the account's actual gross exposure,
        anywhere between 0.8x and 1.8x after vol scaling.  Compared raw, the
        penalty would charge the leverage gap on every name; scaled to the
        same gross it measures the change in composition only.  Gross changes
        are charged by enforce_turnover_cap on the final weights.  Without a
        previous book the penalty anchors on equal weights, as before.
        """
        n = len(selected)
        if prev_weights is None:
            return np.full(n, target_gross / n)
        w_prev = pd.to_numeric(
            prev_weights.reindex(selected), errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
        prev_gross = float(np.abs(prev_weights).sum())
        if prev_gross > 1e-12:
            w_prev = w_prev * (target_gross / prev_gross)
        return w_prev

    def _position_bounds(
        self,
        selected: list[str],
        target_gross: float,
        sector_map: Optional[pd.Series],
    ) -> tuple[float, float]:
        """Per-position bounds that keep the program feasible.

        The minimum weight is dropped to zero when the selected names cannot
        all carry it: either in total against the gross target, or within one
        sector against its cap.
        """
        n = len(selected)
        lower = float(self.min_weight)
        if lower * n > target_gross + 1e-12:
            lower = 0.0
        elif sector_map is not None:
            sectors = self._aligned_sectors(selected, sector_map)
            counts = sectors.value_counts()
            if float(counts.max()) * lower > self.max_sector_weight + 1e-12:
                lower = 0.0
        upper = self._effective_position_cap()
        return lower, max(lower, upper)

    def _linear_constraints(
        self,
        selected: list[str],
        target_gross: float,
        sector_map: Optional[pd.Series],
    ) -> list[dict]:
        """Gross-target equality and sector-cap inequalities with their Jacobians."""
        n = len(selected)
        ones = np.ones(n)
        constraints = [
            {
                "type": "eq",
                "fun": lambda w, target=target_gross: float(np.sum(w) - target),
                "jac": lambda w, g=ones: g,
            },
        ]
        if sector_map is not None:
            sectors = self._aligned_sectors(selected, sector_map)
            cap = float(self.max_sector_weight)
            for sector_name in sectors.unique():
                mask = (sectors == sector_name).to_numpy(dtype=float)
                if mask.sum() > 0:
                    constraints.append({
                        "type": "ineq",
                        "fun": lambda w, m=mask, c=cap: float(c - m @ w),
                        "jac": lambda w, m=mask: -m,
                    })
        return constraints

    def _score_proportional(self, selected: list[str], scores: pd.Series,
                             sector_map: Optional[pd.Series] = None) -> pd.Series:
        """Fallback: weights proportional to alpha scores (shifted positive).

        Sector excess is redistributed only into genuinely available capacity;
        if the selected universe cannot reach 100% without breaching a sector
        cap, the remainder stays in cash.
        """
        s = pd.to_numeric(scores.reindex(selected), errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)
        s = s - s.min() + 1e-6
        target_gross = self._sector_gross_capacity(selected, sector_map)
        w = self._enforce_bounds(s / s.sum(), target_sum=target_gross)
        return self._enforce_sector_caps(w, sector_map, target_sum=target_gross)

    def _enforce_bounds(
        self,
        weights: pd.Series,
        target_sum: float = 1.0,
    ) -> pd.Series:
        """Project weights into configured bounds while preserving target gross."""
        w = weights.astype(float).copy()
        n = len(w)
        if n == 0:
            return w

        target_sum = max(0.0, float(target_sum))
        lower = self.min_weight
        if lower * n > target_sum + 1e-9:
            logger.warning(
                "Minimum weights are infeasible for %.1f%% gross; allowing "
                "smaller positions",
                target_sum * 100,
            )
            lower = 0.0
        position_cap = self._effective_position_cap()
        if position_cap * n < target_sum - 1e-9:
            logger.warning(
                "Maximum weights are infeasible for %.1f%% gross; using the "
                "largest feasible allocation",
                target_sum * 100,
            )
            target_sum = position_cap * n

        total = float(w.sum())
        if total <= 0:
            w[:] = target_sum / n
        else:
            w *= target_sum / total

        for _ in range(20):
            w = w.clip(lower=lower, upper=position_cap)
            deficit = target_sum - float(w.sum())
            if abs(deficit) < 1e-10:
                break

            if deficit > 0:
                free = (position_cap - w).clip(lower=0)
            else:
                free = (w - lower).clip(lower=0)
            free_sum = float(free.sum())
            if free_sum <= 1e-12:
                break
            w = w + deficit * (free / free_sum)

        return w

    def _enforce_sector_caps(
        self,
        weights: pd.Series,
        sector_map: Optional[pd.Series],
        *,
        target_sum: float,
        position_cap: Optional[float] = None,
    ) -> pd.Series:
        """Cap sectors and redistribute only into truly available capacity."""
        if sector_map is None or weights.empty:
            return weights

        w = weights.astype(float).copy()
        sectors = self._aligned_sectors(w.index, sector_map)
        if position_cap is None:
            position_cap = self._effective_position_cap()
        position_cap = max(0.0, float(position_cap))
        for _ in range(20):
            for sector in sectors.unique():
                mask = sectors == sector
                sector_weight = float(w.loc[mask].sum())
                if sector_weight > self.max_sector_weight + 1e-12:
                    w.loc[mask] *= self.max_sector_weight / sector_weight

            deficit = target_sum - float(w.sum())
            if deficit <= 1e-10:
                break

            capacity = (position_cap - w).clip(lower=0.0)
            for sector in sectors.unique():
                mask = sectors == sector
                sector_room = max(
                    0.0, self.max_sector_weight - float(w.loc[mask].sum())
                )
                group_capacity = float(capacity.loc[mask].sum())
                if group_capacity > sector_room and group_capacity > 0:
                    capacity.loc[mask] *= sector_room / group_capacity

            capacity_sum = float(capacity.sum())
            if capacity_sum <= 1e-12:
                break
            addition = min(deficit, capacity_sum) * capacity / capacity_sum
            w += addition

        return w

    def _finalize_weights(
        self,
        weights: pd.Series,
        sector_map: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Apply fallback-safe normalization without perturbing optimized solutions."""
        target_gross = self._sector_gross_capacity(weights.index, sector_map)
        weights = self._enforce_bounds(weights, target_sum=target_gross)
        return self._enforce_sector_caps(
            weights,
            sector_map,
            target_sum=target_gross,
        )

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

    def apply_vol_scaling(
        self,
        weights: pd.Series,
        cov_matrix: pd.DataFrame,
        regime: str = "normal",
        sector_map: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Scale portfolio to target vol with dynamic leverage based on regime.

        In calm markets (low_vol regime), allows leveraging up via margin.
        In stressed markets (high_vol regime), forces de-leveraging.
        """
        selected = weights.index.tolist()
        cov = (
            cov_matrix.reindex(index=selected, columns=selected)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .values
        )
        w = weights.values
        leverage_cap = min(self.regime_caps.get(regime, 1.0), self.max_leverage)

        # Validate covariance before matrix multiplication
        if np.any(np.isnan(cov)):
            logger.warning("Covariance matrix contains NaN, skipping vol scaling")
            return self.apply_hard_exposure_limits(
                weights,
                gross_exposure_cap=leverage_cap,
                sector_map=sector_map,
            )

        port_vol = np.sqrt(w @ cov @ w) * np.sqrt(252)
        if np.isnan(port_vol) or port_vol <= 0:
            logger.warning("Portfolio vol is %.4f, skipping vol scaling", port_vol)
            return self.apply_hard_exposure_limits(
                weights,
                gross_exposure_cap=leverage_cap,
                sector_map=sector_map,
            )
        if port_vol > 0:
            scale = self.target_vol / port_vol
            scale = min(scale, leverage_cap)
            weights = weights * scale
            logger.info("Vol scaling: port_vol=%.1f%%, regime=%s, scale=%.2f, "
                        "invested=%.1f%%", port_vol * 100, regime, scale,
                        weights.sum() * 100)

        # Scaling is not allowed to recreate a position/sector breach that
        # the optimizer already removed. Excess stays in cash if the selected
        # names do not have enough compliant capacity.
        return self.apply_hard_exposure_limits(
            weights,
            gross_exposure_cap=leverage_cap,
            sector_map=sector_map,
        )

    def enforce_turnover_cap(
        self,
        weights: pd.Series,
        prev_weights: Optional[pd.Series],
        min_stub_weight: float = 0.0025,
        gross_exposure_cap: Optional[float] = None,
        sector_map: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Cap TOTAL two-sided turnover vs the previous portfolio.

        The in-optimizer turnover constraint only sees the selected symbols,
        so exit legs (held last period, not selected now) and the post-hoc
        vol/leverage scaling both escape it.  This projection runs on the
        FINAL weights (after vol scaling) over the union of old and new
        holdings, so the configured max_turnover is the real total cap.

        When the cap binds, the budget is split between the sell legs (exits
        and trims) and the buy legs so that the capped book's gross exposure
        never ends below min(previous gross, target gross): sells exceed buys
        only by as much as the target itself de-levers.  Each side then moves
        its legs part of the way toward the target.  Large exits are sold down
        over successive rebalances; small ones are sold in full, smallest
        first.  Transitional stubs below min_stub_weight are liquidated
        outright to avoid dust orders (the tiny cap overshoot is logged).
        """
        weights = self.apply_hard_exposure_limits(
            weights,
            gross_exposure_cap=gross_exposure_cap,
            sector_map=sector_map,
        )
        if prev_weights is None or weights.empty:
            return weights

        prev = prev_weights[prev_weights.abs() > 1e-12]
        if prev.empty:
            return weights

        # A normal turnover budget must never obstruct an emergency reduction
        # in leverage.  If the account starts above today's regime cap, move
        # directly to the risk-compliant target; blending would leave the
        # portfolio overexposed for several rebalance cycles.
        if (
            gross_exposure_cap is not None
            and float(prev.abs().sum()) > gross_exposure_cap + 1e-9
            and float(weights.abs().sum()) <= gross_exposure_cap + 1e-9
        ):
            logger.warning(
                "Bypassing turnover cap to reduce gross exposure from %.1f%% "
                "to %.1f%% (risk cap %.1f%%)",
                float(prev.abs().sum()) * 100,
                float(weights.abs().sum()) * 100,
                gross_exposure_cap * 100,
            )
            return weights

        if sector_map is not None:
            sectors = self._aligned_sectors(
                prev.index.union(weights.index),
                sector_map,
            )
            prev_aligned = prev.reindex(sectors.index).fillna(0.0)
            new_aligned = weights.reindex(sectors.index).fillna(0.0)
            for sector in sectors.unique():
                mask = sectors == sector
                previous_sector = float(prev_aligned.loc[mask].sum())
                new_sector = float(new_aligned.loc[mask].sum())
                if (
                    previous_sector > self.max_sector_weight + 1e-9
                    and new_sector <= self.max_sector_weight + 1e-9
                ):
                    logger.warning(
                        "Bypassing turnover cap to reduce %s sector exposure "
                        "from %.1f%% to %.1f%% (risk cap %.1f%%)",
                        sector,
                        previous_sector * 100,
                        new_sector * 100,
                        self.max_sector_weight * 100,
                    )
                    return weights

        if self.max_turnover >= 1.0:
            return weights

        union = weights.index.union(prev.index)
        w_new = weights.reindex(union).fillna(0.0)
        w_old = prev.reindex(union).fillna(0.0)
        turnover = float((w_new - w_old).abs().sum())
        if turnover <= self.max_turnover:
            return weights

        cap = self.max_turnover
        delta = w_new - w_old
        exits = union[(w_new <= 1e-12) & (w_old > 1e-12)]
        others = union.difference(exits)
        buy_legs = others[delta.loc[others] > 1e-12]
        trim_legs = others[delta.loc[others] < -1e-12]
        exit_full = float(w_old.loc[exits].sum())
        trim_full = float(-delta.loc[trim_legs].sum())
        buy_full = float(delta.loc[buy_legs].sum())

        # Split the budget between the two sides before spending it.  The cap
        # may slow the move toward the target but must not change where the
        # book's gross exposure is heading: sells may exceed buys only by as
        # much as the target itself de-levers.  Spending one blended budget
        # leg by leg did not hold that.  Exits were sold in full while entries
        # moved a fraction of the way, so every capped rebalance was a net
        # sale: on 15 September 2026 a book asked to go to 110% would have
        # landed at 41%, and after a first fix the 23 September preview still
        # took a 77% book to 68% against a 98% target.
        #
        # The same holds the other way round.  Splitting the cap evenly kept
        # an under-invested book where it was: on 23 September a 77% book was
        # asked for 98% and sold 20% to buy 20%, and the 22 October preview
        # took a 67% book to 67% against a 77% target.  So the buys may
        # exceed the sells by as much as the target itself re-levers; when
        # that gap is the whole cap or more, the cap goes to the buys and the
        # exits wait for the next rebalance.
        slack = max(0.0, float(w_old.sum()) - float(w_new.sum()))
        gap = max(0.0, float(w_new.sum()) - float(w_old.sum()))
        sell_budget = min(exit_full + trim_full, cap,
                          max(0.0, (cap + slack - gap) / 2.0),
                          buy_full + slack)

        # Sell side.  Pro rata, every exit and trim would move the same fraction
        # of the way, but blending sells an exiting position down geometrically,
        # (1 - lam) of it surviving each rebalance, so a concentrated book
        # accumulated a long tail of holdings far below the mandate's minimum
        # weight, some held against their own signal.  When the pro-rata sale
        # would leave exits below min_weight, the exits may take up to their
        # share of the budget from the trims to sell those in full, and the
        # exit sale goes to the smallest positions first, each sold in full,
        # the rest spread pro rata over the larger ones.
        sell_full = exit_full + trim_full
        lam_sell = sell_budget / sell_full if sell_full > 1e-12 else 0.0
        blended = w_old.copy()
        exit_order = w_old.loc[exits].sort_values()
        exit_sale = exit_full * lam_sell
        remainders = exit_order * (1.0 - lam_sell)
        stub_extra = float(remainders[remainders < self.min_weight].sum())
        if stub_extra > 1e-12:
            exit_budget = min(cap * self.exit_turnover_share, sell_budget)
            exit_sale = max(exit_sale, min(exit_budget, exit_sale + stub_extra))
        liquidated = []
        left = exit_sale
        for symbol, weight in exit_order.items():
            if weight > left + 1e-12:
                break
            blended.loc[symbol] = 0.0
            left -= float(weight)
            liquidated.append(symbol)
        rest = exit_order.index.difference(liquidated)
        rest_full = float(w_old.loc[rest].sum())
        if rest_full > 1e-12:
            blended.loc[rest] = w_old.loc[rest] * (1.0 - min(1.0, max(left, 0.0) / rest_full))
        lam_trim = (
            min(1.0, max(sell_budget - exit_sale, 0.0) / trim_full)
            if trim_full > 1e-12 else 0.0
        )
        blended.loc[trim_legs] = w_old.loc[trim_legs] + lam_trim * delta.loc[trim_legs]
        sold = exit_sale + lam_trim * trim_full

        # Buy side: whatever the sells left of the cap, all legs alike.
        lam_buy = (
            min(1.0, max(cap - sold, 0.0) / buy_full)
            if buy_full > 1e-12 else 0.0
        )
        blended.loc[buy_legs] = w_old.loc[buy_legs] + lam_buy * delta.loc[buy_legs]

        stubs = blended[(blended > 0) & (blended < min_stub_weight)].index
        if len(stubs) > 0:
            blended.loc[stubs] = 0.0
        blended = blended[blended.abs() > 1e-12]

        logger.info(
            "Turnover cap engaged: target turnover %.1f%% > cap %.1f%%, "
            "sells %.1f%% (exits %.1f%% with %d sold in full, trims lam=%.2f), "
            "buys %.1f%% (lam=%.2f); invested %.1f%% -> %.1f%% (target %.1f%%), "
            "%d transitional position(s), %d stub(s) liquidated",
            turnover * 100, cap * 100, sold * 100, exit_sale * 100,
            len(liquidated), lam_trim, lam_buy * buy_full * 100, lam_buy,
            float(w_old.sum()) * 100, float(blended.sum()) * 100,
            float(w_new.sum()) * 100,
            int((~blended.index.isin(weights.index)).sum()), len(stubs),
        )
        return self.apply_hard_exposure_limits(
            blended,
            gross_exposure_cap=gross_exposure_cap,
            sector_map=sector_map,
        )

    def apply_hard_exposure_limits(
        self,
        weights: pd.Series,
        gross_exposure_cap: Optional[float] = None,
        sector_map: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Enforce final broker-compatible position and gross limits.

        This is deliberately applied after volatility scaling and turnover
        blending: either post-processing step can otherwise recreate a breach
        that the optimizer itself never sees.  Excess remains in cash.
        """
        if weights is None or weights.empty:
            return weights

        limited = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        limited = limited.clip(lower=0.0)

        if self.max_position_pct_safety is not None:
            limited = limited.clip(upper=self._effective_position_cap())

        if gross_exposure_cap is None:
            gross_exposure_cap = self.max_leverage
        gross_exposure_cap = max(0.0, min(float(gross_exposure_cap), self.max_leverage))
        gross = float(limited.abs().sum())
        if gross > gross_exposure_cap + 1e-12 and gross > 0:
            limited *= gross_exposure_cap / gross

        final_position_cap = (
            self._effective_position_cap()
            if self.max_position_pct_safety is not None
            else max(float(limited.sum()), float(limited.max()))
        )
        limited = self._enforce_sector_caps(
            limited,
            sector_map,
            target_sum=float(limited.sum()),
            position_cap=final_position_cap,
        )

        return limited[limited.abs() > 1e-12]

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
        equity_curve = pd.to_numeric(equity_curve, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
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
        ret = pd.to_numeric(returns, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
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

        sectors = self._aligned_sectors(weights.index, sector_map)
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

    @staticmethod
    def _aligned_sectors(
        symbols: pd.Index,
        sector_map: pd.Series,
    ) -> pd.Series:
        """Treat missing classifications as a single monitored risk bucket."""
        sectors = sector_map.reindex(symbols).astype("object")
        return sectors.where(sectors.notna(), "__UNKNOWN__")

    @staticmethod
    def _complete_portfolio_returns(
        weights: pd.Series,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """Portfolio returns without silently treating missing assets as flat."""
        if weights.empty:
            return pd.Series(dtype=float)
        panel = returns.reindex(columns=weights.index).apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        weighted = panel.mul(weights, axis=1)
        return weighted.sum(axis=1, min_count=len(weights)).dropna()

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
        port_returns = self._complete_portfolio_returns(weights, returns)

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
        port_ret = RiskMonitor._complete_portfolio_returns(weights, returns)

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
