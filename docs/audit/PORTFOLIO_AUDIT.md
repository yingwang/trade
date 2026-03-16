# Portfolio Construction & Risk Management Audit

**Auditor**: Portfolio & Risk Agent (ex-DE Shaw portfolio construction and risk management)
**Date**: 2026-03-15
**Scope**: MVO optimizer, risk management, constraints, vol scaling, transaction costs

---

## Executive Summary

The portfolio construction module has a sound architectural skeleton -- MVO optimization with position bounds, vol-targeting with regime detection, stop-loss logic, and a basic risk monitor. However, the implementation has **5 high-severity issues** and **several moderate gaps** that collectively undermine portfolio stability, inflate backtest performance, and create meaningful risk in live deployment.

The most impactful problems are:

1. **Raw sample covariance** with a T/N ratio of 7 (126 days / 18 assets) produces unstable weight estimates that can flip dramatically between rebalances.
2. **No transaction costs in the optimization objective**, meaning the optimizer freely churns the portfolio while the backtest silently charges 15bps per rebalance.
3. **Sector constraint is configured but never enforced** in the optimizer -- a silent config-to-code disconnect.
4. **RiskMonitor exists but is never called** in the backtest loop, making the 20% drawdown halt and 12% stop-loss into dead code for backtesting.
5. **Alpha scores used directly as expected returns** in MVO, but they are z-scored composites (mean 0, std 1), not return forecasts. This distorts the risk-return tradeoff in the objective function.

---

## 1. MVO Implementation Audit

### 1.1 Expected Return Estimation

**Location**: `optimizer.py` lines 58, 61-67

```python
alpha = scores.reindex(selected).fillna(0).values
# Objective: maximize alpha'w - lambda * w'Cov*w
```

**Finding (HIGH)**: The `alpha` vector fed into MVO is the raw composite z-score from the signal generator. These z-scores have mean ~0 and std ~1 across the cross-section, and represent *relative rankings*, not return forecasts in units of expected return. The covariance matrix `cov` is in units of (daily return)^2.

This creates a unit mismatch in the objective function `alpha'w - lambda * w'Cov*w`. The alpha term dominates because z-scores are ~O(1) while daily covariance entries are ~O(0.0001). With `risk_aversion = 1.0`, the risk penalty is negligible relative to the alpha signal, causing the optimizer to chase high-alpha stocks with minimal risk consideration.

**Impact**: The optimizer effectively degenerates into a score-proportional allocation capped by position bounds, ignoring covariance structure. The "risk-parity-tilted" claim in the docstring is misleading.

**Recommendation**: Either (a) scale alpha scores to be in the same units as expected daily returns (multiply by an empirically calibrated IC * cross-sectional std of returns), or (b) increase risk_aversion to ~50-200 to bring the two terms into balance. Option (b) is simpler and more robust.

### 1.2 Covariance Matrix Estimation

**Location**: `strategy.py` line 106-110

```python
ret_window = returns.loc[:date].tail(126)
cov = ret_window[selected].cov()
```

**Finding (HIGH)**: Raw sample covariance with T=126 observations and N=18 assets.

- **T/N ratio**: 126/18 = 7.0. The Marchenko-Pastur distribution theory says that with T/N < ~10, a significant fraction of eigenvalues in the sample covariance are pure noise. Rule of thumb: need T/N > 10 for reasonable estimation, T/N > 20 for reliable optimization.
- **Condition number**: For a 126x18 sample covariance, the condition number is typically 50-200, meaning small perturbations in the data cause large changes in the inverse (or in optimized weights).
- **Estimation error**: Jobson & Korkie (1980) showed that with T/N = 7, the expected squared error of sample covariance relative to true covariance is substantial. The resulting MVO weights have ~50% more variance than necessary.

**Impact**: Portfolio weights are unstable between rebalances, driving unnecessary turnover and eroding returns through transaction costs. The optimizer may produce "corner solutions" that concentrate in 2-3 stocks at the upper bound.

**Fix delivered**: Ledoit-Wolf shrinkage covariance estimator. Shrinks the sample covariance toward a scaled identity matrix, dramatically reducing estimation error and improving the condition number. This is the single highest-impact improvement for portfolio stability.

### 1.3 Risk Aversion Parameter

**Location**: `optimizer.py` line 62

```python
risk_aversion = 1.0
```

**Finding (MEDIUM)**: Hardcoded at 1.0 with no sensitivity analysis. Given the unit mismatch described in 1.1, this value produces an alpha-dominated objective. In a properly calibrated system, typical risk aversion values are:
- 0.5-2.0 when alpha is in return units and covariance is annualized
- 50-500 when alpha is in z-score units and covariance is daily

**Recommendation**: Make risk_aversion a configurable parameter and document its interpretation. Consider running a grid search over [0.5, 1, 2, 5, 10, 50] and measuring weight stability and out-of-sample Sharpe.

### 1.4 Optimizer Solver

**Location**: `optimizer.py` lines 78-80

```python
result = minimize(neg_utility, w0, method="SLSQP",
                  bounds=bounds, constraints=constraints,
                  options={"maxiter": 500, "ftol": 1e-10})
```

**Finding (MEDIUM)**:
- SLSQP is a general-purpose NLP solver, not specialized for quadratic programs. For MVO (a convex QP), cvxpy with OSQP/ECOS would provide guaranteed global optimality and faster convergence.
- `ftol=1e-10` is extremely tight and may cause premature "convergence failure" reports. For portfolio weights accurate to 0.01%, `ftol=1e-8` is sufficient.
- `maxiter=500` is reasonable but may not be enough with sector constraints added.
- The initial point `w0 = 1/n` is feasible (satisfies sum-to-1 and bounds) -- this is correct.

**Recommendation**: Consider switching to cvxpy for production. For now, SLSQP with regularized covariance is acceptable. Increased maxiter to 1000 in fix.

### 1.5 Fallback Mechanism

**Location**: `optimizer.py` lines 83-88, 96-101

**Finding (LOW)**: Score-proportional weights as fallback is reasonable. The shift `s = s - s.min() + 1e-6` ensures all weights are positive, and clipping + normalization ensures constraints are respected. However, this fallback completely ignores risk and covariance, so it should be logged prominently and tracked as a metric (how often does fallback trigger?).

---

## 2. Constraint Audit

### 2.1 Position Weight Bounds

**Config**: `min_position_weight: 0.02` (2%), `max_position_weight: 0.10` (10%)

**Analysis**: For an 18-position portfolio:
- Minimum total allocation: 18 * 2% = 36% (remaining 64% is cash after vol scaling)
- Maximum total allocation: 18 * 10% = 180% (impossible due to sum-to-1)
- Equal weight: 18 * 5.56% = 100%

The bounds [2%, 10%] are reasonable for 18 positions. The 2% minimum prevents dust positions that are expensive to rebalance. The 10% maximum limits single-stock risk.

**Issue**: After vol scaling, weights may sum to > 1.0 (leveraged) or < 1.0 (de-leveraged). The bounds are enforced *before* vol scaling, so individual position weights can exceed 10% in a leveraged portfolio (e.g., 10% * 1.3x = 13%). The safety config has `max_position_pct_of_portfolio: 0.15` as a separate guard, but this is not checked in the optimizer -- only in the execution safety layer.

### 2.2 Sum-to-1 Constraint

**Finding (MEDIUM)**: The optimizer enforces `sum(w) = 1.0`, which is correct pre-vol-scaling. After vol scaling, `strategy.py` line 120 comments "Do NOT renormalize -- the remainder is held as cash." This is correct behavior: the backtest engine computes `target_shares = portfolio_value * target / px`, so if weights sum to 0.8, only 80% of capital is invested and 20% is cash. The margin interest logic (engine.py line 115) charges interest when `cash < 0`, correctly handling leverage.

**Verified**: The backtest engine handles sub-1.0 and super-1.0 weight sums correctly.

### 2.3 Sector Constraint -- NOT ENFORCED (BUG)

**Config**: `max_sector_weight: 0.40`

**Finding (HIGH / BUG)**: The `max_sector_weight` parameter is read into `self.max_sector_weight` in the optimizer constructor (line 31) but is **never used** in `optimize_weights()`. The SLSQP constraints list contains only the sum-to-1 equality constraint. There is no inequality constraint preventing a single sector from exceeding 40%.

Given the universe is heavily tech-weighted (35 of 100 stocks are in Mega-Cap Tech + Tech/Software + Semiconductors), the optimizer can easily allocate 60-80% to tech stocks if momentum and quality favor them, violating the intended 40% sector cap.

**Fix delivered**: Added sector inequality constraints to the optimizer. When `sector_map` is provided, the optimizer adds `sum(w[sector]) <= max_sector_weight` for each sector.

### 2.4 Turnover Constraint -- NONE EXISTS (BUG)

**Finding (HIGH)**: There is no turnover constraint or penalty in the optimization. The `transaction_cost_bps` parameter is loaded but only used in the backtest engine's execution simulation. This means:

1. The optimizer freely churns the portfolio, unaware of transaction costs.
2. Each rebalance may produce a completely different portfolio (especially with noisy sample covariance).
3. The backtest charges 15bps per rebalance (10bps txn + 5bps slippage), but the optimizer doesn't account for this.
4. No "no-trade zone" exists -- even a 0.01% weight change triggers a trade.

**Impact estimate**: With 18 positions and monthly rebalance, if turnover averages 50% per rebalance (common with unstable covariance), that is 50% * 15bps * 12 = 90bps annual drag. With proper turnover awareness, this could be reduced to 30-40bps.

**Fix delivered**: Added turnover penalty `gamma * |w - w_prev|` to the objective function. The penalty coefficient defaults to `2 * txn_cost_bps / 10000` (approximating round-trip cost).

---

## 3. Transaction Cost Handling

### 3.1 Current State

Transaction costs exist in two places:
- **Config**: `transaction_cost_bps: 10` (10bps)
- **Backtest engine**: Lines 96-98 apply cost as `trade_value * (txn_cost_bps + slippage_bps) / 10000`

But they are **absent** from the optimization objective. This is the classic "optimizer's curse" -- the optimizer produces a portfolio that looks optimal before costs but is suboptimal after costs.

### 3.2 No-Trade Zone

**Finding (MEDIUM)**: There is no minimum trade threshold. A weight change from 5.00% to 5.01% triggers a trade that costs more in transaction costs than it gains in alpha improvement. A no-trade zone of ~50bps around each target weight would prevent this.

**Recommendation**: In the backtest engine, add a filter: skip trades where `|target_shares - current_shares| * price < threshold` (e.g., $500 minimum trade size). This is a downstream fix, not in the optimizer.

### 3.3 Market Impact

**Finding (LOW for current scale)**: With $1M capital and 18 positions, average position size is ~$56K. With the 1% ADV limit from safety config, this is appropriate for large-cap stocks with >$50M daily volume. Market impact is negligible at this scale. However, if capital scales to $10M+, market impact modeling (e.g., Almgren-Chriss) would become necessary.

---

## 4. Risk Management Audit

### 4.1 Stop-Loss Logic -- EXISTS BUT NEVER CALLED

**Location**: `optimizer.py` lines 142-157

**Finding (HIGH)**: `check_stop_losses()` is implemented correctly -- it zeros out positions that have dropped more than `stop_loss_pct` (12%) from entry price and renormalizes remaining weights.

However, this function is **never called** in:
- `strategy.py` -- not called in `run_backtest()` or `get_current_portfolio()`
- `engine.py` -- not called in the backtest loop
- `paper_trade.py` -- not called in the rebalance flow

The stop-loss is dead code. It exists only in tests.

**Impact**: In a severe drawdown, the portfolio continues holding losing positions without intervention. The 12% stop-loss that appears in the config and documentation is never enforced.

**Recommendation**: Integrate stop-loss checking into the backtest loop (at each daily step, check entry prices vs current prices) and into the paper trading rebalance flow.

### 4.2 Drawdown Monitoring -- EXISTS BUT NOT INTEGRATED

**Location**: `optimizer.py` lines 170-181

**Finding (MEDIUM)**: `RiskMonitor.check_drawdown()` is implemented correctly -- it detects when the equity curve has fallen more than `max_drawdown_limit` (20%) from its peak. But like stop-loss, it is **never called** in the backtest loop.

The `RiskMonitor` is instantiated in `strategy.py` line 36, but `check_drawdown()` is never invoked. In the backtest engine, even if max drawdown is hit, the simulation continues trading normally.

**Fix delivered**: Added drawdown check at the end of `run_backtest()` in `strategy.py`. For full integration, the backtest engine itself would need to be modified to halt or reduce exposure when drawdown is breached -- this is a more invasive change left for future work.

### 4.3 Factor Exposure Monitoring -- DOES NOT EXIST

**Finding (MEDIUM)**: There is no Fama-French factor exposure analysis. The portfolio may have unintended exposure to market beta, size, value, momentum, or profitability factors that the operator is not aware of.

**Fix delivered**: Added `RiskMonitor.compute_factor_exposures()` that performs OLS regression of portfolio returns on factor returns. Accepts any factor return DataFrame (e.g., Fama-French 5-factor model from Ken French's website).

### 4.4 Concentration Monitoring -- PARTIAL

**Finding (MEDIUM)**: Position-level bounds (2-10%) exist, but:
- No HHI (Herfindahl-Hirschman Index) computation
- Sector constraint not enforced (see 2.3)
- No monitoring of effective N (1/HHI)

**Fix delivered**: Added `compute_hhi()` and `check_sector_concentration()` to RiskMonitor, plus a comprehensive `compute_risk_report()` that aggregates all risk metrics.

### 4.5 Liquidity Risk -- NOT ADDRESSED

**Finding (LOW for current scale)**: No check of position size vs average daily volume. The safety config has `max_adv_fraction: 0.01` but this is only enforced in the execution safety layer, not in portfolio construction. The optimizer could theoretically allocate 10% to an illiquid stock.

For a $1M portfolio with large-cap stocks, this is unlikely to be a problem. For scaling, the optimizer should incorporate a liquidity constraint.

---

## 5. Vol Scaling / Leverage Audit

### 5.1 Mechanism

**Location**: `optimizer.py` lines 118-140

```python
port_vol = sqrt(w' * Cov * w) * sqrt(252)
scale = target_vol / port_vol
scale = min(scale, leverage_cap)
weights = weights * scale
```

**Finding (MEDIUM)**: The vol scaling mechanism is conceptually sound -- it targets 15% annualized portfolio volatility and caps leverage by regime. However:

1. **Same noisy covariance**: The portfolio vol estimate `w'Cov*w` uses the same sample covariance that produced unstable weights. If covariance is noisy, vol estimates are noisy, and scaling amplifies errors. Fix: using Ledoit-Wolf shrinkage for covariance (now delivered).

2. **Vol-of-vol risk**: In transitioning regimes, the scaling factor can swing dramatically between rebalances. E.g., going from low_vol (scale=2.0x) to high_vol (scale=0.7x) in one rebalance means selling ~65% of the portfolio in one day.

3. **Leverage in stressed markets**: The `high_vol` regime cap of 0.7x is good (forced de-leverage). But the transition from `normal` (1.3x) to `high_vol` (0.7x) is abrupt. Consider a linear interpolation between regime caps based on actual SPY vol.

### 5.2 Cash Handling After Vol Scaling

**Location**: `strategy.py` lines 120-121

```python
# Do NOT renormalize after vol scaling -- the remainder is held as cash.
```

**Verified CORRECT**: When weights sum to < 1.0 (de-leveraged), the backtest engine correctly holds the remainder as cash (`cash -= trade_cash` where `trade_cash < portfolio_value`). When weights sum to > 1.0 (leveraged), cash goes negative, and the engine charges margin interest at 6% annual (line 115-117).

### 5.3 Regime Detection

**Location**: `optimizer.py` lines 103-116

**Finding (LOW)**: Regime detection uses trailing 63-day SPY realized vol, which is backward-looking. It will classify regimes *after* the vol spike has occurred, not before. This is acceptable for de-leveraging (reactive protection) but means the system enters high-vol positions during the initial spike before the regime is detected.

---

## 6. Backtest Engine -- Portfolio-Relevant Issues

### 6.1 Rebalance Execution

**Location**: `engine.py` lines 87-103

The engine executes at the rebalance-day close price (same-day close signal and execution). This was flagged by the Data Audit as a MEDIUM issue. From a portfolio construction perspective, this means backtest Sharpe is slightly inflated because in reality there is a 1-day lag.

### 6.2 No Intra-Period Risk Management

The backtest engine loops daily but only takes action on rebalance dates. Between rebalances (21 trading days), the portfolio drifts freely. There is no intra-period check for:
- Stop-loss triggers
- Drawdown limits
- Sector constraint violations due to price moves

This is a significant gap for a system that claims stop-loss and drawdown limits.

---

## 7. compute_covariance Method -- DEAD CODE

**Location**: `optimizer.py` lines 159-161

```python
def compute_covariance(self, returns: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    """Exponentially-weighted covariance matrix."""
    return returns.ewm(span=window).cov().iloc[-len(returns.columns):]
```

**Finding (LOW)**: This method exists but is never called anywhere in the codebase. `strategy.py` computes covariance inline via `ret_window[selected].cov()`. The method itself computes EWM covariance (which would be better than sample covariance) but its output slicing `iloc[-len(returns.columns):]` extracts only the last N rows of the multi-level EWM cov result, which gives the covariance matrix for the last date. This is correct but unused.

**Fix delivered**: Updated `compute_covariance()` to support three methods: `ledoit_wolf` (default), `ewm`, and `sample`.

---

## 8. Risk Matrix

| Issue | Severity | Impact (Backtest) | Impact (Live) | Fix Status |
|-------|----------|-------------------|---------------|------------|
| Raw sample covariance (T/N=7) | HIGH | Unstable weights, excess turnover | Same + real trading costs | FIXED (Ledoit-Wolf) |
| No turnover penalty in optimizer | HIGH | Overstated returns by ~50-90bps/yr | Excess trading costs | FIXED (turnover penalty) |
| Sector constraint not enforced | HIGH | Unintended tech concentration | Same | FIXED (sector constraints in optimizer) |
| Alpha z-scores as return inputs | HIGH | Risk penalty negligible in objective | Same | DOCUMENTED (needs calibration) |
| Stop-loss never called | HIGH | False sense of risk protection | Positions held through crashes | DOCUMENTED (needs engine integration) |
| Drawdown monitor not integrated | MEDIUM | No halt on 20% drawdown | Same | PARTIALLY FIXED (post-backtest check) |
| Risk aversion hardcoded at 1.0 | MEDIUM | Suboptimal risk-return balance | Same | DOCUMENTED |
| No factor exposure monitoring | MEDIUM | Unknown systematic exposures | Same | FIXED (factor exposure method) |
| No concentration metrics | MEDIUM | No HHI/effective-N tracking | Same | FIXED (HHI + sector monitoring) |
| Vol scaling uses noisy covariance | MEDIUM | Unstable leverage scaling | Same | FIXED (Ledoit-Wolf) |
| No no-trade zone | LOW | Unnecessary small trades | Same | DOCUMENTED |
| Regime transition is abrupt | LOW | Sudden leverage changes | Same | DOCUMENTED |
| compute_covariance is dead code | LOW | Wasted code | N/A | FIXED (updated + usable) |

---

## 9. Recommendations (Priority Order)

### Immediate (delivered in this audit)
1. **Ledoit-Wolf shrinkage covariance** -- dramatically improves weight stability
2. **Turnover penalty in optimizer** -- reduces unnecessary trading
3. **Sector constraint enforcement** -- prevents unintended concentration
4. **Enhanced RiskMonitor** -- VaR/CVaR, HHI, factor exposures, sector monitoring
5. **Drawdown check integration** -- at least post-backtest

### Short-term (requires further work)
6. **Calibrate risk aversion** -- grid search or analytical scaling of alpha to return units
7. **Integrate stop-loss into backtest loop** -- check daily, not just at rebalance
8. **Add no-trade zone** -- minimum trade size in backtest engine
9. **Smooth regime transitions** -- linear interpolation of leverage caps

### Medium-term
10. **Switch to cvxpy** -- guaranteed convergence for convex QP
11. **Factor exposure limits** -- constrain market beta, sector tilts
12. **Liquidity constraints** -- integrate ADV into optimizer bounds
13. **Alpha-to-return calibration** -- use trailing IC to convert z-scores to expected returns
