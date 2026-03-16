# Backtest Credibility Assessment

**Auditor**: Backtest & QA Agent (7 years AQR strategy verification)
**Date**: 2026-03-15
**Approach**: Adversarial -- my job is to try as hard as possible to prove this backtest does NOT work. If I cannot break it, it might be valid.

---

## Executive Summary

This backtest has **fundamental structural biases that cannot be patched away** without replacing the data infrastructure. The improvements made by the four preceding agents are real and meaningful -- Ledoit-Wolf shrinkage, turnover penalties, sector constraints, and safety checks all improve the system. But they address second-order problems while the first-order problems (look-ahead bias and survivorship bias) remain, rendering any reported Sharpe ratio unreliable.

**Overall Credibility Score: 3 out of 10.**

A score of 3 means: "The architecture is sound, the code is competent, but the backtest results should not be used for any capital allocation decision."

---

## 1. Known Biases That Inflate Performance

### 1.1 Look-Ahead Bias in Fundamental Factors (CRITICAL)

**What**: `yfinance.Ticker.info` returns current (March 2026) fundamental ratios. These are broadcast to every historical date from January 2016 onward via `np.tile()`. A stock that had PE=50 in 2016 but PE=15 today is scored as "cheap" throughout the entire backtest.

**Affected signal weight**: Value (5%) + Quality (25%) = 30% of composite signal.

**Quantification**: Quality at 25% weight is the second-largest factor. In a cross-section of 100 stocks, knowing which companies will have the best ROE, margins, and earnings growth 10 years in the future is an enormous informational advantage. Conservative estimate:

- Quality factor IC with perfect foresight: ~0.15-0.25 (vs typical IC of 0.02-0.04 with point-in-time data)
- Contribution to composite IC: 0.25 * 0.15 = ~0.038 (vs 0.25 * 0.03 = ~0.008 without look-ahead)
- Excess IC from look-ahead: ~0.030
- Sharpe inflation from quality look-ahead alone: **~0.15-0.30 Sharpe units**

The value factor at 5% weight has smaller impact but the same directional bias: ~0.02-0.05 additional Sharpe.

**Total Sharpe inflation from fundamental look-ahead: ~0.2-0.35**

### 1.2 Survivorship Bias (CRITICAL)

**What**: The universe is 100 currently-active large-cap US stocks. Stocks that declined, went bankrupt, were acquired, or were delisted between 2016 and 2026 are excluded.

**Notable omissions** for a 2016-2026 backtest:
- GE: Major decline and restructuring
- XLNX: Acquired by AMD 2022
- ATVI: Acquired by MSFT 2023
- TWTR: Taken private 2022
- FRC, SIVB: Bank failures 2023
- LUMN, T (significant decliners): May or may not be in universe

**Inclusion of future winners**:
- NVDA was ~$30B market cap in 2016 (mid-cap), now ~$3T. Including it from day one in a large-cap universe is survivorship bias.
- CRWD IPO'd in 2019. Including it in a backtest starting 2016 means yfinance returns NaN for 2016-2019, but the stock is available from IPO onward -- capturing its entire post-IPO run.
- SNOW IPO'd in 2020. Same issue.

**Quantification**: Academic literature (Brown, Goetzmann, Ibbotson, and Ross, 1992) estimates survivorship bias in large-cap mutual fund studies at 0.5-1.5% per year. For a concentrated 18-position portfolio selected from a survivorship-biased universe of 100 stocks over 10 years:

- Annual return inflation: **~1.0-2.0%**
- Sharpe inflation (assuming 15% vol): **~0.07-0.13 Sharpe units**

### 1.3 Same-Day Execution Bias (MEDIUM)

**What**: On rebalance day `t`, the signal is computed from closing prices through `t`, and the backtest executes trades at the close price on `t`. In practice, you cannot observe the close and trade at the close simultaneously.

**Quantification**: For a monthly rebalance strategy, the close-to-open gap is typically 5-15bps per trade. With ~50% turnover per rebalance and 12 rebalances per year:

- Annual drag understatement: 12 * 0.50 * 0.001 = ~0.6% per year
- Sharpe inflation: **~0.04**

This is small relative to the fundamental data biases.

### 1.4 Static Factor Weights (MEDIUM)

**What**: Factor weights (momentum=0.45, quality=0.25, volatility=0.10, value=0.05) are fixed and were presumably chosen based on some combination of intuition and in-sample analysis. There is no walk-forward optimization or IC-based dynamic weighting.

**Impact**: If the weights were optimized on the backtest period (even informally by the developer), the reported performance overstates what would be achieved out-of-sample. The degree of overstatement depends on how much tuning was done.

**Estimated impact**: Hard to quantify without knowing the developer's process. For a 4-factor system with one free weight parameter per factor, the degrees of freedom are low enough that this may be a ~0.05-0.15 Sharpe inflation if there was material in-sample tuning.

### 1.5 Total Estimated Sharpe Inflation

| Source | Sharpe Inflation (low) | Sharpe Inflation (high) |
|--------|------------------------|-------------------------|
| Fundamental look-ahead bias | 0.20 | 0.35 |
| Survivorship bias | 0.07 | 0.13 |
| Same-day execution | 0.03 | 0.05 |
| Weight tuning (uncertain) | 0.00 | 0.15 |
| **Total** | **0.30** | **0.68** |

**Interpretation**: If the backtest reports a Sharpe of 1.0, the true out-of-sample Sharpe after correcting for these biases is likely in the range of **0.32 to 0.70**. A reported Sharpe below 0.7 is consistent with zero true alpha after bias correction.

---

## 2. Fixes That Improve Credibility

### 2.1 Ledoit-Wolf Covariance Shrinkage (HIGH IMPACT)

**What**: Replaced sample covariance (T/N = 126/18 = 7.0) with Ledoit-Wolf shrinkage estimator.

**Why it matters**: This is the single most impactful improvement for portfolio construction. With T/N = 7, the sample covariance is dominated by estimation noise. Ledoit-Wolf shrinks toward a structured target (scaled identity), reducing the condition number by 3-10x and producing dramatically more stable portfolio weights.

**Credibility improvement**: This fix does NOT inflate backtest performance -- it actually reduces apparent returns slightly (by preventing the optimizer from exploiting estimation noise). That is a good sign. Ledoit-Wolf shrinkage is standard practice at every serious quantitative firm (AQR, DE Shaw, Two Sigma all use some form of shrinkage estimation).

### 2.2 Transaction Cost Penalty in Optimizer (HIGH IMPACT)

**What**: Added L1 turnover penalty `gamma * |w - w_prev|` to the optimization objective.

**Why it matters**: Without this, the optimizer freely churns the portfolio each month, unaware that each trade costs 15bps. The backtest charges these costs but the optimizer does not account for them, systematically overstating the achievable return. With the penalty, the optimizer explicitly trades off alpha improvement against transaction costs.

**Credibility improvement**: This closes a real gap between the optimizer's world and the backtest's world. Estimated reduction in annual drag: 30-60bps. This is a real improvement that makes the backtest more realistic.

### 2.3 Sector Constraint Enforcement (MEDIUM IMPACT)

**What**: Added `sum(w[sector]) <= max_sector_weight` constraints to SLSQP.

**Why it matters**: The universe is heavily tech-weighted (~35/100 stocks). Without sector constraints, the optimizer naturally overweights tech (strongest momentum and quality signals in recent years). This creates unintended sector concentration risk that would not be tolerated in a real portfolio.

**Credibility improvement**: This is a genuine risk management improvement. It likely reduces backtest returns slightly (constraining the optimizer away from the optimal-in-backtest solution) while improving robustness. Net positive for credibility.

### 2.4 Factor Bug Fixes (MEDIUM IMPACT)

- **Negative score filter fix**: `_apply_filter_safe()` now correctly handles negative composite scores. Previously, multiplying a negative score by 0.5 would halve the penalty (making the stock look better), which is the opposite of intended behavior.
- **fillna(0) bias fix**: Now normalizes by available factor weights per-cell, so stocks with partial data are not automatically penalized toward zero.
- **Sector neutralization min size**: Sectors with <5 stocks now fall back to cross-sectional z-scoring instead of producing unstable z-scores from 2 observations.

**Credibility improvement**: These are genuine bug fixes that make the signal cleaner. They do not directly address the fundamental data biases but they improve the integrity of the price-based factors (momentum, volatility, trend).

### 2.5 Forward-Fill Bug Fix in Backtest Engine (LOW-MEDIUM IMPACT)

**What**: Fixed `prices.loc[date, symbols].ffill()` which was forward-filling across symbols alphabetically (meaningless) instead of forward-filling each symbol's price forward in time.

**Credibility improvement**: This was a real bug that could randomly assign one stock's price to another. The fix is straightforward and correct. Impact depends on how often missing prices occurred -- for the 100 most liquid US stocks, probably rare but non-zero.

### 2.6 Data Quality Checks and Bias Warnings (LOW DIRECT IMPACT)

**What**: Added `DataQualityChecker`, `PointInTimeDataManager`, and `warn_survivorship_bias()`.

**Credibility improvement**: These do not change backtest results but they explicitly document the limitations. This is intellectually honest and important for anyone interpreting the results. The fact that the system now loudly warns about look-ahead bias and survivorship bias is a credibility positive -- it shows self-awareness.

---

## 3. What a Credible Backtest Would Require

### 3.1 Point-in-Time Fundamental Data (ESSENTIAL)

- **Source**: Compustat via WRDS, Sharadar via Nasdaq Data Link, or SimFin bulk downloads.
- **Requirement**: For each stock on each date, the fundamental ratios must reflect only information that was publicly available on or before that date.
- **Implementation**: Use filing dates (SEC EDGAR 10-K/10-Q filing dates), not fiscal period end dates. Earnings reported on 2024-02-15 should not be available in the backtest until 2024-02-15, even if the fiscal period ended 2023-12-31.
- **Reporting lag**: Model a 1-2 day processing lag after the filing date to account for data vendor ingestion time.

### 3.2 Historical Universe Constituents (ESSENTIAL)

- **Source**: S&P 500 historical constituents from Compustat or similar. Alternatively, reconstruct from market cap and volume screens applied point-in-time.
- **Requirement**: On each rebalance date, the universe should contain only stocks that would have passed the selection screen on that date -- not stocks that are large-cap today.
- **Delisting handling**: When a stock is delisted (acquisition, bankruptcy, merger), the backtest must handle the exit at the actual delisting price/terms, not simply drop the stock from the universe.

### 3.3 Next-Day Execution (IMPORTANT)

- **Implementation**: Signal computed from day `t` close prices. Trades executed at day `t+1` open prices (or VWAP of day `t+1`).
- **Impact**: This removes the close-to-close execution assumption and models a realistic execution lag.

### 3.4 Out-of-Sample Validation (ESSENTIAL)

- **Methodology**: Split the 2016-2026 period into in-sample (2016-2021) and out-of-sample (2022-2026).
- **Requirement**: ALL parameter choices (factor weights, rebalance frequency, position count, vol target, risk aversion) must be fixed on the in-sample period. The out-of-sample period must be run exactly once with no parameter re-tuning.
- **Walk-forward**: Alternatively, use a rolling walk-forward: train on years 1-3, test on year 4; train on years 1-4, test on year 5; etc. Average the out-of-sample Sharpes.

### 3.5 Bootstrap Confidence Intervals (IMPORTANT)

- **Methodology**: Block bootstrap (blocks of 21-63 trading days to preserve serial correlation) of the daily return series.
- **Output**: 95% confidence interval on Sharpe ratio. If the lower bound of the 95% CI is below 0.0, the strategy has no statistically significant alpha.
- **Sample size**: With ~2,500 daily observations over 10 years and a monthly rebalance, there are effectively ~120 independent observations. This is marginal for statistical significance -- a Sharpe of 0.5 would have a t-stat of ~1.7, which is not significant at 95%.

### 3.6 Factor Attribution and Decomposition (IMPORTANT)

- **Methodology**: Regress strategy returns on Fama-French 5 factors (Mkt-RF, SMB, HML, RMW, CMA) plus momentum (UMD).
- **Purpose**: Determine how much of the strategy's return is explained by known systematic factors vs genuine alpha. A strategy that loads heavily on momentum and quality factors is not generating alpha -- it is delivering factor exposure that can be obtained more cheaply via ETFs.
- **Expected finding**: Given the 45% momentum weight and 25% quality weight, the strategy likely has significant loading on UMD and RMW/CMA. True alpha (intercept of the regression) may be small.

---

## 4. Overall Credibility Score: 3 / 10

### Scoring Rubric

| Score | Meaning |
|-------|---------|
| 1-2 | Backtest is fundamentally broken; results are meaningless |
| 3-4 | Architecture is sound but critical biases invalidate reported performance |
| 5-6 | Moderate biases present; results directionally useful but magnitudes unreliable |
| 7-8 | Minor biases only; results suitable for preliminary capital allocation decisions |
| 9-10 | Institutional-grade backtest suitable for production deployment |

### Justification for Score of 3

**Why not lower (1-2)**:
- The code architecture is competent and well-organized
- The factor definitions are academically grounded (proper skip-month momentum, Bollinger z-score, low-vol anomaly)
- The Phase 1-4 improvements are genuine: Ledoit-Wolf, turnover penalty, sector constraints, safety checks
- The system explicitly warns about its own biases (look-ahead, survivorship), showing intellectual honesty
- The price-based factors (momentum, volatility, trend filter) are computed correctly with no look-ahead bias
- 70% of the composite signal weight (momentum + volatility) uses only price data and is bias-free

**Why not higher (5+)**:
- 30% of the composite signal (quality 25% + value 5%) uses fundamentals with severe look-ahead bias
- Quality at 25% weight is the second-largest factor -- this is not a rounding error
- Survivorship bias is structural and cannot be mitigated by code changes
- No out-of-sample validation exists anywhere
- No factor attribution against known risk factors
- No bootstrap confidence intervals
- The reported Sharpe is inflated by an estimated 0.30-0.68 units
- Stop-loss is advertised in config but is dead code (risk management theater)
- The strategy has never been tested on data it has not already seen

### What Would Raise the Score

| Action | Score Impact |
|--------|-------------|
| Replace yfinance fundamentals with point-in-time data (Compustat/Sharadar) | +1.5 |
| Use historical index constituents for universe | +1.0 |
| Implement next-day execution model | +0.5 |
| Conduct proper out-of-sample validation (no re-tuning) | +1.5 |
| Add bootstrap confidence intervals | +0.5 |
| Factor attribution showing alpha net of known factors | +1.0 |
| Integrate stop-loss in backtest loop (or remove from config) | +0.5 |
| **Total potential improvement** | **+6.5 (to 9.5)** |

---

## 5. Recommendations

### For the Developer

1. **Do not deploy real capital based on current backtest results.** The look-ahead bias alone can account for most or all of the apparent alpha.

2. **Quick test to quantify the bias**: Set quality and value weights to 0 (as the Data Audit recommended) and re-run the backtest. Compare the Sharpe with momentum+volatility only (70% of weight, bias-free) vs the full signal. If the Sharpe drops significantly, the apparent alpha was driven by the biased factors.

3. **The momentum and volatility factors are likely real**: Both are well-documented academic anomalies, computed correctly from price data with no look-ahead bias. A strategy based on momentum (0.65) + volatility (0.15) + trend filter + blowoff filter, with Ledoit-Wolf covariance and turnover penalties, is a reasonable starting point for a medium-term equity strategy. But it needs proper out-of-sample validation.

4. **Invest in data infrastructure**: The single highest-ROI investment is obtaining point-in-time fundamental data. Sharadar via Nasdaq Data Link is ~$500/year and provides exactly what is needed. This would convert the value and quality factors from liabilities into assets.

### For Anyone Reviewing This System

- The code quality is good. The architecture is sound. The improvements from the four audit agents are real and well-implemented.
- The problem is not the code -- it is the data. With proper data infrastructure, this system could produce a credible backtest.
- Do not confuse "well-written code" with "valid backtest." A perfectly implemented strategy with biased data produces a perfectly implemented biased backtest.

---

## Appendix: Bias Interaction Effects

The biases described above are not independent. They interact in ways that can amplify or partially offset each other:

1. **Survivorship + look-ahead amplification**: Stocks that survived and prospered (e.g., NVDA) have both (a) good current fundamentals (captured by look-ahead) and (b) strong past price momentum (captured correctly). These signals reinforce each other, double-counting the "winner" effect. A stock that survived AND grew its earnings would score high on momentum, quality, AND value -- triple-counting the same underlying outcome.

2. **Survivorship + sector concentration**: The universe is tech-heavy because tech stocks have performed well. The sector constraint limits tech to 40%, but even 40% in a sector that benefited from survivorship bias is still contaminated.

3. **Ledoit-Wolf partially offsets optimizer noise bias**: Without shrinkage, the noisy sample covariance could cause the optimizer to accidentally concentrate in stocks that happened to have low measured covariance (an estimation error, not a real property). Ledoit-Wolf reduces this, which actually makes the backtest more conservative. This is one of the few interactions that works in the right direction.
