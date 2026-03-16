# Alpha Factor Audit Report

**Auditor**: Alpha Research Agent (ex-Citadel Global Quantitative Strategies)
**Date**: 2026-03-15
**Codebase**: `/Users/ying/claude/trade/quant/signals/factors.py`
**Config**: `/Users/ying/claude/trade/config.yaml`

---

## Executive Summary

The system implements 6 alpha factors (momentum, mean reversion, trend, volatility, value, quality) combined into a weighted composite with multiplicative post-filters. The overall architecture is sound for a medium-term equity strategy. However, I have identified **3 critical issues**, **5 high-severity issues**, and **several medium-priority concerns** that must be addressed before any live deployment.

### Critical Findings

| # | Issue | Severity | Location |
|---|-------|----------|----------|
| C1 | Look-ahead bias in value/quality factors | CRITICAL | `SignalGenerator.generate()` L258-271 |
| C2 | Momentum computation is correct but poorly tested for edge cases | HIGH | `momentum_factor()` L89 |
| C3 | Sector neutralization unstable for small sectors | HIGH | `neutralize_by_sector()` L62-67 |

---

## Factor 1: Momentum

### 1.1 Definition Accuracy

**Implementation** (lines 76-95):
```python
ret = prices.shift(skip).pct_change(w)  # skip=21, w in [63, 126, 252]
```

**Academic Reference**: Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency." The canonical momentum factor uses returns over months 2-12 (or 2-6), skipping the most recent month to avoid the short-term reversal effect documented by Jegadeesh (1990).

**Verification of `prices.shift(skip).pct_change(w)`**:
- `prices.shift(21)` at row `t` yields the price at `t-21` (21 trading days ago).
- `pct_change(w)` on the shifted series computes `(P[t-21] - P[t-21-w]) / P[t-21-w]`.
- For `w=252`: return from `t-273` to `t-21`, i.e., ~12 months of return ending 1 month ago.
- **Verdict: CORRECT.** This properly implements skip-1-month momentum.

**Concern**: The `pct_change(w)` approach computes simple returns, not log returns. For large windows (252 days), simple returns can be heavily right-skewed, potentially biasing cross-sectional z-scores. Using log returns would be more appropriate for longer windows, though the winsorization step partially mitigates this.

**Concern**: All three windows (63, 126, 252) are equally weighted in the composite (`pd.concat(scores).groupby(level=0).mean()`). Empirical evidence (e.g., Moskowitz, Ooi, and Pedersen 2012) suggests 12-month momentum has the strongest signal. Consider weighting windows by IC or assigning explicit weights.

### 1.2 Preprocessing

- **Z-scoring**: Cross-sectional z-score per day per window. Correct approach.
- **NaN handling**: `pct_change()` returns NaN for the first `w` rows; z-score propagates NaN. In the composite, `groupby.mean()` will average available windows. A stock that only has a 3-month window available (not yet 6m or 12m) will have a noisier score. This is acceptable but could be improved with window availability weighting.
- **Winsorization**: Applied post-hoc via `winsorize_zscore(clip_val=3.0)`. Standard practice.
- **Industry neutralization**: Applied. This removes sector momentum effects (e.g., all tech rising together), leaving stock-specific momentum.

### 1.3 Factor Quality Framework

To validate this factor in production:

- **IC (Information Coefficient)**: Rank correlation between momentum z-score at time `t` and forward 21-day returns. Expect IC of 0.02-0.06 for medium-term momentum.
- **ICIR (IC Information Ratio)**: `mean(IC) / std(IC)`. Target > 0.4 for a production-grade signal.
- **Quantile returns**: Sort universe into 5 quintiles by momentum score each month. Top quintile should outperform bottom by 5-10% annualized (pre-transaction cost).
- **Factor turnover**: With 21-day rebalance and 3m/6m/12m windows, turnover should be moderate (~15-25% per month). Monitor for excessive turnover in the 3m window.
- **Decay**: IC should remain positive for at least 21 trading days (the rebalance horizon). Expect IC to peak at 5-10 day forward horizon and decay by ~60 days.

### 1.4 Issues Found

**ISSUE M1 (Medium)**: The composite averaging via `pd.concat(scores).groupby(level=0).mean()` performs an unweighted average. When one window has NaN and others do not, the stock's score is computed from fewer windows. This creates a subtle bias where recently-IPO'd stocks (only short windows available) have different score distributions.

---

## Factor 2: Mean Reversion

### 2.1 Definition Accuracy

**Implementation** (lines 98-109):
```python
zscore = (prices - rolling_mean) / rolling_std
return -zscore  # oversold (negative z) -> positive signal
```

**Academic Reference**: Bollinger Bands (Bollinger 1992). The z-score measures deviation from a rolling mean, normalized by rolling standard deviation. The negation is correct: oversold stocks (price below moving average) receive positive scores.

**Verdict: CORRECT.** Standard Bollinger z-score with proper sign convention.

### 2.2 Preprocessing

- The `zscore_threshold` parameter is passed to the function but **never used** inside `mean_reversion_factor()`. It is only used in `blowoff_filter()`. This is not a bug (the threshold is a config parameter shared between them), but it is confusing API design.
- No winsorization within the function itself (handled by the pipeline's `winsorize_zscore` call).

### 2.3 Redundancy with Blowoff Filter

The `mean_reversion_factor()` and `blowoff_filter()` compute the **exact same Bollinger z-score** using the same window (20 days). The mean reversion factor has weight=0 (disabled as a scored factor), and the blowoff filter uses the z-score > 3.0 threshold to penalize overextended stocks.

**ISSUE MR1 (Low)**: The Bollinger z-score is computed twice: once in `mean_reversion_factor()` (which is neutralized and winsorized but ultimately multiplied by weight=0) and once in `blowoff_filter()`. This is wasteful but not incorrect. The factor computation for mean_reversion could be skipped entirely when weight=0 to save computation.

### 2.4 Factor Quality Framework

Mean reversion is configured as a filter, not a scored factor. If it were to be used as a scored factor:

- **IC**: Expect IC of 0.01-0.03. Mean reversion works best at very short horizons (1-5 days) and decays rapidly.
- **Interaction with momentum**: Mean reversion and momentum are conceptually opposite. Using mean reversion as a blowoff filter (penalizing extreme winners) while scoring momentum is a reasonable architecture that captures the short-term reversal at the tails while preserving medium-term momentum in the body of the distribution.

---

## Factor 3: Trend

### 3.1 Definition Accuracy

**Implementation** (lines 112-124):
```python
ratio = sma_short / sma_long  # 50d SMA / 200d SMA
zs = ratio.sub(ratio.mean(axis=1), axis=0).div(ratio.std(axis=1), axis=0)
```

**Academic Reference**: Moving average crossover strategies have a long history (Brock, Lakonishok, and LeBaron 1992). The ratio of short/long SMA is a continuous version of the classic golden cross / death cross signal.

**Verdict: CORRECT.** The ratio formulation is standard. Cross-sectional z-scoring converts absolute SMA ratios into relative rankings.

### 3.2 Usage as Filter

The trend factor has weight=0 and is instead applied as `trend_filter()`:
```python
above = prices >= sma_long
return above.astype(float).replace(0.0, penalty)  # 1.0 or 0.5
```

This binary filter penalizes stocks below their 200d SMA by halving their composite score. This is a defensible approach: trend acts as a regime indicator rather than a continuous signal.

**ISSUE T1 (Medium)**: The filter is binary (1.0 or 0.5), creating a cliff effect. A stock at 200d SMA + $0.01 gets full score; at SMA - $0.01 it gets half. Consider a sigmoid or linear ramp around the SMA to smooth this transition.

### 3.3 Factor Quality Framework

- **As a filter**: Measure the hit rate: what fraction of stocks below 200d SMA underperform over the next 21 days? Expect 52-55% accuracy.
- **Interaction with momentum**: Trend and momentum are highly correlated. Stocks with strong 6m/12m momentum will almost certainly be above their 200d SMA. The filter primarily catches stocks whose momentum score is still positive but whose price has recently crossed below the SMA (early trend break). This is a reasonable safety net.

---

## Factor 4: Volatility

### 4.1 Definition Accuracy

**Implementation** (lines 153-160):
```python
vol = returns.rolling(window).std() * np.sqrt(252)
zs = vol.sub(vol.mean(axis=1), axis=0).div(vol.std(axis=1), axis=0)
return -zs  # invert: low vol -> high score
```

**Academic Reference**: The low-volatility anomaly (Baker, Bradley, and Wurgler 2011, "Benchmarks as Limits to Arbitrage: Understanding the Low-Volatility Anomaly"). Low-volatility stocks systematically outperform high-volatility stocks on a risk-adjusted basis.

**Verdict: CORRECT.** Annualized realized volatility, cross-sectionally z-scored, inverted so low vol = high score.

### 4.2 Preprocessing

- The input is `returns` (daily simple returns), not prices. Correct: volatility should be computed from returns.
- Window of 63 trading days (~3 months) is standard for realized vol estimation.
- Annualization via `sqrt(252)` assumes i.i.d. returns. This is a standard approximation, though serial correlation in daily returns can make this slightly inaccurate.

### 4.3 Factor Quality Framework

- **IC**: Expect IC of 0.01-0.04 for low-vol factor. Signal is stable but weak.
- **Turnover**: Low-vol rankings are very stable (low turnover), which is favorable for transaction costs.
- **Interaction with momentum**: Momentum and volatility can be negatively correlated (high-momentum stocks often have higher vol), creating useful diversification.
- **Interaction with quality**: High-quality stocks tend to have lower volatility. Monitor for collinearity (VIF > 5 would be concerning).

---

## Factor 5: Value

### 5.1 Definition Accuracy

**Implementation** (lines 167-184):
```python
for col in ["trailingPE", "forwardPE", "priceToBook"]:
    inv = 1.0 / vals.replace(0, np.nan)
    scores[col] = (inv - inv.mean()) / inv.std()
```

**Academic Reference**: Fama and French (1992, 1993). Value factors traditionally use book-to-market (B/M) ratio. The inverse of P/E (earnings yield) and inverse of P/B (book-to-price) are equivalent formulations.

**Verdict: PARTIALLY CORRECT.** The inversion is correct (lower PE/PB = higher value = higher score). However:

### 5.2 CRITICAL ISSUE: Look-Ahead Bias (C1)

**CRITICAL BUG**: `value_factor()` receives fundamentals from `yfinance.Ticker.info`, which returns the **current** (as of today) snapshot of fundamental data. In the backtest, this data is broadcast to ALL historical dates via:

```python
df = pd.DataFrame(
    np.tile(series.values, (len(px), 1)),
    index=px.index, columns=series.index,
)
```

This means the backtest uses **2026 fundamentals** when making decisions in **2016-2025**. A stock that had PE=30 in 2018 but PE=15 today would be scored as cheap throughout the entire backtest.

**Impact**: This creates massive look-ahead bias. The value and quality factors (combined weight: 0.30) are using information from the future. Backtest results for these factors are completely unreliable.

**Remediation**:
1. **Short-term**: Source point-in-time fundamental data. Options: SEC EDGAR filings with filing dates, quarterly earnings data from yfinance `.quarterly_financials`, or a paid data provider with point-in-time coverage.
2. **Medium-term**: Use `yfinance.Ticker.quarterly_financials` and `.quarterly_balance_sheet` with proper as-of-date alignment (use filing date, not period end date).
3. **For current live trading**: The current implementation is valid for generating today's signal only (since current fundamentals ARE the most recent data). The bias only affects backtesting.

### 5.3 Additional Issues

**ISSUE V1 (Medium)**: Negative PE ratios (loss-making companies) are handled via `1.0 / vals.replace(0, np.nan)`, which preserves negative values. Inverting a negative PE gives a negative earnings yield, correctly ranking loss-making companies as expensive. However, the distribution of 1/PE is highly asymmetric around zero. Companies with PE of -1 and -100 get very different inverse values (-1.0 vs -0.01) despite both being loss-making. Consider using earnings yield (E/P) directly rather than inverting PE, or capping negative PE handling.

**ISSUE V2 (Low)**: Only 3 value metrics are used (trailing PE, forward PE, price-to-book). Academic composites typically include sales-to-price (S/P) and cash-flow-to-price (CF/P). See Asness, Moskowitz, and Pedersen (2013) "Value and Momentum Everywhere."

---

## Factor 6: Quality

### 6.1 Definition Accuracy

**Implementation** (lines 187-199):
```python
for col in ["returnOnEquity", "profitMargins", "earningsGrowth"]:
    scores[col] = (vals - vals.mean()) / vals.std()
```

**Academic Reference**: Novy-Marx (2013) "The Other Side of Value: The Gross Profitability Premium." Asness, Frazzini, and Pedersen (2019) "Quality Minus Junk." Quality factors typically include profitability, growth, and safety (low leverage).

**Verdict: PARTIALLY CORRECT.** The z-score computation is standard. However:

### 6.2 Same Look-Ahead Bias as Value (C1)

See Section 5.2. Quality factor suffers from the identical look-ahead bias. Current ROE, margins, and earnings growth are broadcast to all historical dates.

### 6.3 Additional Issues

**ISSUE Q1 (Medium)**: The quality composite uses only 3 metrics. Missing components that would strengthen the factor:
- **Accruals**: (Net income - operating cash flow) / total assets. High accruals predict negative future returns (Sloan 1996).
- **Earnings stability**: Standard deviation of quarterly EPS over trailing 5 years.
- **Financial strength**: Debt-to-equity is available in the data (`debtToEquity` field) but not used. Low leverage is a quality signal.
- **Piotroski F-Score**: Composite of 9 binary signals. Strong academic backing (Piotroski 2000).

**ISSUE Q2 (Low)**: `earningsGrowth` from yfinance is a single point estimate (year-over-year), not a trend. A more robust approach would use multi-year earnings growth consistency.

---

## Preprocessing Pipeline Audit

### Winsorization (`winsorize_zscore`)

**Implementation** (lines 23-34):
```python
clipped = df.clip(lower=-clip_val, upper=clip_val, axis=1)
mean = clipped.mean(axis=1)
std = clipped.std(axis=1).replace(0, 1)
return clipped.sub(mean, axis=0).div(std, axis=0)
```

**Verdict: CORRECT but with a subtlety.** The function clips to [-3, 3] then re-standardizes. This is proper two-pass winsorization. The `axis=1` in `.clip()` is effectively ignored for scalar bounds (all values are clipped to the same range). The `replace(0, 1)` prevents division by zero when all values in a row are identical after clipping.

**ISSUE W1 (Low)**: After re-z-scoring, values can exceed the clip bounds again (though the distribution is now more compressed). If strict bounds are required, a second clip pass would be needed. In practice, this is not a concern.

### Industry Neutralization (`neutralize_by_sector`)

**Implementation** (lines 37-69):
```python
if len(mask) < 2:
    continue  # skip single-stock sectors
sector_data = factor_df[mask]
mean = sector_data.mean(axis=1)
std = sector_data.std(axis=1).replace(0, 1)
result[mask] = sector_data.sub(mean, axis=0).div(std, axis=0)
```

**ISSUE N1 (HIGH)**: Sectors with only 2 stocks have `std` computed from 2 observations, which is extremely noisy. The z-score of a 2-stock sector is always {-0.707, +0.707} (or close to it with `ddof=1` in pandas), regardless of the actual factor values. This forces a binary ranking within the sector.

For the production universe of 100 stocks across ~9 sectors, sector sizes range from 2 (Utilities, Real Estate) to 15 (Mega-Cap Tech + Tech/Software + Semiconductors combined if GICS sectors). The 2-stock sectors (Utilities: NEE, DUK; Real Estate: AMT, PLD) will always produce z-scores of approximately +/- 0.71, which is artificial.

**Remediation**:
1. Set a minimum sector size threshold of 5 stocks. For smaller sectors, use the overall cross-sectional z-score instead of within-sector.
2. Alternatively, use sector dummy regression: regress factor values on sector dummies and use the residuals. This handles small sectors more gracefully.

**ISSUE N2 (Medium)**: Stocks not present in `sector_map` retain their raw (non-neutralized) scores, which are on a different scale than the neutralized scores. The function should either neutralize all stocks or warn about missing sector assignments.

### Composite Signal Construction

**Implementation** (lines 288-307):
```python
composite += weight * f.fillna(0)  # NaN -> 0
...
composite = composite * tf  # trend filter (multiply)
composite = composite * bf  # blowoff filter (multiply)
```

**ISSUE CS1 (HIGH)**: `fillna(0)` for missing factor values is problematic. A stock with missing momentum data (e.g., recently IPO'd, less than 252 trading days) gets momentum score = 0 (neutral). This is reasonable. However, a stock missing ALL factor data gets composite = 0, which after filters could become non-zero if other factors contribute. More importantly, the zero-filling biases the composite toward zero for stocks with partial data, effectively under-weighting them even if the available factors are strong.

**Better approach**: Weight only the available (non-NaN) factors for each stock, normalizing by the sum of available factor weights rather than total weight.

**ISSUE CS2 (Medium)**: The multiplicative filters can create non-linear interactions. Example: A stock with composite z-score of -1.5 (mild underweight) that is below its 200d SMA gets score -1.5 * 0.5 = -0.75 (less negative). The trend filter **helps** underweight stocks that are in downtrends, which is the opposite of the intended effect. Multiplicative filters work correctly only for positive scores.

**Remediation**: Apply filters only to positive composite scores, or use an additive penalty instead of multiplicative.

---

## Factor Interaction Analysis

### Expected Correlation Structure

| Factor Pair | Expected Correlation | Concern Level |
|-------------|---------------------|---------------|
| Momentum - Trend | High positive (0.5-0.7) | HIGH: trend is used as filter, mitigating this |
| Momentum - Volatility | Moderate negative (-0.2 to -0.4) | LOW: provides diversification |
| Quality - Volatility | Moderate positive (0.2-0.4) | MEDIUM: some collinearity |
| Value - Quality | Low to moderate negative (-0.1 to -0.3) | LOW: cheap stocks often lower quality |
| Value - Momentum | Low to moderate negative (-0.1 to -0.3) | LOW: value and momentum are historically uncorrelated to negatively correlated |
| Mean Reversion - Momentum | Strong negative (-0.4 to -0.7) | LOW: mean reversion is disabled as scored factor |

### Multi-Collinearity Assessment Framework

To assess in production:

1. **VIF (Variance Inflation Factor)**: Regress each factor on all others. VIF > 5 indicates problematic collinearity. Expect momentum-trend VIF > 5 (mitigated by trend being a filter).

2. **Factor correlation matrix**: Compute rolling 252-day Spearman rank correlation between all factor pairs. Monitor for regime changes where previously uncorrelated factors become correlated (e.g., during market stress, all factors may become correlated with market beta).

3. **PCA analysis**: Run PCA on the factor matrix. If the first principal component explains > 60% of variance, the factor set lacks diversification.

### Orthogonalization Recommendations

If collinearity is found to be problematic:

1. **Sequential orthogonalization**: Regress each factor on all prior factors and use residuals. Order matters; place the most important factor (momentum) first.
2. **Symmetric orthogonalization**: Use PCA rotation to create uncorrelated factors, then map back to interpretable weights.
3. **IC-weighted combination**: Weight factors by their trailing IC rather than fixed weights. This implicitly down-weights correlated factors when one is more predictive.

---

## Test Coverage Assessment

The existing test suite (`test_factors.py`) covers:
- Shape consistency
- Directional correctness (uptrending stocks rank higher in momentum)
- Oversold/overbought signal direction
- Winsorization bounds
- Neutralization basic behavior
- Composite signal generation

### Missing Test Coverage

1. **No test for momentum computation correctness**: The shift+pct_change logic should be verified against a hand-computed example.
2. **No test for NaN propagation**: What happens when a stock has 50 days of data and momentum uses a 252-day window?
3. **No test for negative PE handling** in value factor.
4. **No test for small sector behavior** (2-3 stocks) in neutralization.
5. **No test for multiplicative filter interaction** with negative composite scores (ISSUE CS2).
6. **No test for the `fillna(0)` behavior** in composite construction (ISSUE CS1).

---

## Summary of All Issues

| ID | Severity | Factor | Description |
|----|----------|--------|-------------|
| C1 | CRITICAL | Value, Quality | Look-ahead bias: current fundamentals broadcast to all historical dates |
| N1 | HIGH | All (neutralization) | Sector neutralization unstable for 2-stock sectors |
| CS1 | HIGH | Composite | `fillna(0)` biases composite for stocks with partial factor data |
| CS2 | MEDIUM | Composite | Multiplicative filters invert effect for negative composite scores |
| M1 | MEDIUM | Momentum | Unweighted window averaging; recently-listed stocks get noisier scores |
| T1 | MEDIUM | Trend Filter | Binary cliff effect at 200d SMA boundary |
| N2 | MEDIUM | All (neutralization) | Stocks without sector assignment retain raw (non-neutralized) scale |
| V1 | MEDIUM | Value | Negative PE inversion creates asymmetric distribution |
| Q1 | MEDIUM | Quality | Missing quality components (accruals, leverage, stability) |
| MR1 | LOW | Mean Reversion | Bollinger z-score computed twice (in factor and in filter) |
| V2 | LOW | Value | Limited value metrics (missing S/P, CF/P) |
| Q2 | LOW | Quality | Single-point earnings growth, not trend |
| W1 | LOW | Preprocessing | Post-winsorize z-scores can slightly exceed clip bounds |
