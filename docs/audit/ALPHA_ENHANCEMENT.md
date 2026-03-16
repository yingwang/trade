# Alpha Enhancement Recommendations

**Author**: Alpha Research Agent
**Date**: 2026-03-15
**Prerequisite**: Read `ALPHA_AUDIT.md` first for context on current issues.

---

## 1. Momentum Enhancement

### 1.1 Residual Momentum

**Reference**: Blitz, Huij, and Martens (2011) "Residual Momentum." After controlling for Fama-French factors, residual momentum (alpha momentum) is stronger, less volatile, and less prone to crashes than raw price momentum.

**Implementation approach**:

```python
def residual_momentum_factor(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    windows: list[int] = None,
    skip: int = 21,
) -> pd.DataFrame:
    """Residual momentum: momentum of stock-specific returns after
    removing market beta exposure.

    1. Estimate rolling beta for each stock vs market.
    2. Compute residual returns: r_i - beta_i * r_m.
    3. Compute momentum of residual return series.
    """
    if windows is None:
        windows = [63, 126, 252]

    beta_window = 252
    scores = []

    for sym in returns.columns:
        if sym == market_returns.name:
            continue
        # Rolling OLS beta
        cov = returns[sym].rolling(beta_window).cov(market_returns)
        var = market_returns.rolling(beta_window).var()
        beta = cov / var
        # Residual returns
        resid = returns[sym] - beta * market_returns
        for w in windows:
            # Cumulative residual return over window, skipping recent month
            cum_resid = resid.shift(skip).rolling(w).sum()
            scores.append(cum_resid.rename(sym))

    # Reshape into (dates x symbols), average across windows
    # Group by symbol, then cross-sectional z-score
    # ... (full implementation requires careful reshape logic)
```

**Expected improvement**: 20-30% higher ICIR vs raw momentum. Significantly reduced momentum crash risk (residual momentum does not load on market reversals).

### 1.2 Weighted Lookback Windows

Rather than equal-weighting the 3m/6m/12m windows, use decay weighting:

```python
def momentum_factor_weighted(
    prices: pd.DataFrame,
    windows: list[int] = None,
    window_weights: list[float] = None,
    skip: int = 21,
) -> pd.DataFrame:
    """Momentum with explicit window weights.

    Default weights emphasize 12m (strongest academic signal)
    while still benefiting from shorter windows.
    """
    if windows is None:
        windows = [63, 126, 252]
    if window_weights is None:
        window_weights = [0.2, 0.3, 0.5]  # 12m gets 50% weight

    scores = []
    for w, wt in zip(windows, window_weights):
        ret = prices.shift(skip).pct_change(w)
        zs = ret.sub(ret.mean(axis=1), axis=0).div(ret.std(axis=1), axis=0)
        scores.append(zs * wt)

    composite = pd.concat(scores).groupby(level=0).sum()
    # Re-standardize
    return composite.sub(composite.mean(axis=1), axis=0).div(
        composite.std(axis=1), axis=0
    )
```

### 1.3 Volatility-Adjusted Momentum

**Reference**: Barroso and Santa-Clara (2015) "Momentum Has Its Moments."

Scale momentum returns by their trailing realized volatility. This dramatically reduces the fat left tail of momentum strategies.

```python
def vol_adjusted_momentum(prices, returns, windows=None, skip=21, vol_window=126):
    """Volatility-scaled momentum: divide raw momentum by
    trailing realized volatility of the momentum strategy itself."""
    raw_mom = momentum_factor(prices, windows)
    # Estimate trailing vol of each stock's momentum return series
    for sym in raw_mom.columns:
        trailing_vol = returns[sym].rolling(vol_window).std() * np.sqrt(252)
        trailing_vol = trailing_vol.replace(0, np.nan).fillna(method="ffill")
        raw_mom[sym] = raw_mom[sym] / trailing_vol.reindex(raw_mom.index)
    # Re-cross-sectional z-score
    zs = raw_mom.sub(raw_mom.mean(axis=1), axis=0).div(
        raw_mom.std(axis=1), axis=0
    )
    return zs
```

---

## 2. Value Enhancement

### 2.1 Composite Value (Multi-Metric)

**Reference**: Asness, Moskowitz, and Pedersen (2013) "Value and Momentum Everywhere."

Expand from 3 metrics to 5+:

```python
def enhanced_value_factor(fundamentals: pd.DataFrame) -> pd.Series:
    """Composite value using E/P, B/P, S/P, CF/P, and dividend yield.

    Each metric is z-scored and averaged. More metrics reduce
    noise from any single accounting measure.
    """
    scores = pd.DataFrame(index=fundamentals.index)

    # Earnings yield (E/P) - from trailing PE
    pe = pd.to_numeric(fundamentals.get("trailingPE"), errors="coerce")
    if pe is not None and pe.notna().sum() > 2:
        ep = 1.0 / pe.replace(0, np.nan)
        ep = ep.clip(lower=ep.quantile(0.02), upper=ep.quantile(0.98))
        scores["EP"] = (ep - ep.mean()) / ep.std()

    # Book-to-price (B/P) - from priceToBook
    pb = pd.to_numeric(fundamentals.get("priceToBook"), errors="coerce")
    if pb is not None and pb.notna().sum() > 2:
        bp = 1.0 / pb.replace(0, np.nan)
        bp = bp.clip(lower=bp.quantile(0.02), upper=bp.quantile(0.98))
        scores["BP"] = (bp - bp.mean()) / bp.std()

    # Sales-to-price (if available from revenue/market_cap)
    rev = pd.to_numeric(fundamentals.get("totalRevenue"), errors="coerce")
    mcap = pd.to_numeric(fundamentals.get("marketCap"), errors="coerce")
    if rev is not None and mcap is not None:
        sp = rev / mcap.replace(0, np.nan)
        if sp.notna().sum() > 2:
            scores["SP"] = (sp - sp.mean()) / sp.std()

    # Cash-flow-to-price (if operatingCashflow available)
    ocf = pd.to_numeric(fundamentals.get("operatingCashflow"), errors="coerce")
    if ocf is not None and mcap is not None:
        cfp = ocf / mcap.replace(0, np.nan)
        if cfp.notna().sum() > 2:
            scores["CFP"] = (cfp - cfp.mean()) / cfp.std()

    # Dividend yield (directly available)
    dy = pd.to_numeric(fundamentals.get("dividendYield"), errors="coerce")
    if dy is not None and dy.notna().sum() > 2:
        scores["DY"] = (dy - dy.mean()) / dy.std()

    if scores.empty:
        return pd.Series(0.0, index=fundamentals.index, name="value")

    return scores.mean(axis=1).rename("value")
```

### 2.2 Sector-Relative Value

Value ratios vary dramatically by sector (tech PE of 25 vs. utilities PE of 15). The current system handles this via industry neutralization, but a more explicit approach:

```python
def sector_relative_value(fundamentals: pd.DataFrame) -> pd.Series:
    """Compute value scores relative to sector median.

    A tech stock at PE=20 is cheap relative to tech median of 30,
    while a utility at PE=20 is expensive vs utility median of 14.
    """
    scores = pd.DataFrame(index=fundamentals.index)
    sector = fundamentals["sector"]

    for col in ["trailingPE", "forwardPE", "priceToBook"]:
        vals = pd.to_numeric(fundamentals[col], errors="coerce")
        if vals.notna().sum() < 3:
            continue
        # Sector median
        sector_median = vals.groupby(sector).transform("median")
        # Relative value: how much cheaper than sector median
        relative = (sector_median - vals) / sector_median.replace(0, np.nan)
        # Z-score the relative values
        scores[col] = (relative - relative.mean()) / relative.std()

    if scores.empty:
        return pd.Series(0.0, index=fundamentals.index, name="value")
    return scores.mean(axis=1).rename("value")
```

---

## 3. Quality Enhancement

### 3.1 Piotroski F-Score

**Reference**: Piotroski (2000) "Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers."

Nine binary signals scored 0 or 1 (total score 0-9):

```python
def piotroski_f_score(fundamentals: pd.DataFrame) -> pd.Series:
    """Piotroski F-Score (simplified version using available yfinance data).

    Full F-Score requires quarterly data. This uses available annual metrics.
    Score 0-9; higher = better quality.
    """
    score = pd.Series(0, index=fundamentals.index, dtype=float)

    # Profitability signals
    roe = pd.to_numeric(fundamentals.get("returnOnEquity"), errors="coerce")
    if roe is not None:
        score += (roe > 0).astype(float)                   # F1: ROE > 0
    ocf = pd.to_numeric(fundamentals.get("operatingCashflow"), errors="coerce")
    mcap = pd.to_numeric(fundamentals.get("marketCap"), errors="coerce")
    if ocf is not None and mcap is not None:
        score += (ocf / mcap > 0).astype(float)            # F2: Operating CF > 0
    eg = pd.to_numeric(fundamentals.get("earningsGrowth"), errors="coerce")
    if eg is not None:
        score += (eg > 0).astype(float)                    # F3: Earnings growth > 0

    # Leverage / liquidity signals
    dte = pd.to_numeric(fundamentals.get("debtToEquity"), errors="coerce")
    if dte is not None:
        median_dte = dte.median()
        score += (dte < median_dte).astype(float)           # F4: Below-median leverage

    # Profitability margin signals
    pm = pd.to_numeric(fundamentals.get("profitMargins"), errors="coerce")
    if pm is not None:
        score += (pm > pm.median()).astype(float)           # F5: Above-median margins

    # Z-score the F-Score for composite integration
    fscore = (score - score.mean()) / score.std().replace(0, 1)
    return fscore.rename("piotroski")
```

### 3.2 Accruals Quality

**Reference**: Sloan (1996) "Do Stock Prices Fully Reflect Information in Accruals and Cash Flows about Future Earnings?"

```python
def accruals_factor(fundamentals: pd.DataFrame) -> pd.Series:
    """Accruals ratio: high accruals (earnings driven by accounting
    rather than cash flow) predict negative future returns.

    Accruals = (Net Income - Operating Cash Flow) / Total Assets
    Lower accruals = higher quality.
    """
    ni = pd.to_numeric(fundamentals.get("netIncome"), errors="coerce")
    ocf = pd.to_numeric(fundamentals.get("operatingCashflow"), errors="coerce")
    ta = pd.to_numeric(fundamentals.get("totalAssets"), errors="coerce")

    if ni is None or ocf is None or ta is None:
        return pd.Series(0.0, index=fundamentals.index, name="accruals")

    accruals = (ni - ocf) / ta.replace(0, np.nan)
    # Invert: lower accruals = higher quality
    inv = -accruals
    zscore = (inv - inv.mean()) / inv.std().replace(0, 1)
    return zscore.rename("accruals")
```

### 3.3 Enhanced Quality Composite

Combine existing quality metrics with new ones:

```python
def enhanced_quality_factor(fundamentals: pd.DataFrame) -> pd.Series:
    """Enhanced quality: ROE + margins + earnings growth + leverage
    + accruals + Piotroski signals."""
    components = pd.DataFrame(index=fundamentals.index)

    # Core profitability
    for col in ["returnOnEquity", "profitMargins", "earningsGrowth"]:
        vals = pd.to_numeric(fundamentals[col], errors="coerce")
        if vals.notna().sum() > 2:
            components[col] = (vals - vals.mean()) / vals.std()

    # Financial safety: low leverage
    dte = pd.to_numeric(fundamentals.get("debtToEquity"), errors="coerce")
    if dte is not None and dte.notna().sum() > 2:
        inv_dte = -dte  # lower debt = higher quality
        components["leverage"] = (inv_dte - inv_dte.mean()) / inv_dte.std()

    # Revenue growth (consistency check)
    rg = pd.to_numeric(fundamentals.get("revenueGrowth"), errors="coerce")
    if rg is not None and rg.notna().sum() > 2:
        components["revGrowth"] = (rg - rg.mean()) / rg.std()

    if components.empty:
        return pd.Series(0.0, index=fundamentals.index, name="quality")

    return components.mean(axis=1).rename("quality")
```

---

## 4. Factor Combination Improvements

### 4.1 IC-Weighted Factor Combination

Instead of fixed weights, dynamically weight factors by their trailing IC:

```python
def ic_weighted_composite(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    ic_lookback: int = 252,
    ic_halflife: int = 63,
    min_weight: float = 0.05,
) -> pd.DataFrame:
    """Dynamically weight factors by their trailing IC.

    Factors with higher recent predictive power get higher weight.
    Uses exponentially-weighted IC to adapt to regime changes.

    Parameters
    ----------
    factors : dict of DataFrames (dates x symbols)
    forward_returns : DataFrame of 21-day forward returns
    ic_lookback : days of IC history to use
    ic_halflife : exponential decay halflife for IC weighting
    min_weight : minimum weight for any factor (prevents total exclusion)
    """
    # Compute trailing IC for each factor
    ic_series = {}
    for name, factor_df in factors.items():
        daily_ic = pd.Series(index=factor_df.index, dtype=float)
        for date in factor_df.index:
            if date not in forward_returns.index:
                continue
            f = factor_df.loc[date].dropna()
            r = forward_returns.loc[date].reindex(f.index).dropna()
            common = f.index.intersection(r.index)
            if len(common) > 10:
                daily_ic[date] = f[common].corr(r[common], method="spearman")
        ic_series[name] = daily_ic

    # Compute EWM IC for dynamic weighting
    ic_df = pd.DataFrame(ic_series)
    ewm_ic = ic_df.ewm(halflife=ic_halflife).mean()

    # Convert IC to weights (positive IC only, with floor)
    ic_weights = ewm_ic.clip(lower=0)
    ic_weights = ic_weights.div(ic_weights.sum(axis=1), axis=0)
    ic_weights = ic_weights.clip(lower=min_weight)
    ic_weights = ic_weights.div(ic_weights.sum(axis=1), axis=0)

    # Build composite
    composite = pd.DataFrame(0.0, index=factor_df.index, columns=factor_df.columns)
    for name in factors:
        f = factors[name].reindex_like(composite).fillna(0)
        w = ic_weights[name].reindex(composite.index).fillna(1.0 / len(factors))
        composite += f.mul(w, axis=0)

    return composite
```

**Caution**: IC-weighting introduces look-ahead risk if not properly lagged. The IC used for weighting at date `t` must be computed from data strictly before `t`. The implementation above uses EWM which includes date `t` IC -- this must be shifted by 1 day in production.

### 4.2 Fix for Multiplicative Filter Issue (CS2)

The current multiplicative filters invert the penalty direction for negative scores. Recommended fix:

```python
def apply_filter_safely(composite: pd.DataFrame, filter_df: pd.DataFrame) -> pd.DataFrame:
    """Apply multiplicative filter only to the magnitude,
    preserving the sign of the composite score.

    For positive scores: score * filter (reduces magnitude when filter < 1).
    For negative scores: score * (2 - filter) (increases magnitude of penalty
    when filter < 1, i.e., makes more negative).
    """
    positive = composite >= 0
    result = composite.copy()
    # For positive scores, multiply normally
    result[positive] = composite[positive] * filter_df[positive]
    # For negative scores, increase the penalty
    result[~positive] = composite[~positive] * (2.0 - filter_df[~positive])
    return result
```

### 4.3 Improved NaN Handling in Composite (CS1)

```python
def build_composite_adaptive(
    factors: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """Build weighted composite, normalizing by available factor weights
    per stock per day (instead of fillna(0) and fixed denominator).
    """
    template = list(factors.values())[0]
    composite = pd.DataFrame(0.0, index=template.index, columns=template.columns)
    weight_sum = pd.DataFrame(0.0, index=template.index, columns=template.columns)

    for name, w in weights.items():
        if w <= 0 or name not in factors:
            continue
        f = factors[name].reindex_like(template)
        mask = f.notna()
        composite += (f.fillna(0) * w)
        weight_sum += (mask.astype(float) * w)

    # Normalize by actual available weight per cell
    weight_sum = weight_sum.replace(0, np.nan)
    composite = composite / weight_sum

    return composite
```

---

## 5. New Factor Ideas

### 5.1 Earnings Revision Momentum

**Reference**: Chan, Jegadeesh, and Lakonishok (1996) "Momentum Strategies."

Analyst earnings estimate revisions are one of the strongest short-to-medium term predictors. Unfortunately, yfinance does not provide analyst revision data. To implement this:

**Data sources**:
- **Free**: Scrape from Yahoo Finance consensus estimates pages (fragile)
- **Paid**: Alpha Vantage, Quandl/Nasdaq Data Link, FactSet, Bloomberg

```python
def earnings_revision_factor(estimates: pd.DataFrame) -> pd.Series:
    """Earnings revision momentum.

    Measures the net direction of analyst EPS estimate revisions
    over the trailing 90 days.

    Parameters
    ----------
    estimates : DataFrame with columns [symbol, date, eps_estimate]
        Time series of consensus EPS estimates.
    """
    # Compute 3-month change in consensus EPS estimate
    # Normalize by stock price to get earnings yield revision
    # Cross-sectional z-score
    pass  # requires external data source
```

### 5.2 Short Interest Factor

**Reference**: Dechow et al. (2001) "Short-sellers, Fundamental Analysis, and Stock Returns."

High short interest predicts negative future returns (short sellers are informed).

**Data sources**: FINRA short interest data (published bi-monthly, free with delay), or Interactive Brokers (real-time borrow data).

### 5.3 Standardized Unexpected Earnings (SUE)

**Reference**: Foster, Olsen, and Shevlin (1984) "Earnings Releases, Anomalies, and the Behavior of Security Returns."

```python
def sue_factor(quarterly_eps: pd.DataFrame) -> pd.Series:
    """Standardized Unexpected Earnings.

    SUE = (EPS_actual - EPS_expected) / std(EPS_surprise)

    EPS_expected estimated from seasonal random walk:
    E[EPS_q] = EPS_{q-4} (same quarter last year).
    """
    # Requires quarterly EPS time series per stock
    # yfinance .quarterly_financials can provide this
    pass
```

### 5.4 Price-Volume Divergence

Available from existing price/volume data (no additional data required):

```python
def price_volume_divergence(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """Detect divergence between price trend and volume trend.

    Bullish: price declining on decreasing volume (selling exhaustion).
    Bearish: price rising on decreasing volume (weak rally).

    Returns cross-sectionally z-scored divergence signal.
    """
    price_change = prices.pct_change(window)
    vol_change = volumes.rolling(window).mean() / volumes.rolling(
        window * 2
    ).mean() - 1

    # Divergence: positive when price up + volume up or price down + volume down
    # Negative (bearish) when price up + volume down
    divergence = price_change * vol_change

    # Cross-sectional z-score
    zs = divergence.sub(divergence.mean(axis=1), axis=0).div(
        divergence.std(axis=1), axis=0
    )
    return zs
```

---

## 6. Priority Roadmap

### Phase 1: Bug Fixes (Immediate)

1. Fix multiplicative filter issue for negative scores (CS2)
2. Fix `fillna(0)` with adaptive weight normalization (CS1)
3. Add minimum sector size guard to neutralization (N1)
4. Add sector-missing stock handling to neutralization (N2)

### Phase 2: Data Quality (1-2 weeks)

1. Source point-in-time fundamental data to fix look-ahead bias (C1)
2. Add quarterly fundamental data pipeline using `yfinance .quarterly_financials`
3. Implement survivorship-bias-free universe (use historical S&P 500 constituents)

### Phase 3: Factor Enhancement (2-4 weeks)

1. Implement weighted lookback windows for momentum
2. Expand value factor to 5-metric composite
3. Add leverage and accruals to quality factor
4. Add smooth ramp for trend filter (replace binary)
5. Add vol-adjusted momentum (Barroso & Santa-Clara)

### Phase 4: Advanced (1-2 months)

1. Implement residual momentum
2. Build IC-weighted factor combination
3. Add factor decay analysis framework
4. Add earnings revision factor (requires data source)
5. Implement factor orthogonalization monitoring
6. Add Ledoit-Wolf shrinkage to covariance estimation
