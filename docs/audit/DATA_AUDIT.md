# Data Engineering Audit Report

**Auditor**: Data Engineer Agent
**Date**: 2026-03-15
**Scope**: Data acquisition, quality, bias analysis for multi-factor quant trading system

---

## Executive Summary

The system has **two critical biases** that invalidate any backtest performance claims, **several moderate data quality gaps**, and a handful of minor issues. The critical biases are (1) look-ahead bias in fundamental data and (2) survivorship bias in the static universe. Both are systemic and cannot be patched away without external data sources. The fixes delivered here add warnings, guardrails, and quality checks, but the fundamental data problem requires a point-in-time database to fully resolve.

---

## 1. Data Source Reliability

### 1.1 Price Data (yfinance adjusted close)

**Source**: `yf.download()` with `auto_adjust=True` (line 39-46, `market_data.py`).

- **Split/dividend adjustment**: `auto_adjust=True` means yfinance returns prices that are retroactively adjusted for splits and dividends. This is correct for total-return backtesting. However, yfinance's adjustment methodology silently changes: Yahoo may revise historical adjusted prices at any time when new corporate actions are filed. There is no versioning or audit trail.
- **Data vendor risk**: yfinance is an unofficial scraper of Yahoo Finance. It has no SLA, no guaranteed uptime, and Yahoo has repeatedly changed its API surface. Rate limiting and transient failures are common (the retry logic at lines 81-88 partially addresses this).
- **Frequency**: Daily bars (`1d`). The `frequency` config parameter is passed directly to yfinance, so intraday data is theoretically possible but untested in this codebase.
- **Close price definition**: The "Close" column from `auto_adjust=True` is the adjusted close. This means the price on any given historical date does NOT match the actual trading price on that date -- it has been retroactively adjusted. This is correct for return calculation but can cause issues if used for order sizing at current prices (see `get_current_portfolio`, line 162 of `strategy.py`, which uses `prices[symbols].iloc[-1]` -- for the most recent date this equals the actual close, so this is fine).

**Verdict**: Acceptable for a research system. Not suitable for production without a more reliable vendor (Polygon, Databento, etc.).

### 1.2 Fundamental Data (yfinance .info)

**Source**: `yf.Ticker(symbol).info` (line 83, `market_data.py`).

- **CRITICAL**: `.info` returns a **single current snapshot** of fundamental ratios. There is no historical time series. Fields like `trailingPE`, `forwardPE`, `priceToBook`, `returnOnEquity`, `profitMargins`, and `earningsGrowth` all reflect the **latest available** values as of the API call date.
- **Coverage**: Not all fields are populated for all symbols. The code handles this with `.get(f)` (line 112), which returns `None` for missing fields. The factor functions then use `errors="coerce"` to handle non-numeric values.
- **Staleness**: Some `.info` fields may lag by days or weeks depending on Yahoo's data pipeline. There is no way to determine the "as-of" date for any given field.

**Verdict**: Unsuitable for backtesting. Acceptable for live forward-looking signals only, with the caveat that data may be stale.

### 1.3 Data Update Delays

- Price data: Yahoo Finance typically publishes end-of-day data 15-30 minutes after market close. For a system rebalancing monthly, this is not a concern.
- Fundamental data: Unknown lag. Earnings data may take hours to days to propagate through Yahoo's pipeline after a company reports.

---

## 2. Look-Ahead Bias Analysis (CRITICAL)

### 2.1 Fundamental Data Look-Ahead Bias

**Severity**: CRITICAL
**Location**: `market_data.py` lines 94-122, `factors.py` lines 167-199, `strategy.py` lines 47-50, `factors.py` lines 262-271.

**The chain of contamination**:

1. `strategy.py` line 47: `fundamentals = self.data.fetch_fundamentals()` fetches a single snapshot of current fundamentals.
2. `factors.py` lines 262-268: The `SignalGenerator.generate()` method broadcasts this static snapshot to **every historical date**:
   ```python
   df = pd.DataFrame(
       np.tile(series.values, (len(px), 1)),
       index=px.index,
       columns=series.index,
   )
   ```
3. This means a backtest starting 2016-01-01 uses 2026 fundamental ratios for stock selection decisions in 2016. A stock that had a PE of 50 in 2016 but a PE of 15 in 2026 would appear "cheap" throughout the entire backtest.

**Impact estimate**: With the current config, value weight = 0.05 and quality weight = 0.25, so 30% of the composite signal is contaminated. Quality has the second-highest weight. This can significantly inflate backtest Sharpe by favoring stocks that turned out to have good fundamentals.

### 2.2 Price Data at Rebalance Time

**Location**: `strategy.py` lines 68-96, `engine.py` lines 78-99.

On a rebalance date `date`:
- `signals.loc[date]` uses the signal computed from prices up to and including `date` (line 72).
- The signal is computed from `prices` which includes the close on `date`.
- The backtest engine then executes trades at `prices.loc[date]` (line 88: `target_shares = (portfolio_value * target / px)`).

**Issue**: The signal uses the **same-day close** to generate a signal, and the trade executes at **that same close price**. In reality, you cannot observe the close price and also trade at that close price. You would need to use the previous day's close for signal generation, or assume next-day-open execution.

**Severity**: MEDIUM. For a monthly rebalance strategy, using close vs next-open is a ~5-15bps difference per rebalance, which compounds over time but is dwarfed by the fundamental data bias.

### 2.3 Momentum Factor Shift Analysis

**Location**: `factors.py` line 89.

```python
ret = prices.shift(skip).pct_change(w)
```

Where `skip = 21` (one month of trading days) and `w` is the lookback window (63, 126, or 252).

**Analysis**: `prices.shift(21)` shifts the entire price series forward by 21 days, then `.pct_change(252)` computes the return over 252 periods of this shifted series. The result on date `t` is:

```
ret[t] = prices[t-21] / prices[t-21-w] - 1
```

This means: on date `t`, the momentum signal uses the return from `t-21-w` to `t-21`. The most recent 21 days are skipped. This is **correct** -- it implements the standard "skip-month" momentum to avoid the short-term reversal effect. No look-ahead bias here.

### 2.4 Rolling Window Factors

- `mean_reversion_factor` (line 105-108): Uses `prices.rolling(window).mean()` which is backward-looking. No bias.
- `trend_factor` (line 118-123): Uses `prices.rolling(short_window).mean()` and `prices.rolling(long_window).mean()`. Backward-looking. No bias.
- `volatility_factor` (line 158): Uses `returns.rolling(window).std()`. Backward-looking. No bias.

### 2.5 Covariance Estimation

**Location**: `strategy.py` line 79.

```python
ret_window = returns.loc[:date].tail(126)
```

This uses returns up to and including `date`. The covariance matrix for portfolio optimization includes the return on the rebalance date itself. This is a minor issue -- the covariance is used for risk estimation, not alpha, and one day's data in a 126-day window has negligible impact.

---

## 3. Survivorship Bias

### 3.1 Static Universe

**Location**: `config.yaml` lines 6-118.

The universe is a hardcoded list of 100 currently-active large-cap US stocks. This list was presumably curated in 2025-2026 based on current market capitalization and trading volume.

**Problems**:
- Stocks that were large-cap in 2016 but subsequently declined, were acquired, or went bankrupt are excluded. Examples for a 2016-2026 backtest:
  - **GE**: Was a Dow component, lost ~75% of its value, split into three companies.
  - **XLNX**: Acquired by AMD in 2022.
  - **ATVI**: Acquired by MSFT in 2023.
  - **TWTR**: Taken private by Elon Musk in 2022.
  - **FRC**: First Republic Bank, collapsed in 2023.
  - **SIVB**: Silicon Valley Bank, collapsed in 2023.
- Stocks that are now large-cap but were small/mid-cap in 2016 are included from the start, giving them an "early bird" advantage. Examples: NVDA was ~$30B market cap in 2016 vs ~$3T now; CRWD IPO'd in 2019.

**Impact estimate**: Academic literature suggests survivorship bias inflates annual returns by 0.5-1.5% for large-cap universes and 2-4% for small-cap. Given this is a large-cap universe with monthly rebalancing, the bias is likely in the 1-2% annual return range -- enough to flip a marginal Sharpe from <1 to >1.

### 3.2 No Dynamic Universe Mechanism

There is no infrastructure for:
- Point-in-time index membership (e.g., "which stocks were in the S&P 500 on 2018-03-15?")
- Handling delistings or acquisitions during a backtest
- Marking stocks as untradeable after corporate events

---

## 4. Data Cleaning

### 4.1 Split Adjustment

`auto_adjust=True` is set in both `yf.download()` (line 44) and `ticker.history()` (line 67). This returns prices adjusted for splits and dividends. **Verified correct.**

However, there is a subtlety: `auto_adjust=True` in recent yfinance versions replaces OHLC with adjusted values AND drops the original "Adj Close" column. The code uses `data["Close"]` (line 49), which is the adjusted close. This is correct.

### 4.2 Missing Value Handling

- `fetch_prices()` line 53: `prices.dropna(how="all")` drops rows where ALL symbols are NaN. This preserves rows where some symbols have data and others don't.
- `compute_returns()` line 131: `pct_change().dropna(how="all")` same approach.
- Factor functions: No explicit NaN handling. `rolling().mean()` and `rolling().std()` propagate NaN for the first `window-1` rows, which is correct (no data, no signal).
- Composite signal (line 294): `f.fillna(0)` fills NaN factor scores with 0 before weighted sum. **This is problematic**: a stock with missing data gets a neutral score rather than being excluded. A stock with data in only one factor effectively has all other factors zeroed out, which distorts the composite.

**Recommendation**: Replace `fillna(0)` with tracking of valid-factor counts per symbol and normalizing by actual contributing weight, or exclude symbols with fewer than N valid factors.

### 4.3 Outlier Detection

- `winsorize_zscore()` (lines 23-34): Clips factor z-scores to [-3, +3] and re-z-scores. This is a reasonable approach for cross-sectional outliers.
- **Missing**: No outlier detection on raw price data. A bad tick (e.g., price drops 99% for one day) would propagate into momentum, mean-reversion, and volatility factors before winsorization can catch it.

### 4.4 Suspended Stock Handling

**Not implemented**. If a stock is halted or suspended:
- yfinance may return the last valid price (forward-fill) or NaN depending on the duration.
- The backtest engine (line 79): `px = prices.loc[date, symbols].ffill()` forward-fills within the cross-section on each date, which incorrectly fills NaN for one symbol with data from the previous symbol alphabetically (since ffill on a Series fills from the prior element). This should be axis-aware -- it should forward-fill along time, not across symbols.

**Bug**: `engine.py` line 79 calls `.ffill()` on a Series (one date, all symbols). This forward-fills across symbols sorted by column order, which is meaningless. It should be done on the time axis before slicing to a single date.

---

## 5. Timezone & Alignment

### 5.1 Price Data Timezone

yfinance returns daily data with a DatetimeIndex that is timezone-naive (midnight timestamps). The code does not set or convert timezones anywhere. This is acceptable for daily US equities (all prices are NYSE close, ~4:00 PM ET) as long as no cross-timezone data is mixed in.

### 5.2 Cross-Source Date Alignment

**Fundamental data** has no date dimension -- it's a single snapshot. When broadcast to all dates in `factors.py` lines 262-268, it's aligned by symbol only. There is no mismatch risk because there's only one set of values.

However, this also means there's no handling of the quarterly cadence of earnings. In a proper system, fundamental data would update quarterly with a reporting lag, and the factor would use the most recently available value at each point in time.

### 5.3 Weekend/Holiday Handling

yfinance returns data only for trading days, so weekends and holidays are automatically excluded from the price DataFrame. The `pct_change()` calculation correctly computes day-over-day returns across trading days (e.g., Friday-to-Monday is one return).

The rebalance logic (`strategy.py` line 61: `prices.index[::rebalance_freq]`) steps through the index by count, not by calendar date. This means rebalance happens every 21 trading days regardless of month boundaries. This is unconventional but not incorrect -- it's a design choice documented in the project overview.

---

## 6. Additional Findings

### 6.1 Forward-Fill Bug in Backtest Engine

**Location**: `engine.py` line 79.

```python
px = prices.loc[date, symbols].ffill()
```

`prices.loc[date, symbols]` returns a `pd.Series` (one row). Calling `.ffill()` on a Series fills NaN values with the preceding non-NaN value in index order (which is alphabetical by symbol). This means if AAPL has a price but ABBV is NaN, ABBV gets AAPL's price. This is a bug.

**Fix**: Forward-fill along the time axis before the daily loop, or use `prices.ffill()` on the full DataFrame (time-axis) before the backtest starts.

### 6.2 Cache Staleness

`_info_cache` in `MarketData` (line 23) caches fundamental data for the lifetime of the object. In a long-running process (e.g., paper trading), stale fundamentals would persist. There is no TTL or invalidation mechanism.

### 6.3 Batch Download vs Single-Symbol Inconsistency

`fetch_prices()` uses `yf.download()` (batch), while `fetch_ohlcv()` uses `yf.Ticker().history()` (single). These may return slightly different adjusted prices due to rounding or timing of the adjustment calculation. The two methods should not be mixed for the same analysis.

---

## 7. Risk Matrix

| Issue | Severity | Impact on Backtest | Impact on Live | Fix Complexity |
|-------|----------|-------------------|----------------|----------------|
| Fundamental look-ahead bias | CRITICAL | Inflates Sharpe by ~0.2-0.5 | None (live uses current data) | HIGH (needs PIT database) |
| Survivorship bias | CRITICAL | Inflates annual return by ~1-2% | None (live uses current universe) | HIGH (needs PIT index membership) |
| Same-day close signal + execution | MEDIUM | ~5-15bps per rebalance | LOW (live can lag by 1 day) | LOW (shift signal by 1 day) |
| ffill bug in backtest engine | MEDIUM | Random price contamination | N/A (backtest only) | LOW (fix one line) |
| fillna(0) in composite | LOW-MEDIUM | Distorts scores for partial-data stocks | Same | LOW |
| No raw-price outlier detection | LOW | Rare but catastrophic when hit | Same | MEDIUM |
| No suspended stock handling | LOW | Occasional bad fills | MEDIUM for live | MEDIUM |

---

## 8. Recommendations (Priority Order)

1. **Immediate**: Set value and quality factor weights to 0 for any backtest used to evaluate strategy performance. Only re-enable after obtaining a point-in-time fundamental database.
2. **Immediate**: Fix the `ffill()` bug in `engine.py` line 79.
3. **Short-term**: Add the `DataQualityChecker` (delivered in this audit) to the backtest pipeline.
4. **Short-term**: Shift signal generation to use `t-1` close for rebalance-day decisions.
5. **Medium-term**: Integrate a point-in-time fundamental database (Sharadar, SimFin, or Compustat).
6. **Medium-term**: Replace the static universe with a point-in-time index membership dataset.
7. **Long-term**: Replace yfinance with a production-grade data vendor for price data.
