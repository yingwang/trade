# Data Fix Log

**Author**: Data Engineer Agent
**Date**: 2026-03-15

---

## Fix 1: DataQualityChecker and PointInTimeDataManager

**File created**: `/quant/data/quality.py`

**Problem**: No data quality validation existed anywhere in the pipeline. Bad ticks, missing data, stale prices, and look-ahead bias in fundamentals could silently corrupt signals and backtest results.

**What was added**:

### DataQualityChecker
- `check_missing_values()`: Reports per-symbol missing data rate. Threshold: 5%.
- `check_extreme_returns()`: Flags daily returns exceeding +/-50% (likely bad ticks or unhandled corporate actions).
- `check_data_continuity()`: Detects gaps in the trading day index exceeding 5 calendar days (data feed outages, unexpected holidays).
- `check_cross_symbol_consistency()`: Reports symbols that start trading significantly later than others (e.g., IPOs mid-backtest like CRWD in 2019).
- `check_stale_prices()`: Detects runs of 5+ identical consecutive closing prices (data feed stalling or exchange halt).
- `run_all_checks()`: Runs all checks and returns a structured report with pass/fail status.

### PointInTimeDataManager
- Wraps fundamental data with explicit documentation of the look-ahead bias.
- In backtest mode, emits a `UserWarning` on first access.
- Supports a configurable `reporting_lag_days` (default 90) to model the delay between fiscal period end and data availability.
- Provides a hook for supplying external point-in-time data in the future.

### warn_survivorship_bias()
- Standalone function that emits a warning when a static universe is used for backtesting.
- Called at the start of `run_backtest()`.

---

## Fix 2: Look-Ahead Bias Warning in fetch_fundamentals()

**File modified**: `/quant/data/market_data.py` (lines 94-110)

**Before**:
```python
def fetch_fundamentals(self, batch_size: int = 50, batch_delay: float = 1.5) -> pd.DataFrame:
    """Return a DataFrame of key fundamental ratios for the universe.
    ..."""
    rows = []
```

**After**:
```python
def fetch_fundamentals(self, batch_size: int = 50, batch_delay: float = 1.5,
                       is_backtest: bool = False) -> pd.DataFrame:
    """Return a DataFrame of key fundamental ratios for the universe.

    WARNING: yfinance .info returns a CURRENT snapshot of fundamentals.
    This data is NOT point-in-time and will cause look-ahead bias if
    used in a historical backtest. ...
    """
    if is_backtest:
        warnings.warn(
            "fetch_fundamentals() returns CURRENT snapshot data ...",
            UserWarning,
            stacklevel=2,
        )
    rows = []
```

**Rationale**: Anyone calling `fetch_fundamentals(is_backtest=True)` now receives an explicit runtime warning. The docstring also documents the limitation permanently.

---

## Fix 3: Strategy Backtest Pipeline Integration

**File modified**: `/quant/strategy.py` (lines 1-50)

**Before**:
```python
from quant.data.market_data import MarketData
from quant.signals.factors import SignalGenerator
...

def run_backtest(self, ...):
    ...
    fundamentals = self.data.fetch_fundamentals()
```

**After**:
```python
from quant.data.market_data import MarketData
from quant.data.quality import (
    DataQualityChecker,
    PointInTimeDataManager,
    warn_survivorship_bias,
)
from quant.signals.factors import SignalGenerator
...

def run_backtest(self, ...):
    # Survivorship bias warning
    warn_survivorship_bias(self.data.symbols, start)

    # Data quality validation
    checker = DataQualityChecker()
    quality_report = checker.run_all_checks(prices)
    ...

    # Point-in-time wrapper
    fundamentals = self.data.fetch_fundamentals(is_backtest=True)
    pit = PointInTimeDataManager(fundamentals, is_backtest=True, reporting_lag_days=90)
    fundamentals = pit.get_fundamentals()
```

**Rationale**: Every backtest run now:
1. Warns about survivorship bias at startup.
2. Validates price data quality before signal generation.
3. Wraps fundamental data with look-ahead bias documentation and warnings.

---

## Fix 4: Forward-Fill Bug in Backtest Engine

**File modified**: `/quant/backtest/engine.py` (line 79)

**Before**:
```python
for date in dates:
    px = prices.loc[date, symbols].ffill()
```

**Problem**: `prices.loc[date, symbols]` returns a `pd.Series` indexed by symbol. Calling `.ffill()` on this Series fills NaN values with the previous symbol's price in alphabetical order. For example, if ABBV is NaN and AAPL is $150, ABBV would get $150 -- a completely meaningless cross-contamination.

**After**:
```python
# Forward-fill missing prices along the TIME axis (not cross-symbol)
prices_ffilled = prices[symbols].ffill()
...
for date in dates:
    px = prices_ffilled.loc[date, symbols]
```

**Rationale**: Forward-filling on the full DataFrame before the loop correctly propagates each symbol's last known price forward in time. A halted stock at $50 keeps $50 until it resumes trading, rather than inheriting another stock's price.

---

## Summary of Changes

| # | File | Type | Severity Addressed |
|---|------|------|--------------------|
| 1 | `quant/data/quality.py` | NEW | CRITICAL (bias warnings), MEDIUM (data quality) |
| 2 | `quant/data/market_data.py` | MODIFIED | CRITICAL (look-ahead documentation) |
| 3 | `quant/strategy.py` | MODIFIED | CRITICAL (pipeline integration) |
| 4 | `quant/backtest/engine.py` | MODIFIED | MEDIUM (ffill bug) |

## Remaining Work (Not Fixed Here)

| Issue | Reason Not Fixed | Recommended Action |
|-------|------------------|--------------------|
| Point-in-time fundamental database | Requires external data subscription | Integrate Sharadar or SimFin |
| Dynamic universe / survivorship-free | Requires historical index membership data | Use S&P 500 historical constituents |
| Same-day close signal + execution | Requires architectural change to signal/execution split | Shift signals to use `t-1` close |
| fillna(0) in composite signal | Requires factor-availability tracking | Track valid factors per symbol, normalize by actual weight |
| Raw price outlier detection | Requires defining policy for corporate actions vs bad ticks | Add pre-factor median-filter or return-cap |
