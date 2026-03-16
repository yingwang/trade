# Execution Layer Fix Log

**Engineer**: Execution Engineer Agent
**Date**: 2026-03-15

Each section documents what was wrong, what was changed, and why.

---

## Fix 1: Pre-trade Safety Checks

**Problem**: Zero validation between the optimizer output and Alpaca order submission. A single bug in signal generation or weight calculation could produce an order of arbitrary dollar value.

**Files changed**: New file `quant/execution/safety.py`, modified `quant/execution/alpaca_broker.py`

**What was added**:
- `SafetyConfig` dataclass with configurable limits (max order value $50k default, max daily trade value $500k, max daily loss $25k, max ADV fraction 1%, min price $1, max position 15% of portfolio)
- `PreTradeCheck` class with `validate()` method that checks all limits before any order is submitted
- `DailyTracker` that accumulates intra-day trade value and P&L, auto-resets each day
- Every order in `AlpacaBroker.submit_order()` now passes through `PreTradeCheck.validate()` before reaching the Alpaca API

**Before**:
```python
def submit_order(self, order: Order) -> Order:
    try:
        alpaca_order = self.api.submit_order(...)  # No checks at all
```

**After**:
```python
def submit_order(self, order: Order, avg_daily_volume: float = None) -> Order:
    # Pre-trade safety check
    passed, reason = self.safety.validate(order, price, portfolio_value, avg_daily_volume)
    if not passed:
        order.status = "rejected"
        order.reject_reason = reason
        self.exec_log.log_safety_block(order, reason)
        return order
    # ... then submit to Alpaca
```

---

## Fix 2: Post-trade Reconciliation

**Problem**: After rebalancing, no comparison between strategy targets and actual Alpaca positions. Partial fills, rejected orders, or external trades would cause silent drift.

**Files changed**: `quant/execution/safety.py`, `quant/execution/alpaca_broker.py`, `paper_trade.py`

**What was added**:
- `PostTradeReconciler` class that computes per-symbol drift between target weights and actual positions
- `AlpacaBroker.reconcile(target_weights)` method that fetches actual positions and prices, runs reconciliation, and logs results to structured JSON
- `paper_trade.py` now calls `broker.reconcile(target_weights)` after every rebalance
- `--reconcile` CLI flag for on-demand reconciliation
- Warning at 2% drift, alert at 5% drift per position

**Before**: No reconciliation existed.

**After**:
```python
# In paper_trade.py, after rebalance:
logger.info("Running post-trade reconciliation...")
drift_df = broker.reconcile(target_weights)
```

---

## Fix 3: TWAP Order Splitting

**Problem**: Large orders submitted as single market orders could move the market against the portfolio, especially in less-liquid names.

**Files changed**: `quant/execution/safety.py`, `quant/execution/alpaca_broker.py`

**What was added**:
- `TWAPSplitter` class that splits orders exceeding 1% of ADV into 5 child slices spread over 30 minutes
- `AlpacaBroker.submit_order()` checks `TWAPSplitter.should_split()` and delegates to `_execute_twap()` when needed
- Each TWAP slice is a separate market order with configurable inter-slice delay
- Parent order aggregates all child fills (weighted average price)

**Before**: Every order submitted as one shot regardless of size.

**After**: Orders > 1% of ADV are automatically split into 5 time-weighted slices.

---

## Fix 4: Limit Order Support

**Problem**: `generate_rebalance_orders()` hardcoded `order_type="market"` with no way to use limit orders.

**Files changed**: `quant/execution/broker.py`

**What was added**:
- `order_type` and `limit_offset_bps` parameters to `generate_rebalance_orders()`
- When `order_type="limit"`, limit prices are calculated as `price +/- offset` (added for buys, subtracted for sells)
- Orders are now sorted sells-first by default (free up cash before buying)
- `signal_price` field populated on each order for slippage tracking

**Before**:
```python
orders.append(Order(symbol=sym, side="buy", quantity=delta, order_type="market"))
```

**After**:
```python
order = Order(
    symbol=sym, side=side, quantity=qty,
    order_type=otype, limit_price=limit_price, signal_price=price,
)
orders.sort(key=lambda o: (0 if o.side == "sell" else 1, o.symbol))
```

---

## Fix 5: Partial Fill Handling

**Problem**: `_wait_for_fill()` only recognized `"filled"` status. Alpaca's `"partially_filled"` status was treated as still-pending, and after timeout the partial fill was lost.

**Files changed**: `quant/execution/alpaca_broker.py`

**What was added**:
- `_wait_for_fill()` now checks for `"partially_filled"` and returns the order on timeout (capturing partial fill data)
- `_execute_single()` detects partial fills and sets `order.status = "partial_fill"`, updating `order.quantity` to actual filled quantity
- Warning log emitted for every partial fill

**Before**:
```python
if o.status == "filled":
    return o
if o.status in ("canceled", "expired", "rejected"):
    return None
```

**After**:
```python
if o.status == "filled":
    return o
if o.status == "partially_filled":
    pass  # keep waiting for market orders, but return info on timeout
if o.status in ("canceled", "expired", "rejected"):
    return None
# On timeout, check one final time for partial fill
```

---

## Fix 6: Retry Logic for Transient Errors

**Problem**: Any API error immediately rejected the order. Network timeouts, rate limits (429), and connection errors are transient and should be retried.

**Files changed**: `quant/execution/alpaca_broker.py`

**What was added**:
- `_execute_single()` retries up to 2 times on transient errors (timeout, 429, connection)
- Exponential backoff (1s, 2s) between retries
- Business logic errors (invalid symbol, insufficient funds) are not retried

**Before**:
```python
except Exception as e:
    order.status = "rejected"
```

**After**:
```python
for attempt in range(max_retries + 1):
    try:
        alpaca_order = self.api.submit_order(...)
        break
    except Exception as e:
        if attempt < max_retries and is_transient(e):
            time.sleep(2 ** attempt)
            continue
        order.status = "rejected"
```

---

## Fix 7: Structured Execution Logging

**Problem**: All trade logs were plain text via Python `logging`. No machine-readable format for monitoring, alerting, or execution quality analysis.

**Files changed**: `quant/execution/safety.py`, `quant/execution/alpaca_broker.py`, `paper_trade.py`

**What was added**:
- `ExecutionLogger` class that writes one JSON object per line to `logs/trade_events.jsonl`
- Events logged: `order_submitted`, `order_filled`, `order_rejected`, `safety_block`, `reconciliation`, `rebalance_start`, `rebalance_complete`
- Slippage tracking: every fill record includes `slippage_bps` computed as `(fill_price - signal_price) / signal_price * 10000`
- `signal_price` field added to `Order` dataclass
- `reject_reason` field added to `Order` dataclass

**Before**: Only `logger.info("Filled: %s %s %d @ $%.2f", ...)` plain text.

**After**: Every event produces a JSON line like:
```json
{"event": "order_filled", "symbol": "AAPL", "side": "buy", "quantity": 100, "filled_price": 151.23, "signal_price": 150.80, "slippage_bps": 2.85, "order_id": "...", "timestamp": "2026-03-15T19:55:12Z"}
```

---

## Fix 8: Paper-mode Safety Gate

**Problem**: Nothing prevented accidental live trading. Passing `paper=False` or setting `ALPACA_PAPER=false` would route orders to real money with no confirmation.

**Files changed**: `quant/execution/safety.py`, `quant/execution/alpaca_broker.py`

**What was added**:
- `require_paper_mode` flag in `SafetyConfig` (default: `True`)
- `AlpacaBroker.__init__()` raises `RuntimeError` if `require_paper_mode=True` and `paper=False`
- To enable live trading, you must explicitly set `require_paper_mode: false` in `config.yaml`

**Before**: `AlpacaBroker(paper=False)` would silently connect to live API.

**After**: `AlpacaBroker(paper=False)` raises:
```
RuntimeError: SAFETY: require_paper_mode is True but paper=False.
Set require_paper_mode=False in SafetyConfig to enable live trading.
```

---

## Fix 9: API Key Wiping

**Problem**: After initialization, API keys remained as plain-text `self.api_key` and `self.secret_key` instance attributes, accessible to any code with a reference to the broker.

**Files changed**: `quant/execution/alpaca_broker.py`

**What was added**:
- After passing keys to `tradeapi.REST()`, both `self.api_key` and `self.secret_key` are overwritten with `"***"`

**Before**:
```python
self.api = tradeapi.REST(self.api_key, self.secret_key, base_url, ...)
# self.api_key still contains the real key
```

**After**:
```python
self.api = tradeapi.REST(self.api_key, self.secret_key, base_url, ...)
self.api_key = "***"
self.secret_key = "***"
```

---

## Fix 10: Lock File for Concurrent Execution

**Problem**: If cron fires while a previous run is still executing (e.g., slow API), two rebalances could execute simultaneously, doubling all positions.

**Files changed**: `paper_trade.py`

**What was added**:
- `acquire_lock()` creates `logs/paper_trade.lock` with PID and start time
- `release_lock()` removes the lock file in a `finally` block
- Stale lock detection: locks older than 1 hour are automatically removed (handles crash recovery)
- Script exits with error if lock cannot be acquired

---

## Fix 11: Market Hours Check

**Problem**: No check whether the market is open before submitting orders.

**Files changed**: `quant/execution/alpaca_broker.py`, `paper_trade.py`

**What was added**:
- `AlpacaBroker.is_market_open()` method using Alpaca's clock API
- `paper_trade.py` checks market status before rebalance (warning log if closed; orders queue for next open via `time_in_force="day"`)

---

## Fix 12: Safety Config in config.yaml

**Problem**: No centralized configuration for safety limits.

**Files changed**: `config.yaml`, `quant/execution/safety.py`

**What was added**:
- New `safety:` section in `config.yaml` with all safety parameters
- `SafetyConfig.from_config()` class method to parse config dict with safe defaults
- `paper_trade.py` passes config-derived `SafetyConfig` to `AlpacaBroker`

---

## Fix 13: Test Suite

**Problem**: No tests for safety checks, reconciliation, or TWAP splitting.

**Files changed**: New file `tests/test_safety.py`

**What was added**:
- 15 tests covering:
  - `PreTradeCheck`: order value limit, share limit, penny stock rejection, position concentration, ADV check, daily cumulative limit, daily loss limit
  - `PostTradeReconciler`: no-drift case, drift detection, missing positions
  - `TWAPSplitter`: small order passthrough, large order splitting, delay progression
  - `SafetyConfig`: default construction, config override

---

## Remaining Work (Not Implemented)

These items were identified but are beyond the scope of this fix pass:

1. **Migrate to `alpaca-py`** -- The legacy SDK works but will eventually lose support. Requires rewriting all API calls.
2. **Automated drawdown circuit breaker** -- Needs a monitoring process or webhook, not just per-rebalance checks.
3. **Trading calendar integration** -- Replace `days_passed * 5/7` with Alpaca's `get_calendar()` API for exact trading day counts.
4. **Atomic state file writes** -- Write to temp file then rename, to prevent corruption on crash.
5. **Secrets manager integration** -- Move API keys from shell RC to a proper secrets manager (e.g., macOS Keychain, AWS Secrets Manager).
6. **ADV data source** -- The TWAP splitter and ADV safety check require average daily volume data to be passed in. Currently this data is not fetched by the pipeline. Need to add volume data retrieval from Alpaca or yfinance.
