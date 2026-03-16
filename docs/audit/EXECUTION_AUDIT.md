# Execution Layer Audit

**Auditor**: Execution Engineer Agent
**Date**: 2026-03-15
**Scope**: Order execution, Alpaca integration, position management, safety controls

---

## 1. Alpaca API Usage

### 1.1 API Version and SDK

| Item | Finding | Severity |
|------|---------|----------|
| SDK | `alpaca-trade-api` (legacy) | MEDIUM |
| API version | v2 (correct) | OK |
| Recommended SDK | `alpaca-py` is the actively maintained replacement | MEDIUM |

**Detail**: The code uses `alpaca-trade-api`, which Alpaca has deprecated in favor of `alpaca-py` (https://github.com/alpacahq/alpaca-py). The legacy SDK still works but receives no new features and may break on API changes. Migration is recommended but not urgent.

### 1.2 Authentication Security

| Item | Finding | Severity |
|------|---------|----------|
| Key source | Environment variables (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`) | OK |
| Key storage | `setup_alpaca.sh` writes keys to `~/.zshrc`/`~/.bashrc` in plain text | HIGH |
| Keys in code | No hardcoded keys in source files | OK |
| Instance attributes | Keys stored as `self.api_key`, `self.secret_key` in plain text on the broker object | HIGH |

**Detail**: The `setup_alpaca.sh` script appends API keys to shell RC files. This means keys are in plain text in `~/.zshrc` or `~/.bashrc`, readable by any process running as the user, and potentially committed to dotfiles repos. The AlpacaBroker constructor also stores keys as instance attributes, making them accessible via `broker.api_key` after initialization.

**Recommendation**: Use a secrets manager or at minimum `~/.config/alpaca/credentials` with `chmod 600`. Wipe key attributes after passing to SDK.

### 1.3 Paper vs Live Environment Separation

| Item | Finding | Severity |
|------|---------|----------|
| Default mode | `paper=True` in AlpacaBroker constructor | OK |
| URL switching | Correct: `paper-api.alpaca.markets` vs `api.alpaca.markets` | OK |
| Accidental live trading | No programmatic guard preventing `paper=False` | CRITICAL |
| Env var logic | `paper` parameter takes precedence over env var -- but `paper if paper else ...` is a bug | MEDIUM |

**Detail**: Line 40 of the original `alpaca_broker.py`:
```python
self.paper = paper if paper else os.environ.get("ALPACA_PAPER", "true").lower() == "true"
```
This is subtly wrong: `paper if paper` evaluates to `True` when `paper=True`, which is the default, meaning the env var `ALPACA_PAPER` is **never consulted** when the default is used. If someone passes `paper=False`, the env var is also ignored. The env var only matters if `paper=None` or `paper=0`, which the type hint doesn't suggest.

### 1.4 Rate Limiting

| Item | Finding | Severity |
|------|---------|----------|
| Rate limit handling | None | MEDIUM |
| API call pattern | `get_current_prices()` loops over symbols one-by-one | MEDIUM |

**Detail**: The `get_current_prices` method makes one API call per symbol. With 100 symbols in the universe, this could hit rate limits. Alpaca's rate limit is 200 requests/minute for paper accounts. No retry-with-backoff logic exists for rate limit (429) responses.

---

## 2. Order Logic

### 2.1 Order Types

| Item | Finding | Severity |
|------|---------|----------|
| Default order type | `"market"` always (hardcoded in `generate_rebalance_orders`) | HIGH |
| Limit order support | `Order` dataclass has `limit_price` field; AlpacaBroker passes it | OK |
| Limit orders used? | Never -- `generate_rebalance_orders` always sets `order_type="market"` | HIGH |

**Detail**: Market orders provide certainty of execution but no price protection. For a $1M portfolio rebalancing ~18 positions, most orders are small enough that market orders are acceptable. However, for larger allocations or less-liquid names, market orders risk significant slippage.

### 2.2 Large Order Handling

| Item | Finding | Severity |
|------|---------|----------|
| Order splitting | None | HIGH |
| TWAP/VWAP | Not implemented | HIGH |
| Max order size | No limit | CRITICAL |

**Detail**: A single order for any dollar amount can be submitted. If a bug in the optimizer produces a 100% weight in one stock, the system would attempt to buy $1M+ of a single name in one market order. No splitting, no participation rate limit.

### 2.3 Partial Fill Handling

| Item | Finding | Severity |
|------|---------|----------|
| `_wait_for_fill` checks | Only checks `"filled"` status | HIGH |
| Partial fill status | `"partially_filled"` is not handled | HIGH |
| Timeout behavior | Returns `None`, order marked as `"submitted"` | MEDIUM |

**Detail**: The `_wait_for_fill` method only returns an order object when `o.status == "filled"`. If Alpaca returns `"partially_filled"`, the method continues polling until timeout, then returns `None`. The parent order is marked `"submitted"` -- losing track of the partial fill entirely. The filled shares are in the Alpaca account but not reflected in the Order object.

### 2.4 Error Handling

| Item | Finding | Severity |
|------|---------|----------|
| Exception handling | Bare `except Exception` catches everything | MEDIUM |
| Network timeout | No retry logic | HIGH |
| Insufficient funds | Caught as generic exception, no specific handling | MEDIUM |
| Order rejection reason | Not captured (only logged as string) | MEDIUM |

**Detail**: The `submit_order` method wraps the entire API call in a single `try/except Exception`. This means network timeouts, rate limits, and business logic errors (insufficient buying power, invalid symbol) all receive the same treatment: the order is marked `"rejected"` and the exception message is logged. No retry is attempted for transient errors.

### 2.5 Trading Session Timing

| Item | Finding | Severity |
|------|---------|----------|
| Market hours check | None | MEDIUM |
| Pre/post-market | `time_in_force="day"` prevents pre/post fills | OK |
| Cron timing | `55 15 * * 1-5` (3:55 PM ET) -- 5 min before close | MEDIUM |

**Detail**: The cron job runs at 3:55 PM ET, giving only 5 minutes for the entire pipeline (signal generation, optimization, order submission, fill polling). If signal computation takes >2 minutes, some orders may not fill before close. `time_in_force="day"` is correct and prevents extended-hours fills, which is good.

---

## 3. Position Management

### 3.1 Reconciliation

| Item | Finding | Severity |
|------|---------|----------|
| Strategy vs actual comparison | Not implemented | CRITICAL |
| Drift monitoring | Not implemented | HIGH |
| Stale position detection | Not implemented | MEDIUM |

**Detail**: After rebalancing, there is no check that actual Alpaca positions match what the strategy intended. If orders are partially filled, rejected, or if external manual trades occur, the portfolio could drift significantly from target without any alert.

### 3.2 Emergency Liquidation

| Item | Finding | Severity |
|------|---------|----------|
| `close_all_positions()` | Exists, calls Alpaca API | OK |
| `cancel_all_orders()` | Exists, calls Alpaca API | OK |
| Automated trigger | None -- no circuit breaker | HIGH |

**Detail**: The methods exist but are never called automatically. The `max_drawdown_limit: 0.20` in config is only used in backtesting, not in live/paper trading. There is no mechanism to halt trading if the portfolio drops 20%.

---

## 4. Safety & Security

### 4.1 API Key Management

| Item | Finding | Severity |
|------|---------|----------|
| Keys in source code | Not found | OK |
| Keys in shell RC files | `setup_alpaca.sh` writes to `~/.zshrc`/`~/.bashrc` | HIGH |
| Keys in git | Not in repo (checked via grep) | OK |
| Key masking in logs | Keys not logged, but stored as instance attributes | MEDIUM |

### 4.2 Pre-trade Safety Limits

| Item | Finding | Severity |
|------|---------|----------|
| Max single order value | NOT IMPLEMENTED | CRITICAL |
| Max daily trade value | NOT IMPLEMENTED | CRITICAL |
| Max daily loss | NOT IMPLEMENTED | CRITICAL |
| Liquidity check (ADV) | NOT IMPLEMENTED | HIGH |
| Position concentration | Handled by optimizer (max 10%) but not enforced at order level | HIGH |

**Detail**: There are zero pre-trade safety checks. Any order generated by the optimizer is submitted directly to Alpaca. A bug in signal generation, optimization, or weight calculation could result in orders of arbitrary size.

### 4.3 Logging

| Item | Finding | Severity |
|------|---------|----------|
| Order submission logged | Yes (logger.info on fill) | OK |
| Order rejection logged | Yes (logger.error) | OK |
| Structured format | No -- plain text only | MEDIUM |
| Slippage tracking | Not implemented | MEDIUM |
| Execution quality metrics | Not implemented | MEDIUM |

---

## 5. Paper Trading Script (`paper_trade.py`)

### 5.1 State Management

| Item | Finding | Severity |
|------|---------|----------|
| State file | `logs/paper_trade_state.json` | OK |
| Crash recovery | None -- if crash occurs mid-rebalance, state is not updated | HIGH |
| Concurrent runs | No lock file protection | HIGH |
| State file corruption | No atomic write (no temp-file-then-rename) | MEDIUM |

**Detail**: If `paper_trade.py` crashes after submitting some orders but before `save_state()`, the next cron run will re-trigger a rebalance, potentially doubling positions. There is no lock file to prevent concurrent execution.

### 5.2 Rebalance Timing

| Item | Finding | Severity |
|------|---------|----------|
| Trading day calculation | `days_passed * 5 / 7` (approximate) | LOW |
| Holiday handling | Not accounted for | LOW |
| Cron schedule | Weekdays only via `1-5` | OK |

**Detail**: The `should_rebalance` function uses `days_passed * 5/7` to approximate trading days. This overestimates trading days (ignores holidays), meaning rebalances happen slightly early. For a monthly rebalance this is acceptable, but using Alpaca's calendar API (`get_calendar()`) would be more precise.

### 5.3 Error Recovery

| Item | Finding | Severity |
|------|---------|----------|
| Mid-rebalance crash | Orders already submitted are orphaned | HIGH |
| Partial execution | No rollback, no state save of partial progress | HIGH |
| Network failure | Script exits, cron re-runs next day | MEDIUM |

---

## 6. Summary of Critical and High Findings

### CRITICAL (must fix before any real-money trading)

1. **No max single order value limit** -- a bug could submit a $1M+ market order
2. **No daily cumulative trade/loss limits** -- no circuit breaker
3. **No position reconciliation** -- drift undetected
4. **No paper-mode safety gate** -- nothing prevents accidental `paper=False`

### HIGH (should fix before production)

5. API keys written to shell RC files by setup script
6. API keys stored as plain-text instance attributes
7. No partial fill handling
8. No retry logic for transient API errors
9. No order splitting for large orders
10. Market orders only -- no limit order support
11. No crash recovery / lock file for paper_trade.py
12. No automated drawdown circuit breaker

---

## 7. Fixes Implemented

See `EXECUTION_FIX_LOG.md` for detailed before/after of each fix.

| # | Fix | File(s) |
|---|-----|---------|
| 1 | Pre-trade safety checks (max order value, daily limits, ADV, concentration) | `safety.py` (new), `alpaca_broker.py` |
| 2 | Post-trade reconciliation | `safety.py`, `alpaca_broker.py`, `paper_trade.py` |
| 3 | TWAP order splitting for large orders | `safety.py`, `alpaca_broker.py` |
| 4 | Limit order support in `generate_rebalance_orders` | `broker.py` |
| 5 | Partial fill handling in `_wait_for_fill` | `alpaca_broker.py` |
| 6 | Retry logic for transient API errors | `alpaca_broker.py` |
| 7 | Structured JSON execution logging with slippage tracking | `safety.py`, `alpaca_broker.py` |
| 8 | Paper-mode safety gate (`require_paper_mode`) | `safety.py`, `alpaca_broker.py` |
| 9 | API key wiping from instance attributes | `alpaca_broker.py` |
| 10 | Lock file for concurrent execution prevention | `paper_trade.py` |
| 11 | Market hours check | `alpaca_broker.py` |
| 12 | Safety config section in `config.yaml` | `config.yaml` |
| 13 | Comprehensive test suite for safety module | `test_safety.py` |
