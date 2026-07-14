"""Shared implementation for the Alpaca paper-trading entry points.

paper_trade.py (multi-factor) and paper_trade_lgbm.py (LightGBM) used to be
95% line-identical copies; every operational fix had to be applied twice and
a missed copy meant silent behavior divergence between the two live accounts.
All logic now lives here, parameterized by a TradeProfile.  The entry scripts
keep thin module-level wrappers (STATE_FILE, load_state, run_rebalance, ...)
because tests and operators patch those names.

Operational behavior implemented here (differences vs the historical copies
are deliberate fixes):

  - Market-closed days abort BEFORE submitting anything.  Submitting market
    orders on a holiday left them queued for the next open; a failed cancel
    plus the next day's fresh rebalance meant double buying.
  - Stop-losses are checked on EVERY daily run.  They used to sit behind the
    "not time to rebalance yet" early return, so they only actually ran on
    rebalance days despite the docs promising daily enforcement.
  - Entry prices are recorded only when a NEW position is established,
    matching the backtest engine's stop-loss semantics.  Adding to a position
    no longer resets its stop-loss base.
  - The account's actual current weights are passed to the strategy so the
    turnover penalty, turnover constraint, and total turnover cap all bind
    against reality (they used to be dead code in the live path).
  - The LightGBM profile persists the score cross-section in the state file,
    activating the score-level turnover penalty across runs.
  - The daily safety counters (total traded value, realised PnL) are
    persisted in the state file, so a manual re-run on the same day resumes
    the cumulative limits instead of resetting them.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from quant.execution.broker import Order, generate_rebalance_orders
from quant.execution.safety import ExecutionLogger, PreTradeCheck, SafetyConfig

logger = logging.getLogger(__name__)


@dataclass
class TradeProfile:
    """Everything that differs between the two paper-trading entry points."""
    name: str
    description: str
    status_banner: str
    portfolio_banner: str
    state_file: Path
    lock_file: Path
    log_prefix: str
    strategy_factory: Callable[[dict], object]
    api_key_env: str = "ALPACA_API_KEY"
    secret_key_env: str = "ALPACA_SECRET_KEY"
    persist_scores: bool = False


def setup_logging(log_prefix: str):
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"logs/{log_prefix}_{datetime.now():%Y%m%d}.log"),
        ],
    )


def load_state(state_file: Path) -> dict:
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            # Starting from an empty state would erase idempotency, entry-price
            # and daily-limit history.  Corruption must stop the workflow.
            raise RuntimeError(f"Trade state is unreadable: {state_file}") from exc
        state.setdefault("last_rebalance", None)
        state.setdefault("trade_history", [])
        return state
    return {"schema_version": 2, "last_rebalance": None, "trade_history": []}


def save_state(state_file: Path, state: dict):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state["schema_version"] = 2
    tmp = state_file.with_suffix(state_file.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str) + "\n")
    os.replace(tmp, state_file)


def acquire_lock(lock_file: Path) -> bool:
    """Simple file-based lock to prevent concurrent rebalances.

    If a previous run crashed, a stale lock file may remain. We consider
    locks older than 1 hour to be stale and remove them.
    """
    if lock_file.exists():
        age = datetime.now().timestamp() - lock_file.stat().st_mtime
        if age > 3600:
            logger.warning("Removing stale lock file (age: %.0f seconds)", age)
            lock_file.unlink()
        else:
            return False
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False

    with os.fdopen(fd, "w") as f:
        json.dump({"pid": os.getpid(), "started": datetime.now().isoformat()}, f)
    return True


def release_lock(lock_file: Path):
    if lock_file.exists():
        lock_file.unlink()


def should_rebalance(state: dict, freq_days: int = 21) -> bool:
    """Check if enough time has passed since last rebalance.

    Uses a conservative calendar-day approximation: multiply calendar days
    by 5/7 to estimate trading days. This slightly overestimates trading days
    (ignores holidays), which means we rebalance slightly early rather than
    late -- a safer default.
    """
    last = state.get("last_rebalance")
    if last is None:
        return True
    last_date = datetime.fromisoformat(last)
    days_passed = (datetime.now() - last_date).days
    trading_days = days_passed * 5 / 7
    return trading_days >= freq_days


def show_status(broker, banner: str = "PAPER TRADING STATUS"):
    """Print current portfolio status."""
    print(f"\n{'='*60}")
    print(f"  {banner}")
    print(f"{'='*60}")
    print(f"  Portfolio Value:  ${broker.get_portfolio_value():>12,.2f}")
    print(f"  Cash:             ${broker.get_cash():>12,.2f}")
    print("  Positions:")

    positions = broker.get_positions()
    if positions.empty:
        print("    (no positions)")
    else:
        prices = broker.get_current_prices(positions.index.tolist())
        for sym, shares in positions.items():
            price = prices.get(sym, 0)
            value = shares * price
            print(f"    {sym:<8} {shares:>6.0f} shares  @ ${price:>8.2f}  = ${value:>10,.2f}")

    print(f"{'='*60}\n")


def current_broker_weights(broker) -> Optional[pd.Series]:
    """The account's actual portfolio weights (shares * price / equity).

    Used as prev_weights for the optimizer's turnover machinery.  Returns
    None when they cannot be derived, so the strategy falls back to its
    no-previous-portfolio behavior instead of binding against garbage.
    """
    try:
        positions = broker.get_positions()
        if positions.empty:
            return pd.Series(dtype=float)
        equity = broker.get_portfolio_value()
        if not equity or equity <= 0:
            return None
        prices = broker.get_current_prices(positions.index.tolist())
        values = pd.Series(
            {sym: shares * prices.get(sym, 0.0) for sym, shares in positions.items()},
            dtype=float,
        )
        return values / equity
    except Exception as e:
        logger.warning("Could not derive current weights from broker: %s", e)
        return None


def check_stop_losses(
    broker,
    optimizer,
    state,
    dry_run=False,
    persist_callback: Optional[Callable[[dict], None]] = None,
    retry_cooldown_seconds: int = 300,
):
    """Check stop-losses against entry prices stored in state.

    Returns list of symbols that were stopped out (and sold if not dry_run).
    """
    entry_prices = state.get("entry_prices", {})
    if not entry_prices:
        return []

    positions = broker.get_positions()
    if positions.empty:
        return []

    held_symbols = [s for s in positions.index if s in entry_prices and positions[s] > 0]
    if not held_symbols:
        return []

    current_prices = broker.get_current_prices(held_symbols)
    stopped = []
    attempts = state.setdefault("stop_loss_attempts", {})
    now = datetime.now()
    for sym in held_symbols:
        entry = entry_prices[sym]
        current = current_prices.get(sym, 0)
        if current > 0 and entry > 0:
            pnl_pct = (current / entry) - 1.0
            if pnl_pct < -optimizer.stop_loss_pct:
                previous = attempts.get(sym, {})
                try:
                    previous_at = datetime.fromisoformat(previous.get("time", ""))
                except (TypeError, ValueError):
                    previous_at = None
                uncertain = previous.get("status") in {
                    "unknown", "submitted", "partial_fill_open"
                }
                if (uncertain and previous_at is not None
                        and (now - previous_at).total_seconds() < retry_cooldown_seconds):
                    logger.error(
                        "Stop-loss for %s is in cooldown because the prior order "
                        "may still be open (status=%s)",
                        sym,
                        previous.get("status"),
                    )
                    continue
                stopped.append(sym)
                logger.warning("Stop-loss triggered for %s: %.1f%% loss (entry=$%.2f, now=$%.2f)",
                               sym, pnl_pct * 100, entry, current)

    if not stopped:
        return []

    if dry_run:
        logger.info("DRY RUN: would sell stopped positions: %s", stopped)
        return stopped

    for sym in stopped:
        shares = positions[sym]
        order = Order(
            symbol=sym,
            side="sell",
            quantity=shares,
            order_type="market",
            purpose="stop_loss",
            signal_price=current_prices.get(sym),
        )
        result = broker.submit_order(order)
        logger.info("Stop-loss sell %s: %s (qty=%d)", sym, result.status, shares)
        attempts[sym] = {
            "time": datetime.now().isoformat(),
            "status": result.status,
            "requested_qty": float(shares),
            "filled_qty": float(getattr(result, "filled_quantity", 0) or 0),
            "order_id": getattr(result, "order_id", ""),
            "client_order_id": getattr(result, "client_order_id", ""),
        }

        # Clear the stop base only after the broker confirms a complete exit.
        # A rejected, cancelled, unknown or partial order keeps both the entry
        # price and retry metadata for the next run.
        fully_exited = result.status == "filled"
        if result.status in {"partial_fill", "partial_fill_open"}:
            fully_exited = False
        if fully_exited:
            try:
                after = broker.get_positions()
                fully_exited = sym not in after.index or float(after.get(sym, 0)) <= 0
            except Exception:
                # A fully-filled sell for the exact pre-order position quantity
                # is sufficient if the follow-up query is unavailable.
                fully_exited = True
        if fully_exited:
            entry_prices.pop(sym, None)
            attempts.pop(sym, None)

        state["entry_prices"] = entry_prices
        state["stop_loss_attempts"] = attempts
        if persist_callback is not None:
            persist_callback(state)

    state["entry_prices"] = entry_prices
    state["stop_loss_attempts"] = attempts

    return stopped


def update_entry_prices(state: dict, filled: list, broker, fallback_prices: dict):
    """Record stop-loss entry prices after a rebalance.

    Matches the backtest engine's semantics: an entry price is recorded only
    when a position goes from zero to positive.  Adding to an existing
    position must NOT reset its stop-loss base (the historical behavior
    overwrote the entry on every buy fill, silently moving the stop).
    Fully-sold positions have their entry price cleared.
    """
    entry_prices = state.get("entry_prices", {})
    positions_after = None

    for trade in filled:
        sym = trade["symbol"]
        if trade["status"] not in ("filled", "partial_fill", "partial_fill_open"):
            continue
        if trade["side"] == "buy":
            if sym not in entry_prices:
                entry_prices[sym] = trade["price"] or fallback_prices.get(sym, 0)
        elif trade["side"] == "sell":
            if positions_after is None:
                positions_after = broker.get_positions()
            if sym not in positions_after.index or positions_after[sym] == 0:
                entry_prices.pop(sym, None)

    state["entry_prices"] = entry_prices


def run_rebalance(strategy, broker, config, dry_run=False,
                  banner: str = "TARGET PORTFOLIO",
                  exec_logger_cls=ExecutionLogger,
                  prev_scores: Optional[pd.Series] = None,
                  order_result_callback: Optional[Callable[[dict], None]] = None):
    """Compute target portfolio and execute rebalance trades.

    Returns (filled, target_weights) where filled is:
      - None if nothing was attempted (dry run, market closed, or daily
        loss kill-switch)
      - [] if the portfolio is already at target (counts as a completed
        rebalance)
      - a list of trade records otherwise
    """
    exec_log = exec_logger_cls()
    capital = broker.get_portfolio_value()
    logger.info("Computing target portfolio for $%.2f...", capital)

    # Market-closed gate: market orders submitted on a holiday queue for the
    # next open; a failed cancel plus the next day's rerun means double
    # buying. Abort outright — the run is retried on the next weekday.
    if not dry_run and hasattr(broker, "is_market_open"):
        if not broker.is_market_open():
            logger.error(
                "Market is closed (holiday or off-hours) — aborting rebalance "
                "instead of queueing orders for the next open."
            )
            return None, None

    # Daily loss kill-switch: if today's loss already exceeds the configured
    # limit, do not trade into a falling market — skip the rebalance entirely.
    if hasattr(broker, "get_daily_pnl"):
        daily_pnl = broker.get_daily_pnl()
        if daily_pnl is not None:
            checker = PreTradeCheck(SafetyConfig.from_config(config))
            ok, reason = checker.check_daily_loss_limit(unrealised_pnl=daily_pnl)
            if not ok:
                logger.error("Rebalance blocked by daily loss limit: %s", reason)
                if not dry_run:
                    return None, None
                print(f"\n  ** DAILY LOSS LIMIT BREACHED ({reason}) — "
                      f"live run would abort here **")

    # The account's real current weights drive the turnover penalty,
    # constraint, and total turnover cap inside the strategy.
    prev_weights = current_broker_weights(broker)

    # Get optimized portfolio
    portfolio_kwargs = {"capital": capital, "prev_weights": prev_weights}
    if prev_scores is not None:
        portfolio_kwargs["prev_scores"] = prev_scores
    portfolio = strategy.get_current_portfolio(**portfolio_kwargs)
    target_weights = portfolio["weight"]

    print(f"\n{'='*65}")
    print(f"  {banner}  |  Capital: ${capital:,.0f}")
    print(f"{'='*65}")
    print(f"  {'Stock':<8} {'Weight':>8} {'Dollars':>10} {'Shares':>8} {'Price':>10}")
    print(f"  {'-'*55}")
    for sym, row in portfolio.iterrows():
        print(f"  {sym:<8} {row['weight_pct']:>7.1f}% ${row['dollars']:>9,.0f} "
              f"{row['shares']:>7.0f}  ${row['price']:>8.2f}")
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<8} {portfolio['weight_pct'].sum():>7.1f}% "
          f"${portfolio['dollars'].sum():>9,.0f}")
    print(f"{'='*65}")

    # Get current positions and prices
    current_positions = broker.get_positions()
    all_symbols = list(set(target_weights.index) | set(current_positions.index))
    prices = broker.get_current_prices(all_symbols)
    missing_prices = [
        sym
        for sym in all_symbols
        if prices.get(sym) is None
        or not pd.notna(prices.get(sym))
        or float(prices.get(sym)) <= 0
    ]
    if missing_prices:
        logger.error(
            "Rebalance aborted: broker quotes unavailable for %s; refusing "
            "to execute an incomplete target",
            sorted(missing_prices),
        )
        return None, None

    # Generate orders (sells first by default via sort in generate_rebalance_orders)
    orders = generate_rebalance_orders(
        current_positions, target_weights, capital, prices
    )

    if not orders:
        print("\n  No trades needed - portfolio already at target.")
        return [], target_weights

    # Show planned trades
    print(f"\n  TRADES TO EXECUTE ({len(orders)} orders):")
    print(f"  {'-'*50}")
    total_trade_value = 0
    for o in orders:
        price = prices.get(o.symbol, 0)
        value = o.quantity * price
        total_trade_value += value
        direction = "BUY " if o.side == "buy" else "SELL"
        print(f"  {direction} {o.symbol:<8} {o.quantity:>6.0f} shares "
              f"(~${value:>9,.0f})")
    print(f"  {'-'*50}")
    print(f"  Total trade value: ${total_trade_value:,.0f}")
    est_cost = total_trade_value * 15 / 10000  # ~15 bps total
    print(f"  Est. cost+slippage: ${est_cost:,.0f}")

    if dry_run:
        print("\n  ** DRY RUN - no orders submitted **\n")
        return None, target_weights

    # Fetch average daily volume so the broker's liquidity check (max ADV
    # fraction) and TWAP splitting can engage. Missing ADV data must not
    # block trading — those orders are simply submitted without the check.
    adv_map = strategy.data.fetch_adv([o.symbol for o in orders])
    if adv_map:
        logger.info("Fetched ADV for %d/%d order symbols", len(adv_map), len(orders))
    else:
        logger.warning("No ADV data available — liquidity checks skipped")

    # Execute trades
    exec_log.log_rebalance_start(capital, len(orders))
    print("\n  Executing trades...")

    filled = []
    n_filled = 0
    n_rejected = 0
    total_value_traded = 0.0

    for o in orders:
        result = broker.submit_order(o, avg_daily_volume=adv_map.get(o.symbol))
        record = {
            "time": datetime.now().isoformat(),
            "symbol": result.symbol,
            "side": result.side,
            "qty": float(getattr(result, "filled_quantity", 0) or 0),
            "requested_qty": float(
                getattr(result, "requested_quantity", None) or o.quantity
            ),
            "status": result.status,
            "price": result.filled_price,
            "signal_price": result.signal_price,
            "order_id": result.order_id,
            "client_order_id": getattr(result, "client_order_id", ""),
            "reject_reason": result.reject_reason,
        }
        filled.append(record)

        if order_result_callback is not None:
            order_result_callback(record)

        if result.status in ("filled", "partial_fill"):
            n_filled += 1
            actual_qty = float(getattr(result, "filled_quantity", 0) or result.quantity)
            total_value_traded += actual_qty * (result.filled_price or 0)
        elif result.status == "rejected":
            n_rejected += 1

    exec_log.log_rebalance_complete(n_filled, n_rejected, total_value_traded)

    print(f"\n  Results: {n_filled} filled, {n_rejected} rejected, "
          f"${total_value_traded:,.0f} traded")

    return filled, target_weights


def _persist_daily_tracker(profile: TradeProfile, broker, state: dict):
    """Write today's cumulative safety counters into the state file.

    Only when something actually happened — otherwise every quiet daily run
    would dirty the state file and produce a noise commit from the workflow.
    """
    daily = getattr(getattr(broker, "safety", None), "daily", None)
    if daily is None:
        return
    if not (daily.orders_submitted or daily.orders_filled
            or daily.orders_rejected or daily.total_value_traded):
        return
    state["daily_tracker"] = daily.to_dict()
    save_state(profile.state_file, state)


def run_main(profile: TradeProfile):
    """Full CLI entry point, shared by both paper-trading scripts."""
    parser = argparse.ArgumentParser(description=profile.description)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show trades without executing")
    parser.add_argument("--status", action="store_true",
                        help="Show current portfolio status")
    parser.add_argument("--force", action="store_true",
                        help="Force rebalance even if not due")
    parser.add_argument("--reconcile", action="store_true",
                        help="Run post-trade reconciliation only")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    setup_logging(profile.log_prefix)

    from quant.utils.config import load_config
    config = load_config("config.yaml")
    strategy = profile.strategy_factory(config)
    safety_config = SafetyConfig.from_config(config)

    from quant.execution.alpaca_broker import AlpacaBroker
    broker = AlpacaBroker(
        api_key=os.environ.get(profile.api_key_env) or None,
        secret_key=os.environ.get(profile.secret_key_env) or None,
        paper=True,
        safety_config=safety_config,
    )

    if args.status:
        show_status(broker, profile.status_banner)
        return

    if args.reconcile:
        state = load_state(profile.state_file)
        last_trades = state.get("trade_history", [])
        if not last_trades:
            print("  No previous rebalance found. Run a rebalance first.")
            return
        capital = broker.get_portfolio_value()
        portfolio = strategy.get_current_portfolio(
            capital=capital, prev_weights=current_broker_weights(broker)
        )
        target_weights = portfolio["weight"]
        drift_df = broker.reconcile(target_weights)
        print("\n  Reconciliation results:")
        print(drift_df.to_string())
        return

    # Acquire lock to prevent concurrent rebalances
    if not args.dry_run:
        if not acquire_lock(profile.lock_file):
            logger.error(
                "Another rebalance is in progress (lock file exists). Exiting."
            )
            sys.exit(1)

    try:
        state = load_state(profile.state_file)
        freq = config["portfolio"]["rebalance_frequency_days"]

        # Resume today's cumulative safety counters on same-day re-runs
        if not args.dry_run and state.get("daily_tracker"):
            if broker.safety.daily.restore(state["daily_tracker"]):
                logger.info(
                    "Restored daily safety counters: $%.0f already traded today",
                    broker.safety.daily.total_value_traded,
                )

        def persist_runtime_state(current: dict):
            """Atomically checkpoint order state and today's safety budget."""
            daily = getattr(getattr(broker, "safety", None), "daily", None)
            if daily is not None:
                current["daily_tracker"] = daily.to_dict()
            save_state(profile.state_file, current)

        # A known but unreconciled split can make order quantities wrong by the
        # split ratio.  Status-only runs remain available, but every trading
        # path fails closed before stop-loss or rebalance orders are built.
        if (not args.dry_run
                and hasattr(broker, "assert_corporate_actions_reconciled")):
            broker.assert_corporate_actions_reconciled()

        # Market-closed gate for the whole live run: neither stop-loss sells
        # nor rebalance orders should queue overnight on a holiday.
        if not args.dry_run and hasattr(broker, "is_market_open"):
            if not broker.is_market_open():
                logger.warning(
                    "Market is closed today — skipping stop-loss check and "
                    "rebalance. The next weekday run will retry."
                )
                show_status(broker, profile.status_banner)
                return

        # Stop-losses run on EVERY daily run, not just rebalance days
        stopped = check_stop_losses(
            broker,
            strategy.optimizer,
            state,
            dry_run=args.dry_run,
            persist_callback=(
                persist_runtime_state
                if not args.dry_run else None
            ),
        )
        if stopped:
            logger.info("Stopped out %d position(s): %s", len(stopped), stopped)
            if not args.dry_run:
                save_state(profile.state_file, state)

        # Check if rebalance is due
        if not args.force and not should_rebalance(state, freq):
            days_since = (
                datetime.now()
                - datetime.fromisoformat(state["last_rebalance"])
            ).days
            print(
                f"  Not time to rebalance yet ({days_since} days since last, "
                f"target is {freq} trading days). Use --force to override."
            )
            if not args.dry_run:
                _persist_daily_tracker(profile, broker, state)
            show_status(broker, profile.status_banner)
            return

        # Previous scores (LightGBM): activates the score-level turnover
        # penalty that the backtest applies between rebalances.
        prev_scores = None
        if profile.persist_scores and state.get("prev_scores"):
            prev_scores = pd.Series(state["prev_scores"], dtype=float)

        # Run rebalance
        pending_records = list(
            state.get("pending_rebalance", {}).get("trades", [])
        )

        def persist_order_result(record: dict):
            pending_records.append(record)
            if (
                record.get("status") in {"filled", "partial_fill", "partial_fill_open"}
                and float(record.get("qty", 0) or 0) > 0
            ):
                update_entry_prices(
                    state,
                    [record],
                    broker,
                    {record["symbol"]: record.get("price", 0)},
                )
            state["pending_rebalance"] = {
                "started_at": state.get("pending_rebalance", {}).get(
                    "started_at", datetime.now().isoformat()
                ),
                "status": "in_progress",
                "trades": pending_records,
            }
            persist_runtime_state(state)

        filled, target_weights = run_rebalance(
            strategy, broker, config, dry_run=args.dry_run,
            banner=profile.portfolio_banner,
            prev_scores=prev_scores,
            order_result_callback=(
                persist_order_result if not args.dry_run else None
            ),
        )

        rebalance_completed = False

        if filled is not None and not args.dry_run and not filled:
            # Portfolio already at target: counts as a completed rebalance,
            # otherwise the strategy re-runs (and re-fetches data) every day.
            state["last_rebalance"] = datetime.now().isoformat()
            rebalance_completed = True
            state.pop("pending_rebalance", None)
            if pending_records:
                state["trade_history"].append({
                    "date": datetime.now().isoformat(),
                    "completed": True,
                    "trades": pending_records,
                })
            save_state(profile.state_file, state)

        if filled and not args.dry_run:
            any_executed = any(
                t["status"] in ("filled", "partial_fill", "partial_fill_open")
                and float(t.get("qty", 0) or 0) > 0
                for t in filled
            )
            all_completed = all(t["status"] == "filled" for t in filled)

            if any_executed:
                prices = broker.get_current_prices(target_weights.index.tolist())
                update_entry_prices(state, filled, broker, prices)

            if all_completed:
                state["last_rebalance"] = datetime.now().isoformat()
                rebalance_completed = True
                state.pop("pending_rebalance", None)
            else:
                logger.warning(
                    "Rebalance incomplete — last_rebalance not updated; "
                    "the next run will recompute and repair the remaining drift"
                )
                state["pending_rebalance"] = {
                    "started_at": state.get("pending_rebalance", {}).get(
                        "started_at", datetime.now().isoformat()
                    ),
                    "updated_at": datetime.now().isoformat(),
                    "status": "incomplete",
                    "trades": pending_records,
                }

            # Always record history for audit trail
            state["trade_history"].append({
                "date": datetime.now().isoformat(),
                "completed": all_completed,
                "trades": pending_records,
            })
            save_state(profile.state_file, state)

            # Post-trade reconciliation
            logger.info("Running post-trade reconciliation...")
            drift_df = broker.reconcile(target_weights)

        # Persist the score cross-section for the next run's turnover penalty
        if (rebalance_completed and profile.persist_scores
                and getattr(strategy, "last_scores_", None) is not None):
            state["prev_scores"] = {
                sym: round(float(v), 6)
                for sym, v in strategy.last_scores_.items()
                if pd.notna(v)
            }
            save_state(profile.state_file, state)

        if not args.dry_run:
            _persist_daily_tracker(profile, broker, state)
            show_status(broker, profile.status_banner)

    finally:
        if not args.dry_run:
            release_lock(profile.lock_file)
