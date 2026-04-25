#!/usr/bin/env python3
"""Paper trading script using Alpaca API.

Usage:
    # One-time setup:
    pip install alpaca-trade-api

    export ALPACA_API_KEY="your-paper-key"
    export ALPACA_SECRET_KEY="your-paper-secret"

    # Run once to see what trades would be made:
    python paper_trade.py --dry-run

    # Execute the rebalance:
    python paper_trade.py

    # Check current status:
    python paper_trade.py --status

    # Run post-trade reconciliation only:
    python paper_trade.py --reconcile

    # Automate with cron (run at 3:55 PM ET every 21 trading days):
    # 55 15 * * 1-5 cd /path/to/trade && python paper_trade.py >> logs/trade.log 2>&1
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from quant.utils.config import load_config
from quant.strategy import MultiFactorStrategy
from quant.execution.broker import generate_rebalance_orders
from quant.execution.safety import SafetyConfig, ExecutionLogger

logger = logging.getLogger(__name__)

STATE_FILE = Path("logs/paper_trade_state.json")
LOCK_FILE = Path("logs/paper_trade.lock")


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"logs/paper_trade_{datetime.now():%Y%m%d}.log"),
        ],
    )


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_rebalance": None, "trade_history": []}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def acquire_lock() -> bool:
    """Simple file-based lock to prevent concurrent rebalances.

    If a previous run crashed, a stale lock file may remain. We consider
    locks older than 1 hour to be stale and remove them.
    """
    if LOCK_FILE.exists():
        # Check for stale lock (older than 1 hour)
        age = datetime.now().timestamp() - LOCK_FILE.stat().st_mtime
        if age > 3600:
            logger.warning("Removing stale lock file (age: %.0f seconds)", age)
            LOCK_FILE.unlink()
        else:
            return False
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False

    with os.fdopen(fd, "w") as lock_file:
        json.dump({
            "pid": os.getpid(),
            "started": datetime.now().isoformat(),
        }, lock_file)
    return True


def release_lock():
    if LOCK_FILE.exists():
        LOCK_FILE.unlink()


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
    # Approximate trading days (weekdays only, ignores holidays)
    trading_days = days_passed * 5 / 7
    return trading_days >= freq_days


def show_status(broker):
    """Print current portfolio status."""
    print(f"\n{'='*60}")
    print(f"  PAPER TRADING STATUS")
    print(f"{'='*60}")
    print(f"  Portfolio Value:  ${broker.get_portfolio_value():>12,.2f}")
    print(f"  Cash:             ${broker.get_cash():>12,.2f}")
    print(f"  Positions:")

    positions = broker.get_positions()
    if positions.empty:
        print(f"    (no positions)")
    else:
        prices = broker.get_current_prices(positions.index.tolist())
        for sym, shares in positions.items():
            price = prices.get(sym, 0)
            value = shares * price
            print(f"    {sym:<8} {shares:>6.0f} shares  @ ${price:>8.2f}  = ${value:>10,.2f}")

    print(f"{'='*60}\n")


def check_stop_losses(broker, optimizer, state, dry_run=False):
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
    for sym in held_symbols:
        entry = entry_prices[sym]
        current = current_prices.get(sym, 0)
        if current > 0 and entry > 0:
            pnl_pct = (current / entry) - 1.0
            if pnl_pct < -optimizer.stop_loss_pct:
                stopped.append(sym)
                logger.warning("Stop-loss triggered for %s: %.1f%% loss (entry=$%.2f, now=$%.2f)",
                               sym, pnl_pct * 100, entry, current)

    if not stopped:
        return []

    if dry_run:
        logger.info("DRY RUN: would sell stopped positions: %s", stopped)
        return stopped

    # Sell stopped positions
    from quant.execution.broker import Order
    for sym in stopped:
        shares = positions[sym]
        order = Order(symbol=sym, side="sell", quantity=shares, order_type="market")
        result = broker.submit_order(order)
        logger.info("Stop-loss sell %s: %s (qty=%d)", sym, result.status, shares)

    # Clear entry prices for stopped symbols
    for sym in stopped:
        entry_prices.pop(sym, None)
    state["entry_prices"] = entry_prices

    return stopped


def run_rebalance(strategy, broker, config, dry_run=False):
    """Compute target portfolio and execute rebalance trades."""
    exec_log = ExecutionLogger()
    capital = broker.get_portfolio_value()
    logger.info("Computing target portfolio for $%.2f...", capital)

    # Check market hours (skip if dry run)
    if not dry_run and hasattr(broker, 'is_market_open'):
        if not broker.is_market_open():
            logger.warning("Market is closed. Orders will queue for next open.")

    # Get optimized portfolio
    portfolio = strategy.get_current_portfolio(capital=capital)
    target_weights = portfolio["weight"]

    print(f"\n{'='*65}")
    print(f"  TARGET PORTFOLIO  |  Capital: ${capital:,.0f}")
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

    # Generate orders (sells first by default via sort in generate_rebalance_orders)
    orders = generate_rebalance_orders(
        current_positions, target_weights, capital, prices
    )

    if not orders:
        print("\n  No trades needed - portfolio already at target.")
        return None, target_weights

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

    # Execute trades
    exec_log.log_rebalance_start(capital, len(orders))
    print("\n  Executing trades...")

    filled = []
    n_filled = 0
    n_rejected = 0
    total_value_traded = 0.0

    for o in orders:
        result = broker.submit_order(o)
        record = {
            "time": datetime.now().isoformat(),
            "symbol": result.symbol,
            "side": result.side,
            "qty": result.quantity,
            "status": result.status,
            "price": result.filled_price,
            "signal_price": result.signal_price,
            "order_id": result.order_id,
            "reject_reason": result.reject_reason,
        }
        filled.append(record)

        if result.status in ("filled", "partial_fill"):
            n_filled += 1
            total_value_traded += result.quantity * (result.filled_price or 0)
        elif result.status == "rejected":
            n_rejected += 1

    exec_log.log_rebalance_complete(n_filled, n_rejected, total_value_traded)

    print(f"\n  Results: {n_filled} filled, {n_rejected} rejected, "
          f"${total_value_traded:,.0f} traded")

    return filled, target_weights


def main():
    parser = argparse.ArgumentParser(description="Paper trading with Alpaca")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show trades without executing")
    parser.add_argument("--status", action="store_true",
                        help="Show current portfolio status")
    parser.add_argument("--force", action="store_true",
                        help="Force rebalance even if not due")
    parser.add_argument("--reconcile", action="store_true",
                        help="Run post-trade reconciliation only")
    parser.add_argument("--capital", type=float, default=None,
                        help="Override capital amount")
    args = parser.parse_args()

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    setup_logging()

    # Load config and strategy
    config = load_config("config.yaml")
    strategy = MultiFactorStrategy(config)

    # Build safety config from project config
    safety_config = SafetyConfig.from_config(config)

    # Connect to Alpaca paper trading
    from quant.execution.alpaca_broker import AlpacaBroker
    broker = AlpacaBroker(paper=True, safety_config=safety_config)

    if args.status:
        show_status(broker)
        return

    if args.reconcile:
        # Load last target weights from state
        state = load_state()
        last_trades = state.get("trade_history", [])
        if not last_trades:
            print("  No previous rebalance found. Run a rebalance first.")
            return
        # Re-compute current target for reconciliation
        capital = broker.get_portfolio_value()
        portfolio = strategy.get_current_portfolio(capital=capital)
        target_weights = portfolio["weight"]
        drift_df = broker.reconcile(target_weights)
        print(f"\n  Reconciliation results:")
        print(drift_df.to_string())
        return

    # Acquire lock to prevent concurrent rebalances
    if not args.dry_run:
        if not acquire_lock():
            logger.error(
                "Another rebalance is in progress (lock file exists). Exiting."
            )
            sys.exit(1)

    try:
        # Check if rebalance is due
        state = load_state()
        freq = config["portfolio"]["rebalance_frequency_days"]

        if not args.force and not should_rebalance(state, freq):
            days_since = (
                datetime.now()
                - datetime.fromisoformat(state["last_rebalance"])
            ).days
            print(
                f"  Not time to rebalance yet ({days_since} days since last, "
                f"target is {freq} trading days). Use --force to override."
            )
            show_status(broker)
            return

        # Check stop-losses before rebalancing
        stopped = check_stop_losses(
            broker, strategy.optimizer, state, dry_run=args.dry_run
        )
        if stopped:
            logger.info("Stopped out %d position(s): %s", len(stopped), stopped)
            save_state(state)

        # Run rebalance
        filled, target_weights = run_rebalance(
            strategy, broker, config, dry_run=args.dry_run
        )

        if filled and not args.dry_run:
            any_executed = any(
                t["status"] in ("filled", "partial_fill") for t in filled
            )

            if any_executed:
                # Update entry prices for new/changed positions
                entry_prices = state.get("entry_prices", {})
                prices = broker.get_current_prices(target_weights.index.tolist())
                for trade in filled:
                    sym = trade["symbol"]
                    if trade["side"] == "buy" and trade["status"] in ("filled", "partial_fill"):
                        entry_prices[sym] = trade["price"] or prices.get(sym, 0)
                    elif trade["side"] == "sell" and trade["status"] in ("filled", "partial_fill"):
                        # If fully sold, remove entry price
                        new_positions = broker.get_positions()
                        if sym not in new_positions.index or new_positions[sym] == 0:
                            entry_prices.pop(sym, None)
                state["entry_prices"] = entry_prices
                state["last_rebalance"] = datetime.now().isoformat()
            else:
                logger.warning("No orders filled — last_rebalance not updated")

            # Always record history for audit trail
            state["trade_history"].append({
                "date": datetime.now().isoformat(),
                "trades": filled,
            })
            save_state(state)

            # Post-trade reconciliation
            logger.info("Running post-trade reconciliation...")
            drift_df = broker.reconcile(target_weights)

        # Show final status
        if not args.dry_run:
            show_status(broker)

    finally:
        if not args.dry_run:
            release_lock()


if __name__ == "__main__":
    main()
