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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/paper_trade_{datetime.now():%Y%m%d}.log"),
    ],
)
logger = logging.getLogger(__name__)

STATE_FILE = Path("logs/paper_trade_state.json")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_rebalance": None, "trade_history": []}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def should_rebalance(state: dict, freq_days: int = 21) -> bool:
    """Check if enough trading days have passed since last rebalance."""
    last = state.get("last_rebalance")
    if last is None:
        return True
    last_date = datetime.fromisoformat(last)
    days_passed = (datetime.now() - last_date).days
    # Approximate trading days (weekdays only)
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


def run_rebalance(strategy, broker, dry_run=False):
    """Compute target portfolio and execute rebalance trades."""
    capital = broker.get_portfolio_value()
    logger.info("Computing target portfolio for $%.2f...", capital)

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

    # Generate orders
    orders = generate_rebalance_orders(
        current_positions, target_weights, capital, prices
    )

    if not orders:
        print("\n  No trades needed - portfolio already at target.")
        return

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
        return

    # Execute: sell first, then buy
    print("\n  Executing trades...")
    sells = [o for o in orders if o.side == "sell"]
    buys = [o for o in orders if o.side == "buy"]

    filled = []
    for o in sells + buys:
        result = broker.submit_order(o)
        filled.append({
            "time": datetime.now().isoformat(),
            "symbol": result.symbol,
            "side": result.side,
            "qty": result.quantity,
            "status": result.status,
            "price": result.filled_price,
            "order_id": result.order_id,
        })

    return filled


def main():
    parser = argparse.ArgumentParser(description="Paper trading with Alpaca")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show trades without executing")
    parser.add_argument("--status", action="store_true",
                        help="Show current portfolio status")
    parser.add_argument("--force", action="store_true",
                        help="Force rebalance even if not due")
    parser.add_argument("--capital", type=float, default=None,
                        help="Override capital amount")
    args = parser.parse_args()

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    # Load config and strategy
    config = load_config("config.yaml")
    strategy = MultiFactorStrategy(config)

    # Connect to Alpaca paper trading
    from quant.execution.alpaca_broker import AlpacaBroker
    broker = AlpacaBroker(paper=True)

    if args.status:
        show_status(broker)
        return

    # Check if rebalance is due
    state = load_state()
    freq = config["portfolio"]["rebalance_frequency_days"]

    if not args.force and not should_rebalance(state, freq):
        days_since = (datetime.now() - datetime.fromisoformat(state["last_rebalance"])).days
        print(f"  Not time to rebalance yet ({days_since} days since last, "
              f"target is {freq} trading days). Use --force to override.")
        show_status(broker)
        return

    # Run rebalance
    filled = run_rebalance(strategy, broker, dry_run=args.dry_run)

    if filled and not args.dry_run:
        state["last_rebalance"] = datetime.now().isoformat()
        state["trade_history"].append({
            "date": datetime.now().isoformat(),
            "trades": filled,
        })
        save_state(state)

    # Show final status
    if not args.dry_run:
        show_status(broker)


if __name__ == "__main__":
    main()
