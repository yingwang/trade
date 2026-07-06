"""Shared helpers for the dashboard generation scripts.

generate_site.py and generate_site_lgbm.py used to carry line-identical
copies of the Alpaca trade-history fetch and the stock-split correction.
The split correction in particular is easy to get wrong and was duplicated;
it now lives here once, with guards that the original copies lacked.
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


# Stock splits not yet reflected in Alpaca paper trading (paper accounts do
# not process corporate actions). Map symbol -> ratio and split date.
STOCK_SPLITS = {
    "BKNG": {"ratio": 25, "date": "2026-04-06"},  # 1:25 split
}


def _looks_presplit(avg_entry_price: float, current_price: float, ratio: float) -> bool:
    """Heuristic: is this position still carried in PRE-split units?

    A position established before the split has an entry price roughly
    ``ratio`` times the post-split market price; one (re)established after
    the split is on the same scale as the market. The ratio/3 threshold
    tolerates large price moves either way without flipping the verdict.
    Without this guard, a re-bought position would be multiplied by the
    split ratio again, silently inflating dashboard equity.
    """
    if avg_entry_price <= 0 or current_price <= 0:
        return False
    return avg_entry_price / current_price > ratio / 3.0


def adjust_position_for_split(symbol, qty, avg_entry_price, cost_basis, current_price):
    """Convert a position Alpaca still carries in pre-split units.

    Applies ONLY when the entry price is on the pre-split scale (see
    _looks_presplit). Positions bought after the split are already correct
    and must not be touched.
    """
    split = STOCK_SPLITS.get(symbol)
    if split and _looks_presplit(avg_entry_price, current_price, split["ratio"]):
        ratio = split["ratio"]
        return qty * ratio, avg_entry_price / ratio, cost_basis
    return qty, avg_entry_price, cost_basis


def _split_sold_credit(filled_orders) -> float:
    """Cash that post-split sells of PRE-split shares failed to book.

    Alpaca books such a sell as qty x price with the pre-split qty, missing
    (ratio-1) x qty x price of proceeds. But a sell of shares RE-BOUGHT
    after the split is booked correctly and must not be credited — so sells
    first consume the quantity visibly bought after the split date (correct
    bookings), and only the unexplained remainder is treated as pre-split
    shares. Order history is finite (last ~200 orders); if a post-split buy
    has rolled out of the window its sell would be over-credited, which we
    accept and log.
    """
    credit = 0.0
    for sym, split in STOCK_SPLITS.items():
        ratio = split["ratio"]
        split_date = split["date"]

        events = []
        for o in filled_orders:
            if o.symbol != sym:
                continue
            fill_dt = str(o.filled_at)[:10] if o.filled_at else str(o.submitted_at)[:10]
            if fill_dt < split_date:
                continue
            qty = float(o.filled_qty)
            price = float(o.filled_avg_price) if o.filled_avg_price else 0.0
            events.append((fill_dt, o.side, qty, price))

        events.sort(key=lambda e: e[0])
        post_split_bought = 0.0
        for fill_dt, side, qty, price in events:
            if side == "buy":
                post_split_bought += qty
            elif side == "sell":
                covered = min(qty, post_split_bought)
                post_split_bought -= covered
                presplit_qty = qty - covered
                if presplit_qty > 0:
                    credit += (ratio - 1) * presplit_qty * price
                    logger.info(
                        "Split sold-credit: %s sell of %.0f pre-split share(s) "
                        "on %s -> +$%.2f", sym, presplit_qty, fill_dt,
                        (ratio - 1) * presplit_qty * price,
                    )
    return credit


def fetch_trade_history(api_key_env: str, secret_key_env: str, state_file: str) -> dict:
    """Trade history + positions + equity history for one Alpaca account.

    Falls back to local log files if API keys are not available.
    """
    api_key = os.environ.get(api_key_env, "")
    secret_key = os.environ.get(secret_key_env, "")

    if api_key and secret_key:
        return _fetch_trades_from_alpaca(api_key, secret_key, state_file)

    logger.warning("No Alpaca keys in %s/%s, falling back to local logs",
                   api_key_env, secret_key_env)
    return parse_local_trade_logs(state_file)


def _fetch_trades_from_alpaca(api_key, secret_key, state_file) -> dict:
    """Pull trade history and current positions from the Alpaca API."""
    try:
        import alpaca_trade_api as tradeapi
    except ImportError:
        logger.warning("alpaca-trade-api not installed, falling back to local logs")
        return parse_local_trade_logs(state_file)

    try:
        api = tradeapi.REST(api_key, secret_key,
                            "https://paper-api.alpaca.markets", api_version="v2")

        # Current account info
        account = api.get_account()
        account_info = {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
        }

        # Portfolio equity history (daily) for actual P&L tracking
        portfolio_history = []
        try:
            ph = api.get_portfolio_history(period="all", timeframe="1D")
            if ph and hasattr(ph, 'equity') and ph.equity:
                import datetime as dt
                for ts, eq, pl in zip(ph.timestamp, ph.equity, ph.profit_loss):
                    d = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                    portfolio_history.append({
                        "date": d,
                        "equity": float(eq) if eq else None,
                        "profit_loss": float(pl) if pl else None,
                    })
                logger.info("Fetched %d days of portfolio history from Alpaca",
                            len(portfolio_history))
        except Exception as e:
            logger.warning("Could not fetch portfolio history: %s", e)

        # Current positions (split-corrected where needed)
        raw_positions = api.list_positions()
        positions = []
        for p in raw_positions:
            qty = float(p.qty)
            avg_entry = float(p.avg_entry_price)
            cost = float(p.cost_basis)
            current = float(p.current_price)

            qty, avg_entry, cost = adjust_position_for_split(
                p.symbol, qty, avg_entry, cost, current
            )

            market_value = qty * current
            total_pl = market_value - cost
            total_pl_pct = (total_pl / cost * 100) if cost else 0

            positions.append({
                "symbol": p.symbol,
                "qty": qty,
                "side": p.side,
                "current_price": current,
                "market_value": round(market_value, 2),
                "avg_entry_price": round(avg_entry, 2),
                "cost_basis": round(cost, 2),
                "today_pl_pct": float(p.unrealized_intraday_plpc) * 100 if hasattr(p, 'unrealized_intraday_plpc') and p.unrealized_intraday_plpc else 0,
                "today_pl": float(p.unrealized_intraday_pl) if hasattr(p, 'unrealized_intraday_pl') and p.unrealized_intraday_pl else 0,
                "total_pl_pct": round(total_pl_pct, 3),
                "total_pl": round(total_pl, 2),
            })

        # Recent orders (last ~200 closed orders)
        orders = api.list_orders(status="closed", limit=200, direction="desc")
        filled_orders = [o for o in orders if o.status == "filled"]

        # Group orders by date into "rebalances"
        by_date = defaultdict(list)
        for o in filled_orders:
            filled = str(o.filled_at) if o.filled_at else str(o.submitted_at)
            date = filled[:10]
            by_date[date].append({
                "symbol": o.symbol,
                "side": o.side,
                "quantity": float(o.filled_qty),
                "price": float(o.filled_avg_price) if o.filled_avg_price else 0,
                "slippage_bps": 0,
            })

        rebalances = []
        for date in sorted(by_date.keys(), reverse=True):
            rebalances.append({
                "date": date,
                "portfolio_value": account_info["equity"],
                "trades": by_date[date],
            })

        logger.info("Fetched %d orders across %d rebalance dates from Alpaca",
                    len(filled_orders), len(rebalances))

        # Adjust account equity and portfolio history for split corrections.
        # Alpaca paper trading doesn't process corporate actions, so equity
        # and history carry wrong market values for split-affected stocks:
        # (a) currently held pre-split positions — market value is understated
        #     by (ratio-1) * qty * price;
        # (b) pre-split shares sold after the split — proceeds were booked at
        #     the pre-split qty, missing (ratio-1) * qty * price of cash.
        held_adjustment = 0
        for p in raw_positions:
            split = STOCK_SPLITS.get(p.symbol)
            if split and _looks_presplit(
                float(p.avg_entry_price), float(p.current_price), split["ratio"]
            ):
                held_adjustment += (
                    float(p.qty) * (split["ratio"] - 1) * float(p.current_price)
                )

        sold_credit = _split_sold_credit(filled_orders)

        equity_adjustment = held_adjustment + sold_credit
        if equity_adjustment:
            total_mv = sum(p["market_value"] for p in positions)
            account_info["equity"] = round(
                account_info["cash"] + sold_credit + total_mv, 2
            )
            logger.info(
                "Adjusted account equity by +$%.2f for stock splits "
                "(held=%.2f, sold=%.2f)",
                equity_adjustment, held_adjustment, sold_credit,
            )
            # Correction start date = max(split_date, first buy date) per symbol
            cutoff_date = None
            for sym, split in STOCK_SPLITS.items():
                split_date = split["date"]
                buy_date = None
                for reb in rebalances:
                    for t in reb.get("trades", []):
                        if t["symbol"] == sym and t["side"] == "buy":
                            if buy_date is None or reb["date"] < buy_date:
                                buy_date = reb["date"]
                effective = max(split_date, buy_date) if buy_date else split_date
                if cutoff_date is None or effective < cutoff_date:
                    cutoff_date = effective
            if cutoff_date:
                for h in portfolio_history:
                    if h["equity"] is not None and h["date"] > cutoff_date:
                        h["equity"] = round(h["equity"] + equity_adjustment, 2)

        # Fetch SPY benchmark aligned to portfolio history dates
        spy_history = _fetch_spy_benchmark(portfolio_history)

        return {
            "account": account_info,
            "positions": positions,
            "portfolio_history": portfolio_history,
            "spy_history": spy_history,
            "rebalances": rebalances,
        }

    except Exception as e:
        logger.error("Failed to fetch from Alpaca: %s, falling back to local logs", e)
        return parse_local_trade_logs(state_file)


def _fetch_spy_benchmark(portfolio_history) -> list:
    """SPY equity curve normalized to the account's starting equity."""
    spy_history = []
    valid_hist = [h for h in portfolio_history if h["equity"]]
    if not valid_hist:
        return spy_history
    try:
        import yfinance as yf
        start_date = valid_hist[0]["date"]
        # yfinance end is exclusive, add 2 days buffer
        end_dt = datetime.strptime(valid_hist[-1]["date"], "%Y-%m-%d") + timedelta(days=2)
        spy = yf.download("SPY", start=start_date, end=end_dt.strftime("%Y-%m-%d"),
                          progress=False)
        if not spy.empty:
            if hasattr(spy.columns, 'levels'):
                spy.columns = spy.columns.get_level_values(0)
            # Build date->close lookup, forward-fill for non-trading days
            spy_by_date = {}
            for idx_date, row in spy.iterrows():
                spy_by_date[idx_date.strftime("%Y-%m-%d")] = float(row["Close"])
            start_equity = valid_hist[0]["equity"]
            first_spy = spy_by_date.get(valid_hist[0]["date"])
            if first_spy is None:
                first_spy = float(spy["Close"].iloc[0])
            last_spy_close = first_spy
            for h in valid_hist:
                d = h["date"]
                if d in spy_by_date:
                    last_spy_close = spy_by_date[d]
                spy_equity = start_equity * last_spy_close / first_spy
                spy_history.append({"date": d, "equity": round(spy_equity, 2)})
            logger.info("Fetched %d days of SPY benchmark data", len(spy_history))
    except Exception as e:
        logger.warning("Could not fetch SPY benchmark: %s", e)
    return spy_history


def parse_local_trade_logs(state_file: str) -> dict:
    """Fallback: parse local log files for trade history."""
    rebalances = []

    # Try trade_events.jsonl first
    events_file = Path("logs/trade_events.jsonl")
    if events_file.exists():
        current = None
        for line in events_file.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "rebalance_start":
                current = {
                    "date": event.get("timestamp", "")[:10],
                    "portfolio_value": event.get("portfolio_value", 0),
                    "trades": [],
                }
                rebalances.append(current)
            elif event.get("event") == "order_filled" and current is not None:
                current["trades"].append({
                    "symbol": event.get("symbol", ""),
                    "side": event.get("side", ""),
                    "quantity": event.get("quantity", 0),
                    "price": event.get("filled_price", 0),
                    "slippage_bps": round(event.get("slippage_bps", 0), 2),
                })

    # Fallback to the strategy's state file
    state_path = Path(state_file)
    if not rebalances and state_path.exists():
        state = json.loads(state_path.read_text())
        for entry in state.get("trade_history", []):
            reb = {
                "date": entry.get("date", "")[:10],
                "portfolio_value": 0,
                "trades": [],
            }
            for t in entry.get("trades", []):
                if t.get("status") == "filled":
                    reb["trades"].append({
                        "symbol": t.get("symbol", ""),
                        "side": t.get("side", ""),
                        "quantity": t.get("qty", 0),
                        "price": t.get("price", 0),
                        "slippage_bps": 0,
                    })
            if reb["trades"]:
                rebalances.append(reb)

    return {"rebalances": rebalances}
