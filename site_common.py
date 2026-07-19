"""Shared helpers for the dashboard generation scripts.

generate_site.py and generate_site_lgbm.py used to carry line-identical
copies of the Alpaca trade-history fetch and the stock-split correction.
The split correction in particular is easy to get wrong and was duplicated;
it now lives here once, with guards that the original copies lacked.
"""

import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from quant.data.corporate_actions import KNOWN_STOCK_SPLITS, looks_presplit

logger = logging.getLogger(__name__)


# Stock splits not yet reflected in Alpaca paper trading (paper accounts do
# not process corporate actions). Map symbol -> ratio and split date.
STOCK_SPLITS = {
    split.symbol: {
        "ratio": split.ratio,
        "effective_date": split.effective_date.isoformat(),
        "first_adjusted_session": split.first_adjusted_session.isoformat(),
        # Backward-compatible alias: order/equity corrections begin when the
        # market first trades on the adjusted basis, not on the legal date.
        "date": split.first_adjusted_session.isoformat(),
    }
    for split in KNOWN_STOCK_SPLITS.values()
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
    return looks_presplit(avg_entry_price, current_price, ratio)


def _enum_value(value):
    """Return stable text for alpaca-py enum fields and test doubles."""
    return getattr(value, "value", value)


def _parse_timestamp(value) -> datetime | None:
    """Normalize Alpaca timestamps for deterministic event ordering."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(value, tz=timezone.utc)
    elif isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    else:
        text = str(value).replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _eastern_date(value) -> str:
    """Convert an Alpaca timestamp to the US market's calendar date."""
    dt = _parse_timestamp(value)
    if dt is None:
        return str(value)[:10] if value is not None else ""
    return dt.astimezone(ZoneInfo("America/New_York")).date().isoformat()


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


def _validated_split_cash_compensations(compensations: dict | None) -> dict:
    """Validate operator-entered cash that already repaired a paper split."""
    result = {}
    for raw_symbol, raw_entry in (compensations or {}).items():
        symbol = str(raw_symbol).upper().strip()
        if symbol not in STOCK_SPLITS:
            raise ValueError(f"Unknown split compensation symbol: {symbol}")
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Split compensation for {symbol} must be a mapping")
        compensation_date = str(raw_entry.get("date", ""))
        try:
            datetime.strptime(compensation_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(
                f"Split compensation for {symbol} needs a YYYY-MM-DD date"
            ) from exc
        amount = float(raw_entry.get("amount", 0))
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError(
                f"Split compensation for {symbol} needs a positive finite amount"
            )
        result[symbol] = {"date": compensation_date, "amount": amount}
    return result


def _split_sold_credit(
    filled_orders,
    cash_compensations: dict | None = None,
    as_of: str | None = None,
) -> float:
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
    compensations = _validated_split_cash_compensations(cash_compensations)
    as_of = as_of or datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    credit = 0.0
    for sym, split in STOCK_SPLITS.items():
        ratio = split["ratio"]
        split_date = split["date"]

        events = []
        for o in filled_orders:
            if o.symbol != sym:
                continue
            timestamp = _parse_timestamp(o.filled_at or o.submitted_at)
            fill_date = _eastern_date(timestamp)
            if timestamp is None or fill_date < split_date:
                continue
            qty = float(o.filled_qty)
            price = float(o.filled_avg_price) if o.filled_avg_price else 0.0
            events.append((timestamp, fill_date, _enum_value(o.side), qty, price))

        events.sort(key=lambda event: event[0])
        post_split_bought = 0.0
        symbol_credit = 0.0
        for _, fill_date, side, qty, price in events:
            if side == "buy":
                post_split_bought += qty
            elif side == "sell":
                covered = min(qty, post_split_bought)
                post_split_bought -= covered
                presplit_qty = qty - covered
                if presplit_qty > 0:
                    inferred = (ratio - 1) * presplit_qty * price
                    symbol_credit += inferred
                    logger.info(
                        "Split sold-credit: %s sell of %.0f pre-split share(s) "
                        "on %s -> +$%.2f", sym, presplit_qty, fill_date,
                        inferred,
                    )
        compensation = compensations.get(sym)
        if compensation and compensation["date"] <= as_of:
            offset = min(symbol_credit, compensation["amount"])
            symbol_credit -= offset
            if abs(symbol_credit) < 1e-9:
                symbol_credit = 0.0
            logger.info(
                "Split sold-credit offset: %s manual cash on %s -> -$%.2f",
                sym,
                compensation["date"],
                offset,
            )
        credit += symbol_credit
    return credit


def _split_history_adjustments(
    portfolio_history,
    raw_positions,
    filled_orders,
    cash_compensations: dict | None = None,
) -> dict:
    """Build date-specific equity corrections for stale paper-account splits.

    Held-share corrections vary with the adjusted market price, while missing
    sale proceeds accumulate only from each sale date onward.  Applying the
    current correction as a constant to the whole history (the old behavior)
    rewrote past returns and drawdowns incorrectly.
    """
    if not portfolio_history:
        return {}

    try:
        import pandas as pd
        import yfinance as yf
    except ImportError:
        return {}

    compensations = _validated_split_cash_compensations(cash_compensations)
    history_dates = pd.DatetimeIndex(
        pd.to_datetime([row["date"] for row in portfolio_history])
    ).normalize().unique().sort_values()
    adjustments = pd.Series(0.0, index=history_dates)
    raw_by_symbol = {p.symbol: p for p in raw_positions}

    for symbol, split in STOCK_SPLITS.items():
        ratio = float(split["ratio"])
        start = pd.Timestamp(split["first_adjusted_session"])
        if history_dates.max() < start:
            continue

        current = raw_by_symbol.get(symbol)
        current_presplit_qty = 0.0
        if current is not None and _looks_presplit(
            float(current.avg_entry_price), float(current.current_price), ratio
        ):
            current_presplit_qty = float(current.qty)

        events = []
        for order in filled_orders:
            if order.symbol != symbol:
                continue
            timestamp = _parse_timestamp(order.filled_at or order.submitted_at)
            event_date = _eastern_date(timestamp)
            if not event_date or event_date < split["first_adjusted_session"]:
                continue
            events.append(
                (
                    timestamp,
                    pd.Timestamp(event_date),
                    _enum_value(order.side),
                    float(order.filled_qty or 0),
                    float(order.filled_avg_price or 0),
                )
            )
        events.sort(key=lambda event: event[0] or datetime.min.replace(tzinfo=timezone.utc))

        post_split_bought = 0.0
        presplit_sales = []
        for _, event_date, side, qty, price in events:
            if side == "buy":
                post_split_bought += qty
            elif side == "sell":
                covered = min(qty, post_split_bought)
                post_split_bought -= covered
                presplit_qty = max(0.0, qty - covered)
                if presplit_qty:
                    presplit_sales.append((event_date, presplit_qty, price))

        initial_presplit_qty = current_presplit_qty + sum(
            qty for _, qty, _ in presplit_sales
        )
        if initial_presplit_qty <= 0 and not presplit_sales:
            continue

        end = history_dates.max() + pd.Timedelta(days=2)
        try:
            downloaded = yf.download(
                symbol,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )
            close = downloaded["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
            close = close.reindex(history_dates.union(close.index)).sort_index().ffill()
            close = close.reindex(history_dates)
        except Exception as exc:
            logger.warning(
                "Could not build date-specific %s split history correction: %s",
                symbol,
                exc,
            )
            continue

        remaining = initial_presplit_qty
        cumulative_credit = 0.0
        sale_index = 0
        for history_date in history_dates:
            if history_date < start:
                continue
            while (
                sale_index < len(presplit_sales)
                and presplit_sales[sale_index][0] <= history_date
            ):
                _, sold_qty, sold_price = presplit_sales[sale_index]
                remaining = max(0.0, remaining - sold_qty)
                cumulative_credit += (ratio - 1.0) * sold_qty * sold_price
                sale_index += 1
            market_price = close.get(history_date)
            if pd.isna(market_price):
                continue
            compensation = compensations.get(symbol)
            compensation_amount = 0.0
            if (
                compensation
                and pd.Timestamp(compensation["date"]) <= history_date
            ):
                compensation_amount = compensation["amount"]
            adjustment = (
                (ratio - 1.0) * remaining * float(market_price)
                + max(0.0, cumulative_credit - compensation_amount)
            )
            adjustments.loc[history_date] += adjustment

    return {
        date.strftime("%Y-%m-%d"): float(value)
        for date, value in adjustments.items()
        if abs(value) > 1e-9
    }


def fetch_trade_history(
    api_key_env: str,
    secret_key_env: str,
    state_file: str,
    split_cash_compensations: dict | None = None,
) -> dict:
    """Trade history + positions + equity history for one Alpaca account.

    Falls back to local log files if API keys are not available.
    """
    api_key = os.environ.get(api_key_env, "")
    secret_key = os.environ.get(secret_key_env, "")

    if api_key and secret_key:
        return _fetch_trades_from_alpaca(
            api_key,
            secret_key,
            state_file,
            split_cash_compensations=split_cash_compensations,
        )

    logger.warning("No Alpaca keys in %s/%s, falling back to local logs",
                   api_key_env, secret_key_env)
    return parse_local_trade_logs(state_file)


def _fetch_trades_from_alpaca(
    api_key,
    secret_key,
    state_file,
    split_cash_compensations: dict | None = None,
) -> dict:
    """Pull trade history and current positions from the Alpaca API."""
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.common.enums import Sort
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest
    except ImportError:
        logger.warning("alpaca-py not installed, falling back to local logs")
        return parse_local_trade_logs(state_file)

    try:
        api = TradingClient(api_key, secret_key, paper=True)

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
            ph = api.get_portfolio_history(
                GetPortfolioHistoryRequest(period="all", timeframe="1D")
            )
            if ph and hasattr(ph, 'equity') and ph.equity:
                for ts, eq, pl in zip(ph.timestamp, ph.equity, ph.profit_loss):
                    d = _eastern_date(ts)
                    portfolio_history.append({
                        "date": d,
                        "equity": float(eq) if eq is not None else None,
                        "profit_loss": float(pl) if pl is not None else None,
                    })
                logger.info("Fetched %d days of portfolio history from Alpaca",
                            len(portfolio_history))
        except Exception as e:
            logger.warning("Could not fetch portfolio history: %s", e)

        # Current positions (split-corrected where needed)
        raw_positions = api.get_all_positions()
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
                "side": _enum_value(p.side),
                "current_price": current,
                "market_value": round(market_value, 2),
                "avg_entry_price": round(avg_entry, 2),
                "cost_basis": round(cost, 2),
                "today_pl_pct": float(p.unrealized_intraday_plpc) * 100 if getattr(p, 'unrealized_intraday_plpc', None) is not None else 0,
                "today_pl": float(p.unrealized_intraday_pl) if getattr(p, 'unrealized_intraday_pl', None) is not None else 0,
                "total_pl_pct": round(total_pl_pct, 3),
                "total_pl": round(total_pl, 2),
            })

        # Recent orders (last ~200 closed orders)
        orders = api.get_orders(
            GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                limit=500,
                direction=Sort.DESC,
            )
        )
        # Include partially filled and later-cancelled orders: actual shares
        # changed hands even when the terminal status is not "filled".
        filled_orders = [o for o in orders if float(o.filled_qty or 0) > 0]

        # Group orders by date into "rebalances"
        by_date = defaultdict(list)
        for o in filled_orders:
            date = _eastern_date(o.filled_at or o.submitted_at)
            by_date[date].append({
                "symbol": o.symbol,
                "side": _enum_value(o.side),
                "quantity": float(o.filled_qty),
                "price": float(o.filled_avg_price) if o.filled_avg_price else 0,
                "slippage_bps": 0,
                "status": str(_enum_value(o.status)),
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

        sold_credit = _split_sold_credit(
            filled_orders,
            cash_compensations=split_cash_compensations,
        )

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

        history_adjustments = _split_history_adjustments(
            portfolio_history,
            raw_positions,
            filled_orders,
            cash_compensations=split_cash_compensations,
        )
        for row in portfolio_history:
            adjustment = history_adjustments.get(row["date"], 0.0)
            if row["equity"] is not None and adjustment:
                row["equity"] = round(row["equity"] + adjustment, 2)

        equity_by_date = {
            row["date"]: row["equity"]
            for row in portfolio_history
            if row["equity"] is not None
        }
        for rebalance in rebalances:
            rebalance["portfolio_value"] = equity_by_date.get(
                rebalance["date"], account_info["equity"]
            )

        # Fetch SPY benchmark aligned to portfolio history dates
        spy_history = _fetch_spy_benchmark(portfolio_history)

        return {
            "source": "alpaca",
            "account": account_info,
            "positions": positions,
            "portfolio_history": portfolio_history,
            "spy_history": spy_history,
            "rebalances": rebalances,
            "corporate_action_adjustments": history_adjustments,
        }

    except Exception as e:
        logger.error("Failed to fetch from Alpaca: %s, falling back to local logs", e)
        return parse_local_trade_logs(state_file)


def _fetch_spy_benchmark(portfolio_history) -> list:
    """SPY equity curve normalized to the account's starting equity."""
    spy_history = []
    valid_hist = [h for h in portfolio_history if h.get("equity") is not None]
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

    return {"source": "local", "rebalances": rebalances}
