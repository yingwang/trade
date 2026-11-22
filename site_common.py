"""Shared helpers for the dashboard generation scripts.

generate_site.py and generate_site_lgbm.py used to carry line-identical
copies of the Alpaca trade-history fetch and the stock-split correction.
The split correction in particular is easy to get wrong and was duplicated;
it now lives here once, with guards that the original copies lacked.
"""

from __future__ import annotations

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

# Transparent, investable returns proxies for the actual-account attribution.
# Each style factor is a long/short ETF spread; sector effects use the relevant
# Select Sector (or equivalent) ETF in excess of SPY.  Stock loadings are
# estimated only from data preceding the attributed day.
ATTRIBUTION_STYLE_PROXIES = {
    "size": ("IWM", "SPY"),
    "value": ("IWD", "IWF"),
    "momentum": ("MTUM", "SPY"),
    "quality": ("QUAL", "SPY"),
    "low_volatility": ("USMV", "SPY"),
}

ATTRIBUTION_SECTOR_PROXIES = {
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
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


def _validated_presplit_sales(sales: dict | None) -> dict:
    """Validate the recorded sales of shares a paper account still carried
    in pre-split units (``dashboard.split_presplit_sales.<book>``).

    ``{SYMBOL: [{date, quantity, price}, ...]}`` with the pre-split quantity
    sold and the post-split fill price; an entry may also carry
    ``post_split_shares_held``, the shares bought after the split and still
    held at that date (0 when absent).  The record is authoritative through
    its last date: the credit no longer depends on those orders staying among
    the newest closed orders Alpaca returns.
    """
    result = {}
    for raw_symbol, raw_entries in (sales or {}).items():
        symbol = str(raw_symbol).upper().strip()
        split = STOCK_SPLITS.get(symbol)
        if split is None:
            raise ValueError(f"Unknown pre-split sale symbol: {symbol}")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError(f"Pre-split sales for {symbol} must be a non-empty list")
        entries = []
        for raw in raw_entries:
            if not isinstance(raw, dict):
                raise ValueError(f"Pre-split sale for {symbol} must be a mapping")
            sale_date = str(raw.get("date", ""))
            try:
                datetime.strptime(sale_date, "%Y-%m-%d")
            except ValueError as exc:
                raise ValueError(
                    f"Pre-split sale for {symbol} needs a YYYY-MM-DD date"
                ) from exc
            if sale_date < split["first_adjusted_session"]:
                raise ValueError(
                    f"Pre-split sale for {symbol} on {sale_date} precedes the split"
                )
            quantity = float(raw.get("quantity", 0))
            price = float(raw.get("price", 0))
            held = float(raw.get("post_split_shares_held", 0))
            if not all(math.isfinite(v) for v in (quantity, price, held)) or quantity <= 0 or price <= 0 or held < 0:
                raise ValueError(
                    f"Pre-split sale for {symbol} needs a positive quantity and price"
                )
            entries.append({
                "date": sale_date, "quantity": quantity, "price": price,
                "post_split_shares_held": held,
            })
        result[symbol] = sorted(entries, key=lambda entry: entry["date"])
    return result


def _split_sold_credit(
    filled_orders,
    cash_compensations: dict | None = None,
    as_of: str | None = None,
    presplit_sales: dict | None = None,
) -> float:
    """Cash that post-split sells of PRE-split shares failed to book.

    Alpaca books such a sell as qty x price with the pre-split qty, missing
    (ratio-1) x qty x price of proceeds. But a sell of shares RE-BOUGHT
    after the split is booked correctly and must not be credited — so sells
    first consume the quantity visibly bought after the split date (correct
    bookings), and only the unexplained remainder is treated as pre-split
    shares. Order history is finite (the newest 500 closed orders), so a
    pre-split sale that ages out of it would take its credit with it; the
    sales recorded in ``presplit_sales`` stand in for every order of that
    symbol up to the last recorded date.
    """
    compensations = _validated_split_cash_compensations(cash_compensations)
    recorded = _validated_presplit_sales(presplit_sales)
    as_of = as_of or datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    credit = 0.0
    for sym, split in STOCK_SPLITS.items():
        ratio = split["ratio"]
        split_date = split["date"]
        ledger = recorded.get(sym, [])
        through = ledger[-1]["date"] if ledger else None

        events = []
        for o in filled_orders:
            if o.symbol != sym:
                continue
            timestamp = _parse_timestamp(o.filled_at or o.submitted_at)
            fill_date = _eastern_date(timestamp)
            if timestamp is None or fill_date < split_date:
                continue
            if through is not None and fill_date <= through:
                continue  # covered by the recorded sales
            qty = float(o.filled_qty)
            price = float(o.filled_avg_price) if o.filled_avg_price else 0.0
            events.append((timestamp, fill_date, _enum_value(o.side), qty, price))

        events.sort(key=lambda event: event[0])
        post_split_bought = ledger[-1]["post_split_shares_held"] if ledger else 0.0
        symbol_credit = 0.0
        for entry in ledger:
            inferred = (ratio - 1) * entry["quantity"] * entry["price"]
            symbol_credit += inferred
            logger.info(
                "Split sold-credit: %s recorded sale of %.0f pre-split share(s) "
                "on %s -> +$%.2f", sym, entry["quantity"], entry["date"], inferred,
            )
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
    presplit_sales: dict | None = None,
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
    recorded = _validated_presplit_sales(presplit_sales)
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

        ledger = recorded.get(symbol, [])
        through = ledger[-1]["date"] if ledger else None
        events = []
        for order in filled_orders:
            if order.symbol != symbol:
                continue
            timestamp = _parse_timestamp(order.filled_at or order.submitted_at)
            event_date = _eastern_date(timestamp)
            if not event_date or event_date < split["first_adjusted_session"]:
                continue
            if through is not None and event_date <= through:
                continue  # covered by the recorded sales
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

        post_split_bought = ledger[-1]["post_split_shares_held"] if ledger else 0.0
        presplit_sales = [
            (pd.Timestamp(entry["date"]), entry["quantity"], entry["price"])
            for entry in ledger
        ]
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
    since: str | None = None,
    split_presplit_sales: dict | None = None,
) -> dict:
    """Trade history + positions + equity history for one Alpaca account.

    ``split_presplit_sales`` is the book's ``dashboard.split_presplit_sales``
    entry: recorded sales of shares still in pre-split units, so the split
    correction outlives the 500-order window (see _validated_presplit_sales).

    ``since`` (YYYY-MM-DD) keeps only the fills and the daily equity from that
    date on, for a book that took over an account another book had traded:
    the earnings book's page must not show the trend book's history as its own.
    The last close before ``since`` stays as the starting equity (see
    ``since_history``).

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
            since=since,
            split_presplit_sales=split_presplit_sales,
        )

    logger.warning("No Alpaca keys in %s/%s, falling back to local logs",
                   api_key_env, secret_key_env)
    return parse_local_trade_logs(state_file)


def current_portfolio_for_display(strategy, prev_weights=None, **kwargs):
    """The strategy's current target for a dashboard, and the held symbols
    the live gate found without price data.

    A live run stops when a holding has no usable price data
    (HeldDataMissingError: the runner stands down rather than sell it as an
    exit).  The page only displays the target, so it does not fail the
    nightly build: it logs the symbols and shows the target computed without
    those holdings, with the symbols published next to it so the page does
    not pass that target off as what the account will trade.
    """
    from quant.data.quality import HeldDataMissingError

    try:
        return strategy.get_current_portfolio(prev_weights=prev_weights, **kwargs), []
    except HeldDataMissingError as exc:
        missing = sorted(exc.symbols)
        logger.warning(
            "Dashboard target: held %s have no usable price data today (a live "
            "run would stand down); showing the target computed without them",
            missing,
        )
        rest = (
            prev_weights.drop(labels=missing, errors="ignore")
            if prev_weights is not None else None
        )
        return strategy.get_current_portfolio(prev_weights=rest, **kwargs), missing


def generate_actual_attribution(
    trades: dict,
    asset_prices,
    fundamentals,
    config: dict,
    benchmark: str = "SPY",
) -> dict:
    """Build the actual-alpha attribution payload used by both dashboards.

    This runs only in the nightly site workflow, after the strategy has already
    fetched its adjusted universe prices and current sector snapshot.  A
    separate compact ETF download supplies investable style/sector returns.
    Failure to fetch a proxy does not invent data: that factor remains zero and
    its stock return stays in security selection or reconciliation residual.
    """

    from quant.attribution import attribute_actual_performance

    portfolio_history = [
        dict(row) for row in (trades.get("portfolio_history", []) or [])
    ]
    current_equity = trades.get("account", {}).get("equity")
    if portfolio_history and current_equity is not None and not trades.get("equity_aligned"):
        # Align the broker's post-close equity with the latest completed
        # session rather than clobbering T-1 when Alpaca's 1D bar lags.
        import pandas as pd

        latest_session = None
        if asset_prices is not None and not getattr(asset_prices, "empty", True):
            latest_session = (
                pd.to_datetime(asset_prices.index)
                .tz_localize(None)
                .normalize()
                .max()
                .strftime("%Y-%m-%d")
            )
        portfolio_history = align_current_equity_to_session(
            portfolio_history, current_equity, latest_session
        )
    if len([row for row in portfolio_history if row.get("equity") is not None]) < 2:
        return {
            "status": "insufficient_data",
            "summary": {"trading_days": 0},
            "diagnostics": {"reason": "Need at least two actual equity values"},
            "components": {},
            "style_breakdown": {},
            "industry_breakdown": {},
            "daily": [],
        }

    try:
        import pandas as pd
        import yfinance as yf

        history_dates = pd.to_datetime(
            [row["date"] for row in portfolio_history if row.get("equity") is not None]
        )
        start = (history_dates.min() - pd.Timedelta(days=550)).strftime("%Y-%m-%d")
        end = (history_dates.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        proxy_symbols = sorted(
            {benchmark}
            | {
                symbol
                for pair in ATTRIBUTION_STYLE_PROXIES.values()
                for symbol in pair
            }
            | set(ATTRIBUTION_SECTOR_PROXIES.values())
        )
        downloaded = yf.download(
            proxy_symbols,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if isinstance(downloaded.columns, pd.MultiIndex):
            proxy_prices = downloaded["Close"]
        elif "Close" in downloaded:
            proxy_prices = downloaded[["Close"]].rename(
                columns={"Close": proxy_symbols[0]}
            )
        else:
            proxy_prices = pd.DataFrame()
    except Exception as exc:
        logger.warning("Could not fetch attribution ETF proxies: %s", exc)
        import pandas as pd

        proxy_prices = pd.DataFrame()

    # Reuse SPY from the exact strategy download if the proxy request failed or
    # returned a shorter calendar.  combine_first preserves proxy observations.
    if asset_prices is not None and benchmark in asset_prices:
        benchmark_frame = asset_prices[[benchmark]]
        proxy_prices = proxy_prices.combine_first(benchmark_frame)

    from quant.utils.config import static_sector_map

    sector_map = static_sector_map(config)
    if sector_map is None and fundamentals is not None and not fundamentals.empty and "sector" in fundamentals:
        sector_map = fundamentals["sector"]

    # Fills and share counts are on the unadjusted (split-adjusted only)
    # basis; the shortfall reference and the holding weights must be too.
    traded = sorted(
        {str(p.get("symbol", "")).upper() for p in trades.get("positions", []) or []}
        | {
            str(t.get("symbol", "")).upper()
            for r in trades.get("rebalances", []) or []
            for t in r.get("trades", []) or []
        }
        - {""}
    )
    reference_prices = None
    if traded:
        equity_dates = pd.to_datetime(
            [row["date"] for row in portfolio_history if row.get("equity") is not None]
        )
        reference_prices = _fetch_unadjusted_closes(
            traded,
            (equity_dates.min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
            (equity_dates.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        )
        if reference_prices is None or reference_prices.empty:
            logger.warning(
                "Unadjusted closes unavailable; trading costs and weights fall back "
                "to dividend-adjusted closes (a fill then shows every later "
                "dividend as cost)"
            )
            reference_prices = None

    attribution_config = config.get("attribution", {})
    result = attribute_actual_performance(
        portfolio_history=portfolio_history,
        rebalances=_attribution_rebalances(
            trades.get("rebalances", []) or []
        ),
        current_positions=trades.get("positions", []) or [],
        asset_prices=asset_prices,
        proxy_prices=proxy_prices,
        sector_map=sector_map,
        benchmark=benchmark,
        style_proxies=ATTRIBUTION_STYLE_PROXIES,
        sector_proxies=ATTRIBUTION_SECTOR_PROXIES,
        lookback=int(attribution_config.get("lookback_days", 252)),
        min_observations=int(attribution_config.get("min_observations", 60)),
        # Alpaca paper cash earns no interest.  This keeps the cash-drag term
        # tied to actual account economics rather than the backtest's 4% hurdle.
        annual_cash_return=float(attribution_config.get("annual_cash_return", 0.0)),
        reference_prices=reference_prices,
    )
    payload = result.to_dict()
    payload.setdefault("methodology", {})
    payload["methodology"] = {
        "benchmark": benchmark,
        "equity_truth": "Alpaca portfolio history; final point uses current account equity",
        "style_proxies": {
            name: f"{long_symbol}-{short_symbol}"
            for name, (long_symbol, short_symbol) in ATTRIBUTION_STYLE_PROXIES.items()
        },
        "sector_proxies": ATTRIBUTION_SECTOR_PROXIES,
        "lookback_days": int(attribution_config.get("lookback_days", 252)),
        "min_observations": int(attribution_config.get("min_observations", 60)),
        "cash_return": float(attribution_config.get("annual_cash_return", 0.0)),
        "trading_costs": (
            "implementation shortfall versus prior unadjusted close"
            if reference_prices is not None
            else "implementation shortfall versus prior adjusted close (unadjusted closes unavailable)"
        ),
        "sector_classification": (
            "current yfinance sector snapshot" if sector_map is not None
            else "none: industry effect not measured for this book"
        ),
        "linking": "Carino",
    }
    return payload


def _fetch_unadjusted_closes(symbols, start: str, end: str):
    """Closes not adjusted for dividends (yfinance ``Close`` with
    ``auto_adjust=False``, split-adjusted), one batched download.

    Dashboard generators only, never the trading path.  None on failure.
    """
    try:
        import pandas as pd
        import yfinance as yf

        downloaded = yf.download(
            list(symbols), start=start, end=end, auto_adjust=False, progress=False,
        )
        if downloaded is None or downloaded.empty:
            return None
        if isinstance(downloaded.columns, pd.MultiIndex):
            return downloaded["Close"]
        if "Close" in downloaded:
            return downloaded[["Close"]].rename(columns={"Close": list(symbols)[0]})
        return None
    except Exception as exc:
        logger.warning("Could not fetch unadjusted closes for attribution: %s", exc)
        return None


def _attribution_rebalances(rebalances: list[dict]) -> list[dict]:
    """Return fills in the split-adjusted units used by Yahoo prices.

    Alpaca paper accounts may retain old share units through a split.  The
    dashboard already corrects the corresponding equity curve; attribution
    must apply the same economic units or the correction would appear as an
    enormous unexplained residual.  Post-split buys are consumed first, just
    like ``_split_sold_credit``: only the unmatched part of a later sell is a
    sale of stale pre-split shares.
    """

    events = []
    for rebalance in rebalances:
        for sequence, raw_trade in enumerate(rebalance.get("trades", []) or []):
            trade = dict(raw_trade)
            timestamp = _parse_timestamp(trade.get("filled_at"))
            date = str(rebalance.get("date", ""))[:10]
            events.append((timestamp, date, sequence, trade))
    events.sort(
        key=lambda item: (
            item[0] or datetime.min.replace(tzinfo=timezone.utc),
            item[1],
            item[2],
        )
    )

    post_split_bought = defaultdict(float)
    by_date = defaultdict(list)
    for _, date, _, trade in events:
        symbol = str(trade.get("symbol", "")).upper()
        split = STOCK_SPLITS.get(symbol)
        if split is not None:
            ratio = float(split["ratio"])
            quantity = float(trade.get("quantity", trade.get("qty", 0)) or 0)
            price = float(trade.get("price", 0) or 0)
            side = str(trade.get("side", "")).lower()
            if date < split["first_adjusted_session"]:
                trade["quantity"] = quantity * ratio
                if price > 0:
                    trade["price"] = price / ratio
            elif side == "buy":
                post_split_bought[symbol] += quantity
            elif side == "sell":
                covered = min(quantity, post_split_bought[symbol])
                post_split_bought[symbol] -= covered
                stale_quantity = quantity - covered
                trade["quantity"] = covered + stale_quantity * ratio
        by_date[date].append(trade)

    return [
        {"date": date, "trades": trades}
        for date, trades in sorted(by_date.items(), reverse=True)
    ]


def since_history(history: list[dict], since: str) -> list[dict]:
    """Daily equity rows for a book that started on ``since``, from its starting point.

    The last row before ``since`` is kept as the equity the book took over: it
    is the book's starting capital, and without it a book with one session of
    its own has a single point, which the page cannot draw (nor compare with
    SPY). Before Alpaca has posted the first session's bar it is the only row.
    """
    before = [row for row in history if row["date"] < since]
    return before[-1:] + [row for row in history if row["date"] >= since]


def _fetch_trades_from_alpaca(
    api_key,
    secret_key,
    state_file,
    split_cash_compensations: dict | None = None,
    since: str | None = None,
    split_presplit_sales: dict | None = None,
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
        if since:
            portfolio_history = since_history(portfolio_history, since)

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
        if since:
            filled_orders = [
                o for o in filled_orders
                if (_eastern_date(_parse_timestamp(o.filled_at or o.submitted_at)) or "") >= since
            ]

        # Group orders by date into "rebalances"
        by_date = defaultdict(list)
        for o in filled_orders:
            fill_timestamp = _parse_timestamp(o.filled_at or o.submitted_at)
            date = _eastern_date(fill_timestamp)
            by_date[date].append({
                "symbol": o.symbol,
                "side": _enum_value(o.side),
                "quantity": float(o.filled_qty),
                "price": float(o.filled_avg_price) if o.filled_avg_price else 0,
                "filled_at": (
                    fill_timestamp.isoformat()
                    if fill_timestamp is not None
                    else None
                ),
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
            presplit_sales=split_presplit_sales,
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
            presplit_sales=split_presplit_sales,
        )
        for row in portfolio_history:
            adjustment = history_adjustments.get(row["date"], 0.0)
            if row["equity"] is not None and adjustment:
                row["equity"] = round(row["equity"] + adjustment, 2)

        # Fetch completed SPY closes once so we know the latest settled US
        # session before aligning current_equity onto portfolio_history.
        valid_hist = [h for h in portfolio_history if h.get("equity") is not None and h.get("date")]
        spy_by_date = _fetch_spy_closes(
            valid_hist[0]["date"],
            valid_hist[-1]["date"],
        ) if valid_hist else {}
        latest_spy_session = max(spy_by_date) if spy_by_date else None

        intraday_open = False
        try:
            clock = api.get_clock()
            intraday_open = bool(getattr(clock, "is_open", False))
        except Exception:
            pass
        market_now = datetime.now(ZoneInfo("America/New_York"))
        market_today = market_now.date().isoformat()
        in_close_grace = (
            latest_spy_session is not None
            and market_today > latest_spy_session
            and market_now.weekday() < 5
            and (16, 0) <= (market_now.hour, market_now.minute) < (16, 15)
        )
        equity_is_post_close = not (
            latest_spy_session is not None
            and market_today > latest_spy_session
            and (intraday_open or in_close_grace)
        )
        if equity_is_post_close and latest_spy_session is not None:
            portfolio_history = align_current_equity_to_session(
                portfolio_history, account_info["equity"], latest_spy_session
            )

        equity_by_date = {
            row["date"]: row["equity"]
            for row in portfolio_history
            if row["equity"] is not None
        }
        for rebalance in rebalances:
            rebalance["portfolio_value"] = equity_by_date.get(
                rebalance["date"], account_info["equity"]
            )

        # Build SPY benchmark aligned to the finalized portfolio history dates
        spy_history = _build_spy_history(portfolio_history, spy_by_date)

        return {
            "source": "alpaca",
            "account": account_info,
            "positions": positions,
            "portfolio_history": portfolio_history,
            "spy_history": spy_history,
            "rebalances": rebalances,
            "corporate_action_adjustments": history_adjustments,
            "equity_aligned": True,
        }

    except Exception as e:
        logger.error("Failed to fetch from Alpaca: %s, falling back to local logs", e)
        return parse_local_trade_logs(state_file)


def align_current_equity_to_session(
    portfolio_history: list[dict],
    current_equity: float | None,
    latest_session: str | None,
    *,
    max_gap_days: int = 10,
) -> list[dict]:
    """Align the broker's post-close equity with ``latest_session``.

    Alpaca's ``1D`` portfolio history often still ends at ``T-1`` right after
    the US close while ``account.equity`` already reflects ``T``. Updating
    ``history[-1]`` in place when ``last_date == latest_session`` and appending
    a ``latest_session`` row when ``last_date < latest_session`` prevents ``T``'s
    P&L from being misattributed to ``T-1`` and keeps ``T``'s fills in-window.
    """
    rows = [dict(row) for row in (portfolio_history or [])]
    if not rows or current_equity is None or not latest_session:
        return rows
    equity_val = round(float(current_equity), 2)
    if not math.isfinite(equity_val):
        return rows

    valid_indices = [
        i for i, r in enumerate(rows) if r.get("equity") is not None and r.get("date")
    ]
    if not valid_indices:
        return rows

    last_idx = valid_indices[-1]
    last_date = str(rows[last_idx]["date"])[:10]
    target_date = str(latest_session)[:10]

    if last_date == target_date:
        prev_eq = (
            float(rows[valid_indices[-2]]["equity"])
            if len(valid_indices) >= 2
            else None
        )
        rows[last_idx]["equity"] = equity_val
        if "profit_loss" in rows[last_idx] and prev_eq is not None:
            rows[last_idx]["profit_loss"] = round(equity_val - prev_eq, 2)
        return rows

    if last_date < target_date:
        try:
            gap = (
                datetime.strptime(target_date, "%Y-%m-%d").date()
                - datetime.strptime(last_date, "%Y-%m-%d").date()
            ).days
        except ValueError:
            return rows
        if 0 < gap <= max_gap_days:
            prev_eq = float(rows[last_idx]["equity"])
            rows.append(
                {
                    "date": target_date,
                    "equity": equity_val,
                    "profit_loss": round(equity_val - prev_eq, 2),
                }
            )
    return rows


def _fetch_spy_closes(start_date: str, last_hist_date: str | None = None) -> dict[str, float]:
    """Return completed daily SPY closes from ``start_date`` through the latest completed session."""
    spy_by_date: dict[str, float] = {}
    try:
        import yfinance as yf
        from quant.data.market_data import MarketData

        exclusive_end = MarketData._exclusive_download_end()
        if last_hist_date:
            hist_end = (
                datetime.strptime(last_hist_date, "%Y-%m-%d") + timedelta(days=2)
            ).strftime("%Y-%m-%d")
            exclusive_end = max(exclusive_end, hist_end)
        spy = yf.download("SPY", start=start_date, end=exclusive_end, progress=False)
        if not spy.empty:
            if hasattr(spy.columns, "levels"):
                spy.columns = spy.columns.get_level_values(0)
            for idx_date, row in spy.iterrows():
                spy_by_date[idx_date.strftime("%Y-%m-%d")] = float(row["Close"])
    except Exception as e:
        logger.warning("Could not fetch SPY benchmark: %s", e)
    return spy_by_date


def _build_spy_history(portfolio_history, spy_by_date: dict[str, float]) -> list:
    """Normalize SPY closes in ``spy_by_date`` onto ``portfolio_history`` dates."""
    spy_history = []
    valid_hist = [h for h in portfolio_history if h.get("equity") is not None]
    if not valid_hist or not spy_by_date:
        return spy_history
    start_equity = valid_hist[0]["equity"]
    first_spy = spy_by_date.get(valid_hist[0]["date"])
    if first_spy is None:
        first_spy = next(iter(spy_by_date.values()))
    last_spy_close = first_spy
    for h in valid_hist:
        d = h["date"]
        if d in spy_by_date:
            last_spy_close = spy_by_date[d]
        spy_equity = start_equity * last_spy_close / first_spy
        spy_history.append({"date": d, "equity": round(spy_equity, 2)})
    logger.info("Fetched %d days of SPY benchmark data", len(spy_history))
    return spy_history


def _fetch_spy_benchmark(portfolio_history) -> list:
    """SPY equity curve normalized to the account's starting equity."""
    valid_hist = [h for h in portfolio_history if h.get("equity") is not None]
    if not valid_hist:
        return []
    spy_by_date = _fetch_spy_closes(valid_hist[0]["date"], valid_hist[-1]["date"])
    return _build_spy_history(portfolio_history, spy_by_date)


_EVENTS_LOG_BY_STATE = {
    "logs/paper_trade_state.json": "logs/trade_events_multi.jsonl",
    "logs/paper_trade_lgbm_state.json": "logs/trade_events_lgbm.jsonl",
    "logs/paper_trade_trend_state.json": "logs/trade_events_trend.jsonl",
    "logs/paper_trade_earnings_state.json": "logs/trade_events_earnings.jsonl",
}


def events_log_for_state(state_file: str) -> Path:
    """Per-book event log path; falls back to the legacy shared file for multi."""
    key = str(state_file).replace('\\', "/")
    mapped = _EVENTS_LOG_BY_STATE.get(key)
    if mapped:
        path = Path(mapped)
        if path.exists() or key != "logs/paper_trade_state.json":
            return path
    return Path("logs/trade_events.jsonl")


def parse_local_trade_logs(state_file: str) -> dict:
    """Fallback: parse local log files for trade history."""
    rebalances = []

    # Prefer the book-specific event log; legacy shared file is multi-only.
    events_file = events_log_for_state(state_file)
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
