#!/usr/bin/env python3
"""Generate static dashboard data for GitHub Pages.

Produces JSON data files in site/data/ that are loaded by the
static HTML dashboard. Run daily after US market close.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from quant.utils.config import load_config
from quant.strategy import MultiFactorStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("site/data")


def generate_portfolio_data(strategy, config):
    """Section 1: Current portfolio recommendation."""
    portfolio = strategy.get_current_portfolio()

    # Get regime
    prices = strategy.data.fetch_prices()
    returns = strategy.data.compute_returns(prices)
    spy_ret = returns.get(strategy.data.benchmark)
    regime = strategy.optimizer.detect_regime(spy_ret) if spy_ret is not None else "normal"

    total_invested = float(portfolio["weight"].sum())

    data = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "regime": regime,
        "total_invested_pct": round(total_invested * 100, 1),
        "cash_pct": round(max(0, (1 - total_invested)) * 100, 1),
        "positions": [],
    }
    for symbol, row in portfolio.iterrows():
        data["positions"].append({
            "symbol": symbol,
            "weight_pct": float(row["weight_pct"]),
            "dollars": float(round(row["dollars"], 0)),
            "shares": int(row["shares"]),
            "price": float(row["price"]),
            "score": round(float(row["score"]), 4),
        })
    return data


def generate_factor_data(strategy):
    """Section 3: Factor score breakdown for held stocks."""
    factors = getattr(strategy.signal_gen, "last_factors_", {})
    if not factors:
        return {"factors": [], "stocks": {}}

    portfolio = strategy.get_current_portfolio()
    held_symbols = portfolio.index.tolist()

    active_factors = [name for name, w in strategy.signal_gen.weights.items()
                      if w > 0 and name in factors]

    data = {"factors": active_factors, "stocks": {}}
    for sym in held_symbols:
        scores = {}
        for f_name in active_factors:
            if f_name in factors and sym in factors[f_name].columns:
                val = factors[f_name][sym].iloc[-1]
                scores[f_name] = round(float(val), 3) if pd.notna(val) else None
            else:
                scores[f_name] = None
        data["stocks"][sym] = scores
    return data


def generate_backtest_data(strategy):
    """Section 2: Historical performance (5-year backtest)."""
    result = strategy.run_backtest()

    # Downsample to weekly for smaller JSON
    eq = result.equity_curve.resample("W").last().dropna()
    bm = result.benchmark_curve.resample("W").last().reindex(eq.index).dropna()

    # Align
    common_idx = eq.index.intersection(bm.index)
    eq = eq.loc[common_idx]
    bm = bm.loc[common_idx]

    # Drawdown
    peak = result.equity_curve.cummax()
    dd = ((result.equity_curve - peak) / peak).resample("W").last().reindex(common_idx)

    data = {
        "dates": [d.strftime("%Y-%m-%d") for d in common_idx],
        "equity": [round(float(v), 2) for v in eq.values],
        "benchmark": [round(float(v), 2) for v in bm.values],
        "drawdown": [round(float(v), 4) for v in dd.values],
        "metrics": {},
    }
    for k, v in result.metrics.items():
        if isinstance(v, (float, np.floating)):
            data["metrics"][k] = round(float(v), 4)
        elif isinstance(v, (int, np.integer)):
            data["metrics"][k] = int(v)
        else:
            data["metrics"][k] = str(v)
    return data


def generate_trade_history():
    """Section 4: Fetch trade history from Alpaca API.

    Falls back to local log files if Alpaca API keys are not available.
    """
    import os

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

    if api_key and secret_key:
        return _fetch_trades_from_alpaca(api_key, secret_key)

    logger.warning("No Alpaca API keys found, falling back to local logs")
    return _parse_local_trade_logs()


def _fetch_trades_from_alpaca(api_key, secret_key):
    """Pull trade history and current positions from Alpaca API."""
    try:
        import alpaca_trade_api as tradeapi
    except ImportError:
        logger.warning("alpaca-trade-api not installed, falling back to local logs")
        return _parse_local_trade_logs()

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
            ph = api.get_portfolio_history(period="1M", timeframe="1D")
            if ph and hasattr(ph, 'equity') and ph.equity:
                import datetime as dt
                for ts, eq, pl in zip(ph.timestamp, ph.equity, ph.profit_loss):
                    d = dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                    portfolio_history.append({
                        "date": d,
                        "equity": float(eq) if eq else None,
                        "profit_loss": float(pl) if pl else None,
                    })
                logger.info("Fetched %d days of portfolio history from Alpaca", len(portfolio_history))
        except Exception as e:
            logger.warning("Could not fetch portfolio history: %s", e)

        # Current positions
        positions = []
        for p in api.list_positions():
            positions.append({
                "symbol": p.symbol,
                "shares": float(p.qty),
                "avg_cost": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_pl_pct": float(p.unrealized_plpc) * 100,
            })

        # Recent orders (last 100 filled orders)
        orders = api.list_orders(status="closed", limit=200, direction="desc")
        filled_orders = [o for o in orders if o.status == "filled"]

        # Group orders by date into "rebalances"
        from collections import defaultdict
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

        return {
            "account": account_info,
            "positions": positions,
            "portfolio_history": portfolio_history,
            "rebalances": rebalances,
        }

    except Exception as e:
        logger.error("Failed to fetch from Alpaca: %s, falling back to local logs", e)
        return _parse_local_trade_logs()


def _parse_local_trade_logs():
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

    # Fallback to paper_trade_state.json
    state_file = Path("logs/paper_trade_state.json")
    if not rebalances and state_file.exists():
        state = json.loads(state_file.read_text())
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    strategy = MultiFactorStrategy(config)

    # Order matters: portfolio first populates last_factors_
    logger.info("Generating portfolio data...")
    portfolio = generate_portfolio_data(strategy, config)

    logger.info("Generating factor data...")
    factors = generate_factor_data(strategy)

    logger.info("Generating backtest data...")
    backtest = generate_backtest_data(strategy)

    logger.info("Generating trade history...")
    trades = generate_trade_history()

    # Write JSON files
    for name, data in [("portfolio", portfolio), ("backtest", backtest),
                       ("factors", factors), ("trades", trades)]:
        path = OUTPUT_DIR / f"{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("Wrote %s (%d bytes)", path, path.stat().st_size)

    logger.info("Done! Open site/index.html to view the dashboard.")


if __name__ == "__main__":
    main()
