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
    """Section 4: Parse trade logs."""
    events_file = Path("logs/trade_events.jsonl")
    rebalances = []

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

    # Also parse paper_trade_state.json as fallback
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
