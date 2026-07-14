#!/usr/bin/env python3
"""Generate static dashboard data for GitHub Pages.

Produces JSON data files in site/data/ that are loaded by the
static HTML dashboard. Run daily after US market close (via the
update-site.yml workflow — do not run locally against the live account).
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from quant.utils.config import load_config
from quant.strategy import MultiFactorStrategy
from site_common import fetch_trade_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("site/data")
MARKET_TZ = ZoneInfo("America/New_York")


def generate_portfolio_data(strategy, portfolio, account_equity=None):
    """Section 1: Current portfolio recommendation."""
    # Reuse the prices already fetched by get_current_portfolio
    prices = strategy.last_prices_ if strategy.last_prices_ is not None else strategy.data.fetch_prices()
    returns = strategy.data.compute_returns(prices)
    spy_ret = returns.get(strategy.data.benchmark)
    regime = strategy.optimizer.detect_regime(spy_ret) if spy_ret is not None else "normal"

    total_invested = float(portfolio["weight"].sum())

    data = {
        "updated_at": datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M %Z"),
        "regime": regime,
        "total_invested_pct": round(total_invested * 100, 1),
        "cash_pct": round(max(0, (1 - total_invested)) * 100, 1),
        "account_equity_basis": (
            round(float(account_equity), 2) if account_equity is not None else None
        ),
        "positions": [],
    }
    for symbol, row in portfolio.iterrows():
        data["positions"].append({
            "symbol": symbol,
            "weight_pct": float(row["weight_pct"]),
            "dollars": float(round(row["dollars"], 0)),
            "shares": int(row["shares"]),
            "price": float(row["price"]),
            "score": round(float(row["score"]), 4) if pd.notna(row["score"]) else None,
        })
    return data


def generate_factor_data(strategy, portfolio):
    """Section 3: Factor score breakdown for held stocks."""
    factors = getattr(strategy.signal_gen, "last_factors_", {})
    if not factors:
        return {"factors": [], "stocks": {}}

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
            data["metrics"][k] = round(float(v), 4) if np.isfinite(v) else None
        elif isinstance(v, (int, np.integer)):
            data["metrics"][k] = int(v)
        else:
            data["metrics"][k] = str(v)
    return data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    strategy = MultiFactorStrategy(config)

    # Pull the account first: target dollars and turnover must be based on the
    # actual paper equity/holdings, not the backtest's configured $1m seed.
    logger.info("Fetching trade history and current account...")
    trades = fetch_trade_history(
        "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "logs/paper_trade_state.json"
    )
    trades["annual_risk_free_rate"] = float(
        config.get("backtest", {}).get("risk_free_rate", 0.0)
    )
    account_equity = trades.get("account", {}).get("equity")
    if os.environ.get("ALPACA_API_KEY") and account_equity is None:
        raise RuntimeError(
            "Alpaca account fetch failed; refusing to publish a fictional $1m target"
        )
    capital = float(
        config["backtest"]["initial_capital"]
        if account_equity is None else account_equity
    )
    if capital <= 0:
        raise RuntimeError("Paper account equity is non-positive; refusing to publish a target")
    current_positions = trades.get("positions", [])
    prev_weights = pd.Series(
        {
            p["symbol"]: float(p.get("market_value", 0)) / capital
            for p in current_positions
            if capital > 0 and float(p.get("market_value", 0)) != 0
        },
        dtype=float,
    )

    # Compute the recommendation once and share it between sections.
    logger.info("Computing current portfolio...")
    current_portfolio = strategy.get_current_portfolio(
        capital=capital,
        prev_weights=prev_weights,
    )

    logger.info("Generating portfolio data...")
    portfolio = generate_portfolio_data(
        strategy, current_portfolio, account_equity=capital
    )

    logger.info("Generating factor data...")
    factors = generate_factor_data(strategy, current_portfolio)

    logger.info("Generating backtest data...")
    backtest = generate_backtest_data(strategy)

    # Write JSON files
    for name, data in [("portfolio", portfolio), ("backtest", backtest),
                       ("factors", factors), ("trades", trades)]:
        path = OUTPUT_DIR / f"{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("Wrote %s (%d bytes)", path, path.stat().st_size)

    logger.info("Done! Open site/index.html to view the dashboard.")


if __name__ == "__main__":
    main()
