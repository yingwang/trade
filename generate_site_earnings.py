#!/usr/bin/env python3
"""Generate static dashboard data for the earnings book (the trend book's
paper account since 2026-10-23).

Produces JSON data files in site/earnings/data/ that the earnings dashboard
loads. Runs nightly in update-site.yml, never locally against the live
account. The current target is computed the way the runner computes it,
which pulls the S&P 500 membership table and Yahoo's earnings calendar for
the whole universe (about five minutes).

Outputs:
  - portfolio.json   — current target book with each name's composite score
  - backtest.json    — backtest curve and metrics since 2016
  - trades.json      — trade history, positions and equity history from Alpaca
  - attribution.json — actual alpha reconciled to market/style/sector effects
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
from quant.earnings_strategy import EarningsStrategy
from site_common import fetch_trade_history, generate_actual_attribution

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("site/earnings/data")
MARKET_TZ = ZoneInfo("America/New_York")


def generate_portfolio_data(strategy, portfolio, account_equity=None):
    decision = strategy.last_decision_ or {}
    total_invested = float(portfolio["weight"].sum())
    data = {
        "updated_at": datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M %Z"),
        "gross_pct": round(total_invested * 100, 1),
        "total_invested_pct": round(total_invested * 100, 1),
        "cash_pct": round((1 - total_invested) * 100, 1),
        "account_equity_basis": round(float(account_equity), 2) if account_equity is not None else None,
        "candidates": decision.get("candidates"),
        "universe_size": len(strategy.data.symbols),
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


def generate_backtest_data(strategy):
    result = strategy.run_backtest()
    eq = result.equity_curve.resample("W").last().dropna()
    bm = result.benchmark_curve.resample("W").last().reindex(eq.index).dropna()
    common_idx = eq.index.intersection(bm.index)
    eq = eq.loc[common_idx]
    bm = bm.loc[common_idx]
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
    config = load_config("config_earnings.yaml")
    strategy = EarningsStrategy(config)

    logger.info("Fetching trade history and current account...")
    # The account traded the trend book until 2026-10-22; the page starts at
    # the takeover (dashboard.account_since in config_earnings.yaml).
    trades = fetch_trade_history(
        "ALPACA_EARNINGS_API_KEY",
        "ALPACA_EARNINGS_SECRET_KEY",
        "logs/paper_trade_earnings_state.json",
        split_cash_compensations=(
            config.get("dashboard", {}).get("split_cash_compensations", {}).get("earnings", {})
        ),
        since=config.get("dashboard", {}).get("account_since"),
    )
    if trades.get("source") == "local":
        # Without account keys, refuse rather than publishing another book's history.
        raise RuntimeError(
            "No Alpaca keys for the earnings book (ALPACA_EARNINGS_API_KEY / "
            "ALPACA_EARNINGS_SECRET_KEY); refusing to publish local logs as its trade history"
        )
    trades["annual_risk_free_rate"] = float(config.get("backtest", {}).get("risk_free_rate", 0.0))
    account_equity = trades.get("account", {}).get("equity")
    if os.environ.get("ALPACA_EARNINGS_API_KEY") and account_equity is None:
        raise RuntimeError("Alpaca account fetch failed; refusing to publish a fictional target")
    capital = float(config["backtest"]["initial_capital"] if account_equity is None else account_equity)
    if capital <= 0:
        raise RuntimeError("Paper account equity is non-positive; refusing to publish a target")
    prev_weights = pd.Series(
        {
            p["symbol"]: float(p.get("market_value", 0)) / capital
            for p in trades.get("positions", [])
            if float(p.get("market_value", 0)) != 0
        },
        dtype=float,
    )

    logger.info("Computing current target book...")
    current_portfolio = strategy.get_current_portfolio(capital=capital, prev_weights=prev_weights)
    portfolio = generate_portfolio_data(strategy, current_portfolio, account_equity=capital)

    logger.info("Generating actual alpha attribution...")
    attribution = generate_actual_attribution(
        trades, strategy.last_prices_, None, config, benchmark=strategy.data.benchmark,
    )

    logger.info("Generating backtest data...")
    # The backtest fetches its own prices for the point-in-time universe; the
    # announcement history fetched for the target above is reused.
    backtest = generate_backtest_data(strategy)

    for name, data in [
        ("portfolio", portfolio),
        ("backtest", backtest),
        ("trades", trades),
        ("attribution", attribution),
    ]:
        path = OUTPUT_DIR / f"{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("Wrote %s (%d bytes)", path, path.stat().st_size)


if __name__ == "__main__":
    main()
