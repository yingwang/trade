#!/usr/bin/env python3
"""Generate static dashboard data for the trend strategy (third account).

Produces JSON data files in site/trend/data/ that the trend dashboard loads.
Runs nightly in update-site.yml, never locally against the live account.

Outputs:
  - portfolio.json   — current target book plus the regime lights it was built under
  - backtest.json    — five-year backtest curve and metrics
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
from quant.trend_strategy import TrendStrategy
from site_common import fetch_trade_history, generate_actual_attribution

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("site/trend/data")
MARKET_TZ = ZoneInfo("America/New_York")


def generate_portfolio_data(strategy, portfolio, account_equity=None):
    decision = strategy.last_decision_ or {}
    lights = decision.get("regime", {})
    on = [k for k in ("benchmark_below_sma", "benchmark_vol_high", "breadth_weak") if lights.get(k)]
    total_invested = float(portfolio["weight"].sum())
    data = {
        "updated_at": datetime.now(MARKET_TZ).strftime("%Y-%m-%d %H:%M %Z"),
        "regime": {"lights_on": on, "count": len(on), "breadth": lights.get("breadth")},
        "regime_cap": decision.get("regime_cap"),
        "gross_pct": round(total_invested * 100, 1),
        "total_invested_pct": round(total_invested * 100, 1),
        "cash_pct": round((1 - total_invested) * 100, 1),
        "account_equity_basis": round(float(account_equity), 2) if account_equity is not None else None,
        "candidates": decision.get("candidates"),
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
    config = load_config("config_trend.yaml")
    strategy = TrendStrategy(config)

    logger.info("Fetching trade history and current account...")
    # Prefer ALPACA_TREND_*; accept legacy ALPACA_CLAUDE_* until secrets are renamed.
    trend_key_env = (
        "ALPACA_TREND_API_KEY"
        if os.environ.get("ALPACA_TREND_API_KEY")
        else "ALPACA_CLAUDE_API_KEY"
    )
    trend_secret_env = (
        "ALPACA_TREND_SECRET_KEY"
        if os.environ.get("ALPACA_TREND_SECRET_KEY")
        else "ALPACA_CLAUDE_SECRET_KEY"
    )
    trades = fetch_trade_history(
        trend_key_env,
        trend_secret_env,
        "logs/paper_trade_trend_state.json",
        split_cash_compensations=(
            config.get("dashboard", {}).get("split_cash_compensations", {}).get("trend", {})
        ),
    )
    if trades.get("source") == "local":
        # Without account keys, refuse rather than publishing another book's history.
        raise RuntimeError(
            "No Alpaca keys for the trend book (ALPACA_TREND_API_KEY / "
            "ALPACA_TREND_SECRET_KEY, or legacy ALPACA_CLAUDE_*); refusing to "
            "publish local logs as its trade history"
        )
    trades["annual_risk_free_rate"] = float(config.get("backtest", {}).get("risk_free_rate", 0.0))
    account_equity = trades.get("account", {}).get("equity")
    if (os.environ.get("ALPACA_TREND_API_KEY") or os.environ.get("ALPACA_CLAUDE_API_KEY")) and account_equity is None:
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
