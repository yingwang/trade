#!/usr/bin/env python3
"""Generate static dashboard data for the LightGBM strategy.

Produces JSON data files in site/lgbm/data/ that are loaded by the
LightGBM dashboard. Run daily after US market close (via the
update-site.yml workflow — do not run locally against the live account).

Outputs 6 JSON files:
  - portfolio.json   — current LightGBM portfolio recommendation
  - backtest.json    — historical backtest performance
  - factors.json     — factor score breakdown (from underlying factor inputs)
  - trades.json      — trade history from Alpaca (LGBM account)
  - feature_importance.json — LightGBM feature importance ranking
  - training_history.json   — model training metadata
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from quant.utils.config import load_config
from quant.signals.lgbm_strategy import LGBMStrategy
from site_common import fetch_trade_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("site/lgbm/data")


def generate_portfolio_data(strategy, portfolio):
    """Section 1: Current LightGBM portfolio recommendation."""
    # Reuse the prices already fetched by get_current_portfolio
    prices = strategy.last_prices_ if strategy.last_prices_ is not None else strategy.data.fetch_prices()
    returns = strategy.data.compute_returns(prices)
    spy_ret = returns.get(strategy.data.benchmark)
    regime = strategy.optimizer.detect_regime(spy_ret) if spy_ret is not None else "normal"

    total_invested = float(portfolio["weight"].sum())

    data = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "regime": regime,
        "strategy": "LightGBM",
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
            "score": round(float(row["score"]), 4) if pd.notna(row["score"]) else None,
        })
    return data


def generate_factor_data(strategy, portfolio):
    """Section 3: Factor score breakdown for held stocks (input features)."""
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
    """Section 2: Historical performance (LightGBM backtest)."""
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


def generate_feature_importance(strategy):
    """ML-specific: LightGBM feature importance ranking.

    The strategy must have been run (get_current_portfolio or run_backtest)
    so the model is trained and feature importance is available.
    """
    model = strategy.model
    feature_names = strategy._feature_names

    if model.feature_importance_ is None or feature_names is None:
        logger.warning("No feature importance available (model not trained yet)")
        return {"features": [], "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}

    imp_df = model.get_feature_importance(feature_names)

    features = []
    for _, row in imp_df.iterrows():
        features.append({
            "feature": str(row["feature"]),
            "importance": round(float(row["importance"]), 4),
            "importance_pct": round(float(row["importance_pct"]), 2),
        })

    return {
        "features": features,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def generate_training_history(strategy):
    """ML-specific: Model training metadata and history."""
    from quant.signals.lgbm_model import LGBM_AVAILABLE, SKLEARN_FALLBACK

    model = strategy.model
    data = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "backend": "lightgbm" if LGBM_AVAILABLE else ("sklearn" if SKLEARN_FALLBACK else "none"),
        "train_window": strategy.train_window,
        "val_window": strategy.val_window,
        "pred_horizon": strategy.pred_horizon,
        "retrain_every": strategy.retrain_every,
        "turnover_penalty": strategy.turnover_penalty,
        "hyperparameters": {},
        "training_runs": [],
    }

    # Extract hyperparameters from model
    if hasattr(model, 'params'):
        for k, v in model.params.items():
            if isinstance(v, (int, float, str, bool)):
                data["hyperparameters"][k] = v

    # Training history from the model (attribute is _train_history; a stale
    # check for 'training_history_' kept this panel permanently empty)
    for run in getattr(model, "_train_history", []) or []:
        entry = {}
        for k, v in run.items():
            if isinstance(v, (np.floating, np.integer)):
                entry[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, (int, float, str, bool)):
                entry[k] = v
        data["training_runs"].append(entry)

    # Number of features
    if strategy._feature_names:
        data["n_features"] = len(strategy._feature_names)

    return data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    strategy = LGBMStrategy(config)

    # Compute the current portfolio ONCE — it fetches prices/fundamentals and
    # trains the model, so the duplicate call the old script made doubled the
    # slowest part of the whole site build.
    logger.info("Computing current LightGBM portfolio...")
    current_portfolio = strategy.get_current_portfolio()

    logger.info("Generating LightGBM portfolio data...")
    portfolio = generate_portfolio_data(strategy, current_portfolio)

    logger.info("Generating factor data...")
    factors = generate_factor_data(strategy, current_portfolio)

    logger.info("Generating LightGBM backtest data...")
    backtest = generate_backtest_data(strategy)

    # After the backtest, _train_history holds the live training run plus the
    # walk-forward retrains, and feature importance reflects the latest model.
    logger.info("Generating feature importance...")
    feature_importance = generate_feature_importance(strategy)

    logger.info("Generating training history...")
    training_history = generate_training_history(strategy)

    logger.info("Generating trade history...")
    trades = fetch_trade_history(
        "ALPACA_LGBM_API_KEY", "ALPACA_LGBM_SECRET_KEY",
        "logs/paper_trade_lgbm_state.json",
    )

    # Write JSON files
    for name, data in [
        ("portfolio", portfolio),
        ("backtest", backtest),
        ("factors", factors),
        ("trades", trades),
        ("feature_importance", feature_importance),
        ("training_history", training_history),
    ]:
        path = OUTPUT_DIR / f"{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("Wrote %s (%d bytes)", path, path.stat().st_size)

    logger.info("Done! Open site/lgbm/index.html to view the LightGBM dashboard.")


if __name__ == "__main__":
    main()
