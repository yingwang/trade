#!/usr/bin/env python3
"""Generate static dashboard data for the LightGBM strategy.

Produces JSON data files in site/lgbm/data/ that are loaded by the
LightGBM dashboard. Run daily after US market close.

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
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from quant.utils.config import load_config
from quant.signals.lgbm_strategy import LGBMStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("site/lgbm/data")


def generate_portfolio_data(strategy, config):
    """Section 1: Current LightGBM portfolio recommendation."""
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
            "score": round(float(row["score"]), 4),
        })
    return data


def generate_factor_data(strategy):
    """Section 3: Factor score breakdown for held stocks (input features)."""
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


def generate_trade_history():
    """Section 4: Fetch trade history from Alpaca API (LGBM account).

    Falls back to local log files if Alpaca API keys are not available.
    """
    api_key = os.environ.get("ALPACA_LGBM_API_KEY", "")
    secret_key = os.environ.get("ALPACA_LGBM_SECRET_KEY", "")

    if api_key and secret_key:
        return _fetch_trades_from_alpaca(api_key, secret_key)

    logger.warning("No Alpaca LGBM API keys found, falling back to local logs")
    return _parse_local_trade_logs()


# Stock splits not yet reflected in Alpaca's avg_entry_price.
# Map symbol -> (split_ratio, effective_date) so we can adjust.
STOCK_SPLITS = {
    "BKNG": 25,  # 1:25 split, June 2024
}


def _adjust_for_splits(symbol, qty, avg_entry_price, cost_basis):
    """Adjust position data for stock splits Alpaca hasn't accounted for."""
    ratio = STOCK_SPLITS.get(symbol)
    if ratio and avg_entry_price > 0:
        # Heuristic: if entry price is ~ratio× current market price, it's unadjusted
        adjusted_entry = avg_entry_price / ratio
        adjusted_qty = qty * ratio
        adjusted_cost = cost_basis  # total cost doesn't change
        return adjusted_qty, adjusted_entry, adjusted_cost
    return qty, avg_entry_price, cost_basis


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
            qty = float(p.qty)
            avg_entry = float(p.avg_entry_price)
            cost = float(p.cost_basis)
            current = float(p.current_price)

            # Adjust for stock splits Alpaca hasn't reflected
            qty, avg_entry, cost = _adjust_for_splits(p.symbol, qty, avg_entry, cost)

            # Recompute P/L from adjusted values
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

        # Adjust account equity and portfolio history for split corrections.
        # Alpaca paper trading doesn't process corporate actions, so equity
        # and history include wrong market values for split-affected stocks.
        equity_adjustment = 0
        for p in api.list_positions():
            sym = p.symbol
            if sym in STOCK_SPLITS:
                ratio = STOCK_SPLITS[sym]
                raw_qty = float(p.qty)
                current = float(p.current_price)
                equity_adjustment += raw_qty * (ratio - 1) * current

        if equity_adjustment:
            # Recompute equity = cash + sum of corrected market values
            total_mv = sum(p["market_value"] for p in positions)
            account_info["equity"] = round(account_info["cash"] + total_mv, 2)
            logger.info("Adjusted account equity by +$%.2f for stock splits", equity_adjustment)
            # Find earliest buy date of any split-affected stock
            split_buy_date = None
            for reb in rebalances:
                for t in reb.get("trades", []):
                    if t["symbol"] in STOCK_SPLITS and t["side"] == "buy":
                        d = reb["date"]
                        if split_buy_date is None or d < split_buy_date:
                            split_buy_date = d
            # Adjust portfolio history only for dates AFTER the buy date
            if split_buy_date:
                for h in portfolio_history:
                    if h["equity"] is not None and h["date"] > split_buy_date:
                        h["equity"] = round(h["equity"] + equity_adjustment, 2)

        # Fetch SPY benchmark aligned to portfolio history dates
        spy_history = []
        valid_hist = [h for h in portfolio_history if h["equity"]]
        if valid_hist:
            try:
                import yfinance as yf
                from datetime import timedelta
                start_date = valid_hist[0]["date"]
                # yfinance end is exclusive, add 2 days buffer
                end_dt = datetime.strptime(valid_hist[-1]["date"], "%Y-%m-%d") + timedelta(days=2)
                spy = yf.download("SPY", start=start_date, end=end_dt.strftime("%Y-%m-%d"), progress=False)
                if not spy.empty:
                    if hasattr(spy.columns, 'levels'):
                        spy.columns = spy.columns.get_level_values(0)
                    # Build date->close lookup, forward-fill for non-trading days
                    spy_by_date = {}
                    last_close = None
                    for idx_date, row in spy.iterrows():
                        last_close = float(row["Close"])
                        spy_by_date[idx_date.strftime("%Y-%m-%d")] = last_close
                    start_equity = valid_hist[0]["equity"]
                    first_spy = spy_by_date.get(valid_hist[0]["date"])
                    if first_spy is None:
                        first_spy = float(spy["Close"].iloc[0])
                    # Emit one SPY point per portfolio history date
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

        return {
            "account": account_info,
            "positions": positions,
            "portfolio_history": portfolio_history,
            "spy_history": spy_history,
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

    # Fallback to paper_trade_lgbm_state.json
    state_file = Path("logs/paper_trade_lgbm_state.json")
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


def generate_feature_importance(strategy):
    """ML-specific: LightGBM feature importance ranking.

    Returns a JSON-serializable dict with feature importance data.
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
    """ML-specific: Model training metadata and history.

    Returns training info including hyperparameters, training metrics,
    and the LightGBM backend being used.
    """
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

    # Training history from model if available
    if hasattr(model, 'training_history_') and model.training_history_:
        for run in model.training_history_:
            entry = {}
            for k, v in run.items():
                if isinstance(v, (int, float, str, bool)):
                    entry[k] = v
                elif isinstance(v, (np.floating, np.integer)):
                    entry[k] = float(v) if isinstance(v, np.floating) else int(v)
            data["training_runs"].append(entry)

    # Number of features
    if strategy._feature_names:
        data["n_features"] = len(strategy._feature_names)

    return data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    strategy = LGBMStrategy(config)

    # Order matters: portfolio first populates model and feature importance
    logger.info("Generating LightGBM portfolio data...")
    portfolio = generate_portfolio_data(strategy, config)

    logger.info("Generating factor data...")
    factors = generate_factor_data(strategy)

    logger.info("Generating LightGBM backtest data...")
    backtest = generate_backtest_data(strategy)

    logger.info("Generating trade history...")
    trades = generate_trade_history()

    logger.info("Generating feature importance...")
    feature_importance = generate_feature_importance(strategy)

    logger.info("Generating training history...")
    training_history = generate_training_history(strategy)

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
