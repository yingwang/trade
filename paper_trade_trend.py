#!/usr/bin/env python3
"""Paper trading entry point for the trend strategy (third account).

All logic lives in paper_trade_common.py, shared with the two factor books. The
differences are the config file (config_trend.yaml), the account
(ALPACA_CLAUDE_API_KEY / ALPACA_CLAUDE_SECRET_KEY) and a rebalance trigger: on
days without a scheduled rebalance the strategy is still asked whether a trailing
stop or a regime cut needs acting on, and if so that action is executed today.

Usage:
    export ALPACA_CLAUDE_API_KEY="your-paper-key"
    export ALPACA_CLAUDE_SECRET_KEY="your-paper-secret"

    python paper_trade_trend.py --dry-run     # preview
    python paper_trade_trend.py               # trade
    python paper_trade_trend.py --status      # account status
"""

from pathlib import Path

from quant.execution.safety import ExecutionLogger  # patched in tests

import paper_trade_common as common
from paper_trade_common import TradeProfile

STATE_FILE = Path("logs/paper_trade_trend_state.json")
LOCK_FILE = Path("logs/paper_trade_trend.lock")


def _strategy_factory(config):
    from quant.trend_strategy import TrendStrategy
    return TrendStrategy(config)


def _entry_dates(state: dict) -> dict:
    """Symbol to the day its position was opened, for the trailing stop's
    peak-since-entry. Positions recorded before entry dates were kept are
    backfilled from the fill history with their earliest buy."""
    dates = dict(state.get("entry_dates", {}))
    missing = [s for s in state.get("entry_prices", {}) if s not in dates]
    if missing:
        first_buy: dict = {}
        for entry in state.get("trade_history", []):
            for trade in entry.get("trades", []):
                if trade.get("side") != "buy" or trade.get("status") not in (
                    "filled", "partial_fill", "partial_fill_open",
                ):
                    continue
                day = str(trade.get("time") or entry.get("date") or "")[:10]
                if day and trade.get("symbol") not in first_buy:
                    first_buy[trade["symbol"]] = day
        for sym in missing:
            if sym in first_buy:
                dates[sym] = first_buy[sym]
    return dates


def _rebalance_trigger(strategy, broker, state):
    """A reason when today's book needs an exit or a de-risking; else None."""
    prev_weights = common.current_broker_weights(broker)
    return strategy.maintenance_needed(prev_weights, _entry_dates(state))


PROFILE = TradeProfile(
    name="trend",
    description="Trend-following paper trading with Alpaca (third account)",
    status_banner="TREND PAPER TRADING STATUS",
    portfolio_banner="TREND TARGET PORTFOLIO",
    state_file=STATE_FILE,
    lock_file=LOCK_FILE,
    log_prefix="paper_trade_trend",
    strategy_factory=_strategy_factory,
    api_key_env="ALPACA_CLAUDE_API_KEY",
    secret_key_env="ALPACA_CLAUDE_SECRET_KEY",
    config_file="config_trend.yaml",
    rebalance_trigger=_rebalance_trigger,
)


def load_state() -> dict:
    return common.load_state(STATE_FILE)


def save_state(state: dict):
    common.save_state(STATE_FILE, state)


def acquire_lock() -> bool:
    return common.acquire_lock(LOCK_FILE)


def release_lock():
    common.release_lock(LOCK_FILE)


should_rebalance = common.should_rebalance
check_stop_losses = common.check_stop_losses


def show_status(broker):
    common.show_status(broker, PROFILE.status_banner)


def run_rebalance(strategy, broker, config, dry_run=False, prev_scores=None,
                  order_result_callback=None):
    return common.run_rebalance(
        strategy, broker, config, dry_run=dry_run,
        banner=PROFILE.portfolio_banner,
        exec_logger_cls=ExecutionLogger,
        prev_scores=prev_scores,
        order_result_callback=order_result_callback,
    )


def main():
    common.run_main(PROFILE)


if __name__ == "__main__":
    main()
