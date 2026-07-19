#!/usr/bin/env python3
"""Paper trading entry point (LightGBM strategy) using the Alpaca API.

All logic lives in paper_trade_common.py, shared with paper_trade.py.
This module keeps thin wrappers around the shared implementation because
tests and operators patch these module-level names.

Uses a separate Alpaca paper account via ALPACA_LGBM_API_KEY /
ALPACA_LGBM_SECRET_KEY, and persists the score cross-section between runs
so the score-level turnover penalty is active in live trading.

Usage:
    # One-time setup:
    python3.12 -m pip install --require-hashes -r requirements.lock

    export ALPACA_LGBM_API_KEY="your-paper-key"
    export ALPACA_LGBM_SECRET_KEY="your-paper-secret"

    # Run once to see what trades would be made:
    python paper_trade_lgbm.py --dry-run

    # Execute the rebalance:
    python paper_trade_lgbm.py

    # Check current status:
    python paper_trade_lgbm.py --status

    # Run post-trade reconciliation only:
    python paper_trade_lgbm.py --reconcile
"""

from pathlib import Path

from quant.execution.safety import ExecutionLogger  # patched in tests

import paper_trade_common as common
from paper_trade_common import TradeProfile

STATE_FILE = Path("logs/paper_trade_lgbm_state.json")
LOCK_FILE = Path("logs/paper_trade_lgbm.lock")


def _strategy_factory(config):
    from quant.signals.lgbm_strategy import LGBMStrategy
    return LGBMStrategy(config)


PROFILE = TradeProfile(
    name="lgbm",
    description="LightGBM paper trading with Alpaca",
    status_banner="LGBM PAPER TRADING STATUS",
    portfolio_banner="LGBM TARGET PORTFOLIO",
    state_file=STATE_FILE,
    lock_file=LOCK_FILE,
    log_prefix="paper_trade_lgbm",
    strategy_factory=_strategy_factory,
    api_key_env="ALPACA_LGBM_API_KEY",
    secret_key_env="ALPACA_LGBM_SECRET_KEY",
    persist_scores=True,
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


def run_rebalance(
    strategy,
    broker,
    config,
    dry_run=False,
    prev_scores=None,
    order_result_callback=None,
):
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
