#!/usr/bin/env python3
"""Paper trading entry point for the earnings book (fourth account).

All logic lives in paper_trade_common.py, shared with the other books. What
differs here: the config file (config_earnings.yaml), the account
(ALPACA_EARNINGS_API_KEY / ALPACA_EARNINGS_SECRET_KEY), and the absence of any
daily action: the book has no stops and no regime lights (see
quant/earnings_strategy.py), so a run on a day without a scheduled rebalance
only records the account and exits. The scheduled rebalance is every five
sessions on the shared anchored calendar; it pulls the S&P 500 membership
table and Yahoo's earnings calendar for the whole universe (about five
minutes) before ranking.

Usage:
    export ALPACA_EARNINGS_API_KEY="your-paper-key"
    export ALPACA_EARNINGS_SECRET_KEY="your-paper-secret"

    python paper_trade_earnings.py --dry-run     # preview
    python paper_trade_earnings.py               # trade
    python paper_trade_earnings.py --status      # account status
"""

from pathlib import Path

from quant.execution.safety import ExecutionLogger  # patched in tests

import paper_trade_common as common
from paper_trade_common import TradeProfile

STATE_FILE = Path("logs/paper_trade_earnings_state.json")
LOCK_FILE = Path("logs/paper_trade_earnings.lock")


def _strategy_factory(config):
    from quant.earnings_strategy import EarningsStrategy
    return EarningsStrategy(config)


PROFILE = TradeProfile(
    name="earnings",
    description="Earnings-book paper trading with Alpaca (fourth account)",
    status_banner="EARNINGS PAPER TRADING STATUS",
    portfolio_banner="EARNINGS TARGET PORTFOLIO",
    state_file=STATE_FILE,
    lock_file=LOCK_FILE,
    log_prefix="paper_trade_earnings",
    strategy_factory=_strategy_factory,
    api_key_env="ALPACA_EARNINGS_API_KEY",
    secret_key_env="ALPACA_EARNINGS_SECRET_KEY",
    config_file="config_earnings.yaml",
    events_log="logs/trade_events_earnings.jsonl",
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
                  order_result_callback=None, portfolio_context=None):
    return common.run_rebalance(
        strategy, broker, config, dry_run=dry_run,
        banner=PROFILE.portfolio_banner,
        exec_logger_cls=ExecutionLogger,
        prev_scores=prev_scores,
        order_result_callback=order_result_callback,
        events_log=PROFILE.events_log,
        portfolio_context=portfolio_context,
    )


def main():
    common.run_main(PROFILE)


if __name__ == "__main__":
    main()
