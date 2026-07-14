# Alpaca Paper Trading Setup Guide

## Prerequisites

- Python 3.12
- An Alpaca paper-trading account and paper API credentials
- This repository's hash-locked dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
```

The integration uses the maintained `alpaca-py` SDK. Do not install the legacy
`alpaca-trade-api` package alongside this lock: its old `websockets` constraint
conflicts with the current market-data stack.

## Credentials

Create keys in Alpaca's Paper Trading dashboard and export them locally:

```bash
export ALPACA_API_KEY="your-paper-api-key"
export ALPACA_SECRET_KEY="your-paper-secret-key"
```

The LightGBM account uses separate variables:

```bash
export ALPACA_LGBM_API_KEY="your-second-paper-api-key"
export ALPACA_LGBM_SECRET_KEY="your-second-paper-secret-key"
```

Never commit keys or paste them into configuration files.

## Verify and operate

```bash
python paper_trade.py --status
python paper_trade.py --dry-run --force
python paper_trade.py
python paper_trade.py --reconcile
```

Use the same flags with `paper_trade_lgbm.py` for the ML account. A normal run
rebalances only when due, while stop-losses are checked on every open-market
weekday run.

## Safety behavior

- Every order has a stable `client_order_id`; retries query Alpaca before a new
  submission.
- Rejected, unknown, and partially filled orders leave the rebalance pending so
  a later run repairs actual drift.
- Stop-loss state is deleted only after the position is confirmed closed.
- Large orders use a 30-minute TWAP schedule. Hosted jobs allow 90 minutes so a
  valid TWAP cannot be killed by the workflow timeout.
- The two trading workflows share one GitHub Actions concurrency group; a local
  lock alone cannot coordinate separate hosted runners.
- Paper mode is mandatory under the default configuration.

## BKNG split guard

Booking Holdings' 25-for-1 split became effective on 2026-04-02 and trading
started on a split-adjusted basis on 2026-04-06. Alpaca paper accounts may not
apply corporate actions. If a BKNG position still has a pre-split average entry
price and share count, the program stops before placing any order: otherwise a
target calculation could be wrong by 25×.

This guard does not prevent commits, pushes, status checks, tests, or backtests.
To resume automated paper trading, reset the affected paper account or manually
repair/close the stale BKNG paper position in Alpaca, then verify with
`--status`.

## GitHub Actions

Store credentials as repository Actions secrets. Scheduled workflows use
Python 3.12, install `requirements.lock` with hash verification, serialize live
runs, cache state, and commit audit state back to the branch.

Run the `dry-run-force` dispatch mode after configuration changes before the
next live scheduled run.
