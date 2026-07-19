# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Multi-factor quantitative equity trading system for medium-term US equities. Two strategies run live against two separate Alpaca paper accounts: a 5-factor momentum composite (paper_trade.py) and a LightGBM cross-sectional ranking model (paper_trade_lgbm.py). Both use mean-variance portfolio optimization, dynamic leverage via market regime detection, and automated execution through the Alpaca API, driven by GitHub Actions.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full backtest
python run.py backtest --start 2021-03-16 --plot
python run.py backtest-lgbm --start 2021-03-16        # LightGBM strategy
python run.py -c config_etf.yaml backtest --start 2007-01-01  # honest ETF control

# View current alpha signals
python run.py signal

# Run tests (fully offline, synthetic fixtures)
pytest tests/
pytest tests/test_factors.py -v        # single module
pytest --cov=quant tests/              # with coverage

# Paper trading (Alpaca) — same flags for paper_trade_lgbm.py
python paper_trade.py --dry-run        # preview trades
python paper_trade.py                  # execute rebalance
python paper_trade.py --status         # check status
python paper_trade.py --force          # force immediate rebalance
python paper_trade.py --reconcile      # post-trade check
```

Do NOT run generate_site.py / generate_site_lgbm.py / refresh_backtest_tables.py locally — they run on GitHub Actions (update-site.yml nightly, readme-backtest.yml on dispatch). Refresh data via workflow_dispatch instead.

## Architecture

```
run.py / paper_trade.py / paper_trade_lgbm.py     ← Entry points
        │            └── paper_trade_common.py     ← shared live-trading logic
quant/strategy.py · quant/signals/lgbm_strategy.py ← Strategy orchestrators
        │
   ┌────┼────────────┬──────────────┐
   │    │            │              │
data/   signals/   portfolio/   execution/   backtest/
```

- **`quant/strategy.py`** — Multi-factor orchestrator: `run_backtest()`, `get_current_signal()`, `get_current_portfolio(capital, prev_weights)`
- **`quant/signals/lgbm_strategy.py`** — LightGBM orchestrator, same interface plus `prev_scores`; live path hard-fails instead of falling back to constant scores
- **`quant/signals/lgbm_model.py`** — LightGBM ranking model + `purged_train_val_split` (de Prado purge/embargo — always use it for walk-forward splits, never slice windows by hand)
- **`quant/data/market_data.py`** — yfinance wrapper for prices and fundamentals
- **`quant/data/quality.py`** — `DataQualityChecker` (logging, backtest path) and `enforce_live_data_quality` (hard gate, live path — drops dead symbols, aborts on breadth collapse)
- **`quant/signals/factors.py`** — Alpha factor calculations. Factors are industry-neutralized and winsorized to ±3σ. `factor_weights` in config is the single source of truth: unlisted factors are 0, no hidden defaults
- **`quant/portfolio/optimizer.py`** — Constrained MVO with Ledoit-Wolf shrinkage. `detect_regime()` (SPY vol), `apply_vol_scaling()` (0.8x–1.8x), and `enforce_turnover_cap()` — the real 40% cap, applied to final weights including exit legs and leverage changes
- **`quant/execution/safety.py`** — `PreTradeCheck` limits; `DailyTracker` counters persist across same-day runs via the state file
- **`quant/backtest/engine.py`** — Event-driven daily simulator: T+1 close execution, daily stop-loss, margin interest, Almgren-Chriss market impact, Sharpe/Sortino vs configured risk-free rate
- **`paper_trade_common.py`** — All live-trading logic (market-closed gate, daily stop-loss check, kill-switch, entry-price semantics, state/lock handling). The two entry scripts are thin wrappers whose module-level names (STATE_FILE, ExecutionLogger, ...) exist for tests to patch
- **`config.yaml`** — All strategy parameters; `config_etf.yaml` — survivorship-free ETF control universe for honest-backtest

## Signal Pipeline

Raw prices → live quality gate → factor scores (momentum 50%, high_proximity 20%, reversal 10%, vol_contraction 10%, trend_persistence 10%) → industry-neutral z-scores → trend filter (< 200d SMA → 0.5x) → blowoff filter (z > 4.0 → 0.5x) → composite score → MVO optimization → vol scaling → total turnover cap → target weights

## Key Constraints in config.yaml

- Max 12 positions, 3%–12% per stock, max 50% per sector
- Rebalance every 21 trading days; 40% cap on TOTAL turnover (exit legs and vol-scaling included; excess blends toward the previous portfolio)
- Transaction cost: 10 bps + Almgren-Chriss market impact; Sharpe/Sortino vs 4% risk-free rate
- Quality and value factors are **disabled** — yfinance only provides current snapshots, causing look-ahead bias in backtests

## Live Operation (GitHub Actions)

- `rebalance.yml` (cron 0 15 UTC) and `rebalance-lgbm.yml` (cron 10 15 UTC, staggered to avoid push races) run every weekday; state is cached AND committed to main `[skip ci]`, with `git pull --rebase` before push
- Market-closed days exit before submitting anything; stop-losses are checked on every daily run, not only rebalance days
- Entry prices back the stop-loss and are recorded only when a position is newly established (adds don't reset the base) — same semantics as the backtest engine
- The LGBM account persists `prev_scores` in its state file so the score-level turnover penalty is active across runs
- `update-site.yml` regenerates dashboards nightly onto gh-pages; site/data on main is gitignored (never commit generated snapshots)
- `honest-backtest.yml` / `readme-backtest.yml` are manual, read-only backtest runners for the ETF control and README tables

## Testing

Tests use synthetic fixtures (3 years of 10 stocks + benchmark) defined in `tests/conftest.py` — fully offline, no API calls needed. Test modules cover factors, portfolio optimization (incl. the turnover cap), backtesting, broker, safety checks, purged splits, live quality gate, and paper-trade operational paths.

## Deployment Notes

- Logs go to `logs/paper_trade_YYYYMMDD.log`; state tracked in `logs/paper_trade_state.json` / `logs/paper_trade_lgbm_state.json` (committed to main by the workflows)
- `logs/paper_trade.lock` prevents concurrent execution (auto-expires after 1 hour)
- Alpaca paper accounts do not process corporate actions — `site_common.STOCK_SPLITS` carries manual split corrections with entry-price-scale guards

## Known Limitations

- **Survivorship bias**: Static 100-stock universe excludes delisted companies; quantify with the honest-backtest ETF control
- **T+1 close execution**: Backtest signals at T close, fills at T+1 close; intraday fill prices still differ in reality
- **No point-in-time fundamentals**: Why quality/value factors are disabled
- **Paper fills**: Alpaca paper fills carry no real spread/impact; live costs will be higher
