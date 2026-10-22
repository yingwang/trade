# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Multi-factor quantitative equity trading system for medium-term US equities. Three strategies run live against three separate Alpaca paper accounts: a 5-factor momentum composite (paper_trade.py), a LightGBM cross-sectional ranking model (paper_trade_lgbm.py), and a leveraged trend-following book (paper_trade_trend.py, `quant/trend_strategy.py`, `config_trend.yaml`) that rebalances every ten sessions and acts on trailing stops and regime lights daily through the runner's `rebalance_trigger`. The two factor books allocate score-proportional weights (then vol scaling, the total turnover cap, sector and position limits), use dynamic leverage via market regime detection, and automated execution through the Alpaca API, driven by GitHub Actions; a constrained mean-variance optimizer is available via `portfolio.allocation: mvo` as a research option.

## Commands

```bash
# Install dependencies
python3.12 -m pip install --require-hashes -r requirements.lock

# Run full backtest
python run.py backtest --start 2021-03-16 --plot
python run.py backtest-lgbm --start 2021-03-16        # LightGBM strategy
python run.py -c config_trend.yaml backtest-trend --start 2021-09-16 --yearly   # trend book (or the Trend Backtest workflow)
python run.py -c config_etf.yaml backtest --start 2007-01-01  # ETF robustness control

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
- **`quant/signals/lgbm_model.py`** — Date-grouped LambdaRank model + `purged_train_val_split`, missingness-aware inputs, Rank IC/ICIR and baseline-skill diagnostics
- **`quant/data/market_data.py`** — yfinance wrapper for prices, same-download opens/volumes, and fundamentals
- **`quant/data/quality.py`** — `DataQualityChecker` (logging, backtest path) and `enforce_live_data_quality` (hard gate, live path — drops dead symbols, aborts on breadth collapse)
- **`quant/signals/factors.py`** — Alpha factor calculations. Factors are industry-neutralized and winsorized to ±3σ. `factor_weights` in config is the single source of truth: unlisted factors are 0, no hidden defaults
- **`quant/portfolio/optimizer.py`** — Portfolio construction. `optimize_weights()` builds score-proportional weights by default (`portfolio.allocation: proportional`, what the live books run); `portfolio.allocation: mvo` runs the constrained mean-variance optimizer instead (Ledoit-Wolf covariance, smooth turnover penalty, SLSQP with analytic gradients, outcome in `last_solve_status`), a research option that backtested below proportional (multi-factor 5y +127.5% MVO vs +151.8% proportional, windows ending 2026-10-22). `detect_regime()` (SPY vol), `apply_vol_scaling()` (0.8x–1.8x), and `enforce_turnover_cap()` — the real 40% cap, applied to final weights including exit legs and leverage changes, the same way for either allocation
- **`quant/execution/safety.py`** — `PreTradeCheck` limits; `DailyTracker` counters persist across same-day runs via the state file
- **`quant/backtest/engine.py`** — Event-driven daily simulator: T-close signal / next-session-open execution, pending missing-bar orders, next-bar stop-losses, delisting returns, margin interest, market impact, and corrected risk metrics
- **`quant/attribution.py`** — Actual-account attribution against SPY. Reconstructs holdings from fills; uses strictly lagged returns-based ETF factors for market/style/industry, keeps selection/cash/implementation-shortfall explicit, exposes reconciliation residual, and Carino-links terminal contributions
- **`paper_trade_common.py`** — All live-trading logic (market-closed gate, daily stop-loss check, kill-switch, entry-price semantics, state/lock handling). The two entry scripts are thin wrappers whose module-level names (STATE_FILE, ExecutionLogger, ...) exist for tests to patch
- **`config.yaml`** — All strategy parameters; `config_etf.yaml` — ETF robustness-control universe (not a survivorship-bias estimate)

## Signal Pipeline

Raw prices → live quality gate → factor scores (momentum 50%, high_proximity 20%, reversal 10%, vol_contraction 10%, trend_persistence 10%) → industry-neutral z-scores → trend filter (< 200d SMA → 0.5x) → blowoff filter (z > 4.0 → 0.5x) → composite score → score-proportional allocation (or MVO when `portfolio.allocation: mvo`) → vol scaling → total turnover cap → target weights

## Key Constraints in config.yaml

- Target portfolio: 12 positions, 3%–12% per stock, max 50% per sector. The held book can carry more names in transition: exits the turnover blend would leave below 3% are sold in full within the budget, larger exits are sold down over successive rebalances
- `portfolio.allocation`: `proportional` (default, live) builds the weights from the scores; `mvo` is the research-only mean-variance path. Vol scaling, the turnover cap and the sector/position limits are applied the same way after either; `alpha_scale`, `risk_aversion` and `turnover_penalty_epsilon` only matter for `mvo`
- `universe.sectors` is the static sector table used by both the backtest and the live path (industry neutralization, sector caps, attribution); the yfinance `.info` snapshot is only a fallback for configs without one
- Rebalance every 21 trading days; 40% cap on TOTAL turnover (exit legs and vol-scaling included). When it binds, the budget is split between sells and buys first, so a capped rebalance slows down but never sells gross exposure below min(current, target)
- Transaction cost: 10 bps + per-symbol volume-participation Almgren-Chriss market impact (turnover fallback when volume is missing); Sharpe/Sortino vs 4% risk-free rate
- Quality and value factors are **disabled** — yfinance only provides current snapshots, causing look-ahead bias in backtests

## Live Operation (GitHub Actions)

- `rebalance.yml`, `rebalance-lgbm.yml` and `rebalance-trend.yml` are scheduled five times each weekday at 09:37/10:37/11:37/12:37/14:37 UTC (+10 and +20 min for the other books): GitHub's scheduler ran them 3.7 to 8.8 hours late in October 2026, so the early slots count on that delay to land in session and the last is in session even on time; every attempt is idempotent. A run that stood down stays green but is flagged with a warning annotation and a job-summary entry (the owner asked for no failure mail on 2026-10-22, so never make a stand-down fail the job): the Python prints `FULL_DAY_MARKET_CLOSED` (five closed-market attempts in a day) or `STAND_DOWN:<reason>` (drawdown gate, open orders, failed trend trigger, daily loss limit, missing quotes or keys) and the workflow greps `run.log` for them; a traceback still fails the step, and a quiet open day ("not time to rebalance yet") is silent. Check the run list for the warnings in the morning. Each book has its own concurrency group (a shared group let GitHub cancel a pending run). State is committed to main `[skip ci]` (the commit step runs even when the trading step failed), with `git pull --rebase` and a retried push; there is no Actions cache any more, git is the record (a restored cache overwrote hand-fixed state twice)
- Market-closed days exit before submitting anything; stop-losses are checked on every daily run, not only rebalance days
- Covered stop-loss/emergency sells bypass entry-oriented size, daily-turnover, penny-stock, and ADV throttles; oversized sells remain blocked
- Entry prices back the stop-loss and are recorded only when a position is newly established (adds don't reset the base) — same semantics as the backtest engine
- The LGBM account persists `prev_scores` in its state file so the score-level turnover penalty is active across runs
- `update-site.yml` regenerates dashboards nightly onto gh-pages; site/data on main is gitignored (never commit generated snapshots)
- Both dashboards publish `attribution.json`: actual equity is the truth set, paper cash earns 0%, and any unexplained P&L stays visible as a reconciliation residual rather than being folded into stock selection
- `honest-backtest.yml` / `readme-backtest.yml` are manual, read-only backtest runners for the ETF control and README tables

## Testing

Tests use synthetic fixtures (3 years of 10 stocks + benchmark) defined in `tests/conftest.py` — fully offline, no API calls needed. Test modules cover factors, portfolio optimization (incl. the turnover cap), backtesting, broker, safety checks, purged splits, live quality gate, and paper-trade operational paths. Three modules check the system rather than one helper: `tests/test_runner_e2e.py` drives `run_main` end to end against an in-memory broker with a frozen clock (a due day, a quiet day, a closed market, a stop-loss day, same-day reruns, a resumed attempt, and the stand-down markers the workflows grep for, matched with the workflows' own pattern); `tests/test_target_equivalence.py` requires the live target to equal the backtest provider's target for the same date and the same book, and records as xfails the two places where it does not yet (an empty book reaches the optimizer as None from the backtest and as an empty Series from the runner; a live fundamentals snapshot industry-neutralizes by its own sector column, which the backtest never does); `tests/test_lookahead.py` checks that every enabled factor, the composite, the trend book's decisions and the engine's book at T do not depend on bars after T.

## Deployment Notes

- Logs go to `logs/paper_trade_YYYYMMDD.log`; state tracked in `logs/paper_trade_state.json` / `logs/paper_trade_lgbm_state.json` (committed to main by the workflows)
- `logs/paper_trade.lock` protects a local process; each book's Actions concurrency group serializes its own hosted runs
- Alpaca paper accounts may not process corporate actions. The dashboard annotates known splits, while the trading path fails closed if BKNG still appears in pre-split units

## Known Limitations

- **Survivorship bias**: Static 100-stock universe excludes delisted companies. The ETF workflow is a robustness control, not a bias estimate; use PIT constituents plus delisting returns for a bias-reduced stock run
- **Next-open execution**: Backtest signals at T close and fills at the next available session open (same-day close fallback only when the open is absent); real fills can still differ
- **No point-in-time fundamentals**: Why quality/value factors are disabled
- **Paper fills**: Alpaca paper fills carry no real spread/impact; live costs will be higher
