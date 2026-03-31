# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Multi-factor quantitative equity trading system for medium-term US equities. Uses 5 price-based alpha factors with mean-variance portfolio optimization, dynamic leverage via market regime detection, and automated execution through Alpaca API.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full backtest (5-year default)
python run.py backtest --start 2021-03-16 --plot

# View current alpha signals
python run.py signal

# Run tests
pytest tests/
pytest tests/test_factors.py -v        # single module
pytest --cov=quant tests/              # with coverage

# Paper trading (Alpaca)
python paper_trade.py --dry-run        # preview trades
python paper_trade.py                  # execute rebalance
python paper_trade.py --status         # check status
python paper_trade.py --force          # force immediate rebalance
python paper_trade.py --reconcile      # post-trade check

# Dashboard generation
python generate_site.py                # outputs JSON to site/data/

# Verbose logging
python run.py -v backtest --start 2021-03-16
```

## Architecture

```
run.py / paper_trade.py          ← Entry points
        │
quant/strategy.py                ← MultiFactorStrategy orchestrator
        │
   ┌────┼────────────┬──────────────┐
   │    │            │              │
data/   signals/   portfolio/   execution/   backtest/
```

- **`quant/strategy.py`** — Orchestrates everything: `run_backtest()`, `get_current_signal()`, `get_current_portfolio()`
- **`quant/data/market_data.py`** — yfinance wrapper for OHLCV and fundamentals
- **`quant/signals/factors.py`** — Alpha factor calculations (momentum, 52-week high, reversal, vol contraction, volume momentum). Factors are industry-neutralized and winsorized to ±3σ
- **`quant/portfolio/optimizer.py`** — Constrained MVO with Ledoit-Wolf shrinkage covariance. Includes `detect_regime()` (SPY vol-based) and `apply_vol_scaling()` for dynamic leverage (0.8x–1.8x)
- **`quant/execution/broker.py`** — `PaperBroker` for simulation; `quant/execution/alpaca_broker.py` for live
- **`quant/execution/safety.py`** — `PreTradeCheck`: order size limits ($50k), daily limits ($500k), liquidity checks (1% ADV), loss limits ($25k/day)
- **`quant/backtest/engine.py`** — Event-driven daily simulator with Almgren-Chriss market impact model
- **`config.yaml`** — All strategy parameters: universe (100 stocks), factor weights, portfolio constraints, risk limits, safety thresholds

## Signal Pipeline

Raw prices → factor scores (momentum 50%, high_proximity 20%, reversal 10%, vol_contraction 10%, volume_momentum 10%) → industry-neutral z-scores → trend filter (< 200d SMA → 0.5x) → blowoff filter (z > 4.0 → 0.5x) → composite score → MVO optimization → target weights

## Key Constraints in config.yaml

- Max 12 positions, 3%–12% per stock, max 50% per sector
- Rebalance every 21 trading days, max 40% turnover
- Transaction cost: 10 bps + Almgren-Chriss market impact
- Quality and value factors are **disabled** — yfinance only provides current snapshots, causing look-ahead bias in backtests

## Testing

Tests use synthetic fixtures (3 years of 10 stocks + benchmark) defined in `tests/conftest.py` — fully offline, no API calls needed. Test modules cover factors, portfolio optimization, backtesting, broker, safety checks, and strategy-vs-baseline comparison.

## Deployment

- `setup_alpaca.sh` — Sets `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` env vars
- `setup_cron.sh` — Installs cron job at 3:55 PM ET weekdays (before market close)
- Logs go to `logs/paper_trade_YYYYMMDD.log`; state tracked in `logs/paper_trade_state.json`
- `logs/paper_trade.lock` prevents concurrent execution (auto-expires after 1 hour)

## Known Limitations

- **Survivorship bias**: Static 100-stock universe excludes delisted companies
- **Same-day execution**: Backtest signals and trades use same closing price
- **Stop-loss**: Enforced daily in both backtest and paper trading (15% threshold)
- **No point-in-time fundamentals**: Why quality/value factors are disabled
