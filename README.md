# Quant Trading System

A multi-factor quantitative trading system for medium-term US equities. Combines six alpha factors with mean-variance portfolio optimization and automated execution via Alpaca.

## Strategy Overview

The system scores stocks using a weighted composite of six factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| Momentum | 30% | Cross-sectional returns over 1/3/6/12 months (skipping most recent month) |
| Trend | 20% | SMA 50/200 ratio — favors stocks in uptrends |
| Value | 15% | Inverse P/E and P/B ratios — favors cheaper stocks |
| Mean Reversion | 15% | Bollinger band z-score — favors oversold names |
| Volatility | 10% | Realized 63-day vol — favors low-volatility stocks |
| Quality | 10% | ROE, profit margins, earnings growth |

The portfolio holds up to 20 positions, rebalances monthly, and targets 15% annualized volatility.

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `yfinance`, `scipy`, `matplotlib`, `seaborn`, `ta`

### Run a Backtest

```bash
# Default backtest (2022-01-01 to today, $1M capital)
python run.py backtest

# Custom dates with plot output
python run.py backtest --start 2023-01-01 --end 2024-12-31 --plot

# Verbose logging
python run.py -v backtest --plot --plot-output my_backtest.png
```

### View Current Signals

```bash
python run.py signal
```

Shows the composite alpha score for every stock in the universe, ranked from strongest to weakest.

---

## Manual Usage (Google Colab)

The notebook `quant_trading_colab.ipynb` gives you a visual, interactive workflow — no local setup needed.

### Setup in Colab

1. Open the notebook in Google Colab
2. Run **Cell 1** (Setup) — clones the repo and installs dependencies
3. Run **Cell 2** (Imports) — loads config and initializes the strategy

### What Each Cell Does

| Cell | Purpose |
|------|---------|
| 1 - Setup | Clone repo, install packages |
| 2 - Imports | Load config, create strategy object |
| 3 - Backtest | Run full historical backtest, print summary metrics (Sharpe, CAGR, max drawdown, etc.) |
| 4 - Equity Curve | Plot equity curve, drawdown, and rolling Sharpe ratio |
| 5 - Monthly Returns | Heatmap of monthly returns by year |
| 6 - Risk Report | Extended stats: skewness, kurtosis, VaR, CVaR, win rate |
| 7 - Current Signals | Fetch today's alpha scores for all stocks |
| 8 - Target Portfolio | **The key cell** — shows exactly what to buy today |

### Using the Target Portfolio (Cell 8)

Edit `MY_CAPITAL` to match your account size:

```python
MY_CAPITAL = 50_000  # <-- your portfolio size in USD
```

Run the cell. Output looks like:

```
=================================================================
  TARGET PORTFOLIO  |  Capital: $50,000
=================================================================
  Stock      Weight    Dollars   Shares      Price     Score
  ------------------------------------------------------------
  NVDA        9.8%    $4,900      36   $136.25    0.842
  META        8.5%    $4,250       7   $607.14    0.731
  ...
  ------------------------------------------------------------
  TOTAL      95.2%   $47,600
  Cash reserve: $2,400
=================================================================
```

Then manually place the trades in your brokerage account.

### Rebalancing Manually

Run Cell 8 monthly (every ~21 trading days). Compare the new target with your current holdings and adjust:
- **Sell** positions that dropped out of the top 20
- **Buy** new positions that entered
- **Resize** existing positions to match target weights

---

## Automated Trading with Alpaca

### 1. Create an Alpaca Account

Sign up at [alpaca.markets](https://alpaca.markets). Start with a **paper trading** account to test without real money.

### 2. Set Environment Variables

```bash
export ALPACA_API_KEY="your-api-key"
export ALPACA_SECRET_KEY="your-secret-key"
```

Paper trading is the default. For live trading (at your own risk), also set:

```bash
export ALPACA_PAPER="false"
```

### 3. Install the Alpaca SDK

```bash
pip install alpaca-trade-api
```

### 4. Paper Trading Commands

```bash
# See what trades would be made (no execution)
python paper_trade.py --dry-run

# Check current portfolio status
python paper_trade.py --status

# Execute rebalance (only runs if 21+ trading days since last)
python paper_trade.py

# Force rebalance regardless of schedule
python paper_trade.py --force
```

### 5. Automate with Cron

Add this to your crontab to auto-rebalance at 3:55 PM ET on weekdays:

```bash
crontab -e
```

```
55 15 * * 1-5 cd /path/to/trade && python paper_trade.py >> logs/trade.log 2>&1
```

The script tracks state in `logs/paper_trade_state.json` and only rebalances when the configured interval (21 trading days) has passed. Use `--force` to override.

### 6. Monitoring

- **Trade logs**: `logs/paper_trade_YYYYMMDD.log` — every order with timestamps and fill prices
- **State file**: `logs/paper_trade_state.json` — last rebalance date and full trade history
- **Quick check**: `python paper_trade.py --status`

---

## Configuration

All parameters live in `config.yaml`:

```yaml
universe:
  symbols: [AAPL, MSFT, GOOGL, ...]   # Stocks to trade
  benchmark: SPY                        # Benchmark index

portfolio:
  max_positions: 20          # Max holdings
  max_position_weight: 0.10  # 10% cap per stock
  min_position_weight: 0.02  # 2% floor per stock
  target_volatility: 0.15    # 15% annual vol target
  rebalance_frequency_days: 21
  transaction_cost_bps: 10

risk:
  max_drawdown_limit: 0.20   # Halt at 20% drawdown
  max_sector_weight: 0.30    # 30% sector cap
  stop_loss_pct: 0.08        # 8% per-stock stop loss
```

### Customizing the Universe

Edit the `symbols` list in `config.yaml`. The system works with any US equities available on Yahoo Finance. The default includes 50 stocks spanning mega-cap tech (NVDA, AMD, CRM, PLTR, etc.) and blue-chip defensives (JNJ, PG, KO, etc.).

### Tuning Factor Weights

Factor weights control the portfolio's style. Edit `signals.factor_weights` in `config.yaml`:

```yaml
signals:
  factor_weights:
    momentum: 0.30       # Recent price winners
    mean_reversion: 0.10 # Oversold bounce candidates
    trend: 0.25          # Stocks above key moving averages
    volatility: 0.05     # Prefer low-vol (higher = more defensive)
    value: 0.15          # Cheap on P/E and P/B
    quality: 0.15        # High ROE, margins, growth
```

**Presets:**

| Style | momentum | trend | value | volatility | mean_rev | quality |
|-------|----------|-------|-------|------------|----------|---------|
| Growth/Tech | 0.35 | 0.30 | 0.05 | 0.00 | 0.10 | 0.20 |
| Balanced (default) | 0.30 | 0.25 | 0.15 | 0.05 | 0.10 | 0.15 |
| Defensive/Income | 0.15 | 0.10 | 0.25 | 0.25 | 0.10 | 0.15 |

Weights must sum to 1.0.

### Adjusting Risk

- **More conservative**: Lower `target_volatility` (e.g., 0.10), lower `max_position_weight`, increase `min_position_weight`
- **More aggressive**: Raise `target_volatility` (e.g., 0.20), allow larger positions

---

## Project Structure

```
trade/
├── config.yaml                 # Strategy configuration
├── run.py                      # CLI: backtest & signals
├── paper_trade.py              # Alpaca paper/live trading
├── quant_trading_colab.ipynb   # Interactive Colab notebook
├── requirements.txt
├── quant/
│   ├── strategy.py             # Main orchestrator
│   ├── data/
│   │   └── market_data.py      # Yahoo Finance data fetching
│   ├── signals/
│   │   └── factors.py          # Alpha factor calculations
│   ├── portfolio/
│   │   └── optimizer.py        # Mean-variance optimization
│   ├── backtest/
│   │   ├── engine.py           # Backtesting engine
│   │   └── report.py           # Analytics & reporting
│   ├── execution/
│   │   ├── broker.py           # Broker interface + paper broker
│   │   └── alpaca_broker.py    # Alpaca API integration
│   └── utils/
│       └── config.py           # Config loader
└── tests/                      # Unit tests
```

## Disclaimer

This software is for educational and research purposes. Past performance does not guarantee future results. Use at your own risk. Always start with paper trading before committing real capital.
