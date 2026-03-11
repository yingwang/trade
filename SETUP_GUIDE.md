# Alpaca Paper Trading Setup Guide

## Prerequisites

- ✓ Python 3.9+ installed
- ✓ All dependencies installed (numpy, pandas, yfinance, matplotlib, seaborn, alpaca-trade-api)
- ✓ Alpaca account (free paper trading account)

## Quick Setup (3 Steps)

### 1. Get Alpaca API Credentials

1. Go to [alpaca.markets](https://alpaca.markets)
2. Create a free account (no credit card required for paper trading)
3. Navigate to **Paper Trading** section in the dashboard
4. Click **Generate API Keys**
5. Save your:
   - **API Key** (starts with `PK...`)
   - **Secret Key** (keep this secure!)

### 2. Configure Environment Variables

**Option A: Use the setup script**
```bash
cd /Users/ying/claude/trade
./setup_alpaca.sh
```

**Option B: Manual setup**
```bash
# Add to ~/.zshrc or ~/.bash_profile
export ALPACA_API_KEY="your-api-key-here"
export ALPACA_SECRET_KEY="your-secret-key-here"

# Reload shell
source ~/.zshrc
```

### 3. Test Your Setup

```bash
cd /Users/ying/claude/trade

# Check if API credentials work
python3 paper_trade.py --status
```

## Usage Commands

### Check Portfolio Status
```bash
python3 paper_trade.py --status
```
Shows current holdings, cash, and portfolio value.

### Dry Run (Preview Trades)
```bash
python3 paper_trade.py --dry-run
```
Calculates what trades would be made WITHOUT executing them.

### Execute Rebalance
```bash
python3 paper_trade.py
```
Executes the portfolio rebalance (only if 21+ trading days have passed).

### Force Rebalance
```bash
python3 paper_trade.py --force
```
Rebalances immediately, ignoring the 21-day interval.

## How It Works

1. **Data Collection**: Fetches 3 years of price data for 30 large-cap stocks
2. **Factor Calculation**: Computes 6 alpha factors:
   - Momentum (30% weight)
   - Trend (20%)
   - Value (15%)
   - Mean Reversion (15%)
   - Volatility (10%)
   - Quality (10%)
3. **Portfolio Optimization**: Selects top 20 stocks using mean-variance optimization
4. **Execution**: Generates and submits orders to Alpaca
5. **Tracking**: Logs all trades and updates state file

## Configuration

Edit `config.yaml` to customize:

- **Stock universe**: Add/remove symbols
- **Position limits**: Max/min weights per stock
- **Rebalance frequency**: Default is 21 trading days (monthly)
- **Risk parameters**: Max drawdown, sector limits, stop loss

## File Structure

```
/Users/ying/claude/trade/
├── paper_trade.py          # Main trading script
├── config.yaml             # Strategy configuration
├── setup_alpaca.sh         # Setup helper script
├── logs/
│   ├── paper_trade_YYYYMMDD.log  # Daily trade logs
│   └── paper_trade_state.json    # Rebalance state tracker
└── quant/
    ├── strategy.py         # Strategy logic
    ├── execution/
    │   └── alpaca_broker.py  # Alpaca API integration
    └── ...
```

## Troubleshooting

### "No module named 'alpaca_trade_api'"
```bash
pip3 install alpaca-trade-api
```

### "Environment variables not set"
```bash
# Check if variables are set
echo $ALPACA_API_KEY
echo $ALPACA_SECRET_KEY

# If empty, run setup again
./setup_alpaca.sh
source ~/.zshrc
```

### "Xcode license agreement error"
```bash
sudo xcodebuild -license
# Type 'agree' when prompted
```

## Safety Notes

- **Paper trading is enabled by default** - no real money is used
- The script only rebalances every 21 trading days to avoid overtrading
- All trades are logged in `logs/` directory
- State is tracked to prevent duplicate trades

## Next Steps

1. Run `--dry-run` first to see what the strategy would do
2. Check the proposed trades make sense
3. Run actual rebalance when ready
4. Monitor via `--status` command
5. Set up a cron job for automation (optional)

## Automation (Optional)

To auto-rebalance at 3:55 PM ET on weekdays:

```bash
crontab -e
```

Add:
```
55 15 * * 1-5 cd /Users/ying/claude/trade && python3 paper_trade.py >> logs/trade.log 2>&1
```

---

**Questions or Issues?**
- Check the main README.md
- Review logs in `logs/` directory
- Verify config.yaml settings
