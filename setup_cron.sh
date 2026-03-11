#!/bin/bash

# Setup script for automatic paper trading cron job

echo "=========================================="
echo "   Setup Automatic Rebalancing"
echo "=========================================="
echo ""
echo "This will add a cron job that runs the paper trading script"
echo "every weekday at 3:55 PM ET (before market close)."
echo ""
echo "The script will only rebalance if 21+ trading days have passed."
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Create logs directory if it doesn't exist
mkdir -p /Users/ying/claude/trade/logs

# Add cron job
CRON_CMD="55 15 * * 1-5 cd /Users/ying/claude/trade && source ~/.bashrc && python3 paper_trade.py >> logs/trade.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "paper_trade.py"; then
    echo ""
    echo "⚠️  Cron job already exists!"
    echo ""
    crontab -l | grep "paper_trade.py"
    echo ""
    read -p "Do you want to replace it? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    # Remove old cron job
    crontab -l | grep -v "paper_trade.py" | crontab -
fi

# Add new cron job
(crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -

echo ""
echo "✅ Cron job installed successfully!"
echo ""
echo "Schedule: Every weekday at 3:55 PM ET"
echo "Log file: /Users/ying/claude/trade/logs/trade.log"
echo ""
echo "To view your cron jobs:"
echo "  crontab -l"
echo ""
echo "To view the log:"
echo "  tail -f /Users/ying/claude/trade/logs/trade.log"
echo ""
echo "To remove the cron job:"
echo "  crontab -e"
echo "  (then delete the line with paper_trade.py)"
echo ""
echo "=========================================="
