#!/bin/bash

# Alpaca Paper Trading Setup Script
# This script helps you configure environment variables for Alpaca paper trading

echo "=========================================="
echo "   Alpaca Paper Trading Setup"
echo "=========================================="
echo ""
echo "Step 1: Get your Alpaca API credentials"
echo "----------------------------------------"
echo "1. Go to https://alpaca.markets"
echo "2. Sign up or log in to your account"
echo "3. Navigate to 'Paper Trading' section"
echo "4. Generate your API keys"
echo ""
echo "You will need:"
echo "  - API Key (starts with 'PK...')"
echo "  - Secret Key"
echo ""
echo "Step 2: Set environment variables"
echo "----------------------------------------"
echo ""
read -p "Enter your Alpaca API Key: " API_KEY
read -p "Enter your Alpaca Secret Key: " SECRET_KEY
echo ""

# Detect shell configuration file
if [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_RC="$HOME/.bash_profile"
else
    SHELL_RC="$HOME/.bashrc"
fi

echo "Adding environment variables to $SHELL_RC..."
echo "" >> "$SHELL_RC"
echo "# Alpaca Paper Trading API credentials" >> "$SHELL_RC"
echo "export ALPACA_API_KEY=\"$API_KEY\"" >> "$SHELL_RC"
echo "export ALPACA_SECRET_KEY=\"$SECRET_KEY\"" >> "$SHELL_RC"

# Also export for current session
export ALPACA_API_KEY="$API_KEY"
export ALPACA_SECRET_KEY="$SECRET_KEY"

echo ""
echo "✓ Environment variables saved to $SHELL_RC"
echo ""
echo "Step 3: Reload your shell configuration"
echo "----------------------------------------"
echo "Run: source $SHELL_RC"
echo ""
echo "Or simply open a new terminal window."
echo ""
echo "Step 4: Test your setup"
echo "----------------------------------------"
echo "Run these commands to test:"
echo ""
echo "  # Check portfolio status"
echo "  python3 paper_trade.py --status"
echo ""
echo "  # Dry run (see what would be traded)"
echo "  python3 paper_trade.py --dry-run"
echo ""
echo "  # Execute actual rebalance"
echo "  python3 paper_trade.py"
echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
