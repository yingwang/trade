#!/bin/bash

# Alpaca Paper Trading Setup Script — LightGBM Strategy
# This script helps you configure environment variables for
# the separate Alpaca paper trading account used by the LightGBM strategy.

echo "=========================================="
echo "   Alpaca LightGBM Paper Trading Setup"
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
read -p "Enter your Alpaca LGBM API Key: " API_KEY
read -s -p "Enter your Alpaca LGBM Secret Key: " SECRET_KEY
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$PROJECT_DIR/.env.lgbm"

cat > "$ENV_FILE" <<EOF
ALPACA_LGBM_API_KEY="$API_KEY"
ALPACA_LGBM_SECRET_KEY="$SECRET_KEY"
EOF

echo ""
echo "Environment variables saved to $ENV_FILE"
echo ""
echo "Step 3: Load the project environment"
echo "----------------------------------------"
echo "Run: source $ENV_FILE"
echo ""
echo "Or configure direnv to load $ENV_FILE automatically."
echo ""
echo "Step 4: Test your setup"
echo "----------------------------------------"
echo "Run these commands to test:"
echo ""
echo "  # Check portfolio status"
echo "  python3 paper_trade_lgbm.py --status"
echo ""
echo "  # Dry run (see what would be traded)"
echo "  python3 paper_trade_lgbm.py --dry-run"
echo ""
echo "  # Execute actual rebalance"
echo "  python3 paper_trade_lgbm.py"
echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
