"""Alpaca broker implementation for paper and live trading.

Requires:
    pip install alpaca-trade-api

Setup:
    1. Create account at https://alpaca.markets
    2. Go to Paper Trading -> API Keys
    3. Set environment variables:
       export ALPACA_API_KEY="your-key"
       export ALPACA_SECRET_KEY="your-secret"
       export ALPACA_PAPER="true"   # "true" for paper, "false" for live
"""

import logging
import os
import time

import pandas as pd

from quant.execution.broker import BaseBroker, Order

logger = logging.getLogger(__name__)


class AlpacaBroker(BaseBroker):
    """Alpaca API broker for paper and live trading."""

    def __init__(self, api_key: str = None, secret_key: str = None,
                 paper: bool = True):
        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            raise ImportError(
                "alpaca-trade-api not installed. Run: pip install alpaca-trade-api"
            )

        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.paper = paper if paper else os.environ.get("ALPACA_PAPER", "true").lower() == "true"

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API keys not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "environment variables, or pass them directly."
            )

        base_url = (
            "https://paper-api.alpaca.markets" if self.paper
            else "https://api.alpaca.markets"
        )

        self.api = tradeapi.REST(
            self.api_key, self.secret_key, base_url, api_version="v2"
        )

        # Verify connection
        account = self.api.get_account()
        mode = "PAPER" if self.paper else "LIVE"
        logger.info(
            "Connected to Alpaca [%s] | Equity: $%s | Cash: $%s",
            mode, account.equity, account.cash,
        )

    def submit_order(self, order: Order) -> Order:
        """Submit order to Alpaca."""
        try:
            alpaca_order = self.api.submit_order(
                symbol=order.symbol,
                qty=int(order.quantity),
                side=order.side,
                type=order.order_type,
                time_in_force="day",
                limit_price=str(order.limit_price) if order.limit_price else None,
            )

            # Wait for fill (up to 30s for market orders)
            if order.order_type == "market":
                filled = self._wait_for_fill(alpaca_order.id, timeout=30)
                if filled:
                    order.status = "filled"
                    order.filled_price = float(filled.filled_avg_price)
                    order.order_id = filled.id
                    logger.info(
                        "Filled: %s %s %d @ $%.2f",
                        order.side.upper(), order.symbol,
                        order.quantity, order.filled_price,
                    )
                else:
                    order.status = "submitted"
                    order.order_id = alpaca_order.id
            else:
                order.status = "submitted"
                order.order_id = alpaca_order.id

        except Exception as e:
            order.status = "rejected"
            logger.error("Order rejected for %s: %s", order.symbol, e)

        return order

    def _wait_for_fill(self, order_id: str, timeout: int = 30):
        """Poll order status until filled or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            o = self.api.get_order(order_id)
            if o.status == "filled":
                return o
            if o.status in ("canceled", "expired", "rejected"):
                return None
            time.sleep(1)
        return None

    def get_positions(self) -> pd.Series:
        """Get current positions as symbol -> shares."""
        positions = self.api.list_positions()
        return pd.Series(
            {p.symbol: float(p.qty) for p in positions}, dtype=float
        )

    def get_portfolio_value(self) -> float:
        account = self.api.get_account()
        return float(account.equity)

    def get_cash(self) -> float:
        account = self.api.get_account()
        return float(account.cash)

    def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get latest prices from Alpaca."""
        prices = {}
        for sym in symbols:
            try:
                trade = self.api.get_latest_trade(sym)
                prices[sym] = float(trade.price)
            except Exception as e:
                logger.warning("Could not get price for %s: %s", sym, e)
        return prices

    def cancel_all_orders(self):
        """Cancel all open orders."""
        self.api.cancel_all_orders()
        logger.info("Cancelled all open orders")

    def close_all_positions(self):
        """Liquidate all positions."""
        self.api.close_all_positions()
        logger.info("Closed all positions")
