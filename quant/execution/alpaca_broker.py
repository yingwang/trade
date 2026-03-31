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

NOTE: alpaca-trade-api is the legacy SDK. Consider migrating to alpaca-py
(https://github.com/alpacahq/alpaca-py) for active maintenance and websocket
support.  This module continues to use alpaca-trade-api for now.
"""

import logging
import os
import time
from typing import Optional

import pandas as pd

from quant.execution.broker import BaseBroker, Order
from quant.execution.safety import (
    ExecutionLogger,
    PostTradeReconciler,
    PreTradeCheck,
    SafetyConfig,
    TWAPSplitter,
)

logger = logging.getLogger(__name__)


class AlpacaBroker(BaseBroker):
    """Alpaca API broker for paper and live trading.

    Now includes:
    - Pre-trade safety checks (max order value, daily limits, ADV)
    - Post-trade reconciliation
    - TWAP order splitting for large orders
    - Structured execution logging with slippage tracking
    - Partial fill handling
    - Paper-mode validation
    """

    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        paper: bool = True,
        safety_config: SafetyConfig = None,
    ):
        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            raise ImportError(
                "alpaca-trade-api not installed. Run: pip install alpaca-trade-api"
            )

        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.paper = (
            paper if paper else
            os.environ.get("ALPACA_PAPER", "true").lower() == "true"
        )

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API keys not found. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables."
            )

        # Do NOT store keys on the instance beyond what the REST client needs.
        # Clear plain-text copies after passing to the SDK.
        base_url = (
            "https://paper-api.alpaca.markets" if self.paper
            else "https://api.alpaca.markets"
        )

        self.api = tradeapi.REST(
            self.api_key, self.secret_key, base_url, api_version="v2"
        )

        # Wipe plain-text key copies from instance attributes
        self.api_key = "***"
        self.secret_key = "***"

        # Safety subsystems
        self.safety = PreTradeCheck(safety_config or SafetyConfig())
        self.reconciler = PostTradeReconciler()
        self.twap = TWAPSplitter()
        self.exec_log = ExecutionLogger()

        # Paper-mode gate
        if self.safety.config.require_paper_mode and not self.paper:
            raise RuntimeError(
                "SAFETY: require_paper_mode is True but paper=False. "
                "Set require_paper_mode=False in SafetyConfig to enable live trading."
            )

        # Verify connection
        account = self.api.get_account()
        mode = "PAPER" if self.paper else "LIVE"
        logger.info(
            "Connected to Alpaca [%s] | Equity: $%s | Cash: $%s",
            mode, account.equity, account.cash,
        )

    # ------------------------------------------------------------------
    # Order submission with safety checks
    # ------------------------------------------------------------------

    def submit_order(self, order: Order, avg_daily_volume: float = None) -> Order:
        """Submit order to Alpaca after safety validation.

        Parameters
        ----------
        order : Order
            The order to submit.
        avg_daily_volume : float, optional
            Recent average daily share volume for the symbol, used for
            liquidity checks and TWAP splitting decisions.
        """
        portfolio_value = self.get_portfolio_value()
        price = order.signal_price or self._get_price_safe(order.symbol)

        if price is None or price <= 0:
            order.status = "rejected"
            order.reject_reason = f"No valid price for {order.symbol}"
            logger.error("Order rejected: %s", order.reject_reason)
            self.exec_log.log_order_rejected(order, order.reject_reason)
            return order

        # Pre-trade safety check
        current_positions = {}
        current_shares = self.get_positions()
        if not current_shares.empty:
            position_prices = self.get_current_prices(current_shares.index.tolist())
            current_positions = {
                symbol: shares * position_prices.get(symbol, 0.0)
                for symbol, shares in current_shares.items()
            }
        passed, reason = self.safety.validate(
            order, price, portfolio_value, current_positions, avg_daily_volume
        )
        if not passed:
            order.status = "rejected"
            order.reject_reason = reason
            self.exec_log.log_safety_block(order, reason)
            self.safety.record_rejection()
            return order

        # TWAP splitting for large orders
        if avg_daily_volume and self.twap.should_split(order.quantity, avg_daily_volume):
            return self._execute_twap(order, avg_daily_volume, price)

        # Single order execution
        return self._execute_single(order, price)

    def _execute_single(self, order: Order, signal_price: float) -> Order:
        """Execute a single order against Alpaca API with retry on transient errors."""
        self.safety.record_submission()
        self.exec_log.log_order_submitted(order, signal_price)

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                alpaca_order = self.api.submit_order(
                    symbol=order.symbol,
                    qty=int(order.quantity),
                    side=order.side,
                    type=order.order_type,
                    time_in_force="day",
                    limit_price=(
                        str(order.limit_price) if order.limit_price else None
                    ),
                )
                break  # success
            except Exception as e:
                err_str = str(e)
                # Retry on network / rate-limit errors, not on business logic errors
                if attempt < max_retries and (
                    "timeout" in err_str.lower()
                    or "429" in err_str
                    or "connection" in err_str.lower()
                ):
                    wait = 2 ** attempt
                    logger.warning(
                        "Transient error submitting %s (attempt %d/%d), "
                        "retrying in %ds: %s",
                        order.symbol, attempt + 1, max_retries + 1, wait, e,
                    )
                    time.sleep(wait)
                    continue
                # Non-retryable error
                order.status = "rejected"
                order.reject_reason = str(e)
                logger.error(
                    "Order rejected for %s: %s", order.symbol, e
                )
                self.exec_log.log_order_rejected(order, str(e))
                self.safety.record_rejection()
                return order

        # Wait for fill
        if order.order_type == "market":
            filled = self._wait_for_fill(alpaca_order.id, timeout=30)
        else:
            # For limit orders give more time
            filled = self._wait_for_fill(alpaca_order.id, timeout=60)

        if filled:
            filled_qty = float(filled.filled_qty)
            order.filled_price = float(filled.filled_avg_price)
            order.order_id = filled.id

            if filled_qty >= order.quantity:
                order.status = "filled"
            else:
                order.status = "partial_fill"
                logger.warning(
                    "Partial fill for %s: %d/%d shares @ $%.2f",
                    order.symbol, int(filled_qty),
                    int(order.quantity), order.filled_price,
                )
                order.quantity = filled_qty  # update to actual filled qty

            logger.info(
                "Filled: %s %s %d @ $%.2f (signal=$%.2f)",
                order.side.upper(), order.symbol,
                int(order.quantity), order.filled_price,
                signal_price,
            )
            self.exec_log.log_order_filled(order, signal_price)
            self.safety.record_fill(order.quantity * order.filled_price)
        else:
            order.status = "submitted"
            order.order_id = alpaca_order.id
            logger.warning(
                "Order for %s not filled within timeout, status=%s",
                order.symbol, "submitted",
            )

        return order

    def _execute_twap(
        self, order: Order, avg_daily_volume: float, signal_price: float
    ) -> Order:
        """Execute order as TWAP slices."""
        slices = self.twap.split_order(order, avg_daily_volume)

        target_quantity = order.quantity
        total_filled_qty = 0.0
        total_filled_value = 0.0
        last_order_id = ""

        for child_order, delay_seconds in slices:
            if delay_seconds > 0:
                logger.info(
                    "TWAP: waiting %ds before slice %s %s %d shares",
                    delay_seconds, child_order.side.upper(),
                    child_order.symbol, int(child_order.quantity),
                )
                time.sleep(delay_seconds)

            result = self._execute_single(child_order, signal_price)
            if result.status in ("filled", "partial_fill"):
                total_filled_qty += result.quantity
                total_filled_value += result.quantity * result.filled_price
                last_order_id = result.order_id

        # Aggregate result into parent order
        if total_filled_qty > 0:
            order.filled_price = total_filled_value / total_filled_qty
            order.quantity = total_filled_qty
            order.status = (
                "filled" if total_filled_qty >= target_quantity else "partial_fill"
            )
            order.order_id = last_order_id
        else:
            order.status = "rejected"
            order.reject_reason = "All TWAP slices failed"

        return order

    # ------------------------------------------------------------------
    # Fill polling (improved: handles partial fills)
    # ------------------------------------------------------------------

    def _wait_for_fill(self, order_id: str, timeout: int = 30):
        """Poll order status until filled, partially filled, or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                o = self.api.get_order(order_id)
            except Exception as e:
                logger.warning("Error polling order %s: %s", order_id, e)
                time.sleep(2)
                continue

            if o.status == "filled":
                return o
            if o.status == "partially_filled":
                # For market orders, keep waiting; they should fill fully
                # but return partial info if timeout
                pass
            if o.status in ("canceled", "expired", "rejected"):
                logger.warning(
                    "Order %s terminal status: %s", order_id, o.status
                )
                return None
            time.sleep(1)

        # Timeout: check one more time for partial fill
        try:
            o = self.api.get_order(order_id)
            if o.status in ("filled", "partially_filled"):
                return o
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Position and account queries
    # ------------------------------------------------------------------

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
            price = self._get_price_safe(sym)
            if price is not None:
                prices[sym] = price
        return prices

    def _get_price_safe(self, symbol: str) -> Optional[float]:
        """Get price for a single symbol, returning None on failure."""
        try:
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)
        except Exception as e:
            logger.warning("Could not get price for %s: %s", symbol, e)
            return None

    # ------------------------------------------------------------------
    # Emergency controls
    # ------------------------------------------------------------------

    def cancel_all_orders(self):
        """Cancel all open orders."""
        self.api.cancel_all_orders()
        logger.info("Cancelled all open orders")

    def close_all_positions(self):
        """Liquidate all positions (emergency use)."""
        self.api.close_all_positions()
        logger.warning("EMERGENCY: Closed all positions")

    # ------------------------------------------------------------------
    # Post-trade reconciliation
    # ------------------------------------------------------------------

    def reconcile(
        self,
        target_weights: pd.Series,
    ) -> pd.DataFrame:
        """Compare strategy targets vs actual Alpaca positions.

        Returns a DataFrame of discrepancies. Also logs to structured log.
        """
        actual_positions = self.get_positions()
        portfolio_value = self.get_portfolio_value()
        all_syms = list(
            set(target_weights.index) | set(actual_positions.index)
        )
        prices = self.get_current_prices(all_syms)

        drift_df = self.reconciler.reconcile(
            target_weights, actual_positions, prices, portfolio_value
        )
        self.exec_log.log_reconciliation(drift_df, portfolio_value)
        return drift_df

    # ------------------------------------------------------------------
    # Market hours check
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.warning("Could not check market clock: %s", e)
            return False
