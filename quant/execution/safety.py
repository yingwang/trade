"""Pre-trade safety checks and post-trade reconciliation.

This module enforces risk guardrails that prevent catastrophic execution errors.
Every order must pass all safety checks before submission to any broker.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Configuration for pre-trade safety limits."""
    # Single order limits
    max_single_order_value: float = 50_000.0   # Max $ value for one order
    max_single_order_shares: int = 10_000       # Max shares per order

    # Daily cumulative limits
    max_daily_trade_value: float = 500_000.0    # Max total $ traded per day
    max_daily_loss: float = 25_000.0            # Max realised + unrealised loss per day

    # Liquidity checks
    max_adv_fraction: float = 0.01              # Max 1% of avg daily volume
    min_price: float = 1.0                      # Reject penny stocks

    # Position concentration limits
    max_position_pct_of_portfolio: float = 0.15  # 15% max single position

    # Environment safety
    require_paper_mode: bool = True              # Block live trading unless explicitly disabled

    @classmethod
    def from_config(cls, config: dict) -> "SafetyConfig":
        """Build SafetyConfig from the project config.yaml dict, with safe defaults."""
        safety = config.get("safety", {})
        return cls(
            max_single_order_value=safety.get("max_single_order_value", 50_000.0),
            max_single_order_shares=safety.get("max_single_order_shares", 10_000),
            max_daily_trade_value=safety.get("max_daily_trade_value", 500_000.0),
            max_daily_loss=safety.get("max_daily_loss", 25_000.0),
            max_adv_fraction=safety.get("max_adv_fraction", 0.01),
            min_price=safety.get("min_price", 1.0),
            max_position_pct_of_portfolio=safety.get(
                "max_position_pct_of_portfolio", 0.15
            ),
            require_paper_mode=safety.get("require_paper_mode", True),
        )


@dataclass
class DailyTracker:
    """Tracks cumulative trading activity for the current day."""
    trade_date: date = field(default_factory=date.today)
    total_value_traded: float = 0.0
    total_realised_pnl: float = 0.0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0

    def reset_if_new_day(self):
        today = date.today()
        if self.trade_date != today:
            self.trade_date = today
            self.total_value_traded = 0.0
            self.total_realised_pnl = 0.0
            self.orders_submitted = 0
            self.orders_filled = 0
            self.orders_rejected = 0


class PreTradeCheck:
    """Validates every order against safety limits before submission.

    Usage:
        checker = PreTradeCheck(SafetyConfig())
        ok, reason = checker.validate(order, price, portfolio_value, {}, adv)
        if not ok:
            reject order with reason
    """

    def __init__(self, config: SafetyConfig = None):
        self.config = config or SafetyConfig()
        self.daily = DailyTracker()

    def validate(
        self,
        order,              # broker.Order
        price: float,
        portfolio_value: float,
        current_positions: Optional[dict[str, float]] = None,
        avg_daily_volume: Optional[float] = None,
    ) -> tuple[bool, str]:
        """Run all pre-trade checks. Returns (passed, reason)."""
        self.daily.reset_if_new_day()

        order_value = abs(order.quantity * price)
        current_position_value = (current_positions or {}).get(order.symbol, 0.0)

        # 1. Min price check
        if price < self.config.min_price:
            reason = (
                f"Price ${price:.2f} below minimum ${self.config.min_price:.2f} "
                f"for {order.symbol}"
            )
            logger.warning("SAFETY BLOCK: %s", reason)
            return False, reason

        # 2. Single order value limit
        if order_value > self.config.max_single_order_value:
            reason = (
                f"Order value ${order_value:,.0f} exceeds max "
                f"${self.config.max_single_order_value:,.0f} for {order.symbol}"
            )
            logger.warning("SAFETY BLOCK: %s", reason)
            return False, reason

        # 3. Single order share limit
        if abs(order.quantity) > self.config.max_single_order_shares:
            reason = (
                f"Order quantity {abs(order.quantity):.0f} exceeds max "
                f"{self.config.max_single_order_shares} shares for {order.symbol}"
            )
            logger.warning("SAFETY BLOCK: %s", reason)
            return False, reason

        # 4. Daily cumulative value limit
        projected_daily = self.daily.total_value_traded + order_value
        if projected_daily > self.config.max_daily_trade_value:
            reason = (
                f"Daily trade value would reach ${projected_daily:,.0f}, "
                f"exceeding limit ${self.config.max_daily_trade_value:,.0f}"
            )
            logger.warning("SAFETY BLOCK: %s", reason)
            return False, reason

        # 5. Position concentration check
        if portfolio_value > 0:
            if order.side == "sell":
                projected_position = max(0, current_position_value - order_value)
            else:
                projected_position = current_position_value + order_value
            position_pct = projected_position / portfolio_value
            if position_pct > self.config.max_position_pct_of_portfolio:
                reason = (
                    f"Position in {order.symbol} would be {position_pct:.1%} of portfolio, "
                    f"exceeding max {self.config.max_position_pct_of_portfolio:.1%}"
                )
                logger.warning("SAFETY BLOCK: %s", reason)
                return False, reason

        # 6. Liquidity / ADV check
        if avg_daily_volume is not None and avg_daily_volume > 0:
            adv_fraction = abs(order.quantity) / avg_daily_volume
            if adv_fraction > self.config.max_adv_fraction:
                reason = (
                    f"Order for {order.symbol} is {adv_fraction:.2%} of ADV "
                    f"({avg_daily_volume:.0f}), exceeding max "
                    f"{self.config.max_adv_fraction:.2%}"
                )
                logger.warning("SAFETY BLOCK: %s", reason)
                return False, reason

        return True, "passed"

    def record_fill(self, order_value: float, realised_pnl: float = 0.0):
        """Update daily tracker after a fill."""
        self.daily.reset_if_new_day()
        self.daily.total_value_traded += abs(order_value)
        self.daily.total_realised_pnl += realised_pnl
        self.daily.orders_filled += 1

    def record_submission(self):
        self.daily.reset_if_new_day()
        self.daily.orders_submitted += 1

    def record_rejection(self):
        self.daily.reset_if_new_day()
        self.daily.orders_rejected += 1

    def check_daily_loss_limit(self, unrealised_pnl: float = 0.0) -> tuple[bool, str]:
        """Check if daily loss limit has been breached."""
        self.daily.reset_if_new_day()
        total_pnl = self.daily.total_realised_pnl + unrealised_pnl
        if total_pnl < -self.config.max_daily_loss:
            reason = (
                f"Daily loss ${abs(total_pnl):,.0f} exceeds limit "
                f"${self.config.max_daily_loss:,.0f}"
            )
            logger.error("DAILY LOSS LIMIT BREACHED: %s", reason)
            return False, reason
        return True, "within limit"


class PostTradeReconciler:
    """Compares strategy target positions against actual broker positions.

    Call reconcile() after each rebalance to detect drift and discrepancies.
    """

    def __init__(self, drift_warn_pct: float = 0.02, drift_alert_pct: float = 0.05):
        self.drift_warn_pct = drift_warn_pct
        self.drift_alert_pct = drift_alert_pct

    def reconcile(
        self,
        target_weights: pd.Series,
        actual_positions: pd.Series,
        prices: dict[str, float],
        portfolio_value: float,
    ) -> pd.DataFrame:
        """Compare target weights vs actual positions.

        Returns a DataFrame with columns:
            target_weight, actual_weight, drift, shares_actual, shares_target
        """
        all_symbols = sorted(
            set(target_weights.index) | set(actual_positions.index)
        )

        rows = []
        for sym in all_symbols:
            target_w = target_weights.get(sym, 0.0)
            actual_shares = actual_positions.get(sym, 0.0)
            price = prices.get(sym, 0.0)
            actual_value = actual_shares * price
            actual_w = actual_value / portfolio_value if portfolio_value > 0 else 0.0
            drift = actual_w - target_w

            target_shares = (
                int(target_w * portfolio_value / price) if price > 0 else 0
            )

            rows.append({
                "symbol": sym,
                "target_weight": target_w,
                "actual_weight": actual_w,
                "drift": drift,
                "drift_abs": abs(drift),
                "shares_actual": actual_shares,
                "shares_target": target_shares,
                "shares_diff": actual_shares - target_shares,
            })

        df = pd.DataFrame(rows).set_index("symbol").sort_values(
            "drift_abs", ascending=False
        )

        # Log discrepancies
        for sym, row in df.iterrows():
            if row["drift_abs"] >= self.drift_alert_pct:
                logger.error(
                    "RECONCILIATION ALERT: %s drift=%.2f%% "
                    "(target=%.2f%%, actual=%.2f%%, diff=%+d shares)",
                    sym,
                    row["drift"] * 100,
                    row["target_weight"] * 100,
                    row["actual_weight"] * 100,
                    int(row["shares_diff"]),
                )
            elif row["drift_abs"] >= self.drift_warn_pct:
                logger.warning(
                    "RECONCILIATION WARN: %s drift=%.2f%% "
                    "(target=%.2f%%, actual=%.2f%%)",
                    sym,
                    row["drift"] * 100,
                    row["target_weight"] * 100,
                    row["actual_weight"] * 100,
                )

        total_drift = df["drift_abs"].sum()
        logger.info(
            "Reconciliation complete: %d positions, total absolute drift=%.2f%%",
            len(df),
            total_drift * 100,
        )

        return df


class TWAPSplitter:
    """Splits large orders into smaller child orders for TWAP execution.

    If an order exceeds `adv_threshold` fraction of average daily volume,
    it is split into `n_slices` child orders to be executed over `duration_minutes`.
    """

    def __init__(
        self,
        adv_threshold: float = 0.01,
        n_slices: int = 5,
        duration_minutes: int = 30,
    ):
        self.adv_threshold = adv_threshold
        self.n_slices = n_slices
        self.duration_minutes = duration_minutes

    def should_split(self, quantity: float, avg_daily_volume: float) -> bool:
        """Check if order is large enough to warrant TWAP splitting."""
        if avg_daily_volume <= 0:
            return False
        return abs(quantity) / avg_daily_volume > self.adv_threshold

    def split_order(self, order, avg_daily_volume: float) -> list:
        """Split a large order into child slices.

        Returns a list of (child_order, delay_seconds) tuples.
        Each child_order is a new Order with reduced quantity.
        """
        from quant.execution.broker import Order

        if not self.should_split(order.quantity, avg_daily_volume):
            return [(order, 0)]

        total_qty = order.quantity
        slice_qty = int(total_qty / self.n_slices)
        remainder = int(total_qty - slice_qty * self.n_slices)
        delay_per_slice = (self.duration_minutes * 60) / self.n_slices

        slices = []
        for i in range(self.n_slices):
            qty = slice_qty + (1 if i < remainder else 0)
            if qty <= 0:
                continue
            child = Order(
                symbol=order.symbol,
                side=order.side,
                quantity=qty,
                order_type=order.order_type,
                limit_price=order.limit_price,
            )
            delay = int(i * delay_per_slice)
            slices.append((child, delay))

        logger.info(
            "TWAP split: %s %s %d shares -> %d slices over %d min "
            "(%.2f%% of ADV %.0f)",
            order.side.upper(),
            order.symbol,
            total_qty,
            len(slices),
            self.duration_minutes,
            (total_qty / avg_daily_volume) * 100,
            avg_daily_volume,
        )

        return slices


class ExecutionLogger:
    """Structured JSON logging for all trade events.

    Writes one JSON object per line to a dedicated trade log file.
    This enables downstream monitoring, alerting, and execution quality analysis.
    """

    def __init__(self, log_path: str = "logs/trade_events.jsonl"):
        from pathlib import Path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path

    def _write(self, event: dict):
        event["timestamp"] = datetime.utcnow().isoformat() + "Z"
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def log_order_submitted(self, order, signal_price: float = None):
        self._write({
            "event": "order_submitted",
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "order_type": order.order_type,
            "limit_price": order.limit_price,
            "signal_price": signal_price,
        })

    def log_order_filled(self, order, signal_price: float = None):
        slippage_bps = None
        if signal_price and order.filled_price and signal_price > 0:
            if order.side == "buy":
                slippage_bps = (
                    (order.filled_price - signal_price) / signal_price
                ) * 10000
            else:
                slippage_bps = (
                    (signal_price - order.filled_price) / signal_price
                ) * 10000

        self._write({
            "event": "order_filled",
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "filled_price": order.filled_price,
            "signal_price": signal_price,
            "slippage_bps": round(slippage_bps, 2) if slippage_bps is not None else None,
            "order_id": order.order_id,
        })

    def log_order_rejected(self, order, reason: str = ""):
        self._write({
            "event": "order_rejected",
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "reason": reason,
        })

    def log_safety_block(self, order, reason: str):
        self._write({
            "event": "safety_block",
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "reason": reason,
        })

    def log_reconciliation(self, drift_df: pd.DataFrame, portfolio_value: float):
        summary = {
            "event": "reconciliation",
            "portfolio_value": portfolio_value,
            "n_positions": len(drift_df),
            "total_abs_drift_pct": round(drift_df["drift_abs"].sum() * 100, 2),
            "max_drift_symbol": drift_df["drift_abs"].idxmax() if len(drift_df) > 0 else None,
            "max_drift_pct": round(drift_df["drift_abs"].max() * 100, 2) if len(drift_df) > 0 else 0,
        }
        self._write(summary)

    def log_rebalance_start(self, portfolio_value: float, n_orders: int):
        self._write({
            "event": "rebalance_start",
            "portfolio_value": portfolio_value,
            "n_orders": n_orders,
        })

    def log_rebalance_complete(
        self, n_filled: int, n_rejected: int, total_value_traded: float
    ):
        self._write({
            "event": "rebalance_complete",
            "n_filled": n_filled,
            "n_rejected": n_rejected,
            "total_value_traded": total_value_traded,
        })
