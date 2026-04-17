"""Order management and broker interface.

Provides a paper-trading broker for simulation and a base class for
plugging in live broker APIs (e.g., Alpaca, Interactive Brokers).

Safety integration: all orders pass through PreTradeCheck before submission.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Order:
    symbol: str
    side: str          # "buy" or "sell"
    quantity: float
    order_type: str    # "market", "limit"
    limit_price: float = None
    status: str = "pending"
    filled_price: float = None
    filled_at: datetime = None
    order_id: str = ""
    signal_price: float = None     # price at signal time (for slippage tracking)
    reject_reason: str = ""        # reason if rejected by safety or broker


class BaseBroker(ABC):
    """Abstract broker interface."""

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        ...

    @abstractmethod
    def get_positions(self) -> pd.Series:
        ...

    @abstractmethod
    def get_portfolio_value(self) -> float:
        ...

    @abstractmethod
    def get_cash(self) -> float:
        ...


class PaperBroker(BaseBroker):
    """Simulated broker for paper trading and strategy validation."""

    def __init__(self, initial_capital: float = 1_000_000,
                 slippage_bps: float = 5, txn_cost_bps: float = 10):
        self.cash = initial_capital
        self.positions: dict[str, float] = {}  # symbol -> shares
        self.slippage_bps = slippage_bps
        self.txn_cost_bps = txn_cost_bps
        self.order_log: list[Order] = []
        self._prices: dict[str, float] = {}
        self._order_counter = 0

    def update_prices(self, prices: dict[str, float]):
        """Feed latest prices into the paper broker."""
        self._prices = prices

    def submit_order(self, order: Order) -> Order:
        price = self._prices.get(order.symbol)
        if price is None:
            order.status = "rejected"
            logger.warning("No price for %s, order rejected", order.symbol)
            return order

        # Apply slippage
        slip = price * self.slippage_bps / 10000
        if order.side == "buy":
            fill_price = price + slip
        else:
            fill_price = price - slip

        if order.order_type == "limit" and order.limit_price is not None:
            buy_miss = order.side == "buy" and fill_price > order.limit_price
            sell_miss = order.side == "sell" and fill_price < order.limit_price
            if buy_miss or sell_miss:
                order.status = "unfilled"
                logger.info(
                    "Limit order not filled: %s %s %.0f @ %.2f (market %.2f)",
                    order.side.upper(),
                    order.symbol,
                    order.quantity,
                    order.limit_price,
                    fill_price,
                )
                return order

        trade_value = fill_price * order.quantity
        cost = trade_value * self.txn_cost_bps / 10000

        if order.side == "buy":
            total_cost = trade_value + cost
            if total_cost > self.cash:
                order.status = "rejected"
                logger.warning("Insufficient cash for %s", order.symbol)
                return order
            self.cash -= total_cost
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.quantity
        else:
            current = self.positions.get(order.symbol, 0)
            if order.quantity > current:
                order.status = "rejected"
                logger.warning("Insufficient shares for %s", order.symbol)
                return order
            self.cash += trade_value - cost
            self.positions[order.symbol] = current - order.quantity
            if self.positions[order.symbol] == 0:
                del self.positions[order.symbol]

        order.status = "filled"
        order.filled_price = fill_price
        order.filled_at = datetime.now()
        self._order_counter += 1
        order.order_id = f"PAPER-{self._order_counter:06d}"
        self.order_log.append(order)

        logger.info("Filled: %s %s %.0f shares @ %.2f",
                     order.side.upper(), order.symbol, order.quantity, fill_price)
        return order

    def get_positions(self) -> pd.Series:
        return pd.Series(self.positions, dtype=float)

    def get_portfolio_value(self) -> float:
        pos_value = sum(
            shares * self._prices.get(sym, 0)
            for sym, shares in self.positions.items()
        )
        return self.cash + pos_value

    def get_cash(self) -> float:
        return self.cash


def generate_rebalance_orders(
    current_positions: pd.Series,
    target_weights: pd.Series,
    portfolio_value: float,
    prices: dict[str, float],
    order_type: str = "market",
    limit_offset_bps: float = 0,
) -> list[Order]:
    """Generate the orders needed to move from current to target portfolio.

    Parameters
    ----------
    order_type : str
        "market" or "limit". If "limit", limit_offset_bps is applied to the
        current price (added for buys, subtracted for sells) to set the limit.
    limit_offset_bps : float
        Basis points offset from current price for limit orders. E.g. 10 means
        willing to pay up to 0.1% above current price on buys.
    """
    orders = []
    all_symbols = set(current_positions.index) | set(target_weights.index)

    for sym in all_symbols:
        current_shares = current_positions.get(sym, 0)
        target_value = portfolio_value * target_weights.get(sym, 0)
        price = prices.get(sym)

        if price is None or price <= 0:
            logger.warning("Skipping %s: no valid price", sym)
            continue

        target_shares = int(target_value / price)
        delta = target_shares - current_shares

        if delta == 0:
            continue

        side = "buy" if delta > 0 else "sell"
        qty = abs(delta)

        limit_price = None
        otype = order_type
        if order_type == "limit" and limit_offset_bps > 0:
            offset = price * limit_offset_bps / 10000
            if side == "buy":
                limit_price = round(price + offset, 2)
            else:
                limit_price = round(price - offset, 2)
        elif order_type == "limit":
            # No offset: use current price as limit
            limit_price = round(price, 2)

        order = Order(
            symbol=sym,
            side=side,
            quantity=qty,
            order_type=otype,
            limit_price=limit_price,
            signal_price=price,
        )
        orders.append(order)

    # Sort: sells first (free up cash), then buys
    orders.sort(key=lambda o: (0 if o.side == "sell" else 1, o.symbol))

    return orders
