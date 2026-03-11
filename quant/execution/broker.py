"""Order management and broker interface.

Provides a paper-trading broker for simulation and a base class for
plugging in live broker APIs (e.g., Alpaca, Interactive Brokers).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

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


def generate_rebalance_orders(current_positions: pd.Series,
                              target_weights: pd.Series,
                              portfolio_value: float,
                              prices: dict[str, float]) -> list[Order]:
    """Generate the orders needed to move from current to target portfolio."""
    orders = []
    all_symbols = set(current_positions.index) | set(target_weights.index)

    for sym in all_symbols:
        current_shares = current_positions.get(sym, 0)
        target_value = portfolio_value * target_weights.get(sym, 0)
        price = prices.get(sym)

        if price is None or price <= 0:
            continue

        target_shares = int(target_value / price)
        delta = target_shares - current_shares

        if delta > 0:
            orders.append(Order(symbol=sym, side="buy", quantity=delta, order_type="market"))
        elif delta < 0:
            orders.append(Order(symbol=sym, side="sell", quantity=abs(delta), order_type="market"))

    return orders
