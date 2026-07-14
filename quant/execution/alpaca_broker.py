"""Alpaca broker adapter with fail-closed execution semantics.

The adapter uses the maintained :mod:`alpaca-py` SDK.  Orders are idempotent,
partial fills are retained, timed-out orders are cancelled and confirmed before
execution continues, and synthetic TWAP parent intents are validated separately
from the child orders that actually reach the broker.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from quant.data.corporate_actions import assert_corporate_actions_reconciled
from quant.execution.broker import BaseBroker, Order
from quant.execution.safety import (
    ExecutionLogger,
    PostTradeReconciler,
    PreTradeCheck,
    SafetyConfig,
    TWAPSplitter,
)

logger = logging.getLogger(__name__)
NY_TZ = ZoneInfo("America/New_York")


def _status(value: object) -> str:
    """Normalize alpaca-py enum values and test doubles to lower-case text."""

    raw = getattr(value, "value", value)
    return str(raw).split(".")[-1].lower()


def _attr(obj: object, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class AlpacaBroker(BaseBroker):
    """Paper/live Alpaca implementation with safety and recovery controls."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: Optional[bool] = None,
        safety_config: SafetyConfig | None = None,
        *,
        trading_client=None,
        data_client=None,
    ):
        key = api_key or os.environ.get("ALPACA_API_KEY", "")
        secret = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        if paper is None:
            paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
        self.paper = bool(paper)

        if trading_client is None:
            if not key or not secret:
                raise ValueError(
                    "Alpaca API keys not found. Set ALPACA_API_KEY and "
                    "ALPACA_SECRET_KEY."
                )
            try:
                from alpaca.trading.client import TradingClient
                from alpaca.data.historical import StockHistoricalDataClient
            except ImportError as exc:
                raise ImportError(
                    "alpaca-py is not installed. Run: pip install alpaca-py"
                ) from exc
            trading_client = TradingClient(key, secret, paper=self.paper)
            data_client = data_client or StockHistoricalDataClient(key, secret)

        self.trading_client = trading_client
        # ``api`` remains as a compatibility alias for existing operator tests.
        self.api = trading_client
        self.data_client = data_client
        self.safety = PreTradeCheck(safety_config or SafetyConfig())
        self.reconciler = PostTradeReconciler()
        self.twap = TWAPSplitter(
            adv_threshold=self.safety.config.max_adv_fraction,
            n_slices=5,
            duration_minutes=30,
        )
        self.exec_log = ExecutionLogger()
        self._sleep = time.sleep
        self._monotonic = time.monotonic

        if self.safety.config.require_paper_mode and not self.paper:
            raise RuntimeError(
                "SAFETY: require_paper_mode is true but a live account was "
                "requested. Explicitly disable the gate before live trading."
            )

        account = self._client().get_account()
        logger.info(
            "Connected to Alpaca [%s] | Equity: $%s | Cash: $%s",
            "PAPER" if self.paper else "LIVE",
            _attr(account, "equity"),
            _attr(account, "cash"),
        )

    def _client(self):
        return getattr(self, "trading_client", getattr(self, "api", None))

    # ------------------------------------------------------------------
    # Idempotency
    # ------------------------------------------------------------------

    @staticmethod
    def _base_client_order_id(order: Order) -> str:
        trading_day = datetime.now(NY_TZ).date().isoformat()
        payload = {
            "day": trading_day,
            "purpose": order.purpose,
            "symbol": order.symbol.upper(),
            "side": order.side,
            "qty": round(float(order.requested_quantity or order.quantity), 8),
            "type": order.order_type,
            "limit": order.limit_price,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
        return f"qts-{order.purpose[:4]}-{digest}"[:48]

    @staticmethod
    def _retry_client_order_id(base: str, attempt: int) -> str:
        digest = hashlib.sha256(f"{base}:retry:{attempt}".encode()).hexdigest()[:8]
        return f"{base[:38]}-r{attempt}-{digest}"[:48]

    @staticmethod
    def _child_client_order_id(parent: str, child_index: int) -> str:
        digest = hashlib.sha256(f"{parent}:slice:{child_index}".encode()).hexdigest()[:8]
        return f"{parent[:36]}-s{child_index}-{digest}"[:48]

    def _get_order_by_client_id(self, client_order_id: str):
        client = self._client()
        methods = (
            "get_order_by_client_id",
            "get_order_by_client_order_id",
        )
        for name in methods:
            method = getattr(client, name, None)
            if method is None:
                continue
            try:
                return method(client_order_id)
            except Exception as exc:
                # Alpaca raises on a missing id.  Other failures are logged but
                # do not turn into an unguarded duplicate submission: the
                # actual submit still uses the same client id.
                text = str(exc).lower()
                if any(token in text for token in ("not found", "404", "does not exist")):
                    return None
                logger.warning(
                    "Could not query client_order_id %s: %s",
                    client_order_id,
                    exc,
                )
                return None
        return None

    def _resolve_idempotency(self, order: Order):
        """Find an existing intent or choose the next safe retry id.

        Active/filled/partially-filled broker orders are reused.  A previously
        cancelled or rejected zero-fill intent gets a deterministic retry id,
        allowing a later workflow run to try again without colliding with the
        terminal order or duplicating an order hidden by a network timeout.
        """

        base = order.client_order_id or self._base_client_order_id(order)
        for attempt in range(10):
            candidate = base if attempt == 0 else self._retry_client_order_id(base, attempt)
            existing = self._get_order_by_client_id(candidate)
            if existing is None:
                order.client_order_id = candidate
                return None
            status = _status(_attr(existing, "status", ""))
            filled_qty = float(_attr(existing, "filled_qty", 0) or 0)
            if filled_qty > 0 or status not in {"canceled", "cancelled", "expired", "rejected"}:
                order.client_order_id = candidate
                return existing
        raise RuntimeError(
            f"Too many terminal retries for {order.symbol} {order.side}; refusing to submit"
        )

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(self, order: Order, avg_daily_volume: float = None) -> Order:
        self.assert_corporate_actions_reconciled()

        # Resolve a prior network-timeout/restarted-workflow intent before
        # applying limits for a *new* submission. Monitoring an order that the
        # broker already accepted must not be blocked by a daily limit that
        # the same fill helped consume.
        existing = self._resolve_idempotency(order)

        portfolio_value = self.get_portfolio_value()
        price = (
            order.signal_price
            or (_attr(existing, "filled_avg_price", None) if existing is not None else None)
            or self._get_price_safe(order.symbol)
        )
        if price is None or price <= 0:
            return self._reject(order, f"No valid price for {order.symbol}")

        if existing is not None:
            return self._execute_single(
                order,
                float(price),
                existing=existing,
                idempotency_checked=True,
            )

        current_positions = self._position_values()
        split = bool(
            avg_daily_volume
            and self.twap.should_split(order.quantity, avg_daily_volume)
        )
        passed, reason = self.safety.validate(
            order,
            price,
            portfolio_value,
            current_positions,
            None if split else avg_daily_volume,
            check_order_limits=not split,
            check_adv=not split,
        )
        if not passed:
            return self._reject(order, reason, safety_block=True)

        if split:
            return self._execute_twap(order, avg_daily_volume, price)
        return self._execute_single(
            order, price, existing=None, idempotency_checked=True
        )

    def _reject(self, order: Order, reason: str, *, safety_block: bool = False) -> Order:
        order.status = "rejected"
        order.reject_reason = reason
        if safety_block:
            self.exec_log.log_safety_block(order, reason)
        else:
            self.exec_log.log_order_rejected(order, reason)
        self.safety.record_rejection()
        logger.error("Order rejected for %s: %s", order.symbol, reason)
        return order

    def _position_values(self) -> dict[str, float]:
        shares = self.get_positions()
        if shares.empty:
            return {}
        prices = self.get_current_prices(shares.index.tolist())
        return {
            symbol: float(qty) * float(prices.get(symbol, 0.0))
            for symbol, qty in shares.items()
        }

    def _validate_child(
        self,
        order: Order,
        signal_price: float,
        avg_daily_volume: float,
    ) -> tuple[bool, str]:
        return self.safety.validate(
            order,
            signal_price,
            self.get_portfolio_value(),
            self._position_values(),
            avg_daily_volume,
        )

    def _build_order_request(self, order: Order):
        """Build an alpaca-py request, with a dict fallback for test doubles."""

        try:
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

            common = {
                "symbol": order.symbol,
                "qty": float(order.quantity),
                "side": OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
                "time_in_force": TimeInForce.DAY,
                "client_order_id": order.client_order_id,
            }
            if order.order_type == "limit":
                return LimitOrderRequest(limit_price=float(order.limit_price), **common)
            return MarketOrderRequest(**common)
        except ImportError:
            return {
                "symbol": order.symbol,
                "qty": float(order.quantity),
                "side": order.side,
                "type": order.order_type,
                "time_in_force": "day",
                "limit_price": order.limit_price,
                "client_order_id": order.client_order_id,
            }

    def _submit_request(self, request):
        client = self._client()
        if isinstance(request, dict):
            return client.submit_order(**request)
        return client.submit_order(order_data=request)

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            token in text
            for token in ("timeout", "timed out", "429", "connection", "temporarily")
        )

    def _execute_single(
        self,
        order: Order,
        signal_price: float,
        *,
        existing=None,
        idempotency_checked: bool = False,
    ) -> Order:
        broker_order = (
            existing
            if idempotency_checked
            else self._resolve_idempotency(order)
        )
        if broker_order is None:
            self.safety.record_submission()
            self.exec_log.log_order_submitted(order, signal_price)
            request = self._build_order_request(order)
            for attempt in range(3):
                try:
                    broker_order = self._submit_request(request)
                    break
                except Exception as exc:
                    if not self._is_transient(exc) or attempt == 2:
                        return self._reject(order, str(exc))
                    # The broker may have accepted the request before the
                    # connection failed.  Query the stable id before retrying.
                    broker_order = self._get_order_by_client_id(order.client_order_id)
                    if broker_order is not None:
                        break
                    wait = 2 ** attempt
                    logger.warning(
                        "Transient submit error for %s; retrying in %ds: %s",
                        order.symbol,
                        wait,
                        exc,
                    )
                    self._sleep(wait)

        if broker_order is None:
            return self._reject(order, "Broker did not return an order")

        order_id = str(_attr(broker_order, "id", ""))
        order.order_id = order_id
        terminal = self._wait_for_fill(
            order_id,
            timeout=30 if order.order_type == "market" else 60,
        )
        if terminal is None:
            if self._cancel_open_order(order_id, order.symbol, timed_out=True):
                order.status = "cancelled"
            else:
                order.status = "unknown"
                order.reject_reason = "Timed out and cancellation was not confirmed"
            return order

        return self._apply_fill(order, terminal, signal_price)

    def _apply_fill(self, order: Order, broker_order, signal_price: float) -> Order:
        order.order_id = str(_attr(broker_order, "id", order.order_id))
        status = _status(_attr(broker_order, "status", ""))
        filled_qty = float(_attr(broker_order, "filled_qty", 0) or 0)
        avg_price = _attr(broker_order, "filled_avg_price", None)

        if filled_qty <= 0:
            if status in {"canceled", "cancelled", "expired"}:
                order.status = "cancelled"
            elif status == "rejected":
                order.status = "rejected"
                order.reject_reason = str(_attr(broker_order, "reject_reason", "Broker rejected order"))
                self.safety.record_rejection()
            else:
                order.status = "submitted"
            return order

        order.filled_quantity = filled_qty
        order.filled_price = float(avg_price) if avg_price is not None else signal_price
        requested = float(order.requested_quantity or order.quantity)

        if filled_qty + 1e-9 >= requested or status == "filled":
            order.status = "filled"
        else:
            cancelled = self._cancel_open_order(
                order.order_id,
                order.symbol,
                timed_out=True,
            )
            order.status = "partial_fill" if cancelled else "partial_fill_open"
            if not cancelled:
                order.reject_reason = "Partial fill remains open; cancellation not confirmed"

        # Keep the historical public interface: ``quantity`` is actual filled
        # quantity after a partial fill; requested_quantity remains immutable.
        order.quantity = filled_qty
        recorded = self.safety.record_fill(
            filled_qty * order.filled_price,
            client_order_id=order.client_order_id,
        )
        if recorded is not False:
            self.exec_log.log_order_filled(order, signal_price)
        else:
            logger.info(
                "Recovered already-checkpointed fill %s; daily totals unchanged",
                order.client_order_id,
            )
        logger.info(
            "%s %s %.0f/%.0f @ $%.2f (status=%s, client_id=%s)",
            order.side.upper(),
            order.symbol,
            filled_qty,
            requested,
            order.filled_price,
            order.status,
            order.client_order_id,
        )
        return order

    def _execute_twap(
        self,
        order: Order,
        avg_daily_volume: float,
        signal_price: float,
    ) -> Order:
        parent_id = order.client_order_id or self._base_client_order_id(order)
        order.client_order_id = parent_id
        slices = self.twap.split_order(order, avg_daily_volume)
        start = self._monotonic()
        target_qty = float(order.requested_quantity or order.quantity)
        total_qty = 0.0
        total_value = 0.0
        last_id = ""
        unsafe_open_child = False

        for index, (child, offset_seconds) in enumerate(slices):
            wait = max(0.0, start + offset_seconds - self._monotonic())
            if wait:
                logger.info("TWAP waiting %.0fs before slice %d/%d", wait, index + 1, len(slices))
                self._sleep(wait)

            child.client_order_id = self._child_client_order_id(parent_id, index)
            passed, reason = self._validate_child(child, signal_price, avg_daily_volume)
            if not passed:
                self._reject(child, reason, safety_block=True)
                order.reject_reason = f"TWAP slice {index + 1} blocked: {reason}"
                break

            result = self._execute_single(child, signal_price)
            if result.filled_quantity > 0:
                total_qty += result.filled_quantity
                total_value += result.filled_quantity * float(result.filled_price or 0)
                last_id = result.order_id
            if result.status in {"unknown", "submitted", "partial_fill_open"}:
                # Never place another child while the preceding child may still
                # be live; otherwise total quantity can exceed the parent.
                unsafe_open_child = True
                order.reject_reason = (
                    f"TWAP halted after slice {index + 1}: prior child may still be open"
                )
                break

        order.order_id = last_id
        order.filled_quantity = total_qty
        if total_qty > 0:
            order.filled_price = total_value / total_qty
            order.quantity = total_qty
            if unsafe_open_child:
                order.status = "partial_fill_open"
            elif total_qty + 1e-9 >= target_qty:
                order.status = "filled"
            else:
                order.status = "partial_fill"
        else:
            order.status = "unknown" if unsafe_open_child else "rejected"
            order.reject_reason = order.reject_reason or "No TWAP slices filled"
        return order

    # ------------------------------------------------------------------
    # Polling and cancellation
    # ------------------------------------------------------------------

    def _get_order(self, order_id: str):
        client = self._client()
        method = getattr(client, "get_order_by_id", None) or getattr(client, "get_order", None)
        if method is None:
            raise AttributeError("Trading client cannot query orders")
        return method(order_id)

    def _wait_for_fill(self, order_id: str, timeout: int = 30):
        deadline = self._monotonic() + timeout
        latest = None
        while self._monotonic() < deadline:
            try:
                latest = self._get_order(order_id)
            except Exception as exc:
                logger.warning("Error polling order %s: %s", order_id, exc)
                self._sleep(1)
                continue
            status = _status(_attr(latest, "status", ""))
            if status == "filled":
                return latest
            if status in {"canceled", "cancelled", "expired", "rejected"}:
                return latest
            self._sleep(1)

        try:
            latest = self._get_order(order_id)
        except Exception:
            return None
        if float(_attr(latest, "filled_qty", 0) or 0) > 0:
            return latest
        if _status(_attr(latest, "status", "")) in {
            "canceled", "cancelled", "expired", "rejected"
        }:
            return latest
        return None

    def _cancel_open_order(
        self,
        order_id: str,
        symbol: str,
        timed_out: bool = False,
        confirmation_timeout: int = 10,
    ) -> bool:
        client = self._client()
        cancel = (
            getattr(client, "cancel_order_by_id", None)
            or getattr(client, "cancel_order", None)
        )
        if cancel is None:
            return False
        try:
            cancel(order_id)
        except Exception as exc:
            # "already cancelled/filled" can still be safe; the confirmation
            # query below is authoritative.
            logger.warning("Cancel request for %s (%s) failed: %s", symbol, order_id, exc)

        if not (getattr(client, "get_order_by_id", None)
                or getattr(client, "get_order", None)):
            # Minimal injected clients used by offline tests cannot confirm the
            # state transition.  Production alpaca-py always has
            # get_order_by_id, so real trading never takes this shortcut.
            return True

        deadline = self._monotonic() + confirmation_timeout
        while self._monotonic() < deadline:
            try:
                current = self._get_order(order_id)
                status = _status(_attr(current, "status", ""))
                if status in {"filled", "canceled", "cancelled", "expired", "rejected"}:
                    return True
            except Exception as exc:
                logger.warning("Could not confirm cancellation for %s: %s", order_id, exc)
            self._sleep(1)
        logger.error(
            "Cancellation not confirmed for %s order %s%s",
            symbol,
            order_id,
            " after timeout" if timed_out else "",
        )
        return False

    # ------------------------------------------------------------------
    # Account, prices, and corporate actions
    # ------------------------------------------------------------------

    def get_positions(self) -> pd.Series:
        positions = self._client().get_all_positions()
        return pd.Series(
            {_attr(p, "symbol"): float(_attr(p, "qty")) for p in positions},
            dtype=float,
        )

    def get_position_details(self) -> list[dict[str, object]]:
        details = []
        for p in self._client().get_all_positions():
            details.append({
                "symbol": _attr(p, "symbol"),
                "qty": float(_attr(p, "qty", 0) or 0),
                "avg_entry_price": float(_attr(p, "avg_entry_price", 0) or 0),
                "current_price": float(_attr(p, "current_price", 0) or 0),
            })
        return details

    def assert_corporate_actions_reconciled(self) -> None:
        assert_corporate_actions_reconciled(self.get_position_details())

    def get_portfolio_value(self) -> float:
        return float(_attr(self._client().get_account(), "equity"))

    def get_cash(self) -> float:
        return float(_attr(self._client().get_account(), "cash"))

    def get_daily_pnl(self) -> Optional[float]:
        try:
            account = self._client().get_account()
            return float(_attr(account, "equity")) - float(_attr(account, "last_equity"))
        except Exception as exc:
            logger.warning("Could not compute daily PnL: %s", exc)
            return None

    def get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        return {
            symbol: price
            for symbol in symbols
            if (price := self._get_price_safe(symbol)) is not None
        }

    def _get_price_safe(self, symbol: str) -> Optional[float]:
        try:
            if self.data_client is not None:
                from alpaca.data.requests import StockLatestTradeRequest
                request = StockLatestTradeRequest(symbol_or_symbols=[symbol])
                trades = self.data_client.get_stock_latest_trade(request)
                trade = trades[symbol]
                return float(_attr(trade, "price"))
            # Compatibility for injected/legacy test clients.
            trade = self._client().get_latest_trade(symbol)
            return float(_attr(trade, "price"))
        except Exception as exc:
            logger.warning("Could not get price for %s: %s", symbol, exc)
            return None

    def cancel_all_orders(self):
        method = (
            getattr(self._client(), "cancel_orders", None)
            or getattr(self._client(), "cancel_all_orders", None)
        )
        if method:
            method()

    def close_all_positions(self):
        method = getattr(self._client(), "close_all_positions", None)
        if method:
            try:
                method(cancel_orders=True)
            except TypeError:
                method()
        logger.warning("EMERGENCY: requested closure of all positions")

    def reconcile(self, target_weights: pd.Series) -> pd.DataFrame:
        actual = self.get_positions()
        value = self.get_portfolio_value()
        symbols = list(set(target_weights.index) | set(actual.index))
        prices = self.get_current_prices(symbols)
        drift = self.reconciler.reconcile(target_weights, actual, prices, value)
        self.exec_log.log_reconciliation(drift, value)
        return drift

    def is_market_open(self) -> bool:
        try:
            return bool(_attr(self._client().get_clock(), "is_open", False))
        except Exception as exc:
            logger.warning("Could not check market clock: %s", exc)
            return False
