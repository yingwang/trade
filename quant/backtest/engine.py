"""Event-driven backtest engine with explicit signal/execution timing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest output metrics and time series."""

    equity_curve: pd.Series = field(default_factory=pd.Series)
    benchmark_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    positions_history: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=" * 60, "BACKTEST RESULTS", "=" * 60]
        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key:30s}: {value:>12.4f}")
            else:
                lines.append(f"  {key:30s}: {value!s:>12s}")
        lines.append("=" * 60)
        return "\n".join(lines)


class BacktestEngine:
    """Daily backtester.

    Signals are observed at date ``T`` close.  Rebalances and stop-losses are
    executed no earlier than the next bar, at that bar's open when an explicit
    ``execution_prices`` frame is supplied.  A missing execution price leaves
    the instruction pending; the engine never substitutes the previous close
    as a fictitious fill.
    """

    def __init__(self, config: dict):
        bt_cfg = config["backtest"]
        port_cfg = config["portfolio"]
        self.initial_capital = float(bt_cfg["initial_capital"])
        self.slippage_bps = float(bt_cfg["slippage_bps"])
        self.txn_cost_bps = float(port_cfg["transaction_cost_bps"])
        self.rebalance_freq = int(port_cfg["rebalance_frequency_days"])
        self.margin_rate = float(config.get("leverage", {}).get("margin_annual_rate", 0.0))
        self.stop_loss_pct = float(config.get("risk", {}).get("stop_loss_pct", 0.0))
        self.annual_risk_free_rate = float(
            bt_cfg.get("risk_free_rate", config.get("risk_free_rate", 0.0))
        )
        self.risk_free_rate = (1.0 + self.annual_risk_free_rate) ** (1.0 / 252.0) - 1.0
        self.impact_coeff = float(bt_cfg.get("market_impact_coeff", 2.5))

    @staticmethod
    def _valid_price(value) -> bool:
        try:
            return bool(np.isfinite(float(value)) and float(value) > 0)
        except (TypeError, ValueError):
            return False

    def _cost(self, trade_value: float, portfolio_value: float) -> float:
        if trade_value <= 0:
            return 0.0
        fixed = trade_value * (self.txn_cost_bps + self.slippage_bps) / 10_000.0
        participation = trade_value / portfolio_value if portfolio_value > 0 else 0.0
        impact = (
            trade_value
            * self.impact_coeff
            * np.sqrt(max(0.0, participation))
            / 10_000.0
        )
        return float(fixed + impact)

    def run(
        self,
        prices: pd.DataFrame,
        target_weights_by_date: dict[str, pd.Series],
        benchmark_col: str = "SPY",
        execution_prices: pd.DataFrame | None = None,
        delisting_returns: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Run a backtest over adjusted closes and optional next-bar opens."""

        result = BacktestResult()
        if prices is None or prices.empty:
            result.metrics = {}
            return result

        prices = prices.sort_index().copy()
        dates = prices.index
        symbols = [column for column in prices.columns if column != benchmark_col]
        close_raw = prices.reindex(columns=symbols).apply(pd.to_numeric, errors="coerce")
        valuation = close_raw.ffill()

        if execution_prices is None:
            execution = close_raw.copy()
            execution_label = "next_close"
        else:
            execution = (
                execution_prices.reindex(index=dates, columns=symbols)
                .apply(pd.to_numeric, errors="coerce")
            )
            # Open is preferred, but a same-day close is a valid next-bar
            # fallback when the vendor omitted only the open.  If both are
            # missing, the instruction stays pending.
            execution = execution.combine_first(close_raw)
            execution_label = "next_open"

        cash = self.initial_capital
        holdings = pd.Series(0.0, index=symbols, dtype=float)
        entry_prices = pd.Series(np.nan, index=symbols, dtype=float)
        equity_history: dict[pd.Timestamp, float] = {}
        pending_target: pd.Series | None = None
        pending_stops: set[str] = set()
        delistings_by_date: dict[pd.Timestamp, list[tuple[str, float]]] = {}
        if delisting_returns is not None and not delisting_returns.empty:
            for row in delisting_returns.itertuples(index=False):
                event_date = pd.Timestamp(row.date)
                delistings_by_date.setdefault(event_date, []).append(
                    (str(row.symbol), float(row.delisting_return))
                )

        rebalance_targets: dict[pd.Timestamp, pd.Series] = {}
        for raw_date, raw_target in target_weights_by_date.items():
            date = pd.Timestamp(raw_date)
            if date in dates:
                rebalance_targets[date] = raw_target.reindex(symbols).fillna(0.0)

        for date in dates:
            close_px = valuation.loc[date]
            exec_px = execution.loc[date]

            # A delisting return is a terminal payoff, not an exchange fill.
            # Apply the supplied event directly and remove the holding; without
            # an external event file the engine makes no invented assumption.
            for symbol, terminal_return in delistings_by_date.get(
                pd.Timestamp(date).normalize(), []
            ):
                if symbol not in holdings or holdings[symbol] <= 0:
                    continue
                reference = valuation.loc[:date, symbol].dropna()
                if reference.empty:
                    raise RuntimeError(
                        f"Cannot apply delisting return for {symbol} on {date}: "
                        "no reference price"
                    )
                payoff = float(holdings[symbol]) * float(reference.iloc[-1]) * (
                    1.0 + terminal_return
                )
                cash += max(0.0, payoff)
                holdings[symbol] = 0.0
                entry_prices[symbol] = np.nan
                pending_stops.discard(symbol)
                result.trades.append(
                    {
                        "date": date,
                        "type": "delisting",
                        "symbols": [symbol],
                        "delisting_return": terminal_return,
                        "proceeds": max(0.0, payoff),
                    }
                )
            open_marks = close_px.copy()
            available_exec = exec_px.map(self._valid_price)
            open_marks.loc[available_exec] = exec_px.loc[available_exec]
            portfolio_at_open = float(cash + (holdings * open_marks).fillna(0).sum())

            # Stop-losses detected at the prior close execute before rebalance
            # activity. Missing bars remain pending.
            executed_stops: list[str] = []
            stop_value = 0.0
            for symbol in sorted(pending_stops):
                qty = float(holdings.get(symbol, 0.0))
                price = exec_px.get(symbol)
                if qty <= 0:
                    executed_stops.append(symbol)
                    continue
                if not self._valid_price(price):
                    continue
                value = qty * float(price)
                cash += value
                stop_value += value
                holdings[symbol] = 0.0
                entry_prices[symbol] = np.nan
                executed_stops.append(symbol)

            if stop_value > 0:
                stop_cost = self._cost(stop_value, portfolio_at_open)
                cash -= stop_cost
                result.trades.append({
                    "date": date,
                    "type": "stop_loss",
                    "symbols": executed_stops.copy(),
                    "turnover": stop_value / portfolio_at_open if portfolio_at_open > 0 else 0.0,
                    "cost": stop_cost,
                    "execution": execution_label,
                })
            pending_stops.difference_update(executed_stops)

            # Stop-loss exits take precedence over portfolio targets. A target
            # computed at the same close can otherwise sell the position as a
            # stop at the next bar and immediately buy it back in the rebalance
            # loop. Pending unpriced stops also force their target weight to
            # zero so unavailable exits block the buy phase.
            blocked_reentries = set(pending_stops) | set(executed_stops)
            if pending_target is not None and blocked_reentries:
                pending_target = pending_target.copy()
                blocked = pending_target.index.intersection(blocked_reentries)
                pending_target.loc[blocked] = 0.0

            # Execute an earlier close's target.  Sells go first.  If a sell is
            # untradeable, buys are deferred to avoid temporary over-exposure.
            if pending_target is not None:
                portfolio_at_open = float(cash + (holdings * open_marks).fillna(0).sum())
                fee_buffer_bps = self.txn_cost_bps + self.slippage_bps + self.impact_coeff
                allocable = portfolio_at_open * max(0.0, 1.0 - fee_buffer_bps / 10_000.0)
                desired = pd.Series(0.0, index=symbols, dtype=float)
                for symbol in symbols:
                    mark = exec_px.get(symbol)
                    if not self._valid_price(mark):
                        mark = close_px.get(symbol)
                    weight = float(pending_target.get(symbol, 0.0))
                    if weight > 0 and self._valid_price(mark):
                        desired[symbol] = np.floor(allocable * weight / float(mark))

                delta = desired - holdings
                required_but_unpriced = {
                    symbol
                    for symbol in symbols
                    if abs(float(delta[symbol])) >= 1.0
                    and not self._valid_price(exec_px.get(symbol))
                }
                unavailable_sells = {
                    symbol for symbol in required_but_unpriced if delta[symbol] < 0
                }

                traded_symbols: list[str] = []
                trade_value = 0.0
                new_positions: dict[str, float] = {}

                for side in ("sell", "buy"):
                    if side == "buy" and unavailable_sells:
                        continue
                    candidates = [
                        symbol
                        for symbol in symbols
                        if (delta[symbol] < -0.5 if side == "sell" else delta[symbol] > 0.5)
                    ]
                    for symbol in candidates:
                        price = exec_px.get(symbol)
                        if not self._valid_price(price):
                            continue
                        qty = abs(float(delta[symbol]))
                        value = qty * float(price)
                        if side == "sell":
                            qty = min(qty, float(holdings[symbol]))
                            value = qty * float(price)
                            holdings[symbol] -= qty
                            cash += value
                            if holdings[symbol] <= 0:
                                holdings[symbol] = 0.0
                                entry_prices[symbol] = np.nan
                        else:
                            was_flat = holdings[symbol] <= 0
                            holdings[symbol] += qty
                            cash -= value
                            if was_flat:
                                new_positions[symbol] = float(price)
                        trade_value += value
                        traded_symbols.append(symbol)

                for symbol, price in new_positions.items():
                    entry_prices[symbol] = price

                if trade_value > 0:
                    cost = self._cost(trade_value, portfolio_at_open)
                    cash -= cost
                    result.trades.append({
                        "date": date,
                        "type": "rebalance",
                        "symbols": sorted(set(traded_symbols)),
                        "turnover": trade_value / portfolio_at_open if portfolio_at_open > 0 else 0.0,
                        "cost": cost,
                        "execution": execution_label,
                    })

                # Recompute remaining instructions after actual fills.  A target
                # that could not trade stays live until a later valid bar.
                remaining = desired - holdings
                unpriced_target = any(
                    float(pending_target.get(symbol, 0.0)) > 0
                    and not self._valid_price(exec_px.get(symbol))
                    and not self._valid_price(close_px.get(symbol))
                    for symbol in symbols
                )
                if (remaining.abs() < 0.5).all() and not unpriced_target:
                    pending_target = None

            # The current close creates a target for a future bar; it never
            # executes on the signal close. A newer target supersedes any stale
            # unfilled target, mirroring a real strategy recomputation.
            if date in rebalance_targets:
                if pending_target is not None:
                    logger.warning("Superseding an unfinished rebalance target on %s", date.date())
                pending_target = rebalance_targets[date].copy()
                blocked = pending_target.index.intersection(blocked_reentries)
                pending_target.loc[blocked] = 0.0

            portfolio_value = float(cash + (holdings * close_px).fillna(0).sum())

            # Detect at close, execute next bar.  Already pending symbols are
            # left untouched until a real execution price appears.
            if self.stop_loss_pct > 0:
                check = (holdings > 0) & entry_prices.notna() & close_px.notna()
                if check.any():
                    pnl = close_px[check] / entry_prices[check] - 1.0
                    newly_stopped = set(pnl[pnl < -self.stop_loss_pct].index)
                    if newly_stopped:
                        logger.info("Stop-loss signal on %s for %s", date.date(), sorted(newly_stopped))
                        pending_stops.update(newly_stopped)
                        if pending_target is not None:
                            blocked = pending_target.index.intersection(newly_stopped)
                            pending_target.loc[blocked] = 0.0

            if cash < 0 and self.margin_rate > 0:
                interest = abs(cash) * self.margin_rate / 252.0
                cash -= interest
                portfolio_value -= interest

            equity_history[date] = portfolio_value
            result.positions_history.append({
                "date": date,
                "holdings": holdings.copy(),
                "cash": cash,
                "value": portfolio_value,
                "pending_stops": sorted(pending_stops),
                "pending_rebalance": pending_target is not None,
            })

        result.equity_curve = pd.Series(equity_history, dtype=float).sort_index()
        if benchmark_col in prices.columns and prices[benchmark_col].notna().any():
            benchmark = pd.to_numeric(prices[benchmark_col], errors="coerce").ffill()
            first = benchmark.dropna().iloc[0]
            result.benchmark_curve = benchmark / first * self.initial_capital

        result.returns = result.equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        result.metrics = self._compute_metrics(result)
        return result

    def _compute_metrics(self, result: BacktestResult) -> dict:
        equity = result.equity_curve.dropna()
        returns = result.returns.dropna()
        if equity.empty:
            return {}

        total_return = equity.iloc[-1] / equity.iloc[0] - 1.0 if len(equity) > 1 else 0.0
        years = len(returns) / 252.0 if len(returns) else 0.0
        cagr = (
            (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
            if years > 0 and equity.iloc[0] > 0 and equity.iloc[-1] > 0
            else 0.0
        )
        annual_vol = returns.std(ddof=1) * np.sqrt(252) if len(returns) > 1 else 0.0
        excess = returns - self.risk_free_rate
        sharpe = (
            excess.mean() / returns.std(ddof=1) * np.sqrt(252)
            if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0
        )

        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_drawdown = float(drawdown.min())
        calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0

        downside = np.minimum(excess.to_numpy(dtype=float), 0.0)
        downside_deviation = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(252)) if len(downside) else 0.0
        annual_excess = float(excess.mean() * 252) if len(excess) else 0.0
        sortino = annual_excess / downside_deviation if downside_deviation > 0 else 0.0

        benchmark_returns = (
            result.benchmark_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            if not result.benchmark_curve.empty else pd.Series(dtype=float)
        )
        aligned = pd.concat(
            [returns.rename("strategy"), benchmark_returns.rename("benchmark")],
            axis=1,
            join="inner",
        ).dropna()
        if len(aligned) > 1:
            active = aligned["strategy"] - aligned["benchmark"]
            information_ratio = (
                active.mean() / active.std(ddof=1) * np.sqrt(252)
                if active.std(ddof=1) > 0 else 0.0
            )
            benchmark_return = (
                result.benchmark_curve.dropna().iloc[-1]
                / result.benchmark_curve.dropna().iloc[0]
                - 1.0
            )
        else:
            information_ratio = 0.0
            benchmark_return = 0.0

        rebalance_turnover = [
            trade["turnover"]
            for trade in result.trades
            if trade.get("type", "rebalance") == "rebalance"
        ]
        return {
            "Total Return": float(total_return),
            "CAGR": float(cagr),
            "Annualized Volatility": float(annual_vol),
            "Sharpe Ratio": float(sharpe),
            "Sortino Ratio": float(sortino),
            "Calmar Ratio": float(calmar),
            "Max Drawdown": max_drawdown,
            "Win Rate (daily)": float((returns > 0).mean()) if len(returns) else 0.0,
            "Avg Rebalance Turnover": float(np.mean(rebalance_turnover)) if rebalance_turnover else 0.0,
            "Information Ratio": float(information_ratio),
            "Benchmark Return": float(benchmark_return),
            "Num Trades": len(result.trades),
        }

    def slice_result(
        self,
        result: BacktestResult,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None = None,
    ) -> BacktestResult:
        """Measure a window from one continuous simulation without rerunning it."""

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end is not None else result.equity_curve.index.max()
        sliced = BacktestResult()
        sliced.equity_curve = result.equity_curve.loc[start_ts:end_ts]
        sliced.benchmark_curve = result.benchmark_curve.loc[start_ts:end_ts]
        sliced.returns = sliced.equity_curve.pct_change().dropna()
        sliced.positions_history = [
            row for row in result.positions_history if start_ts <= row["date"] <= end_ts
        ]
        sliced.trades = [
            trade for trade in result.trades if start_ts <= trade["date"] <= end_ts
        ]
        sliced.metrics = self._compute_metrics(sliced)
        return sliced
