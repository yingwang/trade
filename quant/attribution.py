"""Returns-based attribution for the actual paper-trading accounts.

The account equity curve is the source of truth.  Filled orders reconstruct
end-of-day holdings, adjusted closes measure each holding's economic return,
and a lagged returns-based factor model separates the stock sleeve into market,
style, industry, and security-selection effects.  Cash and implementation
shortfall are explicit.  Anything still not explained (cash flows, dividends,
missing order history, paper-broker corporate-action quirks, or bad prices) is
reported as a reconciliation residual instead of being hidden in selection.

Daily arithmetic contributions reconcile exactly to daily active return once
the residual is included.  Terminal contributions use Carino linking so their
sum also equals the geometric account return minus the geometric benchmark
return over the full reporting period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


COMPONENTS = (
    "market_beta",
    "style",
    "industry",
    "stock_selection",
    "cash_drag",
    "trading_costs",
)
RESIDUAL_COMPONENT = "reconciliation_residual"


@dataclass
class AttributionResult:
    """Daily and linked performance-attribution output."""

    daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    linked_totals: dict[str, float] = field(default_factory=dict)
    style_totals: dict[str, float] = field(default_factory=dict)
    industry_totals: dict[str, float] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-safe dashboard payload."""

        if self.daily.empty:
            return {
                "status": "insufficient_data",
                "summary": self.summary,
                "diagnostics": self.diagnostics,
                "components": {},
                "style_breakdown": {},
                "industry_breakdown": {},
                "daily": [],
            }

        daily_rows = []
        for date, row in self.daily.iterrows():
            values = {
                key: _json_number(value)
                for key, value in row.items()
            }
            values["date"] = pd.Timestamp(date).strftime("%Y-%m-%d")
            daily_rows.append(values)

        return {
            "status": "ok",
            "summary": {
                key: _json_scalar(value)
                for key, value in self.summary.items()
            },
            "diagnostics": {
                key: _json_scalar(value)
                for key, value in self.diagnostics.items()
            },
            "components": {
                key: _json_number(value) for key, value in self.linked_totals.items()
            },
            "style_breakdown": {
                key: _json_number(value) for key, value in self.style_totals.items()
            },
            "industry_breakdown": {
                key: _json_number(value) for key, value in self.industry_totals.items()
            },
            "daily": daily_rows,
        }


def _json_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _json_scalar(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _json_number(value)
    return value


def _as_series(
    rows: Sequence[Mapping[str, object]],
    value_key: str,
) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for row in rows:
        raw_value = row.get(value_key)
        if raw_value is None:
            continue
        try:
            date = pd.Timestamp(row.get("date")).tz_localize(None).normalize()
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if pd.notna(date) and np.isfinite(value):
            values[date] = value
    return pd.Series(values, dtype=float).sort_index()


def _normalise_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    result = prices.copy()
    index = pd.to_datetime(result.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(None)
    result.index = index.normalize()
    result = result.groupby(level=0).last().sort_index()
    return result.apply(pd.to_numeric, errors="coerce")


def _flatten_trades(rebalances: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    records = []
    for rebalance in rebalances or []:
        date = pd.Timestamp(rebalance.get("date")).tz_localize(None).normalize()
        if pd.isna(date):
            continue
        for trade in rebalance.get("trades", []) or []:
            side = str(trade.get("side", "")).lower()
            if side not in {"buy", "sell"}:
                continue
            try:
                quantity = float(trade.get("quantity", trade.get("qty", 0)) or 0)
                price = float(trade.get("price", 0) or 0)
            except (TypeError, ValueError):
                continue
            if quantity <= 0 or not np.isfinite(quantity):
                continue
            records.append(
                {
                    "date": date,
                    "symbol": str(trade.get("symbol", "")).upper(),
                    "side": side,
                    "quantity": quantity,
                    "price": price if np.isfinite(price) and price > 0 else np.nan,
                }
            )
    if not records:
        return pd.DataFrame(columns=["date", "symbol", "side", "quantity", "price"])
    return pd.DataFrame.from_records(records).sort_values(["date", "symbol"])


def _current_quantities(positions: Sequence[Mapping[str, object]]) -> pd.Series:
    values = {}
    for position in positions or []:
        try:
            quantity = float(position.get("qty", 0) or 0)
        except (TypeError, ValueError):
            continue
        if np.isfinite(quantity):
            values[str(position.get("symbol", "")).upper()] = quantity
    return pd.Series(values, dtype=float)


def _reconstruct_holdings(
    dates: pd.DatetimeIndex,
    trades: pd.DataFrame,
    current_positions: Sequence[Mapping[str, object]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Infer the boundary holdings, then replay every known fill."""

    current = _current_quantities(current_positions)
    symbols = current.index.union(
        pd.Index(trades["symbol"].unique()) if not trades.empty else pd.Index([])
    )
    relevant = trades[trades["date"] <= dates.max()] if not trades.empty else trades
    # The boundary condition is holdings at the end of the first equity date.
    # Derive it from today's broker positions by reversing only later fills;
    # older fills may pre-date Alpaca's returned equity window and are already
    # embodied in that boundary holding.
    later = relevant[relevant["date"] > dates.min()] if not relevant.empty else relevant
    net_later = pd.Series(0.0, index=symbols, dtype=float)
    for row in later.itertuples(index=False):
        net_later.loc[row.symbol] += row.quantity if row.side == "buy" else -row.quantity

    opening = current.reindex(symbols).fillna(0.0) - net_later
    negative_opening = opening[opening < -1e-6]
    holdings = opening.copy()
    rows = []
    by_date = {
        date: frame
        for date, frame in later.groupby("date")
    } if not later.empty else {}
    for index, date in enumerate(dates):
        if index == 0:
            rows.append(holdings.copy())
            continue
        for row in by_date.get(date, pd.DataFrame()).itertuples(index=False):
            holdings.loc[row.symbol] += (
                row.quantity if row.side == "buy" else -row.quantity
            )
        holdings[holdings.abs() < 1e-10] = 0.0
        rows.append(holdings.copy())

    history = pd.DataFrame(rows, index=dates).reindex(columns=symbols).fillna(0.0)
    final_error = (
        history.iloc[-1].reindex(symbols).fillna(0.0)
        - current.reindex(symbols).fillna(0.0)
    ) if len(history) else pd.Series(dtype=float)
    diagnostics = {
        "trade_count": int(len(relevant)),
        "opening_positions_inferred": int((opening.abs() > 1e-9).sum()),
        "negative_opening_positions": sorted(negative_opening.index.tolist()),
        "holdings_reconciled": bool(
            negative_opening.empty
            and (final_error.abs() <= 1e-6).all()
        ),
        "max_final_quantity_error": (
            float(final_error.abs().max()) if len(final_error) else 0.0
        ),
    }
    return history, diagnostics


def _factor_returns(
    proxy_prices: pd.DataFrame,
    benchmark: str,
    style_proxies: Mapping[str, tuple[str, str]],
    sector_proxies: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    proxy_returns = _normalise_prices(proxy_prices).pct_change(fill_method=None)
    style = {}
    for name, (long_symbol, short_symbol) in style_proxies.items():
        if long_symbol in proxy_returns and short_symbol in proxy_returns:
            style[name] = proxy_returns[long_symbol] - proxy_returns[short_symbol]
    style_frame = pd.DataFrame(style, index=proxy_returns.index)

    industries = {}
    if benchmark in proxy_returns:
        for sector, symbol in sector_proxies.items():
            if symbol in proxy_returns:
                industries[sector] = proxy_returns[symbol] - proxy_returns[benchmark]
    return style_frame, industries


def _estimate_loadings(
    symbol: str,
    date: pd.Timestamp,
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
    style_returns: pd.DataFrame,
    industry_return: pd.Series | None,
    lookback: int,
    min_observations: int,
) -> tuple[float, dict[str, float], float, int]:
    """Estimate strictly lagged stock loadings with an intercept."""

    pieces = [asset_returns[symbol].rename("asset"), market_returns.rename("market")]
    for name in style_returns.columns:
        pieces.append(style_returns[name].rename(name))
    if industry_return is not None:
        pieces.append(industry_return.rename("industry"))
    sample = pd.concat(pieces, axis=1).loc[lambda frame: frame.index < date]
    sample = sample.tail(lookback).dropna()
    factor_columns = [column for column in sample.columns if column != "asset"]
    if len(sample) < min_observations or not factor_columns:
        return 1.0, {name: 0.0 for name in style_returns.columns}, 0.0, len(sample)

    x = sample[factor_columns].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    y = sample["asset"].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    slopes = {
        name: float(np.clip(coefficients[index + 1], -5.0, 5.0))
        for index, name in enumerate(factor_columns)
    }
    return (
        slopes.get("market", 1.0),
        {name: slopes.get(name, 0.0) for name in style_returns.columns},
        slopes.get("industry", 0.0),
        len(sample),
    )


def _carino_link(
    daily: pd.DataFrame,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Carino-link daily arithmetic contributions to terminal active return."""

    strategy_total = float((1.0 + strategy_returns).prod() - 1.0)
    benchmark_total = float((1.0 + benchmark_returns).prod() - 1.0)
    active_total = strategy_total - benchmark_total

    difference = strategy_returns - benchmark_returns
    numerator = np.log1p(strategy_returns) - np.log1p(benchmark_returns)
    period_k = 1.0 / (1.0 + strategy_returns)
    distinct = difference.abs() > 1e-12
    period_k.loc[distinct] = numerator.loc[distinct] / difference.loc[distinct]
    period_k = period_k.astype(float)
    total_difference = strategy_total - benchmark_total
    total_numerator = np.log1p(strategy_total) - np.log1p(benchmark_total)
    if abs(total_difference) > 1e-12:
        total_k = total_numerator / total_difference
    else:
        total_k = 1.0 / (1.0 + strategy_total)
    if not np.isfinite(total_k) or abs(total_k) < 1e-12:
        total_k = 1.0

    linked = daily.loc[:, columns].mul(period_k / total_k, axis=0)
    totals = {column: float(linked[column].sum()) for column in columns}

    # Floating-point correction belongs to the explicitly named residual.
    linked_sum = sum(totals.values())
    correction = active_total - linked_sum
    if RESIDUAL_COMPONENT in linked.columns and abs(correction) > 0:
        last_date = linked.index[-1]
        linked.loc[last_date, RESIDUAL_COMPONENT] += correction
        totals[RESIDUAL_COMPONENT] += correction
    return linked, totals


def attribute_actual_performance(
    *,
    portfolio_history: Sequence[Mapping[str, object]],
    rebalances: Sequence[Mapping[str, object]],
    current_positions: Sequence[Mapping[str, object]],
    asset_prices: pd.DataFrame,
    proxy_prices: pd.DataFrame,
    sector_map: Mapping[str, str] | pd.Series | None,
    benchmark: str = "SPY",
    style_proxies: Mapping[str, tuple[str, str]] | None = None,
    sector_proxies: Mapping[str, str] | None = None,
    lookback: int = 252,
    min_observations: int = 60,
    annual_cash_return: float = 0.0,
) -> AttributionResult:
    """Attribute actual account alpha against ``benchmark``.

    ``asset_prices`` and ``proxy_prices`` must be adjusted closes.  Loadings
    for date T use observations strictly before T.  ``style_proxies`` maps a
    factor name to a long/short ETF pair; ``sector_proxies`` maps the sector
    strings used by yfinance to sector ETFs.
    """

    result = AttributionResult()
    equity = _as_series(portfolio_history, "equity")
    prices = _normalise_prices(asset_prices)
    proxies = _normalise_prices(proxy_prices)
    if len(equity) < 2 or benchmark not in prices:
        result.summary = {"trading_days": max(0, len(equity) - 1)}
        result.diagnostics = {"reason": "Need two account values and benchmark prices"}
        return result

    dates = equity.index
    aligned_benchmark = prices[benchmark].reindex(dates, method="ffill")
    strategy_returns = equity.pct_change(fill_method=None)
    benchmark_returns = aligned_benchmark.pct_change(fill_method=None)
    valid_dates = dates[1:][
        strategy_returns.iloc[1:].notna().to_numpy()
        & benchmark_returns.iloc[1:].notna().to_numpy()
    ]
    if not len(valid_dates):
        result.summary = {"trading_days": 0}
        result.diagnostics = {"reason": "No aligned account and benchmark returns"}
        return result

    trades = _flatten_trades(rebalances)
    holdings, diagnostics = _reconstruct_holdings(
        dates, trades, current_positions
    )
    style_proxies = style_proxies or {}
    sector_proxies = sector_proxies or {}
    style_returns, industry_returns = _factor_returns(
        proxies, benchmark, style_proxies, sector_proxies
    )

    price_returns = prices.pct_change(fill_method=None)
    market_returns_full = prices[benchmark].pct_change(fill_method=None)
    sectors = (
        pd.Series(dtype="object")
        if sector_map is None
        else pd.Series(sector_map, dtype="object")
    )
    sectors.index = sectors.index.astype(str).str.upper()
    cash_daily = (1.0 + float(annual_cash_return)) ** (1.0 / 252.0) - 1.0

    detail_columns = [f"style_{name}" for name in style_returns.columns]
    sector_names = sorted(set(sectors.dropna().astype(str)))
    detail_columns += [f"industry_{name}" for name in sector_names]
    rows = []
    loading_cache: dict[tuple[str, pd.Timestamp], tuple] = {}
    regression_observations = []
    priced_positions = 0
    total_positions = 0

    for date in valid_dates:
        previous_date = dates[dates.get_loc(date) - 1]
        starting_equity = float(equity.loc[previous_date])
        market_return = float(benchmark_returns.loc[date])
        end_quantities = holdings.loc[date]
        previous_prices = prices.reindex([previous_date], method="ffill").iloc[0]
        day_asset_returns = price_returns.reindex([date]).iloc[0]

        row = {
            "strategy_return": float(strategy_returns.loc[date]),
            "benchmark_return": market_return,
            "actual_alpha": float(strategy_returns.loc[date] - market_return),
            **{component: 0.0 for component in COMPONENTS},
            **{column: 0.0 for column in detail_columns},
        }
        invested_weight = 0.0
        for symbol, quantity in end_quantities.items():
            if abs(float(quantity)) <= 1e-12:
                continue
            total_positions += 1
            if (
                symbol not in previous_prices.index
                or symbol not in day_asset_returns.index
                or not np.isfinite(previous_prices.get(symbol, np.nan))
                or not np.isfinite(day_asset_returns.get(symbol, np.nan))
            ):
                continue
            previous_price = float(previous_prices[symbol])
            asset_return = float(day_asset_returns[symbol])
            if previous_price <= 0 or not np.isfinite(asset_return):
                continue
            priced_positions += 1
            weight = float(quantity) * previous_price / starting_equity
            invested_weight += weight
            sector = str(sectors.get(symbol)) if symbol in sectors.index and pd.notna(sectors.get(symbol)) else None
            industry_series = industry_returns.get(sector) if sector else None
            cache_key = (symbol, pd.Timestamp(date))
            if cache_key not in loading_cache:
                if symbol in price_returns:
                    loading_cache[cache_key] = _estimate_loadings(
                        symbol,
                        pd.Timestamp(date),
                        price_returns,
                        market_returns_full,
                        style_returns,
                        industry_series,
                        int(lookback),
                        int(min_observations),
                    )
                else:
                    loading_cache[cache_key] = (
                        1.0,
                        {name: 0.0 for name in style_returns.columns},
                        0.0,
                        0,
                    )
            market_beta, style_betas, industry_beta, observations = loading_cache[cache_key]
            regression_observations.append(observations)

            market_effect = weight * market_beta * market_return
            style_effect = 0.0
            for name, beta in style_betas.items():
                factor_return = style_returns[name].get(date, np.nan)
                if not np.isfinite(factor_return):
                    continue
                contribution = weight * beta * float(factor_return)
                style_effect += contribution
                row[f"style_{name}"] += contribution
            industry_effect = 0.0
            if industry_series is not None:
                factor_return = industry_series.get(date, np.nan)
                if np.isfinite(factor_return):
                    industry_effect = weight * industry_beta * float(factor_return)
                    if f"industry_{sector}" in row:
                        row[f"industry_{sector}"] += industry_effect

            row["market_beta"] += market_effect - weight * market_return
            row["style"] += style_effect
            row["industry"] += industry_effect
            row["stock_selection"] += weight * asset_return - (
                market_effect + style_effect + industry_effect
            )

        cash_weight = 1.0 - invested_weight
        row["cash_drag"] = cash_weight * (cash_daily - market_return)

        if not trades.empty:
            day_trades = trades[trades["date"] == date]
            implementation_pnl = 0.0
            for trade in day_trades.itertuples(index=False):
                reference = previous_prices.get(trade.symbol, np.nan)
                if not np.isfinite(reference) or not np.isfinite(trade.price):
                    continue
                signed_quantity = trade.quantity if trade.side == "buy" else -trade.quantity
                implementation_pnl += -signed_quantity * (trade.price - float(reference))
            row["trading_costs"] = implementation_pnl / starting_equity

        explained = sum(row[component] for component in COMPONENTS)
        row[RESIDUAL_COMPONENT] = row["actual_alpha"] - explained
        rows.append((date, row))

    daily = pd.DataFrame(
        [row for _, row in rows],
        index=pd.DatetimeIndex([date for date, _ in rows]),
    )
    link_columns = list(COMPONENTS) + [RESIDUAL_COMPONENT]
    linked, linked_totals = _carino_link(
        daily,
        daily["strategy_return"],
        daily["benchmark_return"],
        link_columns,
    )

    detail_totals = {}
    if detail_columns:
        _, detail_totals = _carino_link(
            pd.concat(
                [daily[detail_columns], daily[[RESIDUAL_COMPONENT]]], axis=1
            ),
            daily["strategy_return"],
            daily["benchmark_return"],
            detail_columns + [RESIDUAL_COMPONENT],
        )
        # The residual was included only to preserve the Carino denominator;
        # detail totals report factor subdivisions, not another residual.
        detail_totals.pop(RESIDUAL_COMPONENT, None)

    for column in link_columns:
        daily[f"linked_{column}"] = linked[column]
        daily[f"cumulative_{column}"] = linked[column].cumsum()
    daily["cumulative_actual_alpha"] = linked[link_columns].sum(axis=1).cumsum()

    strategy_total = float((1.0 + daily["strategy_return"]).prod() - 1.0)
    benchmark_total = float((1.0 + daily["benchmark_return"]).prod() - 1.0)
    actual_alpha = strategy_total - benchmark_total
    explained_alpha = sum(linked_totals[component] for component in COMPONENTS)
    residual = linked_totals[RESIDUAL_COMPONENT]
    diagnostics.update(
        {
            "price_coverage": (
                float(priced_positions / total_positions) if total_positions else 1.0
            ),
            "median_regression_observations": (
                float(np.median(regression_observations))
                if regression_observations else 0.0
            ),
            "daily_reconciliation_max_abs": float(
                (
                    daily["actual_alpha"]
                    - daily[link_columns].sum(axis=1)
                ).abs().max()
            ),
        }
    )
    result.daily = daily
    result.linked_totals = linked_totals
    result.style_totals = {
        key.removeprefix("style_"): value
        for key, value in detail_totals.items()
        if key.startswith("style_")
    }
    result.industry_totals = {
        key.removeprefix("industry_"): value
        for key, value in detail_totals.items()
        if key.startswith("industry_")
    }
    result.summary = {
        "period_start": daily.index.min().strftime("%Y-%m-%d"),
        "period_end": daily.index.max().strftime("%Y-%m-%d"),
        "trading_days": int(len(daily)),
        "account_return": strategy_total,
        "benchmark_return": benchmark_total,
        "actual_alpha": actual_alpha,
        "explained_alpha": explained_alpha,
        "reconciliation_residual": residual,
    }
    result.diagnostics = diagnostics
    return result
