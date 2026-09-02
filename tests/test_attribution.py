"""Exact-reconciliation tests for actual-account performance attribution."""

import numpy as np
import pandas as pd
import pytest

from quant.attribution import COMPONENTS, attribute_actual_performance
from site_common import _attribution_rebalances, generate_actual_attribution


def _history(dates, equity):
    return [
        {"date": date.strftime("%Y-%m-%d"), "equity": float(value)}
        for date, value in zip(dates, equity)
    ]


def test_all_cash_alpha_is_reported_as_cash_drag():
    dates = pd.bdate_range("2026-01-05", periods=3)
    prices = pd.DataFrame(
        {"SPY": [100.0, 110.0, 121.0], "AAAA": [50.0, 50.0, 50.0]},
        index=dates,
    )

    result = attribute_actual_performance(
        portfolio_history=_history(dates, [1_000.0, 1_000.0, 1_000.0]),
        rebalances=[],
        current_positions=[],
        asset_prices=prices,
        proxy_prices=prices[["SPY"]],
        sector_map={},
        min_observations=1,
    )

    assert result.summary["account_return"] == pytest.approx(0.0)
    assert result.summary["benchmark_return"] == pytest.approx(0.21)
    assert result.summary["actual_alpha"] == pytest.approx(-0.21)
    assert result.linked_totals["cash_drag"] == pytest.approx(-0.21)
    assert result.linked_totals["reconciliation_residual"] == pytest.approx(0.0)


def test_execution_shortfall_reconciles_trade_day_pnl():
    dates = pd.bdate_range("2026-01-05", periods=2)
    prices = pd.DataFrame(
        {"SPY": [100.0, 100.0], "AAAA": [100.0, 110.0]},
        index=dates,
    )
    rebalances = [
        {
            "date": dates[1].strftime("%Y-%m-%d"),
            "trades": [
                {"symbol": "AAAA", "side": "buy", "quantity": 1, "price": 105.0}
            ],
        }
    ]

    result = attribute_actual_performance(
        portfolio_history=_history(dates, [1_000.0, 1_005.0]),
        rebalances=rebalances,
        current_positions=[{"symbol": "AAAA", "qty": 1}],
        asset_prices=prices,
        proxy_prices=prices[["SPY"]],
        sector_map={},
        min_observations=1,
    )

    day = result.daily.iloc[0]
    assert day["stock_selection"] == pytest.approx(0.01)
    assert day["trading_costs"] == pytest.approx(-0.005)
    assert sum(day[name] for name in COMPONENTS) == pytest.approx(0.005)
    assert day["reconciliation_residual"] == pytest.approx(0.0)
    assert result.diagnostics["holdings_reconciled"] is True
    assert result.to_dict()["diagnostics"]["holdings_reconciled"] is True


def test_lagged_factor_model_separates_market_style_and_industry():
    rng = np.random.default_rng(19)
    dates = pd.bdate_range("2025-01-02", periods=110)
    market = rng.normal(0.0004, 0.007, len(dates))
    size = rng.normal(0.0001, 0.004, len(dates))
    technology = rng.normal(0.0002, 0.005, len(dates))
    asset = 1.2 * market + 0.5 * size + 0.7 * technology

    def curve(returns, start):
        return start * np.cumprod(1.0 + returns)

    prices = pd.DataFrame(
        {
            "SPY": curve(market, 100.0),
            "AAAA": curve(asset, 80.0),
        },
        index=dates,
    )
    proxies = pd.DataFrame(
        {
            "SPY": prices["SPY"],
            "IWM": curve(market + size, 90.0),
            "XLK": curve(market + technology, 95.0),
        },
        index=dates,
    )
    account_dates = dates[-6:]
    quantity = 10.0
    equity = quantity * prices.loc[account_dates, "AAAA"]

    result = attribute_actual_performance(
        portfolio_history=_history(account_dates, equity),
        rebalances=[],
        current_positions=[{"symbol": "AAAA", "qty": quantity}],
        asset_prices=prices,
        proxy_prices=proxies,
        sector_map={"AAAA": "Technology"},
        style_proxies={"size": ("IWM", "SPY")},
        sector_proxies={"Technology": "XLK"},
        lookback=90,
        min_observations=60,
    )

    assert abs(result.linked_totals["market_beta"]) > 1e-6
    assert abs(result.linked_totals["style"]) > 1e-6
    assert abs(result.linked_totals["industry"]) > 1e-6
    assert abs(result.linked_totals["stock_selection"]) < 1e-8
    assert result.diagnostics["median_regression_observations"] >= 60
    assert result.diagnostics["daily_reconciliation_max_abs"] < 1e-12


def test_residual_is_explicit_and_linked_totals_equal_geometric_alpha():
    dates = pd.bdate_range("2026-02-02", periods=4)
    prices = pd.DataFrame(
        {"SPY": [100.0, 101.0, 100.0, 102.0], "AAAA": [50.0] * 4},
        index=dates,
    )
    equity = [1_000.0, 1_003.0, 1_001.0, 1_006.0]

    result = attribute_actual_performance(
        portfolio_history=_history(dates, equity),
        rebalances=[],
        current_positions=[],
        asset_prices=prices,
        proxy_prices=prices[["SPY"]],
        sector_map={},
        min_observations=1,
    )

    terminal_sum = sum(result.linked_totals.values())
    assert terminal_sum == pytest.approx(result.summary["actual_alpha"])
    assert abs(result.linked_totals["reconciliation_residual"]) > 0
    payload = result.to_dict()
    assert payload["status"] == "ok"
    assert payload["daily"][-1]["cumulative_actual_alpha"] == pytest.approx(
        result.summary["actual_alpha"]
    )


def test_known_split_fills_are_converted_to_adjusted_economic_units():
    adjusted = _attribution_rebalances(
        [
            {
                "date": "2026-03-20",
                "trades": [
                    {"symbol": "BKNG", "side": "buy", "quantity": 1, "price": 5_000}
                ],
            },
            {
                "date": "2026-04-10",
                "trades": [
                    {"symbol": "BKNG", "side": "sell", "quantity": 1, "price": 210}
                ],
            },
        ]
    )
    chronological = sorted(adjusted, key=lambda row: row["date"])

    assert chronological[0]["trades"][0]["quantity"] == pytest.approx(25)
    assert chronological[0]["trades"][0]["price"] == pytest.approx(200)
    assert chronological[1]["trades"][0]["quantity"] == pytest.approx(25)


def test_dashboard_payload_wires_actual_history_and_proxy_download(monkeypatch):
    dates = pd.bdate_range("2026-03-02", periods=3)
    asset_prices = pd.DataFrame(
        {"SPY": [100.0, 101.0, 102.0], "AAAA": [50.0, 51.0, 52.0]},
        index=dates,
    )
    proxy_prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0],
            "IWM": [100.0, 101.2, 102.5],
            "IWD": [100.0, 100.5, 101.0],
            "IWF": [100.0, 101.0, 102.0],
            "MTUM": [100.0, 101.1, 102.3],
            "QUAL": [100.0, 101.0, 102.1],
            "USMV": [100.0, 100.6, 101.1],
            "XLK": [100.0, 101.4, 102.8],
        },
        index=dates,
    )
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: pd.concat(
        {"Close": proxy_prices}, axis=1
    ))
    trades = {
        "portfolio_history": _history(dates, [1_000.0, 1_000.0, 1_000.0]),
        "positions": [],
        "rebalances": [],
    }

    payload = generate_actual_attribution(
        trades,
        asset_prices,
        pd.DataFrame({"sector": ["Technology"]}, index=["AAAA"]),
        {"attribution": {"lookback_days": 2, "min_observations": 1}},
    )

    assert payload["status"] == "ok"
    assert payload["methodology"]["benchmark"] == "SPY"
    assert payload["components"]["cash_drag"] < 0
    assert payload["summary"]["actual_alpha"] == pytest.approx(
        -payload["summary"]["benchmark_return"]
    )


def test_fills_after_the_last_equity_date_reconcile_holdings():
    """Today's fills are already in the broker positions but not yet in the
    equity history; they must be reversed out of the opening boundary, not
    left in it (which produced negative opening holdings)."""
    dates = pd.bdate_range("2026-01-05", periods=3)
    prices = pd.DataFrame(
        {"SPY": [100.0, 101.0, 102.0], "AAAA": [50.0, 51.0, 52.0]},
        index=dates,
    )
    after_window = (dates[-1] + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    rebalances = [
        {
            "date": dates[1].strftime("%Y-%m-%d"),
            "trades": [{"symbol": "AAAA", "side": "buy", "quantity": 10, "price": 51.0}],
        },
        {
            "date": after_window,
            "trades": [{"symbol": "AAAA", "side": "sell", "quantity": 4, "price": 53.0}],
        },
    ]
    result = attribute_actual_performance(
        portfolio_history=_history(dates, [1_000.0, 1_000.0, 1_010.0]),
        rebalances=rebalances,
        current_positions=[{"symbol": "AAAA", "qty": 6}],
        asset_prices=prices,
        proxy_prices=prices[["SPY"]],
        sector_map={},
        min_observations=1,
    )
    assert result.diagnostics["negative_opening_positions"] == []
    assert result.diagnostics["holdings_reconciled"] is True
    assert result.diagnostics["trades_after_window"] == 1
    # Holdings inside the window: 0 on day 0, 10 from day 1 on (the sale is after the window).
    assert result.daily.loc[dates[2], "stock_selection"] != 0.0 or result.daily.loc[dates[2], "market_beta"] != 0.0
