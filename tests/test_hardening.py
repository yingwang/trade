"""Regression tests for execution, timing, and research-integrity fixes."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from quant.backtest.calendar import fixed_rebalance_dates
from quant.backtest.engine import BacktestEngine
from quant.data.corporate_actions import (
    UnresolvedCorporateActionError,
    assert_corporate_actions_reconciled,
)
from quant.data.point_in_time import (
    PointInTimeUniverse,
    load_point_in_time_bundle,
)
from quant.execution.broker import Order
from quant.execution.alpaca_broker import AlpacaBroker
from quant.execution.safety import PreTradeCheck, SafetyConfig, TWAPSplitter
from quant.portfolio.optimizer import PortfolioOptimizer
from quant.signals.lgbm_model import LGBMRankingModel
from quant.signals.ml_features import MLFeatureEngine


def _timing_config(config):
    cfg = {**config}
    cfg["portfolio"] = {
        **config["portfolio"],
        "transaction_cost_bps": 0,
    }
    cfg["backtest"] = {
        **config["backtest"],
        "initial_capital": 100_000,
        "slippage_bps": 0,
        "market_impact_coeff": 0,
    }
    cfg["risk"] = {**config["risk"], "stop_loss_pct": 0}
    return cfg


class TestBacktestTiming:
    def test_close_signal_executes_at_next_open(self, config):
        cfg = _timing_config(config)
        dates = pd.bdate_range("2026-01-05", periods=4)
        closes = pd.DataFrame(
            {"AAAA": [100, 100, 100, 100], "BENCH": [100, 100, 100, 100]},
            index=dates,
        )
        opens = pd.DataFrame({"AAAA": [90, 50, 60, 70]}, index=dates)
        result = BacktestEngine(cfg).run(
            closes,
            {str(dates[0].date()): pd.Series({"AAAA": 1.0})},
            benchmark_col="BENCH",
            execution_prices=opens,
        )
        assert result.positions_history[0]["holdings"]["AAAA"] == 0
        assert result.positions_history[1]["holdings"]["AAAA"] == 2000
        assert result.trades[0]["date"] == dates[1]
        assert result.trades[0]["execution"] == "next_open"

    def test_missing_bar_defers_instead_of_using_stale_price(self, config):
        cfg = _timing_config(config)
        dates = pd.bdate_range("2026-01-05", periods=4)
        closes = pd.DataFrame(
            {"AAAA": [100, np.nan, 80, 90], "BENCH": [100, 100, 100, 100]},
            index=dates,
        )
        opens = pd.DataFrame({"AAAA": [100, np.nan, 80, 90]}, index=dates)
        result = BacktestEngine(cfg).run(
            closes,
            {str(dates[0].date()): pd.Series({"AAAA": 1.0})},
            benchmark_col="BENCH",
            execution_prices=opens,
        )
        assert result.positions_history[1]["holdings"]["AAAA"] == 0
        assert result.positions_history[1]["pending_rebalance"] is True
        assert result.trades[0]["date"] == dates[2]

    def test_zero_delta_is_not_recorded_as_trade(self, config):
        cfg = _timing_config(config)
        dates = pd.bdate_range("2026-01-05", periods=3)
        prices = pd.DataFrame(
            {"AAAA": [100, 100, 100], "BENCH": [100, 100, 100]}, index=dates
        )
        result = BacktestEngine(cfg).run(
            prices,
            {str(dates[0].date()): pd.Series({"AAAA": 0.0})},
            benchmark_col="BENCH",
        )
        assert result.trades == []

    def test_delisting_return_is_applied_to_held_position(self, config):
        cfg = _timing_config(config)
        dates = pd.bdate_range("2026-01-05", periods=4)
        prices = pd.DataFrame(
            {"AAAA": [100, 100, np.nan, np.nan], "BENCH": [100] * 4}, index=dates
        )
        events = pd.DataFrame(
            {"date": [dates[2]], "symbol": ["AAAA"], "delisting_return": [-1.0]}
        )
        result = BacktestEngine(cfg).run(
            prices,
            {str(dates[0].date()): pd.Series({"AAAA": 1.0})},
            benchmark_col="BENCH",
            delisting_returns=events,
        )
        assert any(t["type"] == "delisting" for t in result.trades)
        assert result.positions_history[2]["holdings"]["AAAA"] == 0
        assert result.equity_curve.iloc[-1] == pytest.approx(0.0)

    def test_stop_loss_is_not_rebought_by_same_cycle_target(self, config):
        cfg = _timing_config(config)
        cfg["risk"] = {**cfg["risk"], "stop_loss_pct": 0.15}
        dates = pd.bdate_range("2026-01-05", periods=5)
        prices = pd.DataFrame(
            {
                "AAAA": [100.0, 100.0, 70.0, 70.0, 70.0],
                "BENCH": [100.0] * 5,
            },
            index=dates,
        )
        target = pd.Series({"AAAA": 1.0})

        result = BacktestEngine(cfg).run(
            prices,
            {
                str(dates[0].date()): target,
                # This same-close target used to re-buy AAAA immediately after
                # the stop executed at the next bar.
                str(dates[2].date()): target,
            },
            benchmark_col="BENCH",
        )

        assert any(trade["type"] == "stop_loss" for trade in result.trades)
        assert result.positions_history[3]["holdings"]["AAAA"] == 0


class TestRiskReduction:
    def test_concentration_rule_allows_partial_risk_reducing_sell(self):
        checker = PreTradeCheck(
            SafetyConfig(
                max_single_order_value=1_000_000,
                max_single_order_shares=1_000_000,
                max_daily_trade_value=1_000_000,
                max_position_pct_of_portfolio=0.20,
            )
        )
        # Position moves from 25% to 21%, still above the limit but safer.
        order = Order("AAAA", "sell", 400, "market")
        ok, reason = checker.validate(
            order,
            price=100,
            portfolio_value=1_000_000,
            current_positions={"AAAA": 250_000},
        )
        assert (ok, reason) == (True, "passed")

    def test_emergency_deleveraging_bypasses_turnover_cap(self, config):
        cfg = {**config}
        cfg["portfolio"] = {
            **config["portfolio"],
            "max_turnover_per_rebalance": 0.40,
        }
        optimizer = PortfolioOptimizer(cfg)
        previous = pd.Series({"A": 0.9, "B": 0.9})
        target = pd.Series({"A": 0.4, "B": 0.4})
        result = optimizer.enforce_turnover_cap(
            target,
            previous,
            gross_exposure_cap=0.8,
        )
        assert result.sum() == pytest.approx(0.8)

    def test_post_turnover_limits_cap_old_concentrated_stub(self, config):
        cfg = {**config, "safety": {"max_position_pct_of_portfolio": 0.20}}
        cfg["portfolio"] = {
            **config["portfolio"],
            "max_turnover_per_rebalance": 0.10,
        }
        optimizer = PortfolioOptimizer(cfg)
        result = optimizer.enforce_turnover_cap(
            pd.Series({"B": 0.5, "C": 0.5}),
            pd.Series({"A": 0.8, "B": 0.2}),
            gross_exposure_cap=1.0,
        )
        assert result.max() <= 0.20 + 1e-12
        assert result.sum() <= 1.0 + 1e-12

    def test_sector_deconcentration_bypasses_turnover_cap(self, config):
        optimizer = PortfolioOptimizer(config)
        optimizer.max_turnover = 0.10
        sector_map = pd.Series({"A": "Tech", "B": "Tech", "C": "Health"})
        previous = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})
        target = pd.Series({"A": 0.15, "B": 0.15, "C": 0.3})

        result = optimizer.enforce_turnover_cap(
            target,
            previous,
            gross_exposure_cap=1.0,
            sector_map=sector_map,
        )

        pd.testing.assert_series_equal(result, target)


class TestStopLossRecovery:
    def test_rejected_stop_keeps_entry_and_attempt_state(self):
        from paper_trade import check_stop_losses

        broker = MagicMock()
        broker.get_positions.return_value = pd.Series({"AAAA": 100.0})
        broker.get_current_prices.return_value = {"AAAA": 80.0}
        broker.submit_order.return_value = SimpleNamespace(
            status="rejected",
            filled_quantity=0,
            order_id="order-1",
            client_order_id="intent-1",
        )
        optimizer = SimpleNamespace(stop_loss_pct=0.15)
        state = {"entry_prices": {"AAAA": 100.0}}
        check_stop_losses(broker, optimizer, state, dry_run=False)
        assert state["entry_prices"]["AAAA"] == 100.0
        assert state["stop_loss_attempts"]["AAAA"]["status"] == "rejected"


class TestCalendarAndPointInTimeData:
    def test_fixed_calendar_matches_overlapping_windows(self):
        long = pd.bdate_range("2020-01-01", "2026-01-01")
        short = long[long >= "2024-01-01"]
        long_dates = fixed_rebalance_dates(long, 21)
        short_dates = fixed_rebalance_dates(short, 21)
        overlap = [date for date in long_dates if date >= short[0]]
        assert short_dates == overlap

    def test_point_in_time_universe_uses_latest_snapshot(self, tmp_path):
        path = tmp_path / "universe.csv"
        path.write_text(
            "date,symbol,is_member\n"
            "2025-01-01,AAAA,true\n"
            "2025-01-01,BBBB,false\n"
            "2025-02-01,AAAA,false\n"
            "2025-02-01,BBBB,true\n"
        )
        universe = PointInTimeUniverse.from_csv(path)
        assert universe.members_as_of("2025-01-15") == {"AAAA"}
        assert universe.members_as_of("2025-02-15") == {"BBBB"}
        dates = pd.DatetimeIndex(["2025-01-15", "2025-02-15"])
        mask = universe.eligibility_mask(dates, ["AAAA", "BBBB", "CCCC"])
        assert mask.loc[dates[0]].to_dict() == {
            "AAAA": True,
            "BBBB": False,
            "CCCC": False,
        }
        assert mask.loc[dates[1]].to_dict() == {
            "AAAA": False,
            "BBBB": True,
            "CCCC": False,
        }

    def test_point_in_time_universe_rejects_ambiguous_membership(self, tmp_path):
        path = tmp_path / "universe.csv"
        path.write_text("date,symbol,is_member\n2025-01-01,AAAA,tru\n")
        with pytest.raises(ValueError, match="invalid is_member"):
            PointInTimeUniverse.from_csv(path)

    def test_nonmembers_do_not_enter_factor_cross_section(self, config, tmp_path):
        path = tmp_path / "universe.csv"
        path.write_text(
            "date,symbol,is_member\n"
            "2024-01-01,AAAA,true\n"
            "2024-01-01,BBBB,true\n"
            "2024-01-01,CCCC,false\n"
        )
        universe = PointInTimeUniverse.from_csv(path)
        dates = pd.bdate_range("2024-01-01", periods=320)
        prices = pd.DataFrame(
            {
                "AAAA": np.linspace(100, 130, len(dates)),
                "BBBB": np.linspace(100, 120, len(dates)),
                "CCCC": np.linspace(100, 500, len(dates)),
                "BENCH": np.linspace(100, 125, len(dates)),
            },
            index=dates,
        )
        eligibility = universe.eligibility_mask(
            dates, ["AAAA", "BBBB", "CCCC"]
        )
        from quant.signals.factors import SignalGenerator

        signals = SignalGenerator(config).generate(
            prices,
            prices.pct_change(fill_method=None),
            eligibility_mask=eligibility,
        )
        assert pd.isna(signals.iloc[-1]["CCCC"])
        assert signals.iloc[-1][["AAAA", "BBBB"]].notna().all()

    def test_nonmember_price_path_cannot_change_member_scores(self, config):
        dates = pd.bdate_range("2024-01-01", periods=320)
        base = pd.DataFrame(
            {
                "AAAA": np.linspace(100, 140, len(dates)),
                "BBBB": np.linspace(100, 115, len(dates)),
                "CCCC": np.linspace(100, 200, len(dates)),
                "BENCH": np.linspace(100, 125, len(dates)),
            },
            index=dates,
        )
        shocked = base.copy()
        shocked["CCCC"] = np.geomspace(100, 100_000, len(dates))
        eligible = pd.DataFrame(
            {
                "AAAA": True,
                "BBBB": True,
                "CCCC": False,
            },
            index=dates,
        )
        from quant.signals.factors import SignalGenerator

        normal_scores = SignalGenerator(config).generate(
            base,
            base.pct_change(fill_method=None),
            eligibility_mask=eligible,
        )
        shocked_scores = SignalGenerator(config).generate(
            shocked,
            shocked.pct_change(fill_method=None),
            eligibility_mask=eligible,
        )

        pd.testing.assert_series_equal(
            normal_scores.iloc[-1][["AAAA", "BBBB"]],
            shocked_scores.iloc[-1][["AAAA", "BBBB"]],
            check_names=False,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_pit_universe_without_delisting_returns_fails(self, tmp_path):
        path = tmp_path / "universe.csv"
        path.write_text("date,symbol,is_member\n2025-01-01,AAAA,true\n")
        with pytest.raises(ValueError, match="without delisting returns"):
            load_point_in_time_bundle(
                {"data": {"point_in_time_universe_file": str(path)}}
            )


class TestEnsembleFallback:
    def test_constant_strategy_does_not_create_arbitrary_consensus_boost(self, config):
        from quant.strategy_ensemble import StrategyEnsemble

        ensemble = StrategyEnsemble.__new__(StrategyEnsemble)
        ensemble.config = config
        ensemble.weight_a = 0.5
        ensemble.weight_b = 0.5
        ensemble.consensus_boost = 2.0

        factor = pd.Series({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
        unavailable_model = pd.Series(0.5, index=factor.index)
        combined = ensemble._combine_scores(factor, unavailable_model)

        expected = 0.5 * factor.rank(pct=True) + 0.25
        pd.testing.assert_series_equal(
            combined.sort_index(),
            expected.sort_index(),
            check_names=False,
        )


class TestCorporateActions:
    def test_presplit_paper_position_fails_closed(self):
        with pytest.raises(UnresolvedCorporateActionError, match="BKNG"):
            assert_corporate_actions_reconciled(
                [
                    {
                        "symbol": "BKNG",
                        "qty": 10,
                        "avg_entry_price": 5000,
                        "current_price": 200,
                    }
                ],
                as_of=date(2026, 4, 6),
            )

    def test_postsplit_position_is_allowed(self):
        assert_corporate_actions_reconciled(
            [
                {
                    "symbol": "BKNG",
                    "qty": 250,
                    "avg_entry_price": 205,
                    "current_price": 200,
                }
            ],
            as_of=date(2026, 4, 6),
        )

    def test_known_split_with_unusable_price_fails_closed(self):
        with pytest.raises(UnresolvedCorporateActionError, match="BKNG"):
            assert_corporate_actions_reconciled(
                [
                    {
                        "symbol": "BKNG",
                        "qty": 25,
                        "avg_entry_price": 205,
                        "current_price": None,
                    }
                ],
            as_of=date(2026, 4, 6),
        )

    def test_known_split_with_nonfinite_quantity_fails_closed(self):
        with pytest.raises(UnresolvedCorporateActionError, match="BKNG"):
            assert_corporate_actions_reconciled(
                [
                    {
                        "symbol": "BKNG",
                        "qty": float("nan"),
                        "avg_entry_price": 205,
                        "current_price": 210,
                    }
                ],
                as_of=date(2026, 4, 6),
            )


class TestRankingModel:
    def test_lambda_rank_uses_date_groups_and_reports_skill(self):
        rng = np.random.default_rng(7)
        train_t, val_t, n, f = 500, 30, 12, 5
        X_train = rng.normal(size=(train_t, n, f)).astype(np.float32)
        X_val = rng.normal(size=(val_t, n, f)).astype(np.float32)
        y_train = np.argsort(np.argsort(X_train[:, :, 0], axis=1), axis=1) / (n - 1)
        y_val = np.argsort(np.argsort(X_val[:, :, 0], axis=1), axis=1) / (n - 1)
        model = LGBMRankingModel(n_estimators=30, early_stopping_rounds=5)
        info = model.train(
            X_train,
            y_train,
            X_val,
            y_val,
            feature_names=[f"f{i}" for i in range(f)],
        )
        assert info["status"] == "ok"
        assert info["objective"] in {"lambdarank", "regression_fallback"}
        assert info["train_groups"] > 3
        assert "val_rank_ic" in info
        assert "baseline_skill" in info

    def test_feature_matrix_has_explicit_missing_indicators(self, config):
        dates = pd.bdate_range("2025-01-01", periods=80)
        prices = pd.DataFrame(
            {
                "AAAA": np.linspace(100, 110, 80),
                "BBBB": np.linspace(90, 100, 80),
                "BENCH": np.linspace(100, 105, 80),
            },
            index=dates,
        )
        prices.loc[dates[10], "AAAA"] = np.nan
        returns = prices.pct_change(fill_method=None)
        engine = MLFeatureEngine(config)
        _, names, _, _ = engine.build_feature_matrix(prices, returns)
        assert any(name.endswith("_missing") for name in names)

    def test_cross_sectional_target_ranks_only_eligible_members(self, config):
        dates = pd.bdate_range("2025-01-01", periods=3)
        returns = pd.DataFrame(
            {
                "AAAA": [0.0, 0.10, 0.0],
                "BBBB": [0.0, 0.20, 0.0],
                "CCCC": [0.0, 0.90, 0.0],
                "BENCH": [0.0, 0.0, 0.0],
            },
            index=dates,
        )
        eligible = pd.DataFrame(
            {
                "AAAA": [True] * 3,
                "BBBB": [True] * 3,
                "CCCC": [False] * 3,
            },
            index=dates,
        )
        target = MLFeatureEngine(config).get_cross_sectional_target(
            returns, horizon=1, eligibility_mask=eligible
        )
        assert target.loc[dates[0], "AAAA"] == pytest.approx(0.5)
        assert target.loc[dates[0], "BBBB"] == pytest.approx(1.0)
        assert pd.isna(target.loc[dates[0], "CCCC"])

    def test_ml_target_retains_terminal_delisting_loss(self, config):
        dates = pd.bdate_range("2025-01-01", periods=5)
        returns = pd.DataFrame(
            {
                "AAAA": [0.0, 0.01, np.nan, np.nan, np.nan],
                "BBBB": [0.0, 0.01, 0.01, 0.01, 0.01],
                "BENCH": [0.0] * 5,
            },
            index=dates,
        )
        delistings = pd.DataFrame(
            {
                "date": [dates[2]],
                "symbol": ["AAAA"],
                "delisting_return": [-1.0],
            }
        )

        target = MLFeatureEngine(config).get_target(
            returns,
            horizon=3,
            delisting_returns=delistings,
        )

        assert target.loc[dates[0], "AAAA"] == pytest.approx(-1.0)
        assert target.loc[dates[0], "BBBB"] == pytest.approx(1.01 ** 3 - 1.0)


class TestTWAPSchedule:
    def test_last_slice_is_at_configured_duration(self):
        splitter = TWAPSplitter(
            adv_threshold=0.01, n_slices=5, duration_minutes=30
        )
        slices = splitter.split_order(
            Order("AAAA", "buy", 5000, "market"), avg_daily_volume=100_000
        )
        assert [offset for _, offset in slices] == [0, 450, 900, 1350, 1800]


class TestAlpacaIdempotency:
    def test_same_intent_has_stable_client_order_id(self):
        first = Order("AAAA", "sell", 25, "market", purpose="stop_loss")
        second = Order("AAAA", "sell", 25, "market", purpose="stop_loss")
        assert AlpacaBroker._base_client_order_id(first) == AlpacaBroker._base_client_order_id(second)

    def test_partial_stop_retry_remaining_quantity_changes_intent(self):
        original = Order("AAAA", "sell", 25, "market", purpose="stop_loss")
        remaining = Order("AAAA", "sell", 10, "market", purpose="stop_loss")
        assert AlpacaBroker._base_client_order_id(original) != AlpacaBroker._base_client_order_id(remaining)

    def test_official_sdk_request_contains_client_id(self):
        broker = object.__new__(AlpacaBroker)
        order = Order(
            "AAPL",
            "buy",
            5,
            "market",
            client_order_id="qts-regression-test",
        )
        request = broker._build_order_request(order)
        assert request.symbol == "AAPL"
        assert float(request.qty) == 5
        assert request.client_order_id == "qts-regression-test"

    def test_recovered_fill_does_not_double_daily_budget(self):
        checker = PreTradeCheck(SafetyConfig())
        assert checker.record_fill(10_000, client_order_id="intent-1") is True
        assert checker.record_fill(10_000, client_order_id="intent-1") is False
        assert checker.daily.total_value_traded == 10_000
        assert checker.daily.orders_filled == 1

    def test_recovered_partial_fill_counts_only_new_notional(self):
        checker = PreTradeCheck(SafetyConfig())
        assert checker.record_fill(4_000, client_order_id="intent-1") is True
        assert checker.record_fill(10_000, client_order_id="intent-1") is True
        assert checker.record_fill(10_000, client_order_id="intent-1") is False
        assert checker.daily.total_value_traded == 10_000
        assert checker.daily.orders_filled == 1


class TestDashboardDates:
    def test_utc_fill_is_grouped_by_eastern_trade_date(self):
        from site_common import _eastern_date

        assert _eastern_date("2026-07-15T01:00:00+00:00") == "2026-07-14"

    def test_same_day_split_events_use_timestamp_order(self):
        from site_common import _split_sold_credit

        # Alpaca returns closed orders newest-first. Date-only sorting used to
        # process this sell before the earlier post-split buy and invent credit.
        sell = SimpleNamespace(
            symbol="BKNG",
            side="sell",
            filled_qty="3",
            filled_avg_price="180",
            filled_at="2026-04-28T15:00:00-04:00",
            submitted_at=None,
        )
        buy = SimpleNamespace(
            symbol="BKNG",
            side="buy",
            filled_qty="3",
            filled_avg_price="175",
            filled_at="2026-04-28T14:00:00-04:00",
            submitted_at=None,
        )
        assert _split_sold_credit([sell, buy]) == 0.0

    def test_manual_split_cash_prevents_double_credit(self):
        from site_common import _split_sold_credit

        sell = SimpleNamespace(
            symbol="BKNG",
            side="sell",
            filled_qty="3",
            filled_avg_price="175.40",
            filled_at="2026-04-28T15:00:00-04:00",
            submitted_at=None,
        )
        compensation = {
            "BKNG": {"date": "2026-04-29", "amount": 12_628.80}
        }

        assert _split_sold_credit(
            [sell], cash_compensations=compensation, as_of="2026-04-28"
        ) == pytest.approx(12_628.80)
        assert _split_sold_credit(
            [sell], cash_compensations=compensation, as_of="2026-04-29"
        ) == pytest.approx(0.0)

    def test_manual_split_cash_stops_history_credit_from_deposit_date(
        self, monkeypatch
    ):
        from site_common import _split_history_adjustments

        prices = pd.DataFrame(
            {"Close": [175.40, 176.00]},
            index=pd.to_datetime(["2026-04-28", "2026-04-29"]),
        )
        monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: prices)
        sell = SimpleNamespace(
            symbol="BKNG",
            side="sell",
            filled_qty="3",
            filled_avg_price="175.40",
            filled_at="2026-04-28T15:00:00-04:00",
            submitted_at=None,
        )
        history = [
            {"date": "2026-04-28", "equity": 100_000},
            {"date": "2026-04-29", "equity": 100_000},
        ]

        adjustments = _split_history_adjustments(
            history,
            [],
            [sell],
            cash_compensations={
                "BKNG": {"date": "2026-04-29", "amount": 12_628.80}
            },
        )

        assert adjustments == {"2026-04-28": pytest.approx(12_628.80)}

    def test_split_history_adjustment_tracks_daily_price(self, monkeypatch):
        from site_common import _split_history_adjustments

        prices = pd.DataFrame(
            {"Close": [200.0, 210.0]},
            index=pd.to_datetime(["2026-04-06", "2026-04-07"]),
        )
        monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: prices)
        history = [
            {"date": "2026-04-06", "equity": 100_000},
            {"date": "2026-04-07", "equity": 100_000},
        ]
        position = SimpleNamespace(
            symbol="BKNG",
            qty="10",
            avg_entry_price="5000",
            current_price="210",
        )
        adjustments = _split_history_adjustments(history, [position], [])
        assert adjustments["2026-04-06"] == pytest.approx(48_000)
        assert adjustments["2026-04-07"] == pytest.approx(50_400)
