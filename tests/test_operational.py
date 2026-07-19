"""Tests for operational code paths: config, paper_trade helpers, stop-loss integration."""

import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant.utils.config import load_config
from quant.backtest.engine import BacktestEngine


# ── Config loading ──────────────────────────────────────────────────────────

class TestConfigLoading:
    def test_load_config_from_path(self):
        """load_config reads a YAML file and returns a dict."""
        config = load_config("config.yaml")
        assert isinstance(config, dict)
        assert "universe" in config
        assert "portfolio" in config
        assert "risk" in config

    def test_load_config_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_config_default_path(self):
        """Default path resolution finds config.yaml relative to module."""
        config = load_config()
        assert isinstance(config, dict)

    def test_duplicate_universe_symbols_are_rejected(self, tmp_path):
        source = load_config()
        source["universe"]["symbols"] = ["AAAA", "AAAA"]
        source["portfolio"]["max_positions"] = 1
        path = tmp_path / "bad.yaml"
        import yaml

        path.write_text(yaml.safe_dump(source, sort_keys=False))
        with pytest.raises(ValueError, match="duplicate tickers"):
            load_config(path)

    def test_nonfinite_risk_limit_is_rejected(self, tmp_path):
        source = load_config()
        source["risk"]["max_sector_weight"] = float("nan")
        path = tmp_path / "bad.yaml"
        import yaml

        path.write_text(yaml.safe_dump(source, sort_keys=False))
        with pytest.raises(ValueError, match="must be finite"):
            load_config(path)


# ── paper_trade helpers ─────────────────────────────────────────────────────

class TestPaperTradeHelpers:
    def test_load_state_empty(self, tmp_path):
        """load_state returns default dict when file doesn't exist."""
        from paper_trade import load_state
        # Patch STATE_FILE to a temp location
        fake_state = tmp_path / "state.json"
        with patch("paper_trade.STATE_FILE", fake_state):
            state = load_state()
            assert state["last_rebalance"] is None
            assert state["trade_history"] == []

    def test_load_and_save_state(self, tmp_path):
        from paper_trade import load_state, save_state
        fake_state = tmp_path / "logs" / "state.json"
        with patch("paper_trade.STATE_FILE", fake_state):
            state = {"last_rebalance": "2026-03-01T10:00:00", "trade_history": [{"x": 1}]}
            save_state(state)
            loaded = load_state()
            assert loaded["last_rebalance"] == "2026-03-01T10:00:00"
            assert len(loaded["trade_history"]) == 1

    def test_should_rebalance_first_time(self):
        from paper_trade import should_rebalance
        assert should_rebalance({"last_rebalance": None}, 21) is True

    def test_should_rebalance_too_soon(self):
        from paper_trade import should_rebalance
        recent = (datetime.now() - timedelta(days=5)).isoformat()
        assert should_rebalance({"last_rebalance": recent}, 21) is False

    def test_should_rebalance_due(self):
        from paper_trade import should_rebalance
        old = (datetime.now() - timedelta(days=35)).isoformat()
        assert should_rebalance({"last_rebalance": old}, 21) is True

    def test_acquire_release_lock(self, tmp_path):
        from paper_trade import acquire_lock, release_lock
        fake_lock = tmp_path / "paper_trade.lock"
        with patch("paper_trade.LOCK_FILE", fake_lock):
            assert acquire_lock() is True
            assert fake_lock.exists()
            # Second acquire should fail
            assert acquire_lock() is False
            release_lock()
            assert not fake_lock.exists()

    def test_stale_lock_removed(self, tmp_path):
        from paper_trade import acquire_lock, release_lock
        fake_lock = tmp_path / "paper_trade.lock"
        # Create a "stale" lock by writing it and backdating mtime
        fake_lock.write_text('{"pid": 99999}')
        old_time = datetime.now().timestamp() - 7200  # 2 hours ago
        os.utime(fake_lock, (old_time, old_time))

        with patch("paper_trade.LOCK_FILE", fake_lock):
            assert acquire_lock() is True
            release_lock()


# ── Rebalance safety wiring ─────────────────────────────────────────────────

@pytest.fixture
def quiet_exec_log():
    """Keep run_rebalance from writing events to the real logs/ directory."""
    with patch("paper_trade.ExecutionLogger", MagicMock()):
        yield


class TestRebalanceSafety:
    def _config_with_safety(self, config):
        return {**config, "safety": {"max_daily_loss": 25_000}}

    def test_daily_loss_kill_switch_blocks_rebalance(self, config, quiet_exec_log):
        """When today's loss exceeds the limit, no orders are computed or sent."""
        from paper_trade import run_rebalance

        broker = MagicMock()
        broker.get_portfolio_value.return_value = 1_000_000
        broker.get_daily_pnl.return_value = -30_000  # breaches $25k limit

        strategy = MagicMock()

        filled, target = run_rebalance(
            strategy, broker, self._config_with_safety(config), dry_run=False
        )
        assert filled is None and target is None
        strategy.get_current_portfolio.assert_not_called()
        broker.submit_order.assert_not_called()

    def test_daily_loss_within_limit_proceeds(self, config, quiet_exec_log):
        """A normal down day must not block the rebalance."""
        from paper_trade import run_rebalance

        broker = MagicMock()
        broker.get_portfolio_value.return_value = 1_000_000
        broker.get_daily_pnl.return_value = -5_000
        broker.get_positions.return_value = pd.Series({"AAAA": 500.0})
        broker.get_current_prices.return_value = {"AAAA": 100.0}
        broker.is_market_open.return_value = True

        strategy = MagicMock()
        # Target matches current holdings exactly -> no orders needed
        portfolio = pd.DataFrame({
            "score": [1.0], "weight": [0.05], "weight_pct": [5.0],
            "dollars": [50_000.0], "shares": [500], "price": [100.0],
        }, index=["AAAA"])
        strategy.get_current_portfolio.return_value = portfolio

        filled, target = run_rebalance(
            strategy, broker, self._config_with_safety(config), dry_run=False
        )
        # Empty list (not None) signals "at target": main() records the
        # rebalance as done instead of retrying daily.
        assert filled == []
        assert target is not None
        broker.submit_order.assert_not_called()

    def test_market_closed_aborts_rebalance(self, config, quiet_exec_log):
        """Holiday runs must not queue market orders for the next open —
        a failed cancel plus the next day's rerun meant double buying."""
        from paper_trade import run_rebalance

        broker = MagicMock()
        broker.get_portfolio_value.return_value = 1_000_000
        broker.is_market_open.return_value = False

        strategy = MagicMock()

        filled, target = run_rebalance(
            strategy, broker, self._config_with_safety(config), dry_run=False
        )
        assert filled is None and target is None
        strategy.get_current_portfolio.assert_not_called()
        broker.submit_order.assert_not_called()

    def test_actual_weights_passed_to_strategy(self, config, quiet_exec_log):
        """The account's real weights must reach the strategy so the
        turnover machinery binds against reality."""
        from paper_trade import run_rebalance

        broker = MagicMock()
        broker.get_portfolio_value.return_value = 1_000_000
        broker.get_daily_pnl.return_value = 0.0
        broker.get_positions.return_value = pd.Series({"AAAA": 500.0})
        broker.get_current_prices.return_value = {"AAAA": 100.0}
        broker.is_market_open.return_value = True

        strategy = MagicMock()
        portfolio = pd.DataFrame({
            "score": [1.0], "weight": [0.05], "weight_pct": [5.0],
            "dollars": [50_000.0], "shares": [500], "price": [100.0],
        }, index=["AAAA"])
        strategy.get_current_portfolio.return_value = portfolio

        run_rebalance(strategy, broker, self._config_with_safety(config), dry_run=False)

        _, kwargs = strategy.get_current_portfolio.call_args
        prev_w = kwargs["prev_weights"]
        assert prev_w["AAAA"] == pytest.approx(0.05)  # 500 * $100 / $1M

    def test_adv_passed_to_broker(self, config, quiet_exec_log):
        """ADV reaches submit_order so liquidity checks / TWAP can engage."""
        from paper_trade import run_rebalance

        broker = MagicMock()
        broker.get_portfolio_value.return_value = 1_000_000
        broker.get_daily_pnl.return_value = 0.0
        broker.get_positions.return_value = pd.Series(dtype=float)
        broker.get_current_prices.return_value = {"AAAA": 100.0}
        broker.is_market_open.return_value = True
        fill = MagicMock()
        fill.status = "filled"
        fill.quantity = 500
        fill.filled_price = 100.0
        broker.submit_order.return_value = fill

        strategy = MagicMock()
        portfolio = pd.DataFrame({
            "score": [1.0], "weight": [0.05], "weight_pct": [5.0],
            "dollars": [50_000.0], "shares": [500], "price": [100.0],
        }, index=["AAAA"])
        strategy.get_current_portfolio.return_value = portfolio
        strategy.data.fetch_adv.return_value = {"AAAA": 2_000_000.0}

        run_rebalance(strategy, broker, self._config_with_safety(config), dry_run=False)

        strategy.data.fetch_adv.assert_called_once()
        _, kwargs = broker.submit_order.call_args
        assert kwargs["avg_daily_volume"] == 2_000_000.0

    def test_failed_sell_blocks_subsequent_buys(self, config, quiet_exec_log):
        """A target must not assume failed exits produced cash/risk capacity."""
        from paper_trade import run_rebalance

        broker = MagicMock()
        broker.get_portfolio_value.return_value = 100_000
        broker.get_daily_pnl.return_value = 0.0
        broker.get_positions.return_value = pd.Series({"BBBB": 500.0})
        broker.get_current_prices.return_value = {"AAAA": 100.0, "BBBB": 100.0}
        broker.is_market_open.return_value = True

        def reject_sell(order, avg_daily_volume=None):
            assert order.side == "sell"
            order.status = "rejected"
            order.reject_reason = "simulated broker rejection"
            return order

        broker.submit_order.side_effect = reject_sell

        strategy = MagicMock()
        strategy.get_current_portfolio.return_value = pd.DataFrame(
            {
                "score": [1.0],
                "weight": [0.5],
                "weight_pct": [50.0],
                "dollars": [50_000.0],
                "shares": [500],
                "price": [100.0],
            },
            index=["AAAA"],
        )
        strategy.data.fetch_adv.return_value = {
            "AAAA": 2_000_000.0,
            "BBBB": 2_000_000.0,
        }

        records, _ = run_rebalance(
            strategy, broker, self._config_with_safety(config), dry_run=False
        )

        assert broker.submit_order.call_count == 1
        assert [record["status"] for record in records] == [
            "rejected",
            "not_submitted",
        ]
        assert records[1]["side"] == "buy"

    def test_missing_held_position_mark_aborts_before_signal_generation(
        self, config, quiet_exec_log
    ):
        from paper_trade import run_rebalance

        broker = MagicMock()
        broker.get_portfolio_value.return_value = 100_000
        broker.get_daily_pnl.return_value = 0.0
        broker.get_positions.return_value = pd.Series({"AAAA": 100.0})
        broker.get_current_prices.return_value = {}
        broker.is_market_open.return_value = True
        strategy = MagicMock()

        with pytest.raises(RuntimeError, match="missing valid marks"):
            run_rebalance(
                strategy,
                broker,
                self._config_with_safety(config),
                dry_run=False,
            )

        strategy.get_current_portfolio.assert_not_called()


# ── ADV fetching ────────────────────────────────────────────────────────────

class TestFetchADV:
    def _market_data(self, config):
        from quant.data.market_data import MarketData
        return MarketData(config)

    def test_fetch_adv_returns_means(self, config):
        md = self._market_data(config)
        dates = pd.bdate_range("2026-01-01", periods=40)
        cols = pd.MultiIndex.from_product([["Close", "Volume"], ["AAAA", "BBBB"]])
        data = pd.DataFrame(1.0, index=dates, columns=cols)
        data[("Volume", "AAAA")] = 1_000_000.0
        data[("Volume", "BBBB")] = 2_000_000.0

        with patch("quant.data.market_data.yf.download", return_value=data):
            adv = md.fetch_adv(["AAAA", "BBBB"], window=30)
        assert adv["AAAA"] == pytest.approx(1_000_000.0)
        assert adv["BBBB"] == pytest.approx(2_000_000.0)

    def test_fetch_adv_failure_returns_empty(self, config):
        """Network failures must not block trading — empty dict means
        'no liquidity data', and the ADV check is skipped downstream."""
        md = self._market_data(config)
        with patch("quant.data.market_data.yf.download", side_effect=Exception("boom")):
            assert md.fetch_adv(["AAAA"]) == {}

    def test_fetch_adv_empty_symbols(self, config):
        assert self._market_data(config).fetch_adv([]) == {}


# ── Stop-loss in backtest engine ────────────────────────────────────────────

class TestBacktestStopLoss:
    def test_stop_loss_triggers_on_large_drop(self, config, synthetic_prices):
        """When a position drops > stop_loss_pct, it should be sold."""
        # Use a tight stop-loss to guarantee triggering
        config["risk"]["stop_loss_pct"] = 0.05  # 5%
        engine = BacktestEngine(config)

        # Put 100% into a downtrending stock
        first_date = synthetic_prices.index[5]
        weights = pd.Series({"DDDD": 0.5, "JJJJ": 0.5})  # both have negative drift
        targets = {str(first_date.date()): weights}

        result = engine.run(synthetic_prices, targets, benchmark_col="BENCH")
        stop_trades = [t for t in result.trades if t.get("type") == "stop_loss"]
        assert len(stop_trades) > 0, "Expected stop-loss trades for downtrending stocks"

    def test_no_stop_loss_when_disabled(self, config, synthetic_prices):
        """With stop_loss_pct=0, no stop-loss trades should occur."""
        config["risk"]["stop_loss_pct"] = 0.0
        engine = BacktestEngine(config)

        first_date = synthetic_prices.index[5]
        weights = pd.Series({"DDDD": 0.5, "JJJJ": 0.5})
        targets = {str(first_date.date()): weights}

        result = engine.run(synthetic_prices, targets, benchmark_col="BENCH")
        stop_trades = [t for t in result.trades if t.get("type") == "stop_loss"]
        assert len(stop_trades) == 0

    def test_stop_loss_increases_cash(self, config, synthetic_prices):
        """After stop-loss, cash should increase (position was sold)."""
        config["risk"]["stop_loss_pct"] = 0.05
        engine = BacktestEngine(config)

        first_date = synthetic_prices.index[5]
        weights = pd.Series({"DDDD": 1.0})
        targets = {str(first_date.date()): weights}

        result = engine.run(synthetic_prices, targets, benchmark_col="BENCH")
        stop_trades = [t for t in result.trades if t.get("type") == "stop_loss"]

        if stop_trades:
            # Find the position history entry right after stop-loss
            stop_date = stop_trades[0]["date"]
            for ph in result.positions_history:
                if ph["date"] == stop_date:
                    # After stop-loss, DDDD holdings should be 0
                    assert ph["holdings"]["DDDD"] == 0.0
                    break

    def test_stop_loss_records_symbols(self, config, synthetic_prices):
        """Stop-loss trade entries should list which symbols were stopped."""
        config["risk"]["stop_loss_pct"] = 0.05
        engine = BacktestEngine(config)

        first_date = synthetic_prices.index[5]
        weights = pd.Series({"DDDD": 0.5, "JJJJ": 0.5})
        targets = {str(first_date.date()): weights}

        result = engine.run(synthetic_prices, targets, benchmark_col="BENCH")
        stop_trades = [t for t in result.trades if t.get("type") == "stop_loss"]
        for st in stop_trades:
            assert "symbols" in st
            assert len(st["symbols"]) > 0


# ── Stop-loss in paper_trade ────────────────────────────────────────────────

class TestPaperTradeStopLoss:
    def test_check_stop_losses_triggers(self):
        """check_stop_losses identifies positions below threshold."""
        from paper_trade import check_stop_losses

        broker = MagicMock()
        broker.get_positions.return_value = pd.Series({"AAAA": 100, "BBBB": 200})
        broker.get_current_prices.return_value = {"AAAA": 80.0, "BBBB": 110.0}

        optimizer = MagicMock()
        optimizer.stop_loss_pct = 0.15

        state = {
            "entry_prices": {"AAAA": 100.0, "BBBB": 100.0}  # AAAA is down 20%
        }

        stopped = check_stop_losses(broker, optimizer, state, dry_run=True)
        assert "AAAA" in stopped
        assert "BBBB" not in stopped  # only up 10%

    def test_check_stop_losses_no_entry_prices(self):
        """No entry prices in state means no stop-losses."""
        from paper_trade import check_stop_losses

        broker = MagicMock()
        optimizer = MagicMock()
        optimizer.stop_loss_pct = 0.15
        state = {}

        stopped = check_stop_losses(broker, optimizer, state, dry_run=True)
        assert stopped == []

    def test_check_stop_losses_sells_when_not_dry_run(self):
        """In live mode, stopped positions are sold via broker."""
        from paper_trade import check_stop_losses

        broker = MagicMock()
        broker.get_positions.side_effect = [
            pd.Series({"AAAA": 100}),
            pd.Series(dtype=float),
        ]
        broker.get_current_prices.return_value = {"AAAA": 80.0}
        mock_result = MagicMock()
        mock_result.status = "filled"
        mock_result.filled_quantity = 100
        broker.submit_order.return_value = mock_result

        optimizer = MagicMock()
        optimizer.stop_loss_pct = 0.15

        state = {"entry_prices": {"AAAA": 100.0}}

        stopped = check_stop_losses(broker, optimizer, state, dry_run=False)
        assert "AAAA" in stopped
        broker.submit_order.assert_called_once()
        # Entry price should be cleared
        assert "AAAA" not in state["entry_prices"]


# ── Entry price semantics ───────────────────────────────────────────────────

class TestEntryPriceSemantics:
    """Entry prices back the stop-loss; semantics must match the backtest
    engine: record on 0 -> positive only, never reset on adds."""

    def test_add_does_not_reset_entry(self):
        from paper_trade_common import update_entry_prices

        broker = MagicMock()
        broker.get_positions.return_value = pd.Series({"AAAA": 300.0})
        state = {"entry_prices": {"AAAA": 100.0}}
        filled = [
            {"symbol": "AAAA", "side": "buy", "status": "filled", "price": 150.0},
            {"symbol": "BBBB", "side": "buy", "status": "filled", "price": 50.0},
        ]
        update_entry_prices(state, filled, broker, {})
        assert state["entry_prices"]["AAAA"] == 100.0  # add kept original base
        assert state["entry_prices"]["BBBB"] == 50.0   # new position recorded

    def test_full_sell_clears_entry(self):
        from paper_trade_common import update_entry_prices

        broker = MagicMock()
        broker.get_positions.return_value = pd.Series(dtype=float)
        state = {"entry_prices": {"AAAA": 100.0}}
        filled = [
            {"symbol": "AAAA", "side": "sell", "status": "filled", "price": 120.0},
        ]
        update_entry_prices(state, filled, broker, {})
        assert "AAAA" not in state["entry_prices"]

    def test_partial_sell_keeps_entry(self):
        from paper_trade_common import update_entry_prices

        broker = MagicMock()
        broker.get_positions.return_value = pd.Series({"AAAA": 50.0})
        state = {"entry_prices": {"AAAA": 100.0}}
        filled = [
            {"symbol": "AAAA", "side": "sell", "status": "filled", "price": 120.0},
        ]
        update_entry_prices(state, filled, broker, {})
        assert state["entry_prices"]["AAAA"] == 100.0


# ── Daily safety counter persistence ────────────────────────────────────────

class TestDailyTrackerPersistence:
    def test_same_day_roundtrip(self):
        from quant.execution.safety import DailyTracker

        t = DailyTracker()
        t.total_value_traded = 12_345.0
        t.orders_filled = 3
        snapshot = t.to_dict()

        t2 = DailyTracker()
        assert t2.restore(snapshot) is True
        assert t2.total_value_traded == 12_345.0
        assert t2.orders_filled == 3

    def test_stale_snapshot_ignored(self):
        from quant.execution.safety import DailyTracker

        t = DailyTracker()
        snapshot = t.to_dict()
        snapshot["trade_date"] = "2020-01-01"
        snapshot["total_value_traded"] = 99_999.0

        t2 = DailyTracker()
        assert t2.restore(snapshot) is False
        assert t2.total_value_traded == 0.0

    def test_garbage_snapshot_ignored(self):
        from quant.execution.safety import DailyTracker

        t = DailyTracker()
        assert t.restore({}) is False
        assert t.restore({"trade_date": "not-a-date"}) is False

    def test_same_day_nonfinite_snapshot_fails_closed(self):
        from quant.execution.safety import DailyTracker

        t = DailyTracker()
        snapshot = t.to_dict()
        snapshot["total_value_traded"] = float("nan")

        with pytest.raises(ValueError, match="unsafe counters"):
            DailyTracker().restore(snapshot)


# ── Market-data timing and live quality gate ───────────────────────────────

class TestMarketDataTiming:
    def test_explicit_end_date_is_inclusive(self):
        from quant.data.market_data import MarketData

        assert MarketData._exclusive_download_end("2026-07-16") == "2026-07-17"

    def test_default_excludes_intraday_candle_before_close(self):
        from quant.data.market_data import MarketData

        assert MarketData._exclusive_download_end(
            now=datetime(2026, 7, 16, 11, 0)
        ) == "2026-07-16"

    def test_default_includes_session_after_close_grace(self):
        from quant.data.market_data import MarketData

        assert MarketData._exclusive_download_end(
            now=datetime(2026, 7, 16, 16, 30)
        ) == "2026-07-17"

    def test_returns_do_not_bridge_missing_prices(self):
        from quant.data.market_data import MarketData

        prices = pd.DataFrame({"AAAA": [100.0, np.nan, 110.0]})
        assert MarketData.compute_returns(prices).empty


# ── Live data quality gate ──────────────────────────────────────────────────

class TestLiveQualityGate:
    def _prices(self, n_days=250, n_syms=10):
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        data = {
            f"S{i}": np.linspace(100, 130, n_days) + i * 3
            for i in range(n_syms)
        }
        data["BENCH"] = np.linspace(100, 115, n_days)
        return pd.DataFrame(data, index=dates)

    def test_clean_data_passes_untouched(self):
        from quant.data.quality import enforce_live_data_quality

        prices = self._prices()
        out = enforce_live_data_quality(prices, benchmark="BENCH")
        assert set(out.columns) == set(prices.columns)

    def test_dead_symbol_dropped_not_fatal(self):
        from quant.data.quality import enforce_live_data_quality

        prices = self._prices()
        prices["S0"] = np.nan  # one dead ticker must not brick the run
        out = enforce_live_data_quality(prices, benchmark="BENCH")
        assert "S0" not in out.columns
        assert "S1" in out.columns

    def test_breadth_collapse_aborts(self):
        from quant.data.quality import enforce_live_data_quality

        prices = self._prices()
        for i in range(5):  # half the universe dead -> broken feed
            prices[f"S{i}"] = np.nan
        with pytest.raises(RuntimeError, match="quality gate"):
            enforce_live_data_quality(prices, benchmark="BENCH")

    def test_dead_benchmark_aborts(self):
        from quant.data.quality import enforce_live_data_quality

        prices = self._prices()
        prices["BENCH"] = np.nan
        with pytest.raises(RuntimeError, match="benchmark"):
            enforce_live_data_quality(prices, benchmark="BENCH")

    def test_missing_benchmark_aborts(self):
        from quant.data.quality import enforce_live_data_quality

        prices = self._prices().drop(columns=["BENCH"])
        with pytest.raises(RuntimeError, match="benchmark BENCH is missing"):
            enforce_live_data_quality(prices, benchmark="BENCH")

    def test_infinite_quote_is_sanitized(self):
        from quant.data.quality import enforce_live_data_quality

        prices = self._prices()
        prices.loc[prices.index[100], "S0"] = np.inf
        out = enforce_live_data_quality(prices, benchmark="BENCH")
        assert pd.isna(out.loc[out.index[100], "S0"])

    def test_short_history_aborts(self):
        from quant.data.quality import enforce_live_data_quality

        prices = self._prices(n_days=60)  # a half-fetched window
        with pytest.raises(RuntimeError, match="quality gate"):
            enforce_live_data_quality(prices, benchmark="BENCH")
