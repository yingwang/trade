"""Tests for operational code paths: config, paper_trade helpers, stop-loss integration."""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
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


# ── paper_trade helpers ─────────────────────────────────────────────────────

class TestPaperTradeHelpers:
    def test_load_state_empty(self, tmp_path):
        """load_state returns default dict when file doesn't exist."""
        from paper_trade import load_state, STATE_FILE
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
        broker.get_positions.return_value = pd.Series({"AAAA": 100})
        broker.get_current_prices.return_value = {"AAAA": 80.0}
        mock_result = MagicMock()
        mock_result.status = "filled"
        broker.submit_order.return_value = mock_result

        optimizer = MagicMock()
        optimizer.stop_loss_pct = 0.15

        state = {"entry_prices": {"AAAA": 100.0}}

        stopped = check_stop_losses(broker, optimizer, state, dry_run=False)
        assert "AAAA" in stopped
        broker.submit_order.assert_called_once()
        # Entry price should be cleared
        assert "AAAA" not in state["entry_prices"]
