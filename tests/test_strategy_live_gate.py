"""The live price gate's straggler rule and the held-name check
(review 2026-11-21, F7 and execution finding 5)."""

import logging

import numpy as np
import pandas as pd
import pytest

from quant.data.quality import (
    HeldDataMissingError,
    enforce_live_data_quality,
    fill_one_trailing_bar,
    require_held_symbols_kept,
)

pytestmark = pytest.mark.filterwarnings("ignore:SURVIVORSHIP BIAS WARNING")


def _frame(n_days=260, n_syms=10, end=None):
    end = end if end is not None else pd.Timestamp.now().normalize()
    dates = pd.bdate_range(end=end, periods=n_days)
    data = {f"S{i}": np.linspace(100, 130, n_days) + i * 3 for i in range(n_syms)}
    data["BENCH"] = np.linspace(100, 115, n_days)
    return pd.DataFrame(data, index=dates)


class TestTrailingBar:
    def test_only_a_single_missing_last_bar_is_carried(self, caplog):
        prices = _frame()
        prices.iloc[-1, prices.columns.get_loc("S1")] = np.nan          # straggler
        prices.iloc[100, prices.columns.get_loc("S2")] = np.nan         # interior gap
        prices.iloc[-2:, prices.columns.get_loc("S3")] = np.nan         # two bars missing
        with caplog.at_level(logging.WARNING, logger="quant.data.quality"):
            out = fill_one_trailing_bar(prices)
        assert out["S1"].iloc[-1] == prices["S1"].iloc[-2]
        assert pd.isna(out["S2"].iloc[100])
        assert out["S3"].iloc[-2:].isna().all()
        assert "S1" in caplog.text and "S2" not in caplog.text and "S3" not in caplog.text
        # Nothing else moved.
        pd.testing.assert_frame_equal(out.drop(columns="S1"), prices.drop(columns="S1"))

    def test_gate_returns_the_straggler_with_its_previous_close(self):
        prices = _frame()
        prices.iloc[-1, prices.columns.get_loc("S1")] = np.nan
        out = enforce_live_data_quality(prices, benchmark="BENCH", as_of=pd.Timestamp.now())
        assert "S1" in out.columns
        assert out["S1"].iloc[-1] == pytest.approx(prices["S1"].iloc[-2])
        assert out.iloc[-1].notna().all()


    def test_a_half_formed_last_bar_is_not_filled_but_refused(self):
        """Follow-up 5: more stragglers than the breadth rule tolerates
        (1 - min_valid_frac = 20% by default) is a broken bar, not a lag."""
        prices = _frame()
        for sym in ("S1", "S2", "S3"):                     # 3 of 10: 30%
            prices.iloc[-1, prices.columns.get_loc(sym)] = np.nan
        with pytest.raises(RuntimeError, match="quality gate: 3/10 symbols have no"):
            enforce_live_data_quality(prices, benchmark="BENCH", as_of=pd.Timestamp.now())
        prices.iloc[-1, prices.columns.get_loc("S3")] = prices["S3"].iloc[-2]   # 2 of 10: 20%
        out = enforce_live_data_quality(prices, benchmark="BENCH", as_of=pd.Timestamp.now())
        assert out.iloc[-1].notna().all()


class TestHeldSymbolCheck:
    def test_a_dropped_holding_aborts_and_an_off_universe_one_passes(self):
        kept = _frame().drop(columns=["S0"])
        expected = [f"S{i}" for i in range(10)] + ["BENCH"]
        with pytest.raises(HeldDataMissingError, match=r"held symbol\(s\) \['S0'\]") as raised:
            require_held_symbols_kept(pd.Series({"S0": 0.1, "S1": 0.1}), kept, expected)
        assert raised.value.symbols == ["S0"]
        assert isinstance(raised.value, RuntimeError)
        # A name the run never asked for (it left the universe) is a real exit.
        require_held_symbols_kept(pd.Series({"OLD": 0.1, "S1": 0.1}), kept, expected)
        # A zero weight is not a holding; None is an empty book.
        require_held_symbols_kept(pd.Series({"S0": 0.0}), kept, expected)
        require_held_symbols_kept(None, kept, expected)

    @staticmethod
    def _today_prices(synthetic_prices):
        prices = synthetic_prices.copy()
        prices.index = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=len(prices))
        return prices

    def test_multifactor_live_target_refuses_to_sell_a_holding_with_no_data(
        self, config, synthetic_prices, monkeypatch,
    ):
        from quant.strategy import MultiFactorStrategy

        prices = self._today_prices(synthetic_prices)
        prices["AAAA"] = np.nan  # the fetch failed for a held name
        strategy = MultiFactorStrategy(config)
        monkeypatch.setattr(strategy.data, "fetch_prices", lambda start=None, end=None: prices)
        monkeypatch.setattr(strategy.data, "fetch_fundamentals", lambda **kwargs: pd.DataFrame())
        with pytest.raises(RuntimeError, match="AAAA"):
            strategy.get_current_portfolio(
                capital=1_000_000.0, prev_weights=pd.Series({"AAAA": 0.2, "BBBB": 0.2}),
            )
        # Not held: the dead ticker is dropped and the run goes on.
        table = strategy.get_current_portfolio(
            capital=1_000_000.0, prev_weights=pd.Series({"BBBB": 0.2}),
        )
        assert "AAAA" not in table.index

    def test_lgbm_live_target_refuses_to_sell_a_holding_with_no_data(
        self, config, synthetic_prices, monkeypatch,
    ):
        from quant.signals.lgbm_strategy import LGBMStrategy

        prices = self._today_prices(synthetic_prices)
        prices["AAAA"] = np.nan
        strategy = LGBMStrategy(config)
        monkeypatch.setattr(strategy.data, "fetch_prices", lambda start=None, end=None: prices)
        monkeypatch.setattr(strategy.data, "fetch_fundamentals", lambda **kwargs: pd.DataFrame())
        with pytest.raises(RuntimeError, match="AAAA"):
            strategy.get_current_portfolio(
                capital=1_000_000.0, prev_weights=pd.Series({"AAAA": 0.2, "BBBB": 0.2}),
            )


def test_earnings_live_target_refuses_to_sell_a_holding_with_no_data(config, monkeypatch):
    """The same check on the earnings book, through TrendStrategy's live
    price path.  Needs the F7 patch to quant/trend_strategy.py."""
    from quant.earnings_strategy import EarningsStrategy
    from tests.test_earnings_strategy import SYMBOLS, _config, _events, _prices

    strategy = EarningsStrategy(_config(config))
    prices = _prices()
    prices.index = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=len(prices))
    strategy._snapshots = pd.DataFrame(True, index=[prices.index[0]], columns=SYMBOLS)
    strategy._events = _events(prices.index)
    prices["AAAA"] = np.nan  # the fetch failed for a held name
    monkeypatch.setattr(
        strategy.data, "fetch_prices",
        lambda start=None, end=None: prices[
            [c for c in prices.columns if c in strategy.data.symbols or c == "BENCH"]
        ],
    )
    with pytest.raises(RuntimeError, match="AAAA"):
        strategy.get_current_portfolio(
            capital=100_000.0, prev_weights=pd.Series({"AAAA": 0.04, "BBBB": 0.04}),
        )


class TestDashboardTarget:
    """Follow-up 2: the dashboards display the target and must not fail the
    nightly build when a holding has no price data."""

    def test_held_name_without_data_is_logged_and_the_page_goes_on(
        self, config, synthetic_prices, monkeypatch, caplog,
    ):
        from quant.strategy import MultiFactorStrategy
        from site_common import current_portfolio_for_display

        prices = TestHeldSymbolCheck._today_prices(synthetic_prices)
        prices["AAAA"] = np.nan
        strategy = MultiFactorStrategy(config)
        monkeypatch.setattr(strategy.data, "fetch_prices", lambda start=None, end=None: prices)
        monkeypatch.setattr(strategy.data, "fetch_fundamentals", lambda **kwargs: pd.DataFrame())
        held = pd.Series({"AAAA": 0.2, "BBBB": 0.2})
        with caplog.at_level(logging.WARNING, logger="site_common"):
            table, missing = current_portfolio_for_display(strategy, held, capital=1_000_000.0)
        assert missing == ["AAAA"]
        assert "AAAA" in caplog.text and "stand down" in caplog.text
        assert "AAAA" not in table.index and not table.empty
        # What the attribution and regime sections read is in place.
        assert strategy.last_prices_ is not None

        prices["AAAA"] = synthetic_prices["AAAA"].to_numpy()
        table, missing = current_portfolio_for_display(strategy, held, capital=1_000_000.0)
        assert missing == []

    def test_other_gate_errors_still_fail_the_build(self, config, synthetic_prices, monkeypatch):
        from quant.strategy import MultiFactorStrategy
        from site_common import current_portfolio_for_display

        prices = TestHeldSymbolCheck._today_prices(synthetic_prices)
        for sym in ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"]:
            prices[sym] = np.nan   # a broken feed, not one missing holding
        strategy = MultiFactorStrategy(config)
        monkeypatch.setattr(strategy.data, "fetch_prices", lambda start=None, end=None: prices)
        with pytest.raises(RuntimeError, match="usable") as raised:
            current_portfolio_for_display(strategy, pd.Series({"FFFF": 0.2}), capital=1_000_000.0)
        assert not isinstance(raised.value, HeldDataMissingError)
