"""The earnings book: a fourth paper strategy on the whole S&P 500.

Why it exists.  Measured on the project's own IC tools (research/factor_lab.py
and research/earnings_lab.py, October 2026), the five price factors of the
multi-factor book have no ranking power on the hand-picked 100-name universe,
and that universe's own equal-weighted return, with the books' leverage,
explains the backtests.  On the point-in-time S&P 500 the one set of signals
whose 21-day IC was positive in every sub-period since 2010 was built from
announcements: the consensus earnings surprise, year-on-year EPS growth, the
distance to the next announcement, and intermediate (12-6 month) momentum.
This book trades exactly that composite.  The trend book's machinery
(inverse-volatility sizing, volatility targeting, regime lights, trailing
stop and cooldown) is available through the same parameters, but the
variants run on 2026-10-22 (research/earnings_book.py, runs 39745666138 to
39756596377, docs/research/2026-10-22-alpha-audit.md) showed every one of
those pieces costing return on this ranking: equal weights, no stops, no
lights and a fixed gross exposure, rebalanced every five sessions, on the
three signals above, is what config_earnings.yaml runs (run 39757846026:
2010-2026 CAGR 16.7% against SPY's 14.7%, five years to 2026-10-22 23.1%
against 12.8%, unlevered; about a point of CAGR moves between runs with
Yahoo's coverage of the calendar).

Universe.  The membership history of the S&P 500 (quant/data/earnings.py), so
a backtest ranks each day's members and not today's list.  Delisted names
have no price history and drop out, which still flatters the universe; the
honest comparison for this book is its own paper record.

Signals (all read at the decision close, nothing from the future):

1.  ``sue``: reported EPS minus the consensus estimate, over the standard
    deviation of the last eight such errors, from the latest announcement.
2.  ``eps_growth``: reported EPS against the same quarter a year earlier.
3.  ``days_to_next``: minus the days to the next scheduled announcement, so
    names about to report rank first (the announcement premium).  Weighted
    zero in config_earnings.yaml: in the backtest its gain came from reading
    the realized date before a company would have confirmed it, and with the
    date expected a quarter after the last one until it is near
    (``next_known_within_days``) it cost return.
4.  ``momentum``: the return from twelve to six months ago (skipping the
    most recent six), the horizon that kept a positive IC when 12-1 did not.

Each part is z-scored across that day's eligible members, a missing part
counts as zero, and the parts are averaged with the configured weights; a
name needs at least ``min_parts`` of them.  Selection, sizing, leverage,
regime and exits are the trend book's rules (see quant/trend_strategy.py),
with this book's own parameters under ``earnings:`` in config_earnings.yaml.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from quant.backtest.engine import BacktestResult
from quant.data.earnings import (
    DEFAULT_UNIVERSE_FILE,
    announcement_panels,
    current_members,
    fetch_earnings,
    load_sp500_membership,
    members_since,
    membership_mask,
)
from quant.trend_strategy import TrendStrategy

logger = logging.getLogger(__name__)


class EarningsStrategy(TrendStrategy):
    """See the module docstring."""

    def __init__(self, config: dict):
        e = dict(config.get("earnings", {}))
        # The trend machinery reads its parameters from ``trend``; this book
        # keeps them under ``earnings`` so the two configs cannot be confused.
        super().__init__({**config, "trend": e})
        self.config = config
        weights = e.get("weights", {"sue": 1.0, "eps_growth": 1.0, "days_to_next": 1.0, "momentum": 1.0})
        total = float(sum(float(v) for v in weights.values()))
        self.weights = {k: float(v) / total for k, v in weights.items() if float(v) != 0.0}
        self.momentum_lookback = int(e.get("momentum_lookback", 252))
        self.momentum_skip = int(e.get("momentum_skip", 126))
        self.sue_lookback = int(e.get("sue_lookback", 8))
        # Backtests only: treat the next announcement date as unknown until it
        # is this close (the realized calendar would otherwise be read early).
        known = e.get("next_known_within_days")
        self.next_known_within_days = None if known in (None, "", False) else float(known)
        self.min_parts = int(e.get("min_parts", 2))
        # equal: every chosen name gets 1/n (the research simulation's book);
        # inverse_vol: the trend book's sizing.
        self.sizing = str(e.get("sizing", "inverse_vol"))
        # The trend book's absolute gate, optional here: only names closing
        # above their long average can be ranked.
        self.require_uptrend = bool(e.get("require_uptrend", False))
        # A fixed gross exposure instead of the volatility target: the book
        # holds this times its equity (capped by the regime and leverage caps).
        fixed = e.get("fixed_gross")
        self.fixed_gross = None if fixed in (None, "", False) else float(fixed)
        self.earnings_limit = int(e.get("earnings_limit", 100))
        self.universe_file = e.get("universe_file", DEFAULT_UNIVERSE_FILE)
        self.universe_lead_days = int(e.get("universe_lead_days", 400))
        self._snapshots: Optional[pd.DataFrame] = None
        # Announcement history: set by run_backtest (or a test), fetched once
        # per process on the live path, and only when a scheduled target
        # needs it (the daily stop check does not).
        self._events: Optional[dict] = None

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    @property
    def snapshots(self) -> pd.DataFrame:
        if self._snapshots is None:
            self._snapshots = load_sp500_membership(self.universe_file)
        return self._snapshots

    def warmup_days(self) -> int:
        return max(self.momentum_lookback, self.sma_long) + 5

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def precompute(self, prices: pd.DataFrame) -> dict:
        pre = super().precompute(prices)
        px = pre["px"]
        pre["momentum"] = px.shift(self.momentum_skip) / px.shift(self.momentum_lookback) - 1.0
        pre["mask"] = membership_mask(self.snapshots, px.index, px.columns) & px.notna()
        pre["announcements"] = None   # built on first use, see _announcements
        return pre

    def _announcements(self, pre: dict) -> dict:
        if pre["announcements"] is None:
            px = pre["px"]
            events = self._events
            if events is None:
                events = self._events = fetch_earnings(list(px.columns), limit=self.earnings_limit)
            pre["announcements"] = announcement_panels(
                events, px.index, px, sue_lookback=self.sue_lookback,
                next_known_within_days=self.next_known_within_days,
            )
        return pre["announcements"]

    def parts_on(self, pre: dict, date) -> pd.DataFrame:
        """The raw signal parts for every member with a price at ``date``."""
        px = pre["px"].loc[date]
        eligible = pre["mask"].loc[date] & px.notna()
        if self.require_uptrend:
            eligible &= px > pre["sma_long"].loc[date]
        names = eligible.index[eligible]
        ann = self._announcements(pre)
        parts = {}
        if "sue" in self.weights:
            parts["sue"] = ann["sue_consensus"].loc[date].reindex(names)
        if "eps_growth" in self.weights:
            parts["eps_growth"] = ann["eps_growth_yoy"].loc[date].reindex(names)
        if "days_to_next" in self.weights:
            parts["days_to_next"] = -ann["days_to_next"].loc[date].reindex(names)
        if "momentum" in self.weights:
            parts["momentum"] = pre["momentum"].loc[date].reindex(names)
        return pd.DataFrame(parts)

    def scores_on(self, pre: dict, date) -> pd.Series:
        """Weighted average of the z-scored parts across the day's members."""
        parts = self.parts_on(pre, date)
        if parts.empty:
            return pd.Series(dtype=float)
        available = parts.notna().sum(axis=1)
        parts = parts[available >= self.min_parts]
        if parts.empty:
            return pd.Series(dtype=float)
        score = pd.Series(0.0, index=parts.index)
        for name, weight in self.weights.items():
            if name not in parts:
                continue
            z = self._zscore(parts[name]).clip(-3.0, 3.0)
            score = score.add(weight * z.reindex(parts.index).fillna(0.0), fill_value=0.0)
        return score.dropna().sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Backtest and live
    # ------------------------------------------------------------------

    def run_backtest(self, start: str = None, end: str = None) -> BacktestResult:
        bt_cfg = self.config["backtest"]
        start = start or bt_cfg["start_date"]
        end = end or bt_cfg.get("end_date")
        symbols = members_since(self.snapshots, start, self.universe_lead_days)
        logger.warning(
            "Point-in-time S&P 500 membership: %d names were members since %s; those "
            "without a price history drop out, so the universe still favours survivors",
            len(symbols), start,
        )
        self.data.symbols = symbols
        prices = self.data.fetch_prices(start=self.warmup_start(start), end=end)
        enough = prices.notna().sum() >= 252
        keep = [c for c in prices.columns if c == self.data.benchmark or enough.get(c, False)]
        prices = prices[keep]
        if self.data.last_open_prices_ is not None:
            self.data.last_open_prices_ = self.data.last_open_prices_.reindex(columns=prices.columns)
        if self.data.last_volumes_ is not None:
            self.data.last_volumes_ = self.data.last_volumes_.reindex(columns=prices.columns)
        if self._events is None:
            self._events = fetch_earnings([c for c in prices.columns if c != self.data.benchmark], limit=self.earnings_limit)
        result = self.backtest_on(prices, start, end)
        result.metrics["Point-in-Time Universe"] = True
        result.metrics["Survivorship Bias Warning"] = True
        return result

    def gross_target(self, returns: pd.DataFrame, weights: pd.Series, cap: float) -> float:
        if self.fixed_gross is None:
            return super().gross_target(returns, weights, cap)
        if weights.empty or cap <= 0:
            return 0.0
        return float(min(self.fixed_gross, cap))

    def trailing_exits(self, pre: dict, date, held, entry_dates: Optional[dict] = None) -> list[str]:
        # trailing_stop 0 turns the trailing stop off (the research book runs
        # without one); the trend book's rule would read 0 as "sell everything".
        if self.trailing_stop <= 0:
            return []
        return super().trailing_exits(pre, date, held, entry_dates)

    def inverse_vol_weights(self, returns: pd.DataFrame, symbols: list[str]) -> pd.Series:
        if self.sizing != "equal":
            return super().inverse_vol_weights(returns, symbols)
        if not symbols:
            return pd.Series(dtype=float)
        w = pd.Series(1.0 / len(symbols), index=list(symbols))
        return w.clip(upper=self.max_weight)

    def _live_prices(self) -> pd.DataFrame:
        self.data.symbols = current_members(self.snapshots)
        return super()._live_prices()
