"""Leveraged trend-following: the third paper account.

A deliberately different animal from the two factor books. It holds a handful of
the strongest trends in the universe, sizes them by inverse volatility, scales the
whole book to a volatility target with margin, and gets out of the way when the
market itself stops trending. Nothing here is optimized in the mean-variance sense;
every rule is a threshold on a price series, so the live path and the backtest
compute the same thing from the same closes.

Rules, in the order they are applied:

1.  Eligibility. A name can be held only while it closes above its 200-day
    average and its six-month (skip-month) return is positive. This is the
    absolute-momentum gate; a stock has to be going up on its own terms before its
    relative rank matters.
2.  Ranking. Cross-sectional z-scores of three momentum horizons, 12-1 month,
    6-1 month and 3 month, blended 40/30/30. The top ``positions`` names are held,
    with a rank buffer: a name already in the book stays as long as it is still
    inside the top ``rank_buffer``, so the book only turns over when a holding has
    genuinely faded, not on every reordering near the cut.
3.  Sizing. Inverse 60-day volatility, each name capped, renormalized to one.
4.  Leverage. The book is scaled so that its trailing 60-day covariance implies
    the target volatility, capped by ``leverage.max_leverage`` and by the regime.
5.  Regime. Three warning lights on the market: the benchmark below its 200-day
    average, benchmark 21-day realized volatility above a threshold, and fewer
    than a set fraction of the universe above their 50-day averages. One light
    caps gross exposure at ``regime_caps[1]``; two or more send the book to cash.
    Exposure is cut the day a light comes on and only rebuilt on the next
    scheduled rebalance: fast out, slow back in.
6.  Exits between rebalances. Every session, any holding that has fallen
    ``trailing_stop`` from its 60-day high is sold at the next open, and the whole
    book is scaled down if the regime cap has dropped below what is held. The
    fixed stop from entry (``risk.stop_loss_pct``) is enforced by the engine and
    the live runner as for the other strategies.

The scheduled rebalance is weekly on the shared anchored calendar. The daily
checks in rule 6 are what the live runner's rebalance trigger asks for; in the
backtest the engine asks every session and the provider answers None whenever
there is nothing to do.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from quant.backtest.calendar import fixed_rebalance_dates
from quant.backtest.engine import BacktestEngine, BacktestResult
from quant.data.market_data import MarketData
from quant.data.quality import (
    DataQualityChecker,
    enforce_live_data_quality,
    warn_survivorship_bias,
)
from quant.portfolio.optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


class TrendStrategy:
    """See the module docstring for the rules."""

    def __init__(self, config: dict):
        self.config = config
        self.data = MarketData(config)
        # Only its risk limits are used: the entry stop the runners enforce, the
        # per-position safety cap and the leverage ceiling.
        self.optimizer = PortfolioOptimizer(config)
        self.backtest_engine = BacktestEngine(config)

        t = config.get("trend", {})
        self.positions = int(t.get("positions", 10))
        self.rank_buffer = int(t.get("rank_buffer", 15))
        self.skip_days = int(t.get("skip_days", 21))
        self.lookbacks = {
            "long": int(t.get("lookback_long", 252)),
            "mid": int(t.get("lookback_mid", 126)),
            "short": int(t.get("lookback_short", 63)),
        }
        blend = t.get("blend", {"long": 0.4, "mid": 0.3, "short": 0.3})
        total = float(sum(blend.values()))
        self.blend = {k: float(v) / total for k, v in blend.items()}
        self.vol_window = int(t.get("vol_window", 60))
        self.sma_long = int(t.get("sma_long", 200))
        self.sma_short = int(t.get("sma_short", 50))
        self.trailing_stop = float(t.get("trailing_stop", 0.15))
        self.trailing_window = int(t.get("trailing_window", 60))
        self.benchmark_vol_window = int(t.get("benchmark_vol_window", 21))
        self.benchmark_vol_high = float(t.get("benchmark_vol_high", 0.30))
        self.breadth_min = float(t.get("breadth_min", 0.30))
        caps = t.get("regime_caps", {0: None, 1: 0.6, 2: 0.0})
        self.regime_caps = {int(k): (None if v is None else float(v)) for k, v in caps.items()}
        self.deleverage_tolerance = float(t.get("deleverage_tolerance", 0.10))

        pcfg = config["portfolio"]
        self.target_vol = float(pcfg["target_volatility"])
        self.max_weight = float(pcfg["max_position_weight"])
        self.rebalance_freq = int(pcfg["rebalance_frequency_days"])
        self.max_leverage = float(config.get("leverage", {}).get("max_leverage", 1.0))
        self.anchor = config.get("backtest", {}).get("rebalance_anchor_date", "2000-01-03")

        # Set by the live runner's trigger before an unscheduled run.
        self.next_mode = "scheduled"
        self.last_prices_: Optional[pd.DataFrame] = None
        self.last_scores_: Optional[pd.Series] = None
        # What the last decision saw, for the dashboard and the logs.
        self.last_decision_: dict = {}
        self._live_cache: Optional[tuple[datetime, pd.DataFrame]] = None
        self._pending_target: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Pure building blocks (all take history up to and including the decision
    # close; nothing looks past the last row)
    # ------------------------------------------------------------------

    def warmup_days(self) -> int:
        return max(self.lookbacks["long"] + self.skip_days, self.sma_long) + 5

    def precompute(self, prices: pd.DataFrame) -> dict:
        """Rolling series shared by every decision date."""
        symbols = [c for c in prices.columns if c != self.data.benchmark]
        px = prices[symbols].apply(pd.to_numeric, errors="coerce")
        bench = pd.to_numeric(prices[self.data.benchmark], errors="coerce") \
            if self.data.benchmark in prices.columns else None
        out = {
            "symbols": symbols,
            "px": px,
            "returns": px.pct_change(fill_method=None),
            "sma_long": px.rolling(self.sma_long, min_periods=self.sma_long).mean(),
            "sma_short": px.rolling(self.sma_short, min_periods=self.sma_short).mean(),
            "rolling_high": px.rolling(self.trailing_window, min_periods=5).max(),
        }
        skip = self.skip_days
        out["mom_long"] = px.shift(skip) / px.shift(self.lookbacks["long"]) - 1.0
        out["mom_mid"] = px.shift(skip) / px.shift(self.lookbacks["mid"]) - 1.0
        out["mom_short"] = px / px.shift(self.lookbacks["short"]) - 1.0
        if bench is not None:
            out["bench"] = bench
            out["bench_sma_long"] = bench.rolling(self.sma_long, min_periods=self.sma_long).mean()
            out["bench_vol"] = (
                bench.pct_change(fill_method=None)
                .rolling(self.benchmark_vol_window, min_periods=self.benchmark_vol_window)
                .std() * np.sqrt(TRADING_DAYS)
            )
        return out

    @staticmethod
    def _zscore(row: pd.Series) -> pd.Series:
        clean = row.dropna()
        if len(clean) < 3 or clean.std(ddof=0) == 0:
            return pd.Series(0.0, index=clean.index)
        return (clean - clean.mean()) / clean.std(ddof=0)

    def scores_on(self, pre: dict, date) -> pd.Series:
        """Blended momentum score for every eligible name at ``date``'s close."""
        px = pre["px"].loc[date]
        eligible = (px > pre["sma_long"].loc[date]) & (pre["mom_mid"].loc[date] > 0)
        parts = []
        for key, weight in self.blend.items():
            series = pre[f"mom_{key}"].loc[date]
            parts.append(weight * self._zscore(series[eligible & series.notna()]))
        if not parts:
            return pd.Series(dtype=float)
        score = pd.concat(parts, axis=1).sum(axis=1, min_count=len(parts))
        return score.dropna().sort_values(ascending=False)

    def select(self, scores: pd.Series, held: Optional[pd.Series]) -> list[str]:
        """Top names with a rank buffer for what is already held."""
        if scores.empty:
            return []
        ranked = list(scores.index)
        rank = {sym: i + 1 for i, sym in enumerate(ranked)}
        held_syms = [s for s in (held.index if held is not None else []) if s in rank]
        keep = sorted(
            [s for s in held_syms if rank[s] <= self.rank_buffer],
            key=lambda s: rank[s],
        )[: self.positions]
        chosen = list(keep)
        for sym in ranked:
            if len(chosen) >= self.positions:
                break
            if sym not in chosen:
                chosen.append(sym)
        return chosen

    def inverse_vol_weights(self, returns: pd.DataFrame, symbols: list[str]) -> pd.Series:
        window = returns[symbols].tail(self.vol_window)
        vol = window.std(ddof=1).replace(0.0, np.nan)
        inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan).dropna()
        if inv.empty:
            return pd.Series(dtype=float)
        w = inv / inv.sum()
        # Cap and redistribute until nothing exceeds the cap.
        for _ in range(10):
            over = w > self.max_weight
            if not over.any():
                break
            excess = float((w[over] - self.max_weight).sum())
            w[over] = self.max_weight
            under = ~over
            if under.any() and w[under].sum() > 0:
                w[under] += excess * w[under] / w[under].sum()
            else:
                break
        return w

    def regime_lights(self, pre: dict, date) -> dict:
        """Which of the three market warnings are on at ``date``."""
        lights = {"benchmark_below_sma": False, "benchmark_vol_high": False, "breadth_weak": False}
        if "bench" in pre:
            b = pre["bench"].loc[date]
            s = pre["bench_sma_long"].loc[date]
            v = pre["bench_vol"].loc[date]
            lights["benchmark_below_sma"] = bool(pd.notna(b) and pd.notna(s) and b < s)
            lights["benchmark_vol_high"] = bool(pd.notna(v) and v > self.benchmark_vol_high)
        px = pre["px"].loc[date]
        sma = pre["sma_short"].loc[date]
        valid = px.notna() & sma.notna()
        if valid.sum() >= 10:
            breadth = float((px[valid] > sma[valid]).mean())
            lights["breadth_weak"] = breadth < self.breadth_min
            lights["breadth"] = breadth
        return lights

    def regime_cap(self, lights: dict) -> float:
        on = sum(1 for k in ("benchmark_below_sma", "benchmark_vol_high", "breadth_weak") if lights.get(k))
        cap = self.regime_caps.get(min(on, max(self.regime_caps)), None)
        return self.max_leverage if cap is None else min(cap, self.max_leverage)

    def gross_target(self, returns: pd.DataFrame, weights: pd.Series, cap: float) -> float:
        if weights.empty or cap <= 0:
            return 0.0
        window = returns[weights.index].tail(self.vol_window).dropna(how="all")
        if len(window) < 20:
            return min(1.0, cap)
        cov = window.cov().fillna(0.0).values
        w = weights.values
        port_vol = float(np.sqrt(max(w @ cov @ w, 0.0)) * np.sqrt(TRADING_DAYS))
        if not np.isfinite(port_vol) or port_vol <= 0:
            return min(1.0, cap)
        return float(min(self.target_vol / port_vol, cap))

    def trailing_exits(self, pre: dict, date, held: Optional[pd.Series]) -> list[str]:
        if held is None or len(held) == 0:
            return []
        px = pre["px"].loc[date]
        high = pre["rolling_high"].loc[date]
        out = []
        for sym in held.index:
            if sym not in px.index:
                continue
            p, h = px.get(sym), high.get(sym)
            if pd.notna(p) and pd.notna(h) and h > 0 and p < h * (1.0 - self.trailing_stop):
                out.append(sym)
        return out

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def scheduled_target(self, pre: dict, date, held: Optional[pd.Series]) -> pd.Series:
        scores = self.scores_on(pre, date)
        self.last_scores_ = scores
        chosen = self.select(scores, held)
        lights = self.regime_lights(pre, date)
        cap = self.regime_cap(lights)
        weights = self.inverse_vol_weights(pre["returns"].loc[:date], chosen) if chosen else pd.Series(dtype=float)
        gross = self.gross_target(pre["returns"].loc[:date], weights, cap)
        target = (weights * gross) if not weights.empty else pd.Series(dtype=float)
        target = self.optimizer.apply_hard_exposure_limits(target, gross_exposure_cap=cap) \
            if not target.empty else target
        self.last_decision_ = {
            "mode": "scheduled", "date": str(pd.Timestamp(date).date()),
            "regime": lights, "regime_cap": cap, "gross": float(target.sum()) if not target.empty else 0.0,
            "candidates": int(len(scores)), "selected": chosen,
        }
        return target[target > 0]

    def maintenance_target(self, pre: dict, date, held: Optional[pd.Series]) -> Optional[pd.Series]:
        """Exits and de-risking only; None when the book can stay as it is."""
        if held is None or len(held) == 0:
            return None
        held = held[held > 1e-6]
        if held.empty:
            return None
        exits = self.trailing_exits(pre, date, held)
        lights = self.regime_lights(pre, date)
        cap = self.regime_cap(lights)
        target = held.drop(labels=exits, errors="ignore").copy()
        gross = float(target.sum())
        scaled = False
        if gross > cap * (1.0 + self.deleverage_tolerance):
            target = target * (cap / gross) if gross > 0 else target
            scaled = True
        if not exits and not scaled:
            return None
        self.last_decision_ = {
            "mode": "maintenance", "date": str(pd.Timestamp(date).date()),
            "regime": lights, "regime_cap": cap, "exits": exits, "deleveraged": scaled,
            "gross": float(target.sum()),
        }
        return target[target > 0]

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def run_backtest(self, start: str = None, end: str = None) -> BacktestResult:
        bt_cfg = self.config["backtest"]
        start = start or bt_cfg["start_date"]
        end = end or bt_cfg.get("end_date")
        warn_survivorship_bias(self.data.symbols, start)

        warm = self.warmup_days()
        warmup_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=int(warm * 1.6))).strftime("%Y-%m-%d") if start else None
        prices = self.data.fetch_prices(start=warmup_start, end=end)
        report = DataQualityChecker().run_all_checks(prices)
        if not report["passed"]:
            logger.error("Data quality check FAILED:\n%s", DataQualityChecker.format_report(report))

        pre = self.precompute(prices)
        not_before = prices.index[0] + pd.Timedelta(days=int(warm * 1.5))
        scheduled = set(fixed_rebalance_dates(
            prices.index, self.rebalance_freq, anchor=self.anchor, not_before=not_before,
        ))
        decision_dates = [d for d in prices.index if d >= not_before]

        def provide(date, prev_weights):
            held = prev_weights if prev_weights is not None and len(prev_weights) else None
            if date in scheduled:
                return self.scheduled_target(pre, date, held)
            return self.maintenance_target(pre, date, held)

        backtest_prices = prices.loc[start:] if start else prices
        execution_prices = self.data.last_open_prices_
        volumes = self.data.last_volumes_
        if execution_prices is not None and start:
            execution_prices = execution_prices.loc[start:]
        if volumes is not None and start:
            volumes = volumes.loc[start:]
        result = self.backtest_engine.run(
            backtest_prices, {}, self.data.benchmark,
            execution_prices=execution_prices, volumes=volumes,
            target_provider=provide, rebalance_dates=decision_dates,
        )
        result.metrics["Survivorship Bias Warning"] = True
        result.metrics["Point-in-Time Universe"] = False
        return result

    # ------------------------------------------------------------------
    # Live
    # ------------------------------------------------------------------

    def _live_prices(self) -> pd.DataFrame:
        now = datetime.now()
        if self._live_cache is not None and (now - self._live_cache[0]) < timedelta(minutes=10):
            return self._live_cache[1]
        prices = enforce_live_data_quality(
            self.data.fetch_prices(), benchmark=self.data.benchmark, as_of=pd.Timestamp.now(),
        )
        self._live_cache = (now, prices)
        self.last_prices_ = prices
        return prices

    def maintenance_needed(self, prev_weights: Optional[pd.Series]) -> Optional[str]:
        """The live runner's trigger: a reason string when the book needs an
        unscheduled action today, else None. The target it computed is kept for
        the get_current_portfolio call that follows."""
        prices = self._live_prices()
        pre = self.precompute(prices)
        date = prices.index[-1]
        target = self.maintenance_target(pre, date, prev_weights)
        if target is None:
            self._pending_target = None
            return None
        self._pending_target = target
        d = self.last_decision_
        reasons = []
        if d.get("exits"):
            reasons.append(f"trailing stop on {', '.join(d['exits'])}")
        if d.get("deleveraged"):
            on = [k for k, v in d["regime"].items() if v is True]
            reasons.append(f"regime cap {d['regime_cap']:.2f} ({', '.join(on)})")
        self.next_mode = "maintenance"
        return "; ".join(reasons) or "maintenance"

    def get_current_signal(self) -> pd.Series:
        prices = self._live_prices()
        pre = self.precompute(prices)
        return self.scores_on(pre, prices.index[-1])

    def get_current_portfolio(self, capital: float = None, prev_weights: pd.Series = None) -> pd.DataFrame:
        if capital is None:
            capital = self.config["backtest"]["initial_capital"]
        prices = self._live_prices()
        pre = self.precompute(prices)
        date = prices.index[-1]
        held = prev_weights if prev_weights is not None and len(prev_weights) else None

        mode = self.next_mode
        self.next_mode = "scheduled"
        if mode == "maintenance":
            target = self._pending_target
            self._pending_target = None
            if target is None:
                target = self.maintenance_target(pre, date, held)
            if target is None:
                # Nothing to do after all: hand back the book unchanged.
                target = held if held is not None else pd.Series(dtype=float)
        else:
            target = self.scheduled_target(pre, date, held)

        latest = pre["px"].iloc[-1].reindex(target.index)
        dollars = target * capital
        shares = (dollars / latest).apply(np.floor).fillna(0).astype(int)
        scores = self.last_scores_ if self.last_scores_ is not None else pd.Series(dtype=float)
        result = pd.DataFrame({
            "score": scores.reindex(target.index),
            "weight": target,
            "weight_pct": (target * 100).round(2),
            "dollars": dollars.round(2),
            "shares": shares,
            "price": latest.round(2),
        }).sort_values("weight", ascending=False)
        logger.info(
            "Trend %s target: %d positions, gross %.1f%%, regime cap %.2f, lights %s",
            mode, len(result), float(target.sum()) * 100,
            self.last_decision_.get("regime_cap", float("nan")),
            {k: v for k, v in self.last_decision_.get("regime", {}).items() if v is True},
        )
        return result
