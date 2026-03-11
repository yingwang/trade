"""Backtesting engine for evaluating trading strategies on historical data."""

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
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:30s}: {v:>12.4f}")
            else:
                lines.append(f"  {k:30s}: {v!s:>12s}")
        lines.append("=" * 60)
        return "\n".join(lines)


class BacktestEngine:
    """Event-driven daily backtester."""

    def __init__(self, config: dict):
        bt_cfg = config["backtest"]
        port_cfg = config["portfolio"]

        self.initial_capital = bt_cfg["initial_capital"]
        self.slippage_bps = bt_cfg["slippage_bps"]
        self.txn_cost_bps = port_cfg["transaction_cost_bps"]
        self.rebalance_freq = port_cfg["rebalance_frequency_days"]

    def run(self, prices: pd.DataFrame, target_weights_by_date: dict[str, pd.Series],
            benchmark_col: str = "SPY") -> BacktestResult:
        """Run backtest.

        Parameters
        ----------
        prices : DataFrame
            Daily adjusted close prices (dates x symbols), must include benchmark.
        target_weights_by_date : dict
            Mapping of rebalance date (str or Timestamp) -> pd.Series of target weights.
        benchmark_col : str
            Column name for the benchmark.
        """
        result = BacktestResult()

        dates = prices.index.sort_values()
        symbols = [c for c in prices.columns if c != benchmark_col]

        cash = self.initial_capital
        holdings = pd.Series(0.0, index=symbols)  # number of shares
        equity_history = {}
        benchmark_start = prices[benchmark_col].iloc[0] if benchmark_col in prices.columns else 1.0

        current_weights = pd.Series(0.0, index=symbols)

        rebalance_dates = set()
        for d in target_weights_by_date:
            rd = pd.Timestamp(d)
            if rd in dates:
                rebalance_dates.add(rd)

        for date in dates:
            px = prices.loc[date, symbols].ffill()
            portfolio_value = cash + (holdings * px).sum()

            # Rebalance if this is a rebalance date
            if date in rebalance_dates:
                target = target_weights_by_date.get(str(date.date()),
                         target_weights_by_date.get(date, pd.Series(dtype=float)))
                target = target.reindex(symbols).fillna(0)

                target_shares = (portfolio_value * target / px).fillna(0).apply(np.floor)
                trades = target_shares - holdings

                # Apply transaction costs and slippage
                trade_value = (trades.abs() * px).sum()
                cost = trade_value * (self.txn_cost_bps + self.slippage_bps) / 10000
                cash -= cost

                # Execute trades
                trade_cash = (trades * px).sum()
                cash -= trade_cash
                holdings = target_shares
                current_weights = target

                result.trades.append({
                    "date": date,
                    "turnover": trade_value / portfolio_value if portfolio_value > 0 else 0,
                    "cost": cost,
                })

            portfolio_value = cash + (holdings * px).sum()
            equity_history[date] = portfolio_value
            result.positions_history.append({
                "date": date,
                "holdings": holdings.copy(),
                "cash": cash,
                "value": portfolio_value,
            })

        result.equity_curve = pd.Series(equity_history).sort_index()
        if benchmark_col in prices.columns:
            result.benchmark_curve = (
                prices[benchmark_col].reindex(result.equity_curve.index)
                / benchmark_start * self.initial_capital
            )

        result.returns = result.equity_curve.pct_change().dropna()
        result.metrics = self._compute_metrics(result)
        return result

    def _compute_metrics(self, result: BacktestResult) -> dict:
        eq = result.equity_curve
        ret = result.returns
        bm_ret = result.benchmark_curve.pct_change().dropna() if not result.benchmark_curve.empty else pd.Series(dtype=float)

        total_return = (eq.iloc[-1] / eq.iloc[0]) - 1 if len(eq) > 1 else 0
        n_years = len(ret) / 252 if len(ret) > 0 else 1
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        annual_vol = ret.std() * np.sqrt(252) if len(ret) > 1 else 0
        sharpe = cagr / annual_vol if annual_vol > 0 else 0

        # Max drawdown
        peak = eq.cummax()
        dd = (eq - peak) / peak
        max_dd = dd.min()

        # Calmar ratio
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # Sortino
        downside = ret[ret < 0].std() * np.sqrt(252) if (ret < 0).any() else 0
        sortino = cagr / downside if downside > 0 else 0

        # Win rate
        win_rate = (ret > 0).mean() if len(ret) > 0 else 0

        # Turnover
        avg_turnover = np.mean([t["turnover"] for t in result.trades]) if result.trades else 0

        # Information ratio vs benchmark
        if len(bm_ret) > 0:
            excess = ret.reindex(bm_ret.index) - bm_ret
            ir = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
            bm_total = (result.benchmark_curve.iloc[-1] / result.benchmark_curve.iloc[0]) - 1
        else:
            ir = 0
            bm_total = 0

        return {
            "Total Return": total_return,
            "CAGR": cagr,
            "Annualized Volatility": annual_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Max Drawdown": max_dd,
            "Win Rate (daily)": win_rate,
            "Avg Rebalance Turnover": avg_turnover,
            "Information Ratio": ir,
            "Benchmark Return": bm_total,
            "Num Trades": len(result.trades),
        }
