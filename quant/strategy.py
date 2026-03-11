"""Strategy orchestrator: ties together data, signals, portfolio, and execution."""

import logging

import numpy as np
import pandas as pd

from quant.data.market_data import MarketData
from quant.signals.factors import SignalGenerator
from quant.portfolio.optimizer import PortfolioOptimizer, RiskMonitor
from quant.backtest.engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


class MultiFactorStrategy:
    """Medium-term multi-factor equity strategy.

    Pipeline:
      1. Fetch historical price and fundamental data
      2. Compute alpha signals (momentum, mean-reversion, trend, vol, value, quality)
      3. On each rebalance date, select top stocks and optimize weights
      4. Pass target weights to backtester or live execution
    """

    def __init__(self, config: dict):
        self.config = config
        self.data = MarketData(config)
        self.signal_gen = SignalGenerator(config)
        self.optimizer = PortfolioOptimizer(config)
        self.risk_monitor = RiskMonitor(config)
        self.backtest_engine = BacktestEngine(config)

    def run_backtest(self, start: str = None, end: str = None) -> BacktestResult:
        """Full backtest pipeline."""
        bt_cfg = self.config["backtest"]
        start = start or bt_cfg["start_date"]
        end = end or bt_cfg.get("end_date")

        # 1. Fetch data
        logger.info("Fetching price data...")
        prices = self.data.fetch_prices(start=start, end=end)
        returns = MarketData.compute_returns(prices)

        logger.info("Fetching fundamentals...")
        try:
            fundamentals = self.data.fetch_fundamentals()
        except Exception as e:
            logger.warning("Could not fetch fundamentals: %s", e)
            fundamentals = pd.DataFrame()

        # 2. Generate signals
        logger.info("Generating signals...")
        signals = self.signal_gen.generate(prices, returns, fundamentals)

        # 3. Build target weights on rebalance dates
        logger.info("Computing target weights...")
        rebalance_freq = self.config["portfolio"]["rebalance_frequency_days"]
        symbols = [c for c in prices.columns if c != self.data.benchmark]

        rebalance_dates = prices.index[::rebalance_freq]
        # Need enough history for signals
        min_history = max(self.config["signals"]["momentum_windows"]) + 42
        rebalance_dates = [d for d in rebalance_dates
                           if (d - prices.index[0]).days > min_history]

        target_weights = {}
        for date in rebalance_dates:
            if date not in signals.index:
                continue

            day_scores = signals.loc[date].dropna()
            if day_scores.empty:
                continue

            selected = self.optimizer.select_top_stocks(day_scores)

            # Covariance from trailing returns
            ret_window = returns.loc[:date].tail(126)
            cov = ret_window[selected].cov() if len(ret_window) > 20 else pd.DataFrame(
                np.eye(len(selected)) * 0.04 / 252,
                index=selected, columns=selected,
            )

            weights = self.optimizer.optimize_weights(
                selected.tolist(), day_scores, cov
            )
            weights = self.optimizer.apply_vol_scaling(weights, cov)

            # Normalize back to sum=1 after vol scaling
            if weights.sum() > 0:
                weights /= weights.sum()

            target_weights[str(date.date())] = weights

        logger.info("Generated %d rebalance points", len(target_weights))

        # 4. Run backtest
        result = self.backtest_engine.run(prices, target_weights, self.data.benchmark)

        return result

    def get_current_signal(self) -> pd.Series:
        """Get the latest composite signal for live/paper trading decisions."""
        prices = self.data.fetch_prices()
        returns = MarketData.compute_returns(prices)
        try:
            fundamentals = self.data.fetch_fundamentals()
        except Exception:
            fundamentals = pd.DataFrame()

        signals = self.signal_gen.generate(prices, returns, fundamentals)
        return signals.iloc[-1].sort_values(ascending=False)
