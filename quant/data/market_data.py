"""Market data acquisition using yfinance."""

import logging
import time
import warnings
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketData:
    """Fetches and caches daily OHLCV + fundamental data for US equities."""

    def __init__(self, config: dict):
        self.symbols: list[str] = config["universe"]["symbols"]
        self.benchmark: str = config["universe"]["benchmark"]
        self.lookback_years: int = config["data"]["lookback_years"]
        self.frequency: str = config["data"]["frequency"]

        self._price_cache: dict[str, pd.DataFrame] = {}
        self._info_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------

    def fetch_prices(self, start: str = None, end: str = None) -> pd.DataFrame:
        """Return a DataFrame of adjusted close prices, columns = symbols."""
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")
        if start is None:
            start = (datetime.today() - timedelta(days=365 * self.lookback_years)).strftime("%Y-%m-%d")

        all_symbols = self.symbols + [self.benchmark]
        logger.info("Fetching price data for %d symbols from %s to %s", len(all_symbols), start, end)

        data = yf.download(
            all_symbols,
            start=start,
            end=end,
            interval=self.frequency,
            auto_adjust=True,
            progress=False,
        )

        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]].rename(columns={"Close": all_symbols[0]})

        prices = prices.dropna(how="all")
        return prices

    def fetch_ohlcv(self, symbol: str, start: str = None, end: str = None) -> pd.DataFrame:
        """Return OHLCV DataFrame for a single symbol."""
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")
        if start is None:
            start = (datetime.today() - timedelta(days=365 * self.lookback_years)).strftime("%Y-%m-%d")

        if symbol in self._price_cache:
            return self._price_cache[symbol]

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=self.frequency, auto_adjust=True)
        self._price_cache[symbol] = df
        return df

    # ------------------------------------------------------------------
    # Fundamental / info data
    # ------------------------------------------------------------------

    def fetch_info(self, symbol: str, max_retries: int = 2) -> dict:
        """Return yfinance .info dict for a symbol (cached, with retry)."""
        if symbol in self._info_cache:
            return self._info_cache[symbol]

        info = {}
        for attempt in range(max_retries + 1):
            try:
                info = yf.Ticker(symbol).info
                break
            except Exception:
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    logger.warning("Failed to fetch info for %s after %d attempts", symbol, max_retries + 1)

        self._info_cache[symbol] = info
        return info

    def fetch_fundamentals(self, batch_size: int = 50, batch_delay: float = 1.5,
                           is_backtest: bool = False) -> pd.DataFrame:
        """Return a DataFrame of key fundamental ratios for the universe.

        Fetches in batches with delays to avoid yfinance rate limits.

        WARNING: yfinance .info returns a CURRENT snapshot of fundamentals.
        This data is NOT point-in-time and will cause look-ahead bias if
        used in a historical backtest.  See quant.data.quality.PointInTimeDataManager
        for mitigation strategies.
        """
        if is_backtest:
            warnings.warn(
                "fetch_fundamentals() returns CURRENT snapshot data from yfinance. "
                "Using this in a backtest introduces look-ahead bias for value "
                "and quality factors. Consider setting their weights to 0 or "
                "supplying point-in-time data.",
                UserWarning,
                stacklevel=2,
            )
        rows = []
        fields = [
            "trailingPE", "forwardPE", "priceToBook", "pegRatio",
            "dividendYield", "profitMargins", "returnOnEquity",
            "debtToEquity", "earningsGrowth", "revenueGrowth",
            "marketCap", "sector",
        ]
        total = len(self.symbols)
        for i, sym in enumerate(self.symbols):
            info = self.fetch_info(sym)
            row = {"symbol": sym}
            for f in fields:
                row[f] = info.get(f)
            rows.append(row)

            # Progress logging and rate-limit delay between batches
            done = i + 1
            if done % batch_size == 0 or done == total:
                logger.info("Fundamentals progress: %d / %d symbols fetched", done, total)
                if done < total:
                    time.sleep(batch_delay)

        return pd.DataFrame(rows).set_index("symbol")

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    @staticmethod
    def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Simple daily returns."""
        return prices.pct_change().dropna(how="all")

    @staticmethod
    def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Log daily returns."""
        import numpy as np
        return np.log(prices / prices.shift(1)).dropna(how="all")
