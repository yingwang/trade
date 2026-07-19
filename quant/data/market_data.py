"""Market data acquisition using yfinance."""

import logging
import time
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
MARKET_TZ = ZoneInfo("America/New_York")


class MarketData:
    """Fetches and caches daily OHLCV + fundamental data for US equities."""

    def __init__(self, config: dict):
        self.symbols: list[str] = config["universe"]["symbols"]
        self.benchmark: str = config["universe"]["benchmark"]
        self.lookback_years: int = config["data"]["lookback_years"]
        self.frequency: str = config["data"]["frequency"]

        self._price_cache: dict[tuple[str, str, str, str], pd.DataFrame] = {}
        self._info_cache: dict[str, dict] = {}
        # Populated by fetch_prices from the exact same adjusted download.
        # Backtests use it to execute a close-generated signal at the next
        # session's open without performing an inconsistent second fetch.
        self.last_open_prices_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------

    @staticmethod
    def _exclusive_download_end(
        inclusive_end: str | None = None,
        *,
        now: datetime | None = None,
    ) -> str:
        """Translate the public inclusive end date to yfinance's exclusive end.

        With no explicit end, only completed US sessions are eligible: before
        16:15 ET today's still-forming daily candle is excluded; after that
        grace period it is included. Weekends/holidays naturally fall back to
        the vendor's latest available session.
        """
        if inclusive_end is not None:
            parsed = pd.Timestamp(inclusive_end)
            if pd.isna(parsed):
                raise ValueError("Market-data end date is invalid")
            return (parsed.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        market_now = now or datetime.now(MARKET_TZ)
        if market_now.tzinfo is None:
            market_now = market_now.replace(tzinfo=MARKET_TZ)
        else:
            market_now = market_now.astimezone(MARKET_TZ)
        after_close_grace = (market_now.hour, market_now.minute) >= (16, 15)
        completed_through = market_now.date()
        if not after_close_grace:
            completed_through -= timedelta(days=1)
        return (completed_through + timedelta(days=1)).isoformat()

    def fetch_prices(self, start: str = None, end: str = None) -> pd.DataFrame:
        """Return adjusted closes; an explicit ``end`` date is inclusive."""
        download_end = self._exclusive_download_end(end)
        if start is None:
            start = (
                datetime.now(MARKET_TZ) - timedelta(days=365 * self.lookback_years)
            ).strftime("%Y-%m-%d")

        all_symbols = list(dict.fromkeys(self.symbols + [self.benchmark]))
        logger.info(
            "Fetching price data for %d symbols from %s through %s",
            len(all_symbols),
            start,
            end or "latest completed session",
        )

        data = yf.download(
            all_symbols,
            start=start,
            end=download_end,
            interval=self.frequency,
            auto_adjust=True,
            progress=False,
        )

        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
            opens = data["Open"] if "Open" in data.columns.get_level_values(0) else None
        else:
            prices = data[["Close"]].rename(columns={"Close": all_symbols[0]})
            opens = (
                data[["Open"]].rename(columns={"Open": all_symbols[0]})
                if "Open" in data.columns
                else None
            )

        prices = prices.dropna(how="all")
        self.last_open_prices_ = (
            opens.reindex(index=prices.index, columns=prices.columns)
            if opens is not None
            else None
        )
        return prices

    def fetch_adv(self, symbols: list[str], window: int = 30) -> dict[str, float]:
        """Average daily share volume over the trailing `window` trading days.

        Used by pre-trade liquidity checks (max ADV fraction) and TWAP
        splitting. Returns an empty dict on failure so callers can treat
        missing ADV as "no liquidity data" rather than blocking trading.
        """
        if not symbols:
            return {}
        try:
            start = (
                datetime.now(MARKET_TZ) - timedelta(days=window * 2 + 10)
            ).strftime("%Y-%m-%d")
            data = yf.download(
                list(symbols),
                start=start,
                end=self._exclusive_download_end(),
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if isinstance(data.columns, pd.MultiIndex):
                volume = data["Volume"]
            else:
                volume = data[["Volume"]].rename(columns={"Volume": symbols[0]})
            adv = volume.tail(window).mean()
            return {sym: float(v) for sym, v in adv.items() if pd.notna(v) and v > 0}
        except Exception as e:
            logger.warning("Could not fetch ADV data: %s", e)
            return {}

    def fetch_ohlcv(self, symbol: str, start: str = None, end: str = None) -> pd.DataFrame:
        """Return OHLCV for one symbol; an explicit ``end`` is inclusive."""
        download_end = self._exclusive_download_end(end)
        if start is None:
            start = (
                datetime.now(MARKET_TZ) - timedelta(days=365 * self.lookback_years)
            ).strftime("%Y-%m-%d")

        cache_key = (symbol, start, download_end, self.frequency)
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start,
            end=download_end,
            interval=self.frequency,
            auto_adjust=True,
        )
        self._price_cache[cache_key] = df
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
        return prices.pct_change(fill_method=None).dropna(how="all")

    @staticmethod
    def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Log daily returns."""
        import numpy as np
        return np.log(prices / prices.shift(1)).dropna(how="all")
