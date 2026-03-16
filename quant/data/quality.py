"""Data quality validation and point-in-time data management.

Provides:
  - DataQualityChecker: validates price/return data for common issues
  - PointInTimeDataManager: wraps fundamental data to prevent look-ahead bias
"""

import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# Data Quality Checker
# ======================================================================

class DataQualityChecker:
    """Validates market data for common quality issues.

    Checks performed:
      1. Missing value rate per symbol
      2. Extreme daily price moves (potential bad ticks or corporate actions)
      3. Data continuity (missing trading days)
      4. Cross-symbol date consistency
      5. Stale price detection (zero-variance runs)
    """

    def __init__(
        self,
        max_missing_rate: float = 0.05,
        extreme_return_threshold: float = 0.50,
        max_gap_days: int = 5,
        stale_window: int = 5,
    ):
        self.max_missing_rate = max_missing_rate
        self.extreme_return_threshold = extreme_return_threshold
        self.max_gap_days = max_gap_days
        self.stale_window = stale_window

    def run_all_checks(self, prices: pd.DataFrame) -> dict:
        """Run all quality checks and return a summary report dict.

        Parameters
        ----------
        prices : DataFrame
            Daily adjusted close prices, index=dates, columns=symbols.

        Returns
        -------
        dict with keys: 'passed', 'warnings', 'failures', 'details'
        """
        report = {
            "passed": True,
            "warnings": [],
            "failures": [],
            "details": {},
        }

        # 1. Missing value check
        missing = self.check_missing_values(prices)
        report["details"]["missing_values"] = missing
        for sym, rate in missing.items():
            if rate > self.max_missing_rate:
                msg = f"{sym}: missing rate {rate:.1%} exceeds threshold {self.max_missing_rate:.1%}"
                report["failures"].append(msg)
                report["passed"] = False
            elif rate > self.max_missing_rate / 2:
                report["warnings"].append(
                    f"{sym}: missing rate {rate:.1%} approaching threshold"
                )

        # 2. Extreme return check
        extremes = self.check_extreme_returns(prices)
        report["details"]["extreme_returns"] = extremes
        if not extremes.empty:
            report["warnings"].append(
                f"Found {len(extremes)} extreme daily returns (>{self.extreme_return_threshold:.0%})"
            )

        # 3. Data continuity check
        gaps = self.check_data_continuity(prices)
        report["details"]["date_gaps"] = gaps
        if gaps:
            report["warnings"].append(
                f"Found {len(gaps)} date gaps exceeding {self.max_gap_days} calendar days"
            )

        # 4. Cross-symbol consistency
        consistency = self.check_cross_symbol_consistency(prices)
        report["details"]["cross_symbol_consistency"] = consistency

        # 5. Stale prices
        stale = self.check_stale_prices(prices)
        report["details"]["stale_prices"] = stale
        if stale:
            report["warnings"].append(
                f"Found {len(stale)} symbols with stale price runs (>= {self.stale_window} identical closes)"
            )

        n_warn = len(report["warnings"])
        n_fail = len(report["failures"])
        logger.info(
            "Data quality check complete: %s | %d warnings, %d failures",
            "PASSED" if report["passed"] else "FAILED",
            n_warn,
            n_fail,
        )
        return report

    def check_missing_values(self, prices: pd.DataFrame) -> dict[str, float]:
        """Return missing-value rate per symbol."""
        total = len(prices)
        if total == 0:
            return {}
        return (prices.isna().sum() / total).to_dict()

    def check_extreme_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Detect daily returns exceeding the extreme threshold.

        Returns DataFrame with columns: date, symbol, return, price_before, price_after.
        """
        returns = prices.pct_change()
        mask = returns.abs() > self.extreme_return_threshold

        records = []
        for sym in prices.columns:
            extreme_dates = returns.index[mask[sym]]
            for dt in extreme_dates:
                idx = prices.index.get_loc(dt)
                records.append({
                    "date": dt,
                    "symbol": sym,
                    "return": returns.loc[dt, sym],
                    "price_before": prices.iloc[idx - 1][sym] if idx > 0 else np.nan,
                    "price_after": prices.loc[dt, sym],
                })

        return pd.DataFrame(records)

    def check_data_continuity(self, prices: pd.DataFrame) -> list[dict]:
        """Detect gaps in the date index exceeding max_gap_days calendar days."""
        gaps = []
        dates = prices.index.sort_values()
        for i in range(1, len(dates)):
            delta = (dates[i] - dates[i - 1]).days
            if delta > self.max_gap_days:
                gaps.append({
                    "from": dates[i - 1],
                    "to": dates[i],
                    "gap_calendar_days": delta,
                })
        return gaps

    def check_cross_symbol_consistency(self, prices: pd.DataFrame) -> dict:
        """Check that all symbols have data spanning similar date ranges."""
        first_valid = prices.apply(lambda s: s.first_valid_index())
        last_valid = prices.apply(lambda s: s.last_valid_index())
        return {
            "earliest_start": first_valid.min(),
            "latest_start": first_valid.max(),
            "earliest_end": last_valid.min(),
            "latest_end": last_valid.max(),
            "symbols_with_late_start": first_valid[
                first_valid > first_valid.min() + pd.Timedelta(days=30)
            ].to_dict(),
        }

    def check_stale_prices(self, prices: pd.DataFrame) -> list[dict]:
        """Detect runs of identical consecutive closing prices (stale data)."""
        stale = []
        for sym in prices.columns:
            series = prices[sym].dropna()
            if len(series) < self.stale_window:
                continue
            # Rolling check: if all values in window are equal to the first
            diff = series.diff().abs()
            run_length = 0
            run_start = None
            for i in range(1, len(diff)):
                if diff.iloc[i] == 0:
                    if run_length == 0:
                        run_start = series.index[i - 1]
                    run_length += 1
                else:
                    if run_length >= self.stale_window:
                        stale.append({
                            "symbol": sym,
                            "start": run_start,
                            "end": series.index[i - 1],
                            "run_length": run_length,
                            "price": series.iloc[i - 1],
                        })
                    run_length = 0
                    run_start = None
            # Check trailing run
            if run_length >= self.stale_window and run_start is not None:
                stale.append({
                    "symbol": sym,
                    "start": run_start,
                    "end": series.index[-1],
                    "run_length": run_length,
                    "price": series.iloc[-1],
                })
        return stale

    @staticmethod
    def format_report(report: dict) -> str:
        """Format the quality report as a human-readable string."""
        lines = [
            "=" * 60,
            "DATA QUALITY REPORT",
            "=" * 60,
            f"Status: {'PASSED' if report['passed'] else 'FAILED'}",
            "",
        ]
        if report["failures"]:
            lines.append("FAILURES:")
            for f in report["failures"]:
                lines.append(f"  [FAIL] {f}")
            lines.append("")
        if report["warnings"]:
            lines.append("WARNINGS:")
            for w in report["warnings"]:
                lines.append(f"  [WARN] {w}")
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# ======================================================================
# Point-in-Time Data Manager
# ======================================================================

class PointInTimeDataManager:
    """Wrapper to mitigate look-ahead bias in fundamental data.

    PROBLEM:
        yfinance .info returns a CURRENT snapshot of fundamental ratios
        (trailingPE, ROE, etc.). Using these in a backtest that spans
        multiple years means every historical date sees TODAY's fundamentals,
        which is severe look-ahead bias.

    MITIGATION (since yfinance does not provide historical point-in-time data):
        1. In backtest mode, emit explicit warnings whenever fundamental data
           is accessed.
        2. Apply a configurable reporting lag (default 90 days) so that
           fundamental data is only available after the assumed reporting
           delay.  This does NOT fix the look-ahead bias — it only adds a
           buffer that would exist in practice.
        3. Provide a mechanism to supply external point-in-time data
           (e.g., from Sharadar, Compustat) that can replace the yfinance
           snapshot.

    RECOMMENDED LONG-TERM FIX:
        Replace yfinance .info with a true point-in-time fundamental database
        (e.g., Sharadar via Nasdaq Data Link, Compustat via WRDS, or
        SimFin bulk downloads).  Until then, the value and quality factor
        weights should be set to 0 in any backtest used for performance
        evaluation.
    """

    def __init__(
        self,
        fundamentals: pd.DataFrame,
        fetch_date: datetime = None,
        reporting_lag_days: int = 90,
        is_backtest: bool = False,
    ):
        """
        Parameters
        ----------
        fundamentals : DataFrame
            Fundamental data indexed by symbol (from MarketData.fetch_fundamentals).
        fetch_date : datetime
            The date when this snapshot was fetched. Defaults to today.
        reporting_lag_days : int
            Assumed delay between fiscal period end and public availability.
            In backtest mode, fundamental data is only "available" after this lag.
        is_backtest : bool
            If True, emit warnings about look-ahead bias.
        """
        self.fundamentals = fundamentals
        self.fetch_date = fetch_date or datetime.today()
        self.reporting_lag_days = reporting_lag_days
        self.is_backtest = is_backtest
        self._warned = False

    def get_fundamentals(self, as_of_date: datetime = None) -> pd.DataFrame:
        """Return fundamental data, with look-ahead bias safeguards.

        Parameters
        ----------
        as_of_date : datetime, optional
            The simulated "current" date in a backtest. If provided and
            is_backtest=True, a warning is emitted and the data is only
            returned if the as_of_date is within reporting_lag_days of
            the fetch_date.

        Returns
        -------
        DataFrame of fundamentals, or empty DataFrame if data is not yet
        "available" at as_of_date.
        """
        if self.is_backtest:
            self._emit_lookahead_warning()

            if as_of_date is not None:
                # The earliest date this snapshot could realistically be known
                available_after = self.fetch_date - timedelta(
                    days=self.reporting_lag_days
                )
                if as_of_date < available_after:
                    logger.debug(
                        "Fundamental data not available at %s (available after %s)",
                        as_of_date,
                        available_after,
                    )
                    return pd.DataFrame()

        return self.fundamentals

    def _emit_lookahead_warning(self):
        if not self._warned:
            msg = (
                "LOOK-AHEAD BIAS WARNING: Fundamental data is from a single "
                f"yfinance snapshot fetched on {self.fetch_date.strftime('%Y-%m-%d')}. "
                "This data reflects CURRENT fundamentals and is being applied "
                "to historical dates in a backtest. Value and quality factor "
                "scores are NOT point-in-time accurate. For reliable backtest "
                "results, either (a) set value/quality weights to 0, or "
                "(b) supply a point-in-time fundamental database."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            logger.warning(msg)
            self._warned = True

    def get_available_date(self) -> datetime:
        """Return the earliest date this data should be considered available."""
        return self.fetch_date - timedelta(days=self.reporting_lag_days)


# ======================================================================
# Universe warnings
# ======================================================================

def warn_survivorship_bias(symbols: list[str], start_date: str) -> None:
    """Emit a warning about survivorship bias in a static universe.

    Should be called at the start of any backtest that uses a fixed
    stock list.

    Parameters
    ----------
    symbols : list[str]
        The static universe of ticker symbols.
    start_date : str
        The backtest start date.
    """
    msg = (
        f"SURVIVORSHIP BIAS WARNING: The universe contains {len(symbols)} "
        f"currently-active stocks applied to a backtest starting {start_date}. "
        "Stocks that were delisted, acquired, or went bankrupt during this "
        "period are excluded, which biases performance upward. Notable "
        "examples of stocks missing from a backtest starting in 2016: "
        "GE (major decline and restructuring), XLNX (acquired by AMD), "
        "ATVI (acquired by MSFT), TWTR (acquired and delisted). "
        "For unbiased results, use a point-in-time universe from a "
        "survivorship-bias-free database."
    )
    warnings.warn(msg, UserWarning, stacklevel=2)
    logger.warning(msg)
