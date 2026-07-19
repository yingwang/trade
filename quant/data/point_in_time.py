"""Validated interfaces for externally supplied point-in-time research data.

The project does not bundle proprietary historical constituents or delisting
returns.  When operators provide them, these loaders validate the schema and
fail loudly on incomplete coverage instead of silently labelling a static
today-universe backtest as bias-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PointInTimeUniverse:
    snapshots: pd.DataFrame

    @classmethod
    def from_csv(cls, path: str | Path) -> "PointInTimeUniverse":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Point-in-time universe file not found: {source}")
        frame = pd.read_csv(source)
        required = {"date", "symbol", "is_member"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Point-in-time universe is missing columns: {sorted(missing)}"
            )
        frame = frame[["date", "symbol", "is_member"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        if frame["symbol"].isna().any():
            raise ValueError("Point-in-time universe contains a missing symbol")
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        if (frame["symbol"] == "").any():
            raise ValueError("Point-in-time universe contains an empty symbol")
        membership_text = frame["is_member"].map(
            lambda value: str(value).strip().lower()
        )
        true_values = {"1", "1.0", "true", "yes", "y"}
        false_values = {"0", "0.0", "false", "no", "n"}
        invalid = ~membership_text.isin(true_values | false_values)
        if invalid.any():
            values = sorted(set(membership_text[invalid].tolist()))
            raise ValueError(
                "Point-in-time universe has invalid is_member values: "
                f"{values}"
            )
        frame["is_member"] = membership_text.isin(true_values)
        if frame[["date", "symbol"]].duplicated().any():
            raise ValueError("Point-in-time universe contains duplicate date/symbol rows")
        if not frame["is_member"].any():
            raise ValueError("Point-in-time universe contains no active members")
        return cls(frame.sort_values(["date", "symbol"]).reset_index(drop=True))

    @property
    def symbols(self) -> set[str]:
        return set(self.snapshots["symbol"])

    def members_as_of(self, date) -> set[str]:
        """Members from the latest complete snapshot available by ``date``."""
        as_of = pd.Timestamp(date).tz_localize(None).normalize()
        eligible = self.snapshots[self.snapshots["date"] <= as_of]
        if eligible.empty:
            raise ValueError(f"No universe snapshot available on or before {as_of.date()}")
        snapshot_date = eligible["date"].max()
        snapshot = eligible[eligible["date"] == snapshot_date]
        return set(snapshot.loc[snapshot["is_member"], "symbol"])

    def eligibility_mask(
        self,
        dates: pd.DatetimeIndex,
        symbols: list[str] | pd.Index,
    ) -> pd.DataFrame:
        """Boolean membership mask using the latest snapshot at every date.

        The mask is consumed before cross-sectional factor ranks and ML labels
        are built. Filtering only the final portfolio is insufficient: future
        constituents would still influence historical z-scores and train the
        ranker before they entered the research universe.
        """

        index = pd.DatetimeIndex(dates)
        normalized = index.tz_localize(None).normalize()
        columns = [str(symbol).upper() for symbol in symbols]
        mask = pd.DataFrame(False, index=index, columns=columns, dtype=bool)

        snapshot_dates = pd.DatetimeIndex(
            self.snapshots["date"].drop_duplicates().sort_values()
        )
        grouped = {
            pd.Timestamp(snapshot_date): set(
                self.snapshots.loc[
                    (self.snapshots["date"] == snapshot_date)
                    & self.snapshots["is_member"],
                    "symbol",
                ]
            )
            for snapshot_date in snapshot_dates
        }

        for row, as_of in enumerate(normalized):
            position = int(snapshot_dates.searchsorted(as_of, side="right")) - 1
            if position < 0:
                continue
            members = grouped[pd.Timestamp(snapshot_dates[position])]
            if members:
                mask.iloc[row] = [symbol in members for symbol in columns]
        return mask


@dataclass(frozen=True)
class DelistingReturns:
    events: pd.DataFrame

    @classmethod
    def from_csv(cls, path: str | Path) -> "DelistingReturns":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Delisting-return file not found: {source}")
        frame = pd.read_csv(source)
        required = {"date", "symbol", "delisting_return"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Delisting returns are missing columns: {sorted(missing)}")
        frame = frame[["date", "symbol", "delisting_return"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        if frame["symbol"].isna().any():
            raise ValueError("Delisting returns contain a missing symbol")
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        if (frame["symbol"] == "").any():
            raise ValueError("Delisting returns contain an empty symbol")
        frame["delisting_return"] = pd.to_numeric(
            frame["delisting_return"], errors="raise"
        )
        if (~frame["delisting_return"].between(-1.0, 10.0)).any():
            raise ValueError("Delisting returns must be decimal returns in [-1, 10]")
        if frame[["date", "symbol"]].duplicated().any():
            raise ValueError("Delisting returns contain duplicate date/symbol rows")
        return cls(frame.sort_values(["date", "symbol"]).reset_index(drop=True))


def load_point_in_time_bundle(config: dict):
    """Load optional PIT files and return ``(universe, delisting_returns)``.

    Supplying a historical universe without delisting outcomes is rejected by
    default because that combination still has a material upward bias.
    """
    data_cfg = config.get("data", {})
    universe_path = data_cfg.get("point_in_time_universe_file")
    delisting_path = data_cfg.get("delisting_returns_file")
    if not universe_path and not delisting_path:
        return None, None
    if not universe_path:
        raise ValueError("delisting_returns_file requires point_in_time_universe_file")
    universe = PointInTimeUniverse.from_csv(universe_path)
    if not delisting_path:
        raise ValueError(
            "A point-in-time universe was configured without delisting returns; "
            "provide data.delisting_returns_file or remove the PIT claim"
        )
    return universe, DelistingReturns.from_csv(delisting_path)
