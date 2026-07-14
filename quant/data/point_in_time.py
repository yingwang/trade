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
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
        frame["is_member"] = frame["is_member"].map(
            lambda value: str(value).strip().lower()
            in {"1", "1.0", "true", "yes", "y"}
        )
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
        as_of = pd.Timestamp(date).normalize()
        eligible = self.snapshots[self.snapshots["date"] <= as_of]
        if eligible.empty:
            raise ValueError(f"No universe snapshot available on or before {as_of.date()}")
        snapshot_date = eligible["date"].max()
        snapshot = eligible[eligible["date"] == snapshot_date]
        return set(snapshot.loc[snapshot["is_member"], "symbol"])


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
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
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
