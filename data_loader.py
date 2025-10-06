# data_loader.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd


@dataclass
class CSVDateTimeLoader:
    """
    Loads CSV with separate Date and Timestamp columns.
    Date format: YYYYMMDD
    Time format: HH:MM:SS
    """
    path: str
    date_col: str = "Date"
    time_col: str = "Timestamp"
    close_col: str = "Close"
    tz: Optional[str] = None
    _cache: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def _load_dataframe(self) -> pd.DataFrame:
        if self._cache is not None:
            return self._cache.copy()

        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found at {p}")

        df = pd.read_csv(p, dtype={self.date_col: str, self.time_col: str}, keep_default_na=False)

        required = {self.date_col, self.time_col, self.close_col}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV missing required columns. Found: {list(df.columns)}")

        dt = pd.to_datetime(
            df[self.date_col].str.strip() + " " + df[self.time_col].str.strip(),
            format="%Y%m%d %H:%M:%S",
            errors="coerce"
        )
        df = df.assign(datetime=dt)
        df = df.dropna(subset=["datetime"])
        df = df.set_index("datetime")

        if self.tz is not None:
            df.index = df.index.tz_localize(self.tz, nonexistent="shift_forward", ambiguous="NaT")

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        df = df.drop(columns=[self.date_col, self.time_col], errors="ignore")

        numeric_cols = [c for c in df.columns]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=[self.close_col])

        self._cache = df
        return df.copy()

    def load_dataframe(self) -> pd.DataFrame:
        return self._load_dataframe()

    def load_close(self) -> pd.Series:
        df = self._load_dataframe()
        close = df[self.close_col].copy()
        return close


@dataclass
class CSVSingleDTLoader:
    """
    Loads CSV with a single datetime column.
    """
    path: str
    dt_col: str = "datetime"
    close_col: str = "Close"
    tz: Optional[str] = None
    _cache: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def _load_dataframe(self) -> pd.DataFrame:
        if self._cache is not None:
            return self._cache.copy()

        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found at {p}")

        df = pd.read_csv(p)
        if self.dt_col not in df.columns or self.close_col not in df.columns:
            raise ValueError(f"CSV missing required columns. Found: {list(df.columns)}")

        df[self.dt_col] = pd.to_datetime(df[self.dt_col], errors="coerce")
        df = df.dropna(subset=[self.dt_col])
        df = df.set_index(self.dt_col)

        if self.tz is not None:
            df.index = df.index.tz_localize(self.tz, nonexistent="shift_forward", ambiguous="NaT")

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        numeric_cols = [c for c in df.columns]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=[self.close_col])

        self._cache = df
        return df.copy()

    def load_dataframe(self) -> pd.DataFrame:
        return self._load_dataframe()

    def load_close(self) -> pd.Series:
        df = self._load_dataframe()
        close = df[self.close_col].copy()
        return close
