# strategies/ma_crossover.py
from __future__ import annotations
import pandas as pd
from typing import Optional

from strategy_base import Strategy


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(window=n, min_periods=n).mean()


def _crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


class MACrossover(Strategy):
    """Simple moving-average crossover with reversal."""

    DEFAULTS = {
        "short_len": 4,
        "long_len": 16,
    }

    def __init__(self, **params):
        merged = {**self.DEFAULTS, **params}
        super().__init__(**merged)

    def _get_params(self) -> tuple[int, int]:
        params = {**self.DEFAULTS, **self.params}
        short_len = int(params.get("short_len", self.DEFAULTS["short_len"]))
        long_len = int(params.get("long_len", self.DEFAULTS["long_len"]))
        if short_len <= 0 or long_len <= 0 or short_len >= long_len:
            raise ValueError("short_len must be positive and less than long_len")
        return short_len, long_len

    def generate(self, price: pd.Series, data: Optional[pd.DataFrame] = None):
        short_len, long_len = self._get_params()

        s = _sma(price, short_len)
        l = _sma(price, long_len)

        long_entry = _crossover(s, l).fillna(False)
        long_exit = _crossunder(s, l).fillna(False)
        short_entry = long_exit
        short_exit = long_entry

        return {
            "entries": long_entry,
            "exits": long_exit,
            "short_entries": short_entry,
            "short_exits": short_exit,
        }
