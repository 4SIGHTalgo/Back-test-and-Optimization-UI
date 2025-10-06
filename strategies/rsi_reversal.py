# strategies/rsi_reversal.py
from __future__ import annotations
import pandas as pd
from typing import Optional

from .strategy_base import Strategy


class RSIReversal(Strategy):
    """Reversal RSI strategy identical to Pine."""

    DEFAULTS = {
        "rsi_len": 14,
        "oversold": 30.0,
        "overbought": 70.0,
    }

    def __init__(self, **params):
        merged = {**self.DEFAULTS, **params}
        super().__init__(**merged)

    @staticmethod
    def _rma(x: pd.Series, length: int) -> pd.Series:
        return x.ewm(alpha=1 / float(length), adjust=False, min_periods=length).mean()

    def _rsi_tv(self, close: pd.Series, length: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = self._rma(gain, length)
        avg_loss = self._rma(loss, length)
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def _crossover(a: pd.Series, b: float) -> pd.Series:
        return (a > b) & (a.shift(1) <= b)

    @staticmethod
    def _crossunder(a: pd.Series, b: float) -> pd.Series:
        return (a < b) & (a.shift(1) >= b)

    def generate(self, price: pd.Series, data: Optional[pd.DataFrame] = None):
        params = {**self.DEFAULTS, **self.params}
        length = int(params.get("rsi_len", self.DEFAULTS["rsi_len"]))
        oversold = float(params.get("oversold", self.DEFAULTS["oversold"]))
        overbought = float(params.get("overbought", self.DEFAULTS["overbought"]))

        rsi = self._rsi_tv(price, length)

        long_entry = self._crossover(rsi, oversold).fillna(False)
        short_entry = self._crossunder(rsi, overbought).fillna(False)

        long_exit = short_entry
        short_exit = long_entry

        return {
            "entries": long_entry,
            "exits": long_exit,
            "short_entries": short_entry,
            "short_exits": short_exit,
        }
