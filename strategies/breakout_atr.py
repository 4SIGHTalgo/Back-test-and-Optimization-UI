# strategies/breakout_atr.py
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from .strategy_base import Strategy


class ATRBreakout(Strategy):
    """Breakout strategy using a rolling ATR envelope for entries and risk levels."""

    DEFAULTS = {
        "period": 14,
        "atr_multiplier": 1.0,
    }

    def __init__(self, **params):
        merged = {**self.DEFAULTS, **params}
        super().__init__(**merged)

    def _compute_atr(self, close: pd.Series, length: int) -> pd.Series:
        tr = close.diff().abs()
        return tr.rolling(length, min_periods=length).mean()

    def generate(self, price: pd.Series, data: Optional[pd.DataFrame] = None):
        params = {**self.DEFAULTS, **self.params}
        period = int(params.get("period", self.DEFAULTS["period"]))
        multiplier = float(params.get("atr_multiplier", self.DEFAULTS["atr_multiplier"]))

        close = price.astype(float)
        atr_raw = self._compute_atr(close, period)
        atr = atr_raw.shift(1)

        upper_band = close.shift(1) + atr * multiplier
        lower_band = close.shift(1) - atr * multiplier

        long_entry = (close > upper_band).fillna(False)
        short_entry = (close < lower_band).fillna(False)

        long_tp = pd.Series(np.nan, index=close.index, dtype=float)
        long_sl = pd.Series(np.nan, index=close.index, dtype=float)
        short_tp = pd.Series(np.nan, index=close.index, dtype=float)
        short_sl = pd.Series(np.nan, index=close.index, dtype=float)

        valid = atr.notna()
        long_mask = long_entry & valid
        short_mask = short_entry & valid

        long_tp.loc[long_mask] = close.loc[long_mask] + atr.loc[long_mask]
        long_sl.loc[long_mask] = close.loc[long_mask] - atr.loc[long_mask]

        short_tp.loc[short_mask] = close.loc[short_mask] - atr.loc[short_mask]
        short_sl.loc[short_mask] = close.loc[short_mask] + atr.loc[short_mask]

        return {
            "entries": long_entry,
            "exits": pd.Series(False, index=close.index, dtype=bool),
            "short_entries": short_entry,
            "short_exits": pd.Series(False, index=close.index, dtype=bool),
            "long_take_profit": long_tp,
            "long_stop_loss": long_sl,
            "short_take_profit": short_tp,
            "short_stop_loss": short_sl,
        }
