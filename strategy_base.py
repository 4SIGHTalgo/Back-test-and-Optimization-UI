# strategy_base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
import pandas as pd

SignalDict = Dict[str, Union[pd.Series, pd.DataFrame]]

class Strategy(ABC):
    """
    A strategy must return either:
      A) entries/exits as booleans with keys:
         'entries', 'exits', 'short_entries', 'short_exits'
         All indexed like the price series.
         Optional risk targets may be supplied via:
         'long_take_profit', 'long_stop_loss', 'short_take_profit', 'short_stop_loss'
         to request exit handling at those price levels.
      B) a fully built 'position' series in {-1, 0, +1} indexed like price.
         When providing a position series directly, you may optionally include
         'exit_reason', 'active_take_profit', and 'active_stop_loss' series for reporting.

    The engine will accept either form.
    """

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate(self, price: pd.Series, data: Optional[pd.DataFrame] = None) -> SignalDict:
        raise NotImplementedError("Strategy.generate must be implemented")
