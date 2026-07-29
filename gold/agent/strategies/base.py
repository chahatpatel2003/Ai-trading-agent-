"""
Base strategy interface.
All strategies must implement generate_signals() and return a standardized DataFrame.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Signal(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class TradeSignal:
    signal: Signal
    entry_price: float
    stop_price: float
    strategy_name: str
    confidence: float = 0.5
    metadata: dict = None


class BaseStrategy(ABC):
    """
    All strategies inherit from this base.
    generate_signals() must return a DataFrame with at minimum:
      - 'signal': +1 (long), -1 (short), 0 (flat)
      - 'stop_price': price level for stop loss
    No future data may be used. All indicators must use .shift(1) or equivalent
    to avoid lookahead bias when generating entry signals.
    """

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given OHLCV data, return the same DataFrame with added columns:
          - signal: int (-1, 0, +1)
          - stop_price: float
          - confidence: float (0-1)
        CRITICAL: Signals on bar i must only use data from bars <= i-1.
        Use .shift(1) on indicator values before comparing.
        """
        pass

    def __repr__(self):
        return f"<Strategy: {self.name}>"
