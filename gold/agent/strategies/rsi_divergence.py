"""
RSI Divergence Strategy
- Detects bullish divergence: price makes lower low but RSI makes higher low
- Detects bearish divergence: price makes higher high but RSI makes lower high
- Uses swing point detection over a lookback window
- ATR-based stops

WHY THIS MAY FAIL:
- Divergence can persist for many bars before price reverses (or never does)
- Swing detection is noisy on intraday data
- False divergences outnumber real ones significantly
"""
import numpy as np
import pandas as pd

from data.indicators import rsi, atr
from strategies.base import BaseStrategy


class RSIDivergenceStrategy(BaseStrategy):
    def __init__(self, cfg: dict):
        super().__init__("rsi_divergence", cfg)
        dcfg = cfg.get("strategies", {}).get("rsi_divergence", {})
        self.rsi_period = dcfg.get("rsi_period", 14)
        self.lookback = dcfg.get("lookback", 20)
        self.atr_stop_mult = dcfg.get("atr_stop_multiple", 2.5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        rsi_vals = rsi(df["close"], self.rsi_period)
        atr_val = atr(df, 14)

        # Shift to avoid lookahead
        rsi_prev = rsi_vals.shift(1)
        atr_prev = atr_val.shift(1)
        prev_close = df["close"].shift(1)

        # Find rolling lows and highs for price and RSI
        price_low = df["low"].shift(1).rolling(self.lookback, min_periods=self.lookback).min()
        price_high = df["high"].shift(1).rolling(self.lookback, min_periods=self.lookback).max()
        rsi_low = rsi_prev.rolling(self.lookback, min_periods=self.lookback).min()
        rsi_high = rsi_prev.rolling(self.lookback, min_periods=self.lookback).max()

        # Previous period lows/highs (shifted further back)
        price_low_prev = df["low"].shift(1 + self.lookback).rolling(self.lookback, min_periods=self.lookback).min()
        price_high_prev = df["high"].shift(1 + self.lookback).rolling(self.lookback, min_periods=self.lookback).max()
        rsi_low_prev = rsi_vals.shift(1 + self.lookback).rolling(self.lookback, min_periods=self.lookback).min()
        rsi_high_prev = rsi_vals.shift(1 + self.lookback).rolling(self.lookback, min_periods=self.lookback).max()

        # Bullish divergence: price lower low, RSI higher low
        bull_div = (price_low < price_low_prev) & (rsi_low > rsi_low_prev) & (rsi_prev < 40)

        # Bearish divergence: price higher high, RSI lower high
        bear_div = (price_high > price_high_prev) & (rsi_high < rsi_high_prev) & (rsi_prev > 60)

        df["signal"] = 0
        df.loc[bull_div, "signal"] = 1
        df.loc[bear_div, "signal"] = -1

        df["stop_price"] = np.where(
            df["signal"] == 1,
            price_low - atr_prev * 0.5,
            np.where(df["signal"] == -1, price_high + atr_prev * 0.5, np.nan),
        )

        # Confidence based on RSI extremity
        df["confidence"] = np.where(
            df["signal"] == 1,
            ((40 - rsi_prev) / 30).clip(0, 1),
            np.where(df["signal"] == -1, ((rsi_prev - 60) / 30).clip(0, 1), 0),
        )

        return df
