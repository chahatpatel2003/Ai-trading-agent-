"""
Smart Money Concept Strategy
- Detects market structure (higher highs/lows for uptrend, lower highs/lows for downtrend)
- Identifies order blocks (last opposing candle before a strong move)
- Enters on return to order block level with structure confirmation
- Targets liquidity above/below swing points

WHY THIS MAY FAIL:
- Market structure breaks frequently with gold's volatility
- Order block concept is subjective; the implementation approximates it
- This is a popular retail concept, so it may be front-run by institutions
"""
import numpy as np
import pandas as pd

from data.indicators import atr, ema
from strategies.base import BaseStrategy


class SmartMoneyStrategy(BaseStrategy):
    def __init__(self, cfg: dict):
        super().__init__("smart_money", cfg)
        smcfg = cfg.get("strategies", {}).get("smart_money", {})
        self.swing_lookback = smcfg.get("swing_lookback", 10)
        self.atr_stop_mult = smcfg.get("atr_stop_multiple", 2.0)

    def _find_swing_points(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Find swing highs and swing lows."""
        lb = self.swing_lookback
        swing_high = pd.Series(np.nan, index=df.index)
        swing_low = pd.Series(np.nan, index=df.index)

        high = df["high"].values
        low = df["low"].values

        for i in range(lb, len(df) - lb):
            # Swing high: highest in window
            if high[i] == max(high[i - lb:i + lb + 1]):
                swing_high.iloc[i] = high[i]
            # Swing low: lowest in window
            if low[i] == min(low[i - lb:i + lb + 1]):
                swing_low.iloc[i] = low[i]

        return swing_high, swing_low

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        swing_high, swing_low = self._find_swing_points(df)
        atr_val = atr(df, 14)
        trend_ema = ema(df["close"], 50)

        # Forward-fill swing points to create support/resistance levels
        last_swing_high = swing_high.ffill().shift(1)
        last_swing_low = swing_low.ffill().shift(1)

        # Market structure: compare consecutive swing points
        prev_swing_high = swing_high.ffill().shift(self.swing_lookback + 1)
        prev_swing_low = swing_low.ffill().shift(self.swing_lookback + 1)

        # Bullish structure: higher highs and higher lows
        bull_structure = (last_swing_high > prev_swing_high) & (last_swing_low > prev_swing_low)
        # Bearish structure: lower highs and lower lows
        bear_structure = (last_swing_high < prev_swing_high) & (last_swing_low < prev_swing_low)

        atr_p = atr_val.shift(1)
        prev_close = df["close"].shift(1)
        trend_p = trend_ema.shift(1)

        # Order block zone: return to last swing low area (for longs)
        # Price pulled back to near the swing low
        near_support = (prev_close - last_swing_low).abs() < atr_p * 1.0
        near_resistance = (prev_close - last_swing_high).abs() < atr_p * 1.0

        # Long: bullish structure + price near swing low support + above trend EMA
        long_cond = bull_structure & near_support & (prev_close > trend_p)
        # Short: bearish structure + price near swing high resistance + below trend EMA
        short_cond = bear_structure & near_resistance & (prev_close < trend_p)

        df["signal"] = 0
        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1

        # Stop below the swing low (long) or above swing high (short)
        df["stop_price"] = np.where(
            df["signal"] == 1,
            last_swing_low - atr_p * 0.5,
            np.where(df["signal"] == -1, last_swing_high + atr_p * 0.5, np.nan),
        )

        # Confidence: structure strength
        df["confidence"] = 0.5
        df.loc[df["signal"] != 0, "confidence"] = 0.6

        return df
