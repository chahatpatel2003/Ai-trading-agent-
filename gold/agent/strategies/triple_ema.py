"""
Triple EMA Ribbon Strategy
- Uses 3 EMAs (8, 21, 55) as a trend ribbon
- Enters when all 3 are stacked in order (bull: 8>21>55, bear: 8<21<55)
- Waits for pullback to middle EMA before entry (not chasing)
- ATR-based stop below the slow EMA

WHY THIS MAY FAIL:
- Triple EMA alignment is slow to form and quick to break in choppy markets
- Pullback entries can fail if the pullback becomes a reversal
- In sideways markets, EMAs converge and generate many false stacks
"""
import numpy as np
import pandas as pd

from data.indicators import ema, atr, rsi
from strategies.base import BaseStrategy


class TripleEMAStrategy(BaseStrategy):
    def __init__(self, cfg: dict):
        super().__init__("triple_ema", cfg)
        tcfg = cfg.get("strategies", {}).get("triple_ema", {})
        self.fast = tcfg.get("fast_ema", 8)
        self.mid = tcfg.get("mid_ema", 21)
        self.slow = tcfg.get("slow_ema", 55)
        self.atr_stop_mult = tcfg.get("atr_stop_multiple", 2.0)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        fast = ema(df["close"], self.fast)
        mid = ema(df["close"], self.mid)
        slow = ema(df["close"], self.slow)
        atr_val = atr(df, 14)
        rsi_vals = rsi(df["close"], 14)

        # Shift to avoid lookahead
        f_p = fast.shift(1)
        m_p = mid.shift(1)
        s_p = slow.shift(1)
        atr_p = atr_val.shift(1)
        rsi_p = rsi_vals.shift(1)
        prev_close = df["close"].shift(1)
        prev_low = df["low"].shift(1)
        prev_high = df["high"].shift(1)

        # Bull ribbon: fast > mid > slow
        bull_ribbon = (f_p > m_p) & (m_p > s_p)
        # Bear ribbon: fast < mid < slow
        bear_ribbon = (f_p < m_p) & (m_p < s_p)

        # Pullback to mid EMA zone (price touched or crossed below mid EMA)
        bull_pullback = (prev_low <= m_p * 1.002) & (prev_close > m_p)
        bear_pullback = (prev_high >= m_p * 0.998) & (prev_close < m_p)

        # RSI filter: not extreme
        rsi_ok_long = (rsi_p > 35) & (rsi_p < 65)
        rsi_ok_short = (rsi_p > 35) & (rsi_p < 65)

        df["signal"] = 0
        df.loc[bull_ribbon & bull_pullback & rsi_ok_long, "signal"] = 1
        df.loc[bear_ribbon & bear_pullback & rsi_ok_short, "signal"] = -1

        df["stop_price"] = np.where(
            df["signal"] == 1,
            s_p - atr_p * 0.5,  # below slow EMA
            np.where(df["signal"] == -1, s_p + atr_p * 0.5, np.nan),
        )

        df["trailing_stop_distance"] = atr_p * 1.5

        # Confidence: EMA spacing relative to ATR (wider = stronger trend)
        spread = ((f_p - s_p).abs() / atr_p).clip(0, 3) / 3
        df["confidence"] = spread.fillna(0.3)

        return df
