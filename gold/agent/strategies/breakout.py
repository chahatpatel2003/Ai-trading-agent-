"""
Session Breakout Strategy
- Measures the Asian session range
- Enters on a confirmed breakout during London or New York session
- Requires N confirmation bars beyond the range

WHY THIS MAY FAIL:
- False breakouts are extremely common in gold
- Asian range can be too tight (whipsaw) or too wide (no breakout)
- News-driven spikes look like breakouts but reverse instantly
- This strategy has low trade frequency, limiting statistical power
"""
import numpy as np
import pandas as pd

from data.indicators import atr
from strategies.base import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    def __init__(self, cfg: dict):
        super().__init__("breakout", cfg)
        scfg = cfg["strategies"]["breakout"]
        self.asian_start = int(scfg["asian_session_start"].split(":")[0])
        self.asian_end = int(scfg["asian_session_end"].split(":")[0])
        self.london_open = int(scfg["london_open"].split(":")[0])
        self.ny_open = int(scfg["ny_open"].split(":")[0])
        self.buffer_pct = scfg["breakout_buffer_pct"]
        self.confirm_bars = scfg["confirmation_bars"]
        self.atr_period = scfg["atr_period"]
        self.stop_atr_mult = scfg["stop_atr_multiple"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Compute ATR
        df["atr"] = atr(df, self.atr_period)

        # Identify Asian session bars and compute session range
        df["hour"] = df.index.hour
        df["date_key"] = df.index.date

        # Calculate Asian session high/low per day
        asian_mask = (df["hour"] >= self.asian_start) & (df["hour"] < self.asian_end)
        asian_data = df[asian_mask].groupby("date_key").agg(
            asian_high=("high", "max"),
            asian_low=("low", "min"),
        )

        df = df.merge(asian_data, left_on="date_key", right_index=True, how="left")

        # Forward-fill so London/NY bars know the Asian range
        # but only AFTER the Asian session ends
        post_asian = df["hour"] >= self.asian_end
        df.loc[~post_asian, "asian_high"] = np.nan
        df.loc[~post_asian, "asian_low"] = np.nan
        df["asian_high"] = df["asian_high"].ffill()
        df["asian_low"] = df["asian_low"].ffill()

        # Breakout levels with buffer
        df["break_high"] = df["asian_high"] * (1 + self.buffer_pct)
        df["break_low"] = df["asian_low"] * (1 - self.buffer_pct)

        # Use previous bar's close to check breakout (no lookahead)
        prev_close = df["close"].shift(1)
        atr_val = df["atr"].shift(1)

        # Confirmation: price must have been above/below for N bars
        above_count = (prev_close > df["break_high"].shift(1)).rolling(self.confirm_bars).sum()
        below_count = (prev_close < df["break_low"].shift(1)).rolling(self.confirm_bars).sum()

        # Only trade during London or NY sessions
        tradeable = (df["hour"] >= self.london_open) & (df["hour"] < 22)

        # Long breakout
        long_cond = (above_count >= self.confirm_bars) & tradeable & df["break_high"].notna()
        # Short breakout
        short_cond = (below_count >= self.confirm_bars) & tradeable & df["break_low"].notna()

        df["signal"] = 0
        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1

        # Stops: ATR-based from the breakout level
        df["stop_price"] = np.where(
            df["signal"] == 1,
            df["asian_high"] - atr_val * self.stop_atr_mult,
            np.where(
                df["signal"] == -1,
                df["asian_low"] + atr_val * self.stop_atr_mult,
                np.nan,
            ),
        )

        # Confidence based on breakout strength relative to range
        asian_range = (df["asian_high"] - df["asian_low"]).replace(0, np.nan)
        breakout_strength = np.where(
            df["signal"] == 1,
            (prev_close - df["break_high"]) / asian_range,
            np.where(
                df["signal"] == -1,
                (df["break_low"] - prev_close) / asian_range,
                0,
            ),
        )
        df["confidence"] = pd.Series(breakout_strength, index=df.index).clip(0, 1)

        # Cleanup
        df.drop(
            columns=["hour", "date_key", "asian_high", "asian_low",
                      "break_high", "break_low", "atr"],
            inplace=True,
        )

        return df
