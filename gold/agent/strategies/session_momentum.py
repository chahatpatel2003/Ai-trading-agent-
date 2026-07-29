"""
Session Momentum Strategy
- Measures momentum during Asian session close
- Enters at London open if Asian session showed clear directional momentum
- Exits before NY close or on stop
- Combines session analysis with momentum indicators

WHY THIS MAY FAIL:
- Asian session can be quiet (no momentum to measure)
- London open can immediately reverse Asian direction
- NFP / FOMC days destroy any session pattern
"""
import numpy as np
import pandas as pd

from data.indicators import rsi, ema, atr
from strategies.base import BaseStrategy


class SessionMomentumStrategy(BaseStrategy):
    def __init__(self, cfg: dict):
        super().__init__("session_momentum", cfg)
        scfg = cfg.get("strategies", {}).get("session_momentum", {})
        self.asian_start = scfg.get("asian_start", 0)
        self.asian_end = scfg.get("asian_end", 8)
        self.london_start = scfg.get("london_start", 8)
        self.london_end = scfg.get("london_end", 16)
        self.min_move_pct = scfg.get("min_asian_move_pct", 0.001)
        self.atr_stop_mult = scfg.get("atr_stop_multiple", 1.5)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        hour = df.index.hour
        df["_date"] = df.index.date

        # Calculate Asian session move per day
        asian_mask = (hour >= self.asian_start) & (hour < self.asian_end)
        asian_data = df[asian_mask].groupby(df[asian_mask]["_date"]).agg(
            asian_open=("open", "first"),
            asian_close=("close", "last"),
            asian_high=("high", "max"),
            asian_low=("low", "min"),
        )
        asian_data["asian_move_pct"] = (asian_data["asian_close"] - asian_data["asian_open"]) / asian_data["asian_open"]

        df = df.merge(asian_data, left_on="_date", right_index=True, how="left")

        # RSI and EMA for confirmation
        rsi_vals = rsi(df["close"], 14).shift(1)
        fast_ema = ema(df["close"], 8).shift(1)
        slow_ema = ema(df["close"], 21).shift(1)
        atr_val = atr(df, 14).shift(1)
        prev_close = df["close"].shift(1)

        # Only signal at London open (first bar of London session)
        london_open_bar = (hour == self.london_start)

        # Asian session was bullish + EMA confirms
        asian_bull = (df["asian_move_pct"] > self.min_move_pct) & (fast_ema > slow_ema)
        asian_bear = (df["asian_move_pct"] < -self.min_move_pct) & (fast_ema < slow_ema)

        df["signal"] = 0
        df.loc[london_open_bar & asian_bull, "signal"] = 1
        df.loc[london_open_bar & asian_bear, "signal"] = -1

        df["stop_price"] = np.where(
            df["signal"] == 1,
            df["asian_low"] - atr_val * 0.5,
            np.where(df["signal"] == -1, df["asian_high"] + atr_val * 0.5, np.nan),
        )

        # Confidence from magnitude of Asian move
        move_strength = df["asian_move_pct"].abs() / 0.005
        df["confidence"] = move_strength.clip(0, 1).fillna(0.3)

        # Cleanup
        df.drop(columns=["_date", "asian_open", "asian_close", "asian_high", "asian_low", "asian_move_pct"],
                inplace=True, errors="ignore")

        return df
