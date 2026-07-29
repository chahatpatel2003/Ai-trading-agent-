"""
Stochastic + RSI Combo Strategy
- Uses Stochastic %K/%D crossover for timing
- RSI as trend/momentum filter
- Only enters at Stochastic extremes confirmed by RSI
- Designed for mean-reversion in ranging markets, trend-following in trending ones

WHY THIS MAY FAIL:
- Double oscillator = double noise in choppy conditions
- Stochastic stays pinned at 0 or 100 in strong trends
- Combining two lagging indicators doesn't create a leading one
"""
import numpy as np
import pandas as pd

from data.indicators import rsi, stochastic, atr, adx
from strategies.base import BaseStrategy


class StochasticRSIStrategy(BaseStrategy):
    def __init__(self, cfg: dict):
        super().__init__("stochastic_rsi", cfg)
        scfg = cfg.get("strategies", {}).get("stochastic_rsi", {})
        self.stoch_k = scfg.get("stoch_k", 14)
        self.stoch_d = scfg.get("stoch_d", 3)
        self.rsi_period = scfg.get("rsi_period", 14)
        self.atr_stop_mult = scfg.get("atr_stop_multiple", 2.0)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        k, d = stochastic(df, self.stoch_k, self.stoch_d)
        rsi_vals = rsi(df["close"], self.rsi_period)
        adx_vals = adx(df, 14)
        atr_val = atr(df, 14)

        k_p = k.shift(1)
        k_p2 = k.shift(2)
        d_p = d.shift(1)
        rsi_p = rsi_vals.shift(1)
        adx_p = adx_vals.shift(1)
        atr_p = atr_val.shift(1)
        prev_close = df["close"].shift(1)

        # Stochastic cross up from oversold zone + RSI confirms
        long_cond = (
            (k_p > d_p)  # %K crossed above %D
            & (k_p2 <= d.shift(2))  # was below last bar
            & (k_p < 30)  # in oversold zone
            & (rsi_p < 45)  # RSI also not overbought
            & (rsi_p > 20)  # but not extreme (might be in freefall)
        )

        # Stochastic cross down from overbought + RSI confirms
        short_cond = (
            (k_p < d_p)
            & (k_p2 >= d.shift(2))
            & (k_p > 70)
            & (rsi_p > 55)
            & (rsi_p < 80)
        )

        df["signal"] = 0
        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1

        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_p * self.atr_stop_mult,
            np.where(df["signal"] == -1, prev_close + atr_p * self.atr_stop_mult, np.nan),
        )

        # Confidence: how deep in the zone + ADX
        stoch_depth = np.where(
            df["signal"] == 1,
            (30 - k_p) / 30,
            np.where(df["signal"] == -1, (k_p - 70) / 30, 0),
        )
        df["confidence"] = pd.Series(stoch_depth, index=df.index).clip(0, 1) * 0.7 + 0.15

        return df
