"""
Volume-Price Confirmation Strategy
- Uses OBV (On-Balance Volume) trend to confirm price direction
- VWAP as dynamic support/resistance
- Volume spike detection for breakout confirmation
- Only enters when price AND volume agree

WHY THIS MAY FAIL:
- OANDA volume is tick volume, not real exchange volume
- OBV can trend while price is flat and vice versa
- VWAP resets daily, causing edge effects
"""
import numpy as np
import pandas as pd

from data.indicators import ema, obv, vwap, atr
from strategies.base import BaseStrategy


class VolumePriceStrategy(BaseStrategy):
    def __init__(self, cfg: dict):
        super().__init__("volume_price", cfg)
        vcfg = cfg.get("strategies", {}).get("volume_price", {})
        self.vol_spike_mult = vcfg.get("volume_spike_multiple", 1.5)
        self.atr_stop_mult = vcfg.get("atr_stop_multiple", 2.0)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # OBV and its EMA
        obv_vals = obv(df)
        obv_ema = ema(obv_vals, 20)

        # VWAP
        vwap_vals = vwap(df, 20)

        # Volume spike: current vol > N * average vol
        vol_avg = df["volume"].rolling(20, min_periods=20).mean()
        vol_spike = df["volume"] > vol_avg * self.vol_spike_mult

        # Price trend (EMA cross)
        fast_ema = ema(df["close"], 10)
        slow_ema = ema(df["close"], 30)

        atr_val = atr(df, 14)

        # Shift everything
        obv_p = obv_vals.shift(1)
        obv_ema_p = obv_ema.shift(1)
        vwap_p = vwap_vals.shift(1)
        vol_spike_p = vol_spike.shift(1)
        fast_p = fast_ema.shift(1)
        slow_p = slow_ema.shift(1)
        prev_close = df["close"].shift(1)
        atr_p = atr_val.shift(1)

        # Long: OBV rising + price above VWAP + volume spike + uptrend
        long_cond = (
            (obv_p > obv_ema_p)
            & (prev_close > vwap_p)
            & vol_spike_p
            & (fast_p > slow_p)
        )

        # Short: OBV falling + price below VWAP + volume spike + downtrend
        short_cond = (
            (obv_p < obv_ema_p)
            & (prev_close < vwap_p)
            & vol_spike_p
            & (fast_p < slow_p)
        )

        df["signal"] = 0
        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1

        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_p * self.atr_stop_mult,
            np.where(df["signal"] == -1, prev_close + atr_p * self.atr_stop_mult, np.nan),
        )

        # Confidence: volume spike strength
        vol_ratio = (df["volume"].shift(1) / vol_avg.shift(1)).clip(0, 3) / 3
        df["confidence"] = vol_ratio.fillna(0.3)

        return df
