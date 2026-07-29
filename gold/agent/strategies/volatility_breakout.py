"""
Volatility Contraction Breakout Strategy
- Detects periods of low volatility (Bollinger Band squeeze)
- Enters when volatility expands (bands widen) with directional confirmation
- Uses Keltner Channel inside Bollinger Band to detect squeeze
- ATR trailing stop

WHY THIS MAY FAIL:
- Not all squeezes lead to meaningful breakouts
- Direction of breakout is a coin flip without other confirmation
- Gold can squeeze for days then expand in the wrong direction
"""
import numpy as np
import pandas as pd

from data.indicators import bollinger_bands, ema, atr, adx
from strategies.base import BaseStrategy


class VolatilityBreakoutStrategy(BaseStrategy):
    def __init__(self, cfg: dict):
        super().__init__("volatility_breakout", cfg)
        vcfg = cfg.get("strategies", {}).get("volatility_breakout", {})
        self.bb_period = vcfg.get("bb_period", 20)
        self.bb_std = vcfg.get("bb_std", 2.0)
        self.keltner_mult = vcfg.get("keltner_multiple", 1.5)
        self.atr_stop_mult = vcfg.get("atr_stop_multiple", 2.0)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = bollinger_bands(df["close"], self.bb_period, self.bb_std)

        # Keltner Channel
        kelt_mid = ema(df["close"], self.bb_period)
        atr_val = atr(df, self.bb_period)
        kelt_upper = kelt_mid + atr_val * self.keltner_mult
        kelt_lower = kelt_mid - atr_val * self.keltner_mult

        # Squeeze: BB inside Keltner
        squeeze = (bb_lower.shift(1) > kelt_lower.shift(1)) & (bb_upper.shift(1) < kelt_upper.shift(1))
        squeeze_prev = (bb_lower.shift(2) > kelt_lower.shift(2)) & (bb_upper.shift(2) < kelt_upper.shift(2))

        # Squeeze release: was squeezing, now not
        squeeze_release = squeeze_prev & ~squeeze

        # Direction from momentum (close relative to midline)
        momentum = df["close"].shift(1) - bb_mid.shift(1)
        mom_increasing = momentum > momentum.shift(1)

        adx_vals = adx(df, 14).shift(1)
        atr_p = atr_val.shift(1)
        prev_close = df["close"].shift(1)

        # Long: squeeze release + price above midline + momentum increasing
        long_cond = squeeze_release & (momentum > 0) & mom_increasing
        # Short: squeeze release + price below midline + momentum decreasing
        short_cond = squeeze_release & (momentum < 0) & ~mom_increasing

        df["signal"] = 0
        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1

        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_p * self.atr_stop_mult,
            np.where(df["signal"] == -1, prev_close + atr_p * self.atr_stop_mult, np.nan),
        )

        df["trailing_stop_distance"] = atr_p * 1.5

        # Confidence: how long the squeeze lasted
        squeeze_duration = squeeze.astype(int).rolling(50, min_periods=1).sum().shift(1)
        df["confidence"] = (squeeze_duration / 20).clip(0, 1).fillna(0.3)

        return df
