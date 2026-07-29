"""
Momentum + Trend Combined Strategy
- Combines trend-following (EMA cross + regime filter) with momentum (MACD + RSI)
- Both systems must agree before entry
- Designed to filter out false signals from either system alone

WHY THIS MAY FAIL:
- Requiring dual confirmation means very few trades
- When both confirm, the move may already be mostly done
- Still vulnerable to sharp reversals in news-driven gold moves
"""
import numpy as np
import pandas as pd

from data.indicators import ema, sma, rsi, macd, atr, adx
from strategies.base import BaseStrategy


class MomentumTrendStrategy(BaseStrategy):
    def __init__(self, cfg: dict):
        super().__init__("momentum_trend", cfg)
        mcfg = cfg.get("strategies", {}).get("momentum_trend", {})
        self.fast_ma = mcfg.get("fast_ma", 10)
        self.slow_ma = mcfg.get("slow_ma", 30)
        self.regime_ma = mcfg.get("regime_ma", 100)
        self.rsi_period = mcfg.get("rsi_period", 14)
        self.atr_stop_mult = mcfg.get("atr_stop_multiple", 2.0)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Trend components
        fast = ema(df["close"], self.fast_ma)
        slow = ema(df["close"], self.slow_ma)
        regime = sma(df["close"], self.regime_ma)

        # Momentum components
        rsi_vals = rsi(df["close"], self.rsi_period)
        macd_line, signal_line, macd_hist = macd(df["close"], 12, 26, 9)
        adx_vals = adx(df, 14)

        atr_val = atr(df, 14)

        # Shift all
        fast_p = fast.shift(1)
        slow_p = slow.shift(1)
        regime_p = regime.shift(1)
        rsi_p = rsi_vals.shift(1)
        macd_h_p = macd_hist.shift(1)
        macd_h_p2 = macd_hist.shift(2)
        adx_p = adx_vals.shift(1)
        atr_p = atr_val.shift(1)
        prev_close = df["close"].shift(1)

        # Trend says long: fast > slow, price > regime
        trend_long = (fast_p > slow_p) & (prev_close > regime_p)
        trend_short = (fast_p < slow_p) & (prev_close < regime_p)

        # Momentum says long: MACD histogram increasing, RSI not overbought, ADX > 20
        mom_long = (macd_h_p > macd_h_p2) & (rsi_p < 70) & (rsi_p > 30) & (adx_p > 20)
        mom_short = (macd_h_p < macd_h_p2) & (rsi_p > 30) & (rsi_p < 70) & (adx_p > 20)

        df["signal"] = 0
        df.loc[trend_long & mom_long, "signal"] = 1
        df.loc[trend_short & mom_short, "signal"] = -1

        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_p * self.atr_stop_mult,
            np.where(df["signal"] == -1, prev_close + atr_p * self.atr_stop_mult, np.nan),
        )

        # Trailing stop
        df["trailing_stop_distance"] = atr_p * 1.5

        # Confidence from ADX + MA separation
        ma_sep = ((fast_p - slow_p).abs() / slow_p).clip(0, 0.05) / 0.05
        adx_score = ((adx_p - 20) / 40).clip(0, 1)
        df["confidence"] = (ma_sep + adx_score) / 2

        return df
