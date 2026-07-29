"""
Momentum Confirmation Strategy
- RSI for overbought/oversold
- MACD for momentum direction
- ADX filter to confirm trend strength before entry

WHY THIS MAY FAIL:
- Momentum indicators all lag; by the time they confirm, the move may be over
- RSI stays overbought/oversold for extended periods in strong trends
- ADX filter reduces trades significantly, lowering sample size
- In choppy markets, MACD generates constant whipsaws
"""
import numpy as np
import pandas as pd

from data.indicators import adx, macd, rsi
from strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    def __init__(self, cfg: dict):
        super().__init__("momentum", cfg)
        scfg = cfg["strategies"]["momentum"]
        self.rsi_period = scfg["rsi_period"]
        self.rsi_ob = scfg["rsi_overbought"]
        self.rsi_os = scfg["rsi_oversold"]
        self.macd_fast = scfg["macd_fast"]
        self.macd_slow = scfg["macd_slow"]
        self.macd_signal_period = scfg["macd_signal"]
        self.adx_period = scfg["adx_period"]
        self.adx_threshold = scfg["adx_threshold"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Compute indicators
        rsi_vals = rsi(df["close"], self.rsi_period)
        macd_line, signal_line, macd_hist = macd(
            df["close"], self.macd_fast, self.macd_slow, self.macd_signal_period
        )
        adx_vals = adx(df, self.adx_period)

        # Shift by 1 to avoid lookahead
        rsi_prev = rsi_vals.shift(1)
        macd_hist_prev = macd_hist.shift(1)
        macd_hist_prev2 = macd_hist.shift(2)
        adx_prev = adx_vals.shift(1)
        prev_close = df["close"].shift(1)

        # ADX filter: only trade when trend is strong enough
        trend_strong = adx_prev > self.adx_threshold

        # Long: RSI was oversold and turning up + MACD histogram increasing + ADX confirms trend
        long_cond = (
            (rsi_prev < self.rsi_os + 10)  # RSI in lower zone (not extreme oversold only)
            & (rsi_prev > self.rsi_os - 5)  # but recently was near oversold
            & (macd_hist_prev > macd_hist_prev2)  # MACD momentum increasing
            & trend_strong
        )

        # Short: RSI was overbought and turning down + MACD decreasing + ADX confirms
        short_cond = (
            (rsi_prev > self.rsi_ob - 10)
            & (rsi_prev < self.rsi_ob + 5)
            & (macd_hist_prev < macd_hist_prev2)
            & trend_strong
        )

        df["signal"] = 0
        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1

        # Stop: 2% from entry (momentum trades need wider stops)
        stop_pct = 0.02
        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close * (1 - stop_pct),
            np.where(
                df["signal"] == -1,
                prev_close * (1 + stop_pct),
                np.nan,
            ),
        )

        # Confidence based on ADX strength and RSI extremity
        adx_score = ((adx_prev - self.adx_threshold) / 50).clip(0, 1)
        rsi_score = np.where(
            df["signal"] == 1,
            ((self.rsi_os + 10) - rsi_prev) / 20,
            np.where(
                df["signal"] == -1,
                (rsi_prev - (self.rsi_ob - 10)) / 20,
                0,
            ),
        )
        df["confidence"] = (
            (adx_score.values + pd.Series(rsi_score, index=df.index).clip(0, 1).values) / 2
        )

        return df
