"""
Multi-Timeframe Consensus Strategy (User's core idea)
- Simulates 5M, 15M, and 30M candle analysis using 1H data
- Resamples 1H bars into 2H (short), 4H (medium), and 8H (long) timeframes
- Only enters when ALL three timeframes agree on direction
- Maximum 3-5 trades per day enforced via daily trade counter

WHY THIS MAY FAIL:
- Requiring 3-timeframe consensus drastically reduces trade count
- Higher timeframes lag more, so entries are late by definition
- Resampled bars from 1H data lose some intra-bar information
"""
import numpy as np
import pandas as pd

from data.indicators import ema, rsi, macd, atr
from strategies.base import BaseStrategy


class MultiTimeframeStrategy(BaseStrategy):
    def __init__(self, cfg: dict):
        super().__init__("multi_timeframe", cfg)
        self.max_daily_trades = cfg.get("strategies", {}).get(
            "multi_timeframe", {}
        ).get("max_daily_trades", 5)
        self.atr_stop_mult = 2.0

    def _resample_ohlcv(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample OHLCV data to a higher timeframe."""
        return df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

    def _compute_tf_signal(self, df_tf: pd.DataFrame) -> pd.Series:
        """Compute directional signal for one timeframe: +1, -1, 0."""
        fast = ema(df_tf["close"], 8)
        slow = ema(df_tf["close"], 21)
        rsi_val = rsi(df_tf["close"], 14)

        # Shift to prevent lookahead
        fast_p = fast.shift(1)
        slow_p = slow.shift(1)
        rsi_p = rsi_val.shift(1)

        sig = pd.Series(0, index=df_tf.index)
        # Bullish: fast > slow AND RSI > 45 (not oversold)
        sig[(fast_p > slow_p) & (rsi_p > 45)] = 1
        # Bearish: fast < slow AND RSI < 55 (not overbought)
        sig[(fast_p < slow_p) & (rsi_p < 55)] = -1
        return sig

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Build 3 timeframes from the base data
        # Short (2H), Medium (4H), Long (8H) — simulating multi-TF analysis
        tf_short = self._resample_ohlcv(df, "2h")
        tf_medium = self._resample_ohlcv(df, "4h")
        tf_long = self._resample_ohlcv(df, "8h")

        sig_short = self._compute_tf_signal(tf_short)
        sig_medium = self._compute_tf_signal(tf_medium)
        sig_long = self._compute_tf_signal(tf_long)

        # Reindex all signals to the base timeframe (forward-fill)
        sig_short = sig_short.reindex(df.index, method="ffill").fillna(0)
        sig_medium = sig_medium.reindex(df.index, method="ffill").fillna(0)
        sig_long = sig_long.reindex(df.index, method="ffill").fillna(0)

        # Consensus: all three must agree
        consensus = pd.Series(0, index=df.index)
        all_long = (sig_short == 1) & (sig_medium == 1) & (sig_long == 1)
        all_short = (sig_short == -1) & (sig_medium == -1) & (sig_long == -1)
        consensus[all_long] = 1
        consensus[all_short] = -1

        # Enforce max daily trades: only allow signal changes, cap per day
        df["signal"] = 0
        df["_date"] = df.index.date
        daily_count = {}
        prev_signal = 0
        for i in range(len(df)):
            day = df["_date"].iloc[i]
            c = consensus.iloc[i]
            if c != prev_signal and c != 0:
                count = daily_count.get(day, 0)
                if count < self.max_daily_trades:
                    df.iloc[i, df.columns.get_loc("signal")] = c
                    daily_count[day] = count + 1
                    prev_signal = c
            elif c == 0 and prev_signal != 0:
                prev_signal = 0

        # ATR-based stops
        atr_val = atr(df, 14).shift(1)
        prev_close = df["close"].shift(1)
        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_val * self.atr_stop_mult,
            np.where(df["signal"] == -1, prev_close + atr_val * self.atr_stop_mult, np.nan),
        )

        # Confidence based on how strong the consensus is
        agreement_strength = (sig_short.abs() + sig_medium.abs() + sig_long.abs()) / 3
        df["confidence"] = agreement_strength.clip(0, 1)

        df.drop(columns=["_date"], inplace=True)
        return df
