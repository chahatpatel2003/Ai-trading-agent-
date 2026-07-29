"""
Trend-Following Strategy
- Moving average crossover with regime filter (price above/below 200 MA)
- ATR-based stop loss
- Trailing stop using ATR multiple

WHY THIS MAY FAIL:
- Gold can range for months, generating many false crossover signals
- Moving averages lag inherently; entries are late, exits are late
- Transaction costs eat into small trend moves
- Regime filter helps but does not prevent whipsaws in transitions
"""
import numpy as np
import pandas as pd

from data.indicators import atr, ema, sma
from strategies.base import BaseStrategy


class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, cfg: dict):
        super().__init__("trend_following", cfg)
        scfg = cfg["strategies"]["trend_following"]
        self.fast_ma = scfg["fast_ma"]
        self.slow_ma = scfg["slow_ma"]
        self.atr_period = scfg["atr_period"]
        self.atr_stop_mult = scfg["atr_stop_multiple"]
        self.trailing_atr_mult = scfg["trailing_stop_atr_multiple"]
        self.regime_ma = scfg["regime_ma"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Compute indicators
        df["fast_ma"] = ema(df["close"], self.fast_ma)
        df["slow_ma"] = ema(df["close"], self.slow_ma)
        df["regime_ma"] = sma(df["close"], self.regime_ma)
        df["atr"] = atr(df, self.atr_period)

        # Shift indicators by 1 to prevent lookahead
        fast = df["fast_ma"].shift(1)
        slow = df["slow_ma"].shift(1)
        regime = df["regime_ma"].shift(1)
        prev_close = df["close"].shift(1)
        atr_val = df["atr"].shift(1)

        # Long: fast MA > slow MA AND price above regime MA
        long_cond = (fast > slow) & (prev_close > regime)
        # Short: fast MA < slow MA AND price below regime MA
        short_cond = (fast < slow) & (prev_close < regime)

        df["signal"] = 0
        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1

        # Stop prices: entry +/- ATR * multiplier
        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_val * self.atr_stop_mult,
            np.where(
                df["signal"] == -1,
                prev_close + atr_val * self.atr_stop_mult,
                np.nan,
            ),
        )

        # Trailing stop distance (used by backtest engine)
        df["trailing_stop_distance"] = atr_val * self.trailing_atr_mult

        # Confidence based on MA separation strength
        ma_diff = (fast - slow).abs() / slow
        df["confidence"] = ma_diff.clip(0, 0.05) / 0.05  # normalize 0-1

        # Clean up indicator columns
        df.drop(columns=["fast_ma", "slow_ma", "regime_ma", "atr"], inplace=True)

        return df
