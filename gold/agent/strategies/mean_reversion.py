"""
Mean-Reversion Strategy
- Bollinger Bands / z-score deviation entry
- Volatility filter to avoid trading during extreme moves
- Exits when price returns to the mean

WHY THIS MAY FAIL:
- Gold can trend strongly for weeks; mean-reversion gets crushed in trends
- "The market can stay irrational longer than you can stay solvent"
- Volatility filter reduces trade count, which hurts statistical significance
- In regime changes, the "mean" itself shifts, invalidating the setup
"""
import numpy as np
import pandas as pd

from data.indicators import bollinger_bands, zscore
from strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    def __init__(self, cfg: dict):
        super().__init__("mean_reversion", cfg)
        scfg = cfg["strategies"]["mean_reversion"]
        self.bb_period = scfg["bb_period"]
        self.bb_std = scfg["bb_std"]
        self.zscore_entry = scfg["zscore_entry"]
        self.zscore_exit = scfg["zscore_exit"]
        self.vol_period = scfg["volatility_filter_period"]
        self.min_vol_pctile = scfg["min_volatility_percentile"]
        self.max_vol_pctile = scfg["max_volatility_percentile"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Bollinger Bands
        upper, middle, lower = bollinger_bands(df["close"], self.bb_period, self.bb_std)

        # Z-score
        zs = zscore(df["close"], self.bb_period)

        # Volatility filter: rolling std percentile
        vol = df["close"].pct_change().rolling(self.vol_period).std()
        vol_rank = vol.rank(pct=True) * 100

        # Shift everything by 1 to avoid lookahead
        zs_prev = zs.shift(1)
        lower_prev = lower.shift(1)
        upper_prev = upper.shift(1)
        middle_prev = middle.shift(1)
        vol_rank_prev = vol_rank.shift(1)
        prev_close = df["close"].shift(1)

        # Volatility filter: only trade in "normal" volatility
        vol_ok = (vol_rank_prev >= self.min_vol_pctile) & (
            vol_rank_prev <= self.max_vol_pctile
        )

        # Long: z-score below threshold (oversold) + vol filter
        long_cond = (zs_prev <= self.zscore_entry) & vol_ok
        # Short: z-score above -threshold (overbought) + vol filter
        short_cond = (zs_prev >= -self.zscore_entry) & vol_ok

        df["signal"] = 0
        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1

        # Stop: beyond the Bollinger Band
        df["stop_price"] = np.where(
            df["signal"] == 1,
            lower_prev - (middle_prev - lower_prev) * 0.5,
            np.where(
                df["signal"] == -1,
                upper_prev + (upper_prev - middle_prev) * 0.5,
                np.nan,
            ),
        )

        # Target is the mean (middle BB)
        df["target_price"] = middle_prev

        # Confidence inversely proportional to distance from mean
        df["confidence"] = (zs_prev.abs() / 3.0).clip(0, 1)

        return df
