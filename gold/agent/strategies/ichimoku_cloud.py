"""
Ichimoku Cloud Strategy
- Uses Tenkan/Kijun cross for entry signals
- Cloud (Senkou A vs B) acts as trend filter — only trade in direction of cloud
- Price must be above/below cloud for confirmation
- ATR-based stop loss

WHY THIS MAY FAIL:
- Ichimoku was designed for daily charts on Japanese equities, not gold intraday
- Cloud lag is significant (26 periods shifted forward)
- In ranging markets, price oscillates around the cloud, generating whipsaws
"""
import numpy as np
import pandas as pd

from data.indicators import ichimoku, atr
from strategies.base import BaseStrategy


class IchimokuCloudStrategy(BaseStrategy):
    def __init__(self, cfg: dict):
        super().__init__("ichimoku_cloud", cfg)
        icfg = cfg.get("strategies", {}).get("ichimoku_cloud", {})
        self.tenkan_period = icfg.get("tenkan_period", 9)
        self.kijun_period = icfg.get("kijun_period", 26)
        self.senkou_b_period = icfg.get("senkou_b_period", 52)
        self.atr_stop_mult = icfg.get("atr_stop_multiple", 2.0)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        tenkan, kijun, senkou_a, senkou_b = ichimoku(
            df, self.tenkan_period, self.kijun_period, self.senkou_b_period
        )
        atr_val = atr(df, 14)

        # Shift to avoid lookahead
        t_prev = tenkan.shift(1)
        k_prev = kijun.shift(1)
        sa_prev = senkou_a.shift(1)
        sb_prev = senkou_b.shift(1)
        prev_close = df["close"].shift(1)
        atr_prev = atr_val.shift(1)

        # Cloud is bullish when senkou_a > senkou_b
        cloud_bull = sa_prev > sb_prev
        cloud_bear = sa_prev < sb_prev

        # Price above cloud for longs, below for shorts
        cloud_top = pd.concat([sa_prev, sb_prev], axis=1).max(axis=1)
        cloud_bottom = pd.concat([sa_prev, sb_prev], axis=1).min(axis=1)
        price_above_cloud = prev_close > cloud_top
        price_below_cloud = prev_close < cloud_bottom

        # Tenkan/Kijun cross
        tk_cross_up = (t_prev > k_prev) & (tenkan.shift(2) <= kijun.shift(2))
        tk_cross_down = (t_prev < k_prev) & (tenkan.shift(2) >= kijun.shift(2))

        # Long: TK cross up + price above cloud + bullish cloud
        long_cond = tk_cross_up & price_above_cloud & cloud_bull
        # Short: TK cross down + price below cloud + bearish cloud
        short_cond = tk_cross_down & price_below_cloud & cloud_bear

        df["signal"] = 0
        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1

        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_prev * self.atr_stop_mult,
            np.where(df["signal"] == -1, prev_close + atr_prev * self.atr_stop_mult, np.nan),
        )

        # Confidence: distance from cloud as % of ATR
        cloud_dist = np.where(
            df["signal"] == 1,
            (prev_close - cloud_top) / atr_prev,
            np.where(df["signal"] == -1, (cloud_bottom - prev_close) / atr_prev, 0),
        )
        df["confidence"] = pd.Series(cloud_dist, index=df.index).clip(0, 1) * 0.5 + 0.3

        return df
