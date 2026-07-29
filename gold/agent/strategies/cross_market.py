"""
Cross-Market Strategies for XAUUSD

These strategies exploit inter-market relationships between gold and other
asset classes (DXY, silver, VIX, oil, copper, BTC, bonds, miners).
Each strategy requires external market data fed via set_market_data() and/or
set_market_history() before calling generate_signals().

IMPORTANT: All cross-market data must be pre-fetched and synchronized to the
gold timeframe before being passed in. These strategies do NOT fetch data
themselves.

WHY CROSS-MARKET STRATEGIES MATTER:
- Gold does not trade in isolation; it reacts to macro forces
- Inter-market divergences often precede large gold moves
- Combining cross-market signals with technical analysis increases conviction

WHY THEY MAY FAIL:
- Correlations are unstable and regime-dependent
- Cross-market data has latency and alignment issues
- Relationships that worked historically can break down
- Many cross-market signals are already priced in by institutional desks
"""
import numpy as np
import pandas as pd
from typing import Optional

from data.indicators import rsi, ema, sma, atr, bollinger_bands
from strategies.base import BaseStrategy


# ---------------------------------------------------------------------------
# Helpers shared across strategies
# ---------------------------------------------------------------------------

def _safe_zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling z-score that handles NaN and zero-std gracefully."""
    rolling_mean = series.rolling(window=period, min_periods=period).mean()
    rolling_std = series.rolling(window=period, min_periods=period).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def _pct_change_n(series: pd.Series, n: int = 1) -> pd.Series:
    """Percentage change over n periods."""
    return series.pct_change(n)


def _has_required_history(history: Optional[dict], keys: list, min_len: int = 20) -> bool:
    """Check that market_history has the required keys with enough data."""
    if history is None:
        return False
    for key in keys:
        if key not in history:
            return False
        val = history[key]
        if not isinstance(val, pd.DataFrame) and not isinstance(val, pd.Series):
            return False
        if len(val) < min_len:
            return False
    return True


def _get_close(data, key: str) -> Optional[pd.Series]:
    """Extract close series from history dict entry (DataFrame or Series)."""
    val = data.get(key)
    if val is None:
        return None
    if isinstance(val, pd.DataFrame):
        if "close" in val.columns:
            return val["close"]
        # Fall back to first column
        return val.iloc[:, 0]
    if isinstance(val, pd.Series):
        return val
    return None


# ===========================================================================
# 1. DXY Divergence Strategy
# ===========================================================================

class DXYDivergenceStrategy(BaseStrategy):
    """
    Exploits the inverse relationship between the US Dollar Index (DXY) and
    gold. When DXY drops sharply but gold hasn't responded, LONG gold.
    When DXY surges but gold hasn't fallen, SHORT gold.

    Uses rolling correlation and z-score of DXY returns vs gold returns.
    Timeframe: 1H / 4H.

    WHY THIS MAY FAIL:
    - The DXY-gold inverse relationship weakens during liquidity crises
      when both can fall together (margin call liquidation).
    - If the divergence is driven by EUR moves within DXY (DXY is 57% EUR),
      gold may legitimately not respond.
    """

    def __init__(self, cfg: dict):
        super().__init__("dxy_divergence", cfg)
        scfg = cfg.get("strategies", {}).get("dxy_divergence", {})
        self.lookback = scfg.get("lookback", 20)
        self.zscore_threshold = scfg.get("zscore_threshold", 1.5)
        self.corr_window = scfg.get("corr_window", 30)
        self.atr_period = scfg.get("atr_period", 14)
        self.atr_stop_mult = scfg.get("atr_stop_multiple", 2.0)
        self.market_data: Optional[dict] = None
        self.market_history: Optional[dict] = None
        self.confidence: float = 0.0
        self.reasons: list = []

    def set_market_data(self, data: dict):
        self.market_data = data

    def set_market_history(self, history: dict):
        self.market_history = history

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["stop_price"] = np.nan
        df["confidence"] = 0.0
        self.confidence = 0.0
        self.reasons = []

        if not _has_required_history(self.market_history, ["dxy"], min_len=self.lookback):
            return df

        dxy_close = _get_close(self.market_history, "dxy")
        if dxy_close is None or len(dxy_close) < self.lookback:
            return df

        # Align DXY data to gold index (use last N rows of each)
        n = min(len(df), len(dxy_close))
        gold_returns = df["close"].pct_change().iloc[-n:]
        dxy_returns = dxy_close.pct_change().iloc[-n:]

        # Reindex DXY returns to match gold DataFrame index
        dxy_returns = dxy_returns.values
        gold_returns_vals = gold_returns.values

        # Compute rolling z-score of DXY returns
        dxy_series = pd.Series(dxy_returns, index=gold_returns.index)
        gold_series = pd.Series(gold_returns_vals, index=gold_returns.index)

        dxy_zscore = _safe_zscore(dxy_series, self.lookback)
        gold_zscore = _safe_zscore(gold_series, self.lookback)

        # Compute rolling correlation
        corr = dxy_series.rolling(self.corr_window, min_periods=self.corr_window).corr(gold_series)

        # ATR for stops
        df["_atr"] = atr(df, self.atr_period)
        atr_val = df["_atr"].shift(1)
        prev_close = df["close"].shift(1)

        # Map z-scores back to full df index
        dxy_z_full = pd.Series(np.nan, index=df.index)
        gold_z_full = pd.Series(np.nan, index=df.index)
        corr_full = pd.Series(np.nan, index=df.index)

        # Fill aligned portion
        overlap_idx = dxy_zscore.index.intersection(df.index)
        if len(overlap_idx) > 0:
            dxy_z_full.loc[overlap_idx] = dxy_zscore.loc[overlap_idx]
            gold_z_full.loc[overlap_idx] = gold_zscore.loc[overlap_idx]
            corr_full.loc[overlap_idx] = corr.loc[overlap_idx]
        else:
            # Fall back: assign by position to last n rows
            tail_idx = df.index[-len(dxy_zscore):]
            min_l = min(len(tail_idx), len(dxy_zscore))
            dxy_z_full.iloc[-min_l:] = dxy_zscore.values[-min_l:]
            gold_z_full.iloc[-min_l:] = gold_zscore.values[-min_l:]
            corr_full.iloc[-min_l:] = corr.values[-min_l:]

        # Shift to avoid lookahead
        dxy_z_prev = dxy_z_full.shift(1)
        gold_z_prev = gold_z_full.shift(1)
        corr_prev = corr_full.shift(1)

        # Signals: DXY dropping sharply (negative z-score), gold not responding
        long_cond = (
            (dxy_z_prev < -self.zscore_threshold) &
            (gold_z_prev.abs() < self.zscore_threshold * 0.5) &
            (corr_prev < 0)  # Inverse correlation is active
        )
        short_cond = (
            (dxy_z_prev > self.zscore_threshold) &
            (gold_z_prev.abs() < self.zscore_threshold * 0.5) &
            (corr_prev < 0)
        )

        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1

        # Stop prices
        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_val * self.atr_stop_mult,
            np.where(
                df["signal"] == -1,
                prev_close + atr_val * self.atr_stop_mult,
                np.nan,
            ),
        )

        # Confidence based on z-score magnitude and correlation strength
        divergence_strength = dxy_z_prev.abs() / (self.zscore_threshold * 3)
        corr_strength = corr_prev.abs()
        df["confidence"] = (divergence_strength * 0.6 + corr_strength * 0.4).clip(0, 1).fillna(0)
        df.loc[df["signal"] == 0, "confidence"] = 0.0

        # Set instance-level confidence from last signal
        last_signal_mask = df["signal"] != 0
        if last_signal_mask.any():
            last_idx = df.loc[last_signal_mask].index[-1]
            self.confidence = float(df.loc[last_idx, "confidence"])
            direction = "LONG" if df.loc[last_idx, "signal"] == 1 else "SHORT"
            self.reasons = [
                f"DXY divergence detected: {direction} gold",
                f"DXY z-score: {dxy_z_prev.loc[last_idx]:.2f}",
                f"Gold z-score: {gold_z_prev.loc[last_idx]:.2f}",
                f"Rolling correlation: {corr_prev.loc[last_idx]:.2f}",
            ]

        df.drop(columns=["_atr"], inplace=True)
        return df


# ===========================================================================
# 2. Silver Lead Strategy
# ===========================================================================

class SilverLeadStrategy(BaseStrategy):
    """
    Silver often leads gold by 1-3 candles. When silver breaks above its
    Bollinger Band upper on 1H with volume confirmation, go LONG gold.
    Uses silver's breakout as an early-warning signal.

    Timeframe: 1H.

    WHY THIS MAY FAIL:
    - Silver is more volatile and produces more false breakouts
    - The lead-lag relationship varies with market conditions
    - Silver can break out on industrial demand news irrelevant to gold
    """

    def __init__(self, cfg: dict):
        super().__init__("silver_lead", cfg)
        scfg = cfg.get("strategies", {}).get("silver_lead", {})
        self.bb_period = scfg.get("bb_period", 20)
        self.bb_std = scfg.get("bb_std", 2.0)
        self.volume_mult = scfg.get("volume_mult", 1.5)
        self.atr_period = scfg.get("atr_period", 14)
        self.atr_stop_mult = scfg.get("atr_stop_multiple", 2.0)
        self.market_data: Optional[dict] = None
        self.market_history: Optional[dict] = None
        self.confidence: float = 0.0
        self.reasons: list = []

    def set_market_data(self, data: dict):
        self.market_data = data

    def set_market_history(self, history: dict):
        self.market_history = history

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["stop_price"] = np.nan
        df["confidence"] = 0.0
        self.confidence = 0.0
        self.reasons = []

        if not _has_required_history(self.market_history, ["silver"], min_len=self.bb_period + 5):
            return df

        silver_hist = self.market_history["silver"]
        if isinstance(silver_hist, pd.Series):
            return df  # Need DataFrame with close and volume

        if not isinstance(silver_hist, pd.DataFrame):
            return df

        if "close" not in silver_hist.columns:
            return df

        silver_close = silver_hist["close"]
        silver_volume = silver_hist.get("volume", silver_hist.get("tick_volume"))
        if silver_volume is None:
            silver_volume = pd.Series(1.0, index=silver_close.index)

        # Compute silver Bollinger Bands
        bb_upper, bb_mid, bb_lower = bollinger_bands(silver_close, self.bb_period, self.bb_std)

        # Volume average
        vol_avg = silver_volume.rolling(self.bb_period, min_periods=self.bb_period).mean()

        # Silver breakout above upper BB with volume
        silver_above_bb = silver_close > bb_upper
        silver_below_bb = silver_close < bb_lower
        silver_high_vol = silver_volume > (vol_avg * self.volume_mult)

        silver_long_signal = silver_above_bb & silver_high_vol
        silver_short_signal = silver_below_bb & silver_high_vol

        # Shift silver signals: silver leads by 1-3 candles, we react 1 candle later
        # Use shift(1) on silver signal to avoid lookahead
        n = min(len(df), len(silver_long_signal))

        long_flags = pd.Series(False, index=df.index)
        short_flags = pd.Series(False, index=df.index)

        silver_long_vals = silver_long_signal.shift(1).values[-n:]
        silver_short_vals = silver_short_signal.shift(1).values[-n:]

        long_flags.iloc[-n:] = silver_long_vals
        short_flags.iloc[-n:] = silver_short_vals

        # ATR for stops
        df["_atr"] = atr(df, self.atr_period)
        atr_val = df["_atr"].shift(1)
        prev_close = df["close"].shift(1)

        df.loc[long_flags.fillna(False), "signal"] = 1
        df.loc[short_flags.fillna(False), "signal"] = -1

        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_val * self.atr_stop_mult,
            np.where(
                df["signal"] == -1,
                prev_close + atr_val * self.atr_stop_mult,
                np.nan,
            ),
        )

        # Confidence: higher when silver BB breakout is stronger
        bb_width_series = (bb_upper - bb_lower) / bb_mid
        bb_width_vals = bb_width_series.shift(1).values[-n:]
        vol_ratio_vals = (silver_volume / vol_avg).shift(1).values[-n:]

        confidence_vals = pd.Series(0.0, index=df.index)
        width_full = pd.Series(np.nan, index=df.index)
        vol_full = pd.Series(np.nan, index=df.index)
        width_full.iloc[-n:] = bb_width_vals
        vol_full.iloc[-n:] = vol_ratio_vals

        # Normalized confidence from volume ratio
        confidence_vals = (vol_full / 3.0).clip(0, 1).fillna(0)
        df["confidence"] = confidence_vals
        df.loc[df["signal"] == 0, "confidence"] = 0.0

        last_signal_mask = df["signal"] != 0
        if last_signal_mask.any():
            last_idx = df.loc[last_signal_mask].index[-1]
            self.confidence = float(df.loc[last_idx, "confidence"])
            direction = "LONG" if df.loc[last_idx, "signal"] == 1 else "SHORT"
            self.reasons = [
                f"Silver lead signal: {direction} gold",
                f"Silver broke {'above upper' if df.loc[last_idx, 'signal'] == 1 else 'below lower'} Bollinger Band",
                "Volume confirmation present",
            ]

        df.drop(columns=["_atr"], inplace=True)
        return df


# ===========================================================================
# 3. VIX Spike Strategy
# ===========================================================================

class VIXSpikeStrategy(BaseStrategy):
    """
    When VIX spikes >20% in one day or crosses above absolute level 25,
    go LONG gold as a safe-haven play. Tight stops, expect a quick move.

    Timeframe: 1H / 4H.

    WHY THIS MAY FAIL:
    - VIX spikes can reverse quickly, trapping late entries
    - In 2008-style crashes, gold initially sold off with everything
    - VIX above 25 can persist without gold rallying (already priced in)
    """

    def __init__(self, cfg: dict):
        super().__init__("vix_spike", cfg)
        scfg = cfg.get("strategies", {}).get("vix_spike", {})
        self.spike_pct = scfg.get("spike_pct", 0.20)
        self.abs_level = scfg.get("abs_level", 25.0)
        self.atr_period = scfg.get("atr_period", 14)
        self.atr_stop_mult = scfg.get("atr_stop_multiple", 1.5)
        self.market_data: Optional[dict] = None
        self.market_history: Optional[dict] = None
        self.confidence: float = 0.0
        self.reasons: list = []

    def set_market_data(self, data: dict):
        self.market_data = data

    def set_market_history(self, history: dict):
        self.market_history = history

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["stop_price"] = np.nan
        df["confidence"] = 0.0
        self.confidence = 0.0
        self.reasons = []

        if not _has_required_history(self.market_history, ["vix"], min_len=5):
            return df

        vix_close = _get_close(self.market_history, "vix")
        if vix_close is None or len(vix_close) < 5:
            return df

        # VIX percentage change (daily)
        vix_pct = vix_close.pct_change()

        # Conditions: spike > threshold OR absolute level above threshold
        vix_spike = vix_pct > self.spike_pct
        vix_elevated = vix_close > self.abs_level

        # Combined: either condition triggers
        vix_trigger = vix_spike | vix_elevated

        # Map to gold timeframe (align by position)
        n = min(len(df), len(vix_trigger))
        trigger_flags = pd.Series(False, index=df.index)
        trigger_flags.iloc[-n:] = vix_trigger.shift(1).fillna(False).values[-n:]  # Shift to avoid lookahead

        # Map VIX values for confidence calculation
        vix_vals_full = pd.Series(np.nan, index=df.index)
        vix_pct_full = pd.Series(np.nan, index=df.index)
        vix_vals_full.iloc[-n:] = vix_close.shift(1).values[-n:]
        vix_pct_full.iloc[-n:] = vix_pct.shift(1).values[-n:]

        # ATR for stops
        df["_atr"] = atr(df, self.atr_period)
        atr_val = df["_atr"].shift(1)
        prev_close = df["close"].shift(1)

        # VIX spike = LONG gold (safe haven)
        df.loc[trigger_flags.fillna(False), "signal"] = 1

        # Tight stops for VIX-based trades
        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_val * self.atr_stop_mult,
            np.nan,
        )

        # Confidence: higher for larger VIX spikes and higher absolute levels
        spike_conf = (vix_pct_full.abs() / 0.40).clip(0, 0.5)  # 40% spike = 0.5
        level_conf = ((vix_vals_full - 20) / 30).clip(0, 0.5)  # VIX 50 = 0.5
        df["confidence"] = (spike_conf + level_conf).clip(0, 1).fillna(0)
        df.loc[df["signal"] == 0, "confidence"] = 0.0

        last_signal_mask = df["signal"] != 0
        if last_signal_mask.any():
            last_idx = df.loc[last_signal_mask].index[-1]
            self.confidence = float(df.loc[last_idx, "confidence"])
            self.reasons = [
                "VIX spike detected: LONG gold (safe haven)",
                f"VIX level: {vix_vals_full.loc[last_idx]:.1f}",
                f"VIX change: {vix_pct_full.loc[last_idx] * 100:.1f}%",
            ]

        df.drop(columns=["_atr"], inplace=True)
        return df


# ===========================================================================
# 4. Copper/Gold Ratio Strategy
# ===========================================================================

class CopperGoldRatioStrategy(BaseStrategy):
    """
    The copper/gold ratio is a proxy for global growth expectations.
    Falling ratio = growth concerns = LONG gold.
    Rising ratio = risk-on = SHORT gold.
    Uses z-score of the ratio relative to its 30-day mean.

    Timeframe: 4H / 1D.

    WHY THIS MAY FAIL:
    - Copper supply disruptions can distort the ratio
    - The ratio works on multi-week horizons; short-term noise is high
    - Gold can rally on inflation even as growth improves (ratio rising)
    """

    def __init__(self, cfg: dict):
        super().__init__("copper_gold_ratio", cfg)
        scfg = cfg.get("strategies", {}).get("copper_gold_ratio", {})
        self.ratio_period = scfg.get("ratio_period", 30)
        self.zscore_threshold = scfg.get("zscore_threshold", 1.5)
        self.atr_period = scfg.get("atr_period", 14)
        self.atr_stop_mult = scfg.get("atr_stop_multiple", 2.5)
        self.market_data: Optional[dict] = None
        self.market_history: Optional[dict] = None
        self.confidence: float = 0.0
        self.reasons: list = []

    def set_market_data(self, data: dict):
        self.market_data = data

    def set_market_history(self, history: dict):
        self.market_history = history

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["stop_price"] = np.nan
        df["confidence"] = 0.0
        self.confidence = 0.0
        self.reasons = []

        if not _has_required_history(self.market_history, ["copper"], min_len=self.ratio_period + 5):
            return df

        copper_close = _get_close(self.market_history, "copper")
        if copper_close is None:
            return df

        # Compute copper/gold ratio using aligned data
        n = min(len(df), len(copper_close))
        gold_vals = df["close"].iloc[-n:].values
        copper_vals = copper_close.iloc[-n:].values

        # Copper/Gold ratio (copper in $/lb, gold in $/oz, ratio is unitless indicator)
        ratio = pd.Series(copper_vals / np.where(gold_vals == 0, np.nan, gold_vals),
                          index=df.index[-n:])

        # Z-score of the ratio
        ratio_zscore = _safe_zscore(ratio, self.ratio_period)

        # Rate of change of ratio (is it falling or rising?)
        ratio_roc = ratio.pct_change(5)  # 5-period rate of change

        # Map to full df
        ratio_z_full = pd.Series(np.nan, index=df.index)
        ratio_roc_full = pd.Series(np.nan, index=df.index)
        ratio_z_full.iloc[-n:] = ratio_zscore.values
        ratio_roc_full.iloc[-n:] = ratio_roc.values

        # Shift to avoid lookahead
        ratio_z_prev = ratio_z_full.shift(1)
        ratio_roc_prev = ratio_roc_full.shift(1)

        # ATR for stops
        df["_atr"] = atr(df, self.atr_period)
        atr_val = df["_atr"].shift(1)
        prev_close = df["close"].shift(1)

        # Falling ratio (negative z-score + negative ROC) = LONG gold
        long_cond = (ratio_z_prev < -self.zscore_threshold) & (ratio_roc_prev < 0)
        # Rising ratio (positive z-score + positive ROC) = SHORT gold
        short_cond = (ratio_z_prev > self.zscore_threshold) & (ratio_roc_prev > 0)

        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1

        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_val * self.atr_stop_mult,
            np.where(
                df["signal"] == -1,
                prev_close + atr_val * self.atr_stop_mult,
                np.nan,
            ),
        )

        # Confidence from z-score magnitude
        df["confidence"] = (ratio_z_prev.abs() / (self.zscore_threshold * 3)).clip(0, 1).fillna(0)
        df.loc[df["signal"] == 0, "confidence"] = 0.0

        last_signal_mask = df["signal"] != 0
        if last_signal_mask.any():
            last_idx = df.loc[last_signal_mask].index[-1]
            self.confidence = float(df.loc[last_idx, "confidence"])
            direction = "LONG" if df.loc[last_idx, "signal"] == 1 else "SHORT"
            self.reasons = [
                f"Copper/Gold ratio signal: {direction} gold",
                f"Ratio z-score: {ratio_z_prev.loc[last_idx]:.2f}",
                f"Ratio 5-period ROC: {ratio_roc_prev.loc[last_idx] * 100:.2f}%",
                f"{'Growth concerns (ratio falling)' if df.loc[last_idx, 'signal'] == 1 else 'Risk-on (ratio rising)'}",
            ]

        df.drop(columns=["_atr"], inplace=True)
        return df


# ===========================================================================
# 5. Oil/Gold Inflation Strategy
# ===========================================================================

class OilGoldInflationStrategy(BaseStrategy):
    """
    Oil rising sharply signals inflation expectations, and gold tends to
    follow with a lag. When oil has a strong 5-day rally AND gold hasn't
    moved, LONG gold.

    Timeframe: 1D.

    WHY THIS MAY FAIL:
    - Oil can spike on supply shocks (geopolitics) without broad inflation
    - Strong dollar can suppress gold even as oil rallies
    - The lag is variable and sometimes gold doesn't follow at all
    """

    def __init__(self, cfg: dict):
        super().__init__("oil_gold_inflation", cfg)
        scfg = cfg.get("strategies", {}).get("oil_gold_inflation", {})
        self.oil_rally_threshold = scfg.get("oil_rally_threshold", 0.05)
        self.gold_quiet_threshold = scfg.get("gold_quiet_threshold", 0.015)
        self.lookback = scfg.get("lookback", 5)
        self.atr_period = scfg.get("atr_period", 14)
        self.atr_stop_mult = scfg.get("atr_stop_multiple", 2.0)
        self.market_data: Optional[dict] = None
        self.market_history: Optional[dict] = None
        self.confidence: float = 0.0
        self.reasons: list = []

    def set_market_data(self, data: dict):
        self.market_data = data

    def set_market_history(self, history: dict):
        self.market_history = history

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["stop_price"] = np.nan
        df["confidence"] = 0.0
        self.confidence = 0.0
        self.reasons = []

        if not _has_required_history(self.market_history, ["oil"], min_len=self.lookback + 5):
            return df

        oil_close = _get_close(self.market_history, "oil")
        if oil_close is None:
            return df

        # Oil N-day return
        oil_return = oil_close.pct_change(self.lookback)

        # Gold N-day return
        gold_return = df["close"].pct_change(self.lookback)

        # Map oil to gold timeframe
        n = min(len(df), len(oil_return))
        oil_ret_full = pd.Series(np.nan, index=df.index)
        oil_ret_full.iloc[-n:] = oil_return.values[-n:]

        # Shift to avoid lookahead
        oil_ret_prev = oil_ret_full.shift(1)
        gold_ret_prev = gold_return.shift(1)

        # ATR for stops
        df["_atr"] = atr(df, self.atr_period)
        atr_val = df["_atr"].shift(1)
        prev_close = df["close"].shift(1)

        # Oil rallying strongly + gold flat = LONG gold
        long_cond = (
            (oil_ret_prev > self.oil_rally_threshold) &
            (gold_ret_prev.abs() < self.gold_quiet_threshold)
        )

        # Oil crashing + gold flat = potential SHORT gold (deflationary signal)
        short_cond = (
            (oil_ret_prev < -self.oil_rally_threshold) &
            (gold_ret_prev.abs() < self.gold_quiet_threshold)
        )

        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1

        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_val * self.atr_stop_mult,
            np.where(
                df["signal"] == -1,
                prev_close + atr_val * self.atr_stop_mult,
                np.nan,
            ),
        )

        # Confidence from oil move magnitude
        df["confidence"] = (oil_ret_prev.abs() / (self.oil_rally_threshold * 3)).clip(0, 1).fillna(0)
        df.loc[df["signal"] == 0, "confidence"] = 0.0

        last_signal_mask = df["signal"] != 0
        if last_signal_mask.any():
            last_idx = df.loc[last_signal_mask].index[-1]
            self.confidence = float(df.loc[last_idx, "confidence"])
            direction = "LONG" if df.loc[last_idx, "signal"] == 1 else "SHORT"
            self.reasons = [
                f"Oil/Gold inflation signal: {direction} gold",
                f"Oil {self.lookback}-day return: {oil_ret_prev.loc[last_idx] * 100:.2f}%",
                f"Gold {self.lookback}-day return: {gold_ret_prev.loc[last_idx] * 100:.2f}%",
                "Gold lagging oil move — expecting catch-up",
            ]

        df.drop(columns=["_atr"], inplace=True)
        return df


# ===========================================================================
# 6. BTC/Gold Capital Flow Strategy
# ===========================================================================

class BTCGoldFlowStrategy(BaseStrategy):
    """
    During fear events, capital rotates from risk assets to safe havens.
    When BTC is dumping AND DXY is weakening AND VIX is rising, all three
    conditions together signal a rotation into gold. LONG gold.

    Timeframe: 4H.

    WHY THIS MAY FAIL:
    - All three conditions rarely align simultaneously, producing few signals
    - During true panic, gold can sell off too (liquidity crunch)
    - BTC/gold correlation has been unstable since 2020
    """

    def __init__(self, cfg: dict):
        super().__init__("btc_gold_flow", cfg)
        scfg = cfg.get("strategies", {}).get("btc_gold_flow", {})
        self.btc_drop_threshold = scfg.get("btc_drop_threshold", -0.05)
        self.dxy_drop_threshold = scfg.get("dxy_drop_threshold", -0.005)
        self.vix_rise_threshold = scfg.get("vix_rise_threshold", 0.10)
        self.lookback = scfg.get("lookback", 5)
        self.atr_period = scfg.get("atr_period", 14)
        self.atr_stop_mult = scfg.get("atr_stop_multiple", 2.0)
        self.market_data: Optional[dict] = None
        self.market_history: Optional[dict] = None
        self.confidence: float = 0.0
        self.reasons: list = []

    def set_market_data(self, data: dict):
        self.market_data = data

    def set_market_history(self, history: dict):
        self.market_history = history

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["stop_price"] = np.nan
        df["confidence"] = 0.0
        self.confidence = 0.0
        self.reasons = []

        required = ["btc", "dxy", "vix"]
        if not _has_required_history(self.market_history, required, min_len=self.lookback + 5):
            return df

        btc_close = _get_close(self.market_history, "btc")
        dxy_close = _get_close(self.market_history, "dxy")
        vix_close = _get_close(self.market_history, "vix")

        if btc_close is None or dxy_close is None or vix_close is None:
            return df

        # Returns over lookback period
        btc_ret = btc_close.pct_change(self.lookback)
        dxy_ret = dxy_close.pct_change(self.lookback)
        vix_ret = vix_close.pct_change(self.lookback)

        # Map to gold timeframe
        n = min(len(df), len(btc_ret), len(dxy_ret), len(vix_ret))

        btc_full = pd.Series(np.nan, index=df.index)
        dxy_full = pd.Series(np.nan, index=df.index)
        vix_full = pd.Series(np.nan, index=df.index)

        btc_full.iloc[-n:] = btc_ret.values[-n:]
        dxy_full.iloc[-n:] = dxy_ret.values[-n:]
        vix_full.iloc[-n:] = vix_ret.values[-n:]

        # Shift to avoid lookahead
        btc_prev = btc_full.shift(1)
        dxy_prev = dxy_full.shift(1)
        vix_prev = vix_full.shift(1)

        # ATR for stops
        df["_atr"] = atr(df, self.atr_period)
        atr_val = df["_atr"].shift(1)
        prev_close = df["close"].shift(1)

        # All three conditions must be true
        fear_rotation = (
            (btc_prev < self.btc_drop_threshold) &   # BTC dumping
            (dxy_prev < self.dxy_drop_threshold) &    # Dollar weakening
            (vix_prev > self.vix_rise_threshold)      # VIX rising
        )

        df.loc[fear_rotation.fillna(False), "signal"] = 1

        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_val * self.atr_stop_mult,
            np.nan,
        )

        # Confidence: average of how extreme each condition is
        btc_conf = (btc_prev.abs() / 0.15).clip(0, 1)
        dxy_conf = (dxy_prev.abs() / 0.02).clip(0, 1)
        vix_conf = (vix_prev / 0.40).clip(0, 1)
        df["confidence"] = ((btc_conf + dxy_conf + vix_conf) / 3).clip(0, 1).fillna(0)
        df.loc[df["signal"] == 0, "confidence"] = 0.0

        last_signal_mask = df["signal"] != 0
        if last_signal_mask.any():
            last_idx = df.loc[last_signal_mask].index[-1]
            self.confidence = float(df.loc[last_idx, "confidence"])
            self.reasons = [
                "Fear rotation detected: LONG gold",
                f"BTC {self.lookback}-period return: {btc_prev.loc[last_idx] * 100:.1f}%",
                f"DXY {self.lookback}-period return: {dxy_prev.loc[last_idx] * 100:.2f}%",
                f"VIX {self.lookback}-period return: {vix_prev.loc[last_idx] * 100:.1f}%",
                "All three conditions met: BTC dump + USD weak + VIX rising",
            ]

        df.drop(columns=["_atr"], inplace=True)
        return df


# ===========================================================================
# 7. Real Yield Strategy
# ===========================================================================

class RealYieldStrategy(BaseStrategy):
    """
    Gold moves inversely to real yields (nominal yield minus inflation
    expectations). When the real yield proxy (10Y minus TIPS) is falling,
    LONG gold. Uses rate of change rather than absolute level.

    Timeframe: 1D.

    WHY THIS MAY FAIL:
    - TIPS data may be stale or unavailable in real-time
    - Real yields can fall while gold is selling off (liquidity events)
    - The relationship is strongest on weekly/monthly horizons, not daily
    """

    def __init__(self, cfg: dict):
        super().__init__("real_yield", cfg)
        scfg = cfg.get("strategies", {}).get("real_yield", {})
        self.roc_period = scfg.get("roc_period", 5)
        self.roc_threshold = scfg.get("roc_threshold", 0.10)  # 10bps change
        self.smoothing = scfg.get("smoothing", 5)
        self.atr_period = scfg.get("atr_period", 14)
        self.atr_stop_mult = scfg.get("atr_stop_multiple", 2.5)
        self.market_data: Optional[dict] = None
        self.market_history: Optional[dict] = None
        self.confidence: float = 0.0
        self.reasons: list = []

    def set_market_data(self, data: dict):
        self.market_data = data

    def set_market_history(self, history: dict):
        self.market_history = history

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["stop_price"] = np.nan
        df["confidence"] = 0.0
        self.confidence = 0.0
        self.reasons = []

        if not _has_required_history(self.market_history, ["us10y", "tips"],
                                     min_len=self.roc_period + self.smoothing + 5):
            return df

        us10y = _get_close(self.market_history, "us10y")
        tips = _get_close(self.market_history, "tips")
        if us10y is None or tips is None:
            return df

        # Real yield proxy = 10Y nominal - TIPS yield
        n_yield = min(len(us10y), len(tips))
        real_yield = pd.Series(
            us10y.values[-n_yield:] - tips.values[-n_yield:],
            index=us10y.index[-n_yield:] if hasattr(us10y, 'index') else range(n_yield),
        )

        # Smooth to reduce noise
        real_yield_smooth = sma(real_yield, self.smoothing)

        # Rate of change of real yield (in absolute terms, not percentage)
        real_yield_roc = real_yield_smooth.diff(self.roc_period)

        # Map to gold timeframe
        n = min(len(df), len(real_yield_roc))
        roc_full = pd.Series(np.nan, index=df.index)
        ry_full = pd.Series(np.nan, index=df.index)
        roc_full.iloc[-n:] = real_yield_roc.values[-n:]
        ry_full.iloc[-n:] = real_yield_smooth.values[-n:]

        # Shift to avoid lookahead
        roc_prev = roc_full.shift(1)
        ry_prev = ry_full.shift(1)

        # ATR for stops
        df["_atr"] = atr(df, self.atr_period)
        atr_val = df["_atr"].shift(1)
        prev_close = df["close"].shift(1)

        # Real yield falling = LONG gold
        long_cond = roc_prev < -self.roc_threshold
        # Real yield rising = SHORT gold
        short_cond = roc_prev > self.roc_threshold

        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1

        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_val * self.atr_stop_mult,
            np.where(
                df["signal"] == -1,
                prev_close + atr_val * self.atr_stop_mult,
                np.nan,
            ),
        )

        # Confidence from ROC magnitude
        df["confidence"] = (roc_prev.abs() / (self.roc_threshold * 4)).clip(0, 1).fillna(0)
        df.loc[df["signal"] == 0, "confidence"] = 0.0

        last_signal_mask = df["signal"] != 0
        if last_signal_mask.any():
            last_idx = df.loc[last_signal_mask].index[-1]
            self.confidence = float(df.loc[last_idx, "confidence"])
            direction = "LONG" if df.loc[last_idx, "signal"] == 1 else "SHORT"
            self.reasons = [
                f"Real yield signal: {direction} gold",
                f"Real yield ROC ({self.roc_period}d): {roc_prev.loc[last_idx]:.3f}%",
                f"Real yield level: {ry_prev.loc[last_idx]:.3f}%",
                f"{'Real yields falling -> bullish gold' if df.loc[last_idx, 'signal'] == 1 else 'Real yields rising -> bearish gold'}",
            ]

        df.drop(columns=["_atr"], inplace=True)
        return df


# ===========================================================================
# 8. Asian Session Setup Strategy
# ===========================================================================

class AsianSessionSetupStrategy(BaseStrategy):
    """
    Asian session (00:00-08:00 UTC) forms a consolidation range. At the
    London open (08:00 UTC), if price breaks above the Asian high, LONG.
    If price breaks below the Asian low, SHORT. Requires 2 candles of
    confirmation outside the range.

    Timeframe: 1H.

    WHY THIS MAY FAIL:
    - Asian range can be very tight, leading to false breakouts
    - News releases at London open can cause whipsaws
    - Works best when Asian session is genuinely quiet; fails on Asia-centric news days
    """

    def __init__(self, cfg: dict):
        super().__init__("asian_session_setup", cfg)
        scfg = cfg.get("strategies", {}).get("asian_session_setup", {})
        self.asian_start_hour = scfg.get("asian_start_hour", 0)
        self.asian_end_hour = scfg.get("asian_end_hour", 8)
        self.confirmation_candles = scfg.get("confirmation_candles", 2)
        self.min_range_pips = scfg.get("min_range_pips", 3.0)  # Min range in dollars
        self.max_range_pips = scfg.get("max_range_pips", 25.0)  # Max range (too wide = skip)
        self.atr_period = scfg.get("atr_period", 14)
        self.atr_stop_mult = scfg.get("atr_stop_multiple", 1.5)
        self.market_data: Optional[dict] = None
        self.market_history: Optional[dict] = None
        self.confidence: float = 0.0
        self.reasons: list = []

    def set_market_data(self, data: dict):
        self.market_data = data

    def set_market_history(self, history: dict):
        self.market_history = history

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["stop_price"] = np.nan
        df["confidence"] = 0.0
        self.confidence = 0.0
        self.reasons = []

        # This strategy requires datetime index to identify sessions
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception:
                return df

        # Ensure timezone-aware (UTC)
        if df.index.tz is None:
            try:
                df.index = df.index.tz_localize("UTC")
            except Exception:
                pass

        df["_atr"] = atr(df, self.atr_period)
        atr_val = df["_atr"].shift(1)

        # Get hour from index
        hours = df.index.hour

        # Identify Asian session candles
        is_asian = (hours >= self.asian_start_hour) & (hours < self.asian_end_hour)

        # Group by date to get Asian range per day
        df["_date"] = df.index.date

        for date in df["_date"].unique():
            date_mask = df["_date"] == date
            asian_mask = date_mask & is_asian

            if asian_mask.sum() == 0:
                continue

            asian_high = df.loc[asian_mask, "high"].max()
            asian_low = df.loc[asian_mask, "low"].min()
            asian_range = asian_high - asian_low

            # Skip if range is too small or too large
            if asian_range < self.min_range_pips or asian_range > self.max_range_pips:
                continue

            # London session candles (after Asian close)
            london_mask = date_mask & (hours >= self.asian_end_hour) & (hours < self.asian_end_hour + 8)
            london_indices = df.index[london_mask]

            if len(london_indices) < self.confirmation_candles + 1:
                continue

            # Check for breakout with confirmation
            for i in range(self.confirmation_candles, len(london_indices)):
                idx = london_indices[i]
                # Check previous N candles all closed above/below range
                confirm_indices = london_indices[i - self.confirmation_candles:i]

                # Long breakout: all confirmation candles closed above Asian high
                all_above = all(df.loc[ci, "close"] > asian_high for ci in confirm_indices)
                # Short breakout: all confirmation candles closed below Asian low
                all_below = all(df.loc[ci, "close"] < asian_low for ci in confirm_indices)

                if all_above and df.loc[idx, "signal"] == 0:
                    df.loc[idx, "signal"] = 1
                    stop = asian_low  # Stop below Asian low
                    atr_stop = df.loc[idx, "_atr"] * self.atr_stop_mult if not pd.isna(df.loc[idx, "_atr"]) else asian_range
                    df.loc[idx, "stop_price"] = max(stop, df.loc[idx, "close"] - atr_stop)
                    # Confidence: tighter range = higher confidence (cleaner breakout)
                    range_conf = 1.0 - (asian_range - self.min_range_pips) / (self.max_range_pips - self.min_range_pips)
                    df.loc[idx, "confidence"] = max(0.3, min(0.9, range_conf))
                    break  # One signal per day

                elif all_below and df.loc[idx, "signal"] == 0:
                    df.loc[idx, "signal"] = -1
                    stop = asian_high  # Stop above Asian high
                    atr_stop = df.loc[idx, "_atr"] * self.atr_stop_mult if not pd.isna(df.loc[idx, "_atr"]) else asian_range
                    df.loc[idx, "stop_price"] = min(stop, df.loc[idx, "close"] + atr_stop)
                    range_conf = 1.0 - (asian_range - self.min_range_pips) / (self.max_range_pips - self.min_range_pips)
                    df.loc[idx, "confidence"] = max(0.3, min(0.9, range_conf))
                    break

        # Set instance-level metadata from last signal
        last_signal_mask = df["signal"] != 0
        if last_signal_mask.any():
            last_idx = df.loc[last_signal_mask].index[-1]
            self.confidence = float(df.loc[last_idx, "confidence"])
            direction = "LONG" if df.loc[last_idx, "signal"] == 1 else "SHORT"
            self.reasons = [
                f"Asian session breakout: {direction}",
                f"Breakout confirmed by {self.confirmation_candles} candles outside range",
            ]

        df.drop(columns=["_atr", "_date"], inplace=True)
        return df


# ===========================================================================
# 9. Monday Gap Strategy
# ===========================================================================

class MondayGapStrategy(BaseStrategy):
    """
    If Monday opens with a gap > 0.3% from Friday's close, fade the gap
    (mean reversion). Gap up -> SHORT targeting Friday close. Gap down ->
    LONG targeting Friday close. Only trades on Mondays.

    Timeframe: 1H.

    WHY THIS MAY FAIL:
    - Some gaps are driven by weekend geopolitical events and do NOT fill
    - The 0.3% threshold is arbitrary; gold can gap 1%+ on war/crisis news
    - Monday liquidity is thinner, increasing slippage risk
    """

    def __init__(self, cfg: dict):
        super().__init__("monday_gap", cfg)
        scfg = cfg.get("strategies", {}).get("monday_gap", {})
        self.gap_threshold = scfg.get("gap_threshold", 0.003)
        self.max_gap = scfg.get("max_gap", 0.015)  # Don't fade gaps > 1.5%
        self.atr_period = scfg.get("atr_period", 14)
        self.atr_stop_mult = scfg.get("atr_stop_multiple", 2.0)
        self.market_data: Optional[dict] = None
        self.market_history: Optional[dict] = None
        self.confidence: float = 0.0
        self.reasons: list = []

    def set_market_data(self, data: dict):
        self.market_data = data

    def set_market_history(self, history: dict):
        self.market_history = history

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["stop_price"] = np.nan
        df["confidence"] = 0.0
        self.confidence = 0.0
        self.reasons = []

        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception:
                return df

        if df.index.tz is None:
            try:
                df.index = df.index.tz_localize("UTC")
            except Exception:
                pass

        df["_atr"] = atr(df, self.atr_period)
        atr_val = df["_atr"].shift(1)

        # Day of week: Monday = 0
        df["_dow"] = df.index.dayofweek
        df["_date"] = df.index.date

        # Find Mondays
        mondays = df[df["_dow"] == 0]["_date"].unique()

        for monday in mondays:
            monday_mask = df["_date"] == monday
            monday_indices = df.index[monday_mask]

            if len(monday_indices) == 0:
                continue

            # Find Friday's close (last candle before Monday)
            pre_monday = df.index[df.index < monday_indices[0]]
            if len(pre_monday) == 0:
                continue

            friday_close = df.loc[pre_monday[-1], "close"]
            monday_open = df.loc[monday_indices[0], "open"]

            # Calculate gap
            gap_pct = (monday_open - friday_close) / friday_close

            if abs(gap_pct) < self.gap_threshold:
                continue  # Gap too small

            if abs(gap_pct) > self.max_gap:
                continue  # Gap too large to fade safely

            # Signal on the first Monday candle (already looking at completed data)
            first_monday_idx = monday_indices[0]

            if gap_pct > self.gap_threshold:
                # Gap up -> SHORT (fade the gap, target Friday close)
                df.loc[first_monday_idx, "signal"] = -1
                current_atr = df.loc[first_monday_idx, "_atr"]
                if pd.isna(current_atr):
                    current_atr = abs(monday_open - friday_close) * 2
                df.loc[first_monday_idx, "stop_price"] = monday_open + current_atr * self.atr_stop_mult

            elif gap_pct < -self.gap_threshold:
                # Gap down -> LONG (fade the gap, target Friday close)
                df.loc[first_monday_idx, "signal"] = 1
                current_atr = df.loc[first_monday_idx, "_atr"]
                if pd.isna(current_atr):
                    current_atr = abs(monday_open - friday_close) * 2
                df.loc[first_monday_idx, "stop_price"] = monday_open - current_atr * self.atr_stop_mult

            # Confidence inversely proportional to gap size (smaller gaps fill more reliably)
            gap_conf = 1.0 - (abs(gap_pct) - self.gap_threshold) / (self.max_gap - self.gap_threshold)
            df.loc[first_monday_idx, "confidence"] = max(0.3, min(0.8, gap_conf))

        last_signal_mask = df["signal"] != 0
        if last_signal_mask.any():
            last_idx = df.loc[last_signal_mask].index[-1]
            self.confidence = float(df.loc[last_idx, "confidence"])
            direction = "LONG" if df.loc[last_idx, "signal"] == 1 else "SHORT"
            self.reasons = [
                f"Monday gap fade: {direction}",
                "Expecting gap fill toward Friday close",
            ]

        df.drop(columns=["_atr", "_dow", "_date"], inplace=True)
        return df


# ===========================================================================
# 10. Round Number Strategy
# ===========================================================================

class RoundNumberStrategy(BaseStrategy):
    """
    Gold respects round numbers ($2500, $2550, $2600, etc.) as psychological
    support/resistance. Near round numbers, if RSI is oversold, LONG for a
    bounce. If RSI is overbought, SHORT for a rejection.

    Uses dynamic detection based on current price level.

    Timeframe: 1H.

    WHY THIS MAY FAIL:
    - Round numbers are obvious, so institutional flow can sweep stops there
    - Works better on first touch; repeated tests weaken the level
    - In strong trends, round numbers are blown through without hesitation
    """

    def __init__(self, cfg: dict):
        super().__init__("round_number", cfg)
        scfg = cfg.get("strategies", {}).get("round_number", {})
        self.round_interval = scfg.get("round_interval", 50)  # $50 increments
        self.proximity_pct = scfg.get("proximity_pct", 0.003)  # 0.3% from round number
        self.rsi_period = scfg.get("rsi_period", 14)
        self.rsi_oversold = scfg.get("rsi_oversold", 35)
        self.rsi_overbought = scfg.get("rsi_overbought", 65)
        self.atr_period = scfg.get("atr_period", 14)
        self.atr_stop_mult = scfg.get("atr_stop_multiple", 1.5)
        self.market_data: Optional[dict] = None
        self.market_history: Optional[dict] = None
        self.confidence: float = 0.0
        self.reasons: list = []

    def set_market_data(self, data: dict):
        self.market_data = data

    def set_market_history(self, history: dict):
        self.market_history = history

    def _nearest_round(self, price: float) -> float:
        """Find the nearest round number to a given price."""
        return round(price / self.round_interval) * self.round_interval

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["stop_price"] = np.nan
        df["confidence"] = 0.0
        self.confidence = 0.0
        self.reasons = []

        # Compute RSI and ATR
        df["_rsi"] = rsi(df["close"], self.rsi_period)
        df["_atr"] = atr(df, self.atr_period)

        # Shift for lookahead prevention
        rsi_prev = df["_rsi"].shift(1)
        atr_prev = df["_atr"].shift(1)
        prev_close = df["close"].shift(1)

        # For each bar, find nearest round number and check proximity
        nearest = prev_close.apply(lambda x: self._nearest_round(x) if not pd.isna(x) else np.nan)
        distance_pct = ((prev_close - nearest) / nearest).abs()

        near_round = distance_pct < self.proximity_pct

        # Price below round number + RSI oversold = LONG (bounce off support)
        long_cond = near_round & (prev_close < nearest) & (rsi_prev < self.rsi_oversold)
        # Price above round number + RSI overbought = SHORT (rejection at resistance)
        short_cond = near_round & (prev_close > nearest) & (rsi_prev > self.rsi_overbought)

        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1

        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_prev * self.atr_stop_mult,
            np.where(
                df["signal"] == -1,
                prev_close + atr_prev * self.atr_stop_mult,
                np.nan,
            ),
        )

        # Confidence: closer to round number + more extreme RSI = higher confidence
        proximity_conf = (1.0 - distance_pct / self.proximity_pct).clip(0, 0.5)
        rsi_conf = pd.Series(0.0, index=df.index)
        # More extreme RSI = more confident
        rsi_conf = np.where(
            rsi_prev < self.rsi_oversold,
            ((self.rsi_oversold - rsi_prev) / self.rsi_oversold).clip(0, 0.5),
            np.where(
                rsi_prev > self.rsi_overbought,
                ((rsi_prev - self.rsi_overbought) / (100 - self.rsi_overbought)).clip(0, 0.5),
                0.0,
            ),
        )
        df["confidence"] = (proximity_conf + rsi_conf).clip(0, 1).fillna(0)
        df.loc[df["signal"] == 0, "confidence"] = 0.0

        last_signal_mask = df["signal"] != 0
        if last_signal_mask.any():
            last_idx = df.loc[last_signal_mask].index[-1]
            self.confidence = float(df.loc[last_idx, "confidence"])
            direction = "LONG" if df.loc[last_idx, "signal"] == 1 else "SHORT"
            round_level = self._nearest_round(prev_close.loc[last_idx])
            self.reasons = [
                f"Round number signal: {direction} near ${round_level:.0f}",
                f"RSI: {rsi_prev.loc[last_idx]:.1f}",
                f"Distance from round: {distance_pct.loc[last_idx] * 100:.2f}%",
            ]

        df.drop(columns=["_rsi", "_atr"], inplace=True)
        return df


# ===========================================================================
# 11. India Seasonal Strategy
# ===========================================================================

class IndiaSeasonalStrategy(BaseStrategy):
    """
    Gold demand peaks in India during key festival/wedding seasons:
    - Oct-Nov: Diwali, Dhanteras
    - Jan-Feb: Pongal, Makar Sankranti
    - April: Akshaya Tritiya

    During these months, adds a LONG bias. This is a bias modifier, not a
    standalone entry signal. It enhances confidence on other long signals
    and suppresses short signals.

    Timeframe: 1D.

    WHY THIS MAY FAIL:
    - Indian demand has been declining relative to Chinese/ETF demand
    - Government import duty changes can override seasonal patterns
    - Seasonal patterns are well-known and may be already priced in
    """

    # Months with elevated Indian gold demand
    BULLISH_MONTHS = {1, 2, 4, 10, 11}
    # Peak months (strongest demand)
    PEAK_MONTHS = {10, 11}

    def __init__(self, cfg: dict):
        super().__init__("india_seasonal", cfg)
        scfg = cfg.get("strategies", {}).get("india_seasonal", {})
        self.bias_strength = scfg.get("bias_strength", 0.3)
        self.peak_strength = scfg.get("peak_strength", 0.5)
        self.atr_period = scfg.get("atr_period", 14)
        self.atr_stop_mult = scfg.get("atr_stop_multiple", 3.0)
        self.market_data: Optional[dict] = None
        self.market_history: Optional[dict] = None
        self.confidence: float = 0.0
        self.reasons: list = []

    def set_market_data(self, data: dict):
        self.market_data = data

    def set_market_history(self, history: dict):
        self.market_history = history

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["stop_price"] = np.nan
        df["confidence"] = 0.0
        self.confidence = 0.0
        self.reasons = []

        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception:
                return df

        if df.index.tz is None:
            try:
                df.index = df.index.tz_localize("UTC")
            except Exception:
                pass

        months = df.index.month

        # Bias signal: +1 during bullish months (this is a bias modifier)
        is_peak = months.isin(self.PEAK_MONTHS)
        is_bullish = months.isin(self.BULLISH_MONTHS)

        df["_atr"] = atr(df, self.atr_period)
        atr_val = df["_atr"].shift(1)
        prev_close = df["close"].shift(1)

        # Set a mild long bias during seasonal months
        df.loc[is_bullish & ~is_peak, "signal"] = 1
        df.loc[is_peak, "signal"] = 1

        # Loose stops for seasonal bias trades (not tight scalps)
        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_val * self.atr_stop_mult,
            np.nan,
        )

        # Confidence: higher during peak months
        df.loc[is_bullish & ~is_peak, "confidence"] = self.bias_strength
        df.loc[is_peak, "confidence"] = self.peak_strength
        df.loc[df["signal"] == 0, "confidence"] = 0.0

        last_signal_mask = df["signal"] != 0
        if last_signal_mask.any():
            last_idx = df.loc[last_signal_mask].index[-1]
            self.confidence = float(df.loc[last_idx, "confidence"])
            month = df.index[df.index == last_idx].month[0]
            month_name = pd.Timestamp(year=2000, month=month, day=1).strftime("%B")
            is_peak_month = month in self.PEAK_MONTHS
            self.reasons = [
                f"India seasonal bias: LONG gold ({month_name})",
                f"{'Peak demand month (Diwali/Dhanteras)' if is_peak_month else 'Elevated demand season'}",
                "Bias modifier: enhances other LONG signals, use with confluence",
            ]

        df.drop(columns=["_atr"], inplace=True)
        return df


# ===========================================================================
# 12. Multi-Market Momentum Strategy
# ===========================================================================

class MultiMarketMomentumStrategy(BaseStrategy):
    """
    When gold, silver, AND GDX (gold miners) are ALL trending up
    (price > EMA20 > EMA50), strong bullish confluence -> LONG.
    When all three are trending down -> SHORT.

    Timeframe: 4H.

    WHY THIS MAY FAIL:
    - By the time all three align, much of the move may be over
    - GDX data may lag or be unavailable in real-time
    - Divergences between gold and miners often signal reversals,
      which this strategy misses by requiring alignment
    """

    def __init__(self, cfg: dict):
        super().__init__("multi_market_momentum", cfg)
        scfg = cfg.get("strategies", {}).get("multi_market_momentum", {})
        self.fast_ema_period = scfg.get("fast_ema", 20)
        self.slow_ema_period = scfg.get("slow_ema", 50)
        self.atr_period = scfg.get("atr_period", 14)
        self.atr_stop_mult = scfg.get("atr_stop_multiple", 2.0)
        self.market_data: Optional[dict] = None
        self.market_history: Optional[dict] = None
        self.confidence: float = 0.0
        self.reasons: list = []

    def set_market_data(self, data: dict):
        self.market_data = data

    def set_market_history(self, history: dict):
        self.market_history = history

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["stop_price"] = np.nan
        df["confidence"] = 0.0
        self.confidence = 0.0
        self.reasons = []

        required = ["silver", "gdx"]
        if not _has_required_history(self.market_history, required, min_len=self.slow_ema_period + 5):
            return df

        silver_close = _get_close(self.market_history, "silver")
        gdx_close = _get_close(self.market_history, "gdx")
        if silver_close is None or gdx_close is None:
            return df

        # Gold EMAs
        gold_ema_fast = ema(df["close"], self.fast_ema_period)
        gold_ema_slow = ema(df["close"], self.slow_ema_period)

        # Silver EMAs
        silver_ema_fast = ema(silver_close, self.fast_ema_period)
        silver_ema_slow = ema(silver_close, self.slow_ema_period)

        # GDX EMAs
        gdx_ema_fast = ema(gdx_close, self.fast_ema_period)
        gdx_ema_slow = ema(gdx_close, self.slow_ema_period)

        # Gold trend conditions (shifted to avoid lookahead)
        gold_up = (df["close"].shift(1) > gold_ema_fast.shift(1)) & (gold_ema_fast.shift(1) > gold_ema_slow.shift(1))
        gold_down = (df["close"].shift(1) < gold_ema_fast.shift(1)) & (gold_ema_fast.shift(1) < gold_ema_slow.shift(1))

        # Silver and GDX trend conditions
        n_silver = min(len(df), len(silver_close))
        n_gdx = min(len(df), len(gdx_close))

        silver_up_full = pd.Series(False, index=df.index)
        silver_down_full = pd.Series(False, index=df.index)
        gdx_up_full = pd.Series(False, index=df.index)
        gdx_down_full = pd.Series(False, index=df.index)

        # Silver conditions
        s_up = (silver_close > silver_ema_fast) & (silver_ema_fast > silver_ema_slow)
        s_down = (silver_close < silver_ema_fast) & (silver_ema_fast < silver_ema_slow)
        silver_up_full.iloc[-n_silver:] = s_up.shift(1).fillna(False).values[-n_silver:]
        silver_down_full.iloc[-n_silver:] = s_down.shift(1).fillna(False).values[-n_silver:]

        # GDX conditions
        g_up = (gdx_close > gdx_ema_fast) & (gdx_ema_fast > gdx_ema_slow)
        g_down = (gdx_close < gdx_ema_fast) & (gdx_ema_fast < gdx_ema_slow)
        gdx_up_full.iloc[-n_gdx:] = g_up.shift(1).fillna(False).values[-n_gdx:]
        gdx_down_full.iloc[-n_gdx:] = g_down.shift(1).fillna(False).values[-n_gdx:]

        # ATR for stops
        df["_atr"] = atr(df, self.atr_period)
        atr_val = df["_atr"].shift(1)
        prev_close = df["close"].shift(1)

        # All three trending up = LONG
        all_up = gold_up & silver_up_full.fillna(False) & gdx_up_full.fillna(False)
        # All three trending down = SHORT
        all_down = gold_down & silver_down_full.fillna(False) & gdx_down_full.fillna(False)

        df.loc[all_up, "signal"] = 1
        df.loc[all_down, "signal"] = -1

        df["stop_price"] = np.where(
            df["signal"] == 1,
            prev_close - atr_val * self.atr_stop_mult,
            np.where(
                df["signal"] == -1,
                prev_close + atr_val * self.atr_stop_mult,
                np.nan,
            ),
        )

        # Confidence: how strongly each is trending (EMA separation)
        gold_sep = ((gold_ema_fast.shift(1) - gold_ema_slow.shift(1)).abs() / gold_ema_slow.shift(1)).fillna(0)

        silver_sep_full = pd.Series(0.0, index=df.index)
        s_sep = ((silver_ema_fast - silver_ema_slow).abs() / silver_ema_slow.replace(0, np.nan)).fillna(0)
        silver_sep_full.iloc[-n_silver:] = s_sep.shift(1).values[-n_silver:]

        gdx_sep_full = pd.Series(0.0, index=df.index)
        g_sep = ((gdx_ema_fast - gdx_ema_slow).abs() / gdx_ema_slow.replace(0, np.nan)).fillna(0)
        gdx_sep_full.iloc[-n_gdx:] = g_sep.shift(1).values[-n_gdx:]

        # Normalize separations (0.02 = 2% EMA separation = max)
        avg_sep = (gold_sep / 0.02 + silver_sep_full / 0.03 + gdx_sep_full / 0.04) / 3
        df["confidence"] = avg_sep.clip(0, 1).fillna(0)
        df.loc[df["signal"] == 0, "confidence"] = 0.0

        last_signal_mask = df["signal"] != 0
        if last_signal_mask.any():
            last_idx = df.loc[last_signal_mask].index[-1]
            self.confidence = float(df.loc[last_idx, "confidence"])
            direction = "LONG" if df.loc[last_idx, "signal"] == 1 else "SHORT"
            self.reasons = [
                f"Multi-market momentum: {direction}",
                "Gold, silver, and GDX all aligned in same trend direction",
                f"Gold EMA separation: {gold_sep.loc[last_idx] * 100:.2f}%",
            ]

        df.drop(columns=["_atr"], inplace=True)
        return df


# ===========================================================================
# Registry: all cross-market strategies accessible by name
# ===========================================================================

CROSS_MARKET_STRATEGIES = {
    "dxy_divergence": DXYDivergenceStrategy,
    "silver_lead": SilverLeadStrategy,
    "vix_spike": VIXSpikeStrategy,
    "copper_gold_ratio": CopperGoldRatioStrategy,
    "oil_gold_inflation": OilGoldInflationStrategy,
    "btc_gold_flow": BTCGoldFlowStrategy,
    "real_yield": RealYieldStrategy,
    "asian_session_setup": AsianSessionSetupStrategy,
    "monday_gap": MondayGapStrategy,
    "round_number": RoundNumberStrategy,
    "india_seasonal": IndiaSeasonalStrategy,
    "multi_market_momentum": MultiMarketMomentumStrategy,
}


def create_cross_market_strategy(name: str, cfg: dict) -> BaseStrategy:
    """Factory function to create a cross-market strategy by name."""
    if name not in CROSS_MARKET_STRATEGIES:
        raise ValueError(
            f"Unknown cross-market strategy: {name}. "
            f"Available: {list(CROSS_MARKET_STRATEGIES.keys())}"
        )
    return CROSS_MARKET_STRATEGIES[name](cfg)


def create_all_cross_market_strategies(cfg: dict) -> list[BaseStrategy]:
    """Create all cross-market strategies with the given config."""
    strategies = []
    for name, cls in CROSS_MARKET_STRATEGIES.items():
        try:
            strategies.append(cls(cfg))
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to create cross-market strategy {name}: {e}"
            )
    return strategies
