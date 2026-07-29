"""
ICT (Inner Circle Trader) Concepts for Gold — Institutional Price Action

Adds smart money concepts on top of existing technical analysis:
1. NWOG (New Week Opening Gap) — weekly key levels
2. Kill Zones — time-based entry windows
3. Liquidity Sweeps — detecting stop hunts at equal highs/lows
4. FVG (Fair Value Gap) — imbalance candles
5. Order Blocks — institutional entry candles
6. OTE (Optimal Trade Entry) — 62-79% fib retracement zone
7. Market Structure Shift — break of structure detection

Reference: ICT / Michael J. Huddleston methodology
"""

import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from logs.logger import get_logger

log = get_logger("ict_concepts")


# ─── Kill Zone Time Windows (ET / UTC-4) ───
# These are the specific windows when institutional moves happen on gold

KILL_ZONES = {
    "asian":      {"start_utc": 0,  "end_utc": 4,   "name": "Asian Kill Zone"},
    "london":     {"start_utc": 7,  "end_utc": 10,  "name": "London Open Kill Zone"},
    "ny_open":    {"start_utc": 12, "end_utc": 14,  "name": "NY Open Kill Zone"},
    "ny_lunch":   {"start_utc": 16, "end_utc": 17,  "name": "NY Lunch / Silver Bullet"},
    "ny_pm":      {"start_utc": 18, "end_utc": 20,  "name": "NY PM Session"},
}

# ICT Macro times (specific reversal windows) in UTC
ICT_MACROS = [
    {"start_utc": (2, 33), "end_utc": (3, 0),   "name": "Asian Macro"},
    {"start_utc": (4, 3),  "end_utc": (4, 15),  "name": "London Pre-Open Macro"},
    {"start_utc": (14, 50), "end_utc": (15, 10), "name": "NY Lunch Macro"},
    {"start_utc": (19, 50), "end_utc": (20, 10), "name": "NY PM Macro"},
]


@dataclass
class ICTLevel:
    """A price level identified by ICT concepts."""
    level: float
    type: str           # nwog_high, nwog_low, liquidity_buy, liquidity_sell, fvg, ob, ote
    timeframe: str
    timestamp: str
    strength: float     # 0-1 how significant
    description: str
    touched: bool = False
    swept: bool = False


@dataclass
class ICTSignal:
    """An ICT-based trade signal."""
    direction: str      # LONG or SHORT
    confidence: float   # 0-1
    entry_zone: Tuple[float, float]  # (low, high) of entry zone
    stop: float
    target: float
    reasons: List[str]
    kill_zone: str      # which kill zone we're in
    concepts: List[str] # which ICT concepts align


class ICTAnalyzer:
    """
    Computes ICT concepts from candle data and generates levels/signals.

    Usage:
        ict = ICTAnalyzer()
        levels = ict.compute_all_levels(df_1h, df_4h, df_1d)
        signal = ict.generate_signal(df_5m, levels, current_price)
    """

    def __init__(self):
        self.nwog = None
        self.current_levels: List[ICTLevel] = []
        self.active_fvgs: List[dict] = []
        self.active_obs: List[dict] = []

    # ═══════════════════════════════════════════════════════════════
    # 1. NWOG — New Week Opening Gap
    # ═══════════════════════════════════════════════════════════════

    def compute_nwog(self, df_daily: pd.DataFrame) -> Optional[dict]:
        """
        Compute New Week Opening Gap from daily data.

        NWOG = gap between Friday's close and Sunday/Monday's open.
        These levels act as support/resistance for the entire week.
        """
        if df_daily is None or len(df_daily) < 5:
            return None

        # Find the last Friday and the following Monday
        df = df_daily.copy()
        df.index = pd.to_datetime(df.index)

        # Get day of week for each bar
        df["dow"] = df.index.dayofweek  # 0=Mon, 4=Fri

        fridays = df[df["dow"] == 4]
        mondays = df[df["dow"] == 0]

        if len(fridays) < 1 or len(mondays) < 1:
            return None

        last_friday = fridays.iloc[-1]
        last_monday = mondays.iloc[-1]

        # NWOG: gap between Friday close and Monday open
        friday_close = float(last_friday["close"])
        monday_open = float(last_monday["open"])

        nwog_high = max(friday_close, monday_open)
        nwog_low = min(friday_close, monday_open)
        nwog_mid = (nwog_high + nwog_low) / 2

        self.nwog = {
            "high": round(nwog_high, 2),
            "low": round(nwog_low, 2),
            "mid": round(nwog_mid, 2),
            "gap_size": round(abs(friday_close - monday_open), 2),
            "friday_close": round(friday_close, 2),
            "monday_open": round(monday_open, 2),
            "direction": "gap_up" if monday_open > friday_close else "gap_down",
        }

        self.current_levels.append(ICTLevel(
            level=nwog_high, type="nwog_high", timeframe="1W",
            timestamp=str(last_monday.name),
            strength=0.8, description=f"NWOG High ${nwog_high:.2f}"))
        self.current_levels.append(ICTLevel(
            level=nwog_low, type="nwog_low", timeframe="1W",
            timestamp=str(last_monday.name),
            strength=0.8, description=f"NWOG Low ${nwog_low:.2f}"))

        log.info("nwog_computed", high=nwog_high, low=nwog_low,
                 gap=self.nwog["gap_size"], direction=self.nwog["direction"])
        return self.nwog

    # ═══════════════════════════════════════════════════════════════
    # 2. Kill Zone Filter
    # ═══════════════════════════════════════════════════════════════

    def get_current_kill_zone(self) -> Optional[dict]:
        """Check if we're currently in an ICT kill zone."""
        now = datetime.utcnow()
        hour = now.hour

        for kz_id, kz in KILL_ZONES.items():
            if kz["start_utc"] <= hour < kz["end_utc"]:
                return {"id": kz_id, **kz, "active": True}

        return None

    def is_in_kill_zone(self) -> bool:
        return self.get_current_kill_zone() is not None

    def get_current_macro(self) -> Optional[dict]:
        """Check if we're in an ICT macro time window."""
        now = datetime.utcnow()
        for macro in ICT_MACROS:
            s_h, s_m = macro["start_utc"]
            e_h, e_m = macro["end_utc"]
            start = time(s_h, s_m)
            end = time(e_h, e_m)
            current = now.time()
            if start <= current <= end:
                return macro
        return None

    # ═══════════════════════════════════════════════════════════════
    # 3. Liquidity Sweep Detector
    # ═══════════════════════════════════════════════════════════════

    def detect_liquidity_levels(self, df: pd.DataFrame,
                                 lookback: int = 50) -> List[dict]:
        """
        Find equal highs (buyside liquidity) and equal lows (sellside liquidity).

        Equal highs/lows = where stops are resting.
        When price sweeps past these levels and reverses, it's a liquidity grab.
        """
        if df is None or len(df) < lookback:
            return []

        highs = df["high"].iloc[-lookback:].values
        lows = df["low"].iloc[-lookback:].values
        closes = df["close"].iloc[-lookback:].values

        levels = []
        tolerance = np.mean(highs - lows) * 0.3  # 30% of avg range

        # Find equal highs (buyside liquidity)
        for i in range(2, len(highs) - 1):
            for j in range(i + 2, min(i + 15, len(highs))):
                if abs(highs[i] - highs[j]) < tolerance:
                    level = max(highs[i], highs[j])
                    levels.append({
                        "type": "buyside_liquidity",
                        "level": round(float(level), 2),
                        "touches": 2,
                        "description": f"Equal highs (buyside liquidity) at ${level:.2f}",
                    })
                    break

        # Find equal lows (sellside liquidity)
        for i in range(2, len(lows) - 1):
            for j in range(i + 2, min(i + 15, len(lows))):
                if abs(lows[i] - lows[j]) < tolerance:
                    level = min(lows[i], lows[j])
                    levels.append({
                        "type": "sellside_liquidity",
                        "level": round(float(level), 2),
                        "touches": 2,
                        "description": f"Equal lows (sellside liquidity) at ${level:.2f}",
                    })
                    break

        # Deduplicate close levels
        unique = []
        for l in sorted(levels, key=lambda x: x["level"]):
            if not unique or abs(l["level"] - unique[-1]["level"]) > tolerance:
                unique.append(l)

        return unique

    def detect_liquidity_sweep(self, df: pd.DataFrame,
                                liquidity_levels: List[dict]) -> List[dict]:
        """
        Detect if price just swept a liquidity level and reversed.

        A sweep = price pokes through the level then closes back inside.
        This is the institutional stop hunt — smart money entry signal.
        """
        if df is None or len(df) < 3:
            return []

        sweeps = []
        last_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]

        for ll in liquidity_levels:
            level = ll["level"]

            if ll["type"] == "buyside_liquidity":
                # Sweep above = wick above level but close below
                if (float(last_bar["high"]) > level and
                    float(last_bar["close"]) < level and
                    float(prev_bar["high"]) < level):
                    sweeps.append({
                        "type": "buyside_sweep",
                        "level": level,
                        "signal": "SHORT",
                        "description": f"Buyside liquidity swept at ${level:.2f} — SHORT signal",
                        "wick_size": round(float(last_bar["high"]) - level, 2),
                    })

            elif ll["type"] == "sellside_liquidity":
                # Sweep below = wick below level but close above
                if (float(last_bar["low"]) < level and
                    float(last_bar["close"]) > level and
                    float(prev_bar["low"]) > level):
                    sweeps.append({
                        "type": "sellside_sweep",
                        "level": level,
                        "signal": "LONG",
                        "description": f"Sellside liquidity swept at ${level:.2f} — LONG signal",
                        "wick_size": round(level - float(last_bar["low"]), 2),
                    })

        return sweeps

    # ═══════════════════════════════════════════════════════════════
    # 4. FVG (Fair Value Gap) — Imbalance Detection
    # ═══════════════════════════════════════════════════════════════

    def detect_fvgs(self, df: pd.DataFrame, lookback: int = 30) -> List[dict]:
        """
        Find Fair Value Gaps — 3-candle patterns where middle candle
        leaves a gap between candle 1's low and candle 3's high (bullish)
        or candle 1's high and candle 3's low (bearish).

        FVGs are magnets — price tends to return to fill them.
        """
        if df is None or len(df) < lookback:
            return []

        fvgs = []
        data = df.iloc[-lookback:]

        for i in range(2, len(data)):
            c1 = data.iloc[i - 2]  # first candle
            c2 = data.iloc[i - 1]  # middle (displacement) candle
            c3 = data.iloc[i]      # third candle

            # Bullish FVG: gap between c1 high and c3 low
            # (c2 moved so fast upward that there's no overlap)
            if float(c3["low"]) > float(c1["high"]):
                gap_size = float(c3["low"]) - float(c1["high"])
                fvgs.append({
                    "type": "bullish_fvg",
                    "top": round(float(c3["low"]), 2),
                    "bottom": round(float(c1["high"]), 2),
                    "mid": round((float(c3["low"]) + float(c1["high"])) / 2, 2),
                    "gap_size": round(gap_size, 2),
                    "index": i,
                    "filled": False,
                    "description": f"Bullish FVG ${float(c1['high']):.2f}-${float(c3['low']):.2f}",
                })

            # Bearish FVG: gap between c3 high and c1 low
            if float(c3["high"]) < float(c1["low"]):
                gap_size = float(c1["low"]) - float(c3["high"])
                fvgs.append({
                    "type": "bearish_fvg",
                    "top": round(float(c1["low"]), 2),
                    "bottom": round(float(c3["high"]), 2),
                    "mid": round((float(c1["low"]) + float(c3["high"])) / 2, 2),
                    "gap_size": round(gap_size, 2),
                    "index": i,
                    "filled": False,
                    "description": f"Bearish FVG ${float(c3['high']):.2f}-${float(c1['low']):.2f}",
                })

        # Check which FVGs are still unfilled (price hasn't traded through them)
        current_price = float(df["close"].iloc[-1])
        unfilled = []
        for fvg in fvgs:
            if fvg["type"] == "bullish_fvg" and current_price > fvg["top"]:
                fvg["filled"] = False  # still unfilled below
                unfilled.append(fvg)
            elif fvg["type"] == "bearish_fvg" and current_price < fvg["bottom"]:
                fvg["filled"] = False  # still unfilled above
                unfilled.append(fvg)

        self.active_fvgs = unfilled[-10:]  # keep closest 10
        return unfilled

    # ═══════════════════════════════════════════════════════════════
    # 5. Order Blocks — Institutional Entry Points
    # ═══════════════════════════════════════════════════════════════

    def detect_order_blocks(self, df: pd.DataFrame,
                             lookback: int = 30) -> List[dict]:
        """
        Find Order Blocks — the last opposite-direction candle before
        a strong displacement move.

        Bullish OB: last bearish candle before a strong bullish move
        Bearish OB: last bullish candle before a strong bearish move

        These mark where institutions placed their orders.
        """
        if df is None or len(df) < lookback:
            return []

        obs = []
        data = df.iloc[-lookback:]
        avg_range = (data["high"] - data["low"]).mean()
        displacement_threshold = avg_range * 2.0  # 2x average range = displacement

        for i in range(2, len(data) - 1):
            c_prev = data.iloc[i - 1]
            c_curr = data.iloc[i]
            c_next = data.iloc[i + 1] if i + 1 < len(data) else None

            curr_range = float(c_curr["high"]) - float(c_curr["low"])
            curr_body = abs(float(c_curr["close"]) - float(c_curr["open"]))

            # Strong bullish displacement
            if (curr_body > displacement_threshold and
                float(c_curr["close"]) > float(c_curr["open"])):
                # The candle before is the bullish OB
                if float(c_prev["close"]) < float(c_prev["open"]):  # prev is bearish
                    obs.append({
                        "type": "bullish_ob",
                        "top": round(float(c_prev["open"]), 2),
                        "bottom": round(float(c_prev["close"]), 2),
                        "mid": round((float(c_prev["open"]) + float(c_prev["close"])) / 2, 2),
                        "displacement": round(curr_body, 2),
                        "description": f"Bullish OB ${float(c_prev['close']):.2f}-${float(c_prev['open']):.2f}",
                    })

            # Strong bearish displacement
            if (curr_body > displacement_threshold and
                float(c_curr["close"]) < float(c_curr["open"])):
                if float(c_prev["close"]) > float(c_prev["open"]):  # prev is bullish
                    obs.append({
                        "type": "bearish_ob",
                        "top": round(float(c_prev["close"]), 2),
                        "bottom": round(float(c_prev["open"]), 2),
                        "mid": round((float(c_prev["close"]) + float(c_prev["open"])) / 2, 2),
                        "displacement": round(curr_body, 2),
                        "description": f"Bearish OB ${float(c_prev['open']):.2f}-${float(c_prev['close']):.2f}",
                    })

        self.active_obs = obs[-10:]
        return obs

    # ═══════════════════════════════════════════════════════════════
    # 6. OTE (Optimal Trade Entry) — Fibonacci Zone
    # ═══════════════════════════════════════════════════════════════

    def compute_ote_zone(self, df: pd.DataFrame,
                          lookback: int = 50) -> Optional[dict]:
        """
        Find the OTE zone: 62-79% fibonacci retracement of the last
        significant swing.

        This is WHERE to enter — the sweet spot between too early and too late.
        """
        if df is None or len(df) < lookback:
            return None

        data = df.iloc[-lookback:]
        highs = data["high"].values
        lows = data["low"].values

        # Find the last significant swing
        swing_high_idx = np.argmax(highs)
        swing_low_idx = np.argmin(lows)
        swing_high = float(highs[swing_high_idx])
        swing_low = float(lows[swing_low_idx])

        swing_range = swing_high - swing_low
        if swing_range < 1.0:
            return None

        current_price = float(df["close"].iloc[-1])

        # Determine if we're in an upswing retracement or downswing retracement
        if swing_high_idx > swing_low_idx:
            # Upswing completed — looking for pullback to buy
            fib_62 = swing_high - swing_range * 0.62
            fib_705 = swing_high - swing_range * 0.705
            fib_79 = swing_high - swing_range * 0.79
            direction = "LONG"
            ote = {
                "direction": "LONG",
                "swing_high": round(swing_high, 2),
                "swing_low": round(swing_low, 2),
                "ote_top": round(fib_62, 2),    # 62% retracement
                "ote_mid": round(fib_705, 2),   # 70.5% sweet spot
                "ote_bottom": round(fib_79, 2), # 79% retracement
                "in_zone": fib_79 <= current_price <= fib_62,
                "description": f"Bullish OTE zone ${fib_79:.2f}-${fib_62:.2f} (buy zone)",
            }
        else:
            # Downswing completed — looking for rally to sell
            fib_62 = swing_low + swing_range * 0.62
            fib_705 = swing_low + swing_range * 0.705
            fib_79 = swing_low + swing_range * 0.79
            direction = "SHORT"
            ote = {
                "direction": "SHORT",
                "swing_high": round(swing_high, 2),
                "swing_low": round(swing_low, 2),
                "ote_top": round(fib_79, 2),    # 79% retracement
                "ote_mid": round(fib_705, 2),   # 70.5% sweet spot
                "ote_bottom": round(fib_62, 2), # 62% retracement
                "in_zone": fib_62 <= current_price <= fib_79,
                "description": f"Bearish OTE zone ${fib_62:.2f}-${fib_79:.2f} (sell zone)",
            }

        return ote

    # ═══════════════════════════════════════════════════════════════
    # 7. Market Structure Shift (MSS) — Trend Change
    # ═══════════════════════════════════════════════════════════════

    def detect_market_structure_shift(self, df: pd.DataFrame,
                                       lookback: int = 20) -> Optional[dict]:
        """
        Detect if market structure has shifted — break of the most
        recent swing high (bullish MSS) or swing low (bearish MSS).
        """
        if df is None or len(df) < lookback:
            return None

        data = df.iloc[-lookback:]

        # Find swing points (simple: local max/min with 2-bar confirmation)
        swing_highs = []
        swing_lows = []

        for i in range(2, len(data) - 2):
            h = float(data["high"].iloc[i])
            if h > float(data["high"].iloc[i-1]) and h > float(data["high"].iloc[i-2]) and \
               h > float(data["high"].iloc[i+1]) and h > float(data["high"].iloc[i+2]):
                swing_highs.append((i, h))

            l = float(data["low"].iloc[i])
            if l < float(data["low"].iloc[i-1]) and l < float(data["low"].iloc[i-2]) and \
               l < float(data["low"].iloc[i+1]) and l < float(data["low"].iloc[i+2]):
                swing_lows.append((i, l))

        if not swing_highs or not swing_lows:
            return None

        last_high = swing_highs[-1]
        last_low = swing_lows[-1]
        current = float(data["close"].iloc[-1])
        prev = float(data["close"].iloc[-2])

        # Bullish MSS: price breaks above the last swing high
        if prev < last_high[1] and current > last_high[1]:
            return {
                "type": "bullish_mss",
                "break_level": round(last_high[1], 2),
                "signal": "LONG",
                "description": f"Bullish MSS — broke above swing high ${last_high[1]:.2f}",
            }

        # Bearish MSS: price breaks below the last swing low
        if prev > last_low[1] and current < last_low[1]:
            return {
                "type": "bearish_mss",
                "break_level": round(last_low[1], 2),
                "signal": "SHORT",
                "description": f"Bearish MSS — broke below swing low ${last_low[1]:.2f}",
            }

        return None

    # ═══════════════════════════════════════════════════════════════
    # Master: Compute All + Generate Confluence Signal
    # ═══════════════════════════════════════════════════════════════

    def analyze(self, df_5m: pd.DataFrame, df_15m: pd.DataFrame,
                df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                df_1d: pd.DataFrame) -> dict:
        """
        Run all ICT analysis and return a comprehensive report.
        """
        self.current_levels = []
        result = {
            "kill_zone": self.get_current_kill_zone(),
            "macro": self.get_current_macro(),
            "in_kill_zone": self.is_in_kill_zone(),
        }

        # NWOG from daily
        result["nwog"] = self.compute_nwog(df_1d)

        # OTE from 1H and 4H
        result["ote_1h"] = self.compute_ote_zone(df_1h, lookback=30)
        result["ote_4h"] = self.compute_ote_zone(df_4h, lookback=20)

        # FVGs from multiple timeframes
        result["fvg_15m"] = self.detect_fvgs(df_15m, lookback=30)
        result["fvg_1h"] = self.detect_fvgs(df_1h, lookback=20)

        # Order blocks from 1H
        result["order_blocks"] = self.detect_order_blocks(df_1h, lookback=30)

        # Liquidity levels from 1H
        liquidity = self.detect_liquidity_levels(df_1h, lookback=50)
        result["liquidity_levels"] = liquidity

        # Liquidity sweeps from 5M (most granular)
        result["sweeps"] = self.detect_liquidity_sweep(df_5m, liquidity)

        # Market structure shift from 15M
        result["mss"] = self.detect_market_structure_shift(df_15m)

        # Count confluences for a signal
        result["confluence"] = self._count_confluence(result)

        log.info("ict_analysis_complete",
                 in_kz=result["in_kill_zone"],
                 nwog=result["nwog"] is not None,
                 fvgs=len(result.get("fvg_1h", [])),
                 obs=len(result.get("order_blocks", [])),
                 sweeps=len(result.get("sweeps", [])),
                 confluence=result["confluence"].get("score", 0))

        return result

    def _count_confluence(self, analysis: dict) -> dict:
        """Count how many ICT concepts align for a directional signal."""
        long_score = 0
        short_score = 0
        reasons_long = []
        reasons_short = []

        # Kill zone bonus
        if analysis.get("in_kill_zone"):
            kz = analysis["kill_zone"]
            long_score += 1
            short_score += 1  # kill zone is directionally neutral

        # NWOG position
        if analysis.get("nwog"):
            nwog = analysis["nwog"]
            # Price above NWOG mid = bullish bias, below = bearish
            # Can't check current price here, will be done by the brain

        # OTE zones
        for key in ("ote_1h", "ote_4h"):
            ote = analysis.get(key)
            if ote and ote.get("in_zone"):
                if ote["direction"] == "LONG":
                    long_score += 2
                    reasons_long.append(f"In bullish OTE zone ({key})")
                else:
                    short_score += 2
                    reasons_short.append(f"In bearish OTE zone ({key})")

        # Liquidity sweeps
        for sweep in analysis.get("sweeps", []):
            if sweep["signal"] == "LONG":
                long_score += 3
                reasons_long.append(sweep["description"])
            else:
                short_score += 3
                reasons_short.append(sweep["description"])

        # FVGs below = bullish magnet, above = bearish magnet
        for fvg in analysis.get("fvg_1h", []):
            if fvg["type"] == "bullish_fvg":
                long_score += 1
                reasons_long.append(f"Unfilled bullish FVG at ${fvg['bottom']:.2f}")
            else:
                short_score += 1
                reasons_short.append(f"Unfilled bearish FVG at ${fvg['top']:.2f}")

        # Order blocks
        for ob in analysis.get("order_blocks", [])[-3:]:
            if ob["type"] == "bullish_ob":
                long_score += 1
                reasons_long.append(ob["description"])
            else:
                short_score += 1
                reasons_short.append(ob["description"])

        # Market structure shift
        mss = analysis.get("mss")
        if mss:
            if mss["signal"] == "LONG":
                long_score += 2
                reasons_long.append(mss["description"])
            else:
                short_score += 2
                reasons_short.append(mss["description"])

        if long_score > short_score:
            direction = "LONG"
            score = long_score
            reasons = reasons_long
        elif short_score > long_score:
            direction = "SHORT"
            score = short_score
            reasons = reasons_short
        else:
            direction = "NEUTRAL"
            score = 0
            reasons = []

        return {
            "direction": direction,
            "score": score,
            "long_score": long_score,
            "short_score": short_score,
            "reasons": reasons,
            "concepts_aligned": len(reasons),
        }
