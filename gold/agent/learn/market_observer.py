"""
Multi-Timeframe Live Market Observer — Watches gold on ALL charts simultaneously.
Fetches 5M, 15M, 1H, 4H, and 1D data, analyzes each, and cross-references.

What it tracks per timeframe:
- Price action patterns (higher highs/lows, ranges, breakouts)
- Volatility regime, BB squeeze
- Momentum (RSI, MACD, Stochastic)
- Volume anomalies and OBV trend
- Support/resistance and key levels

Cross-timeframe analysis:
- Confluence: how many TFs agree on direction
- Divergence: when small TFs disagree with large TFs
- Multi-TF squeeze detection
- Trend alignment score
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data.indicators import atr, ema, sma, rsi, macd, bollinger_bands, adx, stochastic, obv
from logs.logger import get_logger

log = get_logger("learn.observer")

# All timeframes to learn from
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

TF_LABELS = {
    "5m": "5-Minute",
    "15m": "15-Minute",
    "1h": "1-Hour",
    "4h": "4-Hour",
    "1d": "Daily",
}


def _safe_float(val, decimals=2):
    """Convert to float safely, handling NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), decimals)


class MarketObserver:
    """Observes live XAUUSD market data across ALL timeframes."""

    def __init__(self, data_dir: str = "./learn/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.observations = []
        self.daily_stats = []
        self.pattern_counts = {
            "trend_continuation": 0,
            "trend_reversal": 0,
            "range_bound": 0,
            "breakout_success": 0,
            "breakout_failure": 0,
            "london_open_spike": 0,
            "ny_open_spike": 0,
            "asian_range_respect": 0,
            "round_number_reaction": 0,
            "volatility_expansion": 0,
            "volatility_contraction": 0,
            "volume_spike_up": 0,
            "volume_spike_down": 0,
            "multi_tf_confluence_bull": 0,
            "multi_tf_confluence_bear": 0,
            "tf_divergence": 0,
            "multi_tf_squeeze": 0,
        }
        self.support_resistance_levels = []
        self.session_profiles = {"asian": [], "london": [], "new_york": []}
        self.tf_history = {tf: [] for tf in TIMEFRAMES}

        self._load_knowledge()

    def fetch_all_timeframes(self, symbol="XAUUSD", exchange="OANDA"):
        """Fetch data for every timeframe from TradingView."""
        from data.provider import fetch_tradingview

        data = {}
        for tf in TIMEFRAMES:
            try:
                df = fetch_tradingview(
                    symbol=symbol,
                    exchange=exchange,
                    start="2025-01-01",
                    end="2030-12-31",
                    interval=tf,
                )
                data[tf] = df
                log.info("tf_fetched", timeframe=tf, bars=len(df))
            except Exception as e:
                log.warning("tf_fetch_failed", timeframe=tf, error=str(e))
        return data

    def observe_all(self, all_data: dict) -> dict:
        """Analyze ALL timeframes and cross-reference them."""
        result = {
            "timestamp": str(datetime.now()),
            "timeframes": {},
            "cross_tf": {},
        }

        # Analyze each timeframe individually
        for tf, df in all_data.items():
            if df is not None and len(df) >= 30:
                result["timeframes"][tf] = self._analyze_single_tf(df, tf)

        # Cross-timeframe analysis
        if len(result["timeframes"]) >= 2:
            result["cross_tf"] = self._cross_timeframe_analysis(result["timeframes"])

        # Store
        latest_price = None
        for tf in ["5m", "15m", "1h", "4h", "1d"]:
            if tf in all_data and all_data[tf] is not None and len(all_data[tf]) > 0:
                latest_price = float(all_data[tf]["close"].iloc[-1])
                break

        obs_entry = {
            "timestamp": result["timestamp"],
            "latest_price": latest_price,
            "timeframes_analyzed": list(result["timeframes"].keys()),
            "cross_tf": result["cross_tf"],
            "tf_summaries": {
                tf: {
                    "regime": data.get("regime", {}).get("trend"),
                    "rsi": data.get("momentum", {}).get("rsi"),
                    "bb_squeeze": data.get("volatility", {}).get("bb_squeeze"),
                    "macd_dir": data.get("momentum", {}).get("macd_direction"),
                }
                for tf, data in result["timeframes"].items()
            },
        }
        self.observations.append(obs_entry)

        return result

    def _analyze_single_tf(self, df: pd.DataFrame, tf_name: str) -> dict:
        """Full analysis for a single timeframe."""
        obs = {}
        obs["timeframe"] = tf_name
        obs["bars"] = len(df)
        obs["data_range"] = f"{df.index[0]} → {df.index[-1]}"

        obs["regime"] = self._analyze_regime(df, tf_name)
        obs["volatility"] = self._analyze_volatility(df)
        obs["momentum"] = self._analyze_momentum(df)
        obs["volume"] = self._analyze_volume(df)
        obs["patterns"] = self._detect_patterns(df)
        obs["key_levels"] = self._find_key_levels(df)
        obs["price_action"] = self._price_summary(df, tf_name)

        # Session analysis only makes sense for intraday TFs
        if tf_name in ("5m", "15m", "1h"):
            obs["sessions"] = self._analyze_sessions(df, tf_name)

        return obs

    def _analyze_regime(self, df: pd.DataFrame, tf_name: str) -> dict:
        close = df["close"]
        period_50 = min(50, len(df) - 1)
        period_200 = min(200, len(df) - 1)
        if period_50 < 5 or period_200 < 5:
            return {"trend": "unknown", "strength": "unknown", "adx": 0}

        ema50 = ema(close, period_50)
        ema200 = ema(close, period_200)
        adx_period = min(14, len(df) // 3)
        adx_vals = adx(df, max(adx_period, 3)) if len(df) > 30 else pd.Series(0, index=df.index)

        current_price = close.iloc[-1]
        current_adx = adx_vals.iloc[-1] if not pd.isna(adx_vals.iloc[-1]) else 0

        if current_price > ema50.iloc[-1] and current_price > ema200.iloc[-1]:
            trend = "strong_uptrend" if current_adx > 25 else "weak_uptrend"
        elif current_price < ema50.iloc[-1] and current_price < ema200.iloc[-1]:
            trend = "strong_downtrend" if current_adx > 25 else "weak_downtrend"
        else:
            trend = "ranging"

        if current_adx > 40:
            strength = "very_strong"
        elif current_adx > 25:
            strength = "moderate"
        elif current_adx > 15:
            strength = "weak"
        else:
            strength = "no_trend"

        # Track pattern
        if "uptrend" in trend or "downtrend" in trend:
            self.pattern_counts["trend_continuation"] += 1
        else:
            self.pattern_counts["range_bound"] += 1

        return {
            "trend": trend,
            "strength": strength,
            "adx": _safe_float(current_adx),
            "price_vs_ema50": "above" if current_price > ema50.iloc[-1] else "below",
            "ema_cross": "golden" if ema50.iloc[-1] > ema200.iloc[-1] else "death",
        }

    def _analyze_volatility(self, df: pd.DataFrame) -> dict:
        if len(df) < 20:
            return {"state": "insufficient_data", "bb_squeeze": False}

        atr_val = atr(df, min(14, len(df) // 2))
        current_atr = atr_val.iloc[-1]
        avg_atr = atr_val.rolling(min(50, len(atr_val)), min_periods=5).mean().iloc[-1]

        upper, mid, lower = bollinger_bands(df["close"], 20, 2.0)
        bb_width = ((upper - lower) / mid).iloc[-1] if not pd.isna(mid.iloc[-1]) else 0
        avg_bb = ((upper - lower) / mid).rolling(min(50, len(df)), min_periods=5).mean().iloc[-1]

        if pd.isna(current_atr) or pd.isna(avg_atr) or avg_atr == 0:
            return {"state": "unknown", "bb_squeeze": False}

        ratio = current_atr / avg_atr

        if ratio > 1.5:
            state = "high_volatility"
            self.pattern_counts["volatility_expansion"] += 1
        elif ratio < 0.7:
            state = "low_volatility_squeeze"
            self.pattern_counts["volatility_contraction"] += 1
        else:
            state = "normal"

        squeeze = bool(bb_width < avg_bb * 0.7) if not pd.isna(avg_bb) else False

        return {
            "state": state,
            "current_atr": _safe_float(current_atr),
            "avg_atr": _safe_float(avg_atr),
            "atr_ratio": _safe_float(ratio),
            "bb_width": _safe_float(bb_width, 4),
            "bb_squeeze": squeeze,
        }

    def _analyze_momentum(self, df: pd.DataFrame) -> dict:
        if len(df) < 30:
            return {"rsi": None, "rsi_state": "unknown", "macd_histogram": None, "macd_direction": "unknown", "stochastic_k": None}

        rsi_vals = rsi(df["close"], 14)
        macd_line, signal_line, hist = macd(df["close"], 12, 26, 9)
        k, d = stochastic(df, 14, 3)

        current_rsi = rsi_vals.iloc[-1]
        current_hist = hist.iloc[-1]
        prev_hist = hist.iloc[-2] if len(hist) >= 2 else 0
        current_k = k.iloc[-1]

        if pd.isna(current_rsi):
            rsi_state = "unknown"
        elif current_rsi > 70:
            rsi_state = "overbought"
        elif current_rsi < 30:
            rsi_state = "oversold"
        else:
            rsi_state = "neutral"

        macd_dir = "increasing" if (not pd.isna(current_hist) and current_hist > prev_hist) else "decreasing"

        return {
            "rsi": _safe_float(current_rsi),
            "rsi_state": rsi_state,
            "macd_histogram": _safe_float(current_hist, 4),
            "macd_direction": macd_dir,
            "stochastic_k": _safe_float(current_k),
        }

    def _analyze_volume(self, df: pd.DataFrame) -> dict:
        vol = df["volume"]
        if vol.sum() == 0 or len(df) < 20:
            return {"state": "no_volume_data", "obv_trend": "unknown"}

        avg_vol = vol.rolling(20, min_periods=5).mean()
        current_vol = vol.iloc[-1]
        avg = avg_vol.iloc[-1]
        ratio = current_vol / avg if avg > 0 else 1

        if ratio > 2:
            state = "spike"
            if df["close"].iloc[-1] > df["open"].iloc[-1]:
                self.pattern_counts["volume_spike_up"] += 1
            else:
                self.pattern_counts["volume_spike_down"] += 1
        elif ratio > 1.5:
            state = "above_average"
        elif ratio < 0.5:
            state = "very_low"
        else:
            state = "normal"

        obv_vals = obv(df)
        obv_ema_val = ema(obv_vals, 20)
        obv_trend = "bullish" if obv_vals.iloc[-1] > obv_ema_val.iloc[-1] else "bearish"

        return {
            "current": _safe_float(current_vol, 0),
            "average": _safe_float(avg, 0),
            "ratio": _safe_float(ratio),
            "state": state,
            "obv_trend": obv_trend,
        }

    def _detect_patterns(self, df: pd.DataFrame) -> list:
        patterns = []
        if len(df) < 10:
            return patterns

        recent = df.tail(min(20, len(df)))
        swing_highs, swing_lows = [], []
        for i in range(2, len(recent) - 2):
            if recent["high"].iloc[i] > recent["high"].iloc[i-1] and recent["high"].iloc[i] > recent["high"].iloc[i+1]:
                swing_highs.append(recent["high"].iloc[i])
            if recent["low"].iloc[i] < recent["low"].iloc[i-1] and recent["low"].iloc[i] < recent["low"].iloc[i+1]:
                swing_lows.append(recent["low"].iloc[i])

        if len(swing_highs) >= 2:
            patterns.append("higher_highs" if swing_highs[-1] > swing_highs[-2] else "lower_highs")
        if len(swing_lows) >= 2:
            patterns.append("higher_lows" if swing_lows[-1] > swing_lows[-2] else "lower_lows")

        last = df.iloc[-1]
        body = abs(last["close"] - last["open"])
        full_range = last["high"] - last["low"]
        if full_range > 0 and body / full_range < 0.1:
            patterns.append("doji")

        if len(df) >= 2:
            prev = df.iloc[-2]
            if (last["close"] > last["open"] and prev["close"] < prev["open"]
                    and last["close"] > prev["open"] and last["open"] < prev["close"]):
                patterns.append("bullish_engulfing")
            if (last["close"] < last["open"] and prev["close"] > prev["open"]
                    and last["close"] < prev["open"] and last["open"] > prev["close"]):
                patterns.append("bearish_engulfing")

        return patterns

    def _find_key_levels(self, df: pd.DataFrame) -> dict:
        current = df["close"].iloc[-1]
        base = int(current / 100) * 100
        round_levels = [base + i * 50 for i in range(-2, 4)]
        nearest_round = min(round_levels, key=lambda x: abs(x - current))

        lookback = min(50, len(df))
        recent_high = float(df["high"].tail(lookback).max())
        recent_low = float(df["low"].tail(lookback).min())

        atr_val = atr(df, min(14, len(df) // 2))
        if len(atr_val.dropna()) > 0 and abs(current - nearest_round) < atr_val.dropna().iloc[-1] * 0.5:
            self.pattern_counts["round_number_reaction"] += 1

        self.support_resistance_levels = [
            {"level": recent_high, "type": "resistance"},
            {"level": recent_low, "type": "support"},
            {"level": float(nearest_round), "type": "round_number"},
        ]

        return {
            "nearest_round": nearest_round,
            "distance_to_round": _safe_float(current - nearest_round),
            "recent_high": _safe_float(recent_high),
            "recent_low": _safe_float(recent_low),
            "current_price": _safe_float(current),
        }

    def _price_summary(self, df: pd.DataFrame, tf_name: str) -> dict:
        last = df.iloc[-1]
        # Use appropriate lookback per timeframe
        lookback_map = {"5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
        lb = min(lookback_map.get(tf_name, 24), len(df))
        recent = df.tail(lb)

        return {
            "last_close": _safe_float(last["close"]),
            "last_bar_change_pct": _safe_float((last["close"] - last["open"]) / last["open"] * 100, 4),
            "period_high": _safe_float(recent["high"].max()),
            "period_low": _safe_float(recent["low"].min()),
            "period_range": _safe_float(recent["high"].max() - recent["low"].min()),
        }

    def _analyze_sessions(self, df: pd.DataFrame, tf_name: str) -> dict:
        # Adjusted lookback based on timeframe
        lookback_map = {"5m": 576, "15m": 192, "1h": 100}
        lb = min(lookback_map.get(tf_name, 100), len(df))
        recent = df.tail(lb)

        sessions = {}
        for name, start, end in [("asian", 0, 8), ("london", 8, 13), ("new_york", 13, 22)]:
            mask = (recent.index.hour >= start) & (recent.index.hour < end)
            session_data = recent[mask]
            if len(session_data) > 0:
                avg_range = (session_data["high"] - session_data["low"]).mean()
                avg_volume = session_data["volume"].mean()
                sessions[name] = {
                    "avg_range": _safe_float(avg_range),
                    "avg_volume": _safe_float(avg_volume, 0),
                    "bars": len(session_data),
                }
                if name in self.session_profiles:
                    self.session_profiles[name].append(float(avg_range))
        return sessions

    def _cross_timeframe_analysis(self, tf_data: dict) -> dict:
        """The real magic — cross-reference ALL timeframes."""
        directions = {}
        squeezes = {}
        rsi_vals = {}
        macd_dirs = {}

        for tf, data in tf_data.items():
            regime = data.get("regime", {})
            trend = regime.get("trend", "unknown")
            if "uptrend" in trend:
                directions[tf] = 1
            elif "downtrend" in trend:
                directions[tf] = -1
            else:
                directions[tf] = 0

            squeezes[tf] = data.get("volatility", {}).get("bb_squeeze", False)
            rsi_vals[tf] = data.get("momentum", {}).get("rsi")
            macd_dirs[tf] = data.get("momentum", {}).get("macd_direction", "unknown")

        # Confluence score: how many TFs agree
        bull_count = sum(1 for d in directions.values() if d == 1)
        bear_count = sum(1 for d in directions.values() if d == -1)
        total = len(directions)

        if bull_count >= 3:
            confluence = "STRONG_BULLISH"
            self.pattern_counts["multi_tf_confluence_bull"] += 1
        elif bear_count >= 3:
            confluence = "STRONG_BEARISH"
            self.pattern_counts["multi_tf_confluence_bear"] += 1
        elif bull_count >= 2:
            confluence = "LEAN_BULLISH"
        elif bear_count >= 2:
            confluence = "LEAN_BEARISH"
        else:
            confluence = "MIXED"

        # Divergence: small TFs vs large TFs disagree
        small_tf_dir = directions.get("5m", directions.get("15m", 0))
        large_tf_dir = directions.get("1d", directions.get("4h", 0))
        has_divergence = (small_tf_dir != 0 and large_tf_dir != 0 and small_tf_dir != large_tf_dir)
        if has_divergence:
            self.pattern_counts["tf_divergence"] += 1

        # Multi-TF squeeze: squeeze on 2+ timeframes
        squeeze_count = sum(1 for s in squeezes.values() if s)
        if squeeze_count >= 2:
            self.pattern_counts["multi_tf_squeeze"] += 1

        # Trend alignment score (-1 to +1)
        if total > 0:
            alignment_score = sum(directions.values()) / total
        else:
            alignment_score = 0

        # RSI consensus
        valid_rsi = [v for v in rsi_vals.values() if v is not None]
        avg_rsi = np.mean(valid_rsi) if valid_rsi else None

        # MACD consensus
        macd_bull = sum(1 for d in macd_dirs.values() if d == "increasing")
        macd_bear = sum(1 for d in macd_dirs.values() if d == "decreasing")

        return {
            "confluence": confluence,
            "alignment_score": _safe_float(alignment_score, 3),
            "bull_tf_count": bull_count,
            "bear_tf_count": bear_count,
            "neutral_tf_count": total - bull_count - bear_count,
            "directions": directions,
            "tf_divergence": has_divergence,
            "divergence_detail": f"{small_tf_dir} (short TF) vs {large_tf_dir} (long TF)" if has_divergence else "aligned",
            "squeeze_count": squeeze_count,
            "squeezes": squeezes,
            "avg_rsi_all_tf": _safe_float(avg_rsi) if avg_rsi else None,
            "rsi_by_tf": rsi_vals,
            "macd_bull_count": macd_bull,
            "macd_bear_count": macd_bear,
        }

    def generate_multi_tf_report(self, all_data: dict) -> str:
        """Generate the full multi-timeframe learning report."""
        result = self.observe_all(all_data)

        report = []
        report.append("=" * 75)
        report.append("GOLD MARKET MULTI-TIMEFRAME LEARNING REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 75)

        # Cross-TF summary first (most important)
        ct = result.get("cross_tf", {})
        if ct:
            report.append(f"\n{'─' * 75}")
            report.append(f"  MULTI-TIMEFRAME CONSENSUS: {ct.get('confluence', 'N/A')}")
            report.append(f"  Alignment Score: {ct.get('alignment_score', 0)} (-1=full bear, +1=full bull)")
            report.append(f"  Bull TFs: {ct.get('bull_tf_count', 0)}  |  Bear TFs: {ct.get('bear_tf_count', 0)}  |  Neutral: {ct.get('neutral_tf_count', 0)}")
            report.append(f"  TF Divergence: {'YES — ' + ct.get('divergence_detail', '') if ct.get('tf_divergence') else 'No (aligned)'}")
            report.append(f"  BB Squeeze on {ct.get('squeeze_count', 0)} timeframe(s): {ct.get('squeezes', {})}")
            report.append(f"  Avg RSI (all TFs): {ct.get('avg_rsi_all_tf', 'N/A')}")
            report.append(f"  MACD Bull/Bear: {ct.get('macd_bull_count', 0)} increasing / {ct.get('macd_bear_count', 0)} decreasing")
            report.append(f"{'─' * 75}")

        # Per-timeframe breakdown
        for tf in TIMEFRAMES:
            if tf not in result["timeframes"]:
                report.append(f"\n  [{TF_LABELS.get(tf, tf)}] — SKIPPED (fetch failed)")
                continue

            data = result["timeframes"][tf]
            r = data.get("regime", {})
            v = data.get("volatility", {})
            m = data.get("momentum", {})
            vol = data.get("volume", {})
            pa = data.get("price_action", {})
            pats = data.get("patterns", [])

            report.append(f"\n  [{TF_LABELS.get(tf, tf)}]  bars={data.get('bars', 0)}  |  {data.get('data_range', '')}")
            report.append(f"    Regime:    {r.get('trend', '?')} ({r.get('strength', '?')}, ADX={r.get('adx', '?')}), EMA: {r.get('ema_cross', '?')} cross")
            report.append(f"    Momentum:  RSI={m.get('rsi', '?')} ({m.get('rsi_state', '?')})  MACD={m.get('macd_histogram', '?')} ({m.get('macd_direction', '?')})  Stoch={m.get('stochastic_k', '?')}")
            report.append(f"    Volatility: {v.get('state', '?')} (ATR ratio={v.get('atr_ratio', '?')}x)  BB Squeeze: {'YES' if v.get('bb_squeeze') else 'No'}")
            report.append(f"    Volume:    {vol.get('state', '?')} ({vol.get('ratio', '?')}x avg)  OBV: {vol.get('obv_trend', '?')}")
            report.append(f"    Price:     ${pa.get('last_close', '?')}  range=${pa.get('period_range', '?')}  last_bar={pa.get('last_bar_change_pct', '?')}%")
            if pats:
                report.append(f"    Patterns:  {', '.join(pats)}")

            # Session data (intraday only)
            sess = data.get("sessions", {})
            if sess:
                parts = [f"{n}=${ s['avg_range']}" for n, s in sess.items()]
                report.append(f"    Sessions:  {', '.join(parts)}")

        # Insights
        report.append(f"\n{'─' * 75}")
        report.append("  LEARNING INSIGHTS:")
        insights = self._generate_insights_multi(result)
        for ins in insights:
            report.append(f"    • {ins}")

        # Cumulative stats
        report.append(f"\n  CUMULATIVE PATTERN COUNTS:")
        for pattern, count in sorted(self.pattern_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                report.append(f"    {pattern:35s}: {count}")

        report.append("\n" + "=" * 75)
        return "\n".join(report)

    def _generate_insights_multi(self, result: dict) -> list:
        insights = []
        ct = result.get("cross_tf", {})
        tfs = result.get("timeframes", {})

        # Multi-TF confluence
        conf = ct.get("confluence", "MIXED")
        if "STRONG" in conf:
            insights.append(f"STRONG {conf.split('_')[1]} CONFLUENCE across {ct.get('bull_tf_count', 0) + ct.get('bear_tf_count', 0)} timeframes — high-conviction signal")

        # Divergence warning
        if ct.get("tf_divergence"):
            insights.append(f"TF DIVERGENCE: Short-term and long-term charts disagree ({ct.get('divergence_detail')}) — be cautious, reversal possible")

        # Multi-TF squeeze
        sq = ct.get("squeeze_count", 0)
        if sq >= 3:
            insights.append(f"MULTI-TF SQUEEZE on {sq} timeframes — MAJOR volatility expansion imminent")
        elif sq >= 2:
            insights.append(f"BB Squeeze on {sq} timeframes — watch for breakout")

        # RSI extremes across TFs
        rsi_by_tf = ct.get("rsi_by_tf", {})
        ob_count = sum(1 for v in rsi_by_tf.values() if v and v > 70)
        os_count = sum(1 for v in rsi_by_tf.values() if v and v < 30)
        if ob_count >= 2:
            insights.append(f"OVERBOUGHT on {ob_count} timeframes — pullback risk elevated")
        if os_count >= 2:
            insights.append(f"OVERSOLD on {os_count} timeframes — bounce likely")

        # Per-TF specific
        for tf in ["5m", "15m"]:
            if tf in tfs:
                vol = tfs[tf].get("volume", {})
                if vol.get("state") == "spike":
                    insights.append(f"VOLUME SPIKE on {TF_LABELS[tf]} — institutional activity")

        # Daily chart regime as the anchor
        if "1d" in tfs:
            daily = tfs["1d"].get("regime", {})
            if daily.get("adx", 0) and daily["adx"] > 40:
                insights.append(f"DAILY ADX={daily['adx']} — strong trend, favor trend-following strategies")
            elif daily.get("adx", 0) and daily["adx"] < 15:
                insights.append(f"DAILY ADX={daily['adx']} — no trend, favor mean-reversion")

        # 5M vs 1H alignment check
        if "5m" in tfs and "1h" in tfs:
            dir_5m = tfs["5m"].get("regime", {}).get("trend", "")
            dir_1h = tfs["1h"].get("regime", {}).get("trend", "")
            if "uptrend" in dir_5m and "uptrend" in dir_1h:
                insights.append("5M and 1H both bullish — short-term momentum confirms hourly trend")
            elif "downtrend" in dir_5m and "downtrend" in dir_1h:
                insights.append("5M and 1H both bearish — short-term momentum confirms hourly trend")

        if not insights:
            insights.append("Market conditions normal across all timeframes — no special alerts")

        return insights

    def save_knowledge(self):
        """Persist all learned knowledge to disk."""
        knowledge = {
            "last_updated": str(datetime.now()),
            "total_observations": len(self.observations),
            "pattern_counts": self.pattern_counts,
            "support_resistance": self.support_resistance_levels,
            "session_stats": {
                name: {
                    "avg_range": round(np.mean(ranges[-100:]), 2) if ranges else 0,
                    "observations": len(ranges),
                }
                for name, ranges in self.session_profiles.items()
            },
        }

        path = self.data_dir / "knowledge_base.json"
        with open(path, "w") as f:
            json.dump(knowledge, f, indent=2, default=str)

        obs_path = self.data_dir / "observations.json"
        with open(obs_path, "w") as f:
            json.dump(self.observations[-500:], f, indent=2, default=str)

        log.info("knowledge_saved", path=str(path), observations=len(self.observations))

    def _load_knowledge(self):
        kb_path = self.data_dir / "knowledge_base.json"
        if kb_path.exists():
            with open(kb_path) as f:
                data = json.load(f)
            self.pattern_counts.update(data.get("pattern_counts", {}))
            self.support_resistance_levels = data.get("support_resistance", [])
            for name, stats in data.get("session_stats", {}).items():
                if name in self.session_profiles and stats.get("avg_range"):
                    self.session_profiles[name] = [stats["avg_range"]] * min(stats.get("observations", 1), 50)
            log.info("knowledge_loaded", observations=data.get("total_observations", 0))

        obs_path = self.data_dir / "observations.json"
        if obs_path.exists():
            with open(obs_path) as f:
                self.observations = json.load(f)
