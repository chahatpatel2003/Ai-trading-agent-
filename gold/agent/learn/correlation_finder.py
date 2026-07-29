"""
Correlation Finder — Discovers relationships between indicators, sessions, timeframes.

Finds:
- Which indicators predict moves (lead/lag relationships)
- Session correlations (Asian range → London breakout?)
- Day-of-week effects
- Volatility regime transitions
- Cross-timeframe signal propagation
- Round number magnetism
- What happens BEFORE and AFTER big moves
"""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.indicators import rsi, macd, bollinger_bands, atr, ema, sma, adx, obv, stochastic
from logs.logger import get_logger

log = get_logger("learn.correlations")


class CorrelationFinder:
    """Discovers and tracks market correlations and leading indicators."""

    def __init__(self, data_dir: str = "./learn/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Discovered correlations
        self.correlations = {}
        self.session_stats = defaultdict(lambda: {
            "ranges": [], "breakout_sizes": [], "directions": [],
            "follow_through": []
        })
        self.day_of_week_stats = defaultdict(lambda: {
            "returns": [], "ranges": [], "volume": []
        })
        self.pre_move_patterns = []  # what happens before big moves
        self.round_number_reactions = []
        self.volatility_transitions = []
        self.indicator_lead_lag = {}

        self._load_state()

    def analyze_correlations(self, df: pd.DataFrame, timeframe: str) -> dict:
        """Run full correlation analysis on the data."""
        if len(df) < 100:
            return {"status": "insufficient_data", "bars": len(df)}

        results = {}

        results["indicator_correlations"] = self._indicator_return_correlation(df)
        results["session_analysis"] = self._session_correlations(df, timeframe)
        results["day_of_week"] = self._day_of_week_analysis(df)
        results["pre_big_move"] = self._pre_big_move_analysis(df)
        results["round_numbers"] = self._round_number_analysis(df)
        results["volatility_regime"] = self._volatility_regime_analysis(df)
        results["lead_lag"] = self._lead_lag_analysis(df)
        results["momentum_persistence"] = self._momentum_persistence(df)

        return results

    def _indicator_return_correlation(self, df: pd.DataFrame) -> dict:
        """Which indicators correlate with future returns?"""
        close = df["close"]
        future_ret_5 = close.pct_change(5).shift(-5)  # return over next 5 bars
        future_ret_10 = close.pct_change(10).shift(-10)

        rsi_vals = rsi(close, 14)
        macd_l, _, macd_h = macd(close, 12, 26, 9)
        atr_vals = atr(df, 14)
        adx_vals = adx(df, 14) if len(df) > 45 else pd.Series(0, index=df.index)
        obv_vals = obv(df)
        stoch_k, _ = stochastic(df, 14, 3)

        # Normalized ATR (ATR / price)
        norm_atr = atr_vals / close

        indicators = {
            "rsi": rsi_vals,
            "macd_hist": macd_h,
            "adx": adx_vals,
            "norm_atr": norm_atr,
            "stoch_k": stoch_k,
            "obv_change": obv_vals.pct_change(5),
        }

        corr_5bar = {}
        corr_10bar = {}
        for name, series in indicators.items():
            valid = pd.DataFrame({"ind": series, "ret5": future_ret_5, "ret10": future_ret_10}).dropna()
            if len(valid) > 30:
                c5 = valid["ind"].corr(valid["ret5"])
                c10 = valid["ind"].corr(valid["ret10"])
                corr_5bar[name] = round(c5, 4) if not np.isnan(c5) else 0
                corr_10bar[name] = round(c10, 4) if not np.isnan(c10) else 0

        result = {
            "5bar_future_return": corr_5bar,
            "10bar_future_return": corr_10bar,
            "strongest_predictor_5bar": max(corr_5bar, key=lambda k: abs(corr_5bar[k])) if corr_5bar else None,
            "strongest_predictor_10bar": max(corr_10bar, key=lambda k: abs(corr_10bar[k])) if corr_10bar else None,
        }

        self.indicator_lead_lag = result
        return result

    def _session_correlations(self, df: pd.DataFrame, timeframe: str) -> dict:
        """Asian range → London breakout correlation, session profiling."""
        if timeframe not in ("5m", "15m", "1h"):
            return {"note": "session analysis requires intraday data"}

        results = {}

        # Tag sessions
        for session_name, start_h, end_h in [("asian", 0, 8), ("london", 8, 16), ("ny", 13, 22)]:
            mask = (df.index.hour >= start_h) & (df.index.hour < end_h)
            session_data = df[mask]
            if len(session_data) < 10:
                continue

            # Group by date
            daily_groups = session_data.groupby(session_data.index.date)
            for date, group in daily_groups:
                if len(group) < 2:
                    continue
                session_range = float(group["high"].max() - group["low"].min())
                session_return = float(group["close"].iloc[-1] - group["open"].iloc[0])
                direction = "up" if session_return > 0 else "down"

                self.session_stats[session_name]["ranges"].append(session_range)
                self.session_stats[session_name]["directions"].append(direction)

        # Asian → London correlation
        asian_ranges = self.session_stats["asian"]["ranges"]
        london_ranges = self.session_stats["london"]["ranges"]
        min_len = min(len(asian_ranges), len(london_ranges))
        if min_len > 10:
            corr = np.corrcoef(asian_ranges[-min_len:], london_ranges[-min_len:])[0, 1]
            results["asian_london_range_correlation"] = round(corr, 3)
            results["interpretation"] = (
                "TIGHT Asian range → LARGE London move" if corr < -0.2
                else "Asian range predicts London range" if corr > 0.3
                else "Weak relationship between Asian and London ranges"
            )

        # Session statistics
        for name, stats in self.session_stats.items():
            if stats["ranges"]:
                results[f"{name}_avg_range"] = round(np.mean(stats["ranges"][-100:]), 2)
                results[f"{name}_max_range"] = round(np.max(stats["ranges"][-100:]), 2)
                up_pct = sum(1 for d in stats["directions"][-100:] if d == "up") / max(1, len(stats["directions"][-100:]))
                results[f"{name}_bullish_pct"] = round(up_pct * 100, 1)

        return results

    def _day_of_week_analysis(self, df: pd.DataFrame) -> dict:
        """Do certain days of the week behave differently?"""
        df_copy = df.copy()
        df_copy["return"] = df_copy["close"].pct_change()
        df_copy["range"] = df_copy["high"] - df_copy["low"]
        df_copy["dow"] = df_copy.index.dayofweek

        day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
        results = {}

        for dow, name in day_names.items():
            day_data = df_copy[df_copy["dow"] == dow]
            if len(day_data) < 10:
                continue

            avg_ret = float(day_data["return"].mean())
            avg_range = float(day_data["range"].mean())
            win_rate = float((day_data["return"] > 0).mean())

            self.day_of_week_stats[name]["returns"].append(avg_ret)
            self.day_of_week_stats[name]["ranges"].append(avg_range)

            results[name] = {
                "avg_return_pct": round(avg_ret * 100, 4),
                "avg_range": round(avg_range, 2),
                "bullish_pct": round(win_rate * 100, 1),
                "bars": len(day_data),
            }

        # Find best and worst days
        if results:
            best_day = max(results, key=lambda k: results[k]["avg_return_pct"])
            worst_day = min(results, key=lambda k: results[k]["avg_return_pct"])
            most_volatile = max(results, key=lambda k: results[k]["avg_range"])
            results["_best_day"] = best_day
            results["_worst_day"] = worst_day
            results["_most_volatile"] = most_volatile

        return results

    def _pre_big_move_analysis(self, df: pd.DataFrame) -> dict:
        """What happens BEFORE big moves? Find precursors."""
        close = df["close"]
        returns = close.pct_change()
        atr_vals = atr(df, 14)
        rsi_vals = rsi(close, 14)
        bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20, 2.0)
        bb_width = (bb_upper - bb_lower) / bb_mid

        # Define "big move" as > 2x ATR in one bar
        big_move_threshold = 2.0
        big_moves = []

        for i in range(20, len(df) - 1):
            bar_range = float(df["high"].iloc[i] - df["low"].iloc[i])
            current_atr = float(atr_vals.iloc[i]) if not pd.isna(atr_vals.iloc[i]) else 999
            if current_atr > 0 and bar_range > current_atr * big_move_threshold:
                direction = "up" if float(close.iloc[i]) > float(close.iloc[i - 1]) else "down"

                # Look at what happened in the 5 bars BEFORE
                pre_rsi = float(rsi_vals.iloc[i - 1]) if not pd.isna(rsi_vals.iloc[i - 1]) else 50
                pre_bb_width = float(bb_width.iloc[i - 1]) if not pd.isna(bb_width.iloc[i - 1]) else 0
                pre_range_ratio = float(
                    (df["high"].iloc[i - 1] - df["low"].iloc[i - 1]) / current_atr
                ) if current_atr > 0 else 1

                pre_vol_avg = float(df["volume"].iloc[i - 5:i].mean()) if df["volume"].sum() > 0 else 0
                move_vol = float(df["volume"].iloc[i])
                vol_spike = move_vol / pre_vol_avg if pre_vol_avg > 0 else 1

                big_moves.append({
                    "direction": direction,
                    "magnitude_atr": round(bar_range / current_atr, 2),
                    "pre_rsi": round(pre_rsi, 1),
                    "pre_bb_width": round(pre_bb_width, 4),
                    "pre_range_ratio": round(pre_range_ratio, 2),
                    "volume_spike": round(vol_spike, 2),
                    "was_squeeze": pre_bb_width < 0.03,
                })

        if not big_moves:
            return {"big_moves_found": 0}

        self.pre_move_patterns.extend(big_moves[-50:])

        # Aggregate precursor patterns
        squeeze_before = sum(1 for m in big_moves if m["was_squeeze"])
        avg_pre_rsi = np.mean([m["pre_rsi"] for m in big_moves])
        avg_vol_spike = np.mean([m["volume_spike"] for m in big_moves])
        low_range_before = sum(1 for m in big_moves if m["pre_range_ratio"] < 0.7)

        return {
            "big_moves_found": len(big_moves),
            "preceded_by_squeeze_pct": round(squeeze_before / len(big_moves) * 100, 1),
            "avg_pre_move_rsi": round(avg_pre_rsi, 1),
            "avg_volume_spike_on_move": round(avg_vol_spike, 2),
            "preceded_by_low_range_pct": round(low_range_before / len(big_moves) * 100, 1),
            "insight": self._big_move_insight(squeeze_before / len(big_moves), avg_vol_spike,
                                              low_range_before / len(big_moves)),
        }

    def _big_move_insight(self, squeeze_pct, avg_vol_spike, low_range_pct) -> str:
        parts = []
        if squeeze_pct > 0.4:
            parts.append(f"{squeeze_pct*100:.0f}% of big moves came after BB squeezes — WATCH FOR SQUEEZES")
        if avg_vol_spike > 2.0:
            parts.append(f"Big moves typically see {avg_vol_spike:.1f}x volume spike — volume confirms breakouts")
        if low_range_pct > 0.5:
            parts.append(f"{low_range_pct*100:.0f}% preceded by quiet bars — calm before the storm pattern")
        return " | ".join(parts) if parts else "No strong precursor pattern found yet"

    def _round_number_analysis(self, df: pd.DataFrame) -> dict:
        """Does price react to round numbers ($2600, $2650, etc)?"""
        close = df["close"]
        atr_val = float(atr(df, 14).iloc[-1]) if not pd.isna(atr(df, 14).iloc[-1]) else 10

        reactions = {"bounce": 0, "break_through": 0, "magnet": 0}
        round_levels = list(range(int(close.min() // 50) * 50, int(close.max() // 50 + 2) * 50, 50))

        for i in range(1, len(df)):
            price = float(close.iloc[i])
            prev_price = float(close.iloc[i - 1])

            for level in round_levels:
                dist = abs(price - level)
                if dist < atr_val * 0.3:
                    # Price is near a round number
                    if (prev_price < level and price < level) or (prev_price > level and price > level):
                        reactions["bounce"] += 1  # bounced off
                    elif (prev_price < level and price > level) or (prev_price > level and price < level):
                        reactions["break_through"] += 1  # broke through

                    if dist < atr_val * 0.1:
                        reactions["magnet"] += 1  # attracted to level

        total = reactions["bounce"] + reactions["break_through"]
        bounce_rate = reactions["bounce"] / total if total > 0 else 0

        self.round_number_reactions.append({
            "timestamp": str(datetime.now()),
            "bounce_rate": round(bounce_rate, 3),
            "total_interactions": total,
        })

        return {
            "bounce_count": reactions["bounce"],
            "breakthrough_count": reactions["break_through"],
            "magnet_count": reactions["magnet"],
            "bounce_rate": round(bounce_rate * 100, 1),
            "insight": (
                f"Round numbers act as SUPPORT/RESISTANCE ({bounce_rate*100:.0f}% bounce rate)"
                if bounce_rate > 0.55
                else f"Round numbers are often BROKEN ({(1-bounce_rate)*100:.0f}% breakthrough rate)"
                if bounce_rate < 0.45
                else "Round numbers have mixed significance"
            ),
        }

    def _volatility_regime_analysis(self, df: pd.DataFrame) -> dict:
        """Track volatility regime transitions — what follows high/low vol?"""
        atr_vals = atr(df, 14)
        atr_avg = atr_vals.rolling(50, min_periods=10).mean()
        ratio = atr_vals / atr_avg

        # Classify regimes
        regimes = pd.Series("normal", index=df.index)
        regimes[ratio > 1.5] = "high_vol"
        regimes[ratio < 0.7] = "low_vol"

        transitions = {"low_to_high": 0, "high_to_low": 0, "low_to_normal": 0, "high_to_normal": 0}
        returns_after_low = []
        returns_after_high = []

        for i in range(1, len(regimes)):
            prev_r = regimes.iloc[i - 1]
            curr_r = regimes.iloc[i]
            if prev_r != curr_r:
                key = f"{prev_r}_to_{curr_r}"
                if key in transitions:
                    transitions[key] += 1

            # Track returns after regime
            if i < len(df) - 5:
                future_ret = float(df["close"].iloc[i + 5] / df["close"].iloc[i] - 1)
                if regimes.iloc[i] == "low_vol":
                    returns_after_low.append(future_ret)
                elif regimes.iloc[i] == "high_vol":
                    returns_after_high.append(future_ret)

        result = {
            "transitions": transitions,
            "current_regime": str(regimes.iloc[-1]),
            "current_vol_ratio": round(float(ratio.iloc[-1]), 2) if not pd.isna(ratio.iloc[-1]) else 1.0,
        }

        if returns_after_low:
            result["avg_return_after_low_vol"] = round(np.mean(returns_after_low) * 100, 4)
            result["abs_move_after_low_vol"] = round(np.mean(np.abs(returns_after_low)) * 100, 4)

        if returns_after_high:
            result["avg_return_after_high_vol"] = round(np.mean(returns_after_high) * 100, 4)

        self.volatility_transitions.append({
            "regime": str(regimes.iloc[-1]),
            "ratio": result["current_vol_ratio"],
        })

        return result

    def _lead_lag_analysis(self, df: pd.DataFrame) -> dict:
        """Which indicators lead price moves?"""
        close = df["close"]
        returns_5 = close.pct_change(5).shift(-5)

        rsi_vals = rsi(close, 14)
        _, _, macd_h = macd(close, 12, 26, 9)
        obv_vals = obv(df)
        obv_change = obv_vals.pct_change(5)

        # Check if indicator changes PRECEDE price changes
        results = {}
        for name, indicator in [("rsi_change", rsi_vals.diff(3)),
                                 ("macd_hist_change", macd_h.diff(3)),
                                 ("obv_momentum", obv_change)]:
            valid = pd.DataFrame({"ind": indicator, "future_ret": returns_5}).dropna()
            if len(valid) > 30:
                corr = valid["ind"].corr(valid["future_ret"])
                results[name] = {
                    "correlation_with_future": round(corr, 4) if not np.isnan(corr) else 0,
                    "leads_price": abs(corr) > 0.15,
                    "direction": "positive" if corr > 0 else "negative",
                }

        # Find which leads most
        if results:
            best = max(results, key=lambda k: abs(results[k]["correlation_with_future"]))
            results["_best_leading_indicator"] = best

        return results

    def _momentum_persistence(self, df: pd.DataFrame) -> dict:
        """Does momentum persist or mean-revert in gold?"""
        close = df["close"]

        lookbacks = [3, 5, 10, 20]
        results = {}

        for lb in lookbacks:
            past_ret = close.pct_change(lb)
            future_ret = close.pct_change(lb).shift(-lb)

            valid = pd.DataFrame({"past": past_ret, "future": future_ret}).dropna()
            if len(valid) > 30:
                corr = valid["past"].corr(valid["future"])
                results[f"{lb}bar"] = {
                    "autocorrelation": round(corr, 4) if not np.isnan(corr) else 0,
                    "behavior": (
                        "MOMENTUM (trend continues)" if corr > 0.1
                        else "MEAN REVERSION (trend reverses)" if corr < -0.1
                        else "RANDOM (no pattern)"
                    ),
                }

        return results

    def get_report(self) -> str:
        """Generate correlation findings report."""
        lines = []
        lines.append("CORRELATION DISCOVERIES")
        lines.append("=" * 70)

        if self.indicator_lead_lag:
            lines.append("\nINDICATOR PREDICTIVE POWER (correlation with future returns):")
            for horizon in ["5bar_future_return", "10bar_future_return"]:
                corrs = self.indicator_lead_lag.get(horizon, {})
                if corrs:
                    lines.append(f"  {horizon}:")
                    for ind, val in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True):
                        bar = "+" * int(abs(val) * 50)
                        lines.append(f"    {ind:<20s}: {val:+.4f}  {bar}")

        if self.session_stats:
            lines.append("\nSESSION PROFILES:")
            for name, stats in self.session_stats.items():
                if stats["ranges"]:
                    avg = np.mean(stats["ranges"][-50:])
                    lines.append(f"  {name}: avg_range=${avg:.2f}, observations={len(stats['ranges'])}")

        if self.day_of_week_stats:
            lines.append("\nDAY-OF-WEEK EFFECTS:")
            for day, stats in self.day_of_week_stats.items():
                if stats["returns"]:
                    avg_ret = np.mean(stats["returns"])
                    lines.append(f"  {day}: avg_return={avg_ret*100:.4f}%")

        if self.pre_move_patterns:
            squeeze_pct = sum(1 for p in self.pre_move_patterns if p["was_squeeze"]) / len(self.pre_move_patterns)
            lines.append(f"\nPRE-BIG-MOVE ANALYSIS ({len(self.pre_move_patterns)} events):")
            lines.append(f"  BB squeeze before big move: {squeeze_pct*100:.1f}%")

        return "\n".join(lines)

    def save_state(self):
        """Persist correlation findings."""
        state = {
            "last_updated": str(datetime.now()),
            "indicator_lead_lag": self.indicator_lead_lag,
            "session_stats": {
                k: {sk: sv[-200:] if isinstance(sv, list) else sv for sk, sv in v.items()}
                for k, v in self.session_stats.items()
            },
            "day_of_week_stats": dict(self.day_of_week_stats),
            "pre_move_patterns": self.pre_move_patterns[-200:],
            "round_number_reactions": self.round_number_reactions[-100:],
            "volatility_transitions": self.volatility_transitions[-200:],
        }
        path = self.data_dir / "correlations.json"
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self):
        path = self.data_dir / "correlations.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                state = json.load(f)
            self.indicator_lead_lag = state.get("indicator_lead_lag", {})
            for k, v in state.get("session_stats", {}).items():
                self.session_stats[k].update(v)
            for k, v in state.get("day_of_week_stats", {}).items():
                self.day_of_week_stats[k].update(v)
            self.pre_move_patterns = state.get("pre_move_patterns", [])
            self.round_number_reactions = state.get("round_number_reactions", [])
            self.volatility_transitions = state.get("volatility_transitions", [])
            log.info("correlations_loaded")
        except Exception as e:
            log.warning("correlations_load_failed", error=str(e))
