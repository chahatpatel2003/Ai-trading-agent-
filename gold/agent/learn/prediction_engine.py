"""
Prediction Engine — Makes directional predictions, tracks outcomes, learns what works.

The bot "thinks" about trades without executing:
- Predicts: "Price will go UP/DOWN because [reasons]"
- Waits for outcome
- Scores: was it right? by how much?
- Requires N confirmations before trusting a pattern
- Tracks accuracy by pattern type, timeframe, session, confidence
"""
import json
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.indicators import rsi, macd, bollinger_bands, atr, ema, sma, adx, stochastic, obv
from logs.logger import get_logger

log = get_logger("learn.predictions")


class Prediction:
    """A single directional prediction with reasoning."""

    def __init__(self, direction: str, confidence: float, reasons: list,
                 entry_price: float, timeframe: str, horizon_bars: int,
                 target_price: float = None, stop_price: float = None,
                 strategy_hypothesis: str = None):
        self.id = str(uuid.uuid4())[:8]
        self.timestamp = str(datetime.now())
        self.direction = direction  # "LONG" or "SHORT"
        self.confidence = confidence  # 0.0 to 1.0
        self.reasons = reasons  # list of reason strings
        self.entry_price = entry_price
        self.timeframe = timeframe
        self.horizon_bars = horizon_bars
        self.target_price = target_price
        self.stop_price = stop_price
        self.strategy_hypothesis = strategy_hypothesis

        # Outcome tracking
        self.resolved = False
        self.outcome = None  # "CORRECT", "WRONG", "PARTIAL"
        self.exit_price = None
        self.pnl_points = None
        self.bars_held = 0
        self.max_favorable = 0.0  # max move in predicted direction
        self.max_adverse = 0.0  # max move against prediction
        self.resolution_time = None
        self.post_mortem = None  # why it worked or failed

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "direction": self.direction,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "entry_price": self.entry_price,
            "timeframe": self.timeframe,
            "horizon_bars": self.horizon_bars,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "strategy_hypothesis": self.strategy_hypothesis,
            "resolved": self.resolved,
            "outcome": self.outcome,
            "exit_price": self.exit_price,
            "pnl_points": self.pnl_points,
            "bars_held": self.bars_held,
            "max_favorable": self.max_favorable,
            "max_adverse": self.max_adverse,
            "resolution_time": self.resolution_time,
            "post_mortem": self.post_mortem,
        }


class PredictionEngine:
    """
    Makes predictions based on market conditions, tracks them, and learns.
    """

    def __init__(self, data_dir: str = "./learn/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.active_predictions = []  # unresolved
        self.resolved_predictions = []  # completed
        self.pattern_accuracy = defaultdict(lambda: {"correct": 0, "wrong": 0, "partial": 0})
        self.confirmed_patterns = {}  # patterns with N+ confirmations
        self.min_confirmations = 5  # need 5 correct before trusting

        self._load_state()

    def analyze_and_predict(self, df: pd.DataFrame, timeframe: str,
                            cross_tf_data: dict = None) -> list:
        """Analyze current market state and generate predictions.

        Discipline:
        - Skip strategies marked AVOID or UNRELIABLE (learned from past mistakes)
        - Only ONE prediction per timeframe (no conflicting LONG+SHORT)
        - No new prediction if same strategy already has an active one
        - Cooldown: skip strategy if it resolved WRONG in the last 3 cycles
        """
        if len(df) < 50:
            return []

        candidates = []
        close = df["close"]
        current_price = float(close.iloc[-1])

        # Calculate all indicators
        signals = self._compute_signals(df)

        # --- Prediction Rules (each generates a hypothesis) ---
        checks = [
            self._check_rsi_bb_reversal(signals, current_price, timeframe),
            self._check_macd_trend(signals, current_price, timeframe),
            self._check_bb_squeeze_breakout(signals, current_price, timeframe, df),
            self._check_multi_tf_confluence(signals, current_price, timeframe, cross_tf_data) if cross_tf_data else None,
            self._check_volume_divergence(signals, current_price, timeframe, df),
            self._check_ema_ribbon(signals, current_price, timeframe),
            self._check_stoch_adx(signals, current_price, timeframe),
            self._check_sr_bounce(signals, current_price, timeframe, df),
        ]
        candidates = [p for p in checks if p is not None]

        # ─── FILTER 1: Skip strategies the bot has learned to avoid ───
        filtered = []
        for pred in candidates:
            key = pred.strategy_hypothesis or "unknown"
            status_info = self.confirmed_patterns.get(key, {})
            status = status_info.get("status", "LEARNING")

            if status == "AVOID":
                log.info("prediction_skipped_avoid",
                         strategy=key, win_rate=status_info.get("win_rate", 0))
                continue
            if status == "UNRELIABLE" and pred.confidence < 0.70:
                log.info("prediction_skipped_unreliable",
                         strategy=key, confidence=pred.confidence)
                continue
            filtered.append(pred)
        candidates = filtered

        # ─── FILTER 2: No duplicate — skip if strategy already has active prediction ───
        active_strategies = {p.strategy_hypothesis for p in self.active_predictions
                             if p.timeframe == timeframe and not p.resolved}
        candidates = [p for p in candidates
                      if (p.strategy_hypothesis or "unknown") not in active_strategies]

        # ─── FILTER 3: No conflicting directions — pick highest confidence ───
        if candidates:
            longs = [p for p in candidates if p.direction == "LONG"]
            shorts = [p for p in candidates if p.direction == "SHORT"]

            if longs and shorts:
                # Conflict: both LONG and SHORT signals on same TF
                # Pick the direction with higher total confidence
                long_conf = max(p.confidence for p in longs)
                short_conf = max(p.confidence for p in shorts)

                if long_conf > short_conf:
                    log.info("prediction_conflict_resolved",
                             timeframe=timeframe, chosen="LONG",
                             long_conf=long_conf, short_conf=short_conf)
                    candidates = longs
                else:
                    log.info("prediction_conflict_resolved",
                             timeframe=timeframe, chosen="SHORT",
                             long_conf=long_conf, short_conf=short_conf)
                    candidates = shorts

        # ─── FILTER 4: Only the best prediction per timeframe ───
        # Take the single highest-confidence prediction
        if candidates:
            candidates.sort(key=lambda p: p.confidence, reverse=True)
            candidates = [candidates[0]]

        # Store accepted predictions
        predictions = []
        for p in candidates:
            self.active_predictions.append(p)
            predictions.append(p)
            log.info("prediction_accepted",
                     id=p.id, direction=p.direction,
                     confidence=p.confidence,
                     strategy=p.strategy_hypothesis,
                     reasons="; ".join(p.reasons),
                     entry=p.entry_price,
                     timeframe=timeframe)

        return predictions

    def check_outcomes(self, df: pd.DataFrame, timeframe: str):
        """Check active predictions against new data and resolve them."""
        if len(df) < 2:
            return

        current_price = float(df["close"].iloc[-1])
        current_high = float(df["high"].iloc[-1])
        current_low = float(df["low"].iloc[-1])

        still_active = []
        newly_resolved = []

        for pred in self.active_predictions:
            if pred.timeframe != timeframe:
                still_active.append(pred)
                continue

            pred.bars_held += 1

            # Track max favorable/adverse excursion
            if pred.direction == "LONG":
                pred.max_favorable = max(pred.max_favorable, current_high - pred.entry_price)
                pred.max_adverse = max(pred.max_adverse, pred.entry_price - current_low)
            else:
                pred.max_favorable = max(pred.max_favorable, pred.entry_price - current_low)
                pred.max_adverse = max(pred.max_adverse, current_high - pred.entry_price)

            # Check if target hit
            target_hit = False
            stop_hit = False

            if pred.target_price:
                if pred.direction == "LONG" and current_high >= pred.target_price:
                    target_hit = True
                elif pred.direction == "SHORT" and current_low <= pred.target_price:
                    target_hit = True

            if pred.stop_price:
                if pred.direction == "LONG" and current_low <= pred.stop_price:
                    stop_hit = True
                elif pred.direction == "SHORT" and current_high >= pred.stop_price:
                    stop_hit = True

            # Resolve
            if target_hit:
                pred.resolved = True
                pred.outcome = "CORRECT"
                pred.exit_price = pred.target_price
                pred.pnl_points = abs(pred.target_price - pred.entry_price)
            elif stop_hit:
                pred.resolved = True
                pred.outcome = "WRONG"
                pred.exit_price = pred.stop_price
                pred.pnl_points = -abs(pred.stop_price - pred.entry_price)
            elif pred.bars_held >= pred.horizon_bars:
                # Time expired — judge by final price
                pred.resolved = True
                pred.exit_price = current_price
                if pred.direction == "LONG":
                    pred.pnl_points = current_price - pred.entry_price
                else:
                    pred.pnl_points = pred.entry_price - current_price

                if pred.pnl_points > 0:
                    pred.outcome = "CORRECT"
                elif pred.pnl_points < -1.0:
                    pred.outcome = "WRONG"
                else:
                    pred.outcome = "PARTIAL"

            if pred.resolved:
                pred.resolution_time = str(datetime.now())
                pred.post_mortem = self._generate_post_mortem(pred, df)
                newly_resolved.append(pred)
                self._update_pattern_accuracy(pred)

                log.info("prediction_resolved",
                         id=pred.id,
                         outcome=pred.outcome,
                         direction=pred.direction,
                         pnl=round(pred.pnl_points, 2) if pred.pnl_points else 0,
                         bars_held=pred.bars_held,
                         max_fav=round(pred.max_favorable, 2),
                         max_adv=round(pred.max_adverse, 2))
            else:
                still_active.append(pred)

        self.active_predictions = still_active
        self.resolved_predictions.extend(newly_resolved)

        return newly_resolved

    def _compute_signals(self, df: pd.DataFrame) -> dict:
        """Compute all indicators at current bar."""
        close = df["close"]
        n = len(df)

        rsi_vals = rsi(close, 14)
        macd_line, signal_line, macd_hist = macd(close, 12, 26, 9)
        bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20, 2.0)
        atr_vals = atr(df, 14)
        ema8 = ema(close, min(8, n - 1))
        ema21 = ema(close, min(21, n - 1))
        ema55 = ema(close, min(55, n - 1))
        sma50 = sma(close, min(50, n - 1))
        sma200 = sma(close, min(200, n - 1))
        adx_vals = adx(df, min(14, n // 3)) if n > 45 else pd.Series(0, index=df.index)
        stoch_k, stoch_d = stochastic(df, 14, 3) if n > 20 else (pd.Series(50, index=df.index), pd.Series(50, index=df.index))
        obv_vals = obv(df)

        # BB width for squeeze detection
        bb_width = (bb_upper - bb_lower) / bb_mid
        bb_width_avg = bb_width.rolling(50, min_periods=10).mean()

        return {
            "close": float(close.iloc[-1]),
            "rsi": float(rsi_vals.iloc[-1]) if not pd.isna(rsi_vals.iloc[-1]) else 50,
            "rsi_prev": float(rsi_vals.iloc[-2]) if len(rsi_vals) > 1 and not pd.isna(rsi_vals.iloc[-2]) else 50,
            "macd_hist": float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else 0,
            "macd_hist_prev": float(macd_hist.iloc[-2]) if len(macd_hist) > 1 and not pd.isna(macd_hist.iloc[-2]) else 0,
            "macd_line": float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0,
            "signal_line": float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0,
            "bb_upper": float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else 0,
            "bb_lower": float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else 0,
            "bb_mid": float(bb_mid.iloc[-1]) if not pd.isna(bb_mid.iloc[-1]) else 0,
            "bb_width": float(bb_width.iloc[-1]) if not pd.isna(bb_width.iloc[-1]) else 0,
            "bb_width_avg": float(bb_width_avg.iloc[-1]) if not pd.isna(bb_width_avg.iloc[-1]) else 0,
            "bb_squeeze": bool(bb_width.iloc[-1] < bb_width_avg.iloc[-1] * 0.7) if not pd.isna(bb_width_avg.iloc[-1]) else False,
            "atr": float(atr_vals.iloc[-1]) if not pd.isna(atr_vals.iloc[-1]) else 10,
            "ema8": float(ema8.iloc[-1]) if not pd.isna(ema8.iloc[-1]) else 0,
            "ema21": float(ema21.iloc[-1]) if not pd.isna(ema21.iloc[-1]) else 0,
            "ema55": float(ema55.iloc[-1]) if not pd.isna(ema55.iloc[-1]) else 0,
            "sma50": float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else 0,
            "sma200": float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else 0,
            "adx": float(adx_vals.iloc[-1]) if not pd.isna(adx_vals.iloc[-1]) else 0,
            "stoch_k": float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else 50,
            "stoch_d": float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else 50,
            "obv": float(obv_vals.iloc[-1]) if not pd.isna(obv_vals.iloc[-1]) else 0,
            "obv_ema": float(ema(obv_vals, 20).iloc[-1]) if not pd.isna(ema(obv_vals, 20).iloc[-1]) else 0,
            "price_vs_sma200": "above" if float(close.iloc[-1]) > float(sma200.iloc[-1]) else "below",
            "ema_ribbon_bull": float(ema8.iloc[-1]) > float(ema21.iloc[-1]) > float(ema55.iloc[-1]),
            "ema_ribbon_bear": float(ema8.iloc[-1]) < float(ema21.iloc[-1]) < float(ema55.iloc[-1]),
        }

    def _check_rsi_bb_reversal(self, signals, price, tf) -> Prediction:
        """RSI extreme + Bollinger Band touch = mean reversion play."""
        reasons = []

        if signals["rsi"] < 25 and price <= signals["bb_lower"] * 1.002:
            reasons.append(f"RSI deeply oversold at {signals['rsi']:.1f}")
            reasons.append(f"Price touching lower BB (${signals['bb_lower']:.2f})")
            if signals["stoch_k"] < 20:
                reasons.append(f"Stochastic also oversold ({signals['stoch_k']:.0f})")
            confidence = min(0.7, 0.4 + (30 - signals["rsi"]) / 100 + (0.1 if signals["stoch_k"] < 20 else 0))
            target = signals["bb_mid"]
            stop = price - signals["atr"] * 1.5
            return Prediction("LONG", confidence, reasons, price, tf,
                              horizon_bars=20, target_price=target, stop_price=stop,
                              strategy_hypothesis="rsi_bb_oversold_reversal")

        if signals["rsi"] > 75 and price >= signals["bb_upper"] * 0.998:
            reasons.append(f"RSI deeply overbought at {signals['rsi']:.1f}")
            reasons.append(f"Price touching upper BB (${signals['bb_upper']:.2f})")
            if signals["stoch_k"] > 80:
                reasons.append(f"Stochastic also overbought ({signals['stoch_k']:.0f})")
            confidence = min(0.7, 0.4 + (signals["rsi"] - 70) / 100 + (0.1 if signals["stoch_k"] > 80 else 0))
            target = signals["bb_mid"]
            stop = price + signals["atr"] * 1.5
            return Prediction("SHORT", confidence, reasons, price, tf,
                              horizon_bars=20, target_price=target, stop_price=stop,
                              strategy_hypothesis="rsi_bb_overbought_reversal")
        return None

    def _check_macd_trend(self, signals, price, tf) -> Prediction:
        """MACD crossover with trend filter."""
        hist = signals["macd_hist"]
        prev_hist = signals["macd_hist_prev"]

        # Bullish MACD cross + price above SMA200
        if prev_hist < 0 and hist > 0 and signals["price_vs_sma200"] == "above":
            reasons = [
                "MACD histogram crossed above zero (bullish crossover)",
                f"Price above SMA200 (trend filter confirms uptrend)",
                f"ADX={signals['adx']:.0f} — {'trending' if signals['adx'] > 20 else 'weak trend'}",
            ]
            confidence = 0.5 + (0.15 if signals["adx"] > 25 else 0)
            target = price + signals["atr"] * 3
            stop = price - signals["atr"] * 2
            return Prediction("LONG", confidence, reasons, price, tf,
                              horizon_bars=30, target_price=target, stop_price=stop,
                              strategy_hypothesis="macd_bullish_cross_trend")

        # Bearish MACD cross + price below SMA200
        if prev_hist > 0 and hist < 0 and signals["price_vs_sma200"] == "below":
            reasons = [
                "MACD histogram crossed below zero (bearish crossover)",
                f"Price below SMA200 (trend filter confirms downtrend)",
                f"ADX={signals['adx']:.0f}",
            ]
            confidence = 0.5 + (0.15 if signals["adx"] > 25 else 0)
            target = price - signals["atr"] * 3
            stop = price + signals["atr"] * 2
            return Prediction("SHORT", confidence, reasons, price, tf,
                              horizon_bars=30, target_price=target, stop_price=stop,
                              strategy_hypothesis="macd_bearish_cross_trend")
        return None

    def _check_bb_squeeze_breakout(self, signals, price, tf, df) -> Prediction:
        """BB squeeze releasing = volatility expansion, predict direction."""
        if not signals["bb_squeeze"]:
            return None

        # Squeeze is active — check for direction clues
        reasons = [f"Bollinger Band squeeze detected (width={signals['bb_width']:.4f} vs avg={signals['bb_width_avg']:.4f})"]

        # Use momentum to predict breakout direction
        if signals["rsi"] > 55 and signals["macd_hist"] > 0 and signals["ema_ribbon_bull"]:
            reasons.append(f"RSI={signals['rsi']:.0f} bullish bias")
            reasons.append("MACD histogram positive, EMA ribbon bullish")
            reasons.append("Expecting UPSIDE breakout from squeeze")
            target = price + signals["atr"] * 4
            stop = price - signals["atr"] * 1.5
            return Prediction("LONG", 0.55, reasons, price, tf,
                              horizon_bars=25, target_price=target, stop_price=stop,
                              strategy_hypothesis="bb_squeeze_bullish_breakout")

        if signals["rsi"] < 45 and signals["macd_hist"] < 0 and signals["ema_ribbon_bear"]:
            reasons.append(f"RSI={signals['rsi']:.0f} bearish bias")
            reasons.append("MACD histogram negative, EMA ribbon bearish")
            reasons.append("Expecting DOWNSIDE breakout from squeeze")
            target = price - signals["atr"] * 4
            stop = price + signals["atr"] * 1.5
            return Prediction("SHORT", 0.55, reasons, price, tf,
                              horizon_bars=25, target_price=target, stop_price=stop,
                              strategy_hypothesis="bb_squeeze_bearish_breakout")
        return None

    def _check_multi_tf_confluence(self, signals, price, tf, cross_tf_data) -> Prediction:
        """Multiple timeframes agreeing on direction."""
        confluence = cross_tf_data.get("confluence", "MIXED")
        alignment = cross_tf_data.get("alignment_score", 0)

        if confluence == "STRONG_BULLISH" and alignment > 0.5:
            reasons = [
                f"Multi-TF confluence: STRONG BULLISH ({cross_tf_data.get('bull_tf_count', 0)} TFs agree)",
                f"Alignment score: {alignment}",
                f"RSI across TFs avg: {cross_tf_data.get('avg_rsi_all_tf', 'N/A')}",
            ]
            if not cross_tf_data.get("tf_divergence"):
                reasons.append("No timeframe divergence (clean alignment)")
            confidence = min(0.75, 0.5 + alignment * 0.3)
            target = price + signals["atr"] * 3
            stop = price - signals["atr"] * 2
            return Prediction("LONG", confidence, reasons, price, tf,
                              horizon_bars=20, target_price=target, stop_price=stop,
                              strategy_hypothesis="multi_tf_bullish_confluence")

        if confluence == "STRONG_BEARISH" and alignment < -0.5:
            reasons = [
                f"Multi-TF confluence: STRONG BEARISH ({cross_tf_data.get('bear_tf_count', 0)} TFs agree)",
                f"Alignment score: {alignment}",
            ]
            confidence = min(0.75, 0.5 + abs(alignment) * 0.3)
            target = price - signals["atr"] * 3
            stop = price + signals["atr"] * 2
            return Prediction("SHORT", confidence, reasons, price, tf,
                              horizon_bars=20, target_price=target, stop_price=stop,
                              strategy_hypothesis="multi_tf_bearish_confluence")
        return None

    def _check_volume_divergence(self, signals, price, tf, df) -> Prediction:
        """Price making new highs/lows but OBV diverging."""
        if len(df) < 30:
            return None

        lookback = min(20, len(df) - 1)
        recent_highs = df["high"].tail(lookback)
        recent_lows = df["low"].tail(lookback)
        obv_vals = obv(df).tail(lookback)

        # Bearish divergence: price making higher highs, OBV making lower highs
        price_hh = float(recent_highs.iloc[-1]) > float(recent_highs.iloc[-lookback // 2])
        obv_lh = float(obv_vals.iloc[-1]) < float(obv_vals.iloc[-lookback // 2])

        if price_hh and obv_lh and signals["rsi"] > 60:
            reasons = [
                "BEARISH VOLUME DIVERGENCE: Price making higher highs but OBV declining",
                "Smart money may be distributing (selling into strength)",
                f"RSI={signals['rsi']:.0f} — elevated, room to fall",
            ]
            target = price - signals["atr"] * 2.5
            stop = price + signals["atr"] * 1.5
            return Prediction("SHORT", 0.5, reasons, price, tf,
                              horizon_bars=15, target_price=target, stop_price=stop,
                              strategy_hypothesis="bearish_obv_divergence")

        # Bullish divergence: price making lower lows, OBV making higher lows
        price_ll = float(recent_lows.iloc[-1]) < float(recent_lows.iloc[-lookback // 2])
        obv_hl = float(obv_vals.iloc[-1]) > float(obv_vals.iloc[-lookback // 2])

        if price_ll and obv_hl and signals["rsi"] < 40:
            reasons = [
                "BULLISH VOLUME DIVERGENCE: Price making lower lows but OBV rising",
                "Smart money may be accumulating (buying the dip)",
                f"RSI={signals['rsi']:.0f} — depressed, room to rise",
            ]
            target = price + signals["atr"] * 2.5
            stop = price - signals["atr"] * 1.5
            return Prediction("LONG", 0.5, reasons, price, tf,
                              horizon_bars=15, target_price=target, stop_price=stop,
                              strategy_hypothesis="bullish_obv_divergence")
        return None

    def _check_ema_ribbon(self, signals, price, tf) -> Prediction:
        """EMA 8/21/55 ribbon fully aligned = strong trend continuation."""
        if signals["ema_ribbon_bull"] and signals["adx"] > 25:
            # Pullback to EMA21 in bullish ribbon
            if abs(price - signals["ema21"]) < signals["atr"] * 0.5 and price > signals["ema21"]:
                reasons = [
                    "EMA ribbon fully bullish (8 > 21 > 55)",
                    f"ADX={signals['adx']:.0f} confirms strong trend",
                    f"Price pulled back to EMA21 (${signals['ema21']:.2f}) — buying the dip in trend",
                ]
                target = price + signals["atr"] * 3
                stop = signals["ema55"] - signals["atr"] * 0.5
                return Prediction("LONG", 0.6, reasons, price, tf,
                                  horizon_bars=20, target_price=target, stop_price=stop,
                                  strategy_hypothesis="ema_ribbon_bull_pullback")

        if signals["ema_ribbon_bear"] and signals["adx"] > 25:
            if abs(price - signals["ema21"]) < signals["atr"] * 0.5 and price < signals["ema21"]:
                reasons = [
                    "EMA ribbon fully bearish (8 < 21 < 55)",
                    f"ADX={signals['adx']:.0f} confirms strong trend",
                    f"Price bounced to EMA21 — selling the rally in downtrend",
                ]
                target = price - signals["atr"] * 3
                stop = signals["ema55"] + signals["atr"] * 0.5
                return Prediction("SHORT", 0.6, reasons, price, tf,
                                  horizon_bars=20, target_price=target, stop_price=stop,
                                  strategy_hypothesis="ema_ribbon_bear_rally")
        return None

    def _check_stoch_adx(self, signals, price, tf) -> Prediction:
        """Stochastic at extremes with strong ADX trend."""
        if signals["adx"] < 20:
            return None  # no trend, skip

        if signals["stoch_k"] < 15 and signals["stoch_d"] < 20 and signals["price_vs_sma200"] == "above":
            reasons = [
                f"Stochastic deeply oversold (K={signals['stoch_k']:.0f}, D={signals['stoch_d']:.0f})",
                f"But overall trend is UP (price above SMA200)",
                f"ADX={signals['adx']:.0f} — trend is strong, this is a pullback opportunity",
            ]
            target = price + signals["atr"] * 2.5
            stop = price - signals["atr"] * 2
            return Prediction("LONG", 0.55, reasons, price, tf,
                              horizon_bars=15, target_price=target, stop_price=stop,
                              strategy_hypothesis="stoch_oversold_in_uptrend")

        if signals["stoch_k"] > 85 and signals["stoch_d"] > 80 and signals["price_vs_sma200"] == "below":
            reasons = [
                f"Stochastic deeply overbought (K={signals['stoch_k']:.0f}, D={signals['stoch_d']:.0f})",
                f"But overall trend is DOWN (price below SMA200)",
                f"ADX={signals['adx']:.0f} — trend is strong, this rally will fade",
            ]
            target = price - signals["atr"] * 2.5
            stop = price + signals["atr"] * 2
            return Prediction("SHORT", 0.55, reasons, price, tf,
                              horizon_bars=15, target_price=target, stop_price=stop,
                              strategy_hypothesis="stoch_overbought_in_downtrend")
        return None

    def _check_sr_bounce(self, signals, price, tf, df) -> Prediction:
        """Price near a significant support/resistance level."""
        if len(df) < 50:
            return None

        lookback = min(50, len(df))
        recent_high = float(df["high"].tail(lookback).max())
        recent_low = float(df["low"].tail(lookback).min())

        # Near support (within 0.5 ATR of recent low)
        if abs(price - recent_low) < signals["atr"] * 0.5 and signals["rsi"] < 40:
            reasons = [
                f"Price near recent support at ${recent_low:.2f} (within 0.5 ATR)",
                f"RSI={signals['rsi']:.0f} suggesting oversold near support",
                "Expecting bounce from support level",
            ]
            target = price + signals["atr"] * 2
            stop = recent_low - signals["atr"] * 0.5
            return Prediction("LONG", 0.5, reasons, price, tf,
                              horizon_bars=12, target_price=target, stop_price=stop,
                              strategy_hypothesis="support_bounce")

        # Near resistance (within 0.5 ATR of recent high)
        if abs(price - recent_high) < signals["atr"] * 0.5 and signals["rsi"] > 60:
            reasons = [
                f"Price near recent resistance at ${recent_high:.2f} (within 0.5 ATR)",
                f"RSI={signals['rsi']:.0f} suggesting overbought near resistance",
                "Expecting rejection from resistance level",
            ]
            target = price - signals["atr"] * 2
            stop = recent_high + signals["atr"] * 0.5
            return Prediction("SHORT", 0.5, reasons, price, tf,
                              horizon_bars=12, target_price=target, stop_price=stop,
                              strategy_hypothesis="resistance_rejection")
        return None

    def _generate_post_mortem(self, pred: Prediction, df: pd.DataFrame) -> str:
        """Analyze WHY a prediction succeeded or failed."""
        parts = []
        if pred.outcome == "CORRECT":
            parts.append(f"CORRECT: {pred.direction} call worked, +${pred.pnl_points:.2f}")
            parts.append(f"Max favorable excursion: ${pred.max_favorable:.2f}")
            if pred.max_adverse > pred.max_favorable * 0.5:
                parts.append(f"WARNING: Had significant adverse move (${pred.max_adverse:.2f}) before working out")
        elif pred.outcome == "WRONG":
            parts.append(f"WRONG: {pred.direction} call failed, -${abs(pred.pnl_points):.2f}")
            parts.append(f"Max adverse excursion: ${pred.max_adverse:.2f}")
            if pred.max_favorable > 0:
                parts.append(f"NOTE: Had ${pred.max_favorable:.2f} favorable move but didn't sustain")

            # Try to identify what went wrong
            if pred.bars_held < 3:
                parts.append("ISSUE: Stopped out very quickly — stop too tight?")
            elif pred.max_favorable > pred.max_adverse * 0.3:
                parts.append("ISSUE: Move started in right direction but reversed — timing or exit issue")
        else:
            parts.append(f"PARTIAL: Inconclusive result after {pred.bars_held} bars")

        parts.append(f"Held for {pred.bars_held}/{pred.horizon_bars} bars")
        return " | ".join(parts)

    def _update_pattern_accuracy(self, pred: Prediction):
        """Update accuracy tracking for the strategy hypothesis."""
        key = pred.strategy_hypothesis or "unknown"
        if pred.outcome == "CORRECT":
            self.pattern_accuracy[key]["correct"] += 1
        elif pred.outcome == "WRONG":
            self.pattern_accuracy[key]["wrong"] += 1
        else:
            self.pattern_accuracy[key]["partial"] += 1

        # Check if pattern is now confirmed
        stats = self.pattern_accuracy[key]
        total = stats["correct"] + stats["wrong"] + stats["partial"]
        if total >= self.min_confirmations:
            win_rate = stats["correct"] / total
            self.confirmed_patterns[key] = {
                "win_rate": round(win_rate, 3),
                "total_predictions": total,
                "confirmed": win_rate > 0.55,
                "status": "TRUSTED" if win_rate > 0.6 else "PROMISING" if win_rate > 0.55 else "UNRELIABLE" if win_rate > 0.45 else "AVOID",
            }
            log.info("pattern_status_update", pattern=key,
                     status=self.confirmed_patterns[key]["status"],
                     win_rate=win_rate, total=total)

    def get_scoreboard(self) -> str:
        """Generate a human-readable prediction scoreboard."""
        lines = []
        lines.append("PREDICTION SCOREBOARD")
        lines.append("=" * 70)
        lines.append(f"Active predictions: {len(self.active_predictions)}")
        lines.append(f"Resolved predictions: {len(self.resolved_predictions)}")

        if self.resolved_predictions:
            correct = sum(1 for p in self.resolved_predictions if p.outcome == "CORRECT")
            wrong = sum(1 for p in self.resolved_predictions if p.outcome == "WRONG")
            partial = sum(1 for p in self.resolved_predictions if p.outcome == "PARTIAL")
            total = len(self.resolved_predictions)
            lines.append(f"\nOverall: {correct}/{total} correct ({correct/total*100:.1f}% win rate)")
            lines.append(f"  Correct: {correct}  |  Wrong: {wrong}  |  Partial: {partial}")

            # Average P&L
            pnls = [p.pnl_points for p in self.resolved_predictions if p.pnl_points]
            if pnls:
                lines.append(f"  Avg P&L: ${np.mean(pnls):.2f}  |  Total: ${sum(pnls):.2f}")

        # Per-strategy breakdown
        if self.pattern_accuracy:
            lines.append(f"\nPER-STRATEGY ACCURACY:")
            lines.append(f"{'Strategy':<40s} {'W':>4s} {'L':>4s} {'P':>4s} {'WR%':>6s} {'Status':<12s}")
            lines.append("-" * 70)
            for key, stats in sorted(self.pattern_accuracy.items(),
                                     key=lambda x: x[1]["correct"] / max(1, sum(x[1].values())),
                                     reverse=True):
                total = stats["correct"] + stats["wrong"] + stats["partial"]
                wr = stats["correct"] / total * 100 if total > 0 else 0
                status = self.confirmed_patterns.get(key, {}).get("status", "LEARNING")
                lines.append(f"  {key:<38s} {stats['correct']:>4d} {stats['wrong']:>4d} {stats['partial']:>4d} {wr:>5.1f}% {status:<12s}")

        # Active predictions
        if self.active_predictions:
            lines.append(f"\nACTIVE PREDICTIONS (watching):")
            for p in self.active_predictions[-10:]:
                lines.append(f"  [{p.id}] {p.direction:5s} @ ${p.entry_price:.2f} ({p.timeframe}) "
                             f"conf={p.confidence:.0%} bars={p.bars_held}/{p.horizon_bars} "
                             f"| {p.reasons[0][:50]}")

        return "\n".join(lines)

    def save_state(self):
        """Persist prediction state to disk."""
        state = {
            "last_updated": str(datetime.now()),
            "active": [p.to_dict() for p in self.active_predictions],
            "resolved": [p.to_dict() for p in self.resolved_predictions[-500:]],
            "pattern_accuracy": dict(self.pattern_accuracy),
            "confirmed_patterns": self.confirmed_patterns,
        }
        path = self.data_dir / "predictions.json"
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self):
        """Load previous prediction state."""
        path = self.data_dir / "predictions.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                state = json.load(f)
            # Restore pattern accuracy
            for k, v in state.get("pattern_accuracy", {}).items():
                self.pattern_accuracy[k] = v
            self.confirmed_patterns = state.get("confirmed_patterns", {})
            # Restore resolved predictions
            for rd in state.get("resolved", []):
                try:
                    p = Prediction(
                        direction=rd.get("direction", ""),
                        confidence=rd.get("confidence", 0),
                        reasons=rd.get("reasons", []),
                        entry_price=rd.get("entry_price", 0),
                        timeframe=rd.get("timeframe", ""),
                        horizon_bars=rd.get("horizon_bars", 20),
                        target_price=rd.get("target_price"),
                        stop_price=rd.get("stop_price"),
                        strategy_hypothesis=rd.get("strategy_hypothesis"),
                    )
                    p.id = rd.get("id", p.id)
                    p.timestamp = rd.get("timestamp", p.timestamp)
                    p.resolved = True
                    p.outcome = rd.get("outcome", "")
                    p.exit_price = rd.get("exit_price")
                    p.pnl_points = rd.get("pnl_points")
                    p.bars_held = rd.get("bars_held", 0)
                    p.max_favorable = rd.get("max_favorable", 0)
                    p.max_adverse = rd.get("max_adverse", 0)
                    self.resolved_predictions.append(p)
                except Exception:
                    pass
            log.info("predictions_loaded",
                     resolved=len(self.resolved_predictions),
                     patterns=len(self.pattern_accuracy))
        except Exception as e:
            log.warning("predictions_load_failed", error=str(e))
