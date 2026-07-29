"""
Unified Brain — ONE decision maker that sees EVERYTHING.

Instead of 12 separate strategies each deciding independently,
this brain looks at ALL available data and makes ONE call:

  STRONG SHORT / SHORT / NEUTRAL / LONG / STRONG LONG

Every input votes with a weight. The brain only trades when
the signal is decisive — mixed signals = no trade.

Inputs scored:
  1. Higher TF trend (4H/1D)         — heaviest weight
  2. Cross-market pressure            — DXY, yields, oil, copper, VIX
  3. Multi-TF indicator confluence    — RSI, MACD, BB, ADX across all TFs
  4. ICT concepts                     — FVG, OTE, kill zone, liquidity
  5. Regime                           — risk_on, deflation_fear, etc.
  6. COT positioning                  — smart money direction
  7. Order flow                       — OANDA buy/sell pressure
  8. Volume                           — spike detection, OBV trend
  9. Calendar                         — reduce size near NFP/FOMC
  10. Session                         — London/NY preferred
  11. Pattern history                 — what the bot has observed working

Output: ONE direction, ONE confidence, ONE stop, ONE target.
Prediction and paper trade go the SAME direction. Always.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from logs.logger import get_logger

log = get_logger("unified_brain")


@dataclass
class BrainDecision:
    """The single output of the unified brain."""
    direction: str          # LONG, SHORT, or FLAT
    confidence: float       # 0.0 to 1.0
    score: float            # -10 to +10 (negative = short, positive = long)
    entry_price: float
    stop_price: float
    target_price: float
    reasons: List[str]      # why this direction
    against: List[str]      # what argues against (transparency)
    votes: Dict[str, float] # each input's vote
    strength: str           # STRONG, MODERATE, WEAK, FLAT


class UnifiedBrain:
    """
    Looks at everything. Makes one call.
    """

    def __init__(self):
        self.last_decision = None
        self.decision_history = []

    def decide(self,
               all_data: dict,              # {tf: DataFrame}
               market_changes: dict = None, # {market: pct_change}
               regime: str = "",
               gold_pressure: float = 0,
               ict_result: dict = None,
               cot_data: dict = None,
               order_flow: dict = None,
               calendar_events: list = None,
               seasonal: dict = None,
               pattern_counts: dict = None,
               correlations: dict = None,   # RMT-cleaned gold correlations
               lead_lag: dict = None,       # lead-lag relationships
               news_sentiment: dict = None, # live news sentiment
               ) -> BrainDecision:
        """
        Score all inputs, make one decision.
        Returns BrainDecision with direction, confidence, and full reasoning.
        """
        votes = {}
        reasons_long = []
        reasons_short = []

        # Get current price
        price_df = all_data.get("1h")
        if price_df is None or (hasattr(price_df, 'empty') and price_df.empty):
            price_df = all_data.get("5m")
        if price_df is None or len(price_df) < 20:
            return self._flat("No price data")
        current_price = float(price_df["close"].iloc[-1])

        # ═══════════════════════════════════════════════════════
        # 1. HIGHER TIMEFRAME TREND (weight: 3.0) — most important
        # ═══════════════════════════════════════════════════════
        htf_vote = 0
        for tf_name, tf_weight in [("1d", 1.5), ("4h", 1.0), ("1h", 0.5)]:
            df = all_data.get(tf_name)
            if df is not None and len(df) > 50:
                close = float(df["close"].iloc[-1])
                sma20 = float(df["close"].iloc[-20:].mean())
                sma50 = float(df["close"].iloc[-50:].mean())
                if close > sma20 > sma50:
                    htf_vote += tf_weight
                elif close < sma20 < sma50:
                    htf_vote -= tf_weight
        votes["htf_trend"] = round(htf_vote, 2)
        if htf_vote > 1.0:
            reasons_long.append(f"Higher TFs trending UP (score {htf_vote:+.1f})")
        elif htf_vote < -1.0:
            reasons_short.append(f"Higher TFs trending DOWN (score {htf_vote:+.1f})")

        # ═══════════════════════════════════════════════════════
        # 2. CROSS-MARKET PRESSURE (weight: 2.0)
        # ═══════════════════════════════════════════════════════
        xm_vote = 0
        if market_changes:
            dxy = market_changes.get("dxy", 0)
            us10y = market_changes.get("us10y", 0)
            oil = market_changes.get("wti_oil", 0)
            copper = market_changes.get("copper", 0)
            vix = market_changes.get("vix", 0)
            silver = market_changes.get("silver", 0)

            # DXY inverse
            if dxy > 0.3: xm_vote -= 0.5; reasons_short.append(f"DXY rising +{dxy:.2f}%")
            elif dxy < -0.3: xm_vote += 0.5; reasons_long.append(f"DXY falling {dxy:+.2f}%")
            # Yields inverse
            if us10y > 0.3: xm_vote -= 0.4; reasons_short.append(f"Yields rising +{us10y:.2f}%")
            elif us10y < -0.3: xm_vote += 0.4; reasons_long.append(f"Yields falling {us10y:+.2f}%")
            # Oil correlation (deflation signal)
            if oil < -1.5: xm_vote -= 0.3; reasons_short.append(f"Oil crashing {oil:+.1f}%")
            elif oil > 1.5: xm_vote += 0.2
            # Copper (growth fears)
            if copper < -1.0: xm_vote -= 0.3; reasons_short.append(f"Copper falling {copper:+.1f}%")
            elif copper > 1.0: xm_vote += 0.2
            # VIX (fear)
            if vix > 2.0: xm_vote += 0.3; reasons_long.append(f"VIX spiking +{vix:.1f}% (fear)")
            elif vix < -2.0: xm_vote -= 0.2
            # Silver confirmation
            if silver < -1.0 and xm_vote < 0: xm_vote -= 0.2
            elif silver > 1.0 and xm_vote > 0: xm_vote += 0.2

        # Gold pressure score
        if gold_pressure > 20: xm_vote += 0.5; reasons_long.append(f"Gold pressure bullish ({gold_pressure:+.0f})")
        elif gold_pressure < -20: xm_vote -= 0.5; reasons_short.append(f"Gold pressure bearish ({gold_pressure:+.0f})")
        votes["cross_market"] = round(xm_vote, 2)

        # ═══════════════════════════════════════════════════════
        # 3. MULTI-TF INDICATOR CONFLUENCE (weight: 2.0)
        # ═══════════════════════════════════════════════════════
        indicator_vote = 0
        tf_bullish = 0
        tf_bearish = 0
        for tf_name in ["5m", "15m", "1h", "4h"]:
            df = all_data.get(tf_name)
            if df is None or len(df) < 50:
                continue
            close = df["close"]
            c = float(close.iloc[-1])

            # Simple scoring per TF
            tf_score = 0
            # EMA alignment
            ema8 = float(close.ewm(span=8).mean().iloc[-1])
            ema21 = float(close.ewm(span=21).mean().iloc[-1])
            if ema8 > ema21: tf_score += 0.3
            else: tf_score -= 0.3
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi_val = float((100 - 100 / (1 + rs)).iloc[-1]) if not rs.iloc[-1] != rs.iloc[-1] else 50
            if rsi_val > 60: tf_score += 0.2
            elif rsi_val < 40: tf_score -= 0.2
            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_hist = float((ema12 - ema26).iloc[-1] - (ema12 - ema26).ewm(span=9).mean().iloc[-1])
            if macd_hist > 0: tf_score += 0.2
            else: tf_score -= 0.2

            indicator_vote += tf_score * 0.5  # 0.5 per TF
            if tf_score > 0: tf_bullish += 1
            else: tf_bearish += 1

        if tf_bullish >= 3:
            reasons_long.append(f"Indicators bullish on {tf_bullish}/4 TFs")
        elif tf_bearish >= 3:
            reasons_short.append(f"Indicators bearish on {tf_bearish}/4 TFs")
        votes["indicators"] = round(indicator_vote, 2)

        # ═══════════════════════════════════════════════════════
        # 4. ICT CONCEPTS (weight: 1.5)
        # ═══════════════════════════════════════════════════════
        ict_vote = 0
        if ict_result:
            conf = ict_result.get("confluence", {})
            ict_dir = conf.get("direction", "NEUTRAL")
            ict_score = conf.get("score", 0)
            if ict_dir == "LONG" and ict_score >= 2:
                ict_vote = min(1.5, ict_score * 0.3)
                reasons_long.append(f"ICT confluence LONG (score {ict_score})")
            elif ict_dir == "SHORT" and ict_score >= 2:
                ict_vote = -min(1.5, ict_score * 0.3)
                reasons_short.append(f"ICT confluence SHORT (score {ict_score})")

            # Kill zone bonus
            if ict_result.get("in_kill_zone"):
                ict_vote *= 1.3  # 30% more weight during kill zone
                (reasons_long if ict_vote > 0 else reasons_short).append(
                    f"In {ict_result.get('kill_zone', {}).get('name', 'kill zone')}")

            # Liquidity sweep = strong signal
            for sweep in ict_result.get("sweeps", []):
                if sweep.get("signal") == "LONG":
                    ict_vote += 1.0; reasons_long.append(sweep["description"])
                elif sweep.get("signal") == "SHORT":
                    ict_vote -= 1.0; reasons_short.append(sweep["description"])

            # OTE zone
            for ote_key in ("ote_1h", "ote_4h"):
                ote = ict_result.get(ote_key)
                if ote and ote.get("in_zone"):
                    if ote["direction"] == "LONG":
                        ict_vote += 0.5; reasons_long.append(f"In bullish OTE zone ({ote_key})")
                    else:
                        ict_vote -= 0.5; reasons_short.append(f"In bearish OTE zone ({ote_key})")

        votes["ict"] = round(ict_vote, 2)

        # ═══════════════════════════════════════════════════════
        # 5. REGIME (weight: 1.0)
        # ═══════════════════════════════════════════════════════
        regime_vote = 0
        if regime == "risk_off":
            regime_vote = 0.5; reasons_long.append("Regime: risk_off (safe-haven gold)")
        elif regime == "deflation_fear":
            regime_vote = -0.5; reasons_short.append("Regime: deflation_fear (everything falls)")
        elif regime == "inflation_fear":
            regime_vote = 0.5; reasons_long.append("Regime: inflation_fear (gold hedge)")
        votes["regime"] = regime_vote

        # ═══════════════════════════════════════════════════════
        # 6. COT POSITIONING (weight: 0.8)
        # ═══════════════════════════════════════════════════════
        cot_vote = 0
        if cot_data:
            spec_net = cot_data.get("non_commercial_net", 0)
            if spec_net > 50000:
                cot_vote = 0.5; reasons_long.append(f"COT: Specs net long {spec_net:,}")
            elif spec_net < -10000:
                cot_vote = -0.5; reasons_short.append(f"COT: Specs net short {spec_net:,}")
        votes["cot"] = cot_vote

        # ═══════════════════════════════════════════════════════
        # 7. ORDER FLOW (weight: 0.8)
        # ═══════════════════════════════════════════════════════
        flow_vote = 0
        if order_flow:
            pressure = order_flow.get("order_pressure_score", 0)
            if pressure > 0.3:
                flow_vote = 0.4; reasons_long.append(f"Order flow: buying pressure ({pressure:+.2f})")
            elif pressure < -0.3:
                flow_vote = -0.4; reasons_short.append(f"Order flow: selling pressure ({pressure:+.2f})")
            if order_flow.get("extreme_alert"):
                rec = order_flow.get("recommendation", "")
                if "SHORT" in rec:
                    flow_vote -= 0.3; reasons_short.append("Order flow: extreme positioning (contrarian SHORT)")
        votes["order_flow"] = round(flow_vote, 2)

        # ═══════════════════════════════════════════════════════
        # 8. VOLUME (weight: 0.5)
        # ═══════════════════════════════════════════════════════
        vol_vote = 0
        df_1h = all_data.get("1h")
        if df_1h is not None and "volume" in df_1h.columns and len(df_1h) > 20:
            avg_vol = float(df_1h["volume"].iloc[-20:].mean())
            recent_vol = float(df_1h["volume"].iloc[-1])
            if avg_vol > 0 and recent_vol > avg_vol * 2:
                # Volume spike — confirms the current direction
                if float(df_1h["close"].iloc[-1]) > float(df_1h["open"].iloc[-1]):
                    vol_vote = 0.3; reasons_long.append("Volume spike on bullish candle")
                else:
                    vol_vote = -0.3; reasons_short.append("Volume spike on bearish candle")
        votes["volume"] = vol_vote

        # ═══════════════════════════════════════════════════════
        # 9. CALENDAR (modifier, not directional)
        # ═══════════════════════════════════════════════════════
        calendar_factor = 1.0
        if calendar_events:
            for event in calendar_events:
                if event.get("days_away", 99) <= 1 and event.get("impact") == "HIGH":
                    calendar_factor = 0.5  # reduce confidence by half near high-impact events
                    reasons_long.append(f"CAUTION: {event['event']} imminent")
                    reasons_short.append(f"CAUTION: {event['event']} imminent")

        # ═══════════════════════════════════════════════════════
        # 10. PATTERN HISTORY (weight: 0.5)
        # ═══════════════════════════════════════════════════════
        pattern_vote = 0
        if pattern_counts:
            bear_conf = pattern_counts.get("multi_tf_confluence_bear", 0)
            bull_conf = pattern_counts.get("multi_tf_confluence_bull", 0)
            if bear_conf > bull_conf * 5:
                pattern_vote = -0.3; reasons_short.append(f"Pattern history: {bear_conf} bear vs {bull_conf} bull confluences")
            elif bull_conf > bear_conf * 5:
                pattern_vote = 0.3
        votes["patterns"] = pattern_vote

        # ═══════════════════════════════════════════════════════
        # 11. RMT CORRELATIONS — use cleaned correlations to validate moves
        # ═══════════════════════════════════════════════════════
        corr_vote = 0
        if correlations and market_changes:
            # If a highly-correlated market is moving, gold should follow
            # If a highly-inverse market is moving, gold should go opposite
            for mkt, gold_corr in correlations.items():
                if not isinstance(gold_corr, (int, float)) or mkt == "gold":
                    continue
                mkt_change = market_changes.get(mkt, 0)
                if abs(gold_corr) > 0.5 and abs(mkt_change) > 0.5:
                    # Strong correlation + significant move = signal
                    expected_gold_dir = 1 if (gold_corr * mkt_change) > 0 else -1
                    weight = min(0.3, abs(gold_corr) * abs(mkt_change) * 0.1)
                    corr_vote += expected_gold_dir * weight
                    if abs(weight) > 0.1:
                        dir_word = "UP" if expected_gold_dir > 0 else "DOWN"
                        (reasons_long if expected_gold_dir > 0 else reasons_short).append(
                            f"RMT: {mkt} moved {mkt_change:+.1f}% (corr {gold_corr:+.2f}) → gold should go {dir_word}")
        votes["rmt_correlations"] = round(corr_vote, 2)

        # ═══════════════════════════════════════════════════════
        # 12. LEAD-LAG — if a leading market moved yesterday, gold follows today
        # ═══════════════════════════════════════════════════════
        leadlag_vote = 0
        if lead_lag and market_changes:
            for mkt, lag_info in lead_lag.items():
                if not isinstance(lag_info, dict):
                    continue
                lag = lag_info.get("best_lag", 0)
                corr_strength = abs(lag_info.get("correlation", 0))
                if lag > 0 and corr_strength > 0.3:
                    # This market LEADS gold — its move predicts gold's next move
                    mkt_change = market_changes.get(mkt, 0)
                    if abs(mkt_change) > 0.5:
                        lag_corr = lag_info.get("correlation", 0)
                        expected = 1 if (lag_corr * mkt_change) > 0 else -1
                        weight = min(0.4, corr_strength * 0.5)
                        leadlag_vote += expected * weight
                        dir_word = "UP" if expected > 0 else "DOWN"
                        (reasons_long if expected > 0 else reasons_short).append(
                            f"Lead-lag: {mkt} (leads +{lag}d) moved {mkt_change:+.1f}% → gold should go {dir_word}")
        votes["lead_lag"] = round(leadlag_vote, 2)

        # ═══════════════════════════════════════════════════════
        # 13. MOMENTUM (ADX + trend strength) — weight: 1.0
        # ═══════════════════════════════════════════════════════
        momentum_vote = 0
        for tf_name in ["1h", "4h"]:
            df = all_data.get(tf_name)
            if df is None or len(df) < 30:
                continue
            close = df["close"]
            high = df["high"]
            low = df["low"]
            # ADX calculation (simplified)
            tr = pd.concat([high - low,
                           abs(high - close.shift(1)),
                           abs(low - close.shift(1))], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean()
            # Directional movement
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
            minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)
            plus_di = 100 * plus_dm.rolling(14).mean() / atr14
            minus_di = 100 * minus_dm.rolling(14).mean() / atr14
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx_val = float(dx.rolling(14).mean().iloc[-1]) if not np.isnan(dx.rolling(14).mean().iloc[-1]) else 20
            plus_di_val = float(plus_di.iloc[-1]) if not np.isnan(plus_di.iloc[-1]) else 50
            minus_di_val = float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 50

            if adx_val > 25:  # strong trend
                if plus_di_val > minus_di_val:
                    momentum_vote += 0.4; reasons_long.append(f"ADX={adx_val:.0f} strong bullish trend on {tf_name}")
                else:
                    momentum_vote -= 0.4; reasons_short.append(f"ADX={adx_val:.0f} strong bearish trend on {tf_name}")
        votes["momentum"] = round(momentum_vote, 2)

        # ═══════════════════════════════════════════════════════
        # 14. ADVANCED VOLUME (OBV trend + tick volume)
        # ═══════════════════════════════════════════════════════
        adv_vol_vote = 0
        for tf_name in ["15m", "1h"]:
            df = all_data.get(tf_name)
            if df is None or len(df) < 30 or "volume" not in df.columns:
                continue
            # OBV trend
            obv = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
            obv_sma = obv.rolling(20).mean()
            if float(obv.iloc[-1]) > float(obv_sma.iloc[-1]):
                adv_vol_vote += 0.2  # accumulation
            else:
                adv_vol_vote -= 0.2  # distribution
            # Volume trend (increasing on moves = confirmed)
            vol_sma = df["volume"].rolling(20).mean()
            recent_vol = float(df["volume"].iloc[-3:].mean())
            avg_vol = float(vol_sma.iloc[-1]) if not np.isnan(vol_sma.iloc[-1]) else 1
            if avg_vol > 0 and recent_vol > avg_vol * 1.5:
                # High volume — confirms the direction of recent candles
                recent_dir = float(df["close"].iloc[-1]) - float(df["close"].iloc[-3])
                if recent_dir > 0:
                    adv_vol_vote += 0.2; reasons_long.append(f"High volume on bullish move ({tf_name})")
                else:
                    adv_vol_vote -= 0.2; reasons_short.append(f"High volume on bearish move ({tf_name})")
        votes["adv_volume"] = round(adv_vol_vote, 2)

        # ═══════════════════════════════════════════════════════
        # 15. SEASONAL BIAS
        # ═══════════════════════════════════════════════════════
        seasonal_vote = 0
        if seasonal:
            composite = seasonal.get("composite_score", 0)
            direction = seasonal.get("direction", "neutral")
            if direction == "bullish" and composite > 0.3:
                seasonal_vote = 0.2; reasons_long.append(f"Seasonal: bullish bias ({composite:.2f})")
            elif direction == "bearish" and composite < -0.3:
                seasonal_vote = -0.2; reasons_short.append(f"Seasonal: bearish bias ({composite:.2f})")
        votes["seasonal"] = seasonal_vote

        # ═══════════════════════════════════════════════════════
        # 16. NEWS SENTIMENT (if available)
        # ═══════════════════════════════════════════════════════
        news_vote = 0
        if news_sentiment:
            sentiment = news_sentiment.get("sentiment", 0)
            headlines = news_sentiment.get("headlines", [])
            if sentiment > 0.3:
                news_vote = 0.5; reasons_long.append(f"News: bullish sentiment ({sentiment:+.2f})")
            elif sentiment < -0.3:
                news_vote = -0.5; reasons_short.append(f"News: bearish sentiment ({sentiment:+.2f})")
            for h in headlines[:2]:
                (reasons_long if sentiment > 0 else reasons_short).append(f"Headline: {h}")
        votes["news"] = news_vote

        # ═══════════════════════════════════════════════════════
        # TOTAL SCORE
        # ═══════════════════════════════════════════════════════
        total_score = sum(votes.values())
        total_score *= calendar_factor

        # Determine direction and strength
        if total_score >= 4.0:
            direction, strength = "LONG", "STRONG"
            confidence = min(0.95, 0.60 + abs(total_score) * 0.04)
        elif total_score >= 2.0:
            direction, strength = "LONG", "MODERATE"
            confidence = min(0.85, 0.55 + abs(total_score) * 0.04)
        elif total_score <= -4.0:
            direction, strength = "SHORT", "STRONG"
            confidence = min(0.95, 0.60 + abs(total_score) * 0.04)
        elif total_score <= -2.0:
            direction, strength = "SHORT", "MODERATE"
            confidence = min(0.85, 0.55 + abs(total_score) * 0.04)
        elif abs(total_score) <= 1.0:
            direction, strength = "FLAT", "FLAT"
            confidence = 0
        else:
            direction = "LONG" if total_score > 0 else "SHORT"
            strength = "WEAK"
            confidence = 0.45 + abs(total_score) * 0.03

        # Only trade MODERATE or STRONG
        if strength in ("WEAK", "FLAT"):
            return self._flat(
                f"Mixed signals (score {total_score:+.1f}) — waiting for clarity",
                votes=votes, reasons_long=reasons_long, reasons_short=reasons_short)

        # Compute stop and target from ATR
        atr_df = all_data.get("1h")
        if atr_df is None: atr_df = all_data.get("4h")
        if atr_df is not None and len(atr_df) > 14:
            atr_val = float((atr_df["high"] - atr_df["low"]).iloc[-14:].mean())
        else:
            atr_val = 15.0  # fallback

        if direction == "LONG":
            stop_price = current_price - atr_val * 1.5
            target_price = current_price + atr_val * 1.5 * 2.5  # R:R 1:2.5
            reasons = reasons_long
            against = reasons_short
        else:
            stop_price = current_price + atr_val * 1.5
            target_price = current_price - atr_val * 1.5 * 2.5
            reasons = reasons_short
            against = reasons_long

        decision = BrainDecision(
            direction=direction,
            confidence=round(confidence, 3),
            score=round(total_score, 2),
            entry_price=current_price,
            stop_price=round(stop_price, 2),
            target_price=round(target_price, 2),
            reasons=reasons[:8],
            against=against[:4],
            votes=votes,
            strength=strength,
        )

        self.last_decision = decision
        self.decision_history.append({
            "timestamp": datetime.now().isoformat(),
            "direction": direction,
            "score": total_score,
            "confidence": confidence,
            "strength": strength,
            "price": current_price,
        })
        self.decision_history = self.decision_history[-200:]

        log.info("brain_decision",
                 direction=direction, strength=strength,
                 score=round(total_score, 2),
                 confidence=round(confidence, 3),
                 votes={k: v for k, v in votes.items() if v != 0})

        return decision

    def _flat(self, reason, votes=None, reasons_long=None, reasons_short=None):
        return BrainDecision(
            direction="FLAT",
            confidence=0,
            score=0,
            entry_price=0,
            stop_price=0,
            target_price=0,
            reasons=[reason],
            against=[],
            votes=votes or {},
            strength="FLAT",
        )
