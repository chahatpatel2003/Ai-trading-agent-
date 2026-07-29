"""
Market Thinker — the synthesis layer that makes the bot actually think.

Instead of processing each signal independently, this module connects:
  - Market moves -> trade outcomes (WHY did gold move?)
  - Regime -> strategy performance (WHICH strategy works WHEN?)
  - Past failures -> parameter adjustments (ADAPT from mistakes)
  - Session analysis -> actionable lessons (WHAT to do differently)

This is what separates a signal generator from a trader.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from logs.logger import get_logger

log = get_logger("market_thinker")


class MarketThinker:
    """
    Synthesizes market data into actionable intelligence.

    Runs after each cycle to:
    1. Explain price moves using cross-market data
    2. Map strategy performance to market conditions
    3. Adjust confidence dynamically based on regime alignment
    4. Generate analytical session summaries
    5. Learn and accumulate wisdom over time
    """

    def __init__(self, data_dir: str = "./learn/data"):
        self.data_dir = Path(data_dir)
        self._state_path = self.data_dir / "thinker_state.json"

        # Strategy-regime performance matrix
        # {strategy: {regime: {wins: N, losses: N, expired: N, total_pnl: float}}}
        self.strategy_regime_map: Dict[str, Dict[str, dict]] = {}

        # Strategy parameter adjustments learned from outcomes
        # {strategy: {adjustment: str, reason: str, applied: bool}}
        self.strategy_adjustments: Dict[str, list] = {}

        # Accumulated lessons (persisted across restarts)
        self.lessons: List[dict] = []

        # Today's market narrative
        self.narratives: List[str] = []

        # Confidence adjustments per strategy per regime
        # {strategy: {regime: confidence_modifier}}
        self.confidence_modifiers: Dict[str, Dict[str, float]] = {}

        self._load_state()
        log.info("market_thinker_ready",
                 lessons=len(self.lessons),
                 strategies_tracked=len(self.strategy_regime_map))

    # ─── 1. Market Narrative — WHY did price move? ───

    def explain_price_move(self, gold_change_pct: float,
                           market_changes: Dict[str, float],
                           regime: str, gold_pressure: float) -> str:
        """
        Generate a causal narrative for today's gold move by
        comparing cross-market moves.
        """
        parts = []
        gold_dir = "up" if gold_change_pct > 0 else "down"
        parts.append(f"Gold moved {gold_dir} {abs(gold_change_pct):.2f}%.")

        # Find the biggest movers and explain causality
        drivers = []

        dxy_chg = market_changes.get("dxy", 0)
        if abs(dxy_chg) > 0.2:
            if (dxy_chg > 0 and gold_change_pct < 0) or (dxy_chg < 0 and gold_change_pct > 0):
                drivers.append(f"DXY {'rose' if dxy_chg > 0 else 'fell'} {abs(dxy_chg):.2f}% "
                               f"(inverse correlation confirmed)")
            elif (dxy_chg > 0 and gold_change_pct > 0):
                drivers.append(f"DXY rose {dxy_chg:.2f}% but gold ALSO rose — "
                               f"unusual, suggests inflation/safe-haven demand overriding dollar strength")

        us10y_chg = market_changes.get("us10y", 0)
        tips_chg = market_changes.get("tips", 0)
        if abs(us10y_chg) > 0.3:
            real_yield_move = us10y_chg - tips_chg
            if real_yield_move > 0 and gold_change_pct < 0:
                drivers.append(f"Real yields rose (10Y +{us10y_chg:.2f}%) — bearish for gold")
            elif real_yield_move < 0 and gold_change_pct > 0:
                drivers.append(f"Real yields fell — bullish for gold")

        oil_chg = max(abs(market_changes.get("wti_oil", 0)), abs(market_changes.get("brent_oil", 0)))
        oil_dir = market_changes.get("wti_oil", 0)
        if abs(oil_dir) > 1.0:
            if oil_dir < 0 and gold_change_pct < 0:
                drivers.append(f"Oil collapsed {oil_dir:+.2f}% — deflation signal, drags gold")
            elif oil_dir > 0 and gold_change_pct > 0:
                drivers.append(f"Oil surged {oil_dir:+.2f}% — inflation fear lifts gold")

        vix_chg = market_changes.get("vix", 0)
        if abs(vix_chg) > 1.0:
            if vix_chg > 0 and gold_change_pct > 0:
                drivers.append(f"VIX spiked +{vix_chg:.1f}% — fear buying into gold")
            elif vix_chg > 0 and gold_change_pct < 0:
                drivers.append(f"VIX rose +{vix_chg:.1f}% but gold fell — "
                               f"risk-off selling everything including gold")

        silver_chg = market_changes.get("silver", 0)
        if abs(silver_chg) > 1.0:
            if abs(silver_chg) > abs(gold_change_pct) * 1.5:
                drivers.append(f"Silver moved {silver_chg:+.2f}% (stronger than gold) — "
                               f"industrial demand component diverging")

        copper_chg = market_changes.get("copper", 0)
        if abs(copper_chg) > 1.0:
            if copper_chg < -1.0:
                drivers.append(f"Copper dropped {copper_chg:+.2f}% — growth fears")

        sp500_chg = market_changes.get("sp500", 0)
        if abs(sp500_chg) > 0.5:
            if sp500_chg > 0 and gold_change_pct > 0:
                drivers.append(f"S&P500 +{sp500_chg:.2f}% and gold up — risk-on with inflation hedge")
            elif sp500_chg < 0 and gold_change_pct > 0:
                drivers.append(f"S&P500 {sp500_chg:+.2f}% — flight to gold safety")

        if drivers:
            parts.append("Drivers: " + "; ".join(drivers))
        else:
            parts.append("No clear cross-market driver — internally driven move")

        parts.append(f"Regime: {regime} | Gold pressure: {gold_pressure:+.1f}")

        narrative = " ".join(parts)
        self.narratives.append(narrative)
        return narrative

    # ─── 2. Strategy-Regime Learning — WHICH works WHEN? ───

    def record_strategy_outcome(self, strategy: str, regime: str,
                                 outcome: str, pnl: float,
                                 market_context: dict = None):
        """Record a trade outcome mapped to regime for learning."""
        if strategy not in self.strategy_regime_map:
            self.strategy_regime_map[strategy] = {}
        if regime not in self.strategy_regime_map[strategy]:
            self.strategy_regime_map[strategy][regime] = {
                "wins": 0, "losses": 0, "expired": 0,
                "total_pnl": 0.0, "trades": 0
            }

        bucket = self.strategy_regime_map[strategy][regime]
        bucket["trades"] += 1
        bucket["total_pnl"] += pnl
        if outcome == "WON":
            bucket["wins"] += 1
        elif outcome == "LOST":
            bucket["losses"] += 1
        else:
            bucket["expired"] += 1

        # Recompute confidence modifier
        self._update_confidence_modifier(strategy, regime)
        self._save_state()

    def _update_confidence_modifier(self, strategy: str, regime: str):
        """
        Compute a confidence boost/penalty based on how this strategy
        performs in this regime.

        +0.15 = strategy thrives in this regime (boost confidence)
        -0.20 = strategy fails in this regime (penalize confidence)
         0.00 = not enough data yet
        """
        bucket = self.strategy_regime_map.get(strategy, {}).get(regime, {})
        total = bucket.get("trades", 0)
        if total < 3:
            return  # not enough data

        wins = bucket.get("wins", 0)
        losses = bucket.get("losses", 0)
        avg_pnl = bucket.get("total_pnl", 0) / total if total > 0 else 0

        # Use PnL-based assessment, not just win count
        # Expired trades with positive PnL count as "profitable, not losing"
        actual_losses = losses  # only count actual stop-outs as losses
        loss_rate = actual_losses / total if total > 0 else 0

        if strategy not in self.confidence_modifiers:
            self.confidence_modifiers[strategy] = {}

        if avg_pnl > 0 and loss_rate < 0.3:
            self.confidence_modifiers[strategy][regime] = 0.10  # profitable overall
        elif avg_pnl > 0:
            self.confidence_modifiers[strategy][regime] = 0.0   # profitable but risky
        elif loss_rate > 0.5:
            self.confidence_modifiers[strategy][regime] = -0.15  # mostly losing
        elif avg_pnl < 0:
            self.confidence_modifiers[strategy][regime] = -0.10  # unprofitable
        else:
            self.confidence_modifiers[strategy][regime] = 0.0

        log.info("confidence_modifier_updated",
                 strategy=strategy, regime=regime,
                 modifier=self.confidence_modifiers[strategy][regime],
                 win_rate=round(win_rate, 2), trades=total)

    def get_adjusted_confidence(self, strategy: str, regime: str,
                                 base_confidence: float) -> float:
        """Return confidence adjusted by regime performance."""
        modifier = self.confidence_modifiers.get(strategy, {}).get(regime, 0.0)
        adjusted = max(0.0, min(1.0, base_confidence + modifier))
        if modifier != 0:
            log.info("confidence_adjusted",
                     strategy=strategy, regime=regime,
                     base=round(base_confidence, 2),
                     modifier=modifier,
                     adjusted=round(adjusted, 2))
        return adjusted

    # ─── 3. Adaptive Learning — what to adjust ───

    def analyze_expired_trades(self, resolved_trades: list) -> list:
        """
        Analyze expired trades and generate parameter adjustment suggestions.
        If trades keep expiring, the target/stop don't match the timeframe.
        """
        adjustments = []

        # Group by strategy
        by_strategy = defaultdict(list)
        for t in resolved_trades:
            by_strategy[t.get("strategy", "unknown")].append(t)

        for strategy, trades in by_strategy.items():
            expired = [t for t in trades if t.get("outcome") == "EXPIRED"]
            total = len(trades)
            expire_rate = len(expired) / total if total > 0 else 0

            if expire_rate > 0.6 and len(expired) >= 3:
                # Most trades expire — targets too far or stops too wide
                avg_pnl = sum(t.get("pnl_points", 0) for t in expired) / len(expired)
                avg_mfe = sum(t.get("max_favorable", 0) for t in expired) / len(expired)
                avg_mae = sum(abs(t.get("max_adverse", 0)) for t in expired) / len(expired)

                adj = {
                    "strategy": strategy,
                    "issue": f"{len(expired)}/{total} trades expired ({expire_rate:.0%})",
                    "diagnosis": [],
                    "suggestion": [],
                    "timestamp": datetime.now().isoformat(),
                }

                if avg_mfe > 0 and avg_pnl > 0:
                    adj["diagnosis"].append(
                        f"Trades are profitable on avg (+{avg_pnl:.1f} pts) but MFE was +{avg_mfe:.1f} — "
                        f"target could have been tighter to capture the move")
                    adj["suggestion"].append(
                        f"Reduce target to ~{avg_mfe * 0.7:.0f} pts (70% of avg MFE)")

                if avg_mae > avg_mfe * 0.8:
                    adj["diagnosis"].append(
                        f"MAE ({avg_mae:.1f}) nearly as large as MFE ({avg_mfe:.1f}) — "
                        f"stop may be too wide, allowing too much pain")
                    adj["suggestion"].append(
                        f"Tighten stop to ~{avg_mae * 1.2:.0f} pts")

                if avg_mfe < 5 and total >= 5:
                    adj["diagnosis"].append(
                        f"Very small favorable moves (avg MFE +{avg_mfe:.1f} pts) — "
                        f"this strategy may work better as a FILTER, not an entry signal")
                    adj["suggestion"].append(
                        "Demote to filter: use as confluence confirmation, not standalone entry")

                adjustments.append(adj)

                if strategy not in self.strategy_adjustments:
                    self.strategy_adjustments[strategy] = []
                self.strategy_adjustments[strategy].append(adj)

                log.info("strategy_adjustment_suggested",
                         strategy=strategy,
                         issue=adj["issue"],
                         suggestions=adj["suggestion"])

        return adjustments

    # ─── 4. Analytical Session Summary ───

    def generate_session_analysis(self, session: str,
                                  resolved_trades: list,
                                  market_changes: Dict[str, float],
                                  regime: str,
                                  gold_change_pct: float,
                                  knowledge: dict = None) -> str:
        """Generate an analytical session summary — not just a trade list."""
        lines = []
        now = datetime.utcnow().strftime("%H:%M UTC")
        lines.append(f"### {session.upper()} Analysis ({now})\n")

        # Market narrative
        narrative = self.explain_price_move(
            gold_change_pct, market_changes, regime, 0)
        lines.append(f"**Market**: {narrative}\n")

        # Session character
        ss = (knowledge or {}).get("session_stats", {}).get(session.split("_")[0], {})
        if ss:
            avg_range = ss.get("avg_range", 0)
            lines.append(f"**Session character**: avg range ${avg_range:.2f}\n")

        # Trade analysis
        session_trades = [t for t in resolved_trades
                          if self._in_session(t.get("exit_time", ""), session)]
        if session_trades:
            wins = [t for t in session_trades if t.get("outcome") == "WON"]
            losses = [t for t in session_trades if t.get("outcome") == "LOST"]
            expired = [t for t in session_trades if t.get("outcome") == "EXPIRED"]

            lines.append(f"**Trades**: {len(wins)}W-{len(losses)}L-{len(expired)}E\n")

            # WHY did wins work?
            for t in wins:
                ctx = t.get("market_context", {})
                lines.append(f"- WIN: {t['strategy']} {t['direction']} "
                             f"+${t.get('pnl_dollars',0):.2f} — "
                             f"regime was {ctx.get('regime', '?')}, "
                             f"pressure {ctx.get('gold_pressure', '?')}\n")

            # WHY did losses fail?
            for t in losses:
                ctx = t.get("market_context", {})
                mfe = t.get("max_favorable", 0)
                mae = abs(t.get("max_adverse", 0))
                lines.append(f"- LOSS: {t['strategy']} {t['direction']} "
                             f"-${abs(t.get('pnl_dollars',0)):.2f} — ")
                if mfe > mae * 0.3:
                    lines.append(f"  Had +{mfe:.1f} favorable move but reversed. "
                                 f"Exit timing issue.\n")
                else:
                    lines.append(f"  Never moved in our favor (MFE +{mfe:.1f}). "
                                 f"Wrong direction call.\n")

            # Expired analysis
            if len(expired) > len(wins) + len(losses):
                lines.append(f"- WARNING: {len(expired)} expired trades — "
                             f"targets may be too ambitious for this session\n")
        else:
            lines.append("**Trades**: None resolved this session\n")

        # Regime-strategy recommendation
        lines.append(f"\n**Regime insight ({regime})**: ")
        best = self._best_strategy_for_regime(regime)
        worst = self._worst_strategy_for_regime(regime)
        if best:
            lines.append(f"Best performer: {best[0]} "
                         f"({best[1]:.0%} WR, ${best[2]:+.2f} PnL)\n")
        if worst:
            lines.append(f"Worst performer: {worst[0]} "
                         f"({worst[1]:.0%} WR, ${worst[2]:+.2f} PnL) — avoid in this regime\n")

        return "".join(lines)

    # ─── 5. Daily Lessons — what to remember ───

    def generate_daily_lessons(self, resolved_trades: list,
                                regime: str, balance: float) -> list:
        """Generate actionable lessons from today's results."""
        lessons = []

        # Analyze expired trades
        adjustments = self.analyze_expired_trades(resolved_trades)
        for adj in adjustments:
            lessons.append({
                "type": "parameter",
                "lesson": f"{adj['strategy']}: {'; '.join(adj['suggestion'])}",
                "evidence": adj["issue"],
                "timestamp": datetime.now().isoformat(),
            })

        # Regime lesson
        best = self._best_strategy_for_regime(regime)
        worst = self._worst_strategy_for_regime(regime)
        if best and worst:
            lessons.append({
                "type": "regime",
                "lesson": f"In {regime}: favor {best[0]}, avoid {worst[0]}",
                "evidence": f"Best WR={best[1]:.0%}, Worst WR={worst[1]:.0%}",
                "timestamp": datetime.now().isoformat(),
            })

        # Conflict lesson: did we have LONG+SHORT on same day?
        directions = set(t.get("direction") for t in resolved_trades)
        if "LONG" in directions and "SHORT" in directions:
            long_pnl = sum(t.get("pnl_dollars", 0) for t in resolved_trades
                           if t.get("direction") == "LONG")
            short_pnl = sum(t.get("pnl_dollars", 0) for t in resolved_trades
                            if t.get("direction") == "SHORT")
            winner = "LONG" if long_pnl > short_pnl else "SHORT"
            lessons.append({
                "type": "conflict",
                "lesson": f"Mixed signals today: LONG P&L=${long_pnl:+.2f}, SHORT P&L=${short_pnl:+.2f}. "
                          f"{winner} was correct — commit to one direction per regime",
                "evidence": f"regime={regime}",
                "timestamp": datetime.now().isoformat(),
            })

        self.lessons.extend(lessons)
        self._save_state()
        return lessons

    # ─── Internal ───

    def _best_strategy_for_regime(self, regime):
        results = []
        for strat, regimes in self.strategy_regime_map.items():
            data = regimes.get(regime, {})
            total = data.get("trades", 0)
            if total < 3:
                continue
            wr = data.get("wins", 0) / total
            pnl = data.get("total_pnl", 0)
            results.append((strat, wr, pnl, total))
        if not results:
            return None
        return max(results, key=lambda x: (x[1], x[2]))

    def _worst_strategy_for_regime(self, regime):
        results = []
        for strat, regimes in self.strategy_regime_map.items():
            data = regimes.get(regime, {})
            total = data.get("trades", 0)
            if total < 3:
                continue
            wr = data.get("wins", 0) / total
            pnl = data.get("total_pnl", 0)
            results.append((strat, wr, pnl, total))
        if not results:
            return None
        return min(results, key=lambda x: (x[1], x[2]))

    def _in_session(self, timestamp: str, session: str) -> bool:
        try:
            h = datetime.fromisoformat(timestamp).hour
            sessions = {"asian": (0, 8), "london": (8, 16), "new_york": (16, 21)}
            s, e = sessions.get(session, (0, 24))
            return s <= h < e
        except Exception:
            return True

    def get_status(self) -> dict:
        return {
            "strategy_regime_map": self.strategy_regime_map,
            "confidence_modifiers": self.confidence_modifiers,
            "strategy_adjustments": {k: v[-3:] for k, v in self.strategy_adjustments.items()},
            "lessons": self.lessons[-20:],
            "narratives": self.narratives[-5:],
            "strategies_tracked": len(self.strategy_regime_map),
        }

    def _save_state(self):
        state = {
            "strategy_regime_map": self.strategy_regime_map,
            "confidence_modifiers": self.confidence_modifiers,
            "strategy_adjustments": {k: v[-10:] for k, v in self.strategy_adjustments.items()},
            "lessons": self.lessons[-100:],
            "last_saved": datetime.now().isoformat(),
        }
        try:
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            log.error("thinker_save_failed", error=str(e))

    def _load_state(self):
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                state = json.load(f)
            self.strategy_regime_map = state.get("strategy_regime_map", {})
            self.confidence_modifiers = state.get("confidence_modifiers", {})
            self.strategy_adjustments = state.get("strategy_adjustments", {})
            self.lessons = state.get("lessons", [])
            log.info("thinker_state_loaded",
                     lessons=len(self.lessons),
                     strategies=len(self.strategy_regime_map))
        except Exception as e:
            log.warning("thinker_load_failed", error=str(e))
