"""
Self-Optimizer — the feedback loop that makes the bot actually learn.

After each resolved trade, this module:
1. Analyzes WHY it won/lost (market conditions, news, regime)
2. Computes optimal stop/target from accumulated MFE/MAE data
3. Adjusts strategy parameters for next trade
4. Checks if a calendar event caused the loss
5. Maintains a memory of lessons that the bot reads before each trade

This is the difference between a bot that tracks outcomes and one that improves.
"""

import json
import math
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from logs.logger import get_logger

log = get_logger("self_optimizer")


class SelfOptimizer:
    """
    Feedback loop that turns trade outcomes into parameter improvements.

    After each trade resolves:
      1. Diagnose why it won/lost
      2. Check economic calendar for event coincidence
      3. Update optimal stop/target from MFE/MAE history
      4. Maintain strategy parameter overrides
      5. Write to self-improvement memory

    Before each new trade:
      1. Read parameter overrides (adjusted stops/targets)
      2. Check pre-trade blockers (recent loss in same conditions)
      3. Apply regime-aware confidence adjustment
    """

    def __init__(self, data_dir: str = "./learn/data"):
        self.data_dir = Path(data_dir)
        self._state_path = self.data_dir / "optimizer_state.json"

        # Per-strategy MFE/MAE history for parameter optimization
        # {strategy: {direction: {mfe: [list], mae: [list], pnl: [list], bars: [list]}}}
        self.trade_history: Dict[str, Dict[str, dict]] = {}

        # Computed optimal parameters per strategy
        # {strategy: {optimal_target: float, optimal_stop: float, optimal_horizon: int, ...}}
        self.optimal_params: Dict[str, dict] = {}

        # Pre-trade blockers learned from failures
        # {strategy: [{condition: str, reason: str, until: str}]}
        self.blockers: Dict[str, list] = {}

        # Session performance tracking
        self._session_stats: Dict[str, dict] = {}

        # Self-improvement log
        self.improvements: List[dict] = []

        self._load_state()
        log.info("self_optimizer_ready",
                 strategies=len(self.trade_history),
                 improvements=len(self.improvements))

    # ─── After Trade: Learn ───

    def learn_from_trade(self, trade: dict, calendar_events: list = None,
                         market_changes: dict = None,
                         correlations: dict = None,
                         cot_data: dict = None,
                         session: str = None) -> dict:
        """
        Full post-trade analysis. Called after every resolved trade.

        Now uses cross-market data to explain WHY:
          - market_changes: {dxy: +0.3%, oil: -2.4%, ...} at time of trade
          - correlations: gold's correlation with each market
          - cot_data: CFTC speculator positioning
          - session: which trading session (asian/london/ny)

        Returns a diagnosis dict with:
          - why: explanation of what happened
          - cross_market: which markets moved against/for us
          - adjustments: what parameters should change
          - event_impact: if a news event caused the move
          - lesson: what to remember for next time
        """
        strategy = trade.get("strategy", "unknown")
        direction = trade.get("direction", "?")
        outcome = trade.get("outcome", "?")
        entry = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        mfe = trade.get("max_favorable", 0)
        mae = abs(trade.get("max_adverse", 0))
        pnl = trade.get("pnl_points", 0) or trade.get("pnl_dollars", 0)
        bars = trade.get("bars_held", 0)
        target = trade.get("target_price", 0)
        stop = trade.get("stop_price", 0)
        target_dist = abs(target - entry) if target else 0
        stop_dist = abs(stop - entry) if stop else 0
        regime = trade.get("market_context", {}).get("regime", "unknown")
        reasons = trade.get("reasons", [])

        # 1. Record to history
        self._record_history(strategy, direction, mfe, mae, pnl, bars)

        diagnosis = {
            "strategy": strategy,
            "direction": direction,
            "outcome": outcome,
            "why": [],
            "adjustments": [],
            "event_impact": None,
            "lesson": "",
            "regime": regime,
        }

        # 2. Diagnose WHY
        if outcome == "WON":
            diagnosis["why"].append(f"Target hit at +{mfe:.1f} pts in {bars} bars")
            if mae > mfe * 0.5:
                diagnosis["why"].append(
                    f"BUT had significant drawdown (-{mae:.1f}) before winning — "
                    f"stop was nearly hit, trade was lucky")
                diagnosis["adjustments"].append(
                    f"Consider wider stop or smaller position for {strategy}")
            if bars <= 3:
                diagnosis["why"].append("Quick win — strong conviction signal")

        elif outcome == "LOST":
            if mfe < 2.0:
                diagnosis["why"].append(
                    f"Never moved in our favor (MFE only +{mfe:.1f} pts) — "
                    f"WRONG DIRECTION. Signal was noise, not signal.")
                diagnosis["lesson"] = f"{strategy} {direction} in {regime}: signal was noise"
            elif mfe > stop_dist * 0.3:
                diagnosis["why"].append(
                    f"Had +{mfe:.1f} favorable move but reversed and hit stop — "
                    f"RIGHT DIRECTION but bad exit timing")
                diagnosis["adjustments"].append(
                    f"Move stop to breakeven after +{mfe*0.5:.0f} pts")
                diagnosis["lesson"] = f"{strategy}: add trailing stop after +{mfe*0.5:.0f} pts"
            if bars <= 3:
                diagnosis["why"].append(
                    f"Stopped out in {bars} bars — stop too tight or entry timing off")
                diagnosis["adjustments"].append(
                    f"Widen stop by 20% for {strategy}")

        elif outcome == "EXPIRED":
            if target_dist > 0:
                pct_reached = (mfe / target_dist * 100) if target_dist > 0 else 0
                if pct_reached < 30:
                    diagnosis["why"].append(
                        f"Target was {target_dist:.0f} pts away but MFE only +{mfe:.1f} "
                        f"({pct_reached:.0f}% of target) — TARGET UNREACHABLE for this timeframe")
                    diagnosis["adjustments"].append(
                        f"Reduce target to {mfe * 0.7:.0f} pts (70% of avg MFE)")
                elif pct_reached > 60:
                    diagnosis["why"].append(
                        f"Got to {pct_reached:.0f}% of target (+{mfe:.1f}/{target_dist:.0f}) "
                        f"but couldn't close — needs tighter target or trailing stop")
                    diagnosis["adjustments"].append(
                        f"Add take-profit at {mfe * 0.8:.0f} pts")

            if bars >= 15 and abs(pnl) < 3:
                diagnosis["why"].append(
                    "Held for max bars with near-zero PnL — no edge in this setup")

        # 3. Check calendar events
        if calendar_events:
            for event in calendar_events:
                event_date = event.get("date", "")
                days_away = event.get("days_away", 99)
                if days_away <= 1:
                    diagnosis["event_impact"] = (
                        f"{event.get('event','?')} on {event_date} likely caused volatility — "
                        f"{event.get('gold_bias', '')}")
                    if outcome in ("LOST", "EXPIRED"):
                        diagnosis["why"].append(
                            f"NEWS IMPACT: {event.get('event','?')} released — "
                            f"should have reduced position or stayed flat")
                        self._add_blocker(strategy,
                                          f"Avoid {strategy} within 24h of {event.get('event','?')}",
                                          reason=f"Lost during {event.get('event','?')}")

        # 4. Cross-market analysis — what ELSE was happening?
        diagnosis["cross_market"] = []
        if market_changes and outcome in ("LOST", "WRONG"):
            # Find which markets moved against our position
            gold_chg = market_changes.get("gold", 0)
            dxy_chg = market_changes.get("dxy", 0)
            oil_chg = market_changes.get("wti_oil", 0)
            vix_chg = market_changes.get("vix", 0)
            us10y_chg = market_changes.get("us10y", 0)
            silver_chg = market_changes.get("silver", 0)
            copper_chg = market_changes.get("copper", 0)

            if direction == "LONG":
                # We went long but lost — what pushed gold down?
                if dxy_chg > 0.2:
                    diagnosis["cross_market"].append(
                        f"DXY rose +{dxy_chg:.2f}% (headwind for LONG gold)")
                    diagnosis["why"].append(f"DXY strengthened +{dxy_chg:.2f}% — pushed gold down")
                if us10y_chg > 0.3:
                    diagnosis["cross_market"].append(
                        f"US 10Y yields rose +{us10y_chg:.2f}% (bearish for gold)")
                    diagnosis["why"].append(f"Rising yields (+{us10y_chg:.2f}%) killed the LONG")
                if oil_chg < -1.0:
                    diagnosis["cross_market"].append(
                        f"Oil crashed {oil_chg:+.2f}% (deflation signal)")
            else:
                # We went short but lost — what pushed gold up?
                if dxy_chg < -0.2:
                    diagnosis["cross_market"].append(
                        f"DXY fell {dxy_chg:+.2f}% (tailwind for gold, headwind for SHORT)")
                    diagnosis["why"].append(f"DXY weakened {dxy_chg:+.2f}% — gold rallied against our SHORT")
                if vix_chg > 1.0:
                    diagnosis["cross_market"].append(
                        f"VIX spiked +{vix_chg:.1f}% (fear buying into gold)")
                    diagnosis["why"].append(f"VIX spike +{vix_chg:.1f}% triggered safe-haven buying")
                if us10y_chg < -0.3:
                    diagnosis["cross_market"].append(
                        f"Yields fell {us10y_chg:+.2f}% (bullish for gold)")

            if not diagnosis["cross_market"]:
                diagnosis["cross_market"].append("No clear cross-market driver — internal gold move")

        # 5. COT analysis — were we fighting the smart money?
        if cot_data and outcome in ("LOST", "WRONG"):
            spec_net = cot_data.get("non_commercial_net", 0)
            if direction == "LONG" and spec_net < -10000:
                diagnosis["why"].append(
                    f"COT: Speculators net SHORT {spec_net:,} contracts — we went LONG against smart money")
                diagnosis["lesson"] = f"Don't go LONG when specs are heavily short ({spec_net:,})"
            elif direction == "SHORT" and spec_net > 50000:
                diagnosis["why"].append(
                    f"COT: Speculators net LONG {spec_net:,} contracts — we went SHORT against the crowd")

        # 6. Session analysis — did we trade in the right session?
        if session and outcome in ("LOST", "WRONG"):
            # Track session performance
            sess_key = f"{strategy}_{session}"
            if sess_key not in self._session_stats:
                self._session_stats[sess_key] = {"wins": 0, "losses": 0}
            self._session_stats[sess_key]["losses"] += 1
            total_sess = self._session_stats[sess_key]["wins"] + self._session_stats[sess_key]["losses"]
            if total_sess >= 3:
                wr = self._session_stats[sess_key]["wins"] / total_sess
                if wr < 0.3:
                    diagnosis["why"].append(
                        f"Poor performance in {session} session "
                        f"({self._session_stats[sess_key]['wins']}W-{self._session_stats[sess_key]['losses']}L)")
                    diagnosis["adjustments"].append(
                        f"Consider avoiding {strategy} during {session} session")
        elif session and outcome in ("WON", "CORRECT"):
            sess_key = f"{strategy}_{session}"
            if sess_key not in self._session_stats:
                self._session_stats[sess_key] = {"wins": 0, "losses": 0}
            self._session_stats[sess_key]["wins"] += 1

        # 7. Recompute optimal parameters
        self._recompute_optimal(strategy, direction)

        # 8. Store improvement
        if diagnosis["why"] or diagnosis["adjustments"] or diagnosis["cross_market"]:
            improvement = {
                "timestamp": datetime.now().isoformat(),
                "strategy": strategy,
                "outcome": outcome,
                "diagnosis": diagnosis["why"],
                "cross_market": diagnosis["cross_market"],
                "adjustments": diagnosis["adjustments"],
                "lesson": diagnosis["lesson"],
                "regime": regime,
            }
            self.improvements.append(improvement)
            self.improvements = self.improvements[-200:]

        self._save_state()

        log.info("self_optimizer_learned",
                 strategy=strategy, outcome=outcome,
                 why=diagnosis["why"][:2],
                 cross_market=diagnosis["cross_market"][:2],
                 adjustments=diagnosis["adjustments"][:2])

        return diagnosis

    # ─── Before Trade: Advise ───

    def get_optimal_params(self, strategy: str, direction: str = None) -> dict:
        """
        Return optimized parameters for a strategy based on historical MFE/MAE.

        Returns {optimal_target, optimal_stop, optimal_horizon, sample_size}
        or empty dict if not enough data.
        """
        return self.optimal_params.get(strategy, {})

    def check_blockers(self, strategy: str) -> Optional[str]:
        """Check if strategy has any active pre-trade blockers."""
        blockers = self.blockers.get(strategy, [])
        now = datetime.now().isoformat()
        active = [b for b in blockers if b.get("until", "") > now]
        if active:
            return active[0].get("condition", "blocked")
        return None

    def get_pre_trade_advice(self, strategy: str, direction: str,
                              regime: str, market_changes: dict = None,
                              cot_data: dict = None, session: str = None,
                              calendar_events: list = None) -> dict:
        """
        Get advice before taking a trade, using ALL available intelligence.
        """
        advice = {
            "take_trade": True,
            "adjusted_target": None,
            "adjusted_stop": None,
            "adjusted_horizon": None,
            "warnings": [],
            "confidence_modifier": 0.0,
        }

        # Check blockers
        blocker = self.check_blockers(strategy)
        if blocker:
            advice["take_trade"] = False
            advice["warnings"].append(f"BLOCKED: {blocker}")
            return advice

        # Apply optimal parameters (FROZEN until 50+ trades per strategy)
        params = self.optimal_params.get(strategy, {})
        if params.get("sample_size", 0) >= 5:  # Unfrozen: 5 trade minimum for paper trading mandate: freeze until 50+ trades
            advice["adjusted_target"] = params.get("optimal_target")
            advice["adjusted_stop"] = params.get("optimal_stop")
            advice["adjusted_horizon"] = params.get("optimal_horizon")
            advice["warnings"].append(
                f"Using optimized params: target={params.get('optimal_target', '?'):.1f} "
                f"stop={params.get('optimal_stop', '?'):.1f} "
                f"horizon={params.get('optimal_horizon', '?')} "
                f"(from {params['sample_size']} trades)")
        elif params.get("sample_size", 0) >= 3:
            # FROZEN: log what we WOULD change but don't apply
            advice["warnings"].append(
                f"FROZEN optimizer (need 50+ trades, have {params['sample_size']}): "
                f"WOULD use target={params.get('optimal_target', '?'):.1f} "
                f"stop={params.get('optimal_stop', '?'):.1f}")

        # Check recent losses in same regime
        history = self.trade_history.get(strategy, {}).get(direction, {})
        recent_pnl = (history.get("pnl", []) or [])[-3:]
        if len(recent_pnl) >= 3 and all(p < 0 for p in recent_pnl):
            advice["confidence_modifier"] -= 0.15
            advice["warnings"].append(
                "3 consecutive losses for this strategy+direction — reducing confidence")

        # Cross-market headwind check
        if market_changes:
            dxy_chg = market_changes.get("dxy", 0)
            vix_chg = market_changes.get("vix", 0)
            us10y_chg = market_changes.get("us10y", 0)

            if direction == "LONG":
                headwinds = []
                if dxy_chg > 0.3:
                    headwinds.append(f"DXY rising +{dxy_chg:.2f}%")
                if us10y_chg > 0.5:
                    headwinds.append(f"Yields rising +{us10y_chg:.2f}%")
                if headwinds:
                    advice["confidence_modifier"] -= 0.10
                    advice["warnings"].append(
                        f"Cross-market headwind for LONG: {', '.join(headwinds)}")
            elif direction == "SHORT":
                headwinds = []
                if dxy_chg < -0.3:
                    headwinds.append(f"DXY falling {dxy_chg:+.2f}%")
                if vix_chg > 2.0:
                    headwinds.append(f"VIX spiking +{vix_chg:.1f}% (safe-haven bid)")
                if headwinds:
                    advice["confidence_modifier"] -= 0.10
                    advice["warnings"].append(
                        f"Cross-market headwind for SHORT: {', '.join(headwinds)}")

        # COT alignment check
        if cot_data:
            spec_net = cot_data.get("non_commercial_net", 0)
            if direction == "LONG" and spec_net < -20000:
                advice["confidence_modifier"] -= 0.10
                advice["warnings"].append(
                    f"COT warning: Specs heavily SHORT ({spec_net:,}) — going LONG against smart money")
            elif direction == "SHORT" and spec_net > 80000:
                advice["confidence_modifier"] -= 0.05
                advice["warnings"].append(
                    f"COT warning: Specs heavily LONG ({spec_net:,}) — contrarian SHORT may work but risky")
            elif direction == "LONG" and spec_net > 50000:
                advice["confidence_modifier"] += 0.05
                advice["warnings"].append(
                    f"COT tailwind: Specs LONG ({spec_net:,}) — aligned with smart money")

        # Calendar event proximity check
        if calendar_events:
            for event in calendar_events:
                if event.get("days_away", 99) <= 1 and event.get("impact") == "HIGH":
                    advice["confidence_modifier"] -= 0.15
                    advice["warnings"].append(
                        f"HIGH impact event imminent: {event.get('event')} in {event.get('days_away')}d — "
                        f"reduce exposure")

        # Session performance check
        if session:
            sess_key = f"{strategy}_{session}"
            sess = self._session_stats.get(sess_key, {})
            total = sess.get("wins", 0) + sess.get("losses", 0)
            if total >= 3:
                wr = sess.get("wins", 0) / total
                if wr < 0.3:
                    advice["confidence_modifier"] -= 0.10
                    advice["warnings"].append(
                        f"Poor session performance: {strategy} in {session} = "
                        f"{sess['wins']}W-{sess['losses']}L ({wr:.0%})")

        return advice

    # ─── Internal: Parameter Optimization ───

    def _record_history(self, strategy, direction, mfe, mae, pnl, bars):
        if strategy not in self.trade_history:
            self.trade_history[strategy] = {}
        if direction not in self.trade_history[strategy]:
            self.trade_history[strategy][direction] = {
                "mfe": [], "mae": [], "pnl": [], "bars": []
            }
        h = self.trade_history[strategy][direction]
        h["mfe"].append(round(mfe, 2))
        h["mae"].append(round(abs(mae), 2))
        h["pnl"].append(round(pnl, 2))
        h["bars"].append(bars)
        # Keep last 50
        for k in h:
            h[k] = h[k][-50:]

    def _recompute_optimal(self, strategy, direction):
        """Compute optimal stop/target/horizon from MFE/MAE distribution."""
        h = self.trade_history.get(strategy, {}).get(direction, {})
        mfe_list = h.get("mfe", [])
        mae_list = h.get("mae", [])
        bars_list = h.get("bars", [])
        n = len(mfe_list)

        if n < 3:
            return

        # Optimal target: 70th percentile of MFE (achievable most of the time)
        sorted_mfe = sorted(mfe_list)
        p70_idx = int(n * 0.7)
        optimal_target = sorted_mfe[min(p70_idx, n - 1)]

        # Optimal stop: 90th percentile of MAE (rarely exceeded)
        sorted_mae = sorted(mae_list)
        p90_idx = int(n * 0.9)
        optimal_stop = sorted_mae[min(p90_idx, n - 1)]

        # Ensure minimum 1:2 R:R
        if optimal_stop > 0 and optimal_target / optimal_stop < 2.0:
            optimal_target = optimal_stop * 2.0

        # Optimal horizon: median bars of winning trades
        winning_bars = [b for b, p in zip(bars_list, h.get("pnl", [])) if p > 0]
        optimal_horizon = int(sorted(winning_bars)[len(winning_bars) // 2]) if winning_bars else 20

        # Minimum floors
        optimal_target = max(optimal_target, 3.0)
        optimal_stop = max(optimal_stop, 2.0)
        optimal_horizon = max(optimal_horizon, 5)

        self.optimal_params[strategy] = {
            "optimal_target": round(optimal_target, 1),
            "optimal_stop": round(optimal_stop, 1),
            "optimal_horizon": optimal_horizon,
            "avg_mfe": round(sum(mfe_list) / n, 1),
            "avg_mae": round(sum(mae_list) / n, 1),
            "sample_size": n,
            "last_updated": datetime.now().isoformat(),
        }

        log.info("optimal_params_computed",
                 strategy=strategy, direction=direction,
                 target=optimal_target, stop=optimal_stop,
                 horizon=optimal_horizon, samples=n)

    def _add_blocker(self, strategy, condition, reason, hours=24):
        if strategy not in self.blockers:
            self.blockers[strategy] = []
        until = (datetime.now() + timedelta(hours=hours)).isoformat()
        self.blockers[strategy].append({
            "condition": condition,
            "reason": reason,
            "until": until,
            "created": datetime.now().isoformat(),
        })
        # Keep only active blockers
        now = datetime.now().isoformat()
        self.blockers[strategy] = [
            b for b in self.blockers[strategy] if b["until"] > now
        ][-5:]

    # ─── Status ───

    def get_status(self) -> dict:
        return {
            "optimal_params": self.optimal_params,
            "blockers": {k: [b for b in v if b.get("until", "") > datetime.now().isoformat()]
                         for k, v in self.blockers.items()},
            "improvements": self.improvements[-20:],
            "strategies_optimized": len(self.optimal_params),
            "total_improvements": len(self.improvements),
        }

    # ─── Persistence ───

    def _save_state(self):
        state = {
            "trade_history": self.trade_history,
            "optimal_params": self.optimal_params,
            "blockers": self.blockers,
            "session_stats": self._session_stats,
            "improvements": self.improvements[-200:],
            "last_saved": datetime.now().isoformat(),
        }
        try:
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            log.error("optimizer_save_failed", error=str(e))

    def _load_state(self):
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                state = json.load(f)
            self.trade_history = state.get("trade_history", {})
            self.optimal_params = state.get("optimal_params", {})
            self.blockers = state.get("blockers", {})
            self._session_stats = state.get("session_stats", {})
            self.improvements = state.get("improvements", [])
            log.info("optimizer_state_loaded",
                     strategies=len(self.trade_history),
                     improvements=len(self.improvements))
        except Exception as e:
            log.warning("optimizer_load_failed", error=str(e))
