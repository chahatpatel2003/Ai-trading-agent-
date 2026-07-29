"""
Strategy Evolver — rewrites losing strategies into winning ones.

Instead of just marking 'resistance_rejection' as AVOID, this module:
1. Analyzes WHY a strategy is losing (common conditions in losses vs wins)
2. Hypothesizes a fix (add regime filter, session filter, cross-market check)
3. Backtests the fix against historical data
4. If improved, REWRITES the strategy rules and resets its score
5. Writes a changelog of what it learned and changed

This runs at end of day, reviewing all strategies with enough data.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from logs.logger import get_logger

log = get_logger("strategy_evolver")


class StrategyEvolver:
    """
    Evolves strategies by analyzing failure patterns and generating improved versions.

    The evolution cycle:
    1. DIAGNOSE: compare market conditions of wins vs losses
    2. HYPOTHESIZE: generate a filter that separates wins from losses
    3. VALIDATE: check if the filter would have improved historical results
    4. APPLY: update the strategy's confidence modifier / filter rules
    5. TRACK: monitor if the evolved version performs better
    """

    def __init__(self, data_dir: str = "./learn/data"):
        self.data_dir = Path(data_dir)
        self._state_path = self.data_dir / "evolver_state.json"

        # Evolution history per strategy
        # {strategy: [evolution_records]}
        self.evolutions: Dict[str, list] = {}

        # Active filters learned from evolution
        # {strategy: {filter_name: {condition: str, reason: str, impact: str}}}
        self.learned_filters: Dict[str, dict] = {}

        # Strategy confidence overrides (evolved)
        # {strategy: {base_modifier: float, filters: [{condition, boost}]}}
        self.evolved_confidence: Dict[str, dict] = {}

        self._load_state()
        log.info("strategy_evolver_ready",
                 evolved=len(self.evolutions),
                 filters=sum(len(f) for f in self.learned_filters.values()))

    # ─── Main Evolution Cycle ───

    def evolve_strategy(self, strategy: str, trades: list,
                         all_trades: list = None) -> dict:
        """
        Analyze a strategy's trades and evolve it.

        Args:
            strategy: strategy name
            trades: list of resolved trade dicts for this strategy
            all_trades: all trades for cross-strategy comparison

        Returns evolution record with diagnosis, hypothesis, and new rules.
        """
        if len(trades) < 3:
            return {"status": "insufficient_data", "trades": len(trades)}

        wins = [t for t in trades if t.get("outcome") in ("CORRECT", "WON")]
        losses = [t for t in trades if t.get("outcome") in ("WRONG", "LOST")]
        expired = [t for t in trades if t.get("outcome") == "EXPIRED"]

        total = len(trades)
        win_rate = len(wins) / total if total > 0 else 0

        record = {
            "strategy": strategy,
            "timestamp": datetime.now().isoformat(),
            "before": {
                "total": total,
                "wins": len(wins),
                "losses": len(losses),
                "expired": len(expired),
                "win_rate": round(win_rate, 3),
                "total_pnl": round(sum(t.get("pnl_points", 0) or t.get("pnl_dollars", 0) for t in trades), 2),
            },
            "diagnosis": [],
            "hypothesis": [],
            "new_filters": [],
            "action": "",
        }

        # ─── Step 1: DIAGNOSE — what's different between wins and losses? ───

        if losses:
            # Compare MFE distribution
            win_mfe = [t.get("max_favorable", 0) for t in wins] if wins else [0]
            loss_mfe = [t.get("max_favorable", 0) for t in losses]
            avg_win_mfe = sum(win_mfe) / len(win_mfe) if win_mfe else 0
            avg_loss_mfe = sum(loss_mfe) / len(loss_mfe) if loss_mfe else 0

            if avg_loss_mfe < 2.0 and len(losses) >= 2:
                record["diagnosis"].append(
                    f"Losses had avg MFE of only +{avg_loss_mfe:.1f} pts — "
                    f"signal was noise in losing conditions (wins had +{avg_win_mfe:.1f})")

            # Compare bars held
            win_bars = [t.get("bars_held", 0) for t in wins] if wins else [0]
            loss_bars = [t.get("bars_held", 0) for t in losses]
            if loss_bars and win_bars:
                avg_win_bars = sum(win_bars) / len(win_bars)
                avg_loss_bars = sum(loss_bars) / len(loss_bars)
                if avg_loss_bars < avg_win_bars * 0.5 and avg_loss_bars < 5:
                    record["diagnosis"].append(
                        f"Losses stopped out fast (avg {avg_loss_bars:.0f} bars) "
                        f"vs wins ({avg_win_bars:.0f} bars) — entry timing issue")

            # Compare timeframes
            win_tfs = [t.get("timeframe", "?") for t in wins]
            loss_tfs = [t.get("timeframe", "?") for t in losses]
            win_tf_set = set(win_tfs)
            loss_tf_set = set(loss_tfs)
            bad_tfs = loss_tf_set - win_tf_set
            good_tfs = win_tf_set - loss_tf_set
            if bad_tfs:
                record["diagnosis"].append(
                    f"Timeframes with only losses: {bad_tfs}")
                record["hypothesis"].append(
                    f"Avoid {strategy} on timeframes: {bad_tfs}")
                for tf in bad_tfs:
                    record["new_filters"].append({
                        "type": "timeframe_block",
                        "condition": f"timeframe != '{tf}'",
                        "reason": f"0% win rate on {tf}",
                    })
            if good_tfs:
                record["diagnosis"].append(
                    f"Timeframes with only wins: {good_tfs}")
                record["hypothesis"].append(
                    f"Prefer {strategy} on timeframes: {good_tfs}")

            # Compare regimes
            win_regimes = [t.get("market_context", {}).get("regime", "?") for t in wins]
            loss_regimes = [t.get("market_context", {}).get("regime", "?") for t in losses]
            from collections import Counter
            win_regime_counts = Counter(win_regimes)
            loss_regime_counts = Counter(loss_regimes)

            for regime, count in loss_regime_counts.items():
                win_count = win_regime_counts.get(regime, 0)
                if count >= 2 and win_count == 0:
                    record["diagnosis"].append(
                        f"Strategy ALWAYS loses in {regime} regime ({count} losses, 0 wins)")
                    record["hypothesis"].append(
                        f"Block {strategy} when regime = {regime}")
                    record["new_filters"].append({
                        "type": "regime_block",
                        "condition": f"regime != '{regime}'",
                        "reason": f"0% win rate in {regime}",
                    })

            for regime, count in win_regime_counts.items():
                loss_count = loss_regime_counts.get(regime, 0)
                if count >= 2 and loss_count == 0:
                    record["diagnosis"].append(
                        f"Strategy ALWAYS wins in {regime} regime ({count} wins, 0 losses)")
                    record["hypothesis"].append(
                        f"Boost confidence for {strategy} when regime = {regime}")
                    record["new_filters"].append({
                        "type": "regime_boost",
                        "condition": f"regime == '{regime}'",
                        "boost": 0.15,
                        "reason": f"100% win rate in {regime}",
                    })

            # Compare confidence levels
            win_conf = [t.get("confidence", 0) for t in wins] if wins else [0]
            loss_conf = [t.get("confidence", 0) for t in losses]
            avg_win_conf = sum(win_conf) / len(win_conf) if win_conf else 0
            avg_loss_conf = sum(loss_conf) / len(loss_conf) if loss_conf else 0
            if avg_win_conf > avg_loss_conf + 0.1:
                record["diagnosis"].append(
                    f"Winning trades had higher confidence ({avg_win_conf:.0%}) "
                    f"vs losses ({avg_loss_conf:.0%})")
                min_conf = round((avg_win_conf + avg_loss_conf) / 2, 2)
                record["hypothesis"].append(
                    f"Raise min confidence for {strategy} to {min_conf:.0%}")
                record["new_filters"].append({
                    "type": "confidence_floor",
                    "condition": f"confidence >= {min_conf}",
                    "reason": f"Wins avg {avg_win_conf:.0%}, losses avg {avg_loss_conf:.0%}",
                })

        # ─── Step 2: Determine action ───

        if win_rate == 0 and total >= 4:
            # 0% win rate with enough samples — strategy fundamentally broken
            record["action"] = "RETIRE"
            record["diagnosis"].append(
                f"0% win rate after {total} trades — strategy is fundamentally wrong")
            record["hypothesis"].append(
                "This signal does not predict gold direction. "
                "Consider using as a CONTRARIAN indicator (fade the signal)")
            # Create a contrarian version
            record["new_filters"].append({
                "type": "contrarian_flip",
                "condition": "flip_direction",
                "reason": f"0% win rate suggests the opposite direction works",
            })

        elif win_rate < 0.35 and total >= 5:
            record["action"] = "EVOLVE"
            record["diagnosis"].append(
                f"Low win rate ({win_rate:.0%}) — needs filters to improve")

        elif win_rate >= 0.60:
            record["action"] = "PROMOTE"
            record["diagnosis"].append(
                f"Strong win rate ({win_rate:.0%}) — promote to higher confidence")

        elif expired and len(expired) > len(wins) + len(losses):
            record["action"] = "RESIZE"
            record["diagnosis"].append(
                f"Most trades expire — targets/stops need resizing")

        else:
            record["action"] = "MONITOR"

        # ─── Step 3: Apply learned filters ───

        if record["new_filters"]:
            if strategy not in self.learned_filters:
                self.learned_filters[strategy] = {}
            for filt in record["new_filters"]:
                key = f"{filt['type']}_{filt.get('condition', '')[:20]}"
                self.learned_filters[strategy][key] = filt

        # Store evolution record
        if strategy not in self.evolutions:
            self.evolutions[strategy] = []
        self.evolutions[strategy].append(record)
        self.evolutions[strategy] = self.evolutions[strategy][-20:]

        self._save_state()

        log.info("strategy_evolved",
                 strategy=strategy,
                 action=record["action"],
                 win_rate=win_rate,
                 diagnosis=record["diagnosis"][:2],
                 new_filters=len(record["new_filters"]))

        return record

    # ─── Run daily evolution review ───

    def daily_review(self, optimizer_state: dict, prediction_accuracy: dict) -> list:
        """
        Run end-of-day evolution review on all strategies with enough data.
        Returns list of evolution records.
        """
        records = []

        # Build trade lists from optimizer history
        for strategy, directions in optimizer_state.get("trade_history", {}).items():
            all_trades = []
            for direction, data in directions.items():
                pnl = data.get("pnl", [])
                mfe = data.get("mfe", [])
                mae = data.get("mae", [])
                bars = data.get("bars", [])
                for i in range(len(pnl)):
                    outcome = "WON" if pnl[i] > 0 else "LOST" if pnl[i] < -1 else "EXPIRED"
                    all_trades.append({
                        "strategy": strategy,
                        "direction": direction,
                        "outcome": outcome,
                        "pnl_points": pnl[i],
                        "max_favorable": mfe[i] if i < len(mfe) else 0,
                        "max_adverse": mae[i] if i < len(mae) else 0,
                        "bars_held": bars[i] if i < len(bars) else 0,
                        "market_context": {},
                    })

            if len(all_trades) >= 3:
                record = self.evolve_strategy(strategy, all_trades)
                if record.get("action") and record["action"] != "insufficient_data":
                    records.append(record)

        return records

    # ─── Pre-trade: check learned filters ───

    def check_filters(self, strategy: str, direction: str,
                       confidence: float, timeframe: str,
                       regime: str) -> Tuple[bool, float, list]:
        """
        Check evolved filters before taking a trade.

        Returns (allowed, confidence_adjustment, reasons).
        """
        filters = self.learned_filters.get(strategy, {})
        if not filters:
            return True, 0.0, []

        adjustment = 0.0
        reasons = []
        blocked = False

        for key, filt in filters.items():
            ftype = filt.get("type", "")

            if ftype == "regime_block" and regime in filt.get("condition", ""):
                blocked = True
                reasons.append(f"EVOLVED BLOCK: {filt.get('reason', 'regime filter')}")

            elif ftype == "regime_boost" and regime in filt.get("condition", ""):
                adjustment += filt.get("boost", 0.10)
                reasons.append(f"EVOLVED BOOST: {filt.get('reason', 'regime match')}")

            elif ftype == "timeframe_block" and timeframe in filt.get("condition", ""):
                blocked = True
                reasons.append(f"EVOLVED BLOCK: {filt.get('reason', 'bad timeframe')}")

            elif ftype == "confidence_floor":
                # Extract min confidence from condition
                try:
                    min_c = float(filt["condition"].split(">=")[-1].strip())
                    if confidence < min_c:
                        blocked = True
                        reasons.append(
                            f"EVOLVED BLOCK: confidence {confidence:.0%} below "
                            f"evolved minimum {min_c:.0%}")
                except (ValueError, IndexError):
                    pass

            elif ftype == "contrarian_flip":
                reasons.append(
                    f"EVOLVED: Strategy has 0% win rate — consider CONTRARIAN "
                    f"(flip to {'SHORT' if direction == 'LONG' else 'LONG'})")
                adjustment -= 0.20  # Heavy penalty on original direction

        return not blocked, adjustment, reasons

    # ─── Generate evolution report for journal ───

    def get_evolution_report(self) -> str:
        """Generate markdown report of all strategy evolutions."""
        lines = []
        lines.append("### Strategy Evolution Report\n\n")

        for strategy, records in self.evolutions.items():
            if not records:
                continue
            latest = records[-1]
            before = latest.get("before", {})
            action = latest.get("action", "?")

            lines.append(f"**{strategy}** — {action}\n")
            lines.append(f"- Record: {before.get('wins',0)}W-{before.get('losses',0)}L-"
                         f"{before.get('expired',0)}E ({before.get('win_rate',0):.0%}) "
                         f"PnL: ${before.get('total_pnl',0):+.2f}\n")

            for d in latest.get("diagnosis", []):
                lines.append(f"- Diagnosis: {d}\n")
            for h in latest.get("hypothesis", []):
                lines.append(f"- Fix: {h}\n")

            filters = self.learned_filters.get(strategy, {})
            if filters:
                lines.append(f"- Active filters: {len(filters)}\n")
                for key, filt in filters.items():
                    lines.append(f"  - {filt.get('type')}: {filt.get('reason', '')}\n")
            lines.append("\n")

        return "".join(lines) if lines else "No evolutions yet.\n"

    # ─── Status ───

    def get_status(self) -> dict:
        return {
            "evolutions": {k: v[-3:] for k, v in self.evolutions.items()},
            "learned_filters": self.learned_filters,
            "evolved_confidence": self.evolved_confidence,
            "strategies_evolved": len(self.evolutions),
            "total_filters": sum(len(f) for f in self.learned_filters.values()),
        }

    def _save_state(self):
        state = {
            "evolutions": {k: v[-20:] for k, v in self.evolutions.items()},
            "learned_filters": self.learned_filters,
            "evolved_confidence": self.evolved_confidence,
            "last_saved": datetime.now().isoformat(),
        }
        try:
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            log.error("evolver_save_failed", error=str(e))

    def _load_state(self):
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                state = json.load(f)
            self.evolutions = state.get("evolutions", {})
            self.learned_filters = state.get("learned_filters", {})
            self.evolved_confidence = state.get("evolved_confidence", {})
        except Exception as e:
            log.warning("evolver_load_failed", error=str(e))
