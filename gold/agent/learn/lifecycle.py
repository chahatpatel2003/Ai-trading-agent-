"""
Strategy Lifecycle Manager -- Enforces strict trading discipline and manages
the promotion / demotion pipeline for every strategy.

Lifecycle phases:
    HYPOTHESIS -> BACKTEST -> PAPER -> PROMOTE_REVIEW -> LIVE -> MONITORING -> DEMOTED / RETIRED

Hard limits on trade count, drawdown, confidence, and risk-reward are enforced
before any signal is allowed to become a trade.
"""
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from learn.journal import TradeJournal
from logs.logger import get_logger

log = get_logger("learn.lifecycle")

# ============================================================================
# Hard limits -- DO NOT relax these without explicit review
# ============================================================================

MAX_TRADES_PER_DAY = 4              # Total (mental + paper + live)
MAX_LIVE_TRADES_PER_DAY = 4         # Real money only
MAX_CONCURRENT_POSITIONS = 2
MAX_DAILY_DRAWDOWN_PCT = 6.0        # $60 on $1000 = 6%
MAX_WEEKLY_DRAWDOWN_PCT = 12.0
MIN_CONFIDENCE_TO_TRADE = 0.71
MIN_RR_RATIO = 2.5                  # 1:2.5 minimum

# ============================================================================
# Phase definitions and promotion criteria
# ============================================================================

PHASES = [
    "HYPOTHESIS",
    "BACKTEST",
    "PAPER",
    "PROMOTE_REVIEW",
    "LIVE",
    "MONITORING",
    "DEMOTED",
    "RETIRED",
]

# Promotion thresholds -- keyed by the *source* phase
PROMOTION_CRITERIA = {
    "HYPOTHESIS": {
        # Automatic -- just needs to be defined
    },
    "BACKTEST": {
        "win_rate": 0.55,
        "profit_factor": 1.3,
        "min_years_tested": 2,
    },
    "PAPER": {
        "win_rate": 0.60,
        "rr_ratio": 1.5,
        "min_signals": 20,
        "max_consecutive_losses": 5,
    },
    "PROMOTE_REVIEW": {
        # Same criteria as PAPER, plus a manual approval flag
        "win_rate": 0.60,
        "rr_ratio": 1.5,
        "min_signals": 20,
        "max_consecutive_losses": 5,
        "manual_approval": True,
    },
    "LIVE": {
        "auto_after_trades": 10,  # auto-promote to MONITORING after 10 live trades
    },
}

# Demotion thresholds
DEMOTION_CRITERIA = {
    "MONITORING": {
        "win_rate_floor": 0.55,
        "min_trades_to_judge": 50,
    },
    "DEMOTED": {
        "win_rate_floor": 0.50,
        "min_trades_to_judge": 50,  # 50 *more* trades after demotion
    },
}


class StrategyLifecycle:
    """
    Enforces trading discipline and manages strategy promotion / demotion.

    Parameters
    ----------
    journal : TradeJournal
        Shared journal instance for reading and writing trade data.
    data_dir : str
        Directory for auxiliary persistence (counters, approvals, etc.).
    """

    def __init__(self, journal: TradeJournal, data_dir: str = "./learn/data"):
        self.journal = journal
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Daily / session counters  (reset by daily_reset)
        self._daily_trade_count = 0
        self._daily_live_count = 0
        self._current_positions = 0
        self._daily_pnl_pct = 0.0
        self._weekly_pnl_pct = 0.0

        # Manual approval flags for PROMOTE_REVIEW -> LIVE
        self._manual_approvals: set[str] = set()

        # Per-strategy trade counters for demotion tracking
        self._strategy_trade_counts: dict[str, int] = defaultdict(int)
        self._strategy_trades_since_demotion: dict[str, int] = defaultdict(int)

        self._state_path = self.data_dir / "lifecycle_state.json"
        self._load_state()

        log.info("strategy_lifecycle_ready",
                 daily_trades=self._daily_trade_count,
                 positions=self._current_positions)

    # ------------------------------------------------------------------
    # Trade-gate checks
    # ------------------------------------------------------------------

    def can_trade_today(self, current_count: int | None = None) -> bool:
        """
        Return True if under the daily trade limit.

        Parameters
        ----------
        current_count : int, optional
            Override; if provided it is compared directly against the limit.
            Otherwise the internal counter is used.
        """
        count = current_count if current_count is not None else self._daily_trade_count
        allowed = count < MAX_TRADES_PER_DAY
        if not allowed:
            log.warning("daily_trade_limit_reached", count=count,
                        limit=MAX_TRADES_PER_DAY)
        return allowed

    def check_discipline(
        self,
        confidence: float,
        rr_ratio: float,
        current_positions: int | None = None,
        daily_pnl_pct: float | None = None,
        weekly_pnl_pct: float | None = None,
    ) -> tuple[bool, str]:
        """
        Full discipline check before taking a trade.

        Returns (passed, reason).
        """
        positions = current_positions if current_positions is not None else self._current_positions
        dpnl = daily_pnl_pct if daily_pnl_pct is not None else self._daily_pnl_pct
        wpnl = weekly_pnl_pct if weekly_pnl_pct is not None else self._weekly_pnl_pct

        # 1. Confidence
        if confidence < MIN_CONFIDENCE_TO_TRADE:
            reason = (f"Confidence {confidence:.2f} below minimum "
                      f"{MIN_CONFIDENCE_TO_TRADE:.2f}")
            log.info("discipline_rejected", check="confidence", reason=reason)
            return False, reason

        # 2. Risk-reward
        if rr_ratio < MIN_RR_RATIO:
            reason = f"R:R ratio {rr_ratio:.2f} below minimum {MIN_RR_RATIO:.2f}"
            log.info("discipline_rejected", check="rr_ratio", reason=reason)
            return False, reason

        # 3. Concurrent positions
        if positions >= MAX_CONCURRENT_POSITIONS:
            reason = (f"Already {positions} concurrent position(s); "
                      f"max is {MAX_CONCURRENT_POSITIONS}")
            log.info("discipline_rejected", check="positions", reason=reason)
            return False, reason

        # 4. Daily trade limit
        if not self.can_trade_today():
            reason = (f"Daily trade limit reached ({self._daily_trade_count}/"
                      f"{MAX_TRADES_PER_DAY})")
            return False, reason

        # 5. Daily drawdown
        if dpnl <= -MAX_DAILY_DRAWDOWN_PCT:
            reason = (f"Daily drawdown {dpnl:.2f}% exceeds limit "
                      f"-{MAX_DAILY_DRAWDOWN_PCT:.1f}%")
            log.warning("discipline_rejected", check="daily_drawdown",
                        reason=reason)
            return False, reason

        # 6. Weekly drawdown
        if wpnl <= -MAX_WEEKLY_DRAWDOWN_PCT:
            reason = (f"Weekly drawdown {wpnl:.2f}% exceeds limit "
                      f"-{MAX_WEEKLY_DRAWDOWN_PCT:.1f}%")
            log.warning("discipline_rejected", check="weekly_drawdown",
                        reason=reason)
            return False, reason

        return True, "All discipline checks passed"

    # ------------------------------------------------------------------
    # Promotion evaluation
    # ------------------------------------------------------------------

    def evaluate_promotion(self, strategy_name: str) -> tuple[bool, str | None, str]:
        """
        Evaluate whether a strategy should be promoted to the next phase.

        Returns (should_promote, next_phase_or_None, reason_string).
        """
        lifecycle = self._get_lifecycle(strategy_name)
        if lifecycle is None:
            return False, None, f"Strategy '{strategy_name}' not in lifecycle table"

        current = lifecycle["current_phase"]
        perf = self.journal.get_strategy_performance(strategy_name)

        # HYPOTHESIS -> BACKTEST  (automatic)
        if current == "HYPOTHESIS":
            return True, "BACKTEST", "Automatic promotion: hypothesis defined"

        # BACKTEST -> PAPER
        if current == "BACKTEST":
            criteria = PROMOTION_CRITERIA["BACKTEST"]
            wr = lifecycle.get("backtest_win_rate") or 0
            pf = self._compute_profit_factor(strategy_name)
            years = self._estimate_years_tested(strategy_name)

            reasons = []
            met = True

            if wr < criteria["win_rate"]:
                reasons.append(f"Backtest WR {wr:.2%} < {criteria['win_rate']:.0%}")
                met = False
            else:
                reasons.append(f"Backtest WR {wr:.2%} OK")

            if pf < criteria["profit_factor"]:
                reasons.append(f"Profit factor {pf:.2f} < {criteria['profit_factor']}")
                met = False
            else:
                reasons.append(f"Profit factor {pf:.2f} OK")

            if years < criteria["min_years_tested"]:
                reasons.append(f"Tested {years:.1f}y < {criteria['min_years_tested']}y")
                met = False
            else:
                reasons.append(f"Tested {years:.1f}y OK")

            reason_str = "; ".join(reasons)
            if met:
                log.info("promotion_approved", strategy=strategy_name,
                         from_phase="BACKTEST", to_phase="PAPER", reason=reason_str)
                return True, "PAPER", reason_str
            return False, None, reason_str

        # PAPER -> PROMOTE_REVIEW
        if current == "PAPER":
            return self._check_paper_criteria(strategy_name, perf, "PROMOTE_REVIEW")

        # PROMOTE_REVIEW -> LIVE  (needs manual approval)
        if current == "PROMOTE_REVIEW":
            ok, _, reason = self._check_paper_criteria(strategy_name, perf, "LIVE")
            if not ok:
                return False, None, reason

            if strategy_name not in self._manual_approvals:
                return False, None, ("Paper criteria met but manual approval not yet "
                                     f"granted for '{strategy_name}'")

            log.info("promotion_approved", strategy=strategy_name,
                     from_phase="PROMOTE_REVIEW", to_phase="LIVE",
                     reason="Manual approval + paper criteria met")
            return True, "LIVE", "Manual approval + all paper criteria met"

        # LIVE -> MONITORING  (automatic after N trades)
        if current == "LIVE":
            n_live = perf.get("trades", 0)
            threshold = PROMOTION_CRITERIA["LIVE"]["auto_after_trades"]
            if n_live >= threshold:
                return (True, "MONITORING",
                        f"{n_live} live trades >= {threshold}; auto-promoted to MONITORING")
            return (False, None,
                    f"Only {n_live}/{threshold} live trades; need more before MONITORING")

        return False, None, f"No promotion path from phase '{current}'"

    def evaluate_demotion(self, strategy_name: str) -> tuple[bool, str]:
        """
        Evaluate whether a strategy should be demoted.

        Returns (should_demote, reason_string).
        """
        lifecycle = self._get_lifecycle(strategy_name)
        if lifecycle is None:
            return False, f"Strategy '{strategy_name}' not found"

        current = lifecycle["current_phase"]
        perf = self.journal.get_strategy_performance(strategy_name)
        total_trades = perf.get("trades", 0)
        wr = perf.get("win_rate", 0)

        # MONITORING -> DEMOTED
        if current == "MONITORING":
            crit = DEMOTION_CRITERIA["MONITORING"]
            if total_trades >= crit["min_trades_to_judge"]:
                if wr < crit["win_rate_floor"]:
                    reason = (f"Live WR {wr:.2%} < {crit['win_rate_floor']:.0%} "
                              f"over {total_trades} trades -> DEMOTED")
                    log.warning("strategy_demoted", strategy=strategy_name,
                                reason=reason)
                    return True, reason
            return False, (f"WR={wr:.2%} over {total_trades} trades; "
                           f"need {crit['min_trades_to_judge']} trades to judge")

        # DEMOTED -> RETIRED
        if current == "DEMOTED":
            crit = DEMOTION_CRITERIA["DEMOTED"]
            trades_since = self._strategy_trades_since_demotion.get(strategy_name, 0)
            if trades_since >= crit["min_trades_to_judge"]:
                # Re-check WR over the trades since demotion
                if wr < crit["win_rate_floor"]:
                    reason = (f"Still WR {wr:.2%} < {crit['win_rate_floor']:.0%} "
                              f"after {trades_since} more trades -> RETIRED")
                    log.warning("strategy_retired", strategy=strategy_name,
                                reason=reason)
                    return True, reason
                else:
                    # Strategy recovered -- keep as DEMOTED (manual re-promote)
                    return False, (f"WR recovered to {wr:.2%} after {trades_since} "
                                   f"trades; consider re-promotion")
            return False, (f"{trades_since}/{crit['min_trades_to_judge']} "
                           f"trades since demotion; need more data")

        return False, f"No demotion path from phase '{current}'"

    # ------------------------------------------------------------------
    # Active strategies
    # ------------------------------------------------------------------

    def get_active_strategies(self) -> list[dict]:
        """Return strategies currently in PAPER or LIVE phase."""
        conn = self.journal._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM strategy_lifecycle
                   WHERE current_phase IN ('PAPER', 'LIVE', 'MONITORING')
                   ORDER BY current_phase, name"""
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_strategy_weights(self, regime: str) -> dict[str, float]:
        """
        Compute strategy weights for the current regime based on recent
        performance.  Higher win-rate in this regime -> higher weight.
        Strategies not in PAPER / LIVE / MONITORING get zero weight.

        Returns {strategy_name: weight} where weights sum to 1.0.
        """
        active = self.get_active_strategies()
        if not active:
            return {}

        matrix = self.journal.get_regime_performance_matrix()
        raw_weights: dict[str, float] = {}

        for strat in active:
            name = strat["name"]
            regime_data = matrix.get(name, {}).get(regime)
            if regime_data and regime_data["trades"] >= 5:
                # Weight = win_rate * sqrt(sample_size) to reward both accuracy
                # and experience
                wr = regime_data["win_rate"]
                n = regime_data["trades"]
                raw_weights[name] = wr * (n ** 0.5)
            else:
                # Fall back to overall performance
                perf = self.journal.get_strategy_performance(name)
                if perf["trades"] >= 5:
                    raw_weights[name] = perf["win_rate"] * (perf["trades"] ** 0.5)
                else:
                    raw_weights[name] = 0.1  # tiny default for untested

        total = sum(raw_weights.values())
        if total <= 0:
            # Equal weight among all active strategies
            n = len(active)
            return {s["name"]: 1.0 / n for s in active}

        return {name: round(w / total, 4) for name, w in raw_weights.items()}

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    def record_trade(self, strategy_name: str, outcome: str):
        """
        Update daily counters and per-strategy trade counts after a trade.

        Parameters
        ----------
        strategy_name : str
        outcome : str   "CORRECT", "WRONG", or "PARTIAL"
        """
        self._daily_trade_count += 1
        self._strategy_trade_counts[strategy_name] += 1

        # Track trades since demotion for retirement logic
        lifecycle = self._get_lifecycle(strategy_name)
        if lifecycle and lifecycle["current_phase"] == "DEMOTED":
            self._strategy_trades_since_demotion[strategy_name] += 1

        log.info("trade_recorded", strategy=strategy_name, outcome=outcome,
                 daily_count=self._daily_trade_count)
        self._save_state()

    # ------------------------------------------------------------------
    # Resets
    # ------------------------------------------------------------------

    def daily_reset(self):
        """Reset daily counters.  Call at start of each trading day."""
        self._daily_trade_count = 0
        self._daily_live_count = 0
        self._daily_pnl_pct = 0.0
        self._current_positions = 0
        log.info("daily_counters_reset")
        self._save_state()

    # ------------------------------------------------------------------
    # Weekly review
    # ------------------------------------------------------------------

    def weekly_review(self) -> str:
        """
        Generate a weekly lifecycle assessment covering promotions, demotions,
        and an action plan.

        Returns a formatted report string.
        """
        active = self.get_active_strategies()
        conn = self.journal._get_conn()
        try:
            all_strats = conn.execute(
                "SELECT * FROM strategy_lifecycle ORDER BY current_phase, name"
            ).fetchall()
            all_strats = [dict(r) for r in all_strats]
        finally:
            conn.close()

        lines = []
        lines.append("=" * 75)
        lines.append("  WEEKLY LIFECYCLE REVIEW")
        lines.append(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("=" * 75)

        # Phase distribution
        phase_counts = defaultdict(int)
        for s in all_strats:
            phase_counts[s["current_phase"]] += 1

        lines.append("\n  PHASE DISTRIBUTION:")
        for phase in PHASES:
            count = phase_counts.get(phase, 0)
            bar = "#" * count
            lines.append(f"    {phase:<20s} {count:>3d}  {bar}")

        # Promotion candidates
        lines.append("\n  PROMOTION CANDIDATES:")
        promo_found = False
        for s in all_strats:
            if s["current_phase"] in ("RETIRED",):
                continue
            should, next_phase, reason = self.evaluate_promotion(s["name"])
            if should:
                promo_found = True
                lines.append(
                    f"    [PROMOTE] {s['name']:<30s}  "
                    f"{s['current_phase']} -> {next_phase}"
                )
                lines.append(f"              Reason: {reason}")
        if not promo_found:
            lines.append("    (none)")

        # Demotion candidates
        lines.append("\n  DEMOTION CANDIDATES:")
        demo_found = False
        for s in all_strats:
            if s["current_phase"] not in ("MONITORING", "DEMOTED"):
                continue
            should, reason = self.evaluate_demotion(s["name"])
            if should:
                demo_found = True
                target = "DEMOTED" if s["current_phase"] == "MONITORING" else "RETIRED"
                lines.append(
                    f"    [DEMOTE]  {s['name']:<30s}  "
                    f"{s['current_phase']} -> {target}"
                )
                lines.append(f"              Reason: {reason}")
        if not demo_found:
            lines.append("    (none)")

        # Active strategy performance summary
        lines.append("\n  ACTIVE STRATEGY PERFORMANCE:")
        lines.append(
            f"    {'STRATEGY':<35s} {'PHASE':<16s} {'TRADES':>7s} "
            f"{'WR%':>6s} {'PNL':>9s}"
        )
        lines.append("    " + "-" * 75)
        for s in active:
            perf = self.journal.get_strategy_performance(s["name"])
            wr = perf.get("win_rate", 0)
            lines.append(
                f"    {s['name']:<35s} {s['current_phase']:<16s} "
                f"{perf.get('trades', 0):>7d} {wr:>5.1%} "
                f"${perf.get('total_pnl', 0):>+8.2f}"
            )

        # Discipline summary
        lines.append(f"\n  DISCIPLINE LIMITS:")
        lines.append(f"    Max trades/day:        {MAX_TRADES_PER_DAY}")
        lines.append(f"    Max live trades/day:    {MAX_LIVE_TRADES_PER_DAY}")
        lines.append(f"    Max concurrent:         {MAX_CONCURRENT_POSITIONS}")
        lines.append(f"    Min confidence:         {MIN_CONFIDENCE_TO_TRADE}")
        lines.append(f"    Min R:R:                {MIN_RR_RATIO}")
        lines.append(f"    Max daily DD:           {MAX_DAILY_DRAWDOWN_PCT}%")
        lines.append(f"    Max weekly DD:          {MAX_WEEKLY_DRAWDOWN_PCT}%")

        # Action items
        lines.append("\n  ACTION ITEMS:")
        action_count = 0

        for s in all_strats:
            if s["current_phase"] == "PROMOTE_REVIEW":
                if s["name"] not in self._manual_approvals:
                    action_count += 1
                    lines.append(
                        f"    [{action_count}] Review and approve '{s['name']}' "
                        f"for LIVE trading (call approve_for_live())"
                    )

        if promo_found:
            action_count += 1
            lines.append(
                f"    [{action_count}] Execute pending promotions "
                f"(call execute_promotion(strategy_name))"
            )
        if demo_found:
            action_count += 1
            lines.append(
                f"    [{action_count}] Execute pending demotions "
                f"(call execute_demotion(strategy_name))"
            )
        if action_count == 0:
            lines.append("    No action items this week.")

        lines.append("\n" + "=" * 75)
        report = "\n".join(lines)
        log.info("weekly_review_generated", strategies=len(all_strats),
                 active=len(active))
        return report

    # ------------------------------------------------------------------
    # Convenience executors
    # ------------------------------------------------------------------

    def approve_for_live(self, strategy_name: str):
        """Grant manual approval for a PROMOTE_REVIEW strategy to go LIVE."""
        self._manual_approvals.add(strategy_name)
        log.info("manual_approval_granted", strategy=strategy_name)
        self._save_state()

    def execute_promotion(self, strategy_name: str) -> tuple[bool, str]:
        """
        Evaluate and, if approved, promote the strategy and update the journal.
        """
        should, next_phase, reason = self.evaluate_promotion(strategy_name)
        if not should or next_phase is None:
            return False, reason

        perf = self.journal.get_strategy_performance(strategy_name)
        self.journal.update_strategy_lifecycle(
            strategy_name,
            next_phase,
            {
                "backtest_win_rate": perf.get("win_rate") if next_phase == "PAPER" else None,
                "paper_win_rate": perf.get("win_rate") if next_phase in ("PROMOTE_REVIEW", "LIVE") else None,
                "live_win_rate": perf.get("win_rate") if next_phase == "MONITORING" else None,
                "total_signals": perf.get("trades", 0),
                "total_trades": perf.get("trades", 0),
                "total_pnl": perf.get("total_pnl", 0),
                "promotion_criteria_met": {"reason": reason},
            },
        )
        log.info("promotion_executed", strategy=strategy_name,
                 to_phase=next_phase, reason=reason)
        return True, f"Promoted to {next_phase}: {reason}"

    def execute_demotion(self, strategy_name: str) -> tuple[bool, str]:
        """
        Evaluate and, if warranted, demote the strategy and update the journal.
        """
        should, reason = self.evaluate_demotion(strategy_name)
        if not should:
            return False, reason

        lifecycle = self._get_lifecycle(strategy_name)
        current = lifecycle["current_phase"] if lifecycle else "UNKNOWN"
        target = "DEMOTED" if current == "MONITORING" else "RETIRED"

        perf = self.journal.get_strategy_performance(strategy_name)
        self.journal.update_strategy_lifecycle(
            strategy_name,
            target,
            {
                "live_win_rate": perf.get("win_rate"),
                "total_trades": perf.get("trades", 0),
                "total_pnl": perf.get("total_pnl", 0),
                "promotion_criteria_met": {"demotion_reason": reason},
            },
        )

        # Reset demotion trade counter if newly demoted
        if target == "DEMOTED":
            self._strategy_trades_since_demotion[strategy_name] = 0

        log.warning("demotion_executed", strategy=strategy_name,
                    to_phase=target, reason=reason)
        return True, f"Demoted to {target}: {reason}"

    # ------------------------------------------------------------------
    # Position tracking helpers
    # ------------------------------------------------------------------

    def open_position(self):
        """Increment the concurrent positions counter."""
        self._current_positions += 1
        log.info("position_opened", positions=self._current_positions)

    def close_position(self, pnl_pct: float = 0.0):
        """Decrement positions and update daily PnL."""
        self._current_positions = max(0, self._current_positions - 1)
        self._daily_pnl_pct += pnl_pct
        self._weekly_pnl_pct += pnl_pct
        log.info("position_closed", positions=self._current_positions,
                 daily_pnl=round(self._daily_pnl_pct, 3))

    def update_pnl(self, daily_pnl_pct: float, weekly_pnl_pct: float):
        """Externally set current PnL percentages (e.g., from broker)."""
        self._daily_pnl_pct = daily_pnl_pct
        self._weekly_pnl_pct = weekly_pnl_pct

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_lifecycle(self, strategy_name: str) -> dict | None:
        conn = self.journal._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM strategy_lifecycle WHERE name = ?",
                (strategy_name,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def _check_paper_criteria(
        self, strategy_name: str, perf: dict, target_phase: str
    ) -> tuple[bool, str | None, str]:
        """Shared check for PAPER -> PROMOTE_REVIEW and PROMOTE_REVIEW -> LIVE."""
        criteria = PROMOTION_CRITERIA["PAPER"]  # same criteria for both
        wr = perf.get("win_rate", 0)
        total_trades = perf.get("trades", 0)
        avg_win = perf.get("avg_win", 0)
        avg_loss = abs(perf.get("avg_loss", 0))
        rr = avg_win / avg_loss if avg_loss > 0 else float("inf")
        max_cl = perf.get("max_consecutive_losses", 0)

        reasons = []
        met = True

        if wr < criteria["win_rate"]:
            reasons.append(f"WR {wr:.2%} < {criteria['win_rate']:.0%}")
            met = False
        else:
            reasons.append(f"WR {wr:.2%} OK")

        if rr < criteria["rr_ratio"]:
            reasons.append(f"R:R {rr:.2f} < {criteria['rr_ratio']}")
            met = False
        else:
            reasons.append(f"R:R {rr:.2f} OK")

        if total_trades < criteria["min_signals"]:
            reasons.append(f"Signals {total_trades} < {criteria['min_signals']}")
            met = False
        else:
            reasons.append(f"Signals {total_trades} OK")

        if max_cl >= criteria["max_consecutive_losses"]:
            reasons.append(
                f"Max consec losses {max_cl} >= {criteria['max_consecutive_losses']}"
            )
            met = False
        else:
            reasons.append(f"Max consec losses {max_cl} OK")

        reason_str = "; ".join(reasons)
        if met:
            log.info("promotion_approved", strategy=strategy_name,
                     to_phase=target_phase, reason=reason_str)
            return True, target_phase, reason_str
        return False, None, reason_str

    def _compute_profit_factor(self, strategy_name: str) -> float:
        perf = self.journal.get_strategy_performance(strategy_name)
        return perf.get("profit_factor", 0.0)

    def _estimate_years_tested(self, strategy_name: str) -> float:
        """
        Estimate how many years of data a strategy has been tested on by
        looking at its trade timestamps.
        """
        conn = self.journal._get_conn()
        try:
            row = conn.execute(
                """SELECT MIN(timestamp) as first_ts, MAX(timestamp) as last_ts
                   FROM trade_log WHERE strategy = ?""",
                (strategy_name,),
            ).fetchone()
        finally:
            conn.close()

        if row is None or row["first_ts"] is None or row["last_ts"] is None:
            return 0.0

        try:
            first = datetime.fromisoformat(row["first_ts"][:19])
            last = datetime.fromisoformat(row["last_ts"][:19])
            days = (last - first).days
            return max(days / 365.25, 0.0)
        except (ValueError, TypeError):
            return 0.0

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(self):
        state = {
            "daily_trade_count": self._daily_trade_count,
            "daily_live_count": self._daily_live_count,
            "current_positions": self._current_positions,
            "daily_pnl_pct": self._daily_pnl_pct,
            "weekly_pnl_pct": self._weekly_pnl_pct,
            "manual_approvals": list(self._manual_approvals),
            "strategy_trade_counts": dict(self._strategy_trade_counts),
            "strategy_trades_since_demotion": dict(
                self._strategy_trades_since_demotion
            ),
            "last_saved": str(datetime.utcnow()),
        }
        with open(self._state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                state = json.load(f)
            self._daily_trade_count = state.get("daily_trade_count", 0)
            self._daily_live_count = state.get("daily_live_count", 0)
            self._current_positions = state.get("current_positions", 0)
            self._daily_pnl_pct = state.get("daily_pnl_pct", 0.0)
            self._weekly_pnl_pct = state.get("weekly_pnl_pct", 0.0)
            self._manual_approvals = set(state.get("manual_approvals", []))
            self._strategy_trade_counts = defaultdict(
                int, state.get("strategy_trade_counts", {})
            )
            self._strategy_trades_since_demotion = defaultdict(
                int, state.get("strategy_trades_since_demotion", {})
            )
            log.info("lifecycle_state_loaded",
                     daily_count=self._daily_trade_count)
        except Exception as e:
            log.warning("lifecycle_state_load_failed", error=str(e))
