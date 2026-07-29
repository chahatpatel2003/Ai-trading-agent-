"""
Session Journal — writes daily learning summaries at the end of each trading session.

Sessions:
    Asian:    00:00–08:00 UTC
    London:   08:00–16:00 UTC
    New York: 16:00–21:00 UTC

At the end of each session, writes a structured analysis to a daily MD file:
    - Trades taken and outcomes
    - What worked, what failed
    - Discipline adherence
    - Patterns observed
    - Lessons learned
    - What to follow tomorrow

This becomes the bot's memory for paper/live trading.
"""

import json
from datetime import datetime, date
from pathlib import Path

from logs.logger import get_logger

log = get_logger("session_journal")

SESSION_HOURS = {
    "asian":   (0, 8),
    "london":  (8, 16),
    "new_york": (16, 21),
}


class SessionJournal:
    """
    Writes daily learning journals as markdown files.

    Usage:
        journal = SessionJournal(data_dir)
        journal.write_session_summary(
            session="london",
            trades=resolved_trades,
            open_trades=open_trades,
            risk_status=risk_mgr.get_status(),
            knowledge=knowledge_base_dict,
            predictions=predictions_dict,
            tracker_stats=tracker.get_stats(),
        )
    """

    def __init__(self, data_dir: str = "./learn/data"):
        self.journal_dir = Path(data_dir) / "journals"
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self._sessions_written_today: set = set()
        self._last_date: str = ""

    def should_write(self, session: str) -> bool:
        """Check if this session's summary should be written now."""
        now = datetime.utcnow()
        today = str(now.date())

        # Reset tracking on new day
        if today != self._last_date:
            self._sessions_written_today = set()
            self._last_date = today

        if session in self._sessions_written_today:
            return False

        start_h, end_h = SESSION_HOURS.get(session, (0, 0))
        # Write when current hour >= end_h (session just ended)
        return now.hour >= end_h

    def get_current_session(self) -> str:
        """Return which session we're currently in."""
        h = datetime.utcnow().hour
        if h < 8:
            return "asian"
        elif h < 16:
            return "london"
        elif h < 21:
            return "new_york"
        return "after_hours"

    def write_session_summary(
        self,
        session: str,
        trades: list,
        open_trades: list,
        risk_status: dict,
        tracker_stats: dict,
        knowledge: dict = None,
        predictions: dict = None,
    ):
        """Write a session summary section to today's daily journal."""
        today = str(date.today())
        file_path = self.journal_dir / f"{today}.md"

        # Build the section
        section = self._build_section(
            session, trades, open_trades, risk_status,
            tracker_stats, knowledge, predictions)

        # Append to daily file (create header if new)
        is_new = not file_path.exists()
        with open(file_path, "a") as f:
            if is_new:
                f.write(f"# Trading Journal — {today}\n\n")
                f.write(f"**Capital**: ${risk_status.get('balance', 0):.2f}  |  ")
                f.write(f"**Risk/Trade**: ${risk_status.get('risk_per_trade', 0):.2f}  |  ")
                f.write(f"**R:R Min**: 1:{risk_status.get('min_rr', 2.5):.1f}\n\n")
                f.write("---\n\n")
            f.write(section)

        self._sessions_written_today.add(session)
        log.info("session_journal_written",
                 session=session, date=today, path=str(file_path))

    def write_daily_summary(
        self,
        risk_status: dict,
        tracker_stats: dict,
        knowledge: dict = None,
    ):
        """Write end-of-day summary with lessons learned."""
        today = str(date.today())
        file_path = self.journal_dir / f"{today}.md"

        section = self._build_daily_lessons(risk_status, tracker_stats, knowledge)

        with open(file_path, "a") as f:
            f.write(section)

        log.info("daily_summary_written", date=today)

    # ─── Section Builders ───

    def _build_section(self, session, trades, open_trades, risk_status,
                       tracker_stats, knowledge, predictions):
        lines = []
        now = datetime.utcnow().strftime("%H:%M UTC")
        lines.append(f"## {session.upper()} Session ({now})\n\n")

        # Risk status
        today = risk_status.get("today", {})
        lines.append(f"**Balance**: ${risk_status.get('balance', 0):.2f}  |  "
                     f"**Today P&L**: ${today.get('pnl', 0):+.2f}  |  "
                     f"**Trades**: {today.get('trades', 0)}/4  |  "
                     f"**Loss streak**: {today.get('consecutive_losses', 0)}/3\n\n")

        if today.get("locked_out"):
            lines.append(f"> LOCKED OUT: {today.get('lockout_reason', '')}\n\n")

        # Resolved trades this session
        session_trades = [t for t in trades if self._is_session_trade(t, session)]
        if session_trades:
            lines.append("### Trades Resolved\n\n")
            wins = [t for t in session_trades if t.get("outcome") == "WON"]
            losses = [t for t in session_trades if t.get("outcome") == "LOST"]
            expired = [t for t in session_trades if t.get("outcome") == "EXPIRED"]

            lines.append(f"| # | Strategy | Dir | Entry | Exit | P&L | Outcome | Bars |\n")
            lines.append(f"|---|----------|-----|-------|------|-----|---------|------|\n")
            for i, t in enumerate(session_trades, 1):
                pnl_str = f"${t.get('pnl_dollars', 0):+.2f}"
                lines.append(
                    f"| {i} | {t.get('strategy', '')} | {t.get('direction', '')} | "
                    f"${t.get('entry_price', 0):.2f} | ${t.get('exit_price', 0):.2f} | "
                    f"{pnl_str} | {t.get('outcome', '')} | {t.get('bars_held', 0)} |\n")
            lines.append("\n")

            if wins:
                lines.append("**What worked:**\n")
                for t in wins:
                    reasons = t.get("reasons", [])
                    lines.append(f"- {t['strategy']} {t['direction']}: {', '.join(reasons[:2])}\n")
                    lines.append(f"  - MFE: +{t.get('max_favorable', 0):.1f} pts, "
                                 f"MAE: {t.get('max_adverse', 0):.1f} pts\n")
                lines.append("\n")

            if losses:
                lines.append("**What failed:**\n")
                for t in losses:
                    reasons = t.get("reasons", [])
                    lines.append(f"- {t['strategy']} {t['direction']}: {', '.join(reasons[:2])}\n")
                    lines.append(f"  - Hit stop at ${t.get('exit_price', 0):.2f} "
                                 f"after {t.get('bars_held', 0)} bars\n")
                lines.append("\n")

            if expired:
                lines.append("**Expired (closed at market):**\n")
                for t in expired:
                    lines.append(f"- {t['strategy']} {t['direction']}: "
                                 f"${t.get('pnl_dollars', 0):+.2f} after "
                                 f"{t.get('bars_held', 0)} bars\n")
                lines.append("\n")
        else:
            lines.append("### No trades resolved this session\n\n")

        # Open trades
        if open_trades:
            lines.append(f"### Open Positions ({len(open_trades)})\n\n")
            for t in open_trades:
                unrealized = 0
                lines.append(f"- **{t.get('strategy', '')}** {t.get('direction', '')} "
                             f"@ ${t.get('entry_price', 0):.2f} "
                             f"(bar {t.get('bars_elapsed', 0)}/{t.get('horizon_bars', 20)}) "
                             f"MFE: +{t.get('max_favorable', 0):.1f} "
                             f"MAE: {t.get('max_adverse', 0):.1f}\n")
            lines.append("\n")

        # Overall stats
        stats = tracker_stats
        if stats.get("total", 0) > 0:
            lines.append("### Cumulative Stats\n\n")
            lines.append(f"- **Win rate**: {stats.get('win_rate', 0):.0%} "
                         f"({stats.get('wins', 0)}W / {stats.get('losses', 0)}L / "
                         f"{stats.get('expired', 0)}E)\n")
            lines.append(f"- **Total P&L**: ${stats.get('total_pnl', 0):+.2f}\n")
            lines.append(f"- **Avg P&L/trade**: ${stats.get('avg_pnl', 0):+.2f}\n\n")

            by_strat = stats.get("by_strategy", {})
            if by_strat:
                lines.append("| Strategy | W | L | E | P&L | WR |\n")
                lines.append("|----------|---|---|---|-----|----|\n")
                for s, d in sorted(by_strat.items(),
                                   key=lambda x: x[1]["pnl"], reverse=True):
                    total = d["trades"]
                    wr = d["wins"] / total if total > 0 else 0
                    lines.append(f"| {s} | {d['wins']} | {d['losses']} | "
                                 f"{d['expired']} | ${d['pnl']:+.2f} | {wr:.0%} |\n")
                lines.append("\n")

        # Discipline check
        lines.append("### Discipline\n\n")
        violations = []
        if today.get("locked_out"):
            violations.append(f"Locked out: {today.get('lockout_reason')}")
        if today.get("consecutive_losses", 0) >= 2:
            violations.append(f"Loss streak at {today['consecutive_losses']}")
        if today.get("pnl", 0) < -40:
            violations.append(f"Approaching daily loss cap (${abs(today['pnl']):.0f}/$60)")

        if violations:
            for v in violations:
                lines.append(f"- {v}\n")
        else:
            lines.append("- All discipline rules followed\n")
            lines.append(f"- Risk per trade: ${risk_status.get('risk_per_trade', 0):.2f} "
                         f"(2% of ${risk_status.get('balance', 0):.2f})\n")
        lines.append("\n---\n\n")

        return "".join(lines)

    def _build_daily_lessons(self, risk_status, tracker_stats, knowledge):
        lines = []
        lines.append("## END OF DAY — Lessons Learned\n\n")

        today = risk_status.get("today", {})
        lines.append(f"**Final balance**: ${risk_status.get('balance', 0):.2f}  |  "
                     f"**Day P&L**: ${today.get('pnl', 0):+.2f}  |  "
                     f"**Record**: {today.get('wins', 0)}W-{today.get('losses', 0)}L\n\n")

        stats = tracker_stats
        by_strat = stats.get("by_strategy", {})

        # What to follow
        winners = {s: d for s, d in by_strat.items()
                   if d.get("pnl", 0) > 0}
        losers = {s: d for s, d in by_strat.items()
                  if d.get("pnl", 0) < 0}

        if winners:
            lines.append("### Follow (profitable strategies)\n\n")
            for s, d in sorted(winners.items(), key=lambda x: -x[1]["pnl"]):
                wr = d["wins"] / d["trades"] if d["trades"] > 0 else 0
                lines.append(f"- **{s}**: ${d['pnl']:+.2f} ({wr:.0%} win rate, "
                             f"{d['trades']} trades)\n")
            lines.append("\n")

        if losers:
            lines.append("### Improve (losing strategies)\n\n")
            for s, d in sorted(losers.items(), key=lambda x: x[1]["pnl"]):
                wr = d["wins"] / d["trades"] if d["trades"] > 0 else 0
                lines.append(f"- **{s}**: ${d['pnl']:+.2f} ({wr:.0%} win rate) "
                             f"— needs review\n")
            lines.append("\n")

        # Key observations
        if knowledge:
            lines.append("### Market Observations\n\n")
            sr = knowledge.get("support_resistance", [])
            for level in sr:
                lines.append(f"- {level.get('type', '')}: ${level.get('level', 0):.2f}\n")
            ss = knowledge.get("session_stats", {})
            for session, data in ss.items():
                lines.append(f"- {session} avg range: ${data.get('avg_range', 0):.2f}\n")
            lines.append("\n")

        # Discipline review
        lines.append("### Discipline Review\n\n")
        if today.get("locked_out"):
            lines.append(f"- **LOCKOUT triggered**: {today.get('lockout_reason')}\n")
            lines.append("- Lesson: the system works — respect the rules\n")
        elif today.get("losses", 0) == 0 and today.get("wins", 0) > 0:
            lines.append("- Perfect discipline day — all trades followed the plan\n")
        else:
            lines.append("- Discipline maintained within limits\n")

        lines.append(f"- Trades: {today.get('trades', 0)}/4 max\n")
        lines.append(f"- Streak: {today.get('consecutive_losses', 0)}/3 max\n")
        lines.append(f"- Daily loss: ${abs(today.get('pnl', 0)):.2f}/$60 max\n")

        lines.append("\n### Tomorrow\n\n")
        lines.append("- Market benefits patience — wait for high-confidence setups\n")
        lines.append("- Follow the plan, not the fear\n")
        lines.append("- Consistency beats brilliance\n")
        lines.append("\n---\n")

        return "".join(lines)

    def _is_session_trade(self, trade: dict, session: str) -> bool:
        """Check if a trade was resolved during this session."""
        exit_time = trade.get("exit_time", "")
        if not exit_time:
            return False
        try:
            h = datetime.fromisoformat(exit_time).hour
            start_h, end_h = SESSION_HOURS.get(session, (0, 24))
            return start_h <= h < end_h
        except Exception:
            return True  # include if can't parse
