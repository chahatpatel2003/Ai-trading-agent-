"""
Trade Journal -- Full-context SQLite journal that logs EVERYTHING about each
prediction, mental trade, paper trade, discovered pattern, and strategy lifecycle.

Tables:
  trade_log          -- every mental / paper / live trade with full market context
  pattern_log        -- every discovered pattern with conditions and stats
  strategy_lifecycle -- strategy promotion / demotion lifecycle tracking
  daily_summary      -- one row per trading day with aggregate stats
"""
import json
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from logs.logger import get_logger

log = get_logger("learn.journal")

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
_SESSION_RANGES = [
    ("asian", 0, 8),
    ("london", 8, 13),
    ("ny", 13, 22),
]


def _classify_session(hour: int) -> str:
    """Return the trading session name for a given UTC hour."""
    for name, start, end in _SESSION_RANGES:
        if start <= hour < end:
            return name
    return "off_hours"


# ---------------------------------------------------------------------------
# TradeJournal
# ---------------------------------------------------------------------------

class TradeJournal:
    """Persistent, full-context trade journal backed by SQLite."""

    def __init__(self, data_dir: str = "./learn/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "trade_journal.db"
        self._init_db()
        log.info("trade_journal_ready", db=str(self.db_path))

    # ------------------------------------------------------------------
    # Database initialisation
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trade_log (
                    trade_id        TEXT PRIMARY KEY,
                    timestamp       TEXT NOT NULL,
                    strategy        TEXT,
                    direction       TEXT,
                    entry_price     REAL,
                    exit_price      REAL,
                    pnl             REAL,
                    outcome         TEXT,
                    confidence      REAL,
                    reasons         TEXT,
                    timeframe       TEXT,
                    bars_held       INTEGER,
                    max_favorable   REAL,
                    max_adverse     REAL,
                    market_context  TEXT,
                    regime          TEXT,
                    gold_pressure_score REAL,
                    risk_sentiment  TEXT,
                    hour_of_day     INTEGER,
                    day_of_week     INTEGER,
                    session         TEXT,
                    status          TEXT DEFAULT 'MENTAL'
                );

                CREATE TABLE IF NOT EXISTS pattern_log (
                    pattern_id      TEXT PRIMARY KEY,
                    name            TEXT NOT NULL,
                    description     TEXT,
                    discovery_date  TEXT NOT NULL,
                    win_rate        REAL,
                    sample_size     INTEGER DEFAULT 0,
                    avg_pnl         REAL,
                    confidence      REAL,
                    conditions      TEXT,
                    applicable_regimes  TEXT,
                    applicable_sessions TEXT,
                    status          TEXT DEFAULT 'HYPOTHESIS'
                );

                CREATE TABLE IF NOT EXISTS strategy_lifecycle (
                    strategy_id     TEXT PRIMARY KEY,
                    name            TEXT NOT NULL UNIQUE,
                    created_date    TEXT NOT NULL,
                    current_phase   TEXT DEFAULT 'HYPOTHESIS',
                    backtest_win_rate   REAL,
                    paper_win_rate      REAL,
                    live_win_rate       REAL,
                    total_signals       INTEGER DEFAULT 0,
                    total_trades        INTEGER DEFAULT 0,
                    total_pnl           REAL DEFAULT 0.0,
                    last_promotion_date TEXT,
                    last_demotion_date  TEXT,
                    promotion_criteria_met TEXT
                );

                CREATE TABLE IF NOT EXISTS daily_summary (
                    date                TEXT PRIMARY KEY,
                    predictions_made    INTEGER DEFAULT 0,
                    predictions_correct INTEGER DEFAULT 0,
                    predictions_wrong   INTEGER DEFAULT 0,
                    trades_taken        INTEGER DEFAULT 0,
                    regime              TEXT,
                    gold_pressure_score REAL,
                    top_strategy        TEXT,
                    worst_strategy      TEXT,
                    correlations_discovered INTEGER DEFAULT 0,
                    new_hypotheses      INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_trade_strategy
                    ON trade_log(strategy);
                CREATE INDEX IF NOT EXISTS idx_trade_outcome
                    ON trade_log(outcome);
                CREATE INDEX IF NOT EXISTS idx_trade_status
                    ON trade_log(status);
                CREATE INDEX IF NOT EXISTS idx_trade_session
                    ON trade_log(session);
                CREATE INDEX IF NOT EXISTS idx_trade_timestamp
                    ON trade_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_pattern_status
                    ON pattern_log(status);
                CREATE INDEX IF NOT EXISTS idx_lifecycle_phase
                    ON strategy_lifecycle(current_phase);
            """)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Trade logging
    # ------------------------------------------------------------------

    def log_prediction(self, pred_dict: dict, market_context: dict) -> str:
        """
        Log a new prediction / mental trade.

        Parameters
        ----------
        pred_dict : dict
            Must include at minimum: trade_id, strategy, direction, entry_price,
            confidence, reasons, timeframe.  Optional: bars_held, status.
        market_context : dict
            Snapshot of correlated assets at entry time.  Keys such as
            dxy, vix, oil, silver, btc, ten_year, sp500.

        Returns
        -------
        str  trade_id of the newly inserted row.
        """
        now = datetime.utcnow()
        trade_id = pred_dict.get("trade_id") or pred_dict.get("id") or _make_id()
        reasons = pred_dict.get("reasons", [])
        if isinstance(reasons, list):
            reasons = json.dumps(reasons)

        hour = now.hour
        dow = now.weekday()
        session = _classify_session(hour)

        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO trade_log
                   (trade_id, timestamp, strategy, direction, entry_price,
                    confidence, reasons, timeframe, bars_held,
                    max_favorable, max_adverse,
                    market_context, regime, gold_pressure_score,
                    risk_sentiment, hour_of_day, day_of_week, session, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trade_id,
                    pred_dict.get("timestamp", str(now)),
                    pred_dict.get("strategy") or pred_dict.get("strategy_hypothesis"),
                    pred_dict.get("direction"),
                    pred_dict.get("entry_price"),
                    pred_dict.get("confidence"),
                    reasons,
                    pred_dict.get("timeframe"),
                    pred_dict.get("bars_held", 0),
                    pred_dict.get("max_favorable", 0.0),
                    pred_dict.get("max_adverse", 0.0),
                    json.dumps(market_context) if market_context else "{}",
                    market_context.get("regime") if market_context else None,
                    market_context.get("gold_pressure_score") if market_context else None,
                    market_context.get("risk_sentiment") if market_context else None,
                    hour,
                    dow,
                    session,
                    pred_dict.get("status", "MENTAL"),
                ),
            )
            conn.commit()
            log.info("prediction_logged", trade_id=trade_id,
                     strategy=pred_dict.get("strategy"))
        finally:
            conn.close()
        return trade_id

    def log_outcome(self, trade_id: str, outcome_dict: dict):
        """
        Update a trade row with its resolved outcome.

        Parameters
        ----------
        outcome_dict : dict
            Keys: outcome (CORRECT/WRONG/PARTIAL), exit_price, pnl,
            bars_held, max_favorable, max_adverse.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                """UPDATE trade_log
                   SET outcome       = ?,
                       exit_price    = ?,
                       pnl           = ?,
                       bars_held     = ?,
                       max_favorable = ?,
                       max_adverse   = ?
                   WHERE trade_id = ?""",
                (
                    outcome_dict.get("outcome"),
                    outcome_dict.get("exit_price"),
                    outcome_dict.get("pnl"),
                    outcome_dict.get("bars_held"),
                    outcome_dict.get("max_favorable"),
                    outcome_dict.get("max_adverse"),
                    trade_id,
                ),
            )
            conn.commit()
            log.info("outcome_logged", trade_id=trade_id,
                     outcome=outcome_dict.get("outcome"),
                     pnl=outcome_dict.get("pnl"))
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Pattern logging
    # ------------------------------------------------------------------

    def log_pattern(self, pattern_dict: dict) -> str:
        """
        Insert or update a discovered pattern.

        Parameters
        ----------
        pattern_dict : dict
            Keys: pattern_id, name, description, win_rate, sample_size,
            avg_pnl, confidence, conditions (dict/list), applicable_regimes,
            applicable_sessions, status.
        """
        pid = pattern_dict.get("pattern_id") or _make_id()
        conditions = pattern_dict.get("conditions", {})
        if not isinstance(conditions, str):
            conditions = json.dumps(conditions)

        regimes = pattern_dict.get("applicable_regimes", [])
        if isinstance(regimes, list):
            regimes = json.dumps(regimes)

        sessions = pattern_dict.get("applicable_sessions", [])
        if isinstance(sessions, list):
            sessions = json.dumps(sessions)

        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO pattern_log
                   (pattern_id, name, description, discovery_date,
                    win_rate, sample_size, avg_pnl, confidence,
                    conditions, applicable_regimes, applicable_sessions, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    pid,
                    pattern_dict.get("name", "unnamed"),
                    pattern_dict.get("description"),
                    pattern_dict.get("discovery_date", str(datetime.utcnow())),
                    pattern_dict.get("win_rate"),
                    pattern_dict.get("sample_size", 0),
                    pattern_dict.get("avg_pnl"),
                    pattern_dict.get("confidence"),
                    conditions,
                    regimes,
                    sessions,
                    pattern_dict.get("status", "HYPOTHESIS"),
                ),
            )
            conn.commit()
            log.info("pattern_logged", pattern_id=pid,
                     name=pattern_dict.get("name"),
                     status=pattern_dict.get("status"))
        finally:
            conn.close()
        return pid

    # ------------------------------------------------------------------
    # Strategy lifecycle
    # ------------------------------------------------------------------

    def update_strategy_lifecycle(self, strategy_name: str, new_phase: str,
                                  metrics_dict: dict):
        """
        Insert or update a strategy's lifecycle record.

        Parameters
        ----------
        metrics_dict : dict
            May contain backtest_win_rate, paper_win_rate, live_win_rate,
            total_signals, total_trades, total_pnl, promotion_criteria_met.
        """
        now_str = str(datetime.utcnow())
        conn = self._get_conn()
        try:
            # Check if strategy already exists
            row = conn.execute(
                "SELECT strategy_id, current_phase FROM strategy_lifecycle WHERE name = ?",
                (strategy_name,),
            ).fetchone()

            promotion_criteria = metrics_dict.get("promotion_criteria_met", {})
            if not isinstance(promotion_criteria, str):
                promotion_criteria = json.dumps(promotion_criteria)

            if row is None:
                sid = _make_id()
                conn.execute(
                    """INSERT INTO strategy_lifecycle
                       (strategy_id, name, created_date, current_phase,
                        backtest_win_rate, paper_win_rate, live_win_rate,
                        total_signals, total_trades, total_pnl,
                        last_promotion_date, last_demotion_date,
                        promotion_criteria_met)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        sid,
                        strategy_name,
                        now_str,
                        new_phase,
                        metrics_dict.get("backtest_win_rate"),
                        metrics_dict.get("paper_win_rate"),
                        metrics_dict.get("live_win_rate"),
                        metrics_dict.get("total_signals", 0),
                        metrics_dict.get("total_trades", 0),
                        metrics_dict.get("total_pnl", 0.0),
                        now_str if _is_promotion(None, new_phase) else None,
                        None,
                        promotion_criteria,
                    ),
                )
            else:
                old_phase = row["current_phase"]
                promoted = _is_promotion(old_phase, new_phase)
                demoted = _is_demotion(old_phase, new_phase)

                conn.execute(
                    """UPDATE strategy_lifecycle
                       SET current_phase          = ?,
                           backtest_win_rate       = COALESCE(?, backtest_win_rate),
                           paper_win_rate          = COALESCE(?, paper_win_rate),
                           live_win_rate            = COALESCE(?, live_win_rate),
                           total_signals            = COALESCE(?, total_signals),
                           total_trades             = COALESCE(?, total_trades),
                           total_pnl                = COALESCE(?, total_pnl),
                           last_promotion_date      = CASE WHEN ? THEN ? ELSE last_promotion_date END,
                           last_demotion_date       = CASE WHEN ? THEN ? ELSE last_demotion_date END,
                           promotion_criteria_met   = ?
                       WHERE name = ?""",
                    (
                        new_phase,
                        metrics_dict.get("backtest_win_rate"),
                        metrics_dict.get("paper_win_rate"),
                        metrics_dict.get("live_win_rate"),
                        metrics_dict.get("total_signals"),
                        metrics_dict.get("total_trades"),
                        metrics_dict.get("total_pnl"),
                        promoted, now_str,
                        demoted, now_str,
                        promotion_criteria,
                        strategy_name,
                    ),
                )
            conn.commit()
            log.info("strategy_lifecycle_updated", strategy=strategy_name,
                     phase=new_phase)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_winning_patterns(self, min_win_rate: float = 0.6,
                             min_samples: int = 10) -> list[dict]:
        """Return confirmed patterns meeting win-rate and sample thresholds."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM pattern_log
                   WHERE win_rate >= ? AND sample_size >= ?
                         AND status IN ('CONFIRMED', 'TESTING')
                   ORDER BY win_rate DESC""",
                (min_win_rate, min_samples),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_strategy_performance(self, strategy_name: str) -> dict:
        """Full performance statistics for a strategy from the trade log."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT * FROM trade_log
                   WHERE strategy = ? AND outcome IS NOT NULL""",
                (strategy_name,),
            ).fetchall()
            if not rows:
                return {"strategy": strategy_name, "trades": 0}

            trades = [dict(r) for r in rows]
            pnls = [t["pnl"] for t in trades if t["pnl"] is not None]
            wins = [t for t in trades if t["outcome"] == "CORRECT"]
            losses = [t for t in trades if t["outcome"] == "WRONG"]

            total = len(trades)
            win_count = len(wins)
            loss_count = len(losses)
            win_rate = win_count / total if total > 0 else 0.0

            avg_pnl = statistics.mean(pnls) if pnls else 0.0
            total_pnl = sum(pnls) if pnls else 0.0

            avg_win = statistics.mean([t["pnl"] for t in wins if t["pnl"] is not None]) if wins else 0.0
            avg_loss = statistics.mean([t["pnl"] for t in losses if t["pnl"] is not None]) if losses else 0.0

            gross_profit = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            max_fav = [t["max_favorable"] for t in trades if t["max_favorable"] is not None]
            max_adv = [t["max_adverse"] for t in trades if t["max_adverse"] is not None]

            # Consecutive losses
            max_consec_loss = _max_consecutive(trades, "WRONG")

            return {
                "strategy": strategy_name,
                "trades": total,
                "wins": win_count,
                "losses": loss_count,
                "win_rate": round(win_rate, 4),
                "avg_pnl": round(avg_pnl, 2),
                "total_pnl": round(total_pnl, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 3),
                "max_consecutive_losses": max_consec_loss,
                "avg_max_favorable": round(statistics.mean(max_fav), 2) if max_fav else 0.0,
                "avg_max_adverse": round(statistics.mean(max_adv), 2) if max_adv else 0.0,
            }
        finally:
            conn.close()

    def get_regime_performance_matrix(self) -> dict:
        """
        Strategy x Regime -> win_rate matrix.

        Returns dict of {strategy: {regime: {"win_rate": ..., "trades": ...}}}
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT strategy, regime, outcome, COUNT(*) as cnt
                   FROM trade_log
                   WHERE outcome IS NOT NULL AND regime IS NOT NULL
                   GROUP BY strategy, regime, outcome"""
            ).fetchall()
        finally:
            conn.close()

        # Aggregate: strategy -> regime -> {CORRECT: n, WRONG: n, ...}
        raw = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for r in rows:
            raw[r["strategy"]][r["regime"]][r["outcome"]] += r["cnt"]

        matrix = {}
        for strategy, regimes in raw.items():
            matrix[strategy] = {}
            for regime, outcomes in regimes.items():
                total = sum(outcomes.values())
                wins = outcomes.get("CORRECT", 0)
                matrix[strategy][regime] = {
                    "win_rate": round(wins / total, 4) if total > 0 else 0.0,
                    "trades": total,
                }
        return matrix

    def get_time_of_day_performance(self) -> dict:
        """
        Hour -> win_rate for each strategy.

        Returns dict of {strategy: {hour: {"win_rate": ..., "trades": ...}}}
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT strategy, hour_of_day, outcome, COUNT(*) as cnt
                   FROM trade_log
                   WHERE outcome IS NOT NULL AND hour_of_day IS NOT NULL
                   GROUP BY strategy, hour_of_day, outcome"""
            ).fetchall()
        finally:
            conn.close()

        raw = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for r in rows:
            raw[r["strategy"]][r["hour_of_day"]][r["outcome"]] += r["cnt"]

        result = {}
        for strategy, hours in raw.items():
            result[strategy] = {}
            for hour, outcomes in hours.items():
                total = sum(outcomes.values())
                wins = outcomes.get("CORRECT", 0)
                result[strategy][hour] = {
                    "win_rate": round(wins / total, 4) if total > 0 else 0.0,
                    "trades": total,
                }
        return result

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    def get_daily_summary(self, date: str) -> dict | None:
        """Retrieve the stored daily summary for a given date string (YYYY-MM-DD)."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM daily_summary WHERE date = ?", (date,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def generate_daily_summary(self, date: str, data_dict: dict) -> dict:
        """
        Write (or overwrite) a daily summary row.

        Parameters
        ----------
        date : str  "YYYY-MM-DD"
        data_dict : dict  with keys predictions_made, predictions_correct,
            predictions_wrong, trades_taken, regime, gold_pressure_score,
            top_strategy, worst_strategy, correlations_discovered,
            new_hypotheses.
        """
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO daily_summary
                   (date, predictions_made, predictions_correct, predictions_wrong,
                    trades_taken, regime, gold_pressure_score,
                    top_strategy, worst_strategy,
                    correlations_discovered, new_hypotheses)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    date,
                    data_dict.get("predictions_made", 0),
                    data_dict.get("predictions_correct", 0),
                    data_dict.get("predictions_wrong", 0),
                    data_dict.get("trades_taken", 0),
                    data_dict.get("regime"),
                    data_dict.get("gold_pressure_score"),
                    data_dict.get("top_strategy"),
                    data_dict.get("worst_strategy"),
                    data_dict.get("correlations_discovered", 0),
                    data_dict.get("new_hypotheses", 0),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        summary = {"date": date, **data_dict}
        log.info("daily_summary_generated", date=date)
        return summary

    # ------------------------------------------------------------------
    # Advanced analytics
    # ------------------------------------------------------------------

    def find_best_conditions_for_winning(self) -> dict:
        """
        Analyse the market_context JSON of winning trades to find the most
        common context values.  Returns a dict with keys like "common_regime",
        "common_session", "avg_gold_pressure_score", "common_risk_sentiment",
        "top_hours", and aggregated correlated-asset ranges.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT market_context, regime, session, hour_of_day,
                          gold_pressure_score, risk_sentiment
                   FROM trade_log
                   WHERE outcome = 'CORRECT'"""
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {"note": "No winning trades yet"}

        regimes = defaultdict(int)
        sessions = defaultdict(int)
        hours = defaultdict(int)
        sentiments = defaultdict(int)
        pressure_scores = []
        context_keys = defaultdict(list)

        for r in rows:
            if r["regime"]:
                regimes[r["regime"]] += 1
            if r["session"]:
                sessions[r["session"]] += 1
            if r["hour_of_day"] is not None:
                hours[r["hour_of_day"]] += 1
            if r["risk_sentiment"]:
                sentiments[r["risk_sentiment"]] += 1
            if r["gold_pressure_score"] is not None:
                pressure_scores.append(r["gold_pressure_score"])

            # Parse market context JSON for asset snapshots
            try:
                ctx = json.loads(r["market_context"]) if r["market_context"] else {}
            except (json.JSONDecodeError, TypeError):
                ctx = {}
            for key in ("dxy", "vix", "oil", "silver", "btc", "ten_year", "sp500"):
                val = ctx.get(key)
                if val is not None:
                    context_keys[key].append(val)

        def _top(d, n=3):
            return sorted(d.items(), key=lambda x: -x[1])[:n]

        result = {
            "winning_trades_analysed": len(rows),
            "common_regimes": _top(regimes),
            "common_sessions": _top(sessions),
            "top_hours": _top(hours, 5),
            "common_sentiments": _top(sentiments),
            "avg_gold_pressure_score": (
                round(statistics.mean(pressure_scores), 3) if pressure_scores else None
            ),
        }

        for key, vals in context_keys.items():
            if vals:
                result[f"avg_{key}_at_win"] = round(statistics.mean(vals), 4)
                result[f"median_{key}_at_win"] = round(statistics.median(vals), 4)

        return result

    def generate_weekly_report(self) -> str:
        """
        Comprehensive weekly performance report as a formatted string.
        Covers the last 7 calendar days from today.
        """
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=7)
        start_str = str(start_date)
        end_str = str(end_date)

        conn = self._get_conn()
        try:
            # Trades in window
            trades = conn.execute(
                """SELECT * FROM trade_log
                   WHERE timestamp >= ? AND timestamp < ?
                         AND outcome IS NOT NULL
                   ORDER BY timestamp""",
                (start_str, end_str + "T23:59:59"),
            ).fetchall()
            trades = [dict(r) for r in trades]

            # Daily summaries in window
            summaries = conn.execute(
                """SELECT * FROM daily_summary
                   WHERE date >= ? AND date <= ?
                   ORDER BY date""",
                (start_str, end_str),
            ).fetchall()
            summaries = [dict(r) for r in summaries]

            # Strategy lifecycle snapshot
            strategies = conn.execute(
                "SELECT * FROM strategy_lifecycle ORDER BY current_phase"
            ).fetchall()
            strategies = [dict(r) for r in strategies]

            # Patterns
            patterns = conn.execute(
                "SELECT * FROM pattern_log WHERE status = 'CONFIRMED' ORDER BY win_rate DESC LIMIT 10"
            ).fetchall()
            patterns = [dict(r) for r in patterns]
        finally:
            conn.close()

        # Build report
        lines = []
        lines.append("=" * 75)
        lines.append(f"  WEEKLY PERFORMANCE REPORT  {start_str} to {end_str}")
        lines.append("=" * 75)

        # --- Overall numbers ---
        total = len(trades)
        wins = sum(1 for t in trades if t["outcome"] == "CORRECT")
        losses = sum(1 for t in trades if t["outcome"] == "WRONG")
        partials = total - wins - losses
        pnls = [t["pnl"] for t in trades if t["pnl"] is not None]
        total_pnl = sum(pnls) if pnls else 0.0
        win_rate = wins / total if total > 0 else 0.0

        lines.append(f"\n  TRADES: {total}  |  W: {wins}  L: {losses}  P: {partials}")
        lines.append(f"  WIN RATE: {win_rate:.1%}")
        lines.append(f"  TOTAL PNL: ${total_pnl:+.2f}")
        if pnls:
            lines.append(f"  AVG PNL: ${statistics.mean(pnls):+.2f}")
            lines.append(f"  BEST: ${max(pnls):+.2f}  |  WORST: ${min(pnls):+.2f}")

        # --- Per-strategy breakdown ---
        strat_stats = defaultdict(lambda: {"w": 0, "l": 0, "pnl": 0.0})
        for t in trades:
            s = t.get("strategy") or "unknown"
            if t["outcome"] == "CORRECT":
                strat_stats[s]["w"] += 1
            elif t["outcome"] == "WRONG":
                strat_stats[s]["l"] += 1
            if t["pnl"] is not None:
                strat_stats[s]["pnl"] += t["pnl"]

        if strat_stats:
            lines.append(f"\n  {'STRATEGY':<35s} {'W':>4s} {'L':>4s} {'WR%':>6s} {'PNL':>9s}")
            lines.append("  " + "-" * 60)
            for name, ss in sorted(strat_stats.items(),
                                   key=lambda x: x[1]["pnl"], reverse=True):
                t_ = ss["w"] + ss["l"]
                wr = ss["w"] / t_ * 100 if t_ > 0 else 0
                lines.append(f"  {name:<35s} {ss['w']:>4d} {ss['l']:>4d} {wr:>5.1f}% ${ss['pnl']:>+8.2f}")

        # --- Per-session breakdown ---
        sess_stats = defaultdict(lambda: {"w": 0, "l": 0})
        for t in trades:
            s = t.get("session") or "unknown"
            if t["outcome"] == "CORRECT":
                sess_stats[s]["w"] += 1
            elif t["outcome"] == "WRONG":
                sess_stats[s]["l"] += 1
        if sess_stats:
            lines.append("\n  SESSION PERFORMANCE:")
            for name, ss in sess_stats.items():
                t_ = ss["w"] + ss["l"]
                wr = ss["w"] / t_ * 100 if t_ > 0 else 0
                lines.append(f"    {name:<15s}  W={ss['w']}  L={ss['l']}  WR={wr:.1f}%")

        # --- Strategy lifecycle snapshot ---
        if strategies:
            lines.append("\n  STRATEGY LIFECYCLE:")
            for s in strategies:
                lines.append(
                    f"    {s['name']:<35s}  phase={s['current_phase']:<16s} "
                    f"trades={s.get('total_trades', 0)}  pnl=${s.get('total_pnl', 0):+.2f}"
                )

        # --- Confirmed patterns ---
        if patterns:
            lines.append("\n  TOP CONFIRMED PATTERNS:")
            for p in patterns:
                lines.append(
                    f"    {p['name']:<35s}  WR={p.get('win_rate', 0):.1%}  "
                    f"n={p.get('sample_size', 0)}  avg_pnl=${p.get('avg_pnl', 0) or 0:+.2f}"
                )

        # --- Daily summaries ---
        if summaries:
            lines.append("\n  DAILY LOG:")
            for ds in summaries:
                lines.append(
                    f"    {ds['date']}  pred={ds.get('predictions_made', 0)}  "
                    f"correct={ds.get('predictions_correct', 0)}  "
                    f"wrong={ds.get('predictions_wrong', 0)}  "
                    f"trades={ds.get('trades_taken', 0)}  "
                    f"top={ds.get('top_strategy', '-')}"
                )

        lines.append("\n" + "=" * 75)
        report = "\n".join(lines)
        log.info("weekly_report_generated", trades=total, win_rate=round(win_rate, 3))
        return report


# ===========================================================================
# Private helpers
# ===========================================================================

def _make_id() -> str:
    import uuid
    return str(uuid.uuid4())[:12]


_PHASE_ORDER = [
    "HYPOTHESIS", "BACKTEST", "PAPER", "PROMOTE_REVIEW",
    "LIVE", "MONITORING", "DEMOTED", "RETIRED",
]


def _phase_rank(phase: str | None) -> int:
    if phase is None:
        return -1
    try:
        return _PHASE_ORDER.index(phase)
    except ValueError:
        return -1


def _is_promotion(old_phase: str | None, new_phase: str) -> bool:
    return _phase_rank(new_phase) > _phase_rank(old_phase)


def _is_demotion(old_phase: str | None, new_phase: str) -> bool:
    return new_phase in ("DEMOTED", "RETIRED")


def _max_consecutive(trades: list[dict], target_outcome: str) -> int:
    """Count the longest streak of a given outcome."""
    max_streak = 0
    current = 0
    for t in trades:
        if t.get("outcome") == target_outcome:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak
