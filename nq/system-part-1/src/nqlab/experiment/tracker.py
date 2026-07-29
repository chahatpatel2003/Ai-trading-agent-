"""DuckDB-backed experiment tracker.

Every candidate is saved the moment it is evaluated (so an interrupted discovery
run can resume by skipping already-recorded fingerprints). DuckDB fits the
existing Parquet/DuckDB stack and gives SQL leaderboards for free. A connection
is opened per operation to avoid long-lived file locks.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import duckdb

from nqlab.discovery.record import CANDIDATE_COLUMNS

_RUN_COLUMNS: tuple[str, ...] = (
    "run_id", "symbol", "timeframe", "account", "start_date", "end_date",
    "n_trials", "pbo", "dsr_threshold", "status", "created_at",
)

_CANDIDATES_DDL = """
CREATE TABLE IF NOT EXISTS candidates (
    run_id VARCHAR, trial INTEGER, family VARCHAR, strategy_name VARCHAR,
    version VARCHAR, fingerprint VARCHAR, params VARCHAR, feature_set VARCHAR,
    train_start VARCHAR, train_end VARCHAR, val_start VARCHAR, val_end VARCHAR,
    config_hash VARCHAR, data_version VARCHAR, git_hash VARCHAR,
    full_net DOUBLE, in_sample_net DOUBLE, oos_net DOUBLE, max_drawdown DOUBLE,
    risk_of_ruin DOUBLE, walk_forward_frac DOUBLE, positive_years_frac DOUBLE,
    in_sample_sharpe DOUBLE, in_sample_skew DOUBLE, in_sample_kurt DOUBLE, in_sample_n INTEGER,
    trades INTEGER, oos_trades INTEGER, cost_stress_survived BOOLEAN, blown BOOLEAN,
    deflated_sharpe DOUBLE, sr0 DOUBLE, gauntlet_verdict VARCHAR, final_verdict VARCHAR,
    rejection_reason VARCHAR, objective_score DOUBLE, walk_forward_nets VARCHAR,
    diagnostics VARCHAR, artifacts_path VARCHAR, created_at VARCHAR,
    PRIMARY KEY (run_id, fingerprint)
)
"""

_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY, symbol VARCHAR, timeframe VARCHAR, account VARCHAR,
    start_date VARCHAR, end_date VARCHAR, n_trials INTEGER, pbo DOUBLE, dsr_threshold DOUBLE,
    status VARCHAR, created_at VARCHAR
)
"""


class ExperimentTracker:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(_CANDIDATES_DDL)
            con.execute(_RUNS_DDL)
            # Migrate older databases that predate the diagnostics column.
            with contextlib.suppress(duckdb.Error):
                con.execute("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS diagnostics VARCHAR")

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.path))

    def upsert_run(self, row: dict[str, Any]) -> None:
        cols = ", ".join(_RUN_COLUMNS)
        ph = ", ".join("?" for _ in _RUN_COLUMNS)
        with self._connect() as con:
            con.execute(
                f"INSERT OR REPLACE INTO runs ({cols}) VALUES ({ph})",
                [row.get(c) for c in _RUN_COLUMNS],
            )

    def save_candidate(self, row: dict[str, Any]) -> None:
        cols = ", ".join(CANDIDATE_COLUMNS)
        ph = ", ".join("?" for _ in CANDIDATE_COLUMNS)
        with self._connect() as con:
            con.execute(
                f"INSERT OR REPLACE INTO candidates ({cols}) VALUES ({ph})",
                [row.get(c) for c in CANDIDATE_COLUMNS],
            )

    def completed_fingerprints(self, run_id: str) -> set[str]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT fingerprint FROM candidates WHERE run_id = ?", [run_id]
            ).fetchall()
        return {r[0] for r in rows}

    def load_candidates(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as con:
            cur = con.execute(
                "SELECT * FROM candidates WHERE run_id = ? ORDER BY trial", [run_id]
            )
            names = [d[0] for d in cur.description]
            return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]

    def update_deflation(
        self, run_id: str, fingerprint: str, *, deflated_sharpe: float, sr0: float,
        final_verdict: str, rejection_reason: str, objective_score: float,
    ) -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE candidates SET deflated_sharpe=?, sr0=?, final_verdict=?, "
                "rejection_reason=?, objective_score=? WHERE run_id=? AND fingerprint=?",
                [deflated_sharpe, sr0, final_verdict, rejection_reason, objective_score,
                 run_id, fingerprint],
            )

    def list_runs(self) -> list[dict[str, Any]]:
        with self._connect() as con:
            cur = con.execute("SELECT * FROM runs ORDER BY created_at DESC")
            names = [d[0] for d in cur.description]
            return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            cur = con.execute("SELECT * FROM runs WHERE run_id = ?", [run_id])
            names = [d[0] for d in cur.description]
            row = cur.fetchone()
        return dict(zip(names, row, strict=True)) if row else None

    def accepted(self) -> list[dict[str, Any]]:
        with self._connect() as con:
            cur = con.execute(
                "SELECT * FROM candidates WHERE final_verdict = 'ACCEPT' "
                "ORDER BY objective_score DESC"
            )
            names = [d[0] for d in cur.description]
            return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]
