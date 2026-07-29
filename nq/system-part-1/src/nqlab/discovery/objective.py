"""Survival-first objective and leaderboard ranking (operates on stored rows).

Never ranks by raw profit. Candidates that lose out-of-sample or blow the account
are pushed below everything else; the rest are ordered by deflated significance,
walk-forward stability, year consistency, cost-stress survival, and a capped MAR.
"""

from __future__ import annotations

_VERDICT_RANK = {"ACCEPT": 2, "NEEDS_MORE_DATA": 1, "REJECT": 0}


def _f(row: dict[str, object], key: str) -> float:
    return float(row.get(key) or 0.0)  # type: ignore[arg-type]


def objective_score(row: dict[str, object]) -> float:
    oos = _f(row, "oos_net")
    ruin = _f(row, "risk_of_ruin")
    blown = bool(row.get("blown"))
    if oos <= 0 or blown:
        # Push failures below all survivors; still ordered by how bad.
        return -1000.0 - ruin * 100.0 + oos / 1e6

    drawdown = _f(row, "max_drawdown")
    mar = oos / drawdown if drawdown > 0 else oos
    dsr = _f(row, "deflated_sharpe")
    stress = 1.0 if row.get("cost_stress_survived") else 0.0
    return (
        2.0 * dsr
        + 1.5 * _f(row, "walk_forward_frac")
        + 1.0 * _f(row, "positive_years_frac")
        + 0.5 * stress
        + 0.0005 * mar
        - 2.0 * ruin
    )


def rank_key(row: dict[str, object]) -> tuple[int, float, float]:
    verdict = str(row.get("final_verdict", "REJECT"))
    return (_VERDICT_RANK.get(verdict, 0), _f(row, "objective_score"), _f(row, "deflated_sharpe"))


def rank_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Leaderboard order: best (survival-first) first."""
    return sorted(rows, key=rank_key, reverse=True)
