"""Candidate record schema: the flat row stored per tested candidate."""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING

from nqlab.validation.periods import net_of, trades_of

if TYPE_CHECKING:
    from nqlab.validation.gauntlet import ValidationResult

# Ordered column list shared between the record builder and the tracker schema.
CANDIDATE_COLUMNS: tuple[str, ...] = (
    "run_id", "trial", "family", "strategy_name", "version", "fingerprint",
    "params", "feature_set", "train_start", "train_end", "val_start", "val_end",
    "config_hash", "data_version", "git_hash",
    "full_net", "in_sample_net", "oos_net", "max_drawdown", "risk_of_ruin",
    "walk_forward_frac", "positive_years_frac",
    "in_sample_sharpe", "in_sample_skew", "in_sample_kurt", "in_sample_n",
    "trades", "oos_trades", "cost_stress_survived", "blown",
    "deflated_sharpe", "sr0", "gauntlet_verdict", "final_verdict", "rejection_reason",
    "objective_score", "walk_forward_nets", "diagnostics", "artifacts_path", "created_at",
)


def build_candidate_row(
    result: ValidationResult,
    *,
    run_id: str,
    trial: int,
    family: str,
    fingerprint: str,
    params: dict[str, object],
    feature_set: str,
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
    config_hash: str,
    data_version: str,
    git_hash: str,
    artifacts_path: str,
    created_at: str,
) -> dict[str, object]:
    positive_years = sum(1 for s in result.yearly.values() if net_of(s) > 0)
    years_total = len(result.yearly)
    stress_survived = bool(result.cost_stress) and all(s.net_pnl > 0 for s in result.cost_stress)
    reason = "; ".join(result.reasons)
    return {
        "run_id": run_id,
        "trial": trial,
        "family": family,
        "strategy_name": result.strategy,
        "version": result.version,
        "fingerprint": fingerprint,
        "params": json.dumps(dict(sorted(params.items()))),
        "feature_set": feature_set,
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "val_start": val_start.isoformat(),
        "val_end": val_end.isoformat(),
        "config_hash": config_hash,
        "data_version": data_version,
        "git_hash": git_hash,
        "full_net": net_of(result.full),
        "in_sample_net": net_of(result.in_sample),
        "oos_net": net_of(result.out_sample),
        "max_drawdown": float(result.full.get("max_drawdown") or 0.0),  # type: ignore[arg-type]
        "risk_of_ruin": result.monte_carlo.risk_of_ruin,
        "walk_forward_frac": result.walk_forward_positive_frac,
        "positive_years_frac": positive_years / years_total if years_total else 0.0,
        "in_sample_sharpe": result.in_sample_sharpe,
        "in_sample_skew": result.in_sample_skew,
        "in_sample_kurt": result.in_sample_kurt,
        "in_sample_n": result.in_sample_n,
        "trades": trades_of(result.full),
        "oos_trades": trades_of(result.out_sample),
        "cost_stress_survived": stress_survived,
        "blown": bool(result.topstep.blown) if result.topstep else False,
        "deflated_sharpe": None,
        "sr0": None,
        "gauntlet_verdict": result.verdict,
        "final_verdict": result.verdict,
        "rejection_reason": reason,
        "objective_score": 0.0,
        "walk_forward_nets": json.dumps([w.oos_net for w in result.walk_forward]),
        "diagnostics": json.dumps(result.diagnostics),
        "artifacts_path": artifacts_path,
        "created_at": created_at,
    }
