"""Discovery harness: generate -> validate -> deflate -> accept/reject -> save.

Each candidate is run through the existing gauntlet and saved immediately (so the
run resumes by skipping recorded fingerprints). After all candidates complete, a
single multiple-testing pass deflates every in-sample Sharpe for the trial count
(downgrading lucky ACCEPTs to REJECT) and computes a run-level PBO.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from nqlab.config.models import AppConfig, TopstepRules
from nqlab.discovery.candidates import generate_candidates
from nqlab.discovery.deflation import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
    sharpe_variance,
)
from nqlab.discovery.objective import objective_score, rank_rows
from nqlab.discovery.record import build_candidate_row
from nqlab.domain.calendar.sessions import parse_hhmm
from nqlab.experiment.tracker import ExperimentTracker
from nqlab.features.build import FeatureParams
from nqlab.research.runner import prepare_features
from nqlab.validation.gauntlet import run_gauntlet
from nqlab.validation.report import write_validation_report


@dataclass
class DiscoveryResult:
    run_id: str
    n_trials: int
    sr0: float
    pbo: float | None
    leaderboard: list[dict[str, Any]]
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "nogit"
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def _data_version(symbol: str, timeframe: str, start: date, end: date, config_hash: str) -> str:
    blob = f"{symbol}|{timeframe}|{start.isoformat()}|{end.isoformat()}|{config_hash}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _pbo(rows: list[dict[str, Any]]) -> float | None:
    # Ignore rows with no walk-forward matrix (e.g. prefilter short-circuits) so the
    # PBO is computed over the candidates that actually ran the full gauntlet.
    matrix = [m for m in (json.loads(r["walk_forward_nets"] or "[]") for r in rows) if m]
    if not matrix or len({len(m) for m in matrix}) != 1:
        return None
    return probability_of_backtest_overfitting(matrix)


def _deflate(
    tracker: ExperimentTracker, run_id: str, rows: list[dict[str, Any]], threshold: float
) -> float:
    n_trials = len(rows)
    sr0 = expected_max_sharpe(sharpe_variance([r["in_sample_sharpe"] for r in rows]), n_trials)
    for r in rows:
        dsr = deflated_sharpe_ratio(
            r["in_sample_sharpe"], sr0, int(r["in_sample_n"] or 0),
            r["in_sample_skew"], r["in_sample_kurt"],
        )
        final = r["gauntlet_verdict"]
        reason = r["rejection_reason"] or ""
        if final == "ACCEPT" and dsr < threshold:
            final = "REJECT"
            extra = (
                f"fails multiple-testing deflation (DSR={dsr:.2f} < {threshold:.2f} "
                f"over {n_trials} trials)"
            )
            reason = f"{reason}; {extra}" if reason else extra
        r["deflated_sharpe"] = dsr
        r["final_verdict"] = final
        score = objective_score(r)
        r["objective_score"] = score
        tracker.update_deflation(
            run_id, r["fingerprint"], deflated_sharpe=dsr, sr0=sr0,
            final_verdict=final, rejection_reason=reason, objective_score=score,
        )
    return sr0


def run_discovery(
    config: AppConfig,
    *,
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
    rules: TopstepRules,
    tracker: ExperimentTracker,
    run_id: str,
    families: tuple[str, ...] | None = None,
    max_candidates: int | None = None,
    resume: bool = False,
    created_at: str | None = None,
    feature_frame: Any = None,
    candidates: list[Any] | None = None,
    prefilter: Any = None,
) -> DiscoveryResult:
    """Run (or resume) a discovery sweep and return the deflated leaderboard.

    Pass ``candidates`` to use a pre-built list (e.g. percentile-calibrated combos);
    otherwise candidates are generated from ``families``.

    ``prefilter`` (Phase 9): an optional callable run per candidate BEFORE the full
    gauntlet. If it returns a ``ValidationResult`` (verdict already set), that result
    is recorded and the expensive gauntlet is skipped for that candidate; if it
    returns ``None`` the candidate flows into the full gauntlet as usual.
    """
    created_at = created_at or datetime.now(UTC).isoformat()
    frame = (
        feature_frame
        if feature_frame is not None
        else prepare_features(config, symbol=symbol, timeframe=timeframe, start=start, end=end)
    )
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    if candidates is None:
        candidates = list(generate_candidates(rth_start_min=rth_start_min, families=families))
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    v = config.validation
    cfg_hash = config.config_hash()
    data_ver = _data_version(symbol, timeframe, start, end, cfg_hash)
    gh = git_hash()
    feat_set = FeatureParams().version

    run_meta: dict[str, Any] = {
        "run_id": run_id, "symbol": symbol, "timeframe": timeframe, "account": rules.name,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "n_trials": len(candidates), "pbo": None, "dsr_threshold": v.dsr_threshold,
        "status": "running", "created_at": created_at,
    }
    tracker.upsert_run(run_meta)
    done = tracker.completed_fingerprints(run_id) if resume else set()

    for trial, strat in enumerate(candidates):
        fingerprint = strat.params.fingerprint()
        if fingerprint in done:
            continue
        result = None
        if prefilter is not None:
            result = prefilter(
                strat, config=config, symbol=symbol, timeframe=timeframe,
                start=start, end=end, rules=rules, frame=frame,
            )
        artifacts_path = ""
        if result is None:
            result = run_gauntlet(
                config, strat, symbol=symbol, timeframe=timeframe,
                start=start, end=end, rules=rules, feature_frame=frame,
            )
            artifacts = config.data.artifacts_root / "discovery" / run_id / fingerprint
            write_validation_report(result, artifacts)
            artifacts_path = str(artifacts)
        row = build_candidate_row(
            result, run_id=run_id, trial=trial, family=strat.family, fingerprint=fingerprint,
            params=dict(strat.p), feature_set=feat_set, train_start=v.train_start,
            train_end=v.train_end, val_start=v.test_start, val_end=v.test_end,
            config_hash=cfg_hash, data_version=data_ver, git_hash=gh,
            artifacts_path=artifacts_path, created_at=created_at,
        )
        tracker.save_candidate(row)  # saved immediately -> resume-safe

    rows = tracker.load_candidates(run_id)
    sr0 = _deflate(tracker, run_id, rows, v.dsr_threshold)
    pbo = _pbo(rows)
    run_meta.update({"pbo": pbo, "status": "complete", "n_trials": len(rows)})
    tracker.upsert_run(run_meta)

    ranked = rank_rows(tracker.load_candidates(run_id))
    accepted = [r for r in ranked if r["final_verdict"] == "ACCEPT"]
    rejected = [r for r in ranked if r["final_verdict"] == "REJECT"]
    return DiscoveryResult(run_id, len(ranked), sr0, pbo, ranked, accepted, rejected)
