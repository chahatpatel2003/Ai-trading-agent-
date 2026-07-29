from __future__ import annotations

from pathlib import Path

from nqlab.discovery.record import CANDIDATE_COLUMNS
from nqlab.experiment.tracker import ExperimentTracker


def _row(run_id: str, fp: str, **over: object) -> dict[str, object]:
    base = {c: None for c in CANDIDATE_COLUMNS}
    base.update(
        run_id=run_id, trial=0, family="orb", strategy_name="orb", version="1",
        fingerprint=fp, params="{}", in_sample_sharpe=0.1, in_sample_skew=0.0,
        in_sample_kurt=3.0, in_sample_n=50, oos_net=100.0, max_drawdown=50.0,
        risk_of_ruin=0.05, walk_forward_frac=0.6, positive_years_frac=1.0,
        cost_stress_survived=True, blown=False, gauntlet_verdict="ACCEPT",
        final_verdict="ACCEPT", rejection_reason="", objective_score=1.0,
        walk_forward_nets="[]",
    )
    base.update(over)
    return base


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "exp.duckdb")
    tracker.save_candidate(_row("r1", "fp1"))
    tracker.save_candidate(_row("r1", "fp2"))
    rows = tracker.load_candidates("r1")
    assert len(rows) == 2
    assert {r["fingerprint"] for r in rows} == {"fp1", "fp2"}
    assert tracker.completed_fingerprints("r1") == {"fp1", "fp2"}


def test_insert_or_replace_dedupes(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "exp.duckdb")
    tracker.save_candidate(_row("r1", "fp1", oos_net=100.0))
    tracker.save_candidate(_row("r1", "fp1", oos_net=200.0))  # same PK -> replace
    rows = tracker.load_candidates("r1")
    assert len(rows) == 1
    assert rows[0]["oos_net"] == 200.0


def test_update_deflation_and_accepted(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "exp.duckdb")
    tracker.save_candidate(_row("r1", "fp1", final_verdict="ACCEPT"))
    tracker.save_candidate(_row("r1", "fp2", final_verdict="REJECT"))
    tracker.update_deflation(
        "r1", "fp1", deflated_sharpe=0.97, sr0=0.2, final_verdict="ACCEPT",
        rejection_reason="", objective_score=2.0,
    )
    accepted = tracker.accepted()
    assert len(accepted) == 1
    assert accepted[0]["fingerprint"] == "fp1"
    assert accepted[0]["deflated_sharpe"] == 0.97


def test_run_metadata(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "exp.duckdb")
    tracker.upsert_run(
        {"run_id": "r1", "symbol": "NQ.c", "timeframe": "5m", "account": "50K",
         "start_date": "2021-01-01", "end_date": "2025-12-31", "n_trials": 10, "pbo": 0.3,
         "dsr_threshold": 0.9, "status": "complete", "created_at": "t"}
    )
    run = tracker.get_run("r1")
    assert run is not None
    assert run["n_trials"] == 10
    assert run["status"] == "complete"
    assert len(tracker.list_runs()) == 1
