from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from nqlab.config.models import AppConfig, TopstepRules
from nqlab.discovery.harness import run_discovery
from nqlab.experiment.tracker import ExperimentTracker
from nqlab.features.build import build_features
from tests.synth import feature_bars

_START, _END = date(2024, 1, 2), date(2024, 1, 3)


def _frame() -> pl.DataFrame:
    return build_features(feature_bars(days=2), rth_start_min=570)


def _run(config: AppConfig, rules: TopstepRules, tracker: ExperimentTracker, **kw: object):
    return run_discovery(
        config, symbol="NQ.c", timeframe="5m", start=_START, end=_END, rules=rules,
        tracker=tracker, feature_frame=_frame(), created_at="t", **kw,  # type: ignore[arg-type]
    )


def test_every_candidate_saved_with_verdict(
    config: AppConfig, rules_50k: TopstepRules, tmp_path: Path
) -> None:
    tracker = ExperimentTracker(tmp_path / "e.duckdb")
    result = _run(config, rules_50k, tracker, run_id="t1", max_candidates=6)

    rows = tracker.load_candidates("t1")
    assert len(rows) == 6  # every candidate saved
    assert result.n_trials == 6
    assert len(result.leaderboard) == 6
    for r in rows:
        assert r["final_verdict"] in {"ACCEPT", "REJECT", "NEEDS_MORE_DATA"}
        assert r["deflated_sharpe"] is not None  # deflation applied
    run = tracker.get_run("t1")
    assert run is not None and run["status"] == "complete"


def test_discovery_is_deterministic(
    config: AppConfig, rules_50k: TopstepRules, tmp_path: Path
) -> None:
    t1 = ExperimentTracker(tmp_path / "a.duckdb")
    t2 = ExperimentTracker(tmp_path / "b.duckdb")
    r1 = _run(config, rules_50k, t1, run_id="x", max_candidates=6)
    r2 = _run(config, rules_50k, t2, run_id="x", max_candidates=6)

    def key(rows: list[dict[str, object]]) -> list[tuple[object, ...]]:
        return [
            (r["fingerprint"], round(float(r["oos_net"]), 4),  # type: ignore[arg-type]
             round(float(r["deflated_sharpe"]), 6), r["final_verdict"])  # type: ignore[arg-type]
            for r in sorted(rows, key=lambda z: str(z["fingerprint"]))
        ]

    assert key(r1.leaderboard) == key(r2.leaderboard)


def test_resume_skips_completed(
    config: AppConfig, rules_50k: TopstepRules, tmp_path: Path
) -> None:
    tracker = ExperimentTracker(tmp_path / "e.duckdb")
    _run(config, rules_50k, tracker, run_id="r", max_candidates=3)
    assert len(tracker.completed_fingerprints("r")) == 3

    _run(config, rules_50k, tracker, run_id="r", max_candidates=6, resume=True)
    assert len(tracker.completed_fingerprints("r")) == 6  # added the remaining 3

    _run(config, rules_50k, tracker, run_id="r", max_candidates=6, resume=True)
    assert len(tracker.completed_fingerprints("r")) == 6  # nothing new to do


def test_rejected_candidates_carry_reasons(
    config: AppConfig, rules_50k: TopstepRules, tmp_path: Path
) -> None:
    tracker = ExperimentTracker(tmp_path / "e.duckdb")
    _run(config, rules_50k, tracker, run_id="t", max_candidates=8)
    for r in tracker.load_candidates("t"):
        if r["final_verdict"] != "ACCEPT":
            assert r["rejection_reason"]  # non-empty reason for every non-accept
