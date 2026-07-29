from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from nqlab.config.models import AppConfig, TopstepRules
from nqlab.discovery.calibration import calibrate, generate_setup_candidates
from nqlab.discovery.harness import run_discovery
from nqlab.experiment.tracker import ExperimentTracker
from nqlab.features.build import build_features
from nqlab.features.context import build_context_features
from nqlab.features.mbp1_profile import profile_distributions
from tests.synth import add_synth_mbp1_columns, globex_bars


def _setup_frame() -> pl.DataFrame:
    feats = add_synth_mbp1_columns(build_features(globex_bars(days=2), rth_start_min=570))
    return build_context_features(feats, rth_start_min=570)


def test_setup_discovery_runs_and_is_deterministic(
    config: AppConfig, rules_50k: TopstepRules, tmp_path: Path
) -> None:
    frame = _setup_frame()
    candidates = generate_setup_candidates(calibrate(profile_distributions(frame)))[:6]

    def run(db: str):
        return run_discovery(
            config, symbol="NQ.c", timeframe="1m", start=date(2024, 1, 2), end=date(2024, 1, 3),
            rules=rules_50k, tracker=ExperimentTracker(tmp_path / db), run_id="s",
            candidates=list(candidates), feature_frame=frame, created_at="t",
        )

    r1 = run("a.duckdb")
    r2 = run("b.duckdb")
    for row in r1.leaderboard:
        assert row["final_verdict"] in {"ACCEPT", "REJECT", "NEEDS_MORE_DATA"}
        assert "diagnostics" in row  # diagnostics persisted per candidate

    def key(rows: list[dict[str, object]]) -> list[tuple[object, ...]]:
        return sorted(
            (r["fingerprint"], round(float(r["oos_net"]), 4), r["final_verdict"])  # type: ignore[arg-type]
            for r in rows
        )

    assert key(r1.leaderboard) == key(r2.leaderboard)
