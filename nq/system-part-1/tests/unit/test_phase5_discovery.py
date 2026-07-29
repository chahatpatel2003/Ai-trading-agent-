from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from nqlab.config.models import AppConfig, TopstepRules
from nqlab.data.bars_full import build_bars_full
from nqlab.discovery.calibration import calibrate, generate_calibrated_candidates
from nqlab.discovery.harness import run_discovery
from nqlab.discovery.report import run_summary
from nqlab.experiment.tracker import ExperimentTracker
from nqlab.features.build import build_features
from nqlab.features.mbp1_full import build_mbp1_features_full, read_mbp1_features
from nqlab.features.mbp1_profile import profile_distributions
from tests.synth import add_synth_mbp1_columns, feature_bars

_JAN = (date(2024, 1, 1), date(2024, 1, 31))


def _combined() -> pl.DataFrame:
    return add_synth_mbp1_columns(build_features(feature_bars(days=2), rth_start_min=570))


def test_mbp1_features_build_at_1m(config: AppConfig) -> None:
    build_bars_full(config, timeframe="1m", start=_JAN[0], end=_JAN[1], roots=("NQ",))
    result = build_mbp1_features_full(config, root="NQ", timeframe="1m", start=_JAN[0], end=_JAN[1])
    assert "202401" in result.built
    assert read_mbp1_features(config, symbol="NQ.c", timeframe="1m").height > 0


def test_calibrated_discovery_is_deterministic(
    config: AppConfig, rules_50k: TopstepRules, tmp_path: Path
) -> None:
    thresholds = calibrate(profile_distributions(_combined()))
    candidates = generate_calibrated_candidates(thresholds)[:6]

    def run(db: str):
        return run_discovery(
            config, symbol="NQ.c", timeframe="1m", start=date(2024, 1, 2), end=date(2024, 1, 3),
            rules=rules_50k, tracker=ExperimentTracker(tmp_path / db), run_id="p5",
            candidates=list(candidates), feature_frame=_combined(), created_at="t",
        )

    r1 = run("a.duckdb")
    r2 = run("b.duckdb")
    for row in r1.leaderboard:
        assert row["final_verdict"] in {"ACCEPT", "REJECT", "NEEDS_MORE_DATA"}

    def key(rows: list[dict[str, object]]) -> list[tuple[object, ...]]:
        return sorted(
            (r["fingerprint"], round(float(r["oos_net"]), 4), r["final_verdict"])  # type: ignore[arg-type]
            for r in rows
        )

    assert key(r1.leaderboard) == key(r2.leaderboard)


def test_run_summary_for_comparison() -> None:
    rows = [
        {"final_verdict": "REJECT", "objective_score": -5.0, "oos_net": -100.0},
        {"final_verdict": "ACCEPT", "objective_score": 3.0, "oos_net": 200.0},
        {"final_verdict": "NEEDS_MORE_DATA", "objective_score": -1000.0, "oos_net": 0.0},
    ]
    summary = run_summary(rows)
    assert summary["trials"] == 3
    assert summary["accept"] == 1
    assert summary["reject"] == 1
    assert summary["needs_more_data"] == 1
    assert summary["best_oos"] == 200.0
