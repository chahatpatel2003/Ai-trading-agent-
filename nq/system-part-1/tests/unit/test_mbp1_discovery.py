from __future__ import annotations

from datetime import date
from pathlib import Path

from nqlab.config.models import AppConfig, TopstepRules
from nqlab.discovery.candidates import MBP1_FAMILIES
from nqlab.discovery.harness import run_discovery
from nqlab.experiment.tracker import ExperimentTracker
from nqlab.features.build import build_features
from tests.synth import add_synth_mbp1_columns, feature_bars


def _combined():
    return add_synth_mbp1_columns(build_features(feature_bars(days=2), rth_start_min=570))


def _run(config: AppConfig, rules: TopstepRules, db: Path):
    return run_discovery(
        config, symbol="NQ.c", timeframe="5m", start=date(2024, 1, 2), end=date(2024, 1, 3),
        rules=rules, tracker=ExperimentTracker(db), run_id="m", families=tuple(MBP1_FAMILIES),
        max_candidates=6, feature_frame=_combined(), created_at="t",
    )


def test_mbp1_discovery_runs_and_is_deterministic(
    config: AppConfig, rules_50k: TopstepRules, tmp_path: Path
) -> None:
    r1 = _run(config, rules_50k, tmp_path / "a.duckdb")
    r2 = _run(config, rules_50k, tmp_path / "b.duckdb")

    assert r1.n_trials == 6
    for row in r1.leaderboard:
        assert row["final_verdict"] in {"ACCEPT", "REJECT", "NEEDS_MORE_DATA"}

    def key(rows: list[dict[str, object]]) -> list[tuple[object, ...]]:
        return sorted(
            (r["fingerprint"], round(float(r["oos_net"]), 4), r["final_verdict"])  # type: ignore[arg-type]
            for r in rows
        )

    assert key(r1.leaderboard) == key(r2.leaderboard)
