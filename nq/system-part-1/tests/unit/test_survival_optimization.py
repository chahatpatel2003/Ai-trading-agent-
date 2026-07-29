"""Phase 7 survival optimization runs end-to-end, deterministically, with strict verdicts."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from nqlab.config.models import AppConfig, TopstepRules
from nqlab.discovery.harness import run_discovery
from nqlab.discovery.optimization import generate_survival_candidates
from nqlab.experiment.tracker import ExperimentTracker
from nqlab.features.build import build_features
from nqlab.features.context import build_context_features
from tests.synth import add_synth_mbp1_columns, globex_bars


def _setup_frame() -> pl.DataFrame:
    feats = add_synth_mbp1_columns(build_features(globex_bars(days=2), rth_start_min=570))
    return build_context_features(feats, rth_start_min=570)


def test_survival_optimization_runs_and_is_deterministic(
    config: AppConfig, rules_50k: TopstepRules, tmp_path: Path
) -> None:
    frame = _setup_frame()
    # bias_pullback + pd_sweep_rev exercise the exec-config (risk + min-hold) path.
    candidates = generate_survival_candidates(families=("bias_pullback", "pd_sweep_rev"))[:6]
    assert all(c.exec_config is not None for c in candidates)

    def run(db: str):
        return run_discovery(
            config, symbol="NQ.c", timeframe="1m", start=date(2024, 1, 2), end=date(2024, 1, 3),
            rules=rules_50k, tracker=ExperimentTracker(tmp_path / db), run_id="opt",
            candidates=list(candidates), feature_frame=frame, created_at="t",
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
