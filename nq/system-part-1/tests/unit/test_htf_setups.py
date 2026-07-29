from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from nqlab.config.models import AppConfig, TopstepRules
from nqlab.discovery.harness import run_discovery
from nqlab.discovery.htf_setups import PHASE9_FAMILIES, generate_htf_setup_candidates
from nqlab.discovery.prefilter import diagnostics_prefilter
from nqlab.experiment.tracker import ExperimentTracker
from nqlab.features.build import build_features
from nqlab.features.context import build_context_features
from nqlab.features.htf_context import build_htf_context_features
from tests.synth import add_synth_mbp1_columns, globex_bars


def _htf_frame() -> pl.DataFrame:
    feats = add_synth_mbp1_columns(build_features(globex_bars(days=12), rth_start_min=570))
    ctx = build_context_features(feats, rth_start_min=570)
    return build_htf_context_features(ctx, rth_start_min=570, timeframe="5m")


def test_generate_htf_candidates_deterministic_and_unique() -> None:
    a = generate_htf_setup_candidates(timeframe="5m")
    b = generate_htf_setup_candidates(timeframe="5m")
    assert [s.params.fingerprint() for s in a] == [s.params.fingerprint() for s in b]
    assert len({s.params.fingerprint() for s in a}) == len(a)  # no duplicate candidates
    assert {s.family for s in a} == set(PHASE9_FAMILIES)  # all five families present


def test_phase9_discovery_assigns_every_candidate_a_verdict(
    config: AppConfig, rules_50k: TopstepRules, tmp_path: Path
) -> None:
    frame = _htf_frame()
    candidates = generate_htf_setup_candidates(timeframe="5m")[:12]
    result = run_discovery(
        config, symbol="NQ.c", timeframe="5m", start=date(2024, 1, 1), end=date(2024, 1, 31),
        rules=rules_50k, tracker=ExperimentTracker(tmp_path / "p9.duckdb"), run_id="p9",
        candidates=list(candidates), feature_frame=frame, created_at="t",
        prefilter=diagnostics_prefilter(min_trades=5, asymmetry_min=1.2),
    )
    assert len(result.leaderboard) == 12
    for row in result.leaderboard:
        assert row["final_verdict"] in {"ACCEPT", "REJECT", "NEEDS_MORE_DATA"}
        assert "diagnostics" in row


def test_diagnostics_prefilter_short_circuits_low_trade_candidates(
    config: AppConfig, rules_50k: TopstepRules, tmp_path: Path
) -> None:
    """With a high min-trades floor on a short synthetic frame, the prefilter should
    short-circuit candidates to NEEDS_MORE_DATA without running the full gauntlet
    (no walk-forward matrix is recorded for short-circuited rows)."""
    frame = _htf_frame()
    candidates = generate_htf_setup_candidates(timeframe="5m")[:8]
    result = run_discovery(
        config, symbol="NQ.c", timeframe="5m", start=date(2024, 1, 1), end=date(2024, 1, 31),
        rules=rules_50k, tracker=ExperimentTracker(tmp_path / "p9b.duckdb"), run_id="p9b",
        candidates=list(candidates), feature_frame=frame, created_at="t",
        prefilter=diagnostics_prefilter(min_trades=10_000, asymmetry_min=1.2),
    )
    assert result.leaderboard  # ran
    assert all(r["final_verdict"] == "NEEDS_MORE_DATA" for r in result.leaderboard)
    assert all("diagnostics-first" in str(r["rejection_reason"]) for r in result.leaderboard)
