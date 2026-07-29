from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from nqlab.config.loader import load_config
from nqlab.config.models import TopstepRules
from nqlab.discovery.htf_setups import PHASE9_FAMILIES
from nqlab.discovery.lockbox import (
    ACCEPT,
    NEEDS_PAPER_LOCKBOX,
    REJECT,
    FreezeSpec,
    FrozenHypothesis,
    freeze_from_phase9,
    load_registry,
    phase10_verdict,
    run_lockbox,
    save_registry,
)
from nqlab.experiment.tracker import ExperimentTracker
from nqlab.features.build import build_features
from nqlab.features.context import build_context_features
from nqlab.features.htf_context import build_htf_context_features
from tests.synth import add_synth_mbp1_columns, globex_bars, validation_result

_PARAMS = {
    "trigger": "or",
    "bias": "4h",
    "stop_atr": 1.5,
    "target_atr": 3.0,
    "min_hold_bars": 3,
    "time_stop_bars": 48,
    "risk_per_trade": 100.0,
    "sizing_mode": "mnq_only",
    "hour_start": 9,
    "hour_end": 12,
}


def _phase9_fingerprint() -> str:
    strat = PHASE9_FAMILIES["compression_expansion"](dict(_PARAMS), rth_start_min=570)
    return str(strat.params.fingerprint())


def _seed_phase9_candidate(tracker: ExperimentTracker, fp: str) -> None:
    tracker.upsert_run(
        {
            "run_id": "phase9_50k_5m",
            "symbol": "NQ.c",
            "timeframe": "5m",
            "account": "50K",
            "start_date": "2021-01-01",
            "end_date": "2026-12-31",
            "n_trials": 1,
            "pbo": None,
            "dsr_threshold": 0.9,
            "status": "complete",
            "created_at": "t",
        }
    )
    tracker.save_candidate(
        {
            "run_id": "phase9_50k_5m",
            "fingerprint": fp,
            "family": "compression_expansion",
            "params": json.dumps(dict(sorted(_PARAMS.items()))),
            "oos_net": 2899.0,
            "deflated_sharpe": 0.17,
            "trial": 0,
        }
    )


# --- freezing: immutability + no tuning after freeze ---------------------------


def test_freeze_requires_exact_fingerprint_match(tmp_path: Path) -> None:
    fp = _phase9_fingerprint()
    tr = ExperimentTracker(tmp_path / "p9.duckdb")
    _seed_phase9_candidate(tr, fp)
    frozen = freeze_from_phase9(tr, (FreezeSpec("ce", "phase9_50k_5m", fp),), frozen_at="t")
    assert len(frozen) == 1
    assert frozen[0].config_hash() == fp  # reconstructs the Phase-9 strategy exactly
    # A spec pointing at a fingerprint that isn't there is refused.
    with pytest.raises(ValueError):
        freeze_from_phase9(
            tr, (FreezeSpec("bad", "phase9_50k_5m", "deadbeefdeadbeef"),), frozen_at="t"
        )


def test_frozen_registry_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    fp = _phase9_fingerprint()
    fh = FrozenHypothesis(
        "ce",
        "compression_expansion",
        "5m",
        "50k",
        tuple(sorted(_PARAMS.items())),
        "phase9_50k_5m",
        fp,
        2899.0,
        0.17,
        "t",
    )
    path = tmp_path / "reg.json"
    save_registry(path, [fh])
    back = load_registry(path)
    assert len(back) == 1 and back[0].config_hash() == fp

    # Tuning a parameter after freeze must be detected on load (hash mismatch).
    data = json.loads(path.read_text())
    data[0]["params"]["target_atr"] = 99.0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_registry(path)


# --- verdict logic: ACCEPT / REJECT / NEEDS_PAPER_LOCKBOX reachable ------------


def _result(verdict: str, **ov: object):
    r = validation_result(**ov)
    r.verdict = verdict
    r.reasons = ["reason"]
    return r


def test_phase10_verdict_all_paths() -> None:
    thr = 0.9
    assert phase10_verdict(_result("REJECT"), 0.99, thr, pristine=True)[0] == REJECT
    assert (
        phase10_verdict(_result("NEEDS_MORE_DATA"), 0.99, thr, pristine=True)[0]
        == NEEDS_PAPER_LOCKBOX
    )
    # economically clean but not significant at the frozen count -> REJECT
    assert phase10_verdict(_result("ACCEPT"), 0.50, thr, pristine=True)[0] == REJECT
    # clean + significant but the lockbox was not pristine -> needs forward paper
    assert phase10_verdict(_result("ACCEPT"), 0.95, thr, pristine=False)[0] == NEEDS_PAPER_LOCKBOX
    # clean + significant + pristine -> ACCEPT (only reachable with an untouched lockbox)
    assert phase10_verdict(_result("ACCEPT"), 0.95, thr, pristine=True)[0] == ACCEPT


# --- run_lockbox: schema, frozen trial count, date isolation -------------------


def _htf_frame() -> pl.DataFrame:
    feats = add_synth_mbp1_columns(build_features(globex_bars(days=12), rth_start_min=570))
    ctx = build_context_features(feats, rth_start_min=570)
    return build_htf_context_features(ctx, rth_start_min=570, timeframe="5m")


def _lockbox_config(config_dir: Path, tmp_path: Path):
    return load_config(
        profile="ci",
        config_dir=config_dir,
        apply_env=False,
        overrides={
            "data": {
                "nq_data_root": str(tmp_path),
                "curated_root": str(tmp_path / "c"),
                "catalog_path": str(tmp_path / "c" / "cat.duckdb"),
                "artifacts_root": str(tmp_path / "c" / "a"),
            },
            "validation": {
                "train_start": "2024-01-02",
                "train_end": "2024-01-05",
                "test_start": "2024-01-06",
                "test_end": "2024-01-12",
                "walk_train_months": 1,
                "walk_test_months": 1,
                "walk_step_months": 1,
            },
        },
    )


def test_run_lockbox_schema_trial_count_and_isolation(
    config_dir: Path, rules_50k: TopstepRules, tmp_path: Path
) -> None:
    cfg = _lockbox_config(config_dir, tmp_path)
    frozen = [
        FrozenHypothesis(
            "ce_or",
            "compression_expansion",
            "5m",
            "50k",
            tuple(sorted(_PARAMS.items())),
            "phase9_50k_5m",
            "fp1",
            2899.0,
            0.17,
            "t",
        ),
        FrozenHypothesis(
            "od_on",
            "opening_drive",
            "5m",
            "50k",
            tuple(sorted({**_PARAMS, "ref": "on"}.items())),
            "phase9_50k_5m",
            "fp2",
            379.0,
            0.33,
            "t",
        ),
    ]
    frame = _htf_frame()
    rows = run_lockbox(
        cfg,
        frozen,
        tracker=ExperimentTracker(tmp_path / "lb.duckdb"),
        run_id="phase10_lockbox",
        lockbox_start=date(2024, 1, 2),
        lockbox_end=date(2024, 1, 12),
        frame_for=lambda tf: frame,
        rules_for=lambda a: rules_50k,
        pristine=False,
        created_at="t",
    )
    assert len(rows) == len(frozen)
    schema = (
        "strategy_name",
        "oos_net",
        "risk_of_ruin",
        "blown",
        "cost_stress_survived",
        "walk_forward_frac",
        "positive_years_frac",
        "oos_trades",
        "deflated_sharpe",
        "sr0",
        "final_verdict",
        "rejection_reason",
        "val_start",
        "val_end",
    )
    for r in rows:
        assert r["final_verdict"] in {ACCEPT, REJECT, NEEDS_PAPER_LOCKBOX}
        for key in schema:
            assert key in r, f"missing report field {key}"
        # date isolation: the OOS gate window is the lockbox test window, never earlier.
        assert r["val_start"] == "2024-01-06"
        assert r["val_end"] == "2024-01-12"
    # frozen trial count: a single sr0 deflated over len(frozen) candidates, shared by all rows.
    assert len({r["sr0"] for r in rows}) == 1
