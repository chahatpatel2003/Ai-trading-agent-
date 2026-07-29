from __future__ import annotations

import json
from pathlib import Path

from nqlab.discovery.lockbox import (
    ACCEPT_FOR_PAPER,
    NEEDS_MORE_DATA,
    REJECT,
    FrozenHypothesis,
    phase11_verdict,
    select_survivors,
)
from nqlab.discovery.phase11_families import PHASE11_FAMILIES, generate_phase11_candidates
from nqlab.experiment.tracker import ExperimentTracker
from tests.synth import validation_result


def test_generate_phase11_candidates_deterministic_and_unique() -> None:
    a = generate_phase11_candidates(timeframe="5m")
    b = generate_phase11_candidates(timeframe="5m")
    assert [s.params.fingerprint() for s in a] == [s.params.fingerprint() for s in b]
    assert len({s.params.fingerprint() for s in a}) == len(a)
    assert {s.family for s in a} == set(PHASE11_FAMILIES)  # trend_day, range_fade, vwap_revert


def test_phase11_family_frozen_reconstructs_exactly() -> None:
    params = {
        "stop_atr": 1.5,
        "target_atr": 4.0,
        "min_hold_bars": 3,
        "time_stop_bars": 48,
        "risk_per_trade": 100.0,
        "sizing_mode": "mnq_only",
        "hour_start": 9,
        "hour_end": 12,
    }
    fp = str(PHASE11_FAMILIES["trend_day"](dict(params), rth_start_min=570).params.fingerprint())
    fh = FrozenHypothesis(
        "trend_day_5m_100k_r100",
        "trend_day",
        "5m",
        "100k",
        tuple(sorted(params.items())),
        "phase11_search_100k_5m",
        fp,
        1234.0,
        0.20,
        "t",
    )
    assert fh.config_hash() == fp  # Phase-11 family reconstructs without tuning


def _result(verdict: str, **ov: object):
    r = validation_result(**ov)
    r.verdict = verdict
    r.reasons = ["reason"]
    return r


def test_phase11_verdict_all_paths() -> None:
    thr = 0.9
    # paper-ready: clears the economic gauntlet AND DSR on the pristine lockbox
    assert phase11_verdict(_result("ACCEPT"), 0.95, thr, pristine=True)[0] == ACCEPT_FOR_PAPER
    # clean but not significant even at the frozen count -> REJECT
    assert phase11_verdict(_result("ACCEPT"), 0.50, thr, pristine=True)[0] == REJECT
    # outright economic failure on the lockbox -> REJECT
    assert phase11_verdict(_result("REJECT"), 0.99, thr, pristine=True)[0] == REJECT
    # too few lockbox trades -> NEEDS_MORE_DATA
    assert (
        phase11_verdict(_result("NEEDS_MORE_DATA"), 0.99, thr, pristine=True)[0] == NEEDS_MORE_DATA
    )


def _seed(
    tracker: ExperimentTracker, rid: str, fp: str, *, gv: str, risk: float, oos_trades: int
) -> None:
    tracker.upsert_run(
        {
            "run_id": rid,
            "symbol": "NQ.c",
            "timeframe": "5m",
            "account": "100K",
            "start_date": "2021-01-01",
            "end_date": "2024-12-31",
            "n_trials": 1,
            "pbo": None,
            "dsr_threshold": 0.9,
            "status": "complete",
            "created_at": "t",
        }
    )
    tracker.save_candidate(
        {
            "run_id": rid,
            "fingerprint": fp,
            "family": "trend_day",
            "trial": 0,
            "params": json.dumps({"risk_per_trade": risk, "stop_atr": 1.5, "target_atr": 4.0}),
            "oos_net": 1000.0,
            "deflated_sharpe": 0.2,
            "gauntlet_verdict": gv,
            "oos_trades": oos_trades,
        }
    )


def test_select_survivors_filters_to_testable_gauntlet_winners(tmp_path: Path) -> None:
    tr = ExperimentTracker(tmp_path / "p11.duckdb")
    _seed(tr, "phase11_search_100k_5m", "good00000000000a", gv="ACCEPT", risk=100.0, oos_trades=50)
    _seed(tr, "phase11_search_100k_5m", "lowrisk000000b", gv="ACCEPT", risk=50.0, oos_trades=50)
    _seed(tr, "phase11_search_100k_5m", "fewtrades0000c", gv="ACCEPT", risk=100.0, oos_trades=5)
    _seed(tr, "phase11_search_100k_5m", "rejected00000d", gv="REJECT", risk=100.0, oos_trades=50)
    specs = select_survivors(
        tr, ("phase11_search_100k_5m",), top_n=5, min_oos_trades=20, min_risk=100.0
    )
    fps = {s.fingerprint for s in specs}
    assert fps == {"good00000000000a"}  # only the gauntlet-ACCEPT, risk>=100, enough-trades one
