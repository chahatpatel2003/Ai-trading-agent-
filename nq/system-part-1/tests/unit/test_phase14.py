from __future__ import annotations

from nqlab.config.models import InstrumentSpec
from nqlab.discovery.lockbox import FrozenHypothesis
from nqlab.discovery.phase14_families import PHASE14_FAMILIES, generate_phase14_candidates
from nqlab.domain.sizing import size_position


def _specs() -> tuple[InstrumentSpec, InstrumentSpec]:
    nq = InstrumentSpec(root="NQ", point_value=20.0, tick_size=0.25, tick_value=5.0)
    mnq = InstrumentSpec(root="MNQ", point_value=2.0, tick_size=0.25, tick_value=0.5)
    return nq, mnq


def test_sizing_repair_never_zero_within_hard_cap() -> None:
    nq, mnq = _specs()
    # stop 60 pts -> one MNQ contract risks 60*$2 = $120 > $100 budget.
    skipped = size_position(
        stop_points=60, risk_usd=100, nq=nq, mnq=mnq, max_contracts=10, mode="mnq_only"
    )
    assert skipped.skipped  # this is the Phase-13 zero-contract failure

    # mnq_min1 floors at 1 contract (within the $500 hard cap) instead of skipping.
    repaired = size_position(
        stop_points=60,
        risk_usd=100,
        nq=nq,
        mnq=mnq,
        max_contracts=10,
        mode="mnq_min1",
        max_loss_usd=500,
    )
    assert not repaired.skipped
    assert repaired.contracts == 1 and repaired.instrument == "MNQ"

    # mnq_fixed1 is always exactly one contract within the cap.
    fixed = size_position(
        stop_points=60,
        risk_usd=250,
        nq=nq,
        mnq=mnq,
        max_contracts=10,
        mode="mnq_fixed1",
        max_loss_usd=500,
    )
    assert not fixed.skipped and fixed.contracts == 1

    # Skip ONLY when one contract breaches the absolute per-trade max risk.
    too_wide = size_position(
        stop_points=300,
        risk_usd=100,
        nq=nq,
        mnq=mnq,
        max_contracts=10,
        mode="mnq_min1",
        max_loss_usd=500,  # 300*$2 = $600 > $500
    )
    assert too_wide.skipped


def test_mnq_min1_scales_up_when_affordable() -> None:
    nq, mnq = _specs()
    # stop 10 pts -> one MNQ risks $20; a $200 budget affords 10, capped by max_contracts.
    d = size_position(
        stop_points=10,
        risk_usd=200,
        nq=nq,
        mnq=mnq,
        max_contracts=10,
        mode="mnq_min1",
        max_loss_usd=500,
    )
    assert d.contracts == 10 and not d.skipped


def test_generate_phase14_candidates_deterministic_and_unique() -> None:
    a = generate_phase14_candidates(timeframe="5m")
    b = generate_phase14_candidates(timeframe="5m")
    assert [s.params.fingerprint() for s in a] == [s.params.fingerprint() for s in b]
    assert len({s.params.fingerprint() for s in a}) == len(a)
    assert {s.family for s in a} == set(PHASE14_FAMILIES)
    # the sizing-repair modes are actually searched
    modes = {dict(s.params.params).get("sizing_mode") for s in a}
    assert {"mnq_min1", "mnq_fixed1"} <= modes


def test_phase14_family_frozen_reconstructs_exactly() -> None:
    params = {
        "stop_atr": 1.5,
        "target_atr": 4.0,
        "min_hold_bars": 2,
        "time_stop_bars": 24,
        "risk_per_trade": 150.0,
        "sizing_mode": "mnq_min1",
        "hour_start": 9,
        "hour_end": 12,
    }
    fp = str(PHASE14_FAMILIES["bos_retest"](dict(params), rth_start_min=570).params.fingerprint())
    fh = FrozenHypothesis(
        "bos_retest_15m_150k_r150",
        "bos_retest",
        "15m",
        "150k",
        tuple(sorted(params.items())),
        "phase14_search_150k_15m",
        fp,
        1000.0,
        0.5,
        "t",
    )
    assert fh.config_hash() == fp
