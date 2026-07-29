from __future__ import annotations

from nqlab.discovery.lockbox import FrozenHypothesis
from nqlab.discovery.phase13_families import PHASE13_FAMILIES, generate_phase13_candidates


def test_generate_phase13_candidates_deterministic_and_unique() -> None:
    a = generate_phase13_candidates(timeframe="5m")
    b = generate_phase13_candidates(timeframe="5m")
    assert [s.params.fingerprint() for s in a] == [s.params.fingerprint() for s in b]
    assert len({s.params.fingerprint() for s in a}) == len(a)
    assert {s.family for s in a} == set(PHASE13_FAMILIES)  # all 5 structure families


def test_phase13_family_frozen_reconstructs_exactly() -> None:
    params = {
        "stop_atr": 1.0,
        "target_atr": 4.0,
        "min_hold_bars": 3,
        "time_stop_bars": 48,
        "risk_per_trade": 100.0,
        "sizing_mode": "mnq_only",
        "hour_start": 9,
        "hour_end": 12,
    }
    fp = str(
        PHASE13_FAMILIES["liq_sweep_mss"](dict(params), rth_start_min=570).params.fingerprint()
    )
    fh = FrozenHypothesis(
        "liq_sweep_mss_5m_100k_r100",
        "liq_sweep_mss",
        "5m",
        "100k",
        tuple(sorted(params.items())),
        "phase13_search_100k_5m",
        fp,
        1000.0,
        0.2,
        "t",
    )
    assert fh.config_hash() == fp  # Phase-13 family reconstructs from the registry without tuning
