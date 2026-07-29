from __future__ import annotations

from nqlab.discovery.optimization import (
    FOCUS_FAMILIES,
    RISK_GRID,
    generate_exit_candidates,
    generate_risk_candidates,
    generate_survival_candidates,
)
from nqlab.domain.strategy.base import MBP1_REQUIRED


def test_exit_grid_generation() -> None:
    candidates = generate_exit_candidates()
    assert len(candidates) > 0
    fingerprints = [c.params.fingerprint() for c in candidates]
    assert len(fingerprints) == len(set(fingerprints))
    assert {c.family for c in candidates} <= set(FOCUS_FAMILIES)
    params = candidates[0].params.as_dict()
    assert "stop_atr" in params and "target_atr" in params


def test_risk_sweep_sizing() -> None:
    candidates = generate_risk_candidates(families=("bias_pullback",))
    assert len(candidates) == len(RISK_GRID)
    risks = sorted(c.exec_config.risk_per_trade_usd for c in candidates)  # type: ignore[union-attr]
    assert risks == sorted(RISK_GRID)
    assert all(c.exec_config is not None for c in candidates)


def test_survival_grid_is_unique_and_mbp1() -> None:
    candidates = generate_survival_candidates()
    fingerprints = [c.params.fingerprint() for c in candidates]
    assert len(fingerprints) == len(set(fingerprints))
    assert all(c.data_requirement == MBP1_REQUIRED for c in candidates)
    # Sessions seeded from diagnostics appear in the grid.
    assert any(c.params.as_dict().get("hour_start") == 9 for c in candidates)


def test_min_hold_seeded_in_survival() -> None:
    candidates = generate_survival_candidates(families=("pd_sweep_rev",))
    assert all(c.exec_config is not None for c in candidates)
    assert all(c.exec_config.min_hold_bars == 6 for c in candidates)  # type: ignore[union-attr]
