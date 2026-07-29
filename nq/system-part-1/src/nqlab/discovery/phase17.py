"""Phase 17 — business-worthy Topstep payout strategy acceptance.

Reuses the Phase-16 payout-first rig (business stats, two-stage eval+funded simulation, fleet
scaler) with a tighter bar and the high-conviction A+ families. Adds the ``NEEDS_FORWARD_DATA``
verdict for MBP1-required strategies that lack a clean MBP1 lockbox. No forced acceptance.
"""

from __future__ import annotations

from typing import Any

from nqlab.discovery.lockbox import NEEDS_FORWARD_DATA, NEEDS_MORE_DATA
from nqlab.discovery.phase16 import (
    Phase16Thresholds,
    business_stats,
    phase16_verdict,
    run_phase16_lockbox,
    select_phase16_survivors,
)
from nqlab.discovery.phase17_families import PHASE17_FAMILIES, generate_phase17_candidates

__all__ = [
    "PHASE17_FAMILIES",
    "PHASE17_THRESHOLDS",
    "business_stats",
    "generate_phase17_candidates",
    "phase17_verdict",
    "run_phase16_lockbox",
    "select_phase16_survivors",
]

# Phase 17 bar: PF >= 1.30 (prefer 1.50), green months >= 75%, payout-eligible >= 60% of months,
# ruin <= 10%, pass >= 70% preferred. Same scalability + DSR gates as Phase 16.
PHASE17_THRESHOLDS = Phase16Thresholds(
    pf_min=1.30,
    pf_preferred=1.50,
    ruin_max=0.10,
    pass_pref=0.70,
    green_month_min=0.75,
    min_trades=60,
    min_winning_days_200=5,
    min_winning_day_rate=0.10,
    min_payout_eligible_frac=0.60,
    scale_pf_min=1.20,
    min_avg_contracts_scale=1.5,
    dsr_threshold=0.90,
)


def phase17_verdict(
    *,
    gauntlet_verdict: str,
    dsr: float,
    business: dict[str, Any],
    scale: dict[str, Any],
    thresholds: Phase16Thresholds,
    pristine: bool,
) -> tuple[str, str]:
    """Phase-17 verdict = Phase-16 payout-first bar + the MBP1 forward-data ceiling.

    An MBP1-required strategy whose lockbox MBP1 is not clean cannot exceed
    ``NEEDS_FORWARD_DATA`` (the 2026-06 lockbox month has no MBP1). The MBP1 status is read
    from ``business`` (``requires_mbp1`` / ``mbp1_clean``); Phase-17 families are all
    trades-derivable, so this path is dormant for them but enforced for any MBP1 family.
    """
    base_verdict, reason = phase16_verdict(
        gauntlet_verdict=gauntlet_verdict, dsr=dsr, business=business, scale=scale,
        thresholds=thresholds, pristine=pristine,
    )
    requires_mbp1 = bool(business.get("requires_mbp1"))
    mbp1_clean = bool(business.get("mbp1_clean", True))
    if requires_mbp1 and not mbp1_clean and base_verdict not in (NEEDS_MORE_DATA,):
        return NEEDS_FORWARD_DATA, (
            "MBP1-required strategy without a clean MBP1 lockbox (2026-06 has no MBP1); "
            f"cannot exceed NEEDS_FORWARD_DATA. [would-be: {base_verdict} — {reason}]"
        )
    return base_verdict, reason
