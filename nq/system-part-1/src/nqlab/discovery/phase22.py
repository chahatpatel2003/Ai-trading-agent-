"""Phase 22 — targeted 60-70% win-rate, 1:1 R:R, $200-risk Topstep search.

Reuses the Phase-16/18 payout-first rig with a hard WIN-RATE FLOOR of 60% (the explicit Phase-22
acceptance rule: a 1:1 system must win >= 60% to clear PF 1.30). The Phase-18 verdict already caps
acceptance at ACCEPT_FOR_FORWARD_PAPER and emits NEEDS_FORWARD_DATA for the MBP1-required family.
No forced acceptance.
"""

from __future__ import annotations

from nqlab.discovery.phase16 import Phase16Thresholds, select_phase16_survivors
from nqlab.discovery.phase18 import phase18_verdict, run_phase18_lockbox
from nqlab.discovery.phase22_families import PHASE22_FAMILIES, generate_phase22_candidates

__all__ = [
    "PHASE22_FAMILIES",
    "PHASE22_THRESHOLDS",
    "generate_phase22_candidates",
    "phase18_verdict",
    "run_phase18_lockbox",
    "select_phase16_survivors",
]

PHASE22_THRESHOLDS = Phase16Thresholds(
    pf_min=1.30,
    pf_preferred=1.70,
    win_rate_min=0.60,  # the Phase-22 claim: a 1:1 system must win >= 60% after costs
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
