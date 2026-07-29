"""Phase 23 — HTF POI → LTF reclaim/MSS model acceptance.

Reuses the Phase-16/18 payout-first rig. Win-rate floor is 55% (the explicit Phase-23 early-reject
for a 1:1 system). Verdict ceiling is ACCEPT_FOR_FORWARD_PAPER (no live without forward data).
No forced acceptance, and the model is NOT watered down — every candidate is the full sequence
(HTF POI touch + reclaim + the variant's LTF confirmation), not a generic component.
"""

from __future__ import annotations

from nqlab.discovery.phase16 import Phase16Thresholds, select_phase16_survivors
from nqlab.discovery.phase18 import phase18_verdict, run_phase18_lockbox
from nqlab.discovery.phase23_families import PHASE23_FAMILIES, generate_phase23_candidates

__all__ = [
    "PHASE23_FAMILIES",
    "PHASE23_THRESHOLDS",
    "generate_phase23_candidates",
    "phase18_verdict",
    "run_phase18_lockbox",
    "select_phase16_survivors",
]

PHASE23_THRESHOLDS = Phase16Thresholds(
    pf_min=1.30,
    pf_preferred=1.70,
    win_rate_min=0.55,  # Phase-23 early-reject: 1:1 needs >= 55% win after costs
    ruin_max=0.10,
    pass_pref=0.70,
    green_month_min=0.75,
    min_trades=40,  # 1 trade/day, selective POI reactions -> lower frequency tolerated
    min_winning_days_200=5,
    min_winning_day_rate=0.10,
    min_payout_eligible_frac=0.60,
    scale_pf_min=1.20,
    min_avg_contracts_scale=1.5,
    dsr_threshold=0.90,
)
