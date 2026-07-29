"""Phase 25 — daily-bias + HTF-trend + HTF-POI + LTF-reclaim model acceptance.

Reuses the Phase-16/18 payout-first rig. Win-rate floor 55% (the Phase-25 early-reject for 1:1).
Verdict ceiling ACCEPT_FOR_FORWARD_PAPER. Daily-bias + HTF-trend alignment is a MANDATORY gate in
every family — the model is not watered down into generic ICT components. No forced acceptance.
"""

from __future__ import annotations

from nqlab.discovery.phase16 import Phase16Thresholds, select_phase16_survivors
from nqlab.discovery.phase18 import phase18_verdict, run_phase18_lockbox
from nqlab.discovery.phase25_families import PHASE25_FAMILIES, generate_phase25_candidates

__all__ = [
    "PHASE25_FAMILIES",
    "PHASE25_THRESHOLDS",
    "generate_phase25_candidates",
    "phase18_verdict",
    "run_phase18_lockbox",
    "select_phase16_survivors",
]

PHASE25_THRESHOLDS = Phase16Thresholds(
    pf_min=1.30,
    pf_preferred=1.70,
    win_rate_min=0.55,
    ruin_max=0.10,
    pass_pref=0.70,
    green_month_min=0.75,
    min_trades=40,
    min_winning_days_200=5,
    min_winning_day_rate=0.10,
    min_payout_eligible_frac=0.60,
    scale_pf_min=1.20,
    min_avg_contracts_scale=1.5,
    dsr_threshold=0.90,
)
