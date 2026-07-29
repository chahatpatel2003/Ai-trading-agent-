"""Phase 24 — four strict prop-style 1:1 scalping models acceptance.

Reuses the Phase-16/18 payout-first rig with a 60% win-rate floor (the explicit Phase-24 rule).
Verdict ceiling ACCEPT_FOR_FORWARD_PAPER. No forced acceptance; the models are the strict,
filtered prop versions (not watered-down generic VWAP/OR/sweep tests).
"""

from __future__ import annotations

from nqlab.discovery.phase16 import Phase16Thresholds, select_phase16_survivors
from nqlab.discovery.phase18 import phase18_verdict, run_phase18_lockbox
from nqlab.discovery.phase24_families import PHASE24_FAMILIES, generate_phase24_candidates

__all__ = [
    "PHASE24_FAMILIES",
    "PHASE24_THRESHOLDS",
    "generate_phase24_candidates",
    "phase18_verdict",
    "run_phase18_lockbox",
    "select_phase16_survivors",
]

PHASE24_THRESHOLDS = Phase16Thresholds(
    pf_min=1.30,
    pf_preferred=1.70,
    win_rate_min=0.60,  # Phase-24 acceptance: prop 1:1 models must win >= 60% after costs
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
