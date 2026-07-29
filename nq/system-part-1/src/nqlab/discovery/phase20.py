"""Phase 20 — classic indicator strategy acceptance (last non-MBP1, bar-based phase).

Reuses the Phase-16/18 payout-first rig wholesale (business stats, daily-control sweep, two-stage
funded scaling, fleet, forward-paper-ceiling verdict). Only the families and the threshold object
differ. The win-rate floor is left at 0 (trend systems can pass on PF with sub-50% win); PF >= 1.30
after costs, ruin <= 10%, green months >= 75%, payout-eligible >= 60% remain the real gates.
No forced acceptance; ceiling is ACCEPT_FOR_FORWARD_PAPER.
"""

from __future__ import annotations

from nqlab.discovery.phase16 import Phase16Thresholds, select_phase16_survivors
from nqlab.discovery.phase18 import phase18_verdict, run_phase18_lockbox
from nqlab.discovery.phase20_families import PHASE20_FAMILIES, generate_phase20_candidates

__all__ = [
    "PHASE20_FAMILIES",
    "PHASE20_THRESHOLDS",
    "generate_phase20_candidates",
    "phase18_verdict",
    "run_phase18_lockbox",
    "select_phase16_survivors",
]

PHASE20_THRESHOLDS = Phase16Thresholds(
    pf_min=1.30,
    pf_preferred=1.50,
    win_rate_min=0.0,  # not gated globally (trend systems can pass <50% win); PF is the gate
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
