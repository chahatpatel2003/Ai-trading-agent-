"""Phase 21 — MBP1 order-flow microstructure acceptance (likely final NQ-only NY-AM attempt).

Reuses the Phase-16/18 payout-first rig. The Phase-18 verdict already caps acceptance at
``ACCEPT_FOR_FORWARD_PAPER`` and emits ``NEEDS_FORWARD_DATA`` for MBP1-without-clean-lockbox —
which is exactly the right ceiling here: MBP1 ends at 2025-H1, so even a clean pass on the
2025-H1 MBP1 lockbox can only earn forward-paper (there is no post-2025-H1 MBP1 to confirm live).
No forced acceptance.
"""

from __future__ import annotations

from nqlab.discovery.phase16 import Phase16Thresholds, select_phase16_survivors
from nqlab.discovery.phase18 import phase18_verdict, run_phase18_lockbox
from nqlab.discovery.phase21_families import PHASE21_FAMILIES, generate_phase21_candidates

__all__ = [
    "PHASE21_FAMILIES",
    "PHASE21_THRESHOLDS",
    "generate_phase21_candidates",
    "phase18_verdict",
    "run_phase18_lockbox",
    "select_phase16_survivors",
]

PHASE21_THRESHOLDS = Phase16Thresholds(
    pf_min=1.30,
    pf_preferred=1.50,
    win_rate_min=0.0,  # not gated globally; PF is the real filter (reversal families reported)
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
