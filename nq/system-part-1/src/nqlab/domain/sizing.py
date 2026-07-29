"""Position sizing from a fixed dollar risk, with selectable instrument modes.

Default rule (``adaptive``): size NQ to the dollar risk; if even one NQ contract
exceeds the risk budget, fall back to MNQ; if one MNQ contract still exceeds it,
skip the trade. Phase 8 adds explicit modes (NQ-only, MNQ-only, max-1-NQ,
MNQ-scaling) to test whether instrument granularity changes survivability.
Contract counts are capped by ``max_contracts`` (micros allowed 10x in scaling).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from nqlab.config.models import InstrumentSpec

SizingMode = Literal[
    "adaptive", "nq_only", "mnq_only", "max_1_nq", "mnq_scaling", "mnq_min1", "mnq_fixed1"
]
SIZING_MODES: tuple[SizingMode, ...] = (
    "adaptive",
    "nq_only",
    "mnq_only",
    "max_1_nq",
    "mnq_scaling",
)
# Phase 14 sizing-repair modes: never round to 0 contracts when the trade is affordable
# within the hard per-trade max-loss cap (``mnq_min1`` floors at 1; ``mnq_fixed1`` is
# always 1). The fix for high-ATR regimes where floor(risk/contract_risk) == 0.
SIZING_REPAIR_MODES: tuple[SizingMode, ...] = ("mnq_min1", "mnq_fixed1")

# Topstep counts 10 micros against 1 mini of the contract limit; MNQ scaling
# may stack up to this multiple of the mini cap for finer-grained sizing.
_MICRO_PER_MINI = 10


@dataclass(frozen=True)
class SizingDecision:
    instrument: str  # "NQ" | "MNQ" | "" (skipped)
    contracts: int
    per_contract_risk: float
    skipped: bool
    reason: str = ""


def _nq(stop_points: float, risk_usd: float, nq: InstrumentSpec, cap: int) -> SizingDecision | None:
    nq_risk = stop_points * nq.point_value
    n = math.floor(risk_usd / nq_risk)
    if n >= 1:
        return SizingDecision(nq.root, min(n, cap), nq_risk, False)
    return None


def _mnq(stop_points: float, risk_usd: float, mnq: InstrumentSpec, cap: int) -> SizingDecision:
    mnq_risk = stop_points * mnq.point_value
    n = math.floor(risk_usd / mnq_risk)
    if n >= 1:
        return SizingDecision(mnq.root, min(n, cap), mnq_risk, False)
    return SizingDecision("", 0, mnq_risk, True, "one MNQ contract exceeds risk budget")


def size_position(
    *,
    stop_points: float,
    risk_usd: float,
    nq: InstrumentSpec,
    mnq: InstrumentSpec,
    max_contracts: int,
    mode: SizingMode = "adaptive",
    max_loss_usd: float | None = None,
) -> SizingDecision:
    """Choose instrument and contract count for a trade with ``stop_points`` risk.

    ``max_loss_usd`` is the hard per-trade dollar cap used by the repair modes: they
    take >= 1 MNQ contract as long as one contract's stop-risk fits the cap, and skip
    only when the stop is so wide that even one contract breaches it.
    """
    if stop_points <= 0:
        return SizingDecision("", 0, 0.0, True, "non-positive stop distance")

    if mode in SIZING_REPAIR_MODES:
        mnq_risk = stop_points * mnq.point_value
        if max_loss_usd is not None and mnq_risk > max_loss_usd:
            return SizingDecision("", 0, mnq_risk, True, "stop exceeds absolute max risk per trade")
        n = 1 if mode == "mnq_fixed1" else max(1, math.floor(risk_usd / mnq_risk))
        return SizingDecision(mnq.root, min(n, max_contracts), mnq_risk, False)

    if mode == "nq_only":
        return _nq(stop_points, risk_usd, nq, max_contracts) or SizingDecision(
            "", 0, stop_points * nq.point_value, True, "one NQ contract exceeds risk budget"
        )
    if mode == "mnq_only":
        return _mnq(stop_points, risk_usd, mnq, max_contracts)
    if mode == "mnq_scaling":
        return _mnq(stop_points, risk_usd, mnq, max_contracts * _MICRO_PER_MINI)
    if mode == "max_1_nq":
        nq_dec = _nq(stop_points, risk_usd, nq, 1)
        return nq_dec if nq_dec is not None else _mnq(stop_points, risk_usd, mnq, max_contracts)

    # adaptive (default): NQ first, MNQ fallback.
    return _nq(stop_points, risk_usd, nq, max_contracts) or _mnq(
        stop_points, risk_usd, mnq, max_contracts
    )
