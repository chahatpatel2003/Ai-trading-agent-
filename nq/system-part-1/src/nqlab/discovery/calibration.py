"""Percentile-calibrated grids for the combination families.

Thresholds are read from a :class:`DistributionProfile` (built on the training
slice only) — never guessed. Imbalance/spread/liquidity use feature percentiles;
delta and CVD-slope use rolling z-scores (standardized), so their thresholds are
in z-units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from nqlab.discovery.candidates import (
    AbsorptionReclaim,
    BiasAlignedPullback,
    CvdConfirmationPullback,
    DeltaDivergenceVwap,
    FailedBreakoutTrap,
    ImbalanceTrendCont,
    LiquidityRegimeBreakout,
    OpeningRangeContext,
    PrevDaySweepReversal,
    SessionSweepReversal,
    _Family,
)
from nqlab.features.mbp1_profile import DistributionProfile

_RTH_OPEN_DEFAULT = 570
_Z_WINDOW = 100


@dataclass(frozen=True)
class CalibratedThresholds:
    imb: tuple[float, ...]
    spread_max: tuple[float, ...]
    liq_thr: tuple[float, ...]
    z_thr: tuple[float, ...] = (1.0, 1.5, 2.0)
    vwap_k: tuple[float, ...] = (1.5, 2.5)


def _dedup(values: list[float], fallback: float) -> tuple[float, ...]:
    clean = sorted({round(v, 6) for v in values if v is not None and not math.isnan(v)})
    return tuple(clean) if clean else (fallback,)


def calibrate(profile: DistributionProfile) -> CalibratedThresholds:
    """Derive thresholds from training-slice percentiles."""

    def pct(feature: str, q: float, default: float) -> float:
        try:
            value = profile.percentile(feature, q)
        except KeyError:
            return default
        return default if math.isnan(value) else value

    return CalibratedThresholds(
        imb=_dedup([pct("abs_tob_imbalance", q, 0.2) for q in (0.6, 0.7, 0.8, 0.9)], 0.2),
        spread_max=_dedup([pct("spread", q, 1.0) for q in (0.5, 0.75, 0.9)], 1.0),
        liq_thr=_dedup([pct("liq_spread_ratio", q, 1.0) for q in (0.25, 0.5, 0.75)], 1.0),
    )


def generate_calibrated_candidates(
    thresholds: CalibratedThresholds,
    *,
    rth_start_min: int = _RTH_OPEN_DEFAULT,
    z_window: int = _Z_WINDOW,
) -> list[_Family]:
    """Build the combination-family candidate list from calibrated thresholds."""
    t = thresholds
    out: list[_Family] = []

    for z in t.z_thr:
        for stop in (1.0, 1.5):
            out.append(AbsorptionReclaim(
                {"z_thr": z, "z_window": z_window, "stop_atr": stop, "target_atr": 2.0},
                rth_start_min=rth_start_min,
            ))
    for imb in t.imb[:2]:
        for spread_max in t.spread_max[-2:]:
            for ema in (20, 50):
                out.append(ImbalanceTrendCont(
                    {"imb_thr": imb, "spread_max": spread_max, "ema_span": ema,
                     "stop_atr": 1.5, "target_atr": 2.0},
                    rth_start_min=rth_start_min,
                ))
    for k in t.vwap_k:
        for stop in (1.0, 1.5):
            out.append(DeltaDivergenceVwap(
                {"vwap_k": k, "z_window": z_window, "stop_atr": stop, "target_atr": 2.0},
                rth_start_min=rth_start_min,
            ))
    for liq in t.liq_thr[-2:]:
        for spread_max in t.spread_max[-2:]:
            out.append(LiquidityRegimeBreakout(
                {"liq_thr": liq, "spread_max": spread_max, "stop_atr": 1.5, "target_atr": 2.5},
                rth_start_min=rth_start_min,
            ))
    for ema in (20, 50):
        for target in (2.5, 3.5):
            out.append(CvdConfirmationPullback(
                {"ema_span": ema, "rsi_low": 40.0, "rsi_high": 60.0, "z_window": z_window,
                 "stop_atr": 1.5, "target_atr": target},
                rth_start_min=rth_start_min,
            ))
    return out


def generate_setup_candidates(
    thresholds: CalibratedThresholds, *, rth_start_min: int = _RTH_OPEN_DEFAULT
) -> list[_Family]:
    """Build the context-setup candidate list (small focused grids)."""
    t = thresholds
    out: list[_Family] = []
    for session in ("asia", "london"):
        for stop in (1.0, 1.5):
            out.append(SessionSweepReversal(
                {"session": session, "stop_atr": stop, "target_atr": 2.0},
                rth_start_min=rth_start_min,
            ))
    for stop in (1.0, 1.5):
        for target in (1.5, 2.5):
            out.append(PrevDaySweepReversal(
                {"stop_atr": stop, "target_atr": target}, rth_start_min=rth_start_min
            ))
    for target in (2.5, 3.5):
        out.append(BiasAlignedPullback(
            {"stop_atr": 1.5, "target_atr": target}, rth_start_min=rth_start_min
        ))
    for liq in t.liq_thr[-2:]:
        for spread_max in t.spread_max[-2:]:
            out.append(OpeningRangeContext(
                {"liq_thr": liq, "spread_max": spread_max, "stop_atr": 1.5, "target_atr": 2.0},
                rth_start_min=rth_start_min,
            ))
    for stop in (1.0, 1.5):
        out.append(FailedBreakoutTrap(
            {"stop_atr": stop, "target_atr": 2.0}, rth_start_min=rth_start_min
        ))
    return out
