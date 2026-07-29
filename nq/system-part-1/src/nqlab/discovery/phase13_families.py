"""Phase 13 setup families — NQ-only market structure + liquidity + displacement.

A new signal source (not the retired Phase 6-8 order-flow pullbacks, not the Phase 9
HTF asymmetric candidates, not the Phase 11/12 trend_day variants). Built on the
PIT-safe structure layer (:mod:`nqlab.features.structure`): swings, BOS/CHoCH/MSS,
liquidity sweeps + reclaims, displacement, fair value gaps, premium/discount.

All NQ-only, trades-derivable (so the 2025-2026 lockbox stays clean), NY-AM, and
asymmetric (tight stop beyond the sweep/FVG, wide target into opposing liquidity). The
diagnostics-first prefilter still rejects anything whose realized MAE >= MFE.
"""

from __future__ import annotations

import polars as pl

from nqlab.discovery.candidates import _Family

_RTH_OPEN_DEFAULT = 570

P13_EXIT_BY_TF: dict[str, dict[str, object]] = {
    "5m": {"min_hold_bars": 3, "time_stop_bars": 48},
    "15m": {"min_hold_bars": 2, "time_stop_bars": 24},
    "1m": {"min_hold_bars": 6, "time_stop_bars": 90},
}
P13_RISK_GRID: tuple[float, ...] = (50.0, 100.0, 150.0, 200.0)
P13_SIZING_MODES: tuple[str, ...] = ("mnq_only", "adaptive")


class _P13(_Family):
    version = "13"


class LiqSweepMss(_P13):
    """A — Liquidity sweep + MSS/CHoCH reversal: sweep sell-side in discount (buy-side in
    premium), reclaim, structure shifts; reverse with a stop beyond the sweep."""

    family = "liq_sweep_mss"
    required_columns = ("swept_low", "swept_high", "in_discount", "in_premium", "struct_dir")

    def _long(self) -> pl.Expr:
        return (pl.col("swept_low") == 1) & (pl.col("in_discount") == 1)

    def _short(self) -> pl.Expr:
        return (pl.col("swept_high") == 1) & (pl.col("in_premium") == 1)

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": 1.0, "target_atr": tgt} for tgt in (3.0, 4.0)]


class BosContinuation(_P13):
    """B — BOS continuation: HTF bias agrees, break of structure with displacement,
    continuation toward the next liquidity pool."""

    family = "bos_continuation"
    required_columns = ("struct_dir", "bos_up", "bos_dn", "displacement", "trend_4h_bull")

    def _long(self) -> pl.Expr:
        return (
            (pl.col("struct_dir") == 1)
            & (pl.col("bos_up") == 1)
            & (pl.col("displacement") == 1)
            & (pl.col("trend_4h_bull") == 1)
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("struct_dir") == -1)
            & (pl.col("bos_dn") == 1)
            & (pl.col("displacement") == 1)
            & (pl.col("trend_4h_bull") == 0)
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": 1.5, "target_atr": tgt} for tgt in (3.0, 4.0)]


class FailedBreakoutTrap(_P13):
    """C — Failed breakout trap: break the prior-day extreme, fail to continue, reclaim,
    trade the opposite way toward the range midpoint / opposing liquidity."""

    family = "failed_breakout_trap"
    required_columns = ("prev_day_high", "prev_day_low")

    def _long(self) -> pl.Expr:  # failed breakdown of PDL, reclaimed
        return (pl.col("low") < pl.col("prev_day_low")) & (pl.col("close") > pl.col("prev_day_low"))

    def _short(self) -> pl.Expr:  # failed breakout of PDH, rejected
        return (pl.col("high") > pl.col("prev_day_high")) & (
            pl.col("close") < pl.col("prev_day_high")
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": 1.0, "target_atr": tgt} for tgt in (2.0, 3.0)]


class FvgDisplacement(_P13):
    """D — FVG displacement continuation: displacement leaves a fair value gap; with HTF
    bias + structure agreeing, enter on mitigation of the gap toward liquidity."""

    family = "fvg_displacement"
    required_columns = ("in_bull_fvg", "in_bear_fvg", "struct_dir", "trend_4h_bull")

    def _long(self) -> pl.Expr:
        return (
            (pl.col("in_bull_fvg") == 1)
            & (pl.col("struct_dir") == 1)
            & (pl.col("trend_4h_bull") == 1)
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("in_bear_fvg") == 1)
            & (pl.col("struct_dir") == -1)
            & (pl.col("trend_4h_bull") == 0)
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": 1.5, "target_atr": tgt} for tgt in (3.0, 4.0)]


class PremiumDiscount(_P13):
    """E — Premium/discount liquidity model: only long in discount after a sell-side
    sweep, only short in premium after a buy-side sweep, with displacement/MSS confirm."""

    family = "premium_discount"
    required_columns = ("in_discount", "in_premium", "swept_low", "swept_high", "displacement")

    def _long(self) -> pl.Expr:
        return (
            (pl.col("in_discount") == 1)
            & (pl.col("swept_low") == 1)
            & ((pl.col("displacement") == 1) | (pl.col("mss_up") == 1))
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("in_premium") == 1)
            & (pl.col("swept_high") == 1)
            & ((pl.col("displacement") == 1) | (pl.col("mss_dn") == 1))
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": 1.0, "target_atr": tgt} for tgt in (3.0, 4.0)]


PHASE13_FAMILIES: dict[str, type[_P13]] = {
    "liq_sweep_mss": LiqSweepMss,
    "bos_continuation": BosContinuation,
    "failed_breakout_trap": FailedBreakoutTrap,
    "fvg_displacement": FvgDisplacement,
    "premium_discount": PremiumDiscount,
}


def generate_phase13_candidates(
    families: tuple[str, ...] = tuple(PHASE13_FAMILIES),
    *,
    timeframe: str,
    risk_grid: tuple[float, ...] = P13_RISK_GRID,
    sizing_modes: tuple[str, ...] = P13_SIZING_MODES,
    hour_window: tuple[int, int] = (9, 12),
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_P13]:
    """family x grid x risk x sizing, TF-seeded holds, NY-AM session. Deterministic."""
    exit_seed = P13_EXIT_BY_TF.get(timeframe, P13_EXIT_BY_TF["5m"])
    h0, h1 = hour_window
    out: list[_P13] = []
    for fam in families:
        cls = PHASE13_FAMILIES[fam]
        for base in cls.grid():
            for risk in risk_grid:
                for mode in sizing_modes:
                    params: dict[str, object] = {
                        **base,
                        **exit_seed,
                        "risk_per_trade": risk,
                        "sizing_mode": mode,
                        "hour_start": h0,
                        "hour_end": h1,
                    }
                    out.append(cls(params, rth_start_min=rth_start_min))
    return out
