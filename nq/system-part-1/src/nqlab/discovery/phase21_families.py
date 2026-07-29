"""Phase 21 — MBP1 order-flow microstructure families (A-F).

Tests whether order flow contains a short-horizon edge that bars/indicators could not see. Each
family is MBP1-required (declares ``MBP1_REQUIRED`` so it skips bars without clean MBP1), NY-AM
only, and combines order-flow dynamics (OFI/delta z-scores, CVD momentum, book depletion,
imbalance persistence — from :mod:`nqlab.features.orderflow`) with key price levels (swept swings,
recent range, VWAP). ATR-scaled stops/targets, deliberately small grids. None assumed to work.

MBP1 only exists through 2025-H1, so any verdict here is capped at ACCEPT_FOR_FORWARD_PAPER.
"""

from __future__ import annotations

import polars as pl

from nqlab.discovery.candidates import _Family
from nqlab.domain.strategy.base import MBP1_REQUIRED
from nqlab.features.orderflow import add_orderflow_features

_RTH_OPEN_DEFAULT = 570
P21_EXIT_BY_TF: dict[str, dict[str, object]] = {
    "1m": {"min_hold_bars": 1, "time_stop_bars": 60},
    "5m": {"min_hold_bars": 1, "time_stop_bars": 18},
    "15m": {"min_hold_bars": 1, "time_stop_bars": 8},
}
P21_SIZING_MODES: tuple[str, ...] = ("mnq_min1",)
P21_RISK_GRID: tuple[float, ...] = (250.0,)
P21_Z: tuple[float, ...] = (0.75, 1.25)  # order-flow z thresholds
P21_STOPTGT: tuple[tuple[float, float], ...] = ((1.0, 1.5),)


def _grid(*param_sets: dict[str, object]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for ps in param_sets:
        for z in P21_Z:
            for stop, tgt in P21_STOPTGT:
                out.append({**ps, "z": z, "stop_atr": stop, "target_atr": tgt})
    return out


class _P21(_Family):
    version = "21"
    data_requirement = MBP1_REQUIRED

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        # Order-flow dynamics + PRIOR-bar breakout levels (shifted, so a break of the level is
        # actually attainable — the base roll_high/roll_low include the current bar).
        return add_orderflow_features(frame).with_columns(
            pl.col("high").rolling_max(window_size=20).shift(1).alias("_brk_hi"),
            pl.col("low").rolling_min(window_size=20).shift(1).alias("_brk_lo"),
        )


class AbsorptionReversal(_P21):
    """A — Absorption reversal: at a swept swing, aggressive flow continues INTO the level
    (heavy delta) but price stalls/reclaims -> the flow is absorbed -> reverse."""

    family = "p21_absorption_rev"
    required_columns = ("swept_high", "swept_low", "delta", "tob_imbalance", "cvd")

    def _long(self) -> pl.Expr:  # sellers absorbed at a swept low
        return (
            (pl.col("swept_low") == 1)
            & (pl.col("delta_z") < -float(self.p["z"]))  # type: ignore[arg-type]
            & (pl.col("close") > pl.col("open"))
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("swept_high") == 1)
            & (pl.col("delta_z") > float(self.p["z"]))  # type: ignore[arg-type]
            & (pl.col("close") < pl.col("open"))
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({})


class ImbalanceContinuation(_P21):
    """B — Imbalance continuation: strong OFI + persistent top-of-book imbalance pushing price
    through a recent level -> ride the continuation."""

    family = "p21_imbalance_cont"
    required_columns = ("ofi", "tob_imbalance")

    def _long(self) -> pl.Expr:
        return (
            (pl.col("ofi_z") > float(self.p["z"]))  # type: ignore[arg-type]
            & (pl.col("imb_persist_up") == 1)
            & (pl.col("close") > pl.col("_brk_hi"))
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("ofi_z") < -float(self.p["z"]))  # type: ignore[arg-type]
            & (pl.col("imb_persist_dn") == 1)
            & (pl.col("close") < pl.col("_brk_lo"))
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({})


class FailedAggressiveFlow(_P21):
    """C — Failed aggressive-flow trap: a heavy aggressive push pokes a level but fails to
    continue and reclaims -> fade the trapped aggressors."""

    family = "p21_failed_flow"
    required_columns = ("delta",)

    def _long(self) -> pl.Expr:  # sellers pushed below the prior low, failed, reclaimed
        return (
            (pl.col("low") < pl.col("_brk_lo"))
            & (pl.col("close") > pl.col("_brk_lo"))
            & (pl.col("delta_z") < -float(self.p["z"]))  # type: ignore[arg-type]
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("high") > pl.col("_brk_hi"))
            & (pl.col("close") < pl.col("_brk_hi"))
            & (pl.col("delta_z") > float(self.p["z"]))  # type: ignore[arg-type]
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({})


class LiquidityVacuum(_P21):
    """D — Liquidity-vacuum breakout: resting size pulls from one side (depletion) and aggressive
    flow confirms a level break, only when spread/liquidity is acceptable."""

    family = "p21_liq_vacuum"
    required_columns = ("ask_depletion", "bid_depletion", "ofi", "liq_spread_ratio")

    def _long(self) -> pl.Expr:  # ask liquidity pulled + buy OFI + breaks up + clean spread
        return (
            (pl.col("ask_depletion") > pl.col("bid_depletion"))
            & (pl.col("ofi_z") > float(self.p["z"]))  # type: ignore[arg-type]
            & (pl.col("close") > pl.col("_brk_hi"))
            & (pl.col("liq_spread_ratio") < 1.0)
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("bid_depletion") > pl.col("ask_depletion"))
            & (pl.col("ofi_z") < -float(self.p["z"]))  # type: ignore[arg-type]
            & (pl.col("close") < pl.col("_brk_lo"))
            & (pl.col("liq_spread_ratio") < 1.0)
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({})


class DeltaDivergenceReversal(_P21):
    """E — Delta-divergence reversal: price makes a new short-term extreme but aggressive flow
    (delta/CVD) does not confirm -> reverse after the failure."""

    family = "p21_delta_div"
    required_columns = ("delta", "cvd")

    def _long(self) -> pl.Expr:  # new low vs prior bars, but buyers (positive delta) -> divergence
        return (pl.col("low") <= pl.col("_brk_lo")) & (pl.col("delta_z") > float(self.p["z"]))  # type: ignore[arg-type]

    def _short(self) -> pl.Expr:
        return (pl.col("high") >= pl.col("_brk_hi")) & (pl.col("delta_z") < -float(self.p["z"]))  # type: ignore[arg-type]

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({})


class AdverseSelectionFilter(_P21):
    """F — Adverse-selection-avoidance continuation: trade with VWAP trend ONLY when order flow
    confirms (OFI) and liquidity is clean (acceptable spread) — the MBP1 'filter' as a system."""

    family = "p21_adverse_filter"
    required_columns = ("vwap_session", "ofi", "liq_spread_ratio")

    def _long(self) -> pl.Expr:
        return (
            (pl.col("close") > pl.col("vwap_session"))
            & (pl.col("ofi_z") > float(self.p["z"]))  # type: ignore[arg-type]
            & (pl.col("liq_spread_ratio") < 1.0)
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("close") < pl.col("vwap_session"))
            & (pl.col("ofi_z") < -float(self.p["z"]))  # type: ignore[arg-type]
            & (pl.col("liq_spread_ratio") < 1.0)
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({})


PHASE21_FAMILIES: dict[str, type[_P21]] = {
    "p21_absorption_rev": AbsorptionReversal,
    "p21_imbalance_cont": ImbalanceContinuation,
    "p21_failed_flow": FailedAggressiveFlow,
    "p21_liq_vacuum": LiquidityVacuum,
    "p21_delta_div": DeltaDivergenceReversal,
    "p21_adverse_filter": AdverseSelectionFilter,
}


def generate_phase21_candidates(
    families: tuple[str, ...] = tuple(PHASE21_FAMILIES),
    *,
    timeframe: str,
    risk_grid: tuple[float, ...] = P21_RISK_GRID,
    sizing_modes: tuple[str, ...] = P21_SIZING_MODES,
    hour_window: tuple[int, int] = (9, 12),
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_P21]:
    """family x grid x risk x sizing, TF-seeded holds, NY-AM session. Deterministic."""
    exit_seed = P21_EXIT_BY_TF.get(timeframe, P21_EXIT_BY_TF["5m"])
    h0, h1 = hour_window
    out: list[_P21] = []
    for fam in families:
        cls = PHASE21_FAMILIES[fam]
        for base in cls.grid():
            for risk in risk_grid:
                for mode in sizing_modes:
                    out.append(
                        cls(
                            {**base, **exit_seed, "risk_per_trade": risk, "sizing_mode": mode,
                             "hour_start": h0, "hour_end": h1},
                            rth_start_min=rth_start_min,
                        )
                    )
    return out
