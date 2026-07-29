"""Phase 17 — high-conviction, asymmetric A+ NQ-only NY-session families.

Six setups designed to produce LARGER winners (asymmetric R), not thin edges. Every family is
NY-AM only (hour mask 9-12 ∩ is_rth -> 09:30-12:00 ET, inside the 12:30 cutoff), runs on the
confirmable 5m timeframe, and reuses the PIT-safe structure / HTF / context feature layer (no new
lookahead, all trades-derivable -> the trades-only 2025-26 lockbox is clean for them).

Each family is anchored on a distinct moderate-frequency primary signal (sweep, reclaim, MSS,
range-expansion, opening-range break) so it trades often enough to judge, then gated for
conviction (location / structure / bias). None are assumed to work — they are rejected honestly
against the payout-first bar.
"""

from __future__ import annotations

from itertools import product

import polars as pl

from nqlab.discovery.candidates import _Family

_RTH_OPEN_DEFAULT = 570

P17_EXIT_BY_TF: dict[str, dict[str, object]] = {
    "5m": {"min_hold_bars": 3, "time_stop_bars": 48},
    "15m": {"min_hold_bars": 2, "time_stop_bars": 24},
}
# Sizing-repair modes that never round to 0 AND scale with risk, plus adaptive (multi-contract).
P17_SIZING_MODES: tuple[str, ...] = ("mnq_min1", "adaptive")
# Risk grid spans evaluation ($150-$250) through funded/scale sizing ($500).
P17_RISK_GRID: tuple[float, ...] = (150.0, 250.0, 500.0)
# Asymmetric R: tight stop, wide target -> larger winners.
P17_STOP_ATR: tuple[float, ...] = (1.0, 1.5)
P17_TARGET_ATR: tuple[float, ...] = (3.0, 5.0)


def _bias_up() -> pl.Expr:
    return pl.col("trend_4h_bull") == 1


def _bias_dn() -> pl.Expr:
    return pl.col("trend_4h_bull") == 0


class _P17(_Family):
    version = "17"
    required_columns: tuple[str, ...] = ("trend_4h_bull",)

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": s, "target_atr": t} for s, t in product(P17_STOP_ATR, P17_TARGET_ATR)]


class SweepReversalAPlus(_P17):
    """A — A+ liquidity sweep reversal: sweep a liquidity pool inside the discount/premium of the
    dealing range, with a displacement candle confirming the reversal toward opposing liquidity."""

    family = "p17_sweep_reversal"
    required_columns = ("swept_high", "swept_low", "in_discount", "in_premium", "displacement")

    def _long(self) -> pl.Expr:
        return (pl.col("swept_low") == 1) & (pl.col("in_discount") == 1) & (
            pl.col("displacement") == 1
        )

    def _short(self) -> pl.Expr:
        return (pl.col("swept_high") == 1) & (pl.col("in_premium") == 1) & (
            pl.col("displacement") == 1
        )


class ExpansionContinuation(_P17):
    """B — Expansion-day continuation: after a compression setup, a range-expansion structure break
    in the 4h-bias direction (ride the trending NY-AM expansion)."""

    family = "p17_expansion_cont"
    required_columns = ("trend_4h_bull", "range_expansion", "bos_up", "bos_dn")

    def _long(self) -> pl.Expr:
        return _bias_up() & (pl.col("range_expansion") == 1) & (pl.col("bos_up") == 1)

    def _short(self) -> pl.Expr:
        return _bias_dn() & (pl.col("range_expansion") == 1) & (pl.col("bos_dn") == 1)


class VolatilityExpansion(_P17):
    """C — Volatility expansion (no external news): abnormal pre-NY compression releasing into a
    confirmed range-expansion + displacement leg in the structural direction (skip chop)."""

    family = "p17_vol_expansion"
    required_columns = ("compression_setup", "range_expansion", "displacement", "struct_dir")

    def _long(self) -> pl.Expr:
        return (
            (pl.col("compression_setup") == 1)
            & (pl.col("range_expansion") == 1)
            & (pl.col("displacement") == 1)
            & (pl.col("struct_dir") == 1)
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("compression_setup") == 1)
            & (pl.col("range_expansion") == 1)
            & (pl.col("displacement") == 1)
            & (pl.col("struct_dir") == -1)
        )


class OpeningDriveAPlus(_P17):
    """D — Opening-drive A+ continuation: a strong NY opening-range break that ALSO breaks key
    prior-day liquidity in the bias direction (higher conviction than a bare OR break)."""

    family = "p17_opening_drive_aplus"
    required_columns = ("trend_4h_bull", "or_high", "or_low", "prev_day_high", "prev_day_low")

    def _long(self) -> pl.Expr:
        return (
            _bias_up()
            & pl.col("or_high").is_not_null()
            & (pl.col("close") > pl.col("or_high"))
            & (pl.col("close") > pl.col("prev_day_high"))
        )

    def _short(self) -> pl.Expr:
        return (
            _bias_dn()
            & pl.col("or_low").is_not_null()
            & (pl.col("close") < pl.col("or_low"))
            & (pl.col("close") < pl.col("prev_day_low"))
        )


class FailedBreakoutTrap(_P17):
    """E — Failed-breakout trap: a sweep of major liquidity that fails and RECLAIMS with a
    displacement candle (target the range midpoint / opposing liquidity)."""

    family = "p17_failed_breakout"
    required_columns = ("swept_high", "swept_low", "reclaim_up", "reclaim_dn", "displacement")

    def _long(self) -> pl.Expr:
        return (pl.col("swept_low") == 1) & (pl.col("reclaim_up") == 1) & (
            pl.col("displacement") == 1
        )

    def _short(self) -> pl.Expr:
        return (pl.col("swept_high") == 1) & (pl.col("reclaim_dn") == 1) & (
            pl.col("displacement") == 1
        )


class PremiumDiscountModel(_P17):
    """F — Premium/discount model: long only in discount after a market-structure shift up, short
    only in premium after a shift down (confirmed dealing-range mean-reversion + continuation)."""

    family = "p17_premium_discount"
    required_columns = ("in_discount", "in_premium", "mss_up", "mss_dn", "displacement")

    def _long(self) -> pl.Expr:
        return (pl.col("in_discount") == 1) & (pl.col("mss_up") == 1) & (
            pl.col("displacement") == 1
        )

    def _short(self) -> pl.Expr:
        return (pl.col("in_premium") == 1) & (pl.col("mss_dn") == 1) & (
            pl.col("displacement") == 1
        )


PHASE17_FAMILIES: dict[str, type[_P17]] = {
    "p17_sweep_reversal": SweepReversalAPlus,
    "p17_expansion_cont": ExpansionContinuation,
    "p17_vol_expansion": VolatilityExpansion,
    "p17_opening_drive_aplus": OpeningDriveAPlus,
    "p17_failed_breakout": FailedBreakoutTrap,
    "p17_premium_discount": PremiumDiscountModel,
}


def generate_phase17_candidates(
    families: tuple[str, ...] = tuple(PHASE17_FAMILIES),
    *,
    timeframe: str,
    risk_grid: tuple[float, ...] = P17_RISK_GRID,
    sizing_modes: tuple[str, ...] = P17_SIZING_MODES,
    hour_window: tuple[int, int] = (9, 12),
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_P17]:
    """family x grid x risk x sizing, TF-seeded holds, NY-AM session. Deterministic."""
    exit_seed = P17_EXIT_BY_TF.get(timeframe, P17_EXIT_BY_TF["5m"])
    h0, h1 = hour_window
    out: list[_P17] = []
    for fam in families:
        cls = PHASE17_FAMILIES[fam]
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
