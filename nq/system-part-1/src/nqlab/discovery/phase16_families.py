"""Phase 16 — payout-first, scalable NQ-only strategy families.

Built to FIND larger winners (asymmetric R), not to rescue thin edges. Every family is
NY-AM only (hour mask 9-12 ∩ is_rth -> 09:30-12:00 ET, inside the 12:30 cutoff), runs on the
confirmable 5m timeframe, and reuses the PIT-safe structure / HTF / context feature layer (no
new lookahead). Targets are wide (3-5 ATR) against tight stops (1.0-1.5 ATR) so accepted
candidates can produce $200+ winning days and scale past 1 MNQ.

None of these are assumed to work. They are candidates to be judged against the payout-first
business bar and rejected honestly if they fall short.
"""

from __future__ import annotations

from itertools import product

import polars as pl

from nqlab.discovery.candidates import _Family

_RTH_OPEN_DEFAULT = 570

# Exit seeds per timeframe (let winners run; time-stop bounds the hold).
P16_EXIT_BY_TF: dict[str, dict[str, object]] = {
    "5m": {"min_hold_bars": 3, "time_stop_bars": 48},
    "15m": {"min_hold_bars": 2, "time_stop_bars": 24},
}
# Sizing-repair modes that never round to 0 contracts AND scale with risk (mnq_min1),
# plus adaptive for the higher-risk / multi-contract scaling test.
P16_SIZING_MODES: tuple[str, ...] = ("mnq_min1", "adaptive")
# Risk grid spans evaluation ($150) through funded/scale sizing ($400) so scalability is tested.
P16_RISK_GRID: tuple[float, ...] = (150.0, 250.0, 400.0)
# Asymmetric R: tight stop, wide target -> larger winners.
P16_STOP_ATR: tuple[float, ...] = (1.0, 1.5)
P16_TARGET_ATR: tuple[float, ...] = (3.0, 5.0)


def _bias_up() -> pl.Expr:
    return pl.col("trend_4h_bull") == 1


def _bias_dn() -> pl.Expr:
    return pl.col("trend_4h_bull") == 0


class _P16(_Family):
    version = "16"
    required_columns: tuple[str, ...] = ("trend_4h_bull",)

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": s, "target_atr": t} for s, t in product(P16_STOP_ATR, P16_TARGET_ATR)]


class OpeningDriveContinuation(_P16):
    """1 — Opening-drive continuation: break of the opening range in the 4h-bias direction with
    a displacement candle (ride the NY-AM drive)."""

    family = "p16_opening_drive"
    required_columns = ("trend_4h_bull", "or_high", "or_low", "displacement")

    def _long(self) -> pl.Expr:
        return (
            _bias_up()
            & pl.col("or_high").is_not_null()
            & (pl.col("close") > pl.col("or_high"))
            & (pl.col("displacement") == 1)
        )

    def _short(self) -> pl.Expr:
        return (
            _bias_dn()
            & pl.col("or_low").is_not_null()
            & (pl.col("close") < pl.col("or_low"))
            & (pl.col("displacement") == 1)
        )


class SweepDisplacementReversal(_P16):
    """2 — Liquidity sweep + displacement reversal: sweep a swing liquidity pool, then a market
    structure shift with displacement back the other way (asymmetric reversal)."""

    family = "p16_sweep_reversal"
    required_columns = ("swept_high", "swept_low", "mss_up", "mss_dn", "displacement")

    def _long(self) -> pl.Expr:
        return (pl.col("swept_low") == 1) & (pl.col("mss_up") == 1) & (pl.col("displacement") == 1)

    def _short(self) -> pl.Expr:
        return (pl.col("swept_high") == 1) & (pl.col("mss_dn") == 1) & (pl.col("displacement") == 1)


class CompressionExpansion(_P16):
    """3 — Compression into NY expansion: an overnight/prior compression setup that releases into
    a range-expansion candle in the bias direction."""

    family = "p16_compression"
    required_columns = ("trend_4h_bull", "compression_setup", "range_expansion")

    def _long(self) -> pl.Expr:
        return (
            _bias_up()
            & (pl.col("compression_setup") == 1)
            & (pl.col("range_expansion") == 1)
            & (pl.col("close") > pl.col("open"))
        )

    def _short(self) -> pl.Expr:
        return (
            _bias_dn()
            & (pl.col("compression_setup") == 1)
            & (pl.col("range_expansion") == 1)
            & (pl.col("close") < pl.col("open"))
        )


class PriorLevelReaction(_P16):
    """4 — Prior-day level reaction: a confirmed break-and-displace through the prior day's high/
    low (continuation toward the next liquidity), gated by the 4h bias."""

    family = "p16_prior_level"
    required_columns = ("trend_4h_bull", "prev_day_high", "prev_day_low", "displacement")

    def _long(self) -> pl.Expr:
        return (
            _bias_up()
            & pl.col("prev_day_high").is_not_null()
            & (pl.col("close") > pl.col("prev_day_high"))
            & (pl.col("displacement") == 1)
        )

    def _short(self) -> pl.Expr:
        return (
            _bias_dn()
            & pl.col("prev_day_low").is_not_null()
            & (pl.col("close") < pl.col("prev_day_low"))
            & (pl.col("displacement") == 1)
        )


class MssFvgContinuation(_P16):
    """5 — MSS + FVG mitigation continuation: market-structure shift in the bias direction, entered
    on mitigation of a fresh fair-value gap (asymmetric continuation)."""

    family = "p16_mss_fvg"
    required_columns = ("trend_4h_bull", "mss_up", "mss_dn", "in_bull_fvg", "in_bear_fvg")

    def _long(self) -> pl.Expr:
        return _bias_up() & (pl.col("mss_up") == 1) & (pl.col("in_bull_fvg") == 1)

    def _short(self) -> pl.Expr:
        return _bias_dn() & (pl.col("mss_dn") == 1) & (pl.col("in_bear_fvg") == 1)


class TrendExpansionDay(_P16):
    """6 — Trend-expansion day: aligned structure + 4h bias + a range-expansion displacement leg —
    designed to capture the big directional NY-AM days."""

    family = "p16_trend_expansion"
    required_columns = ("trend_4h_bull", "struct_dir", "range_expansion", "displacement")

    def _long(self) -> pl.Expr:
        return (
            _bias_up()
            & (pl.col("struct_dir") == 1)
            & (pl.col("range_expansion") == 1)
            & (pl.col("displacement") == 1)
        )

    def _short(self) -> pl.Expr:
        return (
            _bias_dn()
            & (pl.col("struct_dir") == -1)
            & (pl.col("range_expansion") == 1)
            & (pl.col("displacement") == 1)
        )


PHASE16_FAMILIES: dict[str, type[_P16]] = {
    "p16_opening_drive": OpeningDriveContinuation,
    "p16_sweep_reversal": SweepDisplacementReversal,
    "p16_compression": CompressionExpansion,
    "p16_prior_level": PriorLevelReaction,
    "p16_mss_fvg": MssFvgContinuation,
    "p16_trend_expansion": TrendExpansionDay,
}


def generate_phase16_candidates(
    families: tuple[str, ...] = tuple(PHASE16_FAMILIES),
    *,
    timeframe: str,
    risk_grid: tuple[float, ...] = P16_RISK_GRID,
    sizing_modes: tuple[str, ...] = P16_SIZING_MODES,
    hour_window: tuple[int, int] = (9, 12),
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_P16]:
    """family x grid x risk x sizing, TF-seeded holds, NY-AM session. Deterministic."""
    exit_seed = P16_EXIT_BY_TF.get(timeframe, P16_EXIT_BY_TF["5m"])
    h0, h1 = hour_window
    out: list[_P16] = []
    for fam in families:
        cls = PHASE16_FAMILIES[fam]
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
