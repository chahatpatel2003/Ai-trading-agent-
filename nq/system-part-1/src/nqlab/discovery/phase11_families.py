"""Phase 11 setup families: NEW NQ-only families for a confirmed-enough search.

These are deliberately distinct from the retired Phase 6-8 edges AND from the Phase 9
breakout/continuation families. Phase 9 was momentum-only; Phase 11 adds the missing
classes — explicit trend-day detection, and two MEAN-REVERSION families (range-day
fade and VWAP reversion) — plus keeps everything NY-AM, trades-only (so the 2025-2026
lockbox stays clean), and asymmetric by construction (tight stop / wider target). The
diagnostics-first prefilter still rejects anything whose realized MAE >= MFE.

All families read only the existing PIT-safe feature columns (base + context +
htf_context); they add no new features, so nothing here can introduce lookahead.
"""

from __future__ import annotations

import polars as pl

from nqlab.discovery.candidates import _Family

_RTH_OPEN_DEFAULT = 570

# Tight-stop / wider-target holds per timeframe (asymmetry by construction).
P11_EXIT_BY_TF: dict[str, dict[str, object]] = {
    "5m": {"min_hold_bars": 3, "time_stop_bars": 48},
    "15m": {"min_hold_bars": 2, "time_stop_bars": 24},
    "1m": {"min_hold_bars": 6, "time_stop_bars": 90},
}
# Risk grid emphasises levels that actually size to a position in the 2025-2026
# regime (>=$100); $50 is kept as the floor for lower-ATR years.
P11_RISK_GRID: tuple[float, ...] = (50.0, 100.0, 150.0, 200.0)
P11_SIZING_MODES: tuple[str, ...] = ("mnq_only", "adaptive")

_RANGE_TYPES = ("range", "compression")  # prior-day types that suggest a range day


class _P11(_Family):
    version = "11"


class TrendDay(_P11):
    """Trend-day momentum continuation: HTF-bull, expanding vol, above VWAP, breaking
    the opening range — ride it with a wide target (focus: trend-day detection)."""

    family = "trend_day"
    required_columns = ("or_high", "vwap_session", "trend_4h_bull", "vol_regime")

    def _long(self) -> pl.Expr:
        return (
            (pl.col("trend_4h_bull") == 1)
            & (pl.col("vol_regime") == 1)
            & (pl.col("close") > pl.col("or_high"))
            & (pl.col("close") > pl.col("vwap_session"))
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("trend_4h_bull") == 0)
            & (pl.col("vol_regime") == 1)
            & (pl.col("close") < pl.col("or_low"))
            & (pl.col("close") < pl.col("vwap_session"))
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": 1.5, "target_atr": tgt} for tgt in (3.0, 4.0)]


class RangeFade(_P11):
    """Range-day mean reversion: on a low-vol day after a range/compression prior day,
    fade a poke through the prior-day extreme that reclaims (focus: range-day fade)."""

    family = "range_fade"
    required_columns = ("prev_day_type", "prev_day_low", "prev_day_high", "vol_regime")

    def _context(self) -> pl.Expr:
        return (pl.col("vol_regime") == 0) & pl.col("prev_day_type").is_in(_RANGE_TYPES)

    def _long(self) -> pl.Expr:  # poke below PDL, reclaim -> fade up
        return (
            self._context()
            & (pl.col("low") < pl.col("prev_day_low"))
            & (pl.col("close") > pl.col("prev_day_low"))
        )

    def _short(self) -> pl.Expr:  # poke above PDH, reclaim -> fade down
        return (
            self._context()
            & (pl.col("high") > pl.col("prev_day_high"))
            & (pl.col("close") < pl.col("prev_day_high"))
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": stop, "target_atr": tgt} for stop in (1.0, 1.5) for tgt in (2.0, 3.0)]


class VwapRevert(_P11):
    """VWAP mean reversion: in a low-vol regime, fade an over-extension away from the
    session VWAP back toward it (focus: low-frequency asymmetric reversion)."""

    family = "vwap_revert"
    required_columns = ("dist_vwap", "vol_regime", "atr")

    def _long(self) -> pl.Expr:
        band = float(self.p["entry_atr"]) * pl.col("atr")  # type: ignore[arg-type]
        return (pl.col("vol_regime") == 0) & (pl.col("dist_vwap") < -band)

    def _short(self) -> pl.Expr:
        band = float(self.p["entry_atr"]) * pl.col("atr")  # type: ignore[arg-type]
        return (pl.col("vol_regime") == 0) & (pl.col("dist_vwap") > band)

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [
            {"entry_atr": entry, "stop_atr": 1.5, "target_atr": tgt}
            for entry in (1.5, 2.0)
            for tgt in (2.0, 3.0)
        ]


PHASE11_FAMILIES: dict[str, type[_P11]] = {
    "trend_day": TrendDay,
    "range_fade": RangeFade,
    "vwap_revert": VwapRevert,
}


def generate_phase11_candidates(
    families: tuple[str, ...] = tuple(PHASE11_FAMILIES),
    *,
    timeframe: str,
    risk_grid: tuple[float, ...] = P11_RISK_GRID,
    sizing_modes: tuple[str, ...] = P11_SIZING_MODES,
    hour_window: tuple[int, int] = (9, 12),
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_P11]:
    """family x grid x risk x sizing, TF-seeded holds, NY-AM session. Deterministic."""
    exit_seed = P11_EXIT_BY_TF.get(timeframe, P11_EXIT_BY_TF["5m"])
    h0, h1 = hour_window
    out: list[_P11] = []
    for fam in families:
        cls = PHASE11_FAMILIES[fam]
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
