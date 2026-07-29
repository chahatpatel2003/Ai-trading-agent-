"""Phase 12 — trend_day frequency expansion.

Phase 11's `trend_day` was the cleanest validation lead in the project, but it fired
too rarely on the held-out lockbox to confirm. These five families keep the SAME edge
logic (only trade WITH a confirmed trend day: HTF-4h bull, expanding vol, on the right
side of session VWAP) but change *entry timing* to produce more trades — entering on
pullbacks / retests / second legs within the trend rather than once on the break.

This is a focused expansion of one lead, not random discovery and not curve-fitting:
the trend-day gate is identical; only the trigger differs. NQ-only, trades-derivable,
NY-AM, asymmetric (tight stop / wider target). The diagnostics-first prefilter still
rejects anything whose realized MAE >= MFE. No new features -> no new lookahead risk.
"""

from __future__ import annotations

import polars as pl

from nqlab.discovery.candidates import _Family

_RTH_OPEN_DEFAULT = 570

P12_EXIT_BY_TF: dict[str, dict[str, object]] = {
    "5m": {"min_hold_bars": 3, "time_stop_bars": 48},
    "15m": {"min_hold_bars": 2, "time_stop_bars": 24},
    "1m": {"min_hold_bars": 6, "time_stop_bars": 90},
}
P12_RISK_GRID: tuple[float, ...] = (50.0, 100.0, 150.0, 200.0)
P12_SIZING_MODES: tuple[str, ...] = ("mnq_only", "adaptive")
_FADE_TYPES = ("range", "compression")  # prior-day types to AVOID (do not fade a trend day)


def _trend_up() -> pl.Expr:
    return (pl.col("trend_4h_bull") == 1) & (pl.col("vol_regime") == 1)


def _trend_dn() -> pl.Expr:
    return (pl.col("trend_4h_bull") == 0) & (pl.col("vol_regime") == 1)


class _P12(_Family):
    version = "12"
    required_columns: tuple[str, ...] = ("trend_4h_bull", "vol_regime", "vwap_session")


class TrendPullback(_P12):
    """1 — Trend-day continuation after a pullback to VWAP (re-enter each dip-and-go)."""

    family = "trend_pullback"

    def _long(self) -> pl.Expr:  # uptrend, this bar dipped to VWAP and closed back above
        return (
            _trend_up()
            & (pl.col("low") <= pl.col("vwap_session"))
            & (pl.col("close") > pl.col("vwap_session"))
        )

    def _short(self) -> pl.Expr:
        return (
            _trend_dn()
            & (pl.col("high") >= pl.col("vwap_session"))
            & (pl.col("close") < pl.col("vwap_session"))
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": 1.5, "target_atr": tgt} for tgt in (3.0, 4.0)]


class TrendOpenDrive(_P12):
    """2 — Opening-drive continuation: held above the OR, re-enter on a retest that holds."""

    family = "trend_open_drive"
    required_columns = ("trend_4h_bull", "vol_regime", "or_high")

    def _long(self) -> pl.Expr:
        return (
            _trend_up()
            & (pl.col("close") > pl.col("or_high"))
            & (pl.col("low") <= pl.col("or_high"))
        )

    def _short(self) -> pl.Expr:
        return (
            _trend_dn()
            & (pl.col("close") < pl.col("or_low"))
            & (pl.col("high") >= pl.col("or_low"))
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": 1.5, "target_atr": tgt} for tgt in (3.0, 4.0)]


class TrendRangeExpansion(_P12):
    """3 — Compression -> NY expansion, continuation only if trend-day filters agree."""

    family = "trend_range_expansion"
    required_columns = ("compression_setup", "trend_4h_bull", "vol_regime", "or_high")

    def _long(self) -> pl.Expr:
        return (
            (pl.col("compression_setup") == 1) & _trend_up() & (pl.col("close") > pl.col("or_high"))
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("compression_setup") == 1) & _trend_dn() & (pl.col("close") < pl.col("or_low"))
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": 1.5, "target_atr": tgt} for tgt in (3.0, 4.0)]


class TrendLateConfirm(_P12):
    """4 — Late confirmation: trend confirmed (broke the prior-day extreme), enter the
    second leg on a pullback to VWAP (fewer false starts)."""

    family = "trend_late_confirm"
    required_columns = (
        "trend_4h_bull",
        "vol_regime",
        "vwap_session",
        "prev_day_high",
        "prev_day_low",
    )

    def _long(self) -> pl.Expr:
        return (
            _trend_up()
            & (pl.col("close") > pl.col("prev_day_high"))
            & (pl.col("low") <= pl.col("vwap_session"))
            & (pl.col("close") > pl.col("vwap_session"))
        )

    def _short(self) -> pl.Expr:
        return (
            _trend_dn()
            & (pl.col("close") < pl.col("prev_day_low"))
            & (pl.col("high") >= pl.col("vwap_session"))
            & (pl.col("close") < pl.col("vwap_session"))
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": 1.5, "target_atr": tgt} for tgt in (3.0, 4.0)]


class TrendNoFade(_P12):
    """5 — 'Do not fade a trend day': continuation pullback entries, explicitly excluding
    range/compression prior days (the contexts a fader would have taken)."""

    family = "trend_no_fade"
    required_columns = ("trend_4h_bull", "vol_regime", "vwap_session", "prev_day_type")

    def _not_fade(self) -> pl.Expr:
        return ~pl.col("prev_day_type").is_in(_FADE_TYPES)

    def _long(self) -> pl.Expr:
        return (
            _trend_up()
            & self._not_fade()
            & (pl.col("low") <= pl.col("vwap_session"))
            & (pl.col("close") > pl.col("vwap_session"))
        )

    def _short(self) -> pl.Expr:
        return (
            _trend_dn()
            & self._not_fade()
            & (pl.col("high") >= pl.col("vwap_session"))
            & (pl.col("close") < pl.col("vwap_session"))
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": 1.5, "target_atr": tgt} for tgt in (3.0, 4.0)]


PHASE12_FAMILIES: dict[str, type[_P12]] = {
    "trend_pullback": TrendPullback,
    "trend_open_drive": TrendOpenDrive,
    "trend_range_expansion": TrendRangeExpansion,
    "trend_late_confirm": TrendLateConfirm,
    "trend_no_fade": TrendNoFade,
}


def generate_phase12_candidates(
    families: tuple[str, ...] = tuple(PHASE12_FAMILIES),
    *,
    timeframe: str,
    risk_grid: tuple[float, ...] = P12_RISK_GRID,
    sizing_modes: tuple[str, ...] = P12_SIZING_MODES,
    hour_window: tuple[int, int] = (9, 12),
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_P12]:
    """family x grid x risk x sizing, TF-seeded holds, NY-AM session. Deterministic."""
    exit_seed = P12_EXIT_BY_TF.get(timeframe, P12_EXIT_BY_TF["5m"])
    h0, h1 = hour_window
    out: list[_P12] = []
    for fam in families:
        cls = PHASE12_FAMILIES[fam]
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
