"""Phase 14 — structure/liquidity frequency expansion + sizing repair.

Focused expansion of Phase 13's strongest lead (15m BOS continuation, validation DSR
0.757). The signal quality was fine; the bottleneck was frequency + sizing — the
validated 15m setups sized to ~0 MNQ contracts in the higher-ATR 2025-2026 regime.

These families keep the SAME edge logic (continue WITH a confirmed up/down structure in
HTF bias, toward the next liquidity) but vary the entry trigger to add trades — structure
retest, FVG mitigation, earlier VWAP pullback, second leg, and a 5m trigger inside the
higher-timeframe bias. Combined with the ``mnq_min1`` / ``mnq_fixed1`` sizing-repair
modes (which never round to 0 contracts within the hard per-trade cap), this is the
focused fix for the Phase-13 confirmation failure — not random discovery.

Built on the PIT-safe structure layer; no new features -> no new lookahead risk.
"""

from __future__ import annotations

import polars as pl

from nqlab.discovery.candidates import _Family

_RTH_OPEN_DEFAULT = 570

P14_EXIT_BY_TF: dict[str, dict[str, object]] = {
    "5m": {"min_hold_bars": 3, "time_stop_bars": 48},
    "15m": {"min_hold_bars": 2, "time_stop_bars": 24},
    "1m": {"min_hold_bars": 6, "time_stop_bars": 90},
}
# Sizing-repair modes first (the Phase-14 fix), plus adaptive for comparison.
P14_SIZING_MODES: tuple[str, ...] = ("mnq_min1", "mnq_fixed1", "adaptive")
P14_RISK_GRID: tuple[float, ...] = (100.0, 150.0, 200.0, 250.0)


def _bias_up() -> pl.Expr:
    return (pl.col("struct_dir") == 1) & (pl.col("trend_4h_bull") == 1)


def _bias_dn() -> pl.Expr:
    return (pl.col("struct_dir") == -1) & (pl.col("trend_4h_bull") == 0)


class _P14(_Family):
    version = "14"
    required_columns: tuple[str, ...] = ("struct_dir", "trend_4h_bull")


class BosRetest(_P14):
    """1 — Structure retest: after a BOS, re-enter on a pullback that retests the broken
    swing and holds (more entries than the single break bar)."""

    family = "bos_retest"
    required_columns = ("struct_dir", "trend_4h_bull", "last_swing_high", "last_swing_low")

    def _long(self) -> pl.Expr:
        return (
            _bias_up()
            & (pl.col("low") <= pl.col("last_swing_high"))
            & (pl.col("close") > pl.col("last_swing_high"))
        )

    def _short(self) -> pl.Expr:
        return (
            _bias_dn()
            & (pl.col("high") >= pl.col("last_swing_low"))
            & (pl.col("close") < pl.col("last_swing_low"))
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": s, "target_atr": 4.0} for s in (1.0, 1.5)]


class BosFvgMitigation(_P14):
    """2 — FVG mitigation: in an up/down structure with HTF bias, enter on mitigation of
    a fair value gap in the trend direction."""

    family = "bos_fvg_mitigation"
    required_columns = ("struct_dir", "trend_4h_bull", "in_bull_fvg", "in_bear_fvg")

    def _long(self) -> pl.Expr:
        return _bias_up() & (pl.col("in_bull_fvg") == 1)

    def _short(self) -> pl.Expr:
        return _bias_dn() & (pl.col("in_bear_fvg") == 1)

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": s, "target_atr": 4.0} for s in (1.0, 1.5)]


class BosEarlyPullback(_P14):
    """3 — Earlier pullback: enter the trend continuation on the first pullback to VWAP
    (before the full structure retest)."""

    family = "bos_early_pullback"
    required_columns = ("struct_dir", "trend_4h_bull", "vwap_session")

    def _long(self) -> pl.Expr:
        return (
            _bias_up()
            & (pl.col("low") <= pl.col("vwap_session"))
            & (pl.col("close") > pl.col("vwap_session"))
        )

    def _short(self) -> pl.Expr:
        return (
            _bias_dn()
            & (pl.col("high") >= pl.col("vwap_session"))
            & (pl.col("close") < pl.col("vwap_session"))
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": s, "target_atr": 4.0} for s in (1.0, 1.5)]


class BosSecondLeg(_P14):
    """4 — Second-leg continuation: confirmed structure plus a fresh higher-low /
    lower-high and displacement (fewer false starts, second-leg entry)."""

    family = "bos_second_leg"
    required_columns = ("struct_dir", "trend_4h_bull", "higher_low", "lower_high", "displacement")

    def _long(self) -> pl.Expr:
        return _bias_up() & (pl.col("higher_low") == 1) & (pl.col("displacement") == 1)

    def _short(self) -> pl.Expr:
        return _bias_dn() & (pl.col("lower_high") == 1) & (pl.col("displacement") == 1)

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": s, "target_atr": 4.0} for s in (1.0, 1.5)]


class Bos5mTrigger(_P14):
    """5 — 5m trigger inside the higher-timeframe bias: BOS with displacement, gated by
    the 4h bias proxy (run this family on the 5m timeframe for frequency)."""

    family = "bos_5m_trigger"
    required_columns = ("struct_dir", "trend_4h_bull", "bos_up", "bos_dn", "displacement")

    def _long(self) -> pl.Expr:
        return (
            (pl.col("trend_4h_bull") == 1) & (pl.col("bos_up") == 1) & (pl.col("displacement") == 1)
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("trend_4h_bull") == 0) & (pl.col("bos_dn") == 1) & (pl.col("displacement") == 1)
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": s, "target_atr": 4.0} for s in (1.0, 1.5)]


PHASE14_FAMILIES: dict[str, type[_P14]] = {
    "bos_retest": BosRetest,
    "bos_fvg_mitigation": BosFvgMitigation,
    "bos_early_pullback": BosEarlyPullback,
    "bos_second_leg": BosSecondLeg,
    "bos_5m_trigger": Bos5mTrigger,
}


def generate_phase14_candidates(
    families: tuple[str, ...] = tuple(PHASE14_FAMILIES),
    *,
    timeframe: str,
    risk_grid: tuple[float, ...] = P14_RISK_GRID,
    sizing_modes: tuple[str, ...] = P14_SIZING_MODES,
    hour_window: tuple[int, int] = (9, 12),
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_P14]:
    """family x grid x risk x sizing, TF-seeded holds, NY-AM session. Deterministic."""
    exit_seed = P14_EXIT_BY_TF.get(timeframe, P14_EXIT_BY_TF["5m"])
    h0, h1 = hour_window
    out: list[_P14] = []
    for fam in families:
        cls = PHASE14_FAMILIES[fam]
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
