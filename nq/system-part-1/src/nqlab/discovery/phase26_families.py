"""Phase 26 — non-Topstep high-profit-factor families (full RTH, wider R, all sessions).

Unlike the Topstep phases, these are NOT NY-AM-restricted and use wider asymmetric targets
(2R/3R) with longer holds, aiming for PF >= 2.0 / high return-to-drawdown. Five explainable
high-PF hypotheses across sessions (full RTH and NY-PM), 5m/15m, reusing the PIT-safe feature
layer. None assumed to work — judged against the PF >= 2.0 bar.
"""

from __future__ import annotations

import polars as pl

from nqlab.discovery.candidates import _Family
from nqlab.features.indicators import add_bollinger, add_donchian, ema

_RTH_OPEN_DEFAULT = 570
P26_EXIT_BY_TF: dict[str, dict[str, object]] = {
    "5m": {"min_hold_bars": 1, "time_stop_bars": 60},   # hold for continuation
    "15m": {"min_hold_bars": 1, "time_stop_bars": 24},
}
P26_SIZING_MODES: tuple[str, ...] = ("mnq_fixed1",)   # 1 MNQ -> clean per-contract PF
P26_RISK_GRID: tuple[float, ...] = (150.0,)           # risk irrelevant to PF at fixed 1 MNQ
# Sessions: full RTH (09:30-16:00) and NY-PM (12:00-16:00).
P26_SESSIONS: tuple[tuple[int, int], ...] = ((9, 16), (12, 16))
P26_STOP = 1.5  # ATR
P26_R: tuple[float, ...] = (2.0, 3.0)  # wide asymmetric targets


class _P26(_Family):
    version = "26"
    required_columns: tuple[str, ...] = ("atr",)

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"stop_atr": P26_STOP, "target_atr": P26_STOP * r} for r in P26_R]


class TrendFollow(_P26):
    """1 — High-PF trend-following: EMA stack up (21>50) + pullback to EMA21 reclaim; ride a wide
    continuation target."""

    family = "p26_trend"

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(ema(21).alias("_e21"), ema(50).alias("_e50"))

    def _long(self) -> pl.Expr:
        return (
            (pl.col("_e21") > pl.col("_e50"))
            & (pl.col("low") <= pl.col("_e21"))
            & (pl.col("close") > pl.col("_e21"))
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("_e21") < pl.col("_e50"))
            & (pl.col("high") >= pl.col("_e21"))
            & (pl.col("close") < pl.col("_e21"))
        )


class BreakoutContinuation(_P26):
    """2 — Breakout continuation: close beyond the prior 20-bar Donchian channel (pullback handled
    by next-open fill); ride the expansion."""

    family = "p26_breakout"

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_donchian(frame, n=20)

    def _long(self) -> pl.Expr:
        return pl.col("_dc_up").is_not_null() & (pl.col("close") > pl.col("_dc_up"))

    def _short(self) -> pl.Expr:
        return pl.col("_dc_dn").is_not_null() & (pl.col("close") < pl.col("_dc_dn"))


class MeanReversion(_P26):
    """3 — Mean reversion: extreme Bollinger deviation with an exhaustion reclaim back inside the
    band; target the mean (wide R only via the fixed target_atr)."""

    family = "p26_meanrev"

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_bollinger(frame, n=20, k=2.5)

    def _long(self) -> pl.Expr:
        return (pl.col("low") <= pl.col("_bb_dn")) & (pl.col("close") > pl.col("_bb_dn"))

    def _short(self) -> pl.Expr:
        return (pl.col("high") >= pl.col("_bb_up")) & (pl.col("close") < pl.col("_bb_up"))


class VolExpansion(_P26):
    """4 — Volatility-regime expansion: compression setup releasing into a range-expansion
    displacement leg in the structural direction."""

    family = "p26_vol_expansion"
    required_columns = ("atr", "compression_setup", "range_expansion", "displacement", "struct_dir")

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


class HtfPoiSwing(_P26):
    """5 — HTF POI swing: structure-aligned BOS + displacement in the 4h bias, held for a wide
    target (the non-Topstep version of the bos trigger, no NY-AM cap)."""

    family = "p26_htf_poi"
    required_columns = ("atr", "trend_4h_bull", "bos_up", "bos_dn", "displacement")

    def _long(self) -> pl.Expr:
        return (
            (pl.col("trend_4h_bull") == 1)
            & (pl.col("bos_up") == 1)
            & (pl.col("displacement") == 1)
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("trend_4h_bull") == 0)
            & (pl.col("bos_dn") == 1)
            & (pl.col("displacement") == 1)
        )


PHASE26_FAMILIES: dict[str, type[_P26]] = {
    "p26_trend": TrendFollow,
    "p26_breakout": BreakoutContinuation,
    "p26_meanrev": MeanReversion,
    "p26_vol_expansion": VolExpansion,
    "p26_htf_poi": HtfPoiSwing,
}


def generate_phase26_candidates(
    families: tuple[str, ...] = tuple(PHASE26_FAMILIES),
    *,
    timeframe: str,
    risk_grid: tuple[float, ...] = P26_RISK_GRID,
    sizing_modes: tuple[str, ...] = P26_SIZING_MODES,
    sessions: tuple[tuple[int, int], ...] = P26_SESSIONS,
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_P26]:
    """family x grid x session x risk x sizing, full-RTH/NY-PM, wide R. Deterministic."""
    exit_seed = P26_EXIT_BY_TF.get(timeframe, P26_EXIT_BY_TF["5m"])
    out: list[_P26] = []
    for fam in families:
        cls = PHASE26_FAMILIES[fam]
        for base in cls.grid():
            for h0, h1 in sessions:
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
