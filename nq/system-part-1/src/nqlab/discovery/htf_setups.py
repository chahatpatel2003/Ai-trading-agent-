"""Phase 9 setup families: NQ-only higher-timeframe context + asymmetric liquidity.

Five families, each built on the Phase-9 HTF-context layer
(:mod:`nqlab.features.htf_context`). All are NQ-only and trades-derivable (they do
not require MBP1), so they validate over the full 2024-2026 out-of-sample window.

Every family is *asymmetric by construction*: a tight stop (``stop_atr``) and a wide
target (``target_atr``) so the intended MFE is much larger than the MAE. The
diagnostics-first prefilter (:mod:`nqlab.discovery.prefilter`) then rejects any
candidate whose *realized* trade distribution is not actually asymmetric.

These do NOT reuse the retired Phase 6-8 edges (bias_pullback / pd_sweep_rev /
cvd_pullback): they are new context/liquidity setups.
"""

from __future__ import annotations

import polars as pl

from nqlab.discovery.candidates import _Family

_RTH_OPEN_DEFAULT = 570

# Tight-stop / wide-target seeds + holds per timeframe (asymmetry by construction).
HTF_EXIT_BY_TF: dict[str, dict[str, object]] = {
    "1m": {"min_hold_bars": 6, "time_stop_bars": 90},
    "5m": {"min_hold_bars": 3, "time_stop_bars": 48},
    "15m": {"min_hold_bars": 2, "time_stop_bars": 24},
}
HTF_RISK_GRID: tuple[float, ...] = (25.0, 50.0, 100.0, 200.0)
HTF_SIZING_MODES: tuple[str, ...] = ("mnq_only", "adaptive")

# level key -> (low column, high column)
_LEVELS: dict[str, tuple[str, str]] = {
    "pd": ("prev_day_low", "prev_day_high"),
    "asia": ("asia_low", "asia_high"),
    "london": ("london_low", "london_high"),
    "pw": ("prev_week_low", "prev_week_high"),
    "pm": ("prev_month_low", "prev_month_high"),
    "on": ("overnight_low", "overnight_high"),
    "or": ("or_low", "or_high"),
}


def _bias_up(p: dict[str, object]) -> pl.Expr:
    col = {"1h": "trend_1h_bull", "4h": "trend_4h_bull", "htf": "htf_bull"}[
        str(p.get("bias", "4h"))
    ]
    return pl.col(col) == 1


def _bias_dn(p: dict[str, object]) -> pl.Expr:
    col = {"1h": "trend_1h_bull", "4h": "trend_4h_bull", "htf": "htf_bull"}[
        str(p.get("bias", "4h"))
    ]
    return pl.col(col) == 0


class _Htf(_Family):
    """Common base: version tag + asymmetric default stop/target from params."""

    version = "9"
    required_columns: tuple[str, ...] = ("trend_4h_bull", "overnight_high")


class HtfSweepReclaim(_Htf):
    """A — HTF bias + sweep a liquidity level then reclaim (fade the stop-run)."""

    family = "htf_sweep_reclaim"

    def _lo_hi(self) -> tuple[str, str]:
        return _LEVELS[str(self.p["level"])]

    def _long(self) -> pl.Expr:
        lo, _ = self._lo_hi()
        return _bias_up(self.p) & (pl.col("low") < pl.col(lo)) & (pl.col("close") > pl.col(lo))

    def _short(self) -> pl.Expr:
        _, hi = self._lo_hi()
        return _bias_dn(self.p) & (pl.col("high") > pl.col(hi)) & (pl.col("close") < pl.col(hi))

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [
            {"level": lvl, "bias": "4h", "stop_atr": 1.5, "target_atr": tgt}
            for lvl in ("pd", "london", "pw")
            for tgt in (3.0, 4.0)
        ]


class OpeningDrive(_Htf):
    """B — NY open drives through the overnight/London range with bias + vol confirm."""

    family = "opening_drive"

    def _lo_hi(self) -> tuple[str, str]:
        return _LEVELS[str(self.p["ref"])]

    def _long(self) -> pl.Expr:
        _, hi = self._lo_hi()
        return _bias_up(self.p) & (pl.col("vol_regime") == 1) & (pl.col("close") > pl.col(hi))

    def _short(self) -> pl.Expr:
        lo, _ = self._lo_hi()
        return _bias_dn(self.p) & (pl.col("vol_regime") == 1) & (pl.col("close") < pl.col(lo))

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [
            {"ref": ref, "bias": "4h", "stop_atr": 1.5, "target_atr": tgt}
            for ref in ("on", "london")
            for tgt in (3.0, 4.0)
        ]


class GapSetup(_Htf):
    """C — gap-and-go (continuation) vs gap-fill (fade toward prior close)."""

    family = "gap_setup"
    required_columns = ("gap_dir", "gap_filled", "prev_day_open")

    def _long(self) -> pl.Expr:
        if self.p["mode"] == "go":  # gap up, still open, driving up, bias up
            return (
                (pl.col("gap_dir") == 1)
                & (pl.col("gap_filled") == 0)
                & (pl.col("close") > pl.col("prev_day_open"))
                & _bias_up(self.p)
            )
        # fade: gap DOWN, not yet filled -> expect retrace up toward prior close
        return (pl.col("gap_dir") == -1) & (pl.col("gap_filled") == 0) & _bias_up(self.p)

    def _short(self) -> pl.Expr:
        if self.p["mode"] == "go":
            return (
                (pl.col("gap_dir") == -1)
                & (pl.col("gap_filled") == 0)
                & (pl.col("close") < pl.col("prev_day_open"))
                & _bias_dn(self.p)
            )
        return (pl.col("gap_dir") == 1) & (pl.col("gap_filled") == 0) & _bias_dn(self.p)

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [
            {"mode": mode, "bias": "4h", "stop_atr": 1.5, "target_atr": tgt}
            for mode in ("go", "fade")
            for tgt in (3.0, 4.0)
        ]


class WeekMonthLiquidity(_Htf):
    """D — reaction at a prior-week/month level: sweep+reclaim or breakout+hold."""

    family = "wk_month_liq"

    def _lo_hi(self) -> tuple[str, str]:
        return _LEVELS[str(self.p["level"])]

    def _long(self) -> pl.Expr:
        lo, hi = self._lo_hi()
        if self.p["mode"] == "sweep":
            return (pl.col("low") < pl.col(lo)) & (pl.col("close") > pl.col(lo))
        return _bias_up(self.p) & (pl.col("close") > pl.col(hi))  # breakout + hold

    def _short(self) -> pl.Expr:
        lo, hi = self._lo_hi()
        if self.p["mode"] == "sweep":
            return (pl.col("high") > pl.col(hi)) & (pl.col("close") < pl.col(hi))
        return _bias_dn(self.p) & (pl.col("close") < pl.col(lo))

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [
            {"level": lvl, "mode": mode, "bias": "4h", "stop_atr": 1.5, "target_atr": 4.0}
            for lvl in ("pw", "pm")
            for mode in ("sweep", "breakout")
        ]


class CompressionExpansion(_Htf):
    """E — prior compression (NR day / quiet overnight) then a NY expansion break."""

    family = "compression_expansion"
    required_columns = ("compression_setup", "trend_4h_bull")

    def _lo_hi(self) -> tuple[str, str]:
        return _LEVELS[str(self.p["trigger"])]

    def _long(self) -> pl.Expr:
        _, hi = self._lo_hi()
        return (
            (pl.col("compression_setup") == 1) & _bias_up(self.p) & (pl.col("close") > pl.col(hi))
        )

    def _short(self) -> pl.Expr:
        lo, _ = self._lo_hi()
        return (
            (pl.col("compression_setup") == 1) & _bias_dn(self.p) & (pl.col("close") < pl.col(lo))
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [
            {"trigger": trig, "bias": "4h", "stop_atr": 1.5, "target_atr": tgt}
            for trig in ("or", "on")
            for tgt in (3.0, 4.0)
        ]


PHASE9_FAMILIES: dict[str, type[_Htf]] = {
    "htf_sweep_reclaim": HtfSweepReclaim,
    "opening_drive": OpeningDrive,
    "gap_setup": GapSetup,
    "wk_month_liq": WeekMonthLiquidity,
    "compression_expansion": CompressionExpansion,
}


def generate_htf_setup_candidates(
    families: tuple[str, ...] = tuple(PHASE9_FAMILIES),
    *,
    timeframe: str,
    risk_grid: tuple[float, ...] = HTF_RISK_GRID,
    sizing_modes: tuple[str, ...] = HTF_SIZING_MODES,
    hour_window: tuple[int, int] = (9, 12),
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_Htf]:
    """family x grid x risk x sizing, with TF-seeded holds and a NY-AM session filter.

    Deterministic order (family, grid, risk, sizing) for reproducible fingerprints.
    """
    exit_seed = HTF_EXIT_BY_TF.get(timeframe, HTF_EXIT_BY_TF["5m"])
    h0, h1 = hour_window
    out: list[_Htf] = []
    for fam in families:
        cls = PHASE9_FAMILIES[fam]
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
