"""Phase 18 — high-win-rate, high-frequency NY-AM scalp / intraday-income families.

A change of direction from the Phase 16/17 large-R approach: instead of rare 3-5 ATR winners,
these aim for MANY controlled green trades with small fixed-point (or tight ATR) targets and
tight stops (R:R ~0.8-1.5), so the equity is smooth and $200+ winning days are frequent. All are
NY-AM only (hour mask 9-12 ∩ is_rth -> 09:30-12:00 ET), trades-derivable (clean trades-only
lockbox), and reuse the PIT-safe feature layer.

Daily risk controls (profit target, max loss, stop-after-N-trades/losses) are applied as a
real-account overlay at evaluation time (engine ``ExecConfig``), not baked into the signal — the
search measures the raw scalp edge, the lockbox sweeps the control matrix.
"""

from __future__ import annotations

import polars as pl

from nqlab.discovery.candidates import _Family

_RTH_OPEN_DEFAULT = 570
_atr = pl.col("atr")

P18_EXIT_BY_TF: dict[str, dict[str, object]] = {
    "5m": {"min_hold_bars": 0, "time_stop_bars": 12},
    "1m": {"min_hold_bars": 0, "time_stop_bars": 45},
}
P18_SIZING_MODES: tuple[str, ...] = ("mnq_min1",)
P18_RISK_GRID: tuple[float, ...] = (150.0, 300.0)
# Fixed-point scalp stop/target (NQ points) spanning R:R 0.8 -> 1.5.
P18_FIXED_GRID: tuple[dict[str, object], ...] = (
    {"fix_stop": 8.0, "fix_target": 10.0},  # 1.25R
    {"fix_stop": 8.0, "fix_target": 8.0},   # 1.0R
    {"fix_stop": 10.0, "fix_target": 8.0},  # 0.8R
    {"fix_stop": 6.0, "fix_target": 9.0},   # 1.5R
)


def _bias_up() -> pl.Expr:
    return pl.col("trend_4h_bull") == 1


def _bias_dn() -> pl.Expr:
    return pl.col("trend_4h_bull") == 0


class _P18(_Family):
    version = "18"
    required_columns: tuple[str, ...] = ("trend_4h_bull",)

    def _stop_pts(self) -> pl.Expr:
        if "fix_stop" in self.p:
            return pl.lit(float(self.p["fix_stop"]))  # type: ignore[arg-type]
        return float(self.p["stop_atr"]) * _atr  # type: ignore[arg-type]

    def _target_pts(self) -> pl.Expr:
        if "fix_target" in self.p:
            return pl.lit(float(self.p["fix_target"]))  # type: ignore[arg-type]
        return float(self.p["target_atr"]) * _atr  # type: ignore[arg-type]

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [dict(g) for g in P18_FIXED_GRID]


class VwapReclaimScalp(_P18):
    """A — VWAP reclaim scalp: price pokes through session VWAP and closes back across it
    (exhaustion/reclaim) -> fade back toward VWAP/midline with a small target."""

    family = "p18_vwap_reclaim"
    required_columns = ("vwap_session",)

    def _long(self) -> pl.Expr:
        return (pl.col("low") <= pl.col("vwap_session")) & (
            pl.col("close") > pl.col("vwap_session")
        )

    def _short(self) -> pl.Expr:
        return (pl.col("high") >= pl.col("vwap_session")) & (
            pl.col("close") < pl.col("vwap_session")
        )


class OpeningRangeMeanReversion(_P18):
    """B — Opening-range mean reversion: a failed break of the opening range that reclaims back
    inside -> fade toward the range midpoint."""

    family = "p18_or_meanrev"
    required_columns = ("or_high", "or_low")

    def _long(self) -> pl.Expr:
        return (
            pl.col("or_low").is_not_null()
            & (pl.col("low") < pl.col("or_low"))
            & (pl.col("close") > pl.col("or_low"))
        )

    def _short(self) -> pl.Expr:
        return (
            pl.col("or_high").is_not_null()
            & (pl.col("high") > pl.col("or_high"))
            & (pl.col("close") < pl.col("or_high"))
        )


class PullbackContinuationScalp(_P18):
    """C — Pullback continuation scalp: in the 4h-bias direction, a shallow pullback that tags and
    reclaims session VWAP -> continuation with a small target."""

    family = "p18_pullback_cont"
    required_columns = ("trend_4h_bull", "vwap_session")

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


class SweepMicroReversal(_P18):
    """D — Liquidity sweep micro-reversal: sweep a very recent swing high/low and immediately
    reclaim -> small reversal target (multiple per session)."""

    family = "p18_sweep_micro"
    required_columns = ("swept_high", "swept_low", "reclaim_up", "reclaim_dn")

    def _long(self) -> pl.Expr:
        return (pl.col("swept_low") == 1) & (pl.col("reclaim_up") == 1)

    def _short(self) -> pl.Expr:
        return (pl.col("swept_high") == 1) & (pl.col("reclaim_dn") == 1)


class RangeRotation(_P18):
    """E — Range rotation: on a balanced (non-expansion) NY-AM range, buy the discount end / sell
    the premium end toward the midpoint. Trend-day kill switch = require no range expansion."""

    family = "p18_range_rotation"
    required_columns = ("range_expansion", "in_discount", "in_premium")

    def _long(self) -> pl.Expr:
        return (
            (pl.col("range_expansion") == 0)
            & (pl.col("in_discount") == 1)
            & (pl.col("close") > pl.col("open"))
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("range_expansion") == 0)
            & (pl.col("in_premium") == 1)
            & (pl.col("close") < pl.col("open"))
        )


class TrendDayScalp(_P18):
    """F — Trend-day scalp: on an early trend day (aligned structure + bias + expansion), trade
    pullbacks ONLY in the trend direction (FVG mitigation) with small repeated targets."""

    family = "p18_trendday_scalp"
    required_columns = ("trend_4h_bull", "struct_dir", "range_expansion", "in_bull_fvg",
                        "in_bear_fvg")

    def _long(self) -> pl.Expr:
        return (
            _bias_up()
            & (pl.col("struct_dir") == 1)
            & (pl.col("range_expansion") == 1)
            & (pl.col("in_bull_fvg") == 1)
        )

    def _short(self) -> pl.Expr:
        return (
            _bias_dn()
            & (pl.col("struct_dir") == -1)
            & (pl.col("range_expansion") == 1)
            & (pl.col("in_bear_fvg") == 1)
        )


PHASE18_FAMILIES: dict[str, type[_P18]] = {
    "p18_vwap_reclaim": VwapReclaimScalp,
    "p18_or_meanrev": OpeningRangeMeanReversion,
    "p18_pullback_cont": PullbackContinuationScalp,
    "p18_sweep_micro": SweepMicroReversal,
    "p18_range_rotation": RangeRotation,
    "p18_trendday_scalp": TrendDayScalp,
}


def generate_phase18_candidates(
    families: tuple[str, ...] = tuple(PHASE18_FAMILIES),
    *,
    timeframe: str,
    risk_grid: tuple[float, ...] = P18_RISK_GRID,
    sizing_modes: tuple[str, ...] = P18_SIZING_MODES,
    hour_window: tuple[int, int] = (9, 12),
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_P18]:
    """family x fixed-point grid x risk x sizing, TF-seeded holds, NY-AM session. Deterministic."""
    exit_seed = P18_EXIT_BY_TF.get(timeframe, P18_EXIT_BY_TF["5m"])
    h0, h1 = hour_window
    out: list[_P18] = []
    for fam in families:
        cls = PHASE18_FAMILIES[fam]
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
