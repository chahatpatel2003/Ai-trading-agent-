"""Phase 25 — daily-bias + HTF-trend gated HTF-POI -> LTF-reclaim model (complete sequence).

MANDATORY gate: a trade is allowed ONLY when the daily bias and HTF trend agree (long only in
bullish alignment, short only in bearish). On top of that, price must have traded into an HTF POI
(support/resistance, from :mod:`nqlab.features.htf_poi`) and the LTF must confirm
(reclaim / CISD / MSS-CHoCH / sweep+reclaim / displacement+FVG / conservative combo), plus two
secondary models (momentum breakout-pullback, pivot/S-R reclaim) under the same mandatory gate.

NY-AM only, 1 trade/day, fixed 1:1 or 1.5R, two daily-bias definitions tested separately. None
assumed to work — judged against a 55% win floor. Daily-bias/HTF-trend alignment is NOT optional.
"""

from __future__ import annotations

from itertools import product

import polars as pl

from nqlab.discovery.candidates import _Family
from nqlab.features.daily_bias import add_daily_bias
from nqlab.features.htf_poi import add_htf_poi

_RTH_OPEN_DEFAULT = 570
P25_EXIT_BY_TF: dict[str, dict[str, object]] = {
    "5m": {"min_hold_bars": 0, "time_stop_bars": 24},
    "15m": {"min_hold_bars": 0, "time_stop_bars": 10},
}
P25_SIZING_MODES: tuple[str, ...] = ("mnq_min1",)
P25_RISK_GRID: tuple[float, ...] = (200.0,)
_POI_TOL = 5.0
P25_BIAS_DEFS: tuple[str, ...] = ("a", "b")    # daily-structure / gap+prior-day
P25_R: tuple[float, ...] = (1.0, 1.5)          # 1:1 and 1.5R
_STOP = 10.0
_BASE_INPUTS = ("trend_1h_bull", "trend_4h_bull", "displacement", "in_bull_fvg", "in_bear_fvg",
                "fvg_bull", "fvg_bear", "prev_day_low", "prev_day_high", "overnight_low",
                "overnight_high")


class _P25(_Family):
    version = "25"
    required_columns: tuple[str, ...] = _BASE_INPUTS

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_daily_bias(add_htf_poi(frame, tol_pts=_POI_TOL))

    def _stop_pts(self) -> pl.Expr:
        return pl.lit(float(self.p["fix_stop"]))  # type: ignore[arg-type]

    def _target_pts(self) -> pl.Expr:
        return pl.lit(float(self.p["fix_target"]))  # type: ignore[arg-type]

    # MANDATORY daily-bias + HTF-trend alignment gate (per bias definition).
    def _aligned_long(self) -> pl.Expr:
        return pl.col(f"aligned_long_{self.p['bias_def']}") == 1

    def _aligned_short(self) -> pl.Expr:
        return pl.col(f"aligned_short_{self.p['bias_def']}") == 1

    # Price must have reached the relevant HTF POI within the last few bars.
    def _at_long_poi(self) -> pl.Expr:
        return pl.col("at_support_poi_recent") == 1

    def _at_short_poi(self) -> pl.Expr:
        return pl.col("at_resistance_poi_recent") == 1

    def signals(self, frame: pl.DataFrame) -> pl.DataFrame:
        sig = super().signals(frame)
        nz = pl.col("signal") != 0
        first_of_day = nz & (nz.cast(pl.Int32).cum_sum().over("trading_day") == 1)
        return sig.with_columns(
            pl.when(first_of_day).then(pl.col("signal")).otherwise(0).cast(pl.Int8).alias("signal")
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [
            {"bias_def": b, "fix_stop": _STOP, "fix_target": _STOP * r}
            for b, r in product(P25_BIAS_DEFS, P25_R)
        ]


# ---- Core entry variants (POI touch + LTF confirmation, under the mandatory gate) -------------
class IfvgReclaim(_P25):
    """A — IFVG reclaim entry."""

    family = "p25_ifvg"
    required_columns = (*_BASE_INPUTS, "reclaim_up", "reclaim_dn")

    def _long(self) -> pl.Expr:
        return self._aligned_long() & self._at_long_poi() & (pl.col("ifvg_bull") == 1)

    def _short(self) -> pl.Expr:
        return self._aligned_short() & self._at_short_poi() & (pl.col("ifvg_bear") == 1)


class CisdConfirm(_P25):
    """B — CISD (change in state of delivery) confirmation entry."""

    family = "p25_cisd"

    def _long(self) -> pl.Expr:
        return self._aligned_long() & self._at_long_poi() & (pl.col("cisd_up") == 1)

    def _short(self) -> pl.Expr:
        return self._aligned_short() & self._at_short_poi() & (pl.col("cisd_dn") == 1)


class MssChochConfirm(_P25):
    """C — MSS / CHoCH confirmation entry."""

    family = "p25_mss"
    required_columns = (*_BASE_INPUTS, "mss_up", "mss_dn", "choch_up", "choch_dn")

    def _long(self) -> pl.Expr:
        return self._aligned_long() & self._at_long_poi() & (
            (pl.col("mss_up") == 1) | (pl.col("choch_up") == 1)
        )

    def _short(self) -> pl.Expr:
        return self._aligned_short() & self._at_short_poi() & (
            (pl.col("mss_dn") == 1) | (pl.col("choch_dn") == 1)
        )


class SweepReclaim(_P25):
    """D — Sweep + reclaim entry."""

    family = "p25_sweep"
    required_columns = (*_BASE_INPUTS, "swept_high", "swept_low", "reclaim_up", "reclaim_dn")

    def _long(self) -> pl.Expr:
        return self._aligned_long() & self._at_long_poi() & (pl.col("swept_low") == 1) & (
            pl.col("reclaim_up") == 1
        )

    def _short(self) -> pl.Expr:
        return self._aligned_short() & self._at_short_poi() & (pl.col("swept_high") == 1) & (
            pl.col("reclaim_dn") == 1
        )


class DisplacementFvg(_P25):
    """E — Displacement + LTF FVG retrace entry."""

    family = "p25_disp_fvg"

    def _long(self) -> pl.Expr:
        return self._aligned_long() & self._at_long_poi() & (pl.col("displacement") == 1) & (
            pl.col("in_bull_fvg") == 1
        )

    def _short(self) -> pl.Expr:
        return self._aligned_short() & self._at_short_poi() & (pl.col("displacement") == 1) & (
            pl.col("in_bear_fvg") == 1
        )


class ConservativeCombo(_P25):
    """F — Conservative combo: POI touch + sweep/reclaim + (MSS or CISD), all aligned."""

    family = "p25_combo"
    required_columns = (*_BASE_INPUTS, "swept_high", "swept_low", "reclaim_up", "reclaim_dn",
                        "mss_up", "mss_dn", "choch_up", "choch_dn")

    def _long(self) -> pl.Expr:
        return (
            self._aligned_long()
            & self._at_long_poi()
            & (pl.col("swept_low") == 1)
            & (pl.col("reclaim_up") == 1)
            & ((pl.col("mss_up") == 1) | (pl.col("choch_up") == 1) | (pl.col("cisd_up") == 1))
        )

    def _short(self) -> pl.Expr:
        return (
            self._aligned_short()
            & self._at_short_poi()
            & (pl.col("swept_high") == 1)
            & (pl.col("reclaim_dn") == 1)
            & ((pl.col("mss_dn") == 1) | (pl.col("choch_dn") == 1) | (pl.col("cisd_dn") == 1))
        )


# ---- Secondary models (same mandatory gate) ---------------------------------------------------
class MomentumPullback(_P25):
    """G — Momentum breakout + pullback: aligned trend, RSI momentum, shallow VWAP pullback reclaim
    (continuation, not breakout chase)."""

    family = "p25_momentum"
    required_columns = (*_BASE_INPUTS, "rsi", "vwap_session")

    def _long(self) -> pl.Expr:
        return (
            self._aligned_long()
            & (pl.col("rsi") > 55.0)
            & (pl.col("low") <= pl.col("vwap_session"))
            & (pl.col("close") > pl.col("vwap_session"))
        )

    def _short(self) -> pl.Expr:
        return (
            self._aligned_short()
            & (pl.col("rsi") < 45.0)
            & (pl.col("high") >= pl.col("vwap_session"))
            & (pl.col("close") < pl.col("vwap_session"))
        )


class PivotSupportResistance(_P25):
    """H — Pivot / support-resistance reclaim: aligned trend + reclaim of a key level (the support/
    resistance POI), LTF confirmed by the reclaim close."""

    family = "p25_pivot_sr"
    required_columns = (*_BASE_INPUTS, "reclaim_up", "reclaim_dn")

    def _long(self) -> pl.Expr:
        return self._aligned_long() & (pl.col("at_support_poi_recent") == 1) & (
            (pl.col("reclaim_up") == 1) | (pl.col("close") > pl.col("open"))
        )

    def _short(self) -> pl.Expr:
        return self._aligned_short() & (pl.col("at_resistance_poi_recent") == 1) & (
            (pl.col("reclaim_dn") == 1) | (pl.col("close") < pl.col("open"))
        )


PHASE25_FAMILIES: dict[str, type[_P25]] = {
    "p25_ifvg": IfvgReclaim,
    "p25_cisd": CisdConfirm,
    "p25_mss": MssChochConfirm,
    "p25_sweep": SweepReclaim,
    "p25_disp_fvg": DisplacementFvg,
    "p25_combo": ConservativeCombo,
    "p25_momentum": MomentumPullback,
    "p25_pivot_sr": PivotSupportResistance,
}


def generate_phase25_candidates(
    families: tuple[str, ...] = tuple(PHASE25_FAMILIES),
    *,
    timeframe: str,
    risk_grid: tuple[float, ...] = P25_RISK_GRID,
    sizing_modes: tuple[str, ...] = P25_SIZING_MODES,
    hour_window: tuple[int, int] = (9, 12),
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_P25]:
    """family x (bias-def x R-multiple) x risk x sizing, NY-AM, 1 trade/day. Deterministic."""
    exit_seed = P25_EXIT_BY_TF.get(timeframe, P25_EXIT_BY_TF["5m"])
    h0, h1 = hour_window
    out: list[_P25] = []
    for fam in families:
        cls = PHASE25_FAMILIES[fam]
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
