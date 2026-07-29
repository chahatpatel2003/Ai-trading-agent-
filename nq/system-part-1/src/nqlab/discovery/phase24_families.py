"""Phase 24 — four strict prop-style 1:1 scalping models.

Tested as filtered, payout-focused prop models (not the rough generic versions from earlier):
  1. Opening Range Fade (high win-rate failed-breakout reclaim + OR-size/sweep-extension filters)
  2. Liquidity Sweep Reversal (sweep + fast reclaim + structure/displacement confirmation)
  3. VWAP Reversion with a real distance filter (ATR-stretch + skip very-strong-trend ADX)
  4. Micro Pullback Scalps in Trend (trend + ADX trending + shallow VWAP pullback)

All are NQ NY-AM only, one trade/day (first qualifying signal), fixed 1:1 R:R (8/10/12/15 pt), with
ATR/ADX computed PIT-safe. Judged against a 60% win-rate floor. None assumed to work.
"""

from __future__ import annotations

import polars as pl

from nqlab.discovery.candidates import _Family
from nqlab.features.indicators import add_adx

_RTH_OPEN_DEFAULT = 570
P24_EXIT_BY_TF: dict[str, dict[str, object]] = {
    "1m": {"min_hold_bars": 0, "time_stop_bars": 90},
    "5m": {"min_hold_bars": 0, "time_stop_bars": 24},
}
P24_SIZING_MODES: tuple[str, ...] = ("mnq_min1",)
P24_RISK_GRID: tuple[float, ...] = (200.0,)
P24_FIXED: tuple[tuple[float, float], ...] = ((8.0, 8.0), (10.0, 10.0), (12.0, 12.0), (15.0, 15.0))
_SWEEP_THR = 2.0   # min points a sweep must extend beyond the level (real liquidity run)
_VWAP_DIST_ATR = 1.0   # price must be this many ATR from VWAP to fade
_VWAP_ADX_MAX = 35.0   # VWAP fade: skip only VERY strong trends (big stretches happen in trends)
_ADX_TREND = 20.0  # pullback model: ADX >= this = trending enough (not chop)


class _P24(_Family):
    version = "24"

    def _stop_pts(self) -> pl.Expr:
        return pl.lit(float(self.p["fix_stop"]))  # type: ignore[arg-type]

    def _target_pts(self) -> pl.Expr:
        return pl.lit(float(self.p["fix_target"]))  # type: ignore[arg-type]

    def signals(self, frame: pl.DataFrame) -> pl.DataFrame:
        sig = super().signals(frame)
        nz = pl.col("signal") != 0
        first_of_day = nz & (nz.cast(pl.Int32).cum_sum().over("trading_day") == 1)
        return sig.with_columns(
            pl.when(first_of_day).then(pl.col("signal")).otherwise(0).cast(pl.Int8).alias("signal")
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"fix_stop": s, "fix_target": t} for s, t in P24_FIXED]


class OpeningRangeFade(_P24):
    """1 — Opening Range Fade (high win-rate): the first clean failed breakout of the opening range
    that sweeps beyond it by a threshold and closes back inside; skip too-small/too-large OR and
    strong-impulse (range-expansion) opens."""

    family = "p24_orf"
    required_columns = ("or_high", "or_low", "atr", "range_expansion")

    def _or_ok(self) -> pl.Expr:
        rng = pl.col("or_high") - pl.col("or_low")
        return (rng >= 5.0) & (rng <= 6.0 * pl.col("atr")) & (pl.col("range_expansion") == 0)

    def _long(self) -> pl.Expr:  # failed breakdown: swept below OR low, closed back inside
        return (
            pl.col("or_low").is_not_null()
            & (pl.col("low") < pl.col("or_low") - _SWEEP_THR)
            & (pl.col("close") > pl.col("or_low"))
            & self._or_ok()
        )

    def _short(self) -> pl.Expr:
        return (
            pl.col("or_high").is_not_null()
            & (pl.col("high") > pl.col("or_high") + _SWEEP_THR)
            & (pl.col("close") < pl.col("or_high"))
            & self._or_ok()
        )


class LiquiditySweepReversal(_P24):
    """2 — Liquidity Sweep Reversal: sweep a swing liquidity pool, reclaim fast, and confirm with a
    structure shift / displacement away from the sweep."""

    family = "p24_sweep_rev"
    required_columns = ("swept_high", "swept_low", "reclaim_up", "reclaim_dn", "displacement",
                        "choch_up", "choch_dn")

    def _long(self) -> pl.Expr:
        return (
            (pl.col("swept_low") == 1)
            & (pl.col("reclaim_up") == 1)
            & ((pl.col("displacement") == 1) | (pl.col("choch_up") == 1))
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("swept_high") == 1)
            & (pl.col("reclaim_dn") == 1)
            & ((pl.col("displacement") == 1) | (pl.col("choch_dn") == 1))
        )


class VwapReversionDistance(_P24):
    """3 — VWAP Reversion with distance filter: price stretched >= 1.5 ATR from VWAP, an exhaustion/
    reclaim candle, and ADX below the trend threshold (skip strong trends that walk the band)."""

    family = "p24_vwap_rev"
    required_columns = ("vwap_session", "dist_vwap", "atr")

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_adx(frame, n=14)

    def _long(self) -> pl.Expr:  # stretched below VWAP, bullish reclaim candle, not a violent trend
        return (
            (pl.col("dist_vwap") < -_VWAP_DIST_ATR * pl.col("atr"))
            & (pl.col("close") > pl.col("open"))
            & (pl.col("adx") < _VWAP_ADX_MAX)
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("dist_vwap") > _VWAP_DIST_ATR * pl.col("atr"))
            & (pl.col("close") < pl.col("open"))
            & (pl.col("adx") < _VWAP_ADX_MAX)
        )


class MicroPullbackTrend(_P24):
    """4 — Micro Pullback Scalps in Trend: with a confirmed NY trend (4h bias + ADX trending), a
    shallow pullback that tags and reclaims VWAP in the trend direction."""

    family = "p24_pullback"
    required_columns = ("trend_4h_bull", "vwap_session")

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_adx(frame, n=14)

    def _long(self) -> pl.Expr:
        return (
            (pl.col("trend_4h_bull") == 1)
            & (pl.col("adx") >= _ADX_TREND)
            & (pl.col("low") <= pl.col("vwap_session"))
            & (pl.col("close") > pl.col("vwap_session"))
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("trend_4h_bull") == 0)
            & (pl.col("adx") >= _ADX_TREND)
            & (pl.col("high") >= pl.col("vwap_session"))
            & (pl.col("close") < pl.col("vwap_session"))
        )


PHASE24_FAMILIES: dict[str, type[_P24]] = {
    "p24_orf": OpeningRangeFade,
    "p24_sweep_rev": LiquiditySweepReversal,
    "p24_vwap_rev": VwapReversionDistance,
    "p24_pullback": MicroPullbackTrend,
}


def generate_phase24_candidates(
    families: tuple[str, ...] = tuple(PHASE24_FAMILIES),
    *,
    timeframe: str,
    risk_grid: tuple[float, ...] = P24_RISK_GRID,
    sizing_modes: tuple[str, ...] = P24_SIZING_MODES,
    hour_window: tuple[int, int] = (9, 12),
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_P24]:
    """family x fixed 1:1 grid x risk x sizing, NY-AM, 1 trade/day. Deterministic."""
    exit_seed = P24_EXIT_BY_TF.get(timeframe, P24_EXIT_BY_TF["5m"])
    h0, h1 = hour_window
    out: list[_P24] = []
    for fam in families:
        cls = PHASE24_FAMILIES[fam]
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
