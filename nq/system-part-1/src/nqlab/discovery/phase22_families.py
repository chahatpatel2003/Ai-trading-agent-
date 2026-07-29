"""Phase 22 — selective one-trade-per-day, fixed 1:1 R:R Topstep families.

Tests the specific claim: can a ~60-70% win-rate, 1:1 R:R, $200-risk NQ NY-AM setup exist? Each
family takes at most the FIRST qualifying signal per trading day (signal-level gate, so the
gauntlet measures the true 1/day edge), with fixed-point stops/targets (10/10 default; also 8/8,
12/12, 15/15). NY-AM only. None assumed to work — judged against a 60% win-rate floor.
"""

from __future__ import annotations

import polars as pl

from nqlab.discovery.candidates import _Family
from nqlab.domain.strategy.base import MBP1_REQUIRED, attach_signals
from nqlab.features.orderflow import add_orderflow_features

_RTH_OPEN_DEFAULT = 570
P22_EXIT_BY_TF: dict[str, dict[str, object]] = {
    "5m": {"min_hold_bars": 0, "time_stop_bars": 24},
    "15m": {"min_hold_bars": 0, "time_stop_bars": 10},
}
P22_SIZING_MODES: tuple[str, ...] = ("mnq_min1",)
P22_RISK_GRID: tuple[float, ...] = (200.0,)
# Fixed 1:1 R:R in NQ points (10/10 primary).
P22_FIXED: tuple[tuple[float, float], ...] = ((10.0, 10.0), (8.0, 8.0), (12.0, 12.0), (15.0, 15.0))


def _bias_up() -> pl.Expr:
    return pl.col("trend_4h_bull") == 1


def _bias_dn() -> pl.Expr:
    return pl.col("trend_4h_bull") == 0


class _P22(_Family):
    version = "22"

    def _stop_pts(self) -> pl.Expr:
        return pl.lit(float(self.p["fix_stop"]))  # type: ignore[arg-type]

    def _target_pts(self) -> pl.Expr:
        return pl.lit(float(self.p["fix_target"]))  # type: ignore[arg-type]

    def signals(self, frame: pl.DataFrame) -> pl.DataFrame:
        sig = super().signals(frame)
        # Keep ONLY the first non-zero signal per trading day (1 trade/day at the signal level).
        nz = pl.col("signal") != 0
        first_of_day = nz & (nz.cast(pl.Int32).cum_sum().over("trading_day") == 1)
        return sig.with_columns(
            pl.when(first_of_day).then(pl.col("signal")).otherwise(0).cast(pl.Int8).alias("signal")
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [{"fix_stop": s, "fix_target": t} for s, t in P22_FIXED]


class FirstCleanBias(_P22):
    """A — First clean NY bias trade: the first bar where 4h bias, VWAP, and the opening range all
    agree on direction."""

    family = "p22_first_bias"
    required_columns = ("trend_4h_bull", "vwap_session", "or_high", "or_low")

    def _long(self) -> pl.Expr:
        return (
            _bias_up()
            & (pl.col("close") > pl.col("vwap_session"))
            & pl.col("or_high").is_not_null()
            & (pl.col("close") > pl.col("or_high"))
        )

    def _short(self) -> pl.Expr:
        return (
            _bias_dn()
            & (pl.col("close") < pl.col("vwap_session"))
            & pl.col("or_low").is_not_null()
            & (pl.col("close") < pl.col("or_low"))
        )


class ContinuationScalp(_P22):
    """B — 10-point continuation scalp: in the 4h-bias direction, the first pullback that tags and
    reclaims VWAP."""

    family = "p22_continuation"
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


class SweepReversal(_P22):
    """C — 10-point reversal after sweep: the first swept swing that reclaims back."""

    family = "p22_sweep_rev"
    required_columns = ("swept_high", "swept_low", "reclaim_up", "reclaim_dn")

    def _long(self) -> pl.Expr:
        return (pl.col("swept_low") == 1) & (pl.col("reclaim_up") == 1)

    def _short(self) -> pl.Expr:
        return (pl.col("swept_high") == 1) & (pl.col("reclaim_dn") == 1)


class OrFailedBreakout(_P22):
    """D — Opening-range failed-breakout 1:1: the first failed OR break that reclaims into range."""

    family = "p22_or_failbrk"
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


class VwapReclaim(_P22):
    """E — VWAP reclaim 1:1: the first bar that deviates below/above VWAP and reclaims it."""

    family = "p22_vwap_reclaim"
    required_columns = ("vwap_session",)

    def _long(self) -> pl.Expr:
        return (pl.col("low") <= pl.col("vwap_session")) & (
            pl.col("close") > pl.col("vwap_session")
        )

    def _short(self) -> pl.Expr:
        return (pl.col("high") >= pl.col("vwap_session")) & (
            pl.col("close") < pl.col("vwap_session")
        )


class Mbp1Confirmed(_P22):
    """F — MBP1-confirmed 1:1: VWAP reclaim that order flow (OFI) confirms. MBP1-required, so the
    verdict is capped at NEEDS_FORWARD_DATA without a complete MBP1 lockbox."""

    family = "p22_mbp1_confirm"
    data_requirement = MBP1_REQUIRED
    required_columns = ("vwap_session", "ofi")

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_orderflow_features(frame)

    def _long(self) -> pl.Expr:
        return (
            (pl.col("low") <= pl.col("vwap_session"))
            & (pl.col("close") > pl.col("vwap_session"))
            & (pl.col("ofi_z") > 0)
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("high") >= pl.col("vwap_session"))
            & (pl.col("close") < pl.col("vwap_session"))
            & (pl.col("ofi_z") < 0)
        )

    def signals(self, frame: pl.DataFrame) -> pl.DataFrame:
        # _prepare adds order-flow; reuse the base 1/day gate via attach + first-of-day.
        if self._missing_required(frame):
            return frame.with_columns(
                pl.lit(0, dtype=pl.Int8).alias("signal"),
                pl.lit(None, dtype=pl.Float64).alias("stop_pts"),
                pl.lit(None, dtype=pl.Float64).alias("target_pts"),
            )
        prepared = self._prepare(frame)
        from nqlab.discovery.candidates import _filter_mask

        mask = _filter_mask(self.p)
        sig = attach_signals(
            prepared, long=self._long() & mask, short=self._short() & mask,
            stop_pts=self._stop_pts(), target_pts=self._target_pts(),
        )
        nz = pl.col("signal") != 0
        first_of_day = nz & (nz.cast(pl.Int32).cum_sum().over("trading_day") == 1)
        return sig.with_columns(
            pl.when(first_of_day).then(pl.col("signal")).otherwise(0).cast(pl.Int8).alias("signal")
        )


PHASE22_FAMILIES: dict[str, type[_P22]] = {
    "p22_first_bias": FirstCleanBias,
    "p22_continuation": ContinuationScalp,
    "p22_sweep_rev": SweepReversal,
    "p22_or_failbrk": OrFailedBreakout,
    "p22_vwap_reclaim": VwapReclaim,
    "p22_mbp1_confirm": Mbp1Confirmed,
}


def generate_phase22_candidates(
    families: tuple[str, ...] = tuple(PHASE22_FAMILIES),
    *,
    timeframe: str,
    risk_grid: tuple[float, ...] = P22_RISK_GRID,
    sizing_modes: tuple[str, ...] = P22_SIZING_MODES,
    hour_window: tuple[int, int] = (9, 12),
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_P22]:
    """family x fixed 1:1 grid x risk x sizing, NY-AM, 1 trade/day. Deterministic."""
    exit_seed = P22_EXIT_BY_TF.get(timeframe, P22_EXIT_BY_TF["5m"])
    h0, h1 = hour_window
    out: list[_P22] = []
    for fam in families:
        cls = PHASE22_FAMILIES[fam]
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
