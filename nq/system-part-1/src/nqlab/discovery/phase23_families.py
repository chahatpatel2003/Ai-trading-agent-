"""Phase 23 — HTF POI reaction + LTF reclaim/MSS confirmation (the full discretionary sequence).

The complete sequence, not generic components:
  HTF POI (support/resistance) → price trades INTO and reclaims the POI → LTF confirmation
  (reclaim / MSS / displacement / FVG) in the POI direction, inside discount/premium → enter.

Four explainable entry variants (A reclaim-close, B MSS, C displacement+FVG-retrace,
D reclaim+MSS). One trade/day (first qualifying signal), fixed stop with 1:1 / 1.5R / 2R targets,
POI-touch tolerance grid. NY-AM only. PIT-safe (HTF POI from :mod:`nqlab.features.htf_poi`; LTF
confirmation from the 5m structure layer). None assumed to work — judged against a 55% win floor.
"""

from __future__ import annotations

from itertools import product

import polars as pl

from nqlab.discovery.candidates import _Family
from nqlab.features.htf_poi import add_htf_poi

_RTH_OPEN_DEFAULT = 570
P23_EXIT_BY_TF: dict[str, dict[str, object]] = {
    "5m": {"min_hold_bars": 0, "time_stop_bars": 24},
    "15m": {"min_hold_bars": 0, "time_stop_bars": 10},
}
P23_SIZING_MODES: tuple[str, ...] = ("mnq_min1",)
P23_RISK_GRID: tuple[float, ...] = (200.0,)
P23_TOL: tuple[float, ...] = (2.0, 5.0, 10.0)      # POI touch tolerance (points)
P23_STOP = 10.0                                     # fixed stop (points)
P23_R: tuple[float, ...] = (1.0, 1.5, 2.0)          # target R-multiples (1:1, 1.5R, 2R)


class _P23(_Family):
    version = "23"
    # INPUT columns the model needs (at_support_poi/at_resistance_poi are derived in _prepare).
    required_columns: tuple[str, ...] = ("in_discount", "in_premium", "prev_day_low",
                                         "prev_day_high", "overnight_low", "overnight_high")

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_htf_poi(frame, tol_pts=float(self.p["poi_tol"]))  # type: ignore[arg-type]

    def _stop_pts(self) -> pl.Expr:
        return pl.lit(float(self.p["fix_stop"]))  # type: ignore[arg-type]

    def _target_pts(self) -> pl.Expr:
        return pl.lit(float(self.p["fix_target"]))  # type: ignore[arg-type]

    def signals(self, frame: pl.DataFrame) -> pl.DataFrame:
        sig = super().signals(frame)  # _Family.signals calls _prepare + mask + attach
        nz = pl.col("signal") != 0
        first_of_day = nz & (nz.cast(pl.Int32).cum_sum().over("trading_day") == 1)
        return sig.with_columns(
            pl.when(first_of_day).then(pl.col("signal")).otherwise(0).cast(pl.Int8).alias("signal")
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [
            {"poi_tol": tol, "fix_stop": P23_STOP, "fix_target": P23_STOP * r}
            for tol, r in product(P23_TOL, P23_R)
        ]

    # Shared sequence base: reaction AT a support/resistance HTF POI. The support POI (a level at/
    # below price = sell-side liquidity / discount) IS the location context, so we do NOT also AND
    # in_discount at the confirm bar — by then a bullish shift has lifted price out of discount.
    def _long_base(self) -> pl.Expr:
        return pl.col("at_support_poi") == 1

    def _short_base(self) -> pl.Expr:
        return pl.col("at_resistance_poi") == 1

    # "POI touched within the last few bars" — lets the LTF confirmation fire AFTER the touch.
    def _long_recent(self) -> pl.Expr:
        return pl.col("at_support_poi_recent") == 1

    def _short_recent(self) -> pl.Expr:
        return pl.col("at_resistance_poi_recent") == 1


class ReclaimClose(_P23):
    """A — Reclaim-close entry: enter once price has traded into the HTF POI and the LTF candle
    closes back through it (the POI touch flag already encodes the reclaim close)."""

    family = "p23_reclaim_close"

    def _long(self) -> pl.Expr:
        return self._long_base()

    def _short(self) -> pl.Expr:
        return self._short_base()


class MssConfirm(_P23):
    """B — MSS confirmation entry: POI reaction PLUS a lower-timeframe market-structure shift away
    from the POI."""

    family = "p23_mss_confirm"
    required_columns = ("in_discount", "in_premium", "prev_day_low", "prev_day_high",
                        "overnight_low", "overnight_high", "mss_up", "mss_dn", "choch_up",
                        "choch_dn")

    def _long(self) -> pl.Expr:  # LTF MSS or CHoCH (bullish structure shift) after the touch
        return self._long_recent() & ((pl.col("mss_up") == 1) | (pl.col("choch_up") == 1))

    def _short(self) -> pl.Expr:
        return self._short_recent() & ((pl.col("mss_dn") == 1) | (pl.col("choch_dn") == 1))


class DisplacementFvg(_P23):
    """C — Displacement + FVG retrace entry: POI reaction PLUS a displacement candle away from the
    POI that prints an LTF fair-value gap (entered on the mitigation)."""

    family = "p23_disp_fvg"
    required_columns = ("in_discount", "in_premium", "prev_day_low", "prev_day_high",
                        "overnight_low", "overnight_high", "displacement", "in_bull_fvg",
                        "in_bear_fvg")

    def _long(self) -> pl.Expr:
        return self._long_recent() & (pl.col("displacement") == 1) & (pl.col("in_bull_fvg") == 1)

    def _short(self) -> pl.Expr:
        return self._short_recent() & (pl.col("displacement") == 1) & (pl.col("in_bear_fvg") == 1)


class ReclaimMss(_P23):
    """D — Conservative reclaim + MSS entry: require BOTH the LTF reclaim and an MSS to enter."""

    family = "p23_reclaim_mss"
    required_columns = ("in_discount", "in_premium", "prev_day_low", "prev_day_high",
                        "overnight_low", "overnight_high", "reclaim_up", "reclaim_dn",
                        "mss_up", "mss_dn", "choch_up", "choch_dn")

    def _long(self) -> pl.Expr:  # reclaim AND a structure shift (MSS/CHoCH) after the touch
        return (
            self._long_recent()
            & (pl.col("reclaim_up") == 1)
            & ((pl.col("mss_up") == 1) | (pl.col("choch_up") == 1))
        )

    def _short(self) -> pl.Expr:
        return (
            self._short_recent()
            & (pl.col("reclaim_dn") == 1)
            & ((pl.col("mss_dn") == 1) | (pl.col("choch_dn") == 1))
        )


PHASE23_FAMILIES: dict[str, type[_P23]] = {
    "p23_reclaim_close": ReclaimClose,
    "p23_mss_confirm": MssConfirm,
    "p23_disp_fvg": DisplacementFvg,
    "p23_reclaim_mss": ReclaimMss,
}


def generate_phase23_candidates(
    families: tuple[str, ...] = tuple(PHASE23_FAMILIES),
    *,
    timeframe: str,
    risk_grid: tuple[float, ...] = P23_RISK_GRID,
    sizing_modes: tuple[str, ...] = P23_SIZING_MODES,
    hour_window: tuple[int, int] = (9, 12),
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_P23]:
    """family x (POI-tol x R-multiple) x risk x sizing, NY-AM, 1 trade/day. Deterministic."""
    exit_seed = P23_EXIT_BY_TF.get(timeframe, P23_EXIT_BY_TF["5m"])
    h0, h1 = hour_window
    out: list[_P23] = []
    for fam in families:
        cls = PHASE23_FAMILIES[fam]
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
