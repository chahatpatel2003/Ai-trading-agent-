"""Candidate strategy families and their parameter grids.

All families reuse the existing trades-based features. Family-specific indicators
(opening range for a given window, EMA of a given span) are computed inside
``signals`` from the bar columns so the feature store stays fixed. Shared
volatility/time/day filters (deliverable E) gate every family's signals.

None of these are assumed to work — they are candidates to be judged and rejected.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product

import polars as pl

from nqlab.domain.strategy.base import (
    MBP1_REQUIRED,
    BaseStrategy,
    ExecConfig,
    StrategyParams,
    TrailingConfig,
    attach_signals,
)

_RTH_OPEN_DEFAULT = 570  # 09:30 ET
_atr = pl.col("atr")


def _trailing_from_params(params: dict[str, object]) -> TrailingConfig | None:
    if params.get("trail") == "atr":
        return TrailingConfig(mode="atr", trail_atr=float(params.get("trail_atr", 2.0)))  # type: ignore[arg-type]
    if params.get("breakeven_r") is not None:
        return TrailingConfig(breakeven_r=float(params["breakeven_r"]))  # type: ignore[arg-type]
    return None


def _exec_from_params(params: dict[str, object]) -> ExecConfig | None:
    risk = params.get("risk_per_trade")
    min_hold = params.get("min_hold_bars")
    time_stop = params.get("time_stop_bars")
    sizing = params.get("sizing_mode")
    max_losses = params.get("max_daily_losses")
    max_trades = params.get("max_daily_trades")
    profit_target = params.get("daily_profit_target")
    if all(
        v is None
        for v in (risk, min_hold, time_stop, sizing, max_losses, max_trades, profit_target)
    ):
        return None
    return ExecConfig(
        risk_per_trade_usd=float(risk) if risk is not None else None,  # type: ignore[arg-type]
        min_hold_bars=int(min_hold) if min_hold is not None else 0,  # type: ignore[call-overload]
        time_stop_bars=int(time_stop) if time_stop is not None else None,  # type: ignore[call-overload]
        sizing_mode=str(sizing) if sizing is not None else "adaptive",
        max_daily_losses=int(max_losses) if max_losses is not None else None,  # type: ignore[call-overload]
        max_daily_trades=int(max_trades) if max_trades is not None else None,  # type: ignore[call-overload]
        daily_profit_target_usd=float(profit_target)  # type: ignore[arg-type]
        if profit_target is not None
        else None,
    )


def _filter_mask(p: dict[str, object]) -> pl.Expr:
    """Shared volatility / time-of-day / day-of-week filters."""
    mask = pl.lit(value=True)
    if "hour_start" in p and "hour_end" in p:
        start = int(p["hour_start"]) * 60  # type: ignore[call-overload]
        end = int(p["hour_end"]) * 60  # type: ignore[call-overload]
        mask = mask & (pl.col("minute_of_day") >= start) & (pl.col("minute_of_day") < end)
    regime = p.get("vol_regime", "any")
    if regime != "any":
        median = pl.col("realized_vol").rolling_median(window_size=100)
        rv = pl.col("realized_vol")
        mask = mask & (rv > median if regime == "high" else rv <= median)
    skip = p.get("dow_skip")
    if skip is not None:
        mask = mask & (pl.col("dow") != int(skip))  # type: ignore[call-overload]
    return mask


class _Family(BaseStrategy):
    family = "?"
    version = "1"

    def __init__(self, params: dict[str, object], rth_start_min: int = _RTH_OPEN_DEFAULT) -> None:
        self.p = params
        self.rth_start_min = rth_start_min
        items = tuple(sorted(params.items()))
        super().__init__(
            StrategyParams(self.family, self.version, items),  # type: ignore[arg-type]
            trailing=_trailing_from_params(params),
            exec_config=_exec_from_params(params),
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        raise NotImplementedError

    # Subclasses implement the four expressions; _prepare may add columns.
    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    def _long(self) -> pl.Expr: ...
    def _short(self) -> pl.Expr: ...
    def _stop_pts(self) -> pl.Expr:
        return float(self.p["stop_atr"]) * _atr  # type: ignore[arg-type]

    def _target_pts(self) -> pl.Expr:
        return float(self.p["target_atr"]) * _atr  # type: ignore[arg-type]

    def signals(self, frame: pl.DataFrame) -> pl.DataFrame:
        # Data-availability contract: if required (MBP1) columns are absent, skip
        # explicitly — emit no signals rather than silently degrade.
        if self._missing_required(frame):
            return frame.with_columns(
                pl.lit(0, dtype=pl.Int8).alias("signal"),
                pl.lit(None, dtype=pl.Float64).alias("stop_pts"),
                pl.lit(None, dtype=pl.Float64).alias("target_pts"),
            )
        prepared = self._prepare(frame)
        mask = _filter_mask(self.p)
        return attach_signals(
            prepared, long=self._long() & mask, short=self._short() & mask,
            stop_pts=self._stop_pts(), target_pts=self._target_pts(),
        )


def _opening_range(frame: pl.DataFrame, rth_start_min: int, or_minutes: int) -> pl.DataFrame:
    or_end = rth_start_min + or_minutes
    bands = (
        frame.filter(
            pl.col("is_rth")
            & (pl.col("minute_of_day") >= rth_start_min)
            & (pl.col("minute_of_day") < or_end)
        )
        .group_by("trading_day")
        .agg(pl.col("high").max().alias("_orh"), pl.col("low").min().alias("_orl"))
    )
    return frame.join(bands, on="trading_day", how="left").with_columns(
        pl.when(pl.col("minute_of_day") >= or_end).then(pl.col("_orh")).alias("_orh"),
        pl.when(pl.col("minute_of_day") >= or_end).then(pl.col("_orl")).alias("_orl"),
    )


class OpeningRangeFamily(_Family):
    family = "orb"

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return _opening_range(frame, self.rth_start_min, int(self.p["or_minutes"]))  # type: ignore[call-overload]

    def _long(self) -> pl.Expr:
        buf = float(self.p["buffer_atr"]) * _atr  # type: ignore[arg-type]
        cond = pl.col("_orh").is_not_null() & (pl.col("close") > pl.col("_orh") + buf)
        if self.p.get("trend_filter") == "vwap":
            cond = cond & (pl.col("close") > pl.col("vwap_session"))
        return cond

    def _short(self) -> pl.Expr:
        buf = float(self.p["buffer_atr"]) * _atr  # type: ignore[arg-type]
        cond = pl.col("_orl").is_not_null() & (pl.col("close") < pl.col("_orl") - buf)
        if self.p.get("trend_filter") == "vwap":
            cond = cond & (pl.col("close") < pl.col("vwap_session"))
        return cond

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        out = []
        for orm, buf, stop, tgt, trend in product(
            (15, 30, 60), (0.0, 0.5), (1.0, 1.5), (2.0,), ("none", "vwap")
        ):
            out.append({
                "or_minutes": orm, "buffer_atr": buf, "stop_atr": stop,
                "target_atr": tgt, "trend_filter": trend,
            })
        return out


class VwapFamily(_Family):
    family = "vwap"

    def _long(self) -> pl.Expr:
        band = float(self.p["entry_atr"]) * _atr  # type: ignore[arg-type]
        if self.p["mode"] == "reversion":
            return pl.col("dist_vwap") < -band
        return pl.col("dist_vwap") > band

    def _short(self) -> pl.Expr:
        band = float(self.p["entry_atr"]) * _atr  # type: ignore[arg-type]
        if self.p["mode"] == "reversion":
            return pl.col("dist_vwap") > band
        return pl.col("dist_vwap") < -band

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        out = []
        for mode, entry, stop, tgt, regime in product(
            ("reversion", "continuation"), (1.0, 2.0), (1.0, 1.5), (2.0,), ("any", "high")
        ):
            out.append({
                "mode": mode, "entry_atr": entry, "stop_atr": stop,
                "target_atr": tgt, "vol_regime": regime,
            })
        return out


class PrevDayFamily(_Family):
    family = "pdhl"

    def _long(self) -> pl.Expr:
        buf = float(self.p["buffer_atr"]) * _atr  # type: ignore[arg-type]
        if self.p["mode"] == "breakout":
            cond = pl.col("prev_day_high").is_not_null() & (
                pl.col("close") > pl.col("prev_day_high") + buf
            )
            if self.p.get("trend_filter") == "vwap":
                cond = cond & (pl.col("close") > pl.col("vwap_session"))
            return cond
        # failed_breakout: fade a poke below the prior low.
        low = pl.col("prev_day_low")
        return low.is_not_null() & (pl.col("close") < low - buf)

    def _short(self) -> pl.Expr:
        buf = float(self.p["buffer_atr"]) * _atr  # type: ignore[arg-type]
        if self.p["mode"] == "breakout":
            cond = pl.col("prev_day_low").is_not_null() & (
                pl.col("close") < pl.col("prev_day_low") - buf
            )
            if self.p.get("trend_filter") == "vwap":
                cond = cond & (pl.col("close") < pl.col("vwap_session"))
            return cond
        high = pl.col("prev_day_high")
        return high.is_not_null() & (pl.col("close") > high + buf)

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        out = []
        for mode, buf, stop, tgt, trend in product(
            ("breakout", "failed_breakout"), (0.0, 0.5), (1.5,), (2.0, 3.0), ("none", "vwap")
        ):
            out.append({
                "mode": mode, "buffer_atr": buf, "stop_atr": stop,
                "target_atr": tgt, "trend_filter": trend,
            })
        return out


class TrendFamily(_Family):
    family = "trend"

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        span = int(self.p["ema_span"])  # type: ignore[call-overload]
        return frame.with_columns(pl.col("close").ewm_mean(span=span, adjust=False).alias("_ema"))

    def _long(self) -> pl.Expr:
        up = pl.col("close") > pl.col("_ema")
        if self.p["pullback"] == "rsi":
            return up & (pl.col("rsi") < float(self.p["rsi_low"]))  # type: ignore[arg-type]
        return up & (pl.col("low") <= pl.col("_ema"))

    def _short(self) -> pl.Expr:
        down = pl.col("close") < pl.col("_ema")
        if self.p["pullback"] == "rsi":
            return down & (pl.col("rsi") > float(self.p["rsi_high"]))  # type: ignore[arg-type]
        return down & (pl.col("high") >= pl.col("_ema"))

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        out = []
        for span, pull, tgt in product((20, 50), ("rsi", "ema"), (2.5, 3.5)):
            out.append({
                "ema_span": span, "pullback": pull, "rsi_low": 35.0, "rsi_high": 65.0,
                "stop_atr": 1.5, "target_atr": tgt,
            })
        return out


# ---------------------------------------------------------------------------
# MBP-1 order-flow families (declare MBP1_REQUIRED; skip where MBP1 is absent).
# ---------------------------------------------------------------------------


class ImbalanceContinuation(_Family):
    family = "imb_cont"
    data_requirement = MBP1_REQUIRED
    required_columns = ("tob_imbalance",)

    def _long(self) -> pl.Expr:
        cond = pl.col("tob_imbalance") > float(self.p["imb_thr"])  # type: ignore[arg-type]
        if self.p.get("trend_filter") == "vwap":
            cond = cond & (pl.col("close") > pl.col("vwap_session"))
        return cond

    def _short(self) -> pl.Expr:
        cond = pl.col("tob_imbalance") < -float(self.p["imb_thr"])  # type: ignore[arg-type]
        if self.p.get("trend_filter") == "vwap":
            cond = cond & (pl.col("close") < pl.col("vwap_session"))
        return cond

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [
            {"imb_thr": t, "stop_atr": s, "target_atr": 2.0, "trend_filter": tr, "vol_regime": vr}
            for t, s, tr, vr in product((0.2, 0.4), (1.0, 1.5), ("none", "vwap"), ("any", "high"))
        ]


class DeltaReversal(_Family):
    family = "delta_rev"
    data_requirement = MBP1_REQUIRED
    required_columns = ("delta", "roll_high", "roll_low")

    def _long(self) -> pl.Expr:  # new low rejected by positive delta
        return (pl.col("close") <= pl.col("roll_low")) & (pl.col("delta") > 0)

    def _short(self) -> pl.Expr:  # new high rejected by negative delta
        return (pl.col("close") >= pl.col("roll_high")) & (pl.col("delta") < 0)

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [
            {"stop_atr": s, "target_atr": t, "vol_regime": vr}
            for s, t, vr in product((1.0, 1.5), (1.5, 2.5), ("any", "high"))
        ]


class Absorption(_Family):
    family = "absorption"
    data_requirement = MBP1_REQUIRED
    required_columns = ("delta",)

    def _long(self) -> pl.Expr:  # heavy selling absorbed, price holds -> fade up
        return (pl.col("delta") < -float(self.p["delta_thr"])) & (pl.col("close") >= pl.col("open"))  # type: ignore[arg-type]

    def _short(self) -> pl.Expr:  # heavy buying absorbed, price stalls -> fade down
        return (pl.col("delta") > float(self.p["delta_thr"])) & (pl.col("close") <= pl.col("open"))  # type: ignore[arg-type]

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [
            {"delta_thr": d, "stop_atr": s, "target_atr": 2.0}
            for d, s in product((500.0, 1500.0), (1.0, 1.5))
        ]


class LiquidityThinBreakout(_Family):
    family = "liq_breakout"
    data_requirement = MBP1_REQUIRED
    required_columns = ("liq_spread_ratio", "roll_high", "roll_low")

    def _long(self) -> pl.Expr:
        return (pl.col("close") > pl.col("roll_high")) & (
            pl.col("liq_spread_ratio") < float(self.p["liq_thr"])  # type: ignore[arg-type]
        )

    def _short(self) -> pl.Expr:
        return (pl.col("close") < pl.col("roll_low")) & (
            pl.col("liq_spread_ratio") < float(self.p["liq_thr"])  # type: ignore[arg-type]
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [
            {"liq_thr": q, "stop_atr": 1.5, "target_atr": t}
            for q, t in product((1.0, 1.5), (2.0, 3.0))
        ]


class CvdTrendConfirm(_Family):
    family = "cvd_trend"
    data_requirement = MBP1_REQUIRED
    required_columns = ("cvd",)

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        span = int(self.p["ema_span"])  # type: ignore[call-overload]
        return frame.with_columns(pl.col("close").ewm_mean(span=span, adjust=False).alias("_ema"))

    def _long(self) -> pl.Expr:
        return (pl.col("close") > pl.col("_ema")) & (pl.col("cvd") > 0)

    def _short(self) -> pl.Expr:
        return (pl.col("close") < pl.col("_ema")) & (pl.col("cvd") < 0)

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return [
            {"ema_span": span, "pullback": "none", "stop_atr": 1.5, "target_atr": t, "trail": tr}
            for span, t, tr in product((20, 50), (2.5, 3.5), ("none", "atr"))
        ]


FAMILIES: dict[str, type[_Family]] = {
    "orb": OpeningRangeFamily,
    "vwap": VwapFamily,
    "pdhl": PrevDayFamily,
    "trend": TrendFamily,
}

MBP1_FAMILIES: dict[str, type[_Family]] = {
    "imb_cont": ImbalanceContinuation,
    "delta_rev": DeltaReversal,
    "absorption": Absorption,
    "liq_breakout": LiquidityThinBreakout,
    "cvd_trend": CvdTrendConfirm,
}

ALL_FAMILIES: dict[str, type[_Family]] = {**FAMILIES, **MBP1_FAMILIES}


def generate_candidates(
    *, rth_start_min: int = _RTH_OPEN_DEFAULT, families: tuple[str, ...] | None = None
) -> Iterator[_Family]:
    """Deterministically enumerate candidate strategies (family order, grid order)."""
    names = families or tuple(FAMILIES)  # default = trades-only families
    for name in names:
        cls = ALL_FAMILIES[name]
        for params in cls.grid():
            yield cls(params, rth_start_min=rth_start_min)


# ---------------------------------------------------------------------------
# Phase 5: combination families (order-flow + trend/VWAP/liquidity/time). Their
# grids are calibrated from profiled percentiles (see discovery/calibration.py).
# ---------------------------------------------------------------------------


def rolling_zscore(col: pl.Expr, window: int) -> pl.Expr:
    """Trailing rolling z-score (point-in-time safe)."""
    mean = col.rolling_mean(window_size=window)
    std = col.rolling_std(window_size=window)
    return (col - mean) / std


class AbsorptionReclaim(_Family):  # combo A
    family = "absorb_reclaim"
    data_requirement = MBP1_REQUIRED
    required_columns = ("delta", "roll_high", "roll_low")

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            rolling_zscore(pl.col("delta"), int(self.p["z_window"])).alias("_dz")  # type: ignore[call-overload]
        )

    def _long(self) -> pl.Expr:
        z = float(self.p["z_thr"])  # type: ignore[arg-type]
        new_low = pl.col("close") <= pl.col("roll_low")
        return new_low & (pl.col("_dz") < -z) & (pl.col("close") > pl.col("open"))

    def _short(self) -> pl.Expr:
        z = float(self.p["z_thr"])  # type: ignore[arg-type]
        new_high = pl.col("close") >= pl.col("roll_high")
        return new_high & (pl.col("_dz") > z) & (pl.col("close") < pl.col("open"))


class ImbalanceTrendCont(_Family):  # combo B
    family = "imb_trend"
    data_requirement = MBP1_REQUIRED
    required_columns = ("tob_imbalance", "spread")

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            pl.col("close").ewm_mean(span=int(self.p["ema_span"]), adjust=False).alias("_ema")  # type: ignore[call-overload]
        )

    def _long(self) -> pl.Expr:
        return (
            (pl.col("tob_imbalance") > float(self.p["imb_thr"]))  # type: ignore[arg-type]
            & (pl.col("close") > pl.col("_ema"))
            & (pl.col("spread") < float(self.p["spread_max"]))  # type: ignore[arg-type]
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("tob_imbalance") < -float(self.p["imb_thr"]))  # type: ignore[arg-type]
            & (pl.col("close") < pl.col("_ema"))
            & (pl.col("spread") < float(self.p["spread_max"]))  # type: ignore[arg-type]
        )


class DeltaDivergenceVwap(_Family):  # combo C
    family = "delta_div_vwap"
    data_requirement = MBP1_REQUIRED
    required_columns = ("delta", "dist_vwap")

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            rolling_zscore(pl.col("delta"), int(self.p["z_window"])).alias("_dz")  # type: ignore[call-overload]
        )

    def _long(self) -> pl.Expr:
        k = float(self.p["vwap_k"])  # type: ignore[arg-type]
        extended = pl.col("dist_vwap") < -(k * _atr)
        return extended & (pl.col("_dz") > 0) & (pl.col("close") > pl.col("open"))

    def _short(self) -> pl.Expr:
        k = float(self.p["vwap_k"])  # type: ignore[arg-type]
        extended = pl.col("dist_vwap") > (k * _atr)
        return extended & (pl.col("_dz") < 0) & (pl.col("close") < pl.col("open"))


class LiquidityRegimeBreakout(_Family):  # combo D
    family = "liq_regime_brk"
    data_requirement = MBP1_REQUIRED
    required_columns = ("liq_spread_ratio", "spread", "roll_high", "roll_low")

    def _long(self) -> pl.Expr:
        return (
            (pl.col("close") > pl.col("roll_high"))
            & (pl.col("liq_spread_ratio") < float(self.p["liq_thr"]))  # type: ignore[arg-type]
            & (pl.col("spread") < float(self.p["spread_max"]))  # type: ignore[arg-type]
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("close") < pl.col("roll_low"))
            & (pl.col("liq_spread_ratio") < float(self.p["liq_thr"]))  # type: ignore[arg-type]
            & (pl.col("spread") < float(self.p["spread_max"]))  # type: ignore[arg-type]
        )


class CvdConfirmationPullback(_Family):  # combo E
    family = "cvd_pullback"
    data_requirement = MBP1_REQUIRED
    required_columns = ("cvd",)

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            pl.col("close").ewm_mean(span=int(self.p["ema_span"]), adjust=False).alias("_ema"),  # type: ignore[call-overload]
            rolling_zscore(
                pl.col("cvd").diff().over("trading_day"), int(self.p["z_window"])  # type: ignore[call-overload]
            ).alias("_cz"),
        )

    def _long(self) -> pl.Expr:
        return (
            (pl.col("close") > pl.col("_ema"))
            & (pl.col("rsi") < float(self.p["rsi_low"]))  # type: ignore[arg-type]
            & (pl.col("_cz") > 0)
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("close") < pl.col("_ema"))
            & (pl.col("rsi") > float(self.p["rsi_high"]))  # type: ignore[arg-type]
            & (pl.col("_cz") < 0)
        )


COMBO_FAMILIES: dict[str, type[_Family]] = {
    "absorb_reclaim": AbsorptionReclaim,
    "imb_trend": ImbalanceTrendCont,
    "delta_div_vwap": DeltaDivergenceVwap,
    "liq_regime_brk": LiquidityRegimeBreakout,
    "cvd_pullback": CvdConfirmationPullback,
}


# ---------------------------------------------------------------------------
# Phase 6: setup families = market context + liquidity location + order-flow
# trigger + strict execution. All require MBP1 (order-flow confirmation) and the
# context columns (session ranges / bias / gap).
# ---------------------------------------------------------------------------


class SessionSweepReversal(_Family):  # setup A — sweep a prior session high/low, reclaim, fade
    family = "session_sweep_rev"
    data_requirement = MBP1_REQUIRED
    required_columns = ("asia_high", "london_high", "delta")

    def _hi(self) -> pl.Expr:
        return pl.col(f"{self.p['session']}_high")

    def _lo(self) -> pl.Expr:
        return pl.col(f"{self.p['session']}_low")

    def _long(self) -> pl.Expr:  # swept the low, reclaimed, buyers stepping in
        swept = (pl.col("low") < self._lo()) & (pl.col("close") > self._lo())
        return swept & (pl.col("delta") > 0)

    def _short(self) -> pl.Expr:
        swept = (pl.col("high") > self._hi()) & (pl.col("close") < self._hi())
        return swept & (pl.col("delta") < 0)


class PrevDaySweepReversal(_Family):  # setup B
    family = "pd_sweep_rev"
    data_requirement = MBP1_REQUIRED
    required_columns = ("prev_day_high", "prev_day_low", "delta")

    def _long(self) -> pl.Expr:
        low = pl.col("prev_day_low")
        return (pl.col("low") < low) & (pl.col("close") > low) & (pl.col("delta") > 0)

    def _short(self) -> pl.Expr:
        high = pl.col("prev_day_high")
        return (pl.col("high") > high) & (pl.col("close") < high) & (pl.col("delta") < 0)


class BiasAlignedPullback(_Family):  # setup C
    family = "bias_pullback"
    data_requirement = MBP1_REQUIRED
    required_columns = ("htf_bull", "cvd")

    def _long(self) -> pl.Expr:  # HTF up, pulled back below VWAP, CVD confirms
        pullback = (pl.col("htf_bull") == 1) & (pl.col("close") < pl.col("vwap_session"))
        return pullback & (pl.col("cvd") > 0)

    def _short(self) -> pl.Expr:
        pullback = (pl.col("htf_bull") == 0) & (pl.col("close") > pl.col("vwap_session"))
        return pullback & (pl.col("cvd") < 0)


class OpeningRangeContext(_Family):  # setup D — OR break with bias + good liquidity
    family = "or_context"
    data_requirement = MBP1_REQUIRED
    required_columns = ("or_high", "or_low", "gap", "liq_spread_ratio", "spread")

    def _liquid(self) -> pl.Expr:
        return (pl.col("liq_spread_ratio") < float(self.p["liq_thr"])) & (  # type: ignore[arg-type]
            pl.col("spread") < float(self.p["spread_max"])  # type: ignore[arg-type]
        )

    def _long(self) -> pl.Expr:
        bias = (pl.col("gap") > 0) | (pl.col("htf_bull") == 1)
        return (pl.col("close") > pl.col("or_high")) & bias & self._liquid()

    def _short(self) -> pl.Expr:
        bias = (pl.col("gap") < 0) | (pl.col("htf_bull") == 0)
        return (pl.col("close") < pl.col("or_low")) & bias & self._liquid()


class FailedBreakoutTrap(_Family):  # setup E
    family = "failed_brk_trap"
    data_requirement = MBP1_REQUIRED
    required_columns = ("roll_high", "roll_low", "delta")

    def _long(self) -> pl.Expr:  # poke below recent low, no selling, reclaim -> long
        reclaim = (pl.col("low") < pl.col("roll_low")) & (pl.col("close") > pl.col("roll_low"))
        return reclaim & (pl.col("delta") > 0)

    def _short(self) -> pl.Expr:
        reclaim = (pl.col("high") > pl.col("roll_high")) & (pl.col("close") < pl.col("roll_high"))
        return reclaim & (pl.col("delta") < 0)


SETUP_FAMILIES: dict[str, type[_Family]] = {
    "session_sweep_rev": SessionSweepReversal,
    "pd_sweep_rev": PrevDaySweepReversal,
    "bias_pullback": BiasAlignedPullback,
    "or_context": OpeningRangeContext,
    "failed_brk_trap": FailedBreakoutTrap,
}
