"""Phase 32 — conditional edge map: which PIT-safe NQ market states produce asymmetric movement.

Conditions are boolean expressions over PIT-safe feature columns ONLY (never ``fwd_*``). For each
condition + direction we aggregate the forward outcomes (from ``nqlab.research.forward``) to score
how cleanly NQ moves afterward: MFE/MAE, P(+T before -T), mean/median return, clean-path & chop
scores, a standardized 20pt-1:1 after-cost expectancy, and year/month stability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import polars as pl

_C = pl.col

# Standardized 20pt 1:1 economics (1 MNQ, audited costs): win +20pt=$40-$1.00; loss -$40-$1.50.
WIN20, LOSS20 = 39.0, -41.50


def add_regime_buckets(frame: pl.DataFrame) -> pl.DataFrame:
    """PIT-safe volatility-regime buckets via TRAILING atr quantiles + helper labels."""
    w = 2000
    frame = frame.with_columns(
        _C("atr").rolling_quantile(quantile=0.2, window_size=w, min_samples=500).alias("_q20"),
        _C("atr").rolling_quantile(quantile=0.4, window_size=w, min_samples=500).alias("_q40"),
        _C("atr").rolling_quantile(quantile=0.6, window_size=w, min_samples=500).alias("_q60"),
        _C("atr").rolling_quantile(quantile=0.8, window_size=w, min_samples=500).alias("_q80"),
    )
    atr = _C("atr")
    bucket = (
        pl.when(atr <= _C("_q20"))
        .then(0)
        .when(atr <= _C("_q40"))
        .then(1)
        .when(atr <= _C("_q60"))
        .then(2)
        .when(atr <= _C("_q80"))
        .then(3)
        .otherwise(4)
    )
    prev_loc = (_C("prev_day_close") - _C("prev_day_low")) / (
        _C("prev_day_high") - _C("prev_day_low")
    )
    return frame.with_columns(
        bucket.alias("atr_bucket"),
        prev_loc.alias("prev_close_loc"),
        ((_C("or_high") - _C("or_low")) / _C("atr")).alias("or_range_atr"),
    )


@dataclass(frozen=True)
class Condition:
    name: str
    bucket: str
    direction: str  # long | short | both
    expr: pl.Expr
    requires: tuple[str, ...] = ()


def _c(
    name: str, bucket: str, direction: str, expr: pl.Expr, requires: tuple[str, ...] = ()
) -> Condition:
    return Condition(name, bucket, direction, expr, requires)


def _window(lo: int, hi: int) -> pl.Expr:
    return (_C("minute_of_day") >= lo) & (_C("minute_of_day") < hi)


def conditions() -> list[Condition]:
    rth = _C("is_rth")
    conds: list[Condition] = [
        # --- A. Time of day (both) ---
        _c("time_open_0930_0945", "time", "both", _window(570, 585)),
        _c("time_0945_1000", "time", "both", _window(585, 600)),
        _c("time_1000_1030", "time", "both", _window(600, 630)),
        _c("time_1030_1100", "time", "both", _window(630, 660)),
        _c("time_1100_1130", "time", "both", _window(660, 690)),
        _c("time_1130_1200", "time", "both", _window(690, 720)),
        _c("time_lunch_1200_1330", "time", "both", _window(720, 810)),
        _c("time_nypm_1330_1500", "time", "both", _window(810, 900)),
        _c("time_lasthour_1500_1600", "time", "both", _window(900, 960)),
        _c("time_premkt_0700_0930", "time", "both", _window(420, 570)),
        # --- B. Prior-day context (apply to current RTH) ---
        _c("pday_trend_up", "priorday", "long", rth & (_C("prev_day_type") == "trend_up")),
        _c("pday_trend_down", "priorday", "short", rth & (_C("prev_day_type") == "trend_down")),
        _c("pday_range", "priorday", "both", rth & (_C("prev_day_type") == "range")),
        _c("pday_compression", "priorday", "both", rth & (_C("prev_day_type") == "compression")),
        _c("pday_expansion", "priorday", "both", rth & (_C("prev_day_type") == "expansion")),
        _c("pday_inside", "priorday", "both", rth & (_C("prev_inside_day") == 1)),
        _c("pday_outside", "priorday", "both", rth & (_C("prev_outside_day") == 1)),
        _c("pday_narrowrange", "priorday", "both", rth & (_C("prev_nr_day") == 1)),
        _c("pday_wuderange", "priorday", "both", rth & (_C("prev_wr_day") == 1)),
        _c("pday_close_near_high", "priorday", "long", rth & (_C("prev_close_loc") > 0.8)),
        _c("pday_close_near_low", "priorday", "short", rth & (_C("prev_close_loc") < 0.2)),
        _c(
            "pday_close_mid",
            "priorday",
            "both",
            rth & (_C("prev_close_loc") >= 0.4) & (_C("prev_close_loc") <= 0.6),
        ),
        # --- C. Gap / open (first 2h) ---
        _c("gap_up", "gap", "both", _window(570, 690) & (_C("gap_dir") == 1)),
        _c("gap_down", "gap", "both", _window(570, 690) & (_C("gap_dir") == -1)),
        _c(
            "gap_up_unfilled",
            "gap",
            "both",
            _window(570, 690) & (_C("gap_dir") == 1) & (_C("gap_filled") == 0),
        ),
        _c(
            "gap_down_unfilled",
            "gap",
            "both",
            _window(570, 690) & (_C("gap_dir") == -1) & (_C("gap_filled") == 0),
        ),
        _c("open_above_value", "gap", "both", _window(570, 660) & (_C("vp_open_loc") == 1)),
        _c("open_below_value", "gap", "both", _window(570, 660) & (_C("vp_open_loc") == -1)),
        _c("open_inside_value", "gap", "both", _window(570, 660) & (_C("vp_open_loc") == 0)),
        # --- D. Volume / market profile ---
        _c("vp_above_value", "vp", "both", rth & (_C("vp_pos") == 1)),
        _c("vp_below_value", "vp", "both", rth & (_C("vp_pos") == -1)),
        _c("vp_inside_value", "vp", "both", rth & (_C("vp_pos") == 0)),
        _c("vp_above_poc", "vp", "long", rth & (_C("vp_d_poc") > 0)),
        _c("vp_below_poc", "vp", "short", rth & (_C("vp_d_poc") < 0)),
        _c("vp_near_vah", "vp", "both", rth & ((_C("vp_d_vah").abs() / _C("atr")) < 0.5)),
        _c("vp_near_val", "vp", "both", rth & ((_C("vp_d_val").abs() / _C("atr")) < 0.5)),
        _c("vp_vah_break", "vp", "long", rth & (_C("vp_vah_break_up") == 1)),
        _c("vp_val_break", "vp", "short", rth & (_C("vp_val_break_dn") == 1)),
        _c("vp_val_reclaim", "vp", "long", rth & (_C("vp_val_reclaim_up") == 1)),
        _c("vp_vah_reject", "vp", "short", rth & (_C("vp_vah_reject_dn") == 1)),
        _c("vp_poc_reclaim_up", "vp", "long", rth & (_C("vp_poc_reclaim_up") == 1)),
        _c("vp_devpoc_rising", "vp", "long", rth & (_C("vp_dev_poc_shift") > 0)),
        _c(
            "vp_clean_path_up",
            "vp",
            "long",
            rth & (_C("vp_dist_hvn_up_atr").is_null() | (_C("vp_dist_hvn_up_atr") > 1.5)),
        ),
        _c(
            "vp_hvn_blocking_up",
            "vp",
            "long",
            rth & _C("vp_dist_hvn_up_atr").is_not_null() & (_C("vp_dist_hvn_up_atr") < 1.0),
        ),
        # --- E. Volatility regime ---
        _c("vol_atr_q0_20", "vol", "both", rth & (_C("atr_bucket") == 0)),
        _c("vol_atr_q20_40", "vol", "both", rth & (_C("atr_bucket") == 1)),
        _c("vol_atr_q40_60", "vol", "both", rth & (_C("atr_bucket") == 2)),
        _c("vol_atr_q60_80", "vol", "both", rth & (_C("atr_bucket") == 3)),
        _c("vol_atr_q80_100", "vol", "both", rth & (_C("atr_bucket") == 4)),
        _c("vol_compression", "vol", "both", rth & (_C("compression_setup") == 1)),
        _c(
            "vol_range_expand_up",
            "vol",
            "long",
            rth & (_C("range_expansion") == 1) & (_C("close") > _C("open")),
        ),
        _c(
            "vol_range_expand_dn",
            "vol",
            "short",
            rth & (_C("range_expansion") == 1) & (_C("close") < _C("open")),
        ),
        _c("vol_or_wide", "vol", "both", rth & (_C("or_range_atr") > 4.0)),
        _c("vol_or_narrow", "vol", "both", rth & (_C("or_range_atr") < 2.0)),
        # --- F. Trend / structure ---
        _c("trend_daily_up", "trend", "long", rth & (_C("htf_bull") == 1)),
        _c("trend_daily_dn", "trend", "short", rth & (_C("htf_bull") == 0)),
        _c("trend_1h_up", "trend", "long", rth & (_C("trend_1h_bull") == 1)),
        _c("trend_4h_up", "trend", "long", rth & (_C("trend_4h_bull") == 1)),
        _c(
            "trend_mtf_align_up",
            "trend",
            "long",
            rth & (_C("htf_bull") == 1) & (_C("trend_1h_bull") == 1) & (_C("trend_4h_bull") == 1),
        ),
        _c("trend_above_vwap", "trend", "long", rth & (_C("close") > _C("vwap_session"))),
        _c("trend_below_vwap", "trend", "short", rth & (_C("close") < _C("vwap_session"))),
        _c(
            "trend_ema_stack_up",
            "trend",
            "long",
            rth & (_C("ema9") > _C("ema21")) & (_C("ema21") > _C("ema50")),
        ),
        _c(
            "trend_adx_up",
            "trend",
            "long",
            rth & (_C("adx") > 25) & (_C("plus_di") > _C("minus_di")),
        ),
        _c("trend_structdir_up", "trend", "long", rth & (_C("struct_dir") > 0)),
        # --- G. Liquidity / sweep ---
        _c(
            "sweep_low_reclaim",
            "sweep",
            "long",
            rth & (_C("swept_low") == 1) & (_C("reclaim_up") == 1),
        ),
        _c(
            "sweep_high_reclaim",
            "sweep",
            "short",
            rth & (_C("swept_high") == 1) & (_C("reclaim_dn") == 1),
        ),
        _c(
            "sweep_low_noreclaim",
            "sweep",
            "short",
            rth & (_C("swept_low") == 1) & (_C("reclaim_up") == 0),
        ),
        _c(
            "sweep_pdl",
            "sweep",
            "long",
            rth & (_C("low") < _C("prev_day_low")) & (_C("close") > _C("prev_day_low")),
        ),
        _c(
            "sweep_pdh",
            "sweep",
            "short",
            rth & (_C("high") > _C("prev_day_high")) & (_C("close") < _C("prev_day_high")),
        ),
        _c(
            "sweep_onl",
            "sweep",
            "long",
            rth & (_C("low") < _C("overnight_low")) & (_C("close") > _C("overnight_low")),
        ),
        _c(
            "sweep_onh",
            "sweep",
            "short",
            rth & (_C("high") > _C("overnight_high")) & (_C("close") < _C("overnight_high")),
        ),
        _c(
            "displacement_up",
            "sweep",
            "long",
            rth & (_C("displacement") == 1) & (_C("close") > _C("open")),
        ),
        _c("ifvg_bull", "sweep", "long", rth & (_C("ifvg_bull") == 1)),
        _c("mss_up", "sweep", "long", rth & (_C("mss_up") == 1)),
        # --- H. Opening range (after OR formed) ---
        _c(
            "or_break_up",
            "or",
            "long",
            _window(600, 900)
            & (_C("close") > _C("or_high"))
            & (_C("close").shift(1) <= _C("or_high")),
        ),
        _c(
            "or_break_dn",
            "or",
            "short",
            _window(600, 900)
            & (_C("close") < _C("or_low"))
            & (_C("close").shift(1) >= _C("or_low")),
        ),
        _c(
            "or_failed_up",
            "or",
            "short",
            _window(600, 900) & (_C("high") > _C("or_high")) & (_C("close") < _C("or_high")),
        ),
        _c(
            "or_failed_dn",
            "or",
            "long",
            _window(600, 900) & (_C("low") < _C("or_low")) & (_C("close") > _C("or_low")),
        ),
    ]
    return conds


def _panel(sub: pl.DataFrame, direction: str) -> dict[str, Any]:
    """Forward-outcome panel for a condition subset in a given direction."""
    n = sub.height
    if n == 0:
        return {"n": 0}
    fav60, adv60 = ("fwd_up_60", "fwd_dn_60") if direction == "long" else ("fwd_dn_60", "fwd_up_60")
    fav_e, adv_e = (
        ("fwd_up_eod", "fwd_dn_eod") if direction == "long" else ("fwd_dn_eod", "fwd_up_eod")
    )
    tcols = {t: (f"fwd_L{t}" if direction == "long" else f"fwd_S{t}") for t in (10, 20, 30, 50)}
    heat = "fwd_Lheat20" if direction == "long" else "fwd_Sheat20"
    ret60 = _C("fwd_ret_60") * (1 if direction == "long" else -1)
    agg = sub.select(
        pl.col(fav60).mean().alias("mfe60"),
        pl.col(adv60).mean().alias("mae60"),
        pl.col(fav_e).mean().alias("mfe_eod"),
        pl.col(adv_e).mean().alias("mae_eod"),
        *[pl.col(c).mean().alias(f"p{t}") for t, c in tcols.items()],
        ret60.mean().alias("ret60_mean"),
        ret60.median().alias("ret60_med"),
        (ret60 > 0).mean().alias("dir_acc"),
        # clean path (T=20): win weighted by (1 - heat/20). heat is NaN on non-winners -> fill_nan.
        (pl.col(tcols[20]) * (1 - (pl.col(heat).fill_nan(0.0) / 20).clip(0, 1)))
        .mean()
        .alias("clean"),
        # chop: both sides reach 20 by EOD
        ((pl.col("fwd_up_eod") >= 20) & (pl.col("fwd_dn_eod") >= 20)).mean().alias("chop"),
    ).row(0, named=True)
    p20 = agg["p20"] or 0.0
    exp20 = p20 * WIN20 + (1 - p20) * LOSS20
    # year stability: P(+20 before -20) per year
    yr = (
        sub.with_columns(_C("entry_ts").dt.year().alias("_y"))
        if "entry_ts" in sub.columns
        else sub.with_columns(_C("ts").dt.year().alias("_y"))
    )
    yp = yr.group_by("_y").agg(pl.col(tcols[20]).mean().alias("p"), pl.len().alias("n")).sort("_y")
    yps = {int(r["_y"]): round(r["p"], 3) for r in yp.to_dicts()}
    years_pos = sum(1 for v in yps.values() if v * WIN20 + (1 - v) * LOSS20 > 0)
    # month stability
    mp = (
        sub.with_columns(_C("ts").dt.strftime("%Y-%m").alias("_m"))
        .group_by("_m")
        .agg(pl.col(tcols[20]).mean().alias("p"))
    )
    months_pos = sum(
        1 for r in mp.to_dicts() if (r["p"] or 0) * WIN20 + (1 - (r["p"] or 0)) * LOSS20 > 0
    )
    n_months = mp.height
    mfe_mae = (agg["mfe60"] / agg["mae60"]) if agg["mae60"] and agg["mae60"] > 0 else None
    return {
        "n": n,
        "mfe60": round(agg["mfe60"], 2),
        "mae60": round(agg["mae60"], 2),
        "mfe_mae": round(mfe_mae, 3) if mfe_mae else None,
        "mfe_eod": round(agg["mfe_eod"], 1),
        "mae_eod": round(agg["mae_eod"], 1),
        "p10": round(agg["p10"], 4),
        "p20": round(p20, 4),
        "p30": round(agg["p30"], 4),
        "p50": round(agg["p50"], 4),
        "ret60_mean": round(agg["ret60_mean"], 2),
        "ret60_med": round(agg["ret60_med"], 2),
        "dir_acc": round(agg["dir_acc"], 4),
        "clean": round(agg["clean"], 4),
        "chop": round(agg["chop"], 4),
        "exp20_cost": round(exp20, 2),
        "years_pos": years_pos,
        "years_n": len(yps),
        "months_pos_pct": round(months_pos / n_months, 3) if n_months else None,
        "yearly_p20": json.dumps(yps),
        "sample_flag": (
            "robust"
            if n >= 500
            else "strong"
            if n >= 200
            else "usable"
            if n >= 100
            else "weak"
            if n >= 50
            else "tiny"
        ),
    }


def evaluate(frame: pl.DataFrame, conds: list[Condition]) -> list[dict[str, Any]]:
    valid = frame.filter(_C("fwd_valid") == 1)
    rows: list[dict[str, Any]] = []
    for cond in conds:
        if any(r not in frame.columns for r in cond.requires):
            continue
        sub = valid.filter(cond.expr.fill_null(value=False))
        dirs = ("long", "short") if cond.direction == "both" else (cond.direction,)
        for d in dirs:
            p = _panel(sub, d)
            if p.get("n", 0) == 0:
                continue
            rows.append({"condition": cond.name, "bucket": cond.bucket, "direction": d, **p})
    return rows
