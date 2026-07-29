"""Phase 19 — edge-map engine: forward excursions, context buckets, edge aggregation."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl

from nqlab.discovery.edgemap import (
    bucket_edge,
    context_columns,
    forward_excursions,
    join_stability,
)


def _frame(closes: list[float], *, day: date = date(2024, 1, 2)) -> pl.DataFrame:
    rows = []
    t0 = datetime(day.year, day.month, day.day, 9, 30)
    for i, c in enumerate(closes):
        rows.append({
            "ts": t0 + timedelta(minutes=5 * i), "trading_day": day,
            "minute_of_day": 570 + 5 * i, "is_rth": True,
            "open": c, "high": c + 2, "low": c - 2, "close": c,
        })
    return pl.DataFrame(rows)


def test_forward_excursions_long_perspective_within_day() -> None:
    # Rising path: forward returns + MFE positive, MAE small; never crosses day boundary.
    f = _frame([100.0, 102.0, 104.0, 106.0, 108.0])
    out = forward_excursions(f, horizons={"10m": 2})
    r0 = out.row(0, named=True)
    assert r0["fwd_10m"] == 104.0 - 100.0  # close[t+2] - close
    assert r0["mfe_10m"] == 104.0 + 2 - 100.0  # best high over next 2 bars - close
    # Last two rows cannot look 2 bars forward within the day -> null.
    assert out["fwd_10m"][-1] is None
    assert out["fwd_10m"][-2] is None


def test_context_columns_derives_buckets() -> None:
    f = _frame([100.0, 101.0]).with_columns(
        pl.lit(0.5).alias("dist_vwap"), pl.lit(1).alias("trend_4h_bull"),
        pl.lit(0).alias("trend_1h_bull"), pl.lit(1).alias("gap_dir"),
        pl.lit(1).alias("in_premium"), pl.lit(0).alias("in_discount"),
        pl.lit(0).alias("vol_regime"), pl.lit("range").alias("prev_day_type"),
        pl.lit(1).alias("range_expansion"), pl.lit(0).alias("compression_setup"),
    )
    c = context_columns(f).row(0, named=True)
    assert c["tod"] == "0930-0945"
    assert c["vwap_loc"] == "above"
    assert c["bias4h"] == "bull"
    assert c["gap"] == "gap_up"
    assert c["pd_loc"] == "premium"
    assert c["regime"] == "expansion"


def test_bucket_edge_flags_direction_and_after_cost() -> None:
    # Two buckets: 'A' drifts up strongly (long edge), 'B' is flat (no edge after cost).
    rows = []
    for i in range(200):
        rows.append({"bucket": "A", "fwd_30m": 5.0, "mfe_30m": 8.0, "mae_30m": 2.0})
        rows.append({"bucket": "B", "fwd_30m": 0.1, "mfe_30m": 6.0, "mae_30m": 6.0})
    df = pl.DataFrame(rows)
    g = bucket_edge(df, ["bucket"], horizon="30m", cost_pts=0.75)
    by = {r["bucket"]: r for r in g.iter_rows(named=True)}
    assert by["A"]["direction"] == "long"
    assert by["A"]["after_cost"] > 0  # 5.0 - 0.75
    assert by["A"]["mfe_mae"] > 1.0
    assert by["B"]["after_cost"] < 0  # 0.1 - 0.75 -> negative (no edge)
    assert abs(by["B"]["mfe_mae"] - 1.0) < 1e-9


def test_join_stability_requires_same_direction_and_positive_both() -> None:
    cols = ["bucket"]
    train = pl.DataFrame([{"bucket": "A", "n": 500, "direction": "long", "gross_edge": 3.0,
                           "after_cost": 2.0, "mfe_mae": 1.4, "dir_acc": 0.6, "p_mfe10": 0.4}])
    val_ok = pl.DataFrame([{"bucket": "A", "n": 100, "direction": "long", "gross_edge": 2.0,
                            "after_cost": 1.2, "mfe_mae": 1.3, "dir_acc": 0.58}])
    j = join_stability(train, val_ok, cols)
    assert j.row(0, named=True)["stable_edge"] is True
    val_flip = val_ok.with_columns(pl.lit("short").alias("direction"))
    j2 = join_stability(train, val_flip, cols)
    assert j2.row(0, named=True)["stable_edge"] is False  # direction flips -> not stable
