"""Phase 19 — NQ NY-AM edge-discovery map (market behaviour, NOT a strategy backtest).

For every NY-AM bar we measure the *forward* outcome (signed return, max-favourable and
max-adverse excursion) over several horizons, then bucket bars by market context (time-of-day,
HTF bias, VWAP / premium-discount location, day type, overnight location, volatility regime) and
aggregate. A context only has a tradable edge if, before any strategy is built, it shows a
positive gross edge before costs, MFE clearly greater than MAE for one direction, enough sample,
and the same edge in the 2024 validation year.

Nothing here touches the 2025-2026 lockbox, optimises parameters, or executes trades — it maps
where NQ actually behaves directionally, so a later phase can build only edges that exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

# Round-trip cost in NQ points: 1-tick slippage/side (0.25*2=0.5) + $0.85/side commission
# (=$1.70 = 0.085 pt at $20/pt). ~0.6 pt; we use a conservative 0.75 pt for the after-cost edge.
COST_PTS = 0.75
HORIZON_BARS_5M: dict[str, int] = {"5m": 1, "10m": 2, "15m": 3, "30m": 6, "60m": 12}
MFE_TARGETS: tuple[float, ...] = (6.0, 10.0, 15.0, 20.0)

_TOD_WINDOWS: tuple[tuple[str, int, int], ...] = (
    ("0930-0945", 570, 585),
    ("0945-1000", 585, 600),
    ("1000-1030", 600, 630),
    ("1030-1100", 630, 660),
    ("1100-1130", 660, 690),
    ("1130-1200", 690, 720),
    ("1200-1230", 720, 750),
)


def forward_excursions(frame: pl.DataFrame, *, horizons: dict[str, int]) -> pl.DataFrame:
    """Add per-horizon forward return + long-perspective MFE/MAE (points), within trading day.

    For a LONG entry at the bar close: ``mfe`` = best high over the next h bars minus close;
    ``mae`` = close minus worst low over the next h bars; ``fwd`` = close[t+h] - close. A SHORT's
    MFE/MAE are this MAE/MFE (symmetric), so one long-perspective pass covers both directions.
    Forward windows never cross the trading day (no overnight leakage).
    """
    exprs: list[pl.Expr] = []
    for label, h in horizons.items():
        highs = [pl.col("high").shift(-k).over("trading_day") for k in range(1, h + 1)]
        lows = [pl.col("low").shift(-k).over("trading_day") for k in range(1, h + 1)]
        exprs.append((pl.max_horizontal(highs) - pl.col("close")).alias(f"mfe_{label}"))
        exprs.append((pl.col("close") - pl.min_horizontal(lows)).alias(f"mae_{label}"))
        exprs.append(
            (pl.col("close").shift(-h).over("trading_day") - pl.col("close")).alias(f"fwd_{label}")
        )
    return frame.with_columns(exprs)


def _tod_expr() -> pl.Expr:
    e = pl.when(pl.lit(value=False)).then(pl.lit("?"))
    for label, lo, hi in _TOD_WINDOWS:
        e = e.when((pl.col("minute_of_day") >= lo) & (pl.col("minute_of_day") < hi)).then(
            pl.lit(label)
        )
    return e.otherwise(pl.lit("other")).alias("tod")


def context_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Derive discrete context-bucket columns from the PIT feature layer."""
    pd_loc = (
        pl.when(pl.col("in_premium") == 1).then(pl.lit("premium"))
        .when(pl.col("in_discount") == 1).then(pl.lit("discount"))
        .otherwise(pl.lit("mid")).alias("pd_loc")
    )
    regime = (
        pl.when(pl.col("range_expansion") == 1).then(pl.lit("expansion"))
        .when(pl.col("compression_setup") == 1).then(pl.lit("compression"))
        .otherwise(pl.lit("normal")).alias("regime")
    )
    return frame.with_columns(
        _tod_expr(),
        pl.when(pl.col("dist_vwap") >= 0).then(pl.lit("above")).otherwise(pl.lit("below")).alias(
            "vwap_loc"
        ),
        pl.when(pl.col("trend_4h_bull") == 1).then(pl.lit("bull")).otherwise(pl.lit("bear")).alias(
            "bias4h"
        ),
        pl.when(pl.col("trend_1h_bull") == 1).then(pl.lit("bull")).otherwise(pl.lit("bear")).alias(
            "bias1h"
        ),
        pl.when(pl.col("gap_dir") > 0).then(pl.lit("gap_up"))
        .when(pl.col("gap_dir") < 0).then(pl.lit("gap_dn")).otherwise(pl.lit("flat")).alias("gap"),
        pl.when(pl.col("vol_regime") == 1).then(pl.lit("hi_vol")).otherwise(pl.lit("lo_vol")).alias(
            "vol"
        ),
        pl.col("prev_day_type").alias("ptype"),
        pd_loc,
        regime,
    )


@dataclass(frozen=True)
class EdgeRow:
    bucket: str
    n: int
    direction: str  # long | short | none
    gross_edge: float  # points, in the bucket's dominant direction
    after_cost: float
    mfe: float
    mae: float
    mfe_mae: float
    dir_acc: float
    p_mfe10: float
    median_fwd: float


def bucket_edge(
    frame: pl.DataFrame, by: list[str], *, horizon: str = "30m", cost_pts: float = COST_PTS
) -> pl.DataFrame:
    """Aggregate forward outcomes per context bucket for one horizon (long perspective)."""
    fwd, mfe, mae = f"fwd_{horizon}", f"mfe_{horizon}", f"mae_{horizon}"
    g = (
        frame.drop_nulls([fwd, mfe, mae])
        .group_by(by)
        .agg(
            pl.len().alias("n"),
            pl.col(fwd).mean().alias("mean_fwd"),
            pl.col(fwd).median().alias("median_fwd"),
            pl.col(mfe).mean().alias("mfe"),
            pl.col(mae).mean().alias("mae"),
            (pl.col(fwd) > 0).mean().alias("dir_acc"),
            *[(pl.col(mfe) >= t).mean().alias(f"p_mfe{int(t)}") for t in MFE_TARGETS],
            *[(pl.col(mae) >= t).mean().alias(f"p_mae{int(t)}") for t in MFE_TARGETS],
            pl.col(mae).quantile(0.75).alias("mae_p75"),
            pl.col(mfe).quantile(0.50).alias("mfe_p50"),
        )
        .with_columns(
            (pl.col("mfe") / pl.col("mae")).alias("mfe_mae"),
            # Dominant direction = sign of mean forward return; edge = |mean_fwd| - cost.
            pl.when(pl.col("mean_fwd") >= 0).then(pl.lit("long")).otherwise(pl.lit("short")).alias(
                "direction"
            ),
            (pl.col("mean_fwd").abs() - cost_pts).alias("after_cost"),
            pl.col("mean_fwd").abs().alias("gross_edge"),
        )
    )
    return g.sort("n", descending=True)


def join_stability(train: pl.DataFrame, val: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Join train + validation bucket stats; flag buckets whose edge holds in 2024."""
    t = train.select([*by, "n", "direction", "gross_edge", "after_cost", "mfe_mae", "dir_acc",
                       "p_mfe10"])
    v = val.select([*by, "n", "direction", "gross_edge", "after_cost", "mfe_mae", "dir_acc"])
    j = t.join(v, on=by, how="inner", suffix="_val")
    return j.with_columns(
        (
            (pl.col("direction") == pl.col("direction_val"))
            & (pl.col("after_cost") > 0)
            & (pl.col("after_cost_val") > 0)
            & (pl.col("n") >= 100)
            & (pl.col("n_val") >= 30)
        ).alias("stable_edge")
    )
