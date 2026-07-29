"""Phase 23 — PIT-safe higher-timeframe points of interest (POI) for the HTF-POI→LTF-reclaim model.

A POI is a higher-timeframe level price reaches before the lower-timeframe confirmation fires. We
build the POI set from confirmed-and-past HTF liquidity only:

- 1H / 4H rolling swing extremes (12 / 48 prior 5m bars, shifted -> no current-bar lookahead),
- prior day / week high & low, overnight high & low (already PIT-safe upstream),

then expose, per 5m bar, the nearest support POI (highest level at/below price) and resistance POI
(lowest level at/above price), and a "traded into and reclaimed the POI" touch flag within a
tolerance. Everything uses only current-and-past data, so a signal at bar t never reads bar t+1.
"""

from __future__ import annotations

import polars as pl

_NEG_INF = pl.lit(float("-inf"))
_POS_INF = pl.lit(float("inf"))

# HTF POI level columns the model reasons over (support = below price, resistance = above).
_SUPPORT_LEVELS: tuple[str, ...] = ("_h1h_lo", "_h4h_lo", "prev_day_low", "prev_week_low",
                                    "overnight_low")
_RESIST_LEVELS: tuple[str, ...] = ("_h1h_hi", "_h4h_hi", "prev_day_high", "prev_week_high",
                                   "overnight_high")


def add_htf_poi(frame: pl.DataFrame, *, tol_pts: float = 5.0, window: int = 6) -> pl.DataFrame:
    """Add HTF POI levels + nearest support/resistance POI + reclaim-touch flags (PIT-safe).

    Also adds ``at_support_poi_recent`` / ``at_resistance_poi_recent`` — the POI was touched within
    the last ``window`` bars — so the LTF confirmation (MSS/displacement) can fire a few bars AFTER
    the touch, modelling the real sequence rather than demanding everything on one bar.
    """
    f = frame.with_columns(
        pl.col("high").rolling_max(window_size=12).shift(1).alias("_h1h_hi"),
        pl.col("low").rolling_min(window_size=12).shift(1).alias("_h1h_lo"),
        pl.col("high").rolling_max(window_size=48).shift(1).alias("_h4h_hi"),
        pl.col("low").rolling_min(window_size=48).shift(1).alias("_h4h_lo"),
    )
    close = pl.col("close")
    # Nearest support POI = highest HTF level at or below the close; resistance = lowest at/above.
    sup_cands = [
        pl.when(pl.col(c).is_not_null() & (pl.col(c) <= close)).then(pl.col(c)).otherwise(_NEG_INF)
        for c in _SUPPORT_LEVELS
    ]
    res_cands = [
        pl.when(pl.col(c).is_not_null() & (pl.col(c) >= close)).then(pl.col(c)).otherwise(_POS_INF)
        for c in _RESIST_LEVELS
    ]
    f = f.with_columns(
        pl.max_horizontal(sup_cands).alias("support_poi"),
        pl.min_horizontal(res_cands).alias("resistance_poi"),
    )
    f = f.with_columns(
        (close - pl.col("support_poi")).alias("poi_dist_support"),
        (pl.col("resistance_poi") - close).alias("poi_dist_resistance"),
        # Traded INTO the support POI (bar low reached it within tol) AND reclaimed (closed above).
        (
            pl.col("support_poi").is_finite()
            & (pl.col("low") <= pl.col("support_poi") + tol_pts)
            & (pl.col("close") > pl.col("support_poi"))
        ).cast(pl.Int8).alias("at_support_poi"),
        (
            pl.col("resistance_poi").is_finite()
            & (pl.col("high") >= pl.col("resistance_poi") - tol_pts)
            & (pl.col("close") < pl.col("resistance_poi"))
        ).cast(pl.Int8).alias("at_resistance_poi"),
    )
    # POI touched within the last `window` bars (within the trading day) -> confirmation may follow.
    return f.with_columns(
        (pl.col("at_support_poi").rolling_max(window_size=window).over("trading_day") == 1)
        .cast(pl.Int8).alias("at_support_poi_recent"),
        (pl.col("at_resistance_poi").rolling_max(window_size=window).over("trading_day") == 1)
        .cast(pl.Int8).alias("at_resistance_poi_recent"),
    )
