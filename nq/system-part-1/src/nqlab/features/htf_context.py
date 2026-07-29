"""Phase 9 higher-timeframe context + rule-based day-type classifier (PIT-safe).

Adds the higher-timeframe "narrative" layer that the intraday setups lacked:
prior-day / prior-week / prior-month liquidity levels, the overnight (ETH) range
and where the NY session opens within it, a rule-based classification of the
*completed prior day* (trend / range / reversal / compression / expansion) plus
volatility regime, gap direction and an intraday gap-fill flag, 1h/4h trend bias,
and distance from price to the nearest liquidity level.

Point-in-time contract (the whole reason this is a separate, tested layer):

* Prior-day / week / month aggregates are ``shift``-ed by one completed period,
  so a value attached to every bar of day *t* is knowable at the *start* of day
  *t* (it describes days strictly before *t*).
* The overnight range and NY-open location complete at the RTH open, so they are
  exposed only for bars at/after ``rth_start_min`` (null before).
* The gap-fill flag is a within-day cumulative touch — it only ever looks at the
  current and earlier RTH bars of the same day.
* 1h/4h trend EMAs are trailing.

No expression uses a negative shift, so slicing the series at row ``t`` reproduces
every value at ``t`` (verified by ``tests/unit/test_htf_context_pit.py``).
"""

from __future__ import annotations

import polars as pl

HTF_CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    # prior-period liquidity levels
    "prev_day_open",
    "prev_day_range",
    "prev_week_high",
    "prev_week_low",
    "prev_month_high",
    "prev_month_low",
    # overnight / opening location
    "overnight_high",
    "overnight_low",
    "overnight_range",
    "ny_open_loc",
    # prior-day classification (rule-based, fully completed day)
    "prev_inside_day",
    "prev_outside_day",
    "prev_nr_day",
    "prev_wr_day",
    "prev_day_type",
    # current-day regime flags (knowable at/after the open, cumulative)
    "gap_dir",
    "gap_filled",
    "compression_setup",
    "vol_regime",
    # higher-timeframe trend bias
    "trend_1h_bull",
    "trend_4h_bull",
    # distance to nearest liquidity (raw points; normalise by atr downstream)
    "res_dist",
    "sup_dist",
)

_NEXT_OPEN_MIN = 1080  # 18:00 ET — next session / next trading day starts here
_NR_FRAC = 0.7  # prior range < 0.7 * trailing-median range -> narrow-range day
_WR_FRAC = 1.3  # prior range > 1.3 * trailing-median range -> wide-range day
_RANGE_WINDOW = 20  # trailing window (completed days) for the range median


def _tf_minutes(timeframe: str) -> int:
    """Bar length in minutes from a '1m'/'5m'/'15m'/'1h' label (default 5)."""
    tf = timeframe.strip().lower()
    if tf.endswith("h"):
        return int(float(tf[:-1]) * 60)
    if tf.endswith("m"):
        return int(float(tf[:-1]))
    return 5


def _day_type_expr() -> pl.Expr:
    """Classify the *prior completed day* from its shifted aggregates.

    Reads only ``_p*`` columns (already ``shift(1)``-ed onto the daily series), so
    the label is knowable at the start of the current day. Priority: reversal >
    trend > expansion > compression(range) > neutral.
    """
    cir = pl.col("_p_cir")  # prior-day close location in its range (0=low, 1=high)
    rr = pl.col("_p_range_ratio")  # prior-day range / trailing median range
    return (
        pl.when(pl.col("prev_outside_day") == 1)
        .then(pl.lit("reversal"))
        .when((cir > 0.66) & (rr >= 1.0))
        .then(pl.lit("trend_up"))
        .when((cir < 0.34) & (rr >= 1.0))
        .then(pl.lit("trend_down"))
        .when(pl.col("prev_wr_day") == 1)
        .then(pl.lit("expansion"))
        .when((pl.col("prev_nr_day") == 1) | (rr < _NR_FRAC))
        .then(pl.lit("compression"))
        .when(rr < 1.0)
        .then(pl.lit("range"))
        .otherwise(pl.lit("neutral"))
    )


def _daily_context(df: pl.DataFrame, rth_start_min: int) -> pl.DataFrame:
    """One row per trading day with prior-period levels + day-type classification."""
    daily = (
        df.group_by("trading_day")
        .agg(
            pl.col("high").max().alias("_dh"),
            pl.col("low").min().alias("_dl"),
            pl.col("open").filter(pl.col("is_rth")).first().alias("_ro"),
            pl.col("close").filter(pl.col("is_rth")).last().alias("_rc"),
        )
        .sort("trading_day")
    )
    # Monotonic Monday-aligned week buckets and calendar-month buckets (epoch day 0
    # was a Thursday; Monday is epoch day 4, hence the -4 offset).
    day_int = pl.col("trading_day").cast(pl.Int32)
    daily = daily.with_columns(
        ((day_int - 4) / 7).floor().cast(pl.Int64).alias("_week_id"),
        (pl.col("trading_day").dt.year() * 12 + pl.col("trading_day").dt.month()).alias(
            "_month_id"
        ),
        (pl.col("_dh") - pl.col("_dl")).alias("_range"),
    )

    # Prior-completed-period weekly / monthly extremes, mapped back per day.
    weekly = (
        daily.group_by("_week_id")
        .agg(pl.col("_dh").max().alias("_wh"), pl.col("_dl").min().alias("_wl"))
        .sort("_week_id")
        .with_columns(
            pl.col("_wh").shift(1).alias("prev_week_high"),
            pl.col("_wl").shift(1).alias("prev_week_low"),
        )
    )
    monthly = (
        daily.group_by("_month_id")
        .agg(pl.col("_dh").max().alias("_mh"), pl.col("_dl").min().alias("_ml"))
        .sort("_month_id")
        .with_columns(
            pl.col("_mh").shift(1).alias("prev_month_high"),
            pl.col("_ml").shift(1).alias("prev_month_low"),
        )
    )

    range_med = pl.col("_range").rolling_median(window_size=_RANGE_WINDOW)
    daily = daily.with_columns(
        pl.col("_ro").shift(1).alias("prev_day_open"),
        pl.col("_dh").shift(1).alias("_pdh"),
        pl.col("_dl").shift(1).alias("_pdl"),
        pl.col("_dh").shift(2).alias("_pdh2"),
        pl.col("_dl").shift(2).alias("_pdl2"),
        pl.col("_range").shift(1).alias("prev_day_range"),
        range_med.shift(1).alias("_p_range_med"),
        (((pl.col("_rc") - pl.col("_dl")) / pl.col("_range")).shift(1)).alias("_p_cir"),
    )
    daily = daily.with_columns(
        ((pl.col("_pdh") <= pl.col("_pdh2")) & (pl.col("_pdl") >= pl.col("_pdl2")))
        .cast(pl.Int8)
        .alias("prev_inside_day"),
        ((pl.col("_pdh") > pl.col("_pdh2")) & (pl.col("_pdl") < pl.col("_pdl2")))
        .cast(pl.Int8)
        .alias("prev_outside_day"),
        (pl.col("prev_day_range") < _NR_FRAC * pl.col("_p_range_med"))
        .cast(pl.Int8)
        .alias("prev_nr_day"),
        (pl.col("prev_day_range") > _WR_FRAC * pl.col("_p_range_med"))
        .cast(pl.Int8)
        .alias("prev_wr_day"),
        (pl.col("prev_day_range") / pl.col("_p_range_med")).alias("_p_range_ratio"),
    )
    daily = daily.with_columns(_day_type_expr().alias("prev_day_type"))

    daily = daily.join(
        weekly.select("_week_id", "prev_week_high", "prev_week_low"), on="_week_id", how="left"
    )
    daily = daily.join(
        monthly.select("_month_id", "prev_month_high", "prev_month_low"), on="_month_id", how="left"
    )
    keep = [
        "trading_day",
        "prev_day_open",
        "prev_day_range",
        "prev_week_high",
        "prev_week_low",
        "prev_month_high",
        "prev_month_low",
        "prev_inside_day",
        "prev_outside_day",
        "prev_nr_day",
        "prev_wr_day",
        "prev_day_type",
    ]
    return daily.select(keep)


def build_htf_context_features(
    frame: pl.DataFrame, *, rth_start_min: int = 570, timeframe: str = "5m"
) -> pl.DataFrame:
    """Add Phase-9 HTF-context columns. Requires calendar + ``realized_vol``; reuses
    ``prev_day_high/low``, ``prev_day_close``, ``gap``, ``vwap_session`` when present."""
    df = frame.sort("ts")

    df = df.join(_daily_context(df, rth_start_min), on="trading_day", how="left")

    # Overnight (pre-RTH ETH) high/low per trading day, exposed only at/after the open.
    pre_rth = (pl.col("minute_of_day") >= _NEXT_OPEN_MIN) | (
        pl.col("minute_of_day") < rth_start_min
    )
    on = (
        df.filter(pre_rth)
        .group_by("trading_day")
        .agg(pl.col("high").max().alias("_onh"), pl.col("low").min().alias("_onl"))
    )
    rth_open = (
        df.filter(pl.col("is_rth"))
        .group_by("trading_day")
        .agg(pl.col("open").first().alias("_ro_today"))
    )
    df = df.join(on, on="trading_day", how="left").join(rth_open, on="trading_day", how="left")
    open_known = pl.col("minute_of_day") >= rth_start_min
    df = df.with_columns(
        pl.when(open_known).then(pl.col("_onh")).alias("overnight_high"),
        pl.when(open_known).then(pl.col("_onl")).alias("overnight_low"),
    )
    df = df.with_columns(
        (pl.col("overnight_high") - pl.col("overnight_low")).alias("overnight_range")
    )
    df = df.with_columns(
        pl.when(open_known & (pl.col("overnight_range") > 0))
        .then((pl.col("_ro_today") - pl.col("overnight_low")) / pl.col("overnight_range"))
        .alias("ny_open_loc"),
    )

    # 1h / 4h trend bias via trailing EMAs scaled to the bar length.
    tfm = max(1, _tf_minutes(timeframe))
    span_1h = max(2, round(60 / tfm))
    span_4h = max(2, round(240 / tfm))
    df = df.with_columns(
        (pl.col("close") > pl.col("close").ewm_mean(span=span_1h, adjust=False))
        .cast(pl.Int8)
        .alias("trend_1h_bull"),
        (pl.col("close") > pl.col("close").ewm_mean(span=span_4h, adjust=False))
        .cast(pl.Int8)
        .alias("trend_4h_bull"),
    )

    # Current-day regime flags. ``gap`` is already exposed only at/after the open.
    has_gap = "gap" in df.columns
    gap_dir = pl.col("gap").sign().cast(pl.Int8) if has_gap else pl.lit(None, dtype=pl.Int8)
    df = df.with_columns(gap_dir.alias("gap_dir"))
    if "prev_day_close" in df.columns:
        touched = (
            open_known
            & (pl.col("low") <= pl.col("prev_day_close"))
            & (pl.col("high") >= pl.col("prev_day_close"))
        ).cast(pl.Int8)
        df = df.with_columns(
            touched.cum_max().over("trading_day").alias("gap_filled"),
        )
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Int8).alias("gap_filled"))

    narrow_on = pl.col("overnight_range") < _NR_FRAC * pl.col("overnight_range").rolling_median(
        window_size=200
    )
    df = df.with_columns(
        ((pl.col("prev_nr_day") == 1) | narrow_on.fill_null(value=False))
        .cast(pl.Int8)
        .alias("compression_setup"),
    )
    # Volatility regime from the trailing realized-vol ratio (1 = high, 0 = low).
    vol_ref = pl.col("realized_vol").rolling_median(window_size=200)
    df = df.with_columns((pl.col("realized_vol") > vol_ref).cast(pl.Int8).alias("vol_regime"))

    # Distance to the nearest liquidity level above (resistance) / below (support).
    levels = [
        "prev_day_high",
        "prev_day_low",
        "prev_week_high",
        "prev_week_low",
        "prev_month_high",
        "prev_month_low",
        "overnight_high",
        "overnight_low",
        "asia_high",
        "asia_low",
        "london_high",
        "london_low",
    ]
    present = [c for c in levels if c in df.columns]
    above = [
        pl.when(pl.col(c) > pl.col("close")).then(pl.col(c) - pl.col("close")).otherwise(None)
        for c in present
    ]
    below = [
        pl.when(pl.col(c) < pl.col("close")).then(pl.col("close") - pl.col(c)).otherwise(None)
        for c in present
    ]
    df = df.with_columns(
        (pl.min_horizontal(*above) if above else pl.lit(None)).alias("res_dist"),
        (pl.min_horizontal(*below) if below else pl.lit(None)).alias("sup_dist"),
    )

    drop = [c for c in ("_onh", "_onl", "_ro_today") if c in df.columns]
    return df.drop(drop).sort("ts")
