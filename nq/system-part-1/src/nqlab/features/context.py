"""Contextual market features (trades-derivable, point-in-time safe).

Adds the market-context layer that simple signals lacked: prior-session ranges
(Asia/London high/low, exposed only after the session closes), previous-day
close + RTH gap (exposed only at/after the RTH open), a higher-timeframe trend
proxy (long EMA), and a volatility regime ratio. Nothing is exposed before it is
knowable, so slicing the series at row t reproduces every value at t.
"""

from __future__ import annotations

import polars as pl

CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    "asia_high", "asia_low", "london_high", "london_low",
    "prev_day_close", "gap", "htf_ema", "htf_bull", "vol_regime_ratio",
)

_ASIA_END_MIN = 180  # 03:00 ET
_LONDON_DEFAULT_END = 570  # 09:30 ET
_NEXT_OPEN_MIN = 1080  # 18:00 ET — start of the next session (and of the next trading day)


def _session_hl(df: pl.DataFrame, session: str, prefix: str, end_min: int) -> pl.DataFrame:
    """Attach a session's high/low per trading day, known only after it closes.

    The window ``[end_min, next_open)`` matters because the Asia session wraps
    midnight: its evening bars have a large minute-of-day, so a simple
    ``minute >= end_min`` would leak the range *during* the session.
    """
    band = (
        df.filter(pl.col("session") == session)
        .group_by("trading_day")
        .agg(pl.col("high").max().alias("_h"), pl.col("low").min().alias("_l"))
    )
    closed = (pl.col("minute_of_day") >= end_min) & (pl.col("minute_of_day") < _NEXT_OPEN_MIN)
    df = df.join(band, on="trading_day", how="left")
    return df.with_columns(
        pl.when(closed).then(pl.col("_h")).alias(f"{prefix}_high"),
        pl.when(closed).then(pl.col("_l")).alias(f"{prefix}_low"),
    ).drop("_h", "_l")


def build_context_features(
    frame: pl.DataFrame, *, rth_start_min: int = _LONDON_DEFAULT_END,
    htf_span: int = 120, vol_window: int = 200,
) -> pl.DataFrame:
    """Add context columns to a feature frame (requires calendar + ``realized_vol``)."""
    df = frame.sort("ts")
    df = _session_hl(df, "ASIA", "asia", _ASIA_END_MIN)
    df = _session_hl(df, "LONDON", "london", rth_start_min)

    daily = (
        df.group_by("trading_day")
        .agg(
            pl.col("close").filter(pl.col("is_rth")).last().alias("_rth_close"),
            pl.col("open").filter(pl.col("is_rth")).first().alias("_rth_open"),
        )
        .sort("trading_day")
        .with_columns(pl.col("_rth_close").shift(1).alias("prev_day_close"))
    )
    df = df.join(
        daily.select("trading_day", "prev_day_close", "_rth_open"), on="trading_day", how="left"
    )
    df = df.with_columns(
        pl.when(pl.col("minute_of_day") >= rth_start_min)
        .then(pl.col("_rth_open") - pl.col("prev_day_close"))
        .alias("gap"),
        pl.col("close").ewm_mean(span=htf_span, adjust=False).alias("htf_ema"),
    )
    df = df.with_columns(
        (pl.col("close") > pl.col("htf_ema")).cast(pl.Int8).alias("htf_bull"),
        (pl.col("realized_vol") / pl.col("realized_vol").rolling_median(window_size=vol_window))
        .alias("vol_regime_ratio"),
    )
    return df.drop("_rth_open").sort("ts")
