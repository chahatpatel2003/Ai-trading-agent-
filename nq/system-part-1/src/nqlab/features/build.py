"""Point-in-time feature computation on a single-symbol bar series.

Every feature is a function of the current and *past* bars only: rolling/EWM
windows are trailing, ``vwap_session`` is a within-day cumulative, previous-day
high/low use the fully-completed prior trading day, and the opening range is
``null`` until the OR window closes. No expression uses a negative shift, so
slicing the series at row ``t`` reproduces every value at ``t`` (verified by a
no-lookahead test).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, timedelta
from pathlib import Path

import polars as pl

from nqlab.config.models import AppConfig
from nqlab.domain.calendar.sessions import parse_hhmm

FEATURE_NAMES: tuple[str, ...] = (
    "ret_1",
    "atr",
    "roll_high",
    "roll_low",
    "prev_day_high",
    "prev_day_low",
    "vwap_session",
    "dist_vwap",
    "or_high",
    "or_low",
    "minute_of_day",
    "dow",
    "session",
    "rel_vol",
    "sma",
    "ema",
    "rsi",
    "realized_vol",
)


@dataclass(frozen=True)
class FeatureParams:
    """Versioned feature parameters (part of a strategy's reproducibility key)."""

    version: str = "1"
    atr_period: int = 14
    range_window: int = 20
    sma_window: int = 20
    ema_window: int = 20
    vol_window: int = 20
    rv_window: int = 20
    rsi_period: int = 14
    or_minutes: int = 30


def features_root(config: AppConfig) -> Path:
    return config.data.curated_root / "features"


def build_features(
    frame: pl.DataFrame,
    *,
    rth_start_min: int,
    params: FeatureParams | None = None,
) -> pl.DataFrame:
    """Return ``frame`` (single symbol, sorted) plus PIT feature columns."""
    p = params or FeatureParams()
    df = frame.sort("ts")

    prev_close = pl.col("close").shift(1)
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    df = df.with_columns(
        pl.col("close").pct_change().alias("ret_1"),
        true_range.alias("_tr"),
        pl.col("high").rolling_max(window_size=p.range_window).alias("roll_high"),
        pl.col("low").rolling_min(window_size=p.range_window).alias("roll_low"),
        pl.col("close").rolling_mean(window_size=p.sma_window).alias("sma"),
        pl.col("close").ewm_mean(span=p.ema_window, adjust=False).alias("ema"),
        pl.col("volume").rolling_mean(window_size=p.vol_window).alias("vol_sma"),
    )
    df = df.with_columns(
        pl.col("_tr").rolling_mean(window_size=p.atr_period).alias("atr"),
        (pl.col("volume") / pl.col("vol_sma")).alias("rel_vol"),
        pl.col("ret_1").rolling_std(window_size=p.rv_window).alias("realized_vol"),
    )

    delta = pl.col("close").diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    df = df.with_columns(
        gain.ewm_mean(alpha=1.0 / p.rsi_period, adjust=False).alias("_avg_gain"),
        loss.ewm_mean(alpha=1.0 / p.rsi_period, adjust=False).alias("_avg_loss"),
    )
    df = df.with_columns(
        (100.0 - 100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss"))).alias("rsi")
    )

    # Session VWAP (within-day cumulative) and distance from it.
    df = df.with_columns(
        (pl.col("vwap") * pl.col("volume")).cum_sum().over("trading_day").alias("_pv"),
        pl.col("volume").cum_sum().over("trading_day").alias("_cv"),
    )
    df = df.with_columns((pl.col("_pv") / pl.col("_cv")).alias("vwap_session"))
    df = df.with_columns((pl.col("close") - pl.col("vwap_session")).alias("dist_vwap"))

    # Previous-day high/low from the fully completed prior trading day.
    daily = (
        df.group_by("trading_day")
        .agg(pl.col("high").max().alias("dh"), pl.col("low").min().alias("dl"))
        .sort("trading_day")
        .with_columns(
            pl.col("dh").shift(1).alias("prev_day_high"),
            pl.col("dl").shift(1).alias("prev_day_low"),
        )
    )
    df = df.join(
        daily.select("trading_day", "prev_day_high", "prev_day_low"),
        on="trading_day",
        how="left",
    )

    # Opening range: known only once the OR window has closed.
    or_end = rth_start_min + p.or_minutes
    or_df = (
        df.filter(
            pl.col("is_rth")
            & (pl.col("minute_of_day") >= rth_start_min)
            & (pl.col("minute_of_day") < or_end)
        )
        .group_by("trading_day")
        .agg(pl.col("high").max().alias("orh"), pl.col("low").min().alias("orl"))
    )
    df = df.join(or_df, on="trading_day", how="left").with_columns(
        pl.when(pl.col("minute_of_day") >= or_end).then(pl.col("orh")).alias("or_high"),
        pl.when(pl.col("minute_of_day") >= or_end).then(pl.col("orl")).alias("or_low"),
    )

    return df.drop("_tr", "_avg_gain", "_avg_loss", "_pv", "_cv", "orh", "orl").sort("ts")


def _utc(d: date) -> object:
    from datetime import datetime

    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def build_and_write_features(
    config: AppConfig,
    *,
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
    params: FeatureParams | None = None,
) -> int:
    """Compute features for one symbol and materialize under ``curated_root/features``."""
    from nqlab.data.bars_build import read_bars

    rth_start_min = parse_hhmm(config.sessions.rth_start)
    bars = read_bars(config, symbol=symbol, timeframe=timeframe, start=start, end=end)
    feats = build_features(bars, rth_start_min=rth_start_min, params=params)
    feats = feats.with_columns(pl.col("trading_day").dt.strftime("%Y%m").alias("year_month"))
    root = features_root(config)
    for (sym, year_month), group in feats.group_by(["symbol", "year_month"]):
        path = (
            root
            / f"symbol={sym}"
            / f"timeframe={timeframe}"
            / f"year_month={year_month}"
            / "data.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        group.drop("year_month").write_parquet(path)
    return int(feats.height)


def read_features(
    config: AppConfig,
    *,
    symbol: str,
    timeframe: str,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    base = features_root(config) / f"symbol={symbol}" / f"timeframe={timeframe}"
    files = sorted(base.glob("year_month=*/data.parquet"))
    if not files:
        raise FileNotFoundError(f"No features for symbol={symbol} timeframe={timeframe}")
    lf = pl.scan_parquet([str(f) for f in files])
    if start is not None:
        lf = lf.filter(pl.col("ts") >= _utc(start))
    if end is not None:
        lf = lf.filter(pl.col("ts") < _utc(end + timedelta(days=1)))
    return lf.sort("ts").collect()
