"""
convert.py
==========

Build a clean Polars tick frame from decoded Dukascopy records and write the
Parquet outputs (snappy). The original .bi5 files are preserved separately under
raw/YYYY_MM/ (never overwritten); this module produces the converted datasets.

Outputs per month:
  parquet/YYYY_MM.parquet            -> timestamp, bid, ask, bid_volume, ask_volume
  processed/YYYY_MM_processed.parquet-> + spread (ask-bid), mid ((bid+ask)/2)
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

import config

TICK_SCHEMA = {
    "timestamp": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "bid": pl.Float64,
    "ask": pl.Float64,
    "bid_volume": pl.Float64,
    "ask_volume": pl.Float64,
}


def build_frame(records: list[tuple[int, int, int, float, float]], price_factor: int) -> pl.DataFrame:
    """records: (abs_ms, ask_int, bid_int, ask_vol, bid_vol). Returns sorted,
    exact-duplicate-row-deduped tick frame with scaled prices (UTC timestamps)."""
    if not records:
        return pl.DataFrame(schema=TICK_SCHEMA)

    ms, ask_i, bid_i, ask_v, bid_v = zip(*records)
    df = pl.DataFrame(
        {"ms": ms, "ask_i": ask_i, "bid_i": bid_i, "ask_v": ask_v, "bid_v": bid_v},
        schema={"ms": pl.Int64, "ask_i": pl.Int64, "bid_i": pl.Int64,
                "ask_v": pl.Float64, "bid_v": pl.Float64},
    )
    df = df.with_columns(
        pl.from_epoch(pl.col("ms"), time_unit="ms").dt.replace_time_zone("UTC").alias("timestamp"),
        (pl.col("bid_i") / price_factor).alias("bid"),
        (pl.col("ask_i") / price_factor).alias("ask"),
        pl.col("bid_v").alias("bid_volume"),
        pl.col("ask_v").alias("ask_volume"),
    ).select(["timestamp", "bid", "ask", "bid_volume", "ask_volume"])

    df = df.sort("timestamp")
    # Drop exact-duplicate records (Dukascopy occasionally repeats a row). Distinct
    # ticks that merely share a millisecond are KEPT (legitimate in tick data).
    df = df.unique(subset=["timestamp", "bid", "ask", "bid_volume", "ask_volume"],
                   keep="first", maintain_order=True)
    return df


def add_derived(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("ask") - pl.col("bid")).alias("spread"),
        ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
    )


def _atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.partial")
    df.write_parquet(tmp, compression=config.PARQUET_COMPRESSION, statistics=True)
    os.replace(tmp, path)


def write_month(df: pl.DataFrame, ym: str) -> dict[str, Path]:
    """Write parquet/ and processed/ outputs for one month. Returns paths."""
    pq_path = config.PARQUET_DIR / f"{ym}.parquet"
    proc_path = config.PROCESSED_DIR / f"{ym}_processed.parquet"
    _atomic_write_parquet(df, pq_path)
    _atomic_write_parquet(add_derived(df), proc_path)
    return {"parquet": pq_path, "processed": proc_path}
