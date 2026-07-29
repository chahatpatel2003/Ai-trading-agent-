"""Phase 57 — GC/MGC data foundation (self-contained; does NOT touch NQ data).

Loads the local GC 1-min OHLCV bars (~/Desktop/gc), converts to ET,
builds 5-minute bars with a gold-session clock, and caches to the scratchpad.

Gold session clock: a CME gold session runs 18:00 ET -> 17:00 ET next day.
We shift timestamps by +6h so each session maps to one `session_date` and an
intraday minute index `smin` that is monotonic 0..1380 across the session.
We also keep the real ET hour/minute (`et_mod` = hour*60+min) for window logic.
"""

from __future__ import annotations

import glob
import os
import tempfile

import polars as pl

GC_DIR = os.path.expanduser("~/Desktop/gc")
SCRATCH = os.environ.get("NQLAB_SCRATCH", os.path.join(tempfile.gettempdir(), "nqlab-phase57"))
os.makedirs(SCRATCH, exist_ok=True)
CACHE = os.path.join(SCRATCH, "gc_5m.parquet")


def load_1m() -> pl.DataFrame:
    files = sorted(
        f for f in glob.glob(os.path.join(GC_DIR, "*.parquet"))
        if "2024_jan" not in os.path.basename(f)  # exclude duplicate Jan subset
    )
    parts = []
    for f in files:
        parts.append(
            pl.read_parquet(f, columns=["ts_event", "open", "high", "low", "close", "volume"])
        )
    df = pl.concat(parts).unique(subset=["ts_event"], keep="first").sort("ts_event")
    # ET clock
    df = df.with_columns(pl.col("ts_event").dt.convert_time_zone("America/New_York").alias("et"))
    return df


def build_5m(df1: pl.DataFrame) -> pl.DataFrame:
    # session clock: +6h shift so 18:00 ET -> 00:00 of session_date
    df1 = df1.with_columns(
        (pl.col("et") + pl.duration(hours=6)).alias("sclk"),
    )
    df1 = df1.with_columns(
        pl.col("sclk").dt.date().alias("session_date"),
        (pl.col("et").dt.hour().cast(pl.Int32) * 60
         + pl.col("et").dt.minute().cast(pl.Int32)).alias("et_mod"),
    )
    # 5-min bucket on real ET wall clock
    df1 = df1.with_columns(pl.col("et").dt.truncate("5m").alias("b5"))
    g = (
        df1.group_by("b5")
        .agg(
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("session_date").first().alias("session_date"),
            pl.col("et_mod").first().alias("et_mod"),
            pl.col("et").first().alias("et"),
        )
        .sort("b5")
    )
    # smin: minutes since session start (18:00 ET => et_mod 1080). Map to 0..1439.
    g = g.with_columns(
        pl.when(pl.col("et_mod") >= 1080)
        .then(pl.col("et_mod") - 1080)
        .otherwise(pl.col("et_mod") + 360)
        .alias("smin")
    )
    return g


def main() -> None:
    df1 = load_1m()
    print("1m rows:", df1.height, "range:", df1["et"].min(), "->", df1["et"].max())
    g = build_5m(df1)
    g.write_parquet(CACHE)
    print("5m rows:", g.height, "-> cached:", CACHE)
    # sanity: sessions/year, bars/session
    yr = g.with_columns(pl.col("session_date").dt.year().alias("y"))
    per = yr.group_by("y").agg(
        pl.col("session_date").n_unique().alias("sessions"),
        pl.len().alias("bars"),
    ).sort("y")
    print(per)
    # session window check: where does NY pit (08:20 ET = et_mod 500) land in smin?
    print("smin at et_mod 1080 (18:00 ET, Asia open):", g.filter(pl.col("et_mod") == 1080)["smin"].head(1).to_list())
    print("smin at et_mod 500 (08:20 ET, COMEX):", g.filter(pl.col("et_mod") == 500)["smin"].head(1).to_list())


if __name__ == "__main__":
    main()
