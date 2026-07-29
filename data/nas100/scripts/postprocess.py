"""
postprocess.py
==============

Build derived datasets from the per-month tick parquet files, into processed/:

  candles_{kind}_{tf}.parquet     for kind in {bid,ask,mid}, tf in the 11 timeframes
  tickbars_{kind}_{n}.parquet     for n in {100,500,1000,2000}
  spread_stats.json, volatility_stats.json, session_stats.json

Memory-safe: time candles are built one month at a time (candles never cross a
month) and appended; coarser timeframes are resampled from the 1s candles. Tick
bars are counted within each month (a partial bar may close at a month edge).
"""

from __future__ import annotations

import json

import polars as pl
import pyarrow.parquet as papq

import config

_MONTH_GLOB = "*_processed.parquet"

_CANDLE_AGG = [
    pl.col("_p").first().alias("open"),
    pl.col("_p").max().alias("high"),
    pl.col("_p").min().alias("low"),
    pl.col("_p").last().alias("close"),
    pl.len().alias("ticks"),
    (pl.col("bid_volume") + pl.col("ask_volume")).sum().alias("volume"),
]


class _Appender:
    def __init__(self, path):
        self.path = path
        self.writer = None

    def write(self, df: pl.DataFrame):
        if df.height == 0:
            return
        tbl = df.to_arrow()
        if self.writer is None:
            self.writer = papq.ParquetWriter(str(self.path), tbl.schema, compression="snappy")
        self.writer.write_table(tbl)

    def close(self):
        if self.writer:
            self.writer.close()


def _month_files() -> list:
    return sorted(config.PROCESSED_DIR.glob(_MONTH_GLOB))


def _resample(df: pl.DataFrame, every: str) -> pl.DataFrame:
    """Resample 1s candles to a coarser timeframe (OHLCV roll-up)."""
    return (df.group_by_dynamic("timestamp", every=every, closed="left", label="left")
              .agg(pl.col("open").first(), pl.col("high").max(),
                   pl.col("low").min(), pl.col("close").last(),
                   pl.col("ticks").sum(), pl.col("volume").sum()))


def build_candles(log) -> None:
    months = _month_files()
    if not months:
        log.warning("postprocess: no processed month files; skipping candles.")
        return
    for kind in config.PRICE_KINDS:
        one_s_path = config.CANDLES_DIR / f"candles_{kind}_1s.parquet"
        app = _Appender(one_s_path)
        for mf in months:
            df = pl.read_parquet(mf, columns=["timestamp", kind, "bid_volume", "ask_volume"])
            if df.height == 0:
                continue
            c = (df.with_columns(pl.col(kind).alias("_p"))
                   .group_by_dynamic("timestamp", every="1s", closed="left", label="left")
                   .agg(_CANDLE_AGG))
            app.write(c.select(["timestamp", "open", "high", "low", "close", "ticks", "volume"]))
        app.close()
        if not one_s_path.exists():
            continue
        base = pl.read_parquet(one_s_path)
        for alias, every in config.TIMEFRAMES:
            if every == "1s":
                continue
            out = config.CANDLES_DIR / f"candles_{kind}_{alias}.parquet"
            _resample(base, every).write_parquet(out, compression="snappy")
        log.info("postprocess: candles for '%s' done (%d 1s rows).", kind, base.height)


def build_tick_bars(log) -> None:
    months = _month_files()
    for kind in config.PRICE_KINDS:
        for n in config.TICK_BAR_SIZES:
            app = _Appender(config.TICKBARS_DIR / f"tickbars_{kind}_{n}.parquet")
            for mf in months:
                df = pl.read_parquet(mf, columns=["timestamp", kind, "bid_volume", "ask_volume"])
                if df.height == 0:
                    continue
                c = (df.with_row_index("_i")
                       .with_columns((pl.col("_i") // n).alias("_bar"),
                                     pl.col(kind).alias("_p"))
                       .group_by("_bar", maintain_order=True)
                       .agg(pl.col("timestamp").first().alias("timestamp"),
                            pl.col("_p").first().alias("open"), pl.col("_p").max().alias("high"),
                            pl.col("_p").min().alias("low"), pl.col("_p").last().alias("close"),
                            pl.len().alias("ticks"),
                            (pl.col("bid_volume") + pl.col("ask_volume")).sum().alias("volume")))
                app.write(c.select(["timestamp", "open", "high", "low", "close", "ticks", "volume"]))
            app.close()
        log.info("postprocess: tick bars for '%s' done.", kind)


def build_stats(log) -> None:
    months = _month_files()
    if not months:
        return
    lf = pl.scan_parquet([str(m) for m in months])

    # spread statistics
    s = lf.select(
        pl.col("spread").mean().alias("mean"), pl.col("spread").std().alias("std"),
        pl.col("spread").min().alias("min"), pl.col("spread").max().alias("max"),
        pl.col("spread").median().alias("median"),
        pl.col("spread").quantile(0.95).alias("p95"),
        pl.col("spread").quantile(0.99).alias("p99"),
        pl.len().alias("n_ticks"),
    ).collect(engine="streaming")
    (config.STATS_DIR / "spread_stats.json").write_text(
        json.dumps(s.to_dicts()[0], indent=2, default=str), encoding="utf-8")

    # session statistics (UTC-hour buckets: Asia/Europe/US/Other)
    sess = (lf.with_columns(pl.col("timestamp").dt.hour().alias("h"))
              .with_columns(pl.when(pl.col("h") < 7).then(pl.lit("Asia"))
                            .when(pl.col("h") < 13).then(pl.lit("Europe"))
                            .when(pl.col("h") < 21).then(pl.lit("US"))
                            .otherwise(pl.lit("Other")).alias("session"))
              .group_by("session")
              .agg(pl.len().alias("ticks"), pl.col("spread").mean().alias("avg_spread"),
                   pl.col("mid").min().alias("min_mid"), pl.col("mid").max().alias("max_mid"))
              .collect(engine="streaming"))
    (config.STATS_DIR / "session_stats.json").write_text(
        json.dumps(sess.to_dicts(), indent=2, default=str), encoding="utf-8")

    # volatility from daily mid candles (if present)
    dmid = config.CANDLES_DIR / "candles_mid_1d.parquet"
    if dmid.exists():
        d = pl.read_parquet(dmid).with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret"))
        daily_std = float(d["ret"].std() or 0.0)
        vol = {"daily_return_std": daily_std,
               "annualized_vol_252": daily_std * (252 ** 0.5),
               "days": d.height}
        (config.STATS_DIR / "volatility_stats.json").write_text(
            json.dumps(vol, indent=2, default=str), encoding="utf-8")
    log.info("postprocess: statistics done.")


def build_mid_spread_datasets(log) -> None:
    """Per-month dedicated mid-price and spread datasets in processed/mid_price/
    and processed/spread/."""
    for mf in _month_files():
        ym = mf.name.replace("_processed.parquet", "")
        df = pl.read_parquet(mf, columns=["timestamp", "mid", "spread"])
        df.select("timestamp", "mid").write_parquet(
            config.MID_PRICE_DIR / f"{ym}.parquet", compression="snappy")
        df.select("timestamp", "spread").write_parquet(
            config.SPREAD_DIR / f"{ym}.parquet", compression="snappy")
    log.info("postprocess: mid_price + spread datasets done.")


def run(log) -> None:
    for d in (config.MID_PRICE_DIR, config.SPREAD_DIR, config.CANDLES_DIR,
              config.TICKBARS_DIR, config.STATS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    log.info("POST-PROCESSING: mid/spread datasets, candles, tick bars, statistics ...")
    build_mid_spread_datasets(log)
    build_candles(log)
    build_tick_bars(log)
    build_stats(log)
    log.info("POST-PROCESSING complete.")
