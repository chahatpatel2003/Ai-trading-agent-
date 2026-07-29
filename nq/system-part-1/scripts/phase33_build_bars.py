"""Phase 33 — build sub-minute RTH bars (15s/30s/1m) + intraday scalp features from trades.

For an INTRADAY EOD-flat scalper, use the front-month outright's RAW bars per day (front month =
highest-volume NQ outright that day). No back-adjustment is needed: every feature is intraday
(session VWAP, micro ATR, micro range, opening range, dist-from-VWAP, developing POC). PIT-safe by
construction (cumulative / trailing within the day; opening range frozen after it forms). Prior-day
levels come from the same front-month's previous session (roll-adjacent days flagged).

Output: docs/phase33/bars_{tf}.parquet for tf in 15s/30s/1m, all years, RTH (09:30-16:00 ET).
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import polars as pl

RAW = Path("/Volumes/T9/nq data/raw/trades")
OUT = Path("docs/phase33")
OUT.mkdir(parents=True, exist_ok=True)
ET = "America/New_York"
RTH_LO, RTH_HI = 570, 960  # 09:30-16:00 ET
TICK = 0.25


def _raw_rth(files: list[Path]) -> pl.DataFrame:
    return (
        pl.scan_parquet([str(f) for f in files])
        .select(["ts_recv", "price", "size", "symbol"])
        .filter(pl.col("symbol").str.starts_with("NQ") & ~pl.col("symbol").str.contains("-"))
        .with_columns(pl.col("ts_recv").dt.convert_time_zone(ET).alias("et"))
        .with_columns(
            pl.col("et").dt.date().alias("d"),
            (pl.col("et").dt.hour().cast(pl.Int32) * 60 + pl.col("et").dt.minute().cast(pl.Int32)).alias("mod"),
        )
        .filter((pl.col("mod") >= RTH_LO) & (pl.col("mod") < RTH_HI))
        .collect()
    )


def _bars(rth: pl.DataFrame, tf: str) -> pl.DataFrame:
    vol = rth.group_by(["d", "symbol"]).agg(pl.col("size").sum().alias("v")).sort(
        ["d", "v"], descending=[False, True]
    )
    front = vol.unique(subset="d", keep="first", maintain_order=True).select(["d", "symbol"])
    f = rth.join(front, on=["d", "symbol"])
    return (
        f.sort("ts_recv")
        .with_columns(pl.col("ts_recv").dt.truncate(tf).alias("ts"))
        .group_by(["d", "ts"], maintain_order=True)
        .agg(
            pl.col("price").first().alias("open"), pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"), pl.col("price").last().alias("close"),
            pl.col("size").sum().alias("volume"),
            ((pl.col("price") * pl.col("size")).sum() / pl.col("size").sum()).alias("vwap_bar"),
            pl.len().alias("trades"), pl.col("mod").first().alias("mod"),
        )
        .sort(["d", "ts"])
    )


def _features(b: pl.DataFrame, tf_min: float) -> pl.DataFrame:
    """Intraday PIT-safe features (all over('d'), trailing/cumulative)."""
    d = pl.col("d")
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("close").shift(1).over(d)).abs(),
        (pl.col("low") - pl.col("close").shift(1).over(d)).abs(),
    )
    # session cumulative VWAP
    cum_pv = (pl.col("vwap_bar") * pl.col("volume")).cum_sum().over(d)
    cum_v = pl.col("volume").cum_sum().over(d)
    bars_5min = max(int(round(5 / tf_min)), 1)
    bars_10min = max(int(round(10 / tf_min)), 1)
    bars_15min = max(int(round(15 / tf_min)), 1)
    b = b.with_columns(tr.alias("_tr"))
    b = b.with_columns(
        (cum_pv / cum_v).alias("vwap"),
        pl.col("_tr").rolling_mean(window_size=14, min_samples=5).over(d).alias("micro_atr"),
        pl.col("high").rolling_max(window_size=bars_10min).shift(1).over(d).alias("rng_hi"),
        pl.col("low").rolling_min(window_size=bars_10min).shift(1).over(d).alias("rng_lo"),
        pl.col("high").rolling_max(window_size=bars_5min).shift(1).over(d).alias("rng_hi5"),
        pl.col("low").rolling_min(window_size=bars_5min).shift(1).over(d).alias("rng_lo5"),
    )
    # opening range = first 15 minutes high/low, frozen after
    oropen = (
        b.filter(pl.col("mod") < RTH_LO + 15)
        .group_by("d").agg(pl.col("high").max().alias("or_hi"), pl.col("low").min().alias("or_lo"))
    )
    b = b.join(oropen, on="d", how="left")
    b = b.with_columns(
        (pl.col("close") - pl.col("vwap")).alias("dist_vwap"),
        ((pl.col("close") - pl.col("vwap")) / pl.col("micro_atr")).alias("dist_vwap_atr"),
    )
    return b.drop("_tr")


def main() -> None:
    files = sorted(RAW.glob("20*_H*.parquet"))
    print("trade files:", [f.name for f in files])
    for tf, tf_min in (("15s", 0.25), ("30s", 0.5), ("1m", 1.0)):
        t0 = time.time()
        parts = []
        for f in files:
            rth = _raw_rth([f])
            parts.append(_bars(rth, tf))
        allb = pl.concat(parts).unique(subset=["d", "ts"], keep="first").sort(["d", "ts"])
        allb = _features(allb, tf_min)
        allb.write_parquet(OUT / f"bars_{tf}.parquet")
        yrs = allb.with_columns(pl.col("d").dt.year().alias("y"))["y"].value_counts().sort("y")
        print(f"{tf}: {allb.height} bars, {allb['d'].n_unique()} days, {time.time()-t0:.0f}s "
              f"years={yrs.to_dicts()}")


if __name__ == "__main__":
    main()
