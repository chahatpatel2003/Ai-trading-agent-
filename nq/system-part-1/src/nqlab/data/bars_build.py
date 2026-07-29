"""Bars engine: build OHLCV time bars from trades (read-only on raw).

- Buckets trades by ``ts_recv`` (the availability clock) truncated to the
  timeframe; ``ts_event``/``sequence`` only break ties for open/close ordering.
- Outrights only (spreads excluded); NQ/MNQ roots only.
- Adds session labels + trading-day. Materializes under ``curated_root/bars``,
  partitioned ``symbol=/timeframe=/year_month=`` (year-month from trading day).
- Never writes to ``raw/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from nqlab.config.models import AppConfig
from nqlab.data.discovery import Period
from nqlab.domain.calendar.sessions import SessionLabeler
from nqlab.domain.market.bars import polars_every


@dataclass(frozen=True)
class BarBuildResult:
    timeframe: str
    symbols: list[str]
    partitions_written: int
    total_bars: int


def bars_root(config: AppConfig) -> Path:
    return config.data.curated_root / "bars"


def _period_overlaps(period: Period, start: date, end: date) -> bool:
    p_start = date(period.year, 1 if period.half == 1 else 7, 1)
    p_end = date(period.year, 6, 30) if period.half == 1 else date(period.year, 12, 31)
    return not (p_end < start or p_start > end)


def _trade_files_for_range(directory: Path, start: date, end: date) -> list[Path]:
    files: list[Path] = []
    for path in sorted(directory.glob("*.parquet")):
        period = Period.parse(path.name)
        if period is None or _period_overlaps(period, start, end):
            files.append(path)
    return files


def _utc(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def build_bars(
    config: AppConfig,
    *,
    timeframe: str,
    start: date,
    end: date,
    roots: tuple[str, ...] = ("NQ", "MNQ"),
    symbols: list[str] | None = None,
) -> BarBuildResult:
    """Build and materialize OHLCV bars for ``[start, end]`` (inclusive)."""
    labeler = SessionLabeler.from_config(config.sessions)
    files = _trade_files_for_range(config.data.raw_trades_dir, start, end)
    if not files:
        return BarBuildResult(timeframe, [], 0, 0)

    every = polars_every(timeframe)
    start_dt, end_dt = _utc(start), _utc(end + timedelta(days=1))

    lf = (
        pl.scan_parquet([str(f) for f in files])
        .select(["ts_recv", "ts_event", "sequence", "price", "size", "symbol"])
        .filter((pl.col("ts_recv") >= start_dt) & (pl.col("ts_recv") < end_dt))
        .filter(~pl.col("symbol").str.contains("-"))  # outrights only
    )
    if symbols is not None:
        lf = lf.filter(pl.col("symbol").is_in(symbols))
    else:
        root_filter = pl.lit(False)
        for root in roots:
            root_filter = root_filter | pl.col("symbol").str.starts_with(root)
        lf = lf.filter(root_filter)

    bars = (
        lf.sort(["symbol", "ts_recv", "sequence"])
        .with_columns(pl.col("ts_recv").dt.truncate(every).alias("ts"))
        .group_by(["symbol", "ts"], maintain_order=True)
        .agg(
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("size").sum().alias("volume"),
            ((pl.col("price") * pl.col("size")).sum() / pl.col("size").sum()).alias("vwap"),
            pl.len().alias("trades"),
        )
        .sort(["symbol", "ts"])
        .collect()
    )
    if bars.height == 0:
        return BarBuildResult(timeframe, [], 0, 0)

    bars = labeler.with_calendar_columns(bars, "ts")
    built_symbols = sorted(bars["symbol"].unique().to_list())
    partitions = _write_partitioned(bars, bars_root(config), timeframe)
    return BarBuildResult(timeframe, built_symbols, partitions, bars.height)


def _write_partitioned(frame: pl.DataFrame, root: Path, timeframe: str) -> int:
    # Partition by the bar's UTC year-month so a calendar-month build maps to
    # exactly one partition (clean, non-overlapping, month-complete).
    frame = frame.with_columns(pl.col("ts").dt.strftime("%Y%m").alias("year_month"))
    count = 0
    for (symbol, year_month), group in frame.group_by(["symbol", "year_month"]):
        path = (
            root
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / f"year_month={year_month}"
            / "data.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        group.drop("year_month").write_parquet(path)
        count += 1
    return count


def read_bars(
    config: AppConfig,
    *,
    symbol: str,
    timeframe: str,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """Read materialized bars for one symbol/timeframe (optionally date-filtered)."""
    base = bars_root(config) / f"symbol={symbol}" / f"timeframe={timeframe}"
    files = sorted(base.glob("year_month=*/data.parquet"))
    if not files:
        raise FileNotFoundError(f"No bars found for symbol={symbol} timeframe={timeframe} ({base})")
    lf = pl.scan_parquet([str(f) for f in files])
    if start is not None:
        lf = lf.filter(pl.col("ts") >= _utc(start))
    if end is not None:
        lf = lf.filter(pl.col("ts") < _utc(end + timedelta(days=1)))
    return lf.sort("ts").collect()


@dataclass(frozen=True)
class BarPartitionInfo:
    symbol: str
    timeframe: str
    months: list[str]
    rows: int


def profile_bars(config: AppConfig) -> list[BarPartitionInfo]:
    """Summarize materialized bars under ``curated_root/bars``."""
    root = bars_root(config)
    out: list[BarPartitionInfo] = []
    if not root.is_dir():
        return out
    for sym_dir in sorted(root.glob("symbol=*")):
        symbol = sym_dir.name.split("=", 1)[1]
        for tf_dir in sorted(sym_dir.glob("timeframe=*")):
            timeframe = tf_dir.name.split("=", 1)[1]
            months: list[str] = []
            rows = 0
            for part in sorted(tf_dir.glob("year_month=*/data.parquet")):
                months.append(part.parent.name.split("=", 1)[1])
                rows += pl.scan_parquet(str(part)).select(pl.len()).collect().item()
            out.append(BarPartitionInfo(symbol, timeframe, months, rows))
    return out
