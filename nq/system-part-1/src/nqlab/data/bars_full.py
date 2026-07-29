"""Full-history incremental bar materialization (month-complete, resumable).

Builds one calendar month at a time, recording completion in a ledger so a re-run
skips months already built under the same config. Partial/corrupt partitions
(present on disk but not marked complete, or with a row-count mismatch) are
rebuilt. Raw data is never duplicated — only the curated month partitions.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import polars as pl

from nqlab.config.models import AppConfig
from nqlab.data.bars_build import bars_root, build_bars
from nqlab.data.ledger import PartitionLedger


@dataclass
class FullBuildResult:
    timeframe: str
    built: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    rebuilt: list[str] = field(default_factory=list)
    total_rows: int = 0


def months_in_range(start: date, end: date) -> list[tuple[date, date]]:
    """Return (month_start, month_end) for every calendar month overlapping [start, end]."""
    out: list[tuple[date, date]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        last = calendar.monthrange(year, month)[1]
        out.append((date(year, month, 1), date(year, month, last)))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def _ledger_path(config: AppConfig, timeframe: str) -> Path:
    return config.data.curated_root / "_status" / f"bars_{timeframe}.json"


def _month_partition_rows(config: AppConfig, timeframe: str, key: str) -> int | None:
    """Total rows across all symbol partitions for one month, or None if unreadable."""
    root = bars_root(config)
    files = list(root.glob(f"symbol=*/timeframe={timeframe}/year_month={key}/data.parquet"))
    if not files:
        return None
    total = 0
    for path in files:
        try:
            total += pl.scan_parquet(str(path)).select(pl.len()).collect().item()
        except (OSError, pl.exceptions.PolarsError):
            return None  # corrupt partition
    return total


def build_bars_full(
    config: AppConfig,
    *,
    timeframe: str,
    start: date,
    end: date,
    roots: tuple[str, ...] = ("NQ",),
    force: bool = False,
    verify: bool = False,
) -> FullBuildResult:
    """Materialize bars month-by-month for ``[start, end]``, skipping complete months."""
    ledger = PartitionLedger.load(_ledger_path(config, timeframe))
    cfg_hash = config.config_hash()
    result = FullBuildResult(timeframe=timeframe)

    for month_start, month_end in months_in_range(start, end):
        key = f"{month_start.year}{month_start.month:02d}"
        complete = ledger.is_complete(key, cfg_hash)

        if complete and verify:
            actual = _month_partition_rows(config, timeframe, key)
            entry = ledger.get(key)
            if actual is None or (entry is not None and actual != entry.rows):
                complete = False  # partial/corrupt -> rebuild

        if complete and not force:
            result.skipped.append(key)
            continue

        had_partition = _month_partition_rows(config, timeframe, key) is not None
        built = build_bars(
            config, timeframe=timeframe, start=month_start, end=month_end, roots=roots
        )
        ledger.mark_complete(
            key,
            rows=built.total_bars,
            config_hash=cfg_hash,
            symbols=built.symbols,
            month=key,
        )
        ledger.save()  # persist after each month -> resume-safe
        result.total_rows += built.total_bars
        (result.rebuilt if (had_partition and not force) else result.built).append(key)

    return result


def build_status(config: AppConfig, timeframe: str) -> list[str]:
    """Completed month keys for a timeframe."""
    return PartitionLedger.load(_ledger_path(config, timeframe)).completed_keys()
