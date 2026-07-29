"""Incremental, month-by-month feature materialization with warmup + metadata.

Each month is computed from bars that include a warmup buffer of prior days so
rolling/EWM windows, previous-day levels, and the opening range are correct at
the month boundary; only rows inside the month are written. Features stay
point-in-time safe (no negative shifts). Each partition records metadata:
feature version, input data range, warmup bars, created timestamp, config hash.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from nqlab.config.models import AppConfig
from nqlab.data.bars_build import read_bars
from nqlab.data.bars_full import months_in_range
from nqlab.data.ledger import PartitionLedger
from nqlab.domain.calendar.sessions import parse_hhmm
from nqlab.features.build import FeatureParams, build_features, features_root


@dataclass
class FeatureBuildResult:
    symbol: str
    timeframe: str
    built: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    total_rows: int = 0


def _utc(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def _ledger_path(config: AppConfig, symbol: str, timeframe: str) -> Path:
    return config.data.curated_root / "_status" / f"features_{symbol}_{timeframe}.json"


def build_features_full(
    config: AppConfig,
    *,
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
    params: FeatureParams | None = None,
    warmup_days: int = 7,
    force: bool = False,
) -> FeatureBuildResult:
    """Materialize PIT features month-by-month with a warmup buffer."""
    p = params or FeatureParams()
    ledger = PartitionLedger.load(_ledger_path(config, symbol, timeframe))
    # Invalidate on either config change or feature-param-version change.
    version_key = f"{config.config_hash()}:{p.version}"
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    root = features_root(config)
    result = FeatureBuildResult(symbol=symbol, timeframe=timeframe)

    for month_start, month_end in months_in_range(start, end):
        key = f"{month_start.year}{month_start.month:02d}"
        if not force and ledger.is_complete(key, version_key):
            result.skipped.append(key)
            continue

        warmup_start = month_start - timedelta(days=warmup_days)
        try:
            bars = read_bars(
                config, symbol=symbol, timeframe=timeframe, start=warmup_start, end=month_end
            )
        except FileNotFoundError:
            continue
        if bars.height == 0:
            continue

        feats = build_features(bars, rth_start_min=rth_start_min, params=p)
        warmup_bars = feats.filter(pl.col("ts") < _utc(month_start)).height
        month_feats = feats.filter(
            (pl.col("ts") >= _utc(month_start))
            & (pl.col("ts") < _utc(month_end + timedelta(days=1)))
        )
        if month_feats.height == 0:
            continue

        path = (
            root
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / f"year_month={key}"
            / "data.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        month_feats.write_parquet(path)

        ledger.mark_complete(
            key,
            rows=month_feats.height,
            config_hash=version_key,
            feature_version=p.version,
            warmup_bars=warmup_bars,
            input_start=warmup_start.isoformat(),
            input_end=month_end.isoformat(),
            feature_params=asdict(p),
        )
        ledger.save()
        result.total_rows += month_feats.height
        result.built.append(key)

    return result
