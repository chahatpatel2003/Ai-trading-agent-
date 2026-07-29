"""Incremental MBP-1 feature materialization for the continuous front contract.

Builds one month at a time for the active outright(s), aligned to the trades
continuous (``NQ.c``). Months with no MBP-1 data are recorded as *missing*
(``mbp1_present=False``) — never forward-filled — so MBP1-required strategies
skip them. CVD (per-day cumulative) and the liquidity-regime ratio (trailing
rolling) are added after assembly; both are point-in-time safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from nqlab.config.models import AppConfig
from nqlab.data.bars_full import months_in_range
from nqlab.data.continuous import compute_active_map
from nqlab.data.ledger import PartitionLedger
from nqlab.features.mbp1 import MBP1_FEATURE_VERSION, build_mbp1_bar_features


@dataclass
class Mbp1BuildResult:
    symbol: str
    timeframe: str
    built: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    total_rows: int = 0


def mbp1_features_root(config: AppConfig) -> Path:
    return config.data.curated_root / "mbp1_features"


def _ledger_path(config: AppConfig, symbol: str, timeframe: str) -> Path:
    return config.data.curated_root / "_status" / f"mbp1_{symbol}_{timeframe}.json"


def _utc(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def _finalize(df: pl.DataFrame) -> pl.DataFrame:
    """Add CVD (per-day cumulative) and the liquidity-regime ratio/label."""
    df = df.sort("ts").with_columns(
        pl.col("delta").cum_sum().over("trading_day").alias("cvd")
    )
    median = pl.col("spread").rolling_median(window_size=100)
    df = df.with_columns((pl.col("spread") / median).alias("liq_spread_ratio"))
    return df.with_columns(
        pl.when(pl.col("liq_spread_ratio") > 1.5).then(1)  # thin
        .when(pl.col("liq_spread_ratio") < 0.7).then(2)  # thick
        .otherwise(0).cast(pl.Int8).alias("liquidity_regime")
    )


def build_mbp1_features_full(
    config: AppConfig,
    *,
    root: str = "NQ",
    timeframe: str,
    start: date,
    end: date,
    force: bool = False,
) -> Mbp1BuildResult:
    """Materialize MBP-1 features month-by-month for the active front contract."""
    active = compute_active_map(config, root=root, timeframe=timeframe, start=start, end=end)
    cont_symbol = f"{root}.c"
    result = Mbp1BuildResult(symbol=cont_symbol, timeframe=timeframe)
    if not active:
        return result

    ledger = PartitionLedger.load(_ledger_path(config, cont_symbol, timeframe))
    version_key = f"{config.config_hash()}:{MBP1_FEATURE_VERSION}"
    root_dir = mbp1_features_root(config)

    for month_start, month_end in months_in_range(start, end):
        key = f"{month_start.year}{month_start.month:02d}"
        if not force and ledger.is_complete(key, version_key):
            result.skipped.append(key)
            continue

        month_days = [d for d in active if month_start <= d <= month_end]
        active_syms = sorted({active[d] for d in month_days})
        frames: list[pl.DataFrame] = []
        for sym in active_syms:
            feats = build_mbp1_bar_features(
                config, symbol=sym, timeframe=timeframe, start=month_start, end=month_end
            )
            if feats.height == 0:
                continue
            keep = {d for d in month_days if active[d] == sym}
            frames.append(feats.filter(pl.col("trading_day").is_in(list(keep))))

        if not frames:  # no MBP-1 data this month -> record as missing, write nothing.
            ledger.mark_complete(
                key, rows=0, config_hash=version_key, source="MBP1", mbp1_present=False,
                input_start=month_start.isoformat(), input_end=month_end.isoformat(),
            )
            ledger.save()
            result.missing.append(key)
            continue

        month_feats = _finalize(pl.concat(frames)).with_columns(
            pl.lit(cont_symbol).alias("symbol")
        )
        path = (
            root_dir / f"symbol={cont_symbol}" / f"timeframe={timeframe}"
            / f"year_month={key}" / "data.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        month_feats.write_parquet(path)
        ledger.mark_complete(
            key, rows=month_feats.height, config_hash=version_key, source="MBP1",
            mbp1_present=True, feature_version=MBP1_FEATURE_VERSION,
            input_start=month_start.isoformat(), input_end=month_end.isoformat(),
        )
        ledger.save()
        result.total_rows += month_feats.height
        result.built.append(key)

    return result


def read_mbp1_features(
    config: AppConfig,
    *,
    symbol: str,
    timeframe: str,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    base = mbp1_features_root(config) / f"symbol={symbol}" / f"timeframe={timeframe}"
    files = sorted(base.glob("year_month=*/data.parquet"))
    if not files:
        raise FileNotFoundError(f"No MBP1 features for symbol={symbol} timeframe={timeframe}")
    lf = pl.scan_parquet([str(f) for f in files])
    if start is not None:
        lf = lf.filter(pl.col("ts") >= _utc(start))
    if end is not None:
        lf = lf.filter(pl.col("ts") < _utc(end + timedelta(days=1)))
    return lf.sort("ts").collect()


def profile_mbp1(config: AppConfig, symbol: str, timeframe: str) -> dict[str, list[str]]:
    """Built vs missing months from the MBP1 feature ledger."""
    ledger = PartitionLedger.load(_ledger_path(config, symbol, timeframe))
    built: list[str] = []
    missing: list[str] = []
    for key, entry in sorted(ledger.entries.items()):
        if entry.status != "complete":
            continue
        (built if entry.extra.get("mbp1_present") else missing).append(key)
    return {"built": built, "missing": missing}
