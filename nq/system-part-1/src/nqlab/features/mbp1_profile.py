"""Distribution profiling for MBP1 features (for percentile-calibrated grids).

Profiles each microstructure feature's mean + quantiles, overall and broken down
by session, liquidity regime, and hour-of-day. Calibration reads percentiles from
the **training** slice only, so thresholds are never derived from out-of-sample.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl

QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99)

PROFILE_FEATURES: tuple[str, ...] = (
    "tob_imbalance", "abs_tob_imbalance", "ofi", "signed_volume", "delta", "cvd_slope",
    "spread", "quote_updates", "liq_spread_ratio", "bid_depletion", "ask_depletion",
    "bid_replenishment", "ask_replenishment", "mid_vol",
)

_Stats = dict[str, float]


def _qkey(q: float) -> str:
    return f"q{round(q * 100)}"


def augment(frame: pl.DataFrame) -> pl.DataFrame:
    """Add derived profiling columns: ``abs_tob_imbalance`` and ``cvd_slope``."""
    cols = []
    if "tob_imbalance" in frame.columns:
        cols.append(pl.col("tob_imbalance").abs().alias("abs_tob_imbalance"))
    if "cvd" in frame.columns and "trading_day" in frame.columns:
        cols.append(pl.col("cvd").diff().over("trading_day").alias("cvd_slope"))
    return frame.with_columns(cols) if cols else frame


def _stat_exprs(features: list[str]) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    for f in features:
        exprs.append(pl.col(f).mean().alias(f"{f}::mean"))
        for q in QUANTILES:
            exprs.append(pl.col(f).quantile(q).alias(f"{f}::{_qkey(q)}"))
    return exprs


def _row_to_stats(row: dict[str, object], features: list[str]) -> dict[str, _Stats]:
    out: dict[str, _Stats] = {}
    for f in features:
        mean = row.get(f"{f}::mean")
        if mean is None:
            continue
        stats: _Stats = {"mean": cast(float, mean)}
        for q in QUANTILES:
            value = row.get(f"{f}::{_qkey(q)}")
            stats[_qkey(q)] = cast(float, value) if value is not None else float("nan")
        out[f] = stats
    return out


def _grouped(
    frame: pl.DataFrame, features: list[str], group_col: str
) -> dict[str, dict[str, _Stats]]:
    if group_col not in frame.columns:
        return {}
    grouped = frame.group_by(group_col).agg(_stat_exprs(features)).sort(group_col)
    return {
        str(row[group_col]): _row_to_stats(row, features)
        for row in grouped.iter_rows(named=True)
    }


@dataclass
class DistributionProfile:
    features: list[str]
    n_bars: int
    overall: dict[str, _Stats]
    by_session: dict[str, dict[str, _Stats]]
    by_regime: dict[str, dict[str, _Stats]]
    by_hour: dict[str, dict[str, _Stats]]

    def percentile(self, feature: str, q: float) -> float:
        return self.overall[feature][_qkey(q)]

    def to_dict(self) -> dict[str, object]:
        return {
            "features": self.features, "n_bars": self.n_bars, "overall": self.overall,
            "by_session": self.by_session, "by_regime": self.by_regime, "by_hour": self.by_hour,
        }

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def profile_distributions(
    frame: pl.DataFrame, *, features: tuple[str, ...] = PROFILE_FEATURES
) -> DistributionProfile:
    """Profile MBP1 feature distributions overall and by session/regime/hour."""
    df = augment(frame)
    if "minute_of_day" in df.columns:
        df = df.with_columns((pl.col("minute_of_day") // 60).alias("hour"))
    available = [f for f in features if f in df.columns]
    overall_row = df.select(_stat_exprs(available)).row(0, named=True) if available else {}
    return DistributionProfile(
        features=available,
        n_bars=df.height,
        overall=_row_to_stats(overall_row, available),
        by_session=_grouped(df, available, "session"),
        by_regime=_grouped(df, available, "liquidity_regime"),
        by_hour=_grouped(df, available, "hour"),
    )
