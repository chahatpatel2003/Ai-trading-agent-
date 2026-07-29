"""Phase 21 — derived order-flow microstructure features from bar-level MBP1 columns.

The MBP1 layer already provides per-bar ``ofi``, ``delta``, ``cvd``, ``tob_imbalance``,
``bid/ask_depletion``, ``trade_aggression``, ``spread``, ``liq_spread_ratio``. This module derives
the *dynamics* the Phase-21 families need — rolling z-scores (scale-robust thresholds), OFI/delta
acceleration, CVD momentum, and order-book imbalance persistence — using only current-and-past
bars (rolling/diff/shift), so a signal at bar *t* never reads bar *t+1*.

These are bar-resolution aggregates of order flow (delta/CVD/OFI per 5m bar), joined onto the bar
frame upstream; they are present only where clean MBP1 exists (through 2025-H1).
"""

from __future__ import annotations

import polars as pl

ORDERFLOW_DERIVED: tuple[str, ...] = (
    "ofi_z", "delta_z", "cvd_mom_z", "ofi_sum5", "delta_accel", "cvd_slope",
    "imb_persist_up", "imb_persist_dn",
)


def _zscore(col: pl.Expr, win: int) -> pl.Expr:
    """Trailing rolling z-score (point-in-time safe)."""
    mean = col.rolling_mean(window_size=win)
    std = col.rolling_std(window_size=win)
    return (col - mean) / std


def add_orderflow_features(frame: pl.DataFrame, *, z_win: int = 50) -> pl.DataFrame:
    """Derive order-flow dynamics; no-ops gracefully if base MBP1 columns are absent (skipped)."""
    if "delta" not in frame.columns or "ofi" not in frame.columns:
        return frame
    imb_sign = pl.col("tob_imbalance").sign()
    f = frame.with_columns(
        _zscore(pl.col("ofi"), z_win).alias("ofi_z"),
        _zscore(pl.col("delta"), z_win).alias("delta_z"),
        _zscore(pl.col("cvd").diff(), z_win).alias("cvd_mom_z"),
        pl.col("ofi").rolling_sum(window_size=5).alias("ofi_sum5"),
        pl.col("delta").diff().alias("delta_accel"),
        pl.col("cvd").diff().alias("cvd_slope"),
        imb_sign.alias("_imb_sign"),
    )
    # 3-bar top-of-book imbalance persistence (same side three bars running).
    return f.with_columns(
        (
            (pl.col("_imb_sign") > 0)
            & (pl.col("_imb_sign").shift(1) > 0)
            & (pl.col("_imb_sign").shift(2) > 0)
        ).cast(pl.Int8).alias("imb_persist_up"),
        (
            (pl.col("_imb_sign") < 0)
            & (pl.col("_imb_sign").shift(1) < 0)
            & (pl.col("_imb_sign").shift(2) < 0)
        ).cast(pl.Int8).alias("imb_persist_dn"),
    )
