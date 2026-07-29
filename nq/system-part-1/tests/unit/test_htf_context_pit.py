"""Point-in-time (no-lookahead) tests for the Phase-9 HTF context layer."""

from __future__ import annotations

import polars as pl
import pytest

from nqlab.features.build import build_features
from nqlab.features.context import build_context_features
from nqlab.features.htf_context import HTF_CONTEXT_FEATURE_NAMES, build_htf_context_features
from tests.synth import globex_bars

_RTH = 570  # 09:30 ET

# HTF columns whose value at row t must depend only on rows <= t.
_NUMERIC_PIT = [
    "prev_day_open",
    "prev_day_range",
    "prev_week_high",
    "prev_week_low",
    "prev_month_high",
    "prev_month_low",
    "overnight_high",
    "overnight_low",
    "overnight_range",
    "ny_open_loc",
    "prev_inside_day",
    "prev_outside_day",
    "prev_nr_day",
    "prev_wr_day",
    "gap_dir",
    "gap_filled",
    "compression_setup",
    "vol_regime",
    "trend_1h_bull",
    "trend_4h_bull",
    "res_dist",
    "sup_dist",
]
_CATEGORICAL_PIT = ["prev_day_type"]


def _stack(bars: pl.DataFrame) -> pl.DataFrame:
    feats = build_features(bars, rth_start_min=_RTH)
    ctx = build_context_features(feats, rth_start_min=_RTH)
    return build_htf_context_features(ctx, rth_start_min=_RTH, timeframe="5m")


def test_htf_columns_present() -> None:
    full = _stack(globex_bars(days=10))
    assert set(HTF_CONTEXT_FEATURE_NAMES) <= set(full.columns)


def test_pit_test_covers_every_htf_feature() -> None:
    """Guard: the no-lookahead test must check EVERY HTF feature, so a newly added
    column can never silently escape PIT verification."""
    assert set(_NUMERIC_PIT) | set(_CATEGORICAL_PIT) == set(HTF_CONTEXT_FEATURE_NAMES)


def test_no_lookahead_htf_context() -> None:
    """For MANY bars — including the first exposed bar of a day (overnight/open
    boundary) and bars deep into later weeks/months — a feature at row k must be
    identical whether computed on the full series or only on the prefix [0..k]."""
    bars = globex_bars(days=50)  # ~7 weeks / 2 months: prior-week & prior-month exist
    full = _stack(bars).with_row_index("idx")
    candidates = full.filter(
        pl.col("is_rth")
        & (pl.col("minute_of_day") >= 570)  # from the open bar onward (boundary stress)
        & pl.col("prev_month_high").is_not_null()  # monthly + weekly context available
        & pl.col("prev_week_high").is_not_null()
    )
    assert candidates.height > 0
    idxs = candidates["idx"].to_list()
    n = len(idxs)
    # earliest + latest exposed bar + several evenly-spaced interior bars
    sample = sorted({idxs[0], idxs[-1], *(idxs[int(f * (n - 1))] for f in (0.2, 0.4, 0.6, 0.8))})

    for k in sample:
        prefix = _stack(bars.head(k + 1))
        full_row = full.drop("idx").row(k, named=True)
        prefix_row = prefix.row(k, named=True)
        for col in _NUMERIC_PIT:
            a, b = full_row[col], prefix_row[col]
            if a is None and b is None:
                continue
            assert a == pytest.approx(b, rel=1e-9, abs=1e-9), f"lookahead in {col} @k={k}: {a}!={b}"
        for col in _CATEGORICAL_PIT:
            assert full_row[col] == prefix_row[col], f"lookahead in {col} @k={k}"


def test_overnight_and_open_location_null_before_rth_open() -> None:
    full = _stack(globex_bars(days=5))
    before = full.filter(pl.col("minute_of_day") < _RTH)
    assert before["overnight_high"].null_count() == before.height
    assert before["ny_open_loc"].null_count() == before.height
    # After the open (on days with a prior day) the overnight range is known.
    after = full.filter(
        (pl.col("minute_of_day") >= _RTH + 60) & pl.col("overnight_high").is_not_null()
    )
    assert after.height > 0


def test_prev_day_type_values_are_known_labels() -> None:
    full = _stack(globex_bars(days=20))
    labels = set(full["prev_day_type"].drop_nulls().unique().to_list())
    allowed = {"trend_up", "trend_down", "range", "reversal", "expansion", "compression", "neutral"}
    assert labels <= allowed
