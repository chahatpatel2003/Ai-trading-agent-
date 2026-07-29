from __future__ import annotations

import polars as pl
import pytest

from nqlab.features.build import build_features
from nqlab.features.context import CONTEXT_FEATURE_NAMES, build_context_features
from tests.synth import feature_bars, globex_bars


def test_context_columns_added() -> None:
    frame = build_features(feature_bars(days=2), rth_start_min=570)
    out = build_context_features(frame, rth_start_min=570)
    assert set(CONTEXT_FEATURE_NAMES) <= set(out.columns)


def test_no_lookahead_context() -> None:
    frame = build_features(feature_bars(days=2), rth_start_min=570)
    full = build_context_features(frame, rth_start_min=570)
    k = 120
    prefix = build_context_features(frame.head(k + 1), rth_start_min=570)
    for col in ("prev_day_close", "gap", "htf_ema", "vol_regime_ratio"):
        a, b = full.row(k, named=True)[col], prefix.row(k, named=True)[col]
        if a is None and b is None:
            continue
        assert a == pytest.approx(b, rel=1e-9, abs=1e-9), col


def test_session_high_is_null_until_session_closes() -> None:
    bars = globex_bars(days=2).with_columns(pl.lit(1.0).alias("realized_vol"))
    out = build_context_features(bars, rth_start_min=570)
    # During the Asia session itself, asia_high must be null (not yet known).
    asia_rows = out.filter(pl.col("session") == "ASIA")
    assert asia_rows["asia_high"].null_count() == asia_rows.height
    # After 03:00 ET it equals that day's Asia high.
    last_day = out["trading_day"].max()
    day = out.filter(pl.col("trading_day") == last_day)
    expected = day.filter(pl.col("session") == "ASIA")["high"].max()
    seen = day.filter(pl.col("minute_of_day") >= 180)["asia_high"].drop_nulls()
    if seen.len() > 0 and expected is not None:
        assert seen[0] == expected


def test_no_lookahead_session_high_with_wrapping() -> None:
    bars = globex_bars(days=2).with_columns(pl.lit(1.0).alias("realized_vol"))
    full = build_context_features(bars, rth_start_min=570)
    k = full.height - 30  # a late (NY/close) bar of the last day
    prefix = build_context_features(bars.head(k + 1), rth_start_min=570)
    for col in ("asia_high", "london_high"):
        a, b = full.row(k, named=True)[col], prefix.row(k, named=True)[col]
        assert (a is None and b is None) or a == b, col
