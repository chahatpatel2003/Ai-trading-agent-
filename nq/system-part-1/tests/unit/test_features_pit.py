from __future__ import annotations

import pytest

from nqlab.features.build import FEATURE_NAMES, build_features
from tests.synth import feature_bars

_RTH_START_MIN = 570  # 09:30 ET

# Features whose value at row t must depend only on rows <= t.
_PIT_COLUMNS = [
    "ret_1", "atr", "sma", "ema", "rsi", "roll_high", "roll_low",
    "vwap_session", "dist_vwap", "realized_vol", "rel_vol",
    "prev_day_high", "prev_day_low", "or_high", "or_low",
]


def test_all_feature_columns_present() -> None:
    feats = build_features(feature_bars(), rth_start_min=_RTH_START_MIN)
    assert set(FEATURE_NAMES) <= set(feats.columns)


def test_no_lookahead() -> None:
    """A feature at row k must be identical whether computed on the full series
    or only on the prefix [0..k] — proving no future data leaks in."""
    bars = feature_bars(days=2, bars_per_day=78)
    full = build_features(bars, rth_start_min=_RTH_START_MIN)
    k = 120  # well into the second day (past the opening range)

    prefix = build_features(bars.head(k + 1), rth_start_min=_RTH_START_MIN)
    full_row = full.row(k, named=True)
    prefix_row = prefix.row(k, named=True)

    for col in _PIT_COLUMNS:
        a, b = full_row[col], prefix_row[col]
        if a is None and b is None:
            continue
        assert a == pytest.approx(b, rel=1e-9, abs=1e-9), f"lookahead in {col}: {a} != {b}"


def test_opening_range_is_null_before_window_closes() -> None:
    feats = build_features(feature_bars(days=1), rth_start_min=_RTH_START_MIN)
    # First bar of the day (09:30) is inside the OR window -> or_high must be null.
    first = feats.sort("ts").row(0, named=True)
    assert first["or_high"] is None
    # A bar after the 30-minute OR window must have a value.
    later = feats.filter(feats["minute_of_day"] >= _RTH_START_MIN + 30)
    assert later["or_high"].null_count() == 0
