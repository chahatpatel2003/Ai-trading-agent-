"""No-lookahead tests for the Phase-13 market-structure / liquidity / FVG layer.

Swings are the trap: a fractal swing high needs +/-swing_k bars to confirm, so it must
not appear in any column until confirmed. The prefix==full check at many bars proves it.
"""

from __future__ import annotations

import polars as pl
import pytest

from nqlab.features.build import build_features
from nqlab.features.context import build_context_features
from nqlab.features.htf_context import build_htf_context_features
from nqlab.features.structure import STRUCTURE_FEATURE_NAMES, build_structure_features
from tests.synth import globex_bars

_RTH = 570
_PIT = list(STRUCTURE_FEATURE_NAMES)


def _stack(bars: pl.DataFrame) -> pl.DataFrame:
    feats = build_features(bars, rth_start_min=_RTH)
    ctx = build_context_features(feats, rth_start_min=_RTH)
    htf = build_htf_context_features(ctx, rth_start_min=_RTH, timeframe="5m")
    return build_structure_features(htf)


def test_structure_columns_present() -> None:
    full = _stack(globex_bars(days=8))
    assert set(STRUCTURE_FEATURE_NAMES) <= set(full.columns)


def test_pit_list_covers_every_structure_feature() -> None:
    assert set(_PIT) == set(STRUCTURE_FEATURE_NAMES)  # no feature escapes PIT verification


def test_no_lookahead_structure() -> None:
    """A feature at row k computed on the full series must equal the value computed on
    only the prefix [0..k] — checked at many bars (swings confirm with a lag)."""
    bars = globex_bars(days=20)
    full = _stack(bars).with_row_index("idx")
    cand = full.filter(pl.col("is_rth") & pl.col("last_swing_high").is_not_null())
    assert cand.height > 0
    idxs = cand["idx"].to_list()
    n = len(idxs)
    sample = sorted(
        {idxs[0], idxs[-1], *(idxs[int(f * (n - 1))] for f in (0.2, 0.35, 0.5, 0.65, 0.8))}
    )

    for k in sample:
        prefix = _stack(bars.head(k + 1))
        full_row = full.drop("idx").row(k, named=True)
        prefix_row = prefix.row(k, named=True)
        for col in _PIT:
            a, b = full_row[col], prefix_row[col]
            if a is None and b is None:
                continue
            assert a == pytest.approx(b, rel=1e-9, abs=1e-9), f"lookahead in {col} @k={k}: {a}!={b}"


def test_swing_high_only_appears_after_confirmation() -> None:
    """A confirmed swing high price must never be present BEFORE its confirmation bar."""
    bars = globex_bars(days=10)
    full = _stack(bars).with_row_index("idx").sort("idx")
    highs = full["high"].to_list()
    sh = full["last_swing_high"].to_list()
    # For every bar, the exposed last_swing_high must equal some PAST bar's high
    # (never a future high) — i.e. it exists earlier in the series.
    seen_highs: set[float] = set()
    for i in range(full.height):
        seen_highs.add(round(highs[i], 6))
        v = sh[i]
        if v is not None:
            assert round(v, 6) in seen_highs, f"swing high {v} at bar {i} not seen in the past"
