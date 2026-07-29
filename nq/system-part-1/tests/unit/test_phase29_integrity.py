"""Phase 29 — integrity for the NEW factory layers.

(1) Prefix-invariance (no hindsight) for the classic-indicator + Supertrend columns added by
    ``add_phase29_indicators`` and for a representative family signal: a value at bar k computed on
    the FULL series must equal the value computed on the prefix [0..k].
(2) The EOD-flat guarantee: ``intraday_only`` drops any trade whose exit falls on a different CME
    trading day than its entry (early-close holidays where the 15:55 force-flat can't fire).

The underlying feature layers (structure/htf/daily_bias/order-flow) and the engine were already
audited in Phase 27; this covers only what Phase 29 adds.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from nqlab.discovery.phase29 import (
    add_phase29_indicators,
    build_signal_frame,
    generate_candidates,
    intraday_only,
)
from nqlab.features.build import build_features
from nqlab.features.context import build_context_features
from nqlab.features.daily_bias import add_daily_bias
from nqlab.features.htf_context import build_htf_context_features
from nqlab.features.htf_poi import add_htf_poi
from nqlab.features.structure import build_structure_features
from tests.synth import globex_bars

_RTH = 570
_IND_COLS = [
    "ema9",
    "ema21",
    "ema50",
    "_macd",
    "_macd_sig",
    "_macd_hist",
    "_stoch_k",
    "_stoch_d",
    "_bb_up",
    "_bb_dn",
    "_bb_bw",
    "_bb_bw_med",
    "_kc_up",
    "_kc_dn",
    "_dc_up",
    "_dc_dn",
    "adx",
    "plus_di",
    "minus_di",
    "st_dir",
    "vwd",
    "ph",
    "pl",
]


def _full_frame(bars: pl.DataFrame) -> pl.DataFrame:
    f = build_features(bars, rth_start_min=_RTH)
    f = build_context_features(f, rth_start_min=_RTH)
    f = build_htf_context_features(f, rth_start_min=_RTH, timeframe="5m")
    f = build_structure_features(f)
    f = add_daily_bias(add_htf_poi(f, tol_pts=5.0))
    return add_phase29_indicators(f)


def test_indicators_no_lookahead() -> None:
    bars = globex_bars(days=8)
    full = add_phase29_indicators(
        build_structure_features(
            build_htf_context_features(
                build_context_features(
                    build_features(bars, rth_start_min=_RTH), rth_start_min=_RTH
                ),
                rth_start_min=_RTH,
                timeframe="5m",
            )
        )
    )
    n = full.height
    picks = sorted({int(f * (n - 1)) for f in (0.4, 0.55, 0.7, 0.85, 0.95)})
    for k in picks:
        prefix = add_phase29_indicators(
            build_structure_features(
                build_htf_context_features(
                    build_context_features(
                        build_features(bars.head(k + 1), rth_start_min=_RTH), rth_start_min=_RTH
                    ),
                    rth_start_min=_RTH,
                    timeframe="5m",
                )
            )
        )
        fr = full.row(k, named=True)
        pr = prefix.row(k, named=True)
        for c in _IND_COLS:
            a, b = fr[c], pr[c]
            if a is None and b is None:
                continue
            assert a == pytest.approx(b, rel=1e-9, abs=1e-9), f"HINDSIGHT in {c} @k={k}: {a}!={b}"


def test_supertrend_is_causal_direction() -> None:
    """st_dir only ever takes values in {-1, +1} and equals prefix-computed value (covered above);
    here we additionally assert it is well-formed (no nulls, only +/-1)."""
    full = _full_frame(globex_bars(days=6))
    vals = set(full["st_dir"].unique().to_list())
    assert vals.issubset({-1, 1})


def test_family_signal_no_lookahead() -> None:
    bars = globex_bars(days=8)
    cand = next(c for c in generate_candidates() if c.family == "ema_cross_9_21")
    full = build_signal_frame(_full_frame(bars), cand).with_row_index("idx")
    # signal frame is filtered to [0,960]; compare by ts to be index-safe.
    n = full.height
    picks = sorted({int(f * (n - 1)) for f in (0.5, 0.7, 0.9)})
    for k in picks:
        ts_k = full["ts"][k]
        prefix_full = _full_frame(bars).filter(pl.col("ts") <= ts_k)
        prefix_sig = build_signal_frame(prefix_full, cand)
        a = full.filter(pl.col("ts") == ts_k)["signal"][0]
        b = prefix_sig.filter(pl.col("ts") == ts_k)["signal"][0]
        assert a == b, f"signal hindsight @ts={ts_k}: {a}!={b}"


def test_all_families_reference_only_known_columns() -> None:
    """Every family's expressions must resolve against the built frame (no typo'd columns)."""
    frame = _full_frame(globex_bars(days=6))
    for cand in generate_candidates():
        sig = build_signal_frame(frame, cand)
        assert "signal" in sig.columns
        assert set(sig["signal"].unique().to_list()).issubset({-1, 0, 1})


def test_intraday_only_enforces_eod_flat() -> None:
    """A trade whose exit is on a later trading day than entry is dropped (early-close guard)."""
    td = date(2024, 11, 29)  # day-after-Thanksgiving (early close 13:00 ET)
    trades = pl.DataFrame(
        {
            "trading_day": [td, td],
            # 12:45 ET same day (17:45 UTC) -> +6h -> 2024-11-29 == trading_day: KEEP
            # next session 00:05 ET 2024-12-02 -> +6h -> 2024-12-02 != trading_day: DROP
            "exit_ts": [
                datetime(2024, 11, 29, 17, 45, tzinfo=UTC),
                datetime(2024, 12, 2, 5, 5, tzinfo=UTC),
            ],
            "net": [10.0, -50.0],
        }
    )
    kept, dropped = intraday_only(trades)
    assert dropped == 1
    assert kept.height == 1
    assert kept["net"][0] == 10.0
