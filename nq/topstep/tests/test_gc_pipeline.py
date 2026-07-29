from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tslab.audit.leakage import Status, TruncationInvarianceCheck
from tslab.audit.pipeline_gate import BarIntegrityCheck
from tslab.data.gc_pipeline import gc_bars_from_ohlcv
from tslab.features.core import FeatureParams, compute_features, feature_columns


def synth_gc_raw(start_utc: str, periods: int, px0=2050.0, seed=9):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start_utc, periods=periods, freq="1min", tz="UTC")
    close = px0 + np.cumsum(rng.normal(0, 0.2, periods))
    opn = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "open": opn, "high": np.maximum(opn, close) + 0.1,
        "low": np.minimum(opn, close) - 0.1, "close": close,
        "volume": rng.integers(1, 400, periods).astype("uint64"),
        "symbol": "GC.v.0",
    }, index=pd.DatetimeIndex(idx, name="bar_open_utc"))


def test_gc_adapter_schema_and_gate(cfg):
    raw = synth_gc_raw("2024-01-09 00:00", 600)          # Tue session, CST
    out = gc_bars_from_ohlcv(raw, cfg, date(2024, 1, 9), date(2024, 1, 9))
    assert not out.empty
    for col in ("open", "high", "low", "close", "volume", "session", "symbol",
                "info_time", "secs_to_flatten", "entry_ok", "rth"):
        assert col in out.columns
    assert (out["session"] == date(2024, 1, 9)).all()
    res = BarIntegrityCheck(out, cfg).run()
    assert res.status == Status.PASS, res.message


def test_gc_holiday_bars_book_to_next_session(cfg):
    # MLK Monday 2024 morning bars (GC really traded until 13:30 CT) must
    # book to Tuesday, and Monday must never appear as a session
    raw = synth_gc_raw("2024-01-15 14:00", 240)          # Mon 08:00 CT ..
    out = gc_bars_from_ohlcv(raw, cfg, date(2024, 1, 16), date(2024, 1, 16))
    assert not out.empty
    assert set(out["session"]) == {date(2024, 1, 16)}
    assert BarIntegrityCheck(out, cfg).run().status == Status.PASS


def test_gc_range_filter_excludes_other_sessions(cfg):
    raw = pd.concat([synth_gc_raw("2024-01-09 00:00", 300),
                     synth_gc_raw("2024-01-10 00:00", 300)])
    out = gc_bars_from_ohlcv(raw, cfg, date(2024, 1, 9), date(2024, 1, 9))
    assert set(out["session"]) == {date(2024, 1, 9)}


def test_gc_features_same_schema_with_nan_micro(cfg):
    raw = pd.concat([synth_gc_raw("2024-01-09 00:00", 400),
                     synth_gc_raw("2024-01-10 00:00", 400)])
    bars = gc_bars_from_ohlcv(raw, cfg, date(2024, 1, 9), date(2024, 1, 10))
    feats = compute_features(bars, cfg, FeatureParams())
    assert feats["f_delta_15"].isna().all()              # no microstructure
    assert feats["f_spread_ratio"].isna().all()
    assert feats["f_ret_5"].notna().sum() > 0            # price features live
    res = TruncationInvarianceCheck(
        bars, lambda df: compute_features(df, cfg, FeatureParams()),
        n_probes=4).run()
    assert res.status == Status.PASS, res.message


GC_DIR = Path("/Volumes/T9/gc")
needs_drive = pytest.mark.skipif(not GC_DIR.exists(),
                                 reason="T9 drive not mounted")


@needs_drive
def test_gc_real_build_readonly_and_gated(cfg):
    from tslab.data.gc_pipeline import build_gc_bars
    raw_files = sorted(GC_DIR.glob("gc_*.parquet"))
    before = {p: (p.stat().st_size, p.stat().st_mtime) for p in raw_files}
    bars = build_gc_bars(cfg, date(2024, 3, 4), date(2024, 3, 6))
    after = {p: (p.stat().st_size, p.stat().st_mtime) for p in raw_files}
    assert before == after                               # read-only
    assert len(bars) > 3000                              # 3 near-full sessions
    assert set(bars["session"]) == {date(2024, 3, 4), date(2024, 3, 5),
                                    date(2024, 3, 6)}
    assert BarIntegrityCheck(bars, cfg).run().status == Status.PASS
    feats = compute_features(bars, cfg, FeatureParams())
    assert len(feature_columns(feats)) > 30
