from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tslab.data.loader import (load_gc_bars, load_nq_mbp1, load_nq_trades,
                               outright_mask)

GC_DIR = Path("/Volumes/T9/gc")

needs_drive = pytest.mark.skipif(not GC_DIR.exists(),
                                 reason="T9 drive not mounted")


def test_outright_mask():
    s = pd.Series(["NQH4", "NQH4-NQM4", "NQZ6", "NQM4-NQU4", "GC.v.0"])
    assert outright_mask(s).tolist() == [True, False, True, False, True]


def test_nq_loaders_are_explicit_stubs(cfg):
    with pytest.raises(NotImplementedError):
        load_nq_trades(cfg)
    with pytest.raises(NotImplementedError):
        load_nq_mbp1(cfg)


@needs_drive
def test_load_gc_bars_clean(cfg):
    df = load_gc_bars(
        cfg,
        start=pd.Timestamp("2024-03-01", tz="UTC"),
        end=pd.Timestamp("2024-03-08", tz="UTC"),
    )
    assert len(df) > 1000
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()
    assert (df["info_time"] - df.index == pd.Timedelta(minutes=1)).all()
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns
    assert (df["high"] >= df["low"]).all()
