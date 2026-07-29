from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tslab.config import DatasetSpec, PathsConfig
from tslab.data.inventory import (render_markdown, scan_dataset, scan_file,
                                  write_reports)


def write_bars(path: Path, start: str, periods: int, freq="1min"):
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    df = pd.DataFrame({
        "ts_event": idx,
        "open": np.linspace(2000, 2001, periods),
        "high": np.linspace(2000.5, 2001.5, periods),
        "low": np.linspace(1999.5, 2000.5, periods),
        "close": np.linspace(2000.2, 2001.2, periods),
        "volume": np.arange(periods, dtype="int64"),
        "symbol": ["GC.v.0"] * periods,
    })
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    return df


@pytest.fixture()
def data_dir(tmp_path):
    d = tmp_path / "gcdata"
    d.mkdir()
    return d


def fake_cfg(data_dir: Path, tmp_path: Path, pattern="*.parquet"):
    ds = DatasetSpec(name="gc.test", root=data_dir, path=data_dir,
                     pattern=pattern, kind="ohlcv_1m", ts_col="ts_event")
    paths = PathsConfig(datasets={"gc.test": ds},
                        outputs_dir=tmp_path / "outputs",
                        full_scan_max_bytes=300_000_000)
    return SimpleNamespace(paths=paths), ds


def test_scan_file_meta_and_full(data_dir, tmp_path):
    write_bars(data_dir / "a.parquet", "2024-01-02 00:00", 600)
    inv = scan_file(data_dir / "a.parquet", "ts_event", 300_000_000)
    assert inv.num_rows == 600
    assert inv.scan_depth == "full"
    assert inv.ts_min is not None and "2024-01-02" in inv.ts_min
    assert inv.duplicate_ts == 0
    assert inv.non_monotonic is False
    assert inv.symbols == {"GC.v.0": 600}


def test_meta_only_when_file_too_big(data_dir):
    write_bars(data_dir / "a.parquet", "2024-01-02 00:00", 600)
    inv = scan_file(data_dir / "a.parquet", "ts_event", full_scan_max_bytes=10)
    assert inv.scan_depth == "meta"
    assert inv.ts_min is not None          # row-group stats still give range
    assert inv.days_present is None
    assert inv.sample_symbols == ["GC.v.0"]


def test_overlap_detection(data_dir, tmp_path):
    # yearly file + monthly extract inside it (the real gc_2024_jan case)
    write_bars(data_dir / "gc_2024.parquet", "2024-01-01 23:00", 5000)
    write_bars(data_dir / "gc_2024_jan.parquet", "2024-01-01 23:00", 1000)
    cfg, ds = fake_cfg(data_dir, tmp_path)
    out = scan_dataset(ds, cfg)
    assert len(out.overlaps) == 1
    names = {out.overlaps[0]["file_a"], out.overlaps[0]["file_b"]}
    assert names == {"gc_2024.parquet", "gc_2024_jan.parquet"}


def test_no_false_overlap(data_dir, tmp_path):
    write_bars(data_dir / "a.parquet", "2024-01-02 00:00", 100)
    write_bars(data_dir / "b.parquet", "2024-03-02 00:00", 100)
    cfg, ds = fake_cfg(data_dir, tmp_path)
    assert scan_dataset(ds, cfg).overlaps == []


def test_duplicate_timestamps_flagged(data_dir):
    df = write_bars(data_dir / "a.parquet", "2024-01-02 00:00", 100)
    dup = pd.concat([df, df.iloc[[50]]], ignore_index=True)
    pq.write_table(pa.Table.from_pandas(dup, preserve_index=False),
                   data_dir / "a.parquet")
    inv = scan_file(data_dir / "a.parquet", "ts_event", 300_000_000)
    assert inv.duplicate_ts == 1


def test_weekday_gap_detection(data_dir):
    # Mon 2024-01-08 and Wed 2024-01-10, skipping Tue 2024-01-09
    a = write_bars(data_dir / "a.parquet", "2024-01-08 15:00", 60)
    b = write_bars(data_dir / "b.parquet", "2024-01-10 15:00", 60)
    both = pd.concat([a, b], ignore_index=True)
    pq.write_table(pa.Table.from_pandas(both, preserve_index=False),
                   data_dir / "a.parquet")
    (data_dir / "b.parquet").unlink()
    inv = scan_file(data_dir / "a.parquet", "ts_event", 300_000_000)
    assert inv.weekday_gaps == ["2024-01-09"]


def test_schema_drift_detected(data_dir, tmp_path):
    write_bars(data_dir / "a.parquet", "2024-01-02 00:00", 50)
    df = write_bars(data_dir / "b.parquet", "2024-02-02 00:00", 50)
    df["extra_col"] = 1.0
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   data_dir / "b.parquet")
    cfg, ds = fake_cfg(data_dir, tmp_path)
    out = scan_dataset(ds, cfg)
    assert not out.schema_consistent
    assert len(out.schema_variants) == 2


def test_readonly_guard_rejects_foreign_paths(data_dir, tmp_path):
    from tslab.data.inventory import _assert_readonly_path
    cfg, _ = fake_cfg(data_dir, tmp_path)
    with pytest.raises(ValueError):
        _assert_readonly_path(tmp_path / "elsewhere.parquet", cfg)


def test_reports_written(data_dir, tmp_path):
    write_bars(data_dir / "a.parquet", "2024-01-02 00:00", 50)
    cfg, ds = fake_cfg(data_dir, tmp_path)
    out = scan_dataset(ds, cfg)
    from dataclasses import asdict
    inv = {"datasets": {"gc.test": asdict(out)}}
    jp, mp = write_reports(inv, tmp_path / "outputs")
    assert jp.exists() and mp.exists()
    md = render_markdown(inv)
    assert "gc.test" in md and "a.parquet" in md


def test_macos_resource_forks_ignored(data_dir, tmp_path):
    write_bars(data_dir / "a.parquet", "2024-01-02 00:00", 50)
    (data_dir / "._a.parquet").write_bytes(b"\x00\x05\x16\x07junk")
    cfg, ds = fake_cfg(data_dir, tmp_path)
    out = scan_dataset(ds, cfg)
    assert [Path(f.path).name for f in out.files] == ["a.parquet"]
