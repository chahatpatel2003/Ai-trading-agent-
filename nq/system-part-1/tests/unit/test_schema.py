from __future__ import annotations

from pathlib import Path

from nqlab.data.schema import DatasetKind, detect_kind, read_schema
from tests.fixtures import write_mbp1_parquet, write_trades_parquet


def test_detect_trades(tmp_path: Path) -> None:
    path = write_trades_parquet(tmp_path / "t.parquet")
    schema = read_schema(path)
    assert detect_kind(schema) == DatasetKind.TRADES
    assert "price" in schema.names
    assert "bid_px_00" not in schema.names


def test_detect_mbp1(tmp_path: Path) -> None:
    path = write_mbp1_parquet(tmp_path / "m.parquet")
    schema = read_schema(path)
    assert detect_kind(schema) == DatasetKind.MBP1
    assert "bid_px_00" in schema.names
    assert "ask_px_00" in schema.names


def test_fingerprint_stable_and_distinct(tmp_path: Path) -> None:
    t = write_trades_parquet(tmp_path / "t.parquet")
    m = write_mbp1_parquet(tmp_path / "m.parquet")
    f_trades = read_schema(t).fingerprint()
    assert f_trades == read_schema(t).fingerprint()  # stable
    assert f_trades != read_schema(m).fingerprint()  # distinct kinds differ
    assert len(f_trades) == 16


def test_schema_to_dict(tmp_path: Path) -> None:
    schema = read_schema(write_trades_parquet(tmp_path / "t.parquet"))
    as_dict = schema.to_dict()
    assert as_dict["fingerprint"] == schema.fingerprint()
    assert isinstance(as_dict["columns"], list)
