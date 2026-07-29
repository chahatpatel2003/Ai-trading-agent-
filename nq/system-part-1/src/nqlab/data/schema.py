"""Parquet schema reading, fingerprinting, and dataset-kind detection.

Reads only the Parquet footer (schema + metadata) — never scans data rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True, slots=True)
class ColumnSchema:
    name: str
    arrow_type: str


@dataclass(frozen=True, slots=True)
class ParquetSchema:
    """A normalized, hashable view of a Parquet file's schema."""

    columns: tuple[ColumnSchema, ...]
    index_columns: tuple[str, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def fingerprint(self) -> str:
        """Stable short hash of (name, type) pairs — detects schema drift."""
        payload = [[c.name, c.arrow_type] for c in self.columns]
        blob = json.dumps(payload, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        return {
            "fingerprint": self.fingerprint(),
            "index_columns": list(self.index_columns),
            "columns": [{"name": c.name, "type": c.arrow_type} for c in self.columns],
        }


class DatasetKind(StrEnum):
    TRADES = "trades"
    MBP1 = "mbp1"
    UNKNOWN = "unknown"


_MBP1_MARKERS = ("bid_px_00", "ask_px_00")
_TRADE_MARKERS = ("price", "action", "size")


def detect_kind(schema: ParquetSchema) -> DatasetKind:
    """Classify a parquet schema as trades, mbp1, or unknown by its columns."""
    names = set(schema.names)
    if all(marker in names for marker in _MBP1_MARKERS):
        return DatasetKind.MBP1
    if all(marker in names for marker in _TRADE_MARKERS) and "bid_px_00" not in names:
        return DatasetKind.TRADES
    return DatasetKind.UNKNOWN


def _index_columns(arrow_schema: pa.Schema) -> tuple[str, ...]:
    metadata = arrow_schema.metadata or {}
    raw = metadata.get(b"pandas")
    if not raw:
        return ()
    try:
        info = json.loads(raw)
    except (ValueError, TypeError):
        return ()
    cols = info.get("index_columns", [])
    return tuple(c for c in cols if isinstance(c, str))


def read_schema(path: str | Path) -> ParquetSchema:
    """Read the schema (footer only) of a Parquet file."""
    parquet = pq.ParquetFile(str(path))
    arrow_schema = parquet.schema_arrow
    columns = tuple(
        ColumnSchema(name=field.name, arrow_type=str(field.type)) for field in arrow_schema
    )
    return ParquetSchema(columns=columns, index_columns=_index_columns(arrow_schema))
