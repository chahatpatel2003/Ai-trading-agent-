"""Read-only discovery of raw Parquet files.

Scans the trades / mbp1 directories, parses the half-yearly period from each
filename, and reads cheap footer metadata (row counts, schema) plus timestamp
coverage. The files are written sorted on the index clock (``ts_recv``), so the
global timestamp bounds are read from the first and last row groups only — a
tiny single-column read, not a full scan. Never opens or modifies row data.

Nanosecond timestamps are decoded from the raw int64 (``Scalar.value``) rather
than via ``Scalar.as_py()``, which refuses ns->microsecond conversion without
pandas (deliberately not a dependency).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from nqlab.data.schema import DatasetKind, ParquetSchema, detect_kind, read_schema

_PERIOD_RE = re.compile(r"(\d{4})_H([12])")
_DEFAULT_TS_COLUMN = "ts_recv"


@dataclass(frozen=True, slots=True, order=True)
class Period:
    """A half-year coverage period, e.g. 2024 H1. Ordered chronologically."""

    year: int
    half: int

    @classmethod
    def parse(cls, name: str) -> Period | None:
        match = _PERIOD_RE.search(name)
        if match is None:
            return None
        return cls(year=int(match.group(1)), half=int(match.group(2)))

    @property
    def label(self) -> str:
        return f"{self.year}_H{self.half}"

    def next(self) -> Period:
        return Period(self.year + 1, 1) if self.half == 2 else Period(self.year, 2)


@dataclass(frozen=True, slots=True)
class ParquetFileInfo:
    """Footer-level facts about one raw Parquet file."""

    dataset: str
    kind: DatasetKind
    path: Path
    period: Period | None
    size_bytes: int
    num_rows: int
    num_row_groups: int
    schema: ParquetSchema
    ts_min: datetime | None
    ts_max: datetime | None


def _pick_ts_column(schema: ParquetSchema) -> str | None:
    if schema.index_columns:
        return schema.index_columns[0]
    if _DEFAULT_TS_COLUMN in schema.names:
        return _DEFAULT_TS_COLUMN
    return None


# Microseconds per one unit of each Arrow timestamp resolution.
_UNIT_TO_MICROS = {"s": 1_000_000.0, "ms": 1_000.0, "us": 1.0, "ns": 0.001}
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _scalar_to_datetime(scalar: object) -> datetime | None:
    """Decode a pyarrow timestamp Scalar from its raw int value (UTC instant)."""
    raw = getattr(scalar, "value", None)
    if not isinstance(raw, int):
        return None
    unit = getattr(getattr(scalar, "type", None), "unit", "ns")
    return _EPOCH + timedelta(microseconds=int(raw * _UNIT_TO_MICROS.get(unit, 0.001)))


def _ts_bounds(
    parquet: pq.ParquetFile, schema: ParquetSchema
) -> tuple[datetime | None, datetime | None]:
    ts_col = _pick_ts_column(schema)
    if ts_col is None or ts_col not in schema.names:
        return None, None
    n_groups = parquet.num_row_groups
    if n_groups == 0:
        return None, None
    try:
        first = parquet.read_row_group(0, columns=[ts_col]).column(0)
        last = parquet.read_row_group(n_groups - 1, columns=[ts_col]).column(0)
    except (OSError, ValueError, KeyError):
        return None, None
    if len(first) == 0 or len(last) == 0:
        return None, None
    return _scalar_to_datetime(pc.min(first)), _scalar_to_datetime(pc.max(last))


def read_file_info(path: str | Path, dataset: str) -> ParquetFileInfo:
    """Read footer metadata + timestamp bounds for one Parquet file (no full scan)."""
    path = Path(path)
    parquet = pq.ParquetFile(str(path))
    metadata = parquet.metadata
    schema = read_schema(path)
    ts_min, ts_max = _ts_bounds(parquet, schema)
    return ParquetFileInfo(
        dataset=dataset,
        kind=detect_kind(schema),
        path=path,
        period=Period.parse(path.name),
        size_bytes=path.stat().st_size,
        num_rows=metadata.num_rows,
        num_row_groups=metadata.num_row_groups,
        schema=schema,
        ts_min=ts_min,
        ts_max=ts_max,
    )


def discover_dataset(directory: str | Path, dataset: str) -> list[ParquetFileInfo]:
    """Discover all ``*.parquet`` files in ``directory`` (sorted by name)."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = sorted(directory.glob("*.parquet"))
    return [read_file_info(path, dataset) for path in files]


def expected_periods(periods: list[Period]) -> list[Period]:
    """Every half-year period between the earliest and latest present period."""
    if not periods:
        return []
    lo, hi = min(periods), max(periods)
    out: list[Period] = []
    cursor = lo
    while cursor <= hi:
        out.append(cursor)
        cursor = cursor.next()
    return out


def missing_periods(periods: list[Period]) -> list[Period]:
    """Gaps within the present range (e.g. mbp1 missing 2025_H2)."""
    present = set(periods)
    return [p for p in expected_periods(periods) if p not in present]
