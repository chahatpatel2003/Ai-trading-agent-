"""Tiny synthetic Parquet writers so tests never touch the external SSD.

The schemas mirror the real Databento ``trades`` and ``mbp1`` layouts closely
enough to exercise discovery, schema detection, and instrument sampling.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_BASE = datetime(2024, 1, 1, tzinfo=UTC)
_SYMBOL_PATTERN = ("NQH4", "NQH4", "NQH4-NQM4")  # 2 outright : 1 spread


def _timestamps(n: int) -> list[datetime]:
    return [_BASE + timedelta(seconds=i) for i in range(n)]


def _symbols(n: int) -> list[str]:
    return [_SYMBOL_PATTERN[i % len(_SYMBOL_PATTERN)] for i in range(n)]


def _ts_array(ts: list[datetime]) -> pa.Array:
    return pa.array(ts, type=pa.timestamp("ns", tz="UTC"))


def write_trades_parquet(path: str | Path, n: int = 12) -> Path:
    ts = _timestamps(n)
    table = pa.table(
        {
            "ts_event": _ts_array(ts),
            "ts_recv": _ts_array(ts),
            "action": ["T"] * n,
            "side": (["A", "B"] * n)[:n],
            "price": [17000.0 + i * 0.25 for i in range(n)],
            "size": list(range(1, n + 1)),
            "sequence": list(range(n)),
            "instrument_id": [750] * n,
            "symbol": _symbols(n),
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))
    return path


def write_mbp1_parquet(path: str | Path, n: int = 12) -> Path:
    ts = _timestamps(n)
    table = pa.table(
        {
            "ts_event": _ts_array(ts),
            "ts_recv": _ts_array(ts),
            "action": (["A", "M", "C", "T"] * n)[:n],
            "side": (["A", "B"] * n)[:n],
            "price": [17000.0 + i * 0.25 for i in range(n)],
            "size": list(range(1, n + 1)),
            "sequence": list(range(n)),
            "instrument_id": [750] * n,
            "bid_px_00": [16999.75 + i * 0.25 for i in range(n)],
            "ask_px_00": [17000.25 + i * 0.25 for i in range(n)],
            "bid_sz_00": list(range(1, n + 1)),
            "ask_sz_00": list(range(1, n + 1)),
            "symbol": _symbols(n),
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))
    return path
