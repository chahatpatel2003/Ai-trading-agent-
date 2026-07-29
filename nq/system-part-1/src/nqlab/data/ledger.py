"""Partition completion ledger for incremental, resumable materialization.

A ledger is a small JSON file recording which month partitions are *complete*,
with the config hash they were built under and a row count for corruption
checks. Writes are atomic (temp file + replace) so an interrupted run never
leaves a half-written ledger, which is what makes resume safe.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class LedgerEntry:
    status: str
    rows: int
    config_hash: str
    created_at: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "rows": self.rows,
            "config_hash": self.config_hash,
            "created_at": self.created_at,
            **self.extra,
        }


class PartitionLedger:
    """JSON-backed registry of completed partitions keyed by month (``YYYYMM``)."""

    def __init__(self, path: Path, entries: dict[str, LedgerEntry] | None = None) -> None:
        self.path = path
        self.entries: dict[str, LedgerEntry] = entries or {}

    @classmethod
    def load(cls, path: str | Path) -> PartitionLedger:
        path = Path(path)
        if not path.exists():
            return cls(path)
        try:
            raw = json.loads(path.read_text())
        except (ValueError, OSError):
            return cls(path)  # treat unreadable ledger as empty (will rebuild)
        entries: dict[str, LedgerEntry] = {}
        for key, value in raw.get("entries", {}).items():
            reserved = {"status", "rows", "config_hash", "created_at"}
            extra = {k: v for k, v in value.items() if k not in reserved}
            entries[key] = LedgerEntry(
                status=value.get("status", "unknown"),
                rows=int(value.get("rows", 0)),
                config_hash=value.get("config_hash", ""),
                created_at=value.get("created_at", ""),
                extra=extra,
            )
        return cls(path, entries)

    def is_complete(self, key: str, config_hash: str) -> bool:
        entry = self.entries.get(key)
        return (
            entry is not None
            and entry.status == "complete"
            and entry.config_hash == config_hash
        )

    def get(self, key: str) -> LedgerEntry | None:
        return self.entries.get(key)

    def mark_complete(
        self, key: str, *, rows: int, config_hash: str, **extra: Any
    ) -> None:
        self.entries[key] = LedgerEntry(
            status="complete",
            rows=rows,
            config_hash=config_hash,
            created_at=datetime.now(UTC).isoformat(),
            extra=extra,
        )

    def remove(self, key: str) -> None:
        self.entries.pop(key, None)

    def completed_keys(self) -> list[str]:
        return sorted(k for k, e in self.entries.items() if e.status == "complete")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": {k: e.to_dict() for k, e in self.entries.items()}}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.path)
