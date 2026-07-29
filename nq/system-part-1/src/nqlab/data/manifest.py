"""Data-availability manifest.

Combines read-only discovery with a bounded instrument sample to produce a single
JSON artifact describing: instruments present, file paths, date coverage, row
counts, schemas, missing periods, and the trades-vs-mbp1 availability matrix.

Instrument enumeration is **sampled** (first N row groups of one file per
dataset) so the manifest stays fast; a full distinct scan over ~8B rows is a
later, explicit operation.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from nqlab.config.models import AppConfig
from nqlab.data.discovery import (
    ParquetFileInfo,
    discover_dataset,
    missing_periods,
)
from nqlab.domain.instruments.symbology import (
    ContractError,
    Outright,
    classify,
    parse_symbol,
)


@dataclass(frozen=True, slots=True)
class InstrumentObservation:
    symbol: str
    classification: str  # "outright" | "spread"
    root: str | None
    month_code: str | None
    year: int | None
    sampled_occurrences: int


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    name: str
    directory: str
    present_periods: list[str]
    missing_periods: list[str]
    file_count: int
    total_rows: int
    total_bytes: int
    schema_fingerprints: list[str]
    ts_min: str | None
    ts_max: str | None
    files: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class AvailabilityManifest:
    generated_at: str
    config_hash: str
    data_root: str
    canonical_clock: str
    sampled_instruments: bool
    datasets: dict[str, DatasetManifest]
    availability_matrix: list[dict[str, object]]
    instruments: list[InstrumentObservation] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "config_hash": self.config_hash,
            "data_root": self.data_root,
            "canonical_clock": self.canonical_clock,
            "sampled_instruments": self.sampled_instruments,
            "datasets": {name: asdict(ds) for name, ds in self.datasets.items()},
            "availability_matrix": self.availability_matrix,
            "instruments": [asdict(obs) for obs in self.instruments],
        }

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _file_to_dict(info: ParquetFileInfo) -> dict[str, object]:
    return {
        "path": str(info.path),
        "period": info.period.label if info.period else None,
        "kind": info.kind.value,
        "size_bytes": info.size_bytes,
        "num_rows": info.num_rows,
        "num_row_groups": info.num_row_groups,
        "schema_fingerprint": info.schema.fingerprint(),
        "ts_min": _iso(info.ts_min),
        "ts_max": _iso(info.ts_max),
    }


def _build_dataset_manifest(
    name: str, directory: Path, infos: list[ParquetFileInfo]
) -> DatasetManifest:
    periods = sorted({i.period for i in infos if i.period is not None})
    fingerprints = sorted({i.schema.fingerprint() for i in infos})
    ts_mins = [i.ts_min for i in infos if i.ts_min is not None]
    ts_maxs = [i.ts_max for i in infos if i.ts_max is not None]
    return DatasetManifest(
        name=name,
        directory=str(directory),
        present_periods=[p.label for p in periods],
        missing_periods=[p.label for p in missing_periods(periods)],
        file_count=len(infos),
        total_rows=sum(i.num_rows for i in infos),
        total_bytes=sum(i.size_bytes for i in infos),
        schema_fingerprints=fingerprints,
        ts_min=_iso(min(ts_mins)) if ts_mins else None,
        ts_max=_iso(max(ts_maxs)) if ts_maxs else None,
        files=[_file_to_dict(i) for i in infos],
    )


def _sample_instruments(infos: list[ParquetFileInfo], n_row_groups: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not infos:
        return counts
    info = infos[0]
    if "symbol" not in info.schema.names:
        return counts
    parquet = pq.ParquetFile(str(info.path))
    limit = min(n_row_groups, parquet.num_row_groups)
    if limit <= 0:
        return counts
    table = parquet.read_row_groups(list(range(limit)), columns=["symbol"])
    for symbol in table.column("symbol").to_pylist():
        if symbol is not None:
            counts[str(symbol)] += 1
    return counts


def _observe(counts: Counter[str], year_base: int) -> list[InstrumentObservation]:
    observations: list[InstrumentObservation] = []
    for symbol, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        kind = classify(symbol)
        root: str | None = None
        month_code: str | None = None
        year: int | None = None
        if kind == "outright":
            try:
                contract = parse_symbol(symbol, year_base=year_base)
            except ContractError:
                kind = "unknown"
            else:
                # parse_symbol returns an Outright for non-spread input.
                if isinstance(contract, Outright):
                    root = contract.root
                    month_code = contract.month_code
                    year = contract.year
        observations.append(
            InstrumentObservation(
                symbol=symbol,
                classification=kind,
                root=root,
                month_code=month_code,
                year=year,
                sampled_occurrences=n,
            )
        )
    return observations


def build_manifest(config: AppConfig, *, sample_instruments: bool = True) -> AvailabilityManifest:
    """Build the data-availability manifest (read-only)."""
    paths = config.data
    dataset_dirs: dict[str, Path] = {
        "trades": paths.raw_trades_dir,
        "mbp1": paths.raw_mbp1_dir,
    }

    datasets: dict[str, DatasetManifest] = {}
    period_sets: dict[str, set[str]] = {}
    all_infos: dict[str, list[ParquetFileInfo]] = {}
    for name, directory in dataset_dirs.items():
        infos = discover_dataset(directory, name)
        all_infos[name] = infos
        datasets[name] = _build_dataset_manifest(name, directory, infos)
        period_sets[name] = {i.period.label for i in infos if i.period is not None}

    # Availability matrix over the union of observed periods, chronologically.
    union_periods = sorted(
        {p for infos in all_infos.values() for i in infos if (p := i.period) is not None}
    )
    matrix = [
        {
            "period": period.label,
            "trades": period.label in period_sets.get("trades", set()),
            "mbp1": period.label in period_sets.get("mbp1", set()),
        }
        for period in union_periods
    ]

    instruments: list[InstrumentObservation] = []
    if sample_instruments:
        year_base = config.instruments.year_base
        n_rg = config.compute.manifest_sample_row_groups
        counts: Counter[str] = Counter()
        for infos in all_infos.values():
            counts.update(_sample_instruments(infos, n_rg))
        instruments = _observe(counts, year_base)

    return AvailabilityManifest(
        generated_at=datetime.now(UTC).isoformat(),
        config_hash=config.config_hash(),
        data_root=str(paths.nq_data_root),
        canonical_clock=config.clock.canonical.value,
        sampled_instruments=sample_instruments,
        datasets=datasets,
        availability_matrix=matrix,
        instruments=instruments,
    )
