"""Read-only parquet data inventory.

Scans every dataset declared in paths.yaml and produces a machine-readable
JSON inventory plus a human-readable markdown report:

  - file-level: size, rows, row groups, schema, writer, timestamp range
  - timestamp ranges come from parquet row-group STATISTICS (metadata only) —
    no data pages are read unless the file is small enough for a full scan
  - full scan (small files): exact per-day coverage, duplicate timestamps,
    monotonicity, NaN counts, symbol distribution, weekday-gap detection
  - dataset-level: schema drift across files, time-range overlaps between
    files (e.g. a monthly extract duplicating part of a yearly file)

READ-ONLY GUARANTEE: this module opens data files exclusively through
pyarrow.parquet readers and os.stat. It never opens a data path for writing;
outputs go to the configured outputs_dir, which config validation guarantees
is outside every data root.
"""
from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

from ..config import Config, DatasetSpec


@dataclass
class FileInventory:
    path: str
    size_bytes: int
    num_rows: int
    num_row_groups: int
    created_by: str
    columns: Dict[str, str]              # name -> arrow type
    ts_col: str
    ts_min: Optional[str] = None
    ts_max: Optional[str] = None
    scan_depth: str = "meta"             # "meta" | "full"
    # full-scan extras
    days_present: Optional[int] = None
    weekday_gaps: Optional[List[str]] = None      # weekdays with zero rows inside range
    duplicate_ts: Optional[int] = None
    non_monotonic: Optional[bool] = None
    nan_counts: Optional[Dict[str, int]] = None
    symbols: Optional[Dict[str, int]] = None
    sample_symbols: Optional[List[str]] = None    # cheap first-batch peek (meta mode)
    error: Optional[str] = None


@dataclass
class DatasetInventory:
    name: str
    kind: str
    directory: str
    files: List[FileInventory] = field(default_factory=list)
    schema_consistent: bool = True
    schema_variants: List[List[str]] = field(default_factory=list)
    overlaps: List[Dict[str, str]] = field(default_factory=list)
    ts_min: Optional[str] = None
    ts_max: Optional[str] = None
    total_rows: int = 0
    total_bytes: int = 0


def _assert_readonly_path(p: Path, cfg: Config) -> None:
    """Defense in depth: refuse to even construct an inventory entry for a
    path we might be asked to write to."""
    if not any(_is_under(p, root) for root in cfg.paths.data_roots()):
        raise ValueError(f"{p} is not under a declared data root")


def _is_under(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _list_files(ds: DatasetSpec) -> List[Path]:
    if not ds.path.exists():
        return []
    out = [p for p in sorted(ds.path.iterdir())
           if p.is_file() and fnmatch.fnmatch(p.name, ds.pattern)
           and not p.name.startswith("._")]          # macOS resource forks
    return out


def scan_file(path: Path, ts_col: str, full_scan_max_bytes: int) -> FileInventory:
    st = os.stat(path)
    inv = FileInventory(
        path=str(path), size_bytes=st.st_size, num_rows=0, num_row_groups=0,
        created_by="", columns={}, ts_col=ts_col,
    )
    try:
        pf = pq.ParquetFile(path)
        md = pf.metadata
        inv.num_rows = md.num_rows
        inv.num_row_groups = md.num_row_groups
        inv.created_by = md.created_by or ""
        inv.columns = {f.name: str(f.type) for f in pf.schema_arrow}

        if ts_col in inv.columns:
            inv.ts_min, inv.ts_max = _ts_range_from_rg_stats(md, ts_col)

        if st.st_size <= full_scan_max_bytes:
            _full_scan(pf, ts_col, inv)
            inv.scan_depth = "full"
        else:
            inv.sample_symbols = _sample_symbols(pf)
    except Exception as e:  # keep scanning other files
        inv.error = f"{type(e).__name__}: {e}"
    return inv


def _ts_range_from_rg_stats(md, ts_col: str) -> Tuple[Optional[str], Optional[str]]:
    """Min/max of the ts column across ALL row groups, from metadata stats only.
    Row groups are not assumed to be time-ordered."""
    ci = None
    for i in range(md.num_columns):
        if md.schema.column(i).name == ts_col:
            ci = i
            break
    if ci is None:
        return None, None
    lo = hi = None
    for rg in range(md.num_row_groups):
        stats = md.row_group(rg).column(ci).statistics
        if stats is None or not stats.has_min_max:
            continue
        if lo is None or stats.min < lo:
            lo = stats.min
        if hi is None or stats.max > hi:
            hi = stats.max
    return (str(lo) if lo is not None else None,
            str(hi) if hi is not None else None)


def _sample_symbols(pf: pq.ParquetFile) -> Optional[List[str]]:
    if "symbol" not in [f.name for f in pf.schema_arrow]:
        return None
    try:
        batch = next(pf.iter_batches(batch_size=8192, columns=["symbol"]))
        return sorted(set(batch.column(0).to_pylist()))[:50]
    except StopIteration:
        return None


def _full_scan(pf: pq.ParquetFile, ts_col: str, inv: FileInventory) -> None:
    import pandas as pd
    cols = [c for c in [ts_col, "symbol", "open", "high", "low", "close", "volume"]
            if c in inv.columns]
    tbl = pf.read(columns=cols)
    df = tbl.to_pandas()
    # ts may be a column or the stored index
    if ts_col in df.columns:
        ts = pd.DatetimeIndex(df[ts_col])
    elif df.index.name == ts_col:
        ts = pd.DatetimeIndex(df.index)
    else:
        return
    inv.duplicate_ts = int(ts.duplicated().sum())
    inv.non_monotonic = bool(not ts.is_monotonic_increasing)
    inv.nan_counts = {c: int(df[c].isna().sum())
                      for c in df.columns if df[c].isna().any()}
    if "symbol" in df.columns:
        inv.symbols = {str(k): int(v)
                       for k, v in df["symbol"].value_counts().head(50).items()}
    days = sorted(set(ts.tz_convert("America/Chicago").date))
    inv.days_present = len(days)
    if days:
        gaps = []
        d = days[0]
        present = set(days)
        while d <= days[-1]:
            if d.weekday() < 5 and d not in present:   # weekday missing entirely
                gaps.append(d.isoformat())
            d += timedelta(days=1)
        inv.weekday_gaps = gaps[:100]                  # cap report size


def scan_dataset(ds: DatasetSpec, cfg: Config) -> DatasetInventory:
    out = DatasetInventory(name=ds.name, kind=ds.kind, directory=str(ds.path))
    for f in _list_files(ds):
        _assert_readonly_path(f, cfg)
        out.files.append(scan_file(f, ds.ts_col, cfg.paths.full_scan_max_bytes))

    good = [f for f in out.files if f.error is None]
    schemas = []
    for f in good:
        key = sorted(f"{k}:{v}" for k, v in f.columns.items())
        if key not in schemas:
            schemas.append(key)
    out.schema_variants = schemas
    out.schema_consistent = len(schemas) <= 1
    out.total_rows = sum(f.num_rows for f in good)
    out.total_bytes = sum(f.size_bytes for f in out.files)
    # compare as timestamps, never as strings (sub-second digit counts vary)
    import pandas as pd
    ts = pd.Timestamp
    mins = [f.ts_min for f in good if f.ts_min]
    maxs = [f.ts_max for f in good if f.ts_max]
    out.ts_min = min(mins, key=ts) if mins else None
    out.ts_max = max(maxs, key=ts) if maxs else None

    # pairwise time-range overlap detection (parsed comparisons)
    ranged = [(f.path, ts(f.ts_min), ts(f.ts_max))
              for f in good if f.ts_min and f.ts_max]
    for i in range(len(ranged)):
        for j in range(i + 1, len(ranged)):
            a, b = ranged[i], ranged[j]
            if a[1] <= b[2] and b[1] <= a[2]:
                out.overlaps.append({
                    "file_a": Path(a[0]).name, "file_b": Path(b[0]).name,
                    "overlap_start": str(max(a[1], b[1])),
                    "overlap_end": str(min(a[2], b[2])),
                })
    return out


def scan_all(cfg: Config) -> Dict[str, Any]:
    inventories = {name: scan_dataset(ds, cfg)
                   for name, ds in cfg.paths.datasets.items()}
    return {
        "datasets": {k: asdict(v) for k, v in inventories.items()},
    }


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #

def write_reports(inventory: Dict[str, Any], outputs_dir: Path,
                  stamp: str = "latest") -> Tuple[Path, Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    jpath = outputs_dir / f"inventory_{stamp}.json"
    mpath = outputs_dir / f"inventory_{stamp}.md"
    with open(jpath, "w") as f:
        json.dump(inventory, f, indent=2, default=str)
    with open(mpath, "w") as f:
        f.write(render_markdown(inventory))
    return jpath, mpath


def render_markdown(inventory: Dict[str, Any]) -> str:
    lines = ["# Data Inventory", ""]
    for name, ds in inventory["datasets"].items():
        lines += [f"## {name}  (`{ds['kind']}`)", "",
                  f"- directory: `{ds['directory']}`",
                  f"- files: {len(ds['files'])}  |  rows: {ds['total_rows']:,}  |  "
                  f"size: {ds['total_bytes'] / 1e9:.2f} GB",
                  f"- time range: {ds['ts_min']}  →  {ds['ts_max']}",
                  f"- schema consistent across files: {ds['schema_consistent']}"]
        if ds["overlaps"]:
            lines.append(f"- **time-range overlaps detected: {len(ds['overlaps'])}**")
            for ov in ds["overlaps"]:
                lines.append(f"    - `{ov['file_a']}` ∩ `{ov['file_b']}`: "
                             f"{ov['overlap_start']} → {ov['overlap_end']}")
        lines += ["", "| file | rows | size MB | ts min | ts max | depth | flags |",
                  "|---|---:|---:|---|---|---|---|"]
        for fi in ds["files"]:
            flags = []
            if fi.get("error"):
                flags.append(f"ERROR: {fi['error']}")
            if fi.get("duplicate_ts"):
                flags.append(f"dup_ts={fi['duplicate_ts']}")
            if fi.get("non_monotonic"):
                flags.append("NON-MONOTONIC")
            if fi.get("weekday_gaps"):
                flags.append(f"weekday_gaps={len(fi['weekday_gaps'])}")
            lines.append(
                f"| {Path(fi['path']).name} | {fi['num_rows']:,} | "
                f"{fi['size_bytes'] / 1e6:.1f} | {fi['ts_min']} | {fi['ts_max']} | "
                f"{fi['scan_depth']} | {'; '.join(flags)} |")
        lines.append("")
    return "\n".join(lines)
