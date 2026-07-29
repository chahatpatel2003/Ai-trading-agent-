#!/usr/bin/env python
"""Scan the parquet data inventory (READ-ONLY) and write reports.

Usage:
    .venv/bin/python scripts/scan_data.py [--dataset nq.mbp1] [--stamp latest]

Writes outputs/inventory_<stamp>.json and outputs/inventory_<stamp>.md and
prints a summary. Never writes anything under the data roots.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab.config import load_config                     # noqa: E402
from tslab.data.inventory import scan_dataset, write_reports  # noqa: E402
from dataclasses import asdict                           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=None, help="configs dir (default: ./configs)")
    ap.add_argument("--dataset", default=None,
                    help="scan only this dataset key (e.g. gc.ohlcv_1m)")
    ap.add_argument("--stamp", default="latest", help="report filename suffix")
    args = ap.parse_args()

    cfg = load_config(args.configs)

    missing = [str(r) for r in cfg.paths.data_roots() if not r.exists()]
    if missing:
        print(f"WARNING: data roots not mounted: {missing}", file=sys.stderr)

    keys = [args.dataset] if args.dataset else list(cfg.paths.datasets)
    inventory = {"datasets": {}}
    for key in keys:
        ds = cfg.paths.datasets[key]
        print(f"scanning {key}  ({ds.path}) ...", flush=True)
        inv = scan_dataset(ds, cfg)
        inventory["datasets"][key] = asdict(inv)
        print(f"  files={len(inv.files)}  rows={inv.total_rows:,}  "
              f"bytes={inv.total_bytes / 1e9:.2f}GB  "
              f"range={inv.ts_min} -> {inv.ts_max}  "
              f"schema_consistent={inv.schema_consistent}  "
              f"overlaps={len(inv.overlaps)}")
        for f in inv.files:
            if f.error:
                print(f"  ERROR {Path(f.path).name}: {f.error}", file=sys.stderr)

    jpath, mpath = write_reports(inventory, cfg.paths.outputs_dir, args.stamp)
    print(f"\nwrote {jpath}\nwrote {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
