#!/usr/bin/env python
"""Phase 5D: full quote-era micro build — 2021-01-04 .. 2025-06-30
(~1,130 sessions, per-session checkpoints, READ-ONLY, resumable).

Writes outputs/nq_micro_bars_full.parquet."""
from __future__ import annotations

import sys
import time as _time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab.config import load_config                  # noqa: E402
from tslab.data.nq_micro import build_micro_bars      # noqa: E402

START, END = date(2021, 1, 5), date(2025, 6, 30)      # MBP-1 coverage era


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop cleanly after this budget (exit 3 = partial)")
    args = ap.parse_args()
    t0 = _time.time()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    bars = pd.read_parquet(outdir / "nq_bars_full.parquet")
    ck = outdir / "checkpoints" / "micro_full"
    deadline = (t0 + args.max_minutes * 60) if args.max_minutes else None
    aug = build_micro_bars(cfg, bars, START, END, checkpoint_dir=ck,
                           progress=lambda m: print(m, flush=True),
                           deadline_ts=deadline)
    want = bars[(bars["session"] >= START) & (bars["session"] <= END)]
    n_want = want["session"].nunique()
    n_have = aug["session"].nunique()
    if n_have < n_want:
        print(f"PARTIAL {n_have}/{n_want} sessions "
              f"({(_time.time() - t0) / 60:.1f} min)", flush=True)
        return 3
    aug.to_parquet(outdir / "nq_micro_bars_full.parquet")
    cov = aug["imb_last"].notna().mean()
    print(f"rows={len(aug):,} sessions={n_have} coverage={cov:.1%} "
          f"({(_time.time() - t0) / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
