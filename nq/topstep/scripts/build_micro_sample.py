#!/usr/bin/env python
"""Phase 5A: build MBP-1 micro-augmented bars for 2024-H1 (smoke scope —
125 sessions inside the quote-coverage era, well clear of the spent 2026
tail). Checkpoints per session; READ-ONLY on raw data.

Writes outputs/nq_micro_bars_2024H1.parquet."""
from __future__ import annotations

import sys
import time as _time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab.config import load_config                  # noqa: E402
from tslab.data.nq_micro import build_micro_bars      # noqa: E402

START, END = date(2024, 1, 2), date(2024, 6, 28)


def main() -> int:
    t0 = _time.time()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    bars = pd.read_parquet(outdir / "nq_bars_full.parquet")
    aug = build_micro_bars(cfg, bars, START, END,
                           checkpoint_dir=outdir / "checkpoints" / "micro_2024H1",
                           progress=lambda m: print(m, flush=True))
    aug.to_parquet(outdir / "nq_micro_bars_2024H1.parquet")
    micro_cov = aug["imb_last"].notna().mean()
    print(f"rows={len(aug):,} sessions={aug['session'].nunique()} "
          f"micro coverage={micro_cov:.1%} "
          f"({(_time.time() - t0) / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
