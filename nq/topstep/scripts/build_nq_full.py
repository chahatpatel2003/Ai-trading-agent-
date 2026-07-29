#!/usr/bin/env python
"""Phase 3B-1: full-history NQ 1-minute bar build (READ-ONLY on raw data).

Builds quarter-by-quarter with checkpoints (outputs/checkpoints/nq_full/) so
an interrupted run resumes losslessly. Each chunk is an independent
build_nq_bars call seeded by the trade session BEFORE the chunk — the roll
rule never sees same-day or future data regardless of chunking.

After assembly, gates:
  - BarIntegrityCheck        (info_time, session windows, cutoff safety)
  - RollCausalityCheck       (global roll log reconstructed from the bars)
  - window invariance        (two sub-window rebuilds must reproduce the
                              full run bit-identically on shared sessions:
                              a quotes-era window across a roll week and a
                              tail window ending one session early)

Writes outputs/nq_bars_full.parquet + nq_full_build_report.{md,json}.
Exit != 0 on any gate failure.
"""
from __future__ import annotations

import json
import sys
import time as _time
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab.audit.leakage import LeakageAuditor, Status            # noqa: E402
from tslab.audit.pipeline_gate import (BarIntegrityCheck,         # noqa: E402
                                       RollCausalityCheck)
from tslab.config import load_config                              # noqa: E402
from tslab.data.nq_pipeline import (build_nq_bars,                # noqa: E402
                                    load_pipeline_params,
                                    trade_sessions_in_range)

START = date(2021, 1, 4)          # first session with prior-day data context
END = date(2026, 6, 16)           # last COMPLETE session in the trades data
QUOTES_END = date(2025, 6, 30)    # MBP-1 coverage ends here (trade-only after)


def quarter_chunks(start: date, end: date):
    out = []
    y, q = start.year, (start.month - 1) // 3 + 1
    while True:
        c0 = date(y, 3 * q - 2, 1)
        c1 = (date(y + (q == 4), 1 if q == 4 else 3 * q + 1, 1)
              - timedelta(days=1))
        lo, hi = max(c0, start), min(c1, end)
        if lo <= hi:
            out.append((f"{y}Q{q}", lo, hi))
        if c1 >= end:
            break
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def rebuild_window(cfg, params, lo: date, hi: date):
    return build_nq_bars(cfg, lo, hi, include_quotes=True, params=params)


def compare_shared_sessions(full: pd.DataFrame, win: pd.DataFrame,
                            skip_first: int = 2):
    """Bit-compare a window rebuild against the full run on shared sessions,
    skipping the window's first `skip_first` sessions (roll-hysteresis seed).
    Columns compared = intersection (tail windows lack quote columns; the
    full frame carries them as all-NaN there)."""
    sessions = sorted(set(win["session"]))[skip_first:]
    cols = [c for c in win.columns if c in full.columns]
    mismatched = []
    for s in sessions:
        a = full[full["session"] == s][cols]
        b = win[win["session"] == s][cols]
        try:
            # values must match exactly; dtype may differ (chunks without
            # quote columns coerce int quote fields to float64 in the
            # assembled frame — values are unchanged)
            pd.testing.assert_frame_equal(a, b, check_dtype=False)
        except AssertionError:
            mismatched.append(str(s))
    return len(sessions), mismatched


def main() -> int:
    t0 = _time.time()
    cfg = load_config()
    params = load_pipeline_params(cfg.project_root / "configs")
    outdir = cfg.paths.outputs_dir
    ckdir = outdir / "checkpoints" / "nq_full"
    ckdir.mkdir(parents=True, exist_ok=True)

    frames, chunk_rows = [], []
    skipped_sessions = []
    for key, lo, hi in quarter_chunks(START, END):
        ck = ckdir / f"{key}.parquet"
        meta = ckdir / f"{key}.json"
        tc = _time.time()
        if ck.exists() and meta.exists():
            bars = pd.read_parquet(ck)
            info = json.loads(meta.read_text())
            print(f"[{key}] resumed from checkpoint: {len(bars):,} bars",
                  flush=True)
        else:
            res = build_nq_bars(cfg, lo, hi, include_quotes=True,
                                params=params)
            bars = res.bars
            info = {"skipped_sessions": res.stats["skipped_sessions"],
                    "sessions": res.stats["sessions"]}
            bars.to_parquet(ck)
            meta.write_text(json.dumps(info))
            print(f"[{key}] built {len(bars):,} bars, "
                  f"{len(info['sessions'])} sessions, "
                  f"skipped={info['skipped_sessions']}, "
                  f"{_time.time() - tc:.0f}s", flush=True)
        skipped_sessions += info["skipped_sessions"]
        chunk_rows.append({"chunk": key, "bars": len(bars),
                           "secs": round(_time.time() - tc, 1)})
        frames.append(bars)

    full = pd.concat(frames)
    if not full.index.is_monotonic_increasing:
        full = full.sort_index(kind="stable")
    build_secs = _time.time() - t0
    print(f"assembled {len(full):,} bars in {build_secs:.0f}s", flush=True)

    # global roll log reconstructed from the assembled bars (chunk-agnostic)
    front = full.groupby("session", sort=True)["symbol"].first()
    roll_log, prev = [], None
    prev_sess = None
    for s, sym in front.items():
        if sym != prev:
            roll_log.append({"session": s, "front": sym, "previous": prev,
                             "decided_from_session": prev_sess})
            prev = sym
        prev_sess = s
    roll_log[0]["decided_from_session"] = roll_log[0]["session"] - \
        timedelta(days=1)                       # bootstrap seed session

    print("running gates ...", flush=True)
    auditor = (LeakageAuditor()
               .register(BarIntegrityCheck(full, cfg, params.bar_seconds,
                                           entry_sample=4000))
               .register(RollCausalityCheck(roll_log)))
    results = auditor.run_all()

    # window-invariance gates (rebuilds must reproduce the full run)
    sessions = sorted(set(full["session"]))
    winA = rebuild_window(cfg, params, date(2024, 3, 4), date(2024, 3, 13))
    nA, badA = compare_shared_sessions(full, winA.bars)
    winB_sessions = sessions[-13:-1]
    winB = rebuild_window(cfg, params, winB_sessions[0], winB_sessions[-1])
    nB, badB = compare_shared_sessions(full, winB.bars)
    from tslab.audit.leakage import CheckResult
    for label, n, bad in (("window_invariance_2024roll", nA, badA),
                          ("window_invariance_tail_minus1", nB, badB)):
        results.append(CheckResult(
            label,
            Status.PASS if not bad else Status.FAIL,
            (f"{n} shared sessions bit-identical to full run" if not bad
             else f"{len(bad)}/{n} sessions differ: {bad[:5]}")))

    LeakageAuditor.write_report(results, outdir, "nq_full_build")
    for r in results:
        print(f"  [{r.status.value}] {r.name}: {r.message}", flush=True)

    out_path = outdir / "nq_bars_full.parquet"
    full.to_parquet(out_path)

    report = {
        "path": str(out_path),
        "start_session": str(sessions[0]), "end_session": str(sessions[-1]),
        "n_sessions": len(sessions), "n_bars": int(len(full)),
        "skipped_sessions": skipped_sessions,
        "quote_coverage_end": str(QUOTES_END),
        "rolls": [{k: str(v) for k, v in e.items()} for e in roll_log],
        "gates": [{"name": r.name, "status": r.status.value,
                   "message": r.message} for r in results],
        "chunks": chunk_rows,
        "runtime_secs": round(_time.time() - t0, 1),
    }
    with open(outdir / "nq_full_build_report.json", "w") as f:
        json.dump(report, f, indent=2)

    md = ["# NQ full-history build report", "",
          f"- file: `{out_path}`",
          f"- sessions: {report['n_sessions']}  "
          f"({report['start_session']} .. {report['end_session']})",
          f"- bars: {report['n_bars']:,}",
          f"- quote coverage through {QUOTES_END} (trade-only after)",
          f"- skipped sessions: {skipped_sessions}",
          f"- runtime: {report['runtime_secs']:.0f}s", "",
          "## Contract rolls", ""]
    md += [f"- {e['session']}: **{e['front']}** (prev {e['previous']}, "
           f"decided from {e['decided_from_session']})" for e in roll_log]
    md += ["", "## Gates", ""]
    md += [f"- [{g['status']}] {g['name']}: {g['message']}"
           for g in report["gates"]]
    (outdir / "nq_full_build_report.md").write_text("\n".join(md))
    print(f"wrote {out_path}", flush=True)

    return 0 if LeakageAuditor.passed(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
