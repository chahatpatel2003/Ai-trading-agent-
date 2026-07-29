#!/usr/bin/env python
"""Re-run all Phase 3B-1 gates against the saved full-history parquet
(no full rebuild): bar integrity, roll causality, and both window-invariance
rebuilds with the dtype-tolerant comparator. Updates the audit report and the
gates section of nq_full_build_report.{md,json}."""
from __future__ import annotations

import json
import sys
import time as _time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import importlib.util                                              # noqa: E402

from tslab.audit.leakage import CheckResult, LeakageAuditor, Status  # noqa: E402
from tslab.audit.pipeline_gate import (BarIntegrityCheck,          # noqa: E402
                                       RollCausalityCheck)
from tslab.config import load_config                               # noqa: E402
from tslab.data.nq_pipeline import load_pipeline_params            # noqa: E402

spec = importlib.util.spec_from_file_location(
    "bfull", Path(__file__).resolve().parent / "build_nq_full.py")
bfull = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bfull)


def main() -> int:
    t0 = _time.time()
    cfg = load_config()
    params = load_pipeline_params(cfg.project_root / "configs")
    outdir = cfg.paths.outputs_dir
    full = pd.read_parquet(outdir / "nq_bars_full.parquet")
    sessions = sorted(set(full["session"]))

    front = full.groupby("session", sort=True)["symbol"].first()
    roll_log, prev, prev_sess = [], None, None
    for s, sym in front.items():
        if sym != prev:
            roll_log.append({"session": s, "front": sym, "previous": prev,
                             "decided_from_session": prev_sess})
            prev = sym
        prev_sess = s
    roll_log[0]["decided_from_session"] = (roll_log[0]["session"]
                                           - timedelta(days=1))

    results = (LeakageAuditor()
               .register(BarIntegrityCheck(full, cfg, params.bar_seconds,
                                           entry_sample=4000))
               .register(RollCausalityCheck(roll_log))).run_all()

    winA = bfull.rebuild_window(cfg, params, date(2024, 3, 4),
                                date(2024, 3, 13))
    nA, badA = bfull.compare_shared_sessions(full, winA.bars)
    winB_sessions = sessions[-13:-1]
    winB = bfull.rebuild_window(cfg, params, winB_sessions[0],
                                winB_sessions[-1])
    nB, badB = bfull.compare_shared_sessions(full, winB.bars)
    for label, n, bad in (("window_invariance_2024roll", nA, badA),
                          ("window_invariance_tail_minus1", nB, badB)):
        results.append(CheckResult(
            label, Status.PASS if not bad else Status.FAIL,
            (f"{n} shared sessions value-identical to full run" if not bad
             else f"{len(bad)}/{n} sessions differ: {bad[:5]}")))

    LeakageAuditor.write_report(results, outdir, "nq_full_build")
    for r in results:
        print(f"  [{r.status.value}] {r.name}: {r.message}", flush=True)

    rpt_path = outdir / "nq_full_build_report.json"
    report = json.loads(rpt_path.read_text())
    report["gates"] = [{"name": r.name, "status": r.status.value,
                        "message": r.message} for r in results]
    report["regate_secs"] = round(_time.time() - t0, 1)
    rpt_path.write_text(json.dumps(report, indent=2))

    md_path = outdir / "nq_full_build_report.md"
    md = md_path.read_text().split("## Gates")[0]
    md += "## Gates\n\n" + "\n".join(
        f"- [{g['status']}] {g['name']}: {g['message']}"
        for g in report["gates"])
    md_path.write_text(md)
    print(f"re-gate done in {report['regate_secs']:.0f}s")
    return 0 if LeakageAuditor.passed(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
