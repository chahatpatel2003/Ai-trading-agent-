#!/usr/bin/env python
"""Phase 3C driver: full-history NQ walk-forward candidate search.

Reads the Phase 3B artifacts, re-verifies the fold plan, runs the search,
Monte-Carlos the walk-forward stream + top-5 global candidates, sweeps
account configurations for the walk-forward stream, audits winner signals,
and writes outputs/full_search_report.{md,json} + outputs/wf_oos_trades.parquet.

Exit != 0 on any verification/audit FAIL or any rule violation.
"""
from __future__ import annotations

import json
import sys
import time as _time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab.audit.leakage import LeakageAuditor, Status            # noqa: E402
from tslab.config import load_config                              # noqa: E402
from tslab.features.labels import required_embargo                # noqa: E402
from tslab.strategy.evaluate import max_drawdown                  # noqa: E402
from tslab.strategy.full_search import (audit_signal,             # noqa: E402
                                        load_fold_plan, mc_for_pool,
                                        run_search, verify_fold_plan)
from tslab.strategy.rules import full_candidate_grid              # noqa: E402


def main() -> int:
    t0 = _time.time()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    bars = pd.read_parquet(outdir / "nq_bars_full.parquet")
    feats = pd.read_parquet(outdir / "nq_features_full.parquet")
    end_times = pd.read_parquet(outdir / "nq_label_endtimes_full.parquet")
    folds = load_fold_plan(outdir / "fold_plan.json")
    grid = full_candidate_grid()
    print(f"bars={len(bars):,} candidates={len(grid)} folds={len(folds)}",
          flush=True)

    checks = verify_fold_plan(folds, end_times, pd.Timedelta(hours=24))
    for r in checks:
        print(f"  [{r.status.value}] {r.name}: {r.message}", flush=True)
    if any(r.status == Status.FAIL for r in checks):
        return 1

    out = run_search(grid, bars, feats, folds, cfg,
                     progress=lambda m: print(m, flush=True))

    # ---- walk-forward stream stats + MC ----
    wf_net = float(out.wf_daily.sum())
    wf_dd = max_drawdown(out.wf_daily)
    wf_ntrades = sum(len(v) for v in out.wf_trades_by_session.values())
    print(f"WF stream: net=${wf_net:,.0f} trades={wf_ntrades} "
          f"maxDD=${wf_dd:,.0f} over {len(out.wf_sessions)} sessions",
          flush=True)
    wf_mc = mc_for_pool(out.wf_trades_by_session, cfg)
    sweep = {}
    for size in ("50k", "100k", "150k"):
        for pdll in (500.0, 750.0, 1000.0):
            st = mc_for_pool(out.wf_trades_by_session, cfg, size=size,
                             personal_dll=pdll)
            sweep[f"{size}/pdll{int(pdll)}"] = st
    best_cfg = max(sweep, key=lambda k: sweep[k].ev_net_payout)

    # ---- top global candidates: aggregate OOS + MC ----
    top_mc = {}
    for name in out.top_global:
        st = mc_for_pool(out.top_candidate_trades[name], cfg)
        top_mc[name] = st
        print(f"  top-candidate MC {name}: EV=${st.ev_net_payout:,.0f} "
              f"pass={st.pass_prob:.2f} payout_p={st.payout_prob:.2f}",
              flush=True)

    # ---- winner signal audits ----
    fold_winners = [f["winner"] for f in out.fold_table if f["winner"]]
    most_selected = (pd.Series(fold_winners).mode().iloc[0]
                     if fold_winners else None)
    audits = []
    audited = set()
    for name in filter(None, {most_selected, out.top_global[0]}):
        if name in audited:
            continue
        audited.add(name)
        rule = next(r for r in grid if r.name == name)
        a = audit_signal(rule, bars, cfg)
        audits.append(a)
        print(f"  [{a.status.value}] {a.name}: {a.message}", flush=True)

    # ---- persist ----
    wf_rows = [asdict(t) for s in sorted(out.wf_trades_by_session)
               for t in out.wf_trades_by_session[s]]
    if wf_rows:
        pd.DataFrame(wf_rows).to_parquet(outdir / "wf_oos_trades.parquet")

    def st_dict(st):
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in vars(st).items()}

    report = {
        "candidates": len(grid), "folds": len(folds),
        "violations": out.violations,
        "fold_table": out.fold_table,
        "wf": {"net": wf_net, "trades": wf_ntrades, "max_dd": wf_dd,
               "sessions": len(out.wf_sessions),
               "daily_mean": float(out.wf_daily.mean()),
               "daily_std": float(out.wf_daily.std()),
               "positive_folds": sum(1 for f in out.fold_table
                                     if (f["test_net"] or 0) > 0),
               "mc_default": st_dict(wf_mc),
               "mc_sweep": {k: st_dict(v) for k, v in sweep.items()},
               "best_account_config": best_cfg},
        "candidate_summary": out.candidate_summary.head(20).to_dict("records"),
        "top_global": out.top_global,
        "top_global_mc": {k: st_dict(v) for k, v in top_mc.items()},
        "most_selected": most_selected,
        "signal_audits": [{"name": a.name, "status": a.status.value,
                           "message": a.message} for a in audits],
        "runtime_min": round((_time.time() - t0) / 60, 1),
    }
    (outdir / "full_search_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    _write_md(report, grid, outdir / "full_search_report.md")
    print(f"wrote {outdir}/full_search_report.md|.json  "
          f"({report['runtime_min']} min)", flush=True)

    ok = (out.violations == 0
          and all(a.status != Status.FAIL for a in audits))
    return 0 if ok else 1


def _write_md(report, grid, path: Path) -> None:
    rules = {r.name: r for r in grid}
    L = ["# Full-history NQ walk-forward search (Phase 3C)", "",
         f"- candidates: {report['candidates']}  folds: {report['folds']}  "
         f"violations: {report['violations']}",
         f"- runtime: {report['runtime_min']} min", "",
         "## Walk-forward stream (train-selected per fold, tested OOS)", ""]
    wf = report["wf"]
    L += [f"- OOS net: ${wf['net']:,.0f} over {wf['trades']} trades / "
          f"{wf['sessions']} sessions ({wf['positive_folds']}/"
          f"{report['folds']} positive test folds)",
          f"- daily mean ${wf['daily_mean']:.0f}, std ${wf['daily_std']:.0f}, "
          f"max drawdown ${wf['max_dd']:,.0f}", "",
          "| fold | winner | train t | test window | test net $ | trades |",
          "|---:|---|---:|---|---:|---:|"]
    for f in report["fold_table"]:
        L.append(f"| {f['fold']} | {f['winner']} | {f['train_tstat']} | "
                 f"{f['test_start']}..{f['test_end']} | {f['test_net']} | "
                 f"{f['test_trades']} |")
    L += ["", "## Lifecycle Monte Carlo (walk-forward stream)", "",
          "| account config | EV net payout $ | pass prob | payout prob | "
          "E[payouts] | DLL rate | combine fail | funded fail |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, st in wf["mc_sweep"].items():
        L.append(f"| {k} | {st['ev_net_payout']:.0f} | {st['pass_prob']:.2f} "
                 f"| {st['payout_prob']:.2f} | {st['mean_payouts']:.1f} | "
                 f"{st['dll_hit_rate']:.3f} | {st['combine_fail_rate']:.2f} "
                 f"| {st['funded_fail_rate']:.2f} |")
    L += ["", f"Best account config by EV: **{wf['best_account_config']}**",
          "", "## Top-5 global candidates (aggregate OOS, own MC)", "",
          "| candidate | EV net payout $ | pass prob | payout prob | "
          "E[payouts] | DLL rate |", "|---|---:|---:|---:|---:|---:|"]
    for k, st in report.get("top_global_mc", {}).items():
        L.append(f"| {k} | {st['ev_net_payout']:.0f} | {st['pass_prob']:.2f} "
                 f"| {st['payout_prob']:.2f} | {st['mean_payouts']:.1f} | "
                 f"{st['dll_hit_rate']:.3f} |")
    L += ["", "## Candidate stability (top 20 by mean train t-stat)", "",
          "| candidate | mean train t | agg OOS net $ | OOS trades | "
          "+fold share |", "|---|---:|---:|---:|---:|"]
    for r in report["candidate_summary"]:
        L.append(f"| {r['name']} | {r['mean_train_tstat']:.2f} | "
                 f"{r['oos_net']:.0f} | {r['oos_trades']} | "
                 f"{r['pos_fold_share']:.2f} |")
    L += ["", "## Selected rules (exact mechanical definitions)", ""]
    names = {f["winner"] for f in report["fold_table"] if f["winner"]}
    for name in sorted(names):
        r = rules[name]
        L += [f"### {name}", "",
              f"- direction: {'LONG' if r.direction > 0 else 'SHORT'}, "
              f"qty {r.qty} mini",
              f"- entry window (decision time, CT): {r.entry_start_ct}"
              f"..{r.entry_end_ct}",
              f"- stop {r.stop_atr} ATR / target {r.target_atr} ATR "
              f"(RR {r.rr:.2f}), time stop {r.time_stop_bars} bars",
              f"- max {r.max_trades_per_day} trades/day, day stop "
              f"-${r.daily_loss_usd:.0f}, profit lock "
              f"{r.daily_profit_lock_usd}", "- filters:"]
        L += [f"    - {f.feature} {f.op} {f.lo}"
              + (f"..{f.hi}" if f.hi is not None else "")
              for f in r.filters]
        L.append("")
    L += ["## Signal audits", ""]
    L += [f"- [{a['status']}] {a['name']}: {a['message']}"
          for a in report["signal_audits"]]
    L += ["", "## Caveats", "",
          f"- {report['candidates']} candidates were searched: multiple-"
          "testing risk is mitigated by train-only selection + walk-forward "
          "OOS, not eliminated.",
          "- MC bootstraps the OOS pool (assumes stationarity); costs are "
          "conservative (spread+slippage on every fill).",
          "- qty fixed at 1 mini; no scaling-plan exploitation modeled."]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
