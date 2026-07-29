#!/usr/bin/env python
"""Phase 4A driver: lower-frequency, larger-target NQ candidate class over
the same verified fold plan. Same protocol as Phase 3C (train-only selection,
OOS test quarters, full costs/rules), different candidate economics.

VIABILITY GATES (pre-declared BEFORE the run; the class is 'viable' only if
the walk-forward stream satisfies ALL):
    V1  zero Topstep rule violations
    V2  signal truncation audits PASS for selected rules
    V3  WF OOS net > 0
    V4  >= 9 of 17 positive test folds (stability)
    V5  WF trades >= 150 (enough to trust)
    V6  lifecycle MC EV net payout > 0 (best legal account config)
Top individual candidates additionally need: pos_fold_share >= 0.5,
aggregate OOS net > 0, >= 80 OOS trades, own MC EV > 0.

Writes outputs/lowfreq_search_report.{md,json} + wf_lowfreq_trades.parquet.
Exit != 0 on verification/audit FAIL or violations (a NON-VIABLE verdict
with clean audits exits 0 — a negative is a valid result).
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

from tslab.audit.leakage import Status                            # noqa: E402
from tslab.config import load_config                              # noqa: E402
from tslab.strategy.evaluate import max_drawdown                  # noqa: E402
from tslab.strategy.full_search import (audit_signal,             # noqa: E402
                                        load_fold_plan, mc_for_pool,
                                        run_search, verify_fold_plan)
from tslab.strategy.rules import lowfreq_candidate_grid           # noqa: E402

MIN_TRAIN_TRADES = 40      # lower frequency => fewer train trades expected
GATES = {"min_pos_folds": 9, "min_wf_trades": 150,
         "top_min_pos_share": 0.5, "top_min_oos_trades": 80}


def main() -> int:
    t0 = _time.time()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    bars = pd.read_parquet(outdir / "nq_bars_full.parquet")
    feats = pd.read_parquet(outdir / "nq_features_full.parquet")
    end_times = pd.read_parquet(outdir / "nq_label_endtimes_full.parquet")
    folds = load_fold_plan(outdir / "fold_plan.json")
    grid = lowfreq_candidate_grid()
    print(f"bars={len(bars):,} candidates={len(grid)} folds={len(folds)} "
          f"min_train_trades={MIN_TRAIN_TRADES}", flush=True)

    checks = verify_fold_plan(folds, end_times, pd.Timedelta(hours=24))
    for r in checks:
        print(f"  [{r.status.value}] {r.name}: {r.message}", flush=True)
    if any(r.status == Status.FAIL for r in checks):
        return 1

    out = run_search(grid, bars, feats, folds, cfg,
                     progress=lambda m: print(m, flush=True),
                     min_trades=MIN_TRAIN_TRADES)

    wf_net = float(out.wf_daily.sum())
    wf_dd = max_drawdown(out.wf_daily)
    wf_ntrades = sum(len(v) for v in out.wf_trades_by_session.values())
    pos_folds = sum(1 for f in out.fold_table if (f["test_net"] or 0) > 0)
    all_tr = [t for v in out.wf_trades_by_session.values() for t in v]
    risk = [t.risk_usd for t in all_tr]
    print(f"WF stream: net=${wf_net:,.0f} trades={wf_ntrades} "
          f"posFolds={pos_folds}/17 maxDD=${wf_dd:,.0f} "
          f"avgRisk=${np.mean(risk):,.0f}" if risk else "WF stream: no trades",
          flush=True)

    sweep = {}
    for size in ("50k", "100k", "150k"):
        for pdll in (750.0, 1000.0):
            sweep[f"{size}/pdll{int(pdll)}"] = mc_for_pool(
                out.wf_trades_by_session, cfg, size=size, personal_dll=pdll)
    best_cfg = max(sweep, key=lambda k: sweep[k].ev_net_payout)
    wf_mc = sweep[best_cfg]

    # top-candidate gating + MC
    top_rows = []
    for name in out.top_global:
        row = out.candidate_summary[
            out.candidate_summary["name"] == name].iloc[0].to_dict()
        st = mc_for_pool(out.top_candidate_trades[name], cfg)
        qualified = (row["pos_fold_share"] >= GATES["top_min_pos_share"]
                     and row["oos_net"] > 0
                     and row["oos_trades"] >= GATES["top_min_oos_trades"]
                     and st.ev_net_payout > 0)
        top_rows.append({"name": name, **{k: row[k] for k in
                                          ("mean_train_tstat", "oos_net",
                                           "oos_trades", "pos_fold_share")},
                         "mc": {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in vars(st).items()},
                         "qualified": bool(qualified)})
        print(f"  top {name}: oos={row['oos_net']:.0f} "
              f"posShare={row['pos_fold_share']:.2f} EV={st.ev_net_payout:.0f}"
              f" qualified={qualified}", flush=True)

    fold_winners = [f["winner"] for f in out.fold_table if f["winner"]]
    audits = []
    for name in sorted({w for w in fold_winners}
                       | set(out.top_global[:1]))[:4]:
        rule = next(r for r in grid if r.name == name)
        a = audit_signal(rule, bars, cfg)
        audits.append(a)
        print(f"  [{a.status.value}] {a.name}", flush=True)

    audits_ok = all(a.status != Status.FAIL for a in audits)
    verdict = {
        "V1_zero_violations": out.violations == 0,
        "V2_signal_audits_pass": audits_ok,
        "V3_wf_net_positive": wf_net > 0,
        "V4_fold_stability": pos_folds >= GATES["min_pos_folds"],
        "V5_enough_trades": wf_ntrades >= GATES["min_wf_trades"],
        "V6_mc_ev_positive": wf_mc.ev_net_payout > 0,
    }
    viable = all(verdict.values())

    wf_rows = [asdict(t) for s in sorted(out.wf_trades_by_session)
               for t in out.wf_trades_by_session[s]]
    if wf_rows:
        pd.DataFrame(wf_rows).to_parquet(outdir / "wf_lowfreq_trades.parquet")

    def st_dict(st):
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in vars(st).items()}

    report = {
        "candidate_class": "lowfreq_larger_target",
        "candidates": len(grid), "folds": len(folds),
        "min_train_trades": MIN_TRAIN_TRADES, "gates": GATES,
        "violations": out.violations,
        "fold_table": out.fold_table,
        "wf": {"net": wf_net, "trades": wf_ntrades, "max_dd": wf_dd,
               "positive_folds": pos_folds,
               "sessions": len(out.wf_sessions),
               "avg_risk_usd": float(np.mean(risk)) if risk else 0.0,
               "trades_per_session": wf_ntrades / max(len(out.wf_sessions), 1),
               "daily_mean": float(out.wf_daily.mean()),
               "daily_std": float(out.wf_daily.std()),
               "mc_sweep": {k: st_dict(v) for k, v in sweep.items()},
               "best_account_config": best_cfg},
        "top_global": top_rows,
        "candidate_summary": out.candidate_summary.head(20).to_dict("records"),
        "signal_audits": [{"name": a.name, "status": a.status.value,
                           "message": a.message} for a in audits],
        "verdict": verdict, "viable": viable,
        "runtime_min": round((_time.time() - t0) / 60, 1),
    }
    (outdir / "lowfreq_search_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    _write_md(report, grid, outdir / "lowfreq_search_report.md")
    print(f"\nVIABLE={viable}  verdict={verdict}", flush=True)
    print(f"wrote {outdir}/lowfreq_search_report.md|.json "
          f"({report['runtime_min']} min)", flush=True)
    return 0 if (out.violations == 0 and audits_ok) else 1


def _write_md(report, grid, path: Path) -> None:
    rules = {r.name: r for r in grid}
    wf = report["wf"]
    L = ["# Lower-frequency NQ candidate class (Phase 4A)", "",
         f"**VIABLE: {report['viable']}**", "",
         "| gate | result |", "|---|---|"]
    L += [f"| {k} | {'PASS' if v else 'FAIL'} |"
          for k, v in report["verdict"].items()]
    L += ["", "## Walk-forward stream", "",
          f"- OOS net ${wf['net']:,.0f}, {wf['trades']} trades "
          f"({wf['trades_per_session']:.2f}/session), avg risk "
          f"${wf['avg_risk_usd']:,.0f}/trade",
          f"- positive folds {wf['positive_folds']}/17, max drawdown "
          f"${wf['max_dd']:,.0f}, daily mean ${wf['daily_mean']:.0f} "
          f"std ${wf['daily_std']:.0f}", "",
          "| fold | winner | train t | test window | test net $ | trades |",
          "|---:|---|---:|---|---:|---:|"]
    for f in report["fold_table"]:
        L.append(f"| {f['fold']} | {f['winner']} | {f['train_tstat']} | "
                 f"{f['test_start']}..{f['test_end']} | {f['test_net']} | "
                 f"{f['test_trades']} |")
    L += ["", "## Lifecycle Monte Carlo", "",
          "| config | EV $ | pass | payout prob | E[payouts] | DLL rate | "
          "MLL fail c/f |", "|---|---:|---:|---:|---:|---:|---|"]
    for k, st in wf["mc_sweep"].items():
        L.append(f"| {k} | {st['ev_net_payout']:.0f} | {st['pass_prob']:.2f} "
                 f"| {st['payout_prob']:.2f} | {st['mean_payouts']:.1f} | "
                 f"{st['dll_hit_rate']:.3f} | {st['combine_fail_rate']:.2f}/"
                 f"{st['funded_fail_rate']:.2f} |")
    L += [f"", f"Best account config: **{wf['best_account_config']}**", "",
          "## Top-5 global candidates (gated)", "",
          "| candidate | train t | OOS net $ | trades | +share | EV $ | "
          "qualified |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in report["top_global"]:
        L.append(f"| {r['name']} | {r['mean_train_tstat']:.2f} | "
                 f"{r['oos_net']:.0f} | {r['oos_trades']} | "
                 f"{r['pos_fold_share']:.2f} | {r['mc']['ev_net_payout']:.0f}"
                 f" | {r['qualified']} |")
    L += ["", "## Selected rules (exact mechanical definitions)", ""]
    for name in sorted({f["winner"] for f in report["fold_table"]
                        if f["winner"]}):
        r = rules[name]
        L += [f"### {name}",
              f"- {'LONG' if r.direction > 0 else 'SHORT'} 1 mini; decision "
              f"window CT {r.entry_start_ct}..{r.entry_end_ct}; stop "
              f"{r.stop_atr}xATR, target {r.target_atr}xATR (RR {r.rr:.1f}), "
              f"time stop {r.time_stop_bars} bars, max "
              f"{r.max_trades_per_day}/day, day stop -${r.daily_loss_usd:.0f}"
              f", lock {r.daily_profit_lock_usd}",
              "- filters: " + "; ".join(
                  f"{f.feature} {f.op} {f.lo}"
                  + (f"..{f.hi}" if f.hi is not None else "")
                  for f in r.filters), ""]
    L += ["## Signal audits", ""]
    L += [f"- [{a['status']}] {a['name']}: {a['message']}"
          for a in report["signal_audits"]]
    L += ["", "## Caveats", "",
          "- Verdict gates were pre-declared before the run.",
          f"- {report['candidates']} candidates searched; multiple-testing "
          "mitigated by train-only selection + OOS, not eliminated.",
          "- MC bootstraps the OOS pool; costs conservative; qty=1."]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
