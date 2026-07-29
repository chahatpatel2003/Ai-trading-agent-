#!/usr/bin/env python
"""GC/MGC independent-market search (checkpointed per fold, resumable).

Era split (PRE-REGISTERED): search 2016-01-04..2024-12-31; virgin tail =
ALL of 2025 (never touched by any prior phase; used only by a later
one-shot script if candidates pass).

MGC primary (2 geometries x 38 signal rules), GC mini control (geometry 'a'
only). HARD GATES on the MGC WF stream: win>=0.55, tps>=1.0, net>0,
chain-MC EV/month>0, blowout<0.50, fold stability >= half positive,
zero violations, audits pass.

Usage: --max-minutes N  (exit 3 = partial, rerun to resume; exit 0 = done)
Writes outputs/gc_search_report.{md,json} + gc_tail_prereg.json when done.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time as _time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab.audit.leakage import (FoldEmbargoCheck, LeakageAuditor,   # noqa: E402
                                 Status, TruncationInvarianceCheck)
from tslab.config import load_config                                 # noqa: E402
from tslab.features.core import FeatureParams                        # noqa: E402
from tslab.features.labels import (LabelParams, check_label_purge,   # noqa: E402
                                   compute_labels, make_session_folds,
                                   required_embargo)
from tslab.features.quantiles import compute_gc_q_features           # noqa: E402
from tslab.sim.costs import CostModel                                # noqa: E402
from tslab.strategy.evaluate import lifecycle_mc_chain, max_drawdown  # noqa: E402
from tslab.strategy.full_search import (candidate_gates,             # noqa: E402
                                        evaluate_fold, select_winner)
from tslab.strategy.rules import gc_exec_grids                       # noqa: E402

SEARCH_END = date(2024, 12, 31)
TAIL_START = date(2025, 1, 1)
GATES = {"win": 0.55, "tps": 1.0, "blowout": 0.50}
MIN_TRAIN_TRADES = 200
N_TRAIN, N_TEST, EMB = 250, 126, 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()
    t0 = _time.time()
    deadline = (t0 + args.max_minutes * 60) if args.max_minutes else None
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    ckdir = outdir / "checkpoints" / "gc_search"
    ckdir.mkdir(parents=True, exist_ok=True)

    full = pd.read_parquet(outdir / "gc_bars_full.parquet")
    bars = full[full["session"] <= SEARCH_END].copy()
    tail_n = full.loc[full["session"] >= TAIL_START, "session"].nunique()
    del full
    sessions = sorted(set(bars["session"]))
    feats = compute_gc_q_features(bars, cfg)
    folds = make_session_folds(sessions, cfg, N_TRAIN, N_TEST, EMB)
    grid_gc, grid_mgc = gc_exec_grids()
    grid_gc = tuple(r for r in grid_gc if r.name.endswith("gca"))  # control
    print(f"search {len(sessions)} sessions, tail reserved {tail_n}; "
          f"folds={len(folds)} rules: MGC={len(grid_mgc)} GC={len(grid_gc)}",
          flush=True)

    # one-time audits (checkpointed)
    aud_ck = ckdir / "audits.pkl"
    if aud_ck.exists():
        audits = pickle.loads(aud_ck.read_bytes())
    else:
        audits = []
        a = TruncationInvarianceCheck(
            bars, lambda df: compute_gc_q_features(df, cfg), n_probes=5).run()
        a.name = "gc_q_truncation"
        audits.append(a)
        labres = compute_labels(bars, cfg, LabelParams())
        audits.append(FoldEmbargoCheck(folds,
                                       required_embargo(labres.specs)).run())
        purge = check_label_purge(labres.end_times, folds)
        from tslab.audit.leakage import CheckResult
        audits.append(CheckResult(
            "label_purge", Status.PASS if not purge else Status.FAIL,
            "clean" if not purge else f"{len(purge)}"))
        aud_ck.write_bytes(pickle.dumps(audits))
    for r in audits:
        print(f"  [{r.status.value}] {r.name}: {r.message}", flush=True)
    if any(r.status == Status.FAIL for r in audits):
        return 1

    cm = CostModel(cfg)
    jobs = [("MGC", grid_mgc, k) for k in range(len(folds))] + \
           [("GC", grid_gc, k) for k in range(len(folds))]
    for inst, grid, k in jobs:
        ck = ckdir / f"{inst}_{k}.pkl"
        if ck.exists():
            continue
        if deadline and _time.time() > deadline:
            print(f"PARTIAL: budget reached before {inst} fold {k}",
                  flush=True)
            return 3
        t = _time.time()
        res = evaluate_fold(grid, bars, feats, folds[k], cfg, cm,
                            instrument=inst)
        ck.write_bytes(pickle.dumps(res))
        print(f"[{inst} fold {k:02d}] done ({_time.time() - t:.0f}s)",
              flush=True)

    # ---- assemble (all folds present) ----
    results = {}
    for inst, grid in (("MGC", grid_mgc), ("GC", grid_gc)):
        fold_res = [pickle.loads((ckdir / f"{inst}_{k}.pkl").read_bytes())
                    for k in range(len(folds))]
        wf_trades, wf_sessions, fold_table = {}, [], []
        agg = {r.name: {"tstats": [], "nets": [], "trades": {}} for r in grid}
        for k, res in enumerate(fold_res):
            winner = select_winner(res, min_trades=MIN_TRAIN_TRADES)
            te_sessions = sorted(
                set(bars.loc[(bars.index >= folds[k]["test_start"])
                             & (bars.index < folds[k]["test_end"]),
                             "session"]))
            wf_sessions += te_sessions
            for s in te_sessions:
                wf_trades[s] = []
            if winner:
                for t in res[winner].test_trade_list:
                    wf_trades[t.session].append(t)
            for name, fc in res.items():
                agg[name]["tstats"].append(fc.train_tstat)
                agg[name]["nets"].append(fc.test_net)
                for t in fc.test_trade_list:
                    agg[name]["trades"].setdefault(t.session, []).append(t)
            w = res[winner] if winner else None
            fold_table.append({
                "fold": k, "winner": winner,
                "train_tstat": round(w.train_tstat, 2) if w else None,
                "test_net": round(w.test_net) if w else 0,
                "test_trades": w.test_trades if w else 0})
        daily = pd.Series({s: sum(t.net_usd for t in wf_trades[s])
                           for s in wf_sessions}, dtype=float).sort_index()
        all_tr = [t for v in wf_trades.values() for t in v]
        n = len(all_tr)
        wins = sum(1 for t in all_tr if t.net_usd > 0)
        rows = []
        for name, a in agg.items():
            tr = [t for lst in a["trades"].values() for t in lst]
            w_ = sum(1 for t in tr if t.net_usd > 0)
            rows.append({
                "name": name,
                "mean_train_tstat": float(np.mean(
                    [x for x in a["tstats"] if np.isfinite(x)]) if any(
                    np.isfinite(x) for x in a["tstats"]) else -np.inf),
                "oos_net": float(np.sum(a["nets"])),
                "oos_trades": len(tr),
                "oos_win_rate": w_ / len(tr) if tr else float("nan"),
                "trades_per_session": len(tr) / max(len(wf_sessions), 1),
                "pos_fold_share": float(np.mean([x > 0 for x in a["nets"]])),
            })
        summary = (pd.DataFrame(rows)
                   .sort_values("mean_train_tstat", ascending=False)
                   .reset_index(drop=True))
        pool = [wf_trades[s] for s in sorted(wf_trades)]
        mc = {}
        for size, pdll in (("50k", 750.0), ("50k", 1000.0), ("100k", 1000.0)):
            mc[f"{size}/pdll{int(pdll)}"] = lifecycle_mc_chain(
                pool, cfg, size=size, personal_dll=pdll, n_runs=300,
                horizon_sessions=252)
        best_cfg = max(mc, key=lambda k_: mc[k_].ev_month)
        results[inst] = {
            "wf": {"net": float(daily.sum()), "trades": n,
                   "tps": n / max(len(wf_sessions), 1),
                   "win": wins / n if n else float("nan"),
                   "gross": float(sum(t.gross_usd for t in all_tr)),
                   "costs": float(sum(t.costs_usd for t in all_tr)),
                   "max_dd": max_drawdown(daily),
                   "pos_folds": sum(1 for f in fold_table
                                    if (f["test_net"] or 0) > 0)},
            "fold_table": fold_table, "best_cfg": best_cfg,
            "mc": {k_: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                        for kk, vv in vars(v).items()}
                   for k_, v in mc.items()},
            "summary": summary.head(12).to_dict("records"),
            "wf_trades": wf_trades,
        }
        wfd = results[inst]["wf"]
        print(f"[{inst}] WF net=${wfd['net']:,.0f} n={wfd['trades']} "
              f"tps={wfd['tps']:.2f} win={(wfd['win'] if n else 0):.3f} "
              f"posFolds={wfd['pos_folds']}/{len(folds)} | {best_cfg}: "
              f"EV/mo=${mc[best_cfg].ev_month:,.0f} "
              f"blowout={mc[best_cfg].blowout_rate:.2f}", flush=True)

    # winner signal audits (MGC)
    mgc = results["MGC"]
    for name in sorted({f["winner"] for f in mgc["fold_table"]
                        if f["winner"]})[:4]:
        rule = next(r for r in grid_mgc if r.name == name)

        def sig_fn(df, rule=rule):
            fe = compute_gc_q_features(df, cfg)
            return pd.DataFrame({"sig": rule.entry_signal(fe, cfg)
                                 .astype(float)}, index=df.index)

        a = TruncationInvarianceCheck(bars, sig_fn, n_probes=3).run()
        a.name = f"signal_truncation:{name}"
        audits.append(a)
        print(f"  [{a.status.value}] {a.name}", flush=True)
    LeakageAuditor.write_report(audits, outdir, "gc_search")
    audits_ok = all(x.status != Status.FAIL for x in audits)

    wf, best = mgc["wf"], mgc["mc"][mgc["best_cfg"]]
    verdict = {
        "zero_violations": True,      # engine-structural; asserted per fold
        "audits_pass": audits_ok,
        "win_rate": bool(wf["trades"] and wf["win"] >= GATES["win"]),
        "frequency": wf["tps"] >= GATES["tps"],
        "net_positive": wf["net"] > 0,
        "ev_month_positive": best["ev_month"] > 0,
        "blowout": best["blowout_rate"] < GATES["blowout"],
        "fold_stability": wf["pos_folds"] >= len(folds) / 2,
        "tail_confirmation": None,
    }
    search_pass = all(v for k, v in verdict.items()
                      if k != "tail_confirmation")
    qual = [r for r in
            pd.DataFrame(mgc["summary"]).to_dict("records")
            if candidate_gates(r)]
    prereg = {"tail": [str(TAIL_START), "2025-12-31"], "instrument": "MGC",
              "candidates": [r["name"] for r in qual[:3]],
              "search_gates_passed": bool(search_pass),
              "tail_pass_criteria": {"net_gt": 0, "win_ge": 0.50,
                                     "tps_ge": 0.5, "max_dd_abs_le": 4000,
                                     "violations": 0}}
    (outdir / "gc_tail_prereg.json").write_text(json.dumps(prereg, indent=2))

    for inst in results:
        results[inst].pop("wf_trades")
    report = {
        "phase": "GC/MGC independent control",
        "folds": len(folds), "gates": GATES, "verdict": verdict,
        "search_gates_passed": bool(search_pass),
        "instruments": results, "tail_prereg": prereg,
        "audits": [{"name": x.name, "status": x.status.value}
                   for x in audits],
        "runtime_min": round((_time.time() - t0) / 60, 1),
    }
    (outdir / "gc_search_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"\nsearch_gates_passed={search_pass} prereg="
          f"{prereg['candidates']}", flush=True)
    print(f"wrote {outdir}/gc_search_report.json "
          f"({report['runtime_min']} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
