#!/usr/bin/env python
"""Strict RR>=1:1 pass, limit-entry (book-aware) phase — NQ market only.

Same pre-registered protocol as run_strict11_search.py (same 88-rule RR>=1
grids, same 11 folds 2021-2024, same train-only gated/plain selection, same
gates, 2025 tail untouched) — the ONLY change is entry execution:

  MKT    market at open[T+1]           (baseline = strict11 checkpoints)
  L0N1   join near quote, cancel 60s   (through-fill rule)
  L0N3   join near quote, cancel 180s  (through)
  LM3    improve 1 tick if spread>=2, else join; 180s (through)
  LP1N3  rest 1 tick behind quote, 180s (through)
  LP2N3  rest 2 ticks behind quote, 180s (through)
  L0N3q  join near quote, 180s, QUEUE fill rule (realistic secondary)

Through-fill = price must trade strictly through the level (fills any
queue position). Unfilled by deadline -> missed attempt (no chase).
Every attempt carries a market-entry counterfactual for adverse-selection
attribution. MNQ fills are adjudicated on the NQ book path (same index,
identical price grid) with MNQ fees — disclosed model assumption.

Usage: run_strict11_le.py --insts MNQ,NQ --models L0N1,L0N3,...
       [--limit-folds N] [--max-minutes M]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time as _time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tslab.config import load_config                              # noqa: E402
from tslab.features.core import FeatureParams                     # noqa: E402
from tslab.features.labels import make_session_folds              # noqa: E402
from tslab.features.quantiles import compute_micro_q_features     # noqa: E402
from tslab.sim.costs import CostModel                             # noqa: E402
from tslab.strategy.backtest import run_backtest                  # noqa: E402
from tslab.strategy.backtest_limit import (EntryModel,            # noqa: E402
                                           build_fill_lookup,
                                           run_backtest_limit)
from tslab.strategy.evaluate import lifecycle_mc_chain, max_drawdown  # noqa: E402
from tslab.strategy.full_search import train_tstat                # noqa: E402
from run_strict11_search import (GATES, MIN_TRAIN_TRADES,         # noqa: E402
                                 MIN_TRAIN_WIN, SEARCH_END,
                                 nq_strict_grids, select_arm,
                                 stream_stats)

MODELS = {
    "L0N1": EntryModel("L0N1", 0, 1, "thr"),
    "L0N3": EntryModel("L0N3", 0, 3, "thr"),
    "LM3": EntryModel("LM3", -1, 3, "thr"),
    "LP1N3": EntryModel("LP1N3", 1, 3, "thr"),
    "LP2N3": EntryModel("LP2N3", 2, 3, "thr"),
    "L0N3q": EntryModel("L0N3q", 0, 3, "qf"),
}


def evaluate_fold_limit(grid, bars, feats, fold, cfg, cm, lookup, model,
                        instrument):
    tr_m = np.asarray((bars.index >= fold["train_start"])
                      & (bars.index < fold["train_end"]))
    te_m = np.asarray((bars.index >= fold["test_start"])
                      & (bars.index < fold["test_end"]))
    btr, ftr = bars[tr_m], feats[tr_m]
    bte, fte = bars[te_m], feats[te_m]
    tr_sessions = sorted(set(btr["session"]))
    assert not set(tr_sessions) & set(bte["session"])
    out = {}
    for rule in grid:
        r_tr, _ = run_backtest_limit(rule, btr, ftr, cfg, cm, lookup,
                                     model, instrument, False)
        r_te, att = run_backtest_limit(rule, bte, fte, cfg, cm, lookup,
                                       model, instrument, True)
        assert r_tr.violations == 0 and r_te.violations == 0
        out[rule.name] = {
            "train_net": r_tr.total_net,
            "train_trades": len(r_tr.trades),
            "train_wins": sum(1 for t in r_tr.trades if t.net_usd > 0),
            "train_tstat": train_tstat(r_tr.daily_net(), len(tr_sessions)),
            "test_trades": r_te.trades,
            "test_attempts": att,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--insts", default="MNQ,NQ")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit-folds", type=int, default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    t0 = _time.time()
    deadline = (t0 + args.max_minutes * 60) if args.max_minutes else None
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    ckdir = outdir / "checkpoints" / "strict11_le"
    ckdir.mkdir(parents=True, exist_ok=True)
    insts = args.insts.split(",")
    model_names = args.models.split(",")

    full = pd.read_parquet(outdir / "nq_micro_bars_full.parquet")
    bars = full[full["session"] <= SEARCH_END].copy()
    del full
    sessions = sorted(set(bars["session"]))
    feats = compute_micro_q_features(bars, cfg, FeatureParams())
    folds = make_session_folds(sessions, cfg, 250, 63, 1)
    if args.limit_folds:
        folds = folds[:args.limit_folds]
    grids = dict(zip(("NQ", "MNQ"), nq_strict_grids()))
    cm = CostModel(cfg)

    ft_dir = outdir / "fill_table"
    ft = pd.concat([pd.read_parquet(p) for p in sorted(
        ft_dir.glob("filltab_202[1-4]_*.parquet"))], ignore_index=True)
    print(f"fill table rows={len(ft)} bars={ft.bar_ts.nunique()}",
          flush=True)
    lookups = {mn: build_fill_lookup(ft, MODELS[mn])
               for mn in model_names if mn in MODELS}
    del ft

    if not args.assemble_only:
        jobs = [(inst, mn, k) for inst in insts
                for mn in model_names if mn in MODELS
                for k in range(len(folds))]
        for inst, mn, k in jobs:
            ck = ckdir / f"{inst}_{mn}_{k}.pkl"
            if ck.exists():
                continue
            if deadline and _time.time() > deadline:
                print(f"PARTIAL before {inst}/{mn} fold {k}", flush=True)
                return 3
            t = _time.time()
            res = evaluate_fold_limit(grids[inst], bars, feats, folds[k],
                                      cfg, cm, lookups[mn], MODELS[mn],
                                      inst)
            ck.write_bytes(pickle.dumps(res))
            print(f"[{inst}/{mn} fold {k:02d}] done "
                  f"({_time.time() - t:.0f}s)", flush=True)
    return assemble(bars, feats, folds, grids, cfg, cm, insts,
                    model_names, ckdir, outdir, t0)


def assemble(bars, feats, folds, grids, cfg, cm, insts, model_names,
             ckdir, outdir, t0):
    mkt_ck = outdir / "checkpoints" / "strict11_nq"
    report = {"phase": "strict11 limit-entry (book-aware)",
              "folds": len(folds), "gates": GATES,
              "models": model_names, "instruments": {}}
    wf_rows, att_rows, cand_rows = [], [], []
    for inst in insts:
        report["instruments"][inst] = {}
        for mn in ["MKT"] + [m for m in model_names if m in MODELS]:
            fold_res = []
            complete = True
            for k in range(len(folds)):
                p = (mkt_ck / f"{inst}_{k}.pkl" if mn == "MKT"
                     else ckdir / f"{inst}_{mn}_{k}.pkl")
                if not p.exists():
                    complete = False
                    break
                fold_res.append(pickle.loads(p.read_bytes()))
            if not complete:
                continue
            wf_sessions_all = []
            arms_out = {}
            agg = {r.name: {"tstats": [], "nets": [], "trades": {},
                            "att": [0, 0]} for r in grids[inst]}
            fold_sess = []
            for k, res in enumerate(fold_res):
                te_sessions = sorted(
                    set(bars.loc[(bars.index >= folds[k]["test_start"])
                                 & (bars.index < folds[k]["test_end"]),
                                 "session"]))
                fold_sess.append(te_sessions)
                wf_sessions_all += te_sessions
                for name, rec in res.items():
                    agg[name]["tstats"].append(rec["train_tstat"])
                    agg[name]["nets"].append(
                        float(sum(t.net_usd for t in rec["test_trades"])))
                    for t in rec["test_trades"]:
                        agg[name]["trades"].setdefault(
                            t.session, []).append(t)
                    for a in rec.get("test_attempts", []):
                        agg[name]["att"][0] += 1
                        agg[name]["att"][1] += int(a.filled)
                        att_rows.append({**asdict(a), "rule": name,
                                         "instrument": inst, "model": mn,
                                         "fold": k})
            for arm_name, gated in (("gated", True), ("plain", False)):
                wf_trades = {s: [] for s in wf_sessions_all}
                fold_table = []
                for k, res in enumerate(fold_res):
                    winner = select_arm(res, gated)
                    tnet = 0.0
                    if winner:
                        for t in res[winner]["test_trades"]:
                            wf_trades[t.session].append(t)
                            wf_rows.append({**asdict(t), "rule": winner,
                                            "fold": k, "instrument": inst,
                                            "model": mn, "arm": arm_name})
                        tnet = float(sum(t.net_usd for t in
                                         res[winner]["test_trades"]))
                    w = res[winner] if winner else None
                    fold_table.append({
                        "fold": k, "winner": winner,
                        "train_tstat": (round(w["train_tstat"], 2)
                                        if w else None),
                        "train_win": (round(w["train_wins"]
                                            / w["train_trades"], 3)
                                      if w else None),
                        "test_net": round(tnet),
                        "test_trades": (len(w["test_trades"])
                                        if w else 0)})
                all_tr, daily, wf = stream_stats(
                    wf_trades, wf_sessions_all,
                    [f["test_net"] for f in fold_table])
                mc = {}
                if all_tr:
                    pool = [wf_trades[s] for s in sorted(wf_trades)]
                    for size, pdll in (("50k", 750.0), ("50k", 1000.0),
                                       ("100k", 1000.0)):
                        mc[f"{size}/pdll{int(pdll)}"] = lifecycle_mc_chain(
                            pool, cfg, size=size, personal_dll=pdll,
                            n_runs=300, horizon_sessions=252)
                best_cfg = (max(mc, key=lambda k_: mc[k_].ev_month)
                            if mc else None)
                best = mc[best_cfg] if best_cfg else None
                verdict = {
                    "win_rate": bool(wf["trades"]
                                     and wf["win"] >= GATES["win"]),
                    "frequency": wf["tps"] >= GATES["tps"],
                    "net_positive": wf["net"] > 0,
                    "ev_month_positive": bool(best and best.ev_month > 0),
                    "blowout": bool(best and best.blowout_rate
                                    < GATES["blowout"]),
                    "fold_stability": wf["pos_folds"] >= len(folds) / 2,
                }
                arms_out[arm_name] = {
                    "wf": wf, "fold_table": fold_table,
                    "best_cfg": best_cfg,
                    "mc_best": ({kk: (round(vv, 4)
                                      if isinstance(vv, float) else vv)
                                 for kk, vv in vars(best).items()}
                                if best else None),
                    "verdict": verdict,
                    "gates_passed": all(verdict.values()),
                }
                print(f"[{inst}/{mn}/{arm_name}] net=${wf['net']:,.0f} "
                      f"n={wf['trades']} tps={wf['tps']:.2f} "
                      f"win={(wf['win'] if wf['trades'] else 0):.3f} "
                      f"posF={wf['pos_folds']}/{len(folds)}"
                      + (f" EV/mo=${best.ev_month:,.0f} "
                         f"blow={best.blowout_rate:.2f}" if best else ""),
                      flush=True)
            rows = []
            rule_by_name = {r.name: r for r in grids[inst]}
            for name, a in agg.items():
                tr = [t for lst in a["trades"].values() for t in lst]
                w_ = sum(1 for t in tr if t.net_usd > 0)
                rl = rule_by_name[name]
                rows.append({
                    "name": name, "model": mn, "instrument": inst,
                    "rr": rl.rr,
                    "mean_train_tstat": float(np.mean(
                        [x for x in a["tstats"] if np.isfinite(x)])
                        if any(np.isfinite(x) for x in a["tstats"])
                        else -np.inf),
                    "oos_net": float(np.sum(a["nets"])),
                    "oos_trades": len(tr),
                    "oos_win_rate": w_ / len(tr) if tr else float("nan"),
                    "trades_per_session": len(tr)
                    / max(len(wf_sessions_all), 1),
                    "gross": float(sum(t.gross_usd for t in tr)),
                    "costs": float(sum(t.costs_usd for t in tr)),
                    "pos_fold_share": float(np.mean(
                        [x > 0 for x in a["nets"]])),
                    "attempts": a["att"][0], "fills": a["att"][1],
                    "fill_rate": (a["att"][1] / a["att"][0]
                                  if a["att"][0] else float("nan")),
                })
            cand_rows += rows
            cand = pd.DataFrame(rows)
            report["instruments"][inst][mn] = {
                "arms": arms_out,
                "joint_gates": {
                    "win_and_tps_and_net": int(
                        ((cand["oos_win_rate"] >= 0.55)
                         & (cand["trades_per_session"] >= 1.0)
                         & (cand["oos_net"] > 0)).sum()),
                    "win_and_tps": int(((cand["oos_win_rate"] >= 0.55)
                                        & (cand["trades_per_session"]
                                           >= 1.0)).sum()),
                    "win_only": int((cand["oos_win_rate"] >= 0.55).sum()),
                    "tps_only": int((cand["trades_per_session"]
                                     >= 1.0).sum()),
                    "net_only": int((cand["oos_net"] > 0).sum()),
                },
            }
    if wf_rows:
        pd.DataFrame(wf_rows).to_parquet(
            outdir / "strict11_le_wf_trades.parquet")
    if att_rows:
        pd.DataFrame(att_rows).to_parquet(
            outdir / "strict11_le_attempts.parquet")
    if cand_rows:
        pd.DataFrame(cand_rows).to_csv(
            outdir / "strict11_le_candidates.csv", index=False)
    report["runtime_min"] = round((_time.time() - t0) / 60, 1)
    (outdir / "strict11_le_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"wrote strict11_le_report.json "
          f"({report['runtime_min']} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
