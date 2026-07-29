#!/usr/bin/env python
"""First autonomous ML discovery pass (NQ/MNQ primary, GC/MGC track).

Per fold (checkpointed, resumable): fit 6-model zoo on train[:80%], select
family+target+policy on train[80%:] (validation, 1-session purges), evaluate
ONCE on the test half-year. Shuffled-label control on fold 1. Aggregate OOS
stream -> chained renewal MC sweep (50k/100k x DLL 300/500/750/1000).
GC-2025 stays virgin (GC data truncated at 2024-12-31 here).

Usage: --max-minutes N (exit 3 = partial, rerun to resume).
Writes outputs/ml_discovery_report.{md,json}."""
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

from tslab.config import load_config                              # noqa: E402
from tslab.features.core import FeatureParams                     # noqa: E402
from tslab.features.labels import LabelParams, compute_labels     # noqa: E402
from tslab.features.quantiles import (compute_gc_q_features,      # noqa: E402
                                      compute_micro_q_features)
from tslab.ml.discovery import (build_dataset, distill_rules,     # noqa: E402
                                feature_importance, fit_and_score,
                                half_year_folds, make_entries,
                                policy_search, run_signal_backtest)
from tslab.strategy.evaluate import lifecycle_mc_chain, max_drawdown  # noqa: E402

TARGETS = ("y_dir30", "y_tb")
TWO_SIDED = {"y_dir30": True, "y_dir15": True, "y_tb": False}


def prep_track(cfg, track):
    outdir = cfg.paths.outputs_dir
    if track == "NQ":
        bars = pd.read_parquet(outdir / "nq_micro_bars_full.parquet")
        feats = compute_micro_q_features(bars, cfg, FeatureParams())
        inst = "MNQ"
        ft = (2023, 1)
        lt = (2025, 1)
    else:
        bars = pd.read_parquet(outdir / "gc_bars_full.parquet")
        bars = bars[bars["session"] <= date(2024, 12, 31)].copy()  # keep 2025 virgin
        feats = compute_gc_q_features(bars, cfg)
        inst = "MGC"
        ft = (2022, 1)
        lt = (2024, 2)
    labels = compute_labels(bars, cfg, LabelParams()).labels
    tick = cfg.instruments["NQ" if track == "NQ" else "GC"].tick_size
    bars = bars.copy()
    bars["__atr_pts"] = feats["f_atr_ticks"] * tick
    ds = build_dataset(bars, feats, labels, cfg)
    fcols = [c for c in ds.columns if c.startswith(("f_", "fq_"))]
    sessions = sorted(set(bars["session"]))
    folds = half_year_folds(sessions, ft, lt)
    return bars, ds, fcols, folds, inst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()
    t0 = _time.time()
    deadline = (t0 + args.max_minutes * 60) if args.max_minutes else None
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    ck = outdir / "checkpoints" / "ml_discovery"
    ck.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for track in ("NQ", "GC"):
        bars, ds, fcols, folds, inst = prep_track(cfg, track)
        print(f"[{track}] rows={len(ds):,} feats={len(fcols)} "
              f"folds={len(folds)} inst={inst}", flush=True)
        fold_out = []
        for fi, fold in enumerate(folds):
            fck = ck / f"{track}_{fi}.pkl"
            if fck.exists():
                fold_out.append(pickle.loads(fck.read_bytes()))
                continue
            if deadline and _time.time() > deadline:
                print(f"PARTIAL before {track} fold {fi}", flush=True)
                return 3
            t = _time.time()
            cut = int(len(fold["train"]) * 0.8)
            val_s = set(fold["train"][cut:])
            scored = {}
            for tgt in TARGETS:
                scored[tgt] = fit_and_score(ds, fold, tgt, fcols)
            best_tgt, best_fam, best_auc, best_model = None, None, -1, None
            for tgt, fam_d in scored.items():
                for fam, r in fam_d.items():
                    if np.isfinite(r["val_auc"]) and r["val_auc"] > best_auc:
                        best_tgt, best_fam = tgt, fam
                        best_auc, best_model = r["val_auc"], r["model"]
            policy, val_net = policy_search(
                bars, ds.dropna(subset=[best_tgt]), best_model, fcols,
                val_s, cfg, inst, TWO_SIDED[best_tgt])
            rec = {"fold": fold["label"], "best_family": best_fam,
                   "best_target": best_tgt, "val_auc": round(best_auc, 4),
                   "val_aucs": {t_: {f_: round(r_["val_auc"], 4)
                                     for f_, r_ in d_.items()}
                                for t_, d_ in scored.items()},
                   "policy": policy, "val_net": round(val_net)
                   if policy else None, "trades": [], "test_auc": None}
            # test evaluation (single look)
            from sklearn.metrics import roc_auc_score
            dte = ds[ds["session"].isin(set(fold["test"]))].dropna(
                subset=[best_tgt])
            if len(dte) and policy:
                probs = best_model.predict_proba(dte[fcols])[:, 1]
                rec["test_auc"] = round(float(
                    roc_auc_score(dte[best_tgt], probs)), 4)
                ent = make_entries(dte, probs, policy["tau"],
                                   TWO_SIDED[best_tgt])
                bte = bars[bars["session"].isin(set(fold["test"]))]
                trades = run_signal_backtest(
                    bte, ent, cfg, inst, policy["stop_atr"],
                    policy["r_mult"], policy["time_stop"],
                    policy["max_trades"], policy["lock"])
                rec["trades"] = [asdict(x) for x in trades]
            if fi == 0:                            # shuffled-label control
                rng = np.random.default_rng(11)
                dtr = ds[ds["session"].isin(set(fold["train"][:cut - 1]))
                         ].dropna(subset=[best_tgt])
                ysh = rng.permutation(dtr[best_tgt].to_numpy())
                from tslab.ml.discovery import model_zoo
                mdl = model_zoo()[best_fam]
                mdl.fit(dtr[fcols], ysh)
                psh = mdl.predict_proba(dte[fcols])[:, 1]
                rec["shuffled_test_auc"] = round(float(
                    roc_auc_score(dte[best_tgt], psh)), 4)
            # keep importances/rules from the last fold
            rec["importance"] = feature_importance(
                best_model, fcols)
            fck.write_bytes(pickle.dumps(rec))
            fold_out.append(rec)
            print(f"[{track} {fold['label']}] {best_fam}/{best_tgt} "
                  f"valAUC={best_auc:.3f} testAUC={rec['test_auc']} "
                  f"policy={policy} testTrades={len(rec['trades'])} "
                  f"({_time.time() - t:.0f}s)", flush=True)
        all_results[track] = {"inst": inst, "folds": fold_out,
                              "fcols": fcols}

    # ---- aggregate + MC + distill ----
    report = {"models_tested": ["logreg", "rf", "hgb", "lgbm", "mlp",
                                "hgb_cal"],
              "tracks": {}, "runtime_min": round((_time.time() - t0) / 60, 1)}
    from tslab.strategy.backtest import Trade
    for track, res in all_results.items():
        trades = [Trade(**{**t, "session": t["session"]})
                  for f in res["folds"] for t in f["trades"]]
        for t in trades:
            if isinstance(t.session, str):
                t.session = date.fromisoformat(t.session)
        n = len(trades)
        wins = sum(1 for t in trades if t.net_usd > 0)
        daily = {}
        for t in trades:
            daily[t.session] = daily.get(t.session, 0.0) + t.net_usd
        ds_ = pd.Series(daily).sort_index() if daily else pd.Series(dtype=float)
        test_sessions = sorted({s for f in res["folds"]
                                for s in []}) or sorted(daily)
        pool = {s: [] for s in sorted(daily)}
        for t in trades:
            pool[t.session].append(t)
        mc_sweep = {}
        if pool:
            for size in ("50k", "100k"):
                for pdll in (300.0, 500.0, 750.0, 1000.0):
                    mc_sweep[f"{size}/pdll{int(pdll)}"] = {
                        k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in vars(lifecycle_mc_chain(
                            [pool[s] for s in sorted(pool)], cfg, size=size,
                            personal_dll=pdll, n_runs=300,
                            horizon_sessions=252)).items()}
        best_cfg = (max(mc_sweep, key=lambda k: mc_sweep[k]["ev_month"])
                    if mc_sweep else None)
        report["tracks"][track] = {
            "instrument": res["inst"],
            "fold_summary": [{k: f[k] for k in
                              ("fold", "best_family", "best_target",
                               "val_auc", "test_auc", "policy")}
                             for f in res["folds"]],
            "fold_nets": [round(sum(t["net_usd"] for t in f["trades"]))
                          for f in res["folds"]],
            "shuffled_control_auc": res["folds"][0].get("shuffled_test_auc"),
            "oos": {"trades": n,
                    "win_rate": wins / n if n else None,
                    "avg_win": float(np.mean([t.net_usd for t in trades
                                              if t.net_usd > 0]))
                    if wins else 0.0,
                    "avg_loss": float(np.mean([t.net_usd for t in trades
                                               if t.net_usd <= 0]))
                    if n > wins else 0.0,
                    "net": float(sum(t.net_usd for t in trades)),
                    "gross": float(sum(t.gross_usd for t in trades)),
                    "costs": float(sum(t.costs_usd for t in trades)),
                    "max_dd": max_drawdown(ds_) if n else 0.0,
                    "active_sessions": len(pool),
                    "trades_per_active_session": (n / len(pool)
                                                  if pool else 0.0)},
            "mc_sweep": mc_sweep, "best_account_config": best_cfg,
            "importance_lastfold": res["folds"][-1].get("importance"),
        }
    (outdir / "ml_discovery_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"wrote {outdir}/ml_discovery_report.json "
          f"({report['runtime_min']} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
