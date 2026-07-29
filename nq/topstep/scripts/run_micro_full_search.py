#!/usr/bin/env python
"""Phase 5D: full quote-era dual-instrument micro search.

Era split (PRE-REGISTERED, hard-asserted):
    search era:        2021-01-05 .. 2024-12-31  (selection + OOS folds)
    confirmation tail: 2025-01-02 .. 2025-06-30  (NEVER touched here; the
                       tail rows are excluded from every frame this script
                       evaluates — features for the search era are computed
                       on the search-era frame alone)

HARD GATES (search-era WF stream, MNQ primary):
    win >= 0.55 | trades/session >= 1.0 (2-8/day preferred) | net > 0 |
    chain-MC EV/month > 0 | blowout < 0.50 (<0.40 preferred) |
    zero violations | audits pass.  (+ tail confirmation, separate script.)

2024-H1 test windows are flagged smoke-contaminated; WF stats are reported
with and without those folds.

Pre-registers the tail candidate set in outputs/tail_prereg_micro.json:
the WF selection-process arm (trained on the last 250 search-era sessions)
plus the top-3 gate-passing MNQ candidates by mean train t-stat.

Writes outputs/micro_full_search_report.{md,json}, fold_plan_micro.json,
wf_micro_trades.parquet.
"""
from __future__ import annotations

import json
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
from tslab.features.quantiles import compute_micro_q_features        # noqa: E402
from tslab.strategy.evaluate import lifecycle_mc_chain, max_drawdown  # noqa: E402
from tslab.strategy.full_search import (candidate_gates,             # noqa: E402
                                        run_search)
from tslab.strategy.rules import exec_micro_grids                    # noqa: E402

SEARCH_END = date(2024, 12, 31)
TAIL_START = date(2025, 1, 1)
TAIL_END = date(2025, 6, 30)
SMOKE_LO, SMOKE_HI = date(2024, 1, 1), date(2024, 6, 30)
GATES = {"win": 0.55, "tps": 1.0, "blowout": 0.50, "blowout_pref": 0.40}
MIN_TRAIN_TRADES = 200


def stream_stats(out):
    all_tr = [t for v in out.wf_trades_by_session.values() for t in v]
    n = len(all_tr)
    wins = sum(1 for t in all_tr if t.net_usd > 0)
    return all_tr, {
        "net": float(out.wf_daily.sum()), "trades": n,
        "sessions": len(out.wf_sessions),
        "trades_per_session": n / max(len(out.wf_sessions), 1),
        "win_rate": wins / n if n else float("nan"),
        "avg_win": float(np.mean([t.net_usd for t in all_tr
                                  if t.net_usd > 0])) if wins else 0.0,
        "avg_loss": float(np.mean([t.net_usd for t in all_tr
                                   if t.net_usd <= 0])) if n > wins else 0.0,
        "gross": float(sum(t.gross_usd for t in all_tr)),
        "costs": float(sum(t.costs_usd for t in all_tr)),
        "max_dd": max_drawdown(out.wf_daily),
        "positive_folds": sum(1 for f in out.fold_table
                              if (f["test_net"] or 0) > 0),
        "avg_risk": float(np.mean([t.risk_usd for t in all_tr])) if n else 0.0,
    }


def main() -> int:
    t0 = _time.time()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    fp = FeatureParams()
    full = pd.read_parquet(outdir / "nq_micro_bars_full.parquet")
    bars = full[full["session"] <= SEARCH_END].copy()
    tail_sessions = sorted(set(full.loc[full["session"] >= TAIL_START,
                                        "session"]))
    del full
    assert max(set(bars["session"])) <= SEARCH_END          # tail excluded
    sessions = sorted(set(bars["session"]))
    print(f"search era: {len(sessions)} sessions "
          f"({sessions[0]}..{sessions[-1]}); tail reserved: "
          f"{len(tail_sessions)} ({tail_sessions[0]}..{tail_sessions[-1]})",
          flush=True)

    t = _time.time()
    feats = compute_micro_q_features(bars, cfg, fp)
    print(f"features in {_time.time() - t:.0f}s", flush=True)

    audits = []
    a = TruncationInvarianceCheck(
        bars, lambda df: compute_micro_q_features(df, cfg, fp),
        n_probes=5).run()
    a.name = "micro_q_truncation_full"
    audits.append(a)
    print(f"  [{a.status.value}] {a.name}: {a.message}", flush=True)

    folds = make_session_folds(sessions, cfg, n_train=250, n_test=63,
                               embargo_sessions=1)
    labres = compute_labels(bars, cfg, LabelParams())
    audits.append(FoldEmbargoCheck(folds, required_embargo(labres.specs)).run())
    purge = check_label_purge(labres.end_times, folds)
    from tslab.audit.leakage import CheckResult
    audits.append(CheckResult(
        "label_purge_per_fold", Status.PASS if not purge else Status.FAIL,
        "clean" if not purge else f"{len(purge)} violations"))
    for r in audits[1:]:
        print(f"  [{r.status.value}] {r.name}: {r.message}", flush=True)
    if any(r.status == Status.FAIL for r in audits):
        LeakageAuditor.write_report(audits, outdir, "micro_full")
        return 1
    (outdir / "fold_plan_micro.json").write_text(json.dumps({
        "n_folds": len(folds),
        "folds": [{k: str(v) for k, v in f.items()} for f in folds]},
        indent=2))

    grid_nq, grid_mnq = exec_micro_grids()
    results = {}
    for inst, grid, role in (("MNQ", grid_mnq, "primary"),
                             ("NQ", grid_nq, "control")):
        out = run_search(grid, bars, feats, folds, cfg,
                         progress=lambda m, i=inst: print(f"[{i}] {m}",
                                                          flush=True),
                         min_trades=MIN_TRAIN_TRADES, instrument=inst)
        all_tr, wf = stream_stats(out)
        contaminated = [k for k, f in enumerate(out.fold_table)
                        if not (date.fromisoformat(f["test_end"]) < SMOKE_LO
                                or date.fromisoformat(f["test_start"])
                                > SMOKE_HI)]
        clean_daily = out.wf_daily[
            [s for s in out.wf_daily.index
             if not (SMOKE_LO <= s <= SMOKE_HI)]]
        mc = {}
        for size in ("50k", "100k"):
            for pdll in (750.0, 1000.0):
                pool = [out.wf_trades_by_session[s]
                        for s in sorted(out.wf_trades_by_session)]
                mc[f"{size}/pdll{int(pdll)}"] = lifecycle_mc_chain(
                    pool, cfg, size=size, personal_dll=pdll,
                    n_runs=400, horizon_sessions=252)
        best_cfg = max(mc, key=lambda k: mc[k].ev_month)
        results[inst] = {"out": out, "wf": wf, "mc": mc,
                         "best_cfg": best_cfg,
                         "contaminated_folds": contaminated,
                         "clean_net": float(clean_daily.sum())}
        m = mc[best_cfg]
        print(f"[{inst}/{role}] WF net=${wf['net']:,.0f} n={wf['trades']} "
              f"tps={wf['trades_per_session']:.2f} "
              f"win={(wf['win_rate'] if wf['trades'] else 0):.3f} | "
              f"{best_cfg}: EV/mo=${m.ev_month:,.0f} "
              f"blowout={m.blowout_rate:.2f} P(pay1)={m.p_payout1:.2f}",
              flush=True)

    # winner signal audits (MNQ fold winners)
    mnq = results["MNQ"]
    for name in sorted({f["winner"] for f in mnq["out"].fold_table
                        if f["winner"]})[:5]:
        rule = next(r for r in grid_mnq if r.name == name)

        def sig_fn(df, rule=rule):
            fe = compute_micro_q_features(df, cfg, fp)
            return pd.DataFrame({"sig": rule.entry_signal(fe, cfg)
                                 .astype(float)}, index=df.index)

        a = TruncationInvarianceCheck(bars, sig_fn, n_probes=3).run()
        a.name = f"signal_truncation:{name}"
        audits.append(a)
        print(f"  [{a.status.value}] {a.name}", flush=True)
    LeakageAuditor.write_report(audits, outdir, "micro_full")
    audits_ok = all(x.status != Status.FAIL for x in audits)

    wf, best = mnq["wf"], mnq["mc"][mnq["best_cfg"]]
    verdict = {
        "G1_zero_violations": mnq["out"].violations == 0,
        "G2_audits_pass": audits_ok,
        "G3_win_rate": bool(wf["trades"]
                            and wf["win_rate"] >= GATES["win"]),
        "G4_frequency": wf["trades_per_session"] >= GATES["tps"],
        "G5_net_positive": wf["net"] > 0,
        "G6_ev_month_positive": best.ev_month > 0,
        "G7_blowout": best.blowout_rate < GATES["blowout"],
        "G8_tail_confirmation": None,        # separate one-shot script
    }
    search_pass = all(v for k, v in verdict.items()
                      if k != "G8_tail_confirmation")

    # ---- pre-register tail candidates (top-3 gate-passing + process arm)
    summary = mnq["out"].candidate_summary
    qual = [r for _, r in summary.iterrows() if candidate_gates(r.to_dict())]
    top3 = [r["name"] for r in qual[:3]]
    prereg = {
        "tail": {"start": str(TAIL_START), "end": str(TAIL_END)},
        "instrument": "MNQ",
        "candidates": top3,
        "process_arm": {"grid": "exec_micro_grids()[1]",
                        "n_train": 250, "embargo": 1,
                        "min_train_trades": MIN_TRAIN_TRADES},
        "tail_pass_criteria": {"net_gt": 0, "win_ge": 0.50, "tps_ge": 0.5,
                               "max_dd_abs_le": 4000, "violations": 0},
        "selection_rule": "highest search-era chain-MC EV among tail passers;"
                          " process arm reported alongside",
        "search_gates_passed": bool(search_pass),
    }
    (outdir / "tail_prereg_micro.json").write_text(
        json.dumps(prereg, indent=2))

    rows = [asdict(t) for s in sorted(mnq["out"].wf_trades_by_session)
            for t in mnq["out"].wf_trades_by_session[s]]
    if rows:
        pd.DataFrame(rows).to_parquet(outdir / "wf_micro_trades.parquet")

    def mcd(m):
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in vars(m).items()}

    report = {
        "phase": "5D full micro search",
        "search_era": [str(sessions[0]), str(sessions[-1])],
        "tail_reserved": [str(tail_sessions[0]), str(tail_sessions[-1])],
        "folds": len(folds), "gates": GATES, "verdict": verdict,
        "search_gates_passed": bool(search_pass),
        "instruments": {
            i: {"role": ("primary" if i == "MNQ" else "control"),
                "wf": results[i]["wf"],
                "clean_net_ex_2024H1": results[i]["clean_net"],
                "contaminated_folds": results[i]["contaminated_folds"],
                "best_cfg": results[i]["best_cfg"],
                "mc": {k: mcd(v) for k, v in results[i]["mc"].items()},
                "fold_table": results[i]["out"].fold_table,
                "candidate_summary": results[i]["out"].candidate_summary
                .head(12).to_dict("records")} for i in results},
        "tail_prereg": prereg,
        "audits": [{"name": x.name, "status": x.status.value}
                   for x in audits],
        "runtime_min": round((_time.time() - t0) / 60, 1),
    }
    (outdir / "micro_full_search_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"\nsearch_gates_passed={search_pass} prereg={top3}", flush=True)
    print(f"wrote {outdir}/micro_full_search_report.json "
          f"({report['runtime_min']} min)", flush=True)
    return 0 if (mnq["out"].violations == 0 and audits_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
