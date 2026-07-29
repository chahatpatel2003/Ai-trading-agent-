#!/usr/bin/env python
"""Phase 5C smoke: dual-instrument (NQ mini vs MNQ micros) quantile micro
search on 2024-H1. Same signals, instrument-fitted risk geometry.

PRE-DECLARED: per-instrument bar = trades/session >= 1.0 AND win >= 0.55
AND net > 0 on the walk-forward stream; instrument preference = the
bar-clearing stream with the higher chained monthly EV (blowout as
tie-break). Respects 10:1 micro counting; max position 4 micros = 0.4
mini-equivalents (far inside all account limits).

Writes outputs/micro_smoke3_report.{md,json}.
"""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab.audit.leakage import (LeakageAuditor, Status,          # noqa: E402
                                 TruncationInvarianceCheck)
from tslab.config import load_config                              # noqa: E402
from tslab.features.core import FeatureParams                     # noqa: E402
from tslab.features.labels import make_session_folds              # noqa: E402
from tslab.features.quantiles import compute_micro_q_features     # noqa: E402
from tslab.strategy.evaluate import lifecycle_mc_chain, max_drawdown  # noqa: E402
from tslab.strategy.full_search import run_search                 # noqa: E402
from tslab.strategy.rules import exec_micro_grids                 # noqa: E402

BAR = {"tps": 1.0, "win": 0.55}


def stream_stats(out):
    all_tr = [t for v in out.wf_trades_by_session.values() for t in v]
    n = len(all_tr)
    wins = sum(1 for t in all_tr if t.net_usd > 0)
    net = float(out.wf_daily.sum())
    return {
        "net": net, "trades": n, "sessions": len(out.wf_sessions),
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
    bars = pd.read_parquet(outdir / "nq_micro_bars_2024H1.parquet")
    sessions = sorted(set(bars["session"]))
    feats = compute_micro_q_features(bars, cfg, fp)

    aud = TruncationInvarianceCheck(
        bars, lambda df: compute_micro_q_features(df, cfg, fp),
        n_probes=4).run()
    aud.name = "quantile_feature_truncation"
    print(f"  [{aud.status.value}] {aud.name}", flush=True)
    if aud.status == Status.FAIL:
        return 1

    grid_nq, grid_mnq = exec_micro_grids()
    folds = make_session_folds(sessions, cfg, n_train=60, n_test=21,
                               embargo_sessions=1)
    print(f"candidates: NQ={len(grid_nq)} MNQ={len(grid_mnq)} "
          f"mini-folds={len(folds)}", flush=True)

    results = {}
    audits = [aud]
    for inst, grid in (("NQ", grid_nq), ("MNQ", grid_mnq)):
        out = run_search(grid, bars, feats, folds, cfg,
                         progress=lambda m, i=inst: print(f"[{i}] {m}",
                                                          flush=True),
                         min_trades=40, instrument=inst)
        wf = stream_stats(out)
        pool = [out.wf_trades_by_session[s]
                for s in sorted(out.wf_trades_by_session)]
        mc = lifecycle_mc_chain(pool, cfg, n_runs=300, horizon_sessions=252)
        clears = (out.violations == 0 and wf["trades"] > 0
                  and wf["trades_per_session"] >= BAR["tps"]
                  and wf["win_rate"] >= BAR["win"] and wf["net"] > 0)
        results[inst] = {"out": out, "wf": wf, "mc": mc, "clears": clears}
        print(f"[{inst}] WF net=${wf['net']:,.0f} n={wf['trades']} "
              f"tps={wf['trades_per_session']:.2f} "
              f"win={(wf['win_rate'] if wf['trades'] else 0):.3f} "
              f"avgRisk=${wf['avg_risk']:.0f} | EV/mo=${mc.ev_month:,.0f} "
              f"blowout={mc.blowout_rate:.2f} clears={clears}", flush=True)
        for name in sorted({f["winner"] for f in out.fold_table
                            if f["winner"]})[:3]:
            rule = next(r for r in grid if r.name == name)

            def sig_fn(df, rule=rule):
                fe = compute_micro_q_features(df, cfg, fp)
                return pd.DataFrame({"sig": rule.entry_signal(fe, cfg)
                                     .astype(float)}, index=df.index)

            a = TruncationInvarianceCheck(bars, sig_fn, n_probes=3).run()
            a.name = f"signal_truncation:{name}"
            audits.append(a)
            print(f"  [{a.status.value}] {a.name}", flush=True)

    LeakageAuditor.write_report(audits, outdir, "micro_smoke3")
    audits_ok = all(a.status != Status.FAIL for a in audits)

    clearing = [i for i in results if results[i]["clears"]]
    preferred = (max(clearing, key=lambda i: results[i]["mc"].ev_month)
                 if clearing else None)

    def mcd(mc):
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in vars(mc).items()}

    report = {
        "phase": "5C dual-instrument micro smoke", "bar": BAR,
        "preference_rule": "bar-clearing stream with higher chain EV/month",
        "instruments": {
            i: {"wf": results[i]["wf"], "chain_mc": mcd(results[i]["mc"]),
                "clears_bar": results[i]["clears"],
                "fold_table": results[i]["out"].fold_table,
                "candidate_summary": results[i]["out"].candidate_summary
                .head(10).to_dict("records")}
            for i in results},
        "preferred_instrument": preferred,
        "audits": [{"name": a.name, "status": a.status.value}
                   for a in audits],
        "violations": {i: results[i]["out"].violations for i in results},
        "runtime_min": round((_time.time() - t0) / 60, 1),
    }
    (outdir / "micro_smoke3_report.json").write_text(
        json.dumps(report, indent=2, default=str))

    md = ["# Dual-instrument micro smoke (Phase 5C)", "",
          f"**preferred instrument: {preferred or 'none — neither cleared'}**",
          ""]
    for i in results:
        wf, mc = results[i]["wf"], results[i]["mc"]
        md += [f"## {i}  (clears bar: {results[i]['clears']})", "",
               f"- net ${wf['net']:,.0f} (gross ${wf['gross']:,.0f} costs "
               f"${wf['costs']:,.0f}), {wf['trades']} trades "
               f"({wf['trades_per_session']:.2f}/session), win "
               f"{(wf['win_rate'] if wf['trades'] else 0):.3f}, avg risk "
               f"${wf['avg_risk']:.0f}, maxDD ${wf['max_dd']:,.0f}, +folds "
               f"{wf['positive_folds']}/{len(folds)}",
               f"- chain MC: EV/mo ${mc.ev_month:,.0f}, P(pay1/2/3) "
               f"{mc.p_payout1:.2f}/{mc.p_payout2:.2f}/{mc.p_payout3:.2f}, "
               f"blowout {mc.blowout_rate:.2f}, DLL {mc.dll_hit_rate:.3f}",
               ""]
        md += ["| fold | winner | test net $ | trades |", "|---:|---|---:|---:|"]
        md += [f"| {f['fold']} | {f['winner']} | {f['test_net']} | "
               f"{f['test_trades']} |" for f in results[i]["out"].fold_table]
        md.append("")
    md += ["## Audits", ""]
    md += [f"- [{a['status']}] {a['name']}" for a in report["audits"]]
    (outdir / "micro_smoke3_report.md").write_text("\n".join(md))
    print(f"\npreferred={preferred} wrote outputs/micro_smoke3_report.md|.json"
          f" ({report['runtime_min']} min)", flush=True)
    return 0 if (audits_ok and all(v == 0 for v in
                                   report["violations"].values())) else 1


if __name__ == "__main__":
    raise SystemExit(main())
