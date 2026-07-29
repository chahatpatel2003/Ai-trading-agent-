#!/usr/bin/env python
"""Phase 5B re-smoke: quantile-thresholded micro candidates on 2024-H1.

Same protocol as Phase 5A's smoke (3 mini-folds, train-only selection,
full costs/rules, chained renewal MC), with the self-calibrating fq_*
indicator battery. PRE-DECLARED BAR for the full-history micro build:
walk-forward stream must show trades/session >= 1.0 AND win rate >= 0.55
AND net > 0 (EV/blowout reported; the full search remains the decider).

Writes outputs/micro_smoke2_report.{md,json}.
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
from tslab.strategy.rules import quantile_micro_grid              # noqa: E402

BAR = {"tps": 1.0, "win": 0.55}


def main() -> int:
    t0 = _time.time()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    fp = FeatureParams()
    bars = pd.read_parquet(outdir / "nq_micro_bars_2024H1.parquet")
    sessions = sorted(set(bars["session"]))
    print(f"bars={len(bars):,} sessions={len(sessions)}", flush=True)

    t = _time.time()
    feats = compute_micro_q_features(bars, cfg, fp)
    qcols = [c for c in feats.columns if c.startswith("fq_")]
    live = {c: float(feats[c].mean()) for c in qcols if feats[c].notna().any()}
    print(f"features: {len([c for c in feats.columns if c.startswith('f')])} "
          f"({len(qcols)} fq indicators, {len(live)} live) "
          f"in {_time.time() - t:.0f}s", flush=True)

    aud = TruncationInvarianceCheck(
        bars, lambda df: compute_micro_q_features(df, cfg, fp),
        n_probes=5).run()
    aud.name = "quantile_feature_truncation"
    print(f"  [{aud.status.value}] {aud.name}: {aud.message}", flush=True)
    if aud.status == Status.FAIL:
        LeakageAuditor.write_report([aud], outdir, "micro_smoke2")
        return 1

    grid = quantile_micro_grid()
    folds = make_session_folds(sessions, cfg, n_train=60, n_test=21,
                               embargo_sessions=1)
    print(f"candidates={len(grid)} mini-folds={len(folds)}", flush=True)
    out = run_search(grid, bars, feats, folds, cfg,
                     progress=lambda m: print(m, flush=True), min_trades=40)

    all_tr = [t_ for v in out.wf_trades_by_session.values() for t_ in v]
    n = len(all_tr)
    wins = sum(1 for t_ in all_tr if t_.net_usd > 0)
    wf_net = float(out.wf_daily.sum())
    wf = {
        "net": wf_net, "trades": n, "sessions": len(out.wf_sessions),
        "trades_per_session": n / max(len(out.wf_sessions), 1),
        "win_rate": wins / n if n else float("nan"),
        "avg_win": float(np.mean([t_.net_usd for t_ in all_tr
                                  if t_.net_usd > 0])) if wins else 0.0,
        "avg_loss": float(np.mean([t_.net_usd for t_ in all_tr
                                   if t_.net_usd <= 0])) if n > wins else 0.0,
        "gross": float(sum(t_.gross_usd for t_ in all_tr)),
        "costs": float(sum(t_.costs_usd for t_ in all_tr)),
        "max_dd": max_drawdown(out.wf_daily),
        "positive_folds": sum(1 for f in out.fold_table
                              if (f["test_net"] or 0) > 0),
    }
    pool = [out.wf_trades_by_session[s]
            for s in sorted(out.wf_trades_by_session)]
    mc = lifecycle_mc_chain(pool, cfg, n_runs=300, horizon_sessions=252)
    print(f"WF: net=${wf_net:,.0f} n={n} tps={wf['trades_per_session']:.2f} "
          f"win={(wf['win_rate'] if n else 0):.3f} | EV/mo=${mc.ev_month:,.0f}"
          f" blowout={mc.blowout_rate:.2f} P(pay1)={mc.p_payout1:.2f}",
          flush=True)

    audits = [aud]
    for name in sorted({f["winner"] for f in out.fold_table if f["winner"]}):
        rule = next(r for r in grid if r.name == name)

        def sig_fn(df, rule=rule):
            fe = compute_micro_q_features(df, cfg, fp)
            return pd.DataFrame({"sig": rule.entry_signal(fe, cfg)
                                 .astype(float)}, index=df.index)

        a = TruncationInvarianceCheck(bars, sig_fn, n_probes=3).run()
        a.name = f"signal_truncation:{name}"
        audits.append(a)
        print(f"  [{a.status.value}] {a.name}", flush=True)
    LeakageAuditor.write_report(audits, outdir, "micro_smoke2")
    audits_ok = all(a.status != Status.FAIL for a in audits)

    clears_bar = (out.violations == 0 and audits_ok and n > 0
                  and wf["trades_per_session"] >= BAR["tps"]
                  and wf["win_rate"] >= BAR["win"] and wf_net > 0)

    report = {
        "phase": "5B quantile micro re-smoke", "candidates": len(grid),
        "violations": out.violations, "wf": wf,
        "chain_mc": {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in vars(mc).items()},
        "fold_table": out.fold_table,
        "candidate_summary": out.candidate_summary.head(15).to_dict("records"),
        "audits": [{"name": a.name, "status": a.status.value}
                   for a in audits],
        "clears_bar_for_full_build": bool(clears_bar),
        "bar": BAR,
        "runtime_min": round((_time.time() - t0) / 60, 1),
    }
    (outdir / "micro_smoke2_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    md = ["# Quantile micro re-smoke (Phase 5B)", "",
          f"**clears bar for full build: {clears_bar}** "
          f"(bar: tps>={BAR['tps']}, win>={BAR['win']}, net>0)", "",
          f"- WF: net ${wf['net']:,.0f} (gross ${wf['gross']:,.0f} costs "
          f"${wf['costs']:,.0f}), {n} trades "
          f"({wf['trades_per_session']:.2f}/session), win "
          f"{(wf['win_rate'] if n else 0):.3f}, avg win ${wf['avg_win']:.0f} "
          f"/ loss ${wf['avg_loss']:.0f}, maxDD ${wf['max_dd']:,.0f}, +folds "
          f"{wf['positive_folds']}/{len(folds)}",
          f"- chain MC: EV/month ${mc.ev_month:,.0f}, P(pay1/2/3) "
          f"{mc.p_payout1:.2f}/{mc.p_payout2:.2f}/{mc.p_payout3:.2f}, "
          f"blowout {mc.blowout_rate:.2f}, DLL {mc.dll_hit_rate:.3f}",
          "", "| fold | winner | train t | test net $ | trades |",
          "|---:|---|---:|---:|---:|"]
    md += [f"| {f['fold']} | {f['winner']} | {f['train_tstat']} | "
           f"{f['test_net']} | {f['test_trades']} |"
           for f in report["fold_table"]]
    md += ["", "## Top candidates (aggregate mini-OOS)", ""]
    for r in report["candidate_summary"][:8]:
        md.append(f"- {r['name']}: net={r['oos_net']:.0f} "
                  f"win={r['oos_win_rate']:.3f} tps={r['trades_per_session']:.2f}"
                  f" +share={r['pos_fold_share']:.2f}")
    md += ["", "## Audits", ""]
    md += [f"- [{a['status']}] {a['name']}" for a in report["audits"]]
    (outdir / "micro_smoke2_report.md").write_text("\n".join(md))
    print(f"wrote {outdir}/micro_smoke2_report.md|.json "
          f"({report['runtime_min']} min)", flush=True)
    return 0 if (out.violations == 0 and audits_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
