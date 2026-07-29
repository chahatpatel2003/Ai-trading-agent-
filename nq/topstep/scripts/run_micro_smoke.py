#!/usr/bin/env python
"""Phase 5A smoke search: microstructure candidates on 2024-H1 micro bars.

SMOKE SCOPE — machinery + first signal read, NOT a viability verdict:
125 sessions, mini walk-forward (60 train / 1 embargo / 21 test, 3 folds),
train-only selection (min 40 train trades), full costs and account rules,
chained renewal MC on the mini-OOS pool (labelled high-variance).

Gate thresholds reported against the Phase 5 targets (win>=55%, >=1
trade/session, monthly EV>0, blowout<50%) but the verdict field is
'smoke_promising' / 'smoke_negative' — full-history search decides later.

Writes outputs/micro_smoke_report.{md,json}.
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
from tslab.features.micro import compute_micro_features           # noqa: E402
from tslab.features.labels import make_session_folds              # noqa: E402
from tslab.strategy.evaluate import lifecycle_mc_chain, max_drawdown  # noqa: E402
from tslab.strategy.full_search import run_search                 # noqa: E402
from tslab.strategy.rules import micro_candidate_grid             # noqa: E402

TARGETS = {"win": 0.55, "tps": 1.0, "blowout": 0.50}


def main() -> int:
    t0 = _time.time()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    fp = FeatureParams()
    bars = pd.read_parquet(outdir / "nq_micro_bars_2024H1.parquet")
    sessions = sorted(set(bars["session"]))
    print(f"micro bars={len(bars):,} sessions={len(sessions)} "
          f"micro coverage={bars['imb_last'].notna().mean():.1%}", flush=True)

    feats = compute_micro_features(bars, cfg, fp)
    mcols = [c for c in feats.columns if c.startswith('f_m_')]
    print(f"features: {len([c for c in feats.columns if c.startswith('f_')])}"
          f" total, {len(mcols)} micro", flush=True)

    # leakage gate on the COMBINED feature function (base + micro)
    aud = TruncationInvarianceCheck(
        bars, lambda df: compute_micro_features(df, cfg, fp), n_probes=6).run()
    aud.name = "micro_feature_truncation"
    print(f"  [{aud.status.value}] {aud.name}: {aud.message}", flush=True)
    if aud.status == Status.FAIL:
        LeakageAuditor.write_report([aud], outdir, "micro_smoke")
        return 1

    grid = micro_candidate_grid()
    folds = make_session_folds(sessions, cfg, n_train=60, n_test=21,
                               embargo_sessions=1)
    print(f"candidates={len(grid)} mini-folds={len(folds)}", flush=True)
    out = run_search(grid, bars, feats, folds, cfg,
                     progress=lambda m: print(m, flush=True), min_trades=40)

    all_tr = [t for v in out.wf_trades_by_session.values() for t in v]
    n = len(all_tr)
    wins = sum(1 for t in all_tr if t.net_usd > 0)
    wf_net = float(out.wf_daily.sum())
    wf = {
        "net": wf_net, "trades": n,
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
    }
    pool = [out.wf_trades_by_session[s] for s in sorted(out.wf_trades_by_session)]
    mc = lifecycle_mc_chain(pool, cfg, n_runs=300, horizon_sessions=252)
    print(f"WF: net=${wf_net:,.0f} n={n} tps={wf['trades_per_session']:.2f} "
          f"win={wf['win_rate'] if n else 0:.3f} | chainMC EV/mo="
          f"${mc.ev_month:,.0f} blowout={mc.blowout_rate:.2f}", flush=True)

    # audit signals of fold winners
    from tslab.strategy.rules import TradeRule  # noqa: F401
    audits = [aud]
    for name in sorted({f["winner"] for f in out.fold_table if f["winner"]}):
        rule = next(r for r in grid if r.name == name)

        def sig_fn(df, rule=rule):
            fe = compute_micro_features(df, cfg, fp)
            return pd.DataFrame({"sig": rule.entry_signal(fe, cfg)
                                 .astype(float)}, index=df.index)

        a = TruncationInvarianceCheck(bars, sig_fn, n_probes=4).run()
        a.name = f"signal_truncation:{name}"
        audits.append(a)
        print(f"  [{a.status.value}] {a.name}", flush=True)
    LeakageAuditor.write_report(audits, outdir, "micro_smoke")
    audits_ok = all(a.status != Status.FAIL for a in audits)

    promising = (out.violations == 0 and audits_ok and n > 0
                 and wf["win_rate"] >= TARGETS["win"]
                 and wf["trades_per_session"] >= TARGETS["tps"]
                 and wf_net > 0 and mc.ev_month > 0
                 and mc.blowout_rate < TARGETS["blowout"])

    report = {
        "phase": "5A micro smoke", "scope": "2024-H1, 3 mini-folds",
        "candidates": len(grid), "violations": out.violations,
        "wf": wf,
        "chain_mc": {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in vars(mc).items()},
        "fold_table": out.fold_table,
        "candidate_summary": out.candidate_summary.head(15).to_dict("records"),
        "audits": [{"name": a.name, "status": a.status.value} for a in audits],
        "verdict": "smoke_promising" if promising else "smoke_negative",
        "runtime_min": round((_time.time() - t0) / 60, 1),
    }
    (outdir / "micro_smoke_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    md = ["# Micro smoke search (Phase 5A)", "",
          f"**verdict: {report['verdict']}** (smoke scope — not a final "
          "viability call)", "",
          f"- WF: net ${wf['net']:,.0f} (gross ${wf['gross']:,.0f} costs "
          f"${wf['costs']:,.0f}), {n} trades "
          f"({wf['trades_per_session']:.2f}/session), win "
          f"{wf['win_rate'] if n else 0:.3f}, maxDD ${wf['max_dd']:,.0f}, "
          f"+folds {wf['positive_folds']}/{len(folds)}",
          f"- chain MC: EV/month ${mc.ev_month:,.0f}, P(pay1/2/3) "
          f"{mc.p_payout1:.2f}/{mc.p_payout2:.2f}/{mc.p_payout3:.2f}, "
          f"blowout {mc.blowout_rate:.2f}, DLL rate {mc.dll_hit_rate:.3f}",
          "", "| fold | winner | train t | test net $ | trades |",
          "|---:|---|---:|---:|---:|"]
    md += [f"| {f['fold']} | {f['winner']} | {f['train_tstat']} | "
           f"{f['test_net']} | {f['test_trades']} |"
           for f in report["fold_table"]]
    md += ["", "## Audits", ""]
    md += [f"- [{a['status']}] {a['name']}" for a in report["audits"]]
    (outdir / "micro_smoke_report.md").write_text("\n".join(md))
    print(f"wrote {outdir}/micro_smoke_report.md|.json "
          f"({report['runtime_min']} min)", flush=True)
    return 0 if (out.violations == 0 and audits_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
