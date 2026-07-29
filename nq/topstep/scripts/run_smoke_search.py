#!/usr/bin/env python
"""Phase 3A smoke test: tiny walk-forward candidate search on the 9-session
NQ sample. NOT a research result — it validates the machinery end to end
(rules -> backtest -> costs -> TopstepSimulator -> lifecycle MC -> audits).

Writes outputs/smoke_search_results.md / .json. Exit != 0 if the leakage
audit of the selected candidate fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab.config import load_config                          # noqa: E402
from tslab.features.core import FeatureParams, compute_features  # noqa: E402
from tslab.strategy.rules import candidate_grid               # noqa: E402
from tslab.strategy.search import walkforward_search          # noqa: E402


def main() -> int:
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    bars = pd.read_parquet(outdir / "nq_bars_sample.parquet")
    feats = compute_features(bars, cfg, FeatureParams())
    cands = candidate_grid()
    print(f"bars={len(bars):,} sessions={bars['session'].nunique()} "
          f"candidates={len(cands)}")

    res = walkforward_search(cands, bars, feats, cfg,
                             n_train=5, n_test=2, embargo_sessions=1,
                             mc_runs=200, mc_horizon=250)

    lines = ["# Smoke search results (NOT a research result)", "",
             f"- candidates: {len(cands)}  folds: {len(res.fold_report)}  "
             f"sessions: {bars['session'].nunique()}",
             "- selection: mean TRAIN net; OOS = held-out test sessions; "
             "lifecycle MC bootstraps OOS sessions (200 runs x 250 sessions)",
             ""]
    for fr in res.fold_report:
        lines += [f"## Fold {fr['fold']}",
                  f"- train: {[str(s) for s in fr['train_sessions']]}",
                  f"- test: {[str(s) for s in fr['test_sessions']]}",
                  "", "| candidate | train net $ | train trades | "
                  "test net $ | test trades |", "|---|---:|---:|---:|---:|"]
        for name, trn, ntr, ten, nte in fr["ranking"]:
            lines.append(f"| {name} | {trn:.0f} | {ntr} | {ten:.0f} | {nte} |")
        lines.append("")

    lines += ["## Candidate scores (ordered by train selection)", "",
              "| candidate | oos trades | oos net $ | EV net payout $ | "
              "pass prob | DLL rate | feasible | notes |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for sc in res.scores:
        life = sc.lifecycle
        lines.append(
            f"| {sc.rule_name} | {sc.oos_trades} | {sc.oos_net:.0f} | "
            f"{life.ev_net_payout:.0f} | {life.pass_prob:.2f} | "
            f"{life.dll_hit_rate:.2f} | {sc.feasible} | "
            f"{'; '.join(sc.reasons)} |"
            if life else
            f"| {sc.rule_name} | 0 | 0 | - | - | - | {sc.feasible} | "
            f"{'; '.join(sc.reasons) or 'no OOS trades'} |")

    lines += ["", "## Selected candidate + audit", ""]
    if res.best:
        lines.append(f"- best feasible by train selection: "
                     f"**{res.best.rule_name}** (OOS net "
                     f"${res.best.oos_net:.0f}, EV net payout "
                     f"${res.best.lifecycle.ev_net_payout:.0f})")
    else:
        lines.append("- no feasible candidate on this tiny sample "
                     "(expected for a smoke test)")
    for r in res.audit_results:
        lines.append(f"- [{r.status.value}] {r.name}: {r.message}")
        print(f"  [{r.status.value}] {r.name}: {r.message}")
    lines += ["", "Caveats: 9 sessions, 2 test sessions per fold, fixed "
              "thresholds, qty=1 — numbers are machinery validation only."]

    (outdir / "smoke_search_results.md").write_text("\n".join(lines))
    with open(outdir / "smoke_search_results.json", "w") as f:
        json.dump({
            "best": res.best.rule_name if res.best else None,
            "audited_ok": res.audited_ok,
            "scores": [{
                "rule": sc.rule_name, "oos_trades": sc.oos_trades,
                "oos_net": sc.oos_net, "violations": sc.violations,
                "feasible": sc.feasible, "reasons": sc.reasons,
                "train_net_by_fold": sc.train_net_by_fold,
                "test_net_by_fold": sc.test_net_by_fold,
                "lifecycle": (vars(sc.lifecycle) if sc.lifecycle else None),
            } for sc in res.scores],
        }, f, indent=2, default=str)
    print(f"wrote {outdir}/smoke_search_results.md and .json")
    if res.best and not res.audited_ok:
        print("LEAKAGE AUDIT FAILED for selected candidate", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
