#!/usr/bin/env python
"""High-frequency / high-win-rate NQ search (renewal-process objective).

PRE-REGISTERED GATES (walk-forward stream, declared before the run):
    G1  zero Topstep violations        G2  signal audits PASS
    G3  OOS win rate >= 0.55           G4  trades/session >= 1.0
    G5  OOS net > 0 (trading costs)    G6  chained-MC monthly EV > 0
    G7  blowout rate < 0.50            G8  DLL hit rate <= 0.15
    G9  >= 9/17 positive folds         G10 |maxDD| <= 0.6 x net
Candidate-level qualification for chain-MC: win rate >= 0.55,
trades/session >= 1.0, OOS net > 0, pos_fold_share >= 0.5.
Chained MC: 300 runs x 252 sessions; combine MLL fail -> paid reset;
funded blowout -> new combine. Reserved tail (>= 2026-04-10) NOT used.

Writes outputs/hf_search_report.{md,json} + wf_hf_trades.parquet.
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
from tslab.strategy.evaluate import (lifecycle_mc_chain,          # noqa: E402
                                     max_drawdown)
from tslab.strategy.full_search import (audit_signal,             # noqa: E402
                                        load_fold_plan,
                                        run_search, verify_fold_plan)
from tslab.strategy.rules import hf_candidate_grid                # noqa: E402

MIN_TRAIN_TRADES = 200          # >= ~0.8/day on train, "enough to trust"
RESERVED_TAIL_START = pd.Timestamp("2026-04-10", tz="UTC")
GATES = {"min_win_rate": 0.55, "min_tps": 1.0, "min_pos_folds": 9,
         "max_blowout": 0.50, "max_dll_rate": 0.15, "dd_ratio": 0.6,
         "top_min_pos_share": 0.5}


def chain(pool_by_session, cfg, size="50k", pdll=1000.0):
    pool = [pool_by_session[s] for s in sorted(pool_by_session)]
    return lifecycle_mc_chain(pool, cfg, size=size, personal_dll=pdll)


def main() -> int:
    t0 = _time.time()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    bars = pd.read_parquet(outdir / "nq_bars_full.parquet")
    feats = pd.read_parquet(outdir / "nq_features_full.parquet")
    end_times = pd.read_parquet(outdir / "nq_label_endtimes_full.parquet")
    folds = load_fold_plan(outdir / "fold_plan.json")
    assert all(f["test_end"] < RESERVED_TAIL_START for f in folds)
    grid = hf_candidate_grid()
    print(f"bars={len(bars):,} candidates={len(grid)} folds={len(folds)} "
          f"min_train_trades={MIN_TRAIN_TRADES}", flush=True)

    checks = verify_fold_plan(folds, end_times, pd.Timedelta(hours=24))
    for r in checks:
        print(f"  [{r.status.value}] {r.name}", flush=True)
    if any(r.status == Status.FAIL for r in checks):
        return 1

    out = run_search(grid, bars, feats, folds, cfg,
                     progress=lambda m: print(m, flush=True),
                     min_trades=MIN_TRAIN_TRADES)

    all_tr = [t for v in out.wf_trades_by_session.values() for t in v]
    wf_net = float(out.wf_daily.sum())
    wf_dd = max_drawdown(out.wf_daily)
    n = len(all_tr)
    wins = sum(1 for t in all_tr if t.net_usd > 0)
    wf = {
        "net": wf_net, "trades": n,
        "trades_per_session": n / max(len(out.wf_sessions), 1),
        "win_rate": wins / n if n else float("nan"),
        "avg_win": float(np.mean([t.net_usd for t in all_tr if t.net_usd > 0]))
        if wins else 0.0,
        "avg_loss": float(np.mean([t.net_usd for t in all_tr
                                   if t.net_usd <= 0]))
        if n > wins else 0.0,
        "gross": float(sum(t.gross_usd for t in all_tr)),
        "costs": float(sum(t.costs_usd for t in all_tr)),
        "max_dd": wf_dd,
        "positive_folds": sum(1 for f in out.fold_table
                              if (f["test_net"] or 0) > 0),
        "sessions": len(out.wf_sessions),
    }
    print(f"WF: net=${wf_net:,.0f} n={n} tps={wf['trades_per_session']:.2f} "
          f"win={wf['win_rate']:.3f} dd=${wf_dd:,.0f} "
          f"posFolds={wf['positive_folds']}/17", flush=True)

    mc = {}
    for pdll in (750.0, 1000.0):
        mc[f"50k/pdll{int(pdll)}"] = chain(out.wf_trades_by_session, cfg,
                                           pdll=pdll)
    best_cfg = max(mc, key=lambda k: mc[k].ev_month)
    wf_mc = mc[best_cfg]
    print(f"WF chain-MC best {best_cfg}: EV/month=${wf_mc.ev_month:,.0f} "
          f"blowout={wf_mc.blowout_rate:.2f} P(pay1)={wf_mc.p_payout1:.2f}",
          flush=True)

    # candidate-level qualification -> chain MC only for qualifiers
    qual_rows = []
    summary = out.candidate_summary
    for _, row in summary.iterrows():
        q = (row["oos_win_rate"] >= GATES["min_win_rate"]
             and row["trades_per_session"] >= GATES["min_tps"]
             and row["oos_net"] > 0
             and row["pos_fold_share"] >= GATES["top_min_pos_share"])
        if q:
            st = chain(out.top_candidate_trades.get(row["name"], {}), cfg) \
                if row["name"] in out.top_candidate_trades else None
            qual_rows.append({**row.to_dict(),
                              "mc": None if st is None else
                              {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in vars(st).items()}})
    print(f"candidate qualifiers: {[r['name'] for r in qual_rows]}",
          flush=True)

    fold_winners = [f["winner"] for f in out.fold_table if f["winner"]]
    audits = []
    for name in sorted(set(fold_winners))[:5]:
        rule = next(r for r in grid if r.name == name)
        a = audit_signal(rule, bars, cfg)
        audits.append(a)
        print(f"  [{a.status.value}] {a.name}", flush=True)
    audits_ok = all(a.status != Status.FAIL for a in audits)

    verdict = {
        "G1_zero_violations": out.violations == 0,
        "G2_audits_pass": audits_ok,
        "G3_win_rate": bool(n and wf["win_rate"] >= GATES["min_win_rate"]),
        "G4_frequency": wf["trades_per_session"] >= GATES["min_tps"],
        "G5_net_positive": wf_net > 0,
        "G6_monthly_ev_positive": wf_mc.ev_month > 0,
        "G7_blowout_ok": wf_mc.blowout_rate < GATES["max_blowout"],
        "G8_dll_ok": wf_mc.dll_hit_rate <= GATES["max_dll_rate"],
        "G9_fold_stability": wf["positive_folds"] >= GATES["min_pos_folds"],
        "G10_dd_ok": wf_net > 0 and abs(wf_dd) <= GATES["dd_ratio"] * wf_net,
    }
    viable = all(verdict.values())

    rows = [asdict(t) for s in sorted(out.wf_trades_by_session)
            for t in out.wf_trades_by_session[s]]
    if rows:
        pd.DataFrame(rows).to_parquet(outdir / "wf_hf_trades.parquet")

    report = {
        "candidate_class": "hf_high_winrate", "candidates": len(grid),
        "gates": GATES, "min_train_trades": MIN_TRAIN_TRADES,
        "violations": out.violations, "fold_table": out.fold_table,
        "wf": wf,
        "wf_chain_mc": {k: {kk: (round(vv, 4) if isinstance(vv, float)
                                 else vv) for kk, vv in vars(v).items()}
                        for k, v in mc.items()},
        "best_account_config": best_cfg,
        "qualified_candidates": qual_rows,
        "candidate_summary": summary.head(25).to_dict("records"),
        "signal_audits": [{"name": a.name, "status": a.status.value}
                          for a in audits],
        "verdict": verdict, "viable": viable,
        "runtime_min": round((_time.time() - t0) / 60, 1),
    }
    (outdir / "hf_search_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    _write_md(report, grid, outdir / "hf_search_report.md")
    print(f"\nVIABLE={viable}", flush=True)
    print(f"verdict={verdict}", flush=True)
    print(f"wrote {outdir}/hf_search_report.md|.json "
          f"({report['runtime_min']} min)", flush=True)
    return 0 if (out.violations == 0 and audits_ok) else 1


def _write_md(report, grid, path: Path) -> None:
    rules = {r.name: r for r in grid}
    wf = report["wf"]
    L = ["# High-frequency / high-win-rate NQ search", "",
         f"**VIABLE: {report['viable']}**", "",
         "| gate | result |", "|---|---|"]
    L += [f"| {k} | {'PASS' if v else 'FAIL'} |"
          for k, v in report["verdict"].items()]
    L += ["", "## Walk-forward stream", "",
          f"- net ${wf['net']:,.0f} (gross ${wf['gross']:,.0f}, costs "
          f"${wf['costs']:,.0f}) over {wf['trades']} trades "
          f"({wf['trades_per_session']:.2f}/session)",
          f"- win rate {wf['win_rate']:.3f}, avg win ${wf['avg_win']:.0f} / "
          f"avg loss ${wf['avg_loss']:.0f}, maxDD ${wf['max_dd']:,.0f}, "
          f"positive folds {wf['positive_folds']}/17", "",
          "## Renewal-process Monte Carlo (chained accounts, 12 months)", "",
          "| config | EV/month $ | EV/yr $ | P(pass) | P(pay>=1/2/3) | "
          "E[payouts/yr] | blowout | DLL rate |",
          "|---|---:|---:|---:|---|---:|---:|---:|"]
    for k, st in report["wf_chain_mc"].items():
        L.append(f"| {k} | {st['ev_month']:.0f} | {st['ev_year']:.0f} | "
                 f"{st['p_pass']:.2f} | {st['p_payout1']:.2f}/"
                 f"{st['p_payout2']:.2f}/{st['p_payout3']:.2f} | "
                 f"{st['mean_payouts_year']:.1f} | {st['blowout_rate']:.2f} "
                 f"| {st['dll_hit_rate']:.3f} |")
    L += ["", "| fold | winner | train t | test net $ | trades |",
          "|---:|---|---:|---:|---:|"]
    for f in report["fold_table"]:
        L.append(f"| {f['fold']} | {f['winner']} | {f['train_tstat']} | "
                 f"{f['test_net']} | {f['test_trades']} |")
    L += ["", "## Qualified candidates (win>=55%, tps>=1, net>0, share>=0.5)",
          ""]
    if report["qualified_candidates"]:
        for r in report["qualified_candidates"]:
            L.append(f"- {r['name']}: net={r['oos_net']:.0f} "
                     f"win={r['oos_win_rate']:.3f} tps="
                     f"{r['trades_per_session']:.2f} mc={r['mc']}")
    else:
        L.append("- none")
    L += ["", "## Selected rules", ""]
    for name in sorted({f["winner"] for f in report["fold_table"]
                        if f["winner"]}):
        r = rules[name]
        L += [f"### {name}",
              f"- {'LONG' if r.direction > 0 else 'SHORT'}; CT "
              f"{r.entry_start_ct}..{r.entry_end_ct}; stop {r.stop_atr}xATR "
              f"target {r.target_atr}xATR (RR {r.rr:.2f}); ts "
              f"{r.time_stop_bars}; max {r.max_trades_per_day}/day; day stop "
              f"-${r.daily_loss_usd:.0f}",
              "- filters: " + "; ".join(
                  f"{f.feature} {f.op} {f.lo}"
                  + (f"..{f.hi}" if f.hi is not None else "")
                  for f in r.filters), ""]
    L += ["## Audits", ""]
    L += [f"- [{a['status']}] {a['name']}" for a in report["signal_audits"]]
    L += ["", "## Caveats", "",
          "- Gates pre-declared; reserved 2026 tail untouched; MC bootstraps "
          "the OOS pool; monthly EV includes subscriptions, activation, "
          "resets, and the 90/10 split."]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
