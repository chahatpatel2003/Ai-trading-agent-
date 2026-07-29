#!/usr/bin/env python
"""Phase 4B-2: ONE-SHOT reserved-tail confirmation (2026-04-10..2026-06-16).

PRE-REGISTRATION (declared before evaluation; nothing below is tuned on the
tail, and nothing will be iterated afterward):

  Candidates (exact Phase 4B-1 rules, unmodified, pulled from the same
  deterministic grid): pmT2_L_r2.0, orPM_L_r2.0, pmT2_L_r3.0, orPM_L_r3.0,
  pmT_L_w2_r1.5. Plus the walk-forward SELECTION PROCESS arm: select among
  the full unmodified 72-candidate grid by train t-stat (min 50 train
  trades) on the 250 sessions ending 1 embargo session before the tail.

  Per-candidate pass criteria:
    C1 tail net P&L > 0
    C2 zero Topstep violations
    C3 signal truncation audit PASS
    C4 tail |max drawdown| <= $4,000 (prior WF DD was $8.4k over 1,071
       sessions; a $4k drawdown inside ~45 sessions = collapse)
    C5 trade count >= 5 (below that: INSUFFICIENT, not FAILED)
    C6 no collapse vs prior OOS profile: tail avg net/trade >= -$150
  Verdict per candidate: CONFIRMED (all pass) / INSUFFICIENT (C5 fails,
  others pass) / FAILED.

  Strategy selection if >= 1 CONFIRMED: highest PRIOR Phase 4B-1 MC EV
  among confirmed — orPM_L_r2.0 ($612) > orPM_L_r3.0 ($504) > pmT2_L_r2.0
  ($444) > pmT_L_w2_r1.5 ($317) > pmT2_L_r3.0 ($312). Tail results are a
  confirm/deny gate only, NEVER a ranking key.

  Account pass: one deterministic TopstepSimulator pass (50k, personal DLL
  $1,000) over each candidate's tail trades for DLL/MLL behavior. No new
  Monte Carlo on tail data (too short; would recycle the confirmation set).

Writes outputs/tail_confirmation_report.{md,json}. Exit != 0 only on
violations or audit failure.
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

from tslab.audit.leakage import Status                            # noqa: E402
from tslab.config import load_config                              # noqa: E402
from tslab.sim.costs import CostModel                             # noqa: E402
from tslab.strategy.backtest import run_backtest                  # noqa: E402
from tslab.strategy.evaluate import max_drawdown, simulate_account  # noqa: E402
from tslab.strategy.full_search import (audit_signal, select_winner,  # noqa: E402
                                        tail_split, train_tstat)
from tslab.strategy.rules import pm_candidate_grid                # noqa: E402

TAIL_START = date(2026, 4, 10)
CANDIDATES = ["pmT2_L_r2.0", "orPM_L_r2.0", "pmT2_L_r3.0", "orPM_L_r3.0",
              "pmT_L_w2_r1.5"]
PRIOR_EV_ORDER = ["orPM_L_r2.0", "orPM_L_r3.0", "pmT2_L_r2.0",
                  "pmT_L_w2_r1.5", "pmT2_L_r3.0"]
CRIT = {"max_dd_usd": 4000.0, "min_trades": 5, "min_avg_net": -150.0}
MIN_TRAIN_TRADES = 50


def main() -> int:
    t0 = _time.time()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    bars = pd.read_parquet(outdir / "nq_bars_full.parquet")
    feats = pd.read_parquet(outdir / "nq_features_full.parquet")
    grid = pm_candidate_grid()
    rules = {r.name: r for r in grid}
    for name in CANDIDATES:
        assert name in rules, f"pre-registered candidate {name} missing"

    all_sessions = sorted(set(bars["session"]))
    train_sess, tail_sess = tail_split(all_sessions, TAIL_START)
    assert not set(train_sess) & set(tail_sess)
    assert tail_sess[-1] == date(2026, 6, 16)
    tr_m = bars["session"].isin(train_sess).to_numpy()
    ta_m = bars["session"].isin(tail_sess).to_numpy()
    btr, ftr = bars[tr_m], feats[tr_m]
    bta, fta = bars[ta_m], feats[ta_m]
    cm = CostModel(cfg)
    print(f"tail: {len(tail_sess)} sessions "
          f"({tail_sess[0]}..{tail_sess[-1]}), train: {len(train_sess)} "
          f"(ends {train_sess[-1]}, embargo 1 session)", flush=True)

    results = {}
    audits = []
    for name in CANDIDATES:
        rule = rules[name]
        bt = run_backtest(rule, bta, fta, cfg, cm)
        daily = bt.daily_net().reindex(tail_sess).fillna(0.0)
        dd = max_drawdown(daily)
        n = len(bt.trades)
        gross = sum(t.gross_usd for t in bt.trades)
        costs = sum(t.costs_usd for t in bt.trades)
        wins = sum(1 for t in bt.trades if t.net_usd > 0)
        acct = simulate_account(bt.by_session(), cfg, size="50k",
                                personal_dll=1000.0) if n else None
        a = audit_signal(rule, bars, cfg)
        audits.append(a)
        crit = {
            "C1_net_positive": bt.total_net > 0,
            "C2_zero_violations": bt.violations == 0,
            "C3_audit_pass": a.status != Status.FAIL,
            "C4_dd_ok": abs(dd) <= CRIT["max_dd_usd"],
            "C5_enough_trades": n >= CRIT["min_trades"],
            "C6_no_collapse": (n == 0
                               or bt.total_net / n >= CRIT["min_avg_net"]),
        }
        others = all(v for k, v in crit.items() if k != "C5_enough_trades")
        verdict = ("CONFIRMED" if all(crit.values())
                   else "INSUFFICIENT" if others else "FAILED")
        results[name] = {
            "verdict": verdict, "criteria": crit, "trades": n,
            "net": round(bt.total_net), "gross": round(gross),
            "costs": round(costs),
            "win_rate": round(wins / n, 3) if n else None,
            "max_dd": round(dd),
            "avg_net_per_trade": round(bt.total_net / n) if n else None,
            "violations": bt.violations,
            "account": None if acct is None else {
                "phase": acct.phase, "dll_hits": acct.dll_hits,
                "events": [e for e in acct.events
                           if e[0] in ("DLL_HIT", "MLL_BREACH")],
            },
            "trades_detail": [
                {"session": str(t.session), "entry": str(t.entry_time),
                 "dir": t.direction, "reason": t.exit_reason,
                 "net": round(t.net_usd)} for t in bt.trades],
        }
        print(f"{name:16} {verdict:12} net={bt.total_net:>7.0f} n={n:>3} "
              f"dd={dd:>6.0f} win={results[name]['win_rate']}", flush=True)

    # ---- walk-forward selection-process arm ----
    tr_sessions_n = len(train_sess)
    proc_rows = {}
    for rule in grid:
        r_tr = run_backtest(rule, btr, ftr, cfg, cm)
        from tslab.strategy.full_search import FoldCandidate
        proc_rows[rule.name] = FoldCandidate(
            name=rule.name, train_net=r_tr.total_net,
            train_trades=len(r_tr.trades),
            train_tstat=train_tstat(r_tr.daily_net(), tr_sessions_n),
            test_net=0.0, test_trades=0)
    proc_winner = select_winner(proc_rows, min_trades=MIN_TRAIN_TRADES)
    proc = {"winner": proc_winner}
    if proc_winner:
        bt = run_backtest(rules[proc_winner], bta, fta, cfg, cm)
        daily = bt.daily_net().reindex(tail_sess).fillna(0.0)
        proc.update({"tail_net": round(bt.total_net),
                     "tail_trades": len(bt.trades),
                     "tail_dd": round(max_drawdown(daily)),
                     "train_tstat": round(
                         proc_rows[proc_winner].train_tstat, 2),
                     "net_positive": bt.total_net > 0})
    print(f"WF-process arm: winner={proc_winner} -> {proc}", flush=True)

    confirmed = [n for n in CANDIDATES if results[n]["verdict"] == "CONFIRMED"]
    selected = next((n for n in PRIOR_EV_ORDER if n in confirmed), None)

    report = {
        "phase": "4B-2 one-shot tail confirmation",
        "tail": {"start": str(tail_sess[0]), "end": str(tail_sess[-1]),
                 "sessions": len(tail_sess)},
        "preregistration": {
            "criteria": CRIT, "prior_ev_order": PRIOR_EV_ORDER,
            "selection_rule": "highest PRIOR 4B-1 MC EV among CONFIRMED",
            "rules": {n: {**{k: str(v) for k, v in asdict(rules[n]).items()
                             if k != "filters"},
                          "filters": [asdict(f) for f in rules[n].filters]}
                      for n in CANDIDATES}},
        "results": results,
        "process_arm": proc,
        "confirmed": confirmed, "selected": selected,
        "audits": [{"name": a.name, "status": a.status.value,
                    "message": a.message} for a in audits],
        "runtime_min": round((_time.time() - t0) / 60, 1),
    }
    (outdir / "tail_confirmation_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    _write_md(report, outdir / "tail_confirmation_report.md")
    print(f"\nCONFIRMED={confirmed} SELECTED={selected}", flush=True)
    print(f"wrote {outdir}/tail_confirmation_report.md|.json", flush=True)

    bad = (any(r["violations"] for r in results.values())
           or any(a.status == Status.FAIL for a in audits))
    return 1 if bad else 0


def _write_md(report, path: Path) -> None:
    L = ["# Reserved-tail confirmation (Phase 4B-2, one shot)", "",
         f"- tail: {report['tail']['start']} .. {report['tail']['end']} "
         f"({report['tail']['sessions']} sessions)",
         f"- selection rule (pre-registered): "
         f"{report['preregistration']['selection_rule']}", "",
         "| candidate | verdict | net $ | trades | win | maxDD $ | "
         "$/trade | DLL | phase |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for n, r in report["results"].items():
        a = r["account"] or {}
        L.append(f"| {n} | **{r['verdict']}** | {r['net']} | {r['trades']} | "
                 f"{r['win_rate']} | {r['max_dd']} | {r['avg_net_per_trade']}"
                 f" | {a.get('dll_hits', '-')} | {a.get('phase', '-')} |")
    L += ["", f"**Confirmed:** {report['confirmed'] or 'none'}",
          f"**Selected:** {report['selected'] or 'none'}", "",
          "## Walk-forward process arm", "",
          f"```{json.dumps(report['process_arm'], indent=2)}```", "",
          "## Per-candidate criteria", ""]
    for n, r in report["results"].items():
        L.append(f"- {n}: " + ", ".join(
            f"{k}={'PASS' if v else 'FAIL'}" for k, v in r["criteria"].items()))
    L += ["", "## Audits", ""]
    L += [f"- [{a['status']}] {a['name']}" for a in report["audits"]]
    L += ["", "## Pre-registered rules (verbatim)", "",
          "```json", json.dumps(report["preregistration"]["rules"], indent=2),
          "```"]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
