#!/usr/bin/env python
"""Phase 5D one-shot confirmation on the reserved 2025-Q1/Q2 tail.

Runs ONCE, after outputs/tail_prereg_micro.json is frozen by the search
script. Evaluates exactly the pre-registered MNQ candidates + the
selection-process arm on the tail; applies the frozen pass criteria; no
iteration afterward. Selection among tail passers uses SEARCH-ERA chain-MC
EV ordering (never tail performance).

Writes outputs/micro_tail_confirmation.{md,json}.
"""
from __future__ import annotations

import json
import sys
import time as _time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab.config import load_config                                 # noqa: E402
from tslab.features.core import FeatureParams                        # noqa: E402
from tslab.features.quantiles import compute_micro_q_features        # noqa: E402
from tslab.sim.costs import CostModel                                # noqa: E402
from tslab.strategy.backtest import run_backtest                     # noqa: E402
from tslab.strategy.evaluate import (lifecycle_mc_chain,             # noqa: E402
                                     max_drawdown, simulate_account)
from tslab.strategy.full_search import (FoldCandidate,               # noqa: E402
                                        select_winner, tail_split,
                                        train_tstat)
from tslab.strategy.rules import exec_micro_grids                    # noqa: E402


def main() -> int:
    t0 = _time.time()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    prereg = json.loads((outdir / "tail_prereg_micro.json").read_text())
    crit = prereg["tail_pass_criteria"]
    tail_start = date.fromisoformat(prereg["tail"]["start"])
    fp = FeatureParams()

    full = pd.read_parquet(outdir / "nq_micro_bars_full.parquet")
    feats = compute_micro_q_features(full, cfg, fp)   # causal; tail thresholds
    all_sessions = sorted(set(full["session"]))       # need pre-tail history
    train_sess, tail_sess = tail_split(all_sessions, tail_start,
                                       n_train=prereg["process_arm"]["n_train"],
                                       embargo=prereg["process_arm"]["embargo"])
    ta_m = full["session"].isin(tail_sess).to_numpy()
    tr_m = full["session"].isin(train_sess).to_numpy()
    bta, fta = full[ta_m], feats[ta_m]
    btr, ftr = full[tr_m], feats[tr_m]
    cm = CostModel(cfg)
    _, grid_mnq = exec_micro_grids()
    rules = {r.name: r for r in grid_mnq}
    print(f"tail: {len(tail_sess)} sessions ({tail_sess[0]}..{tail_sess[-1]})",
          flush=True)

    def evaluate(name):
        rule = rules[name]
        bt = run_backtest(rule, bta, fta, cfg, cm, instrument="MNQ")
        daily = bt.daily_net().reindex(tail_sess).fillna(0.0)
        n = len(bt.trades)
        wins = sum(1 for t in bt.trades if t.net_usd > 0)
        dd = max_drawdown(daily)
        acct = simulate_account(bt.by_session(), cfg, size="50k",
                                personal_dll=1000.0) if n else None
        res = {
            "trades": n, "tps": n / len(tail_sess),
            "net": round(bt.total_net), "win": round(wins / n, 3) if n else None,
            "gross": round(sum(t.gross_usd for t in bt.trades)),
            "costs": round(sum(t.costs_usd for t in bt.trades)),
            "max_dd": round(dd), "violations": bt.violations,
            "dll_hits": acct.dll_hits if acct else 0,
            "phase": acct.phase if acct else "-",
        }
        res["passes"] = bool(
            bt.total_net > crit["net_gt"]
            and n and wins / n >= crit["win_ge"]
            and res["tps"] >= crit["tps_ge"]
            and abs(dd) <= crit["max_dd_abs_le"]
            and bt.violations == crit["violations"])
        pool = {s: [] for s in tail_sess}
        for t in bt.trades:
            pool[t.session].append(t)
        res["tail_chain_mc_ev_month"] = round(lifecycle_mc_chain(
            [pool[s] for s in tail_sess], cfg, n_runs=200,
            horizon_sessions=252).ev_month, 1)
        return res

    results = {name: evaluate(name) for name in prereg["candidates"]}
    for name, r in results.items():
        print(f"{name:24} passes={r['passes']} net={r['net']:>6} n={r['trades']}"
              f" win={r['win']} tps={r['tps']:.2f} dd={r['max_dd']}", flush=True)

    # process arm: train-only selection on the last 250 pre-tail sessions
    proc_rows = {}
    for rule in grid_mnq:
        r_tr = run_backtest(rule, btr, ftr, cfg, cm, instrument="MNQ")
        proc_rows[rule.name] = FoldCandidate(
            name=rule.name, train_net=r_tr.total_net,
            train_trades=len(r_tr.trades),
            train_tstat=train_tstat(r_tr.daily_net(), len(train_sess)),
            test_net=0.0, test_trades=0)
    proc_winner = select_winner(
        proc_rows, min_trades=prereg["process_arm"]["min_train_trades"])
    proc = {"winner": proc_winner}
    if proc_winner:
        proc.update(evaluate(proc_winner))
    print(f"process arm: {proc}", flush=True)

    passers = [n for n, r in results.items() if r["passes"]]
    # selection by SEARCH-ERA EV ordering = prereg candidate order (already
    # ranked by search-era stability/tstat; never by tail results)
    selected = next((n for n in prereg["candidates"] if n in passers), None)
    out = {
        "prereg": prereg, "results": results, "process_arm": proc,
        "passers": passers, "selected": selected,
        "confirmed": bool(selected is not None),
        "runtime_min": round((_time.time() - t0) / 60, 1),
    }
    (outdir / "micro_tail_confirmation.json").write_text(
        json.dumps(out, indent=2, default=str))
    md = ["# Micro tail confirmation (one shot, 2025-Q1/Q2)", "",
          f"**confirmed: {out['confirmed']}**  selected: {selected}", "",
          "| candidate | passes | net $ | trades | tps | win | maxDD | "
          "DLL | phase | tailMC EV/mo |",
          "|---|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for n, r in results.items():
        md.append(f"| {n} | {r['passes']} | {r['net']} | {r['trades']} | "
                  f"{r['tps']:.2f} | {r['win']} | {r['max_dd']} | "
                  f"{r['dll_hits']} | {r['phase']} | "
                  f"{r['tail_chain_mc_ev_month']} |")
    md += ["", f"process arm: `{json.dumps(proc, default=str)}`"]
    (outdir / "micro_tail_confirmation.md").write_text("\n".join(md))
    print(f"\nconfirmed={out['confirmed']} selected={selected}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
