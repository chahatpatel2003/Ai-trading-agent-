#!/usr/bin/env python
"""Strict first backtest of the HTF-POI-tap -> LTF-displacement hypothesis
(pre-registered v2). 4 LTFs x 2 instruments, one pass each, no optimization.
Baseline comparison: v1 MNQ_strict from outputs/ict_report.json.

Writes outputs/ict_poi_report.{md,json}."""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab.config import load_config                              # noqa: E402
from tslab.strategy.evaluate import lifecycle_mc_chain, max_drawdown  # noqa: E402
from tslab.strategy.ict_displacement import run_ict_poi_backtest  # noqa: E402


def summarize(trades, sessions_n, cfg):
    n = len(trades)
    if not n:
        return {"trades": 0}
    wins = sum(1 for t in trades if t.net_usd > 0)
    daily = {}
    for t in trades:
        daily[t.session] = daily.get(t.session, 0.0) + t.net_usd
    ds = pd.Series(daily).sort_index()
    pool_sessions = sorted(daily)
    pool = {s: [] for s in pool_sessions}
    for t in trades:
        pool[t.session].append(t)
    mc = lifecycle_mc_chain([pool[s] for s in pool_sessions], cfg,
                            n_runs=300, horizon_sessions=252)
    per_year = {}
    for t in trades:
        y = str(t.session.year)
        per_year.setdefault(y, [0, 0])
        per_year[y][0] += t.net_usd
        per_year[y][1] += 1
    return {"trades": n, "trades_per_session": n / sessions_n,
            "win_rate": wins / n,
            "net": float(sum(t.net_usd for t in trades)),
            "gross": float(sum(t.gross_usd for t in trades)),
            "costs": float(sum(t.costs_usd for t in trades)),
            "max_dd": max_drawdown(ds),
            "per_year": {y: [round(v[0]), v[1]] for y, v in
                         sorted(per_year.items())},
            "mc": {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in vars(mc).items()}}


def main() -> int:
    t0 = _time.time()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    bars = pd.read_parquet(outdir / "nq_bars_full.parquet")
    sessions_n = bars["session"].nunique()
    baseline = json.loads((outdir / "ict_report.json").read_text())[
        "variants"]["MNQ_strict"]["summary"]

    out = {}
    for ltf in (1, 2, 3, 5):
        for inst in ("MNQ", "NQ"):
            tag = f"{inst}_{ltf}m"
            t = _time.time()
            trades, fails, meta = run_ict_poi_backtest(bars, cfg, inst, ltf)
            s = summarize(trades, sessions_n, cfg)
            out[tag] = {"summary": s, "failures": fails.as_dict(),
                        "poi_counts": meta["poi_counts"]}
            n = s.get("trades", 0)
            print(f"[{tag}] trades={n} "
                  + (f"tps={s['trades_per_session']:.2f} "
                     f"win={s['win_rate']:.3f} net=${s['net']:,.0f} "
                     f"EV/mo=${s['mc']['ev_month']:,.0f} "
                     f"blowout={s['mc']['blowout_rate']:.2f}" if n else "")
                  + f" ({_time.time() - t:.0f}s)", flush=True)

    report = {
        "hypothesis": "HTF POI tap -> LTF displacement (pre-registered v2; "
                      "strict fractal swings; tap window 30min; POI "
                      "freshness 6 sess (15m) / 15 sess (1h))",
        "span": "2021-01-05..2026-06-16",
        "baseline_v1_MNQ_strict": {k: baseline[k] for k in
                                   ("trades", "win_rate", "net")},
        "variants": out,
        "runtime_min": round((_time.time() - t0) / 60, 1)}
    (outdir / "ict_poi_report.json").write_text(
        json.dumps(report, indent=2, default=str))

    md = ["# HTF-POI-tap -> LTF displacement (v2, strict first pass)", "",
          "| variant | trades | tps | win | net $ | EV/mo $ | blowout | "
          "P(pass) | P(pay1) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, blk in out.items():
        s = blk["summary"]
        if not s.get("trades"):
            md.append(f"| {tag} | 0 | | | | | | | |")
            continue
        m = s["mc"]
        md.append(f"| {tag} | {s['trades']} | {s['trades_per_session']:.2f} "
                  f"| {s['win_rate']:.3f} | {s['net']:,.0f} | "
                  f"{m['ev_month']:.0f} | {m['blowout_rate']:.2f} | "
                  f"{m['p_pass']:.2f} | {m['p_payout1']:.2f} |")
    md += ["", f"Baseline v1 MNQ_strict: {report['baseline_v1_MNQ_strict']}",
           "", "## POI-type breakdown / failures", ""]
    for tag, blk in out.items():
        md += [f"- {tag}: poi={blk['poi_counts']}",
               f"  failures={blk['failures']}"]
    (outdir / "ict_poi_report.md").write_text("\n".join(md))
    print(f"wrote {outdir}/ict_poi_report.md|.json "
          f"({report['runtime_min']} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
