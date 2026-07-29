#!/usr/bin/env python
"""ICT v3 strict first backtest: v2 entries frozen, exits varied.
6 pre-registered exit models x {3m, 5m}, MNQ only, single pass each.
Baselines: v2 MNQ_3m / MNQ_5m. Writes outputs/ict_v3_report.{md,json}."""
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

MODELS = (("r1.5", "fixed", 1.5), ("r2", "fixed", 2.0),
          ("r2.5", "fixed", 2.5), ("r3", "fixed", 3.0),
          ("p1r2", "p1r2", None), ("be1", "be1", None))


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
    return {"trades": n, "tps": n / sessions_n, "win_rate": wins / n,
            "avg_win": float(np.mean([t.net_usd for t in trades
                                      if t.net_usd > 0])) if wins else 0.0,
            "avg_loss": float(np.mean([t.net_usd for t in trades
                                       if t.net_usd <= 0]))
            if n > wins else 0.0,
            "net": float(sum(t.net_usd for t in trades)),
            "gross": float(sum(t.gross_usd for t in trades)),
            "costs": float(sum(t.costs_usd for t in trades)),
            "max_dd": max_drawdown(ds),
            "exits": pd.Series([t.exit_reason for t in trades])
            .value_counts().to_dict(),
            "mc": {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in vars(mc).items()}}


def main() -> int:
    t0 = _time.time()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    bars = pd.read_parquet(outdir / "nq_bars_full.parquet")
    sessions_n = bars["session"].nunique()
    v2 = json.loads((outdir / "ict_poi_report.json").read_text())["variants"]
    base = {t: {k: v2[t]["summary"][k] for k in
                ("trades", "win_rate", "net")} for t in ("MNQ_3m", "MNQ_5m")}

    out = {}
    for ltf in (3, 5):
        for tag, model, k in MODELS:
            name = f"{tag}_{ltf}m"
            t = _time.time()
            trades, fails, _ = run_ict_poi_backtest(
                bars, cfg, "MNQ", ltf, exit_model=model,
                target_r=(k or 1.0))
            s = summarize(trades, sessions_n, cfg)
            out[name] = {"summary": s, "failures": fails.as_dict()}
            n = s.get("trades", 0)
            print(f"[{name}] n={n} "
                  + (f"tps={s['tps']:.2f} win={s['win_rate']:.3f} "
                     f"net=${s['net']:,.0f} EV/mo=${s['mc']['ev_month']:,.0f}"
                     f" blowout={s['mc']['blowout_rate']:.2f}" if n else "")
                  + f" ({_time.time() - t:.0f}s)", flush=True)

    report = {"hypothesis": "ICT v3: v2 entries + asymmetric/managed exits",
              "baselines_v2": base, "variants": out,
              "runtime_min": round((_time.time() - t0) / 60, 1)}
    (outdir / "ict_v3_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    md = ["# ICT v3 — exit models (strict first pass)", "",
          "| variant | n | tps | win | avg win | avg loss | net $ | "
          "EV/mo $ | blowout | P(pass) | P(pay1) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, blk in out.items():
        s = blk["summary"]
        if not s.get("trades"):
            md.append(f"| {name} | 0 | | | | | | | | | |")
            continue
        m = s["mc"]
        md.append(f"| {name} | {s['trades']} | {s['tps']:.2f} | "
                  f"{s['win_rate']:.3f} | {s['avg_win']:.0f} | "
                  f"{s['avg_loss']:.0f} | {s['net']:,.0f} | "
                  f"{m['ev_month']:.0f} | {m['blowout_rate']:.2f} | "
                  f"{m['p_pass']:.2f} | {m['p_payout1']:.2f} |")
    md += ["", f"v2 baselines: {json.dumps(base)}", "",
           "Exit mixes:", ""]
    md += [f"- {name}: {blk['summary'].get('exits')}"
           for name, blk in out.items() if blk["summary"].get("trades")]
    (outdir / "ict_v3_report.md").write_text("\n".join(md))
    print(f"wrote {outdir}/ict_v3_report.md|.json "
          f"({report['runtime_min']} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
