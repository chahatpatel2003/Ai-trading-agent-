#!/usr/bin/env python
"""Strict first backtest of the manual-discretionary ICT displacement
hypothesis, exactly as specified (no optimization, no selection, single
pass over 2021-01-05..2026-06-16). Variants: NQ strict, MNQ strict
(primary), MNQ + HTF-POI filter (separately reported).

Writes outputs/ict_report.{md,json}."""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab.config import load_config                              # noqa: E402
from tslab.strategy.evaluate import (lifecycle_mc_chain,          # noqa: E402
                                     max_drawdown, simulate_account)
from tslab.strategy.ict_displacement import run_ict_backtest      # noqa: E402


def summarize(trades, sessions_n, cfg):
    n = len(trades)
    if not n:
        return {"trades": 0}
    wins = sum(1 for t in trades if t.net_usd > 0)
    daily = {}
    for t in trades:
        daily[t.session] = daily.get(t.session, 0.0) + t.net_usd
    ds = pd.Series(daily).sort_index()
    per_year = {}
    for t in trades:
        y = t.session.year
        per_year.setdefault(y, [0.0, 0])
        per_year[y][0] += t.net_usd
        per_year[y][1] += 1
    pool_sessions = sorted({t.session for t in trades})
    pool = {s: [] for s in pool_sessions}
    for t in trades:
        pool[t.session].append(t)
    mc = lifecycle_mc_chain([pool[s] for s in pool_sessions], cfg,
                            n_runs=300, horizon_sessions=252)
    return {
        "trades": n, "trades_per_session": n / sessions_n,
        "trading_days": len(pool_sessions),
        "win_rate": wins / n,
        "avg_win": float(np.mean([t.net_usd for t in trades if t.net_usd > 0])),
        "avg_loss": float(np.mean([t.net_usd for t in trades
                                   if t.net_usd <= 0])),
        "gross": float(sum(t.gross_usd for t in trades)),
        "costs": float(sum(t.costs_usd for t in trades)),
        "net": float(sum(t.net_usd for t in trades)),
        "avg_risk": float(np.mean([t.risk_usd for t in trades])),
        "avg_qty": float(np.mean([t.qty for t in trades])),
        "max_dd": max_drawdown(ds),
        "per_year": {str(y): {"net": round(v[0]), "trades": v[1]}
                     for y, v in sorted(per_year.items())},
        "exits": pd.Series([t.exit_reason for t in trades])
        .value_counts().to_dict(),
        "mc": {k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in vars(mc).items()},
    }


def main() -> int:
    t0 = _time.time()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    bars = pd.read_parquet(outdir / "nq_bars_full.parquet")
    sessions_n = bars["session"].nunique()
    print(f"1m bars={len(bars):,} sessions={sessions_n}", flush=True)

    out = {}
    for tag, inst, poi in (("NQ_strict", "NQ", False),
                           ("MNQ_strict", "MNQ", False),
                           ("MNQ_poi", "MNQ", True)):
        t = _time.time()
        trades, fails, meta = run_ict_backtest(bars, cfg, inst,
                                               poi_filter=poi)
        s = summarize(trades, sessions_n, cfg)
        out[tag] = {"summary": s, "failures": fails.as_dict(),
                    "examples": meta["examples"]}
        n = s.get("trades", 0)
        print(f"[{tag}] trades={n} "
              + (f"tps={s['trades_per_session']:.2f} win={s['win_rate']:.3f} "
                 f"net=${s['net']:,.0f} EV/mo=${s['mc']['ev_month']:,.0f} "
                 f"blowout={s['mc']['blowout_rate']:.2f}" if n else "")
              + f" ({_time.time() - t:.0f}s)", flush=True)

    report = {"hypothesis": "ICT 15m-bias displacement retracement "
                            "(strict, unoptimized first pass)",
              "span": "2021-01-05..2026-06-16",
              "variants": out,
              "disclosed_constants": {
                  "displacement": "body>=70% of range AND range>=1.5xATR20(3m)",
                  "max_trades_per_day": 3, "per_trade_cap_usd": 333.33,
                  "day_stop_usd": 666.67, "mae_model": "1R stop / 0.75R other",
                  "order_life": "cancel if stop or target trades first; "
                                "expire 14:30 CT; one position at a time"},
              "runtime_min": round((_time.time() - t0) / 60, 1)}
    (outdir / "ict_report.json").write_text(
        json.dumps(report, indent=2, default=str))

    md = ["# ICT displacement hypothesis — strict first backtest", ""]
    for tag, blk in out.items():
        s = blk["summary"]
        md += [f"## {tag}", ""]
        if not s.get("trades"):
            md += ["- no trades", ""]
            continue
        md += [f"- trades {s['trades']} ({s['trades_per_session']:.2f}"
               f"/session over all {sessions_n} sessions; active on "
               f"{s['trading_days']} days), win {s['win_rate']:.3f}",
               f"- net ${s['net']:,.0f} (gross ${s['gross']:,.0f}, costs "
               f"${s['costs']:,.0f}); avg win ${s['avg_win']:.0f} / loss "
               f"${s['avg_loss']:.0f}; avg risk ${s['avg_risk']:.0f} "
               f"(avg qty {s['avg_qty']:.1f}); maxDD ${s['max_dd']:,.0f}",
               f"- exits: {s['exits']}",
               f"- yearly: {s['per_year']}",
               f"- chain MC: EV/mo ${s['mc']['ev_month']:,.0f}, P(pass) "
               f"{s['mc']['p_pass']:.2f}, P(pay1/2/3) {s['mc']['p_payout1']:.2f}"
               f"/{s['mc']['p_payout2']:.2f}/{s['mc']['p_payout3']:.2f}, "
               f"blowout {s['mc']['blowout_rate']:.2f}, DLL rate "
               f"{s['mc']['dll_hit_rate']:.3f}",
               f"- rule failures: {blk['failures']}", ""]
        if blk["examples"]:
            md += ["### example trades (exact levels)", "```json",
                   json.dumps(blk["examples"], indent=2), "```", ""]
    md += ["## Disclosed a-priori constants", "```json",
           json.dumps(report["disclosed_constants"], indent=2), "```"]
    (outdir / "ict_report.md").write_text("\n".join(md))
    print(f"wrote {outdir}/ict_report.md|.json "
          f"({report['runtime_min']} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
