"""Phase 52 — High-win-rate Topstep PASS engine test.

Objective shift: maximize PASS probability (target 70-80%+), accept low RR / small targets / high WR.
Builds GENUINE high-WR low-RR systems (fresh engine runs: 5 fade/MR triggers x RR {0.5,0.75,1.0,2.0})
plus clean5+Phase36 as the real-edge benchmark, computes trade WR + after-cost expectancy, and runs the
Topstep pass sim on 50K/100K/150K (cushion sizing). Tests whether high TRADE win rate raises PASS
probability, or whether pass probability is driven by EDGE + the MLL buffer. Frozen-style costs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nqlab.backtest.engine import run_backtest
from nqlab.config.loader import load_config, load_topstep_rules
from nqlab.discovery import phase29 as P29
from nqlab.domain.strategy.base import ExecConfig, StrategyParams

import scripts.phase38_workhorse as P38
from scripts.phase44 import ACCOUNTS, build_days, cushion, merge, p36_days, sim_combine

OUT = Path("docs/phase52")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(52)
_C = pl.col
P38._RR.update({"rr0.5": 0.5, "rr0.75": 0.75})  # add low-RR (high-WR) exits


def scalp_strats():
    md = _C("minute_of_day")
    rth = (md >= 570) & (md < 870)
    after_or = (md >= 600) & (md < 840)
    close, high, low, atr = _C("close"), _C("high"), _C("low"), _C("atr")
    vwap, pdh, pdl, pdc = _C("vwap_session"), _C("prev_day_high"), _C("prev_day_low"), _C("prev_day_close")
    up = close > close.shift(1).over("trading_day")
    dn = close < close.shift(1).over("trading_day")
    triggers = {
        "vwap_rev": (rth & ((close - vwap) < -1.0 * atr) & up, rth & ((close - vwap) > 1.0 * atr) & dn),
        "asia_sweep": (rth & (low < _C("asia_low")) & (close > _C("asia_low")),
                       rth & (high > _C("asia_high")) & (close < _C("asia_high"))),
        "pdhl_reclaim": (rth & (low < pdl) & (close > pdl), rth & (high > pdh) & (close < pdh)),
        "or_fakeout": (after_or & (low < _C("or_low")) & (close > _C("or_low")),
                       after_or & (high > _C("or_high")) & (close < _C("or_high"))),
        "pdc_reclaim": (rth & (low <= pdc) & (close > pdc) & up, rth & (high >= pdc) & (close < pdc) & dn),
    }
    out = []
    for nm, (lo, sh) in triggers.items():
        for rr in ("rr0.5", "rr0.75", "rr1", "rr2"):
            out.append(P38.Strat(f"{nm}_{rr}", "scalp", lo, sh, 1.0, rr))
    return out


def run_one(fr, s, cfg, rules):
    res = run_backtest(P38.build(fr, s), symbol="NQ.c", timeframe="5m",
                       strategy=StrategyParams("p52", "1"), config=cfg, rules=rules,
                       primary_cutoff=False, entry_mode="next_open", gate_entries=False,
                       slippage_mult=1.0, commission_mult=1.0,
                       exec_config=ExecConfig(sizing_mode="mnq_fixed1", time_stop_bars=10000))
    return [{"day": str(t.trading_day), "ts": t.entry_ts.isoformat(), "net": float(t.net_pnl),
             "mae": float(t.mae) * 2, "side": int(t.side)}
            for t in res.trades if P38._exit_td(t.exit_ts) == t.trading_day]


def build_struct(trades, reg, all_days, cap=3):
    by_day = defaultdict(list)
    for t in trades:
        if reg.get(t["day"], {}).get("trend") == "neutral":
            continue
        by_day[t["day"]].append((0, t["net"], t["mae"], "A", "s"))
    days = []
    for d in all_days:
        days.append(sorted(by_day.get(d, []), key=lambda x: x[0])[:cap])
    return days, [reg.get(d, {}) for d in all_days]


def trade_stats(trades):
    nets = np.array([t["net"] for t in trades])
    if not nets.size:
        return {}
    w, ls = nets[nets > 0], nets[nets < 0]
    return {"n": len(nets), "wr": round(float((nets > 0).mean()), 3),
            "avg_w": round(float(w.mean()), 1) if w.size else 0,
            "avg_l": round(float(ls.mean()), 1) if ls.size else 0,
            "rr": round(float(w.mean() / -ls.mean()), 2) if w.size and ls.size else None,
            "exp": round(float(nets.mean()), 2)}


def pass_sim(days, regimes, acct, sims=6000, hor=90):
    cost, T, M, mc = ACCOUNTS[acct]
    n = len(days)
    starts = RNG.integers(0, n - hor, sims)
    res = [sim_combine(days, regimes, int(s), T, M, cushion(T, mc), -0.375 * M, hor) for s in starts]
    npass = sum(1 for r in res if r[0] == "pass")
    nblow = sum(1 for r in res if r[0] == "blow")
    pdays = np.array([r[1] for r in res if r[0] == "pass"])

    def pby(k):
        return round(sum(1 for r in res if r[0] == "pass" and r[1] <= k) / sims, 3)
    return {"pass": round(npass / sims, 3), "blow": round(nblow / sims, 3),
            "p_pass": round(npass / (npass + nblow), 3) if (npass + nblow) else 0.0,
            "ratio": round(npass / nblow, 2) if nblow else (99.9 if npass else 0.0),
            "p10": pby(10), "p20": pby(20), "p30": pby(30), "p45": pby(45),
            "med_days": round(float(np.median(pdays)), 0) if pdays.size else None,
            "maxdd": round(max(r[2] for r in res), 0), "worst_day": round(min(r[3] for r in res), 0),
            "consist": round(sum(1 for r in res if r[4]) / sims, 3)}


def main():
    cfg = load_config(profile="dev", overrides=P29.PHASE29_OVERRIDES)
    rules = load_topstep_rules("topstep_150k")
    fr = P38.add_cols(pl.read_parquet(cfg.data.artifacts_root / "phase30" / "frame_5m_full_vp.parquet"))
    reg = json.loads((Path("docs/phase39") / "day_regime.json").read_text())
    all_days = sorted(reg.keys())

    entries: dict[str, Any] = {}
    for s in scalp_strats():
        tr = run_one(fr, s, cfg, rules)
        entries[s.name] = tr
    # benchmark
    c5 = build_days(["trend_adx_cont", "ir90_break", "asia_sweep_fade", "ir60_fade", "disp_pb_fvg"], "phase38")
    p36 = p36_days()
    bench_days = merge(c5[1], p36[1])
    bench_trades = []  # for WR stats
    st_b = json.loads((Path("docs/phase38") / "_trades.json").read_text())
    for nm in ["trend_adx_cont", "ir90_break", "asia_sweep_fade", "ir60_fade", "disp_pb_fvg"]:
        bench_trades += st_b[nm]

    rows = {}
    # scalps
    for name, tr in entries.items():
        ts = trade_stats(tr)
        days, regs = build_struct(tr, reg, all_days)
        r150 = pass_sim(days, regs, "150K")
        r50 = pass_sim(days, regs, "50K")
        rows[name] = {**ts, "pass150": r150, "pass50": r50}
    # benchmark
    tsb = trade_stats(bench_trades)
    rows["BENCH_clean5+P36"] = {**tsb,
                                "pass150": pass_sim(bench_days, c5[2], "150K"),
                                "pass50": pass_sim(bench_days, c5[2], "50K")}
    (OUT / "_phase52.json").write_text(json.dumps(rows, default=str, indent=2))

    def line(name, r, acc):
        p = r[acc]
        return (f"{name:18s} WR={r.get('wr')} RR={r.get('rr')} exp=${r.get('exp')} n={r.get('n')} | "
                f"PASS={p['pass']} blow={p['blow']} p_pass={p['p_pass']} ratio={p['ratio']} "
                f"p20={p['p20']} p45={p['p45']} med={p['med_days']} dd={p['maxdd']:.0f} "
                f"wd={p['worst_day']:.0f} cons={p['consist']}")

    print("===== 150K (target $9k, MLL $4.5k) — sorted by pass probability =====")
    for name in sorted(rows, key=lambda k: rows[k]["pass150"]["p_pass"], reverse=True):
        tag = " <BENCH" if "BENCH" in name else ""
        print(line(name, rows[name], "pass150") + tag)
    print("\n===== 50K (target $3k, MLL $2k) — sorted by pass probability =====")
    for name in sorted(rows, key=lambda k: rows[k]["pass50"]["p_pass"], reverse=True):
        tag = " <BENCH" if "BENCH" in name else ""
        print(line(name, rows[name], "pass50") + tag)


if __name__ == "__main__":
    main()
