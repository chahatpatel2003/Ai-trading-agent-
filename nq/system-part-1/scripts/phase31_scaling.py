"""Phase 31 addendum — growth-gated (profit-buffer) scaling on the FROZEN finalists.

Start at 1 MNQ, build cushion, scale only with buffer, de-scale when buffer shrinks. Tests Plans A-E
under BOTH a pure-trailing and a locked trailing $3K drawdown, with Monte-Carlo sequence-risk
(reshuffle, block bootstrap, bad-start, slippage). No rule changes; trades are the frozen finalists.

KEY MECHANIC: with a PURE trailing floor (peak-3000), buffer = 3000 - drawdown_from_peak <= 3000
ALWAYS — building profit cushion does NOT add room (the floor trails up with you), so scaling up
always risks the same $3K from the latest peak. Durable cushion only exists under a LOCKED model
(floor stops rising at the starting balance once the $3K buffer is banked).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

OUT = Path("docs/phase31")
SRC = "docs/phase30_finalist/phase30_finalist_trades.csv"
START, DDL = 100_000.0, 3_000.0
MNQ_PV = 2.0
RNG = np.random.default_rng(311)
N_MC = 800


def _load() -> dict[str, list[dict[str, Any]]]:
    df = pl.read_csv(SRC)
    out: dict[str, list[dict[str, Any]]] = {"idb": [], "swd": []}
    for r in df.iter_rows(named=True):
        out[r["strategy"]].append({
            "entry_ts": datetime.fromisoformat(r["entry_ts"]),
            "exit_ts": r["exit_ts"], "day": r["date"], "month": r["month"], "year": r["year"],
            "net1": float(r["net_pnl"]), "mae1": float(r["mae_pts"]) * MNQ_PV,
            "stop1": float(r["stop_dist_pts"]) * MNQ_PV,  # 1-contract worst-case stop $
        })
    return out


def _units(strategy: str, base: dict[str, list]) -> list[dict[str, Any]]:
    """Build the per-trade stream and each trade's BASE contracts for the strategy."""
    if strategy in ("idb", "swd"):
        rows = [dict(t, base=1) for t in base[strategy]]
    elif strategy == "portfolio":
        rows = [dict(t, base=1) for t in base["idb"] + base["swd"]]
    else:  # hybrid: idb leg sized to $150 fixed risk, swd leg 1 MNQ
        rows = []
        for t in base["idb"]:
            c = max(int(150.0 // t["stop1"]), 1)
            rows.append(dict(t, base=c))
        for t in base["swd"]:
            rows.append(dict(t, base=1))
    return sorted(rows, key=lambda x: x["entry_ts"])


def _floor(peak: float, model: str) -> float:
    if model == "trailing":
        return peak - DDL
    if model == "locked":  # floor trails until it reaches the starting balance, then locks
        return min(peak - DDL, START)
    return START - DDL  # static


# ---- plan sizing: returns contracts for this trade given state ----------------------------------
def _plan_contracts(plan: str, frac: float, t: dict, st: dict) -> int:
    base, stop1, buffer = t["base"], t["stop1"], st["buffer"]
    cushion, equity = st["cushion"], st["equity"]
    if plan == "B":
        c = int((buffer * frac) // stop1)
        return c if c >= 1 else 0
    if plan == "E":
        m = st["month_cap"]
    elif plan in ("C", "D"):
        m = 5 if equity >= 106_000 else 3 if equity >= 104_000 else 2 if equity >= 102_000 else 1
    else:  # A — cushion tiers
        m = 5 if cushion >= 6000 else 3 if cushion >= 4000 else 2 if cushion >= 2000 else 1
    c = m * base
    # universal guard: never let the worst-case stop on this trade exceed the buffer
    while c > base and c * stop1 > buffer:
        c -= base
    if c * stop1 > buffer:  # even base size would risk a breach -> stand down
        return 0
    return c


def simulate(trades: list[dict], plan: str, model: str, frac: float = 0.10,
             lockbox: bool = False) -> dict[str, Any]:
    equity = peak = START
    eq_curve = [equity]
    sizes: list[int] = []
    skipped = ups = downs = 0
    prev_c = 0
    breach = False
    breach_date = None
    min_buffer = DDL
    worst_intraday = 0.0
    closed_peak = closed_dd = 0.0
    first_scale = None
    nets: list[float] = []
    months: dict[str, float] = {}
    # month state for plan E
    cur_month = None
    month_start = equity
    month_peak = equity
    month_cap = 1
    month_stopped = False
    lock_floor = START - DDL  # plan D lockbox ratchet
    for t in trades:
        if plan == "E":
            if t["month"] != cur_month:
                cur_month, month_start, month_peak, month_cap, month_stopped = (
                    t["month"], equity, equity, 1, False)
            mprofit = equity - month_start
            mdd = month_peak - equity
            if mdd >= 1250:
                month_stopped = True
            month_cap = 1 if (mdd >= 750 or month_stopped) else (3 if mprofit >= 2000 else 2 if mprofit >= 1000 else 1)
            if month_stopped:
                skipped += 1
                sizes.append(0)
                eq_curve.append(equity)
                continue
        floor = _floor(peak, model)
        if plan == "D" or lockbox:
            cush = equity - START
            if cush >= 6000:
                lock_floor = max(lock_floor, START + 2000)
            elif cush >= 4000:
                lock_floor = max(lock_floor, START + 1000)
            elif cush >= 2000:
                lock_floor = max(lock_floor, START + 500)
            floor = max(floor, lock_floor)
        st = {"buffer": equity - floor, "cushion": equity - START, "equity": equity,
              "month_cap": month_cap}
        min_buffer = min(min_buffer, st["buffer"])
        c = _plan_contracts(plan, frac, t, st)
        if c < 1:
            skipped += 1
            sizes.append(0)
            eq_curve.append(equity)
            continue
        low = equity - c * t["mae1"]
        worst_intraday = max(worst_intraday, peak - low)
        if low < floor and not breach:
            breach, breach_date = True, str(t["exit_ts"])[:10]
        if c > prev_c and prev_c > 0:
            ups += 1
            if first_scale is None:
                first_scale = str(t["exit_ts"])[:10]
        elif c < prev_c and prev_c > 0:
            downs += 1
        prev_c = c
        net = c * t["net1"]
        equity += net
        peak = max(peak, equity)
        month_peak = max(month_peak, equity)
        nets.append(net)
        sizes.append(c)
        months[t["month"]] = months.get(t["month"], 0.0) + net
        eq_curve.append(equity)
        closed_peak = max(closed_peak, equity)
        closed_dd = max(closed_dd, closed_peak - equity)
    arr = np.array(nets) if nets else np.array([0.0])
    active = [s for s in sizes if s > 0]
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    mvals = list(months.values())
    return {
        "plan": plan, "model": model, "frac": frac if plan == "B" else None,
        "start_size": active[0] if active else 0, "max_size": max(active) if active else 0,
        "avg_size": round(float(np.mean(active)), 2) if active else 0, "trades": len(active),
        "skipped": skipped, "ending_equity": round(equity, 2), "net": round(equity - START, 2),
        "pf": round(float(wins.sum() / -losses.sum()), 3) if losses.size else None,
        "win_rate": round(float((arr > 0).mean()), 4) if nets else 0,
        "closed_dd": round(closed_dd, 2), "intraday_dd": round(worst_intraday, 2),
        "min_buffer": round(min_buffer, 2), "breach": breach, "breach_date": breach_date,
        "ups": ups, "downs": downs, "first_scale": first_scale,
        "monthly_avg": round((equity - START) / 66, 2),
        "best_month": f"{max(months, key=months.get)}:{max(mvals):.0f}" if mvals else "-",
        "worst_month": f"{min(months, key=months.get)}:{min(mvals):.0f}" if mvals else "-",
        "profitable_months": sum(1 for v in mvals if v > 0), "losing_months": sum(1 for v in mvals if v <= 0),
        "max_loss_streak": _streak(nets),
    }


def _streak(nets: list[float]) -> int:
    ml = cur = 0
    for x in nets:
        cur = cur + 1 if x < 0 else 0
        ml = max(ml, cur)
    return ml


def _reorder(trades: list[dict], mode: str) -> list[dict]:
    n = len(trades)
    if mode == "reshuffle":
        return [trades[i] for i in RNG.permutation(n)]
    if mode == "badstart":
        nets = np.array([t["net1"] for t in trades])
        q = np.quantile(nets, 0.25)
        worst = [t for t in trades if t["net1"] <= q]
        rest = [t for t in trades if t["net1"] > q]
        first20 = [worst[i] for i in RNG.integers(len(worst), size=20)]
        RNG.shuffle(rest)
        return first20 + rest
    if mode in ("week", "month"):
        key = (lambda t: t["day"][:7]) if mode == "month" else (lambda t: t["day"])
        blocks: dict[str, list] = {}
        for t in trades:
            blocks.setdefault(key(t), []).append(t)
        bl = list(blocks.values())
        seq: list[dict] = []
        while len(seq) < n:
            seq.extend(bl[RNG.integers(len(bl))])
        return seq[:n]
    return trades


def montecarlo(trades: list[dict], plan: str, model: str, frac: float = 0.10) -> dict[str, Any]:
    res = {"reshuffle": 0, "week": 0, "month": 0, "badstart": 0}
    dds = []
    for mode in res:
        breaches = 0
        for _ in range(N_MC):
            seq = _reorder(trades, mode)
            r = simulate(seq, plan, model, frac)
            if r["breach"]:
                breaches += 1
            if mode == "reshuffle":
                dds.append(r["intraday_dd"])
        res[mode] = round(breaches / N_MC, 4)
    dds_a = np.array(dds)
    return {"breach_reshuffle": res["reshuffle"], "breach_week": res["week"],
            "breach_month": res["month"], "breach_badstart": res["badstart"],
            "p95_dd": round(float(np.percentile(dds_a, 95)), 2),
            "p99_dd": round(float(np.percentile(dds_a, 99)), 2),
            "exp_dd": round(float(dds_a.mean()), 2)}


def classify(sim: dict, mc: dict) -> str:
    if sim["breach"]:
        return "REJECT_NOT_SCALABLE"
    if mc["breach_badstart"] > 0.10 or mc["breach_reshuffle"] > 0.10:
        return "REJECT_TOO_MUCH_SEQUENCE_RISK"
    meaningful = sim["monthly_avg"] >= 300
    safe = mc["breach_reshuffle"] < 0.05 and mc["breach_badstart"] < 0.05 and mc["p99_dd"] < 3200
    if safe and sim["max_size"] >= 2 and meaningful:
        return "SCALABLE_AFTER_BUFFER"
    if safe and sim["max_size"] >= 2:
        return "START_SMALL_FORWARD_PAPER"
    if safe:
        return "RESEARCH_ONLY"
    return "REJECT_TOO_MUCH_SEQUENCE_RISK"


def main() -> None:
    base = _load()
    strategies = {s: _units(s, base) for s in ("swd", "idb", "portfolio", "hybrid")}
    plans = [("A", 0.0), ("B", 0.10), ("B", 0.20), ("C", 0.0), ("D", 0.0), ("E", 0.0)]
    rows: list[dict[str, Any]] = []
    for sname, trades in strategies.items():
        for model in ("trailing", "locked"):
            for plan, frac in plans:
                sim = simulate(trades, plan, model, frac, lockbox=(plan == "D"))
                mc = montecarlo(trades, plan, model, frac)
                rows.append({"strategy": sname, **sim, **mc, "classification": classify(sim, mc)})
    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "phase31_scaling.csv")
    (OUT / "_scaling.json").write_text(json.dumps(rows, indent=2, default=str))
    # print headline: swd + portfolio under both models
    for r in rows:
        if r["strategy"] in ("swd", "portfolio") and r["plan"] in ("A", "C", "D"):
            tag = f"{r['plan']}{'/'+str(r['frac']) if r['frac'] else ''}"
            print(f"{r['strategy']:9s} {r['model']:8s} {tag:4s}: maxSz={r['max_size']} avgSz={r['avg_size']} "
                  f"net=${r['net']:.0f} $/mo=${r['monthly_avg']:.0f} DDintra=${r['intraday_dd']:.0f} "
                  f"breach(hist={r['breach']},resh={r['breach_reshuffle']},bad={r['breach_badstart']}) "
                  f"p99=${r['p99_dd']:.0f} -> {r['classification']}")
    print("\nrows:", len(rows))


if __name__ == "__main__":
    main()
