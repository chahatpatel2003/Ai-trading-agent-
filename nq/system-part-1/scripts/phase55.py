"""Phase 55 — Aggressive cashflow-optimized Topstep cycle (accept high fail; optimize EV/month).

Mandate: stop rejecting on fail rate. Pass within 20-22 trading days, optimize EV/month after monthly
subscription + reset + activation. Fail 40-60% OK; ratio <1.25 OK; large sizing OK; blowing combines OK;
funded stays 1 MNQ safe. Each capped attempt ~= 1 subscription month (cheap, bounded lottery). Simulates
3-month (63-day) cashflow timelines (combine -> pass -> 1-MNQ funded -> payouts -> blow -> restart).
Costs per account (monthly=reset=activation): 50K $49 / 100K $99 / 150K $199. clean5+Phase36 signals.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.phase44 import ACCOUNTS, build_days, merge, p36_days, sim_funded
from scripts.phase54 import attempt

OUT = Path("docs/phase55")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(55)
FEE = {"50K": 49.0, "100K": 99.0, "150K": 199.0}  # monthly = reset = activation (per account)
DPM = 21


def sizer(name, mc):
    base = {"fixed3": 3, "fixed4": 4, "fixed5": 5, "1NQ": 10,
            "front234": None}[name]
    if name == "front234":
        return lambda b: (2 if b < 1500 else (3 if b < 4000 else 4))
    return lambda b: min(base, mc)


def per_attempt(days, T, M, sized, lstop, cap, sims=8000):
    n = len(days)
    outs = [attempt(days, int(RNG.integers(0, n - cap - 1)), sized, lstop, cap, []) for _ in range(sims)]
    npass = sum(1 for o in outs if o[0] == "pass")
    nblow = sum(1 for o in outs if o[0] == "blow")
    pdays = [o[1] for o in outs if o[0] == "pass"]
    return {"pass": round(npass / sims, 3), "blow": round(nblow / sims, 3),
            "reset": round(sum(1 for o in outs if o[0] == "reset") / sims, 3),
            "ratio": round(npass / nblow, 2) if nblow else (99.9 if npass else 0.0),
            "med_pass": round(float(np.median(pdays)), 0) if pdays else None}


def timeline(days, regimes, acct, sized, lstop, cap, horizon=63):
    cost, T, M, mc = ACCOUNTS[acct]
    fee = FEE[acct]
    n = len(days)
    cash = 0.0
    elapsed = 0
    passes = 0
    while elapsed < horizon:
        # combine attempt (<= cap days)
        start = int(RNG.integers(0, n - cap - 1))
        outcome, day = attempt(days, start, sized, lstop, cap, [])
        elapsed += day
        cash -= int(np.ceil(day / DPM)) * fee
        if outcome != "pass":
            cash -= fee  # reset
            continue
        cash -= fee  # activation
        passes += 1
        # funded segment (1 MNQ) for remaining window
        rem = max(horizon - elapsed, 5)
        fstart = int(RNG.integers(0, n - rem - 1))
        fr = sim_funded(days, regimes, fstart, M, mc, -0.375 * M, M, rem)
        cash += fr["wd"]
        elapsed += (fr["day"] if fr["blew"] else rem)
        # if funded blew, loop continues -> new combine
    return cash, passes


def evaluate(days, regimes, acct, sized, lstop, cap, sims=6000):
    pa = per_attempt(days, ACCOUNTS[acct][1], ACCOUNTS[acct][2], sized, lstop, cap)
    nets = np.array([timeline(days, regimes, acct, sized, lstop, cap)[0] for _ in range(sims)])
    # cost per pass (expected): fee * E[months+resets+activation per pass]
    p = pa["pass"]
    cost_per_pass = (FEE[acct] / p * (1 + 1)) + FEE[acct] if p > 0 else None  # ~month+reset per attempt /p + activ
    return {**pa,
            "EV_3mo": round(float(nets.mean()), 0), "EV_month": round(float(nets.mean()) / 3, 0),
            "P_profit_3mo": round(float((nets > 0).mean()), 3),
            "P_loss_3mo": round(float((nets < 0).mean()), 3),
            "worst_3mo": round(float(nets.min()), 0),
            "p5_3mo": round(float(np.quantile(nets, 0.05)), 0),
            "cost_per_pass": round(cost_per_pass, 0) if cost_per_pass else None}


def main():
    c5 = build_days(["trend_adx_cont", "ir90_break", "asia_sweep_fade", "ir60_fade", "disp_pb_fvg"], "phase38")
    p36 = p36_days()
    days = merge(c5[1], p36[1])
    regs = c5[2]

    cols = ["pass", "blow", "reset", "ratio", "med_pass", "cost_per_pass", "EV_month", "EV_3mo",
            "P_profit_3mo", "P_loss_3mo", "worst_3mo", "p5_3mo"]
    results = {}
    for acct in ("50K", "100K", "150K"):
        cost, T, M, mc = ACCOUNTS[acct]
        print(f"\n######### {acct} (target ${T:.0f}, MLL ${M:.0f}, max {mc}, fee ${FEE[acct]:.0f}) #########")
        print(f"{'sizing|cap|ls':24s} " + " ".join(f"{c[:9]:>9s}" for c in cols))
        sizings = ["fixed3", "fixed4", "fixed5", "front234"]
        if mc >= 10:
            sizings.append("1NQ")
        for szn in sizings:
            sf = sizer(szn, mc)
            for cap in (20, 22):
                for ls in (-1000.0,):
                    r = evaluate(days, regs, acct, sf, ls, cap)
                    key = f"{acct}|{szn}|cap{cap}|ls{int(-ls)}"
                    results[key] = r
                    print(f"{szn + '|c' + str(cap):24s} " + " ".join(f"{str(r[c]):>9s}" for c in cols))
    (OUT / "_phase55.json").write_text(json.dumps(results, indent=2))
    best = max(results, key=lambda k: results[k]["EV_month"] or -9)
    print(f"\nBEST EV/month: {best} -> EV/mo ${results[best]['EV_month']}, "
          f"pass {results[best]['pass']}, fail {results[best]['blow']}, "
          f"P(profit 3mo) {results[best]['P_profit_3mo']}, P(loss 3mo) {results[best]['P_loss_3mo']}, "
          f"worst 3mo ${results[best]['worst_3mo']}")


if __name__ == "__main__":
    main()
