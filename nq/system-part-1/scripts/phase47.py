"""Phase 47 — Fast Topstep payout extraction test (frozen signals; wrapper only).

Question: can we pass in 1-2 weeks (5-15 trading days), grab 1-3 payouts fast, and accept blow-after-
payout? Optimize CASH EXTRACTED BEFORE BLOW, not longevity. Flat per-attempt cost (50K $49/100K $99/
150K $199) since a fast cycle accrues no months. Frozen signals only. Tests sizing (1-5 MNQ, 1 NQ,
cushion), 7 strategies, 3 accounts. Kill criteria: fail>pass, ratio<1.5, low first-payout, EV<0,
oversized-risk, consistency.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.phase42 import A_SLEEVES, CLEAN5, build_days
from scripts.phase44 import ACCOUNTS, cushion, flat, merge, p36_days

OUT = Path("docs/phase47")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(47)
WIN_THR, MIN_WIN = 150.0, 5


def sim_combine(days, start, T, M, sized, lstop, horizon=60):
    eq = peak = best = worst = mdd = 0.0
    active = 0
    consist = False
    for di in range(horizon):
        if start + di >= len(days):
            break
        d = 0.0
        traded = False
        for (_et, net1, mae1, _q, _sl) in days[start + di]:
            if lstop is not None and d <= lstop:
                break
            bal = eq + d
            s = sized(bal)
            if bal - mae1 * s <= min(max(peak, bal) - M, 0.0):
                return ("blow", di + 1, mdd, min(worst, d), consist)
            d += net1 * s
            traded = True
            bal = eq + d
            peak = max(peak, bal)
            if bal <= min(peak - M, 0.0):
                return ("blow", di + 1, max(mdd, peak - bal), min(worst, d), consist)
        eq += d
        if traded:
            active += 1
        best = max(best, d)
        worst = min(worst, d)
        mdd = max(mdd, peak - eq)
        if eq >= T and active >= 2:
            if best <= 0.5 * eq:
                return ("pass", di + 1, mdd, worst, consist)
            consist = True
    return ("timeout", horizon, mdd, worst, consist)


def sim_funded(days, start, M, mc, cap, fmode, horizon=180):
    """fmode: 'flat1' or 'scaleup' (1 MNQ until first payout, then 2)."""
    eq = peak = best = worst = mdd = 0.0
    win = pots = 0
    wd = 0.0
    first = None
    locked = False
    for di in range(horizon):
        if start + di >= len(days):
            break
        size = 1 if (fmode == "flat1" or pots == 0) else min(2, mc)
        d = 0.0
        for (_et, net1, mae1, _q, _sl) in days[start + di]:
            bal = eq + d
            floor = 0.0 if locked else min(peak - M, 0.0)
            if bal - mae1 * size <= floor:
                return {"wd": wd, "pots": pots, "first": first, "days1": first,
                        "blew": True, "mdd": max(mdd, peak - (bal - mae1 * size))}
            d += net1 * size
            bal = eq + d
            peak = max(peak, bal)
            if peak >= M:
                locked = True
            if bal <= (0.0 if locked else min(peak - M, 0.0)):
                return {"wd": wd, "pots": pots, "first": first, "days1": first, "blew": True,
                        "mdd": max(mdd, peak - bal)}
        eq += d
        best = max(best, d)
        if d >= WIN_THR:
            win += 1
        if win >= MIN_WIN and eq > 0 and best <= 0.5 * eq:
            amt = min(eq, cap)
            wd += amt
            eq -= amt
            pots += 1
            first = first or (di + 1)
            win = 0
            best = 0.0
    return {"wd": wd, "pots": pots, "first": first, "days1": first, "blew": False, "mdd": mdd}


def evaluate(days, acct, sized_c, lstop_c, fmode="flat1", sims=12000):
    cost, T, M, mc = ACCOUNTS[acct]
    n = len(days)
    cs = RNG.integers(0, n - 60, sims)
    cres = [sim_combine(days, int(s), T, M, sized_c, lstop_c) for s in cs]
    npass = sum(1 for r in cres if r[0] == "pass")
    nblow = sum(1 for r in cres if r[0] == "blow")
    pdays = np.array([r[1] for r in cres if r[0] == "pass"])

    def pass_by(k):
        return round(sum(1 for r in cres if r[0] == "pass" and r[1] <= k) / sims, 3)

    p_pass = npass / (npass + nblow) if (npass + nblow) else 0.0
    fs = RNG.integers(0, n - 180, sims)
    fres = [sim_funded(days, int(s), M, mc, M, fmode) for s in fs]
    p1 = np.mean([1 if r["pots"] >= 1 else 0 for r in fres])
    p2 = np.mean([1 if r["pots"] >= 2 else 0 for r in fres])
    p3 = np.mean([1 if r["pots"] >= 3 else 0 for r in fres])
    e_total = float(np.mean([r["wd"] for r in fres]))
    d1 = [r["first"] for r in fres if r["first"]]
    e_attempts = 1.0 / p_pass if p_pass > 0 else 999.0
    ev_cycle = e_total - cost * e_attempts
    # cycle months: combine days-to-resolution * attempts + funded days-to-blow
    resd = np.array([r[1] for r in cres if r[0] in ("pass", "blow")])
    fblow = [r for r in fres if r["blew"]]
    fdur = float(np.mean([r["first"] or 180 for r in fres]))  # approx active funded duration
    cyc_days = e_attempts * (float(resd.mean()) if resd.size else 60) + fdur
    return {
        "pass5": pass_by(5), "pass10": pass_by(10), "pass15": pass_by(15),
        "fail": round(nblow / sims, 3), "ratio": round(npass / nblow, 2) if nblow else None,
        "avg_days_pass": round(float(pdays.mean()), 1) if pdays.size else None,
        "p_1st": round(float(p1), 3), "p_2nd": round(float(p2), 3), "p_3rd": round(float(p3), 3),
        "E_total_payout": round(e_total, 0),
        "avg_days_1st": round(float(np.mean(d1)), 0) if d1 else None,
        "EV_cycle": round(ev_cycle, 0),
        "EV_month": round(ev_cycle / (cyc_days / 21.0), 0) if cyc_days > 0 else None,
        "worst_day": round(min(r[3] for r in cres), 0),
        "maxdd": round(max(r[2] for r in cres), 0),
        "consist_viol": round(sum(1 for r in cres if r[4]) / sims, 3),
    }


def filt_regime(days_struct):
    ad, days, regs = days_struct
    out = []
    for day, rg in zip(days, regs):
        out.append(day if (rg.get("trend_str") == "adx_hi" or rg.get("vol") == "vol_hi") else [])
    return (ad, out, regs)


def main():
    c5 = build_days(CLEAN5, "frozen")
    ao = build_days(list(A_SLEEVES), "frozen")
    tadx = build_days(["trend_adx_cont"], "frozen")
    ir90t = build_days(["ir90_break", "trend_adx_cont"], "frozen")
    p36 = p36_days()
    combo = (c5[0], merge(c5[1], p36[1]), c5[2])
    hivol = filt_regime(combo)
    STRATS = {"combo": combo[1], "phase36": p36[1], "A_only": ao[1], "trend_adx": tadx[1],
              "ir90+trend": ir90t[1], "hivol_trendday": hivol[1]}

    cols = ["pass5", "pass10", "pass15", "fail", "ratio", "avg_days_pass", "p_1st", "p_2nd", "p_3rd",
            "E_total_payout", "avg_days_1st", "EV_cycle", "EV_month", "worst_day", "maxdd", "consist_viol"]
    results = {}

    # PART A: sizing sweep on combo across accounts
    for acct in ACCOUNTS:
        cost, T, M, mc = ACCOUNTS[acct]
        print(f"\n##### PART A — sizing sweep (combo) | {acct} (T${T:.0f} MLL${M:.0f} max{mc}) #####")
        print(f"{'sizing':16s} " + " ".join(f"{c[:9]:>9s}" for c in cols))
        sizings = {"flat1": flat(1, mc), "flat2": flat(2, mc), "flat3": flat(3, mc),
                   "flat4": flat(4, mc), "flat5": flat(5, mc), "cushion": cushion(T, mc)}
        if acct == "150K":
            sizings["1NQ(=10MNQ)"] = flat(10, mc)
        for sname, sf in sizings.items():
            r = evaluate(combo[1], acct, sf, -0.375 * M)
            results[f"A|{acct}|{sname}"] = r
            print(f"{sname:16s} " + " ".join(f"{str(r[c]):>9s}" for c in cols))

    # PART B: strategy sweep at 150K, fast size (flat3) + cushion + aggressive-combine/conservative-funded
    acct = "150K"
    cost, T, M, mc = ACCOUNTS[acct]
    print(f"\n##### PART B — strategy sweep | {acct}, flat3 combine #####")
    print(f"{'strategy':16s} " + " ".join(f"{c[:9]:>9s}" for c in cols))
    for sname, dd in STRATS.items():
        r = evaluate(dd, acct, flat(3, mc), -0.375 * M)
        results[f"B|{sname}|flat3"] = r
        print(f"{sname:16s} " + " ".join(f"{str(r[c]):>9s}" for c in cols))
    # aggressive combine (flat4) + scaleup funded, on combo
    r = evaluate(combo[1], acct, flat(4, mc), -0.375 * M, fmode="scaleup")
    results["B|combo|aggr4+scaleup"] = r
    print(f"{'aggr4+scaleup':16s} " + " ".join(f"{str(r[c]):>9s}" for c in cols))

    (OUT / "_phase47.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
