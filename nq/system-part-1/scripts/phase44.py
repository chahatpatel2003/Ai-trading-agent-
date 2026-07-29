"""Phase 44 — Topstep 50K vs 100K vs 150K repeatable-cycle EV (frozen signals; wrapper only).

Flat per-attempt costs (50K $49 / 100K $99 / 150K $199, reset = same) => NO monthly bleed, so the
combine only needs pass-before-blow. Renewal cycle: buy combine -> (retry on blow) -> pass -> funded ->
collect payouts until the funded account blows -> restart. Optimize REPEATABLE CASHFLOW EV (EV/cycle,
EV/month), not survival forever. Strategies: clean5 (frozen harsh), Phase-36 infrequent edge (idb+swd),
clean5+P36, A-only, and a high-risk-pass/low-risk-payout mode. Per-account cushion sizing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from scripts.phase42 import A_SLEEVES, CLEAN5, _ts2mod, build_days

OUT = Path("docs/phase44")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(44)
WIN_THR, MIN_WIN = 150.0, 5
ACCOUNTS = {  # (cost, target, mll, max_contracts)
    "50K": (49.0, 3000.0, 2000.0, 5),
    "100K": (99.0, 6000.0, 3000.0, 10),
    "150K": (199.0, 9000.0, 4500.0, 15),
}


def cushion(T, mc, hi=False):
    if hi:  # high-risk pass: start 2, scale faster, cap 4
        s = (2, 3, 4); th = (0.15 * T, 0.30 * T)
    else:
        s = (1, 2, 3); th = (0.22 * T, 0.44 * T)

    def f(bal):
        c = s[0] if bal < th[0] else (s[1] if bal < th[1] else s[2])
        return min(c, mc)
    return f


def flat(k, mc):
    return lambda bal: min(k, mc)


def p36_days():
    df = pl.read_csv("docs/phase30_finalist/phase30_finalist_trades.csv")
    reg = json.loads((Path("docs/phase39") / "day_regime.json").read_text())
    all_days = sorted(reg.keys())
    bd = defaultdict(list)
    for r in df.iter_rows(named=True):
        bd[r["date"]].append((r["entry_ts"], float(r["net_pnl"]), abs(float(r["mae_pts"])) * 2.0, "A", r["strategy"]))
    days = []
    for d in all_days:
        days.append([(0, net, mae, q, sl) for (_t, net, mae, q, sl) in sorted(bd.get(d, []))])
    return all_days, days, [reg.get(d, {}) for d in all_days]


def merge(daysA, daysB):
    return [a + b for a, b in zip(daysA, daysB)]


def sim_combine(days, regimes, start, T, M, sized, lstop, horizon=300):
    equity = peak = best_day = maxdd = worst_day = 0.0
    active = 0
    consist = False
    for di in range(horizon):
        if start + di >= len(days):
            break
        day_pnl = 0.0
        traded = False
        for (_et, net1, mae1, _q, _sl) in days[start + di]:
            if lstop is not None and day_pnl <= lstop:
                break
            bal = equity + day_pnl
            s = sized(bal)
            pk = max(peak, bal)
            if bal - mae1 * s <= min(pk - M, 0.0):
                return ("blow", di + 1, maxdd, min(worst_day, day_pnl), consist)
            day_pnl += net1 * s
            traded = True
            bal = equity + day_pnl
            peak = max(peak, bal)
            if bal <= min(peak - M, 0.0):
                return ("blow", di + 1, max(maxdd, peak - bal), min(worst_day, day_pnl), consist)
        equity += day_pnl
        if traded:
            active += 1
        best_day = max(best_day, day_pnl)
        worst_day = min(worst_day, day_pnl)
        maxdd = max(maxdd, peak - equity)
        if equity >= T and active >= 2:
            if best_day <= 0.5 * equity:
                return ("pass", di + 1, maxdd, worst_day, consist)
            consist = True
    return ("timeout", horizon, maxdd, worst_day, consist)


def sim_funded(days, regimes, start, M, mc, lstop, payout_cap, horizon=252):
    equity = peak = best_day = maxdd = worst_day = 0.0
    win_days = payouts = 0
    withdrawn = 0.0
    first = None
    locked = False
    sized = flat(1, mc)  # low-risk payout mode
    for di in range(horizon):
        if start + di >= len(days):
            break
        day_pnl = 0.0
        for (_et, net1, mae1, _q, _sl) in days[start + di]:
            if lstop is not None and day_pnl <= lstop:
                break
            bal = equity + day_pnl
            s = sized(bal)
            floor = 0.0 if locked else min(peak - M, 0.0)
            if bal - mae1 * s <= floor:
                return {"wd": withdrawn, "pots": payouts, "first": first, "blew": True,
                        "day": di + 1, "maxdd": max(maxdd, peak - (bal - mae1 * s)),
                        "worst": min(worst_day, day_pnl)}
            day_pnl += net1 * s
            bal = equity + day_pnl
            peak = max(peak, bal)
            if peak >= M:
                locked = True
            if bal <= (0.0 if locked else min(peak - M, 0.0)):
                return {"wd": withdrawn, "pots": payouts, "first": first, "blew": True,
                        "day": di + 1, "maxdd": max(maxdd, peak - bal), "worst": min(worst_day, day_pnl)}
        equity += day_pnl
        best_day = max(best_day, day_pnl)
        worst_day = min(worst_day, day_pnl)
        maxdd = max(maxdd, peak - equity)
        if day_pnl >= WIN_THR:
            win_days += 1
        if win_days >= MIN_WIN and equity > 0 and best_day <= 0.5 * equity:
            amt = min(equity, payout_cap)
            withdrawn += amt
            equity -= amt
            payouts += 1
            first = first or (di + 1)
            win_days = 0
            best_day = 0.0
    return {"wd": withdrawn, "pots": payouts, "first": first, "blew": False, "day": horizon,
            "maxdd": maxdd, "worst": worst_day}


def evaluate(days, regimes, acct, sized_combine, lstop_c, lstop_f, sims=10000):
    cost, T, M, mc = ACCOUNTS[acct]
    n = len(days)
    payout_cap = M
    cs = RNG.integers(0, n - 300, sims)
    cres = [sim_combine(days, regimes, int(s), T, M, sized_combine, lstop_c) for s in cs]
    npass = sum(1 for r in cres if r[0] == "pass")
    nblow = sim_blow = sum(1 for r in cres if r[0] == "blow")
    pdays = np.array([r[1] for r in cres if r[0] == "pass"])
    resdays = np.array([r[1] for r in cres if r[0] in ("pass", "blow")])
    p_pass = npass / (npass + nblow) if (npass + nblow) else 0.0

    def noresult(d):
        return round(sum(1 for r in cres if r[1] > d or r[0] == "timeout") / sims, 3) if False else \
            round(sum(1 for r in cres if (r[0] == "timeout") or (r[0] in ("pass", "blow") and r[1] > d)) / sims, 3)

    fs = RNG.integers(0, n - 252, sims)
    fres = [sim_funded(days, regimes, int(s), M, mc, lstop_f, payout_cap) for s in fs]
    reached = [r for r in fres if r["first"]]
    first_prob = len(reached) / sims
    pots = [r["pots"] for r in fres]
    allpots = [r["wd"] / r["pots"] for r in fres if r["pots"] > 0]
    blow_after_first = (sum(1 for r in reached if r["blew"]) / len(reached)) if reached else 0.0
    e_total_payout = float(np.mean([r["wd"] for r in fres]))
    f_days_to_blow = float(np.mean([r["day"] for r in fres if r["blew"]])) if any(r["blew"] for r in fres) else 252.0

    # renewal cycle EV
    e_attempts = 1.0 / p_pass if p_pass > 0 else 999.0
    combine_cost = cost * e_attempts
    ev_cycle = e_total_payout - combine_cost
    combine_days = e_attempts * (float(resdays.mean()) if resdays.size else 300)
    cycle_days = combine_days + f_days_to_blow
    ev_month = ev_cycle / (cycle_days / 21.0) if cycle_days > 0 else None
    return {
        "pass": round(npass / sims, 3), "blow": round(nblow / sims, 3),
        "p_pass_resolved": round(p_pass, 3),
        "ratio": round(npass / nblow, 2) if nblow else None,
        "avg_days_pass": round(float(pdays.mean()), 0) if pdays.size else None,
        "med_days_pass": round(float(np.median(pdays)), 0) if pdays.size else None,
        "noresult90": noresult(90),
        "first_payout_prob": round(first_prob, 3),
        "avg_payout": round(float(np.mean(allpots)), 0) if allpots else 0,
        "blow_after_first": round(blow_after_first, 3),
        "E_total_payout": round(e_total_payout, 0),
        "E_combine_cost": round(combine_cost, 0),
        "EV_cycle": round(ev_cycle, 0),
        "EV_month": round(ev_month, 0) if ev_month else None,
        "cycle_months": round(cycle_days / 21.0, 1),
        "consist_viol": round(sum(1 for r in cres if r[4]) / sims, 3),
        "maxdd": round(max(r[2] for r in cres), 0),
        "worst_day": round(min(r[3] for r in cres), 0),
    }


def main():
    clean5 = build_days(CLEAN5, "frozen")
    aonly = build_days(list(A_SLEEVES), "frozen")
    p36 = p36_days()
    combo = (clean5[0], merge(clean5[1], p36[1]), clean5[2])
    STRATS = {"clean5": clean5, "phase36": p36, "clean5+p36": combo, "A_only": aonly}

    cols = ["pass", "blow", "ratio", "med_days_pass", "noresult90", "first_payout_prob", "avg_payout",
            "blow_after_first", "E_total_payout", "E_combine_cost", "EV_cycle", "EV_month",
            "cycle_months", "consist_viol", "maxdd", "worst_day"]
    results = {}
    for acct in ACCOUNTS:
        cost, T, M, mc = ACCOUNTS[acct]
        print(f"\n========== {acct} (cost ${cost:.0f}, target ${T:.0f}, MLL ${M:.0f}, max {mc}ct) ==========")
        print(f"{'strategy':18s} " + " ".join(f"{c[:9]:>9s}" for c in cols))
        for sname, (_ad, days, regs) in STRATS.items():
            sized = cushion(T, mc)
            r = evaluate(days, regs, acct, sized, -0.375 * M, -0.375 * M)
            results[f"{acct}|{sname}|cushion"] = r
            print(f"{sname:18s} " + " ".join(f"{str(r[c]):>9s}" for c in cols))
        # version 5: high-risk pass mode (clean5), low-risk funded
        sized_hr = cushion(T, mc, hi=True)
        rhr = evaluate(clean5[1], clean5[2], acct, sized_hr, -0.375 * M, -0.375 * M)
        results[f"{acct}|clean5|HIGHRISK_pass"] = rhr
        print(f"{'clean5 HR-pass':18s} " + " ".join(f"{str(rhr[c]):>9s}" for c in cols))
    (OUT / "_phase44.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
