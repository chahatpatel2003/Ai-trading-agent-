"""Phase 56 — Can ANY strategy hit $1,000+/month/account net after Topstep costs?

clean5+Phase36 is the best engine but earns ~$400-500/mo/account — below the income requirement. Search
the higher-expectancy families with AGGRESSIVE funded sizing (now allowed: blow-after-payout OK) for a
full-cycle EV/month >= $1,000. Uses FLAT per-attempt cost (49/99/199 — the most favorable; if flat can't
hit $1k, monthly never will). Combine = cushion (best pass-before-blow). Funded = size S MNQ, payout cap
{M, 2M}. Reports full-cycle EV/month/account; flags >= $1,000. Engines from Phase-38 library + Phase36.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.phase44 import ACCOUNTS, build_days, cushion, merge, p36_days, sim_combine

OUT = Path("docs/phase56")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(56)
WIN_THR, MIN_WIN, DPM = 150.0, 5, 21

ENGINES = {
    "clean5+P36 (champ)": ["trend_adx_cont", "ir90_break", "asia_sweep_fade", "ir60_fade", "disp_pb_fvg"],
    "trend_day_cont": ["trend_adx_cont", "trend_htf_orb", "bos_cont"],
    "high_vol_breakout": ["volhi_orb", "atr_exp_orb"],
    "momentum_stack": ["volhi_orb", "od_cont_struct", "ir30_break"],
    "ir_breakout_stack": ["ir30_break", "ir60_break", "ir90_break"],
    "gap_and_go": ["gap_go"],
}


def sim_funded_sz(days, regimes, start, M, mc, lstop, cap, fsize, horizon=180):
    eq = peak = best = 0.0
    win = pots = 0
    wd = 0.0
    first = None
    locked = False
    for di in range(horizon):
        if start + di >= len(days):
            break
        d = 0.0
        for (_et, net1, mae1, _q, _sl) in days[start + di]:
            if lstop is not None and d <= lstop:
                break
            s = min(fsize, mc)
            bal = eq + d
            floor = 0.0 if locked else min(peak - M, 0.0)
            if bal - mae1 * s <= floor:
                return {"wd": wd, "pots": pots, "first": first, "blew": True, "day": di + 1}
            d += net1 * s
            bal = eq + d
            peak = max(peak, bal)
            if peak >= M:
                locked = True
            if bal <= (0.0 if locked else min(peak - M, 0.0)):
                return {"wd": wd, "pots": pots, "first": first, "blew": True, "day": di + 1}
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
    return {"wd": wd, "pots": pots, "first": first, "blew": False, "day": horizon}


def cycle_ev(days, regimes, acct, fsize, cap_mult, sims=6000):
    cost, T, M, mc = ACCOUNTS[acct]
    n = len(days)
    # combine (cushion, flat cost)
    cs = RNG.integers(0, n - 300, sims)
    cres = [sim_combine(days, regimes, int(s), T, M, cushion(T, mc), -0.375 * M, 300) for s in cs]
    npass = sum(1 for r in cres if r[0] == "pass")
    nblow = sum(1 for r in cres if r[0] == "blow")
    p_pass = npass / (npass + nblow) if (npass + nblow) else 0.0
    resd = np.array([r[1] for r in cres if r[0] in ("pass", "blow")])
    e_att = 1.0 / p_pass if p_pass > 0 else 999.0
    combine_cost = cost * e_att
    combine_days = e_att * (float(resd.mean()) if resd.size else 300)
    # funded
    fs = RNG.integers(0, n - 180, sims)
    fres = [sim_funded_sz(days, regimes, int(s), M, mc, -0.375 * M, cap_mult * M, fsize) for s in fs]
    e_payout = float(np.mean([r["wd"] for r in fres]))
    fdur = float(np.mean([r["day"] for r in fres]))
    surv = float(np.mean([0 if r["blew"] else 1 for r in fres]))
    ev_cycle = e_payout - combine_cost
    cyc_days = combine_days + fdur
    return {"p_pass": round(p_pass, 3), "combine_cost": round(combine_cost, 0),
            "combine_days": round(combine_days, 0), "E_payout": round(e_payout, 0),
            "funded_days": round(fdur, 0), "funded_surv": round(surv, 3),
            "EV_cycle": round(ev_cycle, 0), "cycle_mo": round(cyc_days / DPM, 1),
            "EV_month": round(ev_cycle / (cyc_days / DPM), 0) if cyc_days > 0 else None}


def main():
    p36 = p36_days()
    structs = {}
    for name, sl in ENGINES.items():
        c = build_days(sl, "phase38")
        if name == "clean5+P36 (champ)":
            structs[name] = (c[0], merge(c[1], p36[1]), c[2])
        else:
            structs[name] = c

    cols = ["p_pass", "combine_cost", "E_payout", "funded_surv", "funded_days", "EV_cycle",
            "cycle_mo", "EV_month"]
    results = {}
    hits = []
    for acct in ("50K", "100K", "150K"):
        cost, T, M, mc = ACCOUNTS[acct]
        print(f"\n######### {acct} (target ${T:.0f} MLL ${M:.0f} max {mc}, flat fee ${cost:.0f}) #########")
        print(f"{'engine|fsize|capx':30s} " + " ".join(f"{c[:9]:>9s}" for c in cols))
        for name, st in structs.items():
            for fsize in (1, 2, 3, 5):
                if fsize > mc:
                    continue
                for capx in (1, 2):
                    r = cycle_ev(st[1], st[2], acct, fsize, capx)
                    key = f"{acct}|{name}|{fsize}MNQ|cap{capx}M"
                    results[key] = r
                    flag = "  *** >=$1k" if (r["EV_month"] or 0) >= 1000 else ""
                    if flag:
                        hits.append(key)
                    print(f"{name + '|' + str(fsize) + '|' + str(capx):30s} " +
                          " ".join(f"{str(r[c]):>9s}" for c in cols) + flag)
    (OUT / "_phase56.json").write_text(json.dumps(results, indent=2))
    print(f"\n>>> cells reaching >=$1,000/month/account: {len(hits)}")
    for h in hits:
        print(f"    {h}: EV/mo ${results[h]['EV_month']}")
    best = max(results, key=lambda k: results[k]["EV_month"] or -9)
    print(f"\nBEST overall: {best} -> EV/mo ${results[best]['EV_month']} "
          f"(p_pass {results[best]['p_pass']}, funded_surv {results[best]['funded_surv']}, "
          f"E_payout ${results[best]['E_payout']})")


if __name__ == "__main__":
    main()
