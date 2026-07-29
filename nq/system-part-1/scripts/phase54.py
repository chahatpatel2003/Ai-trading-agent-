"""Phase 54 — Topstep subscription-time-optimized pass plan (150K, clean5+Phase36).

clean5+Phase36 passes ~79% but median ~62 days => ~3 months of MONTHLY subscription per pass, dragging
EV. This optimizes the COMBINE for net EV after monthly billing using early-reset rules: time caps
(20/30/45/60d), progress gates (must be above $X by day D else reset), and front-loaded sizing. A
"campaign" re-attempts until a pass; we measure subscription-months-to-pass and EV per subscription
month (funded payout held constant from a 1-MNQ funded sim). Costs: monthly $150, reset $150, activ $150.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.phase44 import build_days, merge, p36_days, sim_funded

OUT = Path("docs/phase54")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(54)
T, M, MC = 9000.0, 4500.0, 15
MONTHLY, RESET, ACTIV = 150.0, 150.0, 150.0
DAYS_PER_MONTH = 21


def size_cushion(b):
    return 1 if b < 2000 else (2 if b < 4000 else 3)


def size_front23(b):
    return 2 if b < 3000 else 3


def size_aggr234(b):
    return 2 if b < 1500 else (3 if b < 4000 else 4)


SIZINGS = {"cushion123": size_cushion, "front_2to3": size_front23, "aggr_2to3to4": size_aggr234}


def attempt(days, start, sized, lstop, time_cap, gates):
    """One combine attempt. Returns (outcome, day) where outcome in pass|blow|reset."""
    eq = peak = best = 0.0
    active = 0
    gi = 0
    gsorted = sorted(gates)
    for di in range(time_cap):
        if start + di >= len(days):
            return ("reset", di + 1)
        d = 0.0
        traded = False
        for (_et, net1, mae1, _q, _sl) in days[start + di]:
            if lstop is not None and d <= lstop:
                break
            bal = eq + d
            s = sized(bal)
            if bal - mae1 * s <= min(max(peak, bal) - M, 0.0):
                return ("blow", di + 1)
            d += net1 * s
            traded = True
            bal = eq + d
            peak = max(peak, bal)
            if bal <= min(peak - M, 0.0):
                return ("blow", di + 1)
        eq += d
        if traded:
            active += 1
        best = max(best, d)
        if eq >= T and active >= 2 and best <= 0.5 * eq:
            return ("pass", di + 1)
        # progress gate (checked at end of the gate day)
        while gi < len(gsorted) and (di + 1) >= gsorted[gi][0]:
            if eq < gsorted[gi][1]:
                return ("reset", di + 1)
            gi += 1
    return ("reset", time_cap)  # time cap reached without pass


def campaign(days, sized, lstop, time_cap, gates, max_attempts=40):
    n = len(days)
    months = 0.0
    resets = 0
    for _ in range(max_attempts):
        start = int(RNG.integers(0, n - time_cap - 1))
        outcome, day = attempt(days, start, sized, lstop, time_cap, gates)
        months += int(np.ceil(day / DAYS_PER_MONTH))
        if outcome == "pass":
            return {"passed": True, "months": months, "resets": resets, "last_days": day}
        resets += 1
    return {"passed": False, "months": months, "resets": resets, "last_days": time_cap}


def per_attempt(days, sized, lstop, time_cap, gates, sims=8000):
    n = len(days)
    outs = []
    for _ in range(sims):
        start = int(RNG.integers(0, n - time_cap - 1))
        outs.append(attempt(days, start, sized, lstop, time_cap, gates))
    npass = sum(1 for o in outs if o[0] == "pass")
    nblow = sum(1 for o in outs if o[0] == "blow")
    nreset = sum(1 for o in outs if o[0] == "reset")
    pdays = [o[1] for o in outs if o[0] == "pass"]
    return {"pass": round(npass / sims, 3), "blow": round(nblow / sims, 3),
            "reset": round(nreset / sims, 3),
            "med_pass_days": round(float(np.median(pdays)), 0) if pdays else None}


def evaluate(days, sized, lstop, time_cap, gates, funded_payout, funded_months, sims=4000):
    pa = per_attempt(days, sized, lstop, time_cap, gates)
    camps = [campaign(days, sized, lstop, time_cap, gates) for _ in range(sims)]
    passed = [c for c in camps if c["passed"]]
    p_camp_pass = len(passed) / sims
    months_to_pass = float(np.mean([c["months"] for c in passed])) if passed else 99.0
    resets_to_pass = float(np.mean([c["resets"] for c in passed])) if passed else 99.0
    cost_to_pass = months_to_pass * MONTHLY + resets_to_pass * RESET + ACTIV
    ev_attempt = p_camp_pass * funded_payout - (np.mean([c["months"] for c in camps]) * MONTHLY +
                                                np.mean([c["resets"] for c in camps]) * RESET) - ACTIV * p_camp_pass
    ev_campaign = funded_payout - cost_to_pass
    ev_month = ev_campaign / (months_to_pass + funded_months) if (months_to_pass + funded_months) > 0 else None
    return {**pa, "months_to_pass": round(months_to_pass, 2), "resets_to_pass": round(resets_to_pass, 2),
            "cost_to_pass": round(cost_to_pass, 0), "EV_attempt": round(float(ev_attempt), 0),
            "EV_campaign": round(ev_campaign, 0), "EV_month": round(ev_month, 0) if ev_month else None}


def main():
    c5 = build_days(["trend_adx_cont", "ir90_break", "asia_sweep_fade", "ir60_fade", "disp_pb_fvg"], "phase38")
    p36 = p36_days()
    days = merge(c5[1], p36[1])
    regs = c5[2]
    # funded held constant: 1 MNQ clean5+Phase36, 150K
    n = len(days)
    fres = [sim_funded(days, regs, int(RNG.integers(0, n - 180)), M, MC, -0.375 * M, M, 180)
            for _ in range(8000)]
    funded_payout = float(np.mean([r["wd"] for r in fres]))
    funded_months = float(np.mean([(r["first"] or 180) for r in fres])) / DAYS_PER_MONTH
    print(f"funded (held constant): E[payout]=${funded_payout:.0f}  funded_months≈{funded_months:.1f}")

    GATES = {
        "none": [],
        "mild": [(10, 0.0), (20, 1500.0)],
        "med": [(10, 500.0), (20, 2500.0), (30, 4500.0)],
        "aggr": [(10, 1000.0), (20, 3500.0), (30, 6000.0)],
    }
    cols = ["pass", "blow", "reset", "med_pass_days", "months_to_pass", "resets_to_pass",
            "cost_to_pass", "EV_attempt", "EV_campaign", "EV_month"]
    results = {}

    # baseline: no cap (run long), cushion, lstop -750
    base = evaluate(days, size_cushion, -750.0, 120, [], funded_payout, funded_months)
    results["BASELINE cushion cap120 nogate"] = base
    print("\n=== BASELINE (cushion, cap120, no gate, lstop -750) ===")
    print("  " + "  ".join(f"{c}={base[c]}" for c in cols))

    print(f"\n{'plan (sizing|cap|gate|lstop)':40s} " + " ".join(f"{c[:9]:>9s}" for c in cols))
    for szn, sf in SIZINGS.items():
        for cap in (20, 30, 45, 60):
            for gname in ("none", "med", "aggr"):
                for lstop in (-1000.0,):
                    r = evaluate(days, sf, lstop, cap, GATES[gname], funded_payout, funded_months)
                    key = f"{szn}|cap{cap}|{gname}|ls{int(-lstop)}"
                    results[key] = r
                    print(f"{key:40s} " + " ".join(f"{str(r[c]):>9s}" for c in cols))
    (OUT / "_phase54.json").write_text(json.dumps(results, indent=2))

    best = max((k for k in results if k != "BASELINE cushion cap120 nogate"),
               key=lambda k: results[k]["EV_month"] or -9)
    print(f"\nBEST by EV/month: {best} -> {results[best]}")


if __name__ == "__main__":
    main()
