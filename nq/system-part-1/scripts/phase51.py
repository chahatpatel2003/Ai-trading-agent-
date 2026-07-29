"""Phase 51 (user-labeled "Phase 50") — Topstep Combine SPRINT engine search.

Single job: pass the combine FAST (target 10-20 trading days). Not funded EV, not long-term ROI, not
protecting clean5. Searches 14 high-expectancy pass-engine families across 50K/100K/150K and sizing
(cushion, aggressive cushion, fixed, +daily loss stop / profit lock). Reports pass<=5/10/15/20-day
buckets, fail-before-pass, ratio, days, DD, worst-day, consistency. Bar: ratio>=1.25 (ideally 1.5+),
fail<pass. clean5+Phase36 = benchmark to beat. Frozen signals; flat cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.phase44 import ACCOUNTS, build_days, cushion, flat, merge, p36_days

OUT = Path("docs/phase51")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(51)
SIMS, HOR = 6000, 60

ENGINES = {
    "1_hivol_trend": ["volhi_orb", "atr_exp_orb"],
    "2_opening_drive": ["od_cont_struct", "od_cont_or"],
    "3_momentum_exp": ["od_cont_rexp", "volhi_orb"],
    "4_first_pullback": ["disp_pb_fvg", "cisd_cont"],
    "5_failed_brk_rev": ["od_fail_or", "pdhl_fade"],
    "6_sweep_displace": ["od_fail_sweep", "od_fail_mss", "cisd_cont"],
    "7_news_vol": ["active_long_volhi"],
    "8_gap_and_go": ["gap_go"],
    "9_orb_atr_exp": ["atr_exp_orb"],
    "10_liquidation_rev": ["on_sweep_fade", "london_sweep_fade"],
    "11_trend_adx_aggr": ["trend_adx_cont"],
    "12_ir90_aggr": ["ir90_break"],
    "13_momentum_stack": ["volhi_orb", "od_cont_struct", "ir30_break"],
}


def size_fn(name, T, mc):
    if name == "cushion":      # 1 -> 2 (+0.22T) -> 3 (+0.44T)
        return cushion(T, mc)
    if name == "aggr":         # 2 -> 3 (+0.15T) -> 4 (+0.30T)
        return cushion(T, mc, hi=True)
    if name == "flat3":
        return flat(3, mc)
    raise ValueError(name)


def sprint(days, start, T, M, sized, lstop, pstop):
    eq = peak = best = worst = mdd = 0.0
    active = 0
    consist = False
    for di in range(HOR):
        if start + di >= len(days):
            break
        d = 0.0
        traded = False
        for (_et, net1, mae1, _q, _sl) in days[start + di]:
            if lstop is not None and d <= lstop:
                break
            if pstop is not None and d >= pstop:
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
    return ("timeout", HOR, mdd, worst, consist)


def evaluate(days, acct, sized, lstop, pstop=None):
    cost, T, M, mc = ACCOUNTS[acct]
    n = len(days)
    starts = RNG.integers(0, n - HOR, SIMS)
    res = [sprint(days, int(s), T, M, sized, lstop, pstop) for s in starts]
    npass = sum(1 for r in res if r[0] == "pass")
    nblow = sum(1 for r in res if r[0] == "blow")
    pdays = np.array([r[1] for r in res if r[0] == "pass"])

    def pby(k):
        return round(sum(1 for r in res if r[0] == "pass" and r[1] <= k) / SIMS, 3)
    return {
        "pass5": pby(5), "pass10": pby(10), "pass15": pby(15), "pass20": pby(20),
        "fail": round(nblow / SIMS, 3),
        "ratio": round(npass / nblow, 2) if nblow else (99.9 if npass else 0.0),
        "med_days": round(float(np.median(pdays)), 0) if pdays.size else None,
        "avg_days": round(float(pdays.mean()), 1) if pdays.size else None,
        "maxdd": round(max(r[2] for r in res), 0), "worst_day": round(min(r[3] for r in res), 0),
        "consist": round(sum(1 for r in res if r[4]) / SIMS, 3),
    }


def main():
    p36 = p36_days()
    c5 = build_days(["trend_adx_cont", "ir90_break", "asia_sweep_fade", "ir60_fade", "disp_pb_fvg"], "phase38")
    bench = (c5[0], merge(c5[1], p36[1]), c5[2])
    structs = {name: build_days(sl, "phase38") for name, sl in ENGINES.items()}
    structs["14_phase36_Aplus"] = p36
    structs["BENCH_clean5+P36"] = bench

    cols = ["pass5", "pass10", "pass15", "pass20", "fail", "ratio", "med_days", "avg_days",
            "maxdd", "worst_day", "consist"]
    results = {}
    for acct in ("50K", "100K", "150K"):
        cost, T, M, mc = ACCOUNTS[acct]
        print(f"\n########## {acct} (target ${T:.0f}, MLL ${M:.0f}, max {mc}) ##########")
        print(f"{'engine':22s}{'sizing':9s} " + " ".join(f"{c[:8]:>8s}" for c in cols))
        for name, st in structs.items():
            for szn in ("cushion", "aggr"):
                r = evaluate(st[1], acct, size_fn(szn, T, mc), -0.375 * M)
                results[f"{acct}|{name}|{szn}"] = r
                mark = " <BENCH" if "BENCH" in name else ""
                print(f"{name:22s}{szn:9s} " + " ".join(f"{str(r[c]):>8s}" for c in cols) + mark)
    (OUT / "_phase51.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
