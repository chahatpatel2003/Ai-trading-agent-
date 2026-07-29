"""Phase 50 — anti-bias BLIND tournament: try to kill clean5.

100+ anonymous candidate families (S001..) are generated from the Phase-38 library EXCLUDING clean5's
5 sleeves, so no candidate can mimic clean5. clean5 and clean5+Phase36 enter only as hidden benchmarks
(revealed at the very end). Every candidate gets the SAME wrapper (neutral-gate + cap3 + cushion sizing,
150K, flat cost) and the SAME test budget. Ranked blind on EV/month, pass/fail ratio, payout prob, 2023
net, fastest viable pass, and correlation vs clean5. Kill criteria applied at the end.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.phase44 import (
    ACCOUNTS, build_days, cushion, flat, merge, p36_days, sim_combine, sim_funded,
)

OUT = Path("docs/phase50")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(50)
ACCT = "150K"
COST, T, M, MC = ACCOUNTS[ACCT]
LSTOP = -0.375 * M
SIMS, HC, HF = 3000, 180, 150
CLEAN5 = ["trend_adx_cont", "ir90_break", "asia_sweep_fade", "ir60_fade", "disp_pb_fvg"]

# non-clean5 sleeve pool, grouped by search area (clean5 sleeves excluded by construction)
AREAS = {
    "hivol_trend": ["volhi_orb", "atr_exp_orb", "active_long_volhi"],
    "od_cont": ["od_cont_struct", "od_cont_or", "od_cont_rexp"],
    "od_fail": ["od_fail_or", "od_fail_sweep", "od_fail_mss"],
    "first_pullback": ["disp_pb_ema9", "cisd_cont"],
    "failed_brk": ["pdhl_fade", "pdc_reject", "lvn_reject"],
    "vwap": ["vwap_reclaim", "vwap_reject", "vwap_revert", "vwap_pullback_trend"],
    "pdhl_sweep": ["pdhl_sweep_reclaim", "pdhl_break_cont"],
    "asia_london": ["on_sweep_fade", "london_sweep_fade", "on_break_cont"],
    "gap": ["gap_go", "gap_fill", "gap_fade_pdc"],
    "range_exp": ["range_or_fade"],
    "compression": ["comp_break"],
    "power_hour": ["tod_power_hour", "tod_pm_revert", "tod_lunch_fade"],
    "value_area": ["vah_reject", "va_revert", "poc_reclaim_cont", "va_break_cont"],
    "ir_brk": ["ir30_break", "ir60_break"],
    "trend_struct": ["bos_cont", "trend_htf_orb", "aligned_cont"],
    "lowvol_mr": ["bb_fade", "nr_day_revert", "rsi_revert", "atr_contract_revert", "calm_long_vollo"],
    "long_beta": ["beta_long_above_vwap", "beta_long_uptrend", "beta_long_above_pdc",
                  "beta_long_gapup", "beta_long_trend1h"],
    "dow": ["dow_long_mon", "dow_long_tue", "dow_long_wed", "dow_long_thu", "dow_long_fri"],
    "short_hedge": ["BASELINE_short", "on_sweep_fade", "london_sweep_fade"],
}
HEDGES = ["on_sweep_fade", "london_sweep_fade", "range_or_fade", "pdhl_fade", "vah_reject"]


def gen_candidates():
    cands = {}  # label -> list of sleeves (or special 'phase36'/'clean5' tags)
    seen = set()

    def add(sleeves):
        key = tuple(sorted(sleeves))
        if key in seen or not sleeves:
            return
        seen.add(key)
        cands[f"__{len(cands)}"] = list(sleeves)

    # singles (skip rexp_rollbreak: 0 trades)
    for area, sls in AREAS.items():
        for s in sls:
            add([s])
    # area stacks
    for area, sls in AREAS.items():
        if len(sls) >= 2:
            add(sls)
    # long-beta + crash-protect hedge (every beta x a couple hedges)
    for b in AREAS["long_beta"]:
        for h in HEDGES[:3]:
            add([b, h])
    # momentum + hedge cross pairs
    for m in ["volhi_orb", "od_cont_struct", "bos_cont", "ir30_break", "gap_go", "trend_htf_orb"]:
        for h in HEDGES[:2]:
            add([m, h])
    # diversified triples (momentum + MR + beta)
    for m in ["volhi_orb", "ir30_break", "od_cont_struct", "bos_cont"]:
        for mr in ["range_or_fade", "vwap_revert", "on_sweep_fade"]:
            add([m, mr, "beta_long_above_pdc"])
    # value-area + vwap combos
    add(["vwap_reclaim", "poc_reclaim_cont"])
    add(["va_revert", "vwap_revert", "range_or_fade"])
    # gap stacks
    add(["gap_go", "gap_fill"])
    add(["gap_go", "comp_break", "volhi_orb"])
    # dow stacks
    add(["dow_long_tue", "dow_long_mon"])
    # beta family stack + hedge
    add(["beta_long_above_pdc", "beta_long_gapup", "on_sweep_fade", "range_or_fade"])
    return cands


def daily_vec(days, all_days):
    bd = defaultdict(float)
    ad = sorted(all_days)
    for day, dd in zip(ad, days):
        for (_et, net1, _m, _q, _sl) in dd:
            bd[day] += net1
    return np.array([bd.get(d, 0.0) for d in ad]), bd


def quick_eval(days, regimes):
    n = len(days)
    cs = RNG.integers(0, n - HC, SIMS)
    cres = [sim_combine(days, regimes, int(s), T, M, cushion(T, MC), LSTOP, HC) for s in cs]
    npass = sum(1 for r in cres if r[0] == "pass")
    nblow = sum(1 for r in cres if r[0] == "blow")
    pdays = np.array([r[1] for r in cres if r[0] == "pass"])
    resd = np.array([r[1] for r in cres if r[0] in ("pass", "blow")])
    p_pass = npass / (npass + nblow) if (npass + nblow) else 0.0
    fs = RNG.integers(0, n - HF, SIMS)
    fres = [sim_funded(days, regimes, int(s), M, MC, LSTOP, M, HF) for s in fs]
    first = np.mean([1 if r["first"] else 0 for r in fres])
    e_total = float(np.mean([r["wd"] for r in fres]))
    fdur = float(np.mean([r["first"] or HF for r in fres]))
    e_att = 1.0 / p_pass if p_pass > 0 else 999.0
    ev_cycle = e_total - COST * e_att
    cyc_days = e_att * (float(resd.mean()) if resd.size else HC) + fdur
    return {
        "pass": round(npass / SIMS, 3), "blow": round(nblow / SIMS, 3),
        "ratio": round(npass / nblow, 2) if nblow else (99.9 if npass else 0.0),
        "med_days": round(float(np.median(pdays)), 0) if pdays.size else None,
        "first_payout": round(float(first), 3), "E_total_payout": round(e_total, 0),
        "EV_month": round(ev_cycle / (cyc_days / 21.0), 0) if cyc_days > 0 else None,
        "maxdd": round(max(r[2] for r in cres), 0), "worst_day": round(min(r[3] for r in cres), 0),
    }


def main():
    raw = gen_candidates()
    c5 = build_days(CLEAN5, "phase38")
    p36 = p36_days()
    combo = (c5[0], merge(c5[1], p36[1]), c5[2])
    all_days = c5[0]
    clean5_vec, _ = daily_vec(c5[1], all_days)

    # assemble all entries incl hidden benchmarks, then shuffle to anonymous S-IDs
    entries = []  # (truename, days_struct)
    for _lbl, sleeves in raw.items():
        try:
            entries.append(("+".join(sleeves), build_days(sleeves, "phase38")))
        except Exception as e:  # noqa: BLE001
            print("skip", sleeves, repr(e)[:60])
    entries.append(("phase36_only", p36))
    entries.append(("BENCH_clean5", c5))
    entries.append(("BENCH_clean5+phase36", combo))
    order = list(range(len(entries)))
    RNG.shuffle(order)
    sid = {order[i]: f"S{ i+1:03d}" for i in range(len(entries))}

    rows = {}
    reveal = {}
    for i, (truename, struct) in enumerate(entries):
        s = sid[i]
        reveal[s] = truename
        r = quick_eval(struct[1], struct[2])
        v, _ = daily_vec(struct[1], all_days)
        r["corr_clean5"] = round(float(np.corrcoef(v, clean5_vec)[0, 1]), 2) if v.std() > 0 else None
        yb = defaultdict(float)
        for day, dd in zip(sorted(all_days), struct[1]):
            for (_et, net1, _m, _q, _sl) in dd:
                yb[day[:4]] += net1
        r["y2022"] = round(yb.get("2022", 0.0), 0)
        r["y2023"] = round(yb.get("2023", 0.0), 0)
        rows[s] = r
    (OUT / "_phase50.json").write_text(json.dumps({"rows": rows, "reveal": reveal}, indent=2))

    def top(metric, k=20, rev=True, filt=None):
        items = [(s, r) for s, r in rows.items() if (filt is None or filt(r))]
        items = [(s, r) for s, r in items if r.get(metric) is not None]
        items.sort(key=lambda x: x[1][metric], reverse=rev)
        return items[:k]

    bid = next(s for s, t in reveal.items() if t == "BENCH_clean5+phase36")
    c5id = next(s for s, t in reveal.items() if t == "BENCH_clean5")
    B = rows[bid]
    print(f"total candidates: {len(rows)}  (clean5+Phase36 hidden as {bid}, clean5 as {c5id})")
    print(f"\nBENCH clean5+Phase36 ({bid}): EV/mo={B['EV_month']} ratio={B['ratio']} pass={B['pass']} "
          f"med_days={B['med_days']} first_payout={B['first_payout']} 2023={B['y2023']} maxdd={B['maxdd']}")

    print("\n=== TOP 20 by EV/month ===")
    for s, r in top("EV_month"):
        tag = " <<<BENCH" if s in (bid, c5id) else ""
        print(f"  {s} EV/mo={r['EV_month']:>5} ratio={r['ratio']:>5} pass={r['pass']} med={r['med_days']} "
              f"2023={r['y2023']:>6.0f} dd={r['maxdd']:>6.0f} corrC5={r['corr_clean5']}{tag}")
    print("\n=== TOP 20 by pass/fail ratio (EV/mo>=150 filter) ===")
    for s, r in top("ratio", filt=lambda r: (r["EV_month"] or -9) >= 150):
        tag = " <<<BENCH" if s in (bid, c5id) else ""
        print(f"  {s} ratio={r['ratio']:>5} EV/mo={r['EV_month']:>5} pass={r['pass']} 2023={r['y2023']:>6.0f}{tag}")
    print("\n=== TOP 20 by payout probability ===")
    for s, r in top("first_payout"):
        tag = " <<<BENCH" if s in (bid, c5id) else ""
        print(f"  {s} first_payout={r['first_payout']} EV/mo={r['EV_month']:>5} ratio={r['ratio']}{tag}")
    print("\n=== TOP 20 by 2023 net (the discriminator) ===")
    for s, r in top("y2023"):
        tag = " <<<BENCH" if s in (bid, c5id) else ""
        print(f"  {s} 2023={r['y2023']:>7.0f} EV/mo={r['EV_month']:>5} ratio={r['ratio']:>5} corrC5={r['corr_clean5']}{tag}")
    print("\n=== TOP 20 fastest viable pass (ratio>=1.5, EV/mo>=150) ===")
    for s, r in top("med_days", rev=False, filt=lambda r: r["ratio"] >= 1.5 and (r["EV_month"] or -9) >= 150):
        tag = " <<<BENCH" if s in (bid, c5id) else ""
        print(f"  {s} med_days={r['med_days']} ratio={r['ratio']:>5} EV/mo={r['EV_month']:>5} 2023={r['y2023']:>6.0f}{tag}")

    # kill-criteria checks vs benchmark
    print("\n=== KILL-CRITERIA CHECK vs clean5+Phase36 ===")
    bench_ev = B["EV_month"]
    beat_ev10 = [(s, r) for s, r in rows.items() if s not in (bid, c5id)
                 and (r["EV_month"] or -9) >= 1.10 * bench_ev]
    faster_sameEV = [(s, r) for s, r in rows.items() if s not in (bid, c5id)
                     and (r["EV_month"] or -9) >= 0.95 * bench_ev and (r["med_days"] or 999) < (B["med_days"] or 0)
                     and r["ratio"] >= 1.5]
    fix2023 = [(s, r) for s, r in rows.items() if s not in (bid, c5id)
               and r["y2023"] > 0 and (r["EV_month"] or -9) >= 0.90 * bench_ev and r["ratio"] >= 1.5]
    lowerdd_samepayout = [(s, r) for s, r in rows.items() if s not in (bid, c5id)
                          and r["maxdd"] < B["maxdd"] and r["first_payout"] >= B["first_payout"]
                          and (r["EV_month"] or -9) >= 0.95 * bench_ev]
    print(f"  beat EV/mo by >=10% ({round(1.1*bench_ev)}): {[(s,r['EV_month']) for s,r in beat_ev10]}")
    print(f"  similar EV + faster pass + ratio>=1.5: {[(s,r['EV_month'],r['med_days']) for s,r in faster_sameEV]}")
    print(f"  fixes 2023 (+ve) w/ EV>=90% + ratio>=1.5: {[(s,r['y2023'],r['EV_month'],r['ratio']) for s,r in fix2023]}")
    print(f"  lower DD + >= payout + >=95% EV: {[(s,r['maxdd'],r['EV_month']) for s,r in lowerdd_samepayout]}")
    print("\n=== REVEAL (top entries) ===")
    for s, r in top("EV_month", k=12):
        print(f"  {s} = {reveal[s]}")
    print(f"\n  clean5+Phase36 = {bid} | clean5 = {c5id} | phase36_only = "
          f"{next(s for s,t in reveal.items() if t=='phase36_only')}")


if __name__ == "__main__":
    main()
