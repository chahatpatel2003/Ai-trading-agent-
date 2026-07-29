"""Phase 49 — search for better Topstep PASS engines beyond clean5 (frozen wrapper; signals from library).

Benchmark = clean5 and clean5+Phase36. Candidate families are built from the Phase-38 sleeve library on
the SAME basis (neutral-gate + cap3 + cushion sizing, 150K) so the comparison is apples-to-apples. Each
is run through the Phase-44 renewal-cycle evaluator (pass/fail/ratio/days/payout/EV-month/DD/worst-day)
plus a year-by-year 1-MNQ net (regime check incl 2022 bear / 2023 grind). PIT-clean (library is PIT-safe).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.phase44 import ACCOUNTS, build_days, cushion, evaluate, flat, merge, p36_days  # noqa: F401

OUT = Path("docs/phase49")
OUT.mkdir(parents=True, exist_ok=True)

FAMILIES = {
    "clean5 (benchmark)": ["trend_adx_cont", "ir90_break", "asia_sweep_fade", "ir60_fade", "disp_pb_fvg"],
    "opening_drive_mom": ["od_cont_struct", "od_cont_or"],
    "high_vol_breakout": ["volhi_orb", "atr_exp_orb"],
    "trend_day_cont": ["trend_adx_cont", "trend_htf_orb", "bos_cont"],
    "news_vol_proxy": ["active_long_volhi"],
    "first_pullback_impulse": ["disp_pb_fvg", "cisd_cont"],
    "ny_sweep_reclaim": ["pdhl_sweep_reclaim", "od_fail_sweep", "od_fail_mss"],
    "asia_london_sweep": ["asia_sweep_fade", "on_sweep_fade", "on_break_cont"],
    "vwap_reclaim_cont": ["vwap_reclaim", "vwap_pullback_trend"],
    "gap_continuation": ["gap_go"],
    "gap_fill": ["gap_fill"],
    "orb_vol_expansion": ["atr_exp_orb"],
    "failed_breakout": ["pdhl_fade", "od_fail_or"],
    "power_hour_cont": ["tod_power_hour"],
    "long_beta_raw": ["beta_long_above_pdc"],
    "long_beta_crashprot": ["beta_long_above_pdc", "asia_sweep_fade", "ir60_fade"],
    "ir_breakout_stack": ["ir30_break", "ir60_break", "ir90_break"],
}


def yearly_net(days):
    reg_days = json.loads((Path("docs/phase39") / "day_regime.json").read_text())
    all_days = sorted(reg_days.keys())
    out = defaultdict(float)
    for day, dd in zip(all_days, days):
        for (_et, net1, _mae, _q, _sl) in dd:
            out[day[:4]] += net1
    return {y: round(out.get(str(y), 0.0), 0) for y in range(2021, 2027)}


def main():
    acct = "150K"
    cost, T, M, mc = ACCOUNTS[acct]
    # benchmark clean5+phase36
    c5 = build_days(FAMILIES["clean5 (benchmark)"], "phase38")
    p36 = p36_days()
    combo = (c5[0], merge(c5[1], p36[1]), c5[2])

    results = {}
    yearly = {}
    cols = ["pass", "blow", "ratio", "med_days_pass", "first_payout_prob", "avg_payout",
            "E_total_payout", "EV_cycle", "EV_month", "maxdd", "worst_day", "consist_viol"]

    def run(name, struct):
        r = evaluate(struct[1], struct[2], acct, cushion(T, mc), -0.375 * M, -0.375 * M, sims=8000)
        results[name] = r
        yearly[name] = yearly_net(struct[1])

    run("clean5+Phase36 (BENCH)", combo)
    for name, sleeves in FAMILIES.items():
        run(name, build_days(sleeves, "phase38"))

    print(f"{'family':24s} " + " ".join(f"{c[:9]:>9s}" for c in cols))
    for name, r in results.items():
        print(f"{name:24s} " + " ".join(f"{str(r.get(c)):>9s}" for c in cols))
    print("\nYEAR-BY-YEAR net (1 MNQ, gated cap3) — 2022=bear, 2023=low-vol grind:")
    print(f"{'family':24s} {'2021':>8s} {'2022':>8s} {'2023':>8s} {'2024':>8s} {'2025':>8s} {'2026':>8s}")
    for name in results:
        y = yearly[name]
        print(f"{name:24s} " + " ".join(f"{y[yr]:>8.0f}" for yr in range(2021, 2027)))
    (OUT / "_phase49.json").write_text(json.dumps({"results": results, "yearly": yearly}, indent=2))


if __name__ == "__main__":
    main()
