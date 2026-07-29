"""Phase 57 — full GC/MGC Topstep evaluation runner.

For every family + combined portfolios: trade metrics + Topstep combine/funded EV
across 50K/100K/150K (funded sizing swept, best EV/month kept). Writes report+JSON.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.phase57_engine import (
    ACCOUNTS,
    FAMILIES,
    build_days,
    cushion,
    load_sessions,
    sim_combine,
    sim_funded,
    trade_metrics,
)

OUT = Path("docs/phase57")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(57)
SIMS = 3000

# combined portfolios (merge trade lists)
PORTFOLIOS = {
    "P_bestOOS (fpb+gap+trend+vol)": ["11_first_pb", "7_gap_cont", "12_trend_day", "9_vol_break"],
    "P_posOOS (fpb+gap+trend)": ["11_first_pb", "7_gap_cont", "12_trend_day"],
    "P_frequent (drive+lonny+vwap)": ["1_open_drive", "2_lon_ny_break", "5_vwap"],
}


def topstep_full(days, acct):
    """Returns combine stats + best-fsize funded EV/month."""
    cost, T, M, mc = ACCOUNTS[acct]
    n = len(days)
    if n < 320:
        return None
    cs = RNG.integers(0, n - 300, SIMS)
    cres = [sim_combine(days, int(s), T, M, cushion(T, mc), -0.375 * M, 300) for s in cs]
    npass = sum(1 for r in cres if r[0] == "pass")
    nblow = sum(1 for r in cres if r[0] == "blow")
    p_pass = npass / (npass + nblow) if (npass + nblow) else 0.0
    pdays = np.array([r[1] for r in cres if r[0] == "pass"])
    resd = np.array([r[1] for r in cres if r[0] in ("pass", "blow")])
    e_att = 1.0 / p_pass if p_pass > 0 else 999.0
    combine_cost = cost * e_att
    combine_days = e_att * (float(resd.mean()) if resd.size else 300)

    fs_starts = RNG.integers(0, n - 180, SIMS)
    best = None
    for fsize in [1, 2, 3, 5, mc]:
        fsize = min(fsize, mc)
        fres = [sim_funded(days, int(s), M, mc, -0.375 * M, M, fsize) for s in fs_starts]
        e_payout = float(np.mean([r["wd"] for r in fres]))
        surv = float(np.mean([0 if r["blew"] else 1 for r in fres]))
        fdur = float(np.mean([r["day"] for r in fres]))
        first_prob = float(np.mean([1 if r["first"] else 0 for r in fres]))
        ev_cycle = e_payout - combine_cost
        cyc_days = combine_days + fdur
        ev_month = ev_cycle / (cyc_days / 21.0) if cyc_days > 0 else -1e9
        cand = {
            "fsize": fsize, "E_payout": round(e_payout, 0), "funded_surv": round(surv, 3),
            "first_payout_prob": round(first_prob, 3), "EV_cycle": round(ev_cycle, 0),
            "cycle_mo": round(cyc_days / 21.0, 1), "EV_month": round(ev_month, 0),
        }
        if best is None or cand["EV_month"] > best["EV_month"]:
            best = cand
    out = {
        "p_pass": round(p_pass, 3),
        "p_fail": round(nblow / (npass + nblow), 3) if (npass + nblow) else None,
        "pass_rate_all": round(npass / SIMS, 3),
        "blow_rate_all": round(nblow / SIMS, 3),
        "med_days_pass": round(float(np.median(pdays)), 0) if pdays.size else None,
        "combine_cost": round(combine_cost, 0),
    }
    out.update(best)
    return out


def worst_month(trades, fsize=3):
    """Worst calendar-month PnL at given contract size (point estimate of tail risk)."""
    bym = defaultdict(float)
    for (d, _sm, net, _mae) in trades:
        bym[(d.year, d.month)] += net * fsize
    return round(min(bym.values()), 0) if bym else 0.0


def main():
    sess = load_sessions()
    sess_dates = [s["date"] for s in sess]
    fam_trades = {name: fn(sess) for name, fn in FAMILIES.items()}
    for pname, members in PORTFOLIOS.items():
        merged = []
        for m in members:
            merged.extend(fam_trades[m])
        merged.sort(key=lambda t: (t[0], t[1]))
        fam_trades[pname] = merged

    report = {}
    for name, trades in fam_trades.items():
        m = trade_metrics(trades)
        if m is None:
            continue
        days = build_days(trades, sess_dates)
        accts = {}
        for acct in ("50K", "100K", "150K"):
            r = topstep_full(days, acct)
            if r:
                r["worst_month_3c"] = worst_month(trades, 3)
                accts[acct] = r
        report[name] = {"metrics": m, "topstep": accts}

    (OUT / "phase57_results.json").write_text(json.dumps(report, indent=2, default=str))

    # ---- console tables ----
    print("\n================ TRADE-LEVEL METRICS (full 2016-2025) ================")
    print(f"{'candidate':32s} {'trades':>6s} {'/yr':>5s} {'wr':>5s} {'rr':>5s} "
          f"{'exp$':>6s} {'pf':>5s} {'IS_exp':>7s} {'OOS_exp':>7s} {'OOS_pf':>6s}")
    for name, d in report.items():
        m = d["metrics"]
        print(f"{name:32s} {m['trades']:>6d} {m['per_yr']:>5.0f} {m['wr']:>5.2f} {str(m['rr']):>5s} "
              f"{m['expectancy']:>6.1f} {str(m['pf']):>5s} {str(m['is_exp']):>7s} "
              f"{str(m['oos_exp']):>7s} {str(m['oos_pf']):>6s}")

    for acct in ("50K", "100K", "150K"):
        print(f"\n================ TOPSTEP {acct} (best funded sizing) ================")
        print(f"{'candidate':32s} {'pass%':>6s} {'fail%':>6s} {'dpass':>6s} {'1stpay':>6s} "
              f"{'Epay$':>7s} {'surv':>5s} {'EV/cyc':>7s} {'cyc_mo':>6s} {'EV/mo$':>7s} {'wmo$':>7s}")
        rows = []
        for name, d in report.items():
            r = d["topstep"].get(acct)
            if not r:
                continue
            rows.append((name, r))
        rows.sort(key=lambda x: -(x[1]["EV_month"] or -1e9))
        for name, r in rows:
            print(f"{name:32s} {r['p_pass']:>6.2f} {str(r['p_fail']):>6s} "
                  f"{str(r['med_days_pass']):>6s} {r['first_payout_prob']:>6.2f} "
                  f"{r['E_payout']:>7.0f} {r['funded_surv']:>5.2f} {r['EV_cycle']:>7.0f} "
                  f"{r['cycle_mo']:>6.1f} {r['EV_month']:>7.0f} {r['worst_month_3c']:>7.0f}")

    # ---- best overall ----
    print("\n================ BEST EV/MONTH PER CANDIDATE (max over accounts) ================")
    best = []
    for name, d in report.items():
        cand_best = None
        for acct, r in d["topstep"].items():
            if cand_best is None or (r["EV_month"] or -1e9) > cand_best[1]["EV_month"]:
                cand_best = (acct, r)
        if cand_best:
            best.append((name, cand_best[0], cand_best[1]["EV_month"], cand_best[1]["fsize"]))
    best.sort(key=lambda x: -(x[2] or -1e9))
    for name, acct, ev, fs in best:
        flag = "  <-- $1k+" if (ev or 0) >= 1000 else ""
        print(f"{name:32s} best={acct:4s} fsize={fs:>2} EV/mo=${ev:>7.0f}{flag}")


if __name__ == "__main__":
    main()
