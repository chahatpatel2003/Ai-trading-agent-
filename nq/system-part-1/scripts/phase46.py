"""Phase 46 — Multi-account Topstep 150K payout factory (calendar-shared portfolio sim).

Correlation is modeled by construction: all accounts run over the SAME calendar dates, so same-strategy
accounts share that day's PnL (correlated); strategy-mix and staggered starts decorrelate. Each account
is a lifecycle (combine -> funded -> blow -> restart) on a 150K (target $9k, MLL $4.5k, 15ct). MONTHLY
combine cost + activation + reset modeled. Frozen signals only (clean5+gate harsh, Phase-36 idb+swd).
Output: per (account_count x allocation) portfolio monthly cost / payout / EV / tail risk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.phase42 import A_SLEEVES, CLEAN5, build_days
from scripts.phase44 import ACCOUNTS, cushion, flat, merge, p36_days

OUT = Path("docs/phase46")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(46)
T, M, MC = 9000.0, 4500.0, 15        # 150K
WIN_THR, MIN_WIN = 150.0, 5
FEES = {"monthly": 150.0, "activation": 150.0, "reset": 150.0}
SIZE_C = cushion(T, MC)              # combine cushion sizing
SIZE_F = flat(1, MC)                 # funded 1 MNQ
PAYOUT_CAP = M
LSTOP = -0.375 * M


def new_state():
    return {"phase": "combine", "eq": 0.0, "peak": 0.0, "best": 0.0, "cdays": 0,
            "win": 0, "locked": False}


def step(st, trades):
    """Advance one calendar day. Returns (cost, payout). Mutates st."""
    cost = payout = 0.0
    if st["phase"] == "combine":
        cost += FEES["monthly"] / 21.0
        day = 0.0
        traded = False
        for (_et, net1, mae1, _q, _sl) in trades:
            if day <= LSTOP:
                break
            bal = st["eq"] + day
            s = SIZE_C(bal)
            if bal - mae1 * s <= min(max(st["peak"], bal) - M, 0.0):
                cost += FEES["reset"]
                st.update(eq=0.0, peak=0.0, best=0.0, cdays=0)
                return cost, payout
            day += net1 * s
            traded = True
            bal = st["eq"] + day
            st["peak"] = max(st["peak"], bal)
            if bal <= min(st["peak"] - M, 0.0):
                cost += FEES["reset"]
                st.update(eq=0.0, peak=0.0, best=0.0, cdays=0)
                return cost, payout
        st["eq"] += day
        if traded:
            st["cdays"] += 1
        st["best"] = max(st["best"], day)
        if st["eq"] >= T and st["cdays"] >= 2 and st["best"] <= 0.5 * st["eq"]:
            cost += FEES["activation"]
            st.update(phase="funded", eq=0.0, peak=0.0, best=0.0, win=0, locked=False)
    else:  # funded
        day = 0.0
        for (_et, net1, mae1, _q, _sl) in trades:
            if day <= LSTOP:
                break
            bal = st["eq"] + day
            floor = 0.0 if st["locked"] else min(st["peak"] - M, 0.0)
            if bal - mae1 * 1 <= floor:
                cost += FEES["reset"]
                st.update(phase="combine", eq=0.0, peak=0.0, best=0.0, cdays=0)
                return cost, payout
            day += net1 * 1
            bal = st["eq"] + day
            st["peak"] = max(st["peak"], bal)
            if st["peak"] >= M:
                st["locked"] = True
            if bal <= (0.0 if st["locked"] else min(st["peak"] - M, 0.0)):
                cost += FEES["reset"]
                st.update(phase="combine", eq=0.0, peak=0.0, best=0.0, cdays=0)
                return cost, payout
        st["eq"] += day
        st["best"] = max(st["best"], day)
        if day >= WIN_THR:
            st["win"] += 1
        if st["win"] >= MIN_WIN and st["eq"] > 0 and st["best"] <= 0.5 * st["eq"]:
            amt = min(st["eq"], PAYOUT_CAP)
            payout += amt
            st["eq"] -= amt
            st["win"] = 0
            st["best"] = 0.0
    return cost, payout


def run_portfolio(accounts, strat_days, sims=2000, win=252, warm=63):
    """accounts: list of (strategy_name, start_offset). Returns monthly metrics over steady months."""
    n_accts = len(accounts)
    maxoff = max(o for _s, o in accounts)
    n = len(next(iter(strat_days.values())))
    months_net, months_pay, months_cost = [], [], []
    pay_any_month, all_blow_month = 0, 0
    n_months_total = 0
    active_funded, payouts_6mo = [], []
    worst_month = 0.0
    for _ in range(sims):
        t0 = int(RNG.integers(0, n - win - maxoff - 1))
        sts = [new_state() for _ in accounts]
        # monthly accumulators
        m_cost = np.zeros(win // 21)
        m_pay = np.zeros(win // 21)
        m_blowcount = np.zeros(win // 21)
        funded_days = 0
        pay_6 = 0.0
        for d in range(win):
            mi = d // 21
            for k, (sname, off) in enumerate(accounts):
                if d < off:
                    continue
                trades = strat_days[sname][t0 + d]
                ph0 = sts[k]["phase"]
                c, p = step(sts[k], trades)
                m_cost[mi] += c
                m_pay[mi] += p
                if sts[k]["phase"] == "funded":
                    funded_days += 1
                if ph0 == "funded" and sts[k]["phase"] == "combine":
                    m_blowcount[mi] += 1
                if d < 126:
                    pay_6 += p
        steady = range(warm // 21, win // 21)  # steady-state months (after warm-up)
        for mi in steady:
            net = m_pay[mi] - m_cost[mi]
            months_net.append(net)
            months_pay.append(m_pay[mi])
            months_cost.append(m_cost[mi])
            n_months_total += 1
            if m_pay[mi] > 0:
                pay_any_month += 1
            if m_blowcount[mi] >= n_accts:
                all_blow_month += 1
            worst_month = min(worst_month, net)
        active_funded.append(funded_days / win)
        payouts_6mo.append(pay_6)
    mn = np.array(months_net)
    return {
        "accounts": n_accts,
        "monthly_cost": round(float(np.mean(months_cost)), 0),
        "monthly_payout": round(float(np.mean(months_pay)), 0),
        "net_EV_month": round(float(mn.mean()), 0),
        "payout_prob_month": round(pay_any_month / n_months_total, 3),
        "worst_month": round(worst_month, 0),
        "p5_month": round(float(np.quantile(mn, 0.05)), 0),
        "P_all_blow_in_month": round(all_blow_month / n_months_total, 4),
        "avg_active_funded": round(float(np.mean(active_funded)) * n_accts, 2),
        "payouts_per_6mo": round(float(np.mean(payouts_6mo)) / PAYOUT_CAP, 2),
        "EV_per_acct": round(float(mn.mean()) / n_accts, 0),
    }


def make_accounts(count, scheme):
    gap = 21  # ~1 month stagger
    if scheme == "synced_combo":
        return [("combo", 0) for _ in range(count)]
    if scheme == "staggered_combo":
        return [("combo", (i * gap) % 84) for i in range(count)]
    if scheme == "mixed_staggered":  # alternate combo / phase36
        mix = ["combo", "phase36"]
        return [(mix[i % 2], (i * gap) % 84) for i in range(count)]
    if scheme == "grinder_vs_infreq":  # clean5 grinders + phase36 infrequent
        return [(("clean5" if i % 2 == 0 else "phase36"), (i * gap) % 84) for i in range(count)]
    if scheme == "diversified":  # clean5 / phase36 / A_only mix, staggered
        pool = ["clean5", "phase36", "aonly"]
        return [(pool[i % 3], (i * gap) % 84) for i in range(count)]
    raise ValueError(scheme)


def main():
    c5 = build_days(CLEAN5, "frozen")
    ao = build_days(list(A_SLEEVES), "frozen")
    p36 = p36_days()
    combo = (c5[0], merge(c5[1], p36[1]), c5[2])
    strat_days = {"clean5": c5[1], "phase36": p36[1], "combo": combo[1], "aonly": ao[1]}

    schemes = ["synced_combo", "staggered_combo", "mixed_staggered", "grinder_vs_infreq", "diversified"]
    counts = [1, 2, 3, 5, 10]
    cols = ["accounts", "monthly_cost", "monthly_payout", "net_EV_month", "EV_per_acct",
            "payout_prob_month", "worst_month", "p5_month", "P_all_blow_in_month",
            "avg_active_funded", "payouts_per_6mo"]
    results = {}
    for scheme in schemes:
        print(f"\n===== {scheme} =====")
        print(f"{'N':>3s} " + " ".join(f"{c[:10]:>10s}" for c in cols[1:]))
        for cnt in counts:
            if scheme == "synced_combo" and cnt == 1:
                pass
            accts = make_accounts(cnt, scheme)
            r = run_portfolio(accts, strat_days, sims=1500)
            results[f"{scheme}|{cnt}"] = r
            print(f"{cnt:>3d} " + " ".join(f"{str(r[c]):>10s}" for c in cols[1:]))
    (OUT / "_phase46.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
