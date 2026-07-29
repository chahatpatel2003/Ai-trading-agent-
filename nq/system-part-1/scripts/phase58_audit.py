"""Phase 58 — engine/cost/wrapper sensitivity audit (GC, full data control).

Re-runs the best GC families under cost/slippage variations and computes baselines,
isolating whether failure is cost, wrapper, or genuine strategy unprofitability.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import polars as pl

import scripts.phase57_engine as E
from scripts.phase57_engine import (ACCOUNTS, build_days, cushion, load_sessions,
                                    sim_combine, sim_funded, trade_metrics)

SCR = os.environ.get("NQLAB_SCRATCH", os.path.join(tempfile.gettempdir(), "nqlab-phase57"))
PV = 10.0  # MGC


def run_family(fn, comm, slip):
    """Run a family with overridden costs (module globals read by walk())."""
    old_c, old_s = E.COMM_RT, E.SLIP
    E.COMM_RT, E.SLIP = comm, slip
    try:
        tr = fn(load_sessions.__wrapped__() if hasattr(load_sessions, "__wrapped__") else SESS)
    finally:
        E.COMM_RT, E.SLIP = old_c, old_s
    return tr


SESS = load_sessions()


def fam_run(fn, comm, slip):
    old_c, old_s = E.COMM_RT, E.SLIP
    E.COMM_RT, E.SLIP = comm, slip
    try:
        tr = fn(SESS)
    finally:
        E.COMM_RT, E.SLIP = old_c, old_s
    return tr


def expectancy(tr):
    if not tr:
        return None
    n = np.array([t[2] for t in tr])
    w, l = n[n > 0], n[n < 0]
    pf = w.sum() / -l.sum() if l.sum() < 0 else float("inf")
    return n.mean(), pf, n.sum(), len(n)


def wrapper_ev(tr, acct, sims=2500, seed=58):
    """Topstep EV/month for a trade list (best funded sizing)."""
    rng = np.random.default_rng(seed)
    dates = [s["date"] for s in SESS]
    days = build_days(tr, dates)
    cost, T, M, mc = ACCOUNTS[acct]
    cs = rng.integers(0, len(days) - 300, sims)
    cres = [sim_combine(days, int(s), T, M, cushion(T, mc), -0.375 * M, 300) for s in cs]
    npass = sum(1 for r in cres if r[0] == "pass"); nblow = sum(1 for r in cres if r[0] == "blow")
    pp = npass / (npass + nblow) if (npass + nblow) else 0.0
    resd = np.array([r[1] for r in cres if r[0] in ("pass", "blow")])
    e_att = 1 / pp if pp > 0 else 999
    cdays = e_att * (resd.mean() if resd.size else 300)
    fs = rng.integers(0, len(days) - 180, sims)
    best = -1e9
    for fsize in [1, 2, 3, 5, mc]:
        fsize = min(fsize, mc)
        fres = [sim_funded(days, int(s), M, mc, -0.375 * M, M, fsize) for s in fs]
        ep = np.mean([r["wd"] for r in fres]); fdur = np.mean([r["day"] for r in fres])
        ev = (ep - cost * e_att) / ((cdays + fdur) / 21.0) if (cdays + fdur) > 0 else -1e9
        best = max(best, ev)
    return pp, best


# ---------------- baselines ----------------
def baselines():
    feat = pl.read_parquet(f"{SCR}/gc_sessfeat.parquet").filter(~pl.col("roll"))
    ropen = feat["ropen"].to_numpy(); rclose = feat["rclose"].to_numpy()
    rhigh = feat["rhigh"].to_numpy(); rlow = feat["rlow"].to_numpy()
    move = (rclose - ropen) * PV
    cost = E.COMM_RT + 2 * E.SLIP * PV  # 1 RT incl both-side slippage
    print("=== GC BASELINES (RTH pit session, 1 MGC, normal cost) ===")
    print(f"  always-flat              : $0 (control)")
    print(f"  buy-open/sell-close (long every day): exp=${(move-cost).mean():.2f}  "
          f"total=${(move-cost).sum():,.0f}  WR={(move>0).mean():.3f}")
    print(f"  sell-open/buy-close (short every day): exp=${(-move-cost).mean():.2f}  "
          f"total=${(-move-cost).sum():,.0f}")
    rng = np.random.default_rng(58)
    dirs = rng.choice([-1, 1], len(move))
    rr = dirs * move - cost
    print(f"  random long/short        : exp=${rr.mean():.2f}  total=${rr.sum():,.0f}")
    # buy&hold across whole period (no daily flat) — pure beta
    print(f"  buy & hold (2016->2025, 1 MGC): ${(rclose[-1]-ropen[0])*PV:,.0f} (pure gold beta, no Topstep)")
    print(f"  mean |RTH range| = {(rhigh-rlow).mean()*PV:.0f}$/day  (room exists; cost {cost:.2f} is {cost/((rhigh-rlow).mean()*PV)*100:.1f}% of daily range)")


def main():
    print("######## PART A: GC BASELINE SANITY ########")
    baselines()

    print("\n######## PART B: COST/SLIPPAGE SENSITIVITY (raw trade edge, NO wrapper) ########")
    fams = {"7_gap_cont (best completer)": E.fam_gap_cont,
            "11_first_pb (best raw edge)": E.fam_first_pb,
            "1_open_drive (most frequent)": E.fam_open_drive}
    scenarios = [("normal (c1.48,s1tk)", 1.48, 0.1), ("half cost (c0.74,s0.05)", 0.74, 0.05),
                 ("zero comm (c0,s1tk)", 0.0, 0.1), ("zero slip (c1.48,s0)", 1.48, 0.0),
                 ("ZERO ALL (c0,s0)", 0.0, 0.0)]
    print(f"{'family':30s} {'scenario':22s} {'exp$':>7s} {'PF':>5s} {'total$':>9s} {'n':>5s}")
    for fname, fn in fams.items():
        for slab, c, s in scenarios:
            tr = fam_run(fn, c, s)
            e = expectancy(tr)
            if e:
                print(f"{fname:30s} {slab:22s} {e[0]:>7.2f} {e[1]:>5.2f} {e[2]:>9.0f} {e[3]:>5d}")
        print()

    print("######## PART C: NO-WRAPPER vs WRAPPER-ONLY (best GC, 50K) ########")
    print(f"{'family':30s} {'raw exp$/tr':>11s} {'raw PF':>7s} {'wrap pass%':>10s} {'wrap EV/mo$':>11s}")
    for fname, fn in fams.items():
        tr = fam_run(fn, 1.48, 0.1)
        e = expectancy(tr)
        pp, ev = wrapper_ev(tr, "50K")
        print(f"{fname:30s} {e[0]:>11.2f} {e[1]:>7.2f} {pp:>10.2f} {ev:>11.0f}")


if __name__ == "__main__":
    main()
