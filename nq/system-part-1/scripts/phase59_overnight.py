"""Phase 59 — can GC's OVERNIGHT edge be captured inside Topstep rules?

Topstep-legal: flat by 16:10 ET (smin 1330), reopen 18:00 ET (smin 0). In our
session clock the evening reopen and the next-day flatten are the SAME session_date
(18:00 ET -> 17:00 ET), so an overnight hold is modelled with existing arrays.
Full MGC cost; long/short beta split; train/val/test; per-year. Reuses phase57 cost.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

import scripts.phase57_engine as E
from scripts.phase57_engine import (ACCOUNTS, build_days, cushion, load_sessions,
                                    sim_combine, sim_funded)

PV, SLIP, COMM = 10.0, 0.1, 1.48

# smin anchors (ET): reopen 18:00=0, Asia 60-540, London 540-860, COMEX 860,
# pit close 1170, Topstep flatten 16:10 ET = 1330.
REOPEN, ASIA_END, LON_OPEN, COMEX, FLATTEN = 6, 540, 540, 860, 1330
SESS = load_sessions()


def idx_at(s, smv):
    w = np.where(s["smin"] >= smv)[0]
    return int(w[0]) if len(w) else None


def overnight_trade(s, entry_smin, direction, exit_smin, stop_atr=None):
    """Enter next bar after entry_smin, hold to exit_smin (timed) or stop. 1 MGC net+MAE."""
    ei = idx_at(s, entry_smin)
    xi = idx_at(s, exit_smin)
    if ei is None or xi is None or ei + 1 >= len(s["o"]) or xi <= ei:
        return None
    entry = s["o"][ei + 1] + direction * SLIP
    stop_px = (entry - direction * stop_atr * s["atr"]) if stop_atr else None
    mae_px = 0.0
    exit_px = s["c"][xi]
    exit_idx = xi
    for j in range(ei + 1, xi + 1):
        hi, lo = s["h"][j], s["l"][j]
        adv = (entry - lo) if direction > 0 else (hi - entry)
        mae_px = max(mae_px, adv)
        if stop_px is not None:
            if direction > 0 and lo <= stop_px:
                exit_px, exit_idx = stop_px, j
                break
            if direction < 0 and hi >= stop_px:
                exit_px, exit_idx = stop_px, j
                break
        exit_px, exit_idx = s["c"][j], j
    exit_eff = exit_px - direction * SLIP
    net = direction * (exit_eff - entry) * PV - COMM
    mae_d = max(mae_px * PV + SLIP * PV, 0.0)
    return (s["date"], entry_smin, net, mae_d)


# ---------------- strategy variants ----------------
def always(direction, entry_smin, exit_smin, stop_atr=None, filt=None):
    out = []
    for s in SESS:
        if filt and not filt(s):
            continue
        r = overnight_trade(s, entry_smin, direction, exit_smin, stop_atr)
        if r:
            out.append(r)
    return out


def prior_day_dir(s):
    if s["prev"] is None:
        return 0
    ph, pl, pc = s["prev"]
    # prior session close vs its midpoint = prior-day direction
    return 1 if pc > (ph + pl) / 2 else -1


def evening_move_dir(s, by_smin):
    oi = idx_at(s, REOPEN); xi = idx_at(s, by_smin)
    if oi is None or xi is None:
        return 0
    return 1 if s["c"][xi] > s["o"][oi] else -1


def signal_trade(dirfn, entry_smin, exit_smin, stop_atr=None):
    out = []
    for s in SESS:
        d = dirfn(s)
        if d == 0:
            continue
        r = overnight_trade(s, entry_smin, d, exit_smin, stop_atr)
        if r:
            out.append(r)
    return out


def breakout(entry_window_end, exit_smin, stop_atr=1.0):
    """Break of the evening range (reopen..entry_window_end), hold to exit."""
    out = []
    for s in SESS:
        oi = idx_at(s, REOPEN); we = idx_at(s, entry_window_end)
        if oi is None or we is None or we <= oi + 1 or we + 1 >= len(s["o"]):
            continue
        rh = float(s["h"][oi:we].max()); rl = float(s["l"][oi:we].min())
        rng = max(rh - rl, 0.3)
        for j in range(we, idx_at(s, exit_smin) or len(s["o"])):
            if s["c"][j] > rh:
                r = overnight_trade(s, int(s["smin"][j]), 1, exit_smin, stop_atr)
                if r: out.append(r)
                break
            if s["c"][j] < rl:
                r = overnight_trade(s, int(s["smin"][j]), -1, exit_smin, stop_atr)
                if r: out.append(r)
                break
    return out


def meanrev(entry_window_end, exit_smin, stop_atr=1.5):
    """Fade the evening extreme back toward mid."""
    out = []
    for s in SESS:
        oi = idx_at(s, REOPEN); we = idx_at(s, entry_window_end)
        if oi is None or we is None or we <= oi + 1:
            continue
        rh = float(s["h"][oi:we].max()); rl = float(s["l"][oi:we].min())
        mid = (rh + rl) / 2
        for j in range(we, (idx_at(s, exit_smin) or len(s["o"]))):
            if s["c"][j] >= rh:
                r = overnight_trade(s, int(s["smin"][j]), -1, exit_smin, stop_atr)
                if r: out.append(r);
                break
            if s["c"][j] <= rl:
                r = overnight_trade(s, int(s["smin"][j]), 1, exit_smin, stop_atr)
                if r: out.append(r)
                break
    return out


def metrics(tr):
    if not tr or len(tr) < 30:
        return None
    n = np.array([t[2] for t in tr]); yrs = np.array([t[0].year for t in tr])
    w, l = n[n > 0], n[n < 0]
    pf = w.sum() / -l.sum() if l.sum() < 0 else float("inf")
    by = {int(y): round(float(n[yrs == y].sum()), 0) for y in range(2016, 2026) if (yrs == y).any()}
    posy = sum(1 for v in by.values() if v > 0)
    tr_e = n[yrs <= 2020].mean(); va = n[(yrs >= 2021) & (yrs <= 2022)].mean(); te = n[yrs >= 2023].mean()
    # max drawdown of cumulative 1-MGC equity
    eq = np.cumsum(n); dd = float((np.maximum.accumulate(eq) - eq).max())
    return {"n": len(tr), "exp": round(float(n.mean()), 2), "pf": round(pf, 2),
            "total": round(float(n.sum()), 0), "wr": round(float((n > 0).mean()), 3),
            "maxdd": round(dd, 0), "train": round(tr_e, 2), "val": round(va, 2),
            "test": round(te, 2), "posyears": posy, "yearly": by,
            "robust": (tr_e > 0 and va > 0 and te > 0 and posy >= 7)}


def main():
    print("######## GC OVERNIGHT — Topstep-legal (flat by 16:10 ET) ########\n")
    cands = {
        "S1a LONG reopen->COMEX": always(1, REOPEN, COMEX),
        "S1b SHORT reopen->COMEX (beta chk)": always(-1, REOPEN, COMEX),
        "S2a LONG reopen->flatten": always(1, REOPEN, FLATTEN),
        "S2b SHORT reopen->flatten (beta chk)": always(-1, REOPEN, FLATTEN),
        "S1a+stop LONG reopen->COMEX st2": always(1, REOPEN, COMEX, stop_atr=2.0),
        "S3 Asia-trend (eve dir->03:00)": signal_trade(lambda s: evening_move_dir(s, 120), 120, ASIA_END, 2.0),
        "S4 London-trend (ON dir->COMEX)": signal_trade(lambda s: evening_move_dir(s, LON_OPEN), LON_OPEN, COMEX, 2.0),
        "S5 ON breakout (eve range->COMEX)": breakout(60, COMEX, 1.5),
        "S6 ON mean-rev (fade eve->COMEX)": meanrev(120, COMEX, 1.5),
        "S7 prior-day cont reopen->COMEX": signal_trade(prior_day_dir, REOPEN, COMEX, 2.0),
        "S8 LONG Mon-only reopen->COMEX": always(1, REOPEN, COMEX, filt=lambda s: s["date"].weekday() == 0),
    }
    print(f"{'strategy':38s} {'n':>5s} {'exp$':>6s} {'PF':>5s} {'WR':>5s} {'total$':>8s} {'maxDD':>7s} "
          f"{'train':>6s} {'val':>6s} {'test':>6s} {'+yr':>4s} {'ROB':>4s}")
    results = {}
    for lab, tr in cands.items():
        m = metrics(tr)
        if m is None:
            print(f"{lab:38s}  (too few)"); continue
        results[lab] = (tr, m)
        print(f"{lab:38s} {m['n']:>5d} {m['exp']:>6.1f} {m['pf']:>5.2f} {m['wr']:>5.2f} {m['total']:>8.0f} "
              f"{m['maxdd']:>7.0f} {m['train']:>6.1f} {m['val']:>6.1f} {m['test']:>6.1f} {m['posyears']:>4d} "
              f"{'YES' if m['robust'] else '':>4s}")

    print("\n=== YEARLY $ (1 MGC) for the key directional baselines ===")
    for lab in ["S1a LONG reopen->COMEX", "S2a LONG reopen->flatten", "S1a+stop LONG reopen->COMEX st2",
                "S7 prior-day cont reopen->COMEX", "S5 ON breakout (eve range->COMEX)"]:
        if lab in results:
            print(f"  {lab:38s} {results[lab][1]['yearly']}")

    # Topstep EV for the most promising robust/long candidates
    print("\n=== TOPSTEP EV (overnight, best funded sizing) ===")
    print(f"{'strategy':38s} {'acct':5s} {'pass%':>6s} {'EV/mo$':>7s} {'timeout%':>8s} {'worstMo$(3c)':>12s}")
    dates = [s["date"] for s in SESS]
    rng = np.random.default_rng(59)
    pick = [l for l in ["S1a LONG reopen->COMEX", "S2a LONG reopen->flatten",
                        "S1a+stop LONG reopen->COMEX st2", "S7 prior-day cont reopen->COMEX",
                        "S5 ON breakout (eve range->COMEX)"] if l in results]
    for lab in pick:
        tr = results[lab][0]
        days = build_days(tr, dates)
        # worst month at 3 contracts
        bym = defaultdict(float)
        for (d, _sm, net, _m) in tr:
            bym[(d.year, d.month)] += net * 3
        wmo = min(bym.values()) if bym else 0
        for acct in ("50K", "100K", "150K"):
            cost, T, M, mc = ACCOUNTS[acct]
            cs = rng.integers(0, len(days) - 300, 2500)
            cres = [sim_combine(days, int(s), T, M, cushion(T, mc), -0.375 * M, 300) for s in cs]
            npass = sum(1 for r in cres if r[0] == "pass"); nblow = sum(1 for r in cres if r[0] == "blow")
            ntime = sum(1 for r in cres if r[0] == "timeout")
            pp = npass / (npass + nblow) if (npass + nblow) else 0.0
            resd = np.array([r[1] for r in cres if r[0] in ("pass", "blow")])
            e_att = 1 / pp if pp > 0 else 999
            cdays = e_att * (resd.mean() if resd.size else 300)
            fs = rng.integers(0, len(days) - 180, 2500)
            best = -1e9
            for fsize in [1, 2, 3, 5, mc]:
                fsize = min(fsize, mc)
                fres = [sim_funded(days, int(s2), M, mc, -0.375 * M, M, fsize) for s2 in fs]
                ep = np.mean([r["wd"] for r in fres]); fdur = np.mean([r["day"] for r in fres])
                ev = (ep - cost * e_att) / ((cdays + fdur) / 21.0) if (cdays + fdur) > 0 else -1e9
                best = max(best, ev)
            print(f"{lab:38s} {acct:5s} {pp:>6.2f} {best:>7.0f} {ntime/2500:>7.0%} {wmo:>12.0f}")


if __name__ == "__main__":
    main()
