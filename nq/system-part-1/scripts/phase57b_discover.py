"""Phase 57b — unsupervised day-type clustering + rigorous trigger mining.

Clusters PIT-clean RTH paths (numpy k-means) to reveal recurring day-types and
tests whether early-session behavior predicts them. Then mines interpretable
opening-window triggers and validates each with full MGC costs under a strict
train/val/test + per-year regime split. Reuses phase57_engine costs (walk()).
"""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict

import numpy as np

from scripts.phase57_engine import load_sessions, walk, ACCOUNTS, build_days, cushion, sim_combine, sim_funded

SCR = os.environ.get("NQLAB_SCRATCH", os.path.join(tempfile.gettempdir(), "nqlab-phase57"))

NY_OPEN, RTH_END = 860, 1170
FLAT_PIT = 1170          # exit at pit close (13:30 ET) for day-level holds


# ---------------- numpy k-means ----------------
def kmeans(X, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    cen = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - cen[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        new = np.array([X[lab == j].mean(0) if (lab == j).any() else cen[j] for j in range(k)])
        if np.allclose(new, cen):
            break
        cen = new
    return lab, cen


def cluster_daytypes():
    z = np.load(f"{SCR}/gc_rthpaths.npz", allow_pickle=True)
    P, dates, atr = z["paths"], z["dates"], z["atr"]
    yrs = np.array([int(s[:4]) for s in dates])
    # normalize each path by its session ATR -> shape only
    Pn = P / atr[:, None]
    lab, cen = kmeans(Pn, 5, seed=7)
    print("=== DAY-TYPE CLUSTERS (normalized RTH path, k=5) ===")
    print(f"{'type':5s} {'freq':>6s} {'endRTH(xATR)':>12s} {'maxUp':>7s} {'maxDn':>7s}  shape")
    order = np.argsort([cen[j][-1] for j in range(5)])
    names = {}
    for rank, j in enumerate(order):
        end = cen[j][-1]
        mu, md = cen[j].max(), cen[j].min()
        freq = (lab == j).mean()
        # crude shape label
        if end > 0.4:
            nm = "TrendUp"
        elif end < -0.4:
            nm = "TrendDn"
        elif mu > 0.5 and end < 0.2:
            nm = "FadeUp(invV)"
        elif md < -0.5 and end > -0.2:
            nm = "FadeDn(V)"
        else:
            nm = "Range"
        names[j] = nm
        print(f"{nm:12s} {freq:>6.1%} {end:>9.2f} {mu:>7.2f} {md:>7.2f}")
    # per-year stability of trend vs range share
    print("\n  per-year share of |end|>0.4 (trending) vs range:")
    for y in range(2016, 2026):
        m = yrs == y
        trend = np.mean(np.abs(Pn[m][:, -1]) > 0.4)
        print(f"    {y}: trend-day share {trend:.0%}", end="  ")
        if y % 5 == 0:
            print()
    print()
    # predictiveness: does first-30min path (first 6 steps) predict end sign?
    early = Pn[:, :7].mean(1)   # avg of first ~30min normalized displacement
    endsign = np.sign(Pn[:, -1])
    agree = np.mean(np.sign(early) == endsign)
    print(f"\n  P(sign(first30m disp) == sign(RTH end)) = {agree:.3f}")
    # conditional magnitude
    for thr in [0.1, 0.2, 0.3]:
        strong = np.abs(early) > thr
        if strong.sum() > 100:
            cont = np.mean(np.sign(early[strong]) == endsign[strong])
            print(f"    |first30m disp|>{thr}xATR (n={strong.sum()}): continuation={cont:.3f}")
    return dates


# ---------------- trigger validation ----------------
def validate(trades, label):
    """trades: list of (date, smin, net, mae). Strict split + per-year."""
    if len(trades) < 100:
        return None
    nets = np.array([t[2] for t in trades])
    yrs = np.array([t[0].year for t in trades])
    def exp_of(mask):
        return float(nets[mask].mean()) if mask.any() else None
    tr = exp_of(yrs <= 2020)            # train 2016-2020
    va = exp_of((yrs >= 2021) & (yrs <= 2022))  # val 2021-2022
    te = exp_of(yrs >= 2023)            # test 2023-2025
    yr_exp = {y: float(nets[yrs == y].mean()) for y in range(2016, 2026) if (yrs == y).any()}
    pos_years = sum(1 for v in yr_exp.values() if v > 0)
    survive = (tr is not None and va is not None and te is not None
               and tr > 0 and va > 0 and te > 0 and pos_years >= 7 and len(trades) >= 150)
    return {
        "label": label, "n": len(trades), "per_yr": round(len(trades) / 10, 1),
        "exp": round(float(nets.mean()), 2),
        "train": round(tr, 2) if tr is not None else None,
        "val": round(va, 2) if va is not None else None,
        "test": round(te, 2) if te is not None else None,
        "pos_years": pos_years, "survive": survive,
        "yr_exp": {y: round(v, 1) for y, v in yr_exp.items()},
    }


def idx_at(s, smv):
    w = np.where(s["smin"] >= smv)[0]
    return int(w[0]) if len(w) else None


# ---- trigger generators (day-level, anchored to COMEX open structure) ----
def trig_open_cont(sess, enter_smin, dirsrc, atr_stop, hold_to=FLAT_PIT, rmult=None,
                   min_disp_atr=0.0, max_disp_atr=1e9):
    """Enter at `enter_smin` in the direction set by opening structure; stop = atr_stop
    fraction of OR; exit at pit close (or rmult target). Gate on opening displacement
    magnitude (|close-open| at enter, in ATR units)."""
    out = []
    for s in sess:
        oi = idx_at(s, NY_OPEN)
        ei = idx_at(s, enter_smin)
        if oi is None or ei is None or ei + 1 >= len(s["o"]):
            continue
        ropen = s["o"][oi]
        # opening range up to enter
        m = (s["smin"] >= NY_OPEN) & (s["smin"] < enter_smin)
        if m.sum() < 2:
            continue
        orh, orl = float(s["h"][m].max()), float(s["l"][m].min())
        cur = s["c"][ei]
        disp = abs(cur - ropen) / s["atr"]
        if disp < min_disp_atr or disp > max_disp_atr:
            continue
        d = dirsrc(ropen, orh, orl, cur)
        if d == 0:
            continue
        rng = max(orh - orl, 0.3)
        if d > 0:
            stop = cur - atr_stop * rng
            tgt = cur + rmult * atr_stop * rng if rmult else 1e9
        else:
            stop = cur + atr_stop * rng
            tgt = cur - rmult * atr_stop * rng if rmult else -1e9
        r = walk(s, ei, d, stop, tgt, flat_smin=hold_to, time_stop=200)
        if r:
            out.append((s["date"], enter_smin, r[0], r[1]))
    return out


def mfe_mae_profile(sess, enter_smin, min_disp_atr):
    """Post-entry MFE/MAE ($/contract) for strong-open continuation entries, to pit close."""
    mfes, maes, finals = [], [], []
    for s in sess:
        oi = idx_at(s, NY_OPEN); ei = idx_at(s, enter_smin)
        if oi is None or ei is None or ei + 1 >= len(s["o"]):
            continue
        ropen = s["o"][oi]; cur = s["c"][ei]
        if abs(cur - ropen) / s["atr"] < min_disp_atr:
            continue
        d = 1 if cur > ropen else -1
        fwd = (s["smin"] >= enter_smin) & (s["smin"] <= FLAT_PIT)
        hi, lo, last = s["h"][fwd].max(), s["l"][fwd].min(), s["c"][fwd][-1]
        if d > 0:
            mfes.append((hi - cur) * 10); maes.append((cur - lo) * 10); finals.append((last - cur) * 10)
        else:
            mfes.append((cur - lo) * 10); maes.append((hi - cur) * 10); finals.append((cur - last) * 10)
    mfes, maes, finals = np.array(mfes), np.array(maes), np.array(finals)
    print(f"\n=== MFE/MAE PROFILE: strong-open (disp>{min_disp_atr}ATR) cont entries, n={len(mfes)} ===")
    print(f"  MFE$  p50/p75/p90 = {np.median(mfes):.0f}/{np.percentile(mfes,75):.0f}/{np.percentile(mfes,90):.0f}")
    print(f"  MAE$  p50/p75/p90 = {np.median(maes):.0f}/{np.percentile(maes,75):.0f}/{np.percentile(maes,90):.0f}")
    print(f"  final$ mean/median = {finals.mean():.0f}/{np.median(finals):.0f}  (pre-cost, 1 contract)")
    print(f"  P(final>0)={np.mean(finals>0):.2f}  P(MFE>2*MAE_median)={np.mean(mfes>2*np.median(maes)):.2f}")


def main():
    cluster_daytypes()
    sess = load_sessions()
    print("\n=== TRIGGER MINING (full MGC cost; strict train/val/test) ===")

    cands = {}
    dirc = lambda ro, oh, ol, cur: (1 if cur > ro else -1)   # direction = sign of displacement
    fadec = lambda ro, oh, ol, cur: (-1 if cur > ro else 1)

    # MFE/MAE profile for strong-open continuation entries (set exits empirically)
    mfe_mae_profile(sess, 900, 0.3)

    # Conditional continuation: only STRONG opening displacement (the discovered edge)
    cands["C1 cont 09:00 disp>0.2 holdpit"] = trig_open_cont(sess, 900, dirc, 1.0, min_disp_atr=0.2)
    cands["C2 cont 09:00 disp>0.3 holdpit"] = trig_open_cont(sess, 900, dirc, 1.0, min_disp_atr=0.3)
    cands["C3 cont 08:45 disp>0.2 holdpit"] = trig_open_cont(sess, 885, dirc, 1.0, min_disp_atr=0.2)
    cands["C4 cont 09:00 disp>0.3 2R"] = trig_open_cont(sess, 900, dirc, 1.0, rmult=2.0, min_disp_atr=0.3)
    cands["C5 cont 09:00 disp>0.3 3R"] = trig_open_cont(sess, 900, dirc, 1.0, rmult=3.0, min_disp_atr=0.3)
    cands["C6 cont 09:00 disp>0.4 holdpit"] = trig_open_cont(sess, 900, dirc, 1.0, min_disp_atr=0.4)
    cands["C7 cont 09:00 disp>0.3 stop0.5"] = trig_open_cont(sess, 900, dirc, 0.5, min_disp_atr=0.3)
    # Counterpart: FADE on QUIET opens (range-day reversion), small displacement only
    cands["C8 FADE 09:00 disp<0.15 holdpit"] = trig_open_cont(sess, 900, fadec, 1.0, max_disp_atr=0.15)

    survivors = []
    print(f"{'trigger':30s} {'n':>5s} {'/yr':>5s} {'exp$':>6s} {'train':>6s} {'val':>6s} {'test':>6s} {'+yrs':>4s} {'OK':>3s}")
    for lab, tr in cands.items():
        v = validate(tr, lab)
        if v is None:
            print(f"{lab:30s}  (too few: {len(tr)})")
            continue
        ok = "YES" if v["survive"] else ""
        print(f"{lab:30s} {v['n']:>5d} {v['per_yr']:>5.0f} {v['exp']:>6.1f} "
              f"{str(v['train']):>6s} {str(v['val']):>6s} {str(v['test']):>6s} {v['pos_years']:>4d} {ok:>3s}")
        if v["survive"]:
            survivors.append((lab, tr, v))

    print(f"\nSURVIVORS: {[s[0] for s in survivors]}")
    # Topstep EV for survivors
    if survivors:
        dates = [s["date"] for s in sess]
        rng = np.random.default_rng(571)
        print("\n=== SURVIVOR TOPSTEP EV ===")
        print(f"{'trigger':30s} {'acct':5s} {'pass%':>6s} {'EV/mo$':>7s} {'timeout%':>8s}")
        for lab, tr, v in survivors:
            days = build_days(tr, dates)
            for acct in ("50K", "100K", "150K"):
                cost, T, M, mc = ACCOUNTS[acct]
                cs = rng.integers(0, len(days) - 300, 2000)
                cres = [sim_combine(days, int(s), T, M, cushion(T, mc), -0.375 * M, 300) for s in cs]
                npass = sum(1 for r in cres if r[0] == "pass")
                nblow = sum(1 for r in cres if r[0] == "blow")
                ntime = sum(1 for r in cres if r[0] == "timeout")
                pp = npass / (npass + nblow) if (npass + nblow) else 0.0
                resd = np.array([r[1] for r in cres if r[0] in ("pass", "blow")])
                e_att = 1 / pp if pp > 0 else 999
                cdays = e_att * (resd.mean() if resd.size else 300)
                fs = rng.integers(0, len(days) - 180, 2000)
                best_ev = -1e9
                for fsize in [1, 2, 3, 5, mc]:
                    fsize = min(fsize, mc)
                    fres = [sim_funded(days, int(s2), M, mc, -0.375 * M, M, fsize) for s2 in fs]
                    ep = np.mean([r["wd"] for r in fres]); fdur = np.mean([r["day"] for r in fres])
                    ev = (ep - cost * e_att) / ((cdays + fdur) / 21.0) if (cdays + fdur) > 0 else -1e9
                    best_ev = max(best_ev, ev)
                print(f"{lab:30s} {acct:5s} {pp:>6.2f} {best_ev:>7.0f} {ntime/2000:>7.0%}")


if __name__ == "__main__":
    main()
