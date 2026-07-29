"""Phase 42 — Topstep 150K Pass + Payout lifecycle simulator (frozen signals; Topstep wrapper only).

Optimizes PASS probability + FUNDED PAYOUT probability (not net ROI). Combine Pass Mode and Funded
Payout Mode are separate. Base signal engine = FROZEN clean5+gate (Phase 40); boosters drawn from the
Phase-38 sleeve library if needed. NOTHING about the underlying signals is changed. Harsh fills.

Run: python -m scripts.phase42 combine | funded | booster | lifecycle
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from nqlab.config.loader import load_config
from nqlab.discovery import phase29 as P29

OUT = Path("docs/phase42")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)
A_SLEEVES = {"ir90_break", "trend_adx_cont", "ir60_fade"}
CLEAN5 = ["trend_adx_cont", "ir90_break", "asia_sweep_fade", "ir60_fade", "disp_pb_fvg"]
TARGET, MLL = 9000.0, 4500.0


def _ts2mod() -> dict:
    cfg = load_config(profile="dev", overrides=P29.PHASE29_OVERRIDES)
    fr = pl.read_parquet(cfg.data.artifacts_root / "phase30" / "frame_5m_full_vp.parquet")
    sub = fr.select(pl.col("ts").dt.strftime("%Y-%m-%dT%H:%M:%S").alias("s"), "minute_of_day")
    return dict(zip(sub["s"].to_list(), sub["minute_of_day"].to_list()))


def build_days(sleeves: list[str], source: str = "frozen") -> tuple[list, list, list]:
    """Return (ordered all_days, per-day trade lists, per-day regime). source: frozen|phase38."""
    reg = json.loads((Path("docs/phase39") / "day_regime.json").read_text())
    all_days = sorted(reg.keys())
    ts2mod = _ts2mod()
    if source == "frozen":
        d = json.loads((Path("docs/phase40") / "_frozen.json").read_text())["BALANCED_harsh"]
        rows = [(t["sleeve"], t["day"], t["entry_ts"], float(t["net"]), float(t["mae"])) for t in d]
    else:  # phase38 library (1 MNQ, ungated/uncapped) -> apply gate(skip neutral)+cap3 here
        st = json.loads((Path("docs/phase38") / "_trades.json").read_text())
        rows = []
        for nm in sleeves:
            for t in st.get(nm, []):
                rows.append((nm, t["day"], t["ts"], float(t["net"]), float(t["mae"])))
    by_day = defaultdict(list)
    for sl, day, ts, net, mae in rows:
        if sl not in sleeves:
            continue
        if reg.get(day, {}).get("trend") == "neutral":  # frozen gate
            continue
        et = ts2mod.get(ts[:19], 999)
        by_day[day].append((ts, et, net, mae, "A" if sl in A_SLEEVES else "B", sl))
    days, regimes = [], []
    for day in all_days:
        ts_list = sorted(by_day.get(day, []), key=lambda x: x[0])
        if source == "phase38":
            ts_list = ts_list[:3]  # cap 3/day
        days.append([(et, net, mae, q, sl) for (_ts, et, net, mae, q, sl) in ts_list])
        regimes.append(reg.get(day, {}))
    return all_days, days, regimes


def size_fn(name: str):
    f = {
        "flat1": lambda b: 1, "flat2": lambda b: 2, "flat3": lambda b: 3, "flat4": lambda b: 4,
        "flat5": lambda b: 5,
        "cushion234": lambda b: 2 if b < 2000 else (3 if b < 4000 else 4),
        "cushion123": lambda b: 1 if b < 2000 else (2 if b < 4000 else 3),
        "cushion1234": lambda b: 1 if b < 1500 else (2 if b < 3500 else (3 if b < 6000 else 4)),
        "cushion124": lambda b: 1 if b < 2500 else (2 if b < 5000 else 4),
        "mixed_nq": lambda b: 3 if b < 3000 else (6 if b < 6000 else 10),
    }
    return f[name]


def day_ok(rg: dict, regime_filter: str | None) -> bool:
    if regime_filter is None:
        return True
    if regime_filter == "trend_day":
        return rg.get("trend_str") == "adx_hi" or rg.get("vol") == "vol_hi"
    if regime_filter == "skip_weak":
        return not (rg.get("vol") == "vol_lo" or rg.get("chop"))
    return True


def sim(days, regimes, start, horizon, *, size, allowed, time_filter, regime_filter,
        pstop, lstop, target=TARGET, mll=MLL):
    sz = size_fn(size)
    equity = peak = best_day = maxdd = worst_day = 0.0
    active = 0
    consist_block = False
    for di in range(horizon):
        day = days[start + di]
        rg = regimes[start + di]
        if not day_ok(rg, regime_filter):
            best_day = max(best_day, 0.0)
            continue
        day_pnl = 0.0
        traded = False
        for (et, net1, mae1, q, sl) in day:
            if lstop is not None and day_pnl <= lstop:
                break
            if pstop is not None and day_pnl >= pstop:
                break
            if allowed is not None and sl not in allowed:
                continue
            if time_filter == "morning" and et >= 720:  # before 12:00 ET
                continue
            bal = equity + day_pnl
            s = sz(bal)
            pk = max(peak, bal)
            trough = bal - mae1 * s
            maxdd = max(maxdd, pk - trough)
            if trough <= min(pk - mll, 0.0):
                return {"r": "fail", "day": di + 1, "maxdd": maxdd, "worst_day": min(worst_day, day_pnl),
                        "best_day": best_day, "consist_block": consist_block}
            day_pnl += net1 * s
            traded = True
            bal = equity + day_pnl
            peak = max(peak, bal)
            if bal <= min(peak - mll, 0.0):
                return {"r": "fail", "day": di + 1, "maxdd": max(maxdd, peak - bal),
                        "worst_day": min(worst_day, day_pnl), "best_day": best_day,
                        "consist_block": consist_block}
        equity += day_pnl
        if traded:
            active += 1
        best_day = max(best_day, day_pnl)
        worst_day = min(worst_day, day_pnl)
        maxdd = max(maxdd, peak - equity)
        if equity >= target and active >= 2:
            if best_day <= 0.5 * equity:
                return {"r": "pass", "day": di + 1, "maxdd": maxdd, "worst_day": worst_day,
                        "best_day": best_day, "consist_block": consist_block}
            consist_block = True
    return {"r": "timeout", "day": horizon, "maxdd": maxdd, "worst_day": worst_day,
            "best_day": best_day, "consist_block": consist_block}


def agg(res, sims):
    P = [r for r in res if r["r"] == "pass"]
    F = [r for r in res if r["r"] == "fail"]
    pd_ = np.array([r["day"] for r in P]) if P else np.array([])

    def cum(lst, d):
        return sum(1 for r in lst if r["day"] <= d) / sims

    def alive(d):  # still in progress (not passed, not failed) by day d
        return round(1 - cum(P, d) - cum(F, d), 3)
    return {
        "pass30": round(cum(P, 30), 3), "pass45": round(cum(P, 45), 3),
        "pass60": round(cum(P, 60), 3), "pass90": round(cum(P, 90), 3),
        "fail90": round(cum(F, 90), 3),
        "ratio90": round(cum(P, 90) / cum(F, 90), 2) if cum(F, 90) > 0 else None,
        "avg_days": round(float(pd_.mean()), 1) if pd_.size else None,
        "med_days": round(float(np.median(pd_)), 1) if pd_.size else None,
        "worst_dd": round(max(r["maxdd"] for r in res), 0),
        "worst_day": round(min(r["worst_day"] for r in res), 0),
        "best_day": round(max(r["best_day"] for r in res), 0),
        "consist_viol": round(sum(1 for r in res if r["consist_block"]) / sims, 3),
        "alive30": alive(30), "alive60": alive(60), "alive90": alive(90),
    }


def run_combine():
    all_days, days, regimes = build_days(CLEAN5, "frozen")
    n = len(days)
    SIMS, HOR = 20000, 90
    starts = RNG.integers(0, n - HOR, SIMS)

    def go(**kw):
        base = dict(size="flat2", allowed=None, time_filter=None, regime_filter=None,
                    pstop=None, lstop=None)
        base.update(kw)
        res = [sim(days, regimes, int(s), HOR, **base) for s in starts]
        return agg(res, SIMS)

    variants = {}
    for sz in ("flat1", "flat2", "flat3", "flat4", "flat5", "cushion123", "cushion234", "mixed_nq"):
        variants[f"size:{sz}"] = go(size=sz)
    for f, kw in [("A_only", dict(allowed=A_SLEEVES)),
                  ("best_sleeve(trend_adx)", dict(allowed={"trend_adx_cont"})),
                  ("best_sleeve(ir90)", dict(allowed={"ir90_break"})),
                  ("top2(trend+ir90)", dict(allowed={"trend_adx_cont", "ir90_break"})),
                  ("morning_only", dict(time_filter="morning")),
                  ("trend_day_only", dict(regime_filter="trend_day")),
                  ("skip_weak", dict(regime_filter="skip_weak"))]:
        variants[f"filter:{f}@flat2"] = go(**kw)
    for ps in (500, 750, 1000):
        variants[f"pstop:+{ps}@flat2"] = go(pstop=float(ps))
    for ls in (500, 750, 1000, 1250):
        variants[f"lstop:-{ls}@flat2"] = go(lstop=float(-ls))
    # promising combos
    variants["COMBO flat2+Aonly+lstop750"] = go(size="flat2", allowed=A_SLEEVES, lstop=-750.0)
    variants["COMBO flat3+Aonly+lstop1000"] = go(size="flat3", allowed=A_SLEEVES, lstop=-1000.0)
    variants["COMBO cushion234+Aonly+lstop1000+pstop1000"] = go(
        size="cushion234", allowed=A_SLEEVES, lstop=-1000.0, pstop=1000.0)
    variants["COMBO flat2+top2+lstop750"] = go(size="flat2", allowed={"trend_adx_cont", "ir90_break"},
                                               lstop=-750.0)

    (OUT / "_combine.json").write_text(json.dumps(variants, indent=2))
    cols = ["pass30", "pass60", "pass90", "fail90", "ratio90", "med_days", "worst_dd", "worst_day",
            "best_day", "consist_viol", "alive90"]
    print(f"{'variant':42s} " + " ".join(f"{c:>9s}" for c in cols))
    for k, v in variants.items():
        print(f"{k:42s} " + " ".join(f"{str(v[c]):>9s}" for c in cols))


def run_refine():
    all_days, days, regimes = build_days(CLEAN5, "frozen")
    n = len(days)
    SIMS, HOR = 20000, 90
    starts = RNG.integers(0, n - HOR, SIMS)

    def go(**kw):
        base = dict(size="cushion123", allowed=None, time_filter=None, regime_filter=None,
                    pstop=None, lstop=None)
        base.update(kw)
        return agg([sim(days, regimes, int(s), HOR, **base) for s in starts], SIMS)

    v = {
        "cushion123 (base)": go(),
        "cushion1234": go(size="cushion1234"),
        "cushion124": go(size="cushion124"),
        "cushion123+Aonly": go(allowed=A_SLEEVES),
        "cushion123+lstop750": go(lstop=-750.0),
        "cushion123+lstop1000": go(lstop=-1000.0),
        "cushion123+skip_weak": go(regime_filter="skip_weak"),
        "cushion123+Aonly+lstop1000": go(allowed=A_SLEEVES, lstop=-1000.0),
        "cushion1234+lstop1000": go(size="cushion1234", lstop=-1000.0),
    }
    cols = ["pass30", "pass60", "pass90", "fail90", "ratio90", "med_days", "worst_dd", "worst_day",
            "best_day", "consist_viol", "alive90"]
    print(f"{'variant':32s} " + " ".join(f"{c:>9s}" for c in cols))
    for k, val in v.items():
        print(f"{k:32s} " + " ".join(f"{str(val[c]):>9s}" for c in cols))


def sim_funded(days, regimes, start, horizon, *, size, allowed, lstop, win_thr=150.0,
               payout_cap=2500.0, min_win=5, mll=MLL):
    sz = size_fn(size)
    equity = peak = best_day = maxdd = 0.0
    win_days = 0
    withdrawn = 0.0
    payouts = 0
    first_payout = None
    locked = False  # once peak>=mll, floor locks at 0 (start)
    for di in range(horizon):
        day = days[start + di]
        day_pnl = 0.0
        for (et, net1, mae1, q, sl) in day:
            if lstop is not None and day_pnl <= lstop:
                break
            if allowed is not None and sl not in allowed:
                continue
            bal = equity + day_pnl
            s = sz(bal)
            floor = 0.0 if locked else min(peak - mll, 0.0)
            if bal - mae1 * s <= floor:
                return {"survived": False, "day": di + 1, "withdrawn": withdrawn, "payouts": payouts,
                        "first_payout": first_payout, "maxdd": max(maxdd, peak - (bal - mae1 * s))}
            day_pnl += net1 * s
            bal = equity + day_pnl
            peak = max(peak, bal)
            if peak >= mll:
                locked = True
            floor = 0.0 if locked else min(peak - mll, 0.0)
            if bal <= floor:
                return {"survived": False, "day": di + 1, "withdrawn": withdrawn, "payouts": payouts,
                        "first_payout": first_payout, "maxdd": max(maxdd, peak - bal)}
        equity += day_pnl
        best_day = max(best_day, day_pnl)
        maxdd = max(maxdd, peak - equity)
        if day_pnl >= win_thr:
            win_days += 1
        # payout: >=5 winning days, positive profit, consistency best_day<=50% of profit
        if win_days >= min_win and equity > 0 and best_day <= 0.5 * equity:
            amt = min(equity, payout_cap)
            withdrawn += amt
            equity -= amt
            payouts += 1
            if first_payout is None:
                first_payout = di + 1
            win_days = 0
            best_day = 0.0
    return {"survived": True, "day": horizon, "withdrawn": withdrawn, "payouts": payouts,
            "first_payout": first_payout, "maxdd": maxdd}


def run_funded():
    all_days, days, regimes = build_days(CLEAN5, "frozen")
    n = len(days)
    SIMS = 20000
    print(f"{'funded plan':28s} {'pay≤63':>7s} {'pay≤126':>8s} {'avg1stD':>8s} {'pots/6mo':>8s} "
          f"{'$wd/6mo':>9s} {'surv6mo':>8s} {'worstDD':>8s}")
    for label, size, lstop in [("1 MNQ grinder", "flat1", None), ("1 MNQ +lstop750", "flat1", -750.0),
                               ("2 MNQ grinder", "flat2", None), ("2 MNQ +lstop1000", "flat2", -1000.0)]:
        for HOR in (126,):
            starts = RNG.integers(0, n - HOR, SIMS)
            res = [sim_funded(days, regimes, int(s), HOR, size=size, allowed=None, lstop=lstop)
                   for s in starts]
            fp = [r["first_payout"] for r in res if r["first_payout"]]
            p63 = sum(1 for r in res if r["first_payout"] and r["first_payout"] <= 63) / SIMS
            p126 = sum(1 for r in res if r["first_payout"]) / SIMS
            surv = sum(1 for r in res if r["survived"]) / SIMS
            print(f"{label:28s} {p63*100:>6.1f}% {p126*100:>7.1f}% "
                  f"{np.mean(fp) if fp else 0:>8.0f} {np.mean([r['payouts'] for r in res]):>8.2f} "
                  f"{np.mean([r['withdrawn'] for r in res]):>9.0f} {surv*100:>7.1f}% "
                  f"{np.mean([r['maxdd'] for r in res]):>8.0f}")
    (OUT / "_funded.json").write_text(json.dumps({"note": "see stdout"}, indent=2))


def run_lifecycle():
    all_days, days, regimes = build_days(CLEAN5, "frozen")
    n = len(days)
    SIMS = 30000
    # Combine cushion123 to 180 days for pass-eventual + days-to-resolution
    HOR = 180
    starts = RNG.integers(0, n - HOR, SIMS)
    cres = [sim(days, regimes, int(s), HOR, size="cushion123", allowed=None, time_filter=None,
                regime_filter=None, pstop=None, lstop=None) for s in starts]
    pass126 = sum(1 for r in cres if r["r"] == "pass" and r["day"] <= 126) / SIMS
    fail126 = sum(1 for r in cres if r["r"] == "fail" and r["day"] <= 126) / SIMS
    res_days = [min(r["day"], 126) for r in cres]
    avg_months = float(np.mean(res_days)) / 21.0
    pdays = [r["day"] for r in cres if r["r"] == "pass" and r["day"] <= 126]
    # Funded 1 MNQ over 6 mo
    fHOR = 126
    fstarts = RNG.integers(0, n - fHOR, SIMS)
    fres = [sim_funded(days, regimes, int(s), fHOR, size="flat1", allowed=None, lstop=None)
            for s in fstarts]
    e_wd_6mo = float(np.mean([r["withdrawn"] for r in fres]))
    surv6 = float(np.mean([1 if r["survived"] else 0 for r in fres]))
    # EV (assumptions explicit)
    FEE = 165.0           # combine $/month (sensitivity below)
    ACT = 149.0           # funded activation one-time
    cost_attempt = avg_months * FEE
    funded_ev = e_wd_6mo - ACT
    ev_per_attempt = pass126 * funded_ev - cost_attempt
    cost_to_pass = cost_attempt / pass126 if pass126 else None
    ev_lifecycle = funded_ev - cost_to_pass if cost_to_pass else None
    breakeven_fee = (pass126 * funded_ev) / avg_months if avg_months else None
    out = {
        "combine_pass_126d": round(pass126, 3), "combine_fail_126d": round(fail126, 3),
        "combine_alive_126d": round(1 - pass126 - fail126, 3),
        "med_days_to_pass": round(float(np.median(pdays)), 0) if pdays else None,
        "avg_months_paying": round(avg_months, 2),
        "funded_E_withdrawn_6mo_1MNQ": round(e_wd_6mo, 0), "funded_survive_6mo": round(surv6, 3),
        "assumed_combine_fee_mo": FEE, "assumed_activation": ACT,
        "expected_combine_cost_to_pass": round(cost_to_pass, 0) if cost_to_pass else None,
        "EV_per_attempt": round(ev_per_attempt, 0),
        "EV_full_lifecycle(pass+6mo funded)": round(ev_lifecycle, 0) if ev_lifecycle else None,
        "breakeven_combine_fee_mo": round(breakeven_fee, 0) if breakeven_fee else None,
    }
    print(json.dumps(out, indent=2))
    (OUT / "_lifecycle.json").write_text(json.dumps(out, indent=2))


def run_booster():
    # apples-to-apples on phase38 source (normal-fill 1MNQ, gate+cap3). Only if a booster helps.
    SIMS = 15000
    sets = {
        "clean5 (base)": CLEAN5,
        "clean5+gap_go": [*CLEAN5, "gap_go"],
        "clean5+gap_go+dow_tue": [*CLEAN5, "gap_go", "dow_long_tue"],
        "clean5+volhi_orb": [*CLEAN5, "volhi_orb"],
    }
    print(f"{'sleeve set':26s} {'C.pass90':>8s} {'C.fail90':>8s} {'ratio':>6s} {'C.med':>6s} "
          f"{'F.pay126':>8s} {'F.surv6':>8s} {'F.$wd6':>8s}")
    for label, sl in sets.items():
        all_days, days, regimes = build_days(sl, "phase38")
        n = len(days)
        cs = RNG.integers(0, n - 90, SIMS)
        c = agg([sim(days, regimes, int(s), 90, size="cushion123", allowed=None, time_filter=None,
                     regime_filter=None, pstop=None, lstop=-750.0) for s in cs], SIMS)
        fs = RNG.integers(0, n - 126, SIMS)
        fr_ = [sim_funded(days, regimes, int(s), 126, size="flat1", allowed=None, lstop=None) for s in fs]
        surv = np.mean([1 if r["survived"] else 0 for r in fr_])
        pay126 = np.mean([1 if r["first_payout"] else 0 for r in fr_])
        wd = np.mean([r["withdrawn"] for r in fr_])
        print(f"{label:26s} {c['pass90']:>8.3f} {c['fail90']:>8.3f} {str(c['ratio90']):>6s} "
              f"{str(c['med_days']):>6s} {pay126:>8.3f} {surv:>8.3f} {wd:>8.0f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "combine"
    {"combine": run_combine, "refine": run_refine, "funded": run_funded,
     "lifecycle": run_lifecycle, "booster": run_booster}[mode]()
