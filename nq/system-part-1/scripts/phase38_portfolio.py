"""Phase 38 portfolio step — combine independently-tested sleeves into daily-workhorse portfolios.

Reads docs/phase38/_trades.json + _sleeves.json (from phase38_workhorse.py). Combines sleeves (each
1 MNQ, union of trades) and measures portfolio days-traded%, net, DD, MC trailing-DD breach, daily
Sharpe, OOS split PF, per-year net, and a Topstep payout/blow Monte-Carlo. Hand portfolios + greedy
forward selection (maximize ret/DD subject to a $3K-breach constraint, and maximize payout). Compares
every portfolio to the dumb always-long baseline on a RISK-ADJUSTED + Topstep-viability basis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path("docs/phase38")
RNG = np.random.default_rng(380)
TOTAL_DAYS = 1406
SPLITS = {"train": (2021, 2022), "val": (2023, 2024), "test": (2025, 2026)}


def _year(day: str) -> int:
    return int(day[:4])


def _pf(nets: np.ndarray) -> float | None:
    w, ls = nets[nets > 0].sum(), -nets[nets < 0].sum()
    return round(float(w / ls), 3) if ls > 0 else None


def _mc_breach(nets: np.ndarray, maes: np.ndarray, limit: float, sims: int = 1500) -> float:
    n = len(nets)
    if n < 5:
        return 1.0
    br = 0
    for _ in range(sims):
        i = RNG.permutation(n)
        ns, ms = nets[i], maes[i]
        eq = np.cumsum(ns)
        prev = np.concatenate([[0.0], eq[:-1]])
        dd = (np.maximum.accumulate(np.maximum(eq, prev)) - (prev - ms)).max()
        br += dd >= limit
    return round(br / sims, 4)


def _payout_mc(daily: np.ndarray, target: float, mll: float, sims: int = 4000) -> dict[str, float]:
    if daily.size < 20:
        return {"payout": 0.0, "blow": 1.0, "neither": 1.0}
    pay = blow = 0
    horizon = min(daily.size, 120)
    for _ in range(sims):
        eq = np.cumsum(daily[RNG.integers(0, daily.size, horizon)])
        peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[:-1]
        floor = np.minimum(peak - mll, 0.0)
        hp = int(np.argmax(eq >= target)) if (eq >= target).any() else -1
        hb = int(np.argmax(eq <= floor)) if (eq <= floor).any() else -1
        if hp >= 0 and (hb < 0 or hp <= hb):
            pay += 1
        elif hb >= 0:
            blow += 1
    return {"payout": round(pay / sims, 3), "blow": round(blow / sims, 3),
            "neither": round(1 - (pay + blow) / sims, 3)}


def combine(names: list[str], store: dict[str, list[dict]], *, full_mc: bool = False) -> dict[str, Any]:
    trades: list[dict] = []
    for nm in names:
        trades.extend(store.get(nm, []))
    if not trades:
        return {"n": 0}
    trades.sort(key=lambda t: t["ts"])
    nets = np.array([t["net"] for t in trades])
    maes = np.array([t["mae"] for t in trades])
    days = [t["day"] for t in trades]
    uniq_days = sorted(set(days))
    # daily PnL
    by_day: dict[str, float] = {}
    day_mae: dict[str, float] = {}
    for t in trades:
        by_day[t["day"]] = by_day.get(t["day"], 0.0) + t["net"]
        day_mae[t["day"]] = max(day_mae.get(t["day"], 0.0), t["mae"])  # worst single-position MAE that day
    daily = np.array([by_day[d] for d in uniq_days])
    # intratrade DD (identity order)
    eq = np.cumsum(nets)
    prev = np.concatenate([[0.0], eq[:-1]])
    dd_intra = float((np.maximum.accumulate(np.maximum(eq, prev)) - (prev - maes)).max())
    net = float(nets.sum())
    sims = 2500 if full_mc else 800
    out: dict[str, Any] = {
        "sleeves": len(names), "n": len(trades), "uniq_days": len(uniq_days),
        "days_pct": round(len(uniq_days) / TOTAL_DAYS, 3),
        "trades_per_active_day": round(len(trades) / len(uniq_days), 2),
        "max_trades_day": int(max(days.count(d) for d in uniq_days)) if uniq_days else 0,
        "net": round(net, 0), "pf": _pf(nets), "max_dd_intra": round(dd_intra, 0),
        "ret_dd": round(net / dd_intra, 2) if dd_intra > 0 else None,
        "daily_sharpe": round(float(daily.mean() / daily.std() * np.sqrt(252)), 2) if daily.std() > 0 else None,
        "mc_breach_3k": _mc_breach(nets, maes, 3000, sims),
        "mc_breach_4500": _mc_breach(nets, maes, 4500, sims),
    }
    # daily-level conservative breach (reshuffle days; intraday dip = daily_net - worst single MAE)
    dlow = np.array([by_day[d] - day_mae[d] for d in uniq_days])
    out["mc_breach_3k_daily"] = _mc_breach_daily(daily, dlow, 3000, sims)
    # split PF + per-year net
    for sp, (y0, y1) in SPLITS.items():
        m = np.array([t["net"] for t in trades if y0 <= _year(t["day"]) <= y1])
        out[f"{sp}_pf"] = _pf(m) if m.size else None
        out[f"{sp}_net"] = round(float(m.sum()), 0) if m.size else None
    out["yearly_net"] = {y: round(float(sum(t["net"] for t in trades if _year(t["day"]) == y)), 0)
                         for y in range(2021, 2027)}
    pm = _payout_mc(daily, 9000.0, 4500.0, 4000 if full_mc else 2000)
    out["payout"], out["blow"] = pm["payout"], pm["blow"]
    out["names"] = names
    return out


def _mc_breach_daily(daily: np.ndarray, dlow: np.ndarray, limit: float, sims: int) -> float:
    n = len(daily)
    if n < 10:
        return 1.0
    br = 0
    for _ in range(sims):
        i = RNG.permutation(n)
        ns, lo = daily[i], dlow[i]
        eq = np.cumsum(ns)
        prev = np.concatenate([[0.0], eq[:-1]])
        # within-day trough = prev_equity + day_low
        dd = (np.maximum.accumulate(np.maximum(eq, prev)) - (prev + lo)).max()
        br += dd >= limit
    return round(br / sims, 4)


def main() -> None:  # noqa: PLR0915
    store = json.loads((OUT / "_trades.json").read_text())
    sl = {r["strategy"]: r for r in json.loads((OUT / "_sleeves.json").read_text())["rows"]}

    def robust(nm: str) -> bool:
        r = sl.get(nm, {})
        return all((r.get(f"{sp}_pf") or 0) >= 1.0 for sp in SPLITS)

    def positive(nm: str) -> bool:
        r = sl.get(nm, {})
        return (r.get("full_net") or 0) > 0 and (r.get("full_edge_vs_be") or -1) > 0.003 \
            and (r.get("full_n") or 0) >= 80 and not nm.startswith("BASELINE")

    pool = [nm for nm in store if positive(nm)]
    lowbreach = [nm for nm in pool if (sl[nm].get("mc_breach_3k") or 1) <= 0.90]
    print(f"pool(positive,edge>0.3pp,n>=80): {len(pool)} | lowbreach(<=90%): {len(lowbreach)}")

    # ---------- hand-built portfolios ----------
    hand = {
        "H1_robust_mr": ["ir60_fade", "asia_sweep_fade", "gap_fill", "range_or_fade"],
        "H2_reversion_lowbreach": ["gap_fill", "comp_break", "range_or_fade", "rsi_revert",
                                   "pdhl_sweep_reclaim", "ir60_fade", "asia_sweep_fade"],
        "H3_diversified": ["gap_go", "ir60_fade", "asia_sweep_fade", "pdhl_fade", "gap_fill", "comp_break"],
        "H4_beta_plus_hedge": ["beta_long_uptrend", "asia_sweep_fade", "ir60_fade", "gap_fill"],
        "H5_all_robust": [nm for nm in pool if robust(nm)],
        "H6_robust_no_beta": [nm for nm in pool if robust(nm)
                              and not nm.startswith(("beta_", "BASELINE"))
                              and sl[nm]["area"] not in ("15_beta", "16_news_proxy")],
        "H7_lowbreach_all": lowbreach,
    }

    # ---------- greedy forward selection ----------
    def greedy(objective: str, breach_cap: float, pool_: list[str]) -> list[str]:
        chosen: list[str] = []
        best = -1e9
        for _ in range(10):
            cand, cand_score = None, best
            for nm in pool_:
                if nm in chosen:
                    continue
                m = combine([*chosen, nm], store)
                if (m.get("mc_breach_3k") or 1) > breach_cap or (m.get("net") or 0) <= 0:
                    continue
                score = {"ret_dd": m.get("ret_dd") or -9, "payout": m.get("payout") or 0,
                         "sharpe": m.get("daily_sharpe") or -9}[objective]
                if score > cand_score + 1e-6:
                    cand, cand_score = nm, score
            if cand is None:
                break
            chosen.append(cand)
            best = cand_score
        return chosen

    greedy_ret = greedy("ret_dd", 0.50, lowbreach)
    greedy_pay = greedy("payout", 0.50, lowbreach)
    greedy_shp = greedy("sharpe", 0.40, lowbreach)
    print(f"greedy ret_dd: {greedy_ret}\ngreedy payout: {greedy_pay}\ngreedy sharpe: {greedy_shp}")

    portfolios = {**hand, "G1_greedy_retdd": greedy_ret, "G2_greedy_payout": greedy_pay,
                  "G3_greedy_sharpe": greedy_shp}

    results: dict[str, Any] = {}
    for name, names in portfolios.items():
        names = [n for n in dict.fromkeys(names) if n in store]  # noqa: PLW2901
        if not names:
            continue
        results[name] = combine(names, store, full_mc=True)

    # baseline for comparison (single "portfolio")
    base = combine(["BASELINE_long"], store, full_mc=True)
    results["BASELINE_long"] = base

    (OUT / "_portfolios.json").write_text(json.dumps(results, indent=2, default=str))

    cols = ("sleeves", "days_pct", "max_trades_day", "net", "pf", "ret_dd", "daily_sharpe",
            "mc_breach_3k", "mc_breach_3k_daily", "mc_breach_4500", "payout", "blow")
    print(f"\n{'portfolio':22s} " + " ".join(f"{c[:9]:>9s}" for c in cols))
    order = sorted(results, key=lambda k: (results[k].get("payout") or 0), reverse=True)
    for k in order:
        r = results[k]
        print(f"{k:22s} " + " ".join(f"{str(r.get(c)):>9s}" for c in cols))
    print("\nOOS split PF (train/val/test) + per-year net:")
    for k in order:
        r = results[k]
        print(f"  {k:22s} PF {r.get('train_pf')}/{r.get('val_pf')}/{r.get('test_pf')}  "
              f"yearly={r.get('yearly_net')}")
    print("\nportfolio members:")
    for k in order:
        print(f"  {k:22s} {results[k].get('names')}")


if __name__ == "__main__":
    main()
