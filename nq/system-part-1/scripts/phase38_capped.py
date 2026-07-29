"""Phase 38 — capped-concurrency portfolios (the fair test of the user's 0-3 trades/day preference).

The uncapped portfolios reach 99% of days only by stacking 9-15 sleeves = up to 15-21 concurrent MNQ,
which over-leverages a 150K/$4500 account (100% breach). Here we cap each day to the FIRST k signals
(k in 1,2,3), bounding size to <=k MNQ, and re-measure days%, net, trailing-DD breach, and Topstep
payout/blow. This isolates whether limiting leverage to the user's preferred 0-3 trades/day rescues a
trades-most-days, Topstep-viable portfolio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.phase38_portfolio import (  # reuse audited helpers
    SPLITS,
    TOTAL_DAYS,
    _mc_breach,
    _mc_breach_daily,
    _payout_mc,
    _pf,
    _year,
)

OUT = Path("docs/phase38")


def metrics(trades: list[dict], names: list[str]) -> dict[str, Any]:
    if not trades:
        return {"n": 0}
    trades = sorted(trades, key=lambda t: t["ts"])
    nets = np.array([t["net"] for t in trades])
    maes = np.array([t["mae"] for t in trades])
    days = [t["day"] for t in trades]
    uniq = sorted(set(days))
    by_day: dict[str, float] = {}
    day_mae: dict[str, float] = {}
    for t in trades:
        by_day[t["day"]] = by_day.get(t["day"], 0.0) + t["net"]
        day_mae[t["day"]] = max(day_mae.get(t["day"], 0.0), t["mae"])
    daily = np.array([by_day[d] for d in uniq])
    eq = np.cumsum(nets)
    prev = np.concatenate([[0.0], eq[:-1]])
    dd = float((np.maximum.accumulate(np.maximum(eq, prev)) - (prev - maes)).max())
    net = float(nets.sum())
    dlow = np.array([by_day[d] - day_mae[d] for d in uniq])
    out: dict[str, Any] = {
        "sleeves": len(names), "n": len(trades), "days_pct": round(len(uniq) / TOTAL_DAYS, 3),
        "max_trades_day": int(max(days.count(d) for d in uniq)),
        "net": round(net, 0), "pf": _pf(nets), "ret_dd": round(net / dd, 2) if dd > 0 else None,
        "daily_sharpe": round(float(daily.mean() / daily.std() * np.sqrt(252)), 2) if daily.std() > 0 else None,
        "mc_breach_3k": _mc_breach(nets, maes, 3000, 2000),
        "mc_breach_3k_daily": _mc_breach_daily(daily, dlow, 3000, 2000),
        "mc_breach_4500": _mc_breach(nets, maes, 4500, 2000),
    }
    for sp, (y0, y1) in SPLITS.items():
        m = np.array([t["net"] for t in trades if y0 <= _year(t["day"]) <= y1])
        out[f"{sp}_pf"] = _pf(m) if m.size else None
    pm = _payout_mc(daily, 9000.0, 4500.0, 4000)
    out["payout"], out["blow"] = pm["payout"], pm["blow"]
    return out


def cap_day(names: list[str], store: dict, k: int) -> list[dict]:
    """Union sleeve trades, then keep only the first k (by time) on each day."""
    trades: list[dict] = []
    for nm in names:
        trades.extend(store.get(nm, []))
    trades.sort(key=lambda t: t["ts"])
    seen: dict[str, int] = {}
    kept = []
    for t in trades:
        c = seen.get(t["day"], 0)
        if c < k:
            kept.append(t)
            seen[t["day"]] = c + 1
    return kept


def main() -> None:
    store = json.loads((OUT / "_trades.json").read_text())
    pf = json.loads((OUT / "_portfolios.json").read_text())
    pools = {
        "all_robust": pf["H5_all_robust"]["names"],
        "robust_no_beta": pf["H6_robust_no_beta"]["names"],
        "diversified": pf["H3_diversified"]["names"],
        "lowbreach_all": pf["H7_lowbreach_all"]["names"],
    }
    cols = ("sleeves", "days_pct", "max_trades_day", "net", "pf", "ret_dd", "daily_sharpe",
            "mc_breach_3k", "mc_breach_3k_daily", "mc_breach_4500", "payout", "blow")
    res: dict[str, Any] = {}
    print(f"{'pool@cap':28s} " + " ".join(f"{c[:9]:>9s}" for c in cols))
    for pname, names in pools.items():
        for k in (1, 2, 3):
            m = metrics(cap_day(names, store, k), names)
            key = f"{pname}@{k}"
            res[key] = m
            print(f"{key:28s} " + " ".join(f"{str(m.get(c)):>9s}" for c in cols))
    print("\nOOS split PF (train/val/test):")
    for key, m in res.items():
        print(f"  {key:28s} {m.get('train_pf')}/{m.get('val_pf')}/{m.get('test_pf')}")
    (OUT / "_capped.json").write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
