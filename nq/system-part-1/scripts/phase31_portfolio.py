"""Phase 31 — multi-sleeve frequent-portfolio test (the last shot at scalable+frequent within $3K).

Combine diverse FROZEN sleeves (1 MNQ each), measure combined frequency + PF + concurrency-aware
intraday drawdown, then find the max uniform fixed-contract scaling that keeps the 100K/$3K account
safe (MC p99 DD < $3,000) and the resulting $/month. No re-optimization.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nqlab.discovery import phase29 as P29

DIRS = ["/Volumes/T9/nq data/curated/artifacts/phase29/run1/trades",
        "/Volumes/T9/nq data/curated/artifacts/phase30/run1/trades"]
OUT = Path("docs/phase31")
SPAN_MONTHS, DD_LIMIT, MNQ_PV = 66, 3000.0, 2.0
RNG = np.random.default_rng(31)
N_MC = 1000


def _load(name: str) -> pl.DataFrame:
    safe = name.replace("|", "__").replace("/", "_").replace(":", "-")
    for d in DIRS:
        for f in (f"{d}/{safe}.parquet",):
            if glob.os.path.exists(f):
                kept, _ = P29.intraday_only(pl.read_parquet(f))
                return kept.sort("entry_ts")
    raise FileNotFoundError(name)


def _concurrency_dd(trades: list[dict[str, Any]], scale: float) -> float:
    """Conservative concurrency-aware intraday max DD: at each entry, all open trades at worst MAE."""
    ev = sorted(trades, key=lambda t: t["entry_ts"])
    realized = 0.0
    peak = 0.0
    worst = 0.0
    open_mae: list[tuple[Any, float]] = []  # (exit_ts, mae_$)
    exits = sorted(trades, key=lambda t: t["exit_ts"])
    ei = 0
    for t in ev:
        # settle exits that occurred before this entry
        while ei < len(exits) and exits[ei]["exit_ts"] <= t["entry_ts"]:
            realized += exits[ei]["net"] * scale
            peak = max(peak, realized)
            open_mae = [(x, m) for (x, m) in open_mae if x > exits[ei]["exit_ts"]]
            ei += 1
        open_mae.append((t["exit_ts"], t["mae"] * MNQ_PV * scale))
        low = realized - sum(m for _, m in open_mae)
        worst = max(worst, peak - low)
    while ei < len(exits):
        realized += exits[ei]["net"] * scale
        peak = max(peak, realized)
        ei += 1
    return worst


def _records(sleeves: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in sleeves:
        df = _load(s)
        for r in df.iter_rows(named=True):
            out.append({"sleeve": s, "entry_ts": r["entry_ts"], "exit_ts": r["exit_ts"],
                        "trading_day": r["trading_day"], "net": r["net"], "mae": r["mae"]})
    return sorted(out, key=lambda t: t["entry_ts"])


def _mc_day_block(trades: list[dict[str, Any]], scale: float) -> dict[str, float]:
    by_day: dict[Any, list[dict[str, Any]]] = {}
    for t in trades:
        by_day.setdefault(t["trading_day"], []).append(t)
    days = list(by_day.values())
    n_days = len(days)
    dds = np.empty(N_MC)
    for s in range(N_MC):
        idx = RNG.integers(n_days, size=n_days)
        seq = [t for j in idx for t in days[j]]
        # closed-equity DD with per-trade MAE dip (day order preserved within block)
        realized = 0.0
        peak = 0.0
        worst = 0.0
        for t in seq:
            low = realized - t["mae"] * MNQ_PV * scale
            worst = max(worst, peak - low)
            realized += t["net"] * scale
            peak = max(peak, realized)
        dds[s] = worst
    return {"p95": float(np.percentile(dds, 95)), "p99": float(np.percentile(dds, 99)),
            "exp": float(dds.mean()), "p_breach": float((dds >= DD_LIMIT).mean())}


def _eval(name: str, sleeves: list[str]) -> dict[str, Any]:
    rec = _records(sleeves)
    nets = np.array([t["net"] for t in rec])
    n = len(rec)
    days = sorted({t["trading_day"] for t in rec})
    span_days = max((days[-1] - days[0]).days, 1)
    per_week = n / (span_days / 7)
    wins = nets[nets > 0]
    losses = nets[nets < 0]
    pf = float(wins.sum() / -losses.sum()) if losses.size else None
    net = float(nets.sum())
    dd1 = _concurrency_dd(rec, 1.0)
    max_safe = int(DD_LIMIT // dd1) if dd1 > 0 else 0
    # MC at 1x and at max_safe
    mc1 = _mc_day_block(rec, 1.0)
    mc_safe = _mc_day_block(rec, max(max_safe, 1)) if max_safe >= 1 else mc1
    # refine max_safe so MC p99 < 3000
    safe_mc = 0
    for k in range(1, 8):
        if _mc_day_block(rec, k)["p99"] < DD_LIMIT:
            safe_mc = k
        else:
            break
    monthly_1x = net / SPAN_MONTHS
    return {
        "portfolio": name, "sleeves": ", ".join(sleeves), "n": n, "per_week": round(per_week, 2),
        "per_month": round(n / SPAN_MONTHS, 2), "pf": round(pf, 3) if pf else None,
        "win_rate": round(float((nets > 0).mean()), 4), "net_1x": round(net, 2),
        "monthly_1x": round(monthly_1x, 2), "intraday_dd_1x": round(dd1, 2),
        "ret_over_dd": round(net / dd1, 2) if dd1 > 0 else None,
        "max_safe_n_hist": max_safe, "max_safe_n_mc": safe_mc,
        "monthly_at_safe_hist": round(monthly_1x * max_safe, 2),
        "monthly_at_safe_mc": round(monthly_1x * safe_mc, 2),
        "mc_p99_1x": round(mc1["p99"], 2), "mc_p99_at_safe": round(mc_safe["p99"], 2),
        "mc_breach_1x": round(mc1["p_breach"], 4),
    }


PORTFOLIOS = {
    "P30_swd+idb": ["swd_clean_path|RTH|a2.5|toPOC", "idb_base|RTH|a2.5|eod"],
    "swd_family_multi": ["swd_clean_path|RTH|a2.5|toPOC", "sweep_displacement|NY_AM|s1|rr1.5",
                         "sweep_displacement|ALLDAY|s1|rr2"],
    "diverse3": ["swd_clean_path|RTH|a2.5|toPOC", "idb_base|RTH|a2.5|eod",
                 "sweep_displacement|ALLDAY|s1|rr2"],
    "diverse4_trend": ["swd_clean_path|RTH|a2.5|toPOC", "idb_base|RTH|a2.5|eod",
                       "sweep_displacement|ALLDAY|s1|rr2", "ema_cross_21_50|NY_AM|s1.5|rr3"],
    "freq_sweeps": ["sweep_displacement|NY_AM|s1|rr1.5", "sweep_displacement|RTH|s1|rr2",
                    "sweep_displacement|ALLDAY|s1|rr2"],
}


def main() -> None:
    rows = [_eval(name, sleeves) for name, sleeves in PORTFOLIOS.items()]
    pl.DataFrame(rows, infer_schema_length=None).write_csv(OUT / "phase31_portfolios.csv")
    (OUT / "_portfolios.json").write_text(json.dumps(rows, indent=2, default=str))
    for r in rows:
        print(f"\n=== {r['portfolio']} ===")
        print(f"  n={r['n']} per_week={r['per_week']} PF={r['pf']} wr={r['win_rate']} "
              f"net1x=${r['net_1x']} monthly1x=${r['monthly_1x']}")
        print(f"  intradayDD(1x)=${r['intraday_dd_1x']} ret/DD={r['ret_over_dd']} "
              f"max_safe_n hist={r['max_safe_n_hist']} mc={r['max_safe_n_mc']}")
        print(f"  monthly@safe hist=${r['monthly_at_safe_hist']} mc=${r['monthly_at_safe_mc']} "
              f"| MC p99(1x)=${r['mc_p99_1x']} breach(1x)={r['mc_breach_1x']}")


if __name__ == "__main__":
    main()
