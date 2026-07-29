"""Phase 39 step 2 — de-beta'd/de-duped portfolio + regime gate + adaptive sizing + payout sims.

Builds the cleaned book (remove beta sleeve dow_long_mon, collapse the 5 correlated breakout/momentum
sleeves to one, keep the genuine hedges), applies the train-confirmed neutral-trend gate and adaptive
sizing, and runs Topstep payout/blow Monte-Carlo for 50K/100K/150K (real configs, trailing intraday
locked-at-start, consistency 50%, min 2 days). Objective = payout / blow. Reports days%, payout, blow,
ratio for safe/balanced/aggressive picks. Sizing/gates decided on train, validated on val+test (OOS).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

OUT = Path("docs/phase39")
RNG = np.random.default_rng(39)
TOTAL_DAYS = 1406
ACCOUNTS = {"50K": (3000.0, 2000.0, 5), "100K": (6000.0, 3000.0, 10), "150K": (9000.0, 4500.0, 15)}
STORE = json.loads((Path("docs/phase38") / "_trades.json").read_text())
REG = json.loads((OUT / "day_regime.json").read_text())

# cleaned book: beta removed (dow_long_mon), breakout cluster collapsed to ir90_break
CLEAN5 = ["trend_adx_cont", "ir90_break", "asia_sweep_fade", "ir60_fade", "disp_pb_fvg"]
CLEAN4 = ["trend_adx_cont", "ir90_break", "asia_sweep_fade", "ir60_fade"]
RNB2 = ["od_cont_struct", "asia_sweep_fade", "disp_pb_fvg", "trend_adx_cont", "dow_long_mon",
        "ir30_break", "ir60_break", "ir90_break", "ir60_fade"]


def daily_series(names: list[str], size_fn: Callable[[str], float], cap: int,
                 gate_neutral: bool, split: tuple[int, int] | None = None) -> tuple[np.ndarray, int]:
    by_day: dict[str, list[dict]] = {}
    for nm in names:
        for t in STORE[nm]:
            if split and not (split[0] <= int(t["day"][:4]) <= split[1]):
                continue
            by_day.setdefault(t["day"], []).append(t)
    daynets: list[float] = []
    for day, trades in by_day.items():
        rg = REG.get(day)
        if gate_neutral and rg and rg["trend"] == "neutral":
            continue
        s = size_fn(day)
        if s <= 0:
            continue
        trades = sorted(trades, key=lambda t: t["ts"])[:cap]
        daynets.append(sum(t["net"] for t in trades) * s)
    n_total = (sum(1 for d in REG if split is None or split[0] <= int(d[:4]) <= split[1]))
    return np.array(daynets), n_total


def payout_mc(daily: np.ndarray, target: float, mll: float, sims: int = 6000,
              horizon: int = 120, block: int = 0) -> dict[str, float]:
    if daily.size < 20:
        return {"payout": 0.0, "blow": 1.0, "ratio": 0.0}
    pay = blow = 0
    for _ in range(sims):
        if block > 0:  # block bootstrap: preserve streaks
            starts = RNG.integers(0, daily.size, horizon // block + 1)
            d = np.concatenate([daily[s:s + block] if s + block <= daily.size
                                else np.concatenate([daily[s:], daily[:block - (daily.size - s)]])
                                for s in starts])[:horizon]
        else:
            d = daily[RNG.integers(0, daily.size, horizon)]
        eq = np.cumsum(d)
        peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[:-1]
        floor = np.minimum(peak - mll, 0.0)
        blow_i = int(np.argmax(eq <= floor)) if (eq <= floor).any() else 10**9
        # payout: first day eq>=target with >=2 days and consistency (max day <= 50% of eq)
        pay_i = 10**9
        cummax = np.maximum.accumulate(d)
        for i in range(1, horizon):  # i>=1 -> at least 2 days
            if eq[i] >= target and cummax[i] <= 0.5 * eq[i]:
                pay_i = i
                break
        if pay_i < blow_i:
            pay += 1
        elif blow_i < 10**9:
            blow += 1
    p, b = pay / sims, blow / sims
    return {"payout": round(p, 3), "blow": round(b, 3), "ratio": round(p / b, 2) if b > 0 else None}


# ---- sizing schemes (decided on train regime structure) ----
def sz_flat(k: float) -> Callable[[str], float]:
    return lambda day: k


def sz_adaptive(base: float, hi: float) -> Callable[[str], float]:
    # high-confidence = clear trend (bull/bear) AND strong ADX; else base
    def f(day: str) -> float:
        rg = REG.get(day)
        if not rg:
            return base
        if rg["trend"] in ("bull", "bear") and rg["trend_str"] == "adx_hi":
            return hi
        return base
    return f


def evaluate(label: str, names: list[str], size_fn: Callable[[str], float], cap: int,
             gate: bool) -> dict[str, Any]:
    daily, ntot = daily_series(names, size_fn, cap, gate)
    row: dict[str, Any] = {"label": label, "sleeves": len(names), "cap": cap, "gate": gate,
                           "days_pct": round(len(daily) / TOTAL_DAYS, 3),
                           "net6yr": round(float(daily.sum()), 0),
                           "mean120": round(float(daily.mean() * 120), 0) if daily.size else 0,
                           "daily_std": round(float(daily.std()), 0) if daily.size else 0}
    for acc, (tgt, mll, _mx) in ACCOUNTS.items():
        r = payout_mc(daily, tgt, mll)
        row[f"{acc}_pay"], row[f"{acc}_blow"], row[f"{acc}_ratio"] = r["payout"], r["blow"], r["ratio"]
    # OOS payout on val+test only (2023-2026)
    dval, _ = daily_series(names, size_fn, cap, gate, split=(2023, 2026))
    rv50, rv150 = payout_mc(dval, 3000.0, 2000.0), payout_mc(dval, 9000.0, 4500.0)
    row["50K_pay_OOS"], row["50K_blow_OOS"] = rv50["payout"], rv50["blow"]
    row["150K_pay_OOS"], row["150K_blow_OOS"] = rv150["payout"], rv150["blow"]
    # block-bootstrap (streak-preserving) stress on 150K + 50K, full sample
    rb150 = payout_mc(daily, 9000.0, 4500.0, block=10)
    rb50 = payout_mc(daily, 3000.0, 2000.0, block=10)
    row["150K_pay_blk"], row["150K_blow_blk"] = rb150["payout"], rb150["blow"]
    row["50K_pay_blk"], row["50K_blow_blk"] = rb50["payout"], rb50["blow"]
    return row


def main() -> None:
    configs = [
        ("RNB2 ref @2 f1 (P38)", RNB2, sz_flat(1), 2, False),
        ("clean5 nogate @2 f1", CLEAN5, sz_flat(1), 2, False),
        ("clean5 gate @1 f1", CLEAN5, sz_flat(1), 1, True),
        ("clean5 gate @2 f1", CLEAN5, sz_flat(1), 2, True),
        ("clean5 gate @3 f1", CLEAN5, sz_flat(1), 3, True),
        ("clean5 gate @4 f1", CLEAN5, sz_flat(1), 4, True),
        ("clean5 gate @5 f1", CLEAN5, sz_flat(1), 5, True),
        ("clean5 gate @3 adapt12", CLEAN5, sz_adaptive(1, 2), 3, True),
        ("clean5 gate @5 adapt12", CLEAN5, sz_adaptive(1, 2), 5, True),
        ("clean5 gate @3 f2", CLEAN5, sz_flat(2), 3, True),
        ("clean5 gate @2 f2", CLEAN5, sz_flat(2), 2, True),
        ("clean4 gate @3 f1", CLEAN4, sz_flat(1), 3, True),
        ("clean4 gate @5 f1", CLEAN4, sz_flat(1), 5, True),
        ("clean4 gate @3 adapt12", CLEAN4, sz_adaptive(1, 2), 3, True),
    ]
    rows = [evaluate(*c) for c in configs]
    cols = ["label", "days_pct", "net6yr", "mean120", "daily_std",
            "50K_pay", "50K_blow", "50K_ratio", "100K_pay", "100K_blow", "100K_ratio",
            "150K_pay", "150K_blow", "150K_ratio", "50K_pay_OOS", "50K_blow_OOS"]
    w = {c: max(len(c), 8) for c in cols}
    print(" ".join(f"{c:>{w[c]}s}" for c in cols))
    for r in rows:
        print(" ".join(f"{str(r.get(c)):>{w[c]}s}" for c in cols))
    (OUT / "_sim.json").write_text(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
