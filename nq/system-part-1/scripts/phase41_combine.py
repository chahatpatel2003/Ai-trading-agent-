"""Phase 41 — clean5+gate as a Topstep 150K Combine Pass Mode (frozen signals, pass-mode wrapper only).

Wraps the FROZEN Phase-40 clean5+gate BALANCED signals in the user's pass-mode rules and Monte-Carlos
contiguous real day-windows (preserves streaks). NOTHING about the strategy is changed.

Rules: start 2 MNQ/signal; ->3 after +$2,000 cushion; ->4 after +$4,000 cushion (never 1 NQ); hard
daily loss stop -$1,000; after +$1,000 on the day take only A-quality signals; target +$9,000 then stop;
trailing DD $4,500 intraday locked-at-start; consistency: best day < 50% of total profit to pass.
A-quality := high-expectancy sleeves {ir90_break, trend_adx_cont, ir60_fade}.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("docs/phase41")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(41)
A_SLEEVES = {"ir90_break", "trend_adx_cont", "ir60_fade"}
TARGET, MLL = 9000.0, 4500.0
DAILY_STOP = -1000.0
SIMS, HORIZON = 40000, 60


def load_days(which: str) -> list[list[tuple]]:
    d = json.loads((Path("docs/phase40") / "_frozen.json").read_text())
    reg = json.loads((Path("docs/phase39") / "day_regime.json").read_text())
    all_days = sorted(reg.keys())
    by_day: dict[str, list] = defaultdict(list)
    for t in d[which]:
        by_day[t["day"]].append((t["entry_ts"], float(t["net"]), float(t["mae"]),
                                 "A" if t["sleeve"] in A_SLEEVES else "B"))
    for day in by_day:
        by_day[day].sort(key=lambda x: x[0])
    return [by_day.get(day, []) for day in all_days]


def size_for(bal: float) -> int:
    return 2 if bal < 2000 else (3 if bal < 4000 else 4)


def sim_once(seq: list[list[tuple]], start: int) -> dict:
    equity = peak = best_day = maxdd = 0.0
    active_days = 0
    for di in range(HORIZON):
        day = seq[start + di]
        day_pnl = 0.0
        traded = False
        for (_ts, net1, mae1, q) in day:
            if day_pnl <= DAILY_STOP:          # hard daily loss stop
                break
            if day_pnl >= 1000.0 and q != "A":  # after +$1k take only A-quality
                continue
            bal = equity + day_pnl
            size = size_for(bal)
            # intraday trailing-DD breach check (uses scaled MAE)
            pk = max(peak, bal)
            trough = bal - mae1 * size
            maxdd = max(maxdd, pk - trough)
            if trough <= min(pk - MLL, 0.0):
                return {"result": "fail", "day": di + 1, "maxdd": maxdd, "best_day": best_day,
                        "peak_equity": peak}
            # apply trade
            day_pnl += net1 * size
            traded = True
            bal = equity + day_pnl
            peak = max(peak, bal)
            if bal <= min(peak - MLL, 0.0):     # closed-equity breach
                maxdd = max(maxdd, peak - bal)
                return {"result": "fail", "day": di + 1, "maxdd": maxdd, "best_day": best_day,
                        "peak_equity": peak}
        equity += day_pnl
        if traded:
            active_days += 1
        best_day = max(best_day, day_pnl)
        maxdd = max(maxdd, peak - equity)
        # pass: target hit, >=2 active days, consistency (best day < 50% of total)
        if equity >= TARGET and active_days >= 2 and best_day <= 0.5 * equity:
            return {"result": "pass", "day": di + 1, "maxdd": maxdd, "best_day": best_day,
                    "final_equity": equity}
    return {"result": "timeout", "day": HORIZON, "maxdd": maxdd, "best_day": best_day,
            "final_equity": equity}


def run(which: str) -> dict:
    seq = load_days(which)
    n = len(seq)
    res = [sim_once(seq, int(RNG.integers(0, n - HORIZON))) for _ in range(SIMS)]
    passes = [r for r in res if r["result"] == "pass"]
    fails = [r for r in res if r["result"] == "fail"]
    pass_days = np.array([r["day"] for r in passes]) if passes else np.array([])
    fail_days = np.array([r["day"] for r in fails]) if fails else np.array([])
    maxdds = np.array([r["maxdd"] for r in res])
    bestdays = np.array([r["best_day"] for r in res])

    def by(dlimit):
        p = sum(1 for r in passes if r["day"] <= dlimit) / SIMS
        f = sum(1 for r in fails if r["day"] <= dlimit) / SIMS
        return p, f

    out = {
        "fills": which,
        "pass_30": round(by(30)[0], 3), "pass_45": round(by(45)[0], 3), "pass_60": round(by(60)[0], 3),
        "fail_30": round(by(30)[1], 3), "fail_45": round(by(45)[1], 3), "fail_60": round(by(60)[1], 3),
        "timeout_60": round(sum(1 for r in res if r["result"] == "timeout") / SIMS, 3),
        "avg_days_to_pass": round(float(pass_days.mean()), 1) if pass_days.size else None,
        "median_days_to_pass": round(float(np.median(pass_days)), 1) if pass_days.size else None,
        "avg_days_to_fail": round(float(fail_days.mean()), 1) if fail_days.size else None,
        "worst_dd": round(float(maxdds.max()), 0),
        "median_dd": round(float(np.median(maxdds)), 0),
        "p95_dd": round(float(np.quantile(maxdds, 0.95)), 0),
        "best_day_max": round(float(bestdays.max()), 0),
        "best_day_median": round(float(np.median(bestdays)), 0),
        "passers_breaching_mll_dd": round(float(np.mean([1 if r["maxdd"] > MLL else 0 for r in passes])), 3) if passes else None,
    }
    return out


def main() -> None:
    rows = [run("BALANCED_harsh"), run("BALANCED_normal")]
    (OUT / "_combine.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"\n===== Combine Pass Mode | fills={r['fills']} (target +$9k, MLL $4.5k, sims={SIMS}) =====")
        print(f"  PASS within 30d={r['pass_30']*100:.1f}%  45d={r['pass_45']*100:.1f}%  60d={r['pass_60']*100:.1f}%")
        print(f"  FAIL within 30d={r['fail_30']*100:.1f}%  45d={r['fail_45']*100:.1f}%  60d={r['fail_60']*100:.1f}%")
        print(f"  timeout(neither in 60d)={r['timeout_60']*100:.1f}%")
        print(f"  avg days to pass={r['avg_days_to_pass']}  median={r['median_days_to_pass']}  avg days to fail={r['avg_days_to_fail']}")
        print(f"  worst DD=${r['worst_dd']:,.0f}  median DD=${r['median_dd']:,.0f}  p95 DD=${r['p95_dd']:,.0f}")
        print(f"  best day: max=${r['best_day_max']:,.0f}  median=${r['best_day_median']:,.0f}")
        print(f"  share of PASSES that had a DD > $4.5k MLL en route = {r['passers_breaching_mll_dd']}")


if __name__ == "__main__":
    main()
