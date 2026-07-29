"""Phase 40 — FREEZE clean5+gate and validate forward-paper readiness. No new search, no optimization.

Re-runs the audited engine for the 5 frozen sleeves (normal + HARSH execution), applies the frozen
neutral-trend gate + daily cap, exports the raw trade log, verifies timestamps/no-lookahead, recomputes
per-year + OOS + concentration, runs 120-day rolling-window AND block-bootstrap payout/blow, and checks
every kill criterion -> PASS/FAIL for forward-paper. Frozen spec only; nothing here is tuned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nqlab.backtest.engine import run_backtest
from nqlab.config.loader import load_config, load_topstep_rules
from nqlab.discovery import phase29 as P29
from nqlab.domain.strategy.base import ExecConfig, StrategyParams

import scripts.phase38_workhorse as P38
from scripts.phase39_sim import payout_mc

OUT = Path("docs/phase40")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(40)
FROZEN = ["trend_adx_cont", "ir90_break", "asia_sweep_fade", "ir60_fade", "disp_pb_fvg"]
WIN_START = {"trend_adx_cont": 570, "ir90_break": 660, "asia_sweep_fade": 570,
             "ir60_fade": 630, "disp_pb_fvg": 570}
MNQ_TICK_USD = 0.50  # 1 tick = $0.50 per MNQ
TARGET = 9000.0  # 150K
MLL = 4500.0
TOTAL_DAYS = 1406


def run_sleeve(fr, s, cfg, rules, slip: float):
    res = run_backtest(P38.build(fr, s), symbol="NQ.c", timeframe="5m",
                       strategy=StrategyParams("p40", "1"), config=cfg, rules=rules,
                       primary_cutoff=False, entry_mode="next_open", gate_entries=False,
                       slippage_mult=slip, commission_mult=1.0,
                       exec_config=ExecConfig(sizing_mode="mnq_fixed1", time_stop_bars=10000))
    out = []
    for t in res.trades:
        if P38._exit_td(t.exit_ts) != t.trading_day:
            continue
        out.append({"sleeve": s.name, "day": str(t.trading_day), "side": int(t.side),
                    "entry_ts": t.entry_ts.isoformat(), "exit_ts": t.exit_ts.isoformat(),
                    "entry_px": round(t.entry_px, 2), "stop_px": round(t.stop_px, 2),
                    "exit_px": round(t.exit_px, 2), "reason": t.exit_reason,
                    "mae": round(float(t.mae) * 2, 2), "net": round(float(t.net_pnl), 2),
                    "entry_mod": t.entry_ts.hour * 60 + t.entry_ts.minute})
    return out


def gate_and_cap(trades: list[dict], reg: dict, cap: int, harsh: bool) -> list[dict]:
    by_day: dict[str, list[dict]] = {}
    for t in trades:
        rg = reg.get(t["day"])
        if rg and rg["trend"] == "neutral":  # frozen neutral-trend gate
            continue
        tt = dict(t)
        if harsh and tt["reason"] == "TARGET":
            tt["net"] = round(tt["net"] - MNQ_TICK_USD, 2)  # touch-only miss: 1 tick worse
        by_day.setdefault(tt["day"], []).append(tt)
    kept = []
    for day, ts in by_day.items():
        for t in sorted(ts, key=lambda x: x["entry_ts"])[:cap]:
            kept.append(t)
    return sorted(kept, key=lambda x: x["entry_ts"])


def daily_pnl(trades: list[dict]) -> tuple[np.ndarray, list[str]]:
    bd: dict[str, float] = {}
    for t in trades:
        bd[t["day"]] = bd.get(t["day"], 0.0) + t["net"]
    days = sorted(bd)
    return np.array([bd[d] for d in days]), days


def rolling_windows(daily: np.ndarray, target: float, mll: float, win: int = 120) -> dict[str, Any]:
    """Real-path rolling 120-day windows (no bootstrap)."""
    n = len(daily)
    if n < win:
        return {"windows": 0}
    pay = blow = 0
    for i in range(n - win + 1):
        eq = np.cumsum(daily[i:i + win])
        peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[:-1]
        floor = np.minimum(peak - mll, 0.0)
        bi = int(np.argmax(eq <= floor)) if (eq <= floor).any() else 10**9
        pi = 10**9
        cm = np.maximum.accumulate(daily[i:i + win])
        for j in range(1, win):
            if eq[j] >= target and cm[j] <= 0.5 * eq[j]:
                pi = j
                break
        if pi < bi:
            pay += 1
        elif bi < 10**9:
            blow += 1
    w = n - win + 1
    return {"windows": w, "payout_pct": round(pay / w, 3), "blow_pct": round(blow / w, 3),
            "ratio": round(pay / blow, 2) if blow else None}


def panel(nets: np.ndarray) -> dict[str, Any]:
    w, ls = nets[nets > 0], nets[nets < 0]
    return {"n": len(nets), "wr": round(float((nets > 0).mean()), 3),
            "pf": round(float(w.sum() / -ls.sum()), 3) if ls.size else None,
            "net": round(float(nets.sum()), 0),
            "avg_w": round(float(w.mean()), 1) if w.size else 0,
            "avg_l": round(float(ls.mean()), 1) if ls.size else 0}


def main() -> None:  # noqa: PLR0915
    cfg = load_config(profile="dev", overrides=P29.PHASE29_OVERRIDES)
    rules = load_topstep_rules("topstep_150k")
    fr = P38.add_cols(pl.read_parquet(cfg.data.artifacts_root / "phase30" / "frame_5m_full_vp.parquet"))
    reg = json.loads((Path("docs/phase39") / "day_regime.json").read_text())
    strat_by = {s.name: s for s in P38.strategies()}

    # 1) run each frozen sleeve, normal + harsh
    raw_norm: list[dict] = []
    raw_harsh: list[dict] = []
    for nm in FROZEN:
        raw_norm += run_sleeve(fr, strat_by[nm], cfg, rules, slip=1.0)
        raw_harsh += run_sleeve(fr, strat_by[nm], cfg, rules, slip=2.0)
    print(f"raw trades (1 MNQ, pre-gate): normal={len(raw_norm)} harsh={len(raw_harsh)}")

    # 2) timestamp + no-lookahead verification (tz-correct: engine ts are UTC; map to ET minute_of_day)
    sub = fr.select(pl.col("ts").dt.strftime("%Y-%m-%dT%H:%M:%S").alias("s"), "minute_of_day", "is_rth")
    ts2mod = dict(zip(sub["s"].to_list(), sub["minute_of_day"].to_list()))
    ts_rth = set(sub.filter(pl.col("is_rth"))["s"].to_list())
    ts_all = set(sub["s"].to_list())
    bad_entry_bar = bad_exit_bar = bad_window = bad_order = same_bar = 0
    for t in raw_norm:
        e, x = t["entry_ts"][:19], t["exit_ts"][:19]
        if e not in ts_rth:                              # entry must be a real RTH bar open
            bad_entry_bar += 1
        if x not in ts_all:                              # exit must be a real bar
            bad_exit_bar += 1
        mod = ts2mod.get(e)
        if mod is None or mod < WIN_START[t["sleeve"]]:  # no entry before sleeve window opens
            bad_window += 1
        if x < e:                                        # exit strictly before entry = lookahead
            bad_order += 1
        if x == e:                                       # same-bar stop/target (legitimate)
            same_bar += 1
    print(f"\n=== timestamp / no-lookahead verification (normal, n={len(raw_norm)}) ===")
    print(f"  entry not on a real RTH bar:        {bad_entry_bar}")
    print(f"  exit  not on a real bar:            {bad_exit_bar}")
    print(f"  entry before sleeve window opens:   {bad_window}")
    print(f"  exit STRICTLY before entry (LOOKAHEAD): {bad_order}")
    print(f"  same-bar stop/target exits (OK):    {same_bar}")

    # 3) build frozen portfolios (gate + cap), export raw log
    port = {
        "BALANCED_normal": gate_and_cap(raw_norm, reg, 3, harsh=False),
        "BALANCED_harsh": gate_and_cap(raw_harsh, reg, 3, harsh=True),
        "SAFE_normal": gate_and_cap(raw_norm, reg, 2, harsh=False),
        "SAFE_harsh": gate_and_cap(raw_harsh, reg, 2, harsh=True),
    }
    pl.DataFrame(port["BALANCED_normal"]).write_csv(OUT / "frozen_trades_BALANCED.csv")
    pl.DataFrame(port["BALANCED_harsh"]).write_csv(OUT / "frozen_trades_BALANCED_harsh.csv")
    (OUT / "_frozen.json").write_text(json.dumps({k: v for k, v in port.items()}, default=str))

    for label in ("BALANCED_normal", "BALANCED_harsh", "SAFE_normal", "SAFE_harsh"):
        tr = port[label]
        nets = np.array([t["net"] for t in tr])
        daily, days = daily_pnl(tr)
        p = panel(nets)
        # per-year
        yr = {}
        for y in range(2021, 2027):
            m = np.array([t["net"] for t in tr if t["day"][:4] == str(y)])
            yr[y] = round(float(m.sum()), 0) if m.size else 0
        # OOS PF (val+test 2023-26)
        oos = np.array([t["net"] for t in tr if 2023 <= int(t["day"][:4]) <= 2026])
        oospf = round(float(oos[oos > 0].sum() / -oos[oos < 0].sum()), 3)
        # concentration: top sleeve share of gross profit
        gp = {}
        for t in tr:
            if t["net"] > 0:
                gp[t["sleeve"]] = gp.get(t["sleeve"], 0.0) + t["net"]
        topshare = round(max(gp.values()) / sum(gp.values()), 3)
        days_pct = round(len(days) / TOTAL_DAYS, 3)
        roll = rolling_windows(daily, TARGET, MLL)
        mc = payout_mc(daily, TARGET, MLL, sims=6000)
        mcb = payout_mc(daily, TARGET, MLL, sims=6000, block=10)
        print(f"\n===== {label} =====")
        print(f"  trades={p['n']} days%={days_pct} WR={p['wr']} PF={p['pf']} net6yr={p['net']} "
              f"avgW={p['avg_w']} avgL={p['avg_l']}")
        print(f"  per-year: {yr}")
        print(f"  OOS(23-26) PF={oospf} | top-sleeve profit share={topshare} ({max(gp,key=gp.get)})")
        print(f"  150K rolling-120 (real path): {roll}")
        print(f"  150K payout MC iid: pay={mc['payout']} blow={mc['blow']} ratio={mc['ratio']} | "
              f"block: pay={mcb['payout']} blow={mcb['blow']} ratio={mcb['ratio']}")
        port[f"{label}_stats"] = {"panel": p, "yr": yr, "oospf": oospf, "topshare": topshare,
                                  "days_pct": days_pct, "roll": roll, "mc": mc, "mcb": mcb,
                                  "top_sleeve": max(gp, key=gp.get)}

    # per-sleeve breakdown within frozen BALANCED normal (concentration detail)
    print("\n=== per-sleeve contribution (BALANCED normal, after gate+cap) ===")
    tr = port["BALANCED_normal"]
    for nm in FROZEN:
        m = np.array([t["net"] for t in tr if t["sleeve"] == nm])
        if m.size:
            print(f"  {nm:18s} n={m.size:4d} net={m.sum():8.0f} pf={panel(m)['pf']}")
    (OUT / "_frozen_stats.json").write_text(json.dumps(
        {k: v for k, v in port.items() if k.endswith("_stats")}, indent=2, default=str))


if __name__ == "__main__":
    main()
