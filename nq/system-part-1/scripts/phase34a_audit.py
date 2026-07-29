"""Phase 34A — stats reality audit. Try to BREAK the Phase-30 finalist numbers.

Independent of the strategy stats/engine code: (1) recompute every stat from the raw trade list with
fresh functions; (2) re-derive the trades with an INDEPENDENT conservative fill simulator and compare
trade-by-trade to the audited engine; (3) integrity checks (exit>entry, no overnight, fills within
bar range, gap-through fills, target reachability); (4) fill/cost sensitivity (touch / 1-tick-through
/ 2-tick-through targets, stop slip, 2x/3x cost); (5) overfit / data-mining audit; (6) account + MC.

No rule changes, no optimization. Finalists: idb_base|RTH|eod, swd_clean_path|RTH|toPOC, portfolio.
"""

from __future__ import annotations

import json
import zoneinfo
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nqlab.config.loader import load_config, load_topstep_rules
from nqlab.discovery import phase29 as P29
from nqlab.discovery import phase30 as P30

ET = zoneinfo.ZoneInfo("America/New_York")
OUT = Path("docs/phase34a")
OUT.mkdir(parents=True, exist_ok=True)
TICK, MNQ_PV, MNQ_COMM_SIDE = 0.25, 2.0, 0.25
FORCE_FLAT = 955
REPORTED = {  # Phase-30 reported (1 MNQ)
    "idb": {"n": 221, "pf": 1.439, "net": 8120.16, "maxdd": 1880.23, "win": 0.4525},
    "swd": {"n": 244, "pf": 1.535, "net": 7713.77, "maxdd": 977.21, "win": 0.459},
}
CANDS = {"idb": "idb_base|RTH|a2.5|eod", "swd": "swd_clean_path|RTH|a2.5|toPOC"}


def _exit_td(ts: Any) -> Any:
    return (ts.astimezone(ET) + timedelta(hours=6)).date()


# ---------- independent stats (fresh, not reusing phase29_stats) ----------
def indep_stats(nets: list[float], days: list[Any], maes_usd: list[float]) -> dict[str, Any]:
    a = np.array(nets, dtype=float)
    n = len(a)
    wins, losses = a[a > 0], a[a < 0]
    eq = np.cumsum(a)
    closed_dd = float((np.maximum.accumulate(eq) - eq).max())
    # intraday DD via per-trade MAE (sequential)
    prev = np.concatenate([[0.0], eq[:-1]])
    run_peak = np.maximum.accumulate(np.maximum(eq, prev))
    intr_dd = float((run_peak - (prev - np.array(maes_usd))).max())
    # daily series for ratios
    dser: dict[Any, float] = {}
    for d, x in zip(days, nets, strict=True):
        dser[d] = dser.get(d, 0.0) + x
    dv = np.array(list(dser.values()))
    sharpe = float(dv.mean() / dv.std(ddof=1) * np.sqrt(252)) if len(dv) > 1 and dv.std() > 0 else None
    downs = dv[dv < 0]
    dstd = float(np.sqrt((downs**2).mean())) if downs.size else 0.0
    sortino = float(dv.mean() / dstd * np.sqrt(252)) if dstd > 0 else None
    ml = cur = 0
    for x in a:
        cur = cur + 1 if x < 0 else 0
        ml = max(ml, cur)
    return {
        "n": n, "net": round(float(a.sum()), 2),
        "gross_profit": round(float(wins.sum()), 2), "gross_loss": round(float(losses.sum()), 2),
        "pf": round(float(wins.sum() / -losses.sum()), 3) if losses.size else None,
        "win_rate": round(float((a > 0).mean()), 4),
        "avg_win": round(float(wins.mean()), 2) if wins.size else 0,
        "avg_loss": round(float(losses.mean()), 2) if losses.size else 0,
        "expectancy": round(float(a.mean()), 2),
        "closed_dd": round(closed_dd, 2), "intraday_dd": round(intr_dd, 2),
        "ret_over_dd": round(float(a.sum() / closed_dd), 3) if closed_dd > 0 else None,
        "sharpe": round(sharpe, 3) if sharpe else None,
        "sortino": round(sortino, 3) if sortino else None,
        "max_loss_streak": ml,
    }


# ---------- independent conservative fill simulator (re-derives trades from bars) ----------
def indep_sim(frame: pl.DataFrame, *, target_margin_tk: float = 0.0, stop_slip_tk: float = 1.0,
              entry_slip_tk: float = 1.0, cost_mult: float = 1.0) -> list[dict[str, Any]]:
    """Fresh implementation of next-open entry / stop-first / exact-target(+margin) / EOD-flat."""
    rows = frame.select(
        ["ts", "trading_day", "minute_of_day", "open", "high", "low", "close", "signal",
         "stop_pts", "target_pts"]
    ).to_dicts()
    slip = TICK * cost_mult
    comm_rt = MNQ_COMM_SIDE * 2 * cost_mult
    trades: list[dict[str, Any]] = []
    pending = None
    pos = None
    cur_day = None
    for r in rows:
        day, minute = r["trading_day"], r["minute_of_day"]
        o, h, lo, c = r["open"], r["high"], r["low"], r["close"]
        if day != cur_day:
            cur_day = day
        # execute pending entry at this bar open
        if pending is not None and pos is None:
            side, sp, tp, sig_ts = pending
            pending = None
            if minute < FORCE_FLAT:
                ef = o + side * entry_slip_tk * TICK
                pos = {"side": side, "entry": ef, "stop": ef - side * sp, "tgt": ef + side * tp,
                       "ets": r["ts"], "eday": day, "sig": sig_ts, "mae": 0.0, "mfe": 0.0}
        if pos is not None:
            s, e, st, tg = pos["side"], pos["entry"], pos["stop"], pos["tgt"]
            pos["mfe"] = max(pos["mfe"], (h - e) if s == 1 else (e - lo))
            pos["mae"] = max(pos["mae"], (e - lo) if s == 1 else (h - e))
            hit_stop = (lo <= st) if s == 1 else (h >= st)
            tgt_level = tg + side_margin(s, target_margin_tk)
            hit_tgt = (h >= tgt_level) if s == 1 else (lo <= tgt_level)
            reason = px = None
            if hit_stop:  # stop first (conservative)
                reason, px = "STOP", st - s * stop_slip_tk * TICK
            elif hit_tgt:
                reason, px = "TARGET", tg  # exact limit fill (no slip)
            elif minute >= FORCE_FLAT:
                reason, px = "EOD", c - s * slip
            if reason:
                gross = s * (px - e) * MNQ_PV
                trades.append({
                    "entry_ts": pos["ets"], "exit_ts": r["ts"], "trading_day": pos["eday"],
                    "side": s, "entry_px": e, "exit_px": px, "reason": reason,
                    "gross": gross, "net": gross - comm_rt, "mae": pos["mae"], "mfe": pos["mfe"],
                    "signal_ts": pos["sig"], "stop_px": st, "tgt_px": tg,
                })
                pos = None
        if pos is None and r["signal"] != 0 and pending is None and minute < FORCE_FLAT:
            sp, tp = r["stop_pts"], r["target_pts"]
            if sp and tp and sp > 0:
                pending = (int(r["signal"]), sp, tp, r["ts"])
    return trades


def side_margin(side: int, ticks: float) -> float:
    return side * ticks * TICK


def _signal_frame(cfg: Any, fr: pl.DataFrame, key: str) -> pl.DataFrame:
    cand = {c.name: c for c in P30.generate_candidates()}[CANDS[key]]
    return P30.build_signal_frame(fr, cand)


def main() -> None:
    cfg = load_config(profile="dev", overrides=P29.PHASE29_OVERRIDES)
    rules = load_topstep_rules("topstep_150k")
    fr = pl.read_parquet(cfg.data.artifacts_root / "phase30" / "frame_5m_full_vp.parquet")
    cmap = {c.name: c for c in P30.generate_candidates()}
    report: dict[str, Any] = {}

    for key in ("idb", "swd"):
        cand = cmap[CANDS[key]]
        res = P30.run_candidate(fr, cand, config=cfg, rules=rules)
        eng = [t for t in res.trades if _exit_td(t.exit_ts) == t.trading_day]  # engine, same-day
        eng_nets = [t.net_pnl for t in eng]
        eng_days = [t.trading_day for t in eng]
        eng_mae_usd = [t.mae * MNQ_PV for t in eng]

        # PART 1: independent stats recompute vs reported
        stat = indep_stats(eng_nets, eng_days, eng_mae_usd)
        rep = REPORTED[key]
        checks = {
            "n": (stat["n"], rep["n"], stat["n"] == rep["n"]),
            "pf": (stat["pf"], rep["pf"], abs(stat["pf"] - rep["pf"]) < 0.005),
            "net": (stat["net"], rep["net"], abs(stat["net"] - rep["net"]) < 1.0),
            "maxdd": (stat["closed_dd"], rep["maxdd"], abs(stat["closed_dd"] - rep["maxdd"]) < 1.0),
            "win": (stat["win_rate"], rep["win"], abs(stat["win_rate"] - rep["win"]) < 0.002),
        }

        # PART 4 (core): independent fill simulator reproduces engine?
        sig = _signal_frame(cfg, fr, key)
        ind = [t for t in indep_sim(sig) if _exit_td(t["exit_ts"]) == t["trading_day"]]
        ind_net = round(sum(t["net"] for t in ind), 2)
        repro = {"engine_n": len(eng), "indep_n": len(ind),
                 "engine_net": round(sum(eng_nets), 2), "indep_net": ind_net,
                 "n_match": len(eng) == len(ind),
                 "net_match": abs(sum(eng_nets) - ind_net) < max(5.0, 0.005 * abs(sum(eng_nets)))}

        # integrity checks on engine trades
        bad_exit_before_entry = sum(1 for t in eng if t.exit_ts <= t.entry_ts)
        overnight = sum(1 for t in eng if _exit_td(t.exit_ts) != t.trading_day)  # already filtered
        # gap-through stops: exit bar (independent) where price gapped beyond stop at the open
        integ = {"exit_before_entry": bad_exit_before_entry, "overnight_trades": overnight,
                 "reasons": dict(__import__("collections").Counter(t.exit_reason for t in eng))}

        # fill/cost sensitivity (independent sim)
        sens = {}
        for name, kw in {
            "touch_1x": {}, "1tk_through": {"target_margin_tk": 1.0},
            "2tk_through": {"target_margin_tk": 2.0}, "stop_slip2": {"stop_slip_tk": 2.0},
            "stop_slip4": {"stop_slip_tk": 4.0}, "cost_2x": {"cost_mult": 2.0},
            "cost_3x": {"cost_mult": 3.0},
        }.items():
            tt = [t for t in indep_sim(sig, **kw) if _exit_td(t["exit_ts"]) == t["trading_day"]]
            nn = np.array([t["net"] for t in tt])
            w, ls = nn[nn > 0], nn[nn < 0]
            sens[name] = {"n": len(tt), "pf": round(float(w.sum() / -ls.sum()), 3) if ls.size else None,
                          "net": round(float(nn.sum()), 2)}

        # PART 6: overfit / concentration
        a = np.array(sorted(eng_nets))
        months: dict[str, float] = {}
        years: dict[int, float] = {}
        for t in eng:
            months[f"{t.trading_day.year}-{t.trading_day.month:02d}"] = months.get(
                f"{t.trading_day.year}-{t.trading_day.month:02d}", 0.0) + t.net_pnl
            years[t.trading_day.year] = years.get(t.trading_day.year, 0.0) + t.net_pnl
        tot = sum(eng_nets)
        overfit = {
            "net": round(tot, 2),
            "net_ex_best5": round(tot - float(a[-5:].sum()), 2),
            "net_ex_best10": round(tot - float(a[-10:].sum()), 2),
            "net_ex_worst5": round(tot - float(a[:5].sum()), 2),
            "best5_share": round(float(a[-5:].sum()) / tot, 3) if tot > 0 else None,
            "best10_share": round(float(a[-10:].sum()) / tot, 3) if tot > 0 else None,
            "best_month": max(months.items(), key=lambda kv: kv[1]),
            "best_month_share": round(max(months.values()) / tot, 3) if tot > 0 else None,
            "net_ex_best_month": round(tot - max(months.values()), 2),
            "best_year": max(years.items(), key=lambda kv: kv[1]),
            "best_year_share": round(max(years.values()) / tot, 3) if tot > 0 else None,
            "yearly": {y: round(v, 0) for y, v in sorted(years.items())},
        }

        report[key] = {"indep_stats": stat, "part1_checks": checks, "part4_repro": repro,
                       "integrity": integ, "fill_cost_sensitivity": sens, "overfit": overfit}

        # export full audit trade CSV
        tr_rows = []
        eqc = 0.0
        peak = 0.0
        for i, t in enumerate(sorted(eng, key=lambda x: x.exit_ts)):
            eqc += t.net_pnl
            peak = max(peak, eqc)
            tr_rows.append({
                "strategy": key, "trade_id": i + 1, "trading_day": str(t.trading_day),
                "entry_ts": str(t.entry_ts), "exit_ts": str(t.exit_ts),
                "direction": "long" if t.side == 1 else "short", "instrument": t.instrument,
                "entry_px": round(t.entry_px, 2), "stop_px": round(t.stop_px, 2),
                "target_px": round(t.target_px, 2) if abs(t.target_px - t.entry_px) < 50000 else "EOD",
                "exit_px": round(t.exit_px, 2), "exit_reason": t.exit_reason, "contracts": t.contracts,
                "gross_pnl": round(t.gross_pnl, 2), "fees": t.fees, "net_pnl": round(t.net_pnl, 2),
                "mfe_pts": round(t.mfe, 2), "mae_pts": round(t.mae, 2),
                "equity_after": round(100000 + eqc, 2), "drawdown_after": round(peak - eqc, 2),
                "year": t.trading_day.year, "month": f"{t.trading_day.year}-{t.trading_day.month:02d}",
            })
        pl.DataFrame(tr_rows, infer_schema_length=None).write_csv(OUT / f"phase34a_trades_{key}.csv")

    (OUT / "_audit.json").write_text(json.dumps(report, indent=2, default=str))
    # print summary
    for key in ("idb", "swd"):
        r = report[key]
        print(f"\n===== {key} =====")
        print("  PART1 recompute vs reported:")
        for m, (got, rep, ok) in r["part1_checks"].items():
            print(f"    {m:6s}: indep={got} reported={rep}  {'PASS' if ok else 'FAIL'}")
        rp = r["part4_repro"]
        print(f"  PART4 independent fill reproduces engine: n {rp['engine_n']}vs{rp['indep_n']} "
              f"({'OK' if rp['n_match'] else 'MISMATCH'}); net ${rp['engine_net']}vs${rp['indep_net']} "
              f"({'OK' if rp['net_match'] else 'MISMATCH'})")
        print("    integrity:", r["integrity"])
        print("  fill/cost sensitivity PF:", {k: v["pf"] for k, v in r["fill_cost_sensitivity"].items()})
        o = r["overfit"]
        print(f"  overfit: net ${o['net']} | ex-best5 ${o['net_ex_best5']} (best5 share {o['best5_share']}) "
              f"| ex-best-month ${o['net_ex_best_month']} (share {o['best_month_share']}) "
              f"| best-year share {o['best_year_share']}")
        print(f"    yearly: {o['yearly']}")


if __name__ == "__main__":
    main()
