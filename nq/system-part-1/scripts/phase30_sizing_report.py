"""Phase 30 finalist sizing study — fixed-dollar-risk sizing on the FROZEN finalists.

NO rule changes / NO re-optimization. Only the sizing simulation changes. Reuses the exact frozen
trades from scripts/phase30_finalist_report.py (idb 221, swd 244, portfolio 465).

Fixed-$ risk sizing (dynamic — small stops -> more contracts, wide stops -> fewer/skip):
  risk_per_MNQ = stop_distance_points * $2 ;  contracts = floor(risk_budget / risk_per_MNQ)
  skip if contracts < 1 ; net scales linearly (net_1mnq is 1-contract net incl. costs).
  1 NQ uses $20/pt and $1.70 round-trip commission (derived exactly from the point move).

Account: $100,000, $3,000 trailing max drawdown (floor = peak - 3000), intratrade lows via MAE.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# reuse the frozen-trade loader + record builder
_spec = importlib.util.spec_from_file_location("p30fr", "scripts/phase30_finalist_report.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

OUT = Path("docs/phase30_finalist")
START_BAL, DD_LIMIT = 100_000.0, 3_000.0
MNQ_PV, NQ_PV, MNQ_TICK = 2.0, 20.0, 0.50
NQ_COMM_RT = 1.70
RNG = np.random.default_rng(7)
N_MC = 2000
BUDGETS = [50, 100, 150, 200, 250, 300]
MODELS = ["1 MNQ", "$50", "$100", "$150", "$200", "$250", "$300", "1 NQ"]


def _per_trade(rec: dict[str, Any], model: str) -> dict[str, Any] | None:
    """Return sized trade {net,mae_d,contracts,risk_usd,...} or None if skipped."""
    sd = rec["stop_dist_pts"]
    if sd <= 0:
        return None
    if model == "1 MNQ":
        c, net, mae_d, risk = 1, rec["_net1mnq"], rec["mae_pts"] * MNQ_PV, sd * MNQ_PV
    elif model == "1 NQ":
        c = 1
        net = rec["_gross1mnq"] * 10.0 - NQ_COMM_RT
        mae_d, risk = rec["mae_pts"] * NQ_PV, sd * NQ_PV
    else:  # fixed-$ MNQ
        budget = float(model[1:])
        rp = sd * MNQ_PV
        c = int(budget // rp)
        if c < 1:
            return None
        c = min(c, 50)  # sane contract cap
        net = rec["_net1mnq"] * c
        mae_d, risk = rec["mae_pts"] * MNQ_PV * c, rp * c
    return {"entry_ts": rec["entry_ts"], "exit_ts": rec["exit_ts"], "trading_day": rec["date"],
            "year": rec["year"], "month": rec["month"], "net": net, "mae_d": mae_d,
            "contracts": c, "risk_usd": risk, "stop_dist": sd}


def _path(recs: list[dict[str, Any]], model: str) -> tuple[list[dict[str, Any]], int]:
    out, skipped = [], 0
    for r in sorted(recs, key=lambda x: x["exit_ts"]):
        t = _per_trade(r, model)
        if t is None:
            skipped += 1
        else:
            out.append(t)
    return out, skipped


def _portfolio_path(idb: list, swd: list, mode: str) -> tuple[list[dict[str, Any]], int]:
    """mode: 'mnq1' | 'per200' | 'tot200'|'tot300'|'tot500'."""
    if mode == "mnq1":
        p = [t for r in (idb, swd) for t in _path(r, "1 MNQ")[0]]
        return sorted(p, key=lambda x: x["exit_ts"]), 0
    if mode == "per200":
        p = _path(idb, "$200")[0] + _path(swd, "$200")[0]
        return sorted(p, key=lambda x: x["exit_ts"]), 0
    total = {"tot200": 200.0, "tot300": 300.0, "tot500": 500.0}[mode]
    merged = sorted(idb + swd, key=lambda x: x["entry_ts"])
    used: dict[str, float] = {}
    out, skipped = [], 0
    for r in merged:
        sd = r["stop_dist_pts"]
        rp = sd * MNQ_PV
        avail = total - used.get(r["date"], 0.0)
        c = int(avail // rp)
        if c < 1:
            skipped += 1
            continue
        c = min(c, 50)
        used[r["date"]] = used.get(r["date"], 0.0) + c * rp
        out.append({"entry_ts": r["entry_ts"], "exit_ts": r["exit_ts"], "trading_day": r["date"],
                    "year": r["year"], "month": r["month"], "net": r["_net1mnq"] * c,
                    "mae_d": r["mae_pts"] * MNQ_PV * c, "contracts": c, "risk_usd": rp * c,
                    "stop_dist": sd})
    return sorted(out, key=lambda x: x["exit_ts"]), skipped


def _max_dd(eq: np.ndarray) -> float:
    return float((np.maximum.accumulate(eq) - eq).max()) if eq.size else 0.0


def _panel(path: list[dict[str, Any]], skipped: int) -> dict[str, Any]:
    if not path:
        return {"trades": 0, "skipped": skipped, "fail_3k": True}
    nets = np.array([t["net"] for t in path])
    cs = np.array([t["contracts"] for t in path])
    risks = np.array([t["risk_usd"] for t in path])
    eq = np.cumsum(nets)
    wins = nets[nets > 0]
    losses = nets[nets < 0]
    days: dict[str, float] = {}
    months: dict[str, float] = {}
    years: dict[int, float] = {}
    for t in path:
        days[t["trading_day"]] = days.get(t["trading_day"], 0.0) + t["net"]
        months[t["month"]] = months.get(t["month"], 0.0) + t["net"]
        years[t["year"]] = years.get(t["year"], 0.0) + t["net"]
    dvals = np.array(list(days.values()))
    sharpe = float(dvals.mean() / dvals.std(ddof=1) * np.sqrt(252)) if dvals.size > 1 and dvals.std() > 0 else None
    downs = dvals[dvals < 0]
    dstd = float(np.sqrt((downs**2).mean())) if downs.size else 0.0
    sortino = float(dvals.mean() / dstd * np.sqrt(252)) if dstd > 0 else None
    mdd = _max_dd(eq)
    annual = nets.sum() / max(len(days), 1) * 252
    # streaks
    ml = cur = 0
    for x in nets:
        cur = cur + 1 if x < 0 else 0
        ml = max(ml, cur)
    # worst 20
    w20 = float(np.convolve(nets, np.ones(20), "valid").min()) if len(nets) >= 20 else float(nets.sum())
    intr = _trailing_sim(path)
    return {
        "trades": len(path), "skipped": skipped,
        "avg_contracts": round(float(cs.mean()), 2), "median_contracts": int(np.median(cs)),
        "max_contracts": int(cs.max()), "avg_risk": round(float(risks.mean()), 2),
        "avg_loss": round(float(losses.mean()), 2) if losses.size else 0.0,
        "max_loss": round(float(nets.min()), 2), "net": round(float(nets.sum()), 2),
        "pf": round(float(wins.sum() / -losses.sum()), 3) if losses.size else None,
        "win_rate": round(float((nets > 0).mean()), 4),
        "expectancy": round(float(nets.mean()), 2),
        "max_dd": round(mdd, 2), "intraday_dd": intr["intraday_max_dd"],
        "ret_dd": round(float(nets.sum() / mdd), 3) if mdd > 0 else None,
        "sharpe": round(sharpe, 3) if sharpe else None,
        "sortino": round(sortino, 3) if sortino else None,
        "calmar": round(float(annual / mdd), 3) if mdd > 0 else None,
        "max_loss_streak": ml, "worst_20": round(w20, 2),
        "best_month": f"{max(months, key=months.get)}:{max(months.values()):.0f}",
        "worst_month": f"{min(months, key=months.get)}:{min(months.values()):.0f}",
        "profitable_years": sum(1 for v in years.values() if v > 0),
        "losing_years": sum(1 for v in years.values() if v <= 0),
        "fail_3k": not intr["pass"], "min_buffer": intr["min_buffer"],
        "breach_date": intr["breach_date"], "ending_equity": round(START_BAL + float(nets.sum()), 2),
    }


def _trailing_sim(path: list[dict[str, Any]]) -> dict[str, Any]:
    eq = peak = START_BAL
    min_buf = float("inf")
    worst = 0.0
    breach = False
    bdate = None
    for t in path:
        low = eq - t["mae_d"]
        floor = peak - DD_LIMIT
        buf = low - floor
        min_buf = min(min_buf, buf)
        if low < floor and not breach:
            breach, bdate = True, str(t["exit_ts"])[:10]
        worst = max(worst, peak - low)
        eq += t["net"]
        peak = max(peak, eq)
    return {"pass": not breach, "min_buffer": round(min_buf, 2), "breach_date": bdate,
            "intraday_max_dd": round(worst, 2)}


def _mc(path: list[dict[str, Any]]) -> dict[str, Any]:
    if len(path) < 5:
        return {"p3k": None, "p25k": None, "p2k": None, "exp": None, "p95": None, "p99": None,
                "p60": None, "p120": None}
    nets = np.array([t["net"] for t in path])
    maes = np.array([t["mae_d"] for t in path])
    n = len(nets)
    dds = np.empty(N_MC)
    for s in range(N_MC):
        idx = RNG.permutation(n)
        ns, ms = nets[idx], maes[idx]
        extra = RNG.uniform(0, MNQ_TICK, n)  # slippage stress
        ns = ns - extra
        eq = START_BAL + np.cumsum(ns)
        prev = np.concatenate([[START_BAL], eq[:-1]])
        peak = np.maximum.accumulate(np.maximum(eq, prev))
        dds[s] = float((peak - (prev - ms)).max())
    days: dict[str, float] = {}
    for t in path:
        days[t["trading_day"]] = days.get(t["trading_day"], 0.0) + t["net"]
    dvals = np.array(list(days.values()))

    def pp(nd: int) -> float:
        return float((dvals[RNG.integers(len(dvals), size=(N_MC, nd))].sum(1) > 0).mean())

    return {"p3k": round(float((dds >= 3000).mean()), 4), "p25k": round(float((dds >= 2500).mean()), 4),
            "p2k": round(float((dds >= 2000).mean()), 4), "exp": round(float(dds.mean()), 2),
            "p95": round(float(np.percentile(dds, 95)), 2), "p99": round(float(np.percentile(dds, 99)), 2),
            "p60": round(pp(60), 4), "p120": round(pp(120), 4)}


def _max_safe_risk(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Largest fixed-$ MNQ budget with (a) no historical $3K breach, (b) MC p99 maxDD < $3K."""
    hist_ok = mc_ok = 0
    for b in range(50, 1001, 25):
        path, _ = _path(recs, f"${b}")
        if not path:
            continue
        if _trailing_sim(path)["pass"]:
            hist_ok = b
        if (_mc(path)["p99"] or 1e9) < 3000:
            mc_ok = b
    return {"hist_max_safe": hist_ok, "mc_max_safe": mc_ok}


def _plot(path: list[dict[str, Any]], title: str, eq_file: str, dd_file: str) -> None:
    nets = np.array([t["net"] for t in path])
    eq = np.cumsum(nets)
    dd = np.maximum.accumulate(eq) - eq
    x = np.arange(len(eq))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, eq, lw=1.4, color="#1f6feb")
    ax.fill_between(x, 0, eq, alpha=0.08, color="#1f6feb")
    ax.set_title(f"{title} — equity ($, net)  2021→2026H1", fontsize=12)
    ax.set_xlabel("trade #")
    ax.set_ylabel("cumulative net $")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / eq_file, dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(x, 0, -dd, color="#d1242f", alpha=0.5)
    ax.axhline(-3000, ls="--", color="black", lw=1, label="$3K limit")
    ax.set_title(f"{title} — drawdown ($)", fontsize=12)
    ax.set_xlabel("trade #")
    ax.set_ylabel("drawdown $")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / dd_file, dpi=110)
    plt.close(fig)


def main() -> None:
    trades = _m._load_trades()
    recs = {k: _m._records(v, k) for k, v in trades.items()}

    # ---- Part 1: per-strategy sizing comparison + 100K/$3K result + MC ----
    part1: list[dict[str, Any]] = []
    mc_rows: list[dict[str, Any]] = []
    for strat in ("idb", "swd"):
        for model in MODELS:
            path, skipped = _path(recs[strat], model)
            pan = _panel(path, skipped)
            part1.append({"strategy": strat, "sizing": model, **pan})
            mc_rows.append({"strategy": strat, "sizing": model, **_mc(path)})

    # ---- portfolio sizing models ----
    pmodes = [("1 MNQ each", "mnq1"), ("$200/strategy", "per200"),
              ("$200 total/day", "tot200"), ("$300 total/day", "tot300"), ("$500 total/day", "tot500")]
    for label, mode in pmodes:
        path, skipped = _portfolio_path(recs["idb"], recs["swd"], mode)
        pan = _panel(path, skipped)
        part1.append({"strategy": "port", "sizing": label, **pan})
        mc_rows.append({"strategy": "port", "sizing": label, **_mc(path)})

    pl.DataFrame(part1, infer_schema_length=None).write_csv(OUT / "phase30_sizing_comparison.csv")
    pl.DataFrame(mc_rows, infer_schema_length=None).write_csv(OUT / "phase30_sizing_montecarlo.csv")

    # ---- max safe risk ----
    safe = {k: _max_safe_risk(recs[k]) for k in ("idb", "swd")}
    # portfolio max-safe via per-strategy total budget
    psafe = {}
    for total in range(100, 1201, 50):
        path, _ = _portfolio_path(recs["idb"], recs["swd"], f"tot{total}" if total in (200, 300, 500) else "tot200")
    # simpler: sweep total-day budgets explicitly
    port_hist = port_mc = 0
    for total in range(100, 1601, 50):
        merged = sorted(recs["idb"] + recs["swd"], key=lambda x: x["entry_ts"])
        used: dict[str, float] = {}
        path = []
        for r in merged:
            rp = r["stop_dist_pts"] * MNQ_PV
            avail = total - used.get(r["date"], 0.0)
            c = int(avail // rp)
            if c < 1:
                continue
            used[r["date"]] = used.get(r["date"], 0.0) + c * rp
            path.append({"exit_ts": r["exit_ts"], "trading_day": r["date"], "year": r["year"],
                         "month": r["month"], "net": r["_net1mnq"] * c, "mae_d": r["mae_pts"] * MNQ_PV * c,
                         "contracts": c, "risk_usd": rp * c, "stop_dist": r["stop_dist_pts"]})
        path = sorted(path, key=lambda x: x["exit_ts"])
        if path and _trailing_sim(path)["pass"]:
            port_hist = total
        if path and (_mc(path)["p99"] or 1e9) < 3000:
            port_mc = total
    safe["port"] = {"hist_max_safe_total": port_hist, "mc_max_safe_total": port_mc}

    # ---- graphs: portfolio at 1MNQ / $100 / $200 / $300 ----
    gpaths = {
        "1mnq": _portfolio_path(recs["idb"], recs["swd"], "mnq1")[0],
        "100risk": _portfolio_path(recs["idb"], recs["swd"], "tot200")[0],  # placeholder, replaced below
    }
    # per-strategy fixed-$ summed portfolio (idb@B + swd@B) for graph budgets
    for b, tag in ((100, "100risk"), (200, "200risk"), (300, "300risk")):
        p = _path(recs["idb"], f"${b}")[0] + _path(recs["swd"], f"${b}")[0]
        gpaths[tag] = sorted(p, key=lambda x: x["exit_ts"])
    _plot(gpaths["1mnq"], "portfolio 1 MNQ each",
          "phase30_portfolio_1mnq_equity.png", "phase30_portfolio_1mnq_drawdown.png")
    _plot(gpaths["100risk"], "portfolio $100 risk/strategy",
          "phase30_portfolio_100risk_equity.png", "phase30_portfolio_100risk_drawdown.png")
    _plot(gpaths["200risk"], "portfolio $200 risk/strategy",
          "phase30_portfolio_200risk_equity.png", "phase30_portfolio_200risk_drawdown.png")
    _plot(gpaths["300risk"], "portfolio $300 risk/strategy",
          "phase30_portfolio_300risk_equity.png", "phase30_portfolio_300risk_drawdown.png")
    # idb/swd sizing-comparison equity overlays
    for strat in ("idb", "swd"):
        fig, ax = plt.subplots(figsize=(11, 5))
        for model in ("1 MNQ", "$100", "$200", "$300", "1 NQ"):
            path, _ = _path(recs[strat], model)
            if path:
                ax.plot(np.cumsum([t["net"] for t in path]), lw=1.2, label=model)
        ax.set_title(f"{strat} — equity by sizing ($, net)")
        ax.set_xlabel("trade #")
        ax.set_ylabel("cumulative net $")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"phase30_{strat}_sizing_equity.png", dpi=110)
        plt.close(fig)

    (OUT / "_sizing_stats.json").write_text(json.dumps(
        {"part1": part1, "mc": mc_rows, "safe": safe}, indent=2, default=str))
    print("DONE sizing. rows:", len(part1))
    for row in part1:
        if row["strategy"] in ("idb", "swd") and row["sizing"] in ("1 MNQ", "$200", "1 NQ"):
            print(f"  {row['strategy']} {row['sizing']:6s}: trades={row['trades']} skip={row['skipped']} "
                  f"avgC={row.get('avg_contracts')} net=${row.get('net')} pf={row.get('pf')} "
                  f"DD=${row.get('max_dd')} fail3k={row.get('fail_3k')}")


if __name__ == "__main__":
    main()
