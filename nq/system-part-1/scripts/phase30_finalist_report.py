"""Phase 30 finalist deep-dive — exact frozen stats, 100K/$3K feasibility, Monte Carlo, graphs.

NO strategy changes / NO re-optimization. Re-runs the three FROZEN finalists exactly as Phase 30
left them, reproduces the same same-day trade set, and prints/exports everything requested.

Finalists:
  idb  = idb_base | RTH | a2.5 | eod
  swd  = swd_clean_path | RTH | a2.5 | toPOC
  port = idb + swd (1 MNQ each, trades merged in time)

Costs are the audited model embedded in the engine (slippage in the fills; commission in fees):
MNQ win/loss $1.00/$1.50, NQ $6.70/$11.70. Sizing scales per-contract; NQ derived exactly from the
per-trade point move (gross/$2) → gross*10 - $1.70 round-trip commission.
"""

from __future__ import annotations

import json
import zoneinfo
from datetime import timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from nqlab.config.loader import load_config, load_topstep_rules
from nqlab.discovery import phase29 as P29
from nqlab.discovery import phase30 as P30

ET = zoneinfo.ZoneInfo("America/New_York")
OUT = Path("docs/phase30_finalist")
OUT.mkdir(parents=True, exist_ok=True)
START_BAL = 100_000.0
DD_LIMIT = 3_000.0
MNQ_PV, NQ_PV = 2.0, 20.0
MNQ_COMM_RT, NQ_COMM_RT = 0.50, 1.70  # round-trip commission ($/contract)
RNG = np.random.default_rng(42)
N_MC = 2000

EXACT_RULES: dict[str, dict[str, str]] = {
    "idb": {
        "name": "idb_base | RTH | a2.5 | eod",
        "family": "inside_day_break (price-action breakout)",
        "timeframe": "5m", "session": "RTH (entries 09:30-15:50 ET)", "direction": "both",
        "entry": "Prior day was an INSIDE day (prev_inside_day==1). After the opening range forms "
                 "(minute>=10:00 ET; OR=09:30-10:00 high/low), LONG when a 5m CLOSE breaks above "
                 "or_high (and prior close<=or_high); SHORT when close<or_low. Fill next bar open.",
        "signal": "long: prev_inside_day==1 & minute>=600 & close>or_high & close.shift(1)<=or_high; "
                  "short: mirror with or_low.",
        "confirmation": "none beyond inside-day filter + close-through-OR",
        "stop": "2.5 x ATR at the signal bar (ATR = base Wilder ATR). STOP slipped 1 tick.",
        "target": "NONE — hold to end of day (no fixed target).",
        "exit": "EOD_FLAT at 15:55 ET, or the 2.5xATR stop, whichever first. No trailing / time stop.",
        "rr_type": "DYNAMIC (no fixed target; realized R measured)",
        "stop_type": "ATR (2.5x)",
    },
    "swd": {
        "name": "swd_clean_path | RTH | a2.5 | toPOC",
        "family": "sweep_displacement (ICT) + volume-profile clean-path filter",
        "timeframe": "5m", "session": "RTH (entries 09:30-15:50 ET)", "direction": "both",
        "entry": "LONG: swept a confirmed swing LOW (swept_low==1) AND a displacement candle up "
                 "(displacement==1 & close>open); FILTER: entry is below prior POC by >0.5 ATR "
                 "(vp_d_poc_atr<-0.5) AND no opposing prior HVN within 1 ATR above the path "
                 "(vp_dist_hvn_up_atr is null OR >1.0). SHORT: mirror. Fill next bar open.",
        "signal": "long: swept_low==1 & displacement==1 & close>open & vp_d_poc_atr<-0.5 & "
                  "(vp_dist_hvn_up_atr null or >1.0); short: mirror.",
        "confirmation": "displacement candle in trade direction + clean path to POC (no blocking HVN)",
        "stop": "2.5 x ATR at the signal bar. STOP slipped 1 tick.",
        "target": "Prior-session POC (vp_prior_poc), distance clamped to [0.5, 8] x ATR; if POC is "
                  "not in the trade direction, fallback target = 2 x stop. TARGET = exact limit.",
        "exit": "POC target limit, or 2.5xATR stop, or EOD_FLAT 15:55, whichever first.",
        "rr_type": "DYNAMIC (target = POC distance; realized R measured)",
        "stop_type": "ATR (2.5x)",
    },
    "port": {
        "name": "portfolio = idb_base + swd_clean_path (1 MNQ each)",
        "family": "diversified pair (breakout + sweep/VP)",
        "timeframe": "5m", "session": "RTH", "direction": "both",
        "entry": "Run both strategies concurrently, 1 MNQ each, independently.",
        "signal": "union of idb and swd signals", "confirmation": "see idb + swd",
        "stop": "each leg 2.5xATR", "target": "idb=EOD, swd=POC",
        "exit": "each leg per its own rule; both flat by 15:55",
        "rr_type": "DYNAMIC", "stop_type": "ATR (2.5x)",
    },
}


def _exit_td(ts: Any) -> Any:
    return (ts.astimezone(ET) + timedelta(hours=6)).date()


def _load_trades() -> dict[str, list[Any]]:
    cfg = load_config(profile="dev", overrides=P29.PHASE29_OVERRIDES)
    rules = load_topstep_rules("topstep_150k")
    fr = pl.read_parquet(cfg.data.artifacts_root / "phase30" / "frame_5m_full_vp.parquet")
    cmap = {c.name: c for c in P30.generate_candidates()}
    out: dict[str, list[Any]] = {}
    for key, cname in (("idb", "idb_base|RTH|a2.5|eod"), ("swd", "swd_clean_path|RTH|a2.5|toPOC")):
        res = P30.run_candidate(fr, cmap[cname], config=cfg, rules=rules)
        out[key] = [t for t in res.trades if _exit_td(t.exit_ts) == t.trading_day]
    return out


def _records(trades: list[Any], strategy: str) -> list[dict[str, Any]]:
    """Per-trade records (1 MNQ) with prices, R, costs, MFE/MAE — independent of account."""
    rows: list[dict[str, Any]] = []
    big = 50_000.0
    for i, t in enumerate(trades):
        stop_dist = abs(t.entry_px - t.stop_px)
        risk_usd = stop_dist * MNQ_PV  # 1 MNQ initial risk
        pnl_r = t.net_pnl / risk_usd if risk_usd > 0 else 0.0
        point_move = t.gross_pnl / MNQ_PV
        is_eod_target = abs(t.target_px - t.entry_px) > big  # the "no fixed target" sentinel
        hold_min = (t.exit_ts - t.entry_ts).total_seconds() / 60.0
        et = t.entry_ts.astimezone(ET)
        rows.append({
            "strategy": strategy, "trade_id": i + 1, "date": t.trading_day.isoformat(),
            "entry_ts": t.entry_ts.isoformat(), "exit_ts": t.exit_ts.isoformat(),
            "direction": "long" if t.side == 1 else "short",
            "entry_price": round(t.entry_px, 2), "stop_price": round(t.stop_px, 2),
            "target_price": "EOD" if is_eod_target else round(t.target_px, 2),
            "exit_price": round(t.exit_px, 2), "exit_reason": t.exit_reason,
            "contracts": 1, "instrument": "MNQ",
            "gross_pnl": round(t.gross_pnl, 2), "net_pnl": round(t.net_pnl, 2),
            "pnl_R": round(pnl_r, 3), "MFE": round(t.mfe, 2), "MAE": round(t.mae, 2),
            "commission": MNQ_COMM_RT, "slippage_pts": 0.25,
            "stop_dist_pts": round(stop_dist, 2), "risk_usd": round(risk_usd, 2),
            "point_move": round(point_move, 2), "bars_held": t.bars_held,
            "hold_min": round(hold_min, 1), "year": t.trading_day.year,
            "month": f"{t.trading_day.year}-{t.trading_day.month:02d}",
            "session": "RTH", "mae_pts": t.mae, "mfe_pts": t.mfe,
            "_net1mnq": t.net_pnl, "_gross1mnq": t.gross_pnl,
        })
    return rows


# ---- core stats helpers -------------------------------------------------------------------------
def _max_dd(equity: np.ndarray) -> tuple[float, int, int, float]:
    """(max_dd, peak_idx, trough_idx, avg_dd) on an equity path starting from 0-baseline allowed."""
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    trough = int(np.argmax(dd))
    peak_idx = int(np.argmax(equity[: trough + 1])) if trough > 0 else 0
    return float(dd.max()), peak_idx, trough, float(dd.mean())


def _streaks(nets: list[float]) -> dict[str, int]:
    cw = cl = mw = ml = 0
    n3 = n5 = 0
    runl = 0
    for x in nets:
        if x > 0:
            cw += 1
            if runl >= 3:
                n3 += 1
            if runl >= 5:
                n5 += 1
            runl = 0
            cl = 0
        elif x < 0:
            cl += 1
            runl += 1
            cw = 0
        mw, ml = max(mw, cw), max(ml, cl)
    if runl >= 3:
        n3 += 1
    if runl >= 5:
        n5 += 1
    return {"max_consec_wins": mw, "max_consec_losses": ml, "n_3loss": n3, "n_5loss": n5}


def _panel(recs: list[dict[str, Any]], net_key: str = "_net1mnq",
           gross_key: str = "_gross1mnq") -> dict[str, Any]:
    recs = sorted(recs, key=lambda r: r["exit_ts"])
    nets = [r[net_key] for r in recs]
    gross = [r[gross_key] for r in recs]
    n = len(nets)
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x < 0]
    be = [x for x in nets if x == 0]
    eq = np.cumsum(nets)
    mdd, pk, tr, add = _max_dd(eq)
    rs = [r["pnl_R"] for r in recs]
    rwin = [r for r in rs if r > 0]
    rloss = [r for r in rs if r < 0]
    # daily / monthly aggregations
    days: dict[str, float] = {}
    months: dict[str, float] = {}
    for r in recs:
        days[r["date"]] = days.get(r["date"], 0.0) + r[net_key]
        months[r["month"]] = months.get(r["month"], 0.0) + r[net_key]
    n_days = len(days)
    n_months = len(months)
    daily_vals = np.array(list(days.values()))
    sharpe = float(daily_vals.mean() / daily_vals.std(ddof=1) * np.sqrt(252)) if n_days > 1 and daily_vals.std() > 0 else None
    downs = daily_vals[daily_vals < 0]
    dstd = float(np.sqrt((downs**2).mean())) if downs.size else 0.0
    sortino = float(daily_vals.mean() / dstd * np.sqrt(252)) if dstd > 0 else None
    annual = sum(nets) / max(n_days, 1) * 252
    calmar = float(annual / mdd) if mdd > 0 else None
    gw, gl = sum(wins), -sum(losses)
    gw_g, gl_g = sum(x for x in gross if x > 0), -sum(x for x in gross if x < 0)
    # drawdown timing
    eq_dates = [r["exit_ts"][:10] for r in recs]
    dd_range = (eq_dates[pk] if pk < len(eq_dates) else "", eq_dates[tr] if tr < len(eq_dates) else "")
    # recovery: first index after trough where eq exceeds prior peak
    rec_idx = next((j for j in range(tr, n) if eq[j] >= eq[pk]), None)
    recovery = (rec_idx - tr) if rec_idx is not None else None
    monthly_vals = list(months.values())
    pos_m = [m for m in monthly_vals if m > 0]
    yearly: dict[int, float] = {}
    for r in recs:
        yearly[r["year"]] = yearly.get(r["year"], 0.0) + r[net_key]
    panel = {
        "trades": n, "wins": len(wins), "losses": len(losses), "breakeven": len(be),
        "win_rate": round(len(wins) / n, 4) if n else 0,
        "loss_rate": round(len(losses) / n, 4) if n else 0,
        "pf_after": round(gw / gl, 3) if gl else None, "pf_before": round(gw_g / gl_g, 3) if gl_g else None,
        "net": round(sum(nets), 2), "gross_profit": round(gw, 2), "gross_loss": round(-gl, 2),
        "expectancy": round(sum(nets) / n, 2) if n else 0,
        "expectancy_month": round(sum(nets) / n_months, 2) if n_months else 0,
        "avg_win": round(gw / len(wins), 2) if wins else 0, "avg_loss": round(-gl / len(losses), 2) if losses else 0,
        "win_loss_ratio": round((gw / len(wins)) / (gl / len(losses)), 3) if wins and losses else None,
        "max_win": round(max(nets), 2), "max_loss": round(min(nets), 2),
        "avg_R": round(float(np.mean(rs)), 3), "median_R": round(float(np.median(rs)), 3),
        "avg_win_R": round(float(np.mean(rwin)), 3) if rwin else None,
        "avg_loss_R": round(float(np.mean(rloss)), 3) if rloss else None,
        "best_R": round(max(rs), 3), "worst_R": round(min(rs), 3),
        "avg_mfe": round(float(np.mean([r["mfe_pts"] for r in recs])), 2),
        "avg_mae": round(float(np.mean([r["mae_pts"] for r in recs])), 2),
        "mfe_mae": round(float(np.mean([r["mfe_pts"] for r in recs]) / np.mean([r["mae_pts"] for r in recs])), 3),
        "max_closed_dd": round(mdd, 2), "avg_dd": round(add, 2),
        "max_dd_range": f"{dd_range[0]}..{dd_range[1]}", "recovery_trades": recovery,
        "ret_over_maxdd": round(sum(nets) / mdd, 3) if mdd > 0 else None,
        "sharpe": round(sharpe, 3) if sharpe else None, "sortino": round(sortino, 3) if sortino else None,
        "calmar": round(calmar, 3) if calmar else None,
        "trading_days": n_days, "months": n_months,
        "trades_per_week": round(n / (n_days / 5), 2) if n_days else None,
        "trades_per_month": round(n / n_months, 2) if n_months else None,
        "max_trades_day": int(max(__import__("collections").Counter(r["date"] for r in recs).values())) if recs else 0,
        "avg_hold_min": round(float(np.mean([r["hold_min"] for r in recs])), 1),
        "median_hold_min": round(float(np.median([r["hold_min"] for r in recs])), 1),
        "max_hold_min": round(max(r["hold_min"] for r in recs), 1),
        "min_hold_min": round(min(r["hold_min"] for r in recs), 1),
        "winning_months": len(pos_m), "losing_months": n_months - len(pos_m),
        "winning_month_pct": round(len(pos_m) / n_months, 3) if n_months else None,
        "best_month": max(months.items(), key=lambda kv: kv[1]),
        "worst_month": min(months.items(), key=lambda kv: kv[1]),
        "best_year": max(yearly.items(), key=lambda kv: kv[1]),
        "worst_year": min(yearly.items(), key=lambda kv: kv[1]),
        "profitable_years": sum(1 for v in yearly.values() if v > 0),
        "losing_years": sum(1 for v in yearly.values() if v <= 0),
        **_streaks(nets),
    }
    return panel


def _year_table(recs: list[dict[str, Any]], net_key: str = "_net1mnq") -> list[dict[str, Any]]:
    out = []
    for y in sorted({r["year"] for r in recs}):
        rr = [r for r in recs if r["year"] == y]
        nets = [r[net_key] for r in rr]
        wins = [x for x in nets if x > 0]
        losses = [x for x in nets if x < 0]
        eq = np.cumsum(sorted([(r["exit_ts"], r[net_key]) for r in rr]) and [r[net_key] for r in sorted(rr, key=lambda x: x["exit_ts"])])
        mdd, *_ = _max_dd(eq)
        months: dict[str, float] = {}
        for r in rr:
            months[r["month"]] = months.get(r["month"], 0.0) + r[net_key]
        ml = _streaks(nets)["max_consec_losses"]
        out.append({
            "year": y, "trades": len(rr), "net": round(sum(nets), 2),
            "pf": round(sum(wins) / -sum(losses), 3) if losses else None,
            "win_rate": round(len(wins) / len(rr), 3),
            "expectancy": round(sum(nets) / len(rr), 2), "avg_trade": round(sum(nets) / len(rr), 2),
            "max_dd": round(mdd, 2), "ret_dd": round(sum(nets) / mdd, 2) if mdd > 0 else None,
            "max_loss_streak": ml,
            "best_month": f"{max(months, key=months.get)}:{max(months.values()):.0f}",
            "worst_month": f"{min(months, key=months.get)}:{min(months.values()):.0f}",
        })
    return out


def _month_table(recs: list[dict[str, Any]], net_key: str = "_net1mnq") -> list[dict[str, Any]]:
    out = []
    months = sorted({r["month"] for r in recs})
    for m in months:
        rr = [r for r in recs if r["month"] == m]
        nets = [r[net_key] for r in rr]
        wins = [x for x in nets if x > 0]
        losses = [x for x in nets if x < 0]
        eq = np.cumsum([r[net_key] for r in sorted(rr, key=lambda x: x["exit_ts"])])
        mdd, *_ = _max_dd(eq)
        out.append({
            "month": m, "pnl": round(sum(nets), 2), "trades": len(rr),
            "win_rate": round(len(wins) / len(rr), 3) if rr else 0,
            "pf": round(sum(wins) / -sum(losses), 3) if losses else None,
            "max_dd": round(mdd, 2),
        })
    return out


# ---- account sizing + feasibility ---------------------------------------------------------------
def _scaled(recs: list[dict[str, Any]], sizing: str) -> list[tuple[Any, float, float, float]]:
    """Return per-trade (exit_ts, net_$, mae_$, mfe_$) for a given sizing, time-ordered by exit."""
    out = []
    for r in sorted(recs, key=lambda x: x["exit_ts"]):
        g1 = r["_gross1mnq"]
        mae_pts, mfe_pts = r["mae_pts"], r["mfe_pts"]
        if sizing.endswith("MNQ"):
            k = int(sizing.split()[0])
            net = r["_net1mnq"] * k
            mae_d, mfe_d = mae_pts * MNQ_PV * k, mfe_pts * MNQ_PV * k
        else:  # NQ
            k = int(sizing.split()[0])
            net = (g1 * 10.0 - NQ_COMM_RT) * k
            mae_d, mfe_d = mae_pts * NQ_PV * k, mfe_pts * NQ_PV * k
        out.append((r["exit_ts"], net, mae_d, mfe_d))
    return out


def _simulate(path: list[tuple[Any, float, float, float]], model: str) -> dict[str, Any]:
    """Sequential account sim with intratrade lows (via MAE). model in static|trailing|locked."""
    eq = START_BAL
    peak = START_BAL
    min_buffer = float("inf")
    worst_dd = 0.0
    breach = False
    breach_date = None
    closed_eq = START_BAL
    closed_peak = START_BAL
    closed_dd = 0.0
    for ts, net, mae_d, _mfe_d in path:
        low = eq - mae_d  # intratrade low (1 position; conservative for single strategy)
        if model == "static":
            floor = START_BAL - DD_LIMIT
        elif model == "trailing":
            floor = peak - DD_LIMIT
        else:  # locked at start
            floor = min(peak - DD_LIMIT, START_BAL)
        buf = low - floor
        if buf < min_buffer:
            min_buffer = buf
        if low < floor and not breach:
            breach = True
            breach_date = str(ts)[:10]
        worst_dd = max(worst_dd, peak - low)
        eq += net
        peak = max(peak, eq)
        # closed-trade dd
        closed_eq += net
        closed_peak = max(closed_peak, closed_eq)
        closed_dd = max(closed_dd, closed_peak - closed_eq)
    return {
        "pass": not breach, "ending_equity": round(eq, 2), "net": round(eq - START_BAL, 2),
        "closed_max_dd": round(closed_dd, 2), "intraday_max_dd": round(worst_dd, 2),
        "min_buffer": round(min_buffer, 2), "breach_date": breach_date,
    }


def _worst_seq(nets: list[float], k: int) -> float:
    a = np.array(nets)
    if len(a) < k:
        return float(a.sum())
    c = np.convolve(a, np.ones(k), "valid")
    return float(c.min())


def _monte_carlo(recs: list[dict[str, Any]], sizing: str) -> dict[str, Any]:
    path = _scaled(recs, sizing)
    nets = np.array([p[1] for p in path])
    maes = np.array([p[2] for p in path])
    n = len(nets)
    # reshuffle DD distribution (intratrade-aware)
    dds = np.empty(N_MC)
    for s in range(N_MC):
        idx = RNG.permutation(n)
        ns, ms = nets[idx], maes[idx]
        eq = START_BAL + np.cumsum(ns)
        prev = np.concatenate([[START_BAL], eq[:-1]])
        lows = prev - ms
        peak = np.maximum.accumulate(np.maximum(eq, prev))
        dds[s] = float((peak - lows).max())
    # block bootstrap by month
    months: dict[str, list[float]] = {}
    for r in sorted(recs, key=lambda x: x["exit_ts"]):
        g = r["month"]
        months.setdefault(g, [])
        # recompute net for this sizing per trade
    # simpler: rebuild per-trade net list aligned to month blocks
    rec_sorted = sorted(recs, key=lambda x: x["exit_ts"])
    net_by = {id(r): path[i][1] for i, r in enumerate(rec_sorted)}
    mblocks: dict[str, list[float]] = {}
    for r in rec_sorted:
        mblocks.setdefault(r["month"], []).append(net_by[id(r)])
    mkeys = list(mblocks.values())
    boot_dd = np.empty(N_MC)
    for s in range(N_MC):
        seq: list[float] = []
        while len(seq) < n:
            seq.extend(mkeys[RNG.integers(len(mkeys))])
        arr = np.array(seq[:n])
        eq = START_BAL + np.cumsum(arr)
        peak = np.maximum.accumulate(eq)
        boot_dd[s] = float((peak - eq).max())
    # daily bootstrap for time-based profitability
    days: dict[str, float] = {}
    for i, r in enumerate(rec_sorted):
        days[r["date"]] = days.get(r["date"], 0.0) + path[i][1]
    dvals = np.array(list(days.values()))

    def p_profit(ndays: int) -> float:
        sims = dvals[RNG.integers(len(dvals), size=(N_MC, ndays))].sum(axis=1)
        return float((sims > 0).mean())

    return {
        "sizing": sizing,
        "p_breach_3000": round(float((dds >= 3000).mean()), 4),
        "p_breach_2500": round(float((dds >= 2500).mean()), 4),
        "p_breach_2000": round(float((dds >= 2000).mean()), 4),
        "p_breach_3000_boot": round(float((boot_dd >= 3000).mean()), 4),
        "exp_max_dd": round(float(dds.mean()), 2),
        "p95_max_dd": round(float(np.percentile(dds, 95)), 2),
        "p99_max_dd": round(float(np.percentile(dds, 99)), 2),
        "p_profit_60d": round(p_profit(60), 4),
        "p_profit_120d": round(p_profit(120), 4),
        "p_loss_6mo": round(1 - p_profit(126), 4),
        "worst_20": round(_worst_seq([p[1] for p in path], 20), 2),
        "worst_50": round(_worst_seq([p[1] for p in path], 50), 2),
    }


# ---- plotting -----------------------------------------------------------------------------------
def _equity_dd(recs: list[dict[str, Any]], net_key: str) -> tuple[list[Any], np.ndarray, np.ndarray]:
    rr = sorted(recs, key=lambda x: x["exit_ts"])
    dates = [r["exit_ts"][:10] for r in rr]
    eq = np.cumsum([r[net_key] for r in rr])
    peak = np.maximum.accumulate(eq)
    return dates, eq, peak - eq


def _plot_equity(recs, key, title, fname):
    dates, eq, _ = _equity_dd(recs, key)
    x = np.arange(len(eq))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, eq, lw=1.4, color="#1f6feb")
    ax.fill_between(x, 0, eq, alpha=0.08, color="#1f6feb")
    ax.set_title(f"{title} — equity curve (1 MNQ, net $)  2021→2026H1", fontsize=12)
    ax.set_xlabel("trade #")
    ax.set_ylabel("cumulative net $")
    ax.grid(alpha=0.3)
    _yearlines(ax, recs, x)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=110)
    plt.close(fig)


def _plot_dd(recs, key, title, fname):
    _, _, dd = _equity_dd(recs, key)
    x = np.arange(len(dd))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(x, 0, -dd, color="#d1242f", alpha=0.5)
    ax.axhline(-3000, ls="--", color="black", lw=1, label="$3K limit")
    ax.set_title(f"{title} — drawdown (1 MNQ, $)", fontsize=12)
    ax.set_xlabel("trade #")
    ax.set_ylabel("drawdown $")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=110)
    plt.close(fig)


def _yearlines(ax, recs, x):
    rr = sorted(recs, key=lambda r: r["exit_ts"])
    last = None
    for i, r in enumerate(rr):
        if r["year"] != last:
            ax.axvline(i, color="grey", alpha=0.25, lw=0.8)
            ax.text(i, ax.get_ylim()[1], str(r["year"]), fontsize=8, color="grey", va="top")
            last = r["year"]


def _plot_monthly(allrecs: dict[str, list], fname):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
    for ax, key in zip(axes, ("idb", "swd", "port"), strict=True):
        mt = _month_table(allrecs[key])
        labels = [m["month"] for m in mt]
        vals = [m["pnl"] for m in mt]
        colors = ["#2da44e" if v > 0 else "#d1242f" for v in vals]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_title(f"{key} — monthly net $ (1 MNQ)", fontsize=11)
        ax.grid(alpha=0.3, axis="y")
        step = max(1, len(labels) // 12)
        ax.set_xticks(range(0, len(labels), step))
        ax.set_xticklabels(labels[::step], rotation=45, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=110)
    plt.close(fig)


def _plot_yearly(allrecs: dict[str, list], fname):
    fig, ax = plt.subplots(figsize=(10, 5))
    years = sorted({r["year"] for r in allrecs["idb"]} | {r["year"] for r in allrecs["swd"]})
    w = 0.25
    for i, key in enumerate(("idb", "swd", "port")):
        yt = {row["year"]: row["net"] for row in _year_table(allrecs[key])}
        vals = [yt.get(y, 0) for y in years]
        ax.bar([y + (i - 1) * w for y in years], vals, width=w, label=key)
    ax.set_title("Yearly net $ (1 MNQ) — idb vs swd vs portfolio", fontsize=12)
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_xticks(years)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=110)
    plt.close(fig)


def _plot_rolling(allrecs: dict[str, list], fname):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for key in ("idb", "swd", "port"):
        rr = sorted(allrecs[key], key=lambda r: r["exit_ts"])
        nets = np.array([r["_net1mnq"] for r in rr])
        rs = np.array([r["pnl_R"] for r in rr])
        # rolling 20-trade expectancy ($)
        if len(nets) >= 20:
            roll_exp = np.convolve(nets, np.ones(20) / 20, "valid")
            axes[0].plot(roll_exp, label=key, lw=1.2)
        # rolling 50-trade PF
        if len(nets) >= 50:
            pf = []
            for i in range(50, len(nets) + 1):
                w = nets[i - 50:i]
                gw = w[w > 0].sum()
                gl = -w[w < 0].sum()
                pf.append(gw / gl if gl > 0 else np.nan)
            axes[1].plot(pf, label=key, lw=1.2)
        # rolling drawdown
        eq = np.cumsum(nets)
        dd = np.maximum.accumulate(eq) - eq
        axes[2].plot(dd, label=key, lw=1.2)
    axes[0].set_title("Rolling 20-trade expectancy ($/trade, 1 MNQ)")
    axes[1].set_title("Rolling 50-trade profit factor")
    axes[1].axhline(1.0, ls="--", color="grey", lw=0.8)
    axes[2].set_title("Rolling drawdown ($, 1 MNQ)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=110)
    plt.close(fig)


def _plot_hist_streak(allrecs: dict[str, list]):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes, ("idb", "swd", "port"), strict=True):
        rs = [r["pnl_R"] for r in allrecs[key]]
        ax.hist(rs, bins=40, color="#1f6feb", alpha=0.75)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{key} — trade R distribution")
        ax.set_xlabel("R multiple")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "phase30_trade_distribution.png", dpi=110)
    plt.close(fig)
    # streak chart: signed consecutive run lengths for portfolio
    fig, ax = plt.subplots(figsize=(11, 4))
    rr = sorted(allrecs["port"], key=lambda r: r["exit_ts"])
    runs = []
    cur = 0
    for r in rr:
        s = 1 if r["_net1mnq"] > 0 else -1
        cur = cur + s if (cur == 0 or (cur > 0) == (s > 0)) else s
        runs.append(cur)
    ax.bar(range(len(runs)), runs, color=["#2da44e" if v > 0 else "#d1242f" for v in runs])
    ax.set_title("Portfolio — win/loss streak (running run length)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "phase30_streaks.png", dpi=110)
    plt.close(fig)


def main() -> None:
    trades = _load_trades()
    recs = {k: _records(v, k) for k, v in trades.items()}
    # Portfolio reuses the same record objects in time order — do NOT mutate their fields.
    recs["port"] = sorted(recs["idb"] + recs["swd"], key=lambda r: r["entry_ts"])

    # account_equity_after_trade + drawdown_after_trade (1 MNQ) for the trades CSV
    trade_rows: list[dict[str, Any]] = []
    for key in ("idb", "swd"):
        rr = sorted(recs[key], key=lambda x: x["exit_ts"])
        eq = START_BAL
        peak = START_BAL
        for r in rr:
            eq += r["_net1mnq"]
            peak = max(peak, eq)
            r2 = {k: v for k, v in r.items() if not k.startswith("_")}
            r2["strategy"] = key
            r2["account_equity_after_trade"] = round(eq, 2)
            r2["drawdown_after_trade"] = round(peak - eq, 2)
            trade_rows.append(r2)
    pl.DataFrame(trade_rows).write_csv(OUT / "phase30_finalist_trades.csv")

    panels = {k: _panel(recs[k]) for k in ("idb", "swd", "port")}
    ytabs = {k: _year_table(recs[k]) for k in ("idb", "swd", "port")}
    mtabs = {k: _month_table(recs[k]) for k in ("idb", "swd", "port")}

    # exact stats CSV (one row per finalist) — coerce tuples to strings for tabular output
    def _flatten(d: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in d.items():
            out[k] = f"{v[0]}:{v[1]:.0f}" if isinstance(v, tuple) else v
        return out

    pl.DataFrame([{"finalist": k, **_flatten(panels[k])} for k in ("idb", "swd", "port")],
                 infer_schema_length=None).write_csv(OUT / "phase30_finalists_exact_stats.csv")

    # 100K feasibility
    sizings = ["1 MNQ", "2 MNQ", "5 MNQ", "1 NQ"]
    sim_rows: list[dict[str, Any]] = []
    mc_rows: list[dict[str, Any]] = []
    for key in ("idb", "swd", "port"):
        szs = sizings + (["scaled"] if key == "port" else [])
        for sz in (sizings if key != "port" else ["1 MNQ", "2 MNQ", "5 MNQ"]):
            path = _scaled(recs[key], sz)
            nets = [p[1] for p in path]
            for model in ("static", "trailing", "locked"):
                r = _simulate(path, model)
                sim_rows.append({"strategy": key, "sizing": sz, "dd_model": model, **r,
                                 "worst_20": round(_worst_seq(nets, 20), 2),
                                 "worst_50": round(_worst_seq(nets, 50), 2)})
            mc_rows.append({"strategy": key, **_monte_carlo(recs[key], sz)})
    pl.DataFrame(sim_rows, infer_schema_length=None).write_csv(OUT / "phase30_100k_account_sim.csv")
    pl.DataFrame(mc_rows, infer_schema_length=None).write_csv(OUT / "phase30_monte_carlo.csv")

    # graphs
    _plot_equity(recs["idb"], "_net1mnq", "idb_base", "phase30_idb_equity_curve.png")
    _plot_dd(recs["idb"], "_net1mnq", "idb_base", "phase30_idb_drawdown_curve.png")
    _plot_equity(recs["swd"], "_net1mnq", "swd_clean_path", "phase30_swd_equity_curve.png")
    _plot_dd(recs["swd"], "_net1mnq", "swd_clean_path", "phase30_swd_drawdown_curve.png")
    _plot_equity(recs["port"], "_net1mnq", "portfolio", "phase30_portfolio_equity_curve.png")
    _plot_dd(recs["port"], "_net1mnq", "portfolio", "phase30_portfolio_drawdown_curve.png")
    _plot_monthly(recs, "phase30_monthly_pnl.png")
    _plot_yearly(recs, "phase30_yearly_pnl.png")
    _plot_rolling(recs, "phase30_rolling_stats.png")
    _plot_hist_streak(recs)

    # dump everything for the report writer / chat
    (OUT / "_stats.json").write_text(json.dumps({
        "panels": panels, "year_tables": ytabs, "month_tables": mtabs,
        "sims": sim_rows, "mc": mc_rows, "rules": EXACT_RULES,
    }, indent=2, default=str))
    print("DONE. trades idb/swd/port:", len(recs["idb"]), len(recs["swd"]), len(recs["port"]))
    print("panels:", {k: (panels[k]["trades"], panels[k]["pf_after"], panels[k]["max_closed_dd"]) for k in panels})


if __name__ == "__main__":
    main()
