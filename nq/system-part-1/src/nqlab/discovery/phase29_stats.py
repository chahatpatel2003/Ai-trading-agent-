"""Phase 29 — metric panel computed from a stored per-trade table.

Every Phase-29 candidate is run ONCE over full history; the full trade list is persisted with each
trade's entry timestamp, net/gross P&L, MFE and MAE. All downstream views (full-history, per-year,
per-month, leave-one-year-out, walk-forward) are then pure functions of a *subset* of those trades —
no re-running, no leakage. This module computes the panel for any such subset.

Metrics are reported for the clean comparable unit used by the factory: **1 MNQ contract** with the
audited cost model. Dollar-based Sharpe/Sortino/Calmar are computed on the **daily** P&L series
(annualized with 252 trading days); ``return_over_maxdd`` is total net / max drawdown (both $).
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl

_ANNUAL = 252.0


def _streaks(nets: list[float]) -> tuple[int, int]:
    """(longest losing streak, longest winning streak)."""
    cur_l = cur_w = max_l = max_w = 0
    for x in nets:
        if x < 0:
            cur_l += 1
            cur_w = 0
        elif x > 0:
            cur_w += 1
            cur_l = 0
        else:
            cur_l = cur_w = 0
        max_l = max(max_l, cur_l)
        max_w = max(max_w, cur_w)
    return max_l, max_w


def _drawdown(equity: list[float]) -> tuple[float, float]:
    """(max drawdown, average drawdown) on a cumulative-net equity path that starts at 0."""
    peak = 0.0
    max_dd = 0.0
    dd_sum = 0.0
    for e in equity:
        peak = max(peak, e)
        dd = peak - e
        max_dd = max(max_dd, dd)
        dd_sum += dd
    avg_dd = dd_sum / len(equity) if equity else 0.0
    return max_dd, avg_dd


def compute_panel(trades: pl.DataFrame) -> dict[str, Any]:
    """Full metric panel for a trade table sorted (or sortable) by ``entry_ts``.

    Required columns: ``entry_ts`` (datetime), ``trading_day`` (date), ``net`` ($), ``mfe``, ``mae``
    (points, favourable/adverse excursion magnitudes). Returns a flat dict of metrics; an empty
    table yields ``{"n": 0}``.
    """
    if trades.height == 0:
        return {"n": 0}
    t = trades.sort("entry_ts")
    nets = t["net"].to_list()
    n = len(nets)
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x < 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    net_total = sum(nets)

    equity = []
    run = 0.0
    for x in nets:
        run += x
        equity.append(run)
    max_dd, avg_dd = _drawdown(equity)
    max_lstreak, max_wstreak = _streaks(nets)

    # Daily P&L series for risk ratios.
    daily = (
        t.group_by("trading_day")
        .agg(pl.col("net").sum().alias("d"))
        .sort("trading_day")["d"]
        .to_list()
    )
    n_days = len(daily)
    mean_d = sum(daily) / n_days if n_days else 0.0
    if n_days > 1:
        var_d = sum((x - mean_d) ** 2 for x in daily) / (n_days - 1)
        std_d = math.sqrt(var_d)
        downs = [x for x in daily if x < 0]
        dstd = math.sqrt(sum(x * x for x in downs) / n_days) if downs else 0.0
    else:
        std_d = dstd = 0.0
    sharpe = (mean_d / std_d * math.sqrt(_ANNUAL)) if std_d > 0 else None
    sortino = (mean_d / dstd * math.sqrt(_ANNUAL)) if dstd > 0 else None
    span_days = max(n_days, 1)
    annual_net = net_total / span_days * _ANNUAL
    calmar = (annual_net / max_dd) if max_dd > 0 else None

    mfe = t["mfe"].mean()
    mae = t["mae"].mean()
    pf = (gross_win / gross_loss) if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)
    return {
        "n": n,
        "net": round(net_total, 2),
        "pf": round(pf, 3) if math.isfinite(pf) else None,
        "win_rate": round(len(wins) / n, 4),
        "expectancy": round(net_total / n, 2),
        "avg_win": round(gross_win / len(wins), 2) if wins else 0.0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "max_dd": round(max_dd, 2),
        "avg_dd": round(avg_dd, 2),
        "ret_over_maxdd": round(net_total / max_dd, 3) if max_dd > 0 else None,
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "sortino": round(sortino, 3) if sortino is not None else None,
        "calmar": round(calmar, 3) if calmar is not None else None,
        "max_losing_streak": max_lstreak,
        "max_winning_streak": max_wstreak,
        "avg_mfe": round(mfe, 2) if mfe is not None else None,
        "avg_mae": round(mae, 2) if mae is not None else None,
        "mfe_mae": round(mfe / mae, 3) if mae and mae > 0 else None,
        "trading_days": n_days,
        "trades_per_day": round(n / n_days, 3) if n_days else None,
    }


def by_year(trades: pl.DataFrame) -> dict[int, dict[str, Any]]:
    """Panel per calendar year (by entry_ts)."""
    if trades.height == 0:
        return {}
    t = trades.with_columns(pl.col("entry_ts").dt.year().alias("_y"))
    out: dict[int, dict[str, Any]] = {}
    for y in sorted(t["_y"].unique().to_list()):
        out[int(y)] = compute_panel(t.filter(pl.col("_y") == y).drop("_y"))
    return out


def monthly_pnl(trades: pl.DataFrame) -> dict[str, float]:
    """Net P&L per calendar month, keyed ``YYYY-MM`` (by entry_ts)."""
    if trades.height == 0:
        return {}
    t = trades.with_columns(pl.col("entry_ts").dt.strftime("%Y-%m").alias("_m"))
    agg = t.group_by("_m").agg(pl.col("net").sum()).sort("_m")
    return {r["_m"]: round(r["net"], 2) for r in agg.to_dicts()}


def month_stats(trades: pl.DataFrame) -> dict[str, Any]:
    """Winning-month %, red months, best/worst month, and profit concentration."""
    m = monthly_pnl(trades)
    if not m:
        return {"months": 0}
    vals = list(m.values())
    pos = [v for v in vals if v > 0]
    total = sum(vals)
    best_k = max(m, key=lambda k: m[k])
    worst_k = min(m, key=lambda k: m[k])
    # Concentration: share of total net from the single best month / best year.
    best_month_share = (m[best_k] / total) if total > 0 else None
    return {
        "months": len(vals),
        "winning_months": len(pos),
        "winning_month_pct": round(len(pos) / len(vals), 4),
        "red_months": len(vals) - len(pos),
        "best_month": f"{best_k}:{m[best_k]:.0f}",
        "worst_month": f"{worst_k}:{m[worst_k]:.0f}",
        "best_month_share": round(best_month_share, 3) if best_month_share is not None else None,
    }


def year_concentration(trades: pl.DataFrame) -> float | None:
    """Share of total net P&L contributed by the single most profitable year (smoothness guard)."""
    yr = by_year(trades)
    nets = [float(v.get("net", 0.0) or 0.0) for v in yr.values()]
    total = sum(nets)
    if total <= 0 or not nets:
        return None
    return round(max(nets) / total, 3)
