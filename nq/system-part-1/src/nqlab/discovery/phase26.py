"""Phase 26 — non-Topstep high-profit-factor strategy acceptance.

The Topstep payout framework is paused. Here the goal is the highest-quality NQ edge: PF >= 2.0
after costs, controlled drawdown, return/maxDD >= 3, Sharpe >= 1 — ranked by PF and return-to-
drawdown, NOT raw PnL or Topstep pass. Sessions/timeframes/holding periods are unrestricted.
No forced acceptance; weak PF is rejected.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any

from nqlab.backtest.result import Trade

ACCEPT_FOR_PAPER = "ACCEPT_FOR_PAPER"
ACCEPT_FOR_FORWARD_PAPER = "ACCEPT_FOR_FORWARD_PAPER"
NEEDS_MORE_DATA = "NEEDS_MORE_DATA"
REJECT = "REJECT"


@dataclass(frozen=True)
class Phase26Thresholds:
    """High-PF, low-drawdown acceptance bar (after realistic costs)."""

    pf_min: float = 2.00
    pf_preferred: float = 2.50
    ret_dd_min: float = 3.00      # net / max drawdown
    sharpe_min: float = 1.00
    calmar_pref: float = 2.00
    win_months_min: float = 0.70
    min_trades: int = 50
    cost_stress_pf_min: float = 1.50  # PF must hold above this at 2x cost


PHASE26_THRESHOLDS = Phase26Thresholds()


def _losing_streak(nets: list[float]) -> int:
    cur = longest = 0
    for x in nets:
        cur = cur + 1 if x < 0 else 0
        longest = max(longest, cur)
    return longest


def quality_stats(trades: list[Trade], *, years: float) -> dict[str, Any]:
    """PF / return-to-drawdown / Sharpe / Sortino / Calmar / winning-months panel (1-contract $)."""
    n = len(trades)
    if n == 0:
        return {"trade_count": 0}
    ts = sorted(trades, key=lambda t: t.entry_ts)
    nets = [t.net_pnl for t in ts]
    net = sum(nets)
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x < 0]
    gross_win, gross_loss = sum(wins), -sum(losses)
    pf = gross_win / gross_loss if gross_loss > 0 else None

    # Drawdown from the trade-sequence equity curve.
    eq = peak = 0.0
    dds: list[float] = []
    for x in nets:
        eq += x
        peak = max(peak, eq)
        dds.append(peak - eq)
    max_dd = max(dds) if dds else 0.0
    avg_dd = sum(dds) / len(dds) if dds else 0.0

    # Daily P&L series -> Sharpe / Sortino (annualised over 252 trading days).
    by_day: dict[date, float] = defaultdict(float)
    by_month: dict[str, float] = defaultdict(float)
    for t in ts:
        by_day[t.trading_day] += t.net_pnl
        by_month[t.entry_ts.strftime("%Y-%m")] += t.net_pnl
    daily = list(by_day.values())
    mean_d = sum(daily) / len(daily) if daily else 0.0
    sd = math.sqrt(sum((d - mean_d) ** 2 for d in daily) / len(daily)) if len(daily) > 1 else 0.0
    downside = [d for d in daily if d < 0]
    dsd = math.sqrt(sum(d * d for d in downside) / len(downside)) if downside else 0.0
    sharpe = (mean_d / sd) * math.sqrt(252) if sd > 0 else 0.0
    sortino = (mean_d / dsd) * math.sqrt(252) if dsd > 0 else 0.0
    ann_return = net / years if years > 0 else net
    calmar = ann_return / max_dd if max_dd > 0 else None
    mvals = list(by_month.values())
    green_months = sum(1 for v in mvals if v > 0)

    return {
        "trade_count": n,
        "profit_factor": round(pf, 3) if pf is not None else None,
        "win_rate": round(len(wins) / n, 4),
        "expectancy": round(net / n, 2),
        "avg_win": round(gross_win / len(wins), 2) if wins else 0.0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
        "net": round(net, 2),
        "max_drawdown": round(max_dd, 2),
        "avg_drawdown": round(avg_dd, 2),
        "return_over_maxdd": round(net / max_dd, 3) if max_dd > 0 else None,
        "ann_return": round(ann_return, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3) if calmar is not None else None,
        "green_months": green_months,
        "total_months": len(mvals),
        "win_month_frac": round(green_months / len(mvals), 4) if mvals else 0.0,
        "best_month": round(max(mvals), 2) if mvals else 0.0,
        "worst_month": round(min(mvals), 2) if mvals else 0.0,
        "max_losing_streak": _losing_streak(nets),
        "avg_mfe": round(sum(t.mfe for t in ts) / n, 2),
        "avg_mae": round(sum(t.mae for t in ts) / n, 2),
    }


def phase26_verdict(
    *,
    val_stats: dict[str, Any],
    lockbox_stats: dict[str, Any] | None,
    cost2x_pf: float | None,
    thresholds: Phase26Thresholds = PHASE26_THRESHOLDS,
    pristine: bool,
) -> tuple[str, str]:
    """High-PF verdict: PF>=2.0 + return/DD>=3 + Sharpe>=1 + winning months, on val AND lockbox."""
    th = thresholds
    if val_stats.get("trade_count", 0) < th.min_trades:
        return NEEDS_MORE_DATA, f"only {val_stats.get('trade_count', 0)} validation trades"
    fails: list[str] = []
    pf = val_stats.get("profit_factor")
    if pf is None or pf < th.pf_min:
        fails.append(f"validation PF {pf} < {th.pf_min}")
    if val_stats.get("net", 0) <= 0:
        fails.append("validation net <= 0")
    rdd = val_stats.get("return_over_maxdd")
    if rdd is None or rdd < th.ret_dd_min:
        fails.append(f"return/maxDD {rdd} < {th.ret_dd_min}")
    if val_stats.get("sharpe", 0) < th.sharpe_min:
        fails.append(f"Sharpe {val_stats.get('sharpe')} < {th.sharpe_min}")
    if val_stats.get("win_month_frac", 0) < th.win_months_min:
        fails.append(f"winning months {val_stats.get('win_month_frac')} < {th.win_months_min}")
    if cost2x_pf is not None and cost2x_pf < th.cost_stress_pf_min:
        fails.append(f"PF collapses to {cost2x_pf} at 2x cost")
    if fails:
        return REJECT, "; ".join(fails)
    if lockbox_stats is not None:
        lpf = lockbox_stats.get("profit_factor")
        if lockbox_stats.get("trade_count", 0) < 20:
            return NEEDS_MORE_DATA, "too few lockbox trades to confirm"
        if lpf is None or lpf < th.pf_min or lockbox_stats.get("net", 0) <= 0:
            return REJECT, f"lockbox fails (PF {lpf}, net {lockbox_stats.get('net')})"
    verdict = ACCEPT_FOR_PAPER if pristine else ACCEPT_FOR_FORWARD_PAPER
    return verdict, (
        f"PF {pf}, return/maxDD {rdd}, Sharpe {val_stats.get('sharpe')} clear the high-PF bar "
        f"on validation{' and lockbox' if lockbox_stats else ''}"
    )
