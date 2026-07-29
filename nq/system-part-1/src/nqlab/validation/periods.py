"""Trade-level summaries and year/month breakdowns."""

from __future__ import annotations

import math
from datetime import date

from nqlab.backtest.result import Trade


def summarize(trades: list[Trade]) -> dict[str, object]:
    """Trade-level metrics (drawdown from the cumulative trade-PnL curve)."""
    ordered = sorted(trades, key=lambda t: t.exit_ts)
    nets = [t.net_pnl for t in ordered]
    n = len(nets)
    net = sum(nets)
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x < 0]
    gross_loss = -sum(losses)
    if gross_loss > 0:
        pf: float = sum(wins) / gross_loss
    elif wins:
        pf = math.inf
    else:
        pf = 0.0

    cum = peak = mdd = 0.0
    for x in nets:
        cum += x
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)

    streak = max_streak = 0
    for x in nets:
        streak = streak + 1 if x < 0 else 0
        max_streak = max(max_streak, streak)

    return {
        "net_pnl": round(net, 2),
        "trades": n,
        "win_rate": round(len(wins) / n, 4) if n else 0.0,
        "profit_factor": pf if math.isfinite(pf) else None,
        "avg_trade": round(net / n, 2) if n else 0.0,
        "max_drawdown": round(mdd, 2),
        "max_losing_streak": max_streak,
    }


def sharpe_skew_kurt(pnls: list[float]) -> tuple[float, float, float, int]:
    """Per-trade Sharpe (mean/std), sample skew, and kurtosis (normal == 3)."""
    n = len(pnls)
    if n < 2:
        return 0.0, 0.0, 3.0, n
    mean = sum(pnls) / n
    m2 = sum((x - mean) ** 2 for x in pnls) / n
    std = math.sqrt(m2)
    if std <= 0:
        return 0.0, 0.0, 3.0, n
    m3 = sum((x - mean) ** 3 for x in pnls) / n
    m4 = sum((x - mean) ** 4 for x in pnls) / n
    return mean / std, m3 / std**3, m4 / m2**2, n


def net_of(metrics: dict[str, object]) -> float:
    from typing import cast

    return cast(float, metrics.get("net_pnl") or 0.0)


def trades_of(metrics: dict[str, object]) -> int:
    from typing import cast

    return cast(int, metrics.get("trades") or 0)


def trades_between(trades: list[Trade], start: date, end: date) -> list[Trade]:
    return [t for t in trades if start <= t.trading_day <= end]


def year_breakdown(trades: list[Trade]) -> dict[str, list[Trade]]:
    out: dict[str, list[Trade]] = {}
    for t in trades:
        out.setdefault(str(t.trading_day.year), []).append(t)
    return dict(sorted(out.items()))


def month_breakdown(trades: list[Trade]) -> dict[str, list[Trade]]:
    out: dict[str, list[Trade]] = {}
    for t in trades:
        key = f"{t.trading_day.year}-{t.trading_day.month:02d}"
        out.setdefault(key, []).append(t)
    return dict(sorted(out.items()))
