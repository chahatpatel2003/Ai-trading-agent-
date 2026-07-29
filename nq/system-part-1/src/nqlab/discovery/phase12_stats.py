"""Phase 12 — full practical statistics for a paper-readiness profile.

Computes the trading-desk stat panel a strategy needs before paper/shadow trading:
win rate, profit factor, expectancy, ROI, drawdown, a Monte-Carlo Topstep pass rate
and expected-days-to-pass per account size, consistency, and rule violations. All from
the realized trade ledger of a single (locked-buffer) backtest plus a reshuffle MC.
"""

from __future__ import annotations

import random
from typing import Any

from nqlab.backtest.result import Trade
from nqlab.config.models import TopstepRules


def _pass_ruin_mc(
    pnls: list[float], rules: TopstepRules, *, runs: int, seed: int
) -> tuple[float, float | None, float]:
    """Reshuffle trade PnLs; return (pass_fraction, avg trades-to-pass, ruin_fraction).

    A path PASSES if cumulative equity reaches start+profit_target before breaching the
    locked-buffer trailing floor; it is RUINED if it breaches first.
    """
    if not pnls:
        return 0.0, None, 0.0
    rng = random.Random(seed)
    start = rules.account_size
    target = start + rules.profit_target
    trailing = rules.trailing_max_loss
    lock = rules.lock_trailing_at_start
    base = list(pnls)
    passes = blows = 0
    ttp: list[int] = []
    for _ in range(runs):
        perm = base[:]
        rng.shuffle(perm)
        equity = peak = start
        for i, p in enumerate(perm):
            equity += p
            peak = max(peak, equity)
            floor = min(peak - trailing, start) if lock else peak - trailing
            if equity <= floor:
                blows += 1
                break
            if equity >= target:
                passes += 1
                ttp.append(i + 1)
                break
    avg_ttp = (sum(ttp) / len(ttp)) if ttp else None
    return passes / runs, avg_ttp, blows / runs


def practical_stats(
    trades: list[Trade],
    *,
    account_size: float,
    rules: TopstepRules,
    months: float,
    rules_by_account: dict[str, TopstepRules] | None = None,
    runs: int = 2000,
    seed: int = 12345,
) -> dict[str, Any]:
    """Full paper-readiness stat panel from a trade ledger (+ reshuffle MC)."""
    n = len(trades)
    if n == 0:
        return {"trade_count": 0}
    nets = [t.net_pnl for t in trades]
    net = sum(nets)
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x < 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)

    by_day: dict[Any, float] = {}
    for t in trades:
        by_day[t.trading_day] = by_day.get(t.trading_day, 0.0) + t.net_pnl
    daily = list(by_day.values())
    pos_total = sum(d for d in daily if d > 0)
    best_day = max(daily) if daily else 0.0
    trades_per_day = n / len(by_day) if by_day else 0.0

    # Drawdown profile from the trade-sequence equity curve.
    eq = peak = 0.0
    dds: list[float] = []
    for x in nets:
        eq += x
        peak = max(peak, eq)
        dds.append(peak - eq)
    max_dd = max(dds) if dds else 0.0
    avg_dd = sum(dds) / len(dds) if dds else 0.0

    pass_frac, avg_ttp, ruin = _pass_ruin_mc(nets, rules, runs=runs, seed=seed)
    days_to_pass = (avg_ttp / trades_per_day) if (avg_ttp and trades_per_day > 0) else None

    p_pass: dict[str, float] = {}
    for label, r in (rules_by_account or {}).items():
        p_pass[label], _, _ = _pass_ruin_mc(nets, r, runs=runs, seed=seed)

    monthly = net / months if months > 0 else net
    return {
        "trade_count": n,
        "win_rate": round(len(wins) / n, 4),
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else None,
        "expectancy": round(net / n, 2),
        "avg_trade": round(net / n, 2),
        "net_pnl": round(net, 2),
        "max_drawdown": round(max_dd, 2),
        "avg_drawdown": round(avg_dd, 2),
        "monthly_pnl": round(monthly, 2),
        "monthly_roi": round(monthly / account_size, 4),
        "annual_roi": round(12.0 * monthly / account_size, 4),
        "ruin": round(ruin, 4),
        "topstep_pass_rate": round(pass_frac, 4),
        "p_pass_by_account": {k: round(v, 4) for k, v in p_pass.items()},
        "avg_trades_to_pass": round(avg_ttp, 1) if avg_ttp else None,
        "est_days_to_pass": round(days_to_pass, 1) if days_to_pass else None,
        "est_monthly_payout": round(
            monthly, 2
        ),  # gross monthly profit (Topstep split out of scope)
        "largest_win_day_pct": round(best_day / net, 4) if net > 0 else None,
        "consistency_pct": round(best_day / pos_total, 4) if pos_total > 0 else None,
        "trades_per_day": round(trades_per_day, 2),
    }
