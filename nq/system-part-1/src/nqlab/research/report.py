"""Backtest metrics and artifact writing (trades / equity / metrics)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl

from nqlab.backtest.result import BacktestResult


def compute_metrics(result: BacktestResult) -> dict[str, object]:
    """Compute the Phase-1 report metrics from a backtest result."""
    trades = result.trades
    nets = [t.net_pnl for t in trades]
    n = len(nets)
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x < 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    if gross_loss > 0:
        profit_factor: float = gross_win / gross_loss
    elif gross_win > 0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0

    net_pnl = result.final_equity - result.starting_balance
    avg_trade = sum(nets) / n if n else 0.0

    streak = max_streak = 0
    for x in nets:
        streak = streak + 1 if x < 0 else 0
        max_streak = max(max_streak, streak)

    eq = result.equity_curve
    if eq.height:
        peak = eq.select((pl.col("equity").cum_max() - pl.col("equity")).max()).item()
        daily = eq.select(
            (pl.col("equity").cum_max().over("trading_day") - pl.col("equity")).max()
        ).item()
        max_dd = float(peak or 0.0)
        max_daily_dd = float(daily or 0.0)
        trading_days = int(eq["trading_day"].n_unique())
    else:
        max_dd = max_daily_dd = 0.0
        trading_days = 0

    ts = result.topstep
    return {
        "net_pnl": round(net_pnl, 2),
        "trades": n,
        "win_rate": round(len(wins) / n, 4) if n else 0.0,
        "profit_factor": profit_factor if math.isfinite(profit_factor) else None,
        "expectancy": round(avg_trade, 2),
        "avg_trade": round(avg_trade, 2),
        "trades_per_day": round(n / trading_days, 3) if trading_days else 0.0,
        "max_drawdown": round(max_dd, 2),
        "max_daily_drawdown": round(max_daily_dd, 2),
        "max_losing_streak": max_streak,
        "trading_days": trading_days,
        "passed": ts.passed if ts else None,
        "days_to_pass": ts.days_to_pass if ts else None,
        "blown": ts.blown if ts else None,
        "account_failures": ts.account_failures if ts else 0,
        "consistency_ok": ts.consistency_ok if ts else None,
        "best_day": round(ts.best_day, 2) if ts else None,
        "worst_day": round(ts.worst_day, 2) if ts else None,
        "violations": ts.violations if ts else [],
    }


def write_report(
    result: BacktestResult, metrics: dict[str, object], out_dir: Path
) -> dict[str, str]:
    """Write trades, equity curve (parquet + csv), and metrics json to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    trades = result.trades_frame()
    if trades.height:
        trades.write_parquet(out_dir / "trades.parquet")
        trades.write_csv(out_dir / "trades.csv")
        paths["trades_parquet"] = str(out_dir / "trades.parquet")
        paths["trades_csv"] = str(out_dir / "trades.csv")

    result.equity_curve.write_parquet(out_dir / "equity.parquet")
    result.equity_curve.write_csv(out_dir / "equity.csv")
    paths["equity_parquet"] = str(out_dir / "equity.parquet")
    paths["equity_csv"] = str(out_dir / "equity.csv")

    report = {
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "strategy": result.strategy.name,
        "strategy_version": result.strategy.version,
        "strategy_params": result.strategy.as_dict(),
        "strategy_fingerprint": result.strategy.fingerprint(),
        "metrics": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    paths["metrics_json"] = str(out_dir / "metrics.json")
    return paths
