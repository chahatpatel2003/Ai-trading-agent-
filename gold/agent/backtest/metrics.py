"""
Performance metrics computation.
All metrics operate on a trade log or equity curve — no forward-looking data.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    win_rate_pct: float
    expectancy_per_trade: float
    exposure_time_pct: float
    num_trades: int
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    stability: float  # R-squared of equity curve linear fit


def compute_metrics(
    equity_curve: pd.Series,
    trades: list[dict],
    initial_equity: float,
    bars_per_year: float = 252 * 6.5,  # hourly bars for equities; adjust for gold
) -> PerformanceMetrics:
    """Compute all performance metrics from equity curve and trade log."""

    # Total return
    final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_equity
    total_return_pct = ((final_equity - initial_equity) / initial_equity) * 100

    # CAGR
    n_bars = len(equity_curve)
    years = n_bars / bars_per_year if bars_per_year > 0 else 1
    if years > 0 and final_equity > 0 and initial_equity > 0:
        cagr = (final_equity / initial_equity) ** (1 / years) - 1
    else:
        cagr = 0.0
    cagr_pct = cagr * 100

    # Drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown_pct = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0.0

    # Returns series
    returns = equity_curve.pct_change().dropna()

    # Sharpe ratio (annualized)
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    # Sortino ratio (annualized, using downside deviation)
    downside = returns[returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = (returns.mean() / downside.std()) * np.sqrt(bars_per_year)
    else:
        sortino = 0.0

    # Trade-level metrics
    pnls = [t["pnl"] for t in trades] if trades else []
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    num_trades = len(pnls)
    win_rate = (len(wins) / num_trades * 100) if num_trades > 0 else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    # Profit factor
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    # Expectancy per trade
    expectancy = np.mean(pnls) if pnls else 0.0

    # Exposure time
    in_trade_bars = sum(t.get("bars_held", 1) for t in trades)
    exposure_pct = (in_trade_bars / n_bars * 100) if n_bars > 0 else 0.0

    # Consecutive wins/losses
    max_consec_wins = _max_consecutive(pnls, positive=True)
    max_consec_losses = _max_consecutive(pnls, positive=False)

    # Stability: R-squared of equity curve vs. linear fit
    stability = _equity_stability(equity_curve)

    return PerformanceMetrics(
        total_return_pct=round(total_return_pct, 2),
        cagr_pct=round(cagr_pct, 2),
        max_drawdown_pct=round(max_drawdown_pct, 2),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        profit_factor=round(profit_factor, 3),
        win_rate_pct=round(win_rate, 2),
        expectancy_per_trade=round(expectancy, 2),
        exposure_time_pct=round(exposure_pct, 2),
        num_trades=num_trades,
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        max_consecutive_wins=max_consec_wins,
        max_consecutive_losses=max_consec_losses,
        stability=round(stability, 4),
    )


def _max_consecutive(pnls: list[float], positive: bool) -> int:
    max_count = 0
    current = 0
    for p in pnls:
        if (positive and p > 0) or (not positive and p <= 0):
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count


def _equity_stability(equity_curve: pd.Series) -> float:
    """R-squared of log equity vs time — higher = more stable growth."""
    if len(equity_curve) < 10:
        return 0.0
    try:
        log_eq = np.log(equity_curve.values[equity_curve.values > 0])
        if len(log_eq) < 10:
            return 0.0
        x = np.arange(len(log_eq))
        coeffs = np.polyfit(x, log_eq, 1)
        fitted = np.polyval(coeffs, x)
        ss_res = np.sum((log_eq - fitted) ** 2)
        ss_tot = np.sum((log_eq - np.mean(log_eq)) ** 2)
        return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    except (ValueError, np.linalg.LinAlgError):
        return 0.0
