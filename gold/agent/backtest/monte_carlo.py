"""
Monte Carlo Robustness Testing.
Resamples trade returns and shuffles trade order to stress-test results.
Produces confidence intervals for key metrics.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from logs.logger import get_logger

log = get_logger("backtest.monte_carlo")


@dataclass
class MonteCarloResult:
    strategy_name: str
    n_simulations: int
    confidence_level: float
    # Distributions of key metrics across simulations
    sharpe_distribution: np.ndarray
    max_dd_distribution: np.ndarray
    return_distribution: np.ndarray
    # Confidence interval bounds
    sharpe_ci: tuple[float, float]
    max_dd_ci: tuple[float, float]
    return_ci: tuple[float, float]
    # Worst-case estimates
    worst_sharpe: float
    worst_max_dd: float
    worst_return: float
    # Probability of ruin (drawdown > threshold)
    prob_ruin_10pct: float
    prob_ruin_20pct: float


def monte_carlo_test(
    trades: list[dict],
    initial_equity: float,
    n_simulations: int = 1000,
    confidence_level: float = 0.95,
    strategy_name: str = "unknown",
) -> MonteCarloResult:
    """
    Run Monte Carlo simulations by reshuffling trade order.
    This tests whether the strategy's performance depends on specific trade sequencing.
    """
    if not trades:
        return _empty_result(strategy_name, n_simulations, confidence_level)

    pnls = np.array([t["pnl"] for t in trades])
    n_trades = len(pnls)

    log.info(
        "monte_carlo_start",
        strategy=strategy_name,
        simulations=n_simulations,
        trades=n_trades,
    )

    sharpes = []
    max_dds = []
    returns = []
    ruin_10 = 0
    ruin_20 = 0

    rng = np.random.default_rng(42)  # reproducible

    for _ in range(n_simulations):
        # Shuffle trade order
        shuffled = rng.permutation(pnls)

        # Build equity curve
        equity = np.empty(n_trades + 1)
        equity[0] = initial_equity
        for j, p in enumerate(shuffled):
            equity[j + 1] = equity[j] + p

        # Calculate metrics for this simulation
        eq_series = pd.Series(equity)
        ret = (equity[-1] - initial_equity) / initial_equity
        returns.append(ret * 100)

        # Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / np.where(running_max > 0, running_max, 1)
        max_dd = abs(drawdowns.min()) * 100
        max_dds.append(max_dd)

        if max_dd >= 10:
            ruin_10 += 1
        if max_dd >= 20:
            ruin_20 += 1

        # Sharpe (simplified: mean/std of trade returns)
        trade_returns = shuffled / initial_equity
        if trade_returns.std() > 0:
            sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
        sharpes.append(sharpe)

    sharpes = np.array(sharpes)
    max_dds = np.array(max_dds)
    returns = np.array(returns)

    alpha = (1 - confidence_level) / 2

    result = MonteCarloResult(
        strategy_name=strategy_name,
        n_simulations=n_simulations,
        confidence_level=confidence_level,
        sharpe_distribution=sharpes,
        max_dd_distribution=max_dds,
        return_distribution=returns,
        sharpe_ci=(
            round(float(np.percentile(sharpes, alpha * 100)), 3),
            round(float(np.percentile(sharpes, (1 - alpha) * 100)), 3),
        ),
        max_dd_ci=(
            round(float(np.percentile(max_dds, alpha * 100)), 2),
            round(float(np.percentile(max_dds, (1 - alpha) * 100)), 2),
        ),
        return_ci=(
            round(float(np.percentile(returns, alpha * 100)), 2),
            round(float(np.percentile(returns, (1 - alpha) * 100)), 2),
        ),
        worst_sharpe=round(float(sharpes.min()), 3),
        worst_max_dd=round(float(max_dds.max()), 2),
        worst_return=round(float(returns.min()), 2),
        prob_ruin_10pct=round(ruin_10 / n_simulations, 4),
        prob_ruin_20pct=round(ruin_20 / n_simulations, 4),
    )

    log.info(
        "monte_carlo_complete",
        strategy=strategy_name,
        sharpe_ci=result.sharpe_ci,
        max_dd_ci=result.max_dd_ci,
        prob_ruin_10=result.prob_ruin_10pct,
    )

    return result


def _empty_result(name, n_sim, conf):
    empty = np.array([0.0])
    return MonteCarloResult(
        strategy_name=name,
        n_simulations=n_sim,
        confidence_level=conf,
        sharpe_distribution=empty,
        max_dd_distribution=empty,
        return_distribution=empty,
        sharpe_ci=(0.0, 0.0),
        max_dd_ci=(0.0, 0.0),
        return_ci=(0.0, 0.0),
        worst_sharpe=0.0,
        worst_max_dd=0.0,
        worst_return=0.0,
        prob_ruin_10pct=0.0,
        prob_ruin_20pct=0.0,
    )
