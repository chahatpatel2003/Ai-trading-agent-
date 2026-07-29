"""
Walk-Forward Optimization.
Splits data into rolling in-sample / out-of-sample windows.
Trains (generates signals) on in-sample, evaluates on out-of-sample.
Prevents overfitting by never testing on training data.
"""
from dataclasses import dataclass

import pandas as pd

from backtest.engine import BacktestEngine, BacktestResult
from backtest.metrics import PerformanceMetrics, compute_metrics
from logs.logger import get_logger
from strategies.base import BaseStrategy

log = get_logger("backtest.walk_forward")


@dataclass
class WalkForwardResult:
    strategy_name: str
    oos_results: list[BacktestResult]
    combined_metrics: PerformanceMetrics
    window_details: list[dict]


def walk_forward_test(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    cfg: dict,
) -> WalkForwardResult:
    """
    Perform walk-forward analysis.
    - Slide a window across the data
    - For each step: train on in-sample, test on out-of-sample
    - Aggregate out-of-sample results for unbiased evaluation
    """
    wf_cfg = cfg["backtest"]["walk_forward"]
    is_bars = wf_cfg["in_sample_bars"]
    oos_bars = wf_cfg["out_of_sample_bars"]
    step = wf_cfg["step_bars"]

    engine = BacktestEngine(cfg)
    oos_results = []
    window_details = []
    all_oos_trades = []
    all_oos_equity = []

    start = 0
    window_num = 0

    while start + is_bars + oos_bars <= len(df):
        is_end = start + is_bars
        oos_end = is_end + oos_bars

        is_data = df.iloc[start:is_end]
        oos_data = df.iloc[is_end:oos_end]

        log.info(
            "walk_forward_window",
            window=window_num,
            is_range=f"{is_data.index[0]} to {is_data.index[-1]}",
            oos_range=f"{oos_data.index[0]} to {oos_data.index[-1]}",
        )

        # Run backtest on out-of-sample data
        # (In a real optimization, you'd tune params on is_data first,
        #  then apply those params to oos_data. Here we use fixed params.)
        result = engine.run(strategy, oos_data)
        oos_results.append(result)
        all_oos_trades.extend(result.trades)
        all_oos_equity.append(result.equity_curve)

        window_details.append({
            "window": window_num,
            "is_start": str(is_data.index[0]),
            "is_end": str(is_data.index[-1]),
            "oos_start": str(oos_data.index[0]),
            "oos_end": str(oos_data.index[-1]),
            "oos_sharpe": result.metrics.sharpe_ratio,
            "oos_return": result.metrics.total_return_pct,
            "oos_max_dd": result.metrics.max_drawdown_pct,
            "oos_trades": result.metrics.num_trades,
        })

        start += step
        window_num += 1

    if not oos_results:
        raise ValueError("Not enough data for walk-forward analysis")

    # Combine all OOS equity curves
    combined_equity = pd.concat(all_oos_equity)
    combined_equity = combined_equity[~combined_equity.index.duplicated(keep="last")]
    combined_equity.sort_index(inplace=True)

    combined_metrics = compute_metrics(
        combined_equity,
        all_oos_trades,
        cfg["backtest"]["initial_equity"],
        252 * 23,
    )

    log.info(
        "walk_forward_complete",
        strategy=strategy.name,
        windows=window_num,
        combined_sharpe=combined_metrics.sharpe_ratio,
        combined_return=combined_metrics.total_return_pct,
    )

    return WalkForwardResult(
        strategy_name=strategy.name,
        oos_results=oos_results,
        combined_metrics=combined_metrics,
        window_details=window_details,
    )
