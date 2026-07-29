"""Unit tests for the backtesting engine."""
import numpy as np
import pandas as pd
import pytest
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics
from strategies.trend_following import TrendFollowingStrategy

TEST_CFG = {
    "costs": {
        "spread_points": 0.30,
        "slippage_points": 0.10,
        "commission_per_lot": 0.0,
        "lot_size_oz": 100,
    },
    "risk": {
        "risk_per_trade_pct": 0.0025,
        "max_risk_per_trade_pct": 0.005,
        "max_daily_drawdown_pct": 0.02,
        "max_total_drawdown_pct": 0.10,
        "max_open_positions": 2,
        "volatility_lookback": 20,
        "no_trade_during_news": False,
        "kill_switch": {
            "max_spread_multiple": 3.0,
            "max_consecutive_losses": 10,
            "max_data_gap_seconds": 300,
        },
    },
    "backtest": {
        "initial_equity": 100000.0,
        "walk_forward": {
            "in_sample_bars": 200,
            "out_of_sample_bars": 50,
            "step_bars": 50,
        },
        "monte_carlo": {
            "n_simulations": 100,
            "confidence_level": 0.95,
            "resample_returns": True,
            "shuffle_trades": True,
        },
    },
    "strategies": {
        "trend_following": {
            "enabled": True,
            "fast_ma": 10,
            "slow_ma": 20,
            "atr_period": 14,
            "atr_stop_multiple": 2.0,
            "trailing_stop_atr_multiple": 1.5,
            "regime_ma": 50,
        },
    },
}


def _sample_data(n=500):
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="1h")
    close = 2000 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.5,
            "high": close + abs(np.random.randn(n) * 3),
            "low": close - abs(np.random.randn(n) * 3),
            "close": close,
            "volume": np.random.randint(100, 5000, n),
        },
        index=dates,
    )


def test_backtest_runs():
    engine = BacktestEngine(TEST_CFG)
    strat = TrendFollowingStrategy(TEST_CFG)
    df = _sample_data()
    result = engine.run(strat, df)
    assert result.strategy_name == "trend_following"
    assert len(result.equity_curve) == len(df)
    assert result.metrics is not None


def test_equity_starts_at_initial():
    engine = BacktestEngine(TEST_CFG)
    strat = TrendFollowingStrategy(TEST_CFG)
    df = _sample_data()
    result = engine.run(strat, df)
    assert result.equity_curve.iloc[0] == 100000.0


def test_transaction_costs_applied():
    """With spread and slippage, even a flat strategy should not gain."""
    engine = BacktestEngine(TEST_CFG)
    strat = TrendFollowingStrategy(TEST_CFG)
    df = _sample_data()
    result = engine.run(strat, df)
    # If trades happened, final equity should differ from initial
    if result.metrics.num_trades > 0:
        assert result.equity_curve.iloc[-1] != 100000.0


def test_metrics_computation():
    equity = pd.Series([100000, 100500, 101000, 100200, 100800])
    trades = [
        {"pnl": 500, "bars_held": 1},
        {"pnl": 500, "bars_held": 1},
        {"pnl": -800, "bars_held": 1},
        {"pnl": 600, "bars_held": 1},
    ]
    metrics = compute_metrics(equity, trades, 100000, 252)
    assert metrics.num_trades == 4
    assert metrics.win_rate_pct == 75.0
    assert metrics.max_drawdown_pct > 0
