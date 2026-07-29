"""Unit tests for strategy signal generation."""
import numpy as np
import pandas as pd
import pytest
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy

# Minimal config
TEST_CFG = {
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
        "mean_reversion": {
            "enabled": True,
            "bb_period": 20,
            "bb_std": 2.0,
            "zscore_entry": -2.0,
            "zscore_exit": 0.0,
            "volatility_filter_period": 20,
            "min_volatility_percentile": 20,
            "max_volatility_percentile": 80,
        },
        "momentum": {
            "enabled": True,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_period": 14,
            "adx_threshold": 25,
        },
    },
}


def _sample_ohlcv(n=500):
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


def test_trend_following_signals_valid():
    strat = TrendFollowingStrategy(TEST_CFG)
    df = _sample_ohlcv()
    result = strat.generate_signals(df)
    assert "signal" in result.columns
    assert "stop_price" in result.columns
    assert set(result["signal"].dropna().unique()).issubset({-1, 0, 1})


def test_mean_reversion_signals_valid():
    strat = MeanReversionStrategy(TEST_CFG)
    df = _sample_ohlcv()
    result = strat.generate_signals(df)
    assert "signal" in result.columns
    assert set(result["signal"].dropna().unique()).issubset({-1, 0, 1})


def test_momentum_signals_valid():
    strat = MomentumStrategy(TEST_CFG)
    df = _sample_ohlcv()
    result = strat.generate_signals(df)
    assert "signal" in result.columns
    assert set(result["signal"].dropna().unique()).issubset({-1, 0, 1})


def test_no_lookahead_trend():
    """Verify signals only depend on past data."""
    strat = TrendFollowingStrategy(TEST_CFG)
    df = _sample_ohlcv(300)
    result_full = strat.generate_signals(df)

    # Run on first 200 bars only
    result_partial = strat.generate_signals(df.iloc[:200])

    # Signals for bar 199 should be identical in both runs
    assert result_full.iloc[199]["signal"] == result_partial.iloc[199]["signal"]


def test_signals_not_always_same():
    """Strategy should generate different signals, not all zeros."""
    strat = TrendFollowingStrategy(TEST_CFG)
    df = _sample_ohlcv(1000)
    result = strat.generate_signals(df)
    # Should have at least some non-zero signals
    non_zero = (result["signal"] != 0).sum()
    assert non_zero > 0, "Strategy generated no signals at all"
