"""Unit tests for technical indicators."""
import numpy as np
import pandas as pd
import pytest
from data.indicators import atr, bollinger_bands, ema, macd, rsi, sma, zscore


def _sample_df(n=100):
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    close = 2000 + np.cumsum(np.random.randn(n) * 5)
    return pd.DataFrame({
        "open": close + np.random.randn(n),
        "high": close + abs(np.random.randn(n) * 3),
        "low": close - abs(np.random.randn(n) * 3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def test_sma_length():
    s = pd.Series(range(50), dtype=float)
    result = sma(s, 10)
    assert len(result) == 50
    assert result.isna().sum() == 9  # first 9 values are NaN


def test_ema_converges():
    s = pd.Series([100.0] * 50)
    result = ema(s, 10)
    assert abs(result.iloc[-1] - 100.0) < 0.01


def test_atr_positive():
    df = _sample_df()
    result = atr(df, 14)
    valid = result.dropna()
    assert (valid > 0).all()


def test_bollinger_bands_ordering():
    df = _sample_df()
    upper, middle, lower = bollinger_bands(df["close"], 20, 2.0)
    valid_idx = upper.dropna().index
    assert (upper[valid_idx] >= middle[valid_idx]).all()
    assert (middle[valid_idx] >= lower[valid_idx]).all()


def test_rsi_range():
    df = _sample_df()
    result = rsi(df["close"], 14)
    valid = result.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_macd_structure():
    df = _sample_df(200)
    line, signal, hist = macd(df["close"], 12, 26, 9)
    assert len(line) == 200
    # Histogram should equal line - signal
    valid = hist.dropna().index
    diff = (line[valid] - signal[valid] - hist[valid]).abs()
    assert (diff < 1e-10).all()


def test_zscore_mean_zero():
    """Z-score of a random series should have mean ~0 over valid values."""
    np.random.seed(42)
    s = pd.Series(np.random.randn(500))
    z = zscore(s, 20)
    valid = z.dropna()
    assert abs(valid.mean()) < 0.2
