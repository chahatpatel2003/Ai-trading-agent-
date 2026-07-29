"""
Technical indicator calculations used by strategies.
All indicators are computed on historical data only — no future leakage.
"""
import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — measures volatility."""
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower) Bollinger Bands."""
    middle = sma(series, period)
    rolling_std = series.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    return upper, middle, lower


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling z-score of a series."""
    rolling_mean = series.rolling(window=period, min_periods=period).mean()
    rolling_std = series.rolling(window=period, min_periods=period).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator %K and %D."""
    low_min = df["low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(window=k_period, min_periods=k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df["close"] - low_min) / denom
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d


def vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Rolling VWAP approximation (volume-weighted average price)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, 1)
    return (tp * vol).rolling(period, min_periods=period).sum() / vol.rolling(period, min_periods=period).sum()


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
    """Ichimoku Cloud components (tenkan, kijun, senkou_a, senkou_b)."""
    high = df["high"]
    low = df["low"]
    tenkan_sen = (high.rolling(tenkan, min_periods=tenkan).max() + low.rolling(tenkan, min_periods=tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun, min_periods=kijun).max() + low.rolling(kijun, min_periods=kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_val = ((high.rolling(senkou_b, min_periods=senkou_b).max() + low.rolling(senkou_b, min_periods=senkou_b).min()) / 2).shift(kijun)
    return tenkan_sen, kijun_sen, senkou_a, senkou_b_val


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff())
    return (df["volume"] * direction).fillna(0).cumsum()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — measures trend strength."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr_vals = atr(df, period)
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_vals)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_vals)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window=period, min_periods=period).mean()
