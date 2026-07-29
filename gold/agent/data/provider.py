"""
Data providers for XAUUSD OHLCV data.
Supports TradingView (tvDatafeed), yfinance, and CSV file loading.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from logs.logger import get_logger

log = get_logger("data.provider")

# TradingView timeframe mapping (loaded lazily to avoid import errors if not installed)
_TV_INTERVAL_MAP = {
    "1m": "in_1_minute",
    "5m": "in_5_minute",
    "15m": "in_15_minute",
    "1h": "in_1_hour",
    "4h": "in_4_hour",
    "1d": "in_daily",
    "1W": "in_weekly",
}


def fetch_tradingview(
    symbol: str = "XAUUSD",
    exchange: str = "OANDA",
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    interval: str = "1h",
    username: str = None,
    password: str = None,
) -> pd.DataFrame:
    """
    Fetch XAUUSD data from TradingView via tvDatafeed.
    Uses real spot XAUUSD data (not futures proxy).
    """
    try:
        from tvDatafeed import TvDatafeed, Interval
    except ImportError:
        raise ImportError(
            "tvDatafeed is not installed. Install it with: pip install tvDatafeed"
        )

    log.info(
        "fetching_data",
        source="tradingview",
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        start=start,
        end=end,
    )

    # Map our timeframe string to tvDatafeed Interval enum
    interval_key = _TV_INTERVAL_MAP.get(interval)
    if interval_key is None:
        raise ValueError(
            f"Unsupported TradingView interval: {interval}. "
            f"Supported: {list(_TV_INTERVAL_MAP.keys())}"
        )
    tv_interval = getattr(Interval, interval_key)

    # Initialize TvDatafeed (works without credentials for most symbols)
    try:
        if username and password:
            tv = TvDatafeed(username=username, password=password)
        else:
            tv = TvDatafeed()
    except Exception as e:
        log.warning("tradingview_auth_failed", error=str(e))
        raise ConnectionError(
            f"TradingView connection failed: {e}. "
            f"Consider falling back to yfinance (set data.provider: yfinance in config)."
        )

    # tvDatafeed always returns the most recent N bars (max ~5000 per call).
    # We fetch the maximum available and then filter to the requested date range.
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    try:
        df = tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=tv_interval,
            n_bars=5000,
        )
    except Exception as e:
        raise ConnectionError(
            f"TradingView data fetch failed: {e}. "
            f"Check symbol '{symbol}' on exchange '{exchange}'."
        )

    if df is None or df.empty:
        raise ValueError(
            f"No data returned from TradingView for {symbol} on {exchange}. "
            f"Try a different exchange (FOREXCOM, FX_IDC) or check symbol availability."
        )

    # Filter to requested date range
    filtered = df.loc[
        (df.index >= start_dt) & (df.index <= end_dt + pd.Timedelta(days=1))
    ]

    if filtered.empty:
        # If requested range has no data, use all available data and warn
        log.warning(
            "tradingview_date_range_mismatch",
            requested_start=start,
            requested_end=end,
            available_start=str(df.index[0]),
            available_end=str(df.index[-1]),
            msg="Requested date range not available. Using all available data.",
        )
        filtered = df

    df = filtered

    # Drop the 'symbol' column if present (tvDatafeed adds it)
    if "symbol" in df.columns:
        df = df.drop(columns=["symbol"])

    df = _normalize_columns(df)
    log.info("data_fetched", rows=len(df), start=str(df.index[0]), end=str(df.index[-1]))
    return df


def fetch_yfinance(
    symbol: str = "GC=F",
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Fetch gold futures data from yfinance.
    GC=F is the COMEX gold futures continuous contract (closest proxy to XAUUSD).
    """
    log.info("fetching_data", source="yfinance", symbol=symbol, start=start, end=end)

    # yfinance limits 1h data to ~730 days per request; chunk if needed
    if interval in ("1h", "60m"):
        chunks = _chunked_date_ranges(start, end, max_days=720)
        frames = []
        for s, e in chunks:
            df = yf.download(symbol, start=s, end=e, interval=interval, progress=False)
            if not df.empty:
                frames.append(df)
        if not frames:
            raise ValueError(f"No data returned for {symbol} from yfinance")
        df = pd.concat(frames)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)
    else:
        df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {symbol} from yfinance")

    df = _normalize_columns(df)
    log.info("data_fetched", rows=len(df), start=str(df.index[0]), end=str(df.index[-1]))
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV data from a CSV file."""
    log.info("loading_csv", path=path)
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = _normalize_columns(df)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column names: open, high, low, close, volume."""
    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower().strip() for c in df.columns]
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0

    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    # Sanity checks
    assert (df["high"] >= df["low"]).all(), "high < low detected — data corruption"
    assert (df["close"] > 0).all(), "Non-positive close prices detected"

    return df


def _chunked_date_ranges(start: str, end: str, max_days: int) -> list[tuple[str, str]]:
    """Split a date range into chunks of max_days."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    chunks = []
    while s < e:
        chunk_end = min(s + pd.Timedelta(days=max_days), e)
        chunks.append((str(s.date()), str(chunk_end.date())))
        s = chunk_end
    return chunks


def get_data(cfg: dict) -> pd.DataFrame:
    """Main entry point — dispatches to the configured provider."""
    provider = cfg["data"]["provider"]
    if provider == "tradingview":
        return fetch_tradingview(
            symbol=cfg["data"].get("tradingview_symbol", "XAUUSD"),
            exchange=cfg["data"].get("tradingview_exchange", "OANDA"),
            start=cfg["data"]["start_date"],
            end=cfg["data"]["end_date"],
            interval=cfg["general"]["timeframe"],
            username=cfg["data"].get("tradingview_username"),
            password=cfg["data"].get("tradingview_password"),
        )
    elif provider == "yfinance":
        return fetch_yfinance(
            start=cfg["data"]["start_date"],
            end=cfg["data"]["end_date"],
            interval=cfg["general"]["timeframe"],
        )
    elif provider == "csv":
        return load_csv(cfg["data"]["csv_path"])
    else:
        raise ValueError(f"Unknown data provider: {provider}")
