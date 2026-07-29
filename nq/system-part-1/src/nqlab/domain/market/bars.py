"""Timeframe definitions and the OHLCV bar schema."""

from __future__ import annotations

SUPPORTED_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "1h")

_TIMEFRAME_MINUTES: dict[str, int] = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}

# Aggregated value columns produced from trades (besides ts/symbol/calendar).
BAR_VALUE_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trades",
)


def timeframe_minutes(timeframe: str) -> int:
    """Minutes per bar for a supported timeframe. Raises ``ValueError`` otherwise."""
    try:
        return _TIMEFRAME_MINUTES[timeframe]
    except KeyError:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}; supported: {SUPPORTED_TIMEFRAMES}"
        ) from None


def polars_every(timeframe: str) -> str:
    """Polars ``truncate``/``group_by_dynamic`` interval string, e.g. ``'5m'``."""
    return f"{timeframe_minutes(timeframe)}m"
