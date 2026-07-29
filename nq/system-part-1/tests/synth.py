"""Synthetic frame builders for hermetic Phase-1 tests (no external data)."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import polars as pl

from nqlab.config.models import SessionConfig
from nqlab.domain.calendar.sessions import SessionLabeler

_FLOAT_COLS = ("open", "high", "low", "close", "atr", "stop_pts", "target_pts")


def engine_frame(bars: list[dict[str, object]]) -> pl.DataFrame:
    """Build a minimal bar+signal frame for engine tests.

    Each bar dict may set: close, open, high, low, signal, stop_pts, target_pts,
    minute (ET minute-of-day, default 600 = 10:00), day, is_rth. ``open`` defaults
    to ``close``.
    """
    base_day = date(2024, 1, 2)
    t0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    rows = []
    for i, bar in enumerate(bars):
        close = float(bar["close"])  # type: ignore[arg-type]
        rows.append(
            {
                "ts": t0 + timedelta(minutes=5 * i),
                "minute_of_day": int(bar.get("minute", 600)),
                "trading_day": bar.get("day", base_day),
                "is_rth": bool(bar.get("is_rth", True)),
                "open": float(bar.get("open", close)),  # type: ignore[arg-type]
                "high": float(bar["high"]),  # type: ignore[arg-type]
                "low": float(bar["low"]),  # type: ignore[arg-type]
                "close": close,
                "atr": float(bar.get("atr", 10.0)),  # type: ignore[arg-type]
                "signal": int(bar.get("signal", 0)),  # type: ignore[arg-type]
                "stop_pts": bar.get("stop_pts"),
                "target_pts": bar.get("target_pts"),
            }
        )
    overrides = {col: pl.Float64 for col in _FLOAT_COLS}
    return pl.DataFrame(rows, schema_overrides=overrides)


def feature_bars(symbol: str = "NQ.c", days: int = 2, bars_per_day: int = 78) -> pl.DataFrame:
    """Deterministic 5-minute RTH bars across ``days`` (no randomness)."""
    labeler = SessionLabeler.from_config(SessionConfig())
    base = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)  # 09:30 ET
    rows = []
    idx = 0
    for day in range(days):
        day_start = base + timedelta(days=day)
        for b in range(bars_per_day):
            price = 100.0 + 5.0 * math.sin(idx / 7.0) + idx * 0.01
            rows.append(
                {
                    "ts": day_start + timedelta(minutes=5 * b),
                    "symbol": symbol,
                    "open": price,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price,
                    "volume": 100 + (idx % 10),
                    "vwap": price,
                    "trades": 50,
                }
            )
            idx += 1
    frame = pl.DataFrame(rows)
    return labeler.with_calendar_columns(frame, "ts")


def multi_day_bars(symbol: str, days: list[date], bars_per_day: int = 30) -> pl.DataFrame:
    """RTH 5m bars across explicit trading days (EST: 14:30 UTC = 09:30 ET)."""
    labeler = SessionLabeler.from_config(SessionConfig())
    rows = []
    idx = 0
    for d in days:
        day_open = datetime(d.year, d.month, d.day, 14, 30, tzinfo=UTC)
        for b in range(bars_per_day):
            price = 100.0 + 5.0 * math.sin(idx / 7.0) + idx * 0.01
            rows.append(
                {
                    "ts": day_open + timedelta(minutes=5 * b),
                    "symbol": symbol,
                    "open": price,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price,
                    "volume": 100 + (idx % 10),
                    "vwap": price,
                    "trades": 50,
                }
            )
            idx += 1
    return labeler.with_calendar_columns(pl.DataFrame(rows), "ts")


def globex_bars(days: int = 2, symbol: str = "NQ.c") -> pl.DataFrame:
    """5m bars spanning the full Globex day (Asia/London/NY) for context tests."""
    labeler = SessionLabeler.from_config(SessionConfig())
    base = datetime(2024, 1, 1, 23, 0, tzinfo=UTC)  # 18:00 ET (session open)
    rows = []
    for idx in range(int(days * 21 * 12)):  # ~21h/day at 5m
        price = 100.0 + 5.0 * math.sin(idx / 9.0) + idx * 0.005
        rows.append(
            {
                "ts": base + timedelta(minutes=5 * idx),
                "symbol": symbol,
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": 100 + (idx % 10),
                "vwap": price,
                "trades": 10,
                "realized_vol": 1.0 + 0.5 * abs(math.sin(idx / 13.0)),
            }
        )
    return labeler.with_calendar_columns(pl.DataFrame(rows), "ts")


def mbp1_bar_frame(days: int = 2, bars_per_day: int = 80) -> pl.DataFrame:
    """Synthetic MBP1 bar frame (delta + spread + calendar) for finalize/PIT tests."""
    labeler = SessionLabeler.from_config(SessionConfig())
    base = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    rows = []
    idx = 0
    for day in range(days):
        day_start = base + timedelta(days=day)
        for b in range(bars_per_day):
            rows.append(
                {
                    "ts": day_start + timedelta(minutes=5 * b),
                    "delta": 100.0 * math.sin(idx / 5.0),
                    "spread": 0.25 + 0.1 * abs(math.sin(idx / 9.0)),
                }
            )
            idx += 1
    return labeler.with_calendar_columns(pl.DataFrame(rows), "ts")


def add_synth_mbp1_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach deterministic synthetic MBP1 feature columns to a trades feature frame."""
    i = pl.int_range(0, pl.len()).cast(pl.Float64)
    return frame.with_columns(
        (0.5 * (i / 7.0).sin()).alias("tob_imbalance"),
        (100.0 * (i / 5.0).sin()).alias("delta"),
        (100.0 * (i / 5.0).sin()).cum_sum().over("trading_day").alias("cvd"),
        (100.0 * (i / 5.0).sin()).alias("signed_volume"),
        (0.25 + 0.1 * (i / 9.0).sin().abs()).alias("spread"),
        pl.lit(1.0).alias("liq_spread_ratio"),
        pl.lit(0, dtype=pl.Int8).alias("liquidity_regime"),
    )


def validation_result(**overrides: object):
    """A passing ValidationResult; override fields to exercise reject rules."""
    from nqlab.validation.gauntlet import ValidationResult
    from nqlab.validation.montecarlo import MonteCarloResult
    from nqlab.validation.stress import StressResult

    base: dict[str, object] = {
        "strategy": "s", "version": "1", "symbol": "NQ.c", "timeframe": "5m", "account": "50K",
        "full": {"net_pnl": 1000.0, "trades": 100},
        "in_sample": {"net_pnl": 600.0, "trades": 60},
        "out_sample": {"net_pnl": 400.0, "trades": 40},
        "yearly": {
            "2021": {"net_pnl": 300.0, "trades": 50},
            "2022": {"net_pnl": 700.0, "trades": 50},
        },
        "monthly": {
            "2021-01": {"net_pnl": 100.0, "trades": 10},
            "2021-02": {"net_pnl": 200.0, "trades": 10},
            "2021-03": {"net_pnl": 700.0, "trades": 10},
        },
        "walk_forward": [],
        "walk_forward_positive_frac": 0.8,
        "cost_stress": [StressResult("slip2x_comm1x", 2.0, 1.0, 500.0, 50, 1.5, 300.0, False)],
        "monte_carlo": MonteCarloResult(100, 0.02, 0.02, 500.0, 100.0, 300.0, 400.0),
        "topstep": None,
        "unrealistic_fills": False,
        "base_net_realistic": 1000.0,
        "base_net_optimistic": 1100.0,
    }
    base.update(overrides)
    return ValidationResult(**base)  # type: ignore[arg-type]
