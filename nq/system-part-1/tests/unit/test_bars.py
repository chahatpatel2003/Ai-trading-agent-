from __future__ import annotations

from datetime import date

from nqlab.config.models import AppConfig
from nqlab.data.bars_build import build_bars, read_bars


def test_bar_aggregation(config: AppConfig) -> None:
    # The fixture trades file holds 12 trades at 2024-01-01 (prices 17000 + i*0.25,
    # sizes 1..12, symbols NQH4, NQH4, NQH4-NQM4 repeating). NQH4 rows are i%3!=2.
    day = date(2024, 1, 1)
    result = build_bars(config, timeframe="5m", start=day, end=day)
    assert "NQH4" in result.symbols
    assert all("-" not in s for s in result.symbols)  # spreads excluded

    bars = read_bars(config, symbol="NQH4", timeframe="5m", start=day, end=day)
    assert bars.height == 1
    row = bars.row(0, named=True)
    assert row["open"] == 17000.0  # first NQH4 trade (i=0)
    assert row["close"] == 17002.5  # last NQH4 trade (i=10)
    assert row["high"] == 17002.5
    assert row["low"] == 17000.0
    assert row["volume"] == 48  # sizes 1+2+4+5+7+8+10+11
    assert row["trades"] == 8


def test_bars_carry_calendar_columns(config: AppConfig) -> None:
    day = date(2024, 1, 1)
    build_bars(config, timeframe="5m", start=day, end=day)
    bars = read_bars(config, symbol="NQH4", timeframe="5m", start=day, end=day)
    assert {"session", "trading_day", "is_rth", "minute_of_day", "dow"} <= set(bars.columns)


def test_volume_conservation_across_timeframes(config: AppConfig) -> None:
    day = date(2024, 1, 1)
    build_bars(config, timeframe="1m", start=day, end=day)
    build_bars(config, timeframe="5m", start=day, end=day)
    one_min = read_bars(config, symbol="NQH4", timeframe="1m", start=day, end=day)
    five_min = read_bars(config, symbol="NQH4", timeframe="5m", start=day, end=day)
    assert one_min["volume"].sum() == five_min["volume"].sum()
