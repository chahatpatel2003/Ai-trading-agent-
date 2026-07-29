from __future__ import annotations

from datetime import date

from nqlab.config.models import AppConfig
from nqlab.data.bars_full import _ledger_path, build_bars_full, months_in_range
from nqlab.data.ledger import PartitionLedger

_JAN = (date(2024, 1, 1), date(2024, 1, 31))


def test_months_in_range() -> None:
    months = months_in_range(date(2024, 1, 15), date(2024, 3, 2))
    assert [m[0] for m in months] == [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)]
    assert months[0][1] == date(2024, 1, 31)


def test_incremental_build_then_skip(config: AppConfig) -> None:
    first = build_bars_full(config, timeframe="5m", start=_JAN[0], end=_JAN[1], roots=("NQ",))
    assert "202401" in first.built
    assert first.total_rows > 0

    # Resume: the same call must skip the completed month (no rebuild).
    second = build_bars_full(config, timeframe="5m", start=_JAN[0], end=_JAN[1], roots=("NQ",))
    assert second.skipped == ["202401"]
    assert second.built == []


def test_force_rebuilds(config: AppConfig) -> None:
    build_bars_full(config, timeframe="5m", start=_JAN[0], end=_JAN[1], roots=("NQ",))
    forced = build_bars_full(
        config, timeframe="5m", start=_JAN[0], end=_JAN[1], roots=("NQ",), force=True
    )
    assert "202401" in forced.built


def test_verify_rebuilds_on_rowcount_mismatch(config: AppConfig) -> None:
    build_bars_full(config, timeframe="5m", start=_JAN[0], end=_JAN[1], roots=("NQ",))
    ledger = PartitionLedger.load(_ledger_path(config, "5m"))
    ledger.entries["202401"].rows = 999_999  # simulate corruption / partial write
    ledger.save()

    result = build_bars_full(
        config, timeframe="5m", start=_JAN[0], end=_JAN[1], roots=("NQ",), verify=True
    )
    assert "202401" in result.rebuilt
