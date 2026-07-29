from __future__ import annotations

from datetime import date

from nqlab.config.models import AppConfig
from nqlab.data.bars_full import build_bars_full
from nqlab.features.mbp1_full import (
    build_mbp1_features_full,
    profile_mbp1,
    read_mbp1_features,
)

_JAN = (date(2024, 1, 1), date(2024, 1, 31))


def test_missing_mbp1_period_is_recorded_not_filled(config: AppConfig) -> None:
    # Bars (for the active-contract map) come from the fixture trades at 2024-01-01.
    build_bars_full(config, timeframe="5m", start=_JAN[0], end=_JAN[1], roots=("NQ",))

    # Jan has MBP1 fixture data; February has none.
    result = build_mbp1_features_full(
        config, root="NQ", timeframe="5m", start=date(2024, 1, 1), end=date(2024, 2, 29)
    )
    assert "202401" in result.built
    assert "202402" in result.missing  # no silent fill — recorded as missing

    summary = profile_mbp1(config, "NQ.c", "5m")
    assert "202401" in summary["built"]
    assert "202402" in summary["missing"]

    features = read_mbp1_features(config, symbol="NQ.c", timeframe="5m")
    assert features.height > 0
    assert {"spread", "tob_imbalance", "cvd", "liq_spread_ratio"} <= set(features.columns)


def test_rebuild_skips_completed(config: AppConfig) -> None:
    build_bars_full(config, timeframe="5m", start=_JAN[0], end=_JAN[1], roots=("NQ",))
    build_mbp1_features_full(config, root="NQ", timeframe="5m", start=_JAN[0], end=_JAN[1])
    again = build_mbp1_features_full(config, root="NQ", timeframe="5m", start=_JAN[0], end=_JAN[1])
    assert again.built == []  # nothing rebuilt
    assert "202401" in again.skipped
