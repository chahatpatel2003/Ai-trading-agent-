from __future__ import annotations

from datetime import date

import pytest

from nqlab.config.models import AppConfig
from nqlab.data.bars_build import _write_partitioned, bars_root
from nqlab.features.build import build_features, read_features
from nqlab.features.build_full import build_features_full
from tests.synth import multi_day_bars

_RTH_START = 570
# Two trading days in late Dec 2023 + two in early Jan 2024 (crosses month + year).
_DAYS = [date(2023, 12, 28), date(2023, 12, 29), date(2024, 1, 2), date(2024, 1, 3)]


def test_warmup_across_month_boundary(config: AppConfig) -> None:
    """Incremental January features (with a December warmup buffer) must match the
    full-series features at the same rows — proving warmup is handled correctly."""
    bars = multi_day_bars("NQ.c", _DAYS, bars_per_day=30)
    _write_partitioned(bars, bars_root(config), "5m")

    result = build_features_full(
        config, symbol="NQ.c", timeframe="5m",
        start=date(2024, 1, 1), end=date(2024, 1, 31), warmup_days=10,
    )
    assert "202401" in result.built

    # Metadata recorded warmup bars pulled from December.
    from nqlab.data.ledger import PartitionLedger
    from nqlab.features.build_full import _ledger_path

    entry = PartitionLedger.load(_ledger_path(config, "NQ.c", "5m")).get("202401")
    assert entry is not None
    assert int(entry.extra["warmup_bars"]) > 0
    assert entry.extra["feature_version"] == "1"

    incremental = read_features(
        config, symbol="NQ.c", timeframe="5m", start=date(2024, 1, 1), end=date(2024, 1, 31)
    ).sort("ts")
    full = build_features(bars, rth_start_min=_RTH_START).sort("ts")
    full_jan = full.filter(full["trading_day"] >= date(2024, 1, 1))

    assert incremental.height == full_jan.height
    for col in ("ret_1", "atr", "sma", "rsi", "vwap_session", "roll_high"):
        a = incremental[col].to_list()
        b = full_jan[col].to_list()
        for x, y in zip(a, b, strict=True):
            if x is None and y is None:
                continue
            assert x == pytest.approx(y, rel=1e-6, abs=1e-6), col
