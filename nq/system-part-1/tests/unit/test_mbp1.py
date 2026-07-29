from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nqlab.config.loader import load_config
from nqlab.features.mbp1 import build_mbp1_bar_features, trade_sign
from nqlab.features.mbp1_full import _finalize

# (action, side, price, size, bid_px, ask_px, bid_sz, ask_sz)
_EVENTS = [
    ("A", "N", 0.0, 0, 100.0, 100.5, 10, 20),
    ("A", "N", 0.0, 0, 100.0, 100.5, 30, 20),
    ("T", "B", 100.5, 5, 100.0, 100.5, 30, 15),
    ("T", "A", 100.0, 3, 100.0, 100.5, 27, 15),
]


def _write_mbp1(path: Path, events: list[tuple]) -> None:
    base = datetime(2024, 1, 2, 14, 30, 1, tzinfo=UTC)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "ts_recv": pa.array([base + timedelta(seconds=i) for i in range(len(events))],
                                type=pa.timestamp("ns", tz="UTC")),
            "sequence": pa.array(list(range(len(events))), type=pa.uint32()),
            "action": [e[0] for e in events],
            "side": [e[1] for e in events],
            "price": pa.array([e[2] for e in events], type=pa.float64()),
            "size": pa.array([e[3] for e in events], type=pa.uint32()),
            "bid_px_00": pa.array([e[4] for e in events], type=pa.float64()),
            "ask_px_00": pa.array([e[5] for e in events], type=pa.float64()),
            "bid_sz_00": pa.array([e[6] for e in events], type=pa.uint32()),
            "ask_sz_00": pa.array([e[7] for e in events], type=pa.uint32()),
            "symbol": ["NQH4"] * len(events),
        }
    )
    pq.write_table(table, str(path))


def _config(tmp_path: Path, config_dir: Path):
    overrides = {"data": {
        "nq_data_root": str(tmp_path), "curated_root": str(tmp_path / "c"),
        "catalog_path": str(tmp_path / "c/cat.duckdb"), "artifacts_root": str(tmp_path / "c/a"),
    }}
    return load_config(profile="ci", config_dir=config_dir, overrides=overrides, apply_env=False)


def test_trade_sign_classification() -> None:
    assert trade_sign("B") == 1  # buy aggressor
    assert trade_sign("A") == -1  # sell aggressor
    assert trade_sign("N") == 0


def test_spread_mid_microprice_imbalance(tmp_path: Path, config_dir: Path) -> None:
    _write_mbp1(tmp_path / "raw" / "mbp1" / "2024_H1.parquet", _EVENTS)
    cfg = _config(tmp_path, config_dir)
    df = build_mbp1_bar_features(
        cfg, symbol="NQH4", timeframe="5m", start=date(2024, 1, 2), end=date(2024, 1, 2)
    )
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["spread"] == pytest.approx(0.5)
    assert row["mid"] == pytest.approx(100.25)
    assert row["microprice"] == pytest.approx((100.0 * 15 + 100.5 * 27) / 42)
    assert row["tob_imbalance"] == pytest.approx(
        (-10 / 30 + 10 / 50 + 15 / 45 + 12 / 42) / 4
    )


def test_order_flow_aggregation(tmp_path: Path, config_dir: Path) -> None:
    _write_mbp1(tmp_path / "raw" / "mbp1" / "2024_H1.parquet", _EVENTS)
    cfg = _config(tmp_path, config_dir)
    row = build_mbp1_bar_features(
        cfg, symbol="NQH4", timeframe="5m", start=date(2024, 1, 2), end=date(2024, 1, 2)
    ).row(0, named=True)
    assert row["signed_volume"] == pytest.approx(2.0)  # +5 buy, -3 sell
    assert row["delta"] == pytest.approx(2.0)
    assert row["trade_count"] == 2
    assert row["quote_updates"] == 2
    assert row["avg_trade_size"] == pytest.approx(4.0)
    # No uint underflow: sizes are sane positive numbers.
    assert 0 < row["bid_size"] < 1000
    assert row["bid_replenishment"] == pytest.approx(20.0)  # +20 on the second quote
    assert row["ask_depletion"] == pytest.approx(5.0)  # 20 -> 15


def test_no_lookahead_mbp1_finalize() -> None:
    from tests.synth import mbp1_bar_frame

    bars = mbp1_bar_frame(days=2, bars_per_day=80)
    full = _finalize(bars)
    k = 120
    prefix = _finalize(bars.head(k + 1))
    for col in ("cvd", "liq_spread_ratio"):
        a, b = full.row(k, named=True)[col], prefix.row(k, named=True)[col]
        if a is None and b is None:
            continue
        assert a == pytest.approx(b, rel=1e-9, abs=1e-9), col
