"""Phase 24 — four prop-style 1:1 families, NaN-safe ADX, 1-trade/day, 60% win floor."""

from __future__ import annotations

import math

import polars as pl

from nqlab.discovery.phase24 import PHASE24_THRESHOLDS, generate_phase24_candidates
from nqlab.discovery.phase24_families import PHASE24_FAMILIES
from nqlab.features.indicators import add_adx


def _bars(n: int = 120) -> pl.DataFrame:
    px = [20000.0 + (i % 7) - 3 for i in range(n)]
    return pl.DataFrame({
        "close": px, "open": [p - 0.5 for p in px],
        "high": [p + 3 for p in px], "low": [p - 3 for p in px],
    })


def test_adx_is_nan_safe() -> None:
    # The old ADX poisoned the EWM with a 0/0 -> NaN; guard must keep it finite.
    f = add_adx(_bars())
    adx = f["adx"].to_list()
    assert all((v is None) or math.isfinite(v) for v in adx)
    assert f.select((pl.col("adx") >= 0).mean()).item() == 1.0  # finite, non-negative


def test_phase24_four_models() -> None:
    assert set(PHASE24_FAMILIES) == {
        "p24_orf", "p24_sweep_rev", "p24_vwap_rev", "p24_pullback",
    }


def test_phase24_candidates_one_to_one() -> None:
    cands = generate_phase24_candidates(timeframe="5m")
    assert len(cands) == 4 * 4  # 4 models x 4 fixed 1:1 sizes
    for c in cands:
        assert c.p["fix_stop"] == c.p["fix_target"]  # strictly 1:1
        assert c.p["hour_start"] == 9 and c.p["hour_end"] == 12


def test_one_trade_per_day_gate() -> None:
    fam = PHASE24_FAMILIES["p24_vwap_rev"]({
        "fix_stop": 10.0, "fix_target": 10.0, "hour_start": 9, "hour_end": 12,
        "risk_per_trade": 200.0, "sizing_mode": "mnq_min1", "min_hold_bars": 0,
        "time_stop_bars": 24,
    })
    # Two days, oscillating far from VWAP so the raw signal could fire many times.
    from datetime import date, datetime, timedelta
    rows = []
    for d in (date(2024, 1, 2), date(2024, 1, 3)):
        t0 = datetime(d.year, d.month, d.day, 9, 35)
        for i in range(20):
            px = 20000.0 - 40 + (i % 4)
            rows.append({
                "ts": t0 + timedelta(minutes=5 * i), "trading_day": d,
                "minute_of_day": 575 + 5 * i, "is_rth": True,
                "open": px - 1, "high": px + 3, "low": px - 3, "close": px,
                "vwap_session": 20000.0, "dist_vwap": px - 20000.0, "atr": 10.0,
            })
    sig = fam.signals(pl.DataFrame(rows))
    per_day = sig.filter(pl.col("signal") != 0).group_by("trading_day").len()
    assert per_day["len"].max() <= 1  # at most one trade/day


def test_phase24_threshold_60pct() -> None:
    assert PHASE24_THRESHOLDS.win_rate_min == 0.60
    assert PHASE24_THRESHOLDS.pf_min == 1.30
