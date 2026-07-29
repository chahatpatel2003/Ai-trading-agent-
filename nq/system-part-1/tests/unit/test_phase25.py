"""Phase 25 — daily-bias + HTF-trend alignment layer + bias-gated families."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl

from nqlab.discovery.phase25 import PHASE25_THRESHOLDS, generate_phase25_candidates
from nqlab.discovery.phase25_families import PHASE25_FAMILIES
from nqlab.features.daily_bias import add_daily_bias


def _frame(days: int = 5) -> pl.DataFrame:
    rows = []
    base = 20000.0
    for di in range(days):
        d = date(2024, 1, 2 + di)
        t0 = datetime(d.year, d.month, d.day, 9, 35)
        drift = di * 15.0  # each day a bit higher -> daily HH/HL (bullish structure)
        for i in range(12):
            px = base + drift + (i % 4)
            rows.append({
                "ts": t0 + timedelta(minutes=5 * i), "trading_day": d,
                "minute_of_day": 575 + 5 * i, "is_rth": True,
                "open": px - 1, "high": px + 3, "low": px - 3, "close": px,
                "trend_1h_bull": 1, "trend_4h_bull": 1, "displacement": 1 if i == 5 else 0,
                "in_bull_fvg": 0, "in_bear_fvg": 0, "fvg_bull": 0, "fvg_bear": 0,
            })
    return pl.DataFrame(rows)


def test_daily_bias_columns() -> None:
    f = add_daily_bias(_frame())
    for c in ("bias_a", "bias_b", "htf_bull", "htf_bear", "aligned_long_a", "aligned_short_a",
              "aligned_long_b", "aligned_short_b", "cisd_up", "cisd_dn", "ifvg_bull", "ifvg_bear"):
        assert c in f.columns


def test_alignment_requires_bias_and_htf_trend() -> None:
    f = add_daily_bias(_frame(days=6))
    # aligned_long_a implies BOTH bullish daily structure AND HTF bull trend.
    al = f.filter(pl.col("aligned_long_a") == 1)
    if al.height:
        assert (al["bias_a"] == 1).all()
        assert (al["htf_bull"] == 1).all()
    # mandatory: no long alignment when daily bias is unclear (bias_a == 0).
    unclear = f.filter(pl.col("bias_a") == 0)
    if unclear.height:
        assert (unclear["aligned_long_a"] == 0).all()


def test_daily_bias_pit_safe() -> None:
    # bias for day D depends only on prior days -> truncating later days never changes it.
    f5 = add_daily_bias(_frame(days=5))
    f3 = add_daily_bias(_frame(days=3))
    d3 = date(2024, 1, 4)
    b_full = f5.filter(pl.col("trading_day") == d3)["bias_a"][0]
    b_pref = f3.filter(pl.col("trading_day") == d3)["bias_a"][0]
    assert b_full == b_pref


def test_phase25_families_and_grid() -> None:
    assert len(PHASE25_FAMILIES) == 8
    assert {"p25_ifvg", "p25_cisd", "p25_mss", "p25_sweep", "p25_momentum", "p25_pivot_sr"} <= set(
        PHASE25_FAMILIES
    )
    cands = generate_phase25_candidates(timeframe="5m")
    assert len(cands) == 8 * 2 * 2  # families x bias-def x R-multiple
    biases = {c.p["bias_def"] for c in cands}
    assert biases == {"a", "b"}  # both daily-bias definitions tested separately


def test_phase25_threshold() -> None:
    assert PHASE25_THRESHOLDS.win_rate_min == 0.55
    assert PHASE25_THRESHOLDS.pf_min == 1.30
