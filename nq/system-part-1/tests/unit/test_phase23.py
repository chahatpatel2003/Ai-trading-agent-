"""Phase 23 — HTF POI features (PIT-safe) + the full-sequence families."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl

from nqlab.discovery.phase23 import PHASE23_THRESHOLDS, generate_phase23_candidates
from nqlab.discovery.phase23_families import PHASE23_FAMILIES
from nqlab.features.htf_poi import add_htf_poi


def _frame(n: int = 80, *, day: date = date(2024, 1, 2)) -> pl.DataFrame:
    rows = []
    t0 = datetime(day.year, day.month, day.day, 9, 35)
    base = 20000.0
    for i in range(n):
        px = base + (i % 10) - 5  # oscillates so price revisits levels
        rows.append({
            "ts": t0 + timedelta(minutes=5 * i), "trading_day": day,
            "minute_of_day": 575 + 5 * i, "is_rth": True,
            "open": px, "high": px + 4, "low": px - 4, "close": px,
            "prev_day_high": base + 8, "prev_day_low": base - 8,
            "prev_week_high": base + 20, "prev_week_low": base - 20,
            "overnight_high": base + 6, "overnight_low": base - 6,
        })
    return pl.DataFrame(rows)


def test_htf_poi_columns_and_touch_flags() -> None:
    f = add_htf_poi(_frame(), tol_pts=5.0)
    for c in ("support_poi", "resistance_poi", "at_support_poi", "at_resistance_poi",
              "at_support_poi_recent", "at_resistance_poi_recent", "poi_dist_support"):
        assert c in f.columns
    # support POI is at/below close where finite; resistance at/above.
    fin = f.filter(pl.col("support_poi").is_finite())
    assert (fin["support_poi"] <= fin["close"]).all()


def test_htf_poi_is_pit_safe() -> None:
    # The rolling HTF extremes use prior bars only -> truncating the future never changes the past.
    f = _frame(40)
    full = add_htf_poi(f, tol_pts=5.0)["at_support_poi"].to_list()
    pref = add_htf_poi(f.head(20), tol_pts=5.0)["at_support_poi"].to_list()
    assert full[:20] == pref


def test_recent_touch_flag_spans_window() -> None:
    f = add_htf_poi(_frame(60), tol_pts=5.0, window=6)
    # recent flag is a superset of the exact-touch flag.
    sup = f.select((pl.col("at_support_poi") == 1).sum()).item()
    rec = f.select((pl.col("at_support_poi_recent") == 1).sum()).item()
    assert rec >= sup


def test_phase23_four_variants_and_grid() -> None:
    assert set(PHASE23_FAMILIES) == {
        "p23_reclaim_close", "p23_mss_confirm", "p23_disp_fvg", "p23_reclaim_mss",
    }
    cands = generate_phase23_candidates(timeframe="5m")
    assert len(cands) == 4 * 3 * 3  # families x tol(3) x R-multiple(3)
    p = cands[0].p
    assert p["hour_start"] == 9 and p["hour_end"] == 12
    assert p["fix_stop"] == 10.0


def test_grid_has_one_to_one_and_higher_r() -> None:
    grid = PHASE23_FAMILIES["p23_reclaim_close"].grid()
    rr = {round(g["fix_target"] / g["fix_stop"], 1) for g in grid}
    assert 1.0 in rr  # 1:1 tested directly
    assert 1.5 in rr and 2.0 in rr


def test_phase23_threshold_55pct_win() -> None:
    assert PHASE23_THRESHOLDS.win_rate_min == 0.55
    assert PHASE23_THRESHOLDS.pf_min == 1.30
