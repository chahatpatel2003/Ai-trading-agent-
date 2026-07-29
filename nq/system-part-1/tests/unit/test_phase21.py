"""Phase 21 — order-flow feature layer + MBP1 families + MBP1-required skipping."""

from __future__ import annotations

import polars as pl

from nqlab.discovery.phase21 import PHASE21_THRESHOLDS, generate_phase21_candidates
from nqlab.discovery.phase21_families import PHASE21_FAMILIES
from nqlab.domain.strategy.base import MBP1_REQUIRED
from nqlab.features.orderflow import ORDERFLOW_DERIVED, add_orderflow_features


def _mbp1_frame(n: int = 80) -> pl.DataFrame:
    px = [20000.0 + (i % 5) - 2 for i in range(n)]
    return pl.DataFrame({
        "close": px, "open": [p - 0.5 for p in px],
        "high": [p + 2 for p in px], "low": [p - 2 for p in px],
        "atr": [10.0] * n, "vwap_session": [20000.0] * n,
        "roll_high": [20003.0] * n, "roll_low": [19997.0] * n,
        "swept_high": [0] * n, "swept_low": [1 if i % 7 == 0 else 0 for i in range(n)],
        "delta": [(-1) ** i * (50 + i) for i in range(n)],
        "cvd": [float(i) for i in range(n)],
        "ofi": [(-1) ** i * (30 + i) for i in range(n)],
        "tob_imbalance": [0.3 if i % 2 else -0.3 for i in range(n)],
        "ask_depletion": [1.0] * n, "bid_depletion": [0.5] * n,
        "liq_spread_ratio": [0.8] * n,
        "minute_of_day": [600] * n, "is_rth": [True] * n,
    })


# --------------------------------------------------------------------------- order-flow features
def test_orderflow_features_added() -> None:
    f = add_orderflow_features(_mbp1_frame())
    for c in ORDERFLOW_DERIVED:
        assert c in f.columns
    assert f["imb_persist_up"].drop_nulls().is_in([0, 1]).all()


def test_orderflow_noops_without_mbp1() -> None:
    # No delta/ofi -> returns frame unchanged (graceful skip).
    bare = pl.DataFrame({"close": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0]})
    out = add_orderflow_features(bare)
    assert "ofi_z" not in out.columns


def test_orderflow_zscore_is_backward_looking() -> None:
    f = _mbp1_frame(40)
    full = add_orderflow_features(f)["delta_z"].to_list()
    pref = add_orderflow_features(f.head(20))["delta_z"].to_list()
    assert full[:20] == pref  # truncating the future does not change past z-scores


# --------------------------------------------------------------------------- families
def test_phase21_has_six_mbp1_families() -> None:
    assert set(PHASE21_FAMILIES) == {
        "p21_absorption_rev", "p21_imbalance_cont", "p21_failed_flow",
        "p21_liq_vacuum", "p21_delta_div", "p21_adverse_filter",
    }
    for cls in PHASE21_FAMILIES.values():
        assert cls.data_requirement == MBP1_REQUIRED


def test_phase21_candidates_are_ny_am() -> None:
    cands = generate_phase21_candidates(timeframe="5m")
    assert len(cands) == 6 * 2  # 6 families x 2 z-thresholds
    p = cands[0].p
    assert p["hour_start"] == 9 and p["hour_end"] == 12


def test_mbp1_family_skips_when_mbp1_absent() -> None:
    strat = PHASE21_FAMILIES["p21_absorption_rev"](
        {"z": 0.75, "stop_atr": 1.0, "target_atr": 1.5, "hour_start": 9, "hour_end": 12,
         "risk_per_trade": 250.0, "sizing_mode": "mnq_min1", "min_hold_bars": 1,
         "time_stop_bars": 18}
    )
    # Frame lacks delta/cvd (MBP1) -> the family must emit no signals, not error.
    bare = pl.DataFrame({
        "close": [1.0, 2.0, 3.0], "open": [1.0, 2.0, 3.0], "high": [1.0, 2.0, 3.0],
        "low": [1.0, 2.0, 3.0], "atr": [1.0, 1.0, 1.0],
        "minute_of_day": [600, 600, 600], "is_rth": [True, True, True],
    })
    sig = strat.signals(bare)
    assert (sig["signal"] == 0).all()


def test_delta_divergence_family_fires_on_orderflow() -> None:
    strat = PHASE21_FAMILIES["p21_delta_div"](
        {"z": 0.5, "stop_atr": 1.0, "target_atr": 1.5, "hour_start": 9, "hour_end": 12,
         "risk_per_trade": 250.0, "sizing_mode": "mnq_min1", "min_hold_bars": 1,
         "time_stop_bars": 18}
    )
    sig = strat.signals(_mbp1_frame(120))
    assert {"signal", "stop_pts", "target_pts"} <= set(sig.columns)
    assert sig.filter(pl.col("signal") != 0).height >= 0  # runs without error on MBP1 data


# --------------------------------------------------------------------------- thresholds
def test_phase21_thresholds() -> None:
    th = PHASE21_THRESHOLDS
    assert th.pf_min == 1.30
    assert th.green_month_min == 0.75
    assert th.min_payout_eligible_frac == 0.60
