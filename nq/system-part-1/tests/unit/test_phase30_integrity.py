"""Phase 30 — volume/market-profile feature integrity (no hindsight).

The decisive proof is **prefix-invariance**: a VP value at bar k computed on the FULL series must
equal the value computed on only the prefix [0..k]. If adding future bars/sessions ever changes a
past POC/VAH/VAL/HVN/LVN, that is look-ahead → FAIL. Plus: the prior-session profile must equal the
*standalone* profile of the completed previous trading day; the developing profile must exist only
within RTH and grow as bars arrive; and VAL ≤ POC ≤ VAH always.
"""

from __future__ import annotations

import polars as pl
import pytest

from nqlab.features.build import build_features
from nqlab.features.volume_profile import (
    DEFAULT_BIN,
    DEFAULT_VA_PCT,
    _day_profiles,
    add_volume_profile,
)
from tests.synth import globex_bars

_RTH = 570
_LEVELS = [
    "vp_prior_poc", "vp_prior_vah", "vp_prior_val", "vp_prior_vaw",
    "vp_prior_hvn_up", "vp_prior_hvn_dn", "vp_prior_lvn_up", "vp_prior_lvn_dn",
    "vp_dev_poc", "vp_dev_vah", "vp_dev_val", "vp_comp_poc", "vp_comp_vah", "vp_comp_val",
]


def _vp(bars: pl.DataFrame) -> pl.DataFrame:
    return add_volume_profile(build_features(bars, rth_start_min=_RTH))


def test_vp_prefix_invariant_no_future_leakage() -> None:
    bars = globex_bars(days=12)
    full = _vp(bars).with_row_index("idx")
    # Sample mid/late RTH bars where prior + developing profiles are populated.
    cand = full.filter(
        pl.col("is_rth") & pl.col("vp_prior_poc").is_not_null() & pl.col("vp_dev_poc").is_not_null()
    )["idx"].to_list()
    assert len(cand) >= 5
    picks = [cand[int(f * (len(cand) - 1))] for f in (0.4, 0.55, 0.7, 0.85, 0.97)]
    for k in picks:
        prefix = _vp(bars.head(k + 1))
        fr = full.drop("idx").row(k, named=True)
        pr = prefix.row(k, named=True)
        for c in _LEVELS:
            a, b = fr[c], pr[c]
            if a is None and b is None:
                continue
            assert a == pytest.approx(b, rel=1e-9, abs=1e-9), f"LEAK in {c} @k={k}: {a}!={b}"


def test_vp_prior_equals_completed_previous_session() -> None:
    bars = globex_bars(days=10)
    fr = build_features(bars, rth_start_min=_RTH)
    vp = add_volume_profile(fr)
    dp = _day_profiles(fr, DEFAULT_BIN, DEFAULT_VA_PCT)
    days = sorted(dp)
    for i in range(1, len(days)):
        got = vp.filter(pl.col("trading_day") == days[i]).select("vp_prior_poc").to_series()[0]
        assert got == pytest.approx(dp[days[i - 1]].poc), f"{days[i]} prior!=prev POC"


def test_vp_developing_only_in_rth_and_well_formed() -> None:
    vp = _vp(globex_bars(days=8))
    # developing profile is defined ONLY on RTH bars
    non_rth_dev = vp.filter(~pl.col("is_rth") & pl.col("vp_dev_poc").is_not_null()).height
    assert non_rth_dev == 0
    rth_dev = vp.filter(pl.col("is_rth") & pl.col("vp_dev_poc").is_not_null()).height
    assert rth_dev > 0
    # VAL <= POC <= VAH for every populated profile (prior + developing)
    for poc, val, vah in (("vp_prior_poc", "vp_prior_val", "vp_prior_vah"),
                          ("vp_dev_poc", "vp_dev_val", "vp_dev_vah")):
        bad = vp.filter(
            pl.col(poc).is_not_null()
            & ((pl.col(val) > pl.col(poc)) | (pl.col(poc) > pl.col(vah)))
        ).height
        assert bad == 0, f"value-area ordering violated for {poc}"


def test_vp_families_resolve_and_emit_valid_signals() -> None:
    """Every Phase-30 candidate builds a valid {-1,0,1} signal on the full feature+VP stack."""
    from nqlab.discovery.phase29 import add_phase29_indicators
    from nqlab.discovery.phase30 import build_signal_frame, generate_candidates
    from nqlab.features.context import build_context_features
    from nqlab.features.daily_bias import add_daily_bias
    from nqlab.features.htf_context import build_htf_context_features
    from nqlab.features.htf_poi import add_htf_poi
    from nqlab.features.structure import build_structure_features

    bars = globex_bars(days=12)
    f = build_features(bars, rth_start_min=_RTH)
    f = build_context_features(f, rth_start_min=_RTH)
    f = build_htf_context_features(f, rth_start_min=_RTH, timeframe="5m")
    f = build_structure_features(f)
    f = add_daily_bias(add_htf_poi(f, tol_pts=5.0))
    f = add_phase29_indicators(f)
    frame = add_volume_profile(f)
    for cand in generate_candidates():
        sig = build_signal_frame(frame, cand)
        assert "signal" in sig.columns
        assert set(sig["signal"].unique().to_list()).issubset({-1, 0, 1})
        # stop/target distances are strictly positive where a signal fires
        fired = sig.filter(pl.col("signal") != 0)
        if fired.height:
            assert fired.select((pl.col("stop_pts") > 0).all()).item()
            assert fired.select((pl.col("target_pts") > 0).all()).item()


def test_vp_developing_volume_is_cumulative_within_session() -> None:
    """Developing value-area width is defined throughout RTH and the profile only accretes bars
    (proxy: dev VAW is non-null for all RTH bars after the first of each day)."""
    vp = _vp(globex_bars(days=6))
    by_day = vp.filter(pl.col("is_rth")).group_by("trading_day").agg(
        pl.col("vp_dev_vaw").is_not_null().sum().alias("nn"),
        pl.len().alias("n"),
    )
    # Every RTH bar of every day should have a developing profile (first bar already has 1 bar).
    assert by_day.filter(pl.col("nn") < pl.col("n")).height == 0
