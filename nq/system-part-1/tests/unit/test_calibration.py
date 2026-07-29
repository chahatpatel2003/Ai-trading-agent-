from __future__ import annotations

import polars as pl
import pytest

from nqlab.discovery.calibration import calibrate, generate_calibrated_candidates
from nqlab.discovery.candidates import COMBO_FAMILIES, ImbalanceTrendCont, rolling_zscore
from nqlab.domain.strategy.base import MBP1_REQUIRED
from nqlab.features.build import build_features
from nqlab.features.mbp1_profile import profile_distributions
from tests.synth import add_synth_mbp1_columns, feature_bars


def _profile():
    frame = add_synth_mbp1_columns(build_features(feature_bars(2), rth_start_min=570))
    return profile_distributions(frame)


def test_thresholds_come_from_percentiles() -> None:
    t = calibrate(_profile())
    assert len(t.imb) >= 1
    assert list(t.imb) == sorted(t.imb)  # ascending percentiles
    assert all(x >= 0 for x in t.imb)
    assert len(t.spread_max) >= 1


def test_generate_calibrated_candidates() -> None:
    candidates = generate_calibrated_candidates(calibrate(_profile()))
    assert len(candidates) > 0
    fingerprints = [c.params.fingerprint() for c in candidates]
    assert len(fingerprints) == len(set(fingerprints))  # unique
    assert {c.family for c in candidates} == set(COMBO_FAMILIES)
    assert all(c.data_requirement == MBP1_REQUIRED for c in candidates)


def test_combo_requires_mbp1_and_skips_when_absent() -> None:
    frame = build_features(feature_bars(days=2), rth_start_min=570)  # trades-only
    strat = ImbalanceTrendCont(
        {"imb_thr": 0.1, "spread_max": 1.0, "ema_span": 20, "stop_atr": 1.5, "target_atr": 2.0}
    )
    assert "tob_imbalance" in strat.required_columns
    assert int(strat.signals(frame)["signal"].abs().sum()) == 0


def test_rolling_zscore_is_point_in_time() -> None:
    df = pl.DataFrame({"x": [float((i * 13) % 17) for i in range(200)]})
    full = df.with_columns(rolling_zscore(pl.col("x"), 50).alias("z"))
    k = 120
    prefix = df.head(k + 1).with_columns(rolling_zscore(pl.col("x"), 50).alias("z"))
    assert full["z"][k] == pytest.approx(prefix["z"][k], rel=1e-9, abs=1e-9)
