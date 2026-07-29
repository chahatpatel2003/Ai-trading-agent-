from __future__ import annotations

import json

import polars as pl

from nqlab.features.build import build_features
from nqlab.features.mbp1_profile import profile_distributions
from tests.synth import add_synth_mbp1_columns, feature_bars


def _frame() -> pl.DataFrame:
    return add_synth_mbp1_columns(build_features(feature_bars(days=2), rth_start_min=570))


def test_overall_quantiles_are_monotonic() -> None:
    prof = profile_distributions(_frame())
    assert prof.n_bars > 0
    s = prof.overall["tob_imbalance"]
    assert s["q10"] <= s["q50"] <= s["q90"] <= s["q99"]
    assert "mean" in s
    assert "abs_tob_imbalance" in prof.overall  # derived
    assert "cvd_slope" in prof.overall  # derived from cvd


def test_breakdowns_present() -> None:
    prof = profile_distributions(_frame())
    assert prof.by_session
    assert prof.by_hour
    for feats in prof.by_session.values():
        assert "delta" in feats


def test_percentile_accessor() -> None:
    prof = profile_distributions(_frame())
    assert prof.percentile("spread", 0.5) == prof.overall["spread"]["q50"]
    assert prof.percentile("spread", 0.9) == prof.overall["spread"]["q90"]


def test_report_schema_serializes() -> None:
    d = profile_distributions(_frame()).to_dict()
    assert {"features", "n_bars", "overall", "by_session", "by_regime", "by_hour"} <= set(d)
    json.dumps(d)
