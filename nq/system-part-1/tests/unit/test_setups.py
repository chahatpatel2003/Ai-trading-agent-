from __future__ import annotations

import polars as pl

from nqlab.discovery.calibration import calibrate, generate_setup_candidates
from nqlab.discovery.candidates import SETUP_FAMILIES, PrevDaySweepReversal
from nqlab.domain.strategy.base import MBP1_REQUIRED, SIGNAL_COLUMNS
from nqlab.features.build import build_features
from nqlab.features.context import build_context_features
from nqlab.features.mbp1_profile import profile_distributions
from tests.synth import add_synth_mbp1_columns, feature_bars, globex_bars


def _setup_frame() -> pl.DataFrame:
    bars = globex_bars(days=2)
    feats = build_features(bars, rth_start_min=570)
    feats = add_synth_mbp1_columns(feats)
    return build_context_features(feats, rth_start_min=570)


def test_generate_setup_candidates() -> None:
    thresholds = calibrate(profile_distributions(_setup_frame()))
    candidates = generate_setup_candidates(thresholds)
    assert len(candidates) > 0
    assert {c.family for c in candidates} == set(SETUP_FAMILIES)
    assert all(c.data_requirement == MBP1_REQUIRED for c in candidates)
    fps = [c.params.fingerprint() for c in candidates]
    assert len(fps) == len(set(fps))


def test_setup_signals_produce_columns() -> None:
    frame = _setup_frame()
    candidates = generate_setup_candidates(calibrate(profile_distributions(frame)))
    for strat in candidates:
        out = strat.signals(frame)
        assert set(SIGNAL_COLUMNS) <= set(out.columns)


def test_setup_skips_without_required_columns() -> None:
    frame = build_features(feature_bars(days=2), rth_start_min=570)  # no MBP1, no context
    strat = PrevDaySweepReversal({"stop_atr": 1.0, "target_atr": 2.0})
    assert int(strat.signals(frame)["signal"].abs().sum()) == 0
