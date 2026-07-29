from __future__ import annotations

from nqlab.discovery.candidates import FAMILIES, generate_candidates
from nqlab.domain.strategy.base import SIGNAL_COLUMNS
from nqlab.features.build import build_features
from tests.synth import feature_bars


def test_grids_are_nonempty() -> None:
    for cls in FAMILIES.values():
        assert len(cls.grid()) > 0


def test_generation_is_deterministic_and_ordered() -> None:
    a = [c.params.fingerprint() for c in generate_candidates()]
    b = [c.params.fingerprint() for c in generate_candidates()]
    assert a == b  # deterministic order
    assert len(a) == len(set(a))  # unique fingerprints
    assert len(a) == sum(len(cls.grid()) for cls in FAMILIES.values())


def test_family_filter_limits_candidates() -> None:
    only_orb = list(generate_candidates(families=("orb",)))
    assert {c.family for c in only_orb} == {"orb"}
    assert len(only_orb) == len(FAMILIES["orb"].grid())


def test_signals_produce_required_columns() -> None:
    frame = build_features(feature_bars(days=2), rth_start_min=570)
    sample = list(generate_candidates(families=("orb",)))[:1]
    sample += list(generate_candidates(families=("trend",)))[:1]
    for strat in sample:
        out = strat.signals(frame)
        assert set(SIGNAL_COLUMNS) <= set(out.columns)
        assert str(out["signal"].dtype).startswith("Int")
