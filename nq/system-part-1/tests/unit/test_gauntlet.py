from __future__ import annotations

import json
from datetime import date

from nqlab.config.models import AppConfig, TopstepRules
from nqlab.features.build import build_features
from nqlab.research.baselines import build_baseline
from nqlab.validation.gauntlet import run_gauntlet
from nqlab.validation.report import to_dict
from tests.synth import feature_bars

_START, _END = date(2024, 1, 2), date(2024, 1, 3)


def _frame():
    return build_features(feature_bars(days=2), rth_start_min=570)


def test_gauntlet_runs_end_to_end_and_decides(
    app_config: AppConfig, rules_50k: TopstepRules
) -> None:
    result = run_gauntlet(
        app_config, build_baseline("pdhl"), symbol="NQ.c", timeframe="5m",
        start=_START, end=_END, rules=rules_50k, feature_frame=_frame(),
    )
    assert result.verdict in {"ACCEPT", "REJECT", "NEEDS_MORE_DATA"}
    assert result.reasons
    assert result.monte_carlo.runs >= 0


def test_validation_report_schema_serializes(
    app_config: AppConfig, rules_50k: TopstepRules
) -> None:
    result = run_gauntlet(
        app_config, build_baseline("orb"), symbol="NQ.c", timeframe="5m",
        start=_START, end=_END, rules=rules_50k, feature_frame=_frame(),
    )
    d = to_dict(result)
    required = {
        "strategy", "version", "symbol", "timeframe", "account", "full", "in_sample",
        "out_sample", "yearly", "monthly", "walk_forward", "walk_forward_positive_frac",
        "cost_stress", "monte_carlo", "topstep", "unrealistic_fills", "verdict", "reasons",
    }
    assert required <= set(d)
    json.dumps(d, default=str)  # must be serializable
