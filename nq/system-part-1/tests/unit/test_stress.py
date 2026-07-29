from __future__ import annotations

from nqlab.config.models import AppConfig, TopstepRules
from nqlab.domain.strategy.base import StrategyParams
from nqlab.validation.stress import run_stress
from tests.synth import engine_frame

_PARAMS = StrategyParams("t", "1")


def _profitable_frame():
    # Long entered at bar 1 open, target hit on bar 2.
    return engine_frame(
        [
            {"close": 100, "high": 100, "low": 100, "signal": 1, "stop_pts": 10, "target_pts": 20},
            {"open": 100, "high": 100, "low": 100, "close": 100, "minute": 605},
            {"open": 100, "high": 121, "low": 100, "close": 120, "minute": 610},
        ]
    )


def test_higher_slippage_reduces_pnl(app_config: AppConfig, rules_50k: TopstepRules) -> None:
    results = run_stress(
        _profitable_frame(), config=app_config, rules=rules_50k, strategy=_PARAMS,
        symbol="NQ.c", timeframe="5m", slippage_mults=(1.0, 3.0), commission_mults=(1.0,),
    )
    by_slip = {r.slippage_mult: r.net_pnl for r in results}
    assert by_slip[3.0] <= by_slip[1.0]


def test_higher_commission_reduces_pnl(app_config: AppConfig, rules_50k: TopstepRules) -> None:
    results = run_stress(
        _profitable_frame(), config=app_config, rules=rules_50k, strategy=_PARAMS,
        symbol="NQ.c", timeframe="5m", slippage_mults=(1.0,), commission_mults=(1.0, 1.5),
    )
    by_comm = {r.commission_mult: r.net_pnl for r in results}
    assert by_comm[1.5] <= by_comm[1.0]
