from __future__ import annotations

from nqlab.config.models import ValidationConfig
from nqlab.validation.montecarlo import MonteCarloResult
from nqlab.validation.stress import StressResult
from nqlab.validation.verdict import ACCEPT, NEEDS_MORE_DATA, REJECT, decide
from tests.synth import validation_result

_V = ValidationConfig()


def test_accept_when_all_checks_pass() -> None:
    verdict, _ = decide(validation_result(), _V)
    assert verdict == ACCEPT


def test_needs_more_data_when_too_few_trades() -> None:
    verdict, _ = decide(validation_result(full={"net_pnl": 100.0, "trades": 5}), _V)
    assert verdict == NEEDS_MORE_DATA


def test_needs_more_data_when_too_few_oos_trades() -> None:
    verdict, _ = decide(validation_result(out_sample={"net_pnl": 100.0, "trades": 3}), _V)
    assert verdict == NEEDS_MORE_DATA


def test_reject_on_negative_out_of_sample() -> None:
    verdict, reasons = decide(
        validation_result(out_sample={"net_pnl": -100.0, "trades": 40}), _V
    )
    assert verdict == REJECT
    assert any("out-of-sample" in r for r in reasons)


def test_reject_when_only_one_profitable_year() -> None:
    verdict, reasons = decide(
        validation_result(
            yearly={
                "2021": {"net_pnl": -100.0, "trades": 50},
                "2022": {"net_pnl": 1100.0, "trades": 50},
            }
        ),
        _V,
    )
    assert verdict == REJECT
    assert any("years" in r for r in reasons)


def test_reject_on_high_blowup_probability() -> None:
    verdict, reasons = decide(
        validation_result(monte_carlo=MonteCarloResult(100, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0)), _V
    )
    assert verdict == REJECT
    assert any("blow-up" in r for r in reasons)


def test_reject_on_cost_stress_failure() -> None:
    verdict, reasons = decide(
        validation_result(
            cost_stress=[StressResult("slip3x_comm1x", 3.0, 1.0, -200.0, 50, 0.8, 400.0, False)]
        ),
        _V,
    )
    assert verdict == REJECT
    assert any("stress" in r for r in reasons)


def test_reject_on_unrealistic_fills() -> None:
    verdict, reasons = decide(validation_result(unrealistic_fills=True), _V)
    assert verdict == REJECT
    assert any("fills" in r for r in reasons)


def test_reject_on_weak_walk_forward() -> None:
    verdict, reasons = decide(validation_result(walk_forward_positive_frac=0.2), _V)
    assert verdict == REJECT
    assert any("walk-forward" in r for r in reasons)
