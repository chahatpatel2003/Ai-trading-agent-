from __future__ import annotations

from nqlab.config.models import TopstepRules
from nqlab.validation.montecarlo import monte_carlo


def _rules() -> TopstepRules:
    return TopstepRules.model_validate(
        {
            "name": "T", "account_size": 50000.0, "profit_target": 3000.0,
            "trailing_max_loss": 2000.0, "max_contracts": 5,
        }
    )


def test_determinism_same_seed() -> None:
    pnls = [100.0, -50.0, 200.0, -150.0, 80.0, -30.0] * 5
    r1 = monte_carlo(pnls, _rules(), runs=500, seed=42)
    r2 = monte_carlo(pnls, _rules(), runs=500, seed=42)
    assert r1.blow_up_prob == r2.blow_up_prob
    assert r1.median_final_pnl == r2.median_final_pnl
    assert r1.p95_max_drawdown == r2.p95_max_drawdown


def test_blowup_certain_for_ruinous_sequence() -> None:
    # Each -3000 trade; trailing floor is 48000, so the first loss already breaches.
    result = monte_carlo([-3000.0] * 5, _rules(), runs=20, seed=1)
    assert result.blow_up_prob == 1.0
    assert result.risk_of_ruin == 1.0


def test_no_blowup_for_safe_sequence() -> None:
    result = monte_carlo([100.0, 100.0, 100.0], _rules(), runs=20, seed=1)
    assert result.blow_up_prob == 0.0


def test_empty_returns_zeroes() -> None:
    result = monte_carlo([], _rules(), runs=10, seed=1)
    assert result.runs == 0
    assert result.blow_up_prob == 0.0
