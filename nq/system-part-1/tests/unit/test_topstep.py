from __future__ import annotations

from datetime import date

from nqlab.config.models import TopstepRules
from nqlab.domain.risk.topstep import TopstepAccount, simulate_account

_DAY = date(2024, 1, 2)


def _rules(**overrides: object) -> TopstepRules:
    base: dict[str, object] = {
        "name": "T",
        "account_size": 50000.0,
        "profit_target": 3000.0,
        "trailing_max_loss": 2000.0,
        "max_contracts": 5,
        "min_trading_days": 0,
        "consistency_pct": None,
    }
    base.update(overrides)
    return TopstepRules.model_validate(base)


def test_trailing_drawdown_breach() -> None:
    acc = TopstepAccount(_rules())
    acc.mark(50000.0, _DAY)
    acc.mark(51000.0, _DAY)  # peak 51000 -> floor min(49000, 50000) = 49000
    acc.mark(48999.0, _DAY)  # below floor -> blown
    result = acc.finalize()
    assert result.blown
    assert result.account_failures == 1
    assert any("trailing_drawdown_breach" in v for v in result.violations)


def test_no_breach_above_floor() -> None:
    result = simulate_account(
        [(_DAY, 50000.0), (_DAY, 49500.0)], _rules()  # floor 48000; 49500 is safe
    )
    assert not result.blown


def test_lock_trailing_at_start() -> None:
    acc = TopstepAccount(_rules())
    acc.mark(50000.0, _DAY)
    acc.mark(55000.0, _DAY)  # peak 55000 -> floor locks at start 50000 (not 53000)
    acc.mark(49999.0, _DAY)
    assert acc.finalize().blown


def test_daily_loss_halt() -> None:
    acc = TopstepAccount(_rules(daily_loss_limit=1000.0))
    acc.mark(50000.0, _DAY)
    assert acc.can_open()
    acc.mark(48999.0, _DAY)  # -1001 from day start -> halt
    assert not acc.can_open()
    assert any("daily_loss_halt" in v for v in acc.finalize().violations)


def test_pass_detection_and_days_to_pass() -> None:
    result = simulate_account(
        [(_DAY, 50000.0), (_DAY, 53000.0)], _rules(min_trading_days=1)
    )
    assert result.passed
    assert result.days_to_pass == 1
    assert not result.blown


def test_min_trading_days_blocks_pass() -> None:
    result = simulate_account([(_DAY, 53000.0)], _rules(min_trading_days=5))
    assert not result.passed


# --- Locked trailing maximum-loss-limit (Topstep MLL) audit ---------------------
# The drawdown budget is NOT a fixed $2K. For a 50K account
# (trailing_max_loss=2000, lock_trailing_at_start=True) the loss-limit floor is
#   floor = min(peak - 2000, 50000)
# i.e. it starts at 48000, trails UP with the equity peak, LOCKS at the 50000
# starting balance, and after the lock any profit above 50000 is real buffer.


def test_50k_starts_with_48k_loss_limit() -> None:
    # Rule (1): a fresh 50K account can lose down to 48,000 before breaching.
    acc = TopstepAccount(_rules())
    acc.mark(50000.0, _DAY)
    assert acc.trailing_floor == 48000.0  # 50000 - 2000
    # Just above the floor survives; at/below the floor breaches.
    acc.mark(48001.0, _DAY)
    assert not acc.blown
    acc.mark(48000.0, _DAY)
    assert acc.blown


def test_profits_trail_the_loss_limit_upward() -> None:
    # Rule (2): the floor trails the equity PEAK upward (and never falls on a dip).
    acc = TopstepAccount(_rules())
    acc.mark(50000.0, _DAY)
    assert acc.trailing_floor == 48000.0
    acc.mark(50500.0, _DAY)  # peak 50500 -> floor 48500
    assert acc.trailing_floor == 48500.0
    acc.mark(51000.0, _DAY)  # peak 51000 -> floor 49000
    assert acc.trailing_floor == 49000.0
    acc.mark(50200.0, _DAY)  # a dip does NOT lower the floor (it trails the peak)
    assert acc.trailing_floor == 49000.0
    assert not acc.blown


def test_loss_limit_locks_at_starting_balance() -> None:
    # Rule (3): once the peak reaches 52,000 the floor reaches the 50,000 start
    # and locks there permanently, no matter how high the peak climbs after.
    acc = TopstepAccount(_rules())
    acc.mark(50000.0, _DAY)
    acc.mark(52000.0, _DAY)  # peak 52000 -> floor min(50000, 50000) = 50000 (locked)
    assert acc.trailing_floor == 50000.0
    acc.mark(53000.0, _DAY)  # peak 53000 -> floor min(51000, 50000) = 50000 (stays)
    assert acc.trailing_floor == 50000.0
    acc.mark(60000.0, _DAY)  # peak 60000 -> floor still 50000
    assert acc.trailing_floor == 50000.0


def test_after_lock_profits_become_real_buffer() -> None:
    # Rule (4): at peak 55,000 the usable buffer is 5,000 (down to the 50,000 lock),
    # NOT 2,000. A 2,000 drawdown (to 53,000) must not blow the account.
    acc = TopstepAccount(_rules())
    acc.mark(50000.0, _DAY)
    acc.mark(55000.0, _DAY)  # locked floor 50000
    assert acc.trailing_floor == 50000.0
    acc.mark(53000.0, _DAY)  # -2000 from peak: would blow a naive $2K model
    assert not acc.blown
    acc.mark(50001.0, _DAY)  # -4999 from peak: still inside the 5,000 buffer
    assert not acc.blown


def test_50k_to_55k_can_draw_to_50k_not_53k() -> None:
    # Rule (5) — the headline: 50K -> 55K can draw down to 50,000, not 53,000.
    acc = TopstepAccount(_rules())
    acc.mark(50000.0, _DAY)
    acc.mark(55000.0, _DAY)
    assert acc.trailing_floor == 50000.0  # locked at start, not 53000
    acc.mark(53000.0, _DAY)
    assert not acc.blown                   # 53000 is well above the 50000 floor
    acc.mark(50000.0, _DAY)
    assert acc.blown                       # equity <= floor (50000) -> breach


def test_without_lock_floor_does_not_lock_at_start() -> None:
    # Control: disable the lock and the buffer collapses back to a flat $2K trail.
    # At peak 55,000 the floor is 53,000, so a draw to 53,000 DOES blow — proving
    # the lock flag is exactly what turns above-start profit into real buffer.
    acc = TopstepAccount(_rules(lock_trailing_at_start=False))
    acc.mark(50000.0, _DAY)
    acc.mark(55000.0, _DAY)  # peak 55000 -> floor 53000 (no lock)
    assert acc.trailing_floor == 53000.0
    acc.mark(53000.0, _DAY)  # equity 53000 <= floor 53000 -> blown
    assert acc.blown


def test_montecarlo_breach_matches_locked_buffer() -> None:
    # The Monte-Carlo ruin estimator must agree with the live account on the
    # locked-buffer floor. After +5000 of profit (peak 55000, floor locked at
    # 50000) a single -5000 trade sits exactly on the floor and breaches, while
    # -4999 survives. With the lock off, -2000 from the 53000 floor already blows.
    from nqlab.validation.montecarlo import monte_carlo

    rules = _rules(min_trading_days=0)
    # locked: +5000 then -5000 -> equity 50000 == floor 50000 -> breach
    assert monte_carlo([5000.0, -5000.0], rules, runs=1, seed=0).risk_of_ruin == 1.0
    # locked: +5000 then -4999 -> equity 50001 > floor 50000 -> survives
    assert monte_carlo([5000.0, -4999.0], rules, runs=1, seed=0).risk_of_ruin == 0.0
    # lock off: +5000 (peak 55000, floor 53000) then -2000 -> 53000 <= 53000 -> breach
    no_lock = _rules(lock_trailing_at_start=False)
    assert monte_carlo([5000.0, -2000.0], no_lock, runs=1, seed=0).risk_of_ruin == 1.0
