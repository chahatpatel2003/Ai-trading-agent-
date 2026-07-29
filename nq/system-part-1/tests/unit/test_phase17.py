"""Phase 17 — A+ families, funded withdrawal model, and the NEEDS_FORWARD_DATA verdict."""

from __future__ import annotations

from nqlab.config.models import TopstepRules
from nqlab.discovery.lockbox import ACCEPT_FOR_PAPER, NEEDS_FORWARD_DATA, REJECT
from nqlab.discovery.phase17 import (
    PHASE17_FAMILIES,
    PHASE17_THRESHOLDS,
    generate_phase17_candidates,
    phase17_verdict,
)
from nqlab.domain.risk.payout import simulate_funded_withdrawals

_RULES = TopstepRules(
    name="150K", account_size=150_000.0, profit_target=9_000.0, trailing_max_loss=4_500.0,
    max_contracts=15, min_trading_days=2, consistency_pct=0.5,
)


# --------------------------------------------------------------------------- families
def test_phase17_has_six_aplus_families() -> None:
    assert set(PHASE17_FAMILIES) == {
        "p17_sweep_reversal", "p17_expansion_cont", "p17_vol_expansion",
        "p17_opening_drive_aplus", "p17_failed_breakout", "p17_premium_discount",
    }


def test_phase17_candidates_are_ny_am_asymmetric() -> None:
    cands = generate_phase17_candidates(timeframe="5m")
    assert len(cands) == 6 * 4 * 3 * 2  # families x grid x risk x sizing
    p = cands[0].p
    assert p["hour_start"] == 9 and p["hour_end"] == 12
    assert p["target_atr"] >= 3.0 >= p["stop_atr"]  # asymmetric R (wide target, tight stop)


# --------------------------------------------------------------------------- withdrawals
def test_funded_withdrawals_pays_out_then_keeps_buffer() -> None:
    # 10 clean $400 winning days -> >=5 winning days, consistent, well above buffer.
    daily = {f"2025-0{(i // 20) + 1}-{(i % 20) + 1:02d}": 400.0 for i in range(40)}
    res = simulate_funded_withdrawals(daily, rules=_RULES)
    assert res.n_payouts >= 1
    assert res.total_withdrawn > 0
    assert res.blown_after_payout is False
    # equity is held at start + safety buffer after withdrawing.
    assert res.ending_equity <= _RULES.account_size + _RULES.trailing_max_loss + 1e-6


def test_funded_withdrawals_blow_after_payout_when_drawdown_follows() -> None:
    # Build a big buffer + 5 winning days, take a payout, then a deep drawdown blows the
    # now-reduced cushion.
    daily = {f"2025-01-{d:02d}": 2_000.0 for d in range(2, 9)}  # 7 winning days, +14k
    daily.update({"2025-02-03": -9_000.0})  # after withdrawal, a big loss breaches the floor
    res = simulate_funded_withdrawals(daily, rules=_RULES)
    assert res.n_payouts >= 1
    assert res.blown_after_payout is True


# --------------------------------------------------------------------------- verdict
def _good_business() -> dict[str, object]:
    return {
        "trade_count": 140, "profit_factor": 1.6, "ruin": 0.05, "green_month_frac": 0.83,
        "winning_days_200": 45, "winning_day_rate": 0.3, "payout_eligible_months": 13,
        "total_months": 18, "blown": False, "p_pass_by_account": {"50k": 0.82, "150k": 0.78},
    }


def _good_scale() -> dict[str, object]:
    return {"profit_factor": 1.4, "avg_contracts": 3.0, "ruin": 0.06, "blown": False}


def test_phase17_verdict_accepts_strong_aplus() -> None:
    v, reason = phase17_verdict(
        gauntlet_verdict="ACCEPT", dsr=0.95, business=_good_business(), scale=_good_scale(),
        thresholds=PHASE17_THRESHOLDS, pristine=True,
    )
    assert v == ACCEPT_FOR_PAPER, reason


def test_phase17_verdict_rejects_below_75pct_green() -> None:
    biz = _good_business() | {"green_month_frac": 0.6}  # below 0.75 bar
    v, reason = phase17_verdict(
        gauntlet_verdict="ACCEPT", dsr=0.95, business=biz, scale=_good_scale(),
        thresholds=PHASE17_THRESHOLDS, pristine=True,
    )
    assert v == REJECT
    assert "green months" in reason


def test_phase17_verdict_needs_forward_data_for_mbp1_without_clean_lockbox() -> None:
    # An otherwise-accept-worthy MBP1-required strategy is capped at NEEDS_FORWARD_DATA.
    biz = _good_business() | {"requires_mbp1": True, "mbp1_clean": False}
    v, reason = phase17_verdict(
        gauntlet_verdict="ACCEPT", dsr=0.95, business=biz, scale=_good_scale(),
        thresholds=PHASE17_THRESHOLDS, pristine=True,
    )
    assert v == NEEDS_FORWARD_DATA
    assert "MBP1" in reason
