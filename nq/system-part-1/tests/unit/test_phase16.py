"""Phase 16 — payout-first stats, fleet scaling, families, and the strict verdict."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from nqlab.backtest.result import Trade
from nqlab.config.models import TopstepRules
from nqlab.discovery.lockbox import (
    ACCEPT_FOR_PAPER,
    NEEDS_MORE_DATA,
    REJECT,
)
from nqlab.discovery.phase16 import (
    PHASE16_THRESHOLDS,
    business_stats,
    phase16_verdict,
)
from nqlab.discovery.phase16_families import PHASE16_FAMILIES, generate_phase16_candidates
from nqlab.domain.risk.payout import (
    payout_projection,
    scale_funded_fleet,
    winning_day_stats,
)

_RULES = TopstepRules(
    name="150K", account_size=150_000.0, profit_target=9_000.0, trailing_max_loss=4_500.0,
    max_contracts=15, min_trading_days=2, consistency_pct=0.5,
)
_RBA = {
    "50k": TopstepRules(name="50K", account_size=50_000.0, profit_target=3_000.0,
                        trailing_max_loss=2_000.0, max_contracts=5, consistency_pct=0.5),
    "150k": _RULES,
}


def _trade(day: date, net: float, *, contracts: int = 2, minute: int = 600) -> Trade:
    ts = datetime(day.year, day.month, day.day, 9, 35)
    return Trade(
        entry_ts=ts, exit_ts=ts, trading_day=day, side=1, instrument="MNQ", contracts=contracts,
        entry_px=20000.0, exit_px=20000.0 + net, stop_px=19990.0, target_px=20040.0,
        gross_pnl=net, fees=0.0, net_pnl=net, exit_reason="TARGET", bars_held=5,
        entry_minute=minute,
    )


# --------------------------------------------------------------------------- winning days
def test_winning_day_stats_counts_200_and_500_days() -> None:
    daily = {"2025-01-02": 250.0, "2025-01-03": 50.0, "2025-01-06": 600.0, "2025-01-07": -300.0}
    wd = winning_day_stats(daily)
    assert wd.trading_days == 4
    assert wd.winning_days_200 == 2  # 250 and 600
    assert wd.winning_days_500 == 1  # 600
    assert wd.best_day == 600.0
    assert wd.winning_day_rate == pytest.approx(0.5)


# --------------------------------------------------------------------------- payout projection
def test_payout_projection_requires_five_winning_days_and_consistency() -> None:
    # Six clean $300 winning days across two months -> eligible once 5 winning days exist,
    # consistency fine (no day > 50% of total).
    daily = {f"2025-01-{d:02d}": 300.0 for d in range(2, 8)}  # 6 winning days in Jan
    daily.update({f"2025-02-{d:02d}": 300.0 for d in range(3, 8)})  # 5 more in Feb
    proj = payout_projection(daily, rules=_RULES, months_total=2, blown=False)
    assert proj.payout_eligible_months >= 1
    assert proj.total_projected_payout > 0
    assert proj.consistency_ok is True


def test_payout_projection_blocks_when_blown_or_inconsistent() -> None:
    daily = {f"2025-01-{d:02d}": 300.0 for d in range(2, 8)}
    blown = payout_projection(daily, rules=_RULES, months_total=1, blown=True)
    assert blown.payout_eligible_months == 0  # account blown -> no payout
    # One huge day dominating profit -> consistency violation.
    spike = {"2025-01-02": 5_000.0, "2025-01-03": 100.0, "2025-01-06": 100.0,
             "2025-01-07": 100.0, "2025-01-08": 100.0, "2025-01-09": 100.0}
    proj = payout_projection(spike, rules=_RULES, months_total=1, blown=False)
    assert proj.consistency_ok is False
    assert proj.payout_eligible_months == 0


# --------------------------------------------------------------------------- fleet scaling
def test_scale_funded_fleet_is_linear_and_correlated() -> None:
    f = scale_funded_fleet(per_account_monthly_payout=1_000.0, per_account_ruin=0.05, n_accounts=10)
    assert f.fleet_monthly_payout == 10_000.0
    assert f.fleet_annual_payout == 120_000.0
    assert f.per_account_ruin == 0.05  # correlated -> fleet ruin == single-account ruin
    assert f.correlated is True


# --------------------------------------------------------------------------- families
def test_phase16_families_generate_deterministic_candidates() -> None:
    cands = generate_phase16_candidates(timeframe="5m")
    # 6 families x 4 grid x 3 risk x 2 sizing.
    assert len(cands) == len(PHASE16_FAMILIES) * 4 * 3 * 2
    # NY-AM mask + asymmetric R seeded into every candidate's params.
    p = cands[0].p
    assert p["hour_start"] == 9 and p["hour_end"] == 12
    assert p["target_atr"] >= 3.0 and p["stop_atr"] <= 1.5


# --------------------------------------------------------------------------- business stats
def test_business_stats_panel_has_payout_fields() -> None:
    days = [date(2025, 1, 6 + i) for i in range(20)]
    trades = [_trade(d, 260.0 if i % 3 else -120.0) for i, d in enumerate(days)]
    daily = {d.isoformat(): t.net_pnl for d, t in zip(days, trades, strict=True)}
    b = business_stats(
        trades, daily_pnl=daily, blown=False, account="150k", rules=_RULES,
        rules_by_account=_RBA, months=1.0, runs=200,
    )
    for key in ("profit_factor", "winning_days_200", "payout_eligible_months", "est_monthly_payout",
                "pass_by_account", "ruin_by_account", "best_account", "avg_contracts"):
        assert key in b
    assert b["avg_contracts"] == 2.0


# --------------------------------------------------------------------------- verdict
def _good_business() -> dict[str, object]:
    return {
        "trade_count": 120, "profit_factor": 1.45, "ruin": 0.05, "green_month_frac": 0.9,
        "winning_days_200": 40, "winning_day_rate": 0.3, "payout_eligible_months": 14,
        "total_months": 18, "blown": False, "p_pass_by_account": {"50k": 0.8, "150k": 0.75},
    }


def _good_scale() -> dict[str, object]:
    return {"profit_factor": 1.35, "avg_contracts": 3.0, "ruin": 0.06, "blown": False}


def test_phase16_verdict_accepts_only_when_every_gate_clears() -> None:
    v, reason = phase16_verdict(
        gauntlet_verdict="ACCEPT", dsr=0.95, business=_good_business(), scale=_good_scale(),
        thresholds=PHASE16_THRESHOLDS, pristine=True,
    )
    assert v == ACCEPT_FOR_PAPER, reason


def test_phase16_verdict_rejects_thin_pf() -> None:
    biz = _good_business() | {"profit_factor": 1.15}
    v, reason = phase16_verdict(
        gauntlet_verdict="ACCEPT", dsr=0.95, business=biz, scale=_good_scale(),
        thresholds=PHASE16_THRESHOLDS, pristine=True,
    )
    assert v == REJECT
    assert "PF" in reason


def test_phase16_verdict_rejects_tiny_size_scaling() -> None:
    scale = _good_scale() | {"avg_contracts": 1.0}  # depends on 1 MNQ only
    v, reason = phase16_verdict(
        gauntlet_verdict="ACCEPT", dsr=0.95, business=_good_business(), scale=scale,
        thresholds=PHASE16_THRESHOLDS, pristine=True,
    )
    assert v == REJECT
    assert "tiny sizing" in reason


def test_phase16_verdict_needs_more_data_when_too_few_trades() -> None:
    biz = _good_business() | {"trade_count": 10}
    v, _ = phase16_verdict(
        gauntlet_verdict="ACCEPT", dsr=0.95, business=biz, scale=_good_scale(),
        thresholds=PHASE16_THRESHOLDS, pristine=True,
    )
    assert v == NEEDS_MORE_DATA
