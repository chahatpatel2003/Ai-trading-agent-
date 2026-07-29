from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from nqlab.config.models import TopstepRules
from nqlab.discovery.lockbox import (
    ACCEPT_FOR_FORWARD_PAPER,
    NEEDS_MORE_DATA,
    REJECT,
    FrozenHypothesis,
    phase12_verdict,
)
from nqlab.discovery.phase12_families import PHASE12_FAMILIES, generate_phase12_candidates
from nqlab.discovery.phase12_stats import practical_stats
from tests.synth import validation_result


def test_generate_phase12_candidates_deterministic_and_unique() -> None:
    a = generate_phase12_candidates(timeframe="5m")
    b = generate_phase12_candidates(timeframe="5m")
    assert [s.params.fingerprint() for s in a] == [s.params.fingerprint() for s in b]
    assert len({s.params.fingerprint() for s in a}) == len(a)
    assert {s.family for s in a} == set(PHASE12_FAMILIES)  # all 5 trend_day variants


def test_phase12_family_frozen_reconstructs_exactly() -> None:
    params = {
        "stop_atr": 1.5,
        "target_atr": 4.0,
        "min_hold_bars": 3,
        "time_stop_bars": 48,
        "risk_per_trade": 100.0,
        "sizing_mode": "mnq_only",
        "hour_start": 9,
        "hour_end": 12,
    }
    fp = str(
        PHASE12_FAMILIES["trend_pullback"](dict(params), rth_start_min=570).params.fingerprint()
    )
    fh = FrozenHypothesis(
        "trend_pullback_5m_100k_r100",
        "trend_pullback",
        "5m",
        "100k",
        tuple(sorted(params.items())),
        "phase12_search_100k_5m",
        fp,
        1234.0,
        0.20,
        "t",
    )
    assert fh.config_hash() == fp


def _result(verdict: str):
    r = validation_result()
    r.verdict = verdict
    r.reasons = ["reason"]
    return r


def test_phase12_verdict_caps_at_forward_paper() -> None:
    thr = 0.9
    # Even a clean pass with high DSR is FORWARD paper only — never ACCEPT_FOR_PAPER,
    # because the 2025-2026 final test was inspected for trend_day in Phase 11.
    assert (
        phase12_verdict(_result("ACCEPT"), 0.99, thr, pristine=True)[0] == ACCEPT_FOR_FORWARD_PAPER
    )
    assert phase12_verdict(_result("ACCEPT"), 0.50, thr, pristine=True)[0] == REJECT
    assert phase12_verdict(_result("REJECT"), 0.99, thr, pristine=True)[0] == REJECT
    assert (
        phase12_verdict(_result("NEEDS_MORE_DATA"), 0.99, thr, pristine=True)[0] == NEEDS_MORE_DATA
    )


def _trade(net: float, day: date) -> object:
    return SimpleNamespace(net_pnl=net, trading_day=day)


def test_practical_stats_full_panel(rules_50k: TopstepRules) -> None:
    trades = [
        _trade(120.0, date(2025, 1, 2)),
        _trade(-60.0, date(2025, 1, 2)),
        _trade(200.0, date(2025, 1, 3)),
        _trade(-40.0, date(2025, 1, 6)),
        _trade(90.0, date(2025, 1, 7)),
    ]
    accounts = {"50k": rules_50k}
    s = practical_stats(
        trades,
        account_size=rules_50k.account_size,
        rules=rules_50k,
        months=2.0,
        rules_by_account=accounts,
        runs=200,
        seed=1,
    )
    assert s["trade_count"] == 5
    assert 0.0 <= s["win_rate"] <= 1.0
    assert s["profit_factor"] > 0
    for key in (
        "expectancy",
        "avg_trade",
        "net_pnl",
        "monthly_roi",
        "annual_roi",
        "ruin",
        "topstep_pass_rate",
        "p_pass_by_account",
        "est_monthly_payout",
        "consistency_pct",
        "largest_win_day_pct",
        "trades_per_day",
    ):
        assert key in s, f"missing stat {key}"
    assert "50k" in s["p_pass_by_account"]


def test_practical_stats_empty() -> None:
    assert practical_stats([], account_size=50000.0, rules=None, months=2.0)["trade_count"] == 0  # type: ignore[arg-type]
