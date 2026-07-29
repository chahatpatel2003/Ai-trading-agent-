"""Phase 18 — scalp families, fixed-point stops, engine daily controls, win-rate verdict."""

from __future__ import annotations

import polars as pl

from nqlab.config.loader import load_config
from nqlab.config.models import TopstepRules
from nqlab.discovery.lockbox import ACCEPT_FOR_FORWARD_PAPER, REJECT
from nqlab.discovery.phase18 import PHASE18_THRESHOLDS, phase18_verdict
from nqlab.discovery.phase18_families import PHASE18_FAMILIES, generate_phase18_candidates
from nqlab.domain.strategy.base import ExecConfig
from tests.synth import engine_frame, globex_bars  # noqa: F401

_RULES = TopstepRules(
    name="150K", account_size=150_000.0, profit_target=9_000.0, trailing_max_loss=4_500.0,
    max_contracts=15, min_trading_days=2, consistency_pct=0.5,
)


# --------------------------------------------------------------------------- families
def test_phase18_has_six_scalp_families() -> None:
    assert set(PHASE18_FAMILIES) == {
        "p18_vwap_reclaim", "p18_or_meanrev", "p18_pullback_cont",
        "p18_sweep_micro", "p18_range_rotation", "p18_trendday_scalp",
    }


def test_phase18_candidates_use_fixed_point_scalp_targets() -> None:
    cands = generate_phase18_candidates(timeframe="5m")
    assert len(cands) == 6 * 4 * 2 * 1  # families x fixed-grid x risk x sizing
    p = cands[0].p
    assert "fix_stop" in p and "fix_target" in p
    assert p["hour_start"] == 9 and p["hour_end"] == 12
    # R:R spans 0.8-1.5 (small targets, tight stops) — never the 3-5 ATR large-R approach.
    rr = float(p["fix_target"]) / float(p["fix_stop"])
    assert 0.7 <= rr <= 1.6


def test_phase18_fixed_point_stop_target_emitted() -> None:
    # A fixed-point family emits literal stop/target point distances (not ATR-scaled).
    strat = PHASE18_FAMILIES["p18_vwap_reclaim"](
        {"fix_stop": 8.0, "fix_target": 10.0, "hour_start": 9, "hour_end": 12,
         "risk_per_trade": 150.0, "sizing_mode": "mnq_min1", "min_hold_bars": 0,
         "time_stop_bars": 12}
    )
    frame = globex_bars(days=2).with_columns(
        pl.col("close").rolling_mean(window_size=3, min_samples=1).alias("vwap_session"),
        pl.lit(10.0).alias("atr"),
    )
    sig = strat.signals(frame)
    stops = sig.filter(pl.col("signal") != 0)["stop_pts"].unique().to_list()
    targets = sig.filter(pl.col("signal") != 0)["target_pts"].unique().to_list()
    if stops:  # where it fires, the distances are the fixed points
        assert stops == [8.0]
        assert targets == [10.0]


# --------------------------------------------------------------------------- engine daily controls
def _const_signal_frame(n: int = 30) -> pl.DataFrame:
    # One signal per bar, all RTH, same day; alternating winners/losers handled by price path.
    rows = []
    base = 20000.0
    for i in range(n):
        rows.append({
            "ts": None, "minute_of_day": 570 + i, "trading_day": None, "is_rth": True,
            "open": base, "high": base + 5, "low": base - 5, "close": base,
            "signal": 1, "stop_pts": 4.0, "target_pts": 4.0, "atr": 5.0,
        })
    return pl.DataFrame(rows)


def test_exec_config_carries_daily_controls() -> None:
    ex = ExecConfig(max_daily_trades=3, daily_profit_target_usd=300.0, max_daily_losses=2)
    assert ex.max_daily_trades == 3
    assert ex.daily_profit_target_usd == 300.0
    assert ex.max_daily_losses == 2


def test_max_daily_trades_caps_entries() -> None:
    # With a generous signal stream, max_daily_trades=3 must cap entries per day to 3 (gated).
    from datetime import date, datetime

    cfg = load_config(profile="dev")
    day = date(2024, 1, 2)
    rows = []
    px = 20000.0
    ts = datetime(2024, 1, 2, 9, 30)
    for i in range(60):
        rows.append({
            "ts": ts, "minute_of_day": 575 + i, "trading_day": day, "is_rth": True,
            "open": px, "high": px + 20, "low": px - 1, "close": px + 10,
            "signal": 1, "stop_pts": 4.0, "target_pts": 8.0, "atr": 5.0,
        })
    frame = pl.DataFrame(rows)
    from nqlab.backtest.engine import run_backtest
    from nqlab.domain.strategy.base import StrategyParams

    params = StrategyParams("t", "1", ())
    capped = run_backtest(
        frame, symbol="NQ.c", timeframe="5m", strategy=params, config=cfg, rules=_RULES,
        gate_entries=True, exec_config=ExecConfig(sizing_mode="mnq_min1", max_daily_trades=3),
    )
    uncapped = run_backtest(
        frame, symbol="NQ.c", timeframe="5m", strategy=params, config=cfg, rules=_RULES,
        gate_entries=True, exec_config=ExecConfig(sizing_mode="mnq_min1"),
    )
    assert len(capped.trades) <= 3
    assert len(uncapped.trades) > len(capped.trades)


# --------------------------------------------------------------------------- verdict
def _good_scalp_business() -> dict[str, object]:
    return {
        "trade_count": 300, "profit_factor": 1.45, "win_rate": 0.62, "ruin": 0.05,
        "green_month_frac": 0.83, "winning_days_200": 60, "winning_day_rate": 0.4,
        "payout_eligible_months": 14, "total_months": 18, "blown": False,
        "p_pass_by_account": {"50k": 0.85, "150k": 0.8},
    }


def _good_scale() -> dict[str, object]:
    return {"profit_factor": 1.35, "avg_contracts": 3.0, "ruin": 0.06, "blown": False}


def test_phase18_verdict_caps_at_forward_paper() -> None:
    v, _ = phase18_verdict(
        gauntlet_verdict="ACCEPT", dsr=0.95, business=_good_scalp_business(), scale=_good_scale(),
        thresholds=PHASE18_THRESHOLDS, pristine=True,
    )
    assert v == ACCEPT_FOR_FORWARD_PAPER  # never live ACCEPT_FOR_PAPER, even pristine


def test_phase18_verdict_rejects_low_win_rate() -> None:
    biz = _good_scalp_business() | {"win_rate": 0.48}  # below 0.55 bar
    v, reason = phase18_verdict(
        gauntlet_verdict="ACCEPT", dsr=0.95, business=biz, scale=_good_scale(),
        thresholds=PHASE18_THRESHOLDS, pristine=True,
    )
    assert v == REJECT
    assert "win rate" in reason
