"""Phase 7 execution-override behavior: min-hold gating and risk-driven sizing."""

from __future__ import annotations

from nqlab.backtest.engine import run_backtest
from nqlab.config.models import AppConfig, TopstepRules
from nqlab.domain.strategy.base import ExecConfig, StrategyParams
from tests.synth import engine_frame

_PARAMS = StrategyParams("test", "1")


def _run(frame, config, rules, exec_config):
    return run_backtest(
        frame, symbol="NQ.c", timeframe="5m", strategy=_PARAMS,
        config=config, rules=rules, exec_config=exec_config,
    )


def test_min_hold_suppresses_early_target(app_config: AppConfig, rules_50k: TopstepRules) -> None:
    # Target (105.25) is reachable on the entry bar; min-hold should let the winner run.
    bars = [
        {"close": 100, "high": 100, "low": 100, "signal": 1, "stop_pts": 10,
         "target_pts": 5, "minute": 600},
        {"open": 100, "high": 110, "low": 100, "close": 108, "minute": 605},
        {"open": 108, "high": 110, "low": 105, "close": 108, "minute": 610},
        {"open": 108, "high": 108, "low": 108, "close": 108, "minute": 955},  # EOD flat
    ]
    no_hold = _run(engine_frame(bars), app_config, rules_50k, ExecConfig(min_hold_bars=0))
    assert no_hold.trades[0].exit_reason == "TARGET"
    assert no_hold.trades[0].bars_held == 0

    held = _run(engine_frame(bars), app_config, rules_50k, ExecConfig(min_hold_bars=5))
    assert held.trades[0].exit_reason == "EOD_FLAT"
    assert held.trades[0].bars_held >= 2


def test_min_hold_no_lookahead_only_delays(
    app_config: AppConfig, rules_50k: TopstepRules
) -> None:
    # min-hold must never *create* exits or move them earlier; it can only delay TARGET.
    bars = [
        {"close": 100, "high": 100, "low": 100, "signal": 1, "stop_pts": 10,
         "target_pts": 5, "minute": 600},
        {"open": 100, "high": 110, "low": 100, "close": 108, "minute": 605},
        {"open": 108, "high": 108, "low": 108, "close": 108, "minute": 955},
    ]
    base = _run(engine_frame(bars), app_config, rules_50k, ExecConfig(min_hold_bars=0))
    delayed = _run(engine_frame(bars), app_config, rules_50k, ExecConfig(min_hold_bars=3))
    assert base.trades[0].bars_held <= delayed.trades[0].bars_held


def test_risk_override_drives_instrument_and_mnq_fallback(
    app_config: AppConfig, rules_50k: TopstepRules
) -> None:
    # stop_pts=5 -> NQ per-contract risk = $100. $250 affords 2 NQ; $25 falls back to MNQ.
    bars = [
        {"close": 100, "high": 100, "low": 100, "signal": 1, "stop_pts": 5,
         "target_pts": 10, "minute": 600},
        {"open": 100, "high": 100, "low": 100, "close": 100, "minute": 605},
        {"open": 100, "high": 112, "low": 100, "close": 110, "minute": 610},
    ]
    big = _run(engine_frame(bars), app_config, rules_50k, ExecConfig(risk_per_trade_usd=250.0))
    small = _run(engine_frame(bars), app_config, rules_50k, ExecConfig(risk_per_trade_usd=25.0))
    assert big.trades[0].instrument == "NQ"
    assert big.trades[0].contracts == 2
    assert small.trades[0].instrument == "MNQ"


def test_exec_override_is_deterministic(app_config: AppConfig, rules_50k: TopstepRules) -> None:
    bars = [
        {"close": 100, "high": 100, "low": 100, "signal": 1, "stop_pts": 8,
         "target_pts": 12, "minute": 600},
        {"open": 100, "high": 113, "low": 100, "close": 108, "minute": 605},
        {"open": 108, "high": 108, "low": 108, "close": 108, "minute": 955},
    ]
    cfg = ExecConfig(risk_per_trade_usd=75.0, min_hold_bars=2)
    r1 = _run(engine_frame(bars), app_config, rules_50k, cfg)
    r2 = _run(engine_frame(bars), app_config, rules_50k, cfg)
    assert [t.net_pnl for t in r1.trades] == [t.net_pnl for t in r2.trades]
    assert r1.final_equity == r2.final_equity
