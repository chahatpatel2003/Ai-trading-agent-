from __future__ import annotations

from nqlab.backtest.engine import run_backtest
from nqlab.config.models import AppConfig, TopstepRules
from nqlab.domain.strategy.base import StrategyParams
from tests.synth import engine_frame

_PARAMS = StrategyParams("test", "1")


def _run(frame, config: AppConfig, rules: TopstepRules, entry_mode: str | None = None):
    return run_backtest(
        frame, symbol="NQ.c", timeframe="5m", strategy=_PARAMS,
        config=config, rules=rules, entry_mode=entry_mode,
    )


def test_next_open_fills_at_next_bar_open(app_config: AppConfig, rules_50k: TopstepRules) -> None:
    # Signal on bar 0; entry must occur at bar 1's OPEN (110), not bar 0's close (100).
    frame = engine_frame(
        [
            {"close": 100, "high": 100, "low": 100, "signal": 1, "stop_pts": 10, "target_pts": 20},
            {"open": 110, "high": 110, "low": 110, "close": 110, "minute": 605},
            {"open": 110, "high": 110, "low": 110, "close": 110, "minute": 955},  # EOD flat
        ]
    )
    trades = _run(frame, app_config, rules_50k).trades
    assert len(trades) == 1
    assert trades[0].entry_px == 110.25  # open 110 + 1 tick slippage
    assert trades[0].exit_reason == "EOD_FLAT"


def test_next_open_long_stop(app_config: AppConfig, rules_50k: TopstepRules) -> None:
    frame = engine_frame(
        [
            {"close": 100, "high": 100, "low": 100, "signal": 1, "stop_pts": 10, "target_pts": 20},
            {"open": 100, "high": 100, "low": 100, "close": 100, "minute": 605},  # entry 100.25
            {"open": 100, "high": 100, "low": 89, "close": 95, "minute": 610},  # low pierces stop
        ]
    )
    trade = _run(frame, app_config, rules_50k).trades[0]
    assert trade.exit_reason == "STOP"
    assert trade.net_pnl < 0


def test_next_open_long_target(app_config: AppConfig, rules_50k: TopstepRules) -> None:
    frame = engine_frame(
        [
            {"close": 100, "high": 100, "low": 100, "signal": 1, "stop_pts": 10, "target_pts": 20},
            {"open": 100, "high": 100, "low": 100, "close": 100, "minute": 605},
            {"open": 100, "high": 121, "low": 100, "close": 120, "minute": 610},  # reaches target
        ]
    )
    trade = _run(frame, app_config, rules_50k).trades[0]
    assert trade.exit_reason == "TARGET"
    assert trade.net_pnl > 0


def test_current_close_debug_mode(app_config: AppConfig, rules_50k: TopstepRules) -> None:
    # Debug mode fills on the signal bar's close (optimistic) — used by the fills check.
    frame = engine_frame(
        [
            {"close": 100, "high": 100, "low": 100, "signal": 1, "stop_pts": 10, "target_pts": 20},
            {"open": 100, "high": 100, "low": 89, "close": 95, "minute": 605},
        ]
    )
    trade = _run(frame, app_config, rules_50k, entry_mode="current_close").trades[0]
    assert trade.entry_px == 100.25  # signal-bar close + slippage
    assert trade.exit_reason == "STOP"


def test_no_entry_after_primary_cutoff(app_config: AppConfig, rules_50k: TopstepRules) -> None:
    # minute 800 (13:20 ET) is past the 12:30 capture cutoff -> nothing is captured.
    frame = engine_frame(
        [
            {"close": 100, "high": 100, "low": 100, "signal": 1, "stop_pts": 10,
             "target_pts": 20, "minute": 800},
            {"open": 100, "high": 100, "low": 70, "close": 80, "minute": 805},
        ]
    )
    assert _run(frame, app_config, rules_50k).trades == []


def test_backtest_is_deterministic(app_config: AppConfig, rules_50k: TopstepRules) -> None:
    bars = [
        {"close": 100, "high": 100, "low": 100, "signal": 1, "stop_pts": 8, "target_pts": 12},
        {"open": 100, "high": 113, "low": 100, "close": 108, "minute": 605},
        {"open": 108, "high": 108, "low": 100, "close": 100, "signal": -1, "stop_pts": 8,
         "target_pts": 12, "minute": 610},
        {"open": 100, "high": 100, "low": 86, "close": 92, "minute": 615},
        {"open": 92, "high": 92, "low": 92, "close": 92, "minute": 955},
    ]
    r1 = _run(engine_frame(bars), app_config, rules_50k)
    r2 = _run(engine_frame(bars), app_config, rules_50k)
    assert [t.net_pnl for t in r1.trades] == [t.net_pnl for t in r2.trades]
    assert r1.final_equity == r2.final_equity
