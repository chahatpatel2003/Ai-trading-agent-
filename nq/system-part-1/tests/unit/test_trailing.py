from __future__ import annotations

import pytest

from nqlab.backtest.engine import run_backtest
from nqlab.config.models import AppConfig, TopstepRules
from nqlab.domain.strategy.base import StrategyParams, TrailingConfig
from tests.synth import engine_frame

_PARAMS = StrategyParams("t", "1")


def _run(frame, config, rules, trailing):
    return run_backtest(
        frame, symbol="NQ.c", timeframe="5m", strategy=_PARAMS,
        config=config, rules=rules, trailing=trailing,
    )


def test_atr_trailing_stop_locks_in_profit(app_config: AppConfig, rules_50k: TopstepRules) -> None:
    # Long; price runs to 120 then pulls back. Trailing stop (1*ATR=10) should exit
    # well above the original 90.25 stop, banking a profit.
    frame = engine_frame(
        [
            {"close": 100, "high": 100, "low": 100, "atr": 10, "signal": 1,
             "stop_pts": 10, "target_pts": 100},
            {"open": 100, "high": 100, "low": 100, "close": 100, "atr": 10, "minute": 605},
            {"open": 100, "high": 120, "low": 110, "close": 115, "atr": 10, "minute": 610},
            {"open": 110, "high": 111, "low": 109, "close": 110, "atr": 10, "minute": 615},
        ]
    )
    trade = _run(frame, app_config, rules_50k, TrailingConfig(mode="atr", trail_atr=1.0)).trades[0]
    assert trade.exit_reason == "STOP"
    assert trade.net_pnl > 0  # trailing locked profit
    assert trade.stop_px > trade.entry_px - 10  # stop was trailed up


def test_breakeven_stop_moves_to_entry(app_config: AppConfig, rules_50k: TopstepRules) -> None:
    # After a 1R favorable move, the stop moves to entry; the later dip exits at
    # breakeven rather than the full -10pt stop.
    frame = engine_frame(
        [
            {"close": 100, "high": 100, "low": 100, "atr": 10, "signal": 1,
             "stop_pts": 10, "target_pts": 100},
            {"open": 100, "high": 100, "low": 100, "close": 100, "atr": 10, "minute": 605},
            {"open": 105, "high": 111, "low": 105, "close": 110, "atr": 10, "minute": 610},  # +1R
            {"open": 100, "high": 101, "low": 99, "close": 100, "atr": 10, "minute": 615},
        ]
    )
    trade = _run(frame, app_config, rules_50k, TrailingConfig(breakeven_r=1.0)).trades[0]
    assert trade.exit_reason == "STOP"
    assert trade.stop_px == pytest.approx(trade.entry_px)  # moved to breakeven (was entry-10)
