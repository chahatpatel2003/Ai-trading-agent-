from __future__ import annotations

from nqlab.backtest.engine import run_backtest
from nqlab.config.models import AppConfig, SessionConfig, TopstepRules
from nqlab.domain.calendar.sessions import SessionLabeler
from nqlab.domain.strategy.base import StrategyParams
from nqlab.validation.diagnostics import diag_summary, diagnose
from tests.synth import engine_frame


def _result(app_config: AppConfig, rules: TopstepRules):
    frame = engine_frame(
        [
            {"close": 100, "high": 100, "low": 100, "atr": 10, "signal": 1,
             "stop_pts": 10, "target_pts": 50},
            {"open": 100, "high": 100, "low": 100, "close": 100, "atr": 10, "minute": 605},
            {"open": 100, "high": 115, "low": 95, "close": 100, "atr": 10, "minute": 610},
            {"open": 100, "high": 100, "low": 89, "close": 95, "atr": 10, "minute": 615},
        ]
    )
    return run_backtest(
        frame, symbol="NQ.c", timeframe="5m", strategy=StrategyParams("t", "1"),
        config=app_config, rules=rules,
    )


def test_mae_mfe_recorded(app_config: AppConfig, rules_50k: TopstepRules) -> None:
    trade = _result(app_config, rules_50k).trades[0]
    assert trade.mfe > 0  # price went ~15 in favor
    assert trade.mae > 0  # and ~5 against before the stop


def test_diagnose_report(app_config: AppConfig, rules_50k: TopstepRules) -> None:
    trades = _result(app_config, rules_50k).trades
    diag = diagnose(trades, SessionLabeler.from_config(SessionConfig()))
    assert diag["n_trades"] == 1
    assert {"by_session", "by_hour", "by_exit_reason", "by_holding"} <= set(diag)
    assert "avg_mfe" in diag
    assert "direction_edge_before_costs" in diag


def test_diag_summary_compact() -> None:
    assert diag_summary([]) == {"n_trades": 0, "direction_edge_before_costs": False}
