"""Unit tests for the risk management module."""
import pytest
from risk.manager import RiskManager, RiskRejection

# Minimal config for testing
TEST_CFG = {
    "risk": {
        "risk_per_trade_pct": 0.0025,
        "max_risk_per_trade_pct": 0.005,
        "max_daily_drawdown_pct": 0.02,
        "max_total_drawdown_pct": 0.10,
        "max_open_positions": 2,
        "volatility_lookback": 20,
        "no_trade_during_news": True,
        "kill_switch": {
            "max_spread_multiple": 3.0,
            "max_consecutive_losses": 5,
            "max_data_gap_seconds": 300,
        },
    },
    "costs": {
        "spread_points": 0.30,
        "slippage_points": 0.10,
        "commission_per_lot": 0.0,
        "lot_size_oz": 100,
    },
}


def test_position_sizing_basic():
    rm = RiskManager(TEST_CFG, initial_equity=100_000)
    result = rm.calculate_position_size(
        entry_price=2000.0,
        stop_price=1980.0,
    )
    assert result.lots > 0
    assert result.rejection == RiskRejection.NONE
    # Risk should be ~0.25% of 100k = $250
    assert 200 <= result.risk_amount <= 500


def test_zero_stop_rejected():
    rm = RiskManager(TEST_CFG, initial_equity=100_000)
    result = rm.calculate_position_size(
        entry_price=2000.0,
        stop_price=2000.0,
    )
    assert result.lots == 0
    assert result.rejection == RiskRejection.ZERO_STOP


def test_daily_loss_limit():
    rm = RiskManager(TEST_CFG, initial_equity=100_000)
    # Simulate losing 2%+ in one day
    rm.record_trade_result(-2100)
    result = rm.calculate_position_size(
        entry_price=2000.0,
        stop_price=1980.0,
    )
    assert result.lots == 0
    assert result.rejection == RiskRejection.DAILY_LOSS_LIMIT


def test_total_drawdown_limit():
    rm = RiskManager(TEST_CFG, initial_equity=100_000)
    # Simulate losses across multiple days to avoid daily limit firing first
    rm.record_trade_result(-4_000)
    rm.state.daily_pnl = 0.0  # reset daily as if new day
    rm.record_trade_result(-4_000)
    rm.state.daily_pnl = 0.0
    rm.record_trade_result(-3_000)
    rm.state.daily_pnl = 0.0  # daily is clear, but total DD > 10%
    result = rm.calculate_position_size(
        entry_price=2000.0,
        stop_price=1980.0,
    )
    assert result.lots == 0
    assert result.rejection == RiskRejection.TOTAL_DRAWDOWN_LIMIT


def test_max_positions():
    rm = RiskManager(TEST_CFG, initial_equity=100_000)
    rm.on_position_opened()
    rm.on_position_opened()
    result = rm.calculate_position_size(
        entry_price=2000.0,
        stop_price=1980.0,
    )
    assert result.lots == 0
    assert result.rejection == RiskRejection.MAX_POSITIONS


def test_consecutive_losses_kill_switch():
    rm = RiskManager(TEST_CFG, initial_equity=100_000)
    for _ in range(5):
        rm.record_trade_result(-50)
    result = rm.calculate_position_size(
        entry_price=2000.0,
        stop_price=1980.0,
    )
    assert result.lots == 0
    assert result.rejection == RiskRejection.KILL_SWITCH_CONSECUTIVE_LOSSES


def test_spread_kill_switch():
    rm = RiskManager(TEST_CFG, initial_equity=100_000)
    # Normal spread is 0.30, threshold is 3x = 0.90
    result = rm.calculate_position_size(
        entry_price=2000.0,
        stop_price=1980.0,
        current_spread=1.0,  # exceeds 3x
    )
    assert result.lots == 0
    assert result.rejection == RiskRejection.KILL_SWITCH_SPREAD


def test_news_blackout():
    rm = RiskManager(TEST_CFG, initial_equity=100_000)
    result = rm.calculate_position_size(
        entry_price=2000.0,
        stop_price=1980.0,
        is_news_period=True,
    )
    assert result.lots == 0
    assert result.rejection == RiskRejection.NEWS_BLACKOUT


def test_equity_tracking():
    rm = RiskManager(TEST_CFG, initial_equity=100_000)
    rm.record_trade_result(500)
    assert rm.state.equity == 100_500
    assert rm.state.peak_equity == 100_500
    rm.record_trade_result(-200)
    assert rm.state.equity == 100_300
    assert rm.state.peak_equity == 100_500


def test_no_martingale():
    """Verify position sizing doesn't increase after losses."""
    rm = RiskManager(TEST_CFG, initial_equity=100_000)
    size_before = rm.calculate_position_size(2000, 1980)

    rm.record_trade_result(-500)
    size_after = rm.calculate_position_size(2000, 1980)

    # After a loss, equity is lower, so position size should be equal or smaller
    assert size_after.lots <= size_before.lots
