"""Unit tests for the live trading gate — verifying it blocks by default."""
import os
import pytest
from live_trade_disabled.gate import (
    LiveTradingDisabledError,
    block_live_trading,
    check_live_trading_gate,
)


def test_live_trading_blocked_by_default():
    cfg = {"general": {"live_trading_enabled": False}}
    allowed, failures = check_live_trading_gate(cfg)
    assert not allowed
    assert len(failures) > 0


def test_live_trading_blocked_without_env():
    cfg = {"general": {"live_trading_enabled": True}}
    # Ensure env var is not set
    os.environ.pop("ENABLE_LIVE_TRADING", None)
    allowed, failures = check_live_trading_gate(cfg)
    assert not allowed


def test_block_raises_exception():
    with pytest.raises(LiveTradingDisabledError):
        block_live_trading("test reason")


def test_config_false_blocks():
    cfg = {"general": {"live_trading_enabled": False}}
    os.environ["ENABLE_LIVE_TRADING"] = "true"
    allowed, failures = check_live_trading_gate(cfg)
    assert not allowed  # config is false, even if env is true
    os.environ.pop("ENABLE_LIVE_TRADING", None)
