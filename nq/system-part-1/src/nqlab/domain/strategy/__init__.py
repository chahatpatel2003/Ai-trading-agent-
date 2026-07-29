"""Strategy contract: deterministic, PIT-only signal generation."""

from __future__ import annotations

from nqlab.domain.strategy.base import (
    MBP1_OPTIONAL,
    MBP1_REQUIRED,
    SIGNAL_COLUMNS,
    TRADES_ONLY,
    BaseStrategy,
    ExecConfig,
    Strategy,
    StrategyParams,
    TrailingConfig,
    attach_signals,
)

__all__ = [
    "MBP1_OPTIONAL",
    "MBP1_REQUIRED",
    "SIGNAL_COLUMNS",
    "TRADES_ONLY",
    "BaseStrategy",
    "ExecConfig",
    "Strategy",
    "StrategyParams",
    "TrailingConfig",
    "attach_signals",
]
