"""Minimal event-driven backtester (bar resolution)."""

from __future__ import annotations

from nqlab.backtest.engine import run_backtest
from nqlab.backtest.result import BacktestResult, Trade

__all__ = ["BacktestResult", "Trade", "run_backtest"]
