"""
Risk management module.
Enforces position sizing, drawdown limits, daily loss limits, and kill switches.
No martingale. No doubling down. No revenge trades.
"""
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum

import numpy as np
import pandas as pd

from logs.logger import get_logger

log = get_logger("risk.manager")


class RiskRejection(Enum):
    NONE = "none"
    DAILY_LOSS_LIMIT = "daily_loss_limit_hit"
    TOTAL_DRAWDOWN_LIMIT = "total_drawdown_limit_hit"
    MAX_POSITIONS = "max_open_positions"
    KILL_SWITCH_SPREAD = "kill_switch_spread"
    KILL_SWITCH_CONSECUTIVE_LOSSES = "kill_switch_consecutive_losses"
    KILL_SWITCH_DATA_GAP = "kill_switch_data_gap"
    NEWS_BLACKOUT = "news_blackout_period"
    ZERO_STOP = "stop_distance_is_zero"


@dataclass
class PositionSize:
    lots: float
    risk_amount: float
    stop_distance: float
    rejection: RiskRejection = RiskRejection.NONE


@dataclass
class RiskState:
    """Mutable state tracked by the risk manager across the session."""
    equity: float
    peak_equity: float
    daily_pnl: float = 0.0
    current_date: date = None
    open_positions: int = 0
    consecutive_losses: int = 0
    last_data_timestamp: datetime = None
    kill_switch_active: bool = False
    kill_switch_reason: str = ""
    trade_log: list = field(default_factory=list)


class RiskManager:
    def __init__(self, cfg: dict, initial_equity: float):
        risk_cfg = cfg["risk"]
        self.risk_per_trade = risk_cfg["risk_per_trade_pct"]
        self.max_risk_per_trade = risk_cfg["max_risk_per_trade_pct"]
        self.max_daily_dd = risk_cfg["max_daily_drawdown_pct"]
        self.max_total_dd = risk_cfg["max_total_drawdown_pct"]
        self.max_open_positions = risk_cfg["max_open_positions"]
        self.vol_lookback = risk_cfg["volatility_lookback"]
        self.no_trade_during_news = risk_cfg.get("no_trade_during_news", True)

        kill_cfg = risk_cfg.get("kill_switch", {})
        self.max_spread_multiple = kill_cfg.get("max_spread_multiple", 3.0)
        self.max_consecutive_losses = kill_cfg.get("max_consecutive_losses", 5)
        self.max_data_gap_seconds = kill_cfg.get("max_data_gap_seconds", 300)

        self.lot_size_oz = cfg["costs"]["lot_size_oz"]
        self.spread = cfg["costs"]["spread_points"]
        self.slippage = cfg["costs"]["slippage_points"]

        self.state = RiskState(
            equity=initial_equity,
            peak_equity=initial_equity,
            current_date=date.today(),
        )

    def calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        current_spread: float = None,
        current_time: datetime = None,
        is_news_period: bool = False,
    ) -> PositionSize:
        """
        Calculate position size based on volatility and stop distance.
        Returns lots to trade, or 0 with a rejection reason.
        """
        # Check kill switch first
        rejection = self._check_kill_switch(current_spread, current_time, is_news_period)
        if rejection != RiskRejection.NONE:
            log.warning("trade_rejected", reason=rejection.value)
            return PositionSize(lots=0, risk_amount=0, stop_distance=0, rejection=rejection)

        # Check daily drawdown
        if self._daily_loss_limit_hit():
            log.warning("trade_rejected", reason="daily_loss_limit")
            return PositionSize(
                lots=0, risk_amount=0, stop_distance=0,
                rejection=RiskRejection.DAILY_LOSS_LIMIT,
            )

        # Check total drawdown
        if self._total_drawdown_limit_hit():
            log.warning("trade_rejected", reason="total_drawdown_limit")
            return PositionSize(
                lots=0, risk_amount=0, stop_distance=0,
                rejection=RiskRejection.TOTAL_DRAWDOWN_LIMIT,
            )

        # Check max open positions
        if self.state.open_positions >= self.max_open_positions:
            return PositionSize(
                lots=0, risk_amount=0, stop_distance=0,
                rejection=RiskRejection.MAX_POSITIONS,
            )

        # Calculate stop distance in price terms
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return PositionSize(
                lots=0, risk_amount=0, stop_distance=0,
                rejection=RiskRejection.ZERO_STOP,
            )

        # Include transaction costs in effective stop
        effective_stop = stop_distance + self.spread + self.slippage

        # Risk amount = equity * risk percentage
        risk_amount = self.state.equity * self.risk_per_trade

        # Cap at max risk
        max_risk = self.state.equity * self.max_risk_per_trade
        risk_amount = min(risk_amount, max_risk)

        # Position size: risk_amount / (stop_distance_per_oz)
        # lots = risk_amount / (effective_stop * lot_size_oz)
        lots = risk_amount / (effective_stop * self.lot_size_oz)

        # Round down to avoid over-risking
        lots = max(0.0, round(lots, 2))

        # Minimum lot size check
        if lots < 0.01:
            lots = 0.0

        log.info(
            "position_sized",
            lots=lots,
            risk_amount=round(risk_amount, 2),
            stop_distance=round(stop_distance, 2),
            effective_stop=round(effective_stop, 2),
        )
        return PositionSize(
            lots=lots,
            risk_amount=round(risk_amount, 2),
            stop_distance=round(stop_distance, 2),
        )

    def record_trade_result(self, pnl: float) -> None:
        """Update risk state after a trade closes."""
        self.state.equity += pnl
        self.state.daily_pnl += pnl
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
        self.state.open_positions = max(0, self.state.open_positions - 1)

        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        self.state.trade_log.append({
            "pnl": pnl,
            "equity_after": self.state.equity,
            "timestamp": datetime.now().isoformat(),
        })

        log.info(
            "trade_closed",
            pnl=round(pnl, 2),
            equity=round(self.state.equity, 2),
            consecutive_losses=self.state.consecutive_losses,
        )

    def on_new_bar(self, bar_time: datetime = None) -> None:
        """Called each bar — resets daily PnL if the date changed and tracks data timestamp."""
        if bar_time is not None:
            self.state.last_data_timestamp = bar_time
        today = (bar_time.date() if bar_time else date.today())
        if self.state.current_date != today:
            log.info(
                "daily_reset",
                prev_daily_pnl=round(self.state.daily_pnl, 2),
                equity=round(self.state.equity, 2),
            )
            self.state.daily_pnl = 0.0
            self.state.current_date = today

    def on_position_opened(self) -> None:
        self.state.open_positions += 1

    def current_drawdown_pct(self) -> float:
        if self.state.peak_equity <= 0:
            return 0.0
        return (self.state.peak_equity - self.state.equity) / self.state.peak_equity

    def _daily_loss_limit_hit(self) -> bool:
        if self.state.equity <= 0:
            return True
        return abs(self.state.daily_pnl) / self.state.equity >= self.max_daily_dd and self.state.daily_pnl < 0

    def _total_drawdown_limit_hit(self) -> bool:
        return self.current_drawdown_pct() >= self.max_total_dd

    def _check_kill_switch(
        self,
        current_spread: float = None,
        current_time: datetime = None,
        is_news_period: bool = False,
    ) -> RiskRejection:
        """Check all kill switch conditions."""
        if self.state.kill_switch_active:
            return RiskRejection.KILL_SWITCH_SPREAD  # already killed

        # Spread check
        if current_spread is not None:
            if current_spread > self.spread * self.max_spread_multiple:
                self.state.kill_switch_active = True
                self.state.kill_switch_reason = "abnormal_spread"
                return RiskRejection.KILL_SWITCH_SPREAD

        # Consecutive losses
        if self.state.consecutive_losses >= self.max_consecutive_losses:
            return RiskRejection.KILL_SWITCH_CONSECUTIVE_LOSSES

        # Data gap
        if current_time and self.state.last_data_timestamp:
            gap = (current_time - self.state.last_data_timestamp).total_seconds()
            if gap > self.max_data_gap_seconds:
                return RiskRejection.KILL_SWITCH_DATA_GAP

        if current_time:
            self.state.last_data_timestamp = current_time

        # News blackout
        if is_news_period and self.no_trade_during_news:
            return RiskRejection.NEWS_BLACKOUT

        return RiskRejection.NONE

    def reset_kill_switch(self) -> None:
        """Manual reset — requires explicit action."""
        log.warning("kill_switch_reset", previous_reason=self.state.kill_switch_reason)
        self.state.kill_switch_active = False
        self.state.kill_switch_reason = ""

    def get_state_summary(self) -> dict:
        return {
            "equity": round(self.state.equity, 2),
            "peak_equity": round(self.state.peak_equity, 2),
            "drawdown_pct": round(self.current_drawdown_pct() * 100, 2),
            "daily_pnl": round(self.state.daily_pnl, 2),
            "open_positions": self.state.open_positions,
            "consecutive_losses": self.state.consecutive_losses,
            "kill_switch": self.state.kill_switch_active,
            "total_trades": len(self.state.trade_log),
        }
