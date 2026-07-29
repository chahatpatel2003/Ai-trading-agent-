"""Backtest result schema: trades, equity curve, and the Topstep outcome."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import polars as pl

from nqlab.domain.risk.topstep import TopstepResult
from nqlab.domain.strategy.base import StrategyParams


@dataclass(frozen=True)
class Trade:
    entry_ts: datetime
    exit_ts: datetime
    trading_day: date
    side: int  # 1 long, -1 short
    instrument: str
    contracts: int
    entry_px: float
    exit_px: float
    stop_px: float
    target_px: float
    gross_pnl: float
    fees: float
    net_pnl: float
    exit_reason: str
    bars_held: int
    entry_minute: int = 0  # ET minute-of-day at entry (for diagnostics)
    mae: float = 0.0  # max adverse excursion, points
    mfe: float = 0.0  # max favorable excursion, points

    def as_dict(self) -> dict[str, object]:
        return {
            "entry_ts": self.entry_ts,
            "exit_ts": self.exit_ts,
            "trading_day": self.trading_day,
            "side": self.side,
            "instrument": self.instrument,
            "contracts": self.contracts,
            "entry_px": self.entry_px,
            "exit_px": self.exit_px,
            "stop_px": self.stop_px,
            "target_px": self.target_px,
            "gross_pnl": self.gross_pnl,
            "fees": self.fees,
            "net_pnl": self.net_pnl,
            "exit_reason": self.exit_reason,
            "bars_held": self.bars_held,
            "entry_minute": self.entry_minute,
            "mae": self.mae,
            "mfe": self.mfe,
        }


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    strategy: StrategyParams
    starting_balance: float
    final_equity: float
    trades: list[Trade]
    equity_curve: pl.DataFrame  # columns: ts, trading_day, equity
    topstep: TopstepResult | None = None
    meta: dict[str, object] = field(default_factory=dict)

    def trades_frame(self) -> pl.DataFrame:
        if not self.trades:
            return pl.DataFrame()
        return pl.DataFrame([t.as_dict() for t in self.trades])
