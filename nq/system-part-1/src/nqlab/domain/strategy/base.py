"""Strategy interface.

A strategy receives a frame of bars + PIT features and emits, per bar, a discrete
``signal`` (1 long / -1 short / 0 none) plus a ``stop_pts`` and ``target_pts``
(distance in price points). It reads only columns at the current row — never
future rows — so it cannot leak. Parameters are versioned and hashable.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl

SIGNAL_COLUMNS: tuple[str, ...] = ("signal", "stop_pts", "target_pts")

ParamValue = float | int | str | bool

# Data-availability contract.
TRADES_ONLY = "trades_only"
MBP1_REQUIRED = "mbp1_required"
MBP1_OPTIONAL = "mbp1_optional"


@dataclass(frozen=True)
class TrailingConfig:
    """Optional trailing/breakeven stop management (applied by the backtester)."""

    mode: str = "none"  # none | fixed_ticks | atr
    trail_ticks: float = 0.0
    trail_atr: float = 0.0
    breakeven_r: float | None = None  # move stop to entry after this R of favorable move


@dataclass(frozen=True)
class ExecConfig:
    """Per-strategy execution overrides for survival optimization (Phase 7/8)."""

    risk_per_trade_usd: float | None = None  # None -> backtest-config default
    min_hold_bars: int = 0  # suppress TARGET exits before this many bars (let winners run)
    time_stop_bars: int | None = None  # None -> backtest-config default
    sizing_mode: str = "adaptive"  # adaptive|nq_only|mnq_only|max_1_nq|mnq_scaling (Phase 8)
    max_daily_losses: int | None = None  # stop NEW entries after N losing trades in a day
    max_daily_trades: int | None = None  # stop NEW entries after N trades taken in a day (P18)
    daily_profit_target_usd: float | None = None  # stop NEW entries after +$X realized today (P18)


@dataclass(frozen=True)
class StrategyParams:
    """Immutable, hashable strategy identity (name + version + params)."""

    name: str
    version: str
    params: tuple[tuple[str, ParamValue], ...] = ()

    def as_dict(self) -> dict[str, ParamValue]:
        return dict(self.params)

    def fingerprint(self) -> str:
        blob = json.dumps(
            {"name": self.name, "version": self.version, "params": self.as_dict()},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def attach_signals(
    frame: pl.DataFrame,
    *,
    long: pl.Expr,
    short: pl.Expr,
    stop_pts: pl.Expr,
    target_pts: pl.Expr,
) -> pl.DataFrame:
    """Attach the three signal columns derived from boolean long/short conditions.

    Long takes precedence over short on the (degenerate) bar where both fire.
    """
    signal = (
        pl.when(long.fill_null(value=False))
        .then(1)
        .when(short.fill_null(value=False))
        .then(-1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("signal")
    )
    return frame.with_columns(
        signal,
        stop_pts.cast(pl.Float64).alias("stop_pts"),
        target_pts.cast(pl.Float64).alias("target_pts"),
    )


@runtime_checkable
class Strategy(Protocol):
    params: StrategyParams

    def signals(self, frame: pl.DataFrame) -> pl.DataFrame: ...


class BaseStrategy(ABC):
    """Convenience base: subclasses provide long/short conditions and stop/target."""

    data_requirement: str = TRADES_ONLY
    required_columns: tuple[str, ...] = ()

    def __init__(
        self,
        params: StrategyParams,
        trailing: TrailingConfig | None = None,
        exec_config: ExecConfig | None = None,
    ) -> None:
        self.params = params
        self.trailing = trailing
        self.exec_config = exec_config

    def _missing_required(self, frame: pl.DataFrame) -> bool:
        return any(col not in frame.columns for col in self.required_columns)

    @abstractmethod
    def _long(self) -> pl.Expr: ...

    @abstractmethod
    def _short(self) -> pl.Expr: ...

    @abstractmethod
    def _stop_pts(self) -> pl.Expr: ...

    @abstractmethod
    def _target_pts(self) -> pl.Expr: ...

    def signals(self, frame: pl.DataFrame) -> pl.DataFrame:
        return attach_signals(
            frame,
            long=self._long(),
            short=self._short(),
            stop_pts=self._stop_pts(),
            target_pts=self._target_pts(),
        )
