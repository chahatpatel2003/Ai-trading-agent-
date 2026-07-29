"""Cost / slippage stress: re-run the backtest with scaled frictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import polars as pl

from nqlab.backtest.engine import run_backtest
from nqlab.config.models import AppConfig, TopstepRules
from nqlab.domain.strategy.base import ExecConfig, StrategyParams, TrailingConfig
from nqlab.validation.periods import net_of, summarize, trades_of


@dataclass
class StressResult:
    label: str
    slippage_mult: float
    commission_mult: float
    net_pnl: float
    trades: int
    profit_factor: float | None
    max_drawdown: float
    blown: bool


def _one(
    signal_frame: pl.DataFrame,
    *,
    config: AppConfig,
    rules: TopstepRules,
    strategy: StrategyParams,
    symbol: str,
    timeframe: str,
    slip: float,
    comm: float,
    trailing: TrailingConfig | None,
    exec_config: ExecConfig | None,
) -> StressResult:
    result = run_backtest(
        signal_frame, symbol=symbol, timeframe=timeframe, strategy=strategy,
        config=config, rules=rules, slippage_mult=slip, commission_mult=comm,
        gate_entries=False, trailing=trailing, exec_config=exec_config,
    )
    metrics = summarize(result.trades)
    return StressResult(
        label=f"slip{slip:g}x_comm{comm:g}x",
        slippage_mult=slip,
        commission_mult=comm,
        net_pnl=net_of(metrics),
        trades=trades_of(metrics),
        profit_factor=cast("float | None", metrics["profit_factor"]),
        max_drawdown=cast(float, metrics["max_drawdown"]),
        blown=bool(result.topstep.blown) if result.topstep else False,
    )


def run_stress(
    signal_frame: pl.DataFrame,
    *,
    config: AppConfig,
    rules: TopstepRules,
    strategy: StrategyParams,
    symbol: str,
    timeframe: str,
    slippage_mults: tuple[float, ...],
    commission_mults: tuple[float, ...],
    trailing: TrailingConfig | None = None,
    exec_config: ExecConfig | None = None,
) -> list[StressResult]:
    """Stress along slippage (at base commission) then commission (at base slippage)."""
    results: list[StressResult] = []
    base_comm = commission_mults[0] if commission_mults else 1.0
    base_slip = slippage_mults[0] if slippage_mults else 1.0
    for slip in slippage_mults:
        results.append(
            _one(signal_frame, config=config, rules=rules, strategy=strategy, symbol=symbol,
                 timeframe=timeframe, slip=slip, comm=base_comm, trailing=trailing,
                 exec_config=exec_config)
        )
    for comm in commission_mults[1:]:
        results.append(
            _one(signal_frame, config=config, rules=rules, strategy=strategy, symbol=symbol,
                 timeframe=timeframe, slip=base_slip, comm=comm, trailing=trailing,
                 exec_config=exec_config)
        )
    return results
