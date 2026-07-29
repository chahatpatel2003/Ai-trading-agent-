"""Run orchestration: prepare features, run strategies, rank results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import cast

import polars as pl

from nqlab.backtest.engine import run_backtest
from nqlab.backtest.result import BacktestResult
from nqlab.config.models import AppConfig, TopstepRules
from nqlab.data.bars_build import read_bars
from nqlab.domain.calendar.sessions import parse_hhmm
from nqlab.domain.strategy.base import BaseStrategy
from nqlab.features.build import FeatureParams, build_features
from nqlab.research.baselines import BASELINES
from nqlab.research.report import compute_metrics


@dataclass
class StrategyRun:
    name: str
    result: BacktestResult
    metrics: dict[str, object]


def prepare_features(
    config: AppConfig,
    *,
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
    params: FeatureParams | None = None,
) -> pl.DataFrame:
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    bars = read_bars(config, symbol=symbol, timeframe=timeframe, start=start, end=end)
    return build_features(bars, rth_start_min=rth_start_min, params=params)


def prepare_mbp1_features(
    config: AppConfig, *, symbol: str, timeframe: str, start: date, end: date
) -> pl.DataFrame:
    """Trades features LEFT JOIN MBP1 features on ``ts``.

    Where MBP1 is absent (missing periods), the MBP1 columns are null — never
    forward-filled — so MBP1-required strategies skip those bars explicitly.
    """
    from nqlab.features.mbp1 import MBP1_FEATURE_NAMES
    from nqlab.features.mbp1_full import read_mbp1_features

    base = prepare_features(config, symbol=symbol, timeframe=timeframe, start=start, end=end)
    try:
        mbp1 = read_mbp1_features(config, symbol=symbol, timeframe=timeframe, start=start, end=end)
    except FileNotFoundError:
        return base
    extra = ("cvd", "liq_spread_ratio", "liquidity_regime")
    cols = ["ts", *[c for c in (*MBP1_FEATURE_NAMES, *extra) if c in mbp1.columns]]
    return base.join(mbp1.select(cols), on="ts", how="left")


def prepare_setup_features(
    config: AppConfig, *, symbol: str, timeframe: str, start: date, end: date
) -> pl.DataFrame:
    """Trades + MBP1 + context features — the frame the setup families consume."""
    from nqlab.features.context import build_context_features

    frame = prepare_mbp1_features(config, symbol=symbol, timeframe=timeframe, start=start, end=end)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    return build_context_features(frame, rth_start_min=rth_start_min)


def prepare_htf_features(
    config: AppConfig, *, symbol: str, timeframe: str, start: date, end: date
) -> pl.DataFrame:
    """Setup frame + Phase-9 higher-timeframe context (prior-period levels, overnight
    range, day-type, trend bias, liquidity distance). Trades-derivable + PIT-safe."""
    from nqlab.features.htf_context import build_htf_context_features

    frame = prepare_setup_features(config, symbol=symbol, timeframe=timeframe, start=start, end=end)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    return build_htf_context_features(frame, rth_start_min=rth_start_min, timeframe=timeframe)


def prepare_structure_features(
    config: AppConfig, *, symbol: str, timeframe: str, start: date, end: date
) -> pl.DataFrame:
    """HTF frame + Phase-13 market-structure / liquidity / displacement / FVG layer
    (swings, BOS/CHoCH/MSS, sweeps, displacement, fair value gaps). PIT-safe."""
    from nqlab.features.structure import build_structure_features

    frame = prepare_htf_features(config, symbol=symbol, timeframe=timeframe, start=start, end=end)
    return build_structure_features(frame)


def run_strategy(
    config: AppConfig,
    strategy: BaseStrategy,
    *,
    symbol: str,
    timeframe: str,
    rules: TopstepRules,
    feature_frame: pl.DataFrame,
    primary_cutoff: bool = True,
) -> BacktestResult:
    signal_frame = strategy.signals(feature_frame)
    return run_backtest(
        signal_frame,
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy.params,
        config=config,
        rules=rules,
        primary_cutoff=primary_cutoff,
    )


def _score(metrics: dict[str, object]) -> tuple[int, int, float, float]:
    """Survival-first ranking key (higher is better)."""
    passed = 1 if metrics.get("passed") else 0
    not_blown = 0 if metrics.get("blown") else 1
    net = cast(float, metrics.get("net_pnl") or 0.0)
    mdd = cast(float, metrics.get("max_drawdown") or 0.0)
    mar = net / mdd if mdd > 0 else net
    return (passed, not_blown, mar, -mdd)


def rank_runs(runs: list[StrategyRun]) -> list[StrategyRun]:
    return sorted(runs, key=lambda r: _score(r.metrics), reverse=True)


def run_baselines(
    config: AppConfig,
    *,
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
    rules: TopstepRules,
    feature_params: FeatureParams | None = None,
    primary_cutoff: bool = True,
) -> list[StrategyRun]:
    """Run all baseline strategies on one symbol/timeframe and rank them."""
    frame = prepare_features(
        config, symbol=symbol, timeframe=timeframe, start=start, end=end, params=feature_params
    )
    runs: list[StrategyRun] = []
    for name, factory in BASELINES.items():
        result = run_strategy(
            config,
            factory(),
            symbol=symbol,
            timeframe=timeframe,
            rules=rules,
            feature_frame=frame,
            primary_cutoff=primary_cutoff,
        )
        runs.append(StrategyRun(name, result, compute_metrics(result)))
    return rank_runs(runs)
