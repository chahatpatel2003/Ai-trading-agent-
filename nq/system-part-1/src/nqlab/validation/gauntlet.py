"""The validation gauntlet: run every check and attach an accept/reject verdict."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl

from nqlab.backtest.engine import run_backtest
from nqlab.config.models import AppConfig, TopstepRules
from nqlab.domain.risk.topstep import TopstepResult
from nqlab.domain.strategy.base import BaseStrategy
from nqlab.research.report import compute_metrics
from nqlab.research.runner import prepare_features
from nqlab.validation.diagnostics import diag_summary
from nqlab.validation.montecarlo import MonteCarloResult, monte_carlo
from nqlab.validation.periods import (
    month_breakdown,
    net_of,
    sharpe_skew_kurt,
    summarize,
    trades_between,
    trades_of,
    year_breakdown,
)
from nqlab.validation.splits import embargoed_split, walk_forward_splits
from nqlab.validation.stress import StressResult, run_stress
from nqlab.validation.verdict import decide


@dataclass
class WalkWindow:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    oos_net: float
    oos_trades: int
    positive: bool


@dataclass
class ValidationResult:
    strategy: str
    version: str
    symbol: str
    timeframe: str
    account: str
    full: dict[str, object]
    in_sample: dict[str, object]
    out_sample: dict[str, object]
    yearly: dict[str, dict[str, object]]
    monthly: dict[str, dict[str, object]]
    walk_forward: list[WalkWindow]
    walk_forward_positive_frac: float
    cost_stress: list[StressResult]
    monte_carlo: MonteCarloResult
    topstep: TopstepResult | None
    unrealistic_fills: bool
    base_net_realistic: float
    base_net_optimistic: float
    in_sample_sharpe: float = 0.0
    in_sample_skew: float = 0.0
    in_sample_kurt: float = 3.0
    in_sample_n: int = 0
    diagnostics: dict[str, object] = field(default_factory=dict)
    verdict: str = ""
    reasons: list[str] = field(default_factory=list)


def run_gauntlet(
    config: AppConfig,
    strategy: BaseStrategy,
    *,
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
    rules: TopstepRules,
    feature_frame: pl.DataFrame | None = None,
) -> ValidationResult:
    """Run the full validation gauntlet for one strategy and decide a verdict."""
    frame = (
        feature_frame
        if feature_frame is not None
        else prepare_features(config, symbol=symbol, timeframe=timeframe, start=start, end=end)
    )
    signal_frame = strategy.signals(frame)
    params = strategy.params
    # With no overnight holds, RTH bars carry every entry/exit; dropping the
    # flat overnight bars is a large speedup with no effect on trades/Topstep.
    bt_frame = signal_frame.filter(pl.col("is_rth")) if config.backtest.rth_only else signal_frame

    trailing = strategy.trailing
    exec_config = strategy.exec_config
    # Ungated: measure the full trade distribution (the account still records a
    # blow-up via base.topstep, but does not stop trading).
    base = run_backtest(
        bt_frame, symbol=symbol, timeframe=timeframe, strategy=params, config=config, rules=rules,
        entry_mode="next_open", gate_entries=False, trailing=trailing, exec_config=exec_config,
    )
    optimistic = run_backtest(
        bt_frame, symbol=symbol, timeframe=timeframe, strategy=params, config=config, rules=rules,
        entry_mode="current_close", gate_entries=False, trailing=trailing, exec_config=exec_config,
    )
    trades = base.trades
    full = compute_metrics(base)

    v = config.validation
    split = embargoed_split(v.train_start, v.train_end, v.test_start, v.test_end, v.embargo_days)
    in_trades = trades_between(trades, split.train_start, split.train_end)
    in_sample = summarize(in_trades)
    out_sample = summarize(trades_between(trades, split.test_start, split.test_end))
    is_sharpe, is_skew, is_kurt, is_n = sharpe_skew_kurt([t.net_pnl for t in in_trades])
    yearly = {y: summarize(ts) for y, ts in year_breakdown(trades).items()}
    monthly = {m: summarize(ts) for m, ts in month_breakdown(trades).items()}

    windows: list[WalkWindow] = []
    for sp in walk_forward_splits(
        start, end, train_months=v.walk_train_months, test_months=v.walk_test_months,
        step_months=v.walk_step_months, embargo_days=v.embargo_days,
    ):
        s = summarize(trades_between(trades, sp.test_start, sp.test_end))
        net = net_of(s)
        n = trades_of(s)
        windows.append(
            WalkWindow(
                sp.train_start.isoformat(), sp.train_end.isoformat(),
                sp.test_start.isoformat(), sp.test_end.isoformat(),
                net, n, net > 0 and n > 0,
            )
        )
    evaluated = [w for w in windows if w.oos_trades > 0]
    wf_frac = sum(1 for w in evaluated if w.positive) / len(evaluated) if evaluated else 0.0

    stress = run_stress(
        bt_frame, config=config, rules=rules, strategy=params, symbol=symbol,
        timeframe=timeframe, slippage_mults=v.slippage_stress,
        commission_mults=v.commission_stress, trailing=trailing, exec_config=exec_config,
    )
    mc = monte_carlo(
        [t.net_pnl for t in trades], rules, runs=v.monte_carlo_runs, seed=v.monte_carlo_seed
    )

    base_net = net_of(full)
    opt_net = net_of(summarize(optimistic.trades))

    result = ValidationResult(
        strategy=params.name, version=params.version, symbol=symbol, timeframe=timeframe,
        account=rules.name, full=full, in_sample=in_sample, out_sample=out_sample,
        yearly=yearly, monthly=monthly, walk_forward=windows,
        walk_forward_positive_frac=wf_frac, cost_stress=stress, monte_carlo=mc,
        topstep=base.topstep, unrealistic_fills=base_net <= 0 < opt_net,
        base_net_realistic=base_net, base_net_optimistic=opt_net,
        in_sample_sharpe=is_sharpe, in_sample_skew=is_skew, in_sample_kurt=is_kurt,
        in_sample_n=is_n, diagnostics=diag_summary(trades),
    )
    result.verdict, result.reasons = decide(result, v)
    return result
