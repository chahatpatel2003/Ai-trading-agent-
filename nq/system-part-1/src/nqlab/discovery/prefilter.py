"""Diagnostics-first prefilter (Phase 9).

Runs ONE cheap ungated backtest per candidate and short-circuits — before the full
gauntlet (Monte-Carlo, cost stress, walk-forward) — the structural failures the
asymmetric-setup search should never spend validation budget on:

* too few trades              -> NEEDS_MORE_DATA
* no directional edge (gross<=0 before costs) -> REJECT
* MAE >= MFE (not asymmetric) -> REJECT

A short-circuited candidate still gets a stored verdict + diagnostics + the cheap
in/out-sample split (Monte-Carlo / cost-stress fields are left empty because they
were intentionally not run). Candidates that pass return ``None`` and flow into the
full :func:`nqlab.validation.gauntlet.run_gauntlet`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import polars as pl

from nqlab.backtest.engine import run_backtest
from nqlab.config.models import AppConfig, TopstepRules
from nqlab.domain.strategy.base import BaseStrategy
from nqlab.research.report import compute_metrics
from nqlab.validation.diagnostics import diag_summary
from nqlab.validation.gauntlet import ValidationResult
from nqlab.validation.montecarlo import MonteCarloResult
from nqlab.validation.periods import (
    month_breakdown,
    net_of,
    summarize,
    trades_between,
    year_breakdown,
)
from nqlab.validation.splits import embargoed_split

Prefilter = Callable[..., ValidationResult | None]


def diagnostics_prefilter(*, min_trades: int = 30, asymmetry_min: float = 1.2) -> Prefilter:
    """Build a prefilter callable for ``run_discovery``.

    ``asymmetry_min`` is the minimum ``avg_mfe / avg_mae`` ratio to survive (the
    realized trade distribution must be meaningfully asymmetric).
    """

    def _prefilter(
        strategy: BaseStrategy,
        *,
        config: AppConfig,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
        rules: TopstepRules,
        frame: pl.DataFrame,
    ) -> ValidationResult | None:
        signal_frame = strategy.signals(frame)
        bt_frame = (
            signal_frame.filter(pl.col("is_rth")) if config.backtest.rth_only else signal_frame
        )
        base = run_backtest(
            bt_frame,
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy.params,
            config=config,
            rules=rules,
            entry_mode="next_open",
            gate_entries=False,
            trailing=strategy.trailing,
            exec_config=strategy.exec_config,
        )
        trades = base.trades
        diag = diag_summary(trades)
        n = len(trades)

        verdict: str | None = None
        reason = ""
        if n < min_trades:
            verdict = "NEEDS_MORE_DATA"
            reason = f"diagnostics-first: only {n} trades (need >= {min_trades})"
        elif not diag.get("direction_edge_before_costs"):
            verdict = "REJECT"
            reason = "diagnostics-first: no directional edge before costs (gross <= 0)"
        else:
            mae = float(diag.get("avg_mae") or 0.0)  # type: ignore[arg-type]
            mfe = float(diag.get("avg_mfe") or 0.0)  # type: ignore[arg-type]
            if mfe <= asymmetry_min * mae:
                verdict = "REJECT"
                reason = (
                    f"diagnostics-first: MAE>=MFE (avg_mfe {mfe:.1f} <= "
                    f"{asymmetry_min:.2f}*avg_mae {mae:.1f}); not asymmetric"
                )

        if verdict is None:
            return None  # passes the prefilter -> full gauntlet

        v = config.validation
        split = embargoed_split(
            v.train_start, v.train_end, v.test_start, v.test_end, v.embargo_days
        )
        full = compute_metrics(base)
        result = ValidationResult(
            strategy=strategy.params.name,
            version=strategy.params.version,
            symbol=symbol,
            timeframe=timeframe,
            account=rules.name,
            full=full,
            in_sample=summarize(trades_between(trades, split.train_start, split.train_end)),
            out_sample=summarize(trades_between(trades, split.test_start, split.test_end)),
            yearly={y: summarize(ts) for y, ts in year_breakdown(trades).items()},
            monthly={m: summarize(ts) for m, ts in month_breakdown(trades).items()},
            walk_forward=[],
            walk_forward_positive_frac=0.0,
            cost_stress=[],
            monte_carlo=MonteCarloResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            topstep=base.topstep,
            unrealistic_fills=False,
            base_net_realistic=net_of(full),
            base_net_optimistic=0.0,
            diagnostics=diag,
        )
        result.verdict = verdict
        result.reasons = [reason]
        return result

    return _prefilter
