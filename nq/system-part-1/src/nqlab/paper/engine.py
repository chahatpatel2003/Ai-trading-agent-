"""Paper/shadow execution engine (Phase 15).

Runs the hash-verified frozen strategy over a feature frame using the SAME locked-buffer
Topstep backtest semantics (entries, stops, targets, EOD-flatten, conservative
slippage/commission). It places NO live orders — it calls the simulator and records a
forward ledger. A real funded account is modelled (``gate_entries=True``): a blow-up or
daily-loss halt stops new entries.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from nqlab.backtest.engine import run_backtest
from nqlab.config.models import AppConfig, TopstepRules
from nqlab.domain.calendar.sessions import parse_hhmm
from nqlab.paper.frozen import PaperConfig
from nqlab.paper.ledger import trade_to_row


def _setup_reason(side: int) -> str:
    if side == 1:
        return "4h-bull bias + bullish break-of-structure with displacement -> long continuation"
    return "4h-bear bias + bearish break-of-structure with displacement -> short continuation"


def run_paper(
    config: AppConfig,
    paper_cfg: PaperConfig,
    *,
    feature_frame: pl.DataFrame,
    rules: TopstepRules,
    mode: str,
    slippage_mult: float = 1.0,
    commission_mult: float = 1.0,
) -> tuple[list[dict[str, Any]], Any]:
    """Run the frozen strategy on ``feature_frame`` and return (ledger_rows, topstep_result).

    Refuses to run if the frozen config was modified. No order is ever sent.
    """
    paper_cfg.verify()  # hard stop if the frozen config was changed
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    strat = paper_cfg.to_strategy(rth_start_min)
    sig = strat.signals(feature_frame)
    bt_frame = sig.filter(pl.col("is_rth")) if config.backtest.rth_only else sig

    bt = run_backtest(
        bt_frame,
        symbol="NQ.c",
        timeframe=paper_cfg.timeframe,
        strategy=strat.params,
        config=config,
        rules=rules,
        entry_mode="next_open",
        gate_entries=True,
        trailing=strat.trailing,
        exec_config=strat.exec_config,
        slippage_mult=slippage_mult,
        commission_mult=commission_mult,
    )
    ts = bt.topstep
    blow_day = ts.blow_trading_day if ts is not None else None
    violations = list(ts.violations) if ts is not None else []

    rows: list[dict[str, Any]] = []
    equity = rules.account_size
    for trade in bt.trades:
        equity += trade.net_pnl
        blown_now = blow_day is not None and trade.trading_day >= blow_day
        rows.append(
            trade_to_row(
                trade,
                strategy_id=paper_cfg.strategy_id,
                frozen_hash=paper_cfg.frozen_hash,
                mode=mode,
                setup_reason=_setup_reason(trade.side),
                account_equity=equity,
                account_blown=blown_now,
                rule_violations=[
                    v for v in violations if v.endswith(trade.trading_day.isoformat())
                ],
                followed_frozen_rules=True,  # the engine executes the frozen strategy verbatim
            )
        )
    return rows, ts
