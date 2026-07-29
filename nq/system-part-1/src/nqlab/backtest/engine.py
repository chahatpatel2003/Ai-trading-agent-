"""Event-driven (bar-resolution) backtester.

Realism (Phase 2):
- ``next_open`` entry mode (default): a signal at bar *t* is executed at bar
  *t+1*'s open with adverse slippage — no decision uses a price it could not have
  acted on. ``current_close`` (debug only) fills at the signal bar's close.
- Stops/targets evaluated intrabar against high/low (stop assumed first if both
  touch); forced exits (stop, EOD-flat, time-stop, halt, blow-up) take adverse
  slippage; target fills are exact (resting limit).
- Optional intrabar trailing-drawdown: each bar also marks the position's adverse
  extreme (low for longs, high for shorts) so a within-bar breach is caught.
- ``slippage_mult`` / ``commission_mult`` drive cost/slippage stress tests.

No overnight holds: positions are force-flat at the configured cutoff.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import cast

import polars as pl

from nqlab.backtest.result import BacktestResult, Trade
from nqlab.config.models import AppConfig, TopstepRules
from nqlab.domain.calendar.sessions import parse_hhmm
from nqlab.domain.risk.topstep import TopstepAccount
from nqlab.domain.sizing import SizingMode, size_position
from nqlab.domain.strategy.base import ExecConfig, StrategyParams, TrailingConfig

_FORCED_REASONS = frozenset({"STOP", "EOD_FLAT", "TIME_STOP", "HALT", "BLOWN", "END"})
# A single trade may risk at most this fraction of the account's trailing max loss; the
# sizing-repair modes skip a trade only when even one MNQ contract would breach it.
_MAX_TRADE_LOSS_FRAC = 0.25
_COLUMNS = [
    "ts",
    "minute_of_day",
    "trading_day",
    "is_rth",
    "open",
    "high",
    "low",
    "close",
    "signal",
    "stop_pts",
    "target_pts",
    "atr",
]


@dataclass
class _Position:
    side: int
    contracts: int
    instrument: str
    point_value: float
    commission_per_side: float
    entry_px: float
    stop_px: float
    target_px: float
    entry_ts: datetime
    entry_day: date
    entry_fees: float
    initial_risk: float
    best_px: float
    entry_minute: int = 0
    mae: float = 0.0
    mfe: float = 0.0
    bars_held: int = 0
    be_moved: bool = False


def _is_valid(value: float | None) -> bool:
    return value is not None and not math.isnan(value) and value > 0


def run_backtest(
    frame: pl.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    strategy: StrategyParams,
    config: AppConfig,
    rules: TopstepRules,
    primary_cutoff: bool = True,
    entry_mode: str | None = None,
    slippage_mult: float = 1.0,
    commission_mult: float = 1.0,
    gate_entries: bool = True,
    trailing: TrailingConfig | None = None,
    exec_config: ExecConfig | None = None,
) -> BacktestResult:
    """Run a signal frame (one symbol, sorted by ts) through the simulator.

    ``gate_entries`` (default True) models a *real* account: a blow-up or daily-loss
    halt stops new entries (and flattens). Validation sets it False to measure the
    strategy's full trade distribution while the account still *records* breaches.
    """
    bt = config.backtest
    mode = entry_mode or bt.entry_mode
    trail = trailing or TrailingConfig()
    ex = exec_config or ExecConfig()
    nq = config.instruments.specs["NQ"]
    mnq = config.instruments.specs["MNQ"]
    tick = nq.tick_size
    slippage = bt.slippage_ticks * tick * slippage_mult
    risk_usd = ex.risk_per_trade_usd if ex.risk_per_trade_usd is not None else bt.risk_per_trade_usd
    time_stop_bars = ex.time_stop_bars if ex.time_stop_bars is not None else bt.time_stop_bars
    min_hold_bars = ex.min_hold_bars
    sizing_mode = ex.sizing_mode
    max_daily_losses = ex.max_daily_losses  # stop NEW entries after N losing trades/day
    max_daily_trades = ex.max_daily_trades  # stop NEW entries after N trades taken/day
    daily_profit_target = ex.daily_profit_target_usd  # stop NEW entries after +$X realized/day
    max_contracts = rules.max_contracts

    force_flat_min = parse_hhmm(bt.force_flat_et)
    hard_cutoff_min = parse_hhmm(bt.no_new_trade_hard_et)
    capture_cutoff_min = (
        parse_hhmm(bt.no_new_trade_primary_et) if primary_cutoff else hard_cutoff_min
    )

    account = TopstepAccount(rules)
    realized = 0.0
    position: _Position | None = None
    pending: tuple[int, float, float] | None = None  # (side, stop_pts, target_pts)
    trades: list[Trade] = []
    eq_ts: list[datetime] = []
    eq_day: list[date] = []
    eq_val: list[float] = []
    losses_today = 0  # losing-trade count for the current day (for max_daily_losses)
    trades_today = 0  # trades taken today (for max_daily_trades)
    pnl_today = 0.0  # realized P&L today (for daily_profit_target)
    cur_day: date | None = None

    def can_capture(minute: int, is_rth: bool) -> bool:
        return (not bt.rth_only or is_rth) and minute < min(capture_cutoff_min, hard_cutoff_min)

    def can_execute(minute: int, is_rth: bool) -> bool:
        return (not bt.rth_only or is_rth) and minute < hard_cutoff_min

    def open_position(
        fill: float, side: int, sp: float, tp: float, ts: datetime, day: date, minute: int
    ) -> None:
        nonlocal position, trades_today
        decision = size_position(
            stop_points=sp,
            risk_usd=risk_usd,
            nq=nq,
            mnq=mnq,
            max_contracts=max_contracts,
            mode=cast("SizingMode", sizing_mode),
            max_loss_usd=rules.trailing_max_loss * _MAX_TRADE_LOSS_FRAC,
        )
        if decision.skipped:
            return
        pv = nq.point_value if decision.instrument == "NQ" else mnq.point_value
        comm = (nq if decision.instrument == "NQ" else mnq).commission_per_side * commission_mult
        entry_fill = fill + side * slippage
        position = _Position(
            side=side,
            contracts=decision.contracts,
            instrument=decision.instrument,
            point_value=pv,
            commission_per_side=comm,
            entry_px=entry_fill,
            stop_px=entry_fill - side * sp,
            target_px=entry_fill + side * tp,
            entry_ts=ts,
            entry_day=day,
            entry_fees=comm * decision.contracts,
            initial_risk=sp,
            best_px=entry_fill,
            entry_minute=minute,
        )
        trades_today += 1

    def close_position(pos: _Position, ts: datetime, day: date, fill: float, reason: str) -> None:
        nonlocal realized, position, losses_today, pnl_today
        if reason in _FORCED_REASONS:
            fill = fill - pos.side * slippage  # adverse market/stop exit
        gross = pos.side * (fill - pos.entry_px) * pos.point_value * pos.contracts
        fees = pos.entry_fees + pos.commission_per_side * pos.contracts
        net = gross - fees
        realized += net
        pnl_today += net
        if net < 0:
            losses_today += 1
        trades.append(
            Trade(
                entry_ts=pos.entry_ts,
                exit_ts=ts,
                trading_day=pos.entry_day,
                side=pos.side,
                instrument=pos.instrument,
                contracts=pos.contracts,
                entry_px=pos.entry_px,
                exit_px=fill,
                stop_px=pos.stop_px,
                target_px=pos.target_px,
                gross_pnl=gross,
                fees=fees,
                net_pnl=net,
                exit_reason=reason,
                bars_held=pos.bars_held,
                entry_minute=pos.entry_minute,
                mae=pos.mae,
                mfe=pos.mfe,
            )
        )
        position = None

    def unrealized(price: float) -> float:
        if position is None:
            return 0.0
        delta = price - position.entry_px
        return position.side * delta * position.point_value * position.contracts

    for row in frame.select(_COLUMNS).iter_rows(named=True):
        ts, day, minute = row["ts"], row["trading_day"], row["minute_of_day"]
        open_, high, low, close = row["open"], row["high"], row["low"], row["close"]

        if day != cur_day:  # reset the daily guards at each new trading day
            cur_day = day
            losses_today = 0
            trades_today = 0
            pnl_today = 0.0
        daily_halted = (
            (max_daily_losses is not None and losses_today >= max_daily_losses)
            or (max_daily_trades is not None and trades_today >= max_daily_trades)
            or (daily_profit_target is not None and pnl_today >= daily_profit_target)
        )
        # Daily controls + the blow/halt gate model a REAL account, so they apply only when
        # gating entries. The ungated gauntlet measures the raw signal edge (full distribution).
        gate_ok = (account.can_open() and not daily_halted) if gate_entries else True

        # 1) Execute a pending (next_open) entry at this bar's open.
        if mode == "next_open" and pending is not None and position is None:
            side, sp, tp = pending
            pending = None
            if gate_ok and can_execute(minute, row["is_rth"]):
                open_position(open_, side, sp, tp, ts, day, minute)

        # 2) Manage exits against this bar (track MAE/MFE first).
        if position is not None:
            pos = position
            if pos.side == 1:
                pos.mfe = max(pos.mfe, high - pos.entry_px)
                pos.mae = max(pos.mae, pos.entry_px - low)
            else:
                pos.mfe = max(pos.mfe, pos.entry_px - low)
                pos.mae = max(pos.mae, high - pos.entry_px)
            if pos.side == 1:
                hit_stop, hit_target = low <= pos.stop_px, high >= pos.target_px
            else:
                hit_stop, hit_target = high >= pos.stop_px, low <= pos.target_px
            if hit_stop:
                close_position(pos, ts, day, pos.stop_px, "STOP")
            elif hit_target and pos.bars_held >= min_hold_bars:
                close_position(pos, ts, day, pos.target_px, "TARGET")
            elif gate_entries and account.blown:
                close_position(pos, ts, day, close, "BLOWN")
            elif gate_entries and account.halted_today:
                close_position(pos, ts, day, close, "HALT")
            elif minute >= force_flat_min:
                close_position(pos, ts, day, close, "EOD_FLAT")
            elif pos.bars_held >= time_stop_bars:
                close_position(pos, ts, day, close, "TIME_STOP")

        # 3) Mark equity (intrabar adverse extreme first, then close).
        if position is not None and bt.intrabar_dd:
            adverse_px = low if position.side == 1 else high
            account.mark(rules.account_size + realized + unrealized(adverse_px), day)
        equity = rules.account_size + realized + unrealized(close)
        account.mark(equity, day)
        eq_ts.append(ts)
        eq_day.append(day)
        eq_val.append(equity)

        # 4) Generate a new entry intent when flat and allowed.
        if position is None and row["signal"] != 0 and gate_ok:
            sp, tp = row["stop_pts"], row["target_pts"]
            if can_capture(minute, row["is_rth"]) and _is_valid(sp) and _is_valid(tp):
                if mode == "current_close":
                    open_position(close, int(row["signal"]), sp, tp, ts, day, minute)
                else:
                    pending = (int(row["signal"]), sp, tp)

        # 5) Age the position and tighten the stop (breakeven / trailing) using
        #    THIS bar's extreme — applied only for the NEXT bar, so no lookahead.
        if position is not None:
            position.bars_held += 1
            pos = position
            pos.best_px = max(pos.best_px, high) if pos.side == 1 else min(pos.best_px, low)
            favorable = pos.side * (pos.best_px - pos.entry_px)
            if (
                trail.breakeven_r is not None
                and not pos.be_moved
                and favorable >= trail.breakeven_r * pos.initial_risk
            ):
                pos.stop_px = pos.entry_px
                pos.be_moved = True
            trail_amount = 0.0
            if trail.mode == "fixed_ticks":
                trail_amount = trail.trail_ticks * tick
            elif trail.mode == "atr":
                atr = row["atr"]
                trail_amount = trail.trail_atr * atr if _is_valid(atr) else 0.0
            if trail_amount > 0:
                candidate = pos.best_px - pos.side * trail_amount
                pos.stop_px = (
                    max(pos.stop_px, candidate) if pos.side == 1 else min(pos.stop_px, candidate)
                )

    if position is not None and eq_ts:
        close_position(position, eq_ts[-1], eq_day[-1], frame["close"][-1], "END")
        eq_val[-1] = rules.account_size + realized
        account.mark(eq_val[-1], eq_day[-1])

    equity_curve = pl.DataFrame(
        {"ts": eq_ts, "trading_day": eq_day, "equity": eq_val}
    ).with_columns((pl.col("equity") - rules.account_size).alias("pnl"))
    return BacktestResult(
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        starting_balance=rules.account_size,
        final_equity=rules.account_size + realized,
        trades=trades,
        equity_curve=equity_curve,
        topstep=account.finalize(),
        meta={
            "entry_mode": mode,
            "slippage_mult": slippage_mult,
            "commission_mult": commission_mult,
            "bars": len(eq_ts),
        },
    )
