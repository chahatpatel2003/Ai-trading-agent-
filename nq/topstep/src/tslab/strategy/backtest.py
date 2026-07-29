"""Deterministic bar-loop backtester for TradeRule candidates.

Execution model (conservative, mirrors the label layer's conventions):
  - decision at info_time(T) from feature row T; entry at open[T+1], charged
    as a market order (fees + half-spread + slippage from CostModel)
  - stop exits fill at the stop price as a stop order (worst slippage tier);
    target exits fill at the target as a limit order (fees + half-spread —
    conservative for a passive fill); both-in-one-bar resolves STOP-FIRST
  - time stop and forced flatten exit at bar close as market orders
  - a position can NEVER survive past the session's deadline bar (last bar
    closing strictly before the early-close-aware flatten time), and no
    entry is possible at or after it: rule violations are structurally
    impossible, not merely penalized
  - per-day guards: max_trades_per_day, daily_loss_usd, daily_profit_lock_usd
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import sessions as S
from ..config import Config
from ..sim.costs import CostModel
from .rules import TradeRule


@dataclass
class Trade:
    session: date
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    qty: int
    entry_px: float
    exit_px: float
    exit_reason: str            # target | stop | time | flatten | day_guard
    gross_usd: float
    costs_usd: float
    net_usd: float
    mae_usd: float              # worst open-trade excursion (price only)
    risk_usd: float             # $ distance to stop at entry


@dataclass
class BacktestResult:
    rule_name: str
    trades: List[Trade] = field(default_factory=list)
    violations: int = 0         # must always be 0 by construction

    def by_session(self) -> Dict[date, List[Trade]]:
        out: Dict[date, List[Trade]] = {}
        for t in self.trades:
            out.setdefault(t.session, []).append(t)
        return out

    def daily_net(self) -> pd.Series:
        d: Dict[date, float] = {}
        for t in self.trades:
            d[t.session] = d.get(t.session, 0.0) + t.net_usd
        return pd.Series(d, dtype=float).sort_index()

    @property
    def total_net(self) -> float:
        return float(sum(t.net_usd for t in self.trades))


def run_backtest(rule: TradeRule, bars: pd.DataFrame, feats: pd.DataFrame,
                 cfg: Config, cost_model: Optional[CostModel] = None,
                 instrument: str = "NQ") -> BacktestResult:
    cm = cost_model or CostModel(cfg)
    pv = cfg.instruments[instrument].point_value
    tick = cfg.instruments[instrument].tick_size
    sig = rule.entry_signal(feats, cfg).to_numpy()
    entry_ok = bars["entry_ok"].to_numpy()
    atr_pts = (feats["f_atr_ticks"] * tick).to_numpy()
    opn = bars["open"].to_numpy(float)
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    idx = bars.index
    info = pd.DatetimeIndex(bars["info_time"])
    sess_vals = bars["session"].to_numpy()

    entry_cost = cm.side_cost(instrument, rule.qty, "market").total
    stop_cost = cm.side_cost(instrument, rule.qty, "stop").total
    limit_cost = cm.side_cost(instrument, rule.qty, "limit").total
    market_cost = entry_cost

    res = BacktestResult(rule_name=rule.name)
    # session boundaries + deadline (same convention as the label layer)
    starts: Dict[date, int] = {}
    ends: Dict[date, int] = {}
    for i, s in enumerate(sess_vals):
        starts.setdefault(s, i)
        ends[s] = i

    for s in starts:
        flat_utc = S.eod_flatten_utc(s, cfg.sessions)
        dl = ends[s]
        while dl >= starts[s] and info[dl] >= flat_utc:
            dl -= 1
        if dl <= starts[s]:
            continue
        i = starts[s]
        day_net = 0.0
        n_trades = 0
        while i < dl:                        # entry bar must be <= dl
            if not (sig[i] and entry_ok[i]):
                i += 1
                continue
            if n_trades >= rule.max_trades_per_day:
                break
            if day_net <= -rule.daily_loss_usd:
                break
            if (rule.daily_profit_lock_usd is not None
                    and day_net >= rule.daily_profit_lock_usd):
                break
            a = atr_pts[i]
            if not np.isfinite(a) or a <= 0:
                i += 1
                continue
            e = i + 1                        # entry bar
            d = rule.direction
            entry = opn[e]
            stop_px = entry - d * rule.stop_atr * a
            target_px = entry + d * rule.target_atr * a
            exit_j, exit_px, reason = dl, close[dl], "flatten"
            mae_px = 0.0
            for j in range(e, dl + 1):
                adverse = (entry - low[j]) if d > 0 else (high[j] - entry)
                mae_px = max(mae_px, adverse)
                stop_hit = low[j] <= stop_px if d > 0 else high[j] >= stop_px
                tgt_hit = high[j] >= target_px if d > 0 else low[j] <= target_px
                if stop_hit:                 # stop-first tie resolution
                    exit_j, exit_px, reason = j, stop_px, "stop"
                    break
                if tgt_hit:
                    exit_j, exit_px, reason = j, target_px, "target"
                    break
                if j - e + 1 >= rule.time_stop_bars:
                    exit_j, exit_px, reason = j, close[j], "time"
                    break
            mae_px = min(mae_px, rule.stop_atr * a)   # bounded by the stop
            gross = d * (exit_px - entry) * pv * rule.qty
            costs = entry_cost + {"stop": stop_cost, "target": limit_cost,
                                  "time": market_cost,
                                  "flatten": market_cost}[reason]
            net = gross - costs
            res.trades.append(Trade(
                session=s, entry_time=idx[e], exit_time=idx[exit_j],
                direction=d, qty=rule.qty, entry_px=float(entry),
                exit_px=float(exit_px), exit_reason=reason,
                gross_usd=float(gross), costs_usd=float(costs),
                net_usd=float(net), mae_usd=float(mae_px * pv * rule.qty),
                risk_usd=float(rule.stop_atr * a * pv * rule.qty)))
            day_net += net
            n_trades += 1
            if info[exit_j] >= flat_utc:     # cannot happen by construction
                res.violations += 1
            i = exit_j + 1                   # no overlapping positions
    return res
