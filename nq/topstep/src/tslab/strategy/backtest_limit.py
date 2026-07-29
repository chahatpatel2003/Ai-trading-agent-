"""Limit-entry variant of the deterministic bar-loop backtester.

Entry flow (vs backtest.run_backtest, which enters market at open[T+1]):
  - decision at info_time(T); a limit order is placed at t0 = open time of
    bar e = T+1, at a price derived from the standing book at t0 (recorded
    in the pre-built fill table): join near quote (delta 0), improve one
    tick when spread >= 2 (delta -1), or rest 1-2 ticks behind (delta 1/2)
  - the fill table's adjudication against the real MBP-1 path decides IF
    and WHEN the order fills: 'thr' = price traded strictly through the
    level (fills any queue position — conservative), 'qf' = full displayed
    queue at placement traded out first (zero-cancel assumption)
  - unfilled by the cancel deadline (N bars) -> MISSED attempt; no trade,
    no day-guard consumption; scanning resumes at the cancel bar
  - filled -> entry at the limit price, entry cost = exchange+clearing
    fees ONLY (no spread, no slippage: passivity is priced by the path);
    stop/target anchored at the fill price with the signal bar's ATR
  - within the fill bar, exits are checked against POST-FILL trade-price
    extremes only (stop-first); later bars use full OHLC as usual
  - every attempt also carries a market-entry counterfactual (entry at
    open[e], market costs, standard engine path) so fill-selection bias
    (adverse selection) is directly measurable: compare filled vs missed
    counterfactual outcomes.

A position/pending order blocks new signals; flatten-by-deadline remains
structurally impossible to violate (fills after the deadline bar are
treated as cancels; open positions exit at the deadline bar close).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .. import sessions as S
from ..config import Config
from ..sim.costs import CostModel
from .backtest import BacktestResult, Trade
from .rules import TradeRule


@dataclass(frozen=True)
class EntryModel:
    name: str
    delta: int              # -1 improve | 0 join | 1/2 rest behind
    cancel_bars: int        # cancel if unfilled after N minutes
    fill_rule: str          # 'thr' (through) | 'qf' (queue)


@dataclass
class Attempt:
    session: object
    signal_time: pd.Timestamp
    entry_bar_time: pd.Timestamp
    direction: int
    filled: bool
    fill_dt_s: float        # NaN if unfilled
    limit_px: float
    net_usd: float          # NaN if unfilled
    exit_reason: str        # '' if unfilled
    cf_net_usd: float       # market-entry counterfactual net
    cf_exit_reason: str
    spread0_ticks: float


def build_fill_lookup(fill_table: pd.DataFrame, model: EntryModel
                      ) -> Dict[int, Dict[int, tuple]]:
    """{side: {bar_ts_ns: (dt_s, bar_off, pf_min, pf_max, limit_px,
    spread0)}} for the model's delta and fill rule."""
    sub = fill_table[fill_table["delta"] == model.delta]
    pre = "thr" if model.fill_rule == "thr" else "qf"
    out: Dict[int, Dict[int, tuple]] = {1: {}, -1: {}}
    ts = sub["bar_ts"].astype("int64").to_numpy()
    side = sub["side"].to_numpy()
    dt = sub[f"{pre}_dt"].to_numpy(float)
    off = sub[f"{pre}_bar_off"].to_numpy(float)
    pmn = sub[f"{pre}_pf_min"].to_numpy(float)
    pmx = sub[f"{pre}_pf_max"].to_numpy(float)
    lpx = sub["limit_px"].to_numpy(float)
    spr = sub["spread0_ticks"].to_numpy(float)
    for k in range(len(sub)):
        out[int(side[k])][int(ts[k])] = (dt[k], off[k], pmn[k], pmx[k],
                                         lpx[k], spr[k])
    return out


def run_backtest_limit(rule: TradeRule, bars: pd.DataFrame,
                       feats: pd.DataFrame, cfg: Config, cm: CostModel,
                       lookup: Dict[int, Dict[int, tuple]],
                       model: EntryModel, instrument: str = "NQ",
                       collect_attempts: bool = False
                       ) -> Tuple[BacktestResult, list]:
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
    idx_ns = idx.asi8
    info = pd.DatetimeIndex(bars["info_time"])
    sess_vals = bars["session"].to_numpy()

    fees_side = cm.side_cost(instrument, rule.qty, "limit").fees
    mkt_entry_cost = cm.side_cost(instrument, rule.qty, "market").total
    stop_cost = cm.side_cost(instrument, rule.qty, "stop").total
    limit_cost = cm.side_cost(instrument, rule.qty, "limit").total
    market_cost = cm.side_cost(instrument, rule.qty, "market").total
    exit_cost = {"stop": stop_cost, "target": limit_cost,
                 "time": market_cost, "flatten": market_cost}

    d = rule.direction
    side_map = lookup[d]
    cancel_s = model.cancel_bars * 60.0

    res = BacktestResult(rule_name=f"{rule.name}|{model.name}")
    attempts: list = []
    starts: Dict = {}
    ends: Dict = {}
    for i, s in enumerate(sess_vals):
        starts.setdefault(s, i)
        ends[s] = i

    def scan_exit(j0: int, dl: int, entry: float, stop_px: float,
                  target_px: float, first_bar_bounds=None):
        """Exit scan from bar j0; optional (lo,hi) overrides bar j0's
        low/high (post-fill extremes). Returns (j, px, reason, mae_px)."""
        mae = 0.0
        for j in range(j0, dl + 1):
            lo_j, hi_j = low[j], high[j]
            if j == j0 and first_bar_bounds is not None:
                lo_j, hi_j = first_bar_bounds
                if not np.isfinite(lo_j):
                    continue                      # no post-fill trades
            adverse = (entry - lo_j) if d > 0 else (hi_j - entry)
            mae = max(mae, adverse)
            stop_hit = lo_j <= stop_px if d > 0 else hi_j >= stop_px
            tgt_hit = hi_j >= target_px if d > 0 else lo_j <= target_px
            if stop_hit:
                return j, stop_px, "stop", mae
            if tgt_hit:
                return j, target_px, "target", mae
            if j - j0 + 1 >= rule.time_stop_bars:
                return j, close[j], "time", mae
        return dl, close[dl], "flatten", mae

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
        while i < dl:
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
            e = i + 1
            row = side_map.get(idx_ns[e])
            # ---- market-entry counterfactual (always computed) ----
            cf_entry = opn[e]
            cf_j, cf_px, cf_reason, _ = scan_exit(
                e, dl, cf_entry, cf_entry - d * rule.stop_atr * a,
                cf_entry + d * rule.target_atr * a)
            cf_gross = d * (cf_px - cf_entry) * pv * rule.qty
            cf_net = cf_gross - (mkt_entry_cost + exit_cost[cf_reason])

            filled = False
            fill_dt = float("nan")
            limit_px = float("nan")
            spread0 = float("nan")
            net = float("nan")
            reason = ""
            if row is not None:
                dt_s, off, pf_mn, pf_mx, limit_px, spread0 = row
                f = e + int(off) if np.isfinite(off) else 10 ** 9
                if np.isfinite(dt_s) and dt_s < cancel_s and f <= dl:
                    filled = True
                    fill_dt = dt_s
            if filled:
                entry = limit_px
                stop_px = entry - d * rule.stop_atr * a
                target_px = entry + d * rule.target_atr * a
                exit_j, exit_px, reason, mae_px = scan_exit(
                    f, dl, entry, stop_px, target_px,
                    first_bar_bounds=(pf_mn, pf_mx))
                mae_px = min(mae_px, rule.stop_atr * a)
                gross = d * (exit_px - entry) * pv * rule.qty
                costs = fees_side + exit_cost[reason]
                net = gross - costs
                res.trades.append(Trade(
                    session=s, entry_time=idx[f], exit_time=idx[exit_j],
                    direction=d, qty=rule.qty, entry_px=float(entry),
                    exit_px=float(exit_px), exit_reason=reason,
                    gross_usd=float(gross), costs_usd=float(costs),
                    net_usd=float(net),
                    mae_usd=float(mae_px * pv * rule.qty),
                    risk_usd=float(rule.stop_atr * a * pv * rule.qty)))
                day_net += net
                n_trades += 1
                if info[exit_j] >= flat_utc:
                    res.violations += 1
                nxt = exit_j + 1
            else:
                nxt = e + model.cancel_bars
            if collect_attempts:
                attempts.append(Attempt(
                    session=s, signal_time=idx[i], entry_bar_time=idx[e],
                    direction=d, filled=filled, fill_dt_s=fill_dt,
                    limit_px=float(limit_px), net_usd=float(net),
                    exit_reason=reason, cf_net_usd=float(cf_net),
                    cf_exit_reason=cf_reason,
                    spread0_ticks=float(spread0)))
            i = nxt
    return res, attempts
