"""Declarative strategy candidates: filters + trade rule.

A candidate is a frozen, fully-deterministic description — no fitted state,
no randomness. Filters reference ONLY gated feature columns (f_*), which are
causal by construction and audited; a rule therefore inherits the feature
layer's leakage guarantees, and its entry signal is re-checkable with
TruncationInvarianceCheck at any time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config


@dataclass(frozen=True)
class Filter:
    feature: str
    op: str                      # gt | lt | between | abs_gt | abs_lt | eq
    lo: float
    hi: Optional[float] = None

    def mask(self, feats: pd.DataFrame) -> pd.Series:
        s = feats[self.feature]
        if self.op == "gt":
            m = s > self.lo
        elif self.op == "lt":
            m = s < self.lo
        elif self.op == "between":
            m = (s >= self.lo) & (s <= self.hi)
        elif self.op == "abs_gt":
            m = s.abs() > self.lo
        elif self.op == "abs_lt":
            m = s.abs() < self.lo
        elif self.op == "eq":
            m = s == self.lo
        elif self.op == "lt_or_na":
            # regime guard that must not veto when the feature is UNAVAILABLE
            # (e.g. spread features are NaN after MBP-1 coverage ends)
            return (s < self.lo) | s.isna()
        elif self.op == "gt_or_na":
            return (s > self.lo) | s.isna()
        else:
            raise ValueError(f"unknown op {self.op!r}")
        return m & s.notna()     # NaN never satisfies a filter


@dataclass(frozen=True)
class TradeRule:
    name: str
    direction: int                       # +1 long, -1 short
    filters: Tuple[Filter, ...]
    entry_start_ct: time = time(8, 35)
    entry_end_ct: time = time(14, 0)
    stop_atr: float = 1.0
    target_atr: float = 1.5
    time_stop_bars: int = 60
    max_trades_per_day: int = 3
    daily_loss_usd: float = 750.0        # strategy's own day stop (< personal DLL)
    daily_profit_lock_usd: Optional[float] = None
    qty: int = 1                         # minis

    @property
    def rr(self) -> float:
        return self.target_atr / self.stop_atr

    def entry_signal(self, feats: pd.DataFrame, cfg: Config) -> pd.Series:
        """Boolean series over bar rows: all filters pass AND the DECISION
        time (info_time, when the bar is closed) falls inside the rule's
        entry window. Row T signals an entry at open[T+1]."""
        m = pd.Series(True, index=feats.index)
        for f in self.filters:
            m &= f.mask(feats)
        ct = pd.DatetimeIndex(feats["info_time"]).tz_convert(cfg.sessions.timezone)
        tt = pd.Series([t.time() for t in ct], index=feats.index)
        return m & (tt >= self.entry_start_ct) & (tt <= self.entry_end_ct)


def full_candidate_grid() -> Tuple[TradeRule, ...]:
    """Phase 3C search grid (~100 deterministic candidates). Thresholds are
    fixed a priori — nothing is fitted inside a fold; walk-forward selection
    chooses among these using train sessions only.

    Spread/liquidity guards use *_or_na ops so a candidate is not silently
    vetoed where quote features are unavailable (post-2025-06 trade-only
    bars). No strategy family is assumed to work: weak families are expected
    to die in train ranking.
    """
    liq = Filter("f_spread_ratio", "lt_or_na", 1.5)
    vol_any = Filter("f_atr_ticks", "between", 4.0, 120.0)
    vol_hi = Filter("f_atr_ticks", "between", 12.0, 120.0)
    vol_lo = Filter("f_atr_ticks", "between", 4.0, 12.0)
    out: list = []

    def add(name, direction, filters, start, end, stop, target, ts,
            mt=3, dls=600.0, lock=None):
        out.append(TradeRule(
            name=name, direction=direction, filters=tuple(filters),
            entry_start_ct=start, entry_end_ct=end, stop_atr=stop,
            target_atr=target, time_stop_bars=ts, max_trades_per_day=mt,
            daily_loss_usd=dls, daily_profit_lock_usd=lock))

    # ---- A. opening-range breakout (bias via prior-close gap) ----
    for rr in (1.5, 2.5):
        for vtag, vf in (("", vol_any), ("_hv", vol_hi)):
            add(f"orb_L_rr{rr}{vtag}", +1,
                [Filter("f_or_break_up", "eq", 1.0), vf, liq,
                 Filter("f_prior_close_gap", "gt", 0.0)],
                time(9, 0), time(11, 30), 1.0, rr, 60)
            add(f"orb_S_rr{rr}{vtag}", -1,
                [Filter("f_or_break_dn", "eq", 1.0), vf, liq,
                 Filter("f_prior_close_gap", "lt", 0.0)],
                time(9, 0), time(11, 30), 1.0, rr, 60)

    # ---- B. momentum continuation ----
    for thr in (0.0008, 0.0015):
        for rr in (1.5, 2.0):
            add(f"mom_L_t{thr}_rr{rr}", +1,
                [Filter("f_ret_15", "gt", thr),
                 Filter("f_sma_dist_20", "gt", 0.3), vol_any, liq],
                time(8, 35), time(14, 0), 1.0, rr, 45)
            add(f"mom_S_t{thr}_rr{rr}", -1,
                [Filter("f_ret_15", "lt", -thr),
                 Filter("f_sma_dist_20", "lt", -0.3), vol_any, liq],
                time(8, 35), time(14, 0), 1.0, rr, 45)

    # ---- C. trend pullback (trend bias + short-term mean reversion) ----
    for rr in (1.5, 2.0):
        add(f"pull_L_rr{rr}", +1,
            [Filter("f_sma_dist_60", "gt", 1.0), Filter("f_z_30", "lt", -1.0),
             vol_any, liq], time(8, 35), time(14, 0), 1.0, rr, 40)
        add(f"pull_S_rr{rr}", -1,
            [Filter("f_sma_dist_60", "lt", -1.0), Filter("f_z_30", "gt", 1.0),
             vol_any, liq], time(8, 35), time(14, 0), 1.0, rr, 40)

    # ---- D. mean-reversion at z extremes ----
    for w, thr in ((30, 2.0), (30, 2.5), (120, 2.0), (120, 2.5)):
        add(f"mrz_L_w{w}_t{thr}", +1,
            [Filter(f"f_z_{w}", "lt", -thr),
             Filter("f_svwap_dist_atr", "lt", -1.5), vol_any, liq],
            time(8, 35), time(14, 0), 1.2, 1.2, 30)
        add(f"mrz_S_w{w}_t{thr}", -1,
            [Filter(f"f_z_{w}", "gt", thr),
             Filter("f_svwap_dist_atr", "gt", 1.5), vol_any, liq],
            time(8, 35), time(14, 0), 1.2, 1.2, 30)

    # ---- E. VWAP band reversion (with/without flow confirmation) ----
    for tag, extra_l, extra_s in (
            ("", [], []),
            ("_dx", [Filter("f_delta_5", "gt", 0.0)],
             [Filter("f_delta_5", "lt", 0.0)])):
        add(f"vwapfade_L{tag}", +1,
            [Filter("f_svwap_dist_atr", "between", -6.0, -2.5), vol_any, liq]
            + extra_l, time(8, 35), time(14, 0), 1.0, 1.0, 30)
        add(f"vwapfade_S{tag}", -1,
            [Filter("f_svwap_dist_atr", "between", 2.5, 6.0), vol_any, liq]
            + extra_s, time(8, 35), time(14, 0), 1.0, 1.0, 30)

    # ---- F. aggressor-flow momentum ----
    for thr in (0.2, 0.35):
        for rr in (1.5, 2.5):
            add(f"flow_L_t{thr}_rr{rr}", +1,
                [Filter("f_delta_15", "gt", thr), Filter("f_ret_5", "gt", 0.0),
                 Filter("f_vol_ratio", "gt", 1.2), vol_any, liq],
                time(8, 35), time(14, 0), 1.0, rr, 30)
            add(f"flow_S_t{thr}_rr{rr}", -1,
                [Filter("f_delta_15", "lt", -thr), Filter("f_ret_5", "lt", 0.0),
                 Filter("f_vol_ratio", "gt", 1.2), vol_any, liq],
                time(8, 35), time(14, 0), 1.0, rr, 30)

    # ---- G. prior-day levels: breakout vs fade ----
    for rr in (1.5, 2.5):
        add(f"pdh_break_L_rr{rr}", +1,
            [Filter("f_prior_high_dist", "between", 0.0, 1.5),
             Filter("f_ret_5", "gt", 0.0), vol_any, liq],
            time(8, 35), time(13, 0), 1.0, rr, 45)
        add(f"pdl_break_S_rr{rr}", -1,
            [Filter("f_prior_low_dist", "between", -1.5, 0.0),
             Filter("f_ret_5", "lt", 0.0), vol_any, liq],
            time(8, 35), time(13, 0), 1.0, rr, 45)
    add("pdh_fade_S", -1,
        [Filter("f_prior_high_dist", "between", 0.0, 0.8),
         Filter("f_ret_5", "lt", 0.0), vol_any, liq],
        time(8, 35), time(13, 0), 1.2, 1.2, 30)
    add("pdl_fade_L", +1,
        [Filter("f_prior_low_dist", "between", -0.8, 0.0),
         Filter("f_ret_5", "gt", 0.0), vol_any, liq],
        time(8, 35), time(13, 0), 1.2, 1.2, 30)

    # ---- H. opening gap plays (first hour) ----
    for rr in (1.2, 2.0):
        add(f"gapfade_S_rr{rr}", -1,
            [Filter("f_prior_close_gap", "gt", 1.5), vol_any, liq],
            time(8, 35), time(9, 30), 1.2, rr, 45)
        add(f"gapfade_L_rr{rr}", +1,
            [Filter("f_prior_close_gap", "lt", -1.5), vol_any, liq],
            time(8, 35), time(9, 30), 1.2, rr, 45)
        add(f"gapgo_L_rr{rr}", +1,
            [Filter("f_prior_close_gap", "gt", 1.5),
             Filter("f_ret_5", "gt", 0.0), vol_any, liq],
            time(8, 35), time(9, 30), 1.0, rr, 45)
        add(f"gapgo_S_rr{rr}", -1,
            [Filter("f_prior_close_gap", "lt", -1.5),
             Filter("f_ret_5", "lt", 0.0), vol_any, liq],
            time(8, 35), time(9, 30), 1.0, rr, 45)

    # ---- I. dynamic RR: volatility-conditioned stop/target geometry ----
    add("dynrr_mom_L_lv", +1,
        [Filter("f_ret_15", "gt", 0.0010), Filter("f_sma_dist_20", "gt", 0.3),
         vol_lo, liq], time(8, 35), time(14, 0), 1.0, 1.25, 25)
    add("dynrr_mom_L_hv", +1,
        [Filter("f_ret_15", "gt", 0.0010), Filter("f_sma_dist_20", "gt", 0.3),
         vol_hi, liq], time(8, 35), time(14, 0), 1.0, 2.5, 60)
    add("dynrr_mom_S_lv", -1,
        [Filter("f_ret_15", "lt", -0.0010), Filter("f_sma_dist_20", "lt", -0.3),
         vol_lo, liq], time(8, 35), time(14, 0), 1.0, 1.25, 25)
    add("dynrr_mom_S_hv", -1,
        [Filter("f_ret_15", "lt", -0.0010), Filter("f_sma_dist_20", "lt", -0.3),
         vol_hi, liq], time(8, 35), time(14, 0), 1.0, 2.5, 60)
    add("dynrr_orb_L_hv", +1,
        [Filter("f_or_break_up", "eq", 1.0), vol_hi, liq],
        time(9, 0), time(11, 30), 1.0, 2.5, 60)
    add("dynrr_orb_S_hv", -1,
        [Filter("f_or_break_dn", "eq", 1.0), vol_hi, liq],
        time(9, 0), time(11, 30), 1.0, 2.5, 60)

    # ---- J. afternoon trend ----
    for rr in (1.5, 2.0):
        add(f"pm_trend_L_rr{rr}", +1,
            [Filter("f_ret_60", "gt", 0.0015),
             Filter("f_sma_dist_60", "gt", 0.5), vol_any, liq],
            time(12, 30), time(14, 15), 1.0, rr, 60)
        add(f"pm_trend_S_rr{rr}", -1,
            [Filter("f_ret_60", "lt", -0.0015),
             Filter("f_sma_dist_60", "lt", -0.5), vol_any, liq],
            time(12, 30), time(14, 15), 1.0, rr, 60)

    # ---- K. lunch reversion ----
    add("lunch_mr_L", +1, [Filter("f_z_30", "lt", -2.0), vol_any, liq],
        time(11, 30), time(13, 0), 1.0, 1.0, 25)
    add("lunch_mr_S", -1, [Filter("f_z_30", "gt", 2.0), vol_any, liq],
        time(11, 30), time(13, 0), 1.0, 1.0, 25)

    # ---- L. participation-regime continuation ----
    add("part_L", +1,
        [Filter("f_vol_ratio", "gt", 1.5), Filter("f_ntrades_z", "gt_or_na", 1.0),
         Filter("f_ret_5", "gt", 0.0005), vol_any, liq],
        time(8, 35), time(14, 0), 1.0, 2.0, 30)
    add("part_S", -1,
        [Filter("f_vol_ratio", "gt", 1.5), Filter("f_ntrades_z", "gt_or_na", 1.0),
         Filter("f_ret_5", "lt", -0.0005), vol_any, liq],
        time(8, 35), time(14, 0), 1.0, 2.0, 30)

    # ---- C2. slow-trend pullback on the long z-window ----
    for rr in (1.5, 2.0):
        add(f"pull_L_z120_rr{rr}", +1,
            [Filter("f_sma_dist_60", "gt", 1.0), Filter("f_z_120", "lt", -0.8),
             vol_any, liq], time(8, 35), time(14, 0), 1.0, rr, 60)
        add(f"pull_S_z120_rr{rr}", -1,
            [Filter("f_sma_dist_60", "lt", -1.0), Filter("f_z_120", "gt", 0.8),
             vol_any, liq], time(8, 35), time(14, 0), 1.0, rr, 60)

    # ---- E2. tighter VWAP band (earlier fade) ----
    for tag, extra_l, extra_s in (
            ("", [], []),
            ("_dx", [Filter("f_delta_5", "gt", 0.0)],
             [Filter("f_delta_5", "lt", 0.0)])):
        add(f"vwapfade2_L{tag}", +1,
            [Filter("f_svwap_dist_atr", "between", -4.0, -2.0), vol_any, liq]
            + extra_l, time(8, 35), time(14, 0), 1.0, 1.0, 25)
        add(f"vwapfade2_S{tag}", -1,
            [Filter("f_svwap_dist_atr", "between", 2.0, 4.0), vol_any, liq]
            + extra_s, time(8, 35), time(14, 0), 1.0, 1.0, 25)

    # ---- B2. momentum: morning-only window variants ----
    for rr in (1.5, 2.0):
        add(f"mom_L_am_rr{rr}", +1,
            [Filter("f_ret_15", "gt", 0.0010),
             Filter("f_sma_dist_20", "gt", 0.3), vol_any, liq],
            time(8, 35), time(11, 0), 1.0, rr, 45)
        add(f"mom_S_am_rr{rr}", -1,
            [Filter("f_ret_15", "lt", -0.0010),
             Filter("f_sma_dist_20", "lt", -0.3), vol_any, liq],
            time(8, 35), time(11, 0), 1.0, rr, 45)

    # ---- K2 / J2. lunch fade on slow z; afternoon trend with flow ----
    add("lunch_mr_L_z120", +1, [Filter("f_z_120", "lt", -1.8), vol_any, liq],
        time(11, 30), time(13, 0), 1.0, 1.0, 30)
    add("lunch_mr_S_z120", -1, [Filter("f_z_120", "gt", 1.8), vol_any, liq],
        time(11, 30), time(13, 0), 1.0, 1.0, 30)
    add("pm_trend_L_flow", +1,
        [Filter("f_ret_60", "gt", 0.0015), Filter("f_sma_dist_60", "gt", 0.5),
         Filter("f_delta_15", "gt", 0.05), vol_any, liq],
        time(12, 30), time(14, 15), 1.0, 2.0, 60)
    add("pm_trend_S_flow", -1,
        [Filter("f_ret_60", "lt", -0.0015), Filter("f_sma_dist_60", "lt", -0.5),
         Filter("f_delta_15", "lt", -0.05), vol_any, liq],
        time(12, 30), time(14, 15), 1.0, 2.0, 60)

    # ---- M. day-guard variants of core families (profit lock / fewer shots)
    core = {r.name: r for r in out}
    for base in ("orb_L_rr2.5", "orb_S_rr2.5", "mom_L_t0.0015_rr2.0",
                 "mom_S_t0.0015_rr2.0", "mrz_L_w30_t2.5", "mrz_S_w30_t2.5",
                 "flow_L_t0.35_rr2.5", "flow_S_t0.35_rr2.5"):
        b = core[base]
        out.append(TradeRule(
            name=f"{base}_lock", direction=b.direction, filters=b.filters,
            entry_start_ct=b.entry_start_ct, entry_end_ct=b.entry_end_ct,
            stop_atr=b.stop_atr, target_atr=b.target_atr,
            time_stop_bars=b.time_stop_bars, max_trades_per_day=2,
            daily_loss_usd=600.0, daily_profit_lock_usd=900.0))
    return tuple(out)


def lowfreq_candidate_grid() -> Tuple[TradeRule, ...]:
    """Phase 4A candidate class: lower-frequency, larger-target intraday
    rules (1-2 trades/day, 6-10x 1-min-ATR stops, 1.5R/2R/3R targets,
    30min-5.5h holds, forced flat before the deadline by the engine).

    Every rule carries an f_atr_ticks band so worst-case per-trade risk
    (stop_atr x band_hi x $5/tick) stays <= ~$700 — under the $1,000
    personal DLL with room for slippage. HTF-style context (trend/range
    day, VWAP location, vol expansion/contraction) is composed from the
    existing causal features only. Thresholds fixed a priori.
    """
    liq = Filter("f_spread_ratio", "lt_or_na", 1.6)
    band_a = Filter("f_atr_ticks", "between", 5.0, 15.0)   # stop 8 -> <=$600
    band_b = Filter("f_atr_ticks", "between", 5.0, 12.0)   # stop 10 -> <=$600
    band_c = Filter("f_atr_ticks", "between", 8.0, 22.0)   # stop 6 -> <=$660
    band_d = Filter("f_atr_ticks", "between", 5.0, 10.0)   # contraction
    band_e = Filter("f_atr_ticks", "between", 10.0, 20.0)  # expansion
    out: list = []

    def add(name, direction, filters, start, end, stop, r_mult, ts,
            mt=1, lock=None):
        out.append(TradeRule(
            name=name, direction=direction, filters=tuple(filters),
            entry_start_ct=start, entry_end_ct=end, stop_atr=stop,
            target_atr=stop * r_mult, time_stop_bars=ts,
            max_trades_per_day=mt, daily_loss_usd=700.0,
            daily_profit_lock_usd=lock))

    # ---- A. open-drive trend day: OR break + extended off VWAP + gap bias
    for r in (1.5, 2.0, 3.0):
        add(f"od_L_r{r}", +1,
            [Filter("f_or_break_up", "eq", 1.0),
             Filter("f_svwap_dist_atr", "gt", 1.0),
             Filter("f_prior_close_gap", "gt", 0.0), band_a, liq],
            time(9, 0), time(10, 30), 8.0, r, 330)
        add(f"od_S_r{r}", -1,
            [Filter("f_or_break_dn", "eq", 1.0),
             Filter("f_svwap_dist_atr", "lt", -1.0),
             Filter("f_prior_close_gap", "lt", 0.0), band_a, liq],
            time(9, 0), time(10, 30), 8.0, r, 330)

    # ---- B. pullback-to-VWAP hold/reclaim in an established trend ----
    for stop, band in ((6.0, band_c), (8.0, band_a)):
        for r in (2.0, 3.0):
            add(f"vwaphold_L_s{stop:.0f}_r{r}", +1,
                [Filter("f_sma_dist_60", "gt", 1.5),
                 Filter("f_svwap_dist_atr", "between", -0.5, 0.5),
                 Filter("f_delta_15", "gt", 0.0), band, liq],
                time(10, 0), time(12, 0), stop, r, 240, mt=2)
            add(f"vwaphold_S_s{stop:.0f}_r{r}", -1,
                [Filter("f_sma_dist_60", "lt", -1.5),
                 Filter("f_svwap_dist_atr", "between", -0.5, 0.5),
                 Filter("f_delta_15", "lt", 0.0), band, liq],
                time(10, 0), time(12, 0), stop, r, 240, mt=2)

    # ---- C. range-day VWAP-band rejection fade ----
    for r in (1.5, 2.0):
        add(f"rngfade_S_r{r}", -1,
            [Filter("f_sma_dist_60", "between", -0.5, 0.5),
             Filter("f_range_atr", "lt", 1.2),
             Filter("f_svwap_dist_atr", "between", 1.5, 4.0), band_c, liq],
            time(10, 0), time(13, 0), 6.0, r, 240, mt=2)
        add(f"rngfade_L_r{r}", +1,
            [Filter("f_sma_dist_60", "between", -0.5, 0.5),
             Filter("f_range_atr", "lt", 1.2),
             Filter("f_svwap_dist_atr", "between", -4.0, -1.5), band_c, liq],
            time(10, 0), time(13, 0), 6.0, r, 240, mt=2)
    for r in (1.5, 2.0):
        add(f"rngfade_z_S_r{r}", -1,
            [Filter("f_z_120", "gt", 1.8), Filter("f_range_atr", "lt", 1.2),
             band_c, liq], time(10, 0), time(13, 0), 6.0, r, 240, mt=2)
        add(f"rngfade_z_L_r{r}", +1,
            [Filter("f_z_120", "lt", -1.8), Filter("f_range_atr", "lt", 1.2),
             band_c, liq], time(10, 0), time(13, 0), 6.0, r, 240, mt=2)

    # ---- D. prior-day level: breakout hold vs rejection fade ----
    for r in (2.0, 3.0):
        add(f"pdh_hold_L_r{r}", +1,
            [Filter("f_prior_high_dist", "between", 0.0, 2.0),
             Filter("f_ret_60", "gt", 0.002), band_a, liq],
            time(9, 0), time(13, 0), 8.0, r, 330)
        add(f"pdl_hold_S_r{r}", -1,
            [Filter("f_prior_low_dist", "between", -2.0, 0.0),
             Filter("f_ret_60", "lt", -0.002), band_a, liq],
            time(9, 0), time(13, 0), 8.0, r, 330)
    add("pdh_rej_S", -1,
        [Filter("f_prior_high_dist", "between", 0.0, 1.0),
         Filter("f_delta_15", "lt", 0.0), band_c, liq],
        time(9, 0), time(13, 0), 6.0, 2.0, 240, mt=2)
    add("pdl_rej_L", +1,
        [Filter("f_prior_low_dist", "between", -1.0, 0.0),
         Filter("f_delta_15", "gt", 0.0), band_c, liq],
        time(9, 0), time(13, 0), 6.0, 2.0, 240, mt=2)

    # ---- E. lunch/PM with-trend entries ----
    for r in (1.5, 2.0):
        add(f"pm_L_r{r}", +1,
            [Filter("f_ret_60", "gt", 0.0015),
             Filter("f_svwap_dist_atr", "gt", 1.0), band_c, liq],
            time(12, 30), time(14, 0), 6.0, r, 120)
        add(f"pm_S_r{r}", -1,
            [Filter("f_ret_60", "lt", -0.0015),
             Filter("f_svwap_dist_atr", "lt", -1.0), band_c, liq],
            time(12, 30), time(14, 0), 6.0, r, 120)

    # ---- F. power hour: continuation vs exhaustion fade ----
    add("ph_cont_L", +1,
        [Filter("f_svwap_dist_atr", "gt", 2.0), Filter("f_delta_15", "gt", 0.0),
         band_c, liq], time(13, 30), time(14, 30), 6.0, 1.5, 90)
    add("ph_cont_S", -1,
        [Filter("f_svwap_dist_atr", "lt", -2.0), Filter("f_delta_15", "lt", 0.0),
         band_c, liq], time(13, 30), time(14, 30), 6.0, 1.5, 90)
    add("ph_fade_S", -1,
        [Filter("f_svwap_dist_atr", "gt", 3.5), band_c, liq],
        time(13, 30), time(14, 30), 6.0, 1.5, 90)
    add("ph_fade_L", +1,
        [Filter("f_svwap_dist_atr", "lt", -3.5), band_c, liq],
        time(13, 30), time(14, 30), 6.0, 1.5, 90)

    # ---- G. large-gap plays: go vs fill ----
    for r in (2.0, 3.0):
        add(f"gapgo_L_r{r}", +1,
            [Filter("f_prior_close_gap", "gt", 2.0),
             Filter("f_ret_15", "gt", 0.0), band_a, liq],
            time(8, 45), time(9, 45), 8.0, r, 330)
        add(f"gapgo_S_r{r}", -1,
            [Filter("f_prior_close_gap", "lt", -2.0),
             Filter("f_ret_15", "lt", 0.0), band_a, liq],
            time(8, 45), time(9, 45), 8.0, r, 330)
    add("gapfill_S", -1,
        [Filter("f_prior_close_gap", "gt", 2.0), Filter("f_ret_15", "lt", 0.0),
         band_a, liq], time(8, 45), time(9, 45), 8.0, 2.0, 330)
    add("gapfill_L", +1,
        [Filter("f_prior_close_gap", "lt", -2.0), Filter("f_ret_15", "gt", 0.0),
         band_a, liq], time(8, 45), time(9, 45), 8.0, 2.0, 330)

    # ---- H. volatility expansion breakout ----
    for r in (2.0, 3.0):
        add(f"volexp_L_r{r}", +1,
            [Filter("f_range_atr", "gt", 1.5), Filter("f_ret_15", "gt", 0.0),
             Filter("f_vol_ratio", "gt", 1.3), band_e, liq],
            time(9, 0), time(13, 0), 6.0, r, 240)
        add(f"volexp_S_r{r}", -1,
            [Filter("f_range_atr", "gt", 1.5), Filter("f_ret_15", "lt", 0.0),
             Filter("f_vol_ratio", "gt", 1.3), band_e, liq],
            time(9, 0), time(13, 0), 6.0, r, 240)

    # ---- I. volatility contraction range fade ----
    add("volcon_S", -1,
        [Filter("f_range_atr", "lt", 0.8), Filter("f_z_120", "gt", 1.5),
         band_d, liq], time(10, 30), time(13, 30), 6.0, 1.5, 180, mt=2)
    add("volcon_L", +1,
        [Filter("f_range_atr", "lt", 0.8), Filter("f_z_120", "lt", -1.5),
         band_d, liq], time(10, 30), time(13, 30), 6.0, 1.5, 180, mt=2)

    # ---- J. morning impulse, hold into the afternoon ----
    for r in (2.0, 3.0):
        add(f"impulse_L_r{r}", +1,
            [Filter("f_ret_60", "gt", 0.0025),
             Filter("f_sma_dist_60", "gt", 1.0), band_b, liq],
            time(9, 30), time(10, 30), 10.0, r, 300)
        add(f"impulse_S_r{r}", -1,
            [Filter("f_ret_60", "lt", -0.0025),
             Filter("f_sma_dist_60", "lt", -1.0), band_b, liq],
            time(9, 30), time(10, 30), 10.0, r, 300)

    # ---- K. profit-lock variants of the widest-target families ----
    core = {r.name: r for r in out}
    for base in ("od_L_r3.0", "od_S_r3.0", "impulse_L_r3.0", "impulse_S_r3.0",
                 "pdh_hold_L_r3.0", "pdl_hold_S_r3.0"):
        b = core[base]
        out.append(TradeRule(
            name=f"{base}_lock", direction=b.direction, filters=b.filters,
            entry_start_ct=b.entry_start_ct, entry_end_ct=b.entry_end_ct,
            stop_atr=b.stop_atr, target_atr=b.target_atr,
            time_stop_bars=b.time_stop_bars, max_trades_per_day=1,
            daily_loss_usd=700.0, daily_profit_lock_usd=1200.0))
    return tuple(out)


def pm_candidate_grid() -> Tuple[TradeRule, ...]:
    """Phase 4B-1 pre-registered grid: DENSE coverage of the lunch/PM/
    power-hour region only (the sole area with recurring — but underpowered —
    signal in Phase 4A). 2-3 trades/day, wider decision windows, 6x-ATR
    stops with band caps (worst-case risk <= $660), R in {1.5, 2, 2.5, 3},
    vol-conditioned (dynamic-target) variants. Thresholds fixed a priori;
    nothing is fitted inside folds.
    """
    liq = Filter("f_spread_ratio", "lt_or_na", 1.6)
    band_c = Filter("f_atr_ticks", "between", 8.0, 22.0)   # stop 6 -> <=$660
    band_d = Filter("f_atr_ticks", "between", 5.0, 10.0)   # contraction
    band_e = Filter("f_atr_ticks", "between", 10.0, 20.0)  # expansion
    W1 = (time(11, 30), time(14, 0))
    W2 = (time(12, 30), time(14, 30))
    W3 = (time(13, 0), time(14, 30))     # power hour (entry_ok caps at 14:30)
    W4 = (time(11, 0), time(14, 30))
    W5 = (time(12, 0), time(14, 30))
    out: list = []

    def add(name, direction, filters, win, r_mult, ts, stop=6.0, mt=2,
            lock=None):
        out.append(TradeRule(
            name=name, direction=direction, filters=tuple(filters),
            entry_start_ct=win[0], entry_end_ct=win[1], stop_atr=stop,
            target_atr=stop * r_mult, time_stop_bars=ts,
            max_trades_per_day=mt, daily_loss_usd=700.0,
            daily_profit_lock_usd=lock))

    # ---- A. PM with-trend (dense around the 4A recurring winner) ----
    trend1_L = [Filter("f_ret_60", "gt", 0.0015),
                Filter("f_svwap_dist_atr", "gt", 1.0)]
    trend1_S = [Filter("f_ret_60", "lt", -0.0015),
                Filter("f_svwap_dist_atr", "lt", -1.0)]
    for wtag, win in (("w1", W1), ("w2", W2)):
        for r in (1.5, 2.0, 2.5, 3.0):
            add(f"pmT_L_{wtag}_r{r}", +1, trend1_L + [band_c, liq],
                win, r, 120)
            add(f"pmT_S_{wtag}_r{r}", -1, trend1_S + [band_c, liq],
                win, r, 120)
    trend2_L = [Filter("f_sma_dist_60", "gt", 1.0),
                Filter("f_svwap_dist_atr", "gt", 0.5)]
    trend2_S = [Filter("f_sma_dist_60", "lt", -1.0),
                Filter("f_svwap_dist_atr", "lt", -0.5)]
    for r in (2.0, 3.0):
        add(f"pmT2_L_r{r}", +1, trend2_L + [band_c, liq], W2, r, 120)
        add(f"pmT2_S_r{r}", -1, trend2_S + [band_c, liq], W2, r, 120)

    # ---- B. power-hour continuation ----
    for thr in (1.5, 2.5):
        for r in (1.5, 2.0):
            add(f"phC_L_t{thr}_r{r}", +1,
                [Filter("f_svwap_dist_atr", "gt", thr),
                 Filter("f_delta_15", "gt", 0.0), band_c, liq], W3, r, 90)
            add(f"phC_S_t{thr}_r{r}", -1,
                [Filter("f_svwap_dist_atr", "lt", -thr),
                 Filter("f_delta_15", "lt", 0.0), band_c, liq], W3, r, 90)

    # ---- C. VWAP extension exhaustion fades ----
    for thr in (3.0, 3.5, 4.5):
        for r in (1.5, 2.0):
            add(f"vxF_S_t{thr}_r{r}", -1,
                [Filter("f_svwap_dist_atr", "gt", thr), band_c, liq],
                W4, r, 90)
            add(f"vxF_L_t{thr}_r{r}", +1,
                [Filter("f_svwap_dist_atr", "lt", -thr), band_c, liq],
                W4, r, 90)
    for r in (1.5, 2.0):                       # stretched-z co-filter variant
        add(f"vxFz_S_r{r}", -1,
            [Filter("f_svwap_dist_atr", "gt", 3.0),
             Filter("f_z_120", "gt", 2.0), band_c, liq], W4, r, 90)
        add(f"vxFz_L_r{r}", +1,
            [Filter("f_svwap_dist_atr", "lt", -3.0),
             Filter("f_z_120", "lt", -2.0), band_c, liq], W4, r, 90)
    for thr in (3.5,):                          # window variant
        for r in (1.5,):
            add(f"vxF5_S_t{thr}_r{r}", -1,
                [Filter("f_svwap_dist_atr", "gt", thr), band_c, liq],
                W5, r, 90)
            add(f"vxF5_L_t{thr}_r{r}", +1,
                [Filter("f_svwap_dist_atr", "lt", -thr), band_c, liq],
                W5, r, 90)

    # ---- D. prior-day high/low: fade and reclaim/hold ----
    for r in (1.5, 2.0):
        add(f"pdhF_S_r{r}", -1,
            [Filter("f_prior_high_dist", "between", 0.0, 1.0),
             Filter("f_delta_15", "lt", 0.0), band_c, liq], W4, r, 120)
        add(f"pdlF_L_r{r}", +1,
            [Filter("f_prior_low_dist", "between", -1.0, 0.0),
             Filter("f_delta_15", "gt", 0.0), band_c, liq], W4, r, 120)
    for r in (2.0, 3.0):
        add(f"pdhH_L_r{r}", +1,
            [Filter("f_prior_high_dist", "between", 0.0, 1.5),
             Filter("f_ret_15", "gt", 0.0), band_c, liq], W5, r, 150)
        add(f"pdlH_S_r{r}", -1,
            [Filter("f_prior_low_dist", "between", -1.5, 0.0),
             Filter("f_ret_15", "lt", 0.0), band_c, liq], W5, r, 150)

    # ---- E. opening-range continuation into the afternoon ----
    for r in (2.0, 3.0):
        add(f"orPM_L_r{r}", +1,
            [Filter("f_or_break_up", "eq", 1.0),
             Filter("f_svwap_dist_atr", "gt", 1.0), band_c, liq],
            W5, r, 150)
        add(f"orPM_S_r{r}", -1,
            [Filter("f_or_break_dn", "eq", 1.0),
             Filter("f_svwap_dist_atr", "lt", -1.0), band_c, liq],
            W5, r, 150)

    # ---- F. volatility-conditioned (dynamic target) variants ----
    for r in (2.0, 3.0):                        # expansion: wider targets
        add(f"pmT_L_hv_r{r}", +1, trend1_L + [band_e, liq], W2, r, 150)
        add(f"pmT_S_hv_r{r}", -1, trend1_S + [band_e, liq], W2, r, 150)
    add("pmT_L_lv_r1.5", +1, trend1_L + [band_d, liq], W2, 1.5, 90)
    add("pmT_S_lv_r1.5", -1, trend1_S + [band_d, liq], W2, 1.5, 90)
    add("vxF_S_lv_r1.5", -1,
        [Filter("f_svwap_dist_atr", "gt", 3.0), Filter("f_range_atr", "lt", 0.9),
         band_d, liq], W4, 1.5, 90)
    add("vxF_L_lv_r1.5", +1,
        [Filter("f_svwap_dist_atr", "lt", -3.0), Filter("f_range_atr", "lt", 0.9),
         band_d, liq], W4, 1.5, 90)

    # ---- G. max-trades-3 and profit-lock variants of the core ----
    core = {r.name: r for r in out}
    for base in ("pmT_L_w2_r2.0", "pmT_S_w2_r2.0", "vxF_S_t3.5_r1.5",
                 "vxF_L_t3.5_r1.5"):
        b = core[base]
        out.append(TradeRule(
            name=f"{base}_mt3", direction=b.direction, filters=b.filters,
            entry_start_ct=b.entry_start_ct, entry_end_ct=b.entry_end_ct,
            stop_atr=b.stop_atr, target_atr=b.target_atr,
            time_stop_bars=b.time_stop_bars, max_trades_per_day=3,
            daily_loss_usd=700.0))
    for base in ("pmT_L_w2_r3.0", "pmT_S_w2_r3.0"):
        b = core[base]
        out.append(TradeRule(
            name=f"{base}_lock", direction=b.direction, filters=b.filters,
            entry_start_ct=b.entry_start_ct, entry_end_ct=b.entry_end_ct,
            stop_atr=b.stop_atr, target_atr=b.target_atr,
            time_stop_bars=b.time_stop_bars, max_trades_per_day=2,
            daily_loss_usd=700.0, daily_profit_lock_usd=1200.0))
    return tuple(out)


def hf_candidate_grid() -> Tuple[TradeRule, ...]:
    """High-frequency / high-win-rate candidate class. Win rate is made
    STRUCTURAL: sub-1.0 RR (target closer than stop) mean-reversion,
    level sweep/reclaim, and scalp-out momentum — families whose exits are
    mostly cheap limit fills. Risk budget: per-trade worst case <= $500
    (stop_atr x ATR-band cap x $5), day stop -$900 under the $1,000 DLL.
    4-6 trades/day allowed; wide RTH windows for frequency. All thresholds
    a priori; regime gates (trend/range) are explicit filters.
    """
    liq = Filter("f_spread_ratio", "lt_or_na", 1.6)
    b4 = Filter("f_atr_ticks", "between", 4.0, 22.0)    # stop 4 -> <=$440
    b5 = Filter("f_atr_ticks", "between", 4.0, 18.0)    # stop 5 -> <=$450
    rng_regime = [Filter("f_range_atr", "lt", 1.1),
                  Filter("f_sma_dist_60", "between", -0.7, 0.7)]
    W = (time(8, 35), time(14, 15))
    AM = (time(8, 35), time(11, 0))
    PM = (time(11, 0), time(14, 15))
    out: list = []

    def add(name, direction, filters, win, stop, r_mult, ts, mt=5,
            dls=900.0):
        out.append(TradeRule(
            name=name, direction=direction, filters=tuple(filters),
            entry_start_ct=win[0], entry_end_ct=win[1], stop_atr=stop,
            target_atr=stop * r_mult, time_stop_bars=ts,
            max_trades_per_day=mt, daily_loss_usd=dls))

    # ---- A. VWAP band fades (workhorse family) ----
    for thr in (1.5, 2.0, 2.5, 3.0):
        for r in (0.5, 0.75, 1.0):
            add(f"vf_S_t{thr}_r{r}", -1,
                [Filter("f_svwap_dist_atr", "gt", thr), b5, liq],
                W, 5.0, r, 30)
            add(f"vf_L_t{thr}_r{r}", +1,
                [Filter("f_svwap_dist_atr", "lt", -thr), b5, liq],
                W, 5.0, r, 30)

    # ---- B. z-score extreme fades (fast + slow windows) ----
    for w, thr in ((30, 1.5), (30, 2.0), (120, 1.5), (120, 2.0)):
        for r in (0.5, 0.75):
            add(f"zf{w}_S_t{thr}_r{r}", -1,
                [Filter(f"f_z_{w}", "gt", thr), b5, liq], W, 5.0, r, 25)
            add(f"zf{w}_L_t{thr}_r{r}", +1,
                [Filter(f"f_z_{w}", "lt", -thr), b5, liq], W, 5.0, r, 25)

    # ---- C. range-regime gated fades (bias/regime engine) ----
    for r in (0.5, 0.75):
        add(f"rgvf_S_r{r}", -1,
            [Filter("f_svwap_dist_atr", "gt", 1.5)] + rng_regime + [b5, liq],
            W, 5.0, r, 30, mt=6)
        add(f"rgvf_L_r{r}", +1,
            [Filter("f_svwap_dist_atr", "lt", -1.5)] + rng_regime + [b5, liq],
            W, 5.0, r, 30, mt=6)

    # ---- D. prior-day level sweep/reclaim (flow-confirmed) ----
    for r in (0.5, 0.75):
        add(f"swpl_L_r{r}", +1,
            [Filter("f_prior_low_dist", "between", -0.3, 0.5),
             Filter("f_ret_5", "gt", 0.0), Filter("f_delta_5", "gt", 0.0),
             b5, liq], W, 5.0, r, 30)
        add(f"swph_S_r{r}", -1,
            [Filter("f_prior_high_dist", "between", -0.5, 0.3),
             Filter("f_ret_5", "lt", 0.0), Filter("f_delta_5", "lt", 0.0),
             b5, liq], W, 5.0, r, 30)

    # ---- E. opening-range level sweep/reclaim (post-OR) ----
    for r in (0.5, 0.75):
        add(f"orlo_L_r{r}", +1,
            [Filter("f_or_low_dist", "between", -0.3, 0.5),
             Filter("f_ret_5", "gt", 0.0), Filter("f_delta_5", "gt", 0.0),
             b5, liq], (time(9, 5), time(14, 15)), 5.0, r, 30)
        add(f"orhi_S_r{r}", -1,
            [Filter("f_or_high_dist", "between", -0.5, 0.3),
             Filter("f_ret_5", "lt", 0.0), Filter("f_delta_5", "lt", 0.0),
             b5, liq], (time(9, 5), time(14, 15)), 5.0, r, 30)

    # ---- F. VWAP reclaim scalp (cross zone with flow + trend bias) ----
    for r in (0.75, 1.0):
        add(f"vrec_L_r{r}", +1,
            [Filter("f_svwap_dist_atr", "between", -0.3, 0.3),
             Filter("f_sma_dist_20", "gt", 0.0), Filter("f_delta_5", "gt", 0.05),
             b4, liq], W, 4.0, r, 30)
        add(f"vrec_S_r{r}", -1,
            [Filter("f_svwap_dist_atr", "between", -0.3, 0.3),
             Filter("f_sma_dist_20", "lt", 0.0), Filter("f_delta_5", "lt", -0.05),
             b4, liq], W, 4.0, r, 30)

    # ---- G. trend-gated momentum scalp-outs (sub-1 RR grab) ----
    trend_L = [Filter("f_sma_dist_60", "gt", 0.7), Filter("f_ret_5", "gt", 0.0004)]
    trend_S = [Filter("f_sma_dist_60", "lt", -0.7), Filter("f_ret_5", "lt", -0.0004)]
    for r in (0.6, 0.8):
        add(f"msc_L_r{r}", +1, trend_L + [Filter("f_delta_5", "gt", 0.1),
                                          b4, liq], W, 4.0, r, 20)
        add(f"msc_S_r{r}", -1, trend_S + [Filter("f_delta_5", "lt", -0.1),
                                          b4, liq], W, 4.0, r, 20)

    # ---- H. time-of-day slices of the workhorse fades ----
    for tag, win in (("am", AM), ("pm", PM)):
        for r in (0.5, 0.75):
            add(f"vf_{tag}_S_r{r}", -1,
                [Filter("f_svwap_dist_atr", "gt", 2.0), b5, liq],
                win, 5.0, r, 30)
            add(f"vf_{tag}_L_r{r}", +1,
                [Filter("f_svwap_dist_atr", "lt", -2.0), b5, liq],
                win, 5.0, r, 30)

    # ---- I. tighter-stop variants of the best-frequency fades ----
    for thr in (2.0, 2.5):
        add(f"vf4_S_t{thr}", -1,
            [Filter("f_svwap_dist_atr", "gt", thr), b4, liq], W, 4.0, 0.75, 30)
        add(f"vf4_L_t{thr}", +1,
            [Filter("f_svwap_dist_atr", "lt", -thr), b4, liq], W, 4.0, 0.75, 30)
    return tuple(out)


def micro_candidate_grid() -> Tuple[TradeRule, ...]:
    """Phase 5A smoke grid: microstructure-confirmed entries (f_m_*).

    Risk arithmetic per the DLL budget: per-trade worst-case risk
    <= $1,000 / max_trades_per_day, enforced by stop_atr x ATR-band caps;
    daily_loss_usd = 1000 - per-trade cap, so realized day loss + one open
    trade's risk can never exceed the $1,000 DLL.
    """
    liq = Filter("f_spread_ratio", "lt_or_na", 1.6)
    W = (time(8, 35), time(14, 15))
    out: list = []

    def add(name, direction, filters, stop, band_hi, r_mult, ts, mt):
        cap = 1000.0 / mt
        assert stop * band_hi * 5.0 <= cap + 1e-9, (name, stop * band_hi * 5)
        out.append(TradeRule(
            name=name, direction=direction,
            filters=tuple(filters + [Filter("f_atr_ticks", "between",
                                            4.0, band_hi), liq]),
            entry_start_ct=W[0], entry_end_ct=W[1], stop_atr=stop,
            target_atr=stop * r_mult, time_stop_bars=ts,
            max_trades_per_day=mt, daily_loss_usd=1000.0 - cap))

    # 1. book-lean confirmed VWAP fades
    for thr in (2.0, 2.5):
        for r in (0.6, 0.8):
            add(f"mvf_S_t{thr}_r{r}", -1,
                [Filter("f_svwap_dist_atr", "gt", thr),
                 Filter("f_m_imb_z", "lt", -1.0)], 3.0, 16.0, r, 30, 4)
            add(f"mvf_L_t{thr}_r{r}", +1,
                [Filter("f_svwap_dist_atr", "lt", -thr),
                 Filter("f_m_imb_z", "gt", 1.0)], 3.0, 16.0, r, 30, 4)

    # 2. sweep + reclaim with book confirmation
    for r in (0.6, 0.8):
        add(f"mswp_L_r{r}", +1,
            [Filter("f_m_sweep_lo", "eq", 1.0), Filter("f_m_imb", "gt", 0.0)],
            3.0, 16.0, r, 30, 4)
        add(f"mswp_S_r{r}", -1,
            [Filter("f_m_sweep_hi", "eq", 1.0), Filter("f_m_imb", "lt", 0.0)],
            3.0, 16.0, r, 30, 4)

    # 3. failed-push fades (pressure-confirmed)
    for r in (0.6, 0.8):
        add(f"mfp_S_r{r}", -1,
            [Filter("f_m_failed_up", "eq", 1.0),
             Filter("f_m_press_z", "lt", 0.0)], 3.0, 16.0, r, 25, 4)
        add(f"mfp_L_r{r}", +1,
            [Filter("f_m_failed_dn", "eq", 1.0),
             Filter("f_m_press_z", "gt", 0.0)], 3.0, 16.0, r, 25, 4)

    # 4. absorption reversal
    for r in (0.6, 0.8):
        add(f"mabs_S_r{r}", -1,
            [Filter("f_m_absorb", "gt", 2.0), Filter("f_m_mr", "gt", 1.2)],
            3.0, 16.0, r, 25, 4)
        add(f"mabs_L_r{r}", +1,
            [Filter("f_m_absorb", "gt", 2.0), Filter("f_m_mr", "lt", -1.2)],
            3.0, 16.0, r, 25, 4)

    # 5. delta-burst micro momentum
    for r in (0.8, 1.0):
        add(f"mdb_L_r{r}", +1,
            [Filter("f_m_dburst_ratio", "gt", 2.0), Filter("f_m_mom", "gt", 1.0),
             Filter("f_m_imb", "gt", 0.1)], 3.0, 16.0, r, 20, 4)
        add(f"mdb_S_r{r}", -1,
            [Filter("f_m_dburst_ratio", "gt", 2.0), Filter("f_m_mom", "lt", -1.0),
             Filter("f_m_imb", "lt", -0.1)], 3.0, 16.0, r, 20, 4)

    # 6. pressure-shift scalp
    add("mps_L", +1, [Filter("f_m_press_z", "gt", 1.2),
                      Filter("f_m_imb_chg", "gt", 0.2)], 3.0, 16.0, 0.8, 20, 4)
    add("mps_S", -1, [Filter("f_m_press_z", "lt", -1.2),
                      Filter("f_m_imb_chg", "lt", -0.2)], 3.0, 16.0, 0.8, 20, 4)

    # 7. quiet-book mean reversion
    add("mqmr_S", -1, [Filter("f_m_qburst_ratio", "lt", 0.8),
                       Filter("f_m_mr", "gt", 1.5)], 3.0, 16.0, 0.6, 20, 4)
    add("mqmr_L", +1, [Filter("f_m_qburst_ratio", "lt", 0.8),
                       Filter("f_m_mr", "lt", -1.5)], 3.0, 16.0, 0.6, 20, 4)

    # 8. two-shot wide-stop variants of the core fades (mt=2, cap $500)
    for r in (0.6, 0.8):
        add(f"mvf2_S_r{r}", -1,
            [Filter("f_svwap_dist_atr", "gt", 2.0),
             Filter("f_m_imb_z", "lt", -1.0)], 4.0, 25.0, r, 40, 2)
        add(f"mvf2_L_r{r}", +1,
            [Filter("f_svwap_dist_atr", "lt", -2.0),
             Filter("f_m_imb_z", "gt", 1.0)], 4.0, 25.0, r, 40, 2)
    return tuple(out)


def quantile_micro_grid() -> Tuple[TradeRule, ...]:
    """Phase 5B grid: micro families gated by SELF-CALIBRATING trailing
    quantile indicators (fq_*) instead of absolute thresholds. Same DLL
    arithmetic as Phase 5A: per-trade cap = $1,000/max_trades, day stop =
    1000 - cap."""
    liq = Filter("f_spread_ratio", "lt_or_na", 1.6)
    W = (time(8, 35), time(14, 15))
    out: list = []

    def add(name, direction, filters, stop, band_hi, r_mult, ts, mt):
        cap = 1000.0 / mt
        assert stop * band_hi * 5.0 <= cap + 1e-9, name
        out.append(TradeRule(
            name=name, direction=direction,
            filters=tuple(filters + [Filter("f_atr_ticks", "between",
                                            4.0, band_hi), liq]),
            entry_start_ct=W[0], entry_end_ct=W[1], stop_atr=stop,
            target_atr=stop * r_mult, time_stop_bars=ts,
            max_trades_per_day=mt, daily_loss_usd=1000.0 - cap))

    def on(col):
        return Filter(col, "eq", 1.0)

    # 1. imbalance reversal (fade a top/bottom-decile book lean)
    for r in (0.6, 0.8):
        add(f"qimbR_S_r{r}", -1, [on("fq_m_imb_z_hi10")], 3.0, 16.0, r, 25, 4)
        add(f"qimbR_L_r{r}", +1, [on("fq_m_imb_z_lo10")], 3.0, 16.0, r, 25, 4)

    # 2. imbalance continuation (quintile lean + tape agreeing)
    for r in (0.8, 1.0):
        add(f"qimbC_L_r{r}", +1, [on("fq_m_imb_z_hi20"),
                                  Filter("f_ret_5", "gt", 0.0)],
            3.0, 16.0, r, 25, 4)
        add(f"qimbC_S_r{r}", -1, [on("fq_m_imb_z_lo20"),
                                  Filter("f_ret_5", "lt", 0.0)],
            3.0, 16.0, r, 25, 4)

    # 3. quote-burst continuation (TOD-normalized and plain)
    for tag in ("hi10_tod", "hi10"):
        add(f"qqbC_L_{tag}", +1, [on(f"fq_m_qburst_ratio_{tag}"),
                                  on("fq_m_mom_hi20")], 3.0, 16.0, 0.8, 20, 4)
        add(f"qqbC_S_{tag}", -1, [on(f"fq_m_qburst_ratio_{tag}"),
                                  on("fq_m_mom_lo20")], 3.0, 16.0, 0.8, 20, 4)

    # 4. quote-burst exhaustion at a stretched tape
    for r in (0.6, 0.8):
        add(f"qqbX_S_r{r}", -1, [on("fq_m_qburst_ratio_hi10"),
                                 on("fq_svwap_dist_atr_hi20")],
            3.0, 16.0, r, 25, 4)
        add(f"qqbX_L_r{r}", +1, [on("fq_m_qburst_ratio_hi10"),
                                 on("fq_svwap_dist_atr_lo20")],
            3.0, 16.0, r, 25, 4)

    # 5. delta-burst fade (climax against a fast stretch)
    for r in (0.6, 0.8):
        add(f"qdbF_S_r{r}", -1, [on("fq_m_dburst_ratio_hi10"),
                                 on("fq_m_mr_hi10")], 3.0, 16.0, r, 25, 4)
        add(f"qdbF_L_r{r}", +1, [on("fq_m_dburst_ratio_hi10"),
                                 on("fq_m_mr_lo10")], 3.0, 16.0, r, 25, 4)

    # 6. absorption breakout (heavy passive size, book picks direction)
    for r in (0.8, 1.0):
        add(f"qabsB_L_r{r}", +1, [on("fq_m_absorb_hi20"),
                                  Filter("f_m_imb", "gt", 0.1)],
            3.0, 16.0, r, 25, 4)
        add(f"qabsB_S_r{r}", -1, [on("fq_m_absorb_hi20"),
                                  Filter("f_m_imb", "lt", -0.1)],
            3.0, 16.0, r, 25, 4)

    # 7. failed-push reclaim (event + book pressure percentile)
    for r in (0.6, 0.8):
        add(f"qfpR_S_r{r}", -1, [Filter("f_m_failed_up", "eq", 1.0),
                                 on("fq_m_press_z_lo10")], 3.0, 16.0, r, 25, 4)
        add(f"qfpR_L_r{r}", +1, [Filter("f_m_failed_dn", "eq", 1.0),
                                 on("fq_m_press_z_hi10")], 3.0, 16.0, r, 25, 4)

    # 8. sweep/reclaim (event + book lean)
    for r in (0.6, 0.8):
        add(f"qswp_L_r{r}", +1, [Filter("f_m_sweep_lo", "eq", 1.0),
                                 Filter("f_m_imb", "gt", 0.0)],
            3.0, 16.0, r, 30, 4)
        add(f"qswp_S_r{r}", -1, [Filter("f_m_sweep_hi", "eq", 1.0),
                                 Filter("f_m_imb", "lt", 0.0)],
            3.0, 16.0, r, 30, 4)

    # 9. micro mean reversion (decile fast-z, plus vwap-stretch combo)
    for r in (0.6, 0.8):
        add(f"qmr_S_r{r}", -1, [on("fq_m_mr_hi10")], 3.0, 16.0, r, 20, 4)
        add(f"qmr_L_r{r}", +1, [on("fq_m_mr_lo10")], 3.0, 16.0, r, 20, 4)
    add("qmrV_S", -1, [on("fq_m_mr_hi20"), on("fq_svwap_dist_atr_hi20")],
        3.0, 16.0, 0.6, 20, 4)
    add("qmrV_L", +1, [on("fq_m_mr_lo20"), on("fq_svwap_dist_atr_lo20")],
        3.0, 16.0, 0.6, 20, 4)

    # 10. micro momentum (quintile mom + participation)
    for r in (0.8, 1.0):
        add(f"qmom_L_r{r}", +1, [on("fq_m_mom_hi20"), on("fq_vol_ratio_hi20"),
                                 on("fq_m_dburst_ratio_hi20")],
            3.0, 16.0, r, 20, 4)
        add(f"qmom_S_r{r}", -1, [on("fq_m_mom_lo20"), on("fq_vol_ratio_hi20"),
                                 on("fq_m_dburst_ratio_hi20")],
            3.0, 16.0, r, 20, 4)

    # 11. two-shot wide variants of the reversal core (mt=2, cap $500)
    for r in (0.6, 0.8):
        add(f"qimbR2_S_r{r}", -1, [on("fq_m_imb_z_hi10"),
                                   on("fq_svwap_dist_atr_hi20")],
            4.0, 25.0, r, 40, 2)
        add(f"qimbR2_L_r{r}", +1, [on("fq_m_imb_z_lo10"),
                                   on("fq_svwap_dist_atr_lo20")],
            4.0, 25.0, r, 40, 2)
    return tuple(out)


def exec_micro_grids() -> Tuple[Tuple[TradeRule, ...], Tuple[TradeRule, ...]]:
    """Phase 5C: the SAME quantile signal families fitted to two executable
    instruments — (nq_grid, mnq_grid).

    Risk arithmetic per instrument: worst-case risk/trade =
    stop_atr x band_hi x tick_value x qty, required <= $1,000 / max_trades;
    daily_loss_usd = 1000 - actual worst-case risk (realized + open <= DLL).
    MNQ's $0.50 tick lets stops carry realistic ATR bands (up to 60 ticks)
    at qty 2-4 (0.2-0.4 mini-equivalents: far inside the 10:1-counted
    position limits and the XFA tier-1 scaling cap).
    """
    liq = Filter("f_spread_ratio", "lt_or_na", 1.6)
    W = (time(8, 35), time(14, 15))

    def on(col):
        return Filter(col, "eq", 1.0)

    # (base name, direction, signal filters, r_mults, time stop)
    FAMS = (
        ("qimbR_S", -1, [on("fq_m_imb_z_hi10")], (0.6, 0.8), 25),
        ("qimbR_L", +1, [on("fq_m_imb_z_lo10")], (0.6, 0.8), 25),
        ("qimbRV_S", -1, [on("fq_m_imb_z_hi10"),
                          on("fq_svwap_dist_atr_hi20")], (0.6, 0.8), 40),
        ("qimbRV_L", +1, [on("fq_m_imb_z_lo10"),
                          on("fq_svwap_dist_atr_lo20")], (0.6, 0.8), 40),
        ("qimbC_L", +1, [on("fq_m_imb_z_hi20"),
                         Filter("f_ret_5", "gt", 0.0)], (0.8,), 25),
        ("qimbC_S", -1, [on("fq_m_imb_z_lo20"),
                         Filter("f_ret_5", "lt", 0.0)], (0.8,), 25),
        ("qqbC_L", +1, [on("fq_m_qburst_ratio_hi10_tod"),
                        on("fq_m_mom_hi20")], (0.8,), 20),
        ("qqbC_S", -1, [on("fq_m_qburst_ratio_hi10_tod"),
                        on("fq_m_mom_lo20")], (0.8,), 20),
        ("qqbX_S", -1, [on("fq_m_qburst_ratio_hi10"),
                        on("fq_svwap_dist_atr_hi20")], (0.6,), 25),
        ("qqbX_L", +1, [on("fq_m_qburst_ratio_hi10"),
                        on("fq_svwap_dist_atr_lo20")], (0.6,), 25),
        ("qdbF_S", -1, [on("fq_m_dburst_ratio_hi10"), on("fq_m_mr_hi10")],
         (0.6, 0.8), 25),
        ("qdbF_L", +1, [on("fq_m_dburst_ratio_hi10"), on("fq_m_mr_lo10")],
         (0.6, 0.8), 25),
        ("qabsB_L", +1, [on("fq_m_absorb_hi20"),
                         Filter("f_m_imb", "gt", 0.1)], (0.8,), 25),
        ("qabsB_S", -1, [on("fq_m_absorb_hi20"),
                         Filter("f_m_imb", "lt", -0.1)], (0.8,), 25),
        ("qfpR_S", -1, [Filter("f_m_failed_up", "eq", 1.0),
                        on("fq_m_press_z_lo10")], (0.6,), 25),
        ("qfpR_L", +1, [Filter("f_m_failed_dn", "eq", 1.0),
                        on("fq_m_press_z_hi10")], (0.6,), 25),
        ("qswp_L", +1, [Filter("f_m_sweep_lo", "eq", 1.0),
                        Filter("f_m_imb", "gt", 0.0)], (0.6, 0.8), 30),
        ("qswp_S", -1, [Filter("f_m_sweep_hi", "eq", 1.0),
                        Filter("f_m_imb", "lt", 0.0)], (0.6, 0.8), 30),
        ("qmr_S", -1, [on("fq_m_mr_hi10")], (0.6, 0.8), 20),
        ("qmr_L", +1, [on("fq_m_mr_lo10")], (0.6, 0.8), 20),
        ("qmom_L", +1, [on("fq_m_mom_hi20"), on("fq_vol_ratio_hi20"),
                        on("fq_m_dburst_ratio_hi20")], (0.8,), 20),
        ("qmom_S", -1, [on("fq_m_mom_lo20"), on("fq_vol_ratio_hi20"),
                        on("fq_m_dburst_ratio_hi20")], (0.8,), 20),
    )
    # geometry slots: (tag, stop_atr, band_hi_ticks, max_trades, qty)
    GEO = {
        "NQ": (("a", 4.0, 25.0, 2, 1),      # $500 cap, wide-ish band
               ("b", 5.0, 40.0, 1, 1)),     # $1000 cap, one shot, full band
        "MNQ": (("a", 3.0, 40.0, 4, 4),     # $240 <= $250 cap, 4 shots
                ("b", 4.0, 60.0, 2, 4)),    # $480 <= $500 cap, full band
    }

    def build(instrument: str) -> Tuple[TradeRule, ...]:
        tv = 5.0 if instrument == "NQ" else 0.5
        out = []
        for base, direction, sig, r_mults, ts in FAMS:
            for tag, stop, band_hi, mt, qty in GEO[instrument]:
                worst = stop * band_hi * tv * qty
                assert worst <= 1000.0 / mt + 1e-9, (base, instrument, tag)
                for r in r_mults:
                    out.append(TradeRule(
                        name=f"{base}_r{r}_{instrument.lower()}{tag}",
                        direction=direction,
                        filters=tuple(sig + [Filter("f_atr_ticks", "between",
                                                    4.0, band_hi), liq]),
                        entry_start_ct=W[0], entry_end_ct=W[1],
                        stop_atr=stop, target_atr=stop * r,
                        time_stop_bars=ts, max_trades_per_day=mt,
                        daily_loss_usd=1000.0 - worst, qty=qty))
        return tuple(out)

    return build("NQ"), build("MNQ")


def gc_exec_grids() -> Tuple[Tuple[TradeRule, ...], Tuple[TradeRule, ...]]:
    """GC/MGC independent grids — (gc_grid, mgc_grid). Families are
    GC-native a-priori hypotheses (level respect, NY-morning mean reversion,
    London-NY overlap timing, weak breakout follow-through as a probe), NOT
    ported NQ rules. DLL arithmetic: worst risk = stop x band_hi x
    tick_value x qty <= $1,000/max_trades; day stop = 1000 - worst.
    GC tick $10 (mini, control), MGC tick $1 (primary).
    """
    liq = Filter("f_spread_ratio", "lt_or_na", 2.0)   # NaN for GC bars: passes
    W = (time(7, 30), time(13, 0))                    # COMEX day focus
    OV = (time(7, 30), time(10, 30))                  # London-NY overlap
    LATE = (time(10, 30), time(13, 0))

    def on(col):
        return Filter(col, "eq", 1.0)

    FAMS = (
        ("gvf20_S", -1, [on("fq_svwap_dist_atr_hi20")], (0.6, 0.8, 1.0), 30, W),
        ("gvf20_L", +1, [on("fq_svwap_dist_atr_lo20")], (0.6, 0.8, 1.0), 30, W),
        ("gvf10_S", -1, [on("fq_svwap_dist_atr_hi10")], (0.6, 0.8), 30, W),
        ("gvf10_L", +1, [on("fq_svwap_dist_atr_lo10")], (0.6, 0.8), 30, W),
        ("gz30_S", -1, [on("fq_z_30_hi10")], (0.6, 0.8), 25, W),
        ("gz30_L", +1, [on("fq_z_30_lo10")], (0.6, 0.8), 25, W),
        ("gz120_S", -1, [on("fq_z_120_hi10")], (0.6, 0.8), 40, W),
        ("gz120_L", +1, [on("fq_z_120_lo10")], (0.6, 0.8), 40, W),
        ("gpdlF_L", +1, [Filter("f_prior_low_dist", "between", -0.8, 0.3),
                         Filter("f_ret_5", "gt", 0.0)], (0.6, 0.8), 40, W),
        ("gpdhF_S", -1, [Filter("f_prior_high_dist", "between", -0.3, 0.8),
                         Filter("f_ret_5", "lt", 0.0)], (0.6, 0.8), 40, W),
        ("gpdhH_L", +1, [Filter("f_prior_high_dist", "between", 0.0, 1.5),
                         on("fq_ret_15_hi20")], (1.5, 2.0), 60, W),
        ("gpdlH_S", -1, [Filter("f_prior_low_dist", "between", -1.5, 0.0),
                         on("fq_ret_15_lo20")], (1.5, 2.0), 60, W),
        ("gorb_L", +1, [Filter("f_or_break_up", "eq", 1.0),
                        on("fq_vol_ratio_hi20")], (1.5,), 60, W),
        ("gorb_S", -1, [Filter("f_or_break_dn", "eq", 1.0),
                        on("fq_vol_ratio_hi20")], (1.5,), 60, W),
        ("gmom_L", +1, [on("fq_ret_15_hi20"), on("fq_vol_ratio_hi20")],
         (0.8, 1.5), 40, W),
        ("gmom_S", -1, [on("fq_ret_15_lo20"), on("fq_vol_ratio_hi20")],
         (0.8, 1.5), 40, W),
        ("gquiet_S", -1, [on("fq_range_atr_lo20"), on("fq_z_30_hi10")],
         (0.6,), 25, LATE),
        ("gquiet_L", +1, [on("fq_range_atr_lo20"), on("fq_z_30_lo10")],
         (0.6,), 25, LATE),
        ("govf_S", -1, [on("fq_svwap_dist_atr_hi20")], (0.6, 0.8), 30, OV),
        ("govf_L", +1, [on("fq_svwap_dist_atr_lo20")], (0.6, 0.8), 30, OV),
    )
    GEO = {
        "GC": (("a", 3.0, 16.0, 2, 1),     # $480 <= $500, two shots
               ("b", 4.0, 25.0, 1, 1)),    # $1000, one shot (control reality)
        "MGC": (("a", 4.0, 25.0, 4, 2),    # $200 <= $250, four shots
                ("b", 5.0, 40.0, 2, 2),    # $400 <= $500
                ("c", 3.0, 15.0, 4, 5)),   # $225, tight-vol regime
    }

    def build(instrument: str) -> Tuple[TradeRule, ...]:
        tv = 10.0 if instrument == "GC" else 1.0
        out = []
        for base, direction, sig, r_mults, ts, win in FAMS:
            for tag, stop, band_hi, mt, qty in GEO[instrument]:
                worst = stop * band_hi * tv * qty
                assert worst <= 1000.0 / mt + 1e-9, (base, instrument, tag)
                for r in r_mults:
                    out.append(TradeRule(
                        name=f"{base}_r{r}_{instrument.lower()}{tag}",
                        direction=direction,
                        filters=tuple(sig + [Filter("f_atr_ticks", "between",
                                                    3.0, band_hi), liq]),
                        entry_start_ct=win[0], entry_end_ct=win[1],
                        stop_atr=stop, target_atr=stop * r,
                        time_stop_bars=ts, max_trades_per_day=mt,
                        daily_loss_usd=1000.0 - worst, qty=qty))
        return tuple(out)

    return build("GC"), build("MGC")


def candidate_grid() -> Tuple[TradeRule, ...]:
    """Deterministic smoke-test template grid: bias/threshold/time/vol/VWAP/
    momentum/mean-reversion/liquidity building blocks. Thresholds are fixed
    a priori (nothing is fitted to data inside a fold)."""
    liq = Filter("f_spread_ratio", "lt", 1.5)
    vol_ok = Filter("f_atr_ticks", "between", 4.0, 60.0)
    out = []

    for tgt in (1.5, 2.0):
        out.append(TradeRule(
            f"orb_long_rr{tgt}", +1,
            (Filter("f_or_break_up", "eq", 1.0), vol_ok, liq,
             Filter("f_prior_close_gap", "gt", 0.0)),      # long-bias filter
            entry_start_ct=time(9, 0), entry_end_ct=time(11, 30),
            stop_atr=1.0, target_atr=tgt))
        out.append(TradeRule(
            f"orb_short_rr{tgt}", -1,
            (Filter("f_or_break_dn", "eq", 1.0), vol_ok, liq,
             Filter("f_prior_close_gap", "lt", 0.0)),
            entry_start_ct=time(9, 0), entry_end_ct=time(11, 30),
            stop_atr=1.0, target_atr=tgt))

    for tgt in (1.5, 2.0):
        out.append(TradeRule(
            f"mom_long_rr{tgt}", +1,
            (Filter("f_ret_15", "gt", 0.0010), Filter("f_sma_dist_20", "gt", 0.5),
             Filter("f_delta_15", "gt", 0.05), vol_ok, liq),
            stop_atr=1.0, target_atr=tgt, time_stop_bars=45))
        out.append(TradeRule(
            f"mom_short_rr{tgt}", -1,
            (Filter("f_ret_15", "lt", -0.0010), Filter("f_sma_dist_20", "lt", -0.5),
             Filter("f_delta_15", "lt", -0.05), vol_ok, liq),
            stop_atr=1.0, target_atr=tgt, time_stop_bars=45))

    out.append(TradeRule(
        "mr_long_z", +1,
        (Filter("f_z_30", "lt", -2.2), Filter("f_svwap_dist_atr", "lt", -2.0),
         vol_ok, liq),
        stop_atr=1.2, target_atr=1.2, time_stop_bars=30))
    out.append(TradeRule(
        "mr_short_z", -1,
        (Filter("f_z_30", "gt", 2.2), Filter("f_svwap_dist_atr", "gt", 2.0),
         vol_ok, liq),
        stop_atr=1.2, target_atr=1.2, time_stop_bars=30))

    out.append(TradeRule(
        "vwap_revert_long", +1,
        (Filter("f_svwap_dist_atr", "between", -6.0, -2.5),
         Filter("f_delta_5", "gt", 0.0), vol_ok, liq),
        stop_atr=1.0, target_atr=1.0, time_stop_bars=30))
    out.append(TradeRule(
        "vwap_revert_short", -1,
        (Filter("f_svwap_dist_atr", "between", 2.5, 6.0),
         Filter("f_delta_5", "lt", 0.0), vol_ok, liq),
        stop_atr=1.0, target_atr=1.0, time_stop_bars=30))

    out.append(TradeRule(
        "delta_mom_long", +1,
        (Filter("f_delta_15", "gt", 0.25), Filter("f_ret_5", "gt", 0.0),
         Filter("f_vol_ratio", "gt", 1.2), vol_ok, liq),
        stop_atr=1.0, target_atr=1.5, time_stop_bars=30))
    out.append(TradeRule(
        "delta_mom_short", -1,
        (Filter("f_delta_15", "lt", -0.25), Filter("f_ret_5", "lt", 0.0),
         Filter("f_vol_ratio", "gt", 1.2), vol_ok, liq),
        stop_atr=1.0, target_atr=1.5, time_stop_bars=30))
    return tuple(out)
