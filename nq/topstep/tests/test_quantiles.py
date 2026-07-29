from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tslab.audit.leakage import Status, TruncationInvarianceCheck
from tslab.features.core import FeatureParams
from tslab.features.quantiles import (compute_micro_q_features,
                                      trailing_session_quantile,
                                      trailing_tod_quantile)
from tests.test_micro import synth_micro_bars


def toy_series():
    sessions, values = [], []
    for k in range(25):
        s = date(2024, 3, 3) + pd.Timedelta(days=k).to_pytimedelta()
        for v in (k, k + 0.5, k + 1.0):        # 3 bars per session
            sessions.append(s)
            values.append(float(v))
    idx = pd.RangeIndex(len(values))
    return pd.Series(values, index=idx), pd.Series(sessions, index=idx)


def test_trailing_session_quantile_causal_and_correct():
    v, s = toy_series()
    thr = trailing_session_quantile(v, s, q=0.90, n_sessions=20)
    sess_list = list(dict.fromkeys(s))
    # first 20 sessions have no full prior window
    assert thr[s.isin(sess_list[:20])].isna().all()
    # session 21's threshold = q90 of sessions 0..19's values exactly
    s21 = sess_list[20]
    expect = np.quantile(v[s.isin(sess_list[:20])].to_numpy(), 0.90)
    assert thr[s == s21].iloc[0] == pytest.approx(expect)
    # CAUSALITY: perturbing session 21+ values must not change s21 threshold
    v2 = v.copy()
    v2[s.isin(sess_list[20:])] += 1e6
    thr2 = trailing_session_quantile(v2, s, q=0.90, n_sessions=20)
    assert thr2[s == s21].iloc[0] == pytest.approx(expect)
    # and a LATER session whose window is >=20% perturbed must move
    # (q90 is correctly robust to the ~10% contamination at session 22)
    s25 = sess_list[24]
    assert thr2[s == s25].iloc[0] != thr[s == s25].iloc[0]


def test_trailing_tod_quantile_buckets():
    v, s = toy_series()
    hours = pd.Series([9, 10, 9] * 25, index=v.index)   # bar hours
    thr = trailing_tod_quantile(v, s, hours, q=0.90, n_sessions=20)
    sess_list = list(dict.fromkeys(s))
    s21 = sess_list[20]
    m9 = (s.isin(sess_list[:20])) & (hours == 9)
    expect9 = np.quantile(v[m9].to_numpy(), 0.90)
    got = thr[(s == s21) & (hours == 9)]
    assert np.allclose(got, expect9)
    got10 = thr[(s == s21) & (hours == 10)].iloc[0]
    assert got10 != pytest.approx(expect9)              # bucket-specific


def test_quantile_indicators_binary_and_warmup(cfg):
    bars = synth_micro_bars(cfg)                        # 3 sessions only
    feats = compute_micro_q_features(bars, cfg, FeatureParams())
    qcols = [c for c in feats.columns if c.startswith("fq_")]
    assert len(qcols) == 28
    # with only 3 sessions the 20-session warm-up is never satisfied
    for c in qcols:
        assert feats[c].isna().all()


def test_quantile_composition_truncation(cfg):
    # 25 short synthetic sessions so indicators activate after warm-up
    # (skip configured holidays — synth_bars rightly refuses them)
    hs = set(cfg.sessions.holidays)
    sessions = tuple(d.date() for d in pd.bdate_range("2024-01-02", periods=30)
                     if d.date() not in hs)[:25]
    from tests.test_features import synth_bars
    bars = synth_bars(cfg, sessions=sessions, seed=7)
    rng = np.random.default_rng(7)
    n = len(bars)
    for col, lo, hi in (("imb_first", -0.8, 0.8), ("imb_last", -0.8, 0.8),
                        ("imb_mean", -0.8, 0.8), ("sz_bid_mean", 2, 30),
                        ("sz_ask_mean", 2, 30), ("book_flips", 0, 40),
                        ("q_burst", 5, 400), ("wide_share", 0, 0.3),
                        ("d_burst", 1, 300)):
        bars[col] = rng.uniform(lo, hi, n)
    feats = compute_micro_q_features(bars, cfg, FeatureParams())
    active = [c for c in feats.columns if c.startswith("fq_")
              and feats[c].notna().any()]
    assert len(active) >= 20                            # battery live
    res = TruncationInvarianceCheck(
        bars, lambda df: compute_micro_q_features(df, cfg, FeatureParams()),
        n_probes=4).run()
    assert res.status == Status.PASS, res.message


def test_exec_micro_grids_dual_instrument(cfg):
    from tslab.strategy.rules import exec_micro_grids
    g_nq, g_mnq = exec_micro_grids()
    assert (g_nq, g_mnq) == exec_micro_grids()          # deterministic
    feats = compute_micro_q_features(synth_micro_bars(cfg), cfg,
                                     FeatureParams())
    for grid, tv, max_qty in ((g_nq, 5.0, 1), (g_mnq, 0.5, 4)):
        assert len(grid) == len({r.name for r in grid}) >= 30
        for rule in grid:
            band = next(f for f in rule.filters
                        if f.feature == "f_atr_ticks" and f.op == "between")
            worst = rule.stop_atr * band.hi * tv * rule.qty
            assert worst <= 1000.0 / rule.max_trades_per_day + 1e-9
            assert rule.daily_loss_usd == pytest.approx(1000.0 - worst)
            assert rule.qty <= max_qty
            for f in rule.filters:
                assert f.feature in feats.columns, (rule.name, f.feature)
    # 10:1 counting: max MNQ position = 4 micros = 0.4 mini-equivalents,
    # far under the 50k combine limit (5 minis / 50 micros)
    assert max(r.qty for r in g_mnq) / 10 <= cfg.accounts["50k"].max_contracts_minis
    assert max(r.qty for r in g_mnq) <= cfg.accounts["50k"].max_contracts_micros
    # MNQ geometry admits realistic volatility: bands reach 60 ticks
    assert max(next(f.hi for f in r.filters if f.feature == "f_atr_ticks")
               for r in g_mnq) >= 60.0


def test_instrument_passthrough_costs_differ(cfg):
    from tslab.strategy.backtest import run_backtest
    from tslab.strategy.rules import Filter, TradeRule
    from datetime import time as dtime
    bars = synth_micro_bars(cfg)
    bars["entry_ok"] = True
    feats = compute_micro_q_features(bars, cfg, FeatureParams())
    rule = TradeRule(name="x", direction=1,
                     filters=(Filter("f_atr_ticks", "gt", 0.0),),
                     entry_start_ct=dtime(8, 0), entry_end_ct=dtime(11, 0),
                     stop_atr=3.0, target_atr=2.0, time_stop_bars=10,
                     max_trades_per_day=3, daily_loss_usd=1e12, qty=4)
    bt_nq = run_backtest(rule, bars, feats, cfg, instrument="NQ")
    bt_mnq = run_backtest(rule, bars, feats, cfg, instrument="MNQ")
    assert len(bt_nq.trades) == len(bt_mnq.trades) > 0   # same signals/exits
    a, b = bt_nq.trades[0], bt_mnq.trades[0]
    assert a.entry_px == b.entry_px and a.exit_px == b.exit_px
    assert b.gross_usd == pytest.approx(a.gross_usd / 10)  # $2 vs $20 / point
    assert b.costs_usd < a.costs_usd                       # micro fee schedule
    assert b.risk_usd == pytest.approx(a.risk_usd / 10)


def test_gc_exec_grids_arithmetic(cfg):
    from tslab.strategy.rules import gc_exec_grids
    g_gc, g_mgc = gc_exec_grids()
    assert (g_gc, g_mgc) == gc_exec_grids()
    for grid, tv, qty_max, mini_cap in ((g_gc, 10.0, 1, 3), (g_mgc, 1.0, 5, 30)):
        assert len(grid) == len({r.name for r in grid}) >= 30
        for rule in grid:
            band = next(f for f in rule.filters
                        if f.feature == "f_atr_ticks" and f.op == "between")
            worst = rule.stop_atr * band.hi * tv * rule.qty
            assert worst <= 1000.0 / rule.max_trades_per_day + 1e-9
            assert rule.daily_loss_usd == pytest.approx(1000.0 - worst)
            assert rule.qty <= qty_max <= mini_cap  # under volatility caps


def test_gc_q_features_battery(cfg):
    from datetime import date as _d
    from tslab.features.quantiles import compute_gc_q_features
    from tests.test_features import synth_bars
    hs = set(cfg.sessions.holidays)
    sessions = tuple(d.date() for d in pd.bdate_range("2024-01-02", periods=30)
                     if d.date() not in hs)[:25]
    bars = synth_bars(cfg, sessions=sessions, seed=13, with_micro=False)
    feats = compute_gc_q_features(bars, cfg)
    qcols = [c for c in feats.columns if c.startswith("fq_")]
    assert len(qcols) == 14
    assert any(feats[c].notna().any() for c in qcols)   # live after warm-up
    res = TruncationInvarianceCheck(
        bars, lambda df: compute_gc_q_features(df, cfg), n_probes=3).run()
    assert res.status == Status.PASS, res.message


def test_quantile_grid_valid(cfg):
    from tslab.strategy.rules import quantile_micro_grid
    g = quantile_micro_grid()
    assert g == quantile_micro_grid()
    assert len(g) == len({r.name for r in g}) >= 40
    bars = synth_micro_bars(cfg)
    feats = compute_micro_q_features(bars, cfg, FeatureParams())
    for rule in g:
        cap = 1000.0 / rule.max_trades_per_day
        band = next(f for f in rule.filters
                    if f.feature == "f_atr_ticks" and f.op == "between")
        assert rule.stop_atr * band.hi * 5.0 <= cap + 1e-9
        assert rule.daily_loss_usd == pytest.approx(1000.0 - cap)
        for f in rule.filters:
            assert f.feature in feats.columns, (rule.name, f.feature)
            f.mask(feats)
