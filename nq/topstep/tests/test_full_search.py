from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tslab.features.core import FeatureParams, compute_features
from tslab.strategy.evaluate import lifecycle_mc, max_drawdown
from tslab.strategy.full_search import (FoldCandidate, select_winner,
                                        train_tstat)
from tslab.strategy.rules import Filter, full_candidate_grid
from tests.test_features import synth_bars
from tests.test_strategy import make_trade


def test_na_tolerant_filter_ops(cfg):
    f = pd.DataFrame({"f_x": [1.0, 2.0, np.nan]})
    assert Filter("f_x", "lt_or_na", 1.5).mask(f).tolist() == [True, False, True]
    assert Filter("f_x", "gt_or_na", 1.5).mask(f).tolist() == [False, True, True]
    assert Filter("f_x", "lt", 1.5).mask(f).tolist() == [True, False, False]


def test_full_grid_deterministic_and_features_exist(cfg):
    g1, g2 = full_candidate_grid(), full_candidate_grid()
    assert g1 == g2
    assert len(g1) == len({r.name for r in g1})          # unique names
    assert len(g1) >= 90
    feats = compute_features(synth_bars(cfg), cfg, FeatureParams())
    for rule in g1:
        for f in rule.filters:
            assert f.feature in feats.columns, (rule.name, f.feature)
            f.mask(feats)                                # ops all valid
        assert rule.stop_atr > 0 and rule.target_atr > 0
        assert rule.daily_loss_usd <= 750                # under personal DLL


def test_lowfreq_grid_risk_capped_and_valid(cfg):
    from tslab.strategy.rules import lowfreq_candidate_grid
    g = lowfreq_candidate_grid()
    assert g == lowfreq_candidate_grid()                 # deterministic
    assert len(g) == len({r.name for r in g}) >= 50
    feats = compute_features(synth_bars(cfg), cfg, FeatureParams())
    tick_value = cfg.instruments["NQ"].tick_value
    for rule in g:
        assert 1 <= rule.max_trades_per_day <= 5
        assert rule.target_atr / rule.stop_atr >= 1.5    # larger targets
        assert rule.time_stop_bars >= 90                 # 1.5h+ holds
        bands = [f for f in rule.filters
                 if f.feature == "f_atr_ticks" and f.op == "between"]
        assert len(bands) == 1, rule.name                # vol band mandatory
        worst_risk = rule.stop_atr * bands[0].hi * tick_value * rule.qty
        assert worst_risk <= 750.0, (rule.name, worst_risk)   # under PDLL
        for f in rule.filters:
            assert f.feature in feats.columns, (rule.name, f.feature)
            f.mask(feats)


def test_hf_grid_constraints(cfg):
    from tslab.strategy.rules import hf_candidate_grid
    g = hf_candidate_grid()
    assert g == hf_candidate_grid()
    assert len(g) == len({r.name for r in g}) >= 70
    feats = compute_features(synth_bars(cfg), cfg, FeatureParams())
    tick_value = cfg.instruments["NQ"].tick_value
    n_subr = 0
    for rule in g:
        assert 1 <= rule.max_trades_per_day <= 8
        assert rule.daily_loss_usd <= 1000.0             # DLL budget
        bands = [f for f in rule.filters
                 if f.feature == "f_atr_ticks" and f.op == "between"]
        assert len(bands) == 1
        assert rule.stop_atr * bands[0].hi * tick_value <= 500.0
        n_subr += rule.target_atr < rule.stop_atr
        for f in rule.filters:
            assert f.feature in feats.columns, (rule.name, f.feature)
            f.mask(feats)
    assert n_subr >= len(g) // 2       # win rate structural: mostly sub-1 RR


def test_chained_process_and_monthly_stats(cfg):
    from datetime import date as _d, timedelta as _td
    from tslab.strategy.evaluate import lifecycle_mc_chain, simulate_process
    # a stream engineered to blow the combine once, then recover
    days = []
    d = _d(2026, 1, 5)
    seq = [-900.0, -900.0, -400.0] + [400.0] * 60      # MLL fails on day 3
    for pnl in seq:
        while d.weekday() >= 5:
            d += _td(days=1)
        days.append((d, [make_trade(cfg, d, pnl, mae=max(0.0, -pnl) + 50)]))
        d += _td(days=1)
    out = simulate_process(days, cfg)
    assert out["combine_fails"] >= 1                   # reset happened
    assert out["attempts"] >= 2
    assert out["passed_any"]                           # recovered and passed
    assert out["payouts"] >= 1
    pool = [[make_trade(cfg, _d(2026, 1, 5), 400.0, mae=100.0)],
            [make_trade(cfg, _d(2026, 1, 5), -380.0, mae=420.0)]]
    st = lifecycle_mc_chain(pool, cfg, n_runs=20, horizon_sessions=63, seed=5)
    assert st.runs == 20
    assert 0 <= st.p_payout3 <= st.p_payout2 <= st.p_payout1 <= 1
    assert st.ev_month * 3 == pytest.approx(st.ev_year, rel=1e-6)
    again = lifecycle_mc_chain(pool, cfg, n_runs=20, horizon_sessions=63,
                               seed=5)
    assert again.ev_year == st.ev_year                 # deterministic


def test_candidate_summary_winrate_fields(cfg):
    from datetime import date as _d
    from tslab.strategy.full_search import run_search
    from tests.test_strategy import always_rule, prepped
    sessions = tuple(_d(2024, 1, d) for d in (8, 9, 10, 11, 12, 16, 17, 18, 19))
    bars, feats = prepped(cfg, sessions=sessions)
    folds = [{
        "train_start": pd.Timestamp(sessions[0], tz="UTC"),
        "train_end": pd.Timestamp(sessions[4], tz="UTC") + pd.Timedelta(hours=22),
        "test_start": pd.Timestamp(sessions[5], tz="UTC"),
        "test_end": pd.Timestamp(sessions[-1], tz="UTC") + pd.Timedelta(days=1),
    }]
    out = run_search([always_rule()], bars, feats, folds, cfg,
                     progress=lambda m: None, min_trades=1)
    row = out.candidate_summary.iloc[0]
    assert 0.0 <= row["oos_win_rate"] <= 1.0
    assert row["trades_per_session"] > 0
    assert row["avg_win"] >= 0.0 and row["avg_loss"] <= 0.0


def test_tail_split_embargo_and_disjoint():
    from datetime import date as _d, timedelta as _td
    from tslab.strategy.full_search import tail_split
    sessions = [_d(2025, 1, 6) + _td(days=k) for k in range(500)
                if (_d(2025, 1, 6) + _td(days=k)).weekday() < 5]
    tail_start = sessions[-40]
    train, tail = tail_split(sessions, tail_start, n_train=250, embargo=1)
    assert len(train) == 250
    assert tail[0] == tail_start and not set(train) & set(tail)
    # exactly one embargo session between train end and tail start
    gap = [s for s in sessions if train[-1] < s < tail[0]]
    assert len(gap) == 1


def test_tail_candidates_frozen_as_reported(cfg):
    """The 5 pre-registered candidates must exist with the exact Phase 4B-1
    parameters (any drift would invalidate the one-shot confirmation)."""
    from datetime import time
    from tslab.strategy.rules import pm_candidate_grid
    rules = {r.name: r for r in pm_candidate_grid()}
    r = rules["pmT2_L_r2.0"]
    assert (r.direction, r.stop_atr, r.target_atr) == (1, 6.0, 12.0)
    assert (r.entry_start_ct, r.entry_end_ct) == (time(12, 30), time(14, 30))
    assert r.max_trades_per_day == 2 and r.time_stop_bars == 120
    fs = {(f.feature, f.op, f.lo, f.hi) for f in r.filters}
    assert ("f_sma_dist_60", "gt", 1.0, None) in fs
    assert ("f_svwap_dist_atr", "gt", 0.5, None) in fs
    assert ("f_atr_ticks", "between", 8.0, 22.0) in fs
    r = rules["orPM_L_r2.0"]
    assert (r.direction, r.stop_atr, r.target_atr) == (1, 6.0, 12.0)
    assert ("f_or_break_up", "eq", 1.0, None) in \
        {(f.feature, f.op, f.lo, f.hi) for f in r.filters}
    for name in ("pmT2_L_r3.0", "orPM_L_r3.0", "pmT_L_w2_r1.5"):
        assert name in rules


def test_pm_grid_region_and_risk_constraints(cfg):
    from datetime import time
    from tslab.strategy.rules import pm_candidate_grid
    g = pm_candidate_grid()
    assert g == pm_candidate_grid()
    assert len(g) == len({r.name for r in g}) >= 60
    feats = compute_features(synth_bars(cfg), cfg, FeatureParams())
    tick_value = cfg.instruments["NQ"].tick_value
    for rule in g:
        assert time(10, 45) <= rule.entry_start_ct        # region only
        assert rule.entry_end_ct <= time(14, 30)
        assert 1 <= rule.max_trades_per_day <= 3
        assert rule.target_atr / rule.stop_atr >= 1.5
        bands = [f for f in rule.filters
                 if f.feature == "f_atr_ticks" and f.op == "between"]
        assert len(bands) == 1, rule.name
        assert rule.stop_atr * bands[0].hi * tick_value <= 750.0
        for f in rule.filters:
            assert f.feature in feats.columns, (rule.name, f.feature)
            f.mask(feats)


def test_run_search_min_trades_passthrough(cfg):
    # with an absurd min_trades no candidate qualifies -> all folds flat
    from datetime import date as _d
    from tslab.strategy.full_search import run_search
    from tests.test_strategy import always_rule, prepped
    sessions = tuple(_d(2024, 1, d) for d in (8, 9, 10, 11, 12, 16, 17, 18, 19))
    bars, feats = prepped(cfg, sessions=sessions)
    folds = [{
        "train_start": pd.Timestamp(sessions[0], tz="UTC"),
        "train_end": pd.Timestamp(sessions[4], tz="UTC") + pd.Timedelta(hours=22),
        "test_start": pd.Timestamp(sessions[5], tz="UTC"),
        "test_end": pd.Timestamp(sessions[-1], tz="UTC") + pd.Timedelta(days=1),
    }]
    out = run_search([always_rule()], bars, feats, folds, cfg,
                     progress=lambda m: None, min_trades=10_000)
    assert out.fold_table[0]["winner"] is None
    assert all(len(v) == 0 for v in out.wf_trades_by_session.values())


def test_train_only_selection():
    # A is better on TRAIN, B is better on TEST: selection must pick A
    res = {
        "A": FoldCandidate("A", train_net=5000, train_trades=100,
                           train_tstat=3.0, test_net=-500, test_trades=30),
        "B": FoldCandidate("B", train_net=4000, train_trades=100,
                           train_tstat=2.0, test_net=+900, test_trades=30),
        "thin": FoldCandidate("thin", train_net=9000, train_trades=10,
                              train_tstat=9.0, test_net=0, test_trades=0),
        "loser": FoldCandidate("loser", train_net=-100, train_trades=200,
                               train_tstat=1.5, test_net=0, test_trades=0),
    }
    assert select_winner(res) == "A"       # not B (test), not thin (<60 trades)
    assert select_winner({"loser": res["loser"]}) is None


def test_train_tstat_definition():
    d = pd.Series([100.0, -50.0, 200.0])
    t = train_tstat(d, 10)                 # 7 flat sessions count as zeros
    mu = 250.0 / 10
    var = (100**2 + 50**2 + 200**2) / 10 - mu**2
    assert t == pytest.approx(mu / np.sqrt(var / 10))
    assert train_tstat(pd.Series(dtype=float), 10) == float("-inf")


def test_lifecycle_stats_new_fields(cfg):
    pool = [[make_trade(cfg, date(2026, 1, 5), +400.0, mae=150.0)],
            [make_trade(cfg, date(2026, 1, 5), -350.0, mae=400.0)]]
    st = lifecycle_mc(pool, cfg, n_runs=25, horizon_sessions=60, seed=3)
    assert 0.0 <= st.payout_prob <= 1.0
    assert st.mean_payouts >= 0.0
    assert abs(st.combine_fail_rate + st.funded_fail_rate
               - st.fail_rate) < 1e-9


def test_max_drawdown():
    assert max_drawdown(pd.Series([100.0, -300.0, 100.0])) == -300.0
    assert max_drawdown(pd.Series([100.0, 100.0])) == 0.0
    assert max_drawdown(pd.Series(dtype=float)) == 0.0
