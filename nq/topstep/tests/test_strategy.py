from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from tslab.audit.leakage import FoldEmbargoCheck, Status
from tslab.features.core import FeatureParams, compute_features
from tslab.features.labels import make_session_folds, required_embargo
from tslab.sim.costs import CostModel
from tslab.strategy.backtest import run_backtest
from tslab.strategy.evaluate import lifecycle_mc, simulate_account
from tslab.strategy.rules import Filter, TradeRule, candidate_grid
from tslab.strategy.search import walkforward_search
from tests.test_features import synth_bars

FP = FeatureParams()


def prepped(cfg, sessions=(date(2024, 1, 8), date(2024, 1, 9),
                           date(2024, 1, 10), date(2024, 1, 11),
                           date(2024, 1, 12)), seed=3):
    bars = synth_bars(cfg, sessions=sessions, seed=seed)
    bars["entry_ok"] = True
    feats = compute_features(bars, cfg, FP)
    return bars, feats


def always_rule(**kw):
    defaults = dict(direction=1, filters=(Filter("f_atr_ticks", "gt", 0.0),),
                    entry_start_ct=time(8, 0), entry_end_ct=time(11, 30),
                    stop_atr=1.0, target_atr=1.5, time_stop_bars=10,
                    max_trades_per_day=100, daily_loss_usd=1e12)
    defaults.update(kw)
    return TradeRule(name=kw.get("name", "always"), **{
        k: v for k, v in defaults.items() if k != "name"})


# --------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------- #

def test_rule_grid_and_backtest_deterministic(cfg):
    g1, g2 = candidate_grid(), candidate_grid()
    assert [c.name for c in g1] == [c.name for c in g2]
    assert g1 == g2
    bars, feats = prepped(cfg)
    r = always_rule()
    a = run_backtest(r, bars, feats, cfg)
    b = run_backtest(r, bars, feats, cfg)
    assert len(a.trades) == len(b.trades) > 0
    for x, y in zip(a.trades, b.trades):
        assert x == y


# --------------------------------------------------------------------- #
# flatten / cutoff discipline
# --------------------------------------------------------------------- #

def test_no_trades_after_cutoff(cfg):
    bars, feats = prepped(cfg)
    # permissive rule with a huge time stop: positions would run forever
    # unless the engine forces the flatten exit
    r = always_rule(entry_end_ct=time(14, 59), time_stop_bars=10_000,
                    stop_atr=50.0, target_atr=50.0)
    res = run_backtest(r, bars, feats, cfg)
    assert res.violations == 0
    assert len(res.trades) > 0
    from tslab import sessions as S
    for t in res.trades:
        flat = S.eod_flatten_utc(t.session, cfg.sessions)
        assert t.exit_time + pd.Timedelta(seconds=60) <= flat  # closes before
        assert t.entry_time < flat


def test_entry_window_respected(cfg):
    bars, feats = prepped(cfg)
    r = always_rule(entry_start_ct=time(9, 0), entry_end_ct=time(9, 30))
    res = run_backtest(r, bars, feats, cfg)
    ct = [t.entry_time.tz_convert(cfg.sessions.timezone).time()
          for t in res.trades]
    # entry bar opens exactly AT the decision info_time => [09:00, 09:30]
    assert ct
    assert all(time(9, 0) <= x <= time(9, 30) for x in ct)


# --------------------------------------------------------------------- #
# cost application
# --------------------------------------------------------------------- #

def test_costs_applied_per_exit_type(cfg):
    bars, feats = prepped(cfg)
    cm = CostModel(cfg)
    r = always_rule(max_trades_per_day=2)
    res = run_backtest(r, bars, feats, cfg, cm)
    entry = cm.side_cost("NQ", 1, "market").total
    by_reason = {"stop": cm.side_cost("NQ", 1, "stop").total,
                 "target": cm.side_cost("NQ", 1, "limit").total,
                 "time": cm.side_cost("NQ", 1, "market").total,
                 "flatten": cm.side_cost("NQ", 1, "market").total}
    assert res.trades
    for t in res.trades:
        assert t.costs_usd == pytest.approx(entry + by_reason[t.exit_reason])
        assert t.net_usd == pytest.approx(t.gross_usd - t.costs_usd)
        pv = cfg.instruments["NQ"].point_value
        assert t.gross_usd == pytest.approx(
            t.direction * (t.exit_px - t.entry_px) * pv)


# --------------------------------------------------------------------- #
# daily guards
# --------------------------------------------------------------------- #

def test_max_trades_per_day(cfg):
    bars, feats = prepped(cfg)
    r = always_rule(max_trades_per_day=2)
    res = run_backtest(r, bars, feats, cfg)
    per_day = pd.Series([t.session for t in res.trades]).value_counts()
    assert (per_day <= 2).all()


def test_daily_loss_stop_halts_trading(cfg):
    bars, feats = prepped(cfg)
    r_free = always_rule(name="free")
    r_capped = always_rule(name="capped", daily_loss_usd=100.0)
    free = run_backtest(r_free, bars, feats, cfg)
    capped = run_backtest(r_capped, bars, feats, cfg)
    assert len(capped.trades) < len(free.trades)
    for sess, trades in capped.by_session().items():
        run = 0.0
        for k, t in enumerate(trades):
            if k > 0:
                assert run > -100.0     # would have stopped otherwise
            run += t.net_usd


# --------------------------------------------------------------------- #
# fold separation / no train-test leakage
# --------------------------------------------------------------------- #

def test_fold_separation_and_embargo(cfg):
    sessions = [date(2024, 1, d) for d in (8, 9, 10, 11, 12, 16, 17, 18, 19)]
    folds = make_session_folds(sessions, cfg, n_train=5, n_test=2,
                               embargo_sessions=1)
    assert FoldEmbargoCheck(folds, required_embargo([])).run().status \
        == Status.PASS


def test_search_train_test_disjoint_and_oos_only_from_test(cfg):
    sessions = tuple(date(2024, 1, d) for d in (8, 9, 10, 11, 12, 16, 17, 18, 19))
    bars, feats = prepped(cfg, sessions=sessions)
    res = walkforward_search([always_rule()], bars, feats, cfg,
                             n_train=5, n_test=2, embargo_sessions=1,
                             mc_runs=20, mc_horizon=40)
    for fr in res.fold_report:
        tr, te = set(fr["train_sessions"]), set(fr["test_sessions"])
        assert not tr & te
        # embargo session excluded from both sides
        assert len(tr) + len(te) < len(sessions)
    # each fold produced train AND test numbers for the candidate
    assert res.scores[0].train_net_by_fold
    assert len(res.scores[0].train_net_by_fold) == len(res.fold_report)


# --------------------------------------------------------------------- #
# simulator integration
# --------------------------------------------------------------------- #

def make_trade(cfg, session, net, mae=0.0, risk=250.0):
    from tslab.strategy.backtest import Trade
    t0 = pd.Timestamp(session, tz="UTC") + pd.Timedelta(hours=15)
    return Trade(session=session, entry_time=t0,
                 exit_time=t0 + pd.Timedelta(minutes=5), direction=1, qty=1,
                 entry_px=17000.0, exit_px=17000.0 + net / 20.0,
                 exit_reason="target", gross_usd=net, costs_usd=0.0,
                 net_usd=net, mae_usd=mae, risk_usd=risk)


def test_simulator_integration_dll_lockout(cfg):
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    trades = {
        d1: [make_trade(cfg, d1, -600.0, mae=650.0),
             make_trade(cfg, d1, -600.0, mae=650.0),     # breaches DLL 1000
             make_trade(cfg, d1, +900.0)],               # must never execute
        d2: [make_trade(cfg, d2, +400.0)],
    }
    run = simulate_account(trades, cfg, personal_dll=1000)
    assert run.dll_hits == 1
    assert not run.phase.startswith("FAILED")
    # day 1 stopped after the DLL mark; day 2 traded normally
    assert any(k == "DLL_HIT" for k, _ in run.events)
    assert run.net_cashflow < 0                          # only fees so far


def test_simulator_integration_mll_failure_terminal(cfg):
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    trades = {
        d1: [make_trade(cfg, d1, -1800.0, mae=2100.0)],  # deep MAE
        d2: [make_trade(cfg, d2, +400.0)],
    }
    run = simulate_account(trades, cfg, personal_dll=1000)
    # mark(-2100) breaches the 48,000 floor intraday -> terminal, day 2 never runs
    assert run.phase == "FAILED_COMBINE"
    assert run.sessions_used == 1


def test_simulator_integration_pass_and_payout(cfg):
    # 8 sessions of +$400 pass the combine (3,200 >= 3,000, consistency OK),
    # then 5 winning days in the XFA unlock the first (greedy) payout
    sessions = [d.date() for d in pd.bdate_range("2026-01-05", periods=14)
                if d.date() != date(2026, 1, 19)][:13]   # skip MLK 2026
    trades = {s: [make_trade(cfg, s, +400.0, mae=100.0)] for s in sessions}
    run = simulate_account(trades, cfg)
    assert run.passed_combine
    assert run.payouts_net > 0                           # greedy payout fired
    assert all(k != "EOD_FLATTEN_VIOLATION" for k, _ in run.events)


def test_lifecycle_mc_smoke(cfg):
    pool = [[make_trade(cfg, date(2026, 1, 5), +400.0, mae=150.0)],
            [make_trade(cfg, date(2026, 1, 5), -350.0, mae=400.0)],
            [make_trade(cfg, date(2026, 1, 5), +150.0, mae=50.0)]]
    stats = lifecycle_mc(pool, cfg, n_runs=30, horizon_sessions=60, seed=7)
    assert stats.runs == 30
    assert 0.0 <= stats.pass_prob <= 1.0
    assert 0.0 <= stats.dll_hit_rate <= 1.0
    # deterministic under the same seed
    again = lifecycle_mc(pool, cfg, n_runs=30, horizon_sessions=60, seed=7)
    assert again.ev_net_payout == stats.ev_net_payout
