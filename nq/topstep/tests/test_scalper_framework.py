"""Tests for the patched scalper framework gating/report rules
(tslab.strategy.scalper_report, used by scripts/run_scalper_wf.py).

Proves:
  1. rejected (qualifies=False) policies are never aggregated,
  2. RR / breakeven / expectancy are computed correctly,
  3. high-win negative-RR policies are flagged invalid,
  4. a shuffled-label high-win result can never be marked viable.
"""
from __future__ import annotations

import numpy as np
import pytest

from tslab.strategy.scalper_report import (CONTROL_WARNING,
                                           HIGH_WIN_WARNING,
                                           REJECTED_NOTE,
                                           aggregate_folds,
                                           assess_viability,
                                           breakeven_win_rate,
                                           effective_rr, expectancy,
                                           gate_fold_policy,
                                           is_control_model,
                                           policy_econ, realized_rr,
                                           write_scalper_md)

NQ_FEE_RT = 2 * 1.90 / 5.0                      # 0.76 ticks round trip


def make_pol(qualifies=True, T=2, S=12):
    return dict(side="S", T=T, S=S, hold=5, cancel=1.0, flip=False,
                q=0.995, thr=0.55, train_ev_ticks=0.5 if qualifies
                else -3.5, qualifies=qualifies)


def make_stats(trades=300, n_wins=210, avg_win=8.85, avg_loss=-67.49,
               attempts=380, killed=1):
    """Synthetic simulate() output consistent with its identities."""
    n_loss = trades - n_wins
    net = n_wins * avg_win + n_loss * avg_loss
    costs = trades * 3.80
    return dict(trades=trades, attempts=attempts,
                fill_rate=trades / attempts, tps=4.7, net=net,
                gross_usd=net + costs, costs_usd=costs, n_wins=n_wins,
                ev_trade=net / trades,
                ev_day=net / 63, win=n_wins / trades, avg_win=avg_win,
                avg_loss=avg_loss, adv_1s=-1.5, max_dd=-500.0,
                miss_fmid_5s=4.0, killed=killed, exit_mix={})


# ------------------------------------------------------------------ 1
def test_rejected_policy_is_not_traded():
    traded, reason = gate_fold_policy(make_pol(qualifies=False))
    assert traded is False and "rejected" in reason
    traded, reason = gate_fold_policy(None)
    assert traded is False and reason == "no_policy_found"
    traded, reason = gate_fold_policy(make_pol(qualifies=True))
    assert traded is True


def test_only_literal_true_qualifies_is_traded():
    # non-canonical encodings (lossy round-trips, schema drift) must
    # never be traded nor aggregated
    for bad in (None, 0, 1, "True", "False", np.True_):
        pol = make_pol()
        pol["qualifies"] = bad
        traded, _ = gate_fold_policy(pol)
        assert traded is False, f"qualifies={bad!r} was traded"
        with pytest.raises(ValueError, match="rejected policy"):
            aggregate_folds([make_stats()], [pol], n_sess_total=63,
                            n_folds_total=1, fee_rt_ticks=NQ_FEE_RT)
    pol = make_pol()
    del pol["qualifies"]
    traded, reason = gate_fold_policy(pol)
    assert traded is False and reason == \
        "qualifies_flag_missing_or_invalid"
    with pytest.raises(ValueError, match="rejected policy"):
        aggregate_folds([make_stats()], [pol], n_sess_total=63,
                        n_folds_total=1, fee_rt_ticks=NQ_FEE_RT)


def test_aggregate_input_validation():
    with pytest.raises(ValueError, match="aligned"):
        aggregate_folds([make_stats(), make_stats()], [make_pol()],
                        n_sess_total=63, n_folds_total=2,
                        fee_rt_ticks=NQ_FEE_RT)
    with pytest.raises(ValueError, match="aligned"):
        aggregate_folds([], [make_pol()], n_sess_total=63,
                        n_folds_total=1, fee_rt_ticks=NQ_FEE_RT)
    with pytest.raises(ValueError, match="n_folds_total"):
        aggregate_folds([make_stats(), make_stats()],
                        [make_pol(), make_pol()], n_sess_total=63,
                        n_folds_total=1, fee_rt_ticks=NQ_FEE_RT)


def test_rejected_policies_never_aggregated():
    # headline aggregation refuses rejected policies outright
    with pytest.raises(ValueError, match="rejected policy"):
        aggregate_folds([make_stats()], [make_pol(qualifies=False)],
                        n_sess_total=63, n_folds_total=1,
                        fee_rt_ticks=NQ_FEE_RT)
    # mixed folds: gate first (as run_scalper_wf.py does), aggregate
    # only what passed; rejected folds contribute zero trades
    folds = [(make_pol(qualifies=True), make_stats(trades=300)),
             (make_pol(qualifies=False), make_stats(trades=300)),
             (make_pol(qualifies=True), make_stats(trades=200,
                                                   n_wins=140))]
    stats = [s for p, s in folds if gate_fold_policy(p)[0]]
    pols = [p for p, s in folds if gate_fold_policy(p)[0]]
    agg = aggregate_folds(stats, pols, n_sess_total=189,
                          n_folds_total=len(folds),
                          fee_rt_ticks=NQ_FEE_RT)
    assert agg["n_folds"] == 2
    assert agg["n_folds_rejected"] == 1
    assert agg["trades"] == 500                  # 300 + 200, not 800
    # all folds rejected -> headline is explicitly empty, not pooled
    empty = aggregate_folds([], [], n_sess_total=189, n_folds_total=3,
                            fee_rt_ticks=NQ_FEE_RT)
    assert empty["trades"] == 0
    assert empty["viable"] is False
    assert empty["n_folds_rejected"] == 3
    assert empty["note"] == REJECTED_NOTE


# ------------------------------------------------------------------ 2
def test_rr_computed_correctly():
    # geometry RR: (T - fee) / (S + slip + fee), NQ rf diagnosis values
    assert effective_rr(2, 12, NQ_FEE_RT, 1.0) == pytest.approx(
        (2 - 0.76) / (12 + 1 + 0.76))
    assert effective_rr(4, 4, NQ_FEE_RT, 1.0) == pytest.approx(
        3.24 / 5.76)
    # realized RR and breakeven from the verified NQ rf diagnosis
    w, loss = 8.853878583473863, -67.49073275862067
    assert realized_rr(w, loss) == pytest.approx(0.13118658253629564)
    assert breakeven_win_rate(w, loss) == pytest.approx(
        0.8840274588104158)
    assert expectancy(0.7187878787878788, w, loss) == pytest.approx(
        -12.615151515151515)
    # symmetric payoff sanity: RR 1 -> breakeven 50%
    assert realized_rr(10.0, -10.0) == 1.0
    assert breakeven_win_rate(10.0, -10.0) == 0.5
    econ = policy_econ(make_pol(T=2, S=12), NQ_FEE_RT, 1.0)
    assert econ["target_ticks"] == 2 and econ["stop_ticks"] == 12
    assert econ["effective_rr"] == pytest.approx(1.24 / 13.76,
                                                 abs=1e-4)
    assert econ["geom_breakeven_win_rate"] == pytest.approx(
        13.76 / 15.0, abs=1e-4)


# ------------------------------------------------------------------ 3
def test_high_win_negative_rr_flagged_invalid():
    v = assess_viability(trades=3300, win=0.7188, avg_win=8.85,
                         avg_loss=-67.49, ev_trade=-12.62,
                         net=-41630.0, qualified=True)
    assert v["viable"] is False
    assert v["valid_win_metric"] is False
    assert HIGH_WIN_WARNING in v["warnings"]
    # the same flags must surface in the headline aggregate
    agg = aggregate_folds([make_stats()], [make_pol()],
                          n_sess_total=63, n_folds_total=1,
                          fee_rt_ticks=NQ_FEE_RT)
    assert agg["win"] == pytest.approx(0.7, abs=1e-9)
    assert agg["valid_win_metric"] is False
    assert agg["viable"] is False
    assert HIGH_WIN_WARNING in agg["warnings"]
    # positive expectancy legitimizes the win metric and viability
    good = assess_viability(trades=500, win=0.60, avg_win=30.0,
                            avg_loss=-20.0, ev_trade=10.0, net=5000.0,
                            qualified=True, tps=2.0)
    assert good["viable"] is True
    assert good["valid_win_metric"] is True
    assert good["warnings"] == []


def test_viable_enforces_mandate_gates():
    base = dict(trades=500, avg_win=30.0, avg_loss=-20.0,
                ev_trade=10.0, net=5000.0, qualified=True)
    # profitable but sub-gate win rate: valid win metric, NOT viable
    low_win = assess_viability(win=0.30, **base)
    assert low_win["valid_win_metric"] is True
    assert low_win["viable"] is False
    assert low_win["mandate_gates"]["win_ge_gate"] is False
    # profitable, good win rate, but tps below mandate: NOT viable
    low_tps = assess_viability(win=0.60, tps=0.4, **base)
    assert low_tps["viable"] is False
    assert low_tps["mandate_gates"]["tps_ge_1"] is False
    # positive ev but negative net: NOT viable
    neg_net = assess_viability(trades=500, win=0.60, avg_win=30.0,
                               avg_loss=-20.0, ev_trade=10.0,
                               net=-1.0, qualified=True)
    assert neg_net["viable"] is False
    # zero trades: NOT viable
    empty = assess_viability(trades=0, win=None, avg_win=0.0,
                             avg_loss=0.0, ev_trade=None, net=0.0,
                             qualified=True)
    assert empty["viable"] is False


def test_viability_boundaries():
    # RR exactly 1.0 validates the win metric even with ev < 0
    rr1 = assess_viability(trades=100, win=0.90, avg_win=10.0,
                           avg_loss=-10.0, ev_trade=-1.0, net=-100.0,
                           qualified=True)
    assert rr1["valid_win_metric"] is True
    assert HIGH_WIN_WARNING not in rr1["warnings"]
    # ev exactly 0 with RR < 1 does not
    ev0 = assess_viability(trades=100, win=0.70, avg_win=8.0,
                           avg_loss=-18.67, ev_trade=0.0, net=0.0,
                           qualified=True)
    assert ev0["valid_win_metric"] is False
    assert HIGH_WIN_WARNING in ev0["warnings"]
    # win exactly at the gate triggers the warning when unsupported
    at_gate = assess_viability(trades=100, win=0.55, avg_win=5.0,
                               avg_loss=-50.0, ev_trade=-25.0,
                               net=-2500.0, qualified=True)
    assert HIGH_WIN_WARNING in at_gate["warnings"]
    # negative avg_win cannot fake RR >= 1 through abs()
    assert not realized_rr(-80.0, -40.0) == realized_rr(-80.0, -40.0)


def test_is_control_model():
    assert is_control_model("hgb_shuf") is True
    assert is_control_model("rf_shuf") is True
    assert is_control_model("rf") is False
    assert is_control_model("hgb") is False


# ------------------------------------------------------------------ 4
def test_shuffled_high_win_cannot_be_viable():
    # even with fabricated stellar stats, a control is never viable
    v = assess_viability(trades=1000, win=0.90, avg_win=50.0,
                         avg_loss=-10.0, ev_trade=44.0, net=44000.0,
                         qualified=True, is_control=True)
    assert v["viable"] is False
    assert CONTROL_WARNING in v["warnings"]
    agg = aggregate_folds(
        [make_stats(trades=300, n_wins=270, avg_win=50.0,
                    avg_loss=-10.0)],
        [make_pol()], n_sess_total=63, n_folds_total=1,
        fee_rt_ticks=NQ_FEE_RT, is_control=True)
    assert agg["viable"] is False
    assert agg["role"] == "control"
    assert CONTROL_WARNING in agg["warnings"]
    # empty control (all folds rejected) is also non-viable + labeled
    empty = aggregate_folds([], [], n_sess_total=63, n_folds_total=11,
                            fee_rt_ticks=NQ_FEE_RT, is_control=True)
    assert empty["viable"] is False
    assert empty["role"] == "control"
    # a barrier-artifact control gets BOTH warnings, not just the
    # control label (the high-win warning is never suppressed)
    artifact = assess_viability(trades=3300, win=0.65, avg_win=1.63,
                                avg_loss=-45.58, ev_trade=-14.81,
                                net=-48875.0, qualified=True,
                                is_control=True)
    assert artifact["viable"] is False
    assert HIGH_WIN_WARNING in artifact["warnings"]
    assert CONTROL_WARNING in artifact["warnings"]


# ------------------------------------------------------------ report
def test_report_md_shows_required_fields_and_control():
    agg = aggregate_folds([make_stats()], [make_pol()],
                          n_sess_total=63, n_folds_total=2,
                          fee_rt_ticks=NQ_FEE_RT)
    ctrl = aggregate_folds([], [], n_sess_total=63, n_folds_total=2,
                           fee_rt_ticks=NQ_FEE_RT, is_control=True)
    report = {"phase": "test", "instruments":
              {"NQ": {"rf": {"fold_table": [], "arms": {"base": agg}},
                      "hgb_shuf": {"fold_table": [],
                                   "arms": {"base": ctrl}}}},
              "controls": {"NQ": ctrl}}
    md = write_scalper_md(report)
    for token in ("target", "stop", "RR", "avg win", "avg loss",
                  "breakeven win", "observed win", "gross", "net",
                  "viable", "Shuffled-label control"):
        assert token in md, token
    assert HIGH_WIN_WARNING in md
    assert CONTROL_WARNING in md
    # aggregate block carries every mandated field
    for key in ("target_ticks", "stop_ticks", "avg_win", "avg_loss",
                "breakeven_win_rate", "observed_win_rate", "gross",
                "costs", "net", "realized_rr", "viable", "warnings"):
        assert key in agg, key
    assert agg["gross"] == pytest.approx(agg["net"] + agg["costs"])


def test_aggregate_pooled_stats_are_exact():
    # pooled avg win/loss must be trade-weighted, matching a flat
    # recomputation over the underlying trades
    s1 = make_stats(trades=300, n_wins=210, avg_win=8.0,
                    avg_loss=-60.0)
    s2 = make_stats(trades=100, n_wins=90, avg_win=12.0,
                    avg_loss=-70.0)
    agg = aggregate_folds([s1, s2], [make_pol(), make_pol(T=3)],
                          n_sess_total=126, n_folds_total=2,
                          fee_rt_ticks=NQ_FEE_RT)
    wins = np.concatenate([np.full(210, 8.0), np.full(90, 12.0)])
    losses = np.concatenate([np.full(90, -60.0), np.full(10, -70.0)])
    assert agg["avg_win"] == pytest.approx(wins.mean(), abs=1e-4)
    assert agg["avg_loss"] == pytest.approx(losses.mean(), abs=1e-4)
    assert agg["win"] == pytest.approx(300 / 400, abs=1e-4)
    assert agg["net"] == pytest.approx(s1["net"] + s2["net"], abs=1e-6)
    assert agg["ev_trade"] == pytest.approx(
        (s1["net"] + s2["net"]) / 400, abs=1e-4)
    assert agg["target_ticks"] == {"2": 1, "3": 1}
    assert agg["stop_ticks"] == {"12": 2}
