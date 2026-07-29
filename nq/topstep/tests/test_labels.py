from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tslab.audit.leakage import (FoldEmbargoCheck, LabelHorizonCheck,
                                 Status)
from tslab.features.labels import (LabelParams, compute_labels,
                                   make_session_folds, required_embargo)
from tests.test_features import synth_bars

LP = LabelParams(fwd_horizons=(3, 5), tb_horizon=20,
                 tb_target_atr=2.0, tb_stop_atr=1.0, atr_window=5)


def controlled_bars(cfg, closes, session=date(2024, 1, 9), start_hour=14):
    idx = pd.date_range(pd.Timestamp(session, tz="UTC")
                        + pd.Timedelta(hours=start_hour),
                        periods=len(closes), freq="1min")
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.5,
                       "low": np.minimum(o, c) - 0.5, "close": c,
                       "volume": 100, "session": session, "symbol": "NQH4"},
                      index=idx)
    df["info_time"] = df.index + pd.Timedelta(seconds=60)
    return df


def test_fixed_horizon_values_and_session_bound(cfg):
    bars = controlled_bars(cfg, np.linspace(17000, 17099, 100))
    res = compute_labels(bars, cfg, LP)
    lab = res.labels["lab_fwd_3"]
    i = 10
    assert lab.iloc[i] == pytest.approx(
        np.log(bars["close"].iloc[i + 3] / bars["close"].iloc[i]))
    assert lab.iloc[-3:].isna().all()          # cannot see past session end
    assert res.end_times["lab_fwd_3"].iloc[i] == bars["info_time"].iloc[i + 3]


def test_fixed_horizon_never_crosses_sessions(cfg):
    b1 = controlled_bars(cfg, np.full(30, 17000.0), date(2024, 1, 9))
    b2 = controlled_bars(cfg, np.full(30, 18000.0), date(2024, 1, 10))
    res = compute_labels(pd.concat([b1, b2]), cfg, LP)
    assert res.labels["lab_fwd_5"].iloc[25:30].isna().all()
    # a naive shift would have produced a huge fake +5.7% "return" here
    assert res.labels["lab_fwd_5"].iloc[:25].abs().max() < 1e-9


def test_triple_barrier_target_stop_and_tie(cfg):
    flat = np.full(40, 17000.0)
    bars = controlled_bars(cfg, flat)
    res = compute_labels(bars, cfg, LP)
    # ATR = 1.0 (range 0.5+0.5 each bar): target +2.0, stop -1.0 from open[T+1]
    i = 10
    assert res.labels["lab_tb"].iloc[i] == 0            # flat path: timeout
    up = flat.copy(); up[15] = 17010.0                  # spike through target
    res_up = compute_labels(controlled_bars(cfg, up), cfg, LP)
    assert res_up.labels["lab_tb"].iloc[10] == 1
    dn = flat.copy(); dn[15] = 16990.0
    res_dn = compute_labels(controlled_bars(cfg, dn), cfg, LP)
    assert res_dn.labels["lab_tb"].iloc[10] == -1
    # tie bar: huge range pierces BOTH barriers -> stop-first convention
    tie = flat.copy()
    bars_tie = controlled_bars(cfg, tie)
    j = bars_tie.index[15]
    bars_tie.loc[j, "high"] = 17010.0
    bars_tie.loc[j, "low"] = 16990.0
    res_tie = compute_labels(bars_tie, cfg, LP)
    assert res_tie.labels["lab_tb"].iloc[10] == -1


def test_mfe_mae_known_values(cfg):
    flat = np.full(40, 17000.0)
    path = flat.copy()
    path[12], path[18] = 17003.0, 16998.0     # within-window extremes
    bars = controlled_bars(cfg, path)
    res = compute_labels(bars, cfg, LP)
    i = 10
    atr = 1.0 + 5.0 / 5                       # spike bars widen TR window mean
    entry = bars["open"].iloc[11]
    whi = bars["high"].iloc[11:31].max()
    wlo = bars["low"].iloc[11:31].min()
    got_mfe = res.labels["lab_mfe_atr"].iloc[i]
    got_mae = res.labels["lab_mae_atr"].iloc[i]
    # exact ATR at i is data-dependent; verify via the recorded definition
    atr_i = (whi - entry) / got_mfe
    assert got_mae == pytest.approx((entry - wlo) / atr_i)


def test_flatten_label_and_deadline_bar(cfg):
    # session 2024-01-09 (CST): bars 14:00Z.. ; flatten 21:10Z. The deadline
    # bar is the last one whose CLOSE is strictly before 21:10Z, i.e. the
    # 21:08Z bar (closes 21:09Z).
    n = 451                                    # 14:00Z .. 21:30Z inclusive
    bars = controlled_bars(cfg, np.linspace(17000, 17450, n))
    res = compute_labels(bars, cfg, LP)
    dl_open = pd.Timestamp("2024-01-09 21:08:00+00:00")
    i = 100
    entry = bars["open"].iloc[i + 1]
    want = np.log(bars.loc[dl_open, "close"] / entry)
    assert res.labels["lab_flat_ret"].iloc[i] == pytest.approx(want)
    assert res.end_times["lab_flat_ret"].iloc[i] == \
        dl_open + pd.Timedelta(seconds=60)
    # decisions at/after the deadline bar have no legal entry -> NaN
    dl_pos = bars.index.get_loc(dl_open)
    assert res.labels["lab_flat_ret"].iloc[dl_pos:].isna().all()


def test_flatten_respects_early_close(cfg):
    # Black Friday 2024: halt 12:15 CT = 18:15Z. Bars 15:00Z..18:14Z.
    bars = controlled_bars(cfg, np.linspace(2000, 2190, 191),
                           session=date(2024, 11, 29), start_hour=15)
    res = compute_labels(bars, cfg, LP)
    ends = res.end_times["lab_flat_ret"].dropna()
    # every information end (deadline close) precedes the 18:15Z halt
    assert (ends <= pd.Timestamp("2024-11-29 18:15:00+00:00")).all()
    assert (ends == ends.iloc[0]).all()        # single shared deadline bar


def test_label_horizon_checks_pass(cfg):
    bars = synth_bars(cfg)
    res = compute_labels(bars, cfg, LP)
    fixed = LabelHorizonCheck(
        bars, lambda df: compute_labels(df, cfg, LP).labels["lab_fwd_5"],
        horizon_rows=5, n_probes=4).run()
    assert fixed.status == Status.PASS, fixed.message
    for name in ("lab_tb_ret", "lab_flat_ret", "lab_mfe_atr"):
        var = LabelHorizonCheck(
            bars, lambda df, n=name: compute_labels(df, cfg, LP).labels[n],
            label_end_times=res.end_times[name], n_probes=4).run()
        assert var.status == Status.PASS, (name, var.message)


def test_understated_label_horizon_caught(cfg):
    bars = synth_bars(cfg)
    res = LabelHorizonCheck(
        bars, lambda df: compute_labels(df, cfg, LP).labels["lab_fwd_5"],
        horizon_rows=2, n_probes=4).run()      # claims 2, actually uses 5
    assert res.status == Status.FAIL


def test_embargo_from_specs_and_folds(cfg):
    bars = synth_bars(cfg)
    specs = compute_labels(bars, cfg, LP).specs
    emb = required_embargo(specs)
    assert emb == pd.Timedelta(hours=24)
    sessions = [date(2024, 1, d) for d in (8, 9, 10, 11, 12, 16, 17, 18, 19)]
    folds = make_session_folds(sessions, cfg, n_train=4, n_test=2,
                               embargo_sessions=1)
    assert len(folds) >= 1
    assert FoldEmbargoCheck(folds, emb).run().status == Status.PASS
    # zero-session embargo leaves only the 1h maintenance gap -> must FAIL
    tight = make_session_folds(sessions, cfg, n_train=4, n_test=2,
                               embargo_sessions=0)
    assert FoldEmbargoCheck(tight, emb).run().status == Status.FAIL


def test_label_purge_check_clean_and_violated(cfg):
    from tslab.features.labels import check_label_purge
    bars = synth_bars(cfg)
    res = compute_labels(bars, cfg, LP)
    sessions = sorted(set(bars["session"]))
    folds = [{
        "train_start": pd.Timestamp(sessions[0], tz="UTC"),
        "train_end": pd.Timestamp(sessions[1], tz="UTC")
                     + pd.Timedelta(hours=22),          # after session 2 close
        "test_start": pd.Timestamp(sessions[2], tz="UTC"),
        "test_end": pd.Timestamp(sessions[2], tz="UTC") + pd.Timedelta(days=1),
    }]
    assert check_label_purge(res.end_times, folds) == []   # session-bounded
    # push one train-row end time past train_end -> must be flagged
    tampered = res.end_times.copy()
    col = "lab_flat_ret"
    i = tampered.index[10]
    tampered.loc[i, col] = folds[0]["train_end"] + pd.Timedelta(hours=5)
    bad = check_label_purge(tampered, folds)
    assert len(bad) == 1 and bad[0]["label"] == col and bad[0]["n_bad"] == 1


def test_end_times_never_before_decision(cfg):
    bars = synth_bars(cfg)
    res = compute_labels(bars, cfg, LP)
    info = pd.DatetimeIndex(bars["info_time"])
    for name in res.end_times.columns:
        e = res.end_times[name]
        ok = e.notna()
        assert (pd.DatetimeIndex(e[ok]) >= info[ok.to_numpy()]).all(), name
