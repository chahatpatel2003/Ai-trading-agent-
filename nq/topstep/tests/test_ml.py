from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tslab.ml.discovery import (build_dataset, half_year_folds,
                                make_entries, run_signal_backtest)
from tests.test_features import synth_bars
from tslab.features.core import FeatureParams, compute_features
from tslab.features.labels import LabelParams, compute_labels


def prepped(cfg, sessions):
    bars = synth_bars(cfg, sessions=sessions)
    feats = compute_features(bars, cfg, FeatureParams())
    labels = compute_labels(bars, cfg, LabelParams()).labels
    tick = cfg.instruments["NQ"].tick_size
    bars = bars.copy()
    bars["entry_ok"] = True
    bars["__atr_pts"] = feats["f_atr_ticks"] * tick
    return bars, feats, labels


def test_dataset_rows_grid_and_targets(cfg):
    sessions = tuple(date(2024, 1, d) for d in (9, 10, 11))
    bars, feats, labels = prepped(cfg, sessions)
    ds = build_dataset(bars, feats, labels, cfg)
    assert (pd.DatetimeIndex(ds.index).minute % 5 == 0).all()
    assert {"y_dir15", "y_dir30", "y_tb", "session"} <= set(ds.columns)
    ok = ds["y_dir30"].dropna()
    assert set(ok.unique()) <= {0.0, 1.0}
    # target rows align with the audited labels frame exactly
    j = ds["y_dir30"].dropna().index
    assert ((labels.loc[j, "lab_fwd_30"] > 0).astype(float)
            == ds.loc[j, "y_dir30"]).all()


def test_half_year_folds_disjoint_and_purged():
    sessions = [d.date() for d in pd.bdate_range("2021-01-04", "2024-12-31")]
    folds = half_year_folds(sessions, (2023, 1), (2024, 2))
    assert len(folds) == 4
    for f in folds:
        assert not set(f["train"]) & set(f["test"])
        assert max(f["train"]) < min(f["test"])
        # 1-session purge: the session right before test is excluded
        gap = [s for s in sessions if f["train"][-1] < s < f["test"][0]]
        assert len(gap) >= 1
    # expanding: each later fold trains on strictly more sessions
    assert all(len(folds[i + 1]["train"]) > len(folds[i]["train"])
               for i in range(len(folds) - 1))


def test_signal_backtest_respects_guards(cfg):
    sessions = tuple(date(2024, 1, d) for d in (9, 10))
    bars, feats, labels = prepped(cfg, sessions)
    ent = pd.Series(1.0, index=bars.index)          # always-long firehose
    trades = run_signal_backtest(bars, ent, cfg, "MNQ", stop_atr=4.0,
                                 r_mult=1.5, time_stop=10, max_trades=2,
                                 lock=None)
    per_day = pd.Series([t.session for t in trades]).value_counts()
    assert (per_day <= 2).all() and len(trades) > 0
    for t in trades:
        assert t.risk_usd <= 1000.0 / 2 + 1e-6      # cap = DLL / max_trades
    from tslab import sessions as S
    for t in trades:
        assert t.exit_time + pd.Timedelta(seconds=60) <= \
            S.eod_flatten_utc(t.session, cfg.sessions)


def test_make_entries_two_sided(cfg):
    ds = pd.DataFrame(index=pd.RangeIndex(4))
    probs = np.array([0.9, 0.5, 0.1, 0.61])
    e2 = make_entries(ds, probs, 0.6, True)
    assert e2.tolist() == [1.0, 0.0, -1.0, 1.0]
    e1 = make_entries(ds, probs, 0.6, False)
    assert e1.tolist() == [1.0, 0.0, 0.0, 1.0]


def test_shuffled_labels_give_chance_auc(cfg):
    # structural sanity of the whole fit path on synthetic data
    from sklearn.metrics import roc_auc_score
    from tslab.ml.discovery import model_zoo
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(3000, 10)),
                     columns=[f"f_x{i}" for i in range(10)])
    y = rng.integers(0, 2, 3000).astype(float)      # pure noise labels
    mdl = model_zoo()["hgb"]
    mdl.fit(X.iloc[:2000], y[:2000])
    auc = roc_auc_score(y[2000:], mdl.predict_proba(X.iloc[2000:])[:, 1])
    assert abs(auc - 0.5) < 0.06
