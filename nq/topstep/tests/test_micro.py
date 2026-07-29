from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tslab.audit.leakage import Status, TruncationInvarianceCheck
from tslab.data.nq_micro import (MICRO_COLS, quote_micro_aggregates,
                                 trade_burst_aggregates)
from tslab.features.core import FeatureParams
from tslab.features.micro import compute_micro_features
from tests.test_features import synth_bars


def ts(s):
    return pd.Timestamp(s, tz="UTC")


def test_quote_micro_aggregates_values():
    t0 = ts("2024-01-09 15:00:00")
    rows = [
        # (sec, bid_px, ask_px, bid_sz, ask_sz)
        (0, 17000.00, 17000.25, 10, 2),    # imb = 8/12
        (2, 17000.00, 17000.25, 2, 10),    # flip (+ -> -)   imb -8/12
        (4, 17000.00, 17000.25, 9, 3),     # flip (- -> +)
        (7, 17000.00, 17000.75, 5, 5),     # wide (3 ticks), imb 0
        (61, 17000.00, 17000.25, 4, 8),    # next bar
        (62, 17001.00, 17000.25, 5, 5),    # crossed -> dropped
    ]
    q = pd.DataFrame(
        [(t0 + pd.Timedelta(seconds=s), b, a, bs, asz)
         for s, b, a, bs, asz in rows],
        columns=["ts_event", "bid_px_00", "ask_px_00", "bid_sz_00",
                 "ask_sz_00"])
    out = quote_micro_aggregates(q, tick_size=0.25)
    b0 = out.iloc[0]
    assert b0["imb_first"] == pytest.approx(8 / 12)
    assert b0["imb_last"] == pytest.approx(0.0)
    assert b0["book_flips"] == 2                      # zero-sign not a flip
    assert b0["wide_share"] == pytest.approx(0.25)
    assert b0["q_burst"] == 3                         # secs 0,2,4 in one 5s bin
    b1 = out.iloc[1]
    assert b1["imb_last"] == pytest.approx(-4 / 12)   # crossed row dropped
    assert len(out) == 2


def test_trade_burst_aggregates():
    t0 = ts("2024-01-09 15:00:00")
    rows = [(0, 5, "B"), (3, 4, "B"), (12, 2, "A"), (15, 20, "A"), (30, 1, "N")]
    t = pd.DataFrame(
        [(t0 + pd.Timedelta(seconds=s), 17000.0, sz, side, "NQH4")
         for s, sz, side in rows],
        columns=["ts_event", "price", "size", "side", "symbol"])
    out = trade_burst_aggregates(t)
    # 10s bins: [0,10)=+9, [10,20)=-22, [30,40)=0 -> max abs = 22
    assert out["d_burst"].iloc[0] == pytest.approx(22.0)


def synth_micro_bars(cfg, seed=11):
    bars = synth_bars(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    n = len(bars)
    bars["imb_first"] = rng.uniform(-0.8, 0.8, n)
    bars["imb_last"] = rng.uniform(-0.8, 0.8, n)
    bars["imb_mean"] = (bars["imb_first"] + bars["imb_last"]) / 2
    bars["sz_bid_mean"] = rng.uniform(2, 30, n)
    bars["sz_ask_mean"] = rng.uniform(2, 30, n)
    bars["book_flips"] = rng.integers(0, 40, n).astype(float)
    bars["q_burst"] = rng.integers(5, 400, n).astype(float)
    bars["wide_share"] = rng.uniform(0, 0.3, n)
    bars["d_burst"] = rng.integers(1, 300, n).astype(float)
    return bars


def test_micro_features_schema_and_truncation(cfg):
    bars = synth_micro_bars(cfg)
    feats = compute_micro_features(bars, cfg, FeatureParams())
    mcols = [c for c in feats.columns if c.startswith("f_m_")]
    assert len(mcols) == 16
    res = TruncationInvarianceCheck(
        bars, lambda df: compute_micro_features(df, cfg, FeatureParams()),
        n_probes=5).run()
    assert res.status == Status.PASS, res.message


def test_micro_features_nan_without_inputs(cfg):
    bars = synth_bars(cfg)                            # no micro columns
    feats = compute_micro_features(bars, cfg, FeatureParams())
    assert feats["f_m_imb"].isna().all()
    assert feats["f_ret_5"].notna().sum() > 0         # base intact


def test_failed_push_and_sweep_binaries(cfg):
    bars = synth_micro_bars(cfg)
    i = 100
    win_hi = bars["high"].iloc[i - 5:i].max()
    bars.iloc[i, bars.columns.get_loc("high")] = win_hi + 5.0
    bars.iloc[i, bars.columns.get_loc("close")] = win_hi - 2.0
    bars.iloc[i, bars.columns.get_loc("delta")] = 500
    lo20 = bars["low"].iloc[i - 20:i].min()
    bars.iloc[i, bars.columns.get_loc("low")] = lo20 - 5.0
    feats = compute_micro_features(bars, cfg, FeatureParams())
    assert feats["f_m_failed_up"].iloc[i] == 1.0
    assert feats["f_m_sweep_lo"].iloc[i] == 1.0       # closed back above lo20


def test_micro_grid_risk_arithmetic(cfg):
    from tslab.strategy.rules import micro_candidate_grid
    g = micro_candidate_grid()
    assert g == micro_candidate_grid()
    assert len(g) == len({r.name for r in g}) >= 28
    feats = compute_micro_features(synth_micro_bars(cfg), cfg,
                                   FeatureParams())
    for rule in g:
        cap = 1000.0 / rule.max_trades_per_day
        band = next(f for f in rule.filters
                    if f.feature == "f_atr_ticks" and f.op == "between")
        assert rule.stop_atr * band.hi * 5.0 <= cap + 1e-9
        assert rule.daily_loss_usd == pytest.approx(1000.0 - cap)
        for f in rule.filters:
            assert f.feature in feats.columns, (rule.name, f.feature)
            f.mask(feats)
