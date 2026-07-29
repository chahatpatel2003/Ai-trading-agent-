from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tslab.audit.leakage import Status, TruncationInvarianceCheck
from tslab.features.core import (FeatureParams, compute_features,
                                 feature_columns, latest_usable)

P = FeatureParams()


def synth_bars(cfg, sessions=(date(2024, 1, 9), date(2024, 1, 10),
                              date(2024, 1, 11)), with_micro=True, seed=3):
    """Bar frames matching the Phase 2B schema: 3 RTH-ish hours per session."""
    rng = np.random.default_rng(seed)
    frames = []
    for s in sessions:
        idx = pd.date_range(pd.Timestamp(s, tz="UTC") + pd.Timedelta(hours=14),
                            periods=240, freq="1min")   # 08:00-12:00 CT (CST)
        n = len(idx)
        close = 17000 + np.cumsum(rng.normal(0, 3, n))
        opn = np.r_[close[0], close[:-1]]
        hi = np.maximum(opn, close) + rng.uniform(0, 2, n)
        lo = np.minimum(opn, close) - rng.uniform(0, 2, n)
        df = pd.DataFrame({
            "open": opn, "high": hi, "low": lo, "close": close,
            "volume": rng.integers(100, 2000, n),
            "session": s, "symbol": "NQH4",
        }, index=idx)
        if with_micro:
            df["n_trades"] = rng.integers(50, 900, n)
            df["buy_vol"] = (df["volume"] * rng.uniform(0.3, 0.7, n)).astype(int)
            df["sell_vol"] = df["volume"] - df["buy_vol"] \
                - rng.integers(0, 20, n).clip(0, 5)
            df["delta"] = df["buy_vol"] - df["sell_vol"]
            df["rv"] = rng.uniform(0, 1e-7, n)
            df["vwap"] = close + rng.normal(0, 0.5, n)
            df["spread_mean_ticks"] = rng.uniform(1, 2, n)
            df["spread_close_ticks"] = 1.0
            df["n_quotes"] = rng.integers(500, 20000, n)
            df["bid_sz_close"] = rng.integers(1, 20, n)
            df["ask_sz_close"] = rng.integers(1, 20, n)
        df["info_time"] = df.index + pd.Timedelta(seconds=60)
        from tslab import sessions as S
        flat = S.eod_flatten_utc(s, cfg.sessions)
        df["secs_to_flatten"] = (flat - pd.DatetimeIndex(df["info_time"])
                                 ).total_seconds()
        ct = df.index.tz_convert(cfg.sessions.timezone)
        df["rth"] = pd.Series(ct.time, index=df.index) >= \
            cfg.sessions.rth["NQ"]["start_ct"]
        frames.append(df)
    return pd.concat(frames)


def test_truncation_invariance_full_feature_set(cfg):
    bars = synth_bars(cfg)
    res = TruncationInvarianceCheck(
        bars, lambda df: compute_features(df, cfg, P), n_probes=6).run()
    assert res.status == Status.PASS, res.message


def test_no_future_session_data(cfg):
    """Perturbing the LAST session must leave earlier sessions' features
    bit-identical (prior-session refs, rolling windows, everything)."""
    bars = synth_bars(cfg)
    tampered = bars.copy()
    last = sorted(set(bars["session"]))[-1]
    m = tampered["session"] == last
    tampered.loc[m, ["open", "high", "low", "close"]] += 500.0
    a = compute_features(bars, cfg, P)
    b = compute_features(tampered, cfg, P)
    early = a["session"] != last
    pd.testing.assert_frame_equal(a[early], b[early])


def test_no_same_bar_decision_leak(cfg):
    bars = synth_bars(cfg)
    feats = compute_features(bars, cfg, P)
    t_open = bars.index[100]
    mid_bar = t_open + pd.Timedelta(seconds=30)
    row = latest_usable(feats, mid_bar)
    assert row.name == bars.index[99]          # forming bar NOT visible
    at_close = t_open + pd.Timedelta(seconds=60)
    assert latest_usable(feats, at_close).name == t_open  # visible at close


def test_prior_session_levels_strictly_prior(cfg):
    bars = synth_bars(cfg)
    feats = compute_features(bars, cfg, P)
    sessions = sorted(set(bars["session"]))
    first, second = sessions[0], sessions[1]
    assert feats.loc[feats["session"] == first, "f_prior_high_dist"].isna().all()
    s2 = feats["session"] == second
    g1 = bars[bars["session"] == first]
    expect = (bars.loc[s2, "close"] - g1["high"].max())
    got = feats.loc[s2, "f_prior_high_dist"] * feats.loc[s2, "f_atr_ticks"] \
        * cfg.instruments["NQ"].tick_size
    ok = got.notna()
    assert np.allclose(got[ok], expect[ok])


def test_opening_range_hidden_until_complete(cfg):
    bars = synth_bars(cfg)
    feats = compute_features(bars, cfg, P)
    ct = feats.index.tz_convert(cfg.sessions.timezone)
    tt = pd.Series([t.time() for t in ct], index=feats.index)
    from datetime import time
    before = tt < time(8, 59)      # info_time < 09:00 CT => OR not yet known
    after = tt >= time(9, 0)
    assert feats.loc[before, "f_or_high_dist"].isna().all()
    assert feats.loc[after, "f_or_high_dist"].notna().all()


def test_session_anchored_reset(cfg):
    bars = synth_bars(cfg)
    feats = compute_features(bars, cfg, P)
    firsts = feats.groupby("session").head(1)
    # first bar of each synthetic session opens 6h after the 17:00 CT open
    assert np.allclose(firsts["f_min_since_open"], 15 * 60)
    assert (firsts["f_sess_cumvol"]
            == bars.groupby("session").head(1)["volume"]).all()


def test_no_backfill_warmups_are_nan(cfg):
    bars = synth_bars(cfg)
    feats = compute_features(bars, cfg, P)
    assert feats["f_ret_60"].iloc[:60].isna().all()
    assert feats["f_z_120"].iloc[:119].isna().all()
    assert feats["f_atr_ticks"].iloc[:P.atr_window - 1].isna().all()


def test_missing_micro_columns_degrade_to_nan(cfg):
    bars = synth_bars(cfg, with_micro=False)
    feats = compute_features(bars, cfg, P)
    full = compute_features(synth_bars(cfg), cfg, P)
    assert feature_columns(feats) == feature_columns(full)   # same schema
    for c in ("f_delta_15", "f_spread_ratio", "f_book_imb", "f_ntrades_z",
              "f_rv_15", "f_cvd_sess"):
        assert feats[c].isna().all()
    assert feats["f_ret_5"].notna().sum() > 0                # price side intact
    res = TruncationInvarianceCheck(
        bars, lambda df: compute_features(df, cfg, P), n_probes=4).run()
    assert res.status == Status.PASS, res.message


def test_categorical_and_int_features(cfg):
    bars = synth_bars(cfg)
    feats = compute_features(bars, cfg, P)
    assert feats["f_dow"].dtype == "int64"
    assert feats["f_ct_hour"].dtype == "int64"
    assert set(feats["f_dow"].unique()) <= set(range(5))


def test_rolls_do_not_bleed_across_contracts(cfg):
    bars = synth_bars(cfg)
    sessions = sorted(set(bars["session"]))
    bars.loc[bars["session"] == sessions[-1], "symbol"] = "NQM4"
    # a fake 400-pt roll gap between contracts
    bars.loc[bars["session"] == sessions[-1],
             ["open", "high", "low", "close"]] += 400.0
    feats = compute_features(bars, cfg, P)
    first_new = feats[feats["symbol"] == "NQM4"].iloc[0]
    assert np.isnan(first_new["f_ret_1"])      # no cross-contract return
    assert np.isnan(first_new["f_atr_ticks"])  # ATR restarts per contract
