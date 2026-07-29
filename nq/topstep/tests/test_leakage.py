from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tslab.audit.leakage import (FoldEmbargoCheck, LabelHorizonCheck,
                                 LeakageAuditor, Status,
                                 TruncationInvarianceCheck)


@pytest.fixture()
def prices():
    idx = pd.date_range("2024-01-02", periods=400, freq="1min", tz="UTC")
    rng = np.random.default_rng(3)
    close = 17000 + np.cumsum(rng.normal(0, 2, len(idx)))
    return pd.DataFrame({"close": close}, index=idx)


# --------------------------------------------------------------------- #
# TruncationInvarianceCheck
# --------------------------------------------------------------------- #

def causal_features(df):
    return pd.DataFrame({
        "sma20": df["close"].rolling(20).mean(),
        "ret5": df["close"].pct_change(5),
    }, index=df.index)


def centered_features(df):
    # classic bug: centered rolling window sees the future
    return pd.DataFrame({
        "sma20c": df["close"].rolling(20, center=True).mean(),
    }, index=df.index)


def full_sample_zscore(df):
    # classic bug: normalizing with full-sample mean/std
    c = df["close"]
    return pd.DataFrame({"z": (c - c.mean()) / c.std()}, index=df.index)


def test_causal_features_pass(prices):
    res = TruncationInvarianceCheck(prices, causal_features).run()
    assert res.status == Status.PASS, res.message


def test_centered_window_caught(prices):
    res = TruncationInvarianceCheck(prices, centered_features).run()
    assert res.status == Status.FAIL
    assert res.details["mismatches"]


def test_full_sample_normalization_caught(prices):
    res = TruncationInvarianceCheck(prices, full_sample_zscore).run()
    assert res.status == Status.FAIL


def test_wrong_index_caught(prices):
    res = TruncationInvarianceCheck(
        prices, lambda df: causal_features(df).iloc[5:]).run()
    assert res.status == Status.FAIL


def test_warmup_backfill_leak_caught(prices):
    # bfill pulls FUTURE values into the first rows of the warm-up region —
    # invisible at the probe rows themselves, caught by full-prefix comparison
    def leaky(df):
        return pd.DataFrame({"sma": df["close"].rolling(20).mean().bfill()},
                            index=df.index)

    res = TruncationInvarianceCheck(prices, leaky).run()
    assert res.status == Status.FAIL


def test_all_nan_feature_warns_not_passes(prices):
    def sparse(df):
        return pd.DataFrame({"never": np.nan}, index=df.index)

    res = TruncationInvarianceCheck(prices, sparse).run()
    assert res.status == Status.WARN


def test_categorical_feature_supported(prices):
    # non-numeric causal feature must be compared by equality, not crash
    def cat(df):
        up = df["close"].gt(df["close"].shift())
        return pd.DataFrame({"dir": up.map({True: "up", False: "down"})},
                            index=df.index)

    res = TruncationInvarianceCheck(prices, cat).run()
    assert res.status == Status.PASS, res.message


# --------------------------------------------------------------------- #
# FoldEmbargoCheck
# --------------------------------------------------------------------- #

def _ts(s):
    return pd.Timestamp(s, tz="UTC")


def test_good_folds_pass():
    folds = [
        {"train_start": _ts("2023-01-01"), "train_end": _ts("2023-06-30"),
         "test_start": _ts("2023-07-03"), "test_end": _ts("2023-09-29")},
        {"train_start": _ts("2023-01-01"), "train_end": _ts("2023-09-29"),
         "test_start": _ts("2023-10-03"), "test_end": _ts("2023-12-29")},
    ]
    res = FoldEmbargoCheck(folds, pd.Timedelta(days=2)).run()
    assert res.status == Status.PASS


def test_overlapping_folds_fail():
    folds = [{"train_start": _ts("2023-01-01"), "train_end": _ts("2023-08-01"),
              "test_start": _ts("2023-07-01"), "test_end": _ts("2023-09-01")}]
    res = FoldEmbargoCheck(folds, pd.Timedelta(days=1)).run()
    assert res.status == Status.FAIL


def test_embargo_violation_fails():
    folds = [{"train_start": _ts("2023-01-01"), "train_end": _ts("2023-06-30"),
              "test_start": _ts("2023-07-01"), "test_end": _ts("2023-08-01")}]
    res = FoldEmbargoCheck(folds, pd.Timedelta(days=5)).run()
    assert res.status == Status.FAIL


def test_overlapping_test_windows_across_folds_fail():
    # each fold is individually fine; the TEST windows overlap each other,
    # double-counting the same OOS period
    folds = [
        {"train_start": _ts("2023-01-01"), "train_end": _ts("2023-06-30"),
         "test_start": _ts("2023-07-03"), "test_end": _ts("2023-10-15")},
        {"train_start": _ts("2023-01-01"), "train_end": _ts("2023-09-25"),
         "test_start": _ts("2023-10-01"), "test_end": _ts("2023-12-29")},
    ]
    res = FoldEmbargoCheck(folds, pd.Timedelta(days=2)).run()
    assert res.status == Status.FAIL
    assert any("test windows overlap" in p for p in res.details["problems"])


def test_gap_aware_embargo_from_row_horizon():
    # 5-row label horizon; bars span a weekend, so 5 rows can cover > 2 days.
    # A 4-hour calendar embargo passes naively but must FAIL gap-aware.
    fri = pd.date_range("2023-01-06 14:00", periods=60, freq="1min", tz="UTC")
    mon = pd.date_range("2023-01-09 14:00", periods=300, freq="1min", tz="UTC")
    idx = fri.append(mon)
    folds = [{"train_start": _ts("2023-01-02"), "train_end": _ts("2023-01-06 15:00"),
              "test_start": _ts("2023-01-06 19:00"), "test_end": _ts("2023-01-10")}]
    naive = FoldEmbargoCheck(folds, pd.Timedelta(hours=1)).run()
    assert naive.status == Status.PASS
    aware = FoldEmbargoCheck(folds, pd.Timedelta(hours=1),
                             bar_index=idx, horizon_rows=5).run()
    assert aware.status == Status.FAIL


# --------------------------------------------------------------------- #
# LabelHorizonCheck
# --------------------------------------------------------------------- #

def test_honest_forward_return_label_passes(prices):
    h = 5

    def label_fn(df):
        return df["close"].shift(-h) / df["close"] - 1.0

    res = LabelHorizonCheck(prices, label_fn, horizon_rows=h).run()
    assert res.status == Status.PASS, res.message


def test_understated_horizon_caught(prices):
    # label secretly looks 10 rows ahead but claims a 5-row horizon
    def label_fn(df):
        return df["close"].shift(-10) / df["close"] - 1.0

    res = LabelHorizonCheck(prices, label_fn, horizon_rows=5).run()
    assert res.status == Status.FAIL


def test_label_with_no_nan_tail_caught(prices):
    # filling forward hides the unknowable tail
    def label_fn(df):
        return (df["close"].shift(-5) / df["close"] - 1.0).fillna(0.0)

    res = LabelHorizonCheck(prices, label_fn, horizon_rows=5).run()
    assert res.status == Status.FAIL
    assert "non-NaN" in res.message


def test_variable_horizon_label_with_end_times_passes(prices):
    # "return to a known end bar" style label: fully knowable in-sample even
    # for the tail (its end times never exceed the data), so the blanket
    # NaN-tail rule must not apply in end-times mode
    n = len(prices)
    end_pos = np.minimum(np.arange(n) + 3, n - 1)

    def label_fn(df):
        m = len(df)
        ep = np.minimum(np.arange(m) + 3, n - 1)
        ep = np.minimum(ep, m - 1)
        return pd.Series(df["close"].values[ep] / df["close"].values - 1.0,
                         index=df.index)

    # information end = CLOSE time of the last bar used (open + 1min)
    ends = pd.Series(prices.index[end_pos] + pd.Timedelta(minutes=1),
                     index=prices.index)
    res = LabelHorizonCheck(prices, label_fn, label_end_times=ends).run()
    assert res.status == Status.PASS, res.message


def test_variable_horizon_overrun_caught(prices):
    # non-NaN labels claiming end times beyond the available data must fail
    def label_fn(df):
        return pd.Series(0.0, index=df.index)

    ends = pd.Series(prices.index + pd.Timedelta(days=30), index=prices.index)
    res = LabelHorizonCheck(prices, label_fn, label_end_times=ends).run()
    assert res.status == Status.FAIL
    assert "beyond the available data" in res.message


def test_horizon_mode_arguments_are_exclusive(prices):
    with pytest.raises(ValueError):
        LabelHorizonCheck(prices, lambda df: pd.Series(0.0, index=df.index))


# --------------------------------------------------------------------- #
# auditor plumbing
# --------------------------------------------------------------------- #

def test_auditor_runs_all_and_reports(prices, tmp_path):
    aud = (LeakageAuditor()
           .register(TruncationInvarianceCheck(prices, causal_features))
           .register(TruncationInvarianceCheck(prices, centered_features)))
    results = aud.run_all()
    assert len(results) == 2
    assert not LeakageAuditor.passed(results)
    path = LeakageAuditor.write_report(results, tmp_path)
    assert path.exists() and path.stat().st_size > 0


def test_crashing_check_reported_as_fail(prices):
    class Boom(TruncationInvarianceCheck):
        def run(self):
            raise RuntimeError("boom")

    aud = LeakageAuditor().register(Boom(prices, causal_features))
    results = aud.run_all()
    assert results[0].status == Status.FAIL
    assert "boom" in results[0].message


def test_skeleton_checks_skip_cleanly():
    from tslab.audit.leakage import SessionBoundaryCheck, ShuffledTargetCheck
    assert SessionBoundaryCheck().run().status == Status.SKIP
    assert ShuffledTargetCheck().run().status == Status.SKIP
