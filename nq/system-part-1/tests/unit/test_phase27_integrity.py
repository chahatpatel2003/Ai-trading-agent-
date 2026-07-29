"""Phase 27 — backtest integrity audit: prefix-invariance (no hindsight) for the NEW feature
layers (htf_poi, daily_bias, orderflow, indicators) and Phase-26 strategy signals, plus
engine stop-first conservative ordering and next-open entry.

The proof: a value at bar k computed on the FULL series must equal the value computed on only the
prefix [0..k]. If adding future bars ever changes a past value, that is hindsight -> FAIL.
"""

from __future__ import annotations

import polars as pl
import pytest

from nqlab.backtest.engine import run_backtest
from nqlab.config.models import AppConfig, TopstepRules
from nqlab.discovery.phase26_families import PHASE26_FAMILIES
from nqlab.domain.strategy.base import ExecConfig, StrategyParams
from nqlab.features.build import build_features
from nqlab.features.context import build_context_features
from nqlab.features.daily_bias import add_daily_bias
from nqlab.features.htf_context import build_htf_context_features
from nqlab.features.htf_poi import add_htf_poi
from nqlab.features.indicators import add_adx, add_bollinger, add_donchian
from nqlab.features.orderflow import add_orderflow_features
from nqlab.features.structure import build_structure_features
from tests.synth import add_synth_mbp1_columns, engine_frame, globex_bars

_RTH = 570


def _struct(bars: pl.DataFrame) -> pl.DataFrame:
    f = build_features(bars, rth_start_min=_RTH)
    f = build_context_features(f, rth_start_min=_RTH)
    f = build_htf_context_features(f, rth_start_min=_RTH, timeframe="5m")
    return build_structure_features(f)


def _assert_prefix_invariant(
    builder, bars: pl.DataFrame, cols: list[str], *, sample: int = 6
) -> None:
    full = _struct(bars)
    full_built = builder(full).with_row_index("idx")
    rth = full_built.filter(pl.col("is_rth"))
    idxs = rth["idx"].to_list()
    n = len(idxs)
    assert n > 0
    picks = sorted({idxs[int(f * (n - 1))] for f in (0.3, 0.45, 0.6, 0.75, 0.9)})[:sample]
    for k in picks:
        prefix = builder(_struct(bars.head(k + 1)))
        fr = full_built.drop("idx").row(k, named=True)
        pr = prefix.row(k, named=True)
        for c in cols:
            a, b = fr[c], pr[c]
            if a is None and b is None:
                continue
            assert a == pytest.approx(b, rel=1e-9, abs=1e-9), f"HINDSIGHT in {c} @k={k}: {a}!={b}"


def test_htf_poi_no_lookahead() -> None:
    _assert_prefix_invariant(
        lambda f: add_htf_poi(f, tol_pts=5.0), globex_bars(days=18),
        ["support_poi", "resistance_poi", "at_support_poi", "at_resistance_poi",
         "at_support_poi_recent", "at_resistance_poi_recent"],
    )


def test_daily_bias_no_lookahead() -> None:
    _assert_prefix_invariant(
        add_daily_bias, globex_bars(days=18),
        ["bias_a", "bias_b", "htf_bull", "htf_bear", "aligned_long_a", "aligned_short_a",
         "cisd_up", "cisd_dn", "ifvg_bull", "ifvg_bear"],
    )


def test_indicators_no_lookahead() -> None:
    _assert_prefix_invariant(
        lambda f: add_adx(add_donchian(add_bollinger(f))), globex_bars(days=14),
        ["_dc_up", "_dc_dn", "adx", "plus_di", "minus_di", "_bb_up", "_bb_dn"],
    )


def test_orderflow_no_lookahead() -> None:
    def build(f: pl.DataFrame) -> pl.DataFrame:
        # synth MBP1 supplies delta/cvd/tob_imbalance; add ofi (= signed_volume) for the derivation.
        m = add_synth_mbp1_columns(f).with_columns(pl.col("signed_volume").alias("ofi"))
        return add_orderflow_features(m)

    _assert_prefix_invariant(
        build, globex_bars(days=14),
        ["ofi_z", "delta_z", "cvd_mom_z", "imb_persist_up", "imb_persist_dn"],
    )


def test_phase26_signal_no_lookahead() -> None:
    # The end-to-end strategy SIGNAL must be prefix-invariant (the real leakage surface).
    strat = PHASE26_FAMILIES["p26_htf_poi"](
        {"stop_atr": 1.5, "target_atr": 4.5, "hour_start": 9, "hour_end": 16,
         "risk_per_trade": 150.0, "sizing_mode": "mnq_fixed1", "min_hold_bars": 1,
         "time_stop_bars": 60}
    )
    _assert_prefix_invariant(lambda f: strat.signals(f), globex_bars(days=16), ["signal"])


# --------------------------------------------------------------------------- engine ordering
def test_stop_checked_before_target_same_bar(
    app_config: AppConfig, rules_50k: TopstepRules
) -> None:
    """When a single bar touches BOTH stop and target, the engine must take the STOP
    (conservative ordering — no assuming the favorable fill happened first)."""
    bars = [
        {"close": 100, "high": 100, "low": 100, "signal": 1, "stop_pts": 5,
         "target_pts": 5, "minute": 600},
        # next bar opens at 100, then trades through BOTH stop (95) and target (105).
        {"open": 100, "high": 106, "low": 94, "close": 100, "minute": 605},
        {"open": 100, "high": 100, "low": 100, "close": 100, "minute": 955},
    ]
    res = run_backtest(
        engine_frame(bars), symbol="NQ.c", timeframe="5m", strategy=StrategyParams("t", "1"),
        config=app_config, rules=rules_50k, exec_config=ExecConfig(min_hold_bars=0),
    )
    assert res.trades[0].exit_reason == "STOP"  # conservative: loss assumed, not the win


def test_entry_is_next_open_after_signal(
    app_config: AppConfig, rules_50k: TopstepRules
) -> None:
    """A signal at bar t must fill at bar t+1's open (no same-bar signal+entry)."""
    bars = [
        {"close": 100, "high": 100, "low": 100, "signal": 1, "stop_pts": 10,
         "target_pts": 10, "minute": 600},
        {"open": 101, "high": 112, "low": 101, "close": 111, "minute": 605},  # target bar
        {"open": 111, "high": 111, "low": 111, "close": 111, "minute": 955},
    ]
    res = run_backtest(
        engine_frame(bars), symbol="NQ.c", timeframe="5m", strategy=StrategyParams("t", "1"),
        config=app_config, rules=rules_50k, exec_config=ExecConfig(min_hold_bars=0),
    )
    tr = res.trades[0]
    # Entry is on the bar AFTER the signal: filled at the 2nd bar's open (101 + slippage),
    # at minute 605 (not the signal bar's 600 / price 100). Same-bar intrabar target fill is
    # legitimate (price rose to target after the open); the proof of no-hindsight is that the
    # entry uses a price/time strictly after the signal bar.
    assert tr.entry_minute == 605
    assert tr.entry_px >= 101.0
    assert tr.exit_ts >= tr.entry_ts
