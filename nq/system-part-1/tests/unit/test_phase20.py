"""Phase 20 — indicator feature layer + indicator families + thresholds."""

from __future__ import annotations

import polars as pl

from nqlab.discovery.phase20 import PHASE20_THRESHOLDS, generate_phase20_candidates
from nqlab.discovery.phase20_families import PHASE20_FAMILIES
from nqlab.features.indicators import (
    add_adx,
    add_bollinger,
    add_donchian,
    add_macd,
    add_stoch,
    ema,
)


def _bars(n: int = 80) -> pl.DataFrame:
    px = [20000.0 + i * 2 + (5 if i % 3 else -4) for i in range(n)]
    return pl.DataFrame({
        "close": px,
        "high": [p + 3 for p in px],
        "low": [p - 3 for p in px],
        "atr": [10.0] * n,
    })


# --------------------------------------------------------------------------- indicators
def test_indicator_columns_added() -> None:
    f = _bars()
    assert "_macd_hist" in add_macd(f).columns
    assert {"_stoch_k", "_stoch_d"} <= set(add_stoch(f).columns)
    assert {"_bb_up", "_bb_dn", "_bb_bw"} <= set(add_bollinger(f).columns)
    assert {"_dc_up", "_dc_dn"} <= set(add_donchian(f).columns)
    assert {"adx", "plus_di", "minus_di"} <= set(add_adx(f).columns)


def test_ema_is_backward_looking() -> None:
    f = _bars(10).with_columns(ema(3).alias("e"))
    # EMA at row i must not depend on future rows: truncating the frame keeps the prefix EMA.
    full = f["e"].to_list()
    pref = _bars(10).head(5).with_columns(ema(3).alias("e"))["e"].to_list()
    assert full[:5] == pref


def test_donchian_uses_prior_bars_only() -> None:
    # _dc_up at row i = max high over bars [i-n .. i-1] (shifted) -> never includes the current bar.
    f = add_donchian(_bars(30), n=5)
    r = f.row(10, named=True)
    prior_max = max(_bars(30)["high"][5:10].to_list())
    assert abs(r["_dc_up"] - prior_max) < 1e-6


# --------------------------------------------------------------------------- families
def test_phase20_has_all_indicator_classes() -> None:
    fams = set(PHASE20_FAMILIES)
    for needed in ("p20_ema_cross", "p20_bb_meanrev", "p20_rsi_reversal", "p20_macd_cross",
                   "p20_stoch_reversal", "p20_adx_trend", "p20_atr_trend", "p20_donchian_brk",
                   "p20_keltner_meanrev", "p20_vwap_trend"):
        assert needed in fams
    assert len(fams) >= 12


def test_generate_candidates_ny_am_and_bounded() -> None:
    cands = generate_phase20_candidates(timeframe="5m")
    assert 20 <= len(cands) <= 60  # deliberately small grid (no overfitting)
    p = cands[0].p
    assert p["hour_start"] == 9 and p["hour_end"] == 12
    assert p["sizing_mode"] == "mnq_min1"


def test_ema_cross_family_emits_signals() -> None:
    strat = PHASE20_FAMILIES["p20_ema_cross"](
        {"fast": 9, "slow": 21, "stop_atr": 1.0, "target_atr": 1.5, "hour_start": 9,
         "hour_end": 12, "risk_per_trade": 250.0, "sizing_mode": "mnq_min1",
         "min_hold_bars": 1, "time_stop_bars": 18}
    )
    f = _bars(120).with_columns(
        pl.lit(600).alias("minute_of_day"), pl.lit(value=True).alias("is_rth"),
    )
    sig = strat.signals(f)
    assert {"signal", "stop_pts", "target_pts"} <= set(sig.columns)
    assert sig.filter(pl.col("signal") != 0).height > 0


# --------------------------------------------------------------------------- thresholds
def test_phase20_thresholds_are_payout_grade() -> None:
    th = PHASE20_THRESHOLDS
    assert th.pf_min == 1.30
    assert th.green_month_min == 0.75
    assert th.min_payout_eligible_frac == 0.60
    assert th.win_rate_min == 0.0  # not globally gated (trend systems can pass on PF)
