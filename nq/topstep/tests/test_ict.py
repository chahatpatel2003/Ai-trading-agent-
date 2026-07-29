from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from tslab.strategy.ict_displacement import (BODY_MIN, CAP, RANGE_ATR,
                                             build_pois, poi_hit, resample,
                                             run_ict_backtest)


def make_day(cfg, session, closes, start_ct_hour=8, start_min=30, base=17000.0):
    """1m bars for one session starting 08:30 CT (CST => 14:30 UTC)."""
    idx = pd.date_range(pd.Timestamp(session, tz="UTC")
                        + pd.Timedelta(hours=start_ct_hour + 6,
                                       minutes=start_min),
                        periods=len(closes), freq="1min")
    c = np.asarray(closes, float) + base
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.5,
                       "low": np.minimum(o, c) - 0.5, "close": c,
                       "volume": 100, "session": session, "symbol": "NQH4"},
                      index=idx)
    df["info_time"] = df.index + pd.Timedelta(seconds=60)
    return df


def test_resample_3m_causal_grouping(cfg):
    day = make_day(cfg, date(2024, 1, 9), np.arange(9.0))
    b3 = resample(day, 3)
    assert len(b3) == 3
    assert b3["open"].iloc[0] == day["open"].iloc[0]
    assert b3["close"].iloc[0] == day["close"].iloc[2]
    assert b3["high"].iloc[0] == day["high"].iloc[:3].max()


def flat_then_bull_disp():
    """Up first-15m candle (locks LONG bias), calm tape (small ATR), one
    strong 3m up candle, retrace to 0.5, rally through target."""
    closes = list(np.linspace(0.2, 3.0, 15))   # bias candle: 17000 -> 17003
    closes += [3.0] * 75                       # calm (25 x 3m bars, ATR ~1)
    closes += [7.0, 11.0, 15.0]                # displacement 3m bar
    closes += [15.0] * 3                       # leg completes
    closes += [9.0] * 3                        # retrace -> fill at 0.5
    closes += [23.0] * 6                       # through target
    closes += [23.0] * 3
    return closes


def test_bias_long_lock_and_displacement_trade(cfg):
    sess = date(2024, 1, 9)
    day = make_day(cfg, sess, flat_then_bull_disp())
    trades, fail, meta = run_ict_backtest(day, cfg, "MNQ")
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == 1
    assert t.exit_reason == "target"
    # leg 3m candle: o=17003 (prev close), c=17015, wick hi 17015.5,
    # body low = 17003 -> R = 12.5 (body-low -> wick-high, NOT full candle)
    assert t.risk_usd / t.qty == pytest.approx(12.5 * 2.0)  # MNQ $2/pt
    assert t.entry_px == pytest.approx(17003 + 12.5 * 0.5)
    assert meta["examples"]                                  # example captured


def test_counter_bias_never_trades(cfg):
    sess = date(2024, 1, 9)
    closes = flat_then_bull_disp()
    closes[0:15] = list(np.linspace(-0.2, -6, 15))  # first 15m closes DOWN
    closes[15:90] = [-6.0] * 75
    day = make_day(cfg, sess, closes)
    trades, fail, _ = run_ict_backtest(day, cfg, "MNQ")
    assert all(t.direction == -1 for t in trades)  # only short-bias trades
    # the bullish displacement leg must NOT have produced a long
    assert not any(t.direction == 1 for t in trades)


def test_doji_day_no_trades(cfg):
    sess = date(2024, 1, 9)
    closes = flat_then_bull_disp()
    closes[14] = closes[0]                         # 15m close == open (doji)
    day = make_day(cfg, sess, closes)
    trades, fail, _ = run_ict_backtest(day, cfg, "MNQ")
    assert trades == [] and fail.doji_days == 1


def test_stop_before_entry_cancels(cfg):
    sess = date(2024, 1, 9)
    closes = list(np.linspace(0.2, 3.0, 15)) + [3.0] * 75
    closes += [7.0, 11.0, 15.0] + [15.0] * 3
    closes += [-5.0] * 3                           # through entry AND stop
    closes += [0.0] * 6
    day = make_day(cfg, sess, closes)
    trades, fail, _ = run_ict_backtest(day, cfg, "MNQ")
    # same-bar entry+stop resolves conservatively as an immediate stop
    assert len(trades) == 1 and trades[0].exit_reason == "stop"


def test_nq_unsizeable_skipped(cfg):
    sess = date(2024, 1, 9)
    closes = list(np.linspace(0.2, 3.0, 15)) + [3.0] * 75
    closes += [13.0, 23.0, 33.0] + [33.0] * 3      # huge leg
    closes += [18.0] * 3 + [52.0] * 6
    day = make_day(cfg, sess, closes)
    trades, fail, _ = run_ict_backtest(day, cfg, "NQ")
    # R = 30.5 pts -> NQ $610 > $333 cap -> skipped (the later rally bar
    # forms a second unsizeable leg, also correctly skipped)
    assert trades == [] and fail.unsizeable_nq == 2
    trades_m, _, _ = run_ict_backtest(day, cfg, "MNQ")
    assert len(trades_m) == 1                      # MNQ sizes it
    assert trades_m[0].qty == int(CAP // (30.5 * 2.0))


def test_poi_fvg_causal():
    idx = pd.date_range("2024-01-09 14:30", periods=6, freq="15min", tz="UTC")
    h = np.array([10.0, 11, 15, 16, 17, 18])
    l = np.array([8.0, 9, 12.5, 14, 15, 16])      # candle2 low > candle0 high
    htf = pd.DataFrame({"open": l + 0.5, "high": h, "low": l, "close": h - 0.5,
                        "session": date(2024, 1, 9)}, index=idx)
    pois = build_pois(htf, 15)
    fvg = [p for p in pois if p["kind"] == "fvg" and p["side"] == 1]
    assert fvg and fvg[0]["lo"] == 10.0 and fvg[0]["hi"] == 12.5
    # NOT usable before candle 2 completes (14:30 + 45min)
    assert not poi_hit(pois, 1, 11.0, pd.Timestamp("2024-01-09 15:00Z"), 0.1)
    assert poi_hit(pois, 1, 11.0, pd.Timestamp("2024-01-09 15:15Z"), 0.1)


def bull_day_with_fvg_and_tap(cfg, sess=date(2024, 1, 9)):
    """Engineered day: LONG bias; a 15m bullish FVG forms mid-morning; price
    taps it; a displacement leg starts right after the tap."""
    closes = list(np.linspace(0.2, 3.0, 15))       # bias LONG
    closes += [3.0] * 30                           # 2 calm 15m candles
    # FVG construction on 15m: candle A high ~3.5, gap candle, candle C low
    # must exceed A's high: jump the tape to 12 for 15m, then hold
    closes += list(np.linspace(6.0, 12.0, 15))     # candle C: low > 3.5
    closes += [12.0] * 30                          # calm; FVG zone ~[3.5, 6.5]
    closes += [5.0] * 3                            # TAP into the FVG zone
    closes += [9.0, 13.0, 17.0]                    # displacement right after
    closes += [17.0] * 3 + [11.0] * 3 + [25.0] * 9
    return make_day(cfg, sess, closes)


def test_poi_tap_gates_displacement(cfg):
    from tslab.strategy.ict_displacement import run_ict_poi_backtest
    day = bull_day_with_fvg_and_tap(cfg)
    trades, fail, meta = run_ict_poi_backtest(day, cfg, "MNQ", 3)
    assert len(trades) >= 1                        # tap -> displacement -> trade
    assert fail.taps_total >= 1
    assert meta["poi_counts"]                      # POI type recorded
    assert all(k.endswith("15") or k.endswith("60")
               for k in meta["poi_counts"])


def test_no_tap_no_trade(cfg):
    from tslab.strategy.ict_displacement import run_ict_poi_backtest
    # v1 fixture has displacement but NO prior HTF POI tap in the window
    day = make_day(cfg, date(2024, 1, 9), flat_then_bull_disp())
    trades, fail, meta = run_ict_poi_backtest(day, cfg, "MNQ", 3)
    assert fail.legs_no_recent_tap >= 1
    # any trade that did occur must have been tap-gated
    assert len(trades) <= fail.taps_total


def test_ltf_resample_variants(cfg):
    day = make_day(cfg, date(2024, 1, 9), np.arange(20.0))
    assert len(resample(day, 2)) == 10
    assert len(resample(day, 5)) == 4
    b2 = resample(day, 2)
    assert b2["close"].iloc[0] == day["close"].iloc[1]


def tap_day_with_path(cfg, post_fill):
    """FVG-tap day whose post-fill path is caller-controlled (1m closes,
    offsets from base)."""
    closes = list(np.linspace(0.2, 3.0, 15)) + [3.0] * 30
    closes += list(np.linspace(6.0, 12.0, 15)) + [12.0] * 30
    closes += [5.0] * 3                            # tap FVG [3.5, 6.5]
    closes += [9.0, 13.0, 17.0] + [17.0] * 3       # leg: R=12.5 (17004.5->17017)
    closes += [11.0] * 3                           # fill at 0.5 = 17010.75
    closes += list(post_fill)
    return make_day(cfg, date(2024, 1, 9), closes)


def test_v3_fixed_2r_target(cfg):
    from tslab.strategy.ict_displacement import run_ict_poi_backtest
    # riskd = 12.5 -> 2R target = entry + 25 = 17035.75; rally there
    day = tap_day_with_path(cfg, [40.0] * 6)
    tr, _, _ = run_ict_poi_backtest(day, cfg, "MNQ", 3,
                                    exit_model="fixed", target_r=2.0)
    assert len(tr) == 1 and tr[0].exit_reason == "target"
    assert tr[0].exit_px == pytest.approx(tr[0].entry_px + 2 * 12.5)
    assert tr[0].gross_usd == pytest.approx(2 * 12.5 * 2.0 * tr[0].qty)


def test_v3_be1_breakeven_exit(cfg):
    from tslab.strategy.ict_displacement import run_ict_poi_backtest
    # touch +1R (17023.25), NEXT bars fall back to entry -> BE stop, gross ~0
    path = [23.5] * 3 + [10.5] * 3 + [10.5] * 6
    day = tap_day_with_path(cfg, path)
    tr, _, _ = run_ict_poi_backtest(day, cfg, "MNQ", 3, exit_model="be1")
    assert len(tr) == 1 and tr[0].exit_reason == "be"
    assert tr[0].gross_usd == pytest.approx(0.0)
    assert tr[0].net_usd < 0                       # costs only


def test_v3_partial_then_stop(cfg):
    from tslab.strategy.ict_displacement import run_ict_poi_backtest
    # +1R partial fills, then price collapses to the ORIGINAL stop
    path = [23.5] * 3 + [-9.0] * 3 + [0.0] * 6
    day = tap_day_with_path(cfg, path)
    tr, _, _ = run_ict_poi_backtest(day, cfg, "MNQ", 3, exit_model="p1r2")
    assert len(tr) == 1 and tr[0].exit_reason == "partial+stop"
    q = tr[0].qty
    q1, q2 = q // 2, q - q // 2
    want = (12.5 * 2.0 * q1) + (-12.5 * 2.0 * q2)
    assert tr[0].gross_usd == pytest.approx(want)


def test_v3_partial_then_runner_target(cfg):
    from tslab.strategy.ict_displacement import run_ict_poi_backtest
    path = [23.5] * 3 + [40.0] * 6
    day = tap_day_with_path(cfg, path)
    tr, _, _ = run_ict_poi_backtest(day, cfg, "MNQ", 3, exit_model="p1r2")
    assert len(tr) == 1 and tr[0].exit_reason == "partial+target"
    q = tr[0].qty
    q1, q2 = q // 2, q - q // 2
    want = (12.5 * 2.0 * q1) + (2 * 12.5 * 2.0 * q2)
    assert tr[0].gross_usd == pytest.approx(want)


def test_displacement_constants_disclosed():
    assert BODY_MIN == 0.70 and RANGE_ATR == 1.5   # frozen a-priori
