from __future__ import annotations

import os
from datetime import date, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from tslab.audit.leakage import LeakageAuditor, Status
from tslab.audit.pipeline_gate import (BarIntegrityCheck, PrefixInvarianceCheck,
                                       RollCausalityCheck)
from tslab.data.nq_pipeline import (FrontMonthSelector, PipelineParams,
                                    bars_from_trades, clean_quotes,
                                    clean_trades, finalize_bars,
                                    is_quarterly_outright, parse_contract,
                                    prior_trade_session, quote_aggregates,
                                    trade_sessions_in_range)

UTC = ZoneInfo("UTC")
P = PipelineParams()


def ts(s):
    return pd.Timestamp(s, tz="UTC")


# --------------------------------------------------------------------- #
# contracts + roll causality
# --------------------------------------------------------------------- #

def test_parse_contract():
    assert parse_contract("NQH4", 2024) == (2024, 3)
    assert parse_contract("NQZ6", 2024) == (2026, 12)
    assert parse_contract("NQM1", 2021) == (2021, 6)
    assert parse_contract("NQZ9", 2021) == (2029, 12)   # not the past 2019
    assert parse_contract("NQH24", 2024) == (2024, 3)
    assert parse_contract("NQH4-NQM4", 2024) is None    # spread
    assert parse_contract("GCJ4", 2024) is None
    assert is_quarterly_outright("NQU5") and not is_quarterly_outright("NQF5")


def test_roll_uses_prior_session_only():
    sel = FrontMonthSelector()
    d = [date(2024, 3, 4), date(2024, 3, 5), date(2024, 3, 6), date(2024, 3, 7)]
    assert sel.select(d[0]) is None                     # bootstrap: no prior
    sel.observe(d[0], {"NQH4": 1000, "NQM4": 10})
    assert sel.select(d[1]) == "NQH4"
    sel.observe(d[1], {"NQH4": 400, "NQM4": 900})       # volume flips on d1...
    assert sel.select(d[2]) == "NQM4"                   # ...selection moves on d2
    sel.observe(d[2], {"NQH4": 800, "NQM4": 700})       # flip back...
    assert sel.select(d[3]) == "NQM4"                   # ...never rolls backwards
    for e in sel.roll_log:
        assert e["decided_from_session"] < e["session"]


def test_roll_causality_check_flags_bad_log():
    good = [{"session": date(2024, 3, 5), "front": "NQH4", "previous": None,
             "decided_from_session": date(2024, 3, 4)}]
    assert RollCausalityCheck(good).run().status == Status.PASS
    bad = [{"session": date(2024, 3, 5), "front": "NQH4", "previous": None,
            "decided_from_session": date(2024, 3, 5)}]     # same-day info
    assert RollCausalityCheck(bad).run().status == Status.FAIL
    backwards = good + [{"session": date(2024, 3, 6), "front": "NQZ3",
                         "previous": "NQH4",
                         "decided_from_session": date(2024, 3, 5)}]
    assert RollCausalityCheck(backwards).run().status == Status.FAIL


# --------------------------------------------------------------------- #
# trade cleaning + bar building
# --------------------------------------------------------------------- #

def synth_trades():
    t0 = ts("2024-01-09 15:00:00")
    rows = []
    px = 17000.0
    for i in range(180):                       # 3 minutes, 1 trade/second
        side = "B" if i % 3 == 0 else ("A" if i % 3 == 1 else "N")
        rows.append((t0 + pd.Timedelta(seconds=i), px + 0.25 * (i % 5),
                     1 + i % 3, side, "NQH4"))
    return pd.DataFrame(rows, columns=["ts_event", "price", "size", "side",
                                       "symbol"])


def test_bad_trades_filtered_causally():
    df = synth_trades()
    df.loc[50, "price"] = 20000.0              # +17% spike
    df.loc[60, "size"] = 0
    df.loc[70, "price"] = -1.0
    out, stats = clean_trades(df, P)
    assert stats["n_dropped_size_px"] == 2
    assert stats["n_dropped_jump"] == 1
    assert 20000.0 not in out["price"].values
    # filter is causal: the median it compares against is shifted by one row,
    # so a clean first row always survives
    assert out.iloc[0]["price"] == df.iloc[0]["price"]


def test_bars_ordering_and_fields():
    bars = bars_from_trades(synth_trades(), P)
    assert len(bars) == 3
    assert bars.index.is_monotonic_increasing and bars.index.tz is not None
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["volume"] > 0).all()
    b0 = bars.iloc[0]
    assert b0["n_trades"] == 60
    assert b0["delta"] == b0["buy_vol"] - b0["sell_vol"]
    assert b0["buy_vol"] + b0["sell_vol"] < b0["volume"]   # 'N' prints counted
    assert b0["rv"] > 0
    assert bars["low"].iloc[0] <= b0["vwap"] <= bars["high"].iloc[0]


def test_bars_out_of_order_input_handled():
    df = synth_trades().sample(frac=1.0, random_state=1)   # shuffle
    bars = bars_from_trades(df, P)
    assert bars.index.is_monotonic_increasing
    ref = bars_from_trades(synth_trades(), P)
    pd.testing.assert_frame_equal(bars, ref)


# --------------------------------------------------------------------- #
# quote cleaning + aggregates
# --------------------------------------------------------------------- #

def synth_quotes():
    t0 = ts("2024-01-09 15:00:00")
    rows = []
    for i in range(120):
        bid = 17000.0 + 0.25 * (i % 3)
        rows.append((t0 + pd.Timedelta(seconds=i), bid, bid + 0.25,
                     5 + i % 4, 7 - i % 4))
    return pd.DataFrame(rows, columns=["ts_event", "bid_px_00", "ask_px_00",
                                       "bid_sz_00", "ask_sz_00"])


def test_spread_filtering():
    df = synth_quotes()
    df.loc[10, ["bid_px_00", "ask_px_00"]] = [17010.0, 17000.0]  # crossed
    df.loc[20, "ask_px_00"] = df.loc[20, "bid_px_00"] + 100.0    # 400 ticks wide
    clean, stats = clean_quotes(df, P, tick_size=0.25)
    assert stats["n_dropped"] == 1                # crossed dropped entirely
    assert stats["n_wide"] == 1
    aggs = quote_aggregates(clean, P)
    assert np.allclose(aggs["spread_mean_ticks"], 1.0)     # wide excluded
    assert aggs["n_quotes"].iloc[0] == 59         # crossed removed from count
    last = clean[clean["ts_event"] < ts("2024-01-09 15:01:00")].iloc[-1]
    assert aggs["bid_close"].iloc[0] == last["bid_px_00"]


# --------------------------------------------------------------------- #
# session assignment, holidays, info_time discipline
# --------------------------------------------------------------------- #

def test_trade_sessions_skip_weekends_and_holidays(cfg):
    sc = cfg.sessions
    assert trade_sessions_in_range(date(2024, 1, 12), date(2024, 1, 17), sc) \
        == [date(2024, 1, 12), date(2024, 1, 16), date(2024, 1, 17)]  # no MLK
    assert prior_trade_session(date(2024, 1, 16), sc) == date(2024, 1, 12)
    assert prior_trade_session(date(2024, 4, 1), sc) == date(2024, 3, 28)  # GF


def make_finalized(cfg, when, session):
    t0 = ts(when)
    df = pd.DataFrame({
        "ts_event": [t0 + pd.Timedelta(seconds=i) for i in range(120)],
        "price": 17000.0, "size": 1, "side": "B", "symbol": "NQH4"})
    bars = bars_from_trades(df, P)
    return finalize_bars(bars, session, "NQH4", cfg, P)


def test_info_time_and_session_fields(cfg):
    out = make_finalized(cfg, "2024-01-09 15:00:00", date(2024, 1, 9))
    assert (pd.DatetimeIndex(out["info_time"]) ==
            out.index + pd.Timedelta(seconds=60)).all()
    assert (out["session"] == date(2024, 1, 9)).all()
    assert out["rth"].all()                     # 09:00 CT is inside NQ RTH
    # 15:00:xx UTC = 09:00 CT -> inside entry window and before flatten
    assert out["entry_ok"].all()
    flat_utc = ts("2024-01-09 21:10:00")        # 15:10 CST
    assert out["secs_to_flatten"].iloc[0] == pytest.approx(
        (flat_utc - out["info_time"].iloc[0]).total_seconds())


def test_holiday_session_bars_book_to_tuesday(cfg):
    # bars during MLK Monday morning belong to Tuesday's trade date and pass
    # the integrity gate; Sunday-evening bars likewise
    out = pd.concat([
        make_finalized(cfg, "2024-01-15 00:30:00", date(2024, 1, 16)),  # Sun eve CT
        make_finalized(cfg, "2024-01-15 16:00:00", date(2024, 1, 16)),  # Mon 10 CT
    ])
    res = BarIntegrityCheck(out, cfg).run()
    assert res.status == Status.PASS, res.message


def test_integrity_rejects_wrong_session(cfg):
    # the builder itself refuses a holiday trade date outright...
    with pytest.raises(ValueError, match="holiday"):
        make_finalized(cfg, "2024-01-15 16:00:00", date(2024, 1, 15))
    # ...and a frame whose session column is corrupted after the fact FAILs
    out = make_finalized(cfg, "2024-01-15 16:00:00", date(2024, 1, 16))
    out["session"] = date(2024, 1, 15)                    # MLK
    assert BarIntegrityCheck(out, cfg).run().status == Status.FAIL


def test_integrity_rejects_broken_info_time(cfg):
    out = make_finalized(cfg, "2024-01-09 15:00:00", date(2024, 1, 9))
    out.loc[out.index[1], "info_time"] = out["info_time"].iloc[1] + \
        pd.Timedelta(seconds=60)
    assert BarIntegrityCheck(out, cfg).run().status == Status.FAIL


def test_integrity_rejects_entry_after_flatten(cfg):
    out = make_finalized(cfg, "2024-01-09 21:30:00", date(2024, 1, 9))  # 15:30 CT
    assert not out["entry_ok"].any()            # honest frame passes
    assert BarIntegrityCheck(out, cfg).run().status == Status.PASS
    out["entry_ok"] = True                      # forged frame fails
    assert BarIntegrityCheck(out, cfg).run().status == Status.FAIL


def test_early_close_flatten_deadline(cfg):
    # Black Friday 2024: deadline is the 12:15 CT halt (18:15 UTC), and the
    # entry flag must respect it even inside the normal entry window
    out = make_finalized(cfg, "2024-11-29 17:00:00", date(2024, 11, 29))  # 11am CT
    flat_utc = ts("2024-11-29 18:15:00")
    assert out["secs_to_flatten"].iloc[0] == pytest.approx(
        (flat_utc - out["info_time"].iloc[0]).total_seconds())
    late = make_finalized(cfg, "2024-11-29 18:20:00", date(2024, 11, 29))
    assert not late["entry_ok"].any()


def test_prefix_invariance_check():
    a = make_bars_frame(300)
    prefix = a.iloc[:200]
    assert PrefixInvarianceCheck(a, prefix).run().status == Status.PASS
    tampered = a.copy()
    tampered.iloc[10, tampered.columns.get_loc("close")] += 1.0
    assert PrefixInvarianceCheck(tampered, prefix).run().status == Status.FAIL


def make_bars_frame(n):
    idx = pd.date_range("2024-01-09 15:00", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"close": np.arange(n, dtype=float),
                         "info_time": idx + pd.Timedelta(minutes=1)}, index=idx)


# --------------------------------------------------------------------- #
# real-data smoke: read-only guarantee + end-to-end gate (drive required)
# --------------------------------------------------------------------- #

TRADES_DIR = Path("/Volumes/T9/nq data/raw/trades")
needs_drive = pytest.mark.skipif(not TRADES_DIR.exists(),
                                 reason="T9 drive not mounted")


@needs_drive
def test_build_survives_empty_seed_session(cfg):
    # the very first buildable session's seed (2020-12-31) has NO data:
    # the seed must yield empty volumes, the first session gets skipped
    # (no prior information), and the second session builds normally
    from tslab.data.nq_pipeline import build_nq_bars
    res = build_nq_bars(cfg, date(2021, 1, 4), date(2021, 1, 5),
                        include_quotes=False)
    assert res.stats["skipped_sessions"] == ["2021-01-04"]
    assert set(res.bars["session"]) == {date(2021, 1, 5)}
    assert len(res.bars) > 800


@needs_drive
def test_one_session_build_readonly_and_gated(cfg):
    from tslab.data.nq_pipeline import build_nq_bars
    raw = sorted(TRADES_DIR.glob("*.parquet"))
    before = {p: (p.stat().st_size, p.stat().st_mtime) for p in raw}

    res = build_nq_bars(cfg, date(2024, 1, 9), date(2024, 1, 9),
                        include_quotes=False)
    after = {p: (p.stat().st_size, p.stat().st_mtime) for p in raw}
    assert before == after                      # raw data untouched

    b = res.bars
    assert len(b) > 800                         # a full Globex session
    assert set(b["symbol"]) == {"NQH4"}
    assert (b["session"] == date(2024, 1, 9)).all()
    assert b.index.is_monotonic_increasing and not b.index.duplicated().any()
    results = (LeakageAuditor()
               .register(BarIntegrityCheck(b, cfg))
               .register(RollCausalityCheck(res.roll_log))).run_all()
    assert LeakageAuditor.passed(results), [r.message for r in results]
