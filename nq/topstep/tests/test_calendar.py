"""Phase 2A regression tests: the POPULATED 2016-2026 CME calendar in
configs/sessions.yaml, exercised through the real loaded config (no fixture
overrides). Dates/times here were verified against recorded Globex behavior
in the local GC/NQ data."""
from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tslab import sessions as S

UTC = ZoneInfo("UTC")
CT = ZoneInfo("America/Chicago")


def utc(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


def ct(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=CT)


# --------------------------------------------------------------------- #
# calendar contents
# --------------------------------------------------------------------- #

def test_calendar_is_populated(cfg):
    sc = cfg.sessions
    assert len(sc.holidays) == 103
    assert len(sc.early_closes) == 95
    assert min(sc.holidays).year == 2016 and max(sc.holidays).year == 2026


def test_full_closures_present(cfg):
    hs = set(cfg.sessions.holidays)
    ec = cfg.sessions.early_closes
    # all Good Fridays (2026's has a jobs-report mini session, rest are full)
    for d in (date(2016, 3, 25), date(2017, 4, 14), date(2018, 3, 30),
              date(2019, 4, 19), date(2020, 4, 10), date(2021, 4, 2),
              date(2022, 4, 15), date(2023, 4, 7), date(2024, 3, 29),
              date(2025, 4, 18)):
        assert d in hs and d not in ec, d
    assert date(2026, 4, 3) in hs
    assert ec[date(2026, 4, 3)] == time(8, 15)     # measured in NQ trades
    # observance shifts
    assert date(2021, 12, 24) in hs                # Christmas 2021 (Sat -> Fri)
    assert date(2022, 12, 26) in hs                # Christmas 2022 (Sun -> Mon)
    assert date(2023, 1, 2) in hs                  # New Year 2023 (Sun -> Mon)
    assert date(2021, 12, 31) not in hs            # New Year 2022 (Sat -> none)
    assert date(2020, 7, 3) in hs                  # July 4 2020 (Sat -> Fri)
    assert date(2021, 7, 5) in hs                  # July 4 2021 (Sun -> Mon)
    # Juneteenth only from 2022
    assert date(2022, 6, 20) in hs
    assert not any(d.year == 2021 and d.month == 6 for d in hs)


def test_early_close_times_match_measured_halts(cfg):
    ec = cfg.sessions.early_closes
    assert ec[date(2024, 1, 15)] == time(12, 0)    # MLK (NQ last trade 11:59:59)
    assert ec[date(2024, 11, 29)] == time(12, 15)  # Black Friday (12:14:59)
    assert ec[date(2024, 12, 24)] == time(12, 15)  # Christmas Eve
    assert ec[date(2023, 7, 3)] == time(12, 15)    # July 3 equity half day
    assert date(2016, 11, 25) in ec                # Black Fridays back to 2016
    # halt holidays are non-trade-dates; half days are trade dates
    hs = set(cfg.sessions.holidays)
    assert date(2024, 1, 15) in hs
    assert date(2024, 11, 29) not in hs
    assert date(2024, 12, 24) not in hs


# --------------------------------------------------------------------- #
# full holidays: closed, flat, not a trade date
# --------------------------------------------------------------------- #

def test_good_friday_fully_closed(cfg):
    sc = cfg.sessions
    gf = ct(2024, 3, 29, 10, 0)
    assert not S.is_trading_time(gf, sc)
    assert S.must_be_flat(gf, sc)
    assert S.session_date(gf, sc) == date(2024, 4, 1)      # rolls to Monday
    with pytest.raises(ValueError):
        S.eod_flatten_utc(date(2024, 3, 29), sc)


def test_no_session_on_eve_of_full_closure(cfg):
    sc = cfg.sessions
    # Thursday evening before Good Friday: Globex did NOT reopen at 17:00
    assert not S.is_trading_time(ct(2024, 3, 28, 20, 0), sc)
    assert S.must_be_flat(ct(2024, 3, 28, 20, 0), sc)
    # but Thursday's own day session was normal
    assert S.is_trading_time(ct(2024, 3, 28, 10, 0), sc)


def test_full_closure_evening_reopen(cfg):
    sc = cfg.sessions
    # Christmas Day 2024 (Wed): closed all day, reopens 17:00 CT that evening
    assert not S.is_trading_time(ct(2024, 12, 25, 10, 0), sc)
    assert S.is_trading_time(ct(2024, 12, 25, 18, 0), sc)
    # and the Thursday session opened Wednesday 17:00 CT (= 23:00 UTC, CST)
    assert S.session_open_utc(date(2024, 12, 26), sc) == utc(2024, 12, 25, 23, 0)


# --------------------------------------------------------------------- #
# halt holidays and half days
# --------------------------------------------------------------------- #

def test_mlk_halt_and_reopen(cfg):
    sc = cfg.sessions
    assert S.is_trading_time(ct(2024, 1, 15, 10, 0), sc)    # open pre-halt
    assert not S.is_trading_time(ct(2024, 1, 15, 12, 30), sc)
    assert S.must_be_flat(ct(2024, 1, 15, 12, 0), sc)
    assert S.is_trading_time(ct(2024, 1, 15, 17, 30), sc)   # evening reopen


def test_black_friday_is_short_trade_date(cfg):
    sc = cfg.sessions
    d = date(2024, 11, 29)
    assert S.session_date(ct(2024, 11, 29, 10, 0), sc) == d  # IS a trade date
    assert S.is_trading_time(ct(2024, 11, 29, 10, 0), sc)
    assert S.must_be_flat(ct(2024, 11, 29, 13, 0), sc)


# --------------------------------------------------------------------- #
# EOD cutoff correctness
# --------------------------------------------------------------------- #

def test_eod_cutoff_capped_by_early_close(cfg):
    sc = cfg.sessions
    # Black Friday 2024: deadline is the 12:15 CT halt, not 15:10 (CST -> UTC+6)
    assert S.eod_flatten_utc(date(2024, 11, 29), sc) == utc(2024, 11, 29, 18, 15)
    # Christmas Eve 2024 likewise
    assert S.eod_flatten_utc(date(2024, 12, 24), sc) == utc(2024, 12, 24, 18, 15)
    # normal day unchanged: 15:10 CT
    assert S.eod_flatten_utc(date(2024, 11, 26), sc) == utc(2024, 11, 26, 21, 10)


def test_eod_cutoff_rejected_on_holidays(cfg):
    for d in (date(2024, 1, 15), date(2024, 12, 25), date(2026, 4, 3)):
        with pytest.raises(ValueError):
            S.eod_flatten_utc(d, cfg.sessions)


# --------------------------------------------------------------------- #
# DST transitions
# --------------------------------------------------------------------- #

def test_dst_spring_forward_2024(cfg):
    sc = cfg.sessions
    # DST began Sun 2024-03-10. Friday (CST): 15:10 CT = 21:10 UTC.
    assert S.eod_flatten_utc(date(2024, 3, 8), sc) == utc(2024, 3, 8, 21, 10)
    # Monday (CDT): 15:10 CT = 20:10 UTC; session opened Sunday 22:00 UTC.
    assert S.eod_flatten_utc(date(2024, 3, 11), sc) == utc(2024, 3, 11, 20, 10)
    assert S.session_open_utc(date(2024, 3, 11), sc) == utc(2024, 3, 10, 22, 0)
    # roll boundary stays at CT wall clock across the change
    assert S.session_date(utc(2024, 3, 11, 21, 59), sc) == date(2024, 3, 11)
    assert S.session_date(utc(2024, 3, 11, 22, 0), sc) == date(2024, 3, 12)


def test_dst_fall_back_2024(cfg):
    sc = cfg.sessions
    # DST ended Sun 2024-11-03. Friday (CDT): 15:10 CT = 20:10 UTC.
    assert S.eod_flatten_utc(date(2024, 11, 1), sc) == utc(2024, 11, 1, 20, 10)
    # Monday (CST): 21:10 UTC; session opened Sunday 17:00 CST = 23:00 UTC.
    assert S.eod_flatten_utc(date(2024, 11, 4), sc) == utc(2024, 11, 4, 21, 10)
    assert S.session_open_utc(date(2024, 11, 4), sc) == utc(2024, 11, 3, 23, 0)


# --------------------------------------------------------------------- #
# trade-date assignment / no phantom trading days
# --------------------------------------------------------------------- #

def test_mlk_activity_books_to_tuesday(cfg):
    sc = cfg.sessions
    assert S.session_date(ct(2024, 1, 14, 18, 0), sc) == date(2024, 1, 16)
    assert S.session_date(ct(2024, 1, 15, 10, 0), sc) == date(2024, 1, 16)
    assert S.session_date(ct(2024, 1, 15, 18, 0), sc) == date(2024, 1, 16)
    # Tuesday's session opened Sunday 17:00 CST = 23:00 UTC
    assert S.session_open_utc(date(2024, 1, 16), sc) == utc(2024, 1, 14, 23, 0)


def test_no_phantom_trading_days_mlk_week(cfg):
    sc = cfg.sessions
    stamps = pd.date_range(utc(2024, 1, 14, 23, 0), utc(2024, 1, 16, 21, 59),
                           freq="30min")
    seen = {S.session_date(t.to_pydatetime(), sc)
            for t in stamps if S.is_trading_time(t.to_pydatetime(), sc)}
    assert date(2024, 1, 15) not in seen           # MLK is never a trade date
    assert seen == {date(2024, 1, 16)}


def test_no_phantom_trading_days_good_friday_week(cfg):
    sc = cfg.sessions
    stamps = pd.date_range(utc(2024, 3, 28, 0, 0), utc(2024, 4, 1, 20, 59),
                           freq="30min")
    seen = {S.session_date(t.to_pydatetime(), sc)
            for t in stamps if S.is_trading_time(t.to_pydatetime(), sc)}
    assert date(2024, 3, 29) not in seen           # Good Friday
    assert date(2024, 3, 30) not in seen           # Saturday
    assert date(2024, 3, 31) not in seen           # Sunday
    # Thu day session, then nothing until the Sunday 17:00 open of Mon Apr 1
    assert seen == {date(2024, 3, 28), date(2024, 4, 1)}
    assert S.session_date(ct(2024, 3, 31, 18, 0), sc) == date(2024, 4, 1)


def test_gc_bars_never_map_to_holiday_dates(cfg):
    """Every 2024 timestamp the exchange actually traded (per the calendar's
    own trading-time predicate) must map to a non-holiday trade date."""
    sc = cfg.sessions
    hs = set(cfg.sessions.holidays)
    stamps = pd.date_range(utc(2024, 1, 1, 0, 0), utc(2024, 12, 31, 23, 0),
                           freq="4h")
    for t in stamps:
        t = t.to_pydatetime()
        if S.is_trading_time(t, sc):
            sd = S.session_date(t, sc)
            assert sd not in hs, f"{t} -> {sd}"
            assert sd.weekday() < 5
