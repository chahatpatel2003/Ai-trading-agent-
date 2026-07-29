from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from tslab import sessions as S

UTC = ZoneInfo("UTC")
CT = ZoneInfo("America/Chicago")


def utc(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


def ct(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=CT)


def test_session_date_rolls_at_17ct_winter(cfg):
    sc = cfg.sessions
    # Jan (CST, UTC-6): 16:59 CT belongs to same day; 17:00 CT to next
    assert S.session_date(ct(2024, 1, 10, 16, 59), sc) == date(2024, 1, 10)
    assert S.session_date(ct(2024, 1, 10, 17, 0), sc) == date(2024, 1, 11)


def test_session_date_rolls_at_17ct_summer(cfg):
    sc = cfg.sessions
    # Jul (CDT, UTC-5): 22:00 UTC = 17:00 CT
    assert S.session_date(utc(2024, 7, 10, 21, 59), sc) == date(2024, 7, 10)
    assert S.session_date(utc(2024, 7, 10, 22, 0), sc) == date(2024, 7, 11)


def test_sunday_open_is_monday_session(cfg):
    sc = cfg.sessions
    # Sunday 2024-01-07 17:30 CT -> Monday 2024-01-08
    assert S.session_date(ct(2024, 1, 7, 17, 30), sc) == date(2024, 1, 8)


def test_saturday_maps_forward_to_monday(cfg):
    sc = cfg.sessions
    assert S.session_date(ct(2024, 1, 6, 12, 0), sc) == date(2024, 1, 8)


def test_eod_flatten_utc(cfg):
    sc = cfg.sessions
    # 15:10 CT: Jan = 21:10 UTC (CST), Jul = 20:10 UTC (CDT)
    assert S.eod_flatten_utc(date(2024, 1, 10), sc) == utc(2024, 1, 10, 21, 10)
    assert S.eod_flatten_utc(date(2024, 7, 10), sc) == utc(2024, 7, 10, 20, 10)


def test_session_open_utc(cfg):
    sc = cfg.sessions
    # Monday 2024-01-08 session opened Sunday 17:00 CT = 23:00 UTC
    assert S.session_open_utc(date(2024, 1, 8), sc) == utc(2024, 1, 7, 23, 0)


def test_is_trading_time(cfg):
    sc = cfg.sessions
    assert S.is_trading_time(ct(2024, 1, 10, 10, 0), sc)          # Wed morning
    assert not S.is_trading_time(ct(2024, 1, 10, 16, 30), sc)     # maintenance
    assert S.is_trading_time(ct(2024, 1, 10, 17, 30), sc)         # evening reopen
    assert not S.is_trading_time(ct(2024, 1, 6, 12, 0), sc)       # Saturday
    assert not S.is_trading_time(ct(2024, 1, 7, 16, 59), sc)      # Sunday pre-open
    assert S.is_trading_time(ct(2024, 1, 7, 17, 0), sc)           # Sunday open
    assert not S.is_trading_time(ct(2024, 1, 12, 16, 30), sc)     # Friday post-close


def test_must_be_flat_windows(cfg):
    sc = cfg.sessions
    assert not S.must_be_flat(ct(2024, 1, 10, 14, 0), sc)         # mid-afternoon OK
    assert S.must_be_flat(ct(2024, 1, 10, 15, 10), sc)            # cutoff
    assert S.must_be_flat(ct(2024, 1, 10, 16, 30), sc)            # maintenance
    assert not S.must_be_flat(ct(2024, 1, 10, 17, 0), sc)         # reopen
    assert S.must_be_flat(ct(2024, 1, 12, 20, 0), sc)             # Friday night
    assert S.must_be_flat(ct(2024, 1, 6, 3, 0), sc)               # Saturday
    assert S.must_be_flat(ct(2024, 1, 7, 16, 0), sc)              # Sunday pre-open
    assert not S.must_be_flat(ct(2024, 1, 7, 17, 5), sc)          # Sunday open


def test_entry_window(cfg):
    sc = cfg.sessions
    assert S.in_entry_window(ct(2024, 1, 10, 9, 0), sc)
    assert not S.in_entry_window(ct(2024, 1, 10, 15, 0), sc)      # after entry end
    assert not S.in_entry_window(ct(2024, 1, 10, 5, 0), sc)       # before start


def test_naive_datetime_rejected(cfg):
    import pytest
    with pytest.raises(ValueError):
        S.session_date(datetime(2024, 1, 10, 12, 0), cfg.sessions)


# --------------------------------------------------------------------- #
# trade-date argument guards
# --------------------------------------------------------------------- #

def test_trade_date_functions_reject_timestamps(cfg):
    import pytest
    sc = cfg.sessions
    # a datetime is a date subclass: datetime.combine would silently use only
    # the calendar-date part and return the WRONG session's cutoff
    with pytest.raises(TypeError):
        S.eod_flatten_utc(utc(2024, 1, 8, 23, 30), sc)
    with pytest.raises(TypeError):
        S.session_open_utc(datetime(2024, 1, 8, 12, 0), sc)
    with pytest.raises(ValueError):
        S.eod_flatten_utc(date(2024, 1, 6), sc)          # Saturday
    with pytest.raises(ValueError):
        S.globex_close_utc(date(2024, 1, 7), sc)         # Sunday


# --------------------------------------------------------------------- #
# holidays / early closes (Memorial Day 2024-05-27 as the fixture)
# --------------------------------------------------------------------- #

def holiday_cfg(cfg):
    from dataclasses import replace
    from datetime import time as dtime
    mem = date(2024, 5, 27)
    return replace(cfg.sessions, holidays=(mem,),
                   early_closes={mem: dtime(12, 0)}), mem


def test_holiday_is_not_a_trade_date(cfg):
    sc, mem = holiday_cfg(cfg)
    # Monday-holiday morning activity books to TUESDAY (CME clearing closed)
    assert S.session_date(ct(2024, 5, 27, 10, 0), sc) == date(2024, 5, 28)
    # Sunday evening before the holiday also books to Tuesday
    assert S.session_date(ct(2024, 5, 26, 18, 0), sc) == date(2024, 5, 28)


def test_tuesday_after_holiday_opens_sunday(cfg):
    sc, mem = holiday_cfg(cfg)
    # Tuesday 2024-05-28's session opened Sunday 17:00 CT (CDT = 22:00 UTC)
    assert S.session_open_utc(date(2024, 5, 28), sc) == utc(2024, 5, 26, 22, 0)


def test_early_close_halts_trading_and_forces_flat(cfg):
    sc, mem = holiday_cfg(cfg)
    assert S.is_trading_time(ct(2024, 5, 27, 11, 0), sc)      # open pre-halt
    assert not S.is_trading_time(ct(2024, 5, 27, 13, 0), sc)  # halted
    assert not S.must_be_flat(ct(2024, 5, 27, 11, 0), sc)
    assert S.must_be_flat(ct(2024, 5, 27, 12, 0), sc)         # halt onward
    assert S.must_be_flat(ct(2024, 5, 27, 14, 0), sc)
    assert not S.must_be_flat(ct(2024, 5, 27, 17, 30), sc)    # evening reopen
    assert S.is_trading_time(ct(2024, 5, 27, 17, 30), sc)


def test_full_closure_holiday(cfg):
    from dataclasses import replace
    xmas = date(2024, 12, 25)
    sc = replace(cfg.sessions, holidays=(xmas,))
    assert not S.is_trading_time(ct(2024, 12, 25, 10, 0), sc)
    assert S.must_be_flat(ct(2024, 12, 25, 10, 0), sc)
    assert S.session_date(ct(2024, 12, 25, 10, 0), sc) == date(2024, 12, 26)
    import pytest
    with pytest.raises(ValueError):
        S.eod_flatten_utc(xmas, sc)                           # not a trade date


def test_validate_calendar_flags_unpopulated_holidays(cfg):
    import pytest
    from dataclasses import replace
    # an EMPTY calendar must be rejected for ranges containing likely holidays
    empty = replace(cfg.sessions, holidays=(), early_closes={})
    with pytest.raises(ValueError, match="2024-05-27"):
        S.validate_calendar(date(2024, 5, 1), date(2024, 6, 1), empty)
    # a holiday-free window needs no calendar entries
    S.validate_calendar(date(2024, 8, 1), date(2024, 8, 20), empty)


def test_shipped_calendar_covers_2016_through_2026(cfg):
    # the populated calendar must satisfy the validator over the full data
    # range — including observance-shifted years (2017/2023 New Year Monday,
    # 2021 Christmas Friday, 2020/2026 July 3, 2022 Juneteenth Monday)
    S.validate_calendar(date(2016, 1, 1), date(2026, 12, 31), cfg.sessions)


def test_known_us_market_holidays_2024(cfg):
    hs = S.known_us_market_holidays(2024)
    assert date(2024, 5, 27) in hs        # Memorial Day
    assert date(2024, 3, 29) in hs        # Good Friday (Easter 2024-03-31)
    assert date(2024, 11, 28) in hs       # Thanksgiving
    assert date(2024, 12, 25) in hs
