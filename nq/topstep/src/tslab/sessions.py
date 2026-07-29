"""CME/Topstep session clock.

All rule times are defined in America/Chicago (CT); data is tz-aware UTC.
Conventions (CME):
  - A session opens 17:00 CT and is labeled with the NEXT calendar day's date.
    Sunday 17:00 CT open -> Monday trade date.
  - Session closes 16:00 CT; maintenance 16:00-17:00 CT.
  - Topstep: flat by 15:10 CT ("eod_flatten"), no positions until 17:00 CT.

Holiday handling: cfg.holidays lists dates that are NOT valid trade dates
(full closures, and day-session closures like Memorial Day where CME books
the activity to the next trade date). cfg.early_closes maps a calendar date
to its halt time (CT) when the day session stops early. Both must be
populated before backtesting a range — validate_calendar() enforces this
against a computed list of likely US market holidays.

Approximation notes (documented, deliberate for the foundation):
  - On a full-closure holiday we treat the evening reopen as 17:00 CT; some
    holidays reopen later. Refine per-holiday when the calendar is populated.

DST is handled by zoneinfo — rule times are fixed in CT, so UTC offsets shift
automatically.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from .config import SessionConfig

UTC = ZoneInfo("UTC")


def _tz(cfg: SessionConfig) -> ZoneInfo:
    return ZoneInfo(cfg.timezone)


def _require_aware(ts: datetime) -> datetime:
    if not isinstance(ts, datetime):
        raise TypeError(f"expected a timestamp, got {type(ts).__name__}")
    if ts.tzinfo is None:
        raise ValueError("naive datetime passed to sessions module; "
                         "all timestamps must be tz-aware (UTC data).")
    return ts


def _require_trade_date(sess: date, cfg: SessionConfig) -> date:
    """Guard functions that take a TRADE DATE: datetimes (incl. pd.Timestamp,
    a datetime subclass) are rejected — datetime.combine would silently use
    only the calendar-date part and produce a wrong-session answer for any
    evening timestamp. Weekends/holidays are not trade dates either."""
    if isinstance(sess, datetime):
        raise TypeError("pass a trade date (use session_date(ts, cfg)), "
                        "not a timestamp")
    if sess.weekday() >= 5:
        raise ValueError(f"{sess} is a weekend, not a trade date")
    if sess in cfg.holidays:
        raise ValueError(f"{sess} is a configured holiday, not a trade date")
    return sess


def _is_holiday(d: date, cfg: SessionConfig) -> bool:
    return d in cfg.holidays


def to_ct(ts: datetime, cfg: SessionConfig) -> datetime:
    return _require_aware(ts).astimezone(_tz(cfg))


def session_date(ts: datetime, cfg: SessionConfig) -> date:
    """Trade date the timestamp belongs to (17:00 CT roll rule; weekends and
    configured holidays are not trade dates and roll forward — e.g. all
    activity from Sunday evening through a Monday-holiday halt books to
    Tuesday, matching CME clearing).

    Timestamps in dead zones (weekend, maintenance break) still map to the
    nominal next session; use is_trading_time() to check validity.
    """
    ct = to_ct(ts, cfg)
    d = ct.date()
    if ct.time() >= cfg.globex_open_ct:
        d = d + timedelta(days=1)
    while d.weekday() >= 5 or _is_holiday(d, cfg):
        d = d + timedelta(days=1)
    return d


def session_open_utc(sess: date, cfg: SessionConfig) -> datetime:
    """UTC instant this trade date's session opened: 17:00 CT on the previous
    non-blocked calendar day (Sunday for Monday sessions; Sunday for the
    Tuesday after a Monday holiday)."""
    _require_trade_date(sess, cfg)
    prior = sess - timedelta(days=1)
    # Walk back over Saturdays and HALT holidays (holiday + early close):
    # their whole stretch (Sun 17:00 -> holiday halt -> reopen) books to this
    # trade date. FULL-closure holidays are NOT skipped: they reopen at
    # 17:00 CT on the holiday itself (empirically confirmed), so the session
    # after e.g. Christmas Wednesday opens Wednesday 17:00 CT.
    while prior.weekday() == 5 or (_is_holiday(prior, cfg)
                                   and prior in cfg.early_closes):
        prior = prior - timedelta(days=1)
    local = datetime.combine(prior, cfg.globex_open_ct, tzinfo=_tz(cfg))
    return local.astimezone(UTC)


def globex_close_utc(sess: date, cfg: SessionConfig) -> datetime:
    _require_trade_date(sess, cfg)
    local = datetime.combine(sess, cfg.globex_close_ct, tzinfo=_tz(cfg))
    return local.astimezone(UTC)


def eod_flatten_utc(sess: date, cfg: SessionConfig) -> datetime:
    """Topstep flatten deadline for a trade date, in UTC: 15:10 CT normally,
    but on early-close trade dates (day after Thanksgiving, Christmas Eve,
    July 3) the Globex halt comes first — the deadline is the earlier of
    the two."""
    _require_trade_date(sess, cfg)
    t = cfg.eod_flatten_ct
    halt = cfg.early_closes.get(sess)
    if halt is not None and halt < t:
        t = halt
    local = datetime.combine(sess, t, tzinfo=_tz(cfg))
    return local.astimezone(UTC)


def is_trading_time(ts: datetime, cfg: SessionConfig) -> bool:
    """True if the exchange is open: Globex hours (Sun 17:00 CT - Fri 16:00
    CT) minus the daily 16:00-17:00 maintenance break, holiday closures, and
    early-close halts."""
    ct = to_ct(ts, cfg)
    d, wd, t = ct.date(), ct.weekday(), ct.time()
    if wd == 5:                                    # Saturday
        return False
    if wd == 6:                                    # Sunday: open from 17:00
        return t >= cfg.globex_open_ct
    if wd == 4 and t >= cfg.globex_close_ct:       # Friday after close
        return False
    if cfg.globex_close_ct <= t < cfg.globex_open_ct:   # daily maintenance
        return False
    halt = cfg.early_closes.get(d)
    if halt is not None and halt <= t < cfg.reopen_ct:  # early-close halt
        return False
    # FULL-closure holidays kill their entire nominal session, including the
    # PRIOR evening (no Thursday-evening trading before Good Friday; markets
    # reopen 17:00 CT on the holiday itself, which nominally belongs to the
    # next day). Halt holidays (in early_closes) trade through except the halt.
    nominal = d + timedelta(days=1) if t >= cfg.globex_open_ct else d
    if _is_holiday(nominal, cfg) and nominal not in cfg.early_closes:
        return False
    return True


def in_entry_window(ts: datetime, cfg: SessionConfig) -> bool:
    """True if new entries are allowed at ts under the configured research
    window (a strict subset of legal hours, ending before the flatten cutoff)."""
    if not is_trading_time(ts, cfg):
        return False
    t = to_ct(ts, cfg).time()
    return cfg.allowed_entry_start_ct <= t <= cfg.allowed_entry_end_ct


def must_be_flat(ts: datetime, cfg: SessionConfig) -> bool:
    """True if holding a position at ts violates the Topstep EOD rule:
    between 15:10 CT and the 17:00 CT reopen, across the weekend, through a
    holiday closure, and from an early-close halt to the reopen."""
    ct = to_ct(ts, cfg)
    d, wd, t = ct.date(), ct.weekday(), ct.time()
    if wd == 5:
        return True
    if wd == 6:
        return t < cfg.reopen_ct
    if cfg.eod_flatten_ct <= t < cfg.reopen_ct:
        return True
    if wd == 4 and t >= cfg.eod_flatten_ct:        # Friday after cutoff
        return True
    halt = cfg.early_closes.get(d)
    if halt is not None and halt <= t < cfg.reopen_ct:
        return True
    # entire nominal session of a full-closure holiday (incl. prior evening)
    nominal = d + timedelta(days=1) if t >= cfg.globex_open_ct else d
    if _is_holiday(nominal, cfg) and nominal not in cfg.early_closes:
        return True
    return False


def bars_in_session(index_utc, sess: date, cfg: SessionConfig):
    """Boolean mask over a tz-aware DatetimeIndex: bars belonging to a trade date."""
    import pandas as pd
    _require_trade_date(sess, cfg)
    idx = pd.DatetimeIndex(index_utc)
    if idx.tz is None:
        raise ValueError("index must be tz-aware UTC")
    lo = session_open_utc(sess, cfg)
    hi = globex_close_utc(sess, cfg)
    return (idx >= lo) & (idx < hi)


# --------------------------------------------------------------------------- #
# calendar validation
# --------------------------------------------------------------------------- #

def _easter(year: int) -> date:
    """Anonymous Gregorian computus."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 \
        else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date, saturday_to_friday: bool = True) -> Optional[date]:
    """US market observance shift: Sunday -> Monday; Saturday -> Friday
    (or no observance when saturday_to_friday is False, as for New Year's:
    when Jan 1 falls on Saturday the markets do not close on Dec 31)."""
    if d.weekday() == 6:
        return d + timedelta(days=1)
    if d.weekday() == 5:
        return d - timedelta(days=1) if saturday_to_friday else None
    return d


def known_us_market_holidays(year: int) -> List[date]:
    """Likely CME holiday/early-close dates (observance-shifted US market
    holidays). Used only to VALIDATE that the configured calendar is
    populated — not as the calendar itself (observed dates and halt times
    come from CME's recorded/announced schedule)."""
    out = [
        _observed(date(year, 1, 1), saturday_to_friday=False),  # New Year's
        _nth_weekday(year, 1, 0, 3),            # MLK Day
        _nth_weekday(year, 2, 0, 3),            # Presidents' Day
        _easter(year) - timedelta(days=2),      # Good Friday
        _last_weekday(year, 5, 0),              # Memorial Day
        _observed(date(year, 7, 4)),            # Independence Day
        _nth_weekday(year, 9, 0, 1),            # Labor Day
        _nth_weekday(year, 11, 3, 4),           # Thanksgiving
        _observed(date(year, 12, 25)),          # Christmas
    ]
    if year >= 2022:                            # CME first observed in 2022
        out.append(_observed(date(year, 6, 19)))  # Juneteenth
    return [d for d in out if d is not None]


def validate_calendar(start: date, end: date, cfg: SessionConfig) -> None:
    """Raise if [start, end] contains likely market holidays (weekdays) that
    the configured calendar does not cover. Backtests must call this before
    trusting session logic over a date range."""
    missing: List[date] = []
    for year in range(start.year, end.year + 1):
        for h in known_us_market_holidays(year):
            if start <= h <= end and h.weekday() < 5 \
                    and h not in cfg.holidays and h not in cfg.early_closes:
                missing.append(h)
    if missing:
        raise ValueError(
            "session calendar not populated for likely market holidays in "
            f"range {start}..{end}: {[d.isoformat() for d in missing]} — "
            "add them to sessions.yaml (holidays / early_closes)")
