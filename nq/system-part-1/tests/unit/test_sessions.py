from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl

from nqlab.config.models import SessionConfig
from nqlab.domain.calendar.sessions import SessionLabeler, parse_hhmm


def _labeler() -> SessionLabeler:
    return SessionLabeler.from_config(SessionConfig())


def test_parse_hhmm() -> None:
    assert parse_hhmm("09:30") == 570
    assert parse_hhmm("18:00") == 1080


def test_session_of_minute() -> None:
    lab = _labeler()
    assert lab.session_of_minute(parse_hhmm("10:00")) == "NY_AM"
    assert lab.session_of_minute(parse_hhmm("13:00")) == "NY_PM"
    assert lab.session_of_minute(parse_hhmm("20:00")) == "ASIA"
    assert lab.session_of_minute(parse_hhmm("05:00")) == "LONDON"


def test_is_rth_minute() -> None:
    lab = _labeler()
    assert lab.is_rth_minute(parse_hhmm("10:00"))
    assert not lab.is_rth_minute(parse_hhmm("08:00"))
    assert not lab.is_rth_minute(parse_hhmm("16:00"))  # end is exclusive


def test_trading_day_boundary() -> None:
    lab = _labeler()
    # 18:00 ET belongs to the next calendar day's session.
    assert lab.trading_day_of(datetime(2024, 1, 2, 18, 0)) == date(2024, 1, 3)
    assert lab.trading_day_of(datetime(2024, 1, 2, 16, 0)) == date(2024, 1, 2)


def test_vectorized_calendar_columns() -> None:
    lab = _labeler()
    frame = pl.DataFrame({"ts": [datetime(2024, 1, 2, 14, 30, tzinfo=UTC)]})  # 09:30 ET
    row = lab.with_calendar_columns(frame, "ts").row(0, named=True)
    assert row["minute_of_day"] == 570
    assert row["is_rth"] is True
    assert row["session"] == "NY_AM"
    assert row["trading_day"] == date(2024, 1, 2)
