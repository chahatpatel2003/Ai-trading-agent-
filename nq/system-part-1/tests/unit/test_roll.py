from __future__ import annotations

from datetime import date

from nqlab.data.continuous import select_active_contracts

_CONTRACTS = [("NQH4", date(2024, 3, 15)), ("NQM4", date(2024, 6, 21))]


def test_volume_triggers_roll() -> None:
    days = [date(2024, 2, 1), date(2024, 2, 2), date(2024, 2, 3)]
    volume = {
        ("NQH4", days[0]): 100.0, ("NQM4", days[0]): 10.0,
        ("NQH4", days[1]): 50.0, ("NQM4", days[1]): 60.0,  # next exceeds front -> roll
        ("NQH4", days[2]): 10.0, ("NQM4", days[2]): 100.0,
    }
    active, rolls = select_active_contracts(
        _CONTRACTS, volume, days, guard_days=8, ratio=1.0
    )
    assert active[days[0]] == "NQH4"
    assert active[days[1]] == "NQM4"
    assert active[days[2]] == "NQM4"
    assert rolls == [(days[1], "NQH4", "NQM4")]


def test_calendar_guard_forces_roll_against_volume() -> None:
    day = date(2024, 3, 10)  # within 8 days of NQH4 expiry (2024-03-15)
    volume = {("NQH4", day): 1000.0, ("NQM4", day): 1.0}  # volume says stay
    active, rolls = select_active_contracts(
        _CONTRACTS, volume, [day], guard_days=8, ratio=1.0
    )
    assert active[day] == "NQM4"
    assert rolls == [(day, "NQH4", "NQM4")]


def test_no_roll_when_front_dominant() -> None:
    day = date(2024, 2, 1)
    volume = {("NQH4", day): 1000.0, ("NQM4", day): 5.0}
    active, rolls = select_active_contracts(
        _CONTRACTS, volume, [day], guard_days=8, ratio=1.0
    )
    assert active[day] == "NQH4"
    assert rolls == []
