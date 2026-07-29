from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nqlab.domain.instruments.symbology import (
    QUARTERLY_CODES,
    CalendarSpread,
    ContractError,
    Outright,
    classify,
    is_spread,
    parse_outright,
    parse_symbol,
    third_friday,
)


def test_parse_outright_nq() -> None:
    c = parse_outright("NQH4")
    assert c.root == "NQ"
    assert c.month_code == "H"
    assert c.month == 3
    assert c.year == 2024
    assert c.is_quarterly
    assert c.expiry == date(2024, 3, 15)
    assert c.canonical == "NQH2024"


def test_parse_outright_mnq() -> None:
    c = parse_outright("MNQZ5")
    assert c.root == "MNQ"
    assert c.month == 12
    assert c.year == 2025
    assert c.is_quarterly


def test_two_digit_year() -> None:
    assert parse_outright("NQH24").year == 2024


@pytest.mark.parametrize(
    ("year", "month", "expected"),
    [
        (2024, 3, date(2024, 3, 15)),
        (2024, 6, date(2024, 6, 21)),
        (2024, 12, date(2024, 12, 20)),
    ],
)
def test_third_friday(year: int, month: int, expected: date) -> None:
    assert third_friday(year, month) == expected


def test_parse_spread() -> None:
    s = parse_symbol("NQH4-NQM4")
    assert isinstance(s, CalendarSpread)
    assert s.root == "NQ"
    assert s.near.month_code == "H"
    assert s.far.month_code == "M"
    assert is_spread("NQH4-NQM4")
    assert classify("NQH4-NQM4") == "spread"


def test_outright_classification() -> None:
    assert classify("NQH4") == "outright"
    s = parse_symbol("NQU6")
    assert isinstance(s, Outright)
    assert s.year == 2026


def test_invalid_symbols() -> None:
    with pytest.raises(ContractError):
        parse_outright("ESZ4")  # wrong root
    with pytest.raises(ContractError):
        parse_symbol("garbage")
    with pytest.raises(ContractError):
        parse_symbol("NQH4-ESM4")  # inter-product spread unsupported


@given(
    root=st.sampled_from(["NQ", "MNQ"]),
    code=st.sampled_from(list(QUARTERLY_CODES)),
    digit=st.integers(min_value=0, max_value=9),
)
def test_year_inference_roundtrip(root: str, code: str, digit: int) -> None:
    contract = parse_outright(f"{root}{code}{digit}", year_base=2020)
    assert contract.root == root
    assert contract.month_code == code
    assert contract.year == 2020 + digit
