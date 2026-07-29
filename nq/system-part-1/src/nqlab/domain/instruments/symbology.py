"""CME NQ/MNQ futures symbology.

Parses Databento-style symbols (e.g. ``NQH4``, ``MNQZ5``, ``NQH4-NQM4``) into
typed contract objects, distinguishes **outrights** from **calendar spreads**,
and computes contract expiries (third Friday of the contract month).

Pure module: no I/O, no configuration, no global mutable state. The single-digit
year base is passed in by the caller (default 2020) so the function stays pure
and deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

# CME month codes -> calendar month number.
MONTH_CODES: dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
MONTH_BY_NUMBER: dict[int, str] = {v: k for k, v in MONTH_CODES.items()}

# NQ/MNQ list on the quarterly cycle.
QUARTERLY_CODES: tuple[str, ...] = ("H", "M", "U", "Z")

_DEFAULT_YEAR_BASE = 2020

# Root is NQ or MNQ; one month code; 1- or 2-digit year.
_OUTRIGHT_RE = re.compile(r"^(M?NQ)([FGHJKMNQUVXZ])(\d{1,2})$")


class ContractError(ValueError):
    """Raised when a symbol cannot be parsed as a NQ/MNQ contract."""


@dataclass(frozen=True, slots=True)
class Outright:
    """A single outright futures contract (e.g. ``NQH4`` = NQ March 2024)."""

    raw: str
    root: str
    month_code: str
    year: int

    @property
    def month(self) -> int:
        return MONTH_CODES[self.month_code]

    @property
    def expiry(self) -> date:
        """Final settlement reference date: third Friday of the contract month."""
        return third_friday(self.year, self.month)

    @property
    def is_quarterly(self) -> bool:
        return self.month_code in QUARTERLY_CODES

    @property
    def canonical(self) -> str:
        """Normalized identity, decade-explicit: e.g. ``NQH2024``."""
        return f"{self.root}{self.month_code}{self.year}"


@dataclass(frozen=True, slots=True)
class CalendarSpread:
    """A calendar (intra-commodity) spread, e.g. ``NQH4-NQM4``."""

    raw: str
    root: str
    near: Outright
    far: Outright

    @property
    def canonical(self) -> str:
        return f"{self.near.canonical}-{self.far.canonical}"


def third_friday(year: int, month: int) -> date:
    """Return the third Friday of ``year``/``month`` (NQ final-settlement date)."""
    first = date(year, month, 1)
    # Monday=0 ... Friday=4 ... Sunday=6
    offset_to_first_friday = (4 - first.weekday()) % 7
    return first + timedelta(days=offset_to_first_friday + 14)


def _infer_year(digits: str, year_base: int) -> int:
    if len(digits) == 1:
        return year_base + int(digits)
    return 2000 + int(digits)


def is_spread(symbol: str) -> bool:
    """True if the symbol denotes a multi-leg spread (contains ``-``)."""
    return "-" in symbol


def is_outright(symbol: str) -> bool:
    return not is_spread(symbol)


def parse_outright(symbol: str, year_base: int = _DEFAULT_YEAR_BASE) -> Outright:
    """Parse a single outright NQ/MNQ symbol. Raises :class:`ContractError`."""
    token = symbol.strip().upper()
    match = _OUTRIGHT_RE.match(token)
    if match is None:
        raise ContractError(f"Not a valid NQ/MNQ outright symbol: {symbol!r}")
    root, code, digits = match.groups()
    return Outright(raw=symbol, root=root, month_code=code, year=_infer_year(digits, year_base))


def parse_symbol(
    symbol: str, year_base: int = _DEFAULT_YEAR_BASE
) -> Outright | CalendarSpread:
    """Parse any NQ/MNQ symbol into an :class:`Outright` or :class:`CalendarSpread`."""
    token = symbol.strip().upper()
    if is_spread(token):
        legs = token.split("-")
        if len(legs) != 2:
            raise ContractError(f"Unsupported multi-leg spread: {symbol!r}")
        near = parse_outright(legs[0], year_base)
        far = parse_outright(legs[1], year_base)
        if near.root != far.root:
            raise ContractError(f"Inter-product spreads are not supported: {symbol!r}")
        return CalendarSpread(raw=symbol, root=near.root, near=near, far=far)
    return parse_outright(token, year_base)


def classify(symbol: str) -> str:
    """Return ``"spread"`` or ``"outright"`` (cheap; does not validate the root)."""
    return "spread" if is_spread(symbol) else "outright"
