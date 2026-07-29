"""Time-series split logic: embargoed train/test and rolling walk-forward.

No split lets test data precede or touch its training data: an embargo gap is
inserted between train end and test start. Walk-forward windows are generated on
month boundaries and are strictly time-ordered.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class TimeSplit:
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    def is_clean(self) -> bool:
        """No leakage: test strictly after train, train non-empty, test non-empty."""
        return self.train_start <= self.train_end < self.test_start <= self.test_end


def _month_index(d: date) -> int:
    return d.year * 12 + (d.month - 1)


def _from_index(idx: int) -> date:
    return date(idx // 12, idx % 12 + 1, 1)


def _month_end(d: date) -> date:
    return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])


def embargoed_split(
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    embargo_days: int,
) -> TimeSplit:
    """Push ``test_start`` past an embargo gap from ``train_end``."""
    gapped = train_end + timedelta(days=embargo_days + 1)
    return TimeSplit(train_start, train_end, max(test_start, gapped), test_end)


def walk_forward_splits(
    start: date,
    end: date,
    *,
    train_months: int,
    test_months: int,
    step_months: int,
    embargo_days: int,
) -> list[TimeSplit]:
    """Rolling walk-forward windows over [start, end] on month boundaries."""
    splits: list[TimeSplit] = []
    start_idx, end_idx = _month_index(start), _month_index(end)
    cursor = start_idx
    while True:
        train_end_idx = cursor + train_months - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_months - 1
        if test_end_idx > end_idx:
            break
        split = TimeSplit(
            train_start=_from_index(cursor),
            train_end=_month_end(_from_index(train_end_idx)),
            test_start=_from_index(test_start_idx) + timedelta(days=embargo_days),
            test_end=_month_end(_from_index(test_end_idx)),
        )
        splits.append(split)
        cursor += step_months
    return splits
