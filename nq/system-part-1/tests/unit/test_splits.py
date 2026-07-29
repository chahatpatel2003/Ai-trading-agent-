from __future__ import annotations

from datetime import date
from itertools import pairwise

from nqlab.validation.splits import embargoed_split, walk_forward_splits


def test_embargoed_split_has_no_overlap() -> None:
    split = embargoed_split(
        date(2021, 1, 1), date(2024, 12, 31), date(2025, 1, 1), date(2026, 12, 31), embargo_days=3
    )
    assert split.is_clean()
    assert split.test_start > split.train_end
    assert (split.test_start - split.train_end).days >= 3


def test_walk_forward_windows_are_clean_and_ordered() -> None:
    splits = walk_forward_splits(
        date(2021, 1, 1), date(2023, 12, 31),
        train_months=12, test_months=3, step_months=3, embargo_days=2,
    )
    assert splits
    for sp in splits:
        assert sp.is_clean()  # test strictly after train (no lookahead)
        assert sp.test_start > sp.train_end
    for earlier, later in pairwise(splits):
        assert later.train_start > earlier.train_start


def test_walk_forward_stays_within_range() -> None:
    end = date(2022, 12, 31)
    splits = walk_forward_splits(
        date(2021, 1, 1), end, train_months=12, test_months=3, step_months=3, embargo_days=0
    )
    for sp in splits:
        assert sp.test_end <= end


def test_embargo_gap_widens_test_start() -> None:
    tr_end, te_start, te_end = date(2024, 12, 31), date(2025, 1, 1), date(2025, 6, 1)
    no_gap = embargoed_split(date(2021, 1, 1), tr_end, te_start, te_end, 0)
    gapped = embargoed_split(date(2021, 1, 1), tr_end, te_start, te_end, 10)
    assert gapped.test_start > no_gap.test_start
