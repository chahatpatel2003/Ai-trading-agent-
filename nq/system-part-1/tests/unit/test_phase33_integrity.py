"""Phase 33 — scalp-engine integrity.

(1) Scalp families never reference a forward-outcome column (tL_/tS_/sL_/sS_/hL/hS).
(2) The forward-scalp engine uses ONLY same-trading-day future bars: a bar's first-touch offset
    changes with a later same-day bar, and is unchanged by past or next-day bars (no EOD crossing).
"""

from __future__ import annotations

import polars as pl

from nqlab.discovery.phase33 import families
from nqlab.research.scalp import add_scalp_outcomes

_FWD_PREFIX = ("tL_", "tS_", "sL_", "sS_", "hL", "hS")


def test_families_dont_reference_forward_columns() -> None:
    for fam in families():
        for expr in (fam.long, fam.short):
            roots = expr.meta.root_names()
            assert not [r for r in roots if r.startswith(_FWD_PREFIX)], (
                f"{fam.name} leaks forward col"
            )


def _synth() -> pl.DataFrame:
    rows = []
    from datetime import date

    for d in range(2):
        for b in range(30):
            px = 100.0 + d * 50 + b * 0.5
            rows.append(
                {
                    "d": date(2024, 1, 2 + d),
                    "open": px,
                    "high": px + 1,
                    "low": px - 1,
                    "close": px + 0.25,
                }
            )
    return pl.DataFrame(rows)


def test_scalp_outcomes_future_within_day_only() -> None:
    base = add_scalp_outcomes(_synth(), tf_min=0.25, hold_min=2.0)
    from datetime import date

    day1 = base.with_row_index("ix").filter(pl.col("d") == date(2024, 1, 2))
    k = day1["ix"].to_list()[5]
    ref = base["tL_1_0"][k]  # offset to first bar reaching entry+1 long

    def touch(mut: pl.DataFrame) -> int:
        return add_scalp_outcomes(mut, tf_min=0.25, hold_min=2.0)["tL_1_0"][k]

    raw = _synth()
    # mutate a PAST bar -> unchanged
    past = raw.with_columns(
        pl.when(pl.int_range(pl.len()) == k - 2)
        .then(pl.col("high") - 50)
        .otherwise(pl.col("high"))
        .alias("high")
    )
    assert touch(past) == ref
    # mutate NEXT-DAY bar -> unchanged (no EOD crossing)
    nxt = raw.with_columns(
        pl.when(pl.col("d") == date(2024, 1, 3))
        .then(pl.col("high") + 999)
        .otherwise(pl.col("high"))
        .alias("high")
    )
    assert touch(nxt) == ref
    # crush all future same-day highs -> the target can no longer be reached -> offset changes to 0
    fut = raw.with_columns(
        pl.when((pl.col("d") == date(2024, 1, 2)) & (pl.int_range(pl.len()) > k))
        .then(pl.col("high") - 100)
        .otherwise(pl.col("high"))
        .alias("high")
    )
    assert touch(fut) != ref
