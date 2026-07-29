"""Phase 32 — edge-map integrity.

(1) Conditions are PIT-safe: no condition expression may reference a forward (``fwd_*``) column.
(2) The forward-outcome engine uses ONLY same-trading-day FUTURE bars: a bar's forward outcome must
    change when a later same-day bar changes, and must NOT change when a past bar or a next-day bar
    changes (no leakage into the past, no crossing the EOD boundary).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl

from nqlab.discovery.phase32 import conditions
from nqlab.research.forward import add_forward_outcomes


def test_conditions_never_reference_forward_columns() -> None:
    for cond in conditions():
        roots = cond.expr.meta.root_names()
        leaks = [r for r in roots if r.startswith("fwd_")]
        assert not leaks, f"condition {cond.name} leaks forward column(s): {leaks}"


def _synth(n_per_day: int = 40, days: int = 2) -> pl.DataFrame:
    rows = []
    base = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)  # 09:30 ET
    for d in range(days):
        for b in range(n_per_day):
            ts = base + timedelta(days=d, minutes=5 * b)
            px = 100.0 + d * 50 + b  # rising
            rows.append(
                {
                    "ts": ts,
                    "minute_of_day": 570 + 5 * b,
                    "trading_day": date(2024, 1, 2 + d),
                    "is_rth": True,
                    "open": px,
                    "high": px + 2,
                    "low": px - 2,
                    "close": px + 0.5,
                }
            )
    return pl.DataFrame(rows)


def test_forward_uses_only_future_within_day() -> None:
    fr = _synth()
    base = add_forward_outcomes(fr)
    valid = base.filter(pl.col("fwd_valid") == 1).with_row_index("ix")
    # pick a mid-day-1 condition bar
    k = valid.filter(pl.col("trading_day") == date(2024, 1, 2))["ix"].to_list()[5]
    ref = base.row(k, named=True)
    entry = base["open"][k + 1]
    # fwd_up_eod must equal max(high over i+1..end of SAME day) - entry
    day1 = base.filter(pl.col("trading_day") == date(2024, 1, 2))
    fut_high = day1["high"][k + 1 :].max()
    assert abs(ref["fwd_up_eod"] - (fut_high - entry)) < 1e-6

    def fwd_at(mut: pl.DataFrame) -> float:
        return add_forward_outcomes(mut).row(k, named=True)["fwd_up_eod"]

    # mutate a PAST bar (k-2) -> no change
    past = base.with_columns(
        pl.when(pl.int_range(pl.len()) == k - 2)
        .then(pl.col("high") + 999)
        .otherwise(pl.col("high"))
        .alias("high")
    ).drop([c for c in base.columns if c.startswith("fwd_")])
    assert abs(fwd_at(past) - ref["fwd_up_eod"]) < 1e-6, "past bar leaked into forward outcome"

    # mutate a NEXT-DAY bar -> no change (forward caps at EOD)
    nxt = base.with_columns(
        pl.when(pl.col("trading_day") == date(2024, 1, 3))
        .then(pl.col("high") + 9999)
        .otherwise(pl.col("high"))
        .alias("high")
    ).drop([c for c in base.columns if c.startswith("fwd_")])
    assert abs(fwd_at(nxt) - ref["fwd_up_eod"]) < 1e-6, "next-day bar leaked (EOD boundary crossed)"

    # mutate a FUTURE same-day bar -> MUST change
    fut = base.with_columns(
        pl.when(pl.int_range(pl.len()) == k + 3)
        .then(pl.col("high") + 500)
        .otherwise(pl.col("high"))
        .alias("high")
    ).drop([c for c in base.columns if c.startswith("fwd_")])
    assert fwd_at(fut) > ref["fwd_up_eod"] + 100, (
        "future same-day bar did not affect forward outcome"
    )
