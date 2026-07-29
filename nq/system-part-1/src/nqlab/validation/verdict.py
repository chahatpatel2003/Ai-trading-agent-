"""Acceptance rules: turn a ValidationResult into ACCEPT / REJECT / NEEDS_MORE_DATA.

NEEDS_MORE_DATA short-circuits when there are too few trades to judge. Otherwise
every failing reject condition is collected so the report explains *all* the ways
a strategy fell short, not just the first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nqlab.config.models import ValidationConfig
from nqlab.validation.periods import net_of, trades_of

if TYPE_CHECKING:
    from nqlab.validation.gauntlet import ValidationResult

ACCEPT = "ACCEPT"
REJECT = "REJECT"
NEEDS_MORE_DATA = "NEEDS_MORE_DATA"


def decide(result: ValidationResult, v: ValidationConfig) -> tuple[str, list[str]]:
    total = trades_of(result.full)
    if total < v.min_trades:
        return NEEDS_MORE_DATA, [f"only {total} trades (need >= {v.min_trades})"]
    oos_trades = trades_of(result.out_sample)
    if oos_trades < v.min_oos_trades:
        return NEEDS_MORE_DATA, [
            f"only {oos_trades} out-of-sample trades (need >= {v.min_oos_trades})"
        ]

    reasons: list[str] = []

    if len(result.yearly) >= 2:
        positive_years = sum(1 for s in result.yearly.values() if net_of(s) > 0)
        if positive_years < v.min_positive_years:
            reasons.append(
                f"profitable in only {positive_years}/{len(result.yearly)} years"
            )

    if net_of(result.full) > 0 and len(result.monthly) >= 3:
        positive_months = sum(1 for s in result.monthly.values() if net_of(s) > 0)
        if positive_months <= 1:
            reasons.append("profit concentrated in <= 1 month")

    if net_of(result.out_sample) <= 0:
        reasons.append(
            f"out-of-sample net PnL {result.out_sample.get('net_pnl')} <= 0 (fails validation)"
        )

    if result.walk_forward_positive_frac < v.min_walk_forward_positive_frac:
        reasons.append(
            f"walk-forward positive in {result.walk_forward_positive_frac:.0%} of windows "
            f"(< {v.min_walk_forward_positive_frac:.0%})"
        )

    failed = [s.label for s in result.cost_stress if s.net_pnl <= 0]
    if failed:
        reasons.append(f"fails cost/slippage stress at {', '.join(failed)}")

    if result.monte_carlo.blow_up_prob > v.max_blowup_prob:
        reasons.append(
            f"blow-up probability {result.monte_carlo.blow_up_prob:.1%} "
            f"> {v.max_blowup_prob:.0%}"
        )

    if result.unrealistic_fills:
        reasons.append("edge only appears with unrealistic current-close fills")

    if result.topstep is not None and result.topstep.blown:
        reasons.append("blows the Topstep account in the chronological run")

    if reasons:
        return REJECT, reasons
    return ACCEPT, ["survives OOS, walk-forward, cost stress, Monte-Carlo, and Topstep"]
