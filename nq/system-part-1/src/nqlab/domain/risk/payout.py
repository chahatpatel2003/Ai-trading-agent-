"""Topstep payout mechanics — calculated from simulated daily equity, never assumed.

A Topstep *funded* (Express) account pays out on rules driven entirely by the realized
daily P&L of the simulated account, the locked-buffer drawdown, and the consistency rule:

- A **winning day** is a trading day with realized profit >= ``WINNING_DAY_MIN`` ($200).
- A first payout requires at least ``min_winning_days`` winning days (default 5) and the
  account profit above the safety buffer, with the **consistency rule**: the largest single
  winning day must be <= ``consistency_pct`` (50%) of total profit at request time.
- Thereafter a month is payout-eligible when it ends net-positive, the account is not blown,
  enough cumulative winning days exist, and consistency still holds.

Everything here is derived from the daily-P&L series produced by the locked-buffer backtest;
nothing about payouts is hardcoded or assumed. Real Topstep also caps the first few payouts and
applies a profit split — those are noted as caveats, not modelled, so projections are an upper
bound on withdrawable profit, not a promise.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from nqlab.config.models import TopstepRules

WINNING_DAY_MIN = 200.0  # Topstep definition of a qualifying winning day
MIN_WINNING_DAYS = 5  # winning days required before a first payout
BIG_WINNING_DAY = 500.0  # a "strong" payout day


@dataclass(frozen=True)
class WinningDayStats:
    trading_days: int
    winning_days_200: int  # days >= $200
    winning_days_500: int  # days >= $500
    winning_day_rate: float  # $200-days / trading days
    avg_winning_day: float  # mean P&L of $200+ days
    best_day: float


@dataclass(frozen=True)
class PayoutMonth:
    month: str
    net: float
    winning_days_200: int
    eligible: bool
    payout: float
    reason: str


@dataclass(frozen=True)
class PayoutProjection:
    months_total: int
    payout_eligible_months: int
    payout_every_month: bool
    total_projected_payout: float
    avg_monthly_payout: float
    annual_payout: float
    consistency_ok: bool
    largest_win_day_pct: float | None
    months: list[PayoutMonth] = field(default_factory=list)


def _daily_values(daily_pnl: dict[str, float] | list[float]) -> list[float]:
    return list(daily_pnl.values()) if isinstance(daily_pnl, dict) else list(daily_pnl)


def winning_day_stats(
    daily_pnl: dict[str, float] | list[float], *, threshold: float = WINNING_DAY_MIN
) -> WinningDayStats:
    """$200+ winning-day statistics from a per-day realized-P&L series."""
    vals = _daily_values(daily_pnl)
    n = len(vals)
    wins_200 = [d for d in vals if d >= threshold]
    wins_500 = [d for d in vals if d >= BIG_WINNING_DAY]
    return WinningDayStats(
        trading_days=n,
        winning_days_200=len(wins_200),
        winning_days_500=len(wins_500),
        winning_day_rate=round(len(wins_200) / n, 4) if n else 0.0,
        avg_winning_day=round(sum(wins_200) / len(wins_200), 2) if wins_200 else 0.0,
        best_day=round(max(vals), 2) if vals else 0.0,
    )


def _consistency_ok(
    daily_values: list[float], consistency_pct: float | None
) -> tuple[bool, float | None]:
    """Topstep consistency: largest winning day <= consistency_pct of total positive profit."""
    pos_total = sum(d for d in daily_values if d > 0)
    best = max(daily_values) if daily_values else 0.0
    if pos_total <= 0:
        return False, None
    pct = best / pos_total
    if consistency_pct is None:
        return True, round(pct, 4)
    return pct <= consistency_pct, round(pct, 4)


def payout_projection(
    daily_pnl_by_day: dict[str, float],
    *,
    rules: TopstepRules,
    months_total: int,
    blown: bool,
    min_winning_days: int = MIN_WINNING_DAYS,
    threshold: float = WINNING_DAY_MIN,
) -> PayoutProjection:
    """Project withdrawable payouts month-by-month from the daily-equity series.

    ``daily_pnl_by_day`` maps ``YYYY-MM-DD`` -> that day's realized P&L (Topstep ``daily_pnl``).
    A month is payout-eligible when: it ends net-positive, the account is not blown, the
    life-to-date winning-day count has reached ``min_winning_days``, and the consistency rule
    holds life-to-date. The projected payout for an eligible month is that month's net profit
    (an upper bound: real Topstep caps early payouts and applies a split).
    """
    by_month: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for day, pnl in sorted(daily_pnl_by_day.items()):
        by_month[day[:7]].append((day, pnl))

    all_values = [p for _, p in sorted(daily_pnl_by_day.items())]
    cons_ok, win_day_pct = _consistency_ok(all_values, rules.consistency_pct)

    cum_winning = 0
    cum_days: list[float] = []
    months: list[PayoutMonth] = []
    for month in sorted(by_month):
        day_vals = [p for _, p in by_month[month]]
        cum_days.extend(day_vals)
        month_net = sum(day_vals)
        month_wins = sum(1 for d in day_vals if d >= threshold)
        cum_winning += month_wins
        month_cons_ok, _ = _consistency_ok(cum_days, rules.consistency_pct)
        if blown:
            eligible, reason = False, "account blown"
        elif month_net <= 0:
            eligible, reason = False, "month not net-positive"
        elif cum_winning < min_winning_days:
            eligible, reason = False, f"only {cum_winning} winning days (<{min_winning_days})"
        elif not month_cons_ok:
            eligible, reason = False, "consistency rule violated (one day > 50% of profit)"
        else:
            eligible, reason = True, "eligible"
        months.append(
            PayoutMonth(month, round(month_net, 2), month_wins, eligible,
                        round(month_net, 2) if eligible else 0.0, reason)
        )

    elig = [m for m in months if m.eligible]
    total = sum(m.payout for m in elig)
    avg_monthly = total / months_total if months_total > 0 else 0.0
    return PayoutProjection(
        months_total=months_total,
        payout_eligible_months=len(elig),
        payout_every_month=len(elig) == months_total and months_total > 0,
        total_projected_payout=round(total, 2),
        avg_monthly_payout=round(avg_monthly, 2),
        annual_payout=round(12.0 * avg_monthly, 2),
        consistency_ok=cons_ok,
        largest_win_day_pct=win_day_pct,
        months=months,
    )


@dataclass(frozen=True)
class FleetProjection:
    n_accounts: int
    account_size: float
    per_account_monthly_payout: float
    per_account_ruin: float
    fleet_monthly_payout: float
    fleet_annual_payout: float
    correlated: bool
    note: str


@dataclass(frozen=True)
class FundedWithdrawalResult:
    """Stage-B funded lifecycle: payouts withdrawn from equity, then drawdown on the
    reduced buffer (a withdrawal shrinks the cushion above the locked floor)."""

    n_payouts: int
    total_withdrawn: float
    avg_payout: float
    blown_after_payout: bool
    ending_equity: float
    months_to_first_payout: int | None


def simulate_funded_withdrawals(
    daily_pnl_by_day: dict[str, float],
    *,
    rules: TopstepRules,
    min_winning_days: int = MIN_WINNING_DAYS,
    safety_buffer: float | None = None,
    threshold: float = WINNING_DAY_MIN,
) -> FundedWithdrawalResult:
    """Walk the funded-account equity day by day, withdrawing eligible profit monthly.

    The funded trailing drawdown is locked at the starting balance, so the floor is
    ``min(peak - trailing_max_loss, start)``. When a month ends payout-eligible (>= 5
    winning days life-to-date, consistency rule holds, equity above ``start + safety_buffer``),
    the profit above ``start + safety_buffer`` is withdrawn — which REDUCES the cushion above
    the floor. A subsequent drawdown can then blow the account: that is the realistic
    "drawdown after payouts" risk this models.
    """
    start = rules.account_size
    buf = safety_buffer if safety_buffer is not None else rules.trailing_max_loss
    equity = peak = start
    cum_winning = 0
    cum_pos_total = 0.0
    cum_best_day = 0.0
    withdrawn = 0.0
    payouts: list[float] = []
    blown = False
    first_payout_month: int | None = None
    months_seen: set[str] = set()

    by_month: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for day, pnl in sorted(daily_pnl_by_day.items()):
        by_month[day[:7]].append((day, pnl))

    for month_idx, month in enumerate(sorted(by_month)):
        months_seen.add(month)
        for _day, pnl in by_month[month]:
            equity += pnl
            peak = max(peak, equity)
            if pnl >= threshold:
                cum_winning += 1
            if pnl > 0:
                cum_pos_total += pnl
                cum_best_day = max(cum_best_day, pnl)
            floor = min(peak - rules.trailing_max_loss, start)
            if equity <= floor:
                blown = True
                break
        if blown:
            break
        # End-of-month payout if eligible (winning days + consistency + above buffer).
        consistency_ok = (
            rules.consistency_pct is None
            or (cum_pos_total > 0 and cum_best_day <= rules.consistency_pct * cum_pos_total)
        )
        if cum_winning >= min_winning_days and consistency_ok and equity > start + buf:
            amount = equity - (start + buf)
            equity = start + buf
            withdrawn += amount
            payouts.append(amount)
            if first_payout_month is None:
                first_payout_month = month_idx + 1

    return FundedWithdrawalResult(
        n_payouts=len(payouts),
        total_withdrawn=round(withdrawn, 2),
        avg_payout=round(withdrawn / len(payouts), 2) if payouts else 0.0,
        blown_after_payout=blown and len(payouts) > 0,
        ending_equity=round(equity, 2),
        months_to_first_payout=first_payout_month,
    )


def scale_funded_fleet(
    *,
    per_account_monthly_payout: float,
    per_account_ruin: float,
    n_accounts: int,
    account_size: float = 150_000.0,
) -> FleetProjection:
    """Aggregate the same strategy across N funded 150K accounts.

    The accounts trade IDENTICAL signals, so payouts add linearly but risk is perfectly
    CORRELATED: there is no diversification — a regime that blows one account blows them all.
    Fleet ruin therefore equals single-account ruin; only the payout scales by N.
    """
    return FleetProjection(
        n_accounts=n_accounts,
        account_size=account_size,
        per_account_monthly_payout=round(per_account_monthly_payout, 2),
        per_account_ruin=round(per_account_ruin, 4),
        fleet_monthly_payout=round(n_accounts * per_account_monthly_payout, 2),
        fleet_annual_payout=round(12.0 * n_accounts * per_account_monthly_payout, 2),
        correlated=True,
        note=(
            "identical signals across accounts -> payouts scale x N but risk is fully "
            "correlated (no diversification; one bad regime blows the whole fleet)"
        ),
    )
