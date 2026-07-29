"""Funded-account (Topstep-style) lifecycle simulation.

A :class:`TopstepAccount` is stepped with mark-to-market equity per bar. It is the
single source of truth for the funded-account rules and is used both *live* inside
the backtester (gating new entries, forcing flat on breach) and *after the fact*
to evaluate a saved equity curve. Rules come from :class:`TopstepRules` (config).

Trailing max drawdown trails the equity peak; when ``lock_trailing_at_start`` is
set it stops rising once it reaches the starting balance (Topstep behaviour).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from nqlab.config.models import TopstepRules


@dataclass
class TopstepResult:
    rules_name: str
    starting_balance: float
    final_equity: float
    net_profit: float
    passed: bool
    pass_trading_day: date | None
    days_to_pass: int | None
    blown: bool
    blow_trading_day: date | None
    max_drawdown_from_peak: float
    trading_days: int
    daily_pnl: dict[str, float]
    best_day: float
    worst_day: float
    consistency_ok: bool | None
    account_failures: int
    violations: list[str] = field(default_factory=list)


class TopstepAccount:
    """Stepwise funded-account state machine."""

    def __init__(self, rules: TopstepRules) -> None:
        self.rules = rules
        self.start = rules.account_size
        self.equity = rules.account_size
        self.peak = rules.account_size
        self.last_equity = rules.account_size
        self.current_day: date | None = None
        self.day_start_equity = rules.account_size
        self.halted_today = False
        self.blown = False
        self.blow_day: date | None = None
        self.passed = False
        self.pass_day: date | None = None
        self.days_to_pass: int | None = None
        self.max_dd_from_peak = 0.0
        self.trading_days: list[date] = []
        self.daily_pnl: dict[str, float] = {}
        self.violations: list[str] = []

    @property
    def trailing_floor(self) -> float:
        floor = self.peak - self.rules.trailing_max_loss
        if self.rules.lock_trailing_at_start:
            floor = min(floor, self.start)
        return floor

    def can_open(self) -> bool:
        return not self.blown and not self.halted_today

    def _start_day(self, day: date) -> None:
        self.current_day = day
        self.day_start_equity = self.last_equity
        self.halted_today = False
        self.trading_days.append(day)

    def _check_breach(self, equity: float, day: date) -> None:
        self.peak = max(self.peak, equity)
        self.max_dd_from_peak = max(self.max_dd_from_peak, self.peak - equity)
        if equity <= self.trailing_floor and not self.blown:
            self.blown = True
            self.blow_day = day
            self.violations.append(f"trailing_drawdown_breach@{day.isoformat()}")

    def mark(self, equity: float, day: date) -> None:
        """Update the account with mark-to-market ``equity`` on ``day``."""
        if day != self.current_day:
            if self.current_day is not None:
                self.daily_pnl[self.current_day.isoformat()] = (
                    self.last_equity - self.day_start_equity
                )
                if self.rules.trailing_basis == "eod":
                    self._check_breach(self.last_equity, self.current_day)
            self._start_day(day)

        self.equity = equity
        if self.rules.trailing_basis == "intraday":
            self._check_breach(equity, day)

        if (
            self.rules.daily_loss_limit is not None
            and (equity - self.day_start_equity) <= -self.rules.daily_loss_limit
            and not self.halted_today
        ):
            self.halted_today = True
            self.violations.append(f"daily_loss_halt@{day.isoformat()}")

        if (
            not self.passed
            and not self.blown
            and (equity - self.start) >= self.rules.profit_target
            and len(self.trading_days) >= self.rules.min_trading_days
        ):
            self.passed = True
            self.pass_day = day
            self.days_to_pass = len(self.trading_days)

        self.last_equity = equity

    def finalize(self) -> TopstepResult:
        if self.current_day is not None:
            self.daily_pnl[self.current_day.isoformat()] = (
                self.last_equity - self.day_start_equity
            )
            if self.rules.trailing_basis == "eod":
                self._check_breach(self.last_equity, self.current_day)

        net_profit = self.last_equity - self.start
        day_values = list(self.daily_pnl.values())
        best_day = max(day_values) if day_values else 0.0
        worst_day = min(day_values) if day_values else 0.0

        consistency_ok: bool | None = None
        if self.rules.consistency_pct is not None:
            total = sum(v for v in day_values if v > 0)
            consistency_ok = total > 0 and best_day <= self.rules.consistency_pct * total

        return TopstepResult(
            rules_name=self.rules.name,
            starting_balance=self.start,
            final_equity=self.last_equity,
            net_profit=net_profit,
            passed=self.passed and not self.blown,
            pass_trading_day=self.pass_day,
            days_to_pass=self.days_to_pass,
            blown=self.blown,
            blow_trading_day=self.blow_day,
            max_drawdown_from_peak=self.max_dd_from_peak,
            trading_days=len(self.trading_days),
            daily_pnl=dict(self.daily_pnl),
            best_day=best_day,
            worst_day=worst_day,
            consistency_ok=consistency_ok,
            account_failures=1 if self.blown else 0,
            violations=list(self.violations),
        )


def simulate_account(marks: list[tuple[date, float]], rules: TopstepRules) -> TopstepResult:
    """Replay a sequence of ``(trading_day, equity)`` marks through the rules."""
    account = TopstepAccount(rules)
    for day, equity in marks:
        account.mark(equity, day)
    return account.finalize()
