"""Topstep account lifecycle simulator (rules verified 2026-07-02).

Models the pipeline the payout-EV objective needs:

    COMBINE --(dynamic profit target)--> FUNDED (XFA) --(payouts...)--> FAILED

Implemented and unit-tested:
  - EOD-trailing Maximum Loss Limit: floor ratchets on end-of-day balance
    highs, never moves down, locks permanently at the starting balance
    ("locks at $0" in XFA display coordinates). Breach evaluated on every
    intraday mark against equity (realized + unrealized), exactly as Topstep
    monitors it in real time.
  - Daily Loss Limit: effective DLL = min(Topstep DLL if enabled, personal
    DLL from risk_overrides). Hitting it = flatten + locked out until the
    next session (soft breach, not failure), matching current rules.
  - Combine Consistency Target: best day must stay <= 50% of the profit
    target; exceeding it RAISES the target to best_day / 0.50 (not a fail).
  - Min 2 trading days to pass; funded (XFA) starts fresh; activation fee.
  - XFA Standard payout path: 5 winning days ($150+) per cycle, positive net
    profit since last payout (first exempt), max 50% of balance per request,
    per-size caps, $125 minimum, 90/10 split, winning-day counter resets
    after each payout, MLL locks at $0 (XFA) after the first payout.
  - Scaling plan: max mini-equivalent contracts by current XFA balance,
    10 micros = 1 mini, plus temporary per-instrument volatility caps (GC!).
  - Cash ledger: monthly subscription (combine only), activation, resets
    (cost = monthly rebill price), payouts. net_cashflow = trader's true net.

Internal coordinates: the XFA officially starts at $0 balance with MLL at
-mll locking at $0. We keep combine coordinates (offset by start_balance);
`xfa_balance` = balance - start_balance gives the official view.

The trade-execution layer (fills, per-trade costs) lives in sim.costs and the
future backtest engine; this class consumes P&L marks so it can be driven by
a backtest, a replay, or synthetic scenarios.

No rule-violation strategies are supported by design: breaches terminate or
lock the account exactly as Topstep would.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional

from ..config import (AccountSpec, CombineRules, Config, FundedRules,
                      PayoutPolicy, PricingPlan)


def _months_after(last: tuple, until: tuple) -> list:
    """Calendar months strictly after `last` up to and including `until`."""
    out = []
    y, m = last
    while (y, m) < until:
        m += 1
        if m == 13:
            y, m = y + 1, 1
        out.append((y, m))
    return out


class Phase(str, Enum):
    COMBINE = "COMBINE"
    FUNDED = "FUNDED"                  # Express Funded Account
    FAILED_COMBINE = "FAILED_COMBINE"
    FAILED_FUNDED = "FAILED_FUNDED"


class Violation(Exception):
    """Raised on API misuse (e.g. marking a failed account) — never used for
    normal rule breaches, which are recorded as events."""


@dataclass
class Event:
    kind: str                  # MLL_BREACH, DLL_HIT, COMBINE_PASSED, PAYOUT, ...
    session: Optional[date]
    detail: Dict = field(default_factory=dict)


class TopstepSimulator:
    def __init__(self, cfg: Config, size: str, plan: str = "standard",
                 personal_dll: Optional[float] = None,
                 start_date: Optional[date] = None):
        self.cfg = cfg
        self.acct: AccountSpec = cfg.accounts[size]
        self.combine_rules: CombineRules = cfg.combine_rules
        self.funded_rules: FundedRules = cfg.funded_rules
        self.payout_policy: PayoutPolicy = cfg.payout
        self.pricing: PricingPlan = cfg.pricing_plan(plan, size)
        self.effective_dll: float = cfg.effective_dll(size, personal_dll)

        self.phase = Phase.COMBINE
        self.balance: float = self.acct.start_balance      # realized
        self._eod_balance_high: float = self.acct.start_balance
        self.mll_floor: float = self.acct.start_balance - self.acct.mll

        # combine consistency target
        self.best_day_profit: float = 0.0

        self.trading_days = 0
        self.session: Optional[date] = None
        self.day_start_balance: float = self.balance
        self.locked_out: bool = False
        self._day_open: bool = False

        # funded/payout state
        self.winning_days_cycle = 0          # resets each payout
        self.winning_days_total = 0
        self._balance_at_last_payout: Optional[float] = None
        self._first_payout_done = False

        self.fees_paid: float = 0.0
        self.payouts_net: List[float] = []
        self.payouts_gross: List[float] = []
        self.resets = 0
        self.events: List[Event] = []
        self._billed_months: set = set()
        self._last_billed_ym: Optional[tuple] = None
        if start_date is not None:
            self._bill_month(start_date)

    # ------------------------------------------------------------------ #
    # properties
    # ------------------------------------------------------------------ #

    @property
    def failed(self) -> bool:
        return self.phase in (Phase.FAILED_COMBINE, Phase.FAILED_FUNDED)

    @property
    def xfa_balance(self) -> float:
        """Official XFA display balance (starts at $0)."""
        return self.balance - self.acct.start_balance

    @property
    def net_cashflow(self) -> float:
        """What the trader has actually netted: payouts (after split & fees)
        minus all subscription/activation/reset costs."""
        return sum(self.payouts_net) - self.fees_paid

    @property
    def effective_profit_target(self) -> float:
        """Combine target after consistency adjustment: best day must stay
        <= pct * target, else target = best_day / pct."""
        pct = self.combine_rules.consistency_best_day_max_pct_of_target
        return max(self.acct.profit_target, self.best_day_profit / pct)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def start_day(self, session: date) -> None:
        self._assert_active()
        if self.session is not None and session <= self.session:
            raise Violation(f"sessions must advance: {session} <= {self.session}")
        self.session = session
        self.day_start_balance = self.balance
        self.locked_out = False
        self._day_open = True
        self._bill_month(session)

    def can_trade(self) -> bool:
        return not self.failed and not self.locked_out

    def mark(self, unrealized_pnl: float, ts: Optional[datetime] = None) -> None:
        """Intraday equity mark: equity = realized balance + open P&L.
        Checks MLL (terminal) and DLL (soft lockout)."""
        self._assert_active()
        equity = self.balance + unrealized_pnl
        self._check_mll(equity)
        if not self.failed:
            self._check_dll(equity)

    def realize(self, pnl: float) -> None:
        """Book realized P&L (a closed trade, net of trading costs)."""
        self._assert_active()
        if self.locked_out and pnl != 0:
            # the closing flatten produced pnl legitimately; anything after is
            # a runner bug — surface loudly either way.
            self.events.append(Event("TRADE_WHILE_LOCKED", self.session, {"pnl": pnl}))
        self.balance += pnl
        self._check_mll(self.balance)
        if not self.failed:
            self._check_dll(self.balance)

    def flatten_violation(self) -> None:
        """Runner reports a position still open at the EOD cutoff (15:10 CT;
        risk managers start flattening 15:08). Auto-flattened by the firm.
        Recorded; our strategies must never produce it intentionally."""
        self.events.append(Event("EOD_FLATTEN_VIOLATION", self.session))

    def max_contracts_now(self, instrument: Optional[str] = None) -> int:
        """Current max mini-equivalent lots: account cap, XFA scaling plan
        (funded only), and temporary volatility caps (e.g. GC).

        The scaling plan "updates next session only, never mid-session":
        the tier is computed from the day-START balance, so an intraday gain
        cannot raise the limit until the next start_day()."""
        limit = self.acct.max_contracts_minis
        if self.phase == Phase.FUNDED:
            day_start_xfa = self.day_start_balance - self.acct.start_balance
            limit = min(limit, self.funded_rules.scaling_max_minis(
                self.acct.size, day_start_xfa))
        if instrument is not None:
            extra = self.cfg.instrument_max_minis(self.acct.size, instrument)
            if extra is not None:
                limit = min(limit, extra)
        return limit

    def check_position_size(self, contracts_minis: int, contracts_micros: int = 0,
                            instrument: Optional[str] = None) -> bool:
        """True if within limits. Micros count 10:1 toward the mini limit;
        micro-denominated volatility caps (e.g. MGC 30/60/90) are checked
        against the micro count directly."""
        ratio = self.combine_rules.micros_per_mini_ratio
        mini_equiv = contracts_minis + contracts_micros / ratio
        limit = self.max_contracts_now(instrument)
        ok = (mini_equiv <= limit
              and contracts_micros <= self.acct.max_contracts_micros)
        if ok and instrument is not None:
            micro_cap = self.cfg.instrument_max_micros(self.acct.size, instrument)
            if micro_cap is not None and contracts_micros > micro_cap:
                ok = False
        if not ok:
            self.events.append(Event("MAX_CONTRACTS_EXCEEDED", self.session,
                                     {"minis": contracts_minis,
                                      "micros": contracts_micros,
                                      "mini_equiv": mini_equiv, "limit": limit}))
        return ok

    def end_day(self) -> None:
        """Close the session: ratchet the MLL floor on the EOD balance, track
        best day / winning days, check combine pass."""
        self._assert_active()
        if self.session is None or not self._day_open:
            raise Violation("end_day without a matching start_day")
        self._day_open = False
        self.trading_days += 1
        day_pnl = self.balance - self.day_start_balance

        self._ratchet_mll_floor()

        if self.phase == Phase.COMBINE:
            self.best_day_profit = max(self.best_day_profit, day_pnl)
            profit = self.balance - self.acct.start_balance
            if (profit >= self.effective_profit_target
                    and self.trading_days >= self.combine_rules.min_trading_days):
                self._pass_combine()
        elif self.phase == Phase.FUNDED:
            if day_pnl >= self.payout_policy.winning_day_min_profit:
                self.winning_days_cycle += 1
                self.winning_days_total += 1

    # ------------------------------------------------------------------ #
    # combine -> funded, resets, payouts
    # ------------------------------------------------------------------ #

    def _pass_combine(self) -> None:
        self.events.append(Event("COMBINE_PASSED", self.session,
                                 {"days": self.trading_days,
                                  "balance": self.balance,
                                  "effective_target": self.effective_profit_target}))
        self.phase = Phase.FUNDED
        self._charge(self.pricing.activation, "ACTIVATION_FEE")
        # XFA starts fresh (officially $0 balance; combine profits don't carry)
        self.balance = self.acct.start_balance
        self._eod_balance_high = self.acct.start_balance
        self.mll_floor = self.acct.start_balance - self.acct.mll
        self.day_start_balance = self.balance
        self.winning_days_cycle = 0
        self.winning_days_total = 0

    def reset_combine(self) -> None:
        """Paid combine reset (combine stage only; funded accounts cannot be
        reset — Back2Funded is a separate product). Cost = monthly rebill."""
        if self.phase not in (Phase.COMBINE, Phase.FAILED_COMBINE):
            raise Violation("reset only applies to the combine stage")
        self._charge(self.cfg.reset_fee(self.pricing.plan, self.acct.size), "RESET_FEE")
        self.resets += 1
        self.phase = Phase.COMBINE
        self.balance = self.acct.start_balance
        self._eod_balance_high = self.acct.start_balance
        self.mll_floor = self.acct.start_balance - self.acct.mll
        self.day_start_balance = self.balance
        self.trading_days = 0
        self.best_day_profit = 0.0
        self.locked_out = False

    def max_payout_available(self) -> float:
        """Largest payout requestable right now under the XFA Standard path
        (0 if not eligible)."""
        if self.phase != Phase.FUNDED or self.locked_out:
            return 0.0
        p = self.payout_policy
        if self.winning_days_cycle < p.min_winning_days:
            return 0.0
        if (p.require_positive_since_last_payout and self._first_payout_done
                and self.balance <= self._balance_at_last_payout):
            return 0.0
        profit = self.xfa_balance
        if profit <= 0:
            return 0.0
        cap = p.per_payout_cap.get(self.acct.size, float("inf"))
        if p.dll_at_checkout_doubles_caps and self.cfg.risk.use_topstep_dll:
            cap *= 2                      # active promo: DLL at checkout
        allowed = min(p.max_pct_of_balance * profit, cap)
        return allowed if allowed >= p.min_payout else 0.0

    def request_payout(self, amount: float) -> float:
        """Withdraw `amount` (<= max_payout_available()). Returns the net
        amount the trader receives after the 90/10 split and processing fee."""
        allowed = self.max_payout_available()
        if amount <= 0 or amount > allowed + 1e-9:
            raise Violation(f"payout {amount} not allowed (max {allowed})")
        p = self.payout_policy
        if amount < p.min_payout:
            raise Violation(f"payout {amount} below minimum {p.min_payout}")
        self.balance -= amount
        # a withdrawal is not trading P&L: shift the day baseline so winning-
        # day counting and the DLL are unaffected by the balance deduction
        self.day_start_balance -= amount
        net = amount * p.profit_split_trader - self.cfg.payout_fee
        self.payouts_gross.append(amount)
        self.payouts_net.append(net)
        self._first_payout_done = True
        self._balance_at_last_payout = self.balance
        if p.winning_day_counter_resets_after_payout:
            self.winning_days_cycle = 0
        if p.mll_locks_at_zero_after_first_payout:
            self.mll_floor = max(self.mll_floor, self.acct.start_balance)
        self.events.append(Event("PAYOUT", self.session,
                                 {"gross": amount, "net": net,
                                  "xfa_balance_after": self.xfa_balance}))
        return net

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _assert_active(self) -> None:
        if self.failed:
            raise Violation(f"account is terminal: {self.phase.value}")

    def _ratchet_mll_floor(self) -> None:
        stops_at_start = (self.combine_rules.mll_stops_at_start_balance
                          if self.phase == Phase.COMBINE
                          else self.funded_rules.mll_stops_at_start_balance)
        if self.balance > self._eod_balance_high:
            self._eod_balance_high = self.balance
            new_floor = self._eod_balance_high - self.acct.mll
            if stops_at_start:
                new_floor = min(new_floor, self.acct.start_balance)
            self.mll_floor = max(self.mll_floor, new_floor)

    def _check_mll(self, equity: float) -> None:
        if equity <= self.mll_floor:
            self.events.append(Event("MLL_BREACH", self.session,
                                     {"equity": equity, "floor": self.mll_floor}))
            self.phase = (Phase.FAILED_COMBINE if self.phase == Phase.COMBINE
                          else Phase.FAILED_FUNDED)

    def _check_dll(self, equity: float) -> None:
        day_loss = self.day_start_balance - equity
        if day_loss >= self.effective_dll and not self.locked_out:
            policy = (self.combine_rules.dll_breach_policy
                      if self.phase == Phase.COMBINE
                      else self.funded_rules.dll_breach_policy)
            self.events.append(Event("DLL_HIT", self.session,
                                     {"day_loss": day_loss,
                                      "effective_dll": self.effective_dll,
                                      "policy": policy}))
            if policy == "fail":
                self.phase = (Phase.FAILED_COMBINE if self.phase == Phase.COMBINE
                              else Phase.FAILED_FUNDED)
            else:
                self.locked_out = True

    def _bill_month(self, session: date) -> None:
        """Monthly subscription while in the combine (none once funded).
        The subscription rebills every calendar month whether or not the
        account traded, so months skipped between sessions are billed too."""
        ym = (session.year, session.month)
        if self._last_billed_ym is None:
            months = [ym]
        else:
            months = _months_after(self._last_billed_ym, ym)
        for key in months:
            if key in self._billed_months:
                continue
            self._billed_months.add(key)
            fee = (self.pricing.monthly if self.phase == Phase.COMBINE
                   else self.funded_rules.monthly_fee)
            if fee:
                self._charge(fee, "MONTHLY_FEE")
        if self._last_billed_ym is None or ym > self._last_billed_ym:
            self._last_billed_ym = ym

    def _charge(self, amount: float, kind: str) -> None:
        if amount:
            self.fees_paid += amount
            self.events.append(Event(kind, self.session, {"amount": amount}))

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #

    def summary(self) -> Dict:
        return {
            "phase": self.phase.value,
            "balance": self.balance,
            "xfa_balance": self.xfa_balance,
            "mll_floor": self.mll_floor,
            "effective_dll": self.effective_dll,
            "effective_profit_target": (self.effective_profit_target
                                        if self.phase == Phase.COMBINE else None),
            "trading_days": self.trading_days,
            "winning_days_cycle": self.winning_days_cycle,
            "winning_days_total": self.winning_days_total,
            "fees_paid": self.fees_paid,
            "payouts_count": len(self.payouts_net),
            "payouts_gross_total": sum(self.payouts_gross),
            "payouts_net_total": sum(self.payouts_net),
            "net_cashflow": self.net_cashflow,
            "resets": self.resets,
            "events": [(e.kind, str(e.session)) for e in self.events],
        }
