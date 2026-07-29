"""Account-level evaluation: backtest trades -> TopstepSimulator -> scoring
under the fixed objective (max expected legal net payout before failure).

Two layers:
  simulate_account   one deterministic pass of a trade history through the
                     account rules (MLL marks use per-trade MAE, DLL lockouts
                     stop the day, greedy payout requests at day end)
  lifecycle_mc       block-bootstrap sessions (with replacement) into
                     synthetic combine->XFA lifecycles to estimate expected
                     net payout, pass probability, DLL hit rate, failure ages
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..sim.topstep import Phase, TopstepSimulator, Violation
from .backtest import BacktestResult, Trade


@dataclass
class AccountRun:
    phase: str
    net_cashflow: float
    payouts_net: float
    fees: float
    sessions_used: int
    dll_hits: int
    passed_combine: bool
    n_payouts: int = 0
    events: List = field(default_factory=list)


def simulate_account(session_trades: Dict[date, List[Trade]], cfg: Config,
                     size: str = "50k", plan: str = "standard",
                     personal_dll: Optional[float] = None,
                     start_date: Optional[date] = None) -> AccountRun:
    sim = TopstepSimulator(cfg, size, plan=plan, personal_dll=personal_dll,
                           start_date=start_date or min(session_trades))
    dll_hits = 0
    used = 0
    passed = False
    for sess in sorted(session_trades):
        if sim.failed:
            break
        sim.start_day(sess)
        used += 1
        for t in session_trades[sess]:
            if not sim.can_trade():
                break
            sim.mark(-t.mae_usd)            # worst open-trade excursion
            if sim.failed or sim.locked_out:
                # position force-closed on the breach mark: book the stop-out
                if not sim.failed:
                    sim.realize(-min(t.risk_usd, t.mae_usd))
                break
            sim.realize(t.net_usd)
        if sim.failed:
            break
        if sim.locked_out:
            dll_hits += 1
        sim.end_day()
        if sim.phase == Phase.FUNDED:
            passed = True
            amt = sim.max_payout_available()
            if amt > 0:
                sim.request_payout(amt)     # greedy legal extraction
    return AccountRun(
        phase=sim.phase.value, net_cashflow=sim.net_cashflow,
        payouts_net=sum(sim.payouts_net), fees=sim.fees_paid,
        sessions_used=used, dll_hits=dll_hits, passed_combine=passed,
        n_payouts=len(sim.payouts_net),
        events=[(e.kind, str(e.session)) for e in sim.events])


@dataclass
class LifecycleStats:
    ev_net_payout: float
    median_net: float
    pass_prob: float
    fail_rate: float
    dll_hit_rate: float          # DLL lockout days / sessions traded
    median_sessions_to_terminal: float
    runs: int
    payout_prob: float = 0.0     # P(>= 1 payout before failure/horizon)
    mean_payouts: float = 0.0    # expected number of payouts
    combine_fail_rate: float = 0.0
    funded_fail_rate: float = 0.0


def lifecycle_mc(session_pool: Sequence[List[Trade]], cfg: Config,
                 size: str = "50k", plan: str = "standard",
                 personal_dll: Optional[float] = None,
                 n_runs: int = 200, horizon_sessions: int = 250,
                 seed: int = 17) -> LifecycleStats:
    """Bootstrap whole sessions (with replacement) from the OOS pool into a
    synthetic lifecycle on a weekday calendar. Sessions are the resampling
    unit, preserving intraday trade ordering and daily P&L clustering."""
    if not session_pool:
        return LifecycleStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    rng = np.random.default_rng(seed)
    nets, passes, fails, dll_rates, ages = [], [], [], [], []
    n_pay, cfail, ffail = [], [], []
    for _ in range(n_runs):
        picks = rng.integers(0, len(session_pool), size=horizon_sessions)
        trades_by_day: Dict[date, List[Trade]] = {}
        d = date(2026, 1, 5)
        for k in picks:
            while d.weekday() >= 5:
                d += timedelta(days=1)
            trades_by_day[d] = session_pool[k]
            d += timedelta(days=1)
        run = simulate_account(trades_by_day, cfg, size=size, plan=plan,
                               personal_dll=personal_dll,
                               start_date=min(trades_by_day))
        nets.append(run.net_cashflow)
        passes.append(run.passed_combine)
        fails.append(run.phase.startswith("FAILED"))
        dll_rates.append(run.dll_hits / max(run.sessions_used, 1))
        ages.append(run.sessions_used)
        n_pay.append(run.n_payouts)
        cfail.append(run.phase == "FAILED_COMBINE")
        ffail.append(run.phase == "FAILED_FUNDED")
    return LifecycleStats(
        ev_net_payout=float(np.mean(nets)), median_net=float(np.median(nets)),
        pass_prob=float(np.mean(passes)), fail_rate=float(np.mean(fails)),
        dll_hit_rate=float(np.mean(dll_rates)),
        median_sessions_to_terminal=float(np.median(ages)), runs=n_runs,
        payout_prob=float(np.mean([p > 0 for p in n_pay])),
        mean_payouts=float(np.mean(n_pay)),
        combine_fail_rate=float(np.mean(cfail)),
        funded_fail_rate=float(np.mean(ffail)))


def simulate_process(day_stream: Sequence, cfg: Config, size: str = "50k",
                     plan: str = "standard",
                     personal_dll: Optional[float] = 1000.0) -> Dict:
    """The RENEWAL process a real trader runs: one account at a time; a
    combine MLL failure buys a paid reset (cost = monthly rebill), a funded
    blowout starts a fresh combine (new subscription/activation). Returns
    process-level totals over the given (date, trades) stream."""
    total_net = 0.0
    attempts, combine_fails, funded_fails = 1, 0, 0
    payouts = 0
    passes = 0
    dll_days = 0
    sessions = 0
    first_pass_session: Optional[int] = None
    sim: Optional[TopstepSimulator] = None
    for d, trades in day_stream:
        if sim is None:
            sim = TopstepSimulator(cfg, size, plan=plan,
                                   personal_dll=personal_dll, start_date=d)
        sessions += 1
        sim.start_day(d)
        for t in trades:
            if not sim.can_trade():
                break
            sim.mark(-t.mae_usd)
            if sim.failed or sim.locked_out:
                if not sim.failed:
                    sim.realize(-min(t.risk_usd, t.mae_usd))
                break
            sim.realize(t.net_usd)
        if sim.failed:
            if sim.phase == Phase.FAILED_COMBINE:
                combine_fails += 1
                attempts += 1
                sim.reset_combine()          # paid reset, same subscription
            else:
                funded_fails += 1
                total_net += sim.net_cashflow
                sim = None                   # fresh combine from next session
                attempts += 1
            continue
        if sim.locked_out:
            dll_days += 1
        sim.end_day()
        if sim.phase == Phase.FUNDED:
            if first_pass_session is None:
                first_pass_session = sessions
                passes += 1
            amt = sim.max_payout_available()
            if amt > 0:
                sim.request_payout(amt)
                payouts += 1
    if sim is not None:
        total_net += sim.net_cashflow
    return {"total_net": total_net, "payouts": payouts, "attempts": attempts,
            "combine_fails": combine_fails, "funded_fails": funded_fails,
            "passed_any": first_pass_session is not None,
            "first_pass_session": first_pass_session,
            "dll_days": dll_days, "sessions": sessions}


@dataclass
class MonthlyStats:
    ev_month: float                  # E[net $ / month] incl. ALL fees
    ev_year: float
    median_year: float
    p_pass: float                    # P(pass a combine within horizon)
    p_payout1: float
    p_payout2: float
    p_payout3: float
    mean_payouts_year: float
    blowout_rate: float              # MLL failures / account attempts
    dll_hit_rate: float
    median_sessions_to_first_pass: float
    runs: int


def lifecycle_mc_chain(session_pool: Sequence[List[Trade]], cfg: Config,
                       size: str = "50k", plan: str = "standard",
                       personal_dll: Optional[float] = 1000.0,
                       n_runs: int = 300, horizon_sessions: int = 252,
                       seed: int = 31) -> MonthlyStats:
    """Bootstrap whole sessions into 12-month renewal-process runs."""
    if not session_pool:
        return MonthlyStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, float("nan"), 0)
    rng = np.random.default_rng(seed)
    outs = []
    for _ in range(n_runs):
        picks = rng.integers(0, len(session_pool), size=horizon_sessions)
        d = date(2026, 1, 5)
        stream = []
        for k in picks:
            while d.weekday() >= 5:
                d += timedelta(days=1)
            stream.append((d, session_pool[k]))
            d += timedelta(days=1)
        outs.append(simulate_process(stream, cfg, size=size, plan=plan,
                                     personal_dll=personal_dll))
    nets = [o["total_net"] for o in outs]
    pays = [o["payouts"] for o in outs]
    fp = [o["first_pass_session"] for o in outs
          if o["first_pass_session"] is not None]
    months = horizon_sessions / 21.0
    return MonthlyStats(
        ev_month=float(np.mean(nets)) / months,
        ev_year=float(np.mean(nets)), median_year=float(np.median(nets)),
        p_pass=float(np.mean([o["passed_any"] for o in outs])),
        p_payout1=float(np.mean([p >= 1 for p in pays])),
        p_payout2=float(np.mean([p >= 2 for p in pays])),
        p_payout3=float(np.mean([p >= 3 for p in pays])),
        mean_payouts_year=float(np.mean(pays)),
        blowout_rate=float(np.mean(
            [(o["combine_fails"] + o["funded_fails"]) / o["attempts"]
             for o in outs])),
        dll_hit_rate=float(np.mean(
            [o["dll_days"] / max(o["sessions"], 1) for o in outs])),
        median_sessions_to_first_pass=float(np.median(fp)) if fp
        else float("nan"),
        runs=n_runs)


def max_drawdown(daily_net: pd.Series) -> float:
    """Max peak-to-trough drawdown ($) of a cumulative daily P&L curve."""
    if daily_net.empty:
        return 0.0
    eq = daily_net.cumsum()
    return float((eq - eq.cummax()).min())


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

@dataclass
class Score:
    rule_name: str
    train_net_by_fold: List[float]
    test_net_by_fold: List[float]
    oos_trades: int
    oos_net: float
    violations: int
    lifecycle: Optional[LifecycleStats]
    feasible: bool
    reasons: List[str]
    score: float

    @staticmethod
    def evaluate(rule_name: str, train_nets: List[float],
                 test_nets: List[float], oos: BacktestResult,
                 life: Optional[LifecycleStats],
                 max_dll_hit_rate: float = 0.10,
                 min_pass_prob: float = 0.0) -> "Score":
        reasons = []
        if oos.violations != 0:
            reasons.append("rule violations in backtest")
        if life is not None and life.dll_hit_rate > max_dll_hit_rate:
            reasons.append(f"DLL hit rate {life.dll_hit_rate:.2f} > "
                           f"{max_dll_hit_rate}")
        if life is not None and life.pass_prob < min_pass_prob:
            reasons.append(f"pass prob {life.pass_prob:.2f} < {min_pass_prob}")
        if len(test_nets) >= 2 and sum(n > 0 for n in test_nets) == 0:
            reasons.append("no positive walk-forward test fold")
        feasible = not reasons
        score = (life.ev_net_payout if (feasible and life is not None)
                 else float("-inf"))
        return Score(rule_name=rule_name, train_net_by_fold=train_nets,
                     test_net_by_fold=test_nets, oos_trades=len(oos.trades),
                     oos_net=oos.total_net, violations=oos.violations,
                     lifecycle=life, feasible=feasible, reasons=reasons,
                     score=score)
