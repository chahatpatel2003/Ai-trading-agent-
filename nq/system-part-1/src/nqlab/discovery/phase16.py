"""Phase 16 — payout-first, scalable strategy acceptance.

The acceptance bar is reset to a *business* standard: a strategy is only worth paper-trading if
it can actually support Topstep payouts and scale across multiple 150K accounts. This module
holds the thresholds, the payout-first business-stat panel, the two-stage (eval + funded-scale)
simulator, the survivor selector, and the strict verdict. Nothing is accepted unless it clears
every gate — no forced acceptance.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl

from nqlab.backtest.engine import run_backtest
from nqlab.config.models import AppConfig, TopstepRules
from nqlab.discovery.deflation import deflated_sharpe_ratio, expected_max_sharpe, sharpe_variance
from nqlab.discovery.lockbox import (
    _ALL_FAMILIES,
    ACCEPT_FOR_FORWARD_PAPER,
    ACCEPT_FOR_PAPER,
    NEEDS_MORE_DATA,
    REJECT,
    FreezeSpec,
    FrozenHypothesis,
)
from nqlab.discovery.phase12_stats import _pass_ruin_mc, practical_stats
from nqlab.discovery.record import build_candidate_row
from nqlab.domain.calendar.sessions import parse_hhmm
from nqlab.domain.risk.payout import (
    PayoutProjection,
    WinningDayStats,
    payout_projection,
    scale_funded_fleet,
    simulate_funded_withdrawals,
    winning_day_stats,
)
from nqlab.experiment.tracker import ExperimentTracker
from nqlab.validation.gauntlet import run_gauntlet

_RTH_OPEN_DEFAULT = 570


@dataclass(frozen=True)
class Phase16Thresholds:
    """Payout-first minimum acceptance bar (after commissions + slippage)."""

    pf_min: float = 1.30
    pf_preferred: float = 1.50  # preferred (not required) PF — reported, not gated
    win_rate_min: float = 0.0  # min win rate (Phase 18 high-win-rate bar sets 0.55)
    ruin_max: float = 0.10
    pass_pref: float = 0.70  # preferred P(pass) on the best account size
    green_month_min: float = 0.85  # green months / total months
    min_trades: int = 60  # enough to judge
    min_winning_days_200: int = 5  # Topstep first-payout requirement
    min_winning_day_rate: float = 0.10  # >= 10% of trading days clear $200
    min_payout_eligible_frac: float = 0.50  # >= half the months are payout-eligible
    scale_pf_min: float = 1.20  # PF must not collapse at funded (higher-risk) sizing
    min_avg_contracts_scale: float = 1.5  # must use more than tiny 1-MNQ at funded sizing
    dsr_threshold: float = 0.90


PHASE16_THRESHOLDS = Phase16Thresholds()


def _green_months(trades: list[Any]) -> tuple[int, int]:
    by_month: dict[str, float] = defaultdict(float)
    for t in trades:
        by_month[t.entry_ts.strftime("%Y-%m")] += t.net_pnl
    vals = list(by_month.values())
    green = sum(1 for v in vals if v > 0)
    return green, len(vals)


def _avg_contracts(trades: list[Any]) -> float:
    return round(sum(t.contracts for t in trades) / len(trades), 2) if trades else 0.0


def _losing_streaks(trades: list[Any]) -> dict[str, Any]:
    """Consecutive-loss streak profile from the chronological trade sequence."""
    cur = 0
    longest = 0
    runs: list[int] = []
    for t in sorted(trades, key=lambda x: x.entry_ts):
        if t.net_pnl < 0:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
        longest = max(longest, cur)
    if cur > 0:
        runs.append(cur)
    return {
        "max_losing_streak": longest,
        "n_3loss_streaks": sum(1 for r in runs if r >= 3),
        "n_5loss_streaks": sum(1 for r in runs if r >= 5),
        "had_10loss_streak": longest >= 10,
    }


def business_stats(
    trades: list[Any],
    *,
    daily_pnl: dict[str, float],
    blown: bool,
    account: str,
    rules: TopstepRules,
    rules_by_account: dict[str, TopstepRules],
    months: float,
    runs: int = 2000,
    seed: int = 12345,
) -> dict[str, Any]:
    """Full payout-first stat panel from a gated real-account trade ledger + daily P&L."""
    n = len(trades)
    if n == 0:
        return {"trade_count": 0}
    ps = practical_stats(
        trades, account_size=rules.account_size, rules=rules, months=months,
        rules_by_account=rules_by_account, runs=runs, seed=seed,
    )
    green, total_m = _green_months(trades)
    # Per-account pass + ruin (survive 50K/100K/150K) from the trade distribution.
    nets = [t.net_pnl for t in trades]
    pass_by_acct: dict[str, float] = {}
    ruin_by_acct: dict[str, float] = {}
    for label, r in rules_by_account.items():
        p, _, ru = _pass_ruin_mc(nets, r, runs=runs, seed=seed)
        pass_by_acct[label] = round(p, 4)
        ruin_by_acct[label] = round(ru, 4)
    survivable = [a for a, ru in ruin_by_acct.items() if ru <= 0.10]
    best_account = max(survivable, key=lambda a: pass_by_acct[a], default="")
    wd: WinningDayStats = winning_day_stats(daily_pnl)
    proj: PayoutProjection = payout_projection(
        daily_pnl, rules=rules, months_total=round(months), blown=blown
    )
    wdr = simulate_funded_withdrawals(daily_pnl, rules=rules)
    # Green/red day + worst day/week (Phase-18 smooth-equity metrics).
    dvals = list(daily_pnl.values())
    green_days = sum(1 for d in dvals if d > 0)
    red_days = sum(1 for d in dvals if d < 0)
    worst_day = round(min(dvals), 2) if dvals else 0.0
    by_week: dict[tuple[int, int], float] = defaultdict(float)
    for day, val in daily_pnl.items():
        iso = date.fromisoformat(day).isocalendar()
        by_week[(iso[0], iso[1])] += val
    worst_week = round(min(by_week.values()), 2) if by_week else 0.0
    trading_weeks = total_m * 4.33
    out: dict[str, Any] = dict(ps)
    out.update(
        {
            "account": account,
            "blown": blown,
            "avg_contracts": _avg_contracts(trades),
            "trades_per_month": round(n / months, 2) if months > 0 else float(n),
            "trades_per_week": round(n / trading_weeks, 2) if trading_weeks > 0 else float(n),
            "green_months": green,
            "total_months": total_m,
            "green_month_frac": round(green / total_m, 4) if total_m else 0.0,
            "red_months": total_m - green,
            "trading_days": wd.trading_days,
            "green_days": green_days,
            "red_days": red_days,
            "green_day_rate": round(green_days / wd.trading_days, 4) if wd.trading_days else 0.0,
            "worst_day": worst_day,
            "worst_week": worst_week,
            **_losing_streaks(trades),
            "winning_days_200": wd.winning_days_200,
            "winning_days_500": wd.winning_days_500,
            "winning_day_rate": wd.winning_day_rate,
            "avg_winning_day": wd.avg_winning_day,
            "best_day": wd.best_day,
            "pass_by_account": pass_by_acct,
            "ruin_by_account": ruin_by_acct,
            "survivable_accounts": survivable,
            "best_account": best_account,
            "payout_eligible_months": proj.payout_eligible_months,
            "payout_every_month": proj.payout_every_month,
            "est_monthly_payout": proj.avg_monthly_payout,
            "est_annual_payout": proj.annual_payout,
            "total_projected_payout": proj.total_projected_payout,
            "payout_consistency_ok": proj.consistency_ok,
            "funded_n_payouts": wdr.n_payouts,
            "funded_total_withdrawn": wdr.total_withdrawn,
            "funded_avg_payout": wdr.avg_payout,
            "funded_blown_after_payout": wdr.blown_after_payout,
            "funded_months_to_first_payout": wdr.months_to_first_payout,
        }
    )
    return out


def _gated_trades(
    config: AppConfig,
    strat: Any,
    frame: pl.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    rules: TopstepRules,
) -> Any:
    """Run a strategy on a (lockbox) feature frame as a GATED real account."""
    sig = strat.signals(frame)
    bt_frame = sig.filter(pl.col("is_rth")) if config.backtest.rth_only else sig
    return run_backtest(
        bt_frame, symbol=symbol, timeframe=timeframe, strategy=strat.params, config=config,
        rules=rules, entry_mode="next_open", gate_entries=True, trailing=strat.trailing,
        exec_config=strat.exec_config,
    )


def select_phase16_survivors(
    tracker: ExperimentTracker,
    run_ids: tuple[str, ...],
    *,
    top_n: int = 8,
    min_oos_trades: int = 20,
    min_risk: float = 100.0,
) -> list[FreezeSpec]:
    """Freeze the top train/val gauntlet survivors (gauntlet ACCEPT), ranked by OOS net (a payout
    proxy). Uses ONLY the 2021-2024 search results — never the 2025-2026 lockbox."""
    cands: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for rid in run_ids:
        run = tracker.get_run(rid)
        if run is None:
            continue
        for r in tracker.load_candidates(rid):
            if r.get("gauntlet_verdict") != "ACCEPT":
                continue
            if int(r.get("oos_trades") or 0) < min_oos_trades:
                continue
            params = json.loads(str(r.get("params") or "{}"))
            if float(params.get("risk_per_trade") or 0.0) < min_risk:
                continue
            cands.append((rid, r, run, params))
    cands.sort(key=lambda c: float(c[1].get("oos_net") or 0.0), reverse=True)
    specs: list[FreezeSpec] = []
    seen: set[tuple[str, str, str, object, object]] = set()
    for rid, r, run, params in cands:
        fam, tf, acct = str(r["family"]), str(run["timeframe"]), str(run["account"]).lower()
        key = (fam, tf, acct, params.get("risk_per_trade"), params.get("sizing_mode"))
        if key in seen:
            continue
        seen.add(key)
        name = (
            f"{fam}_{tf}_{acct}_r{int(float(params.get('risk_per_trade') or 0))}"
            f"_{params.get('sizing_mode')}"
        )
        specs.append(FreezeSpec(name, rid, str(r["fingerprint"])))
        if len(specs) >= top_n:
            break
    return specs


def phase16_verdict(
    *,
    gauntlet_verdict: str,
    dsr: float,
    business: dict[str, Any],
    scale: dict[str, Any],
    thresholds: Phase16Thresholds,
    pristine: bool,
) -> tuple[str, str]:
    """Strict payout-first verdict. Rejects any candidate that misses ANY business gate."""
    th = thresholds
    if gauntlet_verdict == "NEEDS_MORE_DATA" or business.get("trade_count", 0) < th.min_trades:
        return NEEDS_MORE_DATA, (
            f"too few trades to judge (n={business.get('trade_count', 0)} < {th.min_trades})"
        )
    fails: list[str] = []
    pf = business.get("profit_factor")
    if pf is None or pf < th.pf_min:
        fails.append(f"PF {pf} < {th.pf_min} after costs (tiny edge)")
    if business.get("win_rate", 1.0) < th.win_rate_min:
        fails.append(f"win rate {business.get('win_rate')} < {th.win_rate_min}")
    if business.get("ruin", 1.0) > th.ruin_max:
        fails.append(f"risk of ruin {business.get('ruin')} > {th.ruin_max}")
    if business.get("green_month_frac", 0.0) < th.green_month_min:
        fails.append(
            f"green months {business.get('green_month_frac')} < {th.green_month_min}"
        )
    if business.get("winning_days_200", 0) < th.min_winning_days_200:
        fails.append(f"only {business.get('winning_days_200')} $200+ winning days")
    if business.get("winning_day_rate", 0.0) < th.min_winning_day_rate:
        rate = business.get("winning_day_rate")
        fails.append(f"$200-day rate {rate} < {th.min_winning_day_rate}")
    if business.get("payout_eligible_months", 0) < th.min_payout_eligible_frac * business.get(
        "total_months", 1
    ):
        fails.append(
            f"only {business.get('payout_eligible_months')}/{business.get('total_months')} "
            "payout-eligible months"
        )
    if business.get("blown"):
        fails.append("account blown on lockbox")
    # Scalability (Stage B funded sizing): PF must hold and it must use more than 1 MNQ.
    if scale.get("blown"):
        fails.append("blows account at funded scale sizing")
    if (scale.get("profit_factor") or 0.0) < th.scale_pf_min:
        fails.append(f"PF collapses to {scale.get('profit_factor')} at funded scale sizing")
    if scale.get("avg_contracts", 0.0) < th.min_avg_contracts_scale:
        fails.append(
            f"depends on tiny sizing (avg {scale.get('avg_contracts')} contracts at funded risk)"
        )
    if dsr < th.dsr_threshold:
        fails.append(f"fails Deflated-Sharpe at frozen count (DSR={dsr:.2f} < {th.dsr_threshold})")
    if fails:
        return REJECT, "; ".join(fails)
    # Cleared every business gate.
    p_pass = max((business.get("p_pass_by_account") or {}).values(), default=0.0)
    pref = "" if p_pass >= th.pass_pref else f" (P(pass)={p_pass:.0%} < {th.pass_pref:.0%} pref)"
    if pristine:
        return ACCEPT_FOR_PAPER, (
            f"clears the full payout-first bar on the pristine 2025-2026 lockbox: PF={pf}, "
            f"ruin={business.get('ruin')}, green={business.get('green_month_frac')}, "
            f"{business.get('winning_days_200')} $200-days, scales to "
            f"{scale.get('avg_contracts')} contracts (DSR={dsr:.2f}){pref}"
        )
    return ACCEPT_FOR_FORWARD_PAPER, (
        f"clears the payout-first bar but lockbox not pristine -> forward-confirm{pref}"
    )


def run_phase16_lockbox(
    config: AppConfig,
    frozen: list[FrozenHypothesis],
    *,
    tracker: ExperimentTracker,
    run_id: str,
    lockbox_start: date,
    lockbox_end: date,
    frame_for: Callable[[str], pl.DataFrame],
    rules_for: Callable[[str], TopstepRules],
    rules_by_account: dict[str, TopstepRules],
    months: float,
    scale_risk: float,
    scale_account: str = "150k",
    fleet_sizes: tuple[int, ...] = (5, 10),
    pristine: bool,
    created_at: str,
    symbol: str = "NQ.c",
    thresholds: Phase16Thresholds = PHASE16_THRESHOLDS,
    verdict_fn: Callable[..., tuple[str, str]] = phase16_verdict,
) -> list[dict[str, Any]]:
    """Run frozen Phase-16/17 hypotheses on the lockbox: gauntlet + payout-first business stats +
    two-stage scaling (eval + funded withdrawals), then assign a strict verdict to each.

    ``verdict_fn`` / ``thresholds`` are injected so Phase 17 reuses this with its own bar.
    """
    v = config.validation
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    evaluated: list[tuple[FrozenHypothesis, Any, dict[str, Any], dict[str, Any]]] = []
    for fh in frozen:
        frame = frame_for(fh.timeframe)
        strat = fh.to_strategy(rth_start_min)
        rules = rules_for(fh.account)
        result = run_gauntlet(
            config, strat, symbol=symbol, timeframe=fh.timeframe, start=lockbox_start,
            end=lockbox_end, rules=rules, feature_frame=frame,
        )
        # Stage A: gated real-account view at the frozen (evaluation) sizing.
        bt = _gated_trades(config, strat, frame, symbol=symbol, timeframe=fh.timeframe, rules=rules)
        biz = business_stats(
            bt.trades, daily_pnl=bt.topstep.daily_pnl, blown=bt.topstep.blown, account=fh.account,
            rules=rules, rules_by_account=rules_by_account, months=months,
            runs=v.monte_carlo_runs, seed=v.monte_carlo_seed,
        )
        # Stage B: funded scale sizing on a 150K account (higher risk, multi-contract).
        scale_rules = rules_by_account[scale_account]
        scale_strat = _ALL_FAMILIES[fh.family](
            {**fh.param_dict(), "risk_per_trade": scale_risk}, rth_start_min=rth_start_min
        )
        sbt = _gated_trades(
            config, scale_strat, frame, symbol=symbol, timeframe=fh.timeframe, rules=scale_rules
        )
        sbiz = business_stats(
            sbt.trades, daily_pnl=sbt.topstep.daily_pnl, blown=sbt.topstep.blown,
            account=scale_account, rules=scale_rules, rules_by_account=rules_by_account,
            months=months, runs=v.monte_carlo_runs, seed=v.monte_carlo_seed,
        )
        fleet = {
            n: scale_funded_fleet(
                per_account_monthly_payout=float(sbiz.get("est_monthly_payout") or 0.0),
                per_account_ruin=float(sbiz.get("ruin") or 0.0),
                n_accounts=n, account_size=scale_rules.account_size,
            ).__dict__
            for n in fleet_sizes
        }
        sbiz["fleet"] = fleet
        evaluated.append((fh, result, biz, sbiz))

    n_frozen = len(frozen)
    sr0 = expected_max_sharpe(
        sharpe_variance([r.in_sample_sharpe for _, r, _, _ in evaluated]), n_frozen
    )
    tracker.upsert_run(
        {
            "run_id": run_id, "symbol": symbol, "timeframe": "mixed", "account": "frozen",
            "start_date": lockbox_start.isoformat(), "end_date": lockbox_end.isoformat(),
            "n_trials": n_frozen, "pbo": None, "dsr_threshold": v.dsr_threshold,
            "status": "complete", "created_at": created_at,
        }
    )

    rows: list[dict[str, Any]] = []
    for trial, (fh, result, biz, sbiz) in enumerate(evaluated):
        dsr = deflated_sharpe_ratio(
            result.in_sample_sharpe, sr0, int(result.in_sample_n or 0),
            result.in_sample_skew, result.in_sample_kurt,
        )
        verdict, reason = verdict_fn(
            gauntlet_verdict=result.verdict, dsr=dsr, business=biz, scale=sbiz,
            thresholds=thresholds, pristine=pristine,
        )
        row = build_candidate_row(
            result, run_id=run_id, trial=trial, family=fh.family,
            fingerprint=fh.phase9_fingerprint, params=fh.param_dict(), feature_set="16",
            train_start=v.train_start, train_end=v.train_end, val_start=v.test_start,
            val_end=v.test_end, config_hash=fh.config_hash(), data_version="lockbox",
            git_hash="lockbox", artifacts_path="", created_at=created_at,
        )
        row["deflated_sharpe"] = dsr
        row["sr0"] = sr0
        row["gauntlet_verdict"] = result.verdict
        row["final_verdict"] = verdict
        row["rejection_reason"] = reason
        row["strategy_name"] = fh.name
        row["fingerprint"] = fh.name
        row["business_stats"] = json.dumps(biz, default=str)
        row["scale_stats"] = json.dumps(sbiz, default=str)
        # Persist the key payout numbers in the diagnostics JSON column (the tracker only
        # stores known columns), so `report phase16` can read them back from the DB.
        diag = json.loads(str(row.get("diagnostics") or "{}"))
        diag["business"] = {
            k: biz.get(k)
            for k in (
                "trade_count", "profit_factor", "expectancy", "win_rate", "ruin",
                "green_months", "total_months", "green_month_frac", "red_months",
                "green_days", "red_days", "green_day_rate", "worst_day", "worst_week",
                "winning_days_200", "winning_days_500", "winning_day_rate", "avg_winning_day",
                "best_day", "payout_eligible_months", "payout_every_month", "est_monthly_payout",
                "est_annual_payout", "p_pass_by_account", "pass_by_account", "ruin_by_account",
                "survivable_accounts", "best_account", "avg_contracts", "trades_per_month",
                "trades_per_week", "max_drawdown", "funded_n_payouts", "funded_total_withdrawn",
                "funded_avg_payout", "funded_blown_after_payout", "funded_months_to_first_payout",
            )
        }
        diag["scale"] = {
            k: sbiz.get(k)
            for k in ("profit_factor", "avg_contracts", "ruin", "est_monthly_payout", "blown")
        }
        diag["fleet"] = sbiz.get("fleet")
        row["diagnostics"] = json.dumps(diag, default=str)
        tracker.save_candidate(row)
        rows.append(row)
    return rows
