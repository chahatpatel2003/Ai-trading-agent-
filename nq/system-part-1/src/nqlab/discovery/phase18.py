"""Phase 18 — high-win-rate, high-frequency NY-AM scalp acceptance.

Tests the new hypothesis (smooth equity from many controlled green trades, not rare large
winners). Reuses the Phase-16 payout-first rig but: (1) adds a win-rate floor, (2) sweeps the
daily risk-control matrix (profit target / max loss / stop-after-N) per frozen survivor and keeps
the best, and (3) caps the best verdict at ``ACCEPT_FOR_FORWARD_PAPER`` (this is a forward-paper
hypothesis test). No forced acceptance.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Any

import polars as pl

from nqlab.config.models import AppConfig, TopstepRules
from nqlab.discovery.deflation import deflated_sharpe_ratio, expected_max_sharpe, sharpe_variance
from nqlab.discovery.lockbox import (
    ACCEPT_FOR_FORWARD_PAPER,
    NEEDS_FORWARD_DATA,
    NEEDS_MORE_DATA,
    FrozenHypothesis,
)
from nqlab.discovery.phase16 import (
    Phase16Thresholds,
    _gated_trades,
    business_stats,
    phase16_verdict,
    select_phase16_survivors,
)
from nqlab.discovery.phase18_families import PHASE18_FAMILIES, generate_phase18_candidates
from nqlab.discovery.record import build_candidate_row
from nqlab.domain.calendar.sessions import parse_hhmm
from nqlab.domain.risk.payout import scale_funded_fleet
from nqlab.experiment.tracker import ExperimentTracker
from nqlab.validation.gauntlet import run_gauntlet

__all__ = [
    "PHASE18_FAMILIES",
    "PHASE18_THRESHOLDS",
    "generate_phase18_candidates",
    "phase18_verdict",
    "run_phase18_lockbox",
    "select_phase16_survivors",
]

_RTH_OPEN_DEFAULT = 570
# (frozen hypothesis, gauntlet result, stage-A business, stage-B scale, best-controls)
_EvalRow = tuple[FrozenHypothesis, Any, dict[str, Any], dict[str, Any], dict[str, Any]]

# Phase 18 bar: win rate >= 55%, PF >= 1.30 (prefer 1.50), ruin <= 10%, green months >= 75%,
# payout-eligible >= 60%, pass >= 70% preferred.
PHASE18_THRESHOLDS = Phase16Thresholds(
    pf_min=1.30,
    pf_preferred=1.50,
    win_rate_min=0.55,
    ruin_max=0.10,
    pass_pref=0.70,
    green_month_min=0.75,
    min_trades=60,
    min_winning_days_200=5,
    min_winning_day_rate=0.10,
    min_payout_eligible_frac=0.60,
    scale_pf_min=1.20,
    min_avg_contracts_scale=1.5,
    dsr_threshold=0.90,
)

# Daily risk-control matrix swept at the lockbox (real-account overlay).
_PROFIT_TARGETS: tuple[float | None, ...] = (None, 200.0, 300.0, 500.0)
_MAX_LOSSES: tuple[int | None, ...] = (None, 1, 2)
_MAX_TRADES: tuple[int | None, ...] = (None, 3, 5, 10)


def phase18_verdict(
    *,
    gauntlet_verdict: str,
    dsr: float,
    business: dict[str, Any],
    scale: dict[str, Any],
    thresholds: Phase16Thresholds,
    pristine: bool,
) -> tuple[str, str]:
    """Phase-18 verdict: payout-first bar (+ win-rate floor), capped at ACCEPT_FOR_FORWARD_PAPER.

    This is a forward-paper hypothesis test, so a clean pass earns forward-paper, never live
    ACCEPT_FOR_PAPER. MBP1 forward-data ceiling honoured (Phase-18 families are trades-derivable).
    """
    base_verdict, reason = phase16_verdict(
        gauntlet_verdict=gauntlet_verdict, dsr=dsr, business=business, scale=scale,
        thresholds=thresholds, pristine=pristine,
    )
    if (
        bool(business.get("requires_mbp1"))
        and not bool(business.get("mbp1_clean", True))
        and base_verdict != NEEDS_MORE_DATA
    ):
        return NEEDS_FORWARD_DATA, (
            "MBP1-required without a clean MBP1 lockbox; cannot exceed NEEDS_FORWARD_DATA "
            f"[would-be: {base_verdict} — {reason}]"
        )
    # Cap the ceiling at forward-paper for this hypothesis test.
    if base_verdict.startswith("ACCEPT"):
        return ACCEPT_FOR_FORWARD_PAPER, reason.replace("ACCEPT_FOR_PAPER", "forward-paper")
    return base_verdict, reason


def _best_daily_controls(
    config: AppConfig,
    fh: FrozenHypothesis,
    frame: pl.DataFrame,
    *,
    rules: TopstepRules,
    rules_by_account: dict[str, TopstepRules],
    months: float,
    symbol: str,
    runs: int,
    seed: int,
    rth_start_min: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sweep the daily risk-control matrix on the gated real account; keep the set that maximises
    payout-eligible months (then PF). Returns (best_business_stats, best_control_config)."""
    from nqlab.discovery.lockbox import _ALL_FAMILIES

    base = dict(fh.param_dict())
    best_biz: dict[str, Any] = {}
    best_ctrl: dict[str, Any] = {}
    best_key = (-1, -1.0)
    for pt in _PROFIT_TARGETS:
        for ml in _MAX_LOSSES:
            for mt in _MAX_TRADES:
                strat = _ALL_FAMILIES[fh.family](
                    {
                        **base,
                        "daily_profit_target": pt,
                        "max_daily_losses": ml,
                        "max_daily_trades": mt,
                    },
                    rth_start_min=rth_start_min,
                )
                bt = _gated_trades(
                    config, strat, frame, symbol=symbol, timeframe=fh.timeframe, rules=rules
                )
                if not bt.trades:
                    continue
                biz = business_stats(
                    bt.trades, daily_pnl=bt.topstep.daily_pnl, blown=bt.topstep.blown,
                    account=fh.account, rules=rules, rules_by_account=rules_by_account,
                    months=months, runs=runs, seed=seed,
                )
                key = (biz.get("payout_eligible_months", 0), biz.get("profit_factor") or 0.0)
                if key > best_key:
                    best_key = key
                    best_biz = biz
                    best_ctrl = {"daily_profit_target": pt, "max_daily_losses": ml,
                                 "max_daily_trades": mt}
    return best_biz, best_ctrl


def run_phase18_lockbox(
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
    thresholds: Phase16Thresholds = PHASE18_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Run frozen scalp survivors on the lockbox: gauntlet + best-daily-control business stats +
    funded-scale, then assign the Phase-18 verdict (forward-paper ceiling)."""
    v = config.validation
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    from nqlab.discovery.lockbox import _ALL_FAMILIES

    evaluated: list[_EvalRow] = []
    for fh in frozen:
        frame = frame_for(fh.timeframe)
        strat = fh.to_strategy(rth_start_min)
        rules = rules_for(fh.account)
        result = run_gauntlet(
            config, strat, symbol=symbol, timeframe=fh.timeframe, start=lockbox_start,
            end=lockbox_end, rules=rules, feature_frame=frame,
        )
        # Stage A: best daily-control overlay on the gated real account.
        biz, ctrl = _best_daily_controls(
            config, fh, frame, rules=rules, rules_by_account=rules_by_account, months=months,
            symbol=symbol, runs=v.monte_carlo_runs, seed=v.monte_carlo_seed,
            rth_start_min=rth_start_min,
        )
        biz["best_controls"] = ctrl
        # Stage B: funded scale sizing on a 150K account.
        scale_rules = rules_by_account[scale_account]
        scale_strat = _ALL_FAMILIES[fh.family](
            {**fh.param_dict(), "risk_per_trade": scale_risk, **ctrl}, rth_start_min=rth_start_min
        )
        sbt = _gated_trades(
            config, scale_strat, frame, symbol=symbol, timeframe=fh.timeframe, rules=scale_rules
        )
        sbiz: dict[str, Any] = (
            business_stats(
                sbt.trades, daily_pnl=sbt.topstep.daily_pnl, blown=sbt.topstep.blown,
                account=scale_account, rules=scale_rules, rules_by_account=rules_by_account,
                months=months, runs=v.monte_carlo_runs, seed=v.monte_carlo_seed,
            )
            if sbt.trades
            else {"trade_count": 0, "profit_factor": 0.0, "avg_contracts": 0.0,
                  "blown": True, "est_monthly_payout": 0.0, "ruin": 0.0}
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
        evaluated.append((fh, result, biz, sbiz, ctrl))

    n_frozen = len(frozen)
    sr0 = expected_max_sharpe(
        sharpe_variance([r.in_sample_sharpe for _, r, _, _, _ in evaluated]), n_frozen
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
    for trial, (fh, result, biz, sbiz, ctrl) in enumerate(evaluated):
        dsr = deflated_sharpe_ratio(
            result.in_sample_sharpe, sr0, int(result.in_sample_n or 0),
            result.in_sample_skew, result.in_sample_kurt,
        )
        verdict, reason = phase18_verdict(
            gauntlet_verdict=result.verdict, dsr=dsr, business=biz, scale=sbiz,
            thresholds=thresholds, pristine=pristine,
        )
        row = build_candidate_row(
            result, run_id=run_id, trial=trial, family=fh.family,
            fingerprint=fh.phase9_fingerprint, params=fh.param_dict(), feature_set="18",
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
        diag = json.loads(str(row.get("diagnostics") or "{}"))
        diag["business"] = {
            k: biz.get(k)
            for k in (
                "trade_count", "profit_factor", "expectancy", "win_rate", "ruin", "avg_win",
                "avg_loss", "green_months", "total_months", "green_month_frac", "red_months",
                "green_days", "red_days", "green_day_rate", "worst_day", "worst_week",
                "trades_per_day", "trades_per_month", "trades_per_week", "winning_days_200",
                "winning_days_500", "winning_day_rate", "best_day", "max_drawdown", "avg_drawdown",
                "payout_eligible_months", "est_monthly_payout", "est_annual_payout",
                "p_pass_by_account", "pass_by_account", "ruin_by_account", "best_account",
                "avg_contracts", "est_days_to_pass", "max_losing_streak", "n_3loss_streaks",
                "n_5loss_streaks", "had_10loss_streak",
            )
        }
        diag["best_controls"] = ctrl
        diag["scale"] = {
            k: sbiz.get(k)
            for k in ("profit_factor", "avg_contracts", "ruin", "est_monthly_payout", "blown")
        }
        diag["fleet"] = sbiz.get("fleet")
        row["diagnostics"] = json.dumps(diag, default=str)
        tracker.save_candidate(row)
        rows.append(row)
    return rows
