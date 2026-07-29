"""Phase 8 feasibility generators + sizing-mode flow through the engine."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from nqlab.backtest.engine import run_backtest
from nqlab.config.models import AppConfig, TopstepRules
from nqlab.discovery.harness import run_discovery
from nqlab.discovery.optimization import (
    EXIT_BY_TF,
    RISK_GRID_FEASIBILITY,
    SIZING_MODES,
    account_risk_grid,
    generate_sizing_candidates,
    generate_timeframe_candidates,
)
from nqlab.domain.strategy.base import ExecConfig, StrategyParams
from nqlab.experiment.tracker import ExperimentTracker
from nqlab.features.build import build_features
from nqlab.features.context import build_context_features
from tests.synth import add_synth_mbp1_columns, engine_frame, globex_bars

_PARAMS = StrategyParams("test", "1")


def test_account_risk_grid_caps() -> None:
    assert 500.0 in account_risk_grid("150k")
    assert 500.0 not in account_risk_grid("100k") and 300.0 in account_risk_grid("100k")
    assert max(account_risk_grid("50k")) == 200.0


def test_exit_seeds_exist_for_all_timeframes() -> None:
    assert set(EXIT_BY_TF) == {"1m", "5m", "15m"}
    for tf, seed in EXIT_BY_TF.items():
        assert {"stop_atr", "target_atr", "min_hold_bars", "time_stop_bars"} <= set(seed), tf


def test_sizing_candidate_grid_shape_and_uniqueness() -> None:
    fams = ("bias_pullback", "pd_sweep_rev")
    cands = generate_sizing_candidates(fams, timeframe="5m", risk_grid=account_risk_grid("150k"))
    assert len(cands) == len(fams) * len(SIZING_MODES) * len(RISK_GRID_FEASIBILITY)
    fps = [c.params.fingerprint() for c in cands]
    assert len(fps) == len(set(fps))
    modes = {c.exec_config.sizing_mode for c in cands}  # type: ignore[union-attr]
    assert modes == set(SIZING_MODES)


def test_timeframe_candidate_grid_carries_fixed_sizing() -> None:
    cands = generate_timeframe_candidates(
        ("bias_pullback",), timeframe="15m", sizing_mode="mnq_only"
    )
    assert len(cands) == len(RISK_GRID_FEASIBILITY)
    assert all(c.exec_config.sizing_mode == "mnq_only" for c in cands)  # type: ignore[union-attr]
    # 15m exit seed applied.
    assert all(c.params.as_dict()["stop_atr"] == EXIT_BY_TF["15m"]["stop_atr"] for c in cands)


def test_sizing_mode_changes_engine_instrument(
    app_config: AppConfig, rules_50k: TopstepRules
) -> None:
    bars = [
        {"close": 100, "high": 100, "low": 100, "signal": 1, "stop_pts": 5,
         "target_pts": 10, "minute": 600},
        {"open": 100, "high": 100, "low": 100, "close": 100, "minute": 605},
        {"open": 100, "high": 112, "low": 100, "close": 110, "minute": 610},
    ]

    def run(mode: str):
        return run_backtest(
            engine_frame(bars), symbol="NQ.c", timeframe="5m", strategy=_PARAMS,
            config=app_config, rules=rules_50k,
            exec_config=ExecConfig(risk_per_trade_usd=250.0, sizing_mode=mode),
        )

    assert run("nq_only").trades[0].instrument == "NQ"
    assert run("mnq_only").trades[0].instrument == "MNQ"
    assert run("max_1_nq").trades[0].contracts == 1
    assert run("mnq_scaling").trades[0].instrument == "MNQ"


def _setup_frame() -> pl.DataFrame:
    feats = add_synth_mbp1_columns(build_features(globex_bars(days=2), rth_start_min=570))
    return build_context_features(feats, rth_start_min=570)


def test_feasibility_run_is_deterministic(
    config: AppConfig, rules_50k: TopstepRules, tmp_path: Path
) -> None:
    frame = _setup_frame()
    candidates = generate_sizing_candidates(
        ("bias_pullback",), timeframe="5m", risk_grid=(25.0, 50.0),
        sizing_modes=("mnq_only", "mnq_scaling"),
    )

    def run(db: str):
        return run_discovery(
            config, symbol="NQ.c", timeframe="1m", start=date(2024, 1, 2), end=date(2024, 1, 3),
            rules=rules_50k, tracker=ExperimentTracker(tmp_path / db), run_id="feas",
            candidates=list(candidates), feature_frame=frame, created_at="t",
        )

    r1, r2 = run("a.duckdb"), run("b.duckdb")
    for row in r1.leaderboard:
        assert row["final_verdict"] in {"ACCEPT", "REJECT", "NEEDS_MORE_DATA"}

    def key(rows):
        return sorted((r["fingerprint"], round(float(r["oos_net"]), 4)) for r in rows)

    assert key(r1.leaderboard) == key(r2.leaderboard)
