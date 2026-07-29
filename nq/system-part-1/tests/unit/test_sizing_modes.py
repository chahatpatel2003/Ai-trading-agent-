"""Phase 8 instrument-sizing modes: adaptive / nq_only / mnq_only / max_1_nq / mnq_scaling."""

from __future__ import annotations

import pytest

from nqlab.config.models import AppConfig
from nqlab.domain.sizing import SIZING_MODES, SizingDecision, size_position


def _size(config: AppConfig, *, risk: float, mode: str, stop: float = 5.0,
          cap: int = 5) -> SizingDecision:
    return size_position(
        stop_points=stop, risk_usd=risk, nq=config.instruments.specs["NQ"],
        mnq=config.instruments.specs["MNQ"], max_contracts=cap,
        mode=mode,  # type: ignore[arg-type]
    )


def test_all_modes_are_known() -> None:
    assert set(SIZING_MODES) == {"adaptive", "nq_only", "mnq_only", "max_1_nq", "mnq_scaling"}


def test_adaptive_prefers_nq_then_falls_back(app_config: AppConfig) -> None:
    # stop 5 pts -> NQ risk $100/contract, MNQ $10/contract.
    big = _size(app_config, risk=250, mode="adaptive")
    small = _size(app_config, risk=25, mode="adaptive")
    assert big.instrument == "NQ" and big.contracts == 2
    assert small.instrument == "MNQ" and small.contracts == 2


def test_nq_only_skips_when_unaffordable(app_config: AppConfig) -> None:
    skip = _size(app_config, risk=25, mode="nq_only")
    ok = _size(app_config, risk=100, mode="nq_only")
    assert skip.skipped and skip.instrument == ""
    assert ok.instrument == "NQ" and ok.contracts == 1


def test_mnq_only_never_uses_nq(app_config: AppConfig) -> None:
    d = _size(app_config, risk=250, mode="mnq_only")
    assert d.instrument == "MNQ"
    assert d.contracts == 5  # capped at max_contracts (250/10=25 -> 5)


def test_max_1_nq_caps_nq_then_falls_back(app_config: AppConfig) -> None:
    capped = _size(app_config, risk=250, mode="max_1_nq")
    micro = _size(app_config, risk=25, mode="max_1_nq")
    assert capped.instrument == "NQ" and capped.contracts == 1  # never more than 1 NQ
    assert micro.instrument == "MNQ"  # 1 NQ unaffordable -> micro fallback


def test_mnq_scaling_allows_10x_micro_stack(app_config: AppConfig) -> None:
    d = _size(app_config, risk=250, mode="mnq_scaling")
    assert d.instrument == "MNQ"
    assert d.contracts == 25  # 250/10 = 25, under the 5*10 micro cap


def test_non_positive_stop_is_skipped(app_config: AppConfig) -> None:
    for mode in SIZING_MODES:
        assert _size(app_config, risk=100, mode=mode, stop=0.0).skipped


@pytest.mark.parametrize("mode", SIZING_MODES)
def test_contracts_never_exceed_cap(app_config: AppConfig, mode: str) -> None:
    d = _size(app_config, risk=10_000, mode=mode)
    cap = 50 if mode == "mnq_scaling" else 5
    assert d.contracts <= cap
