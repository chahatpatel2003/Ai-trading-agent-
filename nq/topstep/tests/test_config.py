from __future__ import annotations

import pytest

from tslab.config import ConfigError


def test_pricing_matches_user_numbers(cfg):
    assert cfg.pricing_plan("standard", "50k").monthly == 49
    assert cfg.pricing_plan("standard", "100k").monthly == 99
    assert cfg.pricing_plan("standard", "150k").monthly == 199
    for size in ("50k", "100k", "150k"):
        assert cfg.pricing_plan("standard", size).activation == 149
        assert cfg.pricing_plan("no_activation_fee", size).activation == 0
    assert cfg.pricing_plan("no_activation_fee", "50k").monthly == 95
    assert cfg.pricing_plan("no_activation_fee", "100k").monthly == 149
    assert cfg.pricing_plan("no_activation_fee", "150k").monthly == 229


def test_reset_fee_is_monthly_rebill(cfg):
    assert cfg.reset_fee("standard", "50k") == 49
    assert cfg.reset_fee("no_activation_fee", "150k") == 229


def test_account_parameters(cfg):
    a = cfg.accounts["50k"]
    assert (a.start_balance, a.profit_target, a.mll, a.dll) == (50000, 3000, 2000, 1000)
    a = cfg.accounts["150k"]
    assert (a.profit_target, a.mll, a.max_contracts_minis) == (9000, 4500, 15)


def test_effective_dll_options(cfg):
    assert cfg.effective_dll("50k") == 1000                    # default
    for v in (300, 500, 750, 1000):
        assert cfg.effective_dll("50k", v) == v
    assert cfg.effective_dll("100k", 500) == 500               # personal below account DLL
    assert cfg.effective_dll("100k", 1000) == 1000             # account DLL 2000 > personal
    with pytest.raises(ConfigError):
        cfg.effective_dll("50k", 400)                          # not in allowed set


def test_instrument_specs_consistent(cfg):
    nq, gc = cfg.instruments["NQ"], cfg.instruments["GC"]
    assert nq.tick_size == 0.25 and nq.tick_value == 5.0 and nq.point_value == 20.0
    assert gc.tick_size == 0.10 and gc.tick_value == 10.0 and gc.point_value == 100.0
    assert cfg.instruments["MNQ"].tick_value == 0.5
    assert cfg.instruments["MGC"].tick_value == 1.0


def test_costs_verified_and_match_official_round_turns(cfg):
    # official TopstepX RT totals: NQ 3.80, MNQ 1.24, GC 4.24, MGC 1.74
    expected = {"NQ": 3.80, "MNQ": 1.24, "GC": 4.24, "MGC": 1.74}
    for sym, rt in expected.items():
        c = cfg.costs[sym]
        assert c.verified, f"{sym} costs not marked verified"
        assert 2 * c.fees_per_side == pytest.approx(rt), sym


def test_scaling_plan(cfg):
    fr = cfg.funded_rules
    assert fr.scaling_max_minis("50k", 0) == 2
    assert fr.scaling_max_minis("50k", 1499.99) == 2
    assert fr.scaling_max_minis("50k", 1500) == 3        # "$1,500-$2,000" tier
    assert fr.scaling_max_minis("50k", 2000) == 3        # boundary -> lower tier
    assert fr.scaling_max_minis("50k", 2000.01) == 5     # "above $2,000"
    assert fr.scaling_max_minis("100k", 2500) == 5
    assert fr.scaling_max_minis("100k", 3000) == 5
    assert fr.scaling_max_minis("100k", 3000.01) == 10
    assert fr.scaling_max_minis("150k", 10000) == 15


def test_volatility_restrictions_gold(cfg):
    assert cfg.instrument_max_minis("50k", "GC") == 3
    assert cfg.instrument_max_minis("150k", "GC") == 9
    assert cfg.instrument_max_minis("50k", "NQ") is None
    # MGC caps are micro-denominated: they must NOT surface as mini caps
    assert cfg.instrument_max_minis("50k", "MGC") is None
    assert cfg.instrument_max_micros("50k", "MGC") == 30
    assert cfg.instrument_max_micros("150k", "MGC") == 90
    assert cfg.instrument_max_micros("50k", "GC") is None


def test_outputs_dir_not_under_data_roots(cfg):
    for root in cfg.paths.data_roots():
        assert not str(cfg.paths.outputs_dir).startswith(str(root))


def test_payout_policy_verified_values(cfg):
    p = cfg.payout
    assert p.winning_day_min_profit == 150.0
    assert p.min_winning_days == 5
    assert p.per_payout_cap == {"50k": 2000, "100k": 3000, "150k": 5000}
    assert p.profit_split_trader == 0.90
    assert p.min_payout == 125.0
