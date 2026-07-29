from __future__ import annotations

import pytest

from tslab.sim.costs import CostModel


def test_nq_side_cost_market(cfg):
    cm = CostModel(cfg)
    sc = cm.side_cost("NQ", qty=1, order_type="market")
    assert sc.fees == pytest.approx(1.90)            # official $3.80 RT / 2
    assert sc.spread == pytest.approx(0.5 * 1 * 5.0)  # half a 1-tick spread
    assert sc.slippage == pytest.approx(1 * 5.0)
    assert sc.total == pytest.approx(1.90 + 2.50 + 5.00)


def test_gc_round_trip(cfg):
    cm = CostModel(cfg)
    rt = cm.round_trip_cost("GC", qty=1)
    # per side: 2.12 fees + 5.00 half-spread + 10.00 slippage = 17.12
    assert rt == pytest.approx(2 * 17.12)


def test_limit_orders_no_slippage(cfg):
    cm = CostModel(cfg)
    sc = cm.side_cost("NQ", qty=1, order_type="limit")
    assert sc.slippage == 0


def test_stop_orders_slip_more(cfg):
    cm = CostModel(cfg)
    assert (cm.side_cost("NQ", 1, "stop").slippage
            > cm.side_cost("NQ", 1, "market").slippage)


def test_qty_scaling(cfg):
    cm = CostModel(cfg)
    assert (cm.round_trip_cost("MNQ", qty=3)
            == pytest.approx(3 * cm.round_trip_cost("MNQ", qty=1)))


def test_all_costs_verified(cfg):
    cm = CostModel(cfg)
    assert cm.unverified_instruments == []


def test_bad_inputs(cfg):
    cm = CostModel(cfg)
    with pytest.raises(ValueError):
        cm.side_cost("NQ", qty=0)
    with pytest.raises(ValueError):
        cm.side_cost("NQ", qty=1, order_type="teleport")
    with pytest.raises(KeyError):
        cm.side_cost("ES", qty=1)
