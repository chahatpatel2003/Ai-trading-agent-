from __future__ import annotations

from nqlab.config.models import InstrumentSpec
from nqlab.domain.sizing import size_position

NQ = InstrumentSpec(root="NQ", point_value=20.0, tick_size=0.25, tick_value=5.0)
MNQ = InstrumentSpec(root="MNQ", point_value=2.0, tick_size=0.25, tick_value=0.5)


def _size(stop: float, risk: float, max_contracts: int = 5):
    return size_position(
        stop_points=stop, risk_usd=risk, nq=NQ, mnq=MNQ, max_contracts=max_contracts
    )


def test_uses_nq_when_one_contract_fits() -> None:
    d = _size(stop=10.0, risk=250.0)  # 250 / (10*20) = 1.25 -> 1 NQ
    assert d.instrument == "NQ"
    assert d.contracts == 1
    assert not d.skipped


def test_falls_back_to_mnq_when_nq_too_big() -> None:
    d = _size(stop=20.0, risk=100.0)  # NQ: 100/400 = 0; MNQ: 100/40 = 2
    assert d.instrument == "MNQ"
    assert d.contracts == 2


def test_skips_when_even_one_mnq_too_big() -> None:
    d = _size(stop=100.0, risk=50.0)  # NQ 50/2000=0, MNQ 50/200=0 -> skip
    assert d.skipped
    assert d.instrument == ""


def test_caps_at_max_contracts() -> None:
    d = _size(stop=5.0, risk=10000.0, max_contracts=5)  # 10000/100 = 100 -> cap 5
    assert d.instrument == "NQ"
    assert d.contracts == 5


def test_nonpositive_stop_is_skipped() -> None:
    assert _size(stop=0.0, risk=100.0).skipped
