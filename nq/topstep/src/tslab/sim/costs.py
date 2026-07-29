"""Trading cost model: fees + spread + slippage, in dollars.

Cost of a filled order side =
    fees_per_side * qty
  + half_spread_ticks * tick_value * qty          (crossing the spread)
  + slippage_ticks(order_type) * tick_value * qty

The bar-based fill model ("bar_conservative") charges these on both entry and
exit. A book-based model for NQ (MBP-1) replaces the spread term with actual
quoted prices later; fees and slippage terms remain.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Config, CostSpec, InstrumentSpec


@dataclass(frozen=True)
class SideCost:
    fees: float
    spread: float
    slippage: float

    @property
    def total(self) -> float:
        return self.fees + self.spread + self.slippage


class CostModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        unverified = [c.instrument for c in cfg.costs.values() if not c.verified]
        self.unverified_instruments = unverified

    def _pair(self, instrument: str) -> tuple:
        try:
            return self.cfg.instruments[instrument], self.cfg.costs[instrument]
        except KeyError:
            raise KeyError(f"no instrument/cost spec for {instrument!r}")

    def side_cost(self, instrument: str, qty: int, order_type: str = "market",
                  overnight: bool = False) -> SideCost:
        """Dollar cost of one side (entry OR exit) for qty contracts."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        inst, cost = self._pair(instrument)
        spread_ticks = (cost.spread_ticks_overnight if overnight
                        else cost.spread_ticks_rth)
        slip_ticks = {"market": cost.slippage_ticks_market,
                      "stop": cost.slippage_ticks_stop,
                      "limit": 0.0}.get(order_type)
        if slip_ticks is None:
            raise ValueError(f"unknown order_type {order_type!r}")
        return SideCost(
            fees=cost.fees_per_side * qty,
            spread=0.5 * spread_ticks * inst.tick_value * qty,
            slippage=slip_ticks * inst.tick_value * qty,
        )

    def round_trip_cost(self, instrument: str, qty: int,
                        entry_type: str = "market", exit_type: str = "market",
                        overnight: bool = False) -> float:
        """Total dollar friction for entry + exit of qty contracts."""
        return (self.side_cost(instrument, qty, entry_type, overnight).total
                + self.side_cost(instrument, qty, exit_type, overnight).total)
