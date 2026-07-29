"""Continuous-contract planning: roll method enums, events, and interfaces.

Phase 0 deliberately ships **interfaces and configuration types only**. The
approved methodology is a *volume-based roll with a calendar guard*, producing a
*back-adjusted* continuous series for research while raw per-contract data is
preserved for execution realism. The concrete scheduler/builder are implemented
in Phase 2; their stubs raise :class:`NotImplementedError` with a clear pointer.

This module imports nothing from the config layer at runtime (only under
``TYPE_CHECKING``) to keep the dependency graph acyclic: ``config`` depends on
``domain``, never the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from nqlab.config.models import RollConfig
    from nqlab.domain.instruments.symbology import Outright


class RollMethod(StrEnum):
    """How the roll date from one contract to the next is determined."""

    VOLUME = "volume"
    CALENDAR = "calendar"
    OPEN_INTEREST = "open_interest"


class Adjustment(StrEnum):
    """How price discontinuities at a roll are reconciled in the continuous series."""

    BACK_ADJUSTED = "back_adjusted"
    RATIO = "ratio"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class RollEvent:
    """A single roll from ``from_symbol`` to ``to_symbol`` on ``roll_date``.

    ``price_offset`` is the cumulative additive adjustment (back-adjustment) or
    multiplicative factor (ratio) applied to history at/before the roll so the
    stitched series is continuous in returns. Sign/semantics depend on the
    :class:`Adjustment` in effect.
    """

    roll_date: date
    from_symbol: str
    to_symbol: str
    price_offset: float = 0.0


@runtime_checkable
class RollScheduler(Protocol):
    """Computes the ordered sequence of roll events for a continuous contract."""

    def compute_schedule(
        self, root: str, contracts: Sequence[Outright]
    ) -> list[RollEvent]:
        """Return roll events (chronological) stitching ``contracts`` into one series."""
        ...


@runtime_checkable
class ContinuousContractBuilder(Protocol):
    """Builds a back-adjusted/ratio-adjusted continuous series from roll events."""

    def adjustment(self) -> Adjustment: ...


class VolumeRollScheduler:
    """Volume-based roll with a calendar guard (approved method).

    **Not implemented in Phase 0** — interface and config wiring only. The Phase 2
    implementation will roll to the next contract once next-month traded volume
    exceeds front-month volume by ``config.min_front_volume_ratio``, never holding
    a contract within ``config.calendar_guard_days`` of expiry.
    """

    def __init__(self, config: RollConfig) -> None:
        self.config = config

    def compute_schedule(
        self, root: str, contracts: Sequence[Outright]
    ) -> list[RollEvent]:
        raise NotImplementedError(
            "VolumeRollScheduler is planned for Phase 2 (curated/continuous layer). "
            "Phase 0 ships interfaces and roll configuration only."
        )
