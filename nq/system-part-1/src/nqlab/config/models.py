"""Pydantic v2 configuration models.

All models are ``frozen`` (immutable, hashable) and forbid unknown keys so that
typos fail fast and a fully-resolved config is a stable, content-hashable object
suitable for reproducibility stamping.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from nqlab.domain.instruments.roll import Adjustment, RollMethod


class StrictModel(BaseModel):
    """Base: immutable + reject unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _to_path(value: Any) -> Path:
    return Path(str(value)).expanduser()


PathField = Annotated[Path, BeforeValidator(_to_path)]


class Clock(StrEnum):
    TS_RECV = "ts_recv"
    TS_EVENT = "ts_event"


class ClockConfig(StrictModel):
    """Canonical information-availability clock + event-ordering keys."""

    canonical: Clock = Clock.TS_RECV
    ordering: tuple[str, ...] = ("ts_event", "sequence")


class DataPathsConfig(StrictModel):
    """Filesystem layout. ``raw`` is immutable; derived layers live elsewhere."""

    nq_data_root: PathField
    raw_subdir: str = "raw"
    trades_subdir: str = "trades"
    mbp1_subdir: str = "mbp1"
    curated_root: PathField
    catalog_path: PathField
    artifacts_root: PathField

    @property
    def raw_root(self) -> Path:
        return self.nq_data_root / self.raw_subdir

    @property
    def raw_trades_dir(self) -> Path:
        return self.raw_root / self.trades_subdir

    @property
    def raw_mbp1_dir(self) -> Path:
        return self.raw_root / self.mbp1_subdir


class InstrumentSpec(StrictModel):
    """Contract economics for one product root. Lives in config, never hardcoded."""

    root: str
    description: str = ""
    exchange: str = "GLBX"
    currency: str = "USD"
    point_value: float = Field(gt=0)
    tick_size: float = Field(gt=0)
    tick_value: float = Field(gt=0)
    commission_per_side: float = Field(default=0.0, ge=0)
    cycle_months: tuple[str, ...] = ("H", "M", "U", "Z")
    expiry_rule: Literal["third_friday"] = "third_friday"


class InstrumentsConfig(StrictModel):
    default_root: str = "NQ"
    year_base: int = 2020
    specs: dict[str, InstrumentSpec]

    @property
    def default_spec(self) -> InstrumentSpec:
        return self.specs[self.default_root]


class RollConfig(StrictModel):
    """Continuous-contract roll policy (approved: volume + calendar guard)."""

    method: RollMethod = RollMethod.VOLUME
    adjustment: Adjustment = Adjustment.BACK_ADJUSTED
    calendar_guard_days: int = Field(default=8, ge=0)
    roll_offset_days: int = 0
    min_front_volume_ratio: float = Field(default=1.0, gt=0)


class SessionWindow(StrictModel):
    """A named intraday session window in exchange-local time (HH:MM, end exclusive)."""

    name: str
    start: str
    end: str  # may wrap past midnight (e.g. ASIA 18:00 -> 03:00)


def _default_session_windows() -> tuple[SessionWindow, ...]:
    return (
        SessionWindow(name="ASIA", start="18:00", end="03:00"),
        SessionWindow(name="LONDON", start="03:00", end="09:30"),
        SessionWindow(name="NY_AM", start="09:30", end="12:00"),
        SessionWindow(name="NY_PM", start="12:00", end="16:00"),
    )


class SessionConfig(StrictModel):
    """Session labelling + trading-day convention (CME equity-index Globex)."""

    timezone: str = "America/New_York"
    # A Globex session opens 18:00 ET and is dated to the next calendar day;
    # adding this many hours then taking the date yields the trading day.
    trading_day_offset_hours: int = 6
    rth_start: str = "09:30"
    rth_end: str = "16:00"
    windows: tuple[SessionWindow, ...] = Field(default_factory=_default_session_windows)


class BacktestConfig(StrictModel):
    """Execution + risk defaults for the minimal backtester."""

    timeframe: str = "5m"
    # next_open = enter at the NEXT bar's open (realistic). current_close enters at
    # the signal bar's close (debug only — optimistic, used for the fills check).
    entry_mode: Literal["next_open", "current_close"] = "next_open"
    intrabar_dd: bool = True  # mark the bar's adverse extreme for trailing-DD checks
    risk_per_trade_usd: float = Field(default=250.0, gt=0)
    slippage_ticks: float = Field(default=1.0, ge=0)
    default_stop_atr: float = Field(default=1.5, gt=0)
    default_target_atr: float = Field(default=2.0, gt=0)
    time_stop_bars: int = Field(default=90, ge=1)
    rth_only: bool = True
    allow_overnight: bool = False
    # All ET. Primary tests stop new entries early; hard cutoff + force-flat protect
    # against the Topstep close.
    no_new_trade_primary_et: str = "12:30"
    no_new_trade_hard_et: str = "15:50"
    force_flat_et: str = "15:55"


class ValidationConfig(StrictModel):
    """Out-of-sample split, stress, Monte-Carlo, and acceptance thresholds."""

    train_start: date = date(2021, 1, 1)
    train_end: date = date(2024, 12, 31)
    test_start: date = date(2025, 1, 1)
    test_end: date = date(2026, 12, 31)
    embargo_days: int = Field(default=3, ge=0)
    walk_train_months: int = Field(default=12, ge=1)
    walk_test_months: int = Field(default=3, ge=1)
    walk_step_months: int = Field(default=3, ge=1)
    monte_carlo_runs: int = Field(default=2000, ge=1)
    monte_carlo_seed: int = 12345
    slippage_stress: tuple[float, ...] = (1.0, 2.0, 3.0)
    commission_stress: tuple[float, ...] = (1.0, 1.5)
    # Acceptance thresholds.
    min_trades: int = 30
    min_oos_trades: int = 10
    max_blowup_prob: float = 0.10
    min_walk_forward_positive_frac: float = 0.50
    min_positive_years: int = 2
    # Multiple-testing deflation: a candidate that passes the gauntlet must also
    # clear this Deflated-Sharpe probability given the number of trials searched.
    dsr_threshold: float = 0.90
    # Optional touch-once lockbox window (reserved from selection).
    lockbox_start: date | None = None
    lockbox_end: date | None = None


class TopstepRules(StrictModel):
    """A funded-account rule set. Loaded by name from config/risk/*.yaml."""

    name: str
    account_size: float = Field(gt=0)
    profit_target: float = Field(gt=0)
    trailing_max_loss: float = Field(gt=0)
    trailing_basis: Literal["intraday", "eod"] = "intraday"
    lock_trailing_at_start: bool = True
    daily_loss_limit: float | None = None
    max_contracts: int = Field(gt=0)
    min_trading_days: int = Field(default=0, ge=0)
    consistency_pct: float | None = 0.5


class LoggingConfig(StrictModel):
    level: str = "INFO"
    format: Literal["console", "json"] = "console"
    include_timestamp: bool = True


class ComputeConfig(StrictModel):
    max_workers: int | None = None
    manifest_sample_row_groups: int = Field(default=4, ge=1)


class AppConfig(StrictModel):
    """Root configuration object — the single source of truth for a run."""

    profile: str = "base"
    data: DataPathsConfig
    instruments: InstrumentsConfig
    roll: RollConfig = Field(default_factory=RollConfig)
    clock: ClockConfig = Field(default_factory=ClockConfig)
    sessions: SessionConfig = Field(default_factory=SessionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)

    def config_hash(self) -> str:
        """Deterministic SHA-256 of the fully-resolved config (reproducibility key)."""
        payload = self.model_dump(mode="json")
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
