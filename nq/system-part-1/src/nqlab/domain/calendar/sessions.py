"""Session labelling + trading-day derivation for CME equity-index Globex.

All session logic is driven by :class:`SessionConfig` (config, not hardcoded).
Times are exchange-local (default America/New_York). The Globex session opens at
18:00 ET and is dated to the *next* calendar day, so the trading day is obtained
by adding ``trading_day_offset_hours`` (default 6) to the local time and taking
the date.

Provides both scalar helpers (for tests) and a vectorized Polars helper (for the
bars engine), driven by the same window definitions to avoid divergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import polars as pl

from nqlab.config.models import SessionConfig

OTHER_SESSION = "OTHER"
_MINUTES_PER_DAY = 1440


def parse_hhmm(value: str) -> int:
    """Minutes since local midnight for an ``HH:MM`` string."""
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


@dataclass(frozen=True)
class SessionLabeler:
    """Immutable labeler derived from :class:`SessionConfig`."""

    timezone: str
    trading_day_offset_hours: int
    rth_start_min: int
    rth_end_min: int
    window_names: tuple[str, ...]
    window_starts: tuple[int, ...]
    window_ends: tuple[int, ...]

    @classmethod
    def from_config(cls, cfg: SessionConfig) -> SessionLabeler:
        return cls(
            timezone=cfg.timezone,
            trading_day_offset_hours=cfg.trading_day_offset_hours,
            rth_start_min=parse_hhmm(cfg.rth_start),
            rth_end_min=parse_hhmm(cfg.rth_end),
            window_names=tuple(w.name for w in cfg.windows),
            window_starts=tuple(parse_hhmm(w.start) for w in cfg.windows),
            window_ends=tuple(parse_hhmm(w.end) for w in cfg.windows),
        )

    def minute_label_map(self) -> dict[int, str]:
        """Map every minute-of-day (0..1439) to its session (first window wins)."""
        labels: dict[int, str] = {}
        for name, start, end in zip(
            self.window_names, self.window_starts, self.window_ends, strict=True
        ):
            minutes = (
                range(start, end)
                if start < end
                else [*range(start, _MINUTES_PER_DAY), *range(0, end)]
            )
            for minute in minutes:
                labels.setdefault(minute, name)
        return labels

    def session_of_minute(self, minute_of_day: int) -> str:
        return self.minute_label_map().get(minute_of_day % _MINUTES_PER_DAY, OTHER_SESSION)

    def is_rth_minute(self, minute_of_day: int) -> bool:
        return self.rth_start_min <= minute_of_day < self.rth_end_min

    def trading_day_of(self, local_dt: datetime) -> date:
        return (local_dt + timedelta(hours=self.trading_day_offset_hours)).date()

    def with_calendar_columns(self, frame: pl.DataFrame, ts_col: str = "ts") -> pl.DataFrame:
        """Add ``minute_of_day``, ``dow``, ``trading_day``, ``session``, ``is_rth``.

        ``ts_col`` must be a UTC tz-aware Datetime column.
        """
        local = pl.col(ts_col).dt.convert_time_zone(self.timezone)
        minute_of_day = (
            local.dt.hour().cast(pl.Int32) * 60 + local.dt.minute().cast(pl.Int32)
        )
        frame = frame.with_columns(
            minute_of_day.alias("minute_of_day"),
            local.dt.weekday().cast(pl.Int32).alias("dow"),
            (local + pl.duration(hours=self.trading_day_offset_hours))
            .dt.date()
            .alias("trading_day"),
        )
        return frame.with_columns(
            pl.col("minute_of_day")
            .replace_strict(self.minute_label_map(), default=OTHER_SESSION)
            .alias("session"),
            (
                (pl.col("minute_of_day") >= self.rth_start_min)
                & (pl.col("minute_of_day") < self.rth_end_min)
            ).alias("is_rth"),
        )
