"""Minimal continuous-contract construction for research bars.

Volume-based roll with a calendar guard, producing a back-adjusted continuous
series (additive). Raw per-contract bars are untouched — the continuous series is
written under a synthetic symbol ``<root>.c`` (e.g. ``NQ.c``).

The roll selection (:func:`select_active_contracts`) is a pure function so it can
be unit-tested with synthetic volumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from nqlab.config.models import AppConfig
from nqlab.data.bars_build import _write_partitioned, bars_root, read_bars
from nqlab.domain.instruments.symbology import ContractError, Outright, parse_symbol


@dataclass(frozen=True)
class RollEvent:
    roll_date: date
    from_symbol: str
    to_symbol: str
    price_offset: float = 0.0


@dataclass(frozen=True)
class ContinuousResult:
    symbol: str
    timeframe: str
    rolls: list[RollEvent]
    trading_days: int
    rows: int


def select_active_contracts(
    contracts: list[tuple[str, date]],
    daily_volume: dict[tuple[str, date], float],
    days: list[date],
    *,
    guard_days: int,
    ratio: float,
) -> tuple[dict[date, str], list[tuple[date, str, str]]]:
    """Pick the active (front) contract per day; monotonic forward roll.

    ``contracts`` must be sorted by expiry. Roll to the next contract once its
    volume exceeds ``ratio`` * front volume, or within ``guard_days`` of expiry.
    """
    if not contracts or not days:
        return {}, []
    active: dict[date, str] = {}
    rolls: list[tuple[date, str, str]] = []
    idx = 0
    for day in sorted(days):
        while idx < len(contracts) - 1:
            cur_sym, cur_exp = contracts[idx]
            nxt_sym, _ = contracts[idx + 1]
            cur_vol = daily_volume.get((cur_sym, day), 0.0)
            nxt_vol = daily_volume.get((nxt_sym, day), 0.0)
            near_expiry = day >= cur_exp - timedelta(days=guard_days)
            if nxt_vol > ratio * cur_vol or near_expiry:
                rolls.append((day, cur_sym, nxt_sym))
                idx += 1
            else:
                break
        active[day] = contracts[idx][0]
    return active, rolls


def _quarterly_symbols_with_bars(
    config: AppConfig, root: str, timeframe: str, year_base: int
) -> list[str]:
    base = bars_root(config)
    out: list[str] = []
    if not base.is_dir():
        return out
    for sym_dir in sorted(base.glob("symbol=*")):
        symbol = sym_dir.name.split("=", 1)[1]
        if "." in symbol or "-" in symbol:
            continue
        if not (sym_dir / f"timeframe={timeframe}").is_dir():
            continue
        try:
            contract = parse_symbol(symbol, year_base)
        except ContractError:
            continue
        if isinstance(contract, Outright) and contract.root == root and contract.is_quarterly:
            out.append(symbol)
    return out


def _daily_close(frame: pl.DataFrame) -> dict[date, float]:
    rows = (
        frame.sort("ts")
        .group_by("trading_day", maintain_order=True)
        .agg(pl.col("close").last())
        .rows()
    )
    return {d: float(c) for d, c in rows}


def _adjustment_by_day(
    rolls: list[tuple[date, str, str]],
    daily_close: dict[str, dict[date, float]],
) -> list[RollEvent]:
    """Compute the per-roll additive gap (back-adjustment) at each seam."""
    events: list[RollEvent] = []
    for roll_day, from_sym, to_sym in rolls:
        a_close = daily_close.get(from_sym, {}).get(roll_day)
        b_close = daily_close.get(to_sym, {}).get(roll_day)
        gap = 0.0 if a_close is None or b_close is None else b_close - a_close
        events.append(RollEvent(roll_day, from_sym, to_sym, gap))
    return events


def compute_active_map(
    config: AppConfig, *, root: str, timeframe: str, start: date, end: date
) -> dict[date, str]:
    """Return the active (front) outright per trading day, from per-contract volume.

    Reuses the same volume roll + calendar guard as the continuous bars so MBP1
    features align with NQ.c.
    """
    year_base = config.instruments.year_base
    contracts: list[tuple[str, date]] = []
    daily_volume: dict[tuple[str, date], float] = {}
    for symbol in _quarterly_symbols_with_bars(config, root, timeframe, year_base):
        frame = read_bars(config, symbol=symbol, timeframe=timeframe, start=start, end=end)
        if frame.height == 0:
            continue
        contract = parse_symbol(symbol, year_base)
        assert isinstance(contract, Outright)
        contracts.append((symbol, contract.expiry))
        for d, vol in frame.group_by("trading_day").agg(pl.col("volume").sum()).rows():
            daily_volume[(symbol, d)] = float(vol)
    if not contracts:
        return {}
    contracts.sort(key=lambda item: item[1])
    days = sorted({d for (_s, d) in daily_volume})
    active, _rolls = select_active_contracts(
        contracts, daily_volume, days,
        guard_days=config.roll.calendar_guard_days,
        ratio=config.roll.min_front_volume_ratio,
    )
    return active


def build_continuous(
    config: AppConfig,
    *,
    root: str,
    timeframe: str,
    start: date,
    end: date,
) -> ContinuousResult:
    """Build and materialize the back-adjusted continuous series for ``root``."""
    year_base = config.instruments.year_base
    symbols = _quarterly_symbols_with_bars(config, root, timeframe, year_base)

    frames: dict[str, pl.DataFrame] = {}
    contracts: list[tuple[str, date]] = []
    daily_volume: dict[tuple[str, date], float] = {}
    daily_close: dict[str, dict[date, float]] = {}
    for symbol in symbols:
        frame = read_bars(config, symbol=symbol, timeframe=timeframe, start=start, end=end)
        if frame.height == 0:
            continue
        frames[symbol] = frame
        contract = parse_symbol(symbol, year_base)
        assert isinstance(contract, Outright)
        contracts.append((symbol, contract.expiry))
        for d, vol in frame.group_by("trading_day").agg(pl.col("volume").sum()).rows():
            daily_volume[(symbol, d)] = float(vol)
        daily_close[symbol] = _daily_close(frame)

    cont_symbol = f"{root}.c"
    if not frames:
        return ContinuousResult(cont_symbol, timeframe, [], 0, 0)

    contracts.sort(key=lambda item: item[1])
    days = sorted({d for (_s, d) in daily_volume})
    active, raw_rolls = select_active_contracts(
        contracts,
        daily_volume,
        days,
        guard_days=config.roll.calendar_guard_days,
        ratio=config.roll.min_front_volume_ratio,
    )
    rolls = _adjustment_by_day(raw_rolls, daily_close)

    # adj(day) = sum of gaps for rolls strictly after that day (newest unadjusted).
    def adj_for(day: date) -> float:
        return sum(ev.price_offset for ev in rolls if ev.roll_date > day)

    active_df = pl.DataFrame(
        {"trading_day": list(active.keys()), "active_symbol": list(active.values())}
    )
    adj_df = pl.DataFrame(
        {"trading_day": days, "adj": [adj_for(d) for d in days]},
        schema_overrides={"adj": pl.Float64},
    )

    combined = pl.concat(list(frames.values()))
    selected = (
        combined.join(active_df, on="trading_day", how="inner")
        .filter(pl.col("symbol") == pl.col("active_symbol"))
        .join(adj_df, on="trading_day", how="left")
        .sort("ts")
    )
    adjusted = selected.with_columns(
        (pl.col("open") + pl.col("adj")).alias("open"),
        (pl.col("high") + pl.col("adj")).alias("high"),
        (pl.col("low") + pl.col("adj")).alias("low"),
        (pl.col("close") + pl.col("adj")).alias("close"),
        (pl.col("vwap") + pl.col("adj")).alias("vwap"),
    ).with_columns(pl.lit(cont_symbol).alias("symbol"))

    out = adjusted.drop(["active_symbol", "adj"])
    _write_partitioned(out, bars_root(config), timeframe)
    return ContinuousResult(cont_symbol, timeframe, rolls, len(days), out.height)
