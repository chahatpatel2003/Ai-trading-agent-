"""Market-data adapter interface + historical replay (Phase 15).

This module is DATA-IN ONLY. There is deliberately no order/execution path here — no
broker client, no order placement, no fills sent anywhere. A live feed would implement
:class:`MarketDataFeed`; until one is connected, :class:`ReplayFeed` replays stored 5m
bars so the paper engine can be exercised. A real 1m->5m aggregator (:func:`build_5m_from_1m`)
is provided for when a sub-5m live feed exists; it uses the ``ts_recv`` availability clock.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from typing import Protocol, runtime_checkable

import polars as pl

from nqlab.config.models import AppConfig

# Columns a 5m bar must carry to feed the (calendar-aware) feature pipeline.
_BAR_COLUMNS = ("ts", "symbol", "open", "high", "low", "close", "volume", "vwap", "trades")


@runtime_checkable
class MarketDataFeed(Protocol):
    """A source of completed 5m bars. Implementations MUST NOT place orders."""

    def bars(self) -> Iterator[pl.DataFrame]:  # yields completed 5m bars (1+ rows)
        ...


def build_5m_from_1m(df_1m: pl.DataFrame) -> pl.DataFrame:
    """Aggregate 1-minute bars into completed 5-minute bars (OHLCV + vwap).

    Buckets by the floor of ``ts`` to 5 minutes (``ts_recv`` availability clock); a bar
    is emitted only for a fully-populated bucket. open=first, high=max, low=min,
    close=last, volume/trades=sum, vwap=volume-weighted.
    """
    df = df_1m.sort("ts").with_columns(pl.col("ts").dt.truncate("5m").alias("_b"))
    agg = (
        df.group_by("_b")
        .agg(
            pl.col("symbol").first().alias("symbol"),
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("trades").sum().alias("trades"),
            (pl.col("vwap") * pl.col("volume")).sum().alias("_pv"),
        )
        .sort("_b")
    )
    return agg.with_columns(
        pl.col("_b").alias("ts"),
        (pl.col("_pv") / pl.col("volume")).alias("vwap"),
    ).select(_BAR_COLUMNS)


class ReplayFeed:
    """Replays stored 5m bars for ``[start, end]`` as a :class:`MarketDataFeed`.

    Used to exercise the paper engine without a live connection. Bars are *historical*,
    so trades produced from a replay are flagged ``mode="replay"`` and never count as
    genuine forward confirmation.
    """

    mode = "replay"

    def __init__(self, config: AppConfig, *, symbol: str, timeframe: str, start: date, end: date):
        self._config = config
        self._symbol = symbol
        self._timeframe = timeframe
        self._start = start
        self._end = end

    def bars(self) -> Iterator[pl.DataFrame]:
        from nqlab.data.bars_build import read_bars

        frame = read_bars(
            self._config,
            symbol=self._symbol,
            timeframe=self._timeframe,
            start=self._start,
            end=self._end,
        )
        # One chunk (the engine builds features over the whole replay window).
        yield frame.sort("ts")


class LiveFeed:
    """Placeholder for a real-time feed. NOT connected — refuses by design (no orders,
    no execution API in this phase)."""

    mode = "forward"

    def bars(self) -> Iterator[pl.DataFrame]:
        msg = "live market-data feed is not connected (Phase 15 is paper/shadow only)"
        raise NotImplementedError(msg)
