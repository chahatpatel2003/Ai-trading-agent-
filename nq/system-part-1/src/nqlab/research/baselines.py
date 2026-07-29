"""Baseline test-vehicle strategies.

These exist to prove the engine works end-to-end — they are NOT expected to have
an edge and must never be treated as validated strategies. All use ATR-scaled
stops/targets so position sizing and the Topstep simulator are exercised.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from nqlab.domain.strategy.base import BaseStrategy, StrategyParams

_atr = pl.col("atr")


class OpeningRangeBreakout(BaseStrategy):
    def __init__(self, *, stop_atr: float = 1.0, target_atr: float = 2.0) -> None:
        super().__init__(
            StrategyParams(
                "opening_range_breakout",
                "1",
                (("stop_atr", stop_atr), ("target_atr", target_atr)),
            )
        )
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def _long(self) -> pl.Expr:
        return pl.col("or_high").is_not_null() & (pl.col("close") > pl.col("or_high"))

    def _short(self) -> pl.Expr:
        return pl.col("or_low").is_not_null() & (pl.col("close") < pl.col("or_low"))

    def _stop_pts(self) -> pl.Expr:
        return self.stop_atr * _atr

    def _target_pts(self) -> pl.Expr:
        return self.target_atr * _atr


class VwapMeanReversion(BaseStrategy):
    def __init__(
        self, *, entry_atr: float = 1.5, stop_atr: float = 1.0, target_atr: float = 1.5
    ) -> None:
        super().__init__(
            StrategyParams(
                "vwap_mean_reversion",
                "1",
                (("entry_atr", entry_atr), ("stop_atr", stop_atr), ("target_atr", target_atr)),
            )
        )
        self.entry_atr = entry_atr
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def _long(self) -> pl.Expr:
        return pl.col("dist_vwap") < -(self.entry_atr * _atr)

    def _short(self) -> pl.Expr:
        return pl.col("dist_vwap") > (self.entry_atr * _atr)

    def _stop_pts(self) -> pl.Expr:
        return self.stop_atr * _atr

    def _target_pts(self) -> pl.Expr:
        return self.target_atr * _atr


class TrendPullback(BaseStrategy):
    def __init__(
        self,
        *,
        rsi_low: float = 40.0,
        rsi_high: float = 60.0,
        stop_atr: float = 1.5,
        target_atr: float = 3.0,
    ) -> None:
        super().__init__(
            StrategyParams(
                "trend_pullback",
                "1",
                (
                    ("rsi_low", rsi_low),
                    ("rsi_high", rsi_high),
                    ("stop_atr", stop_atr),
                    ("target_atr", target_atr),
                ),
            )
        )
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def _long(self) -> pl.Expr:
        return (pl.col("close") > pl.col("sma")) & (pl.col("rsi") < self.rsi_low)

    def _short(self) -> pl.Expr:
        return (pl.col("close") < pl.col("sma")) & (pl.col("rsi") > self.rsi_high)

    def _stop_pts(self) -> pl.Expr:
        return self.stop_atr * _atr

    def _target_pts(self) -> pl.Expr:
        return self.target_atr * _atr


class PrevDayBreakout(BaseStrategy):
    def __init__(self, *, stop_atr: float = 1.5, target_atr: float = 3.0) -> None:
        super().__init__(
            StrategyParams(
                "prev_day_breakout",
                "1",
                (("stop_atr", stop_atr), ("target_atr", target_atr)),
            )
        )
        self.stop_atr = stop_atr
        self.target_atr = target_atr

    def _long(self) -> pl.Expr:
        return pl.col("prev_day_high").is_not_null() & (
            pl.col("close") > pl.col("prev_day_high")
        )

    def _short(self) -> pl.Expr:
        return pl.col("prev_day_low").is_not_null() & (
            pl.col("close") < pl.col("prev_day_low")
        )

    def _stop_pts(self) -> pl.Expr:
        return self.stop_atr * _atr

    def _target_pts(self) -> pl.Expr:
        return self.target_atr * _atr


BASELINES: dict[str, Callable[[], BaseStrategy]] = {
    "orb": OpeningRangeBreakout,
    "vwap_rev": VwapMeanReversion,
    "trend_pullback": TrendPullback,
    "pdhl": PrevDayBreakout,
}


def build_baseline(name: str) -> BaseStrategy:
    """Instantiate a baseline strategy by short name."""
    try:
        return BASELINES[name]()
    except KeyError:
        raise KeyError(f"Unknown baseline {name!r}; options: {sorted(BASELINES)}") from None
