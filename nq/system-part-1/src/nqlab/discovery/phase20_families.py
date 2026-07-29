"""Phase 20 — classic indicator strategy families (last non-MBP1, bar-based phase).

Simple, explainable indicator rules across the major systematic classes (MA trend, Bollinger, RSI,
MACD, Stochastic, ADX, ATR-trail trend, Keltner/Donchian, VWAP, and small combos). NY-AM only,
ATR-scaled stops/targets, deliberately small parameter grids (no overfitting). Indicators are
computed PIT-safe in ``_prepare`` via :mod:`nqlab.features.indicators`.

None are assumed to work — they are judged against the payout-first bar and rejected honestly.
"""

from __future__ import annotations

import polars as pl

from nqlab.discovery.candidates import _Family
from nqlab.features.indicators import (
    add_adx,
    add_bollinger,
    add_donchian,
    add_keltner,
    add_macd,
    add_stoch,
    ema,
)

_RTH_OPEN_DEFAULT = 570
P20_EXIT_BY_TF: dict[str, dict[str, object]] = {
    "1m": {"min_hold_bars": 1, "time_stop_bars": 60},
    "5m": {"min_hold_bars": 1, "time_stop_bars": 18},
    "15m": {"min_hold_bars": 1, "time_stop_bars": 8},
}
P20_SIZING_MODES: tuple[str, ...] = ("mnq_min1",)
P20_RISK_GRID: tuple[float, ...] = (250.0,)
# Two stop/target regimes: ~1R and ~1.6R (ATR-scaled). Kept tiny to avoid overfitting.
P20_STOPTGT: tuple[tuple[float, float], ...] = ((1.0, 1.5), (1.5, 2.5))


def _grid(*param_sets: dict[str, object]) -> list[dict[str, object]]:
    """Cross indicator-param sets with the shared stop/target regimes."""
    out: list[dict[str, object]] = []
    for ps in param_sets:
        for stop, tgt in P20_STOPTGT:
            out.append({**ps, "stop_atr": stop, "target_atr": tgt})
    return out


class _P20(_Family):
    version = "20"
    required_columns: tuple[str, ...] = ("atr",)


# --- 1) Moving-average trend systems --------------------------------------------------------
class EmaCross(_P20):
    family = "p20_ema_cross"

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            ema(int(self.p["fast"])).alias("_ef"), ema(int(self.p["slow"])).alias("_es")  # type: ignore[call-overload]
        )

    def _long(self) -> pl.Expr:
        return (pl.col("_ef") > pl.col("_es")) & (pl.col("close") > pl.col("_ef"))

    def _short(self) -> pl.Expr:
        return (pl.col("_ef") < pl.col("_es")) & (pl.col("close") < pl.col("_ef"))

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({"fast": 9, "slow": 21}, {"fast": 21, "slow": 50})


class EmaPullback(_P20):
    family = "p20_ema_pullback"

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(ema(21).alias("_e21"), ema(50).alias("_e50"))

    def _long(self) -> pl.Expr:  # uptrend, pull back to 21-EMA and reclaim
        return (
            (pl.col("_e21") > pl.col("_e50"))
            & (pl.col("low") <= pl.col("_e21"))
            & (pl.col("close") > pl.col("_e21"))
        )

    def _short(self) -> pl.Expr:
        return (
            (pl.col("_e21") < pl.col("_e50"))
            & (pl.col("high") >= pl.col("_e21"))
            & (pl.col("close") < pl.col("_e21"))
        )

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({})


# --- 2) Bollinger systems --------------------------------------------------------------------
class BollingerMeanRev(_P20):
    family = "p20_bb_meanrev"

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_bollinger(frame, n=int(self.p["n"]), k=float(self.p["k"]))  # type: ignore[arg-type,call-overload]

    def _long(self) -> pl.Expr:
        return (pl.col("low") <= pl.col("_bb_dn")) & (pl.col("close") > pl.col("_bb_dn"))

    def _short(self) -> pl.Expr:
        return (pl.col("high") >= pl.col("_bb_up")) & (pl.col("close") < pl.col("_bb_up"))

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({"n": 20, "k": 2.0})


class BollingerBreakout(_P20):
    family = "p20_bb_breakout"

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_bollinger(frame, n=int(self.p["n"]), k=float(self.p["k"]))  # type: ignore[arg-type,call-overload]

    def _long(self) -> pl.Expr:
        return pl.col("close") > pl.col("_bb_up")

    def _short(self) -> pl.Expr:
        return pl.col("close") < pl.col("_bb_dn")

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({"n": 20, "k": 2.0})


# --- 3) RSI systems --------------------------------------------------------------------------
class RsiReversal(_P20):
    family = "p20_rsi_reversal"
    required_columns = ("atr", "rsi")

    def _long(self) -> pl.Expr:
        return pl.col("rsi") < float(self.p["low"])  # type: ignore[arg-type]

    def _short(self) -> pl.Expr:
        return pl.col("rsi") > float(self.p["high"])  # type: ignore[arg-type]

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({"low": 30.0, "high": 70.0}, {"low": 25.0, "high": 75.0})


class RsiTrend(_P20):
    family = "p20_rsi_trend"
    required_columns = ("atr", "rsi")

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(ema(50).alias("_e50"))

    def _long(self) -> pl.Expr:
        return (pl.col("close") > pl.col("_e50")) & (pl.col("rsi") > float(self.p["thr"]))  # type: ignore[arg-type]

    def _short(self) -> pl.Expr:
        return (pl.col("close") < pl.col("_e50")) & (pl.col("rsi") < (100.0 - float(self.p["thr"])))  # type: ignore[arg-type]

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({"thr": 55.0})


# --- 4) MACD systems -------------------------------------------------------------------------
class MacdCross(_P20):
    family = "p20_macd_cross"

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_macd(frame)

    def _long(self) -> pl.Expr:
        return (pl.col("_macd") > pl.col("_macd_sig")) & (pl.col("_macd_hist") > 0)

    def _short(self) -> pl.Expr:
        return (pl.col("_macd") < pl.col("_macd_sig")) & (pl.col("_macd_hist") < 0)

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({})


# --- 5) Stochastic systems -------------------------------------------------------------------
class StochReversal(_P20):
    family = "p20_stoch_reversal"

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_stoch(frame)

    def _long(self) -> pl.Expr:
        lo = float(self.p["low"])  # type: ignore[arg-type]
        return (pl.col("_stoch_k") < lo) & (pl.col("_stoch_k") > pl.col("_stoch_d"))

    def _short(self) -> pl.Expr:
        hi = float(self.p["high"])  # type: ignore[arg-type]
        return (pl.col("_stoch_k") > hi) & (pl.col("_stoch_k") < pl.col("_stoch_d"))

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({"low": 20.0, "high": 80.0})


# --- 6) ADX trend-strength -------------------------------------------------------------------
class AdxTrend(_P20):
    family = "p20_adx_trend"

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_adx(frame, n=14)

    def _long(self) -> pl.Expr:
        return (pl.col("adx") > float(self.p["thr"])) & (pl.col("plus_di") > pl.col("minus_di"))  # type: ignore[arg-type]

    def _short(self) -> pl.Expr:
        return (pl.col("adx") > float(self.p["thr"])) & (pl.col("minus_di") > pl.col("plus_di"))  # type: ignore[arg-type]

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({"thr": 25.0})


# --- 7) ATR-trail trend (Supertrend-style: EMA dir + ATR trailing stop exit) -----------------
class AtrTrend(_P20):
    family = "p20_atr_trend"

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(ema(21).alias("_e21"), ema(50).alias("_e50"))

    def _long(self) -> pl.Expr:
        return (pl.col("_e21") > pl.col("_e50")) & (pl.col("close") > pl.col("_e21"))

    def _short(self) -> pl.Expr:
        return (pl.col("_e21") < pl.col("_e50")) & (pl.col("close") < pl.col("_e21"))

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        # ATR trailing-stop exit via TrailingConfig (trail='atr'), wide target to let it trail.
        return [{"trail": "atr", "trail_atr": 2.0, "stop_atr": 1.5, "target_atr": 4.0}]


# --- 8) Keltner / Donchian -------------------------------------------------------------------
class KeltnerMeanRev(_P20):
    family = "p20_keltner_meanrev"

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_keltner(frame, n=20, k=float(self.p["k"]))  # type: ignore[arg-type]

    def _long(self) -> pl.Expr:
        return (pl.col("low") <= pl.col("_kc_dn")) & (pl.col("close") > pl.col("_kc_dn"))

    def _short(self) -> pl.Expr:
        return (pl.col("high") >= pl.col("_kc_up")) & (pl.col("close") < pl.col("_kc_up"))

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({"k": 1.5})


class DonchianBreakout(_P20):
    family = "p20_donchian_brk"

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_donchian(frame, n=int(self.p["n"]))  # type: ignore[call-overload]

    def _long(self) -> pl.Expr:
        return pl.col("_dc_up").is_not_null() & (pl.col("close") > pl.col("_dc_up"))

    def _short(self) -> pl.Expr:
        return pl.col("_dc_dn").is_not_null() & (pl.col("close") < pl.col("_dc_dn"))

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({"n": 20}, {"n": 40})


# --- 9) VWAP systems -------------------------------------------------------------------------
class VwapTrend(_P20):
    family = "p20_vwap_trend"
    required_columns = ("atr", "vwap_session")

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(ema(9).alias("_e9"), ema(21).alias("_e21"))

    def _long(self) -> pl.Expr:
        return (pl.col("close") > pl.col("vwap_session")) & (pl.col("_e9") > pl.col("_e21"))

    def _short(self) -> pl.Expr:
        return (pl.col("close") < pl.col("vwap_session")) & (pl.col("_e9") < pl.col("_e21"))

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({})


# --- 10) Combos ------------------------------------------------------------------------------
class ComboEmaRsiPullback(_P20):
    family = "p20_combo_ema_rsi"
    required_columns = ("atr", "rsi")

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(ema(50).alias("_e50"))

    def _long(self) -> pl.Expr:  # uptrend + RSI pullback
        return (pl.col("close") > pl.col("_e50")) & (pl.col("rsi") < float(self.p["lo"]))  # type: ignore[arg-type]

    def _short(self) -> pl.Expr:
        return (pl.col("close") < pl.col("_e50")) & (pl.col("rsi") > float(self.p["hi"]))  # type: ignore[arg-type]

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({"lo": 40.0, "hi": 60.0})


class ComboDonchianAdx(_P20):
    family = "p20_combo_donch_adx"

    def _prepare(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_adx(add_donchian(frame, n=20), n=14)

    def _long(self) -> pl.Expr:
        return (pl.col("close") > pl.col("_dc_up")) & (pl.col("adx") > 25.0)

    def _short(self) -> pl.Expr:
        return (pl.col("close") < pl.col("_dc_dn")) & (pl.col("adx") > 25.0)

    @classmethod
    def grid(cls) -> list[dict[str, object]]:
        return _grid({})


PHASE20_FAMILIES: dict[str, type[_P20]] = {
    "p20_ema_cross": EmaCross,
    "p20_ema_pullback": EmaPullback,
    "p20_bb_meanrev": BollingerMeanRev,
    "p20_bb_breakout": BollingerBreakout,
    "p20_rsi_reversal": RsiReversal,
    "p20_rsi_trend": RsiTrend,
    "p20_macd_cross": MacdCross,
    "p20_stoch_reversal": StochReversal,
    "p20_adx_trend": AdxTrend,
    "p20_atr_trend": AtrTrend,
    "p20_keltner_meanrev": KeltnerMeanRev,
    "p20_donchian_brk": DonchianBreakout,
    "p20_vwap_trend": VwapTrend,
    "p20_combo_ema_rsi": ComboEmaRsiPullback,
    "p20_combo_donch_adx": ComboDonchianAdx,
}


def generate_phase20_candidates(
    families: tuple[str, ...] = tuple(PHASE20_FAMILIES),
    *,
    timeframe: str,
    risk_grid: tuple[float, ...] = P20_RISK_GRID,
    sizing_modes: tuple[str, ...] = P20_SIZING_MODES,
    hour_window: tuple[int, int] = (9, 12),
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_P20]:
    """family x grid x risk x sizing, TF-seeded holds, NY-AM session. Deterministic."""
    exit_seed = P20_EXIT_BY_TF.get(timeframe, P20_EXIT_BY_TF["5m"])
    h0, h1 = hour_window
    out: list[_P20] = []
    for fam in families:
        cls = PHASE20_FAMILIES[fam]
        for base in cls.grid():
            for risk in risk_grid:
                for mode in sizing_modes:
                    out.append(
                        cls(
                            {**base, **exit_seed, "risk_per_trade": risk, "sizing_mode": mode,
                             "hour_start": h0, "hour_end": h1},
                            rth_start_min=rth_start_min,
                        )
                    )
    return out
