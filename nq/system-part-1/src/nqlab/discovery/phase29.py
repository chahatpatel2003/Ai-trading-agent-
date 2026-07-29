"""Phase 29 — Unrestricted full-history NQ intraday strategy factory (EOD-flat only).

Search space is deliberately broad: classic indicators, price-action, ICT/SMC, volume/volatility,
order-flow (MBP1, <=2025-05), regime filters, across multiple sessions / RR / exit models — the only
hard rule is **flat by end of day**. Every candidate is run ONCE over full history through the
*audited* ``run_backtest`` (Phase-27 integrity: next-open entry, stop-first same-bar ordering,
target exact / forced-exit slipped, exact cost model). To make EOD-flat correct for non-RTH entries
we run on the contiguous midnight->16:00 ET block (``minute_of_day in [0, 960]``) where the 15:55
force-flat fires once per trading day and nothing crosses days. Sessions are gated in each signal.
Sizing is fixed at **1 MNQ** (``mnq_fixed1``) so PF / win-rate / drawdown are comparable.

This module defines the feature layers, the session windows, the family signal builders, and the
candidate grid. Orchestration (run, persist, analyse) lives in ``scripts/phase29_run.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from nqlab.backtest.engine import run_backtest
from nqlab.backtest.result import BacktestResult
from nqlab.config.models import AppConfig, TopstepRules
from nqlab.domain.strategy.base import ExecConfig, StrategyParams, TrailingConfig, attach_signals
from nqlab.features.daily_bias import add_daily_bias
from nqlab.features.htf_poi import add_htf_poi
from nqlab.features.indicators import (
    add_adx,
    add_bollinger,
    add_donchian,
    add_keltner,
    add_macd,
    add_stoch,
)
from nqlab.research.runner import prepare_structure_features

# --- Sessions: minute-of-day [lo, hi) windows, all within the [0, 960] EOD-flat block. -----------
SESSIONS: dict[str, tuple[int, int]] = {
    "NY_AM": (570, 720),  # 09:30-12:00
    "NY_PM": (720, 950),  # 12:00-15:50
    "RTH": (570, 950),  # 09:30-15:50
    "LONDON": (180, 570),  # 03:00-09:30
    "LONDON_NY": (180, 950),  # 03:00-15:50
    "OVN_EARLY": (0, 570),  # 00:00-09:30 (post-midnight overnight + London)
    "ALLDAY": (0, 950),  # 00:00-15:50 (no session restriction, pre-EOD)
}

EOD_BLOCK = (0, 960)  # minute_of_day kept for the backtest (midnight -> 16:00 ET)


# --- Feature layers -------------------------------------------------------------------------------
def _supertrend(frame: pl.DataFrame, *, mult: float = 3.0) -> pl.Series:
    """Causal Supertrend direction (+1 up / -1 down) over the base ATR. Path-dependent: py loop."""
    high = frame["high"].to_list()
    low = frame["low"].to_list()
    close = frame["close"].to_list()
    atr = frame["atr"].to_list()
    n = len(close)
    direction = [1] * n
    fu_prev = fl_prev = None
    for i in range(n):
        a = atr[i]
        if a is None or (isinstance(a, float) and math.isnan(a)):
            fu_prev = fl_prev = None
            direction[i] = 1
            continue
        hl2 = (high[i] + low[i]) / 2.0
        upper = hl2 + mult * a
        lower = hl2 - mult * a
        if fu_prev is None:
            fu = upper
            fl = lower
            direction[i] = 1
        else:
            fu = min(upper, fu_prev) if close[i - 1] <= fu_prev else upper
            fl = max(lower, fl_prev) if close[i - 1] >= fl_prev else lower
            if close[i] > fu_prev:
                direction[i] = 1
            elif close[i] < fl_prev:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]
        fu_prev, fl_prev = fu, fl
    return pl.Series("st_dir", direction, dtype=pl.Int64)


def add_phase29_indicators(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the classic-indicator + helper columns the families consume (PIT-safe / causal)."""
    f = add_macd(frame)
    f = add_stoch(f)
    f = add_bollinger(f, n=20, k=2.0)
    f = add_keltner(f, n=20, k=1.5)
    f = add_donchian(f, n=20)
    f = add_adx(f, n=14)
    f = f.with_columns(
        pl.col("close").ewm_mean(span=9, adjust=False).alias("ema9"),
        pl.col("close").ewm_mean(span=21, adjust=False).alias("ema21"),
        pl.col("close").ewm_mean(span=50, adjust=False).alias("ema50"),
        (pl.col("close") - pl.col("vwap_session")).alias("vwd"),
        pl.col("high").shift(1).alias("ph"),
        pl.col("low").shift(1).alias("pl"),
        (pl.col("_bb_bw").rolling_median(window_size=50)).alias("_bb_bw_med"),
    )
    f = f.with_columns(_supertrend(f))
    return f


def build_phase29_frame(
    config: AppConfig, *, timeframe: str, start: Any, end: Any, with_mbp1: bool = False
) -> pl.DataFrame:
    """Full Phase-29 feature frame: structure + HTF-POI + daily-bias + classic indicators.

    ``with_mbp1`` is informational only — ``prepare_structure_features`` already left-joins MBP1
    where available (null otherwise), so order-flow families just require non-null cvd.
    """
    f = prepare_structure_features(config, symbol="NQ.c", timeframe=timeframe, start=start, end=end)
    f = add_daily_bias(add_htf_poi(f, tol_pts=5.0))
    f = add_phase29_indicators(f)
    return f


# --- Signal helpers -------------------------------------------------------------------------------
def _cu(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cd(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return (a < b) & (a.shift(1) >= b.shift(1))


_C = pl.col
_FALSE = pl.lit(value=False)


@dataclass(frozen=True)
class Family:
    name: str
    kind: str  # trend | mr | breakout | ict | vol | orderflow
    long: pl.Expr
    short: pl.Expr
    requires: tuple[str, ...] = ()  # columns that must be non-null on the signal bar


def _families() -> dict[str, Family]:
    """All Phase-29 family signal definitions. Direction is symmetric (long/short mirror)."""
    fams: list[Family] = [
        # ---- TREND / MOMENTUM ----
        Family(
            "ema_cross_9_21", "trend", _cu(_C("ema9"), _C("ema21")), _cd(_C("ema9"), _C("ema21"))
        ),
        Family(
            "ema_cross_21_50", "trend", _cu(_C("ema21"), _C("ema50")), _cd(_C("ema21"), _C("ema50"))
        ),
        Family(
            "ema_stack",
            "trend",
            (_C("close") > _C("ema9"))
            & (_C("ema9") > _C("ema21"))
            & (_C("ema21") > _C("ema50"))
            & (_C("ema9") > _C("ema9").shift(1)),
            (_C("close") < _C("ema9"))
            & (_C("ema9") < _C("ema21"))
            & (_C("ema21") < _C("ema50"))
            & (_C("ema9") < _C("ema9").shift(1)),
        ),
        Family(
            "macd_cross",
            "trend",
            _cu(_C("_macd"), _C("_macd_sig")),
            _cd(_C("_macd"), _C("_macd_sig")),
        ),
        Family(
            "macd_hist_expand",
            "trend",
            (_C("_macd_hist") > 0)
            & (_C("_macd_hist") > _C("_macd_hist").shift(1))
            & (_C("_macd_hist").shift(1) > _C("_macd_hist").shift(2)),
            (_C("_macd_hist") < 0)
            & (_C("_macd_hist") < _C("_macd_hist").shift(1))
            & (_C("_macd_hist").shift(1) < _C("_macd_hist").shift(2)),
        ),
        Family(
            "adx_trend",
            "trend",
            (_C("adx") > 20) & (_C("plus_di") > _C("minus_di")) & _cu(_C("close"), _C("ema21")),
            (_C("adx") > 20) & (_C("minus_di") > _C("plus_di")) & _cd(_C("close"), _C("ema21")),
        ),
        Family(
            "supertrend",
            "trend",
            (_C("st_dir") == 1) & (_C("st_dir").shift(1) == -1),
            (_C("st_dir") == -1) & (_C("st_dir").shift(1) == 1),
        ),
        Family(
            "donchian_break",
            "trend",
            (_C("close") > _C("_dc_up")) & (_C("close").shift(1) <= _C("_dc_up").shift(1)),
            (_C("close") < _C("_dc_dn")) & (_C("close").shift(1) >= _C("_dc_dn").shift(1)),
        ),
        Family(
            "rsi_momentum",
            "trend",
            _cu(_C("rsi"), pl.lit(55)) & (_C("rsi") < 70),
            _cd(_C("rsi"), pl.lit(45)) & (_C("rsi") > 30),
        ),
        # ---- MEAN REVERSION ----
        Family("rsi_reversal", "mr", _cu(_C("rsi"), pl.lit(30)), _cd(_C("rsi"), pl.lit(70))),
        Family(
            "bollinger_revert",
            "mr",
            (_C("low") < _C("_bb_dn")) & (_C("close") > _C("_bb_dn")),
            (_C("high") > _C("_bb_up")) & (_C("close") < _C("_bb_up")),
        ),
        Family(
            "keltner_revert",
            "mr",
            (_C("low") < _C("_kc_dn")) & (_C("close") > _C("_kc_dn")),
            (_C("high") > _C("_kc_up")) & (_C("close") < _C("_kc_up")),
        ),
        Family(
            "vwap_revert",
            "mr",
            (_C("vwd") < -1.0 * _C("atr")) & (_C("close") > _C("close").shift(1)),
            (_C("vwd") > 1.0 * _C("atr")) & (_C("close") < _C("close").shift(1)),
        ),
        Family(
            "vwap_reclaim",
            "mr",
            (_C("low") <= _C("vwap_session"))
            & (_C("close") > _C("vwap_session"))
            & (_C("open") < _C("vwap_session")),
            (_C("high") >= _C("vwap_session"))
            & (_C("close") < _C("vwap_session"))
            & (_C("open") > _C("vwap_session")),
        ),
        Family(
            "stoch_reversal", "mr", _cu(_C("_stoch_k"), pl.lit(20)), _cd(_C("_stoch_k"), pl.lit(80))
        ),
        # ---- BREAKOUT / PRICE ACTION ----
        Family(
            "orb_break",
            "breakout",
            (_C("minute_of_day") >= 600)
            & (_C("close") > _C("or_high"))
            & (_C("close").shift(1) <= _C("or_high")),
            (_C("minute_of_day") >= 600)
            & (_C("close") < _C("or_low"))
            & (_C("close").shift(1) >= _C("or_low")),
            requires=("or_high", "or_low"),
        ),
        Family(
            "orb_fade",
            "breakout",
            (_C("minute_of_day") >= 600)
            & (_C("low") < _C("or_low"))
            & (_C("close") > _C("or_low")),
            (_C("minute_of_day") >= 600)
            & (_C("high") > _C("or_high"))
            & (_C("close") < _C("or_high")),
            requires=("or_high", "or_low"),
        ),
        Family(
            "pdhl_break",
            "breakout",
            (_C("close") > _C("prev_day_high")) & (_C("close").shift(1) <= _C("prev_day_high")),
            (_C("close") < _C("prev_day_low")) & (_C("close").shift(1) >= _C("prev_day_low")),
            requires=("prev_day_high", "prev_day_low"),
        ),
        Family(
            "pdhl_fade",
            "breakout",
            (_C("low") < _C("prev_day_low")) & (_C("close") > _C("prev_day_low")),
            (_C("high") > _C("prev_day_high")) & (_C("close") < _C("prev_day_high")),
            requires=("prev_day_high", "prev_day_low"),
        ),
        Family(
            "bollinger_break",
            "breakout",
            (_C("close") > _C("_bb_up")) & (_C("close").shift(1) <= _C("_bb_up").shift(1)),
            (_C("close") < _C("_bb_dn")) & (_C("close").shift(1) >= _C("_bb_dn").shift(1)),
        ),
        Family(
            "squeeze_break",
            "breakout",
            (_C("_bb_bw").shift(1) < 0.7 * _C("_bb_bw_med").shift(1))
            & (_C("close") > _C("_bb_up")),
            (_C("_bb_bw").shift(1) < 0.7 * _C("_bb_bw_med").shift(1))
            & (_C("close") < _C("_bb_dn")),
        ),
        Family(
            "inside_day_break",
            "breakout",
            (_C("prev_inside_day") == 1)
            & (_C("minute_of_day") >= 600)
            & (_C("close") > _C("or_high")),
            (_C("prev_inside_day") == 1)
            & (_C("minute_of_day") >= 600)
            & (_C("close") < _C("or_low")),
            requires=("prev_inside_day", "or_high", "or_low"),
        ),
        Family(
            "gap_fill",
            "breakout",
            (_C("gap_dir") == -1) & (_C("gap_filled") == 0) & (_C("minute_of_day") < 660),
            (_C("gap_dir") == 1) & (_C("gap_filled") == 0) & (_C("minute_of_day") < 660),
            requires=("gap_dir", "gap_filled"),
        ),
        Family(
            "gap_continuation",
            "breakout",
            (_C("gap_dir") == 1) & (_C("close") > _C("open")) & (_C("minute_of_day") < 660),
            (_C("gap_dir") == -1) & (_C("close") < _C("open")) & (_C("minute_of_day") < 660),
            requires=("gap_dir",),
        ),
        # ---- ICT / SMC ----
        Family(
            "sweep_reclaim",
            "ict",
            (_C("swept_low") == 1) & (_C("reclaim_up") == 1),
            (_C("swept_high") == 1) & (_C("reclaim_dn") == 1),
        ),
        Family("mss", "ict", _C("mss_up") == 1, _C("mss_dn") == 1),
        Family("choch", "ict", _C("choch_up") == 1, _C("choch_dn") == 1),
        Family(
            "cisd", "ict", _C("cisd_up") == 1, _C("cisd_dn") == 1, requires=("cisd_up", "cisd_dn")
        ),
        Family(
            "bos_cont",
            "ict",
            (_C("bos_up") == 1) & (_C("struct_dir") > 0),
            (_C("bos_dn") == 1) & (_C("struct_dir") < 0),
        ),
        Family(
            "displacement",
            "ict",
            (_C("displacement") == 1) & (_C("close") > _C("open")),
            (_C("displacement") == 1) & (_C("close") < _C("open")),
        ),
        Family(
            "fvg_cont",
            "ict",
            (_C("fvg_bull") == 1) & (_C("struct_dir") > 0),
            (_C("fvg_bear") == 1) & (_C("struct_dir") < 0),
        ),
        Family(
            "ifvg",
            "ict",
            _C("ifvg_bull") == 1,
            _C("ifvg_bear") == 1,
            requires=("ifvg_bull", "ifvg_bear"),
        ),
        Family(
            "premium_discount",
            "ict",
            (_C("in_discount") == 1) & (_C("htf_bull") == 1) & (_C("reclaim_up") == 1),
            (_C("in_premium") == 1) & (_C("htf_bull") == 0) & (_C("reclaim_dn") == 1),
        ),
        Family(
            "sweep_displacement",
            "ict",
            (_C("swept_low") == 1) & (_C("displacement") == 1) & (_C("close") > _C("open")),
            (_C("swept_high") == 1) & (_C("displacement") == 1) & (_C("close") < _C("open")),
        ),
        Family(
            "htf_poi_reclaim",
            "ict",
            (_C("at_support_poi_recent") == 1) & (_C("reclaim_up") == 1),
            (_C("at_resistance_poi_recent") == 1) & (_C("reclaim_dn") == 1),
            requires=("at_support_poi_recent", "at_resistance_poi_recent"),
        ),
        # ---- VOLUME / VOLATILITY ----
        Family(
            "vol_spike_break",
            "vol",
            (_C("rel_vol") > 1.5) & (_C("close") > _C("ph")) & (_C("close") > _C("open")),
            (_C("rel_vol") > 1.5) & (_C("close") < _C("pl")) & (_C("close") < _C("open")),
        ),
        Family(
            "vol_exhaustion_revert",
            "vol",
            (_C("rel_vol") > 2.0) & (_C("low") < _C("pl")) & (_C("close") > _C("close").shift(1)),
            (_C("rel_vol") > 2.0) & (_C("high") > _C("ph")) & (_C("close") < _C("close").shift(1)),
        ),
        Family(
            "range_expansion_dir",
            "vol",
            (_C("range_expansion") == 1) & (_C("close") > _C("open")) & (_C("adx") > 20),
            (_C("range_expansion") == 1) & (_C("close") < _C("open")) & (_C("adx") > 20),
        ),
        # ---- ORDER FLOW (MBP1, <=2025-05) ----
        Family(
            "cvd_trend",
            "orderflow",
            (_C("cvd") > _C("cvd").shift(5)) & _cu(_C("close"), _C("ema21")),
            (_C("cvd") < _C("cvd").shift(5)) & _cd(_C("close"), _C("ema21")),
            requires=("cvd",),
        ),
        Family(
            "cvd_divergence",
            "orderflow",
            (_C("close") < _C("close").shift(5))
            & (_C("cvd") > _C("cvd").shift(5))
            & (_C("close") > _C("open")),
            (_C("close") > _C("close").shift(5))
            & (_C("cvd") < _C("cvd").shift(5))
            & (_C("close") < _C("open")),
            requires=("cvd",),
        ),
    ]
    return {f.name: f for f in fams}


FAMILIES: dict[str, Family] = _families()

# Type-appropriate session / exit / stop grids keep the candidate count broad but bounded.
_KIND_SESSIONS: dict[str, tuple[str, ...]] = {
    "trend": ("RTH", "NY_AM", "LONDON_NY", "ALLDAY"),
    "mr": ("RTH", "NY_AM", "NY_PM"),
    "breakout": ("RTH", "NY_AM", "LONDON_NY"),
    "ict": ("RTH", "NY_AM", "ALLDAY"),
    "vol": ("RTH", "NY_AM", "ALLDAY"),
    "orderflow": ("RTH", "NY_AM"),
}
# Exit model: ('rr', rr) fixed target = rr*stop; ('trail', k) ATR-trail; ('eod', _) hold to EOD.
_KIND_EXITS: dict[str, tuple[tuple[str, float], ...]] = {
    "trend": (("rr", 2.0), ("rr", 3.0), ("trail", 1.5), ("eod", 0.0)),
    "mr": (("rr", 1.0), ("rr", 1.5), ("rr", 2.0)),
    "breakout": (("rr", 1.5), ("rr", 2.0), ("trail", 1.5), ("eod", 0.0)),
    "ict": (("rr", 1.0), ("rr", 1.5), ("rr", 2.0), ("trail", 1.5)),
    "vol": (("rr", 1.5), ("rr", 2.0), ("trail", 1.5)),
    "orderflow": (("rr", 1.5), ("rr", 2.0), ("trail", 1.5)),
}
_KIND_STOPS: dict[str, tuple[float, ...]] = {
    "trend": (1.5, 2.5),
    "mr": (1.0, 1.5),
    "breakout": (1.5, 2.5),
    "ict": (1.0, 1.5),
    "vol": (1.5, 2.5),
    "orderflow": (1.5,),
}


@dataclass(frozen=True)
class Candidate:
    family: str
    kind: str
    session: str
    exit_model: str  # "rr" | "trail" | "eod"
    rr: float  # target multiple (rr) or trail multiple (trail) — informational for eod
    stop_atr: float
    requires_mbp1: bool

    @property
    def name(self) -> str:
        tag = {"rr": f"rr{self.rr:g}", "trail": f"trail{self.rr:g}", "eod": "eod"}[self.exit_model]
        return f"{self.family}|{self.session}|s{self.stop_atr:g}|{tag}"

    def params(self) -> StrategyParams:
        return StrategyParams(
            "p29",
            "1",
            (
                ("family", self.family),
                ("session", self.session),
                ("exit", self.exit_model),
                ("rr", self.rr),
                ("stop_atr", self.stop_atr),
            ),
        )


def generate_candidates() -> list[Candidate]:
    """Expand every family over its type-appropriate session / exit / stop grid."""
    out: list[Candidate] = []
    for fam in FAMILIES.values():
        mbp1 = fam.kind == "orderflow"
        for session in _KIND_SESSIONS[fam.kind]:
            for stop_atr in _KIND_STOPS[fam.kind]:
                for model, mult in _KIND_EXITS[fam.kind]:
                    out.append(Candidate(fam.name, fam.kind, session, model, mult, stop_atr, mbp1))
    return out


# --- Per-candidate signal frame + run -------------------------------------------------------------
_BIG_TARGET = 100000.0  # "no fixed target" sentinel (trail / eod exits ride to stop/EOD)


def build_signal_frame(frame: pl.DataFrame, cand: Candidate) -> pl.DataFrame:
    """Attach signal/stop_pts/target_pts for one candidate, gated to its session and EOD block."""
    fam = FAMILIES[cand.family]
    lo, hi = SESSIONS[cand.session]
    in_session = (_C("minute_of_day") >= lo) & (_C("minute_of_day") < hi)
    if any(req not in frame.columns for req in fam.requires):
        # Required data layer absent (e.g. MBP1 columns on a trades-only frame) -> no trades.
        long = short = _FALSE
    else:
        valid = _C("atr").is_not_null() & (_C("atr") > 0)
        for req in fam.requires:
            valid = valid & _C(req).is_not_null()
        long = fam.long.fill_null(value=False) & in_session & valid
        short = fam.short.fill_null(value=False) & in_session & valid
    stop_pts = cand.stop_atr * _C("atr")
    target = (
        _BIG_TARGET
        if cand.exit_model in ("trail", "eod")
        else (cand.rr * cand.stop_atr * _C("atr"))
    )
    sig = attach_signals(
        frame,
        long=long,
        short=short,
        stop_pts=stop_pts,
        target_pts=pl.lit(target) if isinstance(target, float) else target,
    )
    return sig.filter((_C("minute_of_day") >= EOD_BLOCK[0]) & (_C("minute_of_day") <= EOD_BLOCK[1]))


def run_candidate(
    frame: pl.DataFrame,
    cand: Candidate,
    *,
    config: AppConfig,
    rules: TopstepRules,
    slippage_mult: float = 1.0,
    commission_mult: float = 1.0,
) -> BacktestResult:
    """Run one candidate through the audited engine (1 MNQ, EOD-flat, full distribution)."""
    sig = build_signal_frame(frame, cand)
    trail = (
        TrailingConfig(mode="atr", trail_atr=cand.rr)
        if cand.exit_model == "trail"
        else TrailingConfig()
    )
    exec_cfg = ExecConfig(sizing_mode="mnq_fixed1", time_stop_bars=10_000)
    return run_backtest(
        sig,
        symbol="NQ.c",
        timeframe=config.backtest.timeframe,
        strategy=cand.params(),
        config=config,
        rules=rules,
        primary_cutoff=False,
        entry_mode="next_open",
        gate_entries=False,
        slippage_mult=slippage_mult,
        commission_mult=commission_mult,
        trailing=trail,
        exec_config=exec_cfg,
    )


def intraday_only(trades: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Enforce the EOD-flat guarantee: keep only trades whose entry and exit fall on the SAME CME
    trading day (entry trading_day == exit trading_day, computed with the +6h Globex offset).

    On early-close holidays RTH ends 13:00 with no 15:55 bar, so the engine's force-flat cannot
    fire; such a position would carry to the next session. Those are dropped (and counted) so that
    every reported trade is a true intraday EOD-flat trade. Returns ``(kept, n_dropped)``.
    """
    if trades.height == 0:
        return trades, 0
    ex = pl.col("exit_ts")
    tz = getattr(trades.schema["exit_ts"], "time_zone", None)
    if tz is None:
        ex = ex.dt.replace_time_zone("UTC")
    t = trades.with_columns(
        (ex.dt.convert_time_zone("America/New_York") + pl.duration(hours=6)).dt.date().alias("_etd")
    )
    kept = t.filter(pl.col("_etd") == pl.col("trading_day")).drop("_etd")
    return kept, trades.height - kept.height


def trades_frame(result: BacktestResult, cand: Candidate) -> pl.DataFrame:
    """Persisted per-trade rows for a candidate (points-based MFE/MAE, $ net)."""
    rows: list[dict[str, Any]] = []
    for t in result.trades:
        rows.append(
            {
                "candidate": cand.name,
                "family": cand.family,
                "kind": cand.kind,
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "trading_day": t.trading_day,
                "side": t.side,
                "net": t.net_pnl,
                "gross": t.gross_pnl,
                "fees": t.fees,
                "mfe": t.mfe,
                "mae": t.mae,
                "exit_reason": t.exit_reason,
                "entry_minute": t.entry_minute,
            }
        )
    schema = {
        "candidate": pl.Utf8,
        "family": pl.Utf8,
        "kind": pl.Utf8,
        "entry_ts": pl.Datetime,
        "exit_ts": pl.Datetime,
        "trading_day": pl.Date,
        "side": pl.Int64,
        "net": pl.Float64,
        "gross": pl.Float64,
        "fees": pl.Float64,
        "mfe": pl.Float64,
        "mae": pl.Float64,
        "exit_reason": pl.Utf8,
        "entry_minute": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


# Config overrides: allow non-RTH entries, EOD-flat at 15:55, no 12:30 primary cutoff.
PHASE29_OVERRIDES: dict[str, Any] = {
    "backtest": {
        "rth_only": False,
        "no_new_trade_primary_et": "15:50",
        "no_new_trade_hard_et": "15:50",
        "force_flat_et": "15:55",
    },
    "validation": {
        "train_start": "2021-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2026-12-31",
    },
}

FEATURE_KINDS: tuple[str, ...] = ("trend", "mr", "breakout", "ict", "vol", "orderflow")


@dataclass
class FactoryConfig:
    """Knobs for a Phase-29 run (kept tiny; the grid is defined above)."""

    timeframe: str = "5m"
    families: tuple[str, ...] = field(default_factory=lambda: tuple(FAMILIES.keys()))
