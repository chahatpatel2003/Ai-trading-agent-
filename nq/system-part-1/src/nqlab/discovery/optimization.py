"""Survival optimization for the positive-OOS setup families (Phase 7).

Sweeps exits (stop/target/trailing/min-hold/time-stop), risk-per-trade, and
session filters on the families that showed positive OOS or direction edge in
Phase 6 — to test whether better risk management makes them Topstep-survivable.
No new strategy families. Grids are seeded from the MAE/MFE diagnostics.
"""

from __future__ import annotations

from itertools import product

from nqlab.discovery.candidates import (
    COMBO_FAMILIES,
    SETUP_FAMILIES,
    _Family,
)

_RTH_OPEN_DEFAULT = 570

# Families that earned a survival sweep (positive OOS / direction edge in Phase 6).
FOCUS_FAMILIES: tuple[str, ...] = ("bias_pullback", "pd_sweep_rev", "cvd_pullback")

# Per-trade dollar risk to sweep (NQ/MNQ sizing chooses the instrument).
RISK_GRID: tuple[float, ...] = (25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0)

# Session windows (ET hours) seeded from diagnostics (NY-AM edge for cvd).
_SESSIONS: dict[str, tuple[int, int] | None] = {
    "full": None,
    "ny_am": (9, 12),  # 09:30-12:00
    "ny_open": (9, 11),  # 09:30-11:00
}

_LOOKUP: dict[str, type[_Family]] = {**SETUP_FAMILIES, **COMBO_FAMILIES}

# Family-specific required params (beyond the swept exit/risk/session knobs).
_BASE_PARAMS: dict[str, dict[str, object]] = {
    "bias_pullback": {},
    "pd_sweep_rev": {},
    "cvd_pullback": {"ema_span": 50, "rsi_low": 40.0, "rsi_high": 60.0, "z_window": 100},
}


def _make(family: str, extra: dict[str, object], rth_start_min: int) -> _Family:
    params = dict(_BASE_PARAMS[family])
    params.update(extra)
    return _LOOKUP[family](params, rth_start_min=rth_start_min)


def _apply_session(params: dict[str, object], session: str) -> None:
    window = _SESSIONS[session]
    if window is not None:
        params["hour_start"], params["hour_end"] = window


def generate_exit_candidates(
    families: tuple[str, ...] = FOCUS_FAMILIES, *, rth_start_min: int = _RTH_OPEN_DEFAULT,
    risk: float = 100.0,
) -> list[_Family]:
    """Sweep stop / target / trailing / min-hold (fixed risk, full session)."""
    out: list[_Family] = []
    for family in families:
        for stop, target, trail, min_hold in product(
            (1.5, 2.5), (2.0, 3.0), ("none", "atr"), (0, 6)
        ):
            extra: dict[str, object] = {
                "stop_atr": stop, "target_atr": target, "risk_per_trade": risk,
                "min_hold_bars": min_hold,
            }
            if trail == "atr":
                extra["trail"] = "atr"
                extra["trail_atr"] = 2.0
            out.append(_make(family, extra, rth_start_min))
    return out


def generate_risk_candidates(
    families: tuple[str, ...] = FOCUS_FAMILIES, *, rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_Family]:
    """Sweep risk-per-trade at a fixed (trailing) exit."""
    out: list[_Family] = []
    for family in families:
        for risk in RISK_GRID:
            out.append(_make(family, {
                "stop_atr": 2.0, "target_atr": 3.0, "trail": "atr", "trail_atr": 2.0,
                "min_hold_bars": 6, "risk_per_trade": risk,
            }, rth_start_min))
    return out


def generate_survival_candidates(
    families: tuple[str, ...] = FOCUS_FAMILIES, *, rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_Family]:
    """Combined survival sweep: stop x target x trailing x risk x session."""
    out: list[_Family] = []
    for family in families:
        for stop, target, trail, risk, session in product(
            (1.5, 2.5), (2.0, 4.0), ("none", "atr"), (50.0, 150.0), ("full", "ny_am")
        ):
            extra: dict[str, object] = {
                "stop_atr": stop, "target_atr": target, "risk_per_trade": risk, "min_hold_bars": 6,
            }
            if trail == "atr":
                extra["trail"] = "atr"
                extra["trail_atr"] = 2.0
            _apply_session(extra, session)
            out.append(_make(family, extra, rth_start_min))
    return out


# ---------------------------------------------------------------------------
# Phase 8: account-size / instrument / timeframe feasibility.
#
# Hypothesis: the Phase 6/7 direction edge is real but its per-trade noise band
# (avg MAE ~15 pts, MFE ~20 pts on NQ 1m) is wider than the 50K $2,000 trailing
# drawdown. Phase 8 changes the battlefield: bigger accounts (more DD budget),
# micro sizing (finer granularity), higher timeframes (smaller noise band vs the
# bar's ATR). Exits are seeded by MAE/MFE-in-ATR per timeframe: stop *beyond* the
# typical MAE, target *near/above* the MFE, a min-hold past the losing early-bar
# window, and a time stop at the end of the favorable holding zone.
# ---------------------------------------------------------------------------

# Expanded, smoother risk grid ($/trade). $500 reserved for 150K (see account cap).
RISK_GRID_FEASIBILITY: tuple[float, ...] = (
    10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0,
)

SIZING_MODES: tuple[str, ...] = (
    "adaptive", "nq_only", "mnq_only", "max_1_nq", "mnq_scaling",
)

# Per-timeframe exit seeds (stop/target in ATR units; holds in bars of that TF).
EXIT_BY_TF: dict[str, dict[str, object]] = {
    "1m": {"stop_atr": 2.5, "target_atr": 4.0, "min_hold_bars": 6, "time_stop_bars": 60},
    "5m": {"stop_atr": 2.0, "target_atr": 4.0, "min_hold_bars": 3, "time_stop_bars": 36},
    "15m": {"stop_atr": 1.5, "target_atr": 3.5, "min_hold_bars": 2, "time_stop_bars": 20},
}


def account_risk_grid(account: str) -> tuple[float, ...]:
    """Risk grid capped to what each account's trailing drawdown can plausibly hold."""
    if account.lower() in ("150k", "topstep_150k"):
        return RISK_GRID_FEASIBILITY  # includes $500
    if account.lower() in ("100k", "topstep_100k"):
        return tuple(r for r in RISK_GRID_FEASIBILITY if r <= 300.0)
    return tuple(r for r in RISK_GRID_FEASIBILITY if r <= 200.0)  # 50K


def _exit_extra(timeframe: str) -> dict[str, object]:
    if timeframe not in EXIT_BY_TF:
        msg = f"no exit seed for timeframe {timeframe!r}"
        raise KeyError(msg)
    extra = dict(EXIT_BY_TF[timeframe])
    extra["hour_start"], extra["hour_end"] = 9, 12  # NY-AM (Phase 7's best session)
    return extra


def generate_sizing_candidates(
    families: tuple[str, ...] = FOCUS_FAMILIES,
    *,
    timeframe: str,
    risk_grid: tuple[float, ...] = RISK_GRID_FEASIBILITY,
    sizing_modes: tuple[str, ...] = SIZING_MODES,
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_Family]:
    """sizing_mode x risk at the TF-seeded NY-AM exit (account sweep candidates)."""
    out: list[_Family] = []
    for family in families:
        for mode in sizing_modes:
            for risk in risk_grid:
                extra = _exit_extra(timeframe)
                extra["risk_per_trade"] = risk
                extra["sizing_mode"] = mode
                out.append(_make(family, extra, rth_start_min))
    return out


def generate_timeframe_candidates(
    families: tuple[str, ...] = FOCUS_FAMILIES,
    *,
    timeframe: str,
    sizing_mode: str = "mnq_scaling",
    risk_grid: tuple[float, ...] = RISK_GRID_FEASIBILITY,
    rth_start_min: int = _RTH_OPEN_DEFAULT,
) -> list[_Family]:
    """Risk sweep at the TF-seeded exit, fixed sizing (timeframe sweep candidates)."""
    out: list[_Family] = []
    for family in families:
        for risk in risk_grid:
            extra = _exit_extra(timeframe)
            extra["risk_per_trade"] = risk
            extra["sizing_mode"] = sizing_mode
            out.append(_make(family, extra, rth_start_min))
    return out
