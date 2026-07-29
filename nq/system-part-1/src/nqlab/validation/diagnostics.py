"""Trade diagnostics: where a strategy loses, and whether it's entries vs exits.

Answers the Phase-6 questions from the trade ledger: direction edge *before*
costs, MAE/MFE distributions, how much of the favorable move is captured,
PnL by entry session / hour / exit reason / holding time, and whether stops look
too tight (favorable excursion typically exceeded before the stop is hit).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from nqlab.backtest.result import Trade
from nqlab.domain.calendar.sessions import SessionLabeler


def _pct(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def _hold_bucket(bars: int) -> str:
    if bars <= 5:
        return "0-5"
    if bars <= 20:
        return "6-20"
    if bars <= 60:
        return "21-60"
    return "60+"


def diag_summary(trades: list[Trade]) -> dict[str, object]:
    """Compact diagnostics stored per candidate (family-level)."""
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "direction_edge_before_costs": False}
    gross = sum(t.gross_pnl for t in trades)
    net = sum(t.net_pnl for t in trades)
    avg_mfe = sum(t.mfe for t in trades) / n
    avg_mae = sum(t.mae for t in trades) / n
    realized = sum(t.side * (t.exit_px - t.entry_px) for t in trades) / n
    return {
        "n_trades": n,
        "gross_pnl": round(gross, 2),
        "cost_drag": round(gross - net, 2),
        "direction_edge_before_costs": gross > 0,
        "avg_mae": round(avg_mae, 2),
        "avg_mfe": round(avg_mfe, 2),
        "mfe_capture": round(realized / avg_mfe, 3) if avg_mfe > 0 else 0.0,
    }


def diagnose(trades: list[Trade], labeler: SessionLabeler) -> dict[str, object]:
    """Full diagnostic report for one strategy's trades."""
    n = len(trades)
    if n == 0:
        return {"n_trades": 0}

    maes = sorted(t.mae for t in trades)
    mfes = sorted(t.mfe for t in trades)
    gross = sum(t.gross_pnl for t in trades)
    net = sum(t.net_pnl for t in trades)

    by_session: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    by_hour: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    by_exit: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    by_hold: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for t in trades:
        by_session[labeler.session_of_minute(t.entry_minute)][0] += t.net_pnl
        by_session[labeler.session_of_minute(t.entry_minute)][1] += 1
        by_hour[t.entry_minute // 60][0] += t.net_pnl
        by_hour[t.entry_minute // 60][1] += 1
        by_exit[t.exit_reason][0] += t.net_pnl
        by_exit[t.exit_reason][1] += 1
        by_hold[_hold_bucket(t.bars_held)][0] += t.net_pnl
        by_hold[_hold_bucket(t.bars_held)][1] += 1

    stop_mfe = sorted(t.mfe for t in trades if t.exit_reason == "STOP")
    realized = sum(t.side * (t.exit_px - t.entry_px) for t in trades) / n
    avg_mfe = sum(mfes) / n
    avg_mae = sum(maes) / n

    def _fmt(d: dict[Any, list[float]]) -> dict[str, dict[str, float]]:
        return {str(k): {"net": round(v[0], 1), "n": int(v[1])} for k, v in sorted(d.items())}

    return {
        "n_trades": n,
        "net_pnl": round(net, 2),
        "gross_pnl": round(gross, 2),
        "cost_drag": round(gross - net, 2),
        "direction_edge_before_costs": gross > 0,
        "net_per_trade": round(net / n, 2),
        "gross_per_trade": round(gross / n, 2),
        "avg_mae": round(avg_mae, 2),
        "avg_mfe": round(avg_mfe, 2),
        "mae_p50": round(_pct(maes, 0.5), 2),
        "mae_p90": round(_pct(maes, 0.9), 2),
        "mfe_p50": round(_pct(mfes, 0.5), 2),
        "mfe_p90": round(_pct(mfes, 0.9), 2),
        "avg_realized_pts": round(realized, 2),
        "mfe_capture": round(realized / avg_mfe, 3) if avg_mfe > 0 else 0.0,
        "stop_mfe_p50": round(_pct(stop_mfe, 0.5), 2),
        "stops_likely_too_tight": bool(stop_mfe) and _pct(stop_mfe, 0.5) > avg_mae,
        "by_session": _fmt(by_session),
        "by_hour": _fmt(by_hour),
        "by_exit_reason": _fmt(by_exit),
        "by_holding": _fmt(by_hold),
        "worst_sessions": [k for k, v in sorted(by_session.items(), key=lambda kv: kv[1][0])[:2]],
        "worst_hours": [str(k) for k, v in sorted(by_hour.items(), key=lambda kv: kv[1][0])[:3]],
    }
