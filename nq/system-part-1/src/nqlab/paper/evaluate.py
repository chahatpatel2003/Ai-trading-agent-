"""Forward evaluation + verdict (Phase 15).

Does NOT declare success until enough *genuinely forward* trades exist. Replay trades
(historical) are tracked for testing but never count toward forward confirmation. The
verdict stays ``PENDING_FORWARD_SAMPLE`` until the forward sample reaches the minimum.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

from nqlab.config.models import TopstepRules
from nqlab.discovery.deflation import deflated_sharpe_ratio, expected_max_sharpe
from nqlab.discovery.phase12_stats import practical_stats
from nqlab.validation.periods import sharpe_skew_kurt

PENDING_FORWARD_SAMPLE = "PENDING_FORWARD_SAMPLE"
FORWARD_ACCEPT_FOR_SHADOW = "FORWARD_ACCEPT_FOR_SHADOW"
FORWARD_REJECT = "FORWARD_REJECT"

# Minimum genuinely-forward trades before any forward verdict is rendered.
MIN_FORWARD_TRADES = 150

# Phase-14 historical reference for the frozen bos_5m_trigger (for comparison only).
PHASE14_REFERENCE: dict[str, Any] = {
    "profit_factor": 1.155,
    "expectancy": 8.94,
    "net_pnl": 4280.0,
    "trades": 479,
    "ruin": 0.01,
    "p_pass_50k": 0.62,
    "dsr": 0.44,
    "verdict": "REJECT (DSR<0.90)",
}


def _stats(
    rows: list[dict[str, Any]],
    *,
    rules: TopstepRules,
    account_size: float,
    months: float,
    rules_by_account: dict[str, TopstepRules] | None,
    frozen_count: int,
) -> dict[str, Any]:
    if not rows:
        return {"trade_count": 0}
    trades = [
        SimpleNamespace(net_pnl=r["pnl"], trading_day=date.fromisoformat(r["trading_day"]))
        for r in rows
    ]
    s = practical_stats(
        trades,  # type: ignore[arg-type]  # duck-typed: only .net_pnl / .trading_day are read
        account_size=account_size,
        rules=rules,
        months=months,
        rules_by_account=rules_by_account,
    )
    pnls = [float(r["pnl"]) for r in rows]
    sharpe, skew, kurt, n = sharpe_skew_kurt(pnls)
    # DSR at the FROZEN candidate count (1 pre-registered strategy -> sr0 ~ 0).
    sr0 = expected_max_sharpe(0.0, frozen_count)
    s["forward_sharpe"] = round(sharpe, 4)
    s["dsr_frozen"] = round(deflated_sharpe_ratio(sharpe, sr0, n, skew, kurt), 4)
    return s


def evaluate_forward(
    rows: list[dict[str, Any]],
    *,
    rules: TopstepRules,
    account_size: float,
    months: float,
    rules_by_account: dict[str, TopstepRules] | None = None,
    dsr_threshold: float = 0.90,
    min_trades: int = MIN_FORWARD_TRADES,
    frozen_count: int = 1,
) -> dict[str, Any]:
    """Render the forward verdict + stats. Replay rows never count as forward confirmation."""
    fwd = [r for r in rows if r.get("mode") == "forward"]
    repl = [r for r in rows if r.get("mode") == "replay"]
    fwd_stats = _stats(
        fwd,
        rules=rules,
        account_size=account_size,
        months=months,
        rules_by_account=rules_by_account,
        frozen_count=frozen_count,
    )
    repl_stats = _stats(
        repl,
        rules=rules,
        account_size=account_size,
        months=months,
        rules_by_account=rules_by_account,
        frozen_count=frozen_count,
    )

    if len(fwd) < min_trades:
        verdict = PENDING_FORWARD_SAMPLE
        reason = (
            f"only {len(fwd)} genuine forward trades (need >= {min_trades}); "
            f"{len(repl)} replay trades do not count toward confirmation"
        )
    else:
        passes = (
            fwd_stats.get("net_pnl", 0.0) > 0
            and fwd_stats.get("ruin", 1.0) <= 0.10
            and fwd_stats.get("dsr_frozen", 0.0) >= dsr_threshold
        )
        verdict = FORWARD_ACCEPT_FOR_SHADOW if passes else FORWARD_REJECT
        reason = (
            "forward sample positive, ruin <= 10%, and clears DSR at the frozen count"
            if passes
            else "forward sample fails DSR and/or economics"
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "forward_trades": len(fwd),
        "replay_trades": len(repl),
        "min_forward_trades": min_trades,
        "forward_stats": fwd_stats,
        "replay_stats": repl_stats,
        "phase14_reference": PHASE14_REFERENCE,
    }
