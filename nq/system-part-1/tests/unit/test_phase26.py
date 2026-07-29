"""Phase 26 — quality-stats panel + high-PF verdict + full-RTH families."""

from __future__ import annotations

from datetime import datetime, timedelta

from nqlab.backtest.result import Trade
from nqlab.discovery.phase26 import (
    ACCEPT_FOR_PAPER,
    NEEDS_MORE_DATA,
    REJECT,
    phase26_verdict,
    quality_stats,
)
from nqlab.discovery.phase26_families import PHASE26_FAMILIES, generate_phase26_candidates


def _tr(day: datetime, net: float) -> Trade:
    return Trade(entry_ts=day, exit_ts=day, trading_day=day.date(), side=1, instrument="MNQ",
                 contracts=1, entry_px=1.0, exit_px=1.0, stop_px=1.0, target_px=1.0,
                 gross_pnl=net, fees=0.0, net_pnl=net, exit_reason="x", bars_held=1,
                 mfe=abs(net), mae=abs(net) / 2)


def test_quality_stats_profit_factor_and_drawdown() -> None:
    # 7 wins of +300, 3 losses of -100 across distinct days -> PF = 2100/300 = 7.0.
    trades = []
    d = datetime(2024, 1, 2, 10, 0)
    for i in range(7):
        trades.append(_tr(d + timedelta(days=i), 300.0))
    for i in range(3):
        trades.append(_tr(d + timedelta(days=7 + i), -100.0))
    q = quality_stats(trades, years=1.0)
    assert q["trade_count"] == 10
    assert q["profit_factor"] == 7.0
    assert q["net"] == 1800.0
    assert q["max_drawdown"] >= 100.0  # the 3 consecutive losses
    assert q["max_losing_streak"] == 3
    assert q["return_over_maxdd"] is not None


def test_phase26_verdict_rejects_low_pf() -> None:
    val = {"trade_count": 200, "profit_factor": 1.2, "net": 5000, "return_over_maxdd": 4.0,
           "sharpe": 1.2, "win_month_frac": 0.8}
    v, reason = phase26_verdict(val_stats=val, lockbox_stats=None, cost2x_pf=None, pristine=True)
    assert v == REJECT
    assert "PF" in reason


def test_phase26_verdict_needs_more_data_few_trades() -> None:
    val = {"trade_count": 20, "profit_factor": 3.0, "net": 5000, "return_over_maxdd": 5.0,
           "sharpe": 2.0, "win_month_frac": 0.9}
    v, _ = phase26_verdict(val_stats=val, lockbox_stats=None, cost2x_pf=None, pristine=True)
    assert v == NEEDS_MORE_DATA


def test_phase26_verdict_accepts_strong_pf_with_lockbox() -> None:
    val = {"trade_count": 120, "profit_factor": 2.6, "net": 8000, "return_over_maxdd": 4.5,
           "sharpe": 1.4, "win_month_frac": 0.8}
    lock = {"trade_count": 40, "profit_factor": 2.3, "net": 3000}
    v, _ = phase26_verdict(val_stats=val, lockbox_stats=lock, cost2x_pf=1.9, pristine=True)
    assert v == ACCEPT_FOR_PAPER  # high PF on validation AND lockbox, survives 2x cost


def test_phase26_families_full_rth_and_wide_r() -> None:
    assert len(PHASE26_FAMILIES) == 5
    cands = generate_phase26_candidates(timeframe="5m")
    assert len(cands) == 5 * 2 * 2  # families x R(2) x session(2)
    # not NY-AM restricted: a full-RTH session variant (09:00-16:00) must exist.
    assert any(c.p["hour_start"] == 9 and c.p["hour_end"] == 16 for c in cands)
    # wide asymmetric targets (target ATR > stop ATR).
    for c in cands:
        assert c.p["target_atr"] > c.p["stop_atr"]
