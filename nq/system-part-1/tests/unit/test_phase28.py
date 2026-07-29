"""Phase 28 — mechanical LTF-confirmation registry + conditional-on-touch evaluation."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl

from nqlab.config.loader import load_topstep_rules
from nqlab.discovery.phase28 import CONFIRMATIONS, confirmation_signal, evaluate_confirmation


def test_all_confirmations_are_mechanical() -> None:
    # Every confirmation must carry an exact mechanical definition + long/short exprs.
    assert len(CONFIRMATIONS) >= 10
    for conf in CONFIRMATIONS.values():
        assert conf.definition and len(conf.definition) > 10
        assert isinstance(conf.long(), pl.Expr)
        assert isinstance(conf.short(), pl.Expr)
    for needed in ("sweep_reclaim", "mss", "choch", "ifvg", "cisd", "displacement",
                   "micro_swing", "vwap_reclaim"):
        assert needed in CONFIRMATIONS


def _frame() -> pl.DataFrame:
    # Minimal frame with the columns the confirmations + POI-touch gate reference.
    rows = []
    d = date(2024, 1, 2)
    t0 = datetime(d.year, d.month, d.day, 9, 35)
    for i in range(40):
        px = 20000.0 + (i % 5)
        rows.append({
            "ts": t0 + timedelta(minutes=5 * i), "trading_day": d, "minute_of_day": 575 + 5 * i,
            "is_rth": True, "open": px - 1, "high": px + 3, "low": px - 3, "close": px,
            "atr": 10.0, "vwap_session": 20000.0,
            "at_support_poi_recent": 1, "at_resistance_poi_recent": 0,
            "swept_low": 1 if i % 6 == 0 else 0, "swept_high": 0,
            "reclaim_up": 1 if i % 6 == 0 else 0, "reclaim_dn": 0,
            "mss_up": 1 if i % 8 == 0 else 0, "mss_dn": 0,
            "choch_up": 0, "choch_dn": 0, "ifvg_bull": 0, "ifvg_bear": 0,
            "cisd_up": 1 if i % 7 == 0 else 0, "cisd_dn": 0, "displacement": 0,
            "in_bull_fvg": 0, "in_bear_fvg": 0,
            "aligned_long_a": 1, "aligned_short_a": 0,
        })
    return pl.DataFrame(rows)


def test_confirmation_signal_long_only_at_support_poi() -> None:
    # A support-POI touch -> only LONG signals from a bullish confirmation; never short.
    sig = confirmation_signal(_frame(), CONFIRMATIONS["sweep_reclaim"], stop_pts=10.0,
                              target_pts=10.0)
    fired = sig.filter(pl.col("signal") != 0)
    assert fired.height > 0
    assert (fired["signal"] == 1).all()  # support-POI touch -> long only, never short


def test_evaluate_confirmation_returns_panel() -> None:
    rules = load_topstep_rules("topstep_150k")
    from nqlab.config.loader import load_config
    cfg = load_config(profile="dev")
    out = evaluate_confirmation(cfg, CONFIRMATIONS["sweep_reclaim"], _frame(), tf="5m", rules=rules)
    # Either no trades (n=0) or a full reliability panel.
    assert "n" in out
    if out["n"] > 0:
        for k in ("win_rate", "pf", "mfe_mae", "max_losing_streak", "expectancy"):
            assert k in out
