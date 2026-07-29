"""Phase 22 — 1-trade/day gate, fixed 1:1 stops, losing-streak stats, 60% win-rate threshold."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl

from nqlab.backtest.result import Trade
from nqlab.discovery.phase16 import _losing_streaks
from nqlab.discovery.phase22 import PHASE22_THRESHOLDS, generate_phase22_candidates
from nqlab.discovery.phase22_families import PHASE22_FAMILIES


def _two_day_frame() -> pl.DataFrame:
    rows = []
    for d in (date(2024, 1, 2), date(2024, 1, 3)):
        t0 = datetime(d.year, d.month, d.day, 9, 35)
        for i in range(12):
            # price keeps reclaiming VWAP so the raw signal fires many times per day
            rows.append({
                "ts": t0 + timedelta(minutes=5 * i), "trading_day": d,
                "minute_of_day": 575 + 5 * i, "is_rth": True,
                "open": 20000.0, "high": 20010.0, "low": 19990.0, "close": 20005.0,
                "atr": 10.0, "vwap_session": 20000.0,
            })
    return pl.DataFrame(rows)


def test_one_trade_per_day_gate() -> None:
    strat = PHASE22_FAMILIES["p22_vwap_reclaim"](
        {"fix_stop": 10.0, "fix_target": 10.0, "hour_start": 9, "hour_end": 12,
         "risk_per_trade": 200.0, "sizing_mode": "mnq_min1", "min_hold_bars": 0,
         "time_stop_bars": 24}
    )
    sig = strat.signals(_two_day_frame())
    per_day = sig.filter(pl.col("signal") != 0).group_by("trading_day").len()
    # At most ONE signal per trading day.
    assert per_day["len"].max() <= 1
    assert per_day.height == 2  # both days produced exactly one


def test_fixed_one_to_one_stops() -> None:
    strat = PHASE22_FAMILIES["p22_vwap_reclaim"](
        {"fix_stop": 10.0, "fix_target": 10.0, "hour_start": 9, "hour_end": 12,
         "risk_per_trade": 200.0, "sizing_mode": "mnq_min1", "min_hold_bars": 0,
         "time_stop_bars": 24}
    )
    sig = strat.signals(_two_day_frame()).filter(pl.col("signal") != 0)
    assert sig["stop_pts"].unique().to_list() == [10.0]
    assert sig["target_pts"].unique().to_list() == [10.0]  # 1:1 R:R


def test_phase22_families_and_grid() -> None:
    assert set(PHASE22_FAMILIES) == {
        "p22_first_bias", "p22_continuation", "p22_sweep_rev",
        "p22_or_failbrk", "p22_vwap_reclaim", "p22_mbp1_confirm",
    }
    cands = generate_phase22_candidates(timeframe="5m")
    assert len(cands) == 6 * 4  # families x 4 fixed 1:1 sizes
    # every grid entry is 1:1.
    for c in cands:
        assert c.p["fix_stop"] == c.p["fix_target"]


def test_losing_streak_stats() -> None:
    def tr(day: int, net: float) -> Trade:
        ts = datetime(2024, 1, day, 9, 35)
        return Trade(entry_ts=ts, exit_ts=ts, trading_day=date(2024, 1, day), side=1,
                     instrument="MNQ", contracts=1, entry_px=1.0, exit_px=1.0, stop_px=1.0,
                     target_px=1.0, gross_pnl=net, fees=0.0, net_pnl=net, exit_reason="x",
                     bars_held=1)
    # L L L W L L W -> longest 3, one 3-run, zero 5-runs
    nets = [-1, -1, -1, 1, -1, -1, 1]
    s = _losing_streaks([tr(i + 1, v) for i, v in enumerate(nets)])
    assert s["max_losing_streak"] == 3
    assert s["n_3loss_streaks"] == 1
    assert s["n_5loss_streaks"] == 0
    assert s["had_10loss_streak"] is False


def test_phase22_threshold_requires_60pct_win() -> None:
    assert PHASE22_THRESHOLDS.win_rate_min == 0.60
    assert PHASE22_THRESHOLDS.pf_min == 1.30
    assert PHASE22_THRESHOLDS.pf_preferred == 1.70
