"""Phase 36 — Topstep payout-first account-lifecycle simulator.

Different objective: probability of reaching FIRST PAYOUT before the account blows, and the EV of the
rinse-and-repeat cycle. Rules are read from config (printed below); winning-day / consistency are
configurable and MUST be verified against the live Topstep dashboard. Topstep first-payout caps and
profit split are NOT modeled, so payout-dollar figures are an UPPER BOUND; the decision metric is
P(payout before blow) + days-to-payout, which are rule-robust.

Strategies use the frozen finalist trade lists (1 MNQ), scaled per sizing. Day P&L applies a daily
target (stop taking trades once hit), daily stop, and max trades/day. The account uses the locked
trailing floor min(peak - MLL, start) with intraday breach via per-trade MAE.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zoneinfo
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nqlab.config.loader import load_config, load_topstep_rules
from nqlab.discovery import phase29 as P29
from nqlab.discovery import phase30 as P30
from nqlab.discovery import phase32 as P32

OUT = Path("docs/phase36")
OUT.mkdir(parents=True, exist_ok=True)
ET = zoneinfo.ZoneInfo("America/New_York")
RNG = np.random.default_rng(36)
N_SIMS = 4000
MAX_DAYS = 150
MNQ_PV = 2.0
# Representative monthly subscription + reset cost (USER MUST VERIFY — Topstep changes these).
FEES = {"50K": {"sub": 49.0, "reset": 49.0}, "100K": {"sub": 99.0, "reset": 99.0},
        "150K": {"sub": 149.0, "reset": 149.0}}
PAYOUT_CAP = {"50K": 2000.0, "100K": 3000.0, "150K": 4500.0}  # caveat: first-payout caps vary


def _exit_td(ts: Any) -> Any:
    return (ts.astimezone(ET) + timedelta(hours=6)).date()


def _load_finalist_trades(cfg: Any, rules: Any) -> dict[str, list[dict[str, Any]]]:
    """Per-trade (net@1MNQ, mae_pts, entry_ts, trading_day) for the finalists."""
    fr = pl.read_parquet(cfg.data.artifacts_root / "phase30" / "frame_5m_full_vp.parquet")
    cmap = {c.name: c for c in P30.generate_candidates()}
    out: dict[str, list[dict[str, Any]]] = {}
    for key, name in (("swd", "swd_clean_path|RTH|a2.5|toPOC"), ("idb", "idb_base|RTH|a2.5|eod")):
        res = P30.run_candidate(fr, cmap[name], config=cfg, rules=rules)
        out[key] = [{"day": t.trading_day, "ts": t.entry_ts, "net": t.net_pnl, "mae": t.mae}
                    for t in res.trades if _exit_td(t.exit_ts) == t.trading_day]
    # breakout_acceptance via phase34b
    spec = importlib.util.spec_from_file_location("p34b", "scripts/phase34b_run.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["p34b"] = m
    spec.loader.exec_module(m)
    frb = P32.add_regime_buckets(fr)
    _C = pl.col
    st = m.Stack("ba", "", (_C("vp_vah_break_up") == 1) & (_C("vp_pos") > 0) & (_C("vp_dev_poc_shift") > 0), m._F)
    from nqlab.backtest.engine import run_backtest
    from nqlab.domain.strategy.base import ExecConfig, StrategyParams
    sig = m.build(frb, st, "poc", 2.5, bans=True)
    res = run_backtest(sig, symbol="NQ.c", timeframe="5m", strategy=StrategyParams("ba", "1"),
                       config=cfg, rules=rules, primary_cutoff=False, entry_mode="next_open",
                       gate_entries=False, exec_config=ExecConfig(sizing_mode="mnq_fixed1", time_stop_bars=10000))
    out["breakout_acceptance"] = [{"day": t.trading_day, "ts": t.entry_ts, "net": t.net_pnl, "mae": t.mae}
                                  for t in res.trades if _exit_td(t.exit_ts) == t.trading_day]
    out["portfolio"] = sorted(out["idb"] + out["swd"], key=lambda x: x["ts"])
    return out


def _all_days(cfg: Any) -> list[Any]:
    fr = pl.read_parquet(cfg.data.artifacts_root / "phase30" / "frame_5m_full_vp.parquet",
                         columns=["trading_day", "is_rth"])
    return sorted(fr.filter(pl.col("is_rth"))["trading_day"].unique().to_list())


def day_series(trades: list[dict], all_days: list[Any], sizing: int, daily_target: float,
               daily_stop: float, max_trades: int) -> tuple[np.ndarray, np.ndarray]:
    """(day_pnl, day_low) over all trading days, scaled to `sizing` MNQ, with daily caps."""
    by_day: dict[Any, list[dict]] = {}
    for t in trades:
        by_day.setdefault(t["day"], []).append(t)
    pnl = np.zeros(len(all_days))
    low = np.zeros(len(all_days))
    for i, d in enumerate(all_days):
        ts = sorted(by_day.get(d, []), key=lambda x: x["ts"])
        cum = 0.0
        lo = 0.0
        n = 0
        for t in ts:
            if n >= max_trades or cum >= daily_target or cum <= -daily_stop:
                break
            net = t["net"] * sizing
            mae_usd = t["mae"] * MNQ_PV * sizing
            lo = min(lo, cum - mae_usd)
            cum += net
            n += 1
        pnl[i] = cum
        low[i] = lo
    return pnl, low


def lifecycle_mc(pnl: np.ndarray, low: np.ndarray, *, start: float, mll: float,
                 win_thresh: float, n_sims: int = N_SIMS, max_days: int = MAX_DAYS) -> dict[str, Any]:
    """Vectorized MC: iid-day bootstrap. Returns payout/blow probs, days-to-payout, profit@payout."""
    npool = len(pnl)
    equity = np.full(n_sims, start)
    peak = np.full(n_sims, start)
    wins = np.zeros(n_sims)
    maxwin = np.zeros(n_sims)
    active = np.ones(n_sims, dtype=bool)
    outcome = np.zeros(n_sims, dtype=np.int8)  # 1 payout, -1 blow, 0 timeout
    days = np.full(n_sims, max_days)
    profit_at = np.zeros(n_sims)
    for t in range(max_days):
        idx = RNG.integers(npool, size=n_sims)
        dp, dl = pnl[idx], low[idx]
        floor = np.minimum(peak - mll, start)
        # intraday breach on still-active accounts
        breach = active & ((equity + dl) < floor)
        outcome[breach] = -1
        days[breach] = t + 1
        active &= ~breach
        # settle the day for survivors
        equity = np.where(active, equity + dp, equity)
        peak = np.maximum(peak, equity)
        is_win = active & (dp >= win_thresh)
        wins += is_win
        maxwin = np.where(is_win, np.maximum(maxwin, dp), maxwin)
        # payout eligibility
        profit = equity - start
        elig = active & (wins >= 5) & (profit > 0) & (maxwin <= 0.5 * profit)
        outcome[elig] = 1
        days[elig] = t + 1
        profit_at[elig] = profit[elig]
        active &= ~elig
        if not active.any():
            break
    pay = outcome == 1
    blow = outcome == -1
    d_pay = days[pay]
    return {
        "p_payout": round(float(pay.mean()), 4), "p_blow": round(float(blow.mean()), 4),
        "p_timeout": round(float((outcome == 0).mean()), 4),
        "days_med": int(np.median(d_pay)) if pay.any() else None,
        "days_p90": int(np.percentile(d_pay, 90)) if pay.any() else None,
        "profit_at_payout": round(float(profit_at[pay].mean()), 2) if pay.any() else None,
    }


def main() -> None:
    cfg = load_config(profile="dev")
    rules_by = {a: load_topstep_rules(f"topstep_{a}") for a in ("50k", "100k", "150k")}
    print("=== Topstep rules (from config) ===")
    for a, r in rules_by.items():
        print(f"  {r.name}: size=${r.account_size:.0f} MLL=${r.trailing_max_loss:.0f} "
              f"target=${r.profit_target:.0f} maxC={r.max_contracts} lock_at_start={r.lock_trailing_at_start} "
              f"basis={r.trailing_basis}")
    print("  payout model: 5 winning days; consistency: best win-day <= 50% of profit; "
          "winning-day threshold tested at $150 and $200 (VERIFY on dashboard).")
    rules150 = rules_by["150k"]
    trades = _load_finalist_trades(cfg, rules150)
    days = _all_days(cfg)
    print(f"\ntrading-day pool: {len(days)} | trades: " +
          ", ".join(f"{k}={len(v)}" for k, v in trades.items()))

    accounts = {"50K": (50000.0, 2000.0), "100K": (100000.0, 3000.0), "150K": (150000.0, 4500.0)}
    rows: list[dict[str, Any]] = []
    for strat in ("swd", "idb", "breakout_acceptance", "portfolio"):
        for sizing in (1, 2, 3, 5):
            pnl, low = day_series(trades[strat], days, sizing, daily_target=10000.0,
                                  daily_stop=10000.0, max_trades=20)  # caps loose; low-freq
            for acct, (start, mll) in accounts.items():
                for wt in (150.0, 200.0):
                    r = lifecycle_mc(pnl, low, start=start, mll=mll, win_thresh=wt)
                    rows.append({"strategy": strat, "sizing": f"{sizing} MNQ", "account": acct,
                                 "win_day": wt, **r})
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(OUT / "phase36_lifecycle.csv")
    (OUT / "_payout.json").write_text(json.dumps(rows, indent=2, default=str))

    print("\n=== TOP by P(payout before blow), $200 winning-day ===")
    cols = ["strategy", "sizing", "account", "win_day", "p_payout", "p_blow", "p_timeout",
            "days_med", "days_p90", "profit_at_payout"]
    with pl.Config(tbl_rows=24, fmt_str_lengths=20, tbl_width_chars=200):
        print(df.filter(pl.col("win_day") == 200).sort("p_payout", descending=True).select(cols).head(20))
    print("\n=== best per account (max P(payout), $200) ===")
    best = (df.filter(pl.col("win_day") == 200).sort("p_payout", descending=True)
            .group_by("account").first().sort("account"))
    print(best.select(cols))


if __name__ == "__main__":
    main()
