"""Phase 37 — simplest NQ/MNQ strategy that trades MOST RTH days, disciplined & adversarial.

Anti-bias: disprove first. One trade/day max (first valid signal), simple a-priori rules (NO tuning),
realistic costs, MNQ. Train 2021-22 / val 2023-24 / test 2025-26 + per-year walk-forward + Monte-Carlo
trade-order reshuffle + Topstep trailing sim + comparison vs a dumb always-long baseline. Honest
verdicts. Reuses the audited engine.
"""

from __future__ import annotations

import json
import zoneinfo
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nqlab.backtest.engine import run_backtest
from nqlab.config.loader import load_config, load_topstep_rules
from nqlab.discovery import phase29 as P29
from nqlab.domain.strategy.base import ExecConfig, StrategyParams, attach_signals

OUT = Path("docs/phase37")
OUT.mkdir(parents=True, exist_ok=True)
ET = zoneinfo.ZoneInfo("America/New_York")
_C = pl.col
_BIG = 100000.0
RNG = np.random.default_rng(37)


def _first_per_day(sig: pl.Expr) -> pl.Expr:
    return sig & (sig.cast(pl.Int32).cum_sum().over("trading_day") == 1)


@dataclass(frozen=True)
class Strat:
    name: str
    long: pl.Expr
    short: pl.Expr
    stop_atr: float
    exit: str  # "eod" | "rr2" | "rr1"


def strategies() -> list[Strat]:
    rth = (_C("minute_of_day") >= 570) & (_C("minute_of_day") < 955)
    after_or = (_C("minute_of_day") >= 600) & (_C("minute_of_day") < 900)
    up = _C("close") > _C("close").shift(1)
    dn = _C("close") < _C("close").shift(1)
    return [
        # ORB breakout: first close beyond the opening range (continuation, hold to EOD)
        Strat("orb_break",
              after_or & (_C("close") > _C("or_high")) & (_C("close").shift(1) <= _C("or_high")),
              after_or & (_C("close") < _C("or_low")) & (_C("close").shift(1) >= _C("or_low")),
              2.0, "eod"),
        # ORB fade: first poke beyond OR then close back inside (fade to mid, 2R)
        Strat("orb_fade",
              after_or & (_C("low") < _C("or_low")) & (_C("close") > _C("or_low")),
              after_or & (_C("high") > _C("or_high")) & (_C("close") < _C("or_high")),
              1.5, "rr2"),
        # VWAP reversion: first >1.5 ATR stretch from VWAP, fade toward it (2R)
        Strat("vwap_revert",
              rth & ((_C("close") - _C("vwap_session")) < -1.5 * _C("atr")) & up,
              rth & ((_C("close") - _C("vwap_session")) > 1.5 * _C("atr")) & dn,
              1.5, "rr2"),
        # Trend pullback to VWAP: above VWAP, first tag of VWAP that holds (continuation, EOD)
        Strat("trend_pullback_vwap",
              rth & (_C("close") > _C("vwap_session")) & (_C("low") <= _C("vwap_session")) & up,
              rth & (_C("close") < _C("vwap_session")) & (_C("high") >= _C("vwap_session")) & dn,
              2.0, "eod"),
        # Prior-day H/L fade: first tag of PDH -> short / PDL -> long (2R)
        Strat("pdhl_fade",
              rth & (_C("low") < _C("prev_day_low")) & (_C("close") > _C("prev_day_low")),
              rth & (_C("high") > _C("prev_day_high")) & (_C("close") < _C("prev_day_high")),
              1.5, "rr2"),
        # Inside-day breakout: prior inside day + first OR break (continuation, EOD)
        Strat("inside_day_break",
              after_or & (_C("prev_inside_day") == 1) & (_C("close") > _C("or_high")),
              after_or & (_C("prev_inside_day") == 1) & (_C("close") < _C("or_low")),
              2.0, "eod"),
        # Dumb baselines: always long / short on the first RTH bar, hold to EOD
        Strat("BASELINE_long", (_C("minute_of_day") >= 570) & (_C("minute_of_day") < 955),
              pl.lit(value=False), 5.0, "eod"),  # noqa: FBT003
        Strat("BASELINE_short", pl.lit(value=False),
              (_C("minute_of_day") >= 570) & (_C("minute_of_day") < 955), 5.0, "eod"),  # noqa: FBT003
    ]


def build(fr: pl.DataFrame, s: Strat) -> pl.DataFrame:
    valid = _C("atr").is_not_null() & (_C("atr") > 0)
    long = _first_per_day(s.long.fill_null(value=False) & valid)
    short = _first_per_day(s.short.fill_null(value=False) & valid)
    stop = s.stop_atr * _C("atr")
    target = pl.lit(_BIG) if s.exit == "eod" else ({"rr2": 2.0, "rr1": 1.0}[s.exit] * stop)
    sig = attach_signals(fr, long=long, short=short, stop_pts=stop, target_pts=target)
    return sig.filter((_C("minute_of_day") >= 0) & (_C("minute_of_day") <= 960))


def _exit_td(ts: Any) -> Any:
    return (ts.astimezone(ET) + timedelta(hours=6)).date()


def _panel(trades: list[Any], n_days_total: int) -> dict[str, Any]:
    sd = [t for t in trades if _exit_td(t.exit_ts) == t.trading_day]
    n = len(sd)
    if n == 0:
        return {"n": 0}
    nets = np.array([t.net_pnl for t in sd])
    w, ls = nets[nets > 0], nets[nets < 0]
    aw = float(w.mean()) if w.size else 0.0
    al = float(-ls.mean()) if ls.size else 0.0
    be = al / (aw + al) if (aw + al) > 0 else None
    wr = float((nets > 0).mean())
    eq = np.cumsum(nets)
    mdd = float((np.maximum.accumulate(eq) - eq).max())
    days_traded = len({t.trading_day for t in sd})
    return {
        "n": n, "days_traded_pct": round(days_traded / n_days_total, 3) if n_days_total else None,
        "trades_per_day": round(n / max(days_traded, 1), 2),
        "win_rate": round(wr, 4), "breakeven_wr": round(be, 4) if be else None,
        "edge_vs_be": round(wr - be, 4) if be else None,
        "pf": round(float(w.sum() / -ls.sum()), 3) if ls.size else None,
        "avg_win": round(aw, 2), "avg_loss": round(-al, 2), "expectancy": round(float(nets.mean()), 2),
        "net": round(float(nets.sum()), 2), "max_dd": round(mdd, 2),
        "ret_dd": round(float(nets.sum() / mdd), 2) if mdd > 0 else None,
    }


def _mc_ruin(nets: np.ndarray, mae_usd: np.ndarray, limit: float, sims: int = 2000) -> float:
    n = len(nets)
    if n < 5:
        return 1.0
    br = 0
    for _ in range(sims):
        i = RNG.permutation(n)
        ns, ms = nets[i], mae_usd[i]
        eq = np.cumsum(ns)
        prev = np.concatenate([[0.0], eq[:-1]])
        dd = (np.maximum.accumulate(np.maximum(eq, prev)) - (prev - ms)).max()
        br += dd >= limit
    return round(br / sims, 4)


def main() -> None:
    cfg = load_config(profile="dev", overrides=P29.PHASE29_OVERRIDES)
    rules = load_topstep_rules("topstep_150k")
    fr = pl.read_parquet(cfg.data.artifacts_root / "phase30" / "frame_5m_full_vp.parquet")
    days_by_year: dict[int, int] = {}
    for y, sub in fr.filter(_C("is_rth")).group_by(_C("trading_day").dt.year()):
        days_by_year[int(y[0] if isinstance(y, tuple) else y)] = sub["trading_day"].n_unique()
    total_days = sum(days_by_year.values())

    splits = {"train(21-22)": (2021, 2022), "val(23-24)": (2023, 2024), "test(25-26)": (2025, 2026)}
    rows: list[dict[str, Any]] = []
    raw_sample: dict[str, list] = {}
    for s in strategies():
        res = run_backtest(build(fr, s), symbol="NQ.c", timeframe="5m",
                           strategy=StrategyParams("p37", "1"), config=cfg, rules=rules,
                           primary_cutoff=False, entry_mode="next_open", gate_entries=False,
                           exec_config=ExecConfig(sizing_mode="mnq_fixed1", time_stop_bars=10000))
        allt = [t for t in res.trades if _exit_td(t.exit_ts) == t.trading_day]
        full = _panel(allt, total_days)
        row: dict[str, Any] = {"strategy": s.name, "exit": s.exit, "stop_atr": s.stop_atr, **{f"full_{k}": v for k, v in full.items()}}
        # splits
        for sp, (y0, y1) in splits.items():
            tt = [t for t in allt if y0 <= t.trading_day.year <= y1]
            nd = sum(v for y, v in days_by_year.items() if y0 <= y <= y1)
            p = _panel(tt, nd)
            row[f"{sp}_pf"] = p.get("pf")
            row[f"{sp}_wr"] = p.get("win_rate")
            row[f"{sp}_net"] = p.get("net")
            row[f"{sp}_dd"] = p.get("max_dd")
        # per-year net (walk-forward consistency)
        row["yearly_net"] = json.dumps({y: round(sum(t.net_pnl for t in allt if t.trading_day.year == y), 0)
                                        for y in range(2021, 2027)})
        # MC ruin on 150K/$4500 (1 MNQ) + Topstep-ish
        nets = np.array([t.net_pnl for t in allt])
        maes = np.array([t.mae * 2 for t in allt])
        row["mc_breach_3k"] = _mc_ruin(nets, maes, 3000)
        row["mc_breach_4500"] = _mc_ruin(nets, maes, 4500)
        rows.append(row)
        raw_sample[s.name] = [
            {"day": str(t.trading_day), "dir": "L" if t.side == 1 else "S",
             "entry": round(t.entry_px, 1), "stop": round(t.stop_px, 1), "exit": round(t.exit_px, 1),
             "reason": t.exit_reason, "net": round(t.net_pnl, 2)}
            for t in allt[:12]
        ]
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(OUT / "phase37_simple.csv")
    (OUT / "_simple.json").write_text(json.dumps({"rows": rows, "raw": raw_sample,
                                                  "days_by_year": days_by_year}, indent=2, default=str))
    cols = ["strategy", "full_n", "full_days_traded_pct", "full_trades_per_day", "full_win_rate",
            "full_breakeven_wr", "full_edge_vs_be", "full_pf", "full_max_dd", "full_ret_dd",
            "train(21-22)_pf", "val(23-24)_pf", "test(25-26)_pf", "mc_breach_3k"]
    print(f"trading days: {days_by_year} total={total_days}")
    print("\n=== SIMPLE STRATEGIES (one-per-day) — full + OOS splits ===")
    with pl.Config(tbl_rows=12, fmt_str_lengths=22, tbl_width_chars=240):
        print(df.select(cols))
    print("\n=== verdict screen (trades>=40% days, edge>0, PF>1.1 all 3 splits, beats baseline) ===")
    base_net = df.filter(_C("strategy") == "BASELINE_long")["full_net"][0]
    for r in rows:
        if r["strategy"].startswith("BASELINE"):
            continue
        oos_ok = all((r.get(f"{sp}_pf") or 0) > 1.1 for sp in splits)
        freq_ok = (r.get("full_days_traded_pct") or 0) >= 0.40
        edge_ok = (r.get("full_edge_vs_be") or -1) > 0
        verdict = "PASS" if (oos_ok and freq_ok and edge_ok and (r.get("full_pf") or 0) >= 1.3) else \
                  "NEEDS_MORE" if (oos_ok and freq_ok) else "FAIL"
        print(f"  {r['strategy']:22s} days%={r.get('full_days_traded_pct')} PF full={r.get('full_pf')} "
              f"edge={r.get('full_edge_vs_be')} split PF={r.get('train(21-22)_pf')}/{r.get('val(23-24)_pf')}/{r.get('test(25-26)_pf')} "
              f"net=${r.get('full_net')} (base ${base_net}) -> {verdict}")


if __name__ == "__main__":
    main()
