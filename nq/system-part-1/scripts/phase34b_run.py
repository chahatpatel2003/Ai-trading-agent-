"""Phase 34B — retail high-win-rate hypothesis-stack lab.

Test COMPLETE retail hypotheses (context + trap/failure + confirmation + target-path + risk control),
not isolated labels. Goal: high win rate with REAL edge over the breakeven win rate (the geometry).
Reuses the audited engine; conservative fills; EOD-flat; all years. No optimization beyond the
declared grid; the headline honesty metric is edge_vs_breakeven (win rate minus the avg-win/avg-loss
breakeven), which strips out "fake" high win rate from tiny targets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nqlab.backtest.engine import run_backtest
from nqlab.config.loader import load_config, load_topstep_rules
from nqlab.discovery import phase29 as P29
from nqlab.discovery import phase32 as P32
from nqlab.domain.strategy.base import ExecConfig, StrategyParams, attach_signals

OUT = Path("docs/phase34b")
OUT.mkdir(parents=True, exist_ok=True)
_C = pl.col
_F = pl.lit(value=False)  # noqa: FBT003
ET = __import__("zoneinfo").ZoneInfo("America/New_York")
from datetime import timedelta  # noqa: E402

# default retail bans (Phase 32): skip first 15 min, last hour, low-vol quintiles
BANS = (_C("minute_of_day") >= 585) & (_C("minute_of_day") < 900) & (_C("atr_bucket") >= 2)


@dataclass(frozen=True)
class Stack:
    name: str
    hypothesis: str
    long: pl.Expr
    short: pl.Expr


def stacks() -> list[Stack]:
    up = _C("close") > _C("close").shift(1).over("trading_day")
    dn = _C("close") < _C("close").shift(1).over("trading_day")
    clean_up = _C("vp_dist_hvn_up_atr").is_null() | (_C("vp_dist_hvn_up_atr") > 1.0)
    clean_dn = _C("vp_dist_hvn_dn_atr").is_null() | (_C("vp_dist_hvn_dn_atr") > 1.0)
    return [
        # A. Failed breakout trap (fade the failed break of an obvious level)
        Stack("failed_pdh_trap", "break PDH, fail back inside -> short to value",
              (_C("low") < _C("prev_day_low")) & (_C("close") > _C("prev_day_low")) & up,
              (_C("high") > _C("prev_day_high")) & (_C("close") < _C("prev_day_high")) & dn),
        Stack("failed_onh_trap", "break overnight high/low, fail -> fade",
              (_C("low") < _C("overnight_low")) & (_C("close") > _C("overnight_low")) & up,
              (_C("high") > _C("overnight_high")) & (_C("close") < _C("overnight_high")) & dn),
        # B. Breakout continuation with acceptance
        Stack("breakout_acceptance", "accept beyond VAH/VAL + dev POC with trend -> continue",
              (_C("vp_vah_break_up") == 1) & (_C("vp_pos") > 0) & (_C("vp_dev_poc_shift") > 0),
              (_C("vp_val_break_dn") == 1) & (_C("vp_pos") < 0) & (_C("vp_dev_poc_shift") < 0)),
        # C. Stop-hunt into value reversion (= swd-style)
        Stack("stophunt_reversion", "sweep swing + reclaim, above/below POC, clean path -> to POC",
              (_C("swept_low") == 1) & (_C("reclaim_up") == 1) & (_C("vp_d_poc_atr") < -0.5) & clean_up,
              (_C("swept_high") == 1) & (_C("reclaim_dn") == 1) & (_C("vp_d_poc_atr") > 0.5) & clean_dn),
        # D. VWAP rubber band (stretch + failure -> snap to VWAP). dist_vwap_atr = vwd/atr
        Stack("vwap_rubber_band", "stretched >1 ATR from VWAP near edge + reclaim -> to VWAP",
              ((_C("vwd") / _C("atr")) < -1.0) & (_C("vp_pos") <= 0) & up,
              ((_C("vwd") / _C("atr")) > 1.0) & (_C("vp_pos") >= 0) & dn),
        # E. POC magnet reversion (above POC + VAH reject -> to POC)
        Stack("poc_magnet", "extended past value + edge rejection, clean path -> to POC",
              (_C("vp_d_poc_atr") < -1.0) & (_C("vp_val_reclaim_up") == 1) & clean_up,
              (_C("vp_d_poc_atr") > 1.0) & (_C("vp_vah_reject_dn") == 1) & clean_dn),
        # F. Opening range trap (failed OR breakout back inside)
        Stack("or_trap", "break OR edge, fail back inside -> to OR mid/VWAP",
              (_C("minute_of_day") >= 600) & (_C("low") < _C("or_low")) & (_C("close") > _C("or_low")),
              (_C("minute_of_day") >= 600) & (_C("high") > _C("or_high")) & (_C("close") < _C("or_high"))),
        # G. Gap trap (gap fails to hold vs VWAP -> fill toward VWAP/POC)
        Stack("gap_trap", "gap fails to hold above/below VWAP -> revert",
              (_C("gap_dir") == -1) & (_C("close") > _C("vwap_session")) & (_C("minute_of_day") < 690),
              (_C("gap_dir") == 1) & (_C("close") < _C("vwap_session")) & (_C("minute_of_day") < 690)),
        # H. Trend pullback to VWAP (continuation)
        Stack("trend_pullback", "1H trend + pullback holds VWAP -> continuation",
              (_C("trend_1h_bull") == 1) & (_C("close") > _C("vwap_session")) & (_C("low") <= _C("vwap_session")) & up,
              (_C("trend_1h_bull") == 0) & (_C("close") < _C("vwap_session")) & (_C("high") >= _C("vwap_session")) & dn),
        # I. Range rotation (range day, fade value edge to POC)
        Stack("range_rotation", "range day + value-edge rejection -> to POC/mid",
              (_C("prev_day_type") == "range") & (_C("vp_val_reclaim_up") == 1),
              (_C("prev_day_type") == "range") & (_C("vp_vah_reject_dn") == 1)),
    ]


def _target_expr(ttype: str, stop: pl.Expr, long: pl.Expr, short: pl.Expr) -> pl.Expr:
    atr, close = _C("atr"), _C("close")
    clamp = lambda x: pl.min_horizontal(pl.max_horizontal(x, 0.5 * atr), 8.0 * atr)  # noqa: E731
    if ttype == "rr1":
        return stop
    if ttype.startswith("fixed"):
        return pl.lit(float(ttype[5:]))
    level = {"vwap": _C("vwap_session"), "poc": _C("vp_prior_poc"),
             "ormid": (_C("or_high") + _C("or_low")) / 2}[ttype]
    lt = pl.when(level > close).then(clamp(level - close)).otherwise(2.0 * stop)
    st = pl.when(level < close).then(clamp(close - level)).otherwise(2.0 * stop)
    return pl.when(long).then(lt).when(short).then(st).otherwise(pl.lit(100000.0))


def build(frame: pl.DataFrame, st: Stack, ttype: str, stop_atr: float,
          bans: bool = True) -> pl.DataFrame:
    valid = _C("atr").is_not_null() & (_C("atr") > 0) & _C("vp_prior_poc").is_not_null()
    gate = (BANS if bans else (_C("minute_of_day") >= 570) & (_C("minute_of_day") < 955)) & valid
    long = st.long.fill_null(value=False) & gate
    short = st.short.fill_null(value=False) & gate
    stop = stop_atr * _C("atr")
    target = _target_expr(ttype, stop, long, short)
    sig = attach_signals(frame, long=long, short=short, stop_pts=stop, target_pts=target)
    return sig.filter((_C("minute_of_day") >= 0) & (_C("minute_of_day") <= 960))


def _panel(trades: list[Any]) -> dict[str, Any]:
    sd = [t for t in trades if (t.exit_ts.astimezone(ET) + timedelta(hours=6)).date() == t.trading_day]
    n = len(sd)
    if n < 30:
        return {"n": n}
    nets = np.array([t.net_pnl for t in sd])
    wins, losses = nets[nets > 0], nets[nets < 0]
    aw = float(wins.mean()) if wins.size else 0.0
    al = float(-losses.mean()) if losses.size else 0.0
    be = al / (aw + al) if (aw + al) > 0 else None
    wr = float((nets > 0).mean())
    eq = np.cumsum(nets)
    mdd = float((np.maximum.accumulate(eq) - eq).max())
    maes = np.array([t.mae * 2 for t in sd])
    prev = np.concatenate([[0.0], eq[:-1]])
    intr = float((np.maximum.accumulate(np.maximum(eq, prev)) - (prev - maes)).max())
    yrs: dict[int, float] = {}
    months: dict[str, float] = {}
    for t in sd:
        yrs[t.trading_day.year] = yrs.get(t.trading_day.year, 0.0) + t.net_pnl
        months[f"{t.trading_day.year}-{t.trading_day.month:02d}"] = months.get(
            f"{t.trading_day.year}-{t.trading_day.month:02d}", 0.0) + t.net_pnl
    ml = cur = 0
    for x in nets:
        cur = cur + 1 if x < 0 else 0
        ml = max(ml, cur)
    return {
        "n": n, "win_rate": round(wr, 4), "breakeven_wr": round(be, 4) if be else None,
        "edge_vs_be": round(wr - be, 4) if be else None,
        "pf": round(float(wins.sum() / -losses.sum()), 3) if losses.size else None,
        "expectancy": round(float(nets.mean()), 2), "avg_win": round(aw, 2), "avg_loss": round(-al, 2),
        "net": round(float(nets.sum()), 2), "max_dd": round(mdd, 2), "intraday_dd": round(intr, 2),
        "ret_dd": round(float(nets.sum() / mdd), 2) if mdd > 0 else None,
        "monthly": round(float(nets.sum()) / 66, 2), "max_loss_streak": ml,
        "years_pos": sum(1 for v in yrs.values() if v > 0), "years_n": len(yrs),
        "best_month_share": round(max(months.values()) / float(nets.sum()), 3) if float(nets.sum()) > 0 else None,
        "yearly": {y: round(v, 0) for y, v in sorted(yrs.items())},
        "_nets": nets, "_maes": maes, "_trades": sd,
    }


def run(frame: pl.DataFrame, cfg: Any, rules: Any, st: Stack, ttype: str, stop_atr: float,
        bans: bool = True, cost_mult: float = 1.0) -> dict[str, Any]:
    sig = build(frame, st, ttype, stop_atr, bans=bans)
    if sig.filter(_C("signal") != 0).height == 0:
        return {"n": 0}
    res = run_backtest(sig, symbol="NQ.c", timeframe="5m", strategy=StrategyParams("p34b", "1"),
                       config=cfg, rules=rules, primary_cutoff=False, entry_mode="next_open",
                       gate_entries=False, exec_config=ExecConfig(sizing_mode="mnq_fixed1", time_stop_bars=10000),
                       slippage_mult=cost_mult, commission_mult=cost_mult)
    return _panel(res.trades)


def main() -> None:
    cfg = load_config(profile="dev", overrides=P29.PHASE29_OVERRIDES)
    rules = load_topstep_rules("topstep_150k")
    fr = P32.add_regime_buckets(pl.read_parquet(cfg.data.artifacts_root / "phase30" / "frame_5m_full_vp.parquet"))
    targets = ["rr1", "vwap", "poc", "ormid", "fixed5", "fixed10"]
    rows: list[dict[str, Any]] = []
    for st in stacks():
        for direction in ("long", "short"):
            stk = Stack(st.name, st.hypothesis, st.long if direction == "long" else _F,
                        st.short if direction == "short" else _F)
            for ttype in targets:
                for stop_atr in (1.5, 2.5):
                    p = run(fr, cfg, rules, stk, ttype, stop_atr)
                    if p.get("n", 0) < 80:
                        continue
                    flat = {k: (json.dumps(v) if isinstance(v, dict) else v)
                            for k, v in p.items() if not k.startswith("_")}
                    rows.append({"stack": st.name, "direction": direction, "target": ttype,
                                 "stop_atr": stop_atr, **flat})
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.sort("edge_vs_be", descending=True).write_csv(OUT / "phase34b_leaderboard.csv")
    (OUT / "_lab.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"candidates: {df.height}")
    cols = ["stack", "direction", "target", "stop_atr", "n", "win_rate", "breakeven_wr", "edge_vs_be",
            "pf", "monthly", "ret_dd", "years_pos", "best_month_share"]
    print("\n=== TOP 15 by EDGE over breakeven (the real-edge metric), n>=120 ===")
    with pl.Config(tbl_rows=17, fmt_str_lengths=20, tbl_width_chars=220):
        print(df.filter(_C("n") >= 120).sort("edge_vs_be", descending=True).select(cols).head(15))
    print("\n=== TOP 12 high win rate (>=0.58) with PF>=1.4, n>=120 ===")
    with pl.Config(tbl_rows=14, fmt_str_lengths=20, tbl_width_chars=220):
        print(df.filter((_C("n") >= 120) & (_C("win_rate") >= 0.58) & (_C("pf") >= 1.4))
              .sort("pf", descending=True).select(cols).head(12))
    hi = df.filter((_C("n") >= 120) & (_C("win_rate") >= 0.6) & (_C("pf") >= 1.4) & (_C("edge_vs_be") >= 0.05))
    print(f"\n>>> setups meeting FULL bar (WR>=60%, PF>=1.4, edge>=5pp, n>=120): {hi.height}")
    if hi.height:
        with pl.Config(tbl_rows=12, fmt_str_lengths=20, tbl_width_chars=220):
            print(hi.sort("edge_vs_be", descending=True).select(cols))


if __name__ == "__main__":
    main()
