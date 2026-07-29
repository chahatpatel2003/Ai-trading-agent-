"""Phase 38 — Daily NQ Workhorse Search: 55+ simple sleeves + portfolio combinations.

Expands Phase 37. Goal: a simple sleeve OR a portfolio of independently-tested rule-based sleeves that
trades MOST RTH days (70-95%), low drawdown, Topstep payout-viable, and beats the dumb always-long
baseline on a RISK-ADJUSTED basis (the baseline wins net $ but breaches a $3K trailing DD ~100% of the
time, so it cannot pass Topstep — that is the opening this phase probes).

Anti-bias: one trade/day per sleeve (first valid signal, no forcing), a-priori rules NO tuning,
realistic costs, MNQ. train 21-22 / val 23-24 / test 25-26 + per-year + MC trade reshuffle + Topstep
trailing + payout sim. Honest verdicts. Audited engine. News calendar (area 16) NOT available -> proxied
by a high-volatility "active day" filter and disclosed.
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

OUT = Path("docs/phase38")
OUT.mkdir(parents=True, exist_ok=True)
ET = zoneinfo.ZoneInfo("America/New_York")
_C = pl.col
_BIG = 100000.0
RNG = np.random.default_rng(38)
_RR = {"rr1": 1.0, "rr1.5": 1.5, "rr2": 2.0, "rr3": 3.0}


def _flag(name: str) -> pl.Expr:
    return _C(name).cast(pl.Int8) == 1


def _cross_up(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return (a > b) & (a.shift(1).over("trading_day") <= b.shift(1).over("trading_day"))


def _cross_dn(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return (a < b) & (a.shift(1).over("trading_day") >= b.shift(1).over("trading_day"))


def _first_per_day(sig: pl.Expr) -> pl.Expr:
    return sig & (sig.cast(pl.Int32).cum_sum().over("trading_day") == 1)


@dataclass(frozen=True)
class Strat:
    name: str
    area: str
    long: pl.Expr
    short: pl.Expr
    stop_atr: float
    exit: str


def strategies() -> list[Strat]:  # noqa: PLR0915
    md = _C("minute_of_day")
    win = lambda a, b: (md >= a) & (md < b)  # noqa: E731
    rth = win(570, 900)
    after_or = win(600, 870)
    close, high, low = _C("close"), _C("high"), _C("low")
    vwap, atr = _C("vwap_session"), _C("atr")
    up = close > close.shift(1).over("trading_day")
    dn = close < close.shift(1).over("trading_day")
    orh, orl = _C("or_high"), _C("or_low")
    pdh, pdl, pdc = _C("prev_day_high"), _C("prev_day_low"), _C("prev_day_close")
    S = []  # noqa: N806
    add = lambda *a: S.append(Strat(*a))  # noqa: E731

    # ---- Area 1: opening drive continuation ----
    add("od_cont_or", "1_open_drive_cont", after_or & _cross_up(close, orh) & _flag("displacement"),
        after_or & _cross_dn(close, orl) & _flag("displacement"), 2.0, "eod")
    add("od_cont_rexp", "1_open_drive_cont", win(600, 720) & _flag("range_expansion") & (close > orh),
        win(600, 720) & _flag("range_expansion") & (close < orl), 1.5, "rr2")
    add("od_cont_struct", "1_open_drive_cont", win(600, 750) & (_C("struct_dir") == 1) & (close > orh),
        win(600, 750) & (_C("struct_dir") == -1) & (close < orl), 2.0, "eod")
    # ---- Area 2: opening drive failure ----
    add("od_fail_sweep", "2_open_drive_fail", win(600, 780) & _flag("swept_low") & _flag("reclaim_up"),
        win(600, 780) & _flag("swept_high") & _flag("reclaim_dn"), 1.5, "rr2")
    add("od_fail_or", "2_open_drive_fail", after_or & (low < orl) & (close > orl),
        after_or & (high > orh) & (close < orh), 1.5, "rr2")
    add("od_fail_mss", "2_open_drive_fail", win(600, 780) & _flag("mss_up"),
        win(600, 780) & _flag("mss_dn"), 1.5, "rr2")
    # ---- Area 3: VWAP reclaim / rejection ----
    add("vwap_reclaim", "3_vwap", rth & _cross_up(close, vwap), rth & _cross_dn(close, vwap), 2.0, "eod")
    add("vwap_reject", "3_vwap", rth & (high >= vwap) & (close < vwap) & dn & (close < pdc),
        rth & (low <= vwap) & (close > vwap) & up & (close > pdc), 1.5, "rr2")
    add("vwap_revert", "3_vwap", rth & ((close - vwap) < -1.5 * atr) & up,
        rth & ((close - vwap) > 1.5 * atr) & dn, 1.5, "rr2")
    add("vwap_pullback_trend", "3_vwap",
        rth & (close > vwap) & (low <= vwap) & _flag("trend_1h_bull") & up,
        rth & (close < vwap) & (high >= vwap) & _flag("htf_bear") & dn, 2.0, "eod")
    # ---- Area 4: PDH/PDL sweep + reclaim ----
    add("pdhl_fade", "4_pdhl", rth & (low < pdl) & (close > pdl), rth & (high > pdh) & (close < pdh),
        1.5, "rr2")
    add("pdhl_sweep_reclaim", "4_pdhl",
        rth & (low < pdl) & (close > pdl) & _flag("reclaim_up"),
        rth & (high > pdh) & (close < pdh) & _flag("reclaim_dn"), 1.5, "rr2")
    add("pdhl_break_cont", "4_pdhl", after_or & _cross_up(close, pdh), after_or & _cross_dn(close, pdl),
        2.0, "eod")
    # ---- Area 5: Asia/London/overnight sweep into NY ----
    onh, onl = _C("overnight_high"), _C("overnight_low")
    add("on_sweep_fade", "5_session_sweep", rth & (low < onl) & (close > onl),
        rth & (high > onh) & (close < onh), 1.5, "rr2")
    add("asia_sweep_fade", "5_session_sweep", rth & (low < _C("asia_low")) & (close > _C("asia_low")),
        rth & (high > _C("asia_high")) & (close < _C("asia_high")), 1.5, "rr2")
    add("london_sweep_fade", "5_session_sweep",
        rth & (low < _C("london_low")) & (close > _C("london_low")),
        rth & (high > _C("london_high")) & (close < _C("london_high")), 1.5, "rr2")
    add("on_break_cont", "5_session_sweep", after_or & _cross_up(close, onh), after_or & _cross_dn(close, onl),
        2.0, "eod")
    # ---- Area 6: first pullback after displacement ----
    add("disp_pb_fvg", "6_disp_pullback", rth & _flag("displacement") & _flag("in_bull_fvg") & up,
        rth & _flag("displacement") & _flag("in_bear_fvg") & dn, 1.5, "rr2")
    add("disp_pb_ema9", "6_disp_pullback",
        win(600, 810) & (_C("struct_dir") == 1) & (low <= _C("ema9")) & (close > _C("ema9")),
        win(600, 810) & (_C("struct_dir") == -1) & (high >= _C("ema9")) & (close < _C("ema9")), 1.5, "rr2")
    add("cisd_cont", "6_disp_pullback", win(600, 810) & _flag("cisd_up"), win(600, 810) & _flag("cisd_dn"),
        1.5, "rr2")
    # ---- Area 7: trend day continuation ----
    add("trend_adx_cont", "7_trend_cont", rth & (_C("adx") > 25) & _flag("trend_1h_bull") & (close > vwap) & up,
        rth & (_C("adx") > 25) & _flag("htf_bear") & (close < vwap) & dn, 2.0, "eod")
    add("trend_htf_orb", "7_trend_cont", after_or & _flag("htf_bull") & (close > orh),
        after_or & _flag("htf_bear") & (close < orl), 2.0, "eod")
    add("bos_cont", "7_trend_cont", win(600, 810) & _flag("bos_up"), win(600, 810) & _flag("bos_dn"),
        2.0, "eod")
    add("aligned_cont", "7_trend_cont", after_or & _flag("aligned_long_a") & (close > orh),
        after_or & _flag("aligned_short_a") & (close < orl), 2.0, "eod")
    # ---- Area 8: range day mean reversion ----
    add("range_or_fade", "8_range_mr", after_or & (_C("vol_regime") == 0) & (high > orh) & (close < orh),
        after_or & (_C("vol_regime") == 0) & (low < orl) & (close > orl), 1.5, "rr1.5")
    add("bb_fade", "8_range_mr", rth & (_C("adx") < 20) & (close < _C("_bb_dn")) & up,
        rth & (_C("adx") < 20) & (close > _C("_bb_up")) & dn, 1.5, "rr2")
    add("nr_day_revert", "8_range_mr", rth & _flag("prev_nr_day") & ((close - vwap) < -1.0 * atr) & up,
        rth & _flag("prev_nr_day") & ((close - vwap) > 1.0 * atr) & dn, 1.5, "rr2")
    add("rsi_revert", "8_range_mr", rth & (_C("adx") < 22) & (_C("rsi") < 30) & up,
        rth & (_C("adx") < 22) & (_C("rsi") > 70) & dn, 1.5, "rr2")
    # ---- Area 9: gap fill / gap continuation ----
    add("gap_fill", "9_gap", win(570, 720) & (_C("gap_dir") == -1) & (_flag("gap_filled").not_()),
        win(570, 720) & (_C("gap_dir") == 1) & (_flag("gap_filled").not_()), 1.5, "rr2")
    add("gap_go", "9_gap", after_or & (_C("gap_dir") == 1) & (close > orh),
        after_or & (_C("gap_dir") == -1) & (close < orl), 2.0, "eod")
    add("gap_fade_pdc", "9_gap", win(570, 780) & (_C("gap_dir") == -1) & (close < pdc) & up,
        win(570, 780) & (_C("gap_dir") == 1) & (close > pdc) & dn, 1.5, "rr2")
    # ---- Area 10: settlement / prev close reclaim ----
    add("pdc_reclaim", "10_pdc", rth & _cross_up(close, pdc), rth & _cross_dn(close, pdc), 2.0, "eod")
    add("pdc_reject", "10_pdc", rth & (low <= pdc) & (close > pdc) & up & (close < vwap),
        rth & (high >= pdc) & (close < pdc) & dn & (close > vwap), 1.5, "rr2")
    # ---- Area 11: POC / value-area reversion ----
    add("vah_reject", "11_value_area", rth & _flag("vp_val_reclaim_up"), rth & _flag("vp_vah_reject_dn"),
        1.5, "rr2")
    add("va_revert", "11_value_area", rth & (_C("vp_pos") == 1) & ((close - vwap) < 0) & up,
        rth & (_C("vp_pos") == -1) & ((close - vwap) > 0) & dn, 1.5, "rr2")
    add("poc_reclaim_cont", "11_value_area", after_or & _flag("vp_poc_reclaim_up"),
        after_or & _flag("vp_poc_reclaim_dn"), 2.0, "eod")
    add("va_break_cont", "11_value_area", after_or & _flag("vp_vah_break_up"),
        after_or & _flag("vp_val_break_dn"), 2.0, "eod")
    add("lvn_reject", "11_value_area", rth & _flag("vp_lvn_reject_up"), rth & _flag("vp_lvn_reject_dn"),
        1.5, "rr2")
    # ---- Area 12: high-volatility breakout ----
    add("volhi_orb", "12_vol_breakout", after_or & (_C("vol_regime") == 1) & (close > orh),
        after_or & (_C("vol_regime") == 1) & (close < orl), 2.0, "eod")
    add("rexp_rollbreak", "12_vol_breakout", after_or & _flag("range_expansion") & _cross_up(close, _C("roll_high")),
        after_or & _flag("range_expansion") & _cross_dn(close, _C("roll_low")), 1.5, "rr2")
    # ---- Area 13: low-volatility chop / compression breakout ----
    add("comp_break", "13_compression", after_or & _flag("compression_setup") & (close > orh),
        after_or & _flag("compression_setup") & (close < orl), 2.0, "eod")
    # ---- Area 14: time-of-day ----
    add("tod_pm_revert", "14_time_of_day", win(780, 840) & ((close - vwap) < -1.0 * atr) & up,
        win(780, 840) & ((close - vwap) > 1.0 * atr) & dn, 1.5, "rr2")
    add("tod_lunch_fade", "14_time_of_day", win(660, 750) & ((close - vwap) < -1.2 * atr) & up,
        win(660, 750) & ((close - vwap) > 1.2 * atr) & dn, 1.5, "rr2")
    add("tod_power_hour", "14_time_of_day", win(840, 895) & (close > vwap) & up & _flag("htf_bull"),
        win(840, 895) & (close < vwap) & dn & _flag("htf_bear"), 2.0, "eod")
    # ---- Area 15: long-only beta-capture filters ----
    add("beta_long_above_vwap", "15_beta", after_or & (close > vwap), pl.lit(value=False), 5.0, "eod")
    add("beta_long_uptrend", "15_beta", rth & _flag("htf_bull"), pl.lit(value=False), 5.0, "eod")
    add("beta_long_above_pdc", "15_beta", rth & (close > pdc), pl.lit(value=False), 5.0, "eod")
    add("beta_long_gapup", "15_beta", rth & (_C("gap_dir") >= 0), pl.lit(value=False), 5.0, "eod")
    add("beta_long_trend1h", "15_beta", after_or & _flag("trend_1h_bull") & (close > orh),
        pl.lit(value=False), 3.0, "eod")
    # ---- Area 16: news proxy (NO calendar) -> high-activity day filter ----
    add("active_long_volhi", "16_news_proxy", rth & (_C("vol_regime") == 1) & (close > vwap),
        pl.lit(value=False), 3.0, "eod")
    add("calm_long_vollo", "16_news_proxy", rth & (_C("vol_regime") == 0) & (close > vwap),
        pl.lit(value=False), 3.0, "eod")
    # ---- Area 17: ATR expansion / contraction regimes ----
    add("atr_exp_orb", "17_atr_regime", after_or & (_C("vol_regime_ratio") > 1.1) & (close > orh),
        after_or & (_C("vol_regime_ratio") > 1.1) & (close < orl), 2.0, "eod")
    add("atr_contract_revert", "17_atr_regime",
        rth & (_C("vol_regime_ratio") < 0.9) & ((close - vwap) < -1.0 * atr) & up,
        rth & (_C("vol_regime_ratio") < 0.9) & ((close - vwap) > 1.0 * atr) & dn, 1.5, "rr2")
    # ---- Area 18: day-of-week (long beta by weekday) ----
    for k, nm in ((1, "mon"), (2, "tue"), (3, "wed"), (4, "thu"), (5, "fri")):
        add(f"dow_long_{nm}", "18_day_of_week", after_or & (_C("dow") == k), pl.lit(value=False), 5.0, "eod")
    # ---- Area 19: first 30/60/90-min range break ----
    add("ir30_break", "19_intraday_range", after_or & _cross_up(close, orh), after_or & _cross_dn(close, orl),
        2.0, "eod")
    add("ir60_break", "19_intraday_range",
        win(630, 870) & _cross_up(close, _C("ir60_high")), win(630, 870) & _cross_dn(close, _C("ir60_low")),
        2.0, "eod")
    add("ir90_break", "19_intraday_range",
        win(660, 870) & _cross_up(close, _C("ir90_high")), win(660, 870) & _cross_dn(close, _C("ir90_low")),
        2.0, "eod")
    add("ir60_fade", "19_intraday_range",
        win(630, 840) & (high > _C("ir60_high")) & (close < _C("ir60_high")),
        win(630, 840) & (low < _C("ir60_low")) & (close > _C("ir60_low")), 1.5, "rr2")
    # ---- baselines ----
    add("BASELINE_long", "0_baseline", win(570, 955), pl.lit(value=False), 5.0, "eod")
    add("BASELINE_short", "0_baseline", pl.lit(value=False), win(570, 955), 5.0, "eod")
    return S


def add_cols(fr: pl.DataFrame) -> pl.DataFrame:
    md = _C("minute_of_day")
    inw = lambda hi: _C("is_rth") & (md < hi)  # noqa: E731
    return fr.sort(["trading_day", "minute_of_day"]).with_columns(
        pl.when(inw(630)).then(_C("high")).max().over("trading_day").alias("ir60_high"),
        pl.when(inw(630)).then(_C("low")).min().over("trading_day").alias("ir60_low"),
        pl.when(inw(660)).then(_C("high")).max().over("trading_day").alias("ir90_high"),
        pl.when(inw(660)).then(_C("low")).min().over("trading_day").alias("ir90_low"),
    )


def build(fr: pl.DataFrame, s: Strat) -> pl.DataFrame:
    valid = _C("atr").is_not_null() & (_C("atr") > 0)
    long = _first_per_day(s.long.fill_null(value=False) & valid)
    short = _first_per_day(s.short.fill_null(value=False) & valid)
    stop = s.stop_atr * _C("atr")
    target = pl.lit(_BIG) if s.exit == "eod" else (_RR[s.exit] * stop)
    sig = attach_signals(fr, long=long, short=short, stop_pts=stop, target_pts=target)
    return sig.filter((_C("minute_of_day") >= 0) & (_C("minute_of_day") <= 960))


def _exit_td(ts: Any) -> Any:
    return (ts.astimezone(ET) + timedelta(hours=6)).date()


def _daily_sharpe(sd: list[Any]) -> float | None:
    by_day: dict[Any, float] = {}
    for t in sd:
        by_day[t.trading_day] = by_day.get(t.trading_day, 0.0) + t.net_pnl
    v = np.array(list(by_day.values()))
    if v.size < 20 or v.std() == 0:
        return None
    return round(float(v.mean() / v.std() * np.sqrt(252)), 2)


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
        "sharpe": _daily_sharpe(sd),
    }


def _mc_breach(nets: np.ndarray, mae_usd: np.ndarray, limit: float, sims: int = 1500) -> float:
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


def _payout_mc(daily: np.ndarray, target: float, mll: float, sims: int = 3000) -> dict[str, float]:
    """Block-bootstrap daily PnL; P(reach +target cumulative before trailing DD breaches mll)."""
    if daily.size < 20:
        return {"payout": 0.0, "blow": 1.0}
    pay = blow = 0
    horizon = min(daily.size, 120)
    for _ in range(sims):
        idx = RNG.integers(0, daily.size, horizon)
        eq = np.cumsum(daily[idx])
        peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[:-1]
        floor = np.minimum(peak - mll, 0.0)  # trailing, locked at start (start floor = -mll)
        hit_pay = np.argmax(eq >= target) if (eq >= target).any() else -1
        hit_blow = np.argmax(eq <= floor) if (eq <= floor).any() else -1
        if hit_pay >= 0 and (hit_blow < 0 or hit_pay <= hit_blow):
            pay += 1
        elif hit_blow >= 0:
            blow += 1
    return {"payout": round(pay / sims, 3), "blow": round(blow / sims, 3)}


def main() -> None:  # noqa: PLR0915
    cfg = load_config(profile="dev", overrides=P29.PHASE29_OVERRIDES)
    rules = load_topstep_rules("topstep_150k")
    fr = add_cols(pl.read_parquet(cfg.data.artifacts_root / "phase30" / "frame_5m_full_vp.parquet"))
    days_by_year: dict[int, int] = {}
    for y, sub in fr.filter(_C("is_rth")).group_by(_C("trading_day").dt.year()):
        days_by_year[int(y[0] if isinstance(y, tuple) else y)] = sub["trading_day"].n_unique()
    total_days = sum(days_by_year.values())
    splits = {"train": (2021, 2022), "val": (2023, 2024), "test": (2025, 2026)}

    rows: list[dict[str, Any]] = []
    trade_store: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    strats = strategies()
    print(f"sleeves: {len(strats)} | RTH days: {days_by_year} total={total_days}")
    for s in strats:
        try:
            res = run_backtest(build(fr, s), symbol="NQ.c", timeframe="5m",
                               strategy=StrategyParams("p38", "1"), config=cfg, rules=rules,
                               primary_cutoff=False, entry_mode="next_open", gate_entries=False,
                               exec_config=ExecConfig(sizing_mode="mnq_fixed1", time_stop_bars=10000))
        except Exception as e:  # noqa: BLE001
            errors[s.name] = repr(e)[:160]
            print(f"  ERR {s.name}: {errors[s.name]}")
            continue
        allt = [t for t in res.trades if _exit_td(t.exit_ts) == t.trading_day]
        full = _panel(allt, total_days)
        row: dict[str, Any] = {"strategy": s.name, "area": s.area, "exit": s.exit, "stop_atr": s.stop_atr,
                               **{f"full_{k}": v for k, v in full.items()}}
        for sp, (y0, y1) in splits.items():
            p = _panel([t for t in allt if y0 <= t.trading_day.year <= y1],
                       sum(v for y, v in days_by_year.items() if y0 <= y <= y1))
            row[f"{sp}_pf"], row[f"{sp}_net"] = p.get("pf"), p.get("net")
        row["yearly_net"] = json.dumps({y: round(sum(t.net_pnl for t in allt if t.trading_day.year == y), 0)
                                        for y in range(2021, 2027)})
        nets = np.array([t.net_pnl for t in allt]) if allt else np.array([])
        maes = np.array([t.mae * 2 for t in allt]) if allt else np.array([])
        row["mc_breach_3k"] = _mc_breach(nets, maes, 3000) if allt else None
        row["mc_breach_4500"] = _mc_breach(nets, maes, 4500) if allt else None
        rows.append(row)
        trade_store[s.name] = [{"day": str(t.trading_day), "ts": t.entry_ts.isoformat(),
                                "net": float(t.net_pnl), "mae": float(t.mae * 2),
                                "side": int(t.side)} for t in allt]
        print(f"  {s.name:24s} n={full.get('n',0):5d} days%={full.get('days_traded_pct')} "
              f"pf={full.get('pf')} edge={full.get('edge_vs_be')} net={full.get('net')} "
              f"retDD={full.get('ret_dd')} br3k={row['mc_breach_3k']}")

    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(OUT / "phase38_sleeves.csv")
    (OUT / "_sleeves.json").write_text(json.dumps({"rows": rows, "errors": errors,
                                                   "days_by_year": days_by_year}, indent=2, default=str))
    (OUT / "_trades.json").write_text(json.dumps(trade_store, default=str))
    print(f"\nerrors: {len(errors)}  ->  {list(errors)}")
    print("done sleeves; portfolio step in phase38_portfolio.py")


if __name__ == "__main__":
    main()
