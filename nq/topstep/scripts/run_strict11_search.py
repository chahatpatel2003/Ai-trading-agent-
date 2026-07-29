#!/usr/bin/env python
"""Strict RR>=1:1 high-win-rate discovery pass (failure-diagnosis oriented).

PRE-REGISTERED protocol (fixed a priori, nothing fitted inside folds):
  - Signal battery: the SAME audited quantile families as Phase 5C/GC passes
    (exec_micro_grids / gc_exec_grids signal bases), but every rule's
    geometry is forced to r_mult in {1.0, 1.5} — target >= stop everywhere.
    Time stops scale with the target: ts' = round(ts_base * (0.5 + r)).
  - Era split: NQ/MNQ search 2021-01-05..2024-12-31 (2025 tail untouched);
    GC/MGC search 2016..2024-12-31 (2025 tail untouched).
  - Walk-forward: session folds (NQ 250/63/1, GC 250/126/1), TRAIN-only
    selection, two arms reported:
      gated: train win>=0.55 AND train net>0 AND train trades>=200,
             ranked by train t-stat  (encodes the high-win-rate mandate)
      plain: train net>0 AND train trades>=200, ranked by train t-stat
  - HARD GATES on each OOS WF stream: win>=0.55 (pref 0.60), tps>=1.0
    (pref 2-8), net>0, chain-MC EV/month>0, blowout<0.50, pos folds>=half,
    zero violations, audits pass.
  - Failure-diagnosis outputs: per-fold max attainable TRAIN win rate,
    full candidate summary, WF stream trades AND all-candidate OOS trades
    parquets (exit reasons, MAE, risk, costs per trade).

Usage: run_strict11_search.py --market {nq,gc} [--limit-folds N]
       [--no-audits] [--max-minutes M]   (exit 3 = partial, resumable)
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time as _time
from dataclasses import asdict
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab.audit.leakage import (CheckResult, FoldEmbargoCheck,   # noqa: E402
                                 LeakageAuditor, Status,
                                 TruncationInvarianceCheck)
from tslab.config import load_config                              # noqa: E402
from tslab.features.core import FeatureParams                     # noqa: E402
from tslab.features.labels import (LabelParams, check_label_purge,  # noqa: E402
                                   compute_labels, make_session_folds,
                                   required_embargo)
from tslab.features.quantiles import (compute_gc_q_features,      # noqa: E402
                                      compute_micro_q_features)
from tslab.sim.costs import CostModel                             # noqa: E402
from tslab.strategy.backtest import run_backtest                  # noqa: E402
from tslab.strategy.evaluate import lifecycle_mc_chain, max_drawdown  # noqa: E402
from tslab.strategy.full_search import train_tstat                # noqa: E402
from tslab.strategy.rules import Filter, TradeRule                # noqa: E402

SEARCH_END = date(2024, 12, 31)
TAIL_START = date(2025, 1, 1)
GATES = {"win": 0.55, "win_pref": 0.60, "tps": 1.0, "tps_pref_lo": 2.0,
         "tps_pref_hi": 8.0, "blowout": 0.50}
MIN_TRAIN_TRADES = 200
MIN_TRAIN_WIN = 0.55
R_MULTS = (1.0, 1.5)                     # STRICT: target >= stop, everywhere


def on(col):
    return Filter(col, "eq", 1.0)


# --------------------------------------------------------------------------- #
# grids: same signal bases as exec_micro_grids()/gc_exec_grids(), RR forced
# --------------------------------------------------------------------------- #

def nq_strict_grids():
    """(nq_grid, mnq_grid) — 22 signal bases x geometry x r in R_MULTS."""
    liq = Filter("f_spread_ratio", "lt_or_na", 1.6)
    W = (time(8, 35), time(14, 15))
    FAMS = (
        ("qimbR_S", -1, [on("fq_m_imb_z_hi10")], 25),
        ("qimbR_L", +1, [on("fq_m_imb_z_lo10")], 25),
        ("qimbRV_S", -1, [on("fq_m_imb_z_hi10"),
                          on("fq_svwap_dist_atr_hi20")], 40),
        ("qimbRV_L", +1, [on("fq_m_imb_z_lo10"),
                          on("fq_svwap_dist_atr_lo20")], 40),
        ("qimbC_L", +1, [on("fq_m_imb_z_hi20"),
                         Filter("f_ret_5", "gt", 0.0)], 25),
        ("qimbC_S", -1, [on("fq_m_imb_z_lo20"),
                         Filter("f_ret_5", "lt", 0.0)], 25),
        ("qqbC_L", +1, [on("fq_m_qburst_ratio_hi10_tod"),
                        on("fq_m_mom_hi20")], 20),
        ("qqbC_S", -1, [on("fq_m_qburst_ratio_hi10_tod"),
                        on("fq_m_mom_lo20")], 20),
        ("qqbX_S", -1, [on("fq_m_qburst_ratio_hi10"),
                        on("fq_svwap_dist_atr_hi20")], 25),
        ("qqbX_L", +1, [on("fq_m_qburst_ratio_hi10"),
                        on("fq_svwap_dist_atr_lo20")], 25),
        ("qdbF_S", -1, [on("fq_m_dburst_ratio_hi10"), on("fq_m_mr_hi10")], 25),
        ("qdbF_L", +1, [on("fq_m_dburst_ratio_hi10"), on("fq_m_mr_lo10")], 25),
        ("qabsB_L", +1, [on("fq_m_absorb_hi20"),
                         Filter("f_m_imb", "gt", 0.1)], 25),
        ("qabsB_S", -1, [on("fq_m_absorb_hi20"),
                         Filter("f_m_imb", "lt", -0.1)], 25),
        ("qfpR_S", -1, [Filter("f_m_failed_up", "eq", 1.0),
                        on("fq_m_press_z_lo10")], 25),
        ("qfpR_L", +1, [Filter("f_m_failed_dn", "eq", 1.0),
                        on("fq_m_press_z_hi10")], 25),
        ("qswp_L", +1, [Filter("f_m_sweep_lo", "eq", 1.0),
                        Filter("f_m_imb", "gt", 0.0)], 30),
        ("qswp_S", -1, [Filter("f_m_sweep_hi", "eq", 1.0),
                        Filter("f_m_imb", "lt", 0.0)], 30),
        ("qmr_S", -1, [on("fq_m_mr_hi10")], 20),
        ("qmr_L", +1, [on("fq_m_mr_lo10")], 20),
        ("qmom_L", +1, [on("fq_m_mom_hi20"), on("fq_vol_ratio_hi20"),
                        on("fq_m_dburst_ratio_hi20")], 20),
        ("qmom_S", -1, [on("fq_m_mom_lo20"), on("fq_vol_ratio_hi20"),
                        on("fq_m_dburst_ratio_hi20")], 20),
    )
    GEO = {"NQ": (("a", 4.0, 25.0, 2, 1), ("b", 5.0, 40.0, 1, 1)),
           "MNQ": (("a", 3.0, 40.0, 4, 4), ("b", 4.0, 60.0, 2, 4))}

    def build(instrument):
        tv = 5.0 if instrument == "NQ" else 0.5
        out = []
        for base, direction, sig, ts in FAMS:
            for tag, stop, band_hi, mt, qty in GEO[instrument]:
                worst = stop * band_hi * tv * qty
                assert worst <= 1000.0 / mt + 1e-9, (base, instrument, tag)
                for r in R_MULTS:
                    out.append(TradeRule(
                        name=f"{base}_r{r}_{instrument.lower()}{tag}",
                        direction=direction,
                        filters=tuple(sig + [Filter("f_atr_ticks", "between",
                                                    4.0, band_hi), liq]),
                        entry_start_ct=W[0], entry_end_ct=W[1],
                        stop_atr=stop, target_atr=stop * r,
                        time_stop_bars=int(round(ts * (0.5 + r))),
                        max_trades_per_day=mt,
                        daily_loss_usd=1000.0 - worst, qty=qty))
        return tuple(out)

    return build("NQ"), build("MNQ")


def gc_strict_grids():
    """(gc_grid, mgc_grid) — 20 signal bases x geometry x r in R_MULTS."""
    liq = Filter("f_spread_ratio", "lt_or_na", 2.0)
    W = (time(7, 30), time(13, 0))
    OV = (time(7, 30), time(10, 30))
    LATE = (time(10, 30), time(13, 0))
    FAMS = (
        ("gvf20_S", -1, [on("fq_svwap_dist_atr_hi20")], 30, W),
        ("gvf20_L", +1, [on("fq_svwap_dist_atr_lo20")], 30, W),
        ("gvf10_S", -1, [on("fq_svwap_dist_atr_hi10")], 30, W),
        ("gvf10_L", +1, [on("fq_svwap_dist_atr_lo10")], 30, W),
        ("gz30_S", -1, [on("fq_z_30_hi10")], 25, W),
        ("gz30_L", +1, [on("fq_z_30_lo10")], 25, W),
        ("gz120_S", -1, [on("fq_z_120_hi10")], 40, W),
        ("gz120_L", +1, [on("fq_z_120_lo10")], 40, W),
        ("gpdlF_L", +1, [Filter("f_prior_low_dist", "between", -0.8, 0.3),
                         Filter("f_ret_5", "gt", 0.0)], 40, W),
        ("gpdhF_S", -1, [Filter("f_prior_high_dist", "between", -0.3, 0.8),
                         Filter("f_ret_5", "lt", 0.0)], 40, W),
        ("gpdhH_L", +1, [Filter("f_prior_high_dist", "between", 0.0, 1.5),
                         on("fq_ret_15_hi20")], 60, W),
        ("gpdlH_S", -1, [Filter("f_prior_low_dist", "between", -1.5, 0.0),
                         on("fq_ret_15_lo20")], 60, W),
        ("gorb_L", +1, [Filter("f_or_break_up", "eq", 1.0),
                        on("fq_vol_ratio_hi20")], 60, W),
        ("gorb_S", -1, [Filter("f_or_break_dn", "eq", 1.0),
                        on("fq_vol_ratio_hi20")], 60, W),
        ("gmom_L", +1, [on("fq_ret_15_hi20"), on("fq_vol_ratio_hi20")], 40, W),
        ("gmom_S", -1, [on("fq_ret_15_lo20"), on("fq_vol_ratio_hi20")], 40, W),
        ("gquiet_S", -1, [on("fq_range_atr_lo20"), on("fq_z_30_hi10")],
         25, LATE),
        ("gquiet_L", +1, [on("fq_range_atr_lo20"), on("fq_z_30_lo10")],
         25, LATE),
        ("govf_S", -1, [on("fq_svwap_dist_atr_hi20")], 30, OV),
        ("govf_L", +1, [on("fq_svwap_dist_atr_lo20")], 30, OV),
    )
    GEO = {"GC": (("a", 3.0, 16.0, 2, 1), ("b", 4.0, 25.0, 1, 1)),
           "MGC": (("a", 4.0, 25.0, 4, 2), ("b", 5.0, 40.0, 2, 2),
                   ("c", 3.0, 15.0, 4, 5))}

    def build(instrument):
        tv = 10.0 if instrument == "GC" else 1.0
        out = []
        for base, direction, sig, ts, win in FAMS:
            for tag, stop, band_hi, mt, qty in GEO[instrument]:
                worst = stop * band_hi * tv * qty
                assert worst <= 1000.0 / mt + 1e-9, (base, instrument, tag)
                for r in R_MULTS:
                    out.append(TradeRule(
                        name=f"{base}_r{r}_{instrument.lower()}{tag}",
                        direction=direction,
                        filters=tuple(sig + [Filter("f_atr_ticks", "between",
                                                    3.0, band_hi), liq]),
                        entry_start_ct=win[0], entry_end_ct=win[1],
                        stop_atr=stop, target_atr=stop * r,
                        time_stop_bars=int(round(ts * (0.5 + r))),
                        max_trades_per_day=mt,
                        daily_loss_usd=1000.0 - worst, qty=qty))
        return tuple(out)

    return build("GC"), build("MGC")


# --------------------------------------------------------------------------- #
# fold evaluation with train win rate (needed by the gated arm)
# --------------------------------------------------------------------------- #

def evaluate_fold_ext(grid, bars, feats, fold, cfg, cm, instrument):
    tr_m = np.asarray((bars.index >= fold["train_start"])
                      & (bars.index < fold["train_end"]))
    te_m = np.asarray((bars.index >= fold["test_start"])
                      & (bars.index < fold["test_end"]))
    btr, ftr = bars[tr_m], feats[tr_m]
    bte, fte = bars[te_m], feats[te_m]
    tr_sessions = sorted(set(btr["session"]))
    assert not set(tr_sessions) & set(bte["session"])   # hard leakage guard
    out = {}
    for rule in grid:
        r_tr = run_backtest(rule, btr, ftr, cfg, cm, instrument=instrument)
        r_te = run_backtest(rule, bte, fte, cfg, cm, instrument=instrument)
        assert r_tr.violations == 0 and r_te.violations == 0
        out[rule.name] = {
            "train_net": r_tr.total_net,
            "train_trades": len(r_tr.trades),
            "train_wins": sum(1 for t in r_tr.trades if t.net_usd > 0),
            "train_tstat": train_tstat(r_tr.daily_net(), len(tr_sessions)),
            "test_trades": r_te.trades,
        }
    return out


def select_arm(res, gated: bool):
    best, best_key = None, (float("-inf"), float("-inf"))
    for name, rec in res.items():
        if rec["train_trades"] < MIN_TRAIN_TRADES or rec["train_net"] <= 0:
            continue
        if gated and rec["train_wins"] / rec["train_trades"] < MIN_TRAIN_WIN:
            continue
        key = (rec["train_tstat"], rec["train_net"])
        if key > best_key:
            best, best_key = name, key
    return best


def stream_stats(wf_trades, wf_sessions, fold_nets):
    all_tr = [t for s in wf_sessions for t in wf_trades.get(s, [])]
    n = len(all_tr)
    wins = sum(1 for t in all_tr if t.net_usd > 0)
    daily = pd.Series({s: sum(t.net_usd for t in wf_trades.get(s, []))
                       for s in wf_sessions}, dtype=float).sort_index()
    return all_tr, daily, {
        "net": float(daily.sum()), "trades": n, "sessions": len(wf_sessions),
        "tps": n / max(len(wf_sessions), 1),
        "win": wins / n if n else float("nan"),
        "gross": float(sum(t.gross_usd for t in all_tr)),
        "costs": float(sum(t.costs_usd for t in all_tr)),
        "avg_win": (float(np.mean([t.net_usd for t in all_tr
                                   if t.net_usd > 0])) if wins else 0.0),
        "avg_loss": (float(np.mean([t.net_usd for t in all_tr
                                    if t.net_usd <= 0]))
                     if n > wins else 0.0),
        "max_dd": max_drawdown(daily),
        "pos_folds": int(sum(1 for x in fold_nets if x > 0)),
        "exit_mix": {r: sum(1 for t in all_tr if t.exit_reason == r)
                     for r in ("target", "stop", "time", "flatten")},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=("nq", "gc"), required=True)
    ap.add_argument("--limit-folds", type=int, default=None)
    ap.add_argument("--no-audits", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()
    t0 = _time.time()
    deadline = (t0 + args.max_minutes * 60) if args.max_minutes else None
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    smoke = args.limit_folds is not None
    tag = f"strict11_{args.market}" + ("_smoke" if smoke else "")
    ckdir = outdir / "checkpoints" / tag
    ckdir.mkdir(parents=True, exist_ok=True)

    if args.market == "nq":
        full = pd.read_parquet(outdir / "nq_micro_bars_full.parquet")
        feat_fn = lambda df: compute_micro_q_features(df, cfg, FeatureParams())  # noqa: E731,E501
        grids = dict(zip(("NQ", "MNQ"), nq_strict_grids()))
        n_train, n_test = 250, 63
        primary, control = "MNQ", "NQ"
    else:
        full = pd.read_parquet(outdir / "gc_bars_full.parquet")
        feat_fn = lambda df: compute_gc_q_features(df, cfg)      # noqa: E731
        grids = dict(zip(("GC", "MGC"), gc_strict_grids()))
        n_train, n_test = 250, 126
        primary, control = "MGC", "GC"

    bars = full[full["session"] <= SEARCH_END].copy()
    tail_n = full.loc[full["session"] >= TAIL_START, "session"].nunique()
    del full
    sessions = sorted(set(bars["session"]))
    print(f"[{tag}] search {sessions[0]}..{sessions[-1]} "
          f"({len(sessions)} sessions), tail reserved {tail_n}; grids: "
          + ", ".join(f"{k}={len(v)}" for k, v in grids.items()), flush=True)

    t = _time.time()
    feats = feat_fn(bars)
    print(f"features in {_time.time() - t:.0f}s", flush=True)
    folds = make_session_folds(sessions, cfg, n_train, n_test, 1)
    if args.limit_folds:
        folds = folds[:args.limit_folds]
    print(f"folds={len(folds)}", flush=True)

    audits = []
    if not args.no_audits:
        aud_ck = ckdir / "audits.pkl"
        if aud_ck.exists():
            audits = pickle.loads(aud_ck.read_bytes())
        else:
            a = TruncationInvarianceCheck(bars, feat_fn, n_probes=3).run()
            a.name = f"{tag}_feature_truncation"
            audits.append(a)
            labres = compute_labels(bars, cfg, LabelParams())
            audits.append(FoldEmbargoCheck(
                folds, required_embargo(labres.specs)).run())
            purge = check_label_purge(labres.end_times, folds)
            audits.append(CheckResult(
                "label_purge", Status.PASS if not purge else Status.FAIL,
                "clean" if not purge else f"{len(purge)}"))
            aud_ck.write_bytes(pickle.dumps(audits))
        for r in audits:
            print(f"  [{r.status.value}] {r.name}: {r.message}", flush=True)
        if any(r.status == Status.FAIL for r in audits):
            return 1

    cm = CostModel(cfg)
    jobs = [(inst, k) for inst in grids for k in range(len(folds))]
    for inst, k in jobs:
        ck = ckdir / f"{inst}_{k}.pkl"
        if ck.exists():
            continue
        if deadline and _time.time() > deadline:
            print(f"PARTIAL: budget reached before {inst} fold {k}", flush=True)
            return 3
        t = _time.time()
        res = evaluate_fold_ext(grids[inst], bars, feats, folds[k], cfg, cm,
                                instrument=inst)
        ck.write_bytes(pickle.dumps(res))
        print(f"[{inst} fold {k:02d}] done ({_time.time() - t:.0f}s)",
              flush=True)

    # ---- assemble ----
    results = {}
    all_trade_rows = []
    for inst, grid in grids.items():
        fold_res = [pickle.loads((ckdir / f"{inst}_{k}.pkl").read_bytes())
                    for k in range(len(folds))]
        arms = {}
        agg = {r.name: {"tstats": [], "nets": [], "trades": {}} for r in grid}
        wf_sessions_all = []
        fold_diag = []
        for k, res in enumerate(fold_res):
            te_sessions = sorted(
                set(bars.loc[(bars.index >= folds[k]["test_start"])
                             & (bars.index < folds[k]["test_end"]),
                             "session"]))
            wf_sessions_all += te_sessions
            # per-fold attainable-train-win diagnostic
            wins_ok = [rec["train_wins"] / rec["train_trades"]
                       for rec in res.values()
                       if rec["train_trades"] >= MIN_TRAIN_TRADES]
            pos_ok = [rec["train_wins"] / rec["train_trades"]
                      for rec in res.values()
                      if rec["train_trades"] >= MIN_TRAIN_TRADES
                      and rec["train_net"] > 0]
            fold_diag.append({
                "fold": k,
                "n_min_trades": len(wins_ok),
                "max_train_win": max(wins_ok) if wins_ok else None,
                "n_train_net_pos": len(pos_ok),
                "max_train_win_netpos": max(pos_ok) if pos_ok else None,
                "n_gated_eligible": sum(
                    1 for rec in res.values()
                    if rec["train_trades"] >= MIN_TRAIN_TRADES
                    and rec["train_net"] > 0
                    and rec["train_wins"] / rec["train_trades"]
                    >= MIN_TRAIN_WIN)})
            for name, rec in res.items():
                agg[name]["tstats"].append(rec["train_tstat"])
                agg[name]["nets"].append(
                    float(sum(t.net_usd for t in rec["test_trades"])))
                for t in rec["test_trades"]:
                    agg[name]["trades"].setdefault(t.session, []).append(t)
                for t in rec["test_trades"]:
                    all_trade_rows.append(
                        {**asdict(t), "rule": name, "instrument": inst,
                         "fold": k})
        for arm_name, gated in (("gated", True), ("plain", False)):
            wf_trades = {s: [] for s in wf_sessions_all}
            arm_rows = []
            fold_table = []
            for k, res in enumerate(fold_res):
                winner = select_arm(res, gated)
                te_sessions = sorted(
                    set(bars.loc[(bars.index >= folds[k]["test_start"])
                                 & (bars.index < folds[k]["test_end"]),
                                 "session"]))
                tnet = 0.0
                if winner:
                    for t in res[winner]["test_trades"]:
                        wf_trades[t.session].append(t)
                        arm_rows.append({**asdict(t), "rule": winner,
                                         "fold": k})
                    tnet = float(sum(t.net_usd
                                     for t in res[winner]["test_trades"]))
                w = res[winner] if winner else None
                fold_table.append({
                    "fold": k, "winner": winner,
                    "train_tstat": round(w["train_tstat"], 2) if w else None,
                    "train_win": (round(w["train_wins"] / w["train_trades"], 3)
                                  if w else None),
                    "train_trades": w["train_trades"] if w else 0,
                    "test_net": round(tnet),
                    "test_trades": (len(w["test_trades"]) if w else 0)})
            all_tr, daily, wf = stream_stats(
                wf_trades, wf_sessions_all,
                [f["test_net"] for f in fold_table])
            mc = {}
            if all_tr:
                pool = [wf_trades[s] for s in sorted(wf_trades)]
                for size, pdll in (("50k", 750.0), ("50k", 1000.0),
                                   ("100k", 1000.0)):
                    mc[f"{size}/pdll{int(pdll)}"] = lifecycle_mc_chain(
                        pool, cfg, size=size, personal_dll=pdll, n_runs=300,
                        horizon_sessions=252)
            best_cfg = (max(mc, key=lambda k_: mc[k_].ev_month)
                        if mc else None)
            best = mc[best_cfg] if best_cfg else None
            verdict = {
                "zero_violations": True,
                "win_rate": bool(wf["trades"] and wf["win"] >= GATES["win"]),
                "win_rate_pref": bool(wf["trades"]
                                      and wf["win"] >= GATES["win_pref"]),
                "frequency": wf["tps"] >= GATES["tps"],
                "frequency_pref": (GATES["tps_pref_lo"] <= wf["tps"]
                                   <= GATES["tps_pref_hi"]),
                "net_positive": wf["net"] > 0,
                "ev_month_positive": bool(best and best.ev_month > 0),
                "blowout": bool(best
                                and best.blowout_rate < GATES["blowout"]),
                "fold_stability": wf["pos_folds"] >= len(folds) / 2,
            }
            hard = ("zero_violations", "win_rate", "frequency",
                    "net_positive", "ev_month_positive", "blowout",
                    "fold_stability")
            arms[arm_name] = {
                "wf": wf, "fold_table": fold_table, "best_cfg": best_cfg,
                "mc": {k_: {kk: (round(vv, 4) if isinstance(vv, float)
                                 else vv) for kk, vv in vars(v).items()}
                       for k_, v in mc.items()},
                "verdict": verdict,
                "gates_passed": all(verdict[k] for k in hard),
                "wf_trades": wf_trades,
                "wf_rows": arm_rows,
            }
            print(f"[{inst}/{arm_name}] net=${wf['net']:,.0f} "
                  f"n={wf['trades']} tps={wf['tps']:.2f} "
                  f"win={(wf['win'] if wf['trades'] else 0):.3f} "
                  f"posFolds={wf['pos_folds']}/{len(folds)} "
                  f"exit={wf['exit_mix']}"
                  + (f" | {best_cfg}: EV/mo=${best.ev_month:,.0f} "
                     f"blowout={best.blowout_rate:.2f}" if best else
                     " | MC: no trades"), flush=True)

        rows = []
        rule_by_name = {r.name: r for r in grid}
        for name, a in agg.items():
            tr = [t for lst in a["trades"].values() for t in lst]
            w_ = sum(1 for t in tr if t.net_usd > 0)
            rl = rule_by_name[name]
            rows.append({
                "name": name, "rr": rl.rr, "stop_atr": rl.stop_atr,
                "qty": rl.qty, "max_trades": rl.max_trades_per_day,
                "mean_train_tstat": float(np.mean(
                    [x for x in a["tstats"] if np.isfinite(x)]) if any(
                    np.isfinite(x) for x in a["tstats"]) else -np.inf),
                "oos_net": float(np.sum(a["nets"])),
                "oos_trades": len(tr),
                "oos_win_rate": w_ / len(tr) if tr else float("nan"),
                "trades_per_session": len(tr) / max(len(wf_sessions_all), 1),
                "avg_win": (float(np.mean([t.net_usd for t in tr
                                           if t.net_usd > 0]))
                            if w_ else 0.0),
                "avg_loss": (float(np.mean([t.net_usd for t in tr
                                            if t.net_usd <= 0]))
                             if len(tr) > w_ else 0.0),
                "gross": float(sum(t.gross_usd for t in tr)),
                "costs": float(sum(t.costs_usd for t in tr)),
                "pos_fold_share": float(np.mean([x > 0 for x in a["nets"]])),
            })
        summary = (pd.DataFrame(rows)
                   .sort_values(["oos_win_rate", "mean_train_tstat"],
                                ascending=False).reset_index(drop=True))
        results[inst] = {"arms": arms, "summary": summary,
                         "fold_diag": fold_diag}

    # winner signal audits (primary instrument, union of arm winners)
    if not args.no_audits:
        prim = results[primary]
        winners = sorted({f["winner"] for a in prim["arms"].values()
                          for f in a["fold_table"] if f["winner"]})[:4]
        for name in winners:
            rule = next(r for r in grids[primary] if r.name == name)

            def sig_fn(df, rule=rule):
                fe = feat_fn(df)
                return pd.DataFrame({"sig": rule.entry_signal(fe, cfg)
                                     .astype(float)}, index=df.index)

            a = TruncationInvarianceCheck(bars, sig_fn, n_probes=3).run()
            a.name = f"signal_truncation:{name}"
            audits.append(a)
            print(f"  [{a.status.value}] {a.name}", flush=True)
        LeakageAuditor.write_report(audits, outdir, tag)
    audits_ok = all(x.status != Status.FAIL for x in audits)

    # ---- persist trade-level data for failure attribution ----
    if all_trade_rows:
        pd.DataFrame(all_trade_rows).to_parquet(
            outdir / f"{tag}_all_candidate_trades.parquet")
    wf_rows = []
    for inst in results:
        for arm_name, arm in results[inst]["arms"].items():
            for row in arm["wf_rows"]:
                wf_rows.append({**row, "instrument": inst, "arm": arm_name})
    if wf_rows:
        pd.DataFrame(wf_rows).to_parquet(outdir / f"{tag}_wf_trades.parquet")
    for inst in results:
        results[inst]["summary"].to_csv(
            outdir / f"{tag}_{inst}_candidates.csv", index=False)

    report = {
        "phase": "strict RR>=1:1 high-win-rate discovery",
        "market": args.market, "primary": primary, "control": control,
        "search_era": [str(sessions[0]), str(sessions[-1])],
        "tail_reserved_sessions": int(tail_n),
        "folds": len(folds), "gates": GATES,
        "min_train_trades": MIN_TRAIN_TRADES,
        "min_train_win": MIN_TRAIN_WIN, "r_mults": list(R_MULTS),
        "grid_sizes": {k: len(v) for k, v in grids.items()},
        "audits_pass": bool(audits_ok),
        "instruments": {
            inst: {
                "arms": {an: {k_: v for k_, v in arm.items()
                              if k_ not in ("wf_trades", "wf_rows")}
                         for an, arm in results[inst]["arms"].items()},
                "fold_diag": results[inst]["fold_diag"],
                "top_candidates": results[inst]["summary"]
                .head(15).to_dict("records"),
            } for inst in results},
        "audits": [{"name": x.name, "status": x.status.value}
                   for x in audits],
        "runtime_min": round((_time.time() - t0) / 60, 1),
    }
    (outdir / f"{tag}_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    passed = {inst: {an: results[inst]["arms"][an]["gates_passed"]
                     for an in results[inst]["arms"]} for inst in results}
    print(f"\ngates_passed={passed}", flush=True)
    print(f"wrote {outdir}/{tag}_report.json ({report['runtime_min']} min)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
