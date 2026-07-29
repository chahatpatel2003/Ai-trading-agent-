#!/usr/bin/env python
"""Failure-attribution diagnosis for the strict RR>=1:1 pass.

Reads strict11_{nq,gc}_report.json + trade parquets and computes the
15-question attribution battery:
  win rate | costs | frequency | avg win vs loss | year/regime | drawdown
  entries-vs-exits (offline MFE vs bars) | direction | vol regime |
  time-of-day | MNQ-vs-NQ | MGC-vs-GC | breakeven win rate | missing gross
  edge per trade | binding constraint.

Writes outputs/strict11_diagnosis.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tslab.config import load_config                      # noqa: E402
from tslab.sim.costs import CostModel                     # noqa: E402
from run_strict11_search import (gc_strict_grids,         # noqa: E402
                                 nq_strict_grids)

CT = "America/Chicago"


def jser(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def stream_table(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"trades": 0}
    wins = df["net_usd"] > 0
    out = {
        "trades": int(n),
        "net": float(df["net_usd"].sum()),
        "gross": float(df["gross_usd"].sum()),
        "costs": float(df["costs_usd"].sum()),
        "win_rate": float(wins.mean()),
        "avg_win": float(df.loc[wins, "net_usd"].mean()) if wins.any() else 0,
        "avg_loss": (float(df.loc[~wins, "net_usd"].mean())
                     if (~wins).any() else 0),
        "avg_risk": float(df["risk_usd"].mean()),
        "cost_per_trade": float(df["costs_usd"].mean()),
        "cost_pct_of_risk": float(df["costs_usd"].mean()
                                  / max(df["risk_usd"].mean(), 1e-9)),
        "exit_mix": df["exit_reason"].value_counts().to_dict(),
        "net_by_exit": df.groupby("exit_reason")["net_usd"]
        .agg(["count", "sum", "mean"]).round(2).to_dict("index"),
    }
    out["deficit_per_trade"] = -out["net"] / n
    # observed-payoff breakeven win rate
    aw, al = out["avg_win"], abs(out["avg_loss"])
    out["breakeven_win_observed"] = al / (aw + al) if (aw + al) > 0 else None
    return out


def barrier_breakeven(cm: CostModel, inst: str, qty: int, risk_usd: float,
                      rr: float) -> dict:
    """Idealized two-barrier breakeven win rate at avg risk R and ratio rr:
    p*(R*rr - Cwin) = (1-p*)(R + Closs)."""
    c_entry = cm.side_cost(inst, qty, "market").total
    c_win = c_entry + cm.side_cost(inst, qty, "limit").total
    c_loss = c_entry + cm.side_cost(inst, qty, "stop").total
    denom = risk_usd * rr - c_win + risk_usd + c_loss
    return {
        "cost_win_rt": c_win, "cost_loss_rt": c_loss,
        "breakeven_win": (risk_usd + c_loss) / denom if denom > 0 else None,
    }


def by_bucket(df, key, min_n=10):
    g = df.groupby(key, observed=True)["net_usd"]
    t = pd.DataFrame({"n": g.size(), "net": g.sum().round(0),
                      "mean": g.mean().round(2),
                      "win": df.assign(w=df["net_usd"] > 0)
                      .groupby(key, observed=True)["w"].mean().round(3)})
    return t[t["n"] >= min_n].to_dict("index")


def mfe_analysis(trades: pd.DataFrame, bars: pd.DataFrame, rules: dict,
                 pv: float) -> dict:
    """Offline MFE: from each trade's entry bar, before the stop level is
    breached (stop-first convention) and within the rule's time-stop horizon
    (capped at session end), how far did price move in the trade's favor?
    Separates entry quality (does the market offer >= target excursion?)
    from exit handling."""
    if trades.empty:
        return {"trades": 0}
    idx = bars.index
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    sess = bars["session"].to_numpy()
    sess_end = {}
    for i, s in enumerate(sess):
        sess_end[s] = i
    pos = idx.get_indexer(pd.DatetimeIndex(trades["entry_time"]))
    reached_tgt, reached_1r, mfe_r, valid = [], [], [], []
    for k, (_, t) in enumerate(trades.iterrows()):
        e = pos[k]
        rl = rules.get(t["rule"]) if "rule" in t else None
        ts_bars = rl["ts"] if rl else 60
        rr = rl["rr"] if rl else 1.0
        if e < 0:
            valid.append(False)
            reached_tgt.append(False)
            reached_1r.append(False)
            mfe_r.append(np.nan)
            continue
        d = int(t["direction"])
        entry = float(t["entry_px"])
        riskpx = float(t["risk_usd"]) / (pv * max(int(t["qty"]), 1))
        stop_px = entry - d * riskpx
        end = min(e + ts_bars, sess_end.get(t["session"], e))
        best = 0.0
        for j in range(e, end + 1):
            stop_hit = (low[j] <= stop_px) if d > 0 else (high[j] >= stop_px)
            if stop_hit:            # stop-first: no favorable credit this bar
                break
            fav = (high[j] - entry) if d > 0 else (entry - low[j])
            best = max(best, fav)
        r = best / riskpx if riskpx > 0 else np.nan
        valid.append(True)
        mfe_r.append(r)
        reached_1r.append(r >= 1.0)
        reached_tgt.append(r >= rr)
    m = pd.DataFrame({"v": valid, "tgt": reached_tgt, "r1": reached_1r,
                      "mfe": mfe_r, "win": trades["net_usd"].values > 0,
                      "reason": trades["exit_reason"].values})
    m = m[m["v"]]
    if m.empty:
        return {"trades": 0}
    tgt_reachers = m[m["tgt"]]
    return {
        "trades": int(len(m)),
        "share_mfe_ge_target_before_stop": float(m["tgt"].mean()),
        "share_mfe_ge_1R_before_stop": float(m["r1"].mean()),
        "median_mfe_R_before_stop": float(m["mfe"].median()),
        "win_rate_given_target_offered": (float(tgt_reachers["win"].mean())
                                          if len(tgt_reachers) else None),
        "time_exits_that_touched_1R": (
            float(m[m["reason"] == "time"]["r1"].mean())
            if (m["reason"] == "time").any() else None),
    }


def main() -> int:
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    cm = CostModel(cfg)
    grids = {}
    for (a, b), names in ((nq_strict_grids(), ("NQ", "MNQ")),
                          (gc_strict_grids(), ("GC", "MGC"))):
        for inst, grid in zip(names, (a, b)):
            grids[inst] = {r.name: {"ts": r.time_stop_bars, "rr": r.rr,
                                    "stop_atr": r.stop_atr, "qty": r.qty}
                           for r in grid}
    bars_cache = {}
    diag = {}
    for mkt, bars_file in (("nq", "nq_micro_bars_full.parquet"),
                           ("gc", "gc_bars_full.parquet")):
        rep = json.loads((outdir / f"strict11_{mkt}_report.json").read_text())
        wf_path = outdir / f"strict11_{mkt}_wf_trades.parquet"
        wf = (pd.read_parquet(wf_path) if wf_path.exists()
              else pd.DataFrame(columns=[
                  "session", "entry_time", "exit_time", "direction", "qty",
                  "entry_px", "exit_px", "exit_reason", "gross_usd",
                  "costs_usd", "net_usd", "mae_usd", "risk_usd", "rule",
                  "fold", "instrument", "arm"]))
        allt = pd.read_parquet(
            outdir / f"strict11_{mkt}_all_candidate_trades.parquet")
        for df in (wf, allt):
            if not len(df):
                df["ct_hour"] = pd.Series(dtype=int)
                df["year"] = pd.Series(dtype=int)
                continue
            et = pd.DatetimeIndex(df["entry_time"])
            df["ct_hour"] = et.tz_convert(CT).hour
            df["year"] = [s.year for s in df["session"]]
        allt["rr"] = allt.apply(
            lambda r: grids[r["instrument"]][r["rule"]]["rr"], axis=1)
        allt["stop_atr"] = allt.apply(
            lambda r: grids[r["instrument"]][r["rule"]]["stop_atr"], axis=1)
        # ATR ticks at entry, derived: risk = stop_atr * atr_pts * pv * qty
        for inst in allt["instrument"].unique():
            m = allt["instrument"] == inst
            pv = cfg.instruments[inst].point_value
            tick = cfg.instruments[inst].tick_size
            allt.loc[m, "atr_ticks"] = (
                allt.loc[m, "risk_usd"]
                / (allt.loc[m, "stop_atr"] * pv * allt.loc[m, "qty"] * tick))
        diag[mkt] = {"report_gates": {
            inst: {arm: rep["instruments"][inst]["arms"][arm]["verdict"]
                   for arm in rep["instruments"][inst]["arms"]}
            for inst in rep["instruments"]},
            "fold_diag": {inst: rep["instruments"][inst]["fold_diag"]
                          for inst in rep["instruments"]},
            "instruments": {}}
        if bars_file not in bars_cache:
            bars_cache[bars_file] = pd.read_parquet(outdir / bars_file)
        bars = bars_cache[bars_file]
        for inst in rep["instruments"]:
            pv = cfg.instruments[inst].point_value
            sub = allt[allt["instrument"] == inst]
            d = {}
            # arms: stream-level tables
            for arm in ("gated", "plain"):
                s = wf[(wf["instrument"] == inst) & (wf["arm"] == arm)]
                d[f"stream_{arm}"] = stream_table(s)
                if len(s):
                    d[f"stream_{arm}"]["by_year"] = by_bucket(s, "year", 1)
                    d[f"stream_{arm}"]["by_hour"] = by_bucket(s, "ct_hour", 5)
                    d[f"stream_{arm}"]["by_direction"] = by_bucket(
                        s, "direction", 5)
            # all-candidate aggregates (the search space itself)
            wins = sub["net_usd"] > 0
            d["all_candidates"] = {
                "trades": int(len(sub)),
                "win_rate": float(wins.mean()) if len(sub) else None,
                "net": float(sub["net_usd"].sum()),
                "gross": float(sub["gross_usd"].sum()),
                "costs": float(sub["costs_usd"].sum()),
                "by_rr": by_bucket(sub, "rr", 10),
                "by_direction": by_bucket(sub, "direction", 10),
                "by_year": by_bucket(sub, "year", 10),
                "by_hour": by_bucket(sub, "ct_hour", 50),
            }
            if len(sub):
                sub = sub.copy()
                sub["vol_tercile"] = pd.qcut(sub["atr_ticks"], 3,
                                             labels=["low", "mid", "high"])
                d["all_candidates"]["by_vol_tercile"] = by_bucket(
                    sub, "vol_tercile", 10)
            # candidate-level joint-gate satisfaction
            cand = pd.read_csv(outdir / f"strict11_{mkt}_{inst}"
                               "_candidates.csv")
            cand["win_ok"] = cand["oos_win_rate"] >= 0.55
            cand["tps_ok"] = cand["trades_per_session"] >= 1.0
            cand["net_ok"] = cand["oos_net"] > 0
            cand["stab_ok"] = cand["pos_fold_share"] >= 0.5
            d["joint_gates"] = {
                "n_candidates": int(len(cand)),
                "win_only": int(cand["win_ok"].sum()),
                "tps_only": int(cand["tps_ok"].sum()),
                "net_only": int(cand["net_ok"].sum()),
                "win_and_tps": int((cand["win_ok"] & cand["tps_ok"]).sum()),
                "win_and_net": int((cand["win_ok"] & cand["net_ok"]).sum()),
                "tps_and_net": int((cand["tps_ok"] & cand["net_ok"]).sum()),
                "all_three": int((cand["win_ok"] & cand["tps_ok"]
                                  & cand["net_ok"]).sum()),
                "all_four": int((cand["win_ok"] & cand["tps_ok"]
                                 & cand["net_ok"] & cand["stab_ok"]).sum()),
            }
            best = cand.sort_values("oos_net", ascending=False).head(5)
            d["best_by_net"] = best[
                ["name", "rr", "oos_net", "oos_trades", "oos_win_rate",
                 "trades_per_session", "avg_win", "avg_loss", "gross",
                 "costs", "pos_fold_share"]].round(3).to_dict("records")
            hi_freq = cand[cand["trades_per_session"] >= 1.0]
            d["best_win_at_tps1"] = (
                hi_freq.sort_values("oos_win_rate", ascending=False).head(5)[
                    ["name", "rr", "oos_net", "oos_trades", "oos_win_rate",
                     "trades_per_session", "avg_win", "avg_loss",
                     "pos_fold_share"]].round(3).to_dict("records"))
            hi_win = cand[cand["oos_win_rate"] >= 0.55]
            d["best_tps_at_win55"] = (
                hi_win.sort_values("trades_per_session", ascending=False)
                .head(5)[["name", "rr", "oos_net", "oos_trades",
                          "oos_win_rate", "trades_per_session",
                          "pos_fold_share"]].round(3).to_dict("records"))
            # breakeven math at observed geometry
            for rr in (1.0, 1.5):
                s_rr = sub[sub["rr"] == rr]
                if len(s_rr):
                    qty = int(s_rr["qty"].mode().iloc[0])
                    d[f"barrier_breakeven_rr{rr}"] = barrier_breakeven(
                        cm, inst, qty, float(s_rr["risk_usd"].mean()), rr)
                    d[f"barrier_breakeven_rr{rr}"]["avg_risk"] = float(
                        s_rr["risk_usd"].mean())
                    d[f"barrier_breakeven_rr{rr}"]["observed_win"] = float(
                        (s_rr["net_usd"] > 0).mean())
                    d[f"barrier_breakeven_rr{rr}"]["observed_target_rate"] = \
                        float((s_rr["exit_reason"] == "target").mean())
            # MFE (entries vs exits): WF plain stream + best-by-net top3
            wf_p = wf[(wf["instrument"] == inst) & (wf["arm"] == "plain")]
            d["mfe_stream_plain"] = mfe_analysis(wf_p, bars, grids[inst], pv)
            top3 = list(best["name"].head(3))
            s3 = sub[sub["rule"].isin(top3)]
            d["mfe_top3_by_net"] = mfe_analysis(s3, bars, grids[inst], pv)
            diag[mkt]["instruments"][inst] = d
        # micro-vs-mini paired comparison on shared signal bases
        pair = ("NQ", "MNQ") if mkt == "nq" else ("GC", "MGC")
        rows = {}
        for inst in pair:
            c = pd.read_csv(outdir / f"strict11_{mkt}_{inst}_candidates.csv")
            c["base"] = c["name"].str.replace(
                r"_r[0-9.]+_[a-z]+[abc]$", "", regex=True)
            g = c.groupby("base").agg(
                net=("oos_net", "sum"), trades=("oos_trades", "sum"),
                gross=("gross", "sum"), costs=("costs", "sum"))
            rows[inst] = g
        j = rows[pair[0]].join(rows[pair[1]], lsuffix=f"_{pair[0]}",
                               rsuffix=f"_{pair[1]}", how="inner")
        diag[mkt]["micro_vs_mini"] = {
            "per_base": j.round(0).to_dict("index"),
            "totals": {i: {"net": float(rows[i]["net"].sum()),
                           "gross": float(rows[i]["gross"].sum()),
                           "costs": float(rows[i]["costs"].sum()),
                           "cost_share_of_gross_moves": (
                               float(rows[i]["costs"].sum()
                                     / max(abs(rows[i]["gross"]).sum(), 1)))}
                       for i in pair}}
    (outdir / "strict11_diagnosis.json").write_text(
        json.dumps(diag, indent=2, default=jser))
    print(f"wrote {outdir}/strict11_diagnosis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
