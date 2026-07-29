#!/usr/bin/env python
"""Walk-forward tick-level passive scalper research (MNQ primary, NQ fee
comparison) on the 1s decision dataset built from MBP-1.

Protocol (pre-registered):
  - folds: 250 train / 63 test sessions, embargo 1, over 2021-2024
    (2025H1 never loaded).
  - models per fold, per side (B=join-bid long, S=join-ask short),
    target = GROSS ticks of the reference policy (T=2,S=8,hold=15s,
    cancel=3s, through fills, 1-tick stop slippage; unfilled = 0):
      logit  SGD logistic on z=(gross>0), standardized, chunked
      rf     RandomForestClassifier on 1.2M-row subsample
      hgb    HistGradientBoostingRegressor (primary EV model)
  - train-only selection per model x instrument: policy grid
    T in {1,2,3,4} x S in {4,6,8,12} x hold in {1,5,15,30}s x cancel in
    {1,3,10}s x flip-flatten in {off,on} x threshold in top
    {0.5,1,2,5}% of train score; objective: max train net EV/day with
    EV/trade>0, fills/session in [2,60], preferring >=5 fills/session.
  - test: SEQUENTIAL simulator: one position, busy while pending/held,
    1s cooldown, day stop (MNQ $75 / NQ $500), kill switch (trailing
    300-trade net < -300 x tick_value), flip-flatten at fill+5s if the
    book turns against (qimb x dir < -0.3), fees per side, stop exits
    slip 1 extra tick.
  - stress (winner model, frozen policy): fees x1.5/x2, fill rule
    through->queue-proxy/touch, 1s-stale features (latency), and a
    shuffled-label control (hgb trained on train-permuted targets).
  - GATING (patched 2026-07-05): a fold policy with qualifies=False
    (no positive train-EV grid cell) is recorded in fold_table but is
    NEVER simulated, never contributes trades, and never enters the
    headline arms aggregate. The old fallback that traded the
    least-negative rejected candidate is removed. The shuffled-label
    control is reported under report["controls"] and can never be
    viable. Viability/warning semantics live in
    tslab.strategy.scalper_report.

Usage: run_scalper_wf.py [--folds-from N] [--folds-to N] [--models ...]
Writes outputs/scalper_report.json, scalper_report.md,
scalper_trades.parquet, checkpoints under outputs/checkpoints/scalper/.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab.config import load_config                      # noqa: E402
from tslab.features.labels import make_session_folds      # noqa: E402
from tslab.strategy.scalper_report import (                # noqa: E402
    aggregate_folds, breakeven_win_rate, gate_fold_policy,
    is_control_model, policy_econ, realized_rr, write_scalper_md)

FEATURES = ["spread_t", "log_bsz", "log_asz", "qimb", "micro_off",
            "ofi_1s", "ofi_5s", "ofi_30s", "tp_1s", "tp_5s", "tp_30s",
            "trct_5s", "qrate_5s", "cancel_5s", "mom_1s", "mom_5s",
            "mom_30s", "rv_30s", "range_30s", "vwap_dist", "open_dist",
            "tod", "t_since_tr", "last_side", "flips_5s"]
UPS = (1, 2, 3, 4)
DNS = (4, 6, 8, 12)
HOLDS = (1, 5, 15, 30)
CANCELS = (1.0, 3.0, 10.0)
THRQ = (0.995, 0.99, 0.98, 0.95)
REF = dict(T=2, S=8, hold=15, cancel=3.0, rule="thr")
STOP_SLIP = 1.0
INST = {"MNQ": dict(tv=0.5, fee_side=0.62, day_stop=75.0, kill=150.0),
        "NQ": dict(tv=5.0, fee_side=1.90, day_stop=500.0, kill=1500.0)}
SEARCH_END = pd.Timestamp("2024-12-31").date()


class Data:
    def __init__(self, outdir: Path):
        files = sorted((outdir / "tick_ds").glob("tickds_202[1-4]_*.parquet"))
        cols = (["ts", "session", "bid", "ask"] + FEATURES
                + [f"{s}_{r}_dt" for s in "BS" for r in ("thr", "qpx",
                                                         "tch")]
                + [f"{s}_up{k}" for s in "BS" for k in UPS]
                + [f"{s}_dn{k}" for s in "BS" for k in DNS]
                + [f"{s}_mark{m}s" for s in "BS" for m in (1, 5, 15, 30)]
                + ["fmid_5s"])
        df = pd.concat([pd.read_parquet(f, columns=cols) for f in files],
                       ignore_index=True)
        n0 = len(df)
        df = df.dropna(subset=FEATURES).reset_index(drop=True)
        print(f"loaded {n0} rows -> {len(df)} after feature-NaN drop",
              flush=True)
        self.sessions = df["session"].to_numpy()
        self.ts = df["ts"].astype("int64").to_numpy()
        self.X = df[FEATURES].to_numpy(np.float32)
        self.qimb = df["qimb"].to_numpy(np.float32)
        self.L = {c: df[c].to_numpy(np.float32) for c in df.columns
                  if c.startswith(("B_", "S_")) or c == "fmid_5s"}
        self.sess_code = pd.factorize(df["session"])[0]
        self.session_list = sorted(set(df["session"]))
        del df

    def sess_mask(self, sess_set):
        return np.isin(self.sessions, list(sess_set))


def policy_gross(D: Data, idx: np.ndarray, side: str, T: int, S: int,
                 hold: float, cancel: float, rule: str, flip: bool
                 ) -> tuple:
    """(gross ticks array, filled mask, exit_dt seconds, reason codes)
    for decision rows idx. Unfilled rows: gross NaN, filled False."""
    L = D.L
    fdt = L[f"{side}_{rule}_dt"][idx]
    filled = np.isfinite(fdt) & (fdt <= cancel)
    inf = np.float32(np.inf)
    t_up = np.where(np.isfinite(L[f"{side}_up{T}"][idx]),
                    L[f"{side}_up{T}"][idx], inf)
    t_dn = np.where(np.isfinite(L[f"{side}_dn{S}"][idx]),
                    L[f"{side}_dn{S}"][idx], inf)
    mark_h = L[f"{side}_mark{int(hold)}s"][idx]
    stop_first = t_dn <= np.minimum(t_up, hold)
    win = ~stop_first & (t_up <= hold)
    gross = np.where(stop_first, -S - STOP_SLIP,
                     np.where(win, float(T), mark_h))
    exit_dt = np.where(stop_first, t_dn, np.where(win, t_up, float(hold)))
    reason = np.where(stop_first, 1, np.where(win, 0, 2)).astype(np.int8)
    if flip and hold > 5:
        undecided = filled & (t_up > 5) & (t_dn > 5)
        off = np.rint(np.nan_to_num(fdt) + 5).astype(np.int64)
        j = idx + off
        ok = (j < len(D.qimb))
        j = np.minimum(j, len(D.qimb) - 1)
        same = D.sess_code[j] == D.sess_code[idx]
        dirn = 1.0 if side == "B" else -1.0
        flipped = undecided & ok & same & (D.qimb[j] * dirn < -0.3)
        gross = np.where(flipped, L[f"{side}_mark5s"][idx], gross)
        exit_dt = np.where(flipped, 5.0, exit_dt)
        reason = np.where(flipped, 3, reason).astype(np.int8)
    gross = np.where(filled, gross, np.nan)
    return gross, filled, exit_dt, reason


def train_models(D: Data, tr_idx: np.ndarray, models: list, seed: int,
                 shuffle_target: bool = False) -> dict:
    """Fit per-side models on the reference-policy GROSS target.
    Returns {model: {side: fitted}} plus scalers."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    out = {"scaler": None}
    fit_idx = tr_idx[::2]                 # fit-time subsample (speed)
    Xtr = D.X[fit_idx]
    ys = {}
    for side in ("B", "S"):
        g, filled, _, _ = policy_gross(D, fit_idx, side, **REF,
                                       flip=False)
        y = np.where(filled, np.nan_to_num(g), 0.0).astype(np.float32)
        if shuffle_target:
            y = y[rng.permutation(len(y))]
        ys[side] = y
    if "logit" in models or True:
        sc = StandardScaler().fit(Xtr[:: max(len(Xtr) // 2_000_000, 1)])
        out["scaler"] = sc
    for m in models:
        out[m] = {}
        for side in ("B", "S"):
            y = ys[side]
            if m == "hgb":
                mdl = HistGradientBoostingRegressor(
                    max_iter=250, learning_rate=0.06, max_leaf_nodes=63,
                    min_samples_leaf=500, early_stopping=True,
                    validation_fraction=0.1, random_state=seed)
                mdl.fit(Xtr, y)
            elif m == "logit":
                mdl = SGDClassifier(loss="log_loss", alpha=1e-6,
                                    max_iter=5, tol=1e-4,
                                    random_state=seed)
                mdl.fit(out["scaler"].transform(Xtr), (y > 0))
            elif m == "rf":
                sub = rng.choice(len(fit_idx),
                                 size=min(1_000_000, len(fit_idx)),
                                 replace=False)
                mdl = RandomForestClassifier(
                    n_estimators=120, max_depth=12, min_samples_leaf=200,
                    n_jobs=-1, random_state=seed)
                mdl.fit(Xtr[sub], (y[sub] > 0))
            out[m][side] = mdl
    return out


def score(D: Data, fitted: dict, m: str, side: str, idx: np.ndarray
          ) -> np.ndarray:
    X = D.X[idx]
    mdl = fitted[m][side]
    if m == "logit":
        return mdl.predict_proba(
            fitted["scaler"].transform(X))[:, 1].astype(np.float32)
    if m == "rf":
        return mdl.predict_proba(X)[:, 1].astype(np.float32)
    return mdl.predict(X).astype(np.float32)


def select_policy(D: Data, tr_idx: np.ndarray, sc_B: np.ndarray,
                  sc_S: np.ndarray, inst: dict, n_sessions: int) -> dict:
    """Train-only (policy, threshold) selection maximizing net EV/day."""
    fee_rt = 2 * inst["fee_side"] / inst["tv"]        # ticks per RT
    best = None
    best_any = None
    for T in UPS:
        for S in DNS:
            for hold in HOLDS:
                for cancel in CANCELS:
                    for flip in (False, True):
                        if flip and hold <= 5:
                            continue
                        for side, sc in (("B", sc_B), ("S", sc_S)):
                            g, filled, _, _ = policy_gross(
                                D, tr_idx, side, T, S, hold, cancel,
                                "thr", flip)
                            net = g - fee_rt
                            for q in THRQ:
                                thr = float(np.quantile(sc, q))
                                m_ = (sc >= thr) & filled
                                n = int(m_.sum())
                                if n < 200:
                                    continue
                                ev_t = float(np.nanmean(net[m_]))
                                fps = n / n_sessions   # vectorized count:
                                # ignores position-blocking; realized tps
                                # comes from the sequential simulator
                                ev_day = ev_t * fps * inst["tv"]
                                cand = dict(
                                    side=side, T=T, S=S, hold=hold,
                                    cancel=cancel, flip=flip, q=q,
                                    thr=thr, train_ev_ticks=ev_t,
                                    train_fps=fps, train_ev_day=ev_day,
                                    pref=fps >= 5)
                                if (best_any is None
                                        or ev_t > best_any[
                                            "train_ev_ticks"]):
                                    best_any = cand
                                if ev_t <= 0 or fps < 2:
                                    continue
                                if best is None or (
                                    (cand["pref"], cand["train_ev_day"])
                                    > (best["pref"],
                                       best["train_ev_day"])):
                                    best = cand
    if best is not None:
        return {**best, "qualifies": True}
    if best_any is not None:
        # Report-only record of the least-negative rejected candidate.
        # gate_fold_policy() guarantees it is never simulated or
        # aggregated (the old fallback traded it anyway — removed).
        return {**best_any, "qualifies": False}
    return None


def simulate(D: Data, te_idx: np.ndarray, sc: np.ndarray, pol: dict,
             inst: dict, fee_mult: float = 1.0, rule: str = "thr",
             latency_rows: int = 0) -> dict:
    """Sequential test simulation of one frozen (side, policy, thr)."""
    side, T, S = pol["side"], pol["T"], pol["S"]
    hold, cancel, flip = pol["hold"], pol["cancel"], pol["flip"]
    fee_rt_t = 2 * inst["fee_side"] * fee_mult / inst["tv"]
    if latency_rows:
        sc = np.concatenate([np.full(latency_rows, -np.inf,
                                     np.float32), sc[:-latency_rows]])
    g, filled, exit_dt, reason = policy_gross(
        D, te_idx, side, T, S, hold, cancel, rule, flip)
    fdt = D.L[f"{side}_{rule}_dt"][te_idx]
    mark1 = D.L[f"{side}_mark1s"][te_idx]
    fmid = D.L["fmid_5s"][te_idx]
    ts = D.ts[te_idx]
    sess = D.sessions[te_idx]
    order = np.argsort(ts, kind="mergesort")
    trades, attempts = [], 0
    misses_fmid = []
    day_pnl, cur_sess = 0.0, None
    killed = False
    tail = []
    busy_until = -1
    halted_day = False
    kill_events = 0
    for i in order:
        if sc[i] < pol["thr"]:
            continue
        if sess[i] != cur_sess:
            cur_sess = sess[i]
            day_pnl = 0.0
            halted_day = False
        if killed or halted_day or ts[i] < busy_until:
            continue
        attempts += 1
        if not filled[i]:
            d = 1.0 if side == "B" else -1.0
            misses_fmid.append(d * float(fmid[i]))
            busy_until = ts[i] + int(cancel * 1e9)
            continue
        net_t = float(g[i]) - fee_rt_t
        net = net_t * inst["tv"]
        day_pnl += net
        trades.append(dict(
            session=str(sess[i]), ts=int(ts[i]), side=side,
            gross_ticks=float(g[i]), net_ticks=net_t, net_usd=net,
            exit=int(reason[i]), hold_s=float(exit_dt[i]),
            fill_dt=float(fdt[i]), adv_1s=float(mark1[i])))
        busy_until = ts[i] + int((fdt[i] + exit_dt[i] + 1.0) * 1e9)
        if day_pnl <= -inst["day_stop"]:
            halted_day = True
        tail.append(net)
        if len(tail) > 300:
            tail.pop(0)
        if len(tail) == 300 and sum(tail) < -inst["kill"]:
            killed = True
            kill_events += 1
    tr = pd.DataFrame(trades)
    n = len(tr)
    n_sess = len(set(sess.tolist()))
    daily = (tr.groupby("session")["net_usd"].sum()
             if n else pd.Series(dtype=float))
    eq = daily.cumsum()
    gross_usd = float((tr["gross_ticks"] * inst["tv"]).sum()) if n \
        else 0.0
    net_usd = float(tr["net_usd"].sum()) if n else 0.0
    return dict(
        trades=n, attempts=attempts,
        fill_rate=n / attempts if attempts else float("nan"),
        tps=n / max(n_sess, 1),
        net=net_usd,
        gross_usd=gross_usd,
        costs_usd=gross_usd - net_usd,
        n_wins=int((tr["net_usd"] > 0).sum()) if n else 0,
        ev_trade=float(tr["net_usd"].mean()) if n else float("nan"),
        ev_day=(float(tr["net_usd"].sum()) if n else 0.0) / max(n_sess, 1),
        win=float((tr["net_usd"] > 0).mean()) if n else float("nan"),
        avg_win=float(tr.loc[tr.net_usd > 0, "net_usd"].mean())
        if n and (tr.net_usd > 0).any() else 0.0,
        avg_loss=float(tr.loc[tr.net_usd <= 0, "net_usd"].mean())
        if n and (tr.net_usd <= 0).any() else 0.0,
        adv_1s=float(tr["adv_1s"].mean()) if n else float("nan"),
        max_dd=float((eq - eq.cummax()).min()) if n else 0.0,
        miss_fmid_5s=float(np.mean(misses_fmid))
        if misses_fmid else float("nan"),
        killed=kill_events,
        exit_mix=(tr["exit"].value_counts().to_dict() if n else {}),
        trade_rows=tr,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="logit,rf,hgb")
    ap.add_argument("--folds-from", type=int, default=0)
    ap.add_argument("--folds-to", type=int, default=99)
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    t0 = _time.time()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    ckdir = outdir / "checkpoints" / "scalper"
    ckdir.mkdir(parents=True, exist_ok=True)
    models = args.models.split(",")
    D = Data(outdir)
    assert max(D.session_list) <= SEARCH_END
    folds = make_session_folds(D.session_list, cfg, 250, 63, 1)
    print(f"{len(D.session_list)} sessions, {len(folds)} folds, "
          f"rows={len(D.X)}", flush=True)

    for k, fold in enumerate(folds):
        if not (args.folds_from <= k <= args.folds_to):
            continue
        ck = ckdir / f"fold_{k}.pkl"
        if ck.exists() or args.assemble_only:
            continue
        t = _time.time()
        tr_sess = [s for s in D.session_list
                   if fold["train_start"].date() <= s
                   < fold["train_end"].date()]
        te_sess = [s for s in D.session_list
                   if fold["test_start"].date() <= s
                   < fold["test_end"].date()]
        assert not set(tr_sess) & set(te_sess)
        tr_idx = np.nonzero(D.sess_mask(tr_sess))[0]
        te_idx = np.nonzero(D.sess_mask(te_sess))[0]
        fitted = train_models(D, tr_idx, models, seed=100 + k)
        shuf = train_models(D, tr_idx, ["hgb"], seed=200 + k,
                            shuffle_target=True)
        tr_sel = tr_idx[::2]              # selection-time subsample
        n_sel_sess = max(len(tr_sess) // 2, 1)
        pack = {"n_train_sessions": len(tr_sess),
                "n_test_sessions": len(te_sess),
                "te_idx": te_idx, "scores": {}, "policies": {},
                "diag": {}}
        pack["tr_scores"] = {}
        pack["n_sel_sessions"] = n_sel_sess
        for m in models:
            sc_trB = score(D, fitted, m, "B", tr_sel)
            sc_trS = score(D, fitted, m, "S", tr_sel)
            pack["tr_scores"][m] = {"B": sc_trB, "S": sc_trS}
            pack["scores"][m] = {
                "B": score(D, fitted, m, "B", te_idx),
                "S": score(D, fitted, m, "S", te_idx)}
            for inst in INST:
                pol = select_policy(D, tr_sel, sc_trB, sc_trS,
                                    INST[inst], n_sel_sess)
                pack["policies"][(m, inst)] = pol
            # model-skill diagnostics on TEST (reference policy, B side)
            g, filled, _, _ = policy_gross(D, te_idx, "B", **REF,
                                           flip=False)
            y = np.where(filled, np.nan_to_num(g), 0.0)
            scB = pack["scores"][m]["B"]
            order = np.argsort(scB)
            decs = np.array_split(order, 10)
            pack["diag"][m] = {
                "test_ic": float(np.corrcoef(scB, y)[0, 1]),
                "test_decile_gross": [
                    float(np.nanmean(np.where(filled[d_], g[d_],
                                              np.nan)))
                    for d_ in decs]}
        pack["scores"]["hgb_shuf"] = {
            "B": score(D, shuf, "hgb", "B", te_idx),
            "S": score(D, shuf, "hgb", "S", te_idx)}
        sc_trB = score(D, shuf, "hgb", "B", tr_sel)
        sc_trS = score(D, shuf, "hgb", "S", tr_sel)
        for inst in INST:
            pack["policies"][("hgb_shuf", inst)] = select_policy(
                D, tr_sel, sc_trB, sc_trS, INST[inst], n_sel_sess)
        ck.write_bytes(pickle.dumps(pack))
        print(f"[fold {k:02d}] trained+selected "
              f"({(_time.time()-t)/60:.1f} min)", flush=True)

    # ---------------- assembly ----------------
    packs = []
    for k in range(len(folds)):
        p = ckdir / f"fold_{k}.pkl"
        if not p.exists():
            print(f"fold {k} missing — partial assembly", flush=True)
            continue
        packs.append((k, pickle.loads(p.read_bytes())))
    report = {"phase": "tick-scalper WF", "folds": len(packs),
              "features": FEATURES, "ref_policy": REF,
              "instruments": {}}
    all_trades = []
    report["controls"] = {}
    for inst in INST:
        rep_i = {}
        fee_rt = 2 * INST[inst]["fee_side"] / INST[inst]["tv"]
        for m in models + ["hgb_shuf"]:
            is_control = is_control_model(m)
            fold_rows, traded_pols = [], []
            arms = {a: [] for a in ("base", "fee1.5", "fee2.0", "qpx",
                                    "tch", "lat1s")}
            for k, pack in packs:
                pol = pack["policies"].get((m, inst))
                traded, gate_reason = gate_fold_policy(pol)
                row = {"fold": k,
                       "policy": dict(pol) if pol else None,
                       "traded": traded, "gate_reason": gate_reason}
                if pol is not None:
                    row["policy_econ"] = policy_econ(pol, fee_rt,
                                                     STOP_SLIP)
                if not traded:
                    # qualifies=False (or no policy): report-only row;
                    # never simulated, never in headline aggregates.
                    fold_rows.append(row)
                    continue
                te_idx = pack["te_idx"]
                sc = pack["scores"][m][pol["side"]]
                res = simulate(D, te_idx, sc, pol, INST[inst])
                for tr_ in res["trade_rows"].itertuples():
                    all_trades.append({
                        **tr_._asdict(), "model": m, "inst": inst,
                        "fold": k,
                        "arm": "control" if is_control else "base"})
                row.update({kk: res[kk] for kk in res
                            if kk != "trade_rows"})
                row["realized_rr"] = realized_rr(res["avg_win"],
                                                 res["avg_loss"])
                row["breakeven_win_rate"] = breakeven_win_rate(
                    res["avg_win"], res["avg_loss"])
                fold_rows.append(row)
                arms["base"].append(res)
                traded_pols.append(pol)
                if m == "hgb":            # stress on primary model only
                    arms["fee1.5"].append(simulate(
                        D, te_idx, sc, pol, INST[inst], fee_mult=1.5))
                    arms["fee2.0"].append(simulate(
                        D, te_idx, sc, pol, INST[inst], fee_mult=2.0))
                    arms["qpx"].append(simulate(
                        D, te_idx, sc, pol, INST[inst], rule="qpx"))
                    arms["tch"].append(simulate(
                        D, te_idx, sc, pol, INST[inst], rule="tch"))
                    arms["lat1s"].append(simulate(
                        D, te_idx, sc, pol, INST[inst], latency_rows=1))

            n_sess_total = sum(pack["n_test_sessions"]
                               for _, pack in packs)
            arms_out = {}
            for a, v in arms.items():
                if a != "base" and not v:
                    continue           # stress arms only when run
                arms_out[a] = aggregate_folds(
                    v, traded_pols, n_sess_total=n_sess_total,
                    n_folds_total=len(packs), fee_rt_ticks=fee_rt,
                    stop_slip_ticks=STOP_SLIP, is_control=is_control)
            rep_i[m] = {"fold_table": fold_rows, "arms": arms_out}
            if is_control:
                report["controls"][inst] = arms_out["base"]
        report["instruments"][inst] = rep_i
    # always (re)write, even when empty: a stale trades file from a
    # previous run must never coexist with a report that traded less
    tr_cols = ["session", "ts", "side", "gross_ticks", "net_ticks",
               "net_usd", "exit", "hold_s", "fill_dt", "adv_1s",
               "model", "inst", "fold", "arm"]
    pd.DataFrame(all_trades, columns=None if all_trades else tr_cols
                 ).drop(columns=["Index"], errors="ignore").to_parquet(
        outdir / "scalper_trades.parquet")
    report["runtime_min"] = round((_time.time() - t0) / 60, 1)
    (outdir / "scalper_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    (outdir / "scalper_report.md").write_text(write_scalper_md(report))
    print(f"wrote scalper_report.json + .md "
          f"({report['runtime_min']} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
