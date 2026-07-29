"""Autonomous ML strategy discovery over the audited causal feature sets.

PROTOCOL (leakage-safe by construction):
  - Decision rows: every 5th minute inside the entry window; features are
    the info_time-disciplined columns already gated by truncation audits.
  - Labels: session-bounded (never cross a session), from the audited label
    layer: direction of forward 15/30/60m returns and the 2:1
    target-before-stop outcome (lab_tb) as the payout-friendly target.
  - Walk-forward: EXPANDING half-year folds. ALL selection (model family,
    probability threshold, policy geometry) happens on the last 20% of the
    TRAIN window (validation); the test half-year is touched exactly once.
    1-session purge between train and validation and between train and test.
  - Shuffled-label control per fold-1: test AUC must collapse to ~0.5.
  - Policy: prob >= tau -> enter next bar open with an R-bracket; micro
    sizing under the $1,000 DLL (cap = 1000/max_trades, day stop =
    1000 - cap); engine makes Topstep violations structurally impossible
    (entry_ok + flatten deadline), same conventions as every prior engine.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import sessions as S
from ..config import Config
from ..sim.costs import CostModel
from ..strategy.backtest import Trade

warnings.filterwarnings("ignore")

GRID_MIN = 5
VAL_FRAC = 0.20
TAUS = (0.55, 0.60, 0.65, 0.70)
R_MULTS = (1.0, 1.5, 2.0, 3.0)
STOPS = (4.0, 8.0)          # x 1m ATR
TIME_STOPS = (30, 60)
MAX_TRADES = (2, 3)
LOCKS = (None, 800.0)
DLL = 1000.0


def build_dataset(bars: pd.DataFrame, feats: pd.DataFrame,
                  labels: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Decision-row dataset: X = f_*/fq_* columns, y_* targets, meta."""
    fcols = [c for c in feats.columns if c.startswith(("f_", "fq_"))]
    m = (bars.index.minute % GRID_MIN == 0) & bars["entry_ok"].to_numpy()
    X = feats.loc[m, fcols].astype(float)
    out = X.copy()
    out["y_dir15"] = (labels.loc[m, "lab_fwd_15"] > 0).astype(float).where(
        labels.loc[m, "lab_fwd_15"].notna())
    out["y_dir30"] = (labels.loc[m, "lab_fwd_30"] > 0).astype(float).where(
        labels.loc[m, "lab_fwd_30"].notna())
    out["y_tb"] = (labels.loc[m, "lab_tb"] == 1).astype(float).where(
        labels.loc[m, "lab_tb"].notna())
    out["session"] = bars.loc[m, "session"]
    out["atr_ticks"] = feats.loc[m, "f_atr_ticks"]
    return out


def half_year_folds(sessions: Sequence[date], first_test: Tuple[int, int],
                    last_test: Tuple[int, int]) -> List[Dict]:
    """Expanding train -> half-year test. first/last_test = (year, half)."""
    halves = []
    y, h = first_test
    while (y, h) <= last_test:
        lo = date(y, 1 if h == 1 else 7, 1)
        hi = date(y, 6, 30) if h == 1 else date(y, 12, 31)
        halves.append((lo, hi))
        y, h = (y, 2) if h == 1 else (y + 1, 1)
    out = []
    for lo, hi in halves:
        tr = [s for s in sessions if s < lo]
        te = [s for s in sessions if lo <= s <= hi]
        if len(tr) >= 250 and te:
            out.append({"train": tr[:-1], "test": te,     # 1-session purge
                        "label": f"{lo.year}H{1 if lo.month == 1 else 2}"})
    return out


def model_zoo(seed: int = 7) -> Dict[str, object]:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import lightgbm as lgb
    imp = SimpleImputer(strategy="median")
    return {
        "logreg": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.5,
                               class_weight="balanced")),
        "rf": make_pipeline(imp, RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=200,
            class_weight="balanced", n_jobs=-1, random_state=seed)),
        "hgb": HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.06,
            min_samples_leaf=200, random_state=seed),
        "lgbm": lgb.LGBMClassifier(
            n_estimators=300, num_leaves=63, learning_rate=0.05,
            min_child_samples=200, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbose=-1),
        "mlp": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32,), max_iter=300,
                          early_stopping=True, random_state=seed)),
        "hgb_cal": CalibratedClassifierCV(
            HistGradientBoostingClassifier(
                max_iter=200, max_depth=6, learning_rate=0.08,
                min_samples_leaf=200, random_state=seed),
            method="isotonic", cv=3),
    }


def fit_and_score(ds: pd.DataFrame, fold: Dict, target: str, fcols: List[str]
                  ) -> Dict:
    """Fit the zoo on train[:80%], score AUC on train[80%:] (validation)."""
    from sklearn.metrics import roc_auc_score
    tr_sessions = fold["train"]
    cut = int(len(tr_sessions) * (1 - VAL_FRAC))
    fit_s = set(tr_sessions[:cut - 1])            # 1-session purge before val
    val_s = set(tr_sessions[cut:])
    d = ds[ds["session"].isin(fit_s | val_s)].dropna(subset=[target])
    fit_m = d["session"].isin(fit_s)
    Xf, yf = d.loc[fit_m, fcols], d.loc[fit_m, target]
    Xv, yv = d.loc[~fit_m, fcols], d.loc[~fit_m, target]
    out = {}
    for name, mdl in model_zoo().items():
        try:
            mdl.fit(Xf, yf)
            p = mdl.predict_proba(Xv)[:, 1]
            out[name] = {"model": mdl, "val_auc": float(roc_auc_score(yv, p))}
        except Exception as e:
            out[name] = {"model": None, "val_auc": float("nan"),
                         "error": str(e)}
    return out


# --------------------------------------------------------------------------- #
# signal-driven backtest (same conventions as every engine in this repo)
# --------------------------------------------------------------------------- #

def run_signal_backtest(bars: pd.DataFrame, entries: pd.Series,
                        cfg: Config, instrument: str, stop_atr: float,
                        r_mult: float, time_stop: int, max_trades: int,
                        lock: Optional[float]) -> List[Trade]:
    cm = CostModel(cfg)
    pv = cfg.instruments[instrument].point_value
    tick = cfg.instruments["NQ" if instrument in ("NQ", "MNQ")
                           else "GC"].tick_size
    cap = DLL / max_trades
    day_stop = DLL - cap
    sig = entries.reindex(bars.index).fillna(0).to_numpy()
    entry_ok = bars["entry_ok"].to_numpy()
    atr_pts = None
    trades: List[Trade] = []
    o = bars["open"].to_numpy(float); h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float); c = bars["close"].to_numpy(float)
    idx = bars.index
    info = pd.DatetimeIndex(bars["info_time"])
    sess_vals = bars["session"].to_numpy()
    atr_col = bars["__atr_pts"].to_numpy(float)
    starts: Dict[date, int] = {}
    ends: Dict[date, int] = {}
    for i, s in enumerate(sess_vals):
        starts.setdefault(s, i)
        ends[s] = i
    for s in starts:
        flat = S.eod_flatten_utc(s, cfg.sessions)
        dl = ends[s]
        while dl >= starts[s] and info[dl] >= flat:
            dl -= 1
        if dl <= starts[s]:
            continue
        i = starts[s]
        day_net, n_tr = 0.0, 0
        while i < dl:
            d = int(sig[i])
            if d == 0 or not entry_ok[i]:
                i += 1
                continue
            if n_tr >= max_trades or day_net <= -day_stop \
                    or (lock is not None and day_net >= lock):
                break
            a = atr_col[i]
            if not np.isfinite(a) or a <= 0:
                i += 1
                continue
            e = i + 1
            entry = o[e]
            riskd = stop_atr * a
            stop_px = entry - d * riskd
            tgt_px = entry + d * r_mult * riskd
            risk_per_ct = riskd * pv
            if instrument in ("MNQ", "MGC"):
                qty = int(min(30, cap // risk_per_ct))
                if qty < 1:
                    i += 1
                    continue
            else:
                qty = 1
                if risk_per_ct > cap:
                    i += 1
                    continue
            exit_j, exit_px, reason = dl, c[dl], "flatten"
            mae_px = 0.0
            for j in range(e, dl + 1):
                adverse = (entry - l[j]) if d > 0 else (h[j] - entry)
                mae_px = max(mae_px, adverse)
                hit_s = l[j] <= stop_px if d > 0 else h[j] >= stop_px
                hit_t = h[j] >= tgt_px if d > 0 else l[j] <= tgt_px
                if hit_s:
                    exit_j, exit_px, reason = j, stop_px, "stop"
                    break
                if hit_t:
                    exit_j, exit_px, reason = j, tgt_px, "target"
                    break
                if j - e + 1 >= time_stop:
                    exit_j, exit_px, reason = j, c[j], "time"
                    break
            mae_px = min(mae_px, riskd)
            gross = d * (exit_px - entry) * pv * qty
            costs = cm.side_cost(instrument, qty, "market").total \
                + cm.side_cost(instrument, qty,
                               "stop" if reason == "stop"
                               else "limit" if reason == "target"
                               else "market").total
            trades.append(Trade(
                session=s, entry_time=idx[e], exit_time=idx[exit_j],
                direction=d, qty=qty, entry_px=float(entry),
                exit_px=float(exit_px), exit_reason=reason,
                gross_usd=float(gross), costs_usd=float(costs),
                net_usd=float(gross - costs),
                mae_usd=float(mae_px * pv * qty),
                risk_usd=float(riskd * pv * qty)))
            day_net += gross - costs
            n_tr += 1
            i = exit_j + 1
    return trades


def make_entries(ds_slice: pd.DataFrame, probs: np.ndarray, tau: float,
                 two_sided: bool) -> pd.Series:
    sig = np.zeros(len(ds_slice))
    sig[probs >= tau] = 1
    if two_sided:
        sig[probs <= 1 - tau] = -1
    return pd.Series(sig, index=ds_slice.index)


def policy_search(bars: pd.DataFrame, ds: pd.DataFrame, model, fcols,
                  sessions_val: set, cfg: Config, instrument: str,
                  two_sided: bool) -> Tuple[Dict, float]:
    """Sweep policy on VALIDATION sessions only; return best policy + net."""
    dval = ds[ds["session"].isin(sessions_val)]
    probs = model.predict_proba(dval[fcols])[:, 1]
    bval = bars[bars["session"].isin(sessions_val)]
    best, best_net = None, -math.inf
    for tau in TAUS:
        ent = make_entries(dval, probs, tau, two_sided)
        if (ent != 0).sum() < 10:
            continue
        for stop in STOPS:
            for r in R_MULTS:
                for ts in TIME_STOPS:
                    for mt in MAX_TRADES:
                        for lk in LOCKS:
                            tr = run_signal_backtest(
                                bval, ent, cfg, instrument, stop, r, ts,
                                mt, lk)
                            net = sum(t.net_usd for t in tr)
                            if len(tr) >= 10 and net > best_net:
                                best_net = net
                                best = {"tau": tau, "stop_atr": stop,
                                        "r_mult": r, "time_stop": ts,
                                        "max_trades": mt, "lock": lk,
                                        "val_trades": len(tr)}
    return best, best_net


def distill_rules(model, X: pd.DataFrame, fcols: List[str]) -> str:
    """Depth-3 tree distillation of the model's decision surface."""
    from sklearn.impute import SimpleImputer
    from sklearn.tree import DecisionTreeClassifier, export_text
    p = model.predict_proba(X[fcols])[:, 1]
    y = (p >= np.quantile(p, 0.8)).astype(int)     # its own top-quintile calls
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    Xi = imp.fit_transform(X[fcols])
    t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=500,
                               random_state=3)
    t.fit(Xi, y)
    return export_text(t, feature_names=fcols, max_depth=3)


def feature_importance(model, fcols: List[str]) -> Dict[str, float]:
    try:
        fi = model.feature_importances_
    except AttributeError:
        try:
            fi = model[-1].feature_importances_
        except Exception:
            return {}
    order = np.argsort(fi)[::-1][:15]
    return {fcols[i]: round(float(fi[i]), 4) for i in order}
