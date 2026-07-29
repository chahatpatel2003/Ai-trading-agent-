"""Phase 5B: self-calibrating causal rolling-quantile thresholds.

A threshold for session S is the q-quantile of a feature over the bars of
the PRIOR `n_sessions` trade sessions (all fully closed before S opens) —
constant within S, recomputed each session. Regime-adaptive by construction
and leakage-free: no bar of S (or later) ever influences S's threshold.
The time-of-day variant computes thresholds per Chicago hour bucket.

Indicators are materialized as binary fq_* columns (1.0 / 0.0 / NaN during
the warm-up or where inputs are NaN) so TradeRule filters stay unchanged
(Filter(fq_..., "eq", 1.0)). The whole composition is gated by the same
TruncationInvarianceCheck as every other feature.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from .core import FeatureParams
from .micro import compute_micro_features

N_SESS = 20     # trailing window (sessions)


def trailing_session_quantile(values: pd.Series, sessions: pd.Series,
                              q: float, n_sessions: int = N_SESS) -> pd.Series:
    """Per-bar threshold: q-quantile of `values` over the prior n_sessions
    sessions. NaN until a full window of prior sessions exists."""
    order = list(dict.fromkeys(sessions))          # preserves chronology
    per = {s: v.dropna().to_numpy()
           for s, v in values.groupby(sessions, sort=False)}
    win: deque = deque()
    thr: Dict = {}
    for s in order:
        if len(win) == n_sessions:
            pool = np.concatenate(win) if win else np.array([])
            thr[s] = float(np.quantile(pool, q)) if pool.size else np.nan
        else:
            thr[s] = np.nan
        win.append(per.get(s, np.array([])))
        if len(win) > n_sessions:
            win.popleft()
    return sessions.map(thr).astype(float)


def trailing_tod_quantile(values: pd.Series, sessions: pd.Series,
                          hours: pd.Series, q: float,
                          n_sessions: int = N_SESS) -> pd.Series:
    """Time-of-day normalized variant: thresholds per (session, CT hour),
    from the same hour bucket of the prior n_sessions sessions."""
    order = list(dict.fromkeys(sessions))
    frame = pd.DataFrame({"v": values, "s": sessions, "h": hours})
    per: Dict = {}
    for (s, h), g in frame.groupby(["s", "h"], sort=False):
        per[(s, h)] = g["v"].dropna().to_numpy()
    buckets = sorted(set(hours))
    win: Dict = {h: deque() for h in buckets}
    thr: Dict = {}
    for s in order:
        for h in buckets:
            w = win[h]
            if len(w) == n_sessions:
                pool = np.concatenate(w) if w else np.array([])
                thr[(s, h)] = (float(np.quantile(pool, q))
                               if pool.size else np.nan)
            else:
                thr[(s, h)] = np.nan
            w.append(per.get((s, h), np.array([])))
            if len(w) > n_sessions:
                w.popleft()
    key = pd.Series(list(zip(sessions, hours)), index=values.index)
    return key.map(thr).astype(float)


# feature -> (quantile spec name, q, side) battery
_BATTERY: Tuple[Tuple[str, str, float, str], ...] = (
    # (feature, tag, quantile, side)  side hi: value > thr; lo: value < thr
    ("f_m_imb_z", "hi10", 0.90, "hi"), ("f_m_imb_z", "lo10", 0.10, "lo"),
    ("f_m_imb_z", "hi20", 0.80, "hi"), ("f_m_imb_z", "lo20", 0.20, "lo"),
    ("f_m_imb_chg", "hi10", 0.90, "hi"), ("f_m_imb_chg", "lo10", 0.10, "lo"),
    ("f_m_press_z", "hi10", 0.90, "hi"), ("f_m_press_z", "lo10", 0.10, "lo"),
    ("f_m_qburst_ratio", "hi10", 0.90, "hi"),
    ("f_m_qburst_ratio", "hi20", 0.80, "hi"),
    ("f_m_qburst_ratio", "lo20", 0.20, "lo"),
    ("f_m_dburst_ratio", "hi10", 0.90, "hi"),
    ("f_m_dburst_ratio", "hi20", 0.80, "hi"),
    ("f_m_absorb", "hi10", 0.90, "hi"), ("f_m_absorb", "hi20", 0.80, "hi"),
    ("f_m_mr", "hi10", 0.90, "hi"), ("f_m_mr", "lo10", 0.10, "lo"),
    ("f_m_mr", "hi20", 0.80, "hi"), ("f_m_mr", "lo20", 0.20, "lo"),
    ("f_m_mom", "hi20", 0.80, "hi"), ("f_m_mom", "lo20", 0.20, "lo"),
    ("f_svwap_dist_atr", "hi10", 0.90, "hi"),
    ("f_svwap_dist_atr", "lo10", 0.10, "lo"),
    ("f_svwap_dist_atr", "hi20", 0.80, "hi"),
    ("f_svwap_dist_atr", "lo20", 0.20, "lo"),
    ("f_vol_ratio", "hi20", 0.80, "hi"),
)
_TOD_BATTERY: Tuple[Tuple[str, str, float, str], ...] = (
    ("f_m_qburst_ratio", "hi10_tod", 0.90, "hi"),
    ("f_m_dburst_ratio", "hi10_tod", 0.90, "hi"),
)


# GC has bars only (no MBP-1): its battery runs over BASE features.
_GC_BATTERY: Tuple[Tuple[str, str, float, str], ...] = (
    ("f_svwap_dist_atr", "hi10", 0.90, "hi"),
    ("f_svwap_dist_atr", "lo10", 0.10, "lo"),
    ("f_svwap_dist_atr", "hi20", 0.80, "hi"),
    ("f_svwap_dist_atr", "lo20", 0.20, "lo"),
    ("f_z_30", "hi10", 0.90, "hi"), ("f_z_30", "lo10", 0.10, "lo"),
    ("f_z_120", "hi10", 0.90, "hi"), ("f_z_120", "lo10", 0.10, "lo"),
    ("f_ret_15", "hi20", 0.80, "hi"), ("f_ret_15", "lo20", 0.20, "lo"),
    ("f_vol_ratio", "hi20", 0.80, "hi"),
    ("f_range_atr", "hi20", 0.80, "hi"), ("f_range_atr", "lo20", 0.20, "lo"),
    ("f_retstd_60", "hi20", 0.80, "hi"),
)


def compute_gc_q_features(bars: pd.DataFrame, cfg: Config,
                          params: FeatureParams = None) -> pd.DataFrame:
    """GC/MGC feature function: base causal features (COMEX opening range at
    07:20 CT) + trailing-quantile indicators over base features only.
    Independent of every NQ-specific assumption; same audit gates."""
    from .core import compute_features
    if params is None:
        from datetime import time as _t
        params = FeatureParams(rth_open_ct=_t(7, 20))
    out = compute_features(bars, cfg, params)
    sess = out["session"]
    for feat, tag, q, side in _GC_BATTERY:
        thr = trailing_session_quantile(out[feat], sess, q)
        ind = (out[feat] > thr) if side == "hi" else (out[feat] < thr)
        out[f"fq_{feat[2:]}_{tag}"] = ind.astype(float).where(
            thr.notna() & out[feat].notna())
    return out


def compute_micro_q_features(bars: pd.DataFrame, cfg: Config,
                             params: FeatureParams = FeatureParams()
                             ) -> pd.DataFrame:
    """Base + micro features + fq_* quantile indicators (the full Phase 5B
    feature function; audited as one composition)."""
    out = compute_micro_features(bars, cfg, params)
    sess = out["session"]
    ct_hour = out["f_ct_hour"]
    for feat, tag, q, side in _BATTERY:
        thr = trailing_session_quantile(out[feat], sess, q)
        ind = (out[feat] > thr) if side == "hi" else (out[feat] < thr)
        out[f"fq_{feat[2:]}_{tag}"] = ind.astype(float).where(
            thr.notna() & out[feat].notna())
    for feat, tag, q, side in _TOD_BATTERY:
        thr = trailing_tod_quantile(out[feat], sess, ct_hour, q)
        ind = (out[feat] > thr) if side == "hi" else (out[feat] < thr)
        out[f"fq_{feat[2:]}_{tag}"] = ind.astype(float).where(
            thr.notna() & out[feat].notna())
    return out
