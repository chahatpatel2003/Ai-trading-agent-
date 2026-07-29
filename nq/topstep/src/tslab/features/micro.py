"""Phase 5A causal microstructure features over micro-augmented bars.

Same contract as features.core: a row at bar-open T uses ONLY bars with
index <= T (raw micro aggregates of bar T are complete at info_time = T+60s);
warm-ups are NaN; rolling ops never cross a contract roll; select decision
rows with latest_usable(). Gate: TruncationInvarianceCheck on the combined
feature function.

Columns (all prefixed f_m_):
    f_m_imb            last top-of-book imbalance of the closed bar
    f_m_imb_chg        change vs the prior bar's last imbalance
    f_m_imb_z          imbalance z-score vs rolling 60 bars
    f_m_press          log(mean bid size / mean ask size)
    f_m_press_z        pressure z-score vs rolling 60
    f_m_flip_z         book-flip count z-score (churn regime)
    f_m_qburst_ratio   5s quote-burst vs rolling median
    f_m_wide_share     share of wide-spread updates in bar
    f_m_dburst_ratio   10s |delta| burst vs rolling median
    f_m_absorb         volume ratio / bar range (absorption: size traded
                       without travel)
    f_m_failed_up      pushed above prior-5-bar high on positive delta but
                       closed back below (failed push)
    f_m_failed_dn      mirror
    f_m_sweep_lo       traded below prior-20-bar low, closed back above
                       (sweep + reclaim)
    f_m_sweep_hi       mirror
    f_m_mom            micro momentum: delta-burst-signed 1-bar return z
    f_m_mr             fast mean reversion: 1-bar return z vs rolling 15
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from .core import FeatureParams, compute_features

MICRO_INPUT = ("imb_first", "imb_last", "imb_mean", "sz_bid_mean",
               "sz_ask_mean", "book_flips", "q_burst", "wide_share",
               "d_burst")


def compute_micro_features(bars: pd.DataFrame, cfg: Config,
                           params: FeatureParams = FeatureParams()
                           ) -> pd.DataFrame:
    """Base 39 features + 16 causal micro features (f_m_*)."""
    out = compute_features(bars, cfg, params)
    block = (bars["symbol"] != bars["symbol"].shift()).cumsum()

    def roll(series, w, fn):
        return series.groupby(block).transform(
            lambda s, w=w, fn=fn: fn(s.rolling(w)))

    def z(series, w):
        m = roll(series, w, lambda r: r.mean())
        sd = roll(series, w, lambda r: r.std())
        return (series - m) / sd

    have = all(c in bars.columns for c in MICRO_INPUT)
    if not have:
        for c in ("f_m_imb", "f_m_imb_chg", "f_m_imb_z", "f_m_press",
                  "f_m_press_z", "f_m_flip_z", "f_m_qburst_ratio",
                  "f_m_wide_share", "f_m_dburst_ratio", "f_m_absorb",
                  "f_m_failed_up", "f_m_failed_dn", "f_m_sweep_lo",
                  "f_m_sweep_hi", "f_m_mom", "f_m_mr"):
            out[c] = np.nan
        return out

    imb = bars["imb_last"].astype(float)
    out["f_m_imb"] = imb
    out["f_m_imb_chg"] = imb.groupby(block).diff()
    out["f_m_imb_z"] = z(imb, 60)
    press = np.log(bars["sz_bid_mean"].astype(float)
                   / bars["sz_ask_mean"].astype(float))
    press = press.replace([np.inf, -np.inf], np.nan)
    out["f_m_press"] = press
    out["f_m_press_z"] = z(press, 60)
    out["f_m_flip_z"] = z(bars["book_flips"].astype(float), 60)
    out["f_m_qburst_ratio"] = (bars["q_burst"].astype(float)
                               / roll(bars["q_burst"].astype(float), 60,
                                      lambda r: r.median()))
    out["f_m_wide_share"] = bars["wide_share"].astype(float)
    out["f_m_dburst_ratio"] = (bars["d_burst"].astype(float)
                               / roll(bars["d_burst"].astype(float), 60,
                                      lambda r: r.median()))
    rng = (bars["high"] - bars["low"]).astype(float)
    tick = cfg.instruments["NQ"].tick_size
    out["f_m_absorb"] = out["f_vol_ratio"] / (rng / tick).clip(lower=1.0)

    hi5 = bars["high"].groupby(block).transform(
        lambda s: s.rolling(5).max().shift(1))
    lo20 = bars["low"].groupby(block).transform(
        lambda s: s.rolling(20).min().shift(1))
    hi20 = bars["high"].groupby(block).transform(
        lambda s: s.rolling(20).max().shift(1))
    lo5 = bars["low"].groupby(block).transform(
        lambda s: s.rolling(5).min().shift(1))
    delta = bars["delta"].astype(float) if "delta" in bars.columns else \
        pd.Series(np.nan, index=bars.index)
    out["f_m_failed_up"] = ((bars["high"] > hi5) & (bars["close"] < hi5)
                            & (delta > 0)).astype(float).where(hi5.notna())
    out["f_m_failed_dn"] = ((bars["low"] < lo5) & (bars["close"] > lo5)
                            & (delta < 0)).astype(float).where(lo5.notna())
    out["f_m_sweep_lo"] = ((bars["low"] < lo20) & (bars["close"] > lo20)
                           ).astype(float).where(lo20.notna())
    out["f_m_sweep_hi"] = ((bars["high"] > hi20) & (bars["close"] < hi20)
                           ).astype(float).where(hi20.notna())

    lc = pd.Series(np.log(bars["close"].to_numpy()), index=bars.index)
    r1 = lc.groupby(block).diff()
    out["f_m_mr"] = r1 / roll(r1, 15, lambda r: r.std())
    out["f_m_mom"] = np.sign(delta) * out["f_m_mr"].abs()

    fcols = [c for c in out.columns if c.startswith("f_")]
    out[fcols] = out[fcols].replace([np.inf, -np.inf], np.nan)
    return out
