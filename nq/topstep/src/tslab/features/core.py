"""Causal feature library over Phase 2B bar frames (NQ or GC).

CONTRACT
  - Input: a bar frame with index bar_open_utc and at least the Phase 2B
    columns (GC frames may lack microstructure columns — those feature
    groups come back as NaN, same schema).
  - A feature row at index T uses ONLY bars with index <= T. Bar T's
    contents are known at info_time = T + 60s, so the feature row at T is
    usable from info_time(T) onward and NEVER at T itself. Use
    latest_usable() to select the correct row for a decision timestamp —
    that is the no-same-bar-leak rule, mechanically enforced.
  - Warm-ups are NaN. Nothing is ever backfilled. Rolling windows never
    cross a contract roll (symbol change); session-anchored quantities
    reset per trade date; prior-session references use strictly earlier
    sessions.

Feature columns are prefixed f_; meta columns (session, symbol, info_time)
are passed through for joining and grouping.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import sessions as S
from ..config import Config

MICRO_COLS = ("vwap", "n_trades", "delta", "buy_vol", "sell_vol", "rv",
              "spread_mean_ticks", "spread_close_ticks", "n_quotes",
              "bid_sz_close", "ask_sz_close")


@dataclass(frozen=True)
class FeatureParams:
    ret_windows: Tuple[int, ...] = (1, 5, 15, 60)
    vol_windows: Tuple[int, ...] = (15, 60)
    atr_window: int = 14
    zscore_windows: Tuple[int, ...] = (30, 120)
    sma_windows: Tuple[int, ...] = (20, 60)
    volume_window: int = 60
    spread_window: int = 120
    delta_windows: Tuple[int, ...] = (5, 15, 30)
    opening_range_minutes: int = 30
    rth_open_ct: time = time(8, 30)


def latest_usable(frame: pd.DataFrame, at: pd.Timestamp) -> Optional[pd.Series]:
    """The most recent row whose info_time <= `at` — the ONLY legal row for
    a decision at `at`. Mid-bar timestamps therefore resolve to the previous
    bar's row, never the still-forming bar's."""
    ok = frame[pd.DatetimeIndex(frame["info_time"]) <= at]
    return None if ok.empty else ok.iloc[-1]


def compute_features(bars: pd.DataFrame, cfg: Config,
                     params: FeatureParams = FeatureParams()) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    df = bars
    p = params
    tick = cfg.instruments[_root_symbol(df)].tick_size
    out = pd.DataFrame(index=df.index)
    out["session"] = df["session"]
    out["symbol"] = df["symbol"]
    out["info_time"] = df["info_time"]

    have = df.columns
    c, h, l = df["close"], df["high"], df["low"]
    lc = pd.Series(np.log(c.to_numpy()), index=df.index)
    # contract blocks: rolling ops must not bleed across a roll
    block = (df["symbol"] != df["symbol"].shift()).cumsum()

    def roll(series, w, fn):
        return series.groupby(block).transform(
            lambda s, w=w, fn=fn: fn(s.rolling(w)))

    def bdiff(series, w):
        return series.groupby(block).transform(lambda s, w=w: s.diff(w))

    # ---- returns ----
    for w in p.ret_windows:
        out[f"f_ret_{w}"] = bdiff(lc, w)

    # ---- volatility / range ----
    prev_c = c.groupby(block).shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()],
                   axis=1).max(axis=1)
    atr = roll(tr, p.atr_window, lambda r: r.mean())
    out["f_atr_ticks"] = atr / tick
    out["f_range_atr"] = (h - l) / atr
    r1 = bdiff(lc, 1)
    for w in p.vol_windows:
        out[f"f_retstd_{w}"] = roll(r1, w, lambda r: r.std())
    if "rv" in have:
        for w in p.vol_windows:
            out[f"f_rv_{w}"] = roll(df["rv"], w, lambda r: r.sum())
    else:
        for w in p.vol_windows:
            out[f"f_rv_{w}"] = np.nan

    # ---- momentum / mean reversion ----
    for w in p.sma_windows:
        sma = roll(c, w, lambda r: r.mean())
        out[f"f_sma_dist_{w}"] = (c - sma) / atr
    for w in p.zscore_windows:
        m = roll(c, w, lambda r: r.mean())
        s = roll(c, w, lambda r: r.std())
        out[f"f_z_{w}"] = (c - m) / s

    # ---- session-anchored ----
    sess = df["session"]
    vol = df["volume"].astype(float)
    vwap_px = df["vwap"] if "vwap" in have else c
    cum_vol = vol.groupby(sess).cumsum()
    svwap = (vwap_px * vol).groupby(sess).cumsum() / cum_vol
    out["f_svwap_dist_atr"] = (c - svwap) / atr
    out["f_sess_cumvol"] = cum_vol
    out["f_vol_ratio"] = vol / roll(vol, p.volume_window, lambda r: r.median())
    if "n_trades" in have:
        nt = df["n_trades"].astype(float)
        m = roll(nt, p.volume_window, lambda r: r.mean())
        s = roll(nt, p.volume_window, lambda r: r.std())
        out["f_ntrades_z"] = (nt - m) / s
    else:
        out["f_ntrades_z"] = np.nan

    # ---- spread / liquidity ----
    if "spread_mean_ticks" in have:
        sp = df["spread_mean_ticks"]
        out["f_spread_ratio"] = sp / roll(sp, p.spread_window,
                                          lambda r: r.median())
        tot = df["bid_sz_close"] + df["ask_sz_close"]
        out["f_book_imb"] = (df["bid_sz_close"] - df["ask_sz_close"]) / tot
        nq_ = df["n_quotes"].astype(float)
        out["f_quote_ratio"] = nq_ / roll(nq_, p.volume_window,
                                          lambda r: r.median())
    else:
        out["f_spread_ratio"] = np.nan
        out["f_book_imb"] = np.nan
        out["f_quote_ratio"] = np.nan

    # ---- microstructure delta ----
    if "delta" in have:
        d = df["delta"].astype(float)
        for w in p.delta_windows:
            out[f"f_delta_{w}"] = (roll(d, w, lambda r: r.sum())
                                   / roll(vol, w, lambda r: r.sum()))
        out["f_cvd_sess"] = d.groupby(sess).cumsum() / cum_vol
        bs = (df["buy_vol"] + df["sell_vol"]).astype(float)
        out["f_buy_share_15"] = (roll(df["buy_vol"].astype(float), 15,
                                      lambda r: r.sum())
                                 / roll(bs, 15, lambda r: r.sum()))
    else:
        for w in p.delta_windows:
            out[f"f_delta_{w}"] = np.nan
        out["f_cvd_sess"] = np.nan
        out["f_buy_share_15"] = np.nan

    # ---- time-of-day / session fields ----
    sc = cfg.sessions
    opens = {s: S.session_open_utc(s, sc) for s in sess.unique()}
    open_utc = sess.map(opens)
    out["f_min_since_open"] = (df.index - pd.DatetimeIndex(open_utc)
                               ).total_seconds() / 60.0
    out["f_min_to_flatten"] = df["secs_to_flatten"] / 60.0
    ct = df.index.tz_convert(sc.timezone)
    out["f_ct_hour"] = ct.hour.astype("int64")
    out["f_dow"] = pd.Series([s.weekday() for s in sess], index=df.index,
                             dtype="int64")
    out["f_rth"] = (df["rth"].astype("int64") if "rth" in have else 0)

    # ---- prior session levels (strictly earlier sessions) ----
    per = df.groupby("session").agg(hi=("high", "max"), lo=("low", "min"),
                                    cl=("close", "last"))
    prior = per.shift(1)
    ph, pl_, pc = (sess.map(prior["hi"]), sess.map(prior["lo"]),
                   sess.map(prior["cl"]))
    out["f_prior_high_dist"] = (c - ph) / atr
    out["f_prior_low_dist"] = (c - pl_) / atr
    out["f_prior_close_gap"] = (c - pc) / atr
    out["f_in_prior_range"] = ((c <= ph) & (c >= pl_)).astype(float).where(
        ph.notna() & pl_.notna())

    # ---- opening range (RTH day of the session; visible only once complete)
    or_start, or_end = _or_window(p)
    day_is_sess = pd.Series(ct.date, index=df.index) == sess
    tt = pd.Series(ct.time, index=df.index)
    in_or = day_is_sess & (tt >= or_start) & (tt < or_end)
    orhi = h.where(in_or).groupby(sess).transform("max")
    orlo = l.where(in_or).groupby(sess).transform("min")
    or_end_utc = sess.map({s: _ct_instant(s, or_end, sc) for s in sess.unique()})
    visible = pd.DatetimeIndex(df["info_time"]) >= pd.DatetimeIndex(or_end_utc)
    out["f_or_high_dist"] = ((c - orhi) / atr).where(visible)
    out["f_or_low_dist"] = ((c - orlo) / atr).where(visible)
    out["f_or_break_up"] = (c > orhi).astype(float).where(visible & orhi.notna())
    out["f_or_break_dn"] = (c < orlo).astype(float).where(visible & orlo.notna())

    fcols = [x for x in out.columns if x.startswith("f_")]
    out[fcols] = out[fcols].replace([np.inf, -np.inf], np.nan)
    return out


def feature_columns(features: pd.DataFrame) -> List[str]:
    return [c for c in features.columns if c.startswith("f_")]


def _root_symbol(df: pd.DataFrame) -> str:
    sym = str(df["symbol"].iloc[0])
    return "GC" if sym.startswith("GC") else "NQ"


def _or_window(p: FeatureParams) -> Tuple[time, time]:
    start = p.rth_open_ct
    end_dt = (datetime(2000, 1, 1, start.hour, start.minute)
              + timedelta(minutes=p.opening_range_minutes))
    return start, end_dt.time()


def _ct_instant(sess_date, t: time, sc) -> pd.Timestamp:
    from zoneinfo import ZoneInfo
    return pd.Timestamp(datetime.combine(sess_date, t,
                                         tzinfo=ZoneInfo(sc.timezone))
                        ).tz_convert("UTC")
