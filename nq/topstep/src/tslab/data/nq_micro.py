"""Phase 5A: per-minute MBP-1 microstructure aggregates for NQ.

Augments the gated Phase 3B bars with intra-minute order-book information
(READ-ONLY streaming per session; front-month symbol comes from the already
audited bar file — no new roll logic). Every aggregate for the bar stamped T
uses only quotes/trades inside [T, T+60s): information complete at
info_time = T+60s, same discipline as every other bar column.

Raw aggregate columns (feature layer builds causal f_* on top):
    imb_first/last/mean   signed top-of-book imbalance (b-a)/(b+a)
    sz_bid_mean/sz_ask_mean   mean displayed sizes
    book_flips            sign changes of (bid_sz - ask_sz) within the bar
    q_burst               max quote updates in any 5s sub-bucket
    wide_share            share of updates with spread > 1 tick
    d_burst               max |aggressor delta| in any 10s sub-bucket
"""
from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from .nq_pipeline import read_session_quotes, read_session_trades

MICRO_COLS = ("imb_first", "imb_last", "imb_mean", "sz_bid_mean",
              "sz_ask_mean", "book_flips", "q_burst", "wide_share", "d_burst")


def quote_micro_aggregates(quotes: pd.DataFrame, tick_size: float,
                           bar_seconds: int = 60) -> pd.DataFrame:
    """Per-minute book aggregates from cleaned top-of-book states."""
    if quotes.empty:
        return pd.DataFrame()
    q = quotes.sort_values("ts_event", kind="stable")
    q = q[(q["bid_px_00"] > 0) & (q["ask_px_00"] > 0)
          & (q["ask_px_00"] >= q["bid_px_00"])]
    if q.empty:
        return pd.DataFrame()
    tot = (q["bid_sz_00"] + q["ask_sz_00"]).astype(float)
    imb = (q["bid_sz_00"] - q["ask_sz_00"]) / tot.where(tot > 0)
    spread_ticks = (q["ask_px_00"] - q["bid_px_00"]) / tick_size
    bar = q["ts_event"].dt.floor(f"{bar_seconds}s")
    sub5 = q["ts_event"].dt.floor("5s")
    g = q.assign(bar=bar, imb=imb, wide=spread_ticks > 1.0).groupby("bar",
                                                                   sort=True)
    out = pd.DataFrame({
        "imb_first": g["imb"].first(),
        "imb_last": g["imb"].last(),
        "imb_mean": g["imb"].mean(),
        "sz_bid_mean": g["bid_sz_00"].mean(),
        "sz_ask_mean": g["ask_sz_00"].mean(),
        "wide_share": g["wide"].mean(),
    })
    sign = np.sign((q["bid_sz_00"] - q["ask_sz_00"]).to_numpy())
    flip = np.zeros(len(q))
    flip[1:] = (sign[1:] != sign[:-1]) & (sign[1:] != 0) & (sign[:-1] != 0)
    flip[np.r_[True, (bar.to_numpy()[1:] != bar.to_numpy()[:-1])]] = 0.0
    out["book_flips"] = pd.Series(flip, index=q.index).groupby(
        bar, sort=True).sum()
    per5 = q.assign(bar=bar, sub=sub5).groupby(["bar", "sub"]).size()
    out["q_burst"] = per5.groupby(level=0).max().astype(float)
    out.index.name = "bar_open_utc"
    return out


def trade_burst_aggregates(trades: pd.DataFrame,
                           bar_seconds: int = 60) -> pd.DataFrame:
    """Per-minute max |10s aggressor delta| from trade prints of one symbol."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.sort_values("ts_event", kind="stable")
    sgn = np.where(t["side"] == "B", 1, np.where(t["side"] == "A", -1, 0))
    d = t["size"].astype(float) * sgn
    bar = t["ts_event"].dt.floor(f"{bar_seconds}s")
    sub = t["ts_event"].dt.floor("10s")
    d10 = d.groupby([bar, sub]).sum()
    out = d10.abs().groupby(level=0).max().to_frame("d_burst")
    out.index.name = "bar_open_utc"
    return out


def build_micro_bars(cfg: Config, bars: pd.DataFrame, start: date, end: date,
                     checkpoint_dir: Optional[Path] = None,
                     progress=print,
                     deadline_ts: Optional[float] = None) -> pd.DataFrame:
    """Augment the gated bar frame with micro aggregates for sessions in
    [start, end] (front symbol per session read from the bars themselves)."""
    tick = cfg.instruments["NQ"].tick_size
    sel = bars[(bars["session"] >= start) & (bars["session"] <= end)]
    sessions = sorted(set(sel["session"]))
    front = sel.groupby("session")["symbol"].first()
    frames: List[pd.DataFrame] = []
    for i, s in enumerate(sessions):
        ck = (checkpoint_dir / f"{s}.parquet") if checkpoint_dir else None
        if ck is not None and ck.exists():
            frames.append(pd.read_parquet(ck))
            continue
        if deadline_ts is not None and _time.time() > deadline_ts:
            progress(f"[micro] deadline reached at {i}/{len(sessions)} — "
                     "returning partial (checkpoints preserved)")
            break
        t0 = _time.time()
        sym = front[s]
        quotes = read_session_quotes(cfg, s, sym)
        qa = quote_micro_aggregates(quotes, tick)
        trades = read_session_trades(cfg, s)
        trades = trades[trades["symbol"] == sym]
        ta = trade_burst_aggregates(trades)
        day = sel[sel["session"] == s]
        aug = day.join(qa, how="left").join(ta, how="left")
        if ck is not None:
            ck.parent.mkdir(parents=True, exist_ok=True)
            aug.to_parquet(ck)
        frames.append(aug)
        if i % 10 == 0 or i == len(sessions) - 1:
            progress(f"[micro {i + 1}/{len(sessions)}] {s} sym={sym} "
                     f"({_time.time() - t0:.0f}s)")
    return pd.concat(frames).sort_index(kind="stable")
