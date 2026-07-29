"""GC compatibility layer: adapt the continuous GC.v.0 1-minute OHLCV bars to
the exact Phase 2B bar schema so the feature/label library and the pipeline
gate run unchanged.

Deliberately NOT optimized yet (Phase 2C scope): no tick-level microstructure
exists for GC in the data, so vwap falls back to close and the trades/quote
columns are absent — compute_features() degrades those groups to NaN with an
identical schema. The GC volatility contract caps (3/6/9) live in the
simulator layer, not here.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from .. import sessions as S
from ..config import Config
from .loader import load_gc_bars


def gc_bars_from_ohlcv(raw: pd.DataFrame, cfg: Config,
                       start: date, end: date) -> pd.DataFrame:
    """Pure adapter: loader-style OHLCV frame (UTC bar-open index, columns
    open/high/low/close/volume/symbol[/info_time]) -> Phase 2B bar schema,
    restricted to trade sessions in [start, end].

    Session assignment uses the holiday-aware calendar: bars during a halt
    holiday (e.g. MLK Monday) book to the next trade date, exactly like NQ.
    """
    sc = cfg.sessions
    S.validate_calendar(start, end, sc)
    df = raw.sort_index(kind="stable")
    df = df[~df.index.duplicated(keep="first")]

    sess = pd.Series([S.session_date(t.to_pydatetime(), sc) for t in df.index],
                     index=df.index)
    keep = (sess >= start) & (sess <= end)
    df, sess = df[keep], sess[keep]
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    for col in ("open", "high", "low", "close"):
        out[col] = df[col].astype(float)
    out["volume"] = df["volume"].astype("int64")
    out["session"] = sess
    out["symbol"] = df["symbol"].astype(str)
    out["info_time"] = out.index + pd.Timedelta(seconds=60)

    flat = sess.map({s: S.eod_flatten_utc(s, sc) for s in sess.unique()})
    out["secs_to_flatten"] = (pd.DatetimeIndex(flat)
                              - pd.DatetimeIndex(out["info_time"])
                              ).total_seconds()
    out["entry_ok"] = [
        S.in_entry_window(t.to_pydatetime(), sc) and t < f
        for t, f in zip(pd.DatetimeIndex(out["info_time"]),
                        pd.DatetimeIndex(flat))]
    ct = out.index.tz_convert(sc.timezone)
    rth = sc.rth.get("GC")
    if rth:
        out["rth"] = ((pd.Series(ct.date, index=out.index) == sess)
                      & (pd.Series(ct.time, index=out.index) >= rth["start_ct"])
                      & (pd.Series(ct.time, index=out.index) <= rth["end_ct"]))
    else:
        out["rth"] = False

    # a session's bars must sit inside its window (drops e.g. stray prints
    # outside Globex hours; none observed in the data, guarded anyway)
    lo = pd.DatetimeIndex(sess.map({s: S.session_open_utc(s, sc)
                                    for s in sess.unique()}))
    hi = pd.DatetimeIndex(sess.map({s: S.globex_close_utc(s, sc)
                                    for s in sess.unique()}))
    out = out[(out.index >= lo) & (out.index < hi)]
    return out


def build_gc_bars(cfg: Config, start: date, end: date) -> pd.DataFrame:
    """Load GC bars from the T9 drive (read-only) and adapt. The loader
    dedups the known gc_2024_jan/gc_2024 overlap."""
    lo = pd.Timestamp(start, tz="UTC") - pd.Timedelta(days=3)
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=2)
    raw = load_gc_bars(cfg, start=lo, end=hi)
    return gc_bars_from_ohlcv(raw, cfg, start, end)
