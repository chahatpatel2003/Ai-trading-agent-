"""Read-only data loaders with leakage-safe conventions.

Conventions enforced here so every downstream layer inherits them:

  1. All timestamps tz-aware UTC.
  2. Bar timestamp semantics are made EXPLICIT: Databento OHLCV bars are
     stamped with the bar OPEN time. A bar's contents are not knowable until
     its close, so every bar frame gets an `info_time` column = open + bar
     duration. Features must be joined/aligned on `info_time`, never on the
     open stamp; the leakage auditor's truncation test is built to catch
     pipelines that violate this convention.
  3. Loaders never write anything anywhere.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow.parquet as pq

from ..config import Config

BAR_DURATION = {"ohlcv_1m": pd.Timedelta(minutes=1)}


class DataError(Exception):
    pass


def load_gc_bars(cfg: Config,
                 start: Optional[datetime] = None,
                 end: Optional[datetime] = None) -> pd.DataFrame:
    """Load GC continuous-contract 1-minute bars, cleaned and leakage-tagged.

    Returns a frame indexed by UTC bar-open time with columns
    [open, high, low, close, volume, symbol, info_time]. Overlapping source
    files (e.g. a monthly extract inside a yearly file) are deduplicated on
    timestamp, keeping the first occurrence.
    """
    ds = cfg.paths.datasets["gc.ohlcv_1m"]
    files = sorted(ds.path.glob(ds.pattern))
    files = [f for f in files if not f.name.startswith("._")]
    if not files:
        raise DataError(f"no files matching {ds.pattern} in {ds.path} "
                        "(is the T9 drive mounted?)")
    frames: List[pd.DataFrame] = []
    for f in files:
        df = pq.read_table(f).to_pandas()
        if df.index.name == ds.ts_col:
            df = df.reset_index()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    ts = pd.DatetimeIndex(df[ds.ts_col])
    if ts.tz is None:
        raise DataError("expected tz-aware UTC timestamps in GC data")
    df = df.set_index(ts.tz_convert("UTC")).drop(columns=[ds.ts_col])
    df = df.sort_index(kind="stable")     # stable so keep='first' is deterministic
    df = df[~df.index.duplicated(keep="first")]

    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index < pd.Timestamp(end)]

    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise DataError(f"GC bars missing column {col}")

    # information availability time: nothing in this row may be used before it
    df["info_time"] = df.index + BAR_DURATION[ds.kind]
    df.index.name = "bar_open_utc"
    return df


def outright_mask(symbols: pd.Series, spread_marker: str = "-") -> pd.Series:
    """True for outright contract symbols (e.g. NQH4), False for calendar
    spreads (NQH4-NQM4) and other multi-leg symbols."""
    return ~symbols.str.contains(spread_marker, regex=False)


def load_nq_trades(cfg: Config, *args, **kwargs):
    """NOT IMPLEMENTED YET (foundation stage).

    Plan: stream half-year trade files with pyarrow iter_batches, filter to
    outright symbols via outright_mask, select the front month per session by
    PRIOR-session volume (roll decisions may only use information available
    before the session being constructed — no lookahead in roll logic), then
    aggregate to bars/imbalance features. Kept out of the foundation on
    purpose: it must be built together with its leakage tests.
    """
    raise NotImplementedError(load_nq_trades.__doc__)


def load_nq_mbp1(cfg: Config, *args, **kwargs):
    """NOT IMPLEMENTED YET (foundation stage). Same contract as load_nq_trades;
    additionally exposes bid/ask for book-aware fill simulation."""
    raise NotImplementedError(load_nq_mbp1.__doc__)
