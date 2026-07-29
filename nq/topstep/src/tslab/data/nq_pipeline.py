"""NQ market-data pipeline: raw Databento trades + MBP-1 -> leakage-safe
1-minute front-month bars with microstructure aggregates.

Design (all reads are READ-ONLY, streamed per session with pyarrow
row-group pruning — never a whole 20 GB file in memory):

  for each trade session S in [start, end]:
      1. read all NQ trades in [session_open(S), globex_close(S))
      2. per-symbol volumes of quarterly outrights -> stored for S+1's roll
      3. front month for S was decided from S-1's volumes ONLY (the first
         requested session uses the session BEFORE the window; a session is
         never selected by its own or later data)
      4. clean trades (causal filters), build 1-min bars for the front month
      5. optionally read MBP-1 quotes for the front month, add quote aggs
      6. stamp session fields + info_time

info_time discipline: a bar stamped with open time T contains information
through T+60s, so info_time = T + bar_seconds is the EARLIEST usable decision
time. Every decision-relevant flag (entry_ok) is evaluated AT info_time.
Downstream joins must use info_time, never the open stamp.

Delta sign convention: Databento `side` is the aggressor side — 'B' = buy
aggressor (lifted the ask), 'A' = sell aggressor (hit the bid), 'N' = unknown.
delta = buy_vol - sell_vol; 'N' volume is in `volume` but in neither leg.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .. import sessions as S
from ..config import Config

QUARTERLY = {"H": 3, "M": 6, "U": 9, "Z": 12}
_CONTRACT_RE = re.compile(r"^NQ([HMUZ])(\d{1,2})$")


# --------------------------------------------------------------------------- #
# params
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PipelineParams:
    bar_seconds: int = 60
    min_size: int = 1
    median_window: int = 50
    max_jump_pct: float = 0.005
    max_spread_ticks: float = 25.0


def load_pipeline_params(configs_dir: Path) -> PipelineParams:
    path = Path(configs_dir) / "pipeline.yaml"
    if not path.exists():
        return PipelineParams()
    raw = (yaml.safe_load(path.read_text()) or {}).get("nq_pipeline", {})
    tf = raw.get("trade_filters", {})
    qf = raw.get("quote_filters", {})
    return PipelineParams(
        bar_seconds=int(raw.get("bar_seconds", 60)),
        min_size=int(tf.get("min_size", 1)),
        median_window=int(tf.get("median_window", 50)),
        max_jump_pct=float(tf.get("max_jump_pct", 0.005)),
        max_spread_ticks=float(qf.get("max_spread_ticks", 25.0)),
    )


# --------------------------------------------------------------------------- #
# contracts and roll
# --------------------------------------------------------------------------- #

def parse_contract(symbol: str, ref_year: int) -> Optional[Tuple[int, int]]:
    """NQH4 -> (2024, 3) resolved against ref_year (contract year is the
    smallest year >= ref_year - 1 with a matching final digit for 1-digit
    codes). Returns None for spreads/non-quarterly/foreign symbols."""
    m = _CONTRACT_RE.match(symbol)
    if not m:
        return None
    month = QUARTERLY[m.group(1)]
    digits = m.group(2)
    if len(digits) == 2:
        return 2000 + int(digits), month
    d = int(digits)
    y = ref_year - 1
    while y % 10 != d:
        y += 1
    return y, month


def is_quarterly_outright(symbol: str) -> bool:
    return _CONTRACT_RE.match(symbol) is not None


class FrontMonthSelector:
    """Selects the front month for session S from session S-1's volumes ONLY.

    Causality contract: select(S) may be called only before observe(S, ...);
    observe() stores S's volumes for use by the NEXT session. Once rolled
    forward the selector never rolls back (monotone contract order)."""

    def __init__(self) -> None:
        self._prior_session: Optional[date] = None
        self._prior_volumes: Optional[Dict[str, int]] = None
        self._current: Optional[str] = None
        self.roll_log: List[Dict] = []

    def select(self, session: date) -> Optional[str]:
        if not self._prior_volumes:
            return None                      # bootstrap: no prior information
        ref = session.year
        best, best_vol = None, -1
        for sym, vol in self._prior_volumes.items():
            c = parse_contract(sym, ref)
            if c is None:
                continue
            if self._current is not None:
                cur = parse_contract(self._current, ref)
                if cur is not None and c < cur:
                    continue                 # never roll backwards
            if vol > best_vol:
                best, best_vol = sym, vol
        if best is None:
            best = self._current
        if best != self._current:
            self.roll_log.append({
                "session": session, "front": best, "previous": self._current,
                "decided_from_session": self._prior_session,
            })
        self._current = best
        return best

    def observe(self, session: date, volumes: Dict[str, int]) -> None:
        self._prior_session = session
        self._prior_volumes = dict(volumes)


# --------------------------------------------------------------------------- #
# session calendar helpers
# --------------------------------------------------------------------------- #

def trade_sessions_in_range(start: date, end: date, sc) -> List[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in sc.holidays:
            out.append(d)
        d += timedelta(days=1)
    return out


def prior_trade_session(d: date, sc) -> date:
    p = d - timedelta(days=1)
    while p.weekday() >= 5 or p in sc.holidays:
        p -= timedelta(days=1)
    return p


# --------------------------------------------------------------------------- #
# cleaning + aggregation (pure functions, unit-tested on synthetic frames)
# --------------------------------------------------------------------------- #

def clean_trades(df: pd.DataFrame, params: PipelineParams) -> Tuple[pd.DataFrame, Dict]:
    """Drop nonpositive/undersized prints and prices > max_jump_pct away from
    the rolling median of STRICTLY PRIOR trades (median shifted by one row —
    causal by construction)."""
    n0 = len(df)
    df = df[(df["price"] > 0) & (df["size"] >= params.min_size)]
    med = (df["price"].rolling(params.median_window, min_periods=5)
           .median().shift(1))
    dev = (df["price"] / med - 1.0).abs()
    keep = dev.isna() | (dev <= params.max_jump_pct)
    out = df[keep]
    return out, {"n_raw": n0, "n_dropped_size_px": n0 - len(df),
                 "n_dropped_jump": int((~keep).sum()), "n_clean": len(out)}


def bars_from_trades(df: pd.DataFrame, params: PipelineParams) -> pd.DataFrame:
    """1-min OHLCV + microstructure from cleaned trades of ONE symbol.
    df: columns ts_event (tz-aware UTC), price, size, side."""
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("ts_event", kind="stable")
    bar = df["ts_event"].dt.floor(f"{params.bar_seconds}s")
    px, sz = df["price"], df["size"].astype("int64")
    buy = sz.where(df["side"] == "B", 0)
    sell = sz.where(df["side"] == "A", 0)
    lp = np.log(px.to_numpy())
    r = np.diff(lp, prepend=lp[0])
    r[~(bar == bar.shift()).to_numpy()] = 0.0        # no cross-bar returns
    g = df.assign(bar=bar, buy=buy, sell=sell, r2=r * r,
                  notional=px * sz).groupby("bar", sort=True)
    out = pd.DataFrame({
        "open": g["price"].first(),
        "high": g["price"].max(),
        "low": g["price"].min(),
        "close": g["price"].last(),
        "volume": g["size"].sum().astype("int64"),
        "n_trades": g["price"].size().astype("int64"),
        "buy_vol": g["buy"].sum().astype("int64"),
        "sell_vol": g["sell"].sum().astype("int64"),
        "rv": g["r2"].sum(),
        "vwap": g["notional"].sum() / g["size"].sum(),
    })
    out["delta"] = out["buy_vol"] - out["sell_vol"]
    out.index.name = "bar_open_utc"
    return out


def clean_quotes(df: pd.DataFrame, params: PipelineParams,
                 tick_size: float) -> Tuple[pd.DataFrame, Dict]:
    """Drop nonpositive and crossed (ask < bid) top-of-book states; tag
    wide spreads (excluded from spread stats, retained for last-quote)."""
    n0 = len(df)
    df = df[(df["bid_px_00"] > 0) & (df["ask_px_00"] > 0)
            & (df["ask_px_00"] >= df["bid_px_00"])]
    df = df.assign(spread_ticks=(df["ask_px_00"] - df["bid_px_00"]) / tick_size)
    df = df.assign(wide=df["spread_ticks"] > params.max_spread_ticks)
    return df, {"n_raw": n0, "n_dropped": n0 - len(df),
                "n_wide": int(df["wide"].sum())}


def quote_aggregates(df: pd.DataFrame, params: PipelineParams) -> pd.DataFrame:
    """Per-minute top-of-book aggregates from cleaned quotes: last state at
    bar close (known by info_time), mean spread over non-wide updates,
    update count."""
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("ts_event", kind="stable")
    bar = df["ts_event"].dt.floor(f"{params.bar_seconds}s")
    g = df.assign(bar=bar).groupby("bar", sort=True)
    ok = df[~df["wide"]].assign(bar=bar[~df["wide"]])
    spread_mean = ok.groupby("bar")["spread_ticks"].mean()
    out = pd.DataFrame({
        "bid_close": g["bid_px_00"].last(),
        "ask_close": g["ask_px_00"].last(),
        "bid_sz_close": g["bid_sz_00"].last().astype("int64"),
        "ask_sz_close": g["ask_sz_00"].last().astype("int64"),
        "spread_close_ticks": g["spread_ticks"].last(),
        "n_quotes": g["spread_ticks"].size().astype("int64"),
    })
    out["spread_mean_ticks"] = spread_mean
    out.index.name = "bar_open_utc"
    return out


def finalize_bars(bars: pd.DataFrame, session: date, symbol: str,
                  cfg: Config, params: PipelineParams) -> pd.DataFrame:
    """Stamp session/time fields. info_time = open + bar_seconds is the
    earliest usable decision time; entry_ok is evaluated AT info_time."""
    sc = cfg.sessions
    out = bars.copy()
    out["session"] = session
    out["symbol"] = symbol
    out["info_time"] = out.index + pd.Timedelta(seconds=params.bar_seconds)
    flatten = S.eod_flatten_utc(session, sc)          # early-close aware
    out["secs_to_flatten"] = (flatten - out["info_time"]).dt.total_seconds()
    out["entry_ok"] = [S.in_entry_window(t.to_pydatetime(), sc)
                       and t < flatten for t in out["info_time"]]
    ct = out.index.tz_convert(sc.timezone)
    rth = cfg.sessions.rth.get("NQ")
    if rth:
        out["rth"] = ((pd.Series(ct.date, index=out.index) == session)
                      & (pd.Series(ct.time, index=out.index) >= rth["start_ct"])
                      & (pd.Series(ct.time, index=out.index) <= rth["end_ct"]))
    return out


# --------------------------------------------------------------------------- #
# streaming reads (READ-ONLY)
# --------------------------------------------------------------------------- #

def _dataset(files: List[Path]):
    import pyarrow.dataset as ds
    return ds.dataset([str(f) for f in files])


def _files(cfg: Config, key: str) -> List[Path]:
    spec = cfg.paths.datasets[key]
    return sorted(p for p in spec.path.glob(spec.pattern)
                  if not p.name.startswith("._"))


def read_session_trades(cfg: Config, session: date) -> pd.DataFrame:
    """All NQ trades of one session window, [session_open, globex_close)."""
    import pyarrow.compute as pc
    lo = S.session_open_utc(session, cfg.sessions)
    hi = S.globex_close_utc(session, cfg.sessions)
    tbl = _dataset(_files(cfg, "nq.trades")).to_table(
        columns=["ts_event", "price", "size", "side", "symbol"],
        filter=(pc.field("ts_event") >= lo) & (pc.field("ts_event") < hi))
    df = tbl.to_pandas()
    df["ts_event"] = pd.DatetimeIndex(df["ts_event"]).tz_convert("UTC")
    return df


def read_session_quotes(cfg: Config, session: date, symbol: str) -> pd.DataFrame:
    import pyarrow.compute as pc
    lo = S.session_open_utc(session, cfg.sessions)
    hi = S.globex_close_utc(session, cfg.sessions)
    tbl = _dataset(_files(cfg, "nq.mbp1")).to_table(
        columns=["ts_event", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"],
        filter=(pc.field("ts_event") >= lo) & (pc.field("ts_event") < hi)
               & (pc.field("symbol") == symbol))
    df = tbl.to_pandas()
    df["ts_event"] = pd.DatetimeIndex(df["ts_event"]).tz_convert("UTC")
    return df


# --------------------------------------------------------------------------- #
# top-level build
# --------------------------------------------------------------------------- #

@dataclass
class BuildResult:
    bars: pd.DataFrame
    roll_log: List[Dict]
    stats: Dict = field(default_factory=dict)


def build_nq_bars(cfg: Config, start: date, end: date,
                  include_quotes: bool = True,
                  params: Optional[PipelineParams] = None) -> BuildResult:
    """Build front-month NQ 1-minute bars for trade sessions in [start, end].

    Reads one extra session BEFORE the window purely to seed the roll rule
    with prior-session volume: no session's front month ever depends on its
    own (or any later) data."""
    params = params or load_pipeline_params(cfg.project_root / "configs")
    sc = cfg.sessions
    S.validate_calendar(start, end, sc)
    sessions = trade_sessions_in_range(start, end, sc)
    if not sessions:
        raise ValueError(f"no trade sessions in {start}..{end}")
    seed = prior_trade_session(sessions[0], sc)
    tick = cfg.instruments["NQ"].tick_size

    selector = FrontMonthSelector()
    frames: List[pd.DataFrame] = []
    stats: Dict = {"sessions": [], "trade_filter": [], "quote_filter": [],
                   "skipped_sessions": []}

    for sess in [seed] + sessions:
        trades = read_session_trades(cfg, sess)
        if trades.empty:
            # e.g. the seed session before the dataset begins; an empty
            # non-bool mask would otherwise select zero COLUMNS, not rows
            outr = trades
            volumes: Dict[str, int] = {}
        else:
            mask = trades["symbol"].map(is_quarterly_outright).astype(bool)
            outr = trades[mask]
            volumes = outr.groupby("symbol")["size"].sum().astype(int).to_dict()

        if sess != seed:
            front = selector.select(sess)
            if front is None or front not in set(outr["symbol"]):
                stats["skipped_sessions"].append(str(sess))
            else:
                fr = outr[outr["symbol"] == front]
                fr, tstats = clean_trades(fr, params)
                bars = bars_from_trades(fr, params)
                if include_quotes and not bars.empty:
                    quotes = read_session_quotes(cfg, sess, front)
                    if not quotes.empty:
                        quotes, qstats = clean_quotes(quotes, params, tick)
                        bars = bars.join(quote_aggregates(quotes, params),
                                         how="left")
                        stats["quote_filter"].append({str(sess): qstats})
                if not bars.empty:
                    frames.append(finalize_bars(bars, sess, front, cfg, params))
                stats["trade_filter"].append({str(sess): tstats})
                stats["sessions"].append(str(sess))
        selector.observe(sess, volumes)

    bars = (pd.concat(frames).sort_index(kind="stable")
            if frames else pd.DataFrame())
    return BuildResult(bars=bars, roll_log=selector.roll_log, stats=stats)
