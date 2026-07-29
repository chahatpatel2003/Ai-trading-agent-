"""Label library: session-bounded intraday labels over Phase 2B bar frames.

DESIGN RULES
  - The decision for row T happens at info_time(T); the assumed entry is the
    NEXT bar's open (open[T+1]) — no same-bar fills.
  - Every label is SESSION-BOUNDED: its information window ends at or before
    its session's last usable bar (last bar whose close time is strictly
    before the early-close-aware Topstep flatten deadline). Labels never see
    the next session, a weekend, or a holiday gap — which also pins the
    walk-forward embargo to whole-session boundaries.
  - Each label column has a LabelSpec carrying its max horizon; end-time
    Series (per-row information end) feed LabelHorizonCheck and the fold
    embargo. Unknowable rows are NaN — never filled.

Labels (long convention; short is the mirror image):
  lab_fwd_{h}       log(close[T+h] / close[T]), NaN if T+h leaves the session
  lab_tb            triple barrier: +1 target first, -1 stop first, 0 timeout
                    (both-in-one-bar resolved STOP-FIRST, conservative)
  lab_tb_ret        log return of the barrier trade (entry open[T+1] ->
                    barrier price or timeout close)
  lab_mfe_atr       max favorable excursion over the window, in entry-ATRs
  lab_mae_atr       max adverse excursion over the window, in entry-ATRs
  lab_flat_ret      log(close[deadline bar] / open[T+1]) — the forced
                    hold-to-flatten outcome
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .. import sessions as S
from ..config import Config


@dataclass(frozen=True)
class LabelParams:
    fwd_horizons: Tuple[int, ...] = (5, 15, 30)
    tb_horizon: int = 60              # max bars in the barrier/MFE/MAE window
    tb_target_atr: float = 2.0
    tb_stop_atr: float = 1.0
    atr_window: int = 14


@dataclass(frozen=True)
class LabelSpec:
    name: str
    kind: str                          # "fixed" | "variable"
    horizon_rows: int                  # max bars of future info used
    session_bounded: bool = True
    description: str = ""


@dataclass
class LabelResult:
    labels: pd.DataFrame
    end_times: pd.DataFrame            # per-label information end time
    specs: List[LabelSpec] = field(default_factory=list)


def compute_labels(bars: pd.DataFrame, cfg: Config,
                   params: LabelParams = LabelParams()) -> LabelResult:
    p = params
    n = len(bars)
    idx = bars.index
    info = pd.DatetimeIndex(bars["info_time"])
    close = bars["close"].to_numpy(float)
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    opn = bars["open"].to_numpy(float)

    # causal ATR at decision time (same formula family as the feature layer)
    prev_c = bars["close"].groupby(
        (bars["symbol"] != bars["symbol"].shift()).cumsum()).shift(1)
    tr = pd.concat([(bars["high"] - bars["low"]),
                    (bars["high"] - prev_c).abs(),
                    (bars["low"] - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(p.atr_window).mean().to_numpy(float)

    # per-session positions and flatten-deadline position
    sess_vals = bars["session"].to_numpy()
    starts: Dict[date, int] = {}
    ends: Dict[date, int] = {}
    for i, s in enumerate(sess_vals):
        starts.setdefault(s, i)
        ends[s] = i
    deadline_pos: Dict[date, int] = {}
    for s, lastpos in ends.items():
        flat = S.eod_flatten_utc(s, cfg.sessions)
        j = lastpos
        while j >= starts[s] and info[j] >= flat:   # close time must be < flat
            j -= 1
        deadline_pos[s] = j                          # may be < start (no bars)

    cols = {f"lab_fwd_{h}": np.full(n, np.nan) for h in p.fwd_horizons}
    for name in ("lab_tb", "lab_tb_ret", "lab_mfe_atr", "lab_mae_atr",
                 "lab_flat_ret"):
        cols[name] = np.full(n, np.nan)
    ends_map = {name: np.full(n, np.datetime64("NaT", "ns"))
                for name in list(cols)}
    info_ns = info.tz_convert("UTC").tz_localize(None).to_numpy()

    for i in range(n):
        s = sess_vals[i]
        last = ends[s]
        dl = deadline_pos[s]
        for h in p.fwd_horizons:
            j = i + h
            if j <= last and sess_vals[j] == s:
                cols[f"lab_fwd_{h}"][i] = np.log(close[j] / close[i])
                ends_map[f"lab_fwd_{h}"][i] = info_ns[j]
        if i + 1 > dl or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue                                  # no legal entry bar
        entry = opn[i + 1]
        target = entry + p.tb_target_atr * atr[i]
        stop = entry - p.tb_stop_atr * atr[i]
        jend = min(i + p.tb_horizon, dl)
        hit = 0
        exit_px, exit_j = close[jend], jend
        for j in range(i + 1, jend + 1):
            if low[j] <= stop:                        # stop-first convention
                hit, exit_px, exit_j = -1, stop, j
                break
            if high[j] >= target:
                hit, exit_px, exit_j = +1, target, j
                break
        cols["lab_tb"][i] = hit
        cols["lab_tb_ret"][i] = np.log(exit_px / entry)
        ends_map["lab_tb"][i] = info_ns[exit_j]
        ends_map["lab_tb_ret"][i] = info_ns[exit_j]
        w_hi = high[i + 1: jend + 1].max()
        w_lo = low[i + 1: jend + 1].min()
        cols["lab_mfe_atr"][i] = (w_hi - entry) / atr[i]
        cols["lab_mae_atr"][i] = (entry - w_lo) / atr[i]
        ends_map["lab_mfe_atr"][i] = info_ns[jend]
        ends_map["lab_mae_atr"][i] = info_ns[jend]
        cols["lab_flat_ret"][i] = np.log(close[dl] / entry)
        ends_map["lab_flat_ret"][i] = info_ns[dl]

    labels = pd.DataFrame(cols, index=idx)
    end_times = pd.DataFrame(
        {k: pd.DatetimeIndex(v).tz_localize("UTC") for k, v in ends_map.items()},
        index=idx)

    specs = ([LabelSpec(f"lab_fwd_{h}", "fixed", h,
                        description=f"{h}-bar forward log return, session-bounded")
              for h in p.fwd_horizons]
             + [LabelSpec("lab_tb", "variable", p.tb_horizon,
                          description="triple barrier ±ATR, stop-first ties"),
                LabelSpec("lab_tb_ret", "variable", p.tb_horizon,
                          description="barrier trade log return from open[T+1]"),
                LabelSpec("lab_mfe_atr", "variable", p.tb_horizon,
                          description="max favorable excursion / entry ATR"),
                LabelSpec("lab_mae_atr", "variable", p.tb_horizon,
                          description="max adverse excursion / entry ATR"),
                LabelSpec("lab_flat_ret", "variable", 1440,
                          description="hold to flatten deadline from open[T+1]")])
    return LabelResult(labels=labels, end_times=end_times, specs=specs)


# --------------------------------------------------------------------------- #
# embargo integration
# --------------------------------------------------------------------------- #

def required_embargo(specs: Sequence[LabelSpec]) -> pd.Timedelta:
    """Session-bounded labels cannot leak across a session boundary, so one
    full boundary (>= 24h wall clock: 16:00 close -> next day) is sufficient
    and necessary. Any non-session-bounded spec forces gap-aware handling
    via FoldEmbargoCheck(bar_index=..., horizon_rows=...)."""
    if all(s.session_bounded for s in specs):
        return pd.Timedelta(hours=24)
    raise ValueError("non-session-bounded labels present: use "
                     "FoldEmbargoCheck with bar_index + horizon_rows")


def check_label_purge(end_times: pd.DataFrame, folds: Sequence[Dict]
                      ) -> List[Dict]:
    """Exact, data-driven embargo verification: for every fold, no label
    DECIDED inside the train window may have an information end time past
    train_end (and thus into the embargo/test period). This is the rigorous
    'gap-aware' check for session-bounded labels — durations are measured
    from the actual end times (holiday sessions can span 40h+ wall-clock,
    which naive gap arithmetic mishandles). Returns a list of violations
    (empty = purge-clean)."""
    idx = end_times.index
    problems: List[Dict] = []
    for k, f in enumerate(folds):
        tr = (idx >= f["train_start"]) & (idx < f["train_end"])
        for col in end_times.columns:
            e = end_times.loc[tr, col].dropna()
            n_bad = int((e > f["train_end"]).sum())
            if n_bad:
                problems.append({"fold": k, "label": col, "n_bad": n_bad,
                                 "worst_end": str(e.max())})
    return problems


def make_session_folds(sessions: Sequence[date], cfg: Config,
                       n_train: int, n_test: int,
                       embargo_sessions: int = 1) -> List[Dict]:
    """Walk-forward folds aligned to whole sessions: train on n_train
    sessions, skip embargo_sessions, test on n_test; step by n_test."""
    sc = cfg.sessions
    out: List[Dict] = []
    i = 0
    while i + n_train + embargo_sessions + n_test <= len(sessions):
        tr = sessions[i: i + n_train]
        te = sessions[i + n_train + embargo_sessions:
                      i + n_train + embargo_sessions + n_test]
        out.append({
            "train_start": pd.Timestamp(S.session_open_utc(tr[0], sc)),
            "train_end": pd.Timestamp(S.globex_close_utc(tr[-1], sc)),
            "test_start": pd.Timestamp(S.session_open_utc(te[0], sc)),
            "test_end": pd.Timestamp(S.globex_close_utc(te[-1], sc)),
        })
        i += n_test
    return out
