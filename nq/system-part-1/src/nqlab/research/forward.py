"""PIT-safe forward-outcome engine (Phase 32) — measure what happens AFTER each bar.

This is *measurement*, not a trading decision: forward MFE/MAE and target-before-stop deliberately
look at FUTURE bars to characterize how cleanly NQ moves after a (PIT-safe) condition. The contract:
forward columns are all prefixed ``fwd_`` and must NEVER be used as a condition/signal — only the
PIT-safe feature columns may condition. Forward windows are capped at the END OF THE TRADING DAY
(EOD-flat), and entry is the NEXT bar's open (the first tradeable price after the condition bar).

Per condition bar i (minute_of_day in [lo, 955]):
  entry = open[i+1]; forward bars = i+1 .. last bar of the day with minute<=960.
  fwd_up_{m}/fwd_dn_{m} = best favorable up / down excursion (points) over m-minute horizon + EOD.
  fwd_ret_{m} = close-at-horizon minus entry (long-perspective signed return).
  fwd_L{T}/fwd_S{T} = 1 if the long/short +T point target is reached BEFORE the -T stop (same-bar
    double-touch counts as the STOP -> conservative), for T in {10,20,30,50}; fwd_*{T}t = bars-to-
    resolve (-1 if neither by EOD). fwd_Lheat20/fwd_Sheat20 = adverse heat (points) before the +20
    target when it is reached.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

HORIZONS = [("5", 1), ("10", 2), ("15", 3), ("30", 6), ("60", 12), ("120", 24)]
TARGETS = [10, 20, 30, 50]
COND_MIN_LO = 420  # 07:00 ET — condition bars from late-premarket through RTH
COND_MIN_HI = 955  # last entry 15:50 ET
DAY_END_MAX = 960  # forward window capped at 16:00 ET (EOD-flat)


def _fwd_cols() -> list[str]:
    cols: list[str] = ["fwd_valid"]
    for m, _ in HORIZONS:
        cols += [f"fwd_up_{m}", f"fwd_dn_{m}", f"fwd_ret_{m}"]
    cols += ["fwd_up_eod", "fwd_dn_eod", "fwd_ret_eod"]
    for t in TARGETS:
        cols += [f"fwd_L{t}", f"fwd_S{t}", f"fwd_L{t}t", f"fwd_S{t}t"]
    cols += ["fwd_Lheat20", "fwd_Sheat20"]
    return cols


def add_forward_outcomes(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach ``fwd_*`` outcome columns. ``frame`` needs ts + OHLC + calendar cols."""
    frame = frame.sort("ts").filter(
        (pl.col("minute_of_day") >= 0) & (pl.col("minute_of_day") <= DAY_END_MAX)
    )
    n = frame.height
    out: dict[str, np.ndarray] = {c: np.full(n, np.nan) for c in _fwd_cols()}
    out["fwd_valid"] = np.zeros(n, dtype=np.float64)
    o = frame["open"].to_numpy()
    h = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    c = frame["close"].to_numpy()
    minute = frame["minute_of_day"].to_numpy()
    day = frame["trading_day"].to_numpy()

    # day boundaries
    start = 0
    for end in [*np.where(day[1:] != day[:-1])[0] + 1, n]:
        _process_day(out, o, h, low, c, minute, start, end)
        start = end
    return frame.with_columns([pl.Series(k, v) for k, v in out.items()])


def _process_day(
    out: dict[str, np.ndarray], o: np.ndarray, h: np.ndarray, low: np.ndarray, c: np.ndarray,
    minute: np.ndarray, s: int, e: int,
) -> None:
    for i in range(s, e - 1):
        if not (COND_MIN_LO <= minute[i] <= COND_MIN_HI):
            continue
        entry = o[i + 1]
        j0, j1 = i + 1, e  # forward bars [j0, j1)
        if j0 >= j1:
            continue
        out["fwd_valid"][i] = 1.0
        fh = h[j0:j1]
        fl = low[j0:j1]
        fc = c[j0:j1]
        # horizon excursions / returns
        for m, bars in HORIZONS:
            k = min(bars, len(fh))
            out[f"fwd_up_{m}"][i] = fh[:k].max() - entry
            out[f"fwd_dn_{m}"][i] = entry - fl[:k].min()
            out[f"fwd_ret_{m}"][i] = fc[k - 1] - entry
        out["fwd_up_eod"][i] = fh.max() - entry
        out["fwd_dn_eod"][i] = entry - fl.min()
        out["fwd_ret_eod"][i] = fc[-1] - entry
        # single forward scan resolving all target-before-stop levels (both directions)
        l_done = {t: False for t in TARGETS}
        s_done = {t: False for t in TARGETS}
        min_low = entry
        max_high = entry
        for off in range(len(fh)):
            hi_j, lo_j = fh[off], fl[off]
            min_low = min(min_low, lo_j)
            max_high = max(max_high, hi_j)
            for t in TARGETS:
                if not l_done[t]:
                    if lo_j <= entry - t:  # stop first (conservative)
                        out[f"fwd_L{t}"][i] = 0.0
                        out[f"fwd_L{t}t"][i] = off + 1
                        l_done[t] = True
                    elif hi_j >= entry + t:
                        out[f"fwd_L{t}"][i] = 1.0
                        out[f"fwd_L{t}t"][i] = off + 1
                        if t == 20:
                            out["fwd_Lheat20"][i] = entry - min_low
                        l_done[t] = True
                if not s_done[t]:
                    if hi_j >= entry + t:  # short stop first (conservative)
                        out[f"fwd_S{t}"][i] = 0.0
                        out[f"fwd_S{t}t"][i] = off + 1
                        s_done[t] = True
                    elif lo_j <= entry - t:
                        out[f"fwd_S{t}"][i] = 1.0
                        out[f"fwd_S{t}t"][i] = off + 1
                        if t == 20:
                            out["fwd_Sheat20"][i] = max_high - entry
                        s_done[t] = True
            if all(l_done.values()) and all(s_done.values()):
                break
        for (
            t
        ) in TARGETS:  # neither hit by EOD -> mark time -1 (value left NaN -> treat as no-touch)
            if not l_done[t]:
                out[f"fwd_L{t}t"][i] = -1.0
                out[f"fwd_L{t}"][i] = 0.0
            if not s_done[t]:
                out[f"fwd_S{t}t"][i] = -1.0
                out[f"fwd_S{t}"][i] = 0.0


def ensure_forward_frame(vp_frame_path: Path, out_path: Path) -> pl.DataFrame:
    """Build (or load cached) the VP frame + forward outcomes."""
    if out_path.exists():
        return pl.read_parquet(out_path)
    fr = add_forward_outcomes(pl.read_parquet(vp_frame_path))
    fr.write_parquet(out_path)
    return fr
