"""Phase 33 — sub-minute conservative-fill scalp outcome engine.

For each bar i, entry = next bar's open (i+1). Over a forward window capped at ``hold_bars`` AND the
end of the trading day, compute the first-touch offset for every target (under fill margins 0/1/2
ticks = touch / 1-/2-tick-through) and every stop, plus the raw move at the hold limit.
Same-bar target+stop -> STOP wins (conservative). These offsets let any (target, stop, fill-rule,
stop-slip, cost) combination be priced analytically and cheaply afterwards.

Realism: target = exact limit (no slip) but requires the fill margin to actually trade through; stop
and time exits are slipped; entry slipped 1 tick. Forward values are *outcomes* (never
fed back into a signal). Windows never cross the trading day (EOD-flat).
"""

from __future__ import annotations

import numpy as np
import polars as pl

TICK = 0.25
TARGETS = [1, 2, 3, 4, 5, 6, 8]
STOPS = [3, 4, 5, 6, 8, 10]
MARGINS = [0, 1, 2]


def _first_touch(fwd: np.ndarray, level: np.ndarray, *, above: bool) -> np.ndarray:
    """First offset (1..H) where fwd[i, :] crosses level[i]; 0 if none. fwd shape (N, H)."""
    mask = fwd >= level[:, None] if above else fwd <= level[:, None]
    any_hit = mask.any(axis=1)
    first = mask.argmax(axis=1) + 1
    return np.where(any_hit, first, 0).astype(np.int32)


def add_scalp_outcomes(
    bars: pl.DataFrame, *, tf_min: float, hold_min: float = 15.0
) -> pl.DataFrame:
    """Attach first-touch offset columns (``tL_*``/``sL_*``/``hL`` and short mirror) per bar."""
    hold = max(round(hold_min / tf_min), 1)
    n = bars.height
    out: dict[str, np.ndarray] = {}
    for col in (
        [f"tL_{t}_{m}" for t in TARGETS for m in MARGINS]
        + [f"sL_{s}" for s in STOPS]
        + [f"tS_{t}_{m}" for t in TARGETS for m in MARGINS]
        + [f"sS_{s}" for s in STOPS]
    ):
        out[col] = np.zeros(n, dtype=np.int32)
    out["hL"] = np.full(n, np.nan)
    out["hS"] = np.full(n, np.nan)
    out["scalp_valid"] = np.zeros(n, dtype=np.float64)

    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    day = bars["d"].to_numpy()

    start = 0
    for end in [*np.where(day[1:] != day[:-1])[0] + 1, n]:
        _proc(out, o, h, low, c, start, end, hold)
        start = end
    cols = [pl.Series(k, v) for k, v in out.items()]
    return bars.with_columns(cols)


def _proc(
    out: dict[str, np.ndarray], o: np.ndarray, h: np.ndarray, low: np.ndarray, c: np.ndarray,
    s: int, e: int, hold: int,
) -> None:
    if e - s < 3:
        return
    entry = o[s + 1 : e]  # entry[i] = open of bar s+1+i ; aligns to signal bar s+i
    ns = len(entry)  # number of signal bars (last bar of day can't enter)
    # forward highs/lows starting at the entry bar itself (entry bar can hit target/stop too)
    fh = h[s + 1 : e]
    fl = low[s + 1 : e]
    fc = c[s + 1 : e]
    pad = hold
    fh_p = np.concatenate([fh, np.full(pad, -np.inf)])
    fl_p = np.concatenate([fl, np.full(pad, np.inf)])
    swh = np.lib.stride_tricks.sliding_window_view(fh_p, hold)[:ns]
    swl = np.lib.stride_tricks.sliding_window_view(fl_p, hold)[:ns]
    idx = slice(s, s + ns)
    out["scalp_valid"][idx] = 1.0
    for t in TARGETS:
        for m in MARGINS:
            out[f"tL_{t}_{m}"][idx] = _first_touch(swh, entry + t + m * TICK, above=True)
            out[f"tS_{t}_{m}"][idx] = _first_touch(swl, entry - t - m * TICK, above=False)
    for st in STOPS:
        out[f"sL_{st}"][idx] = _first_touch(swl, entry - st, above=False)
        out[f"sS_{st}"][idx] = _first_touch(swh, entry + st, above=True)
    # raw move at the hold limit (last bar in window)
    hold_close = fc[np.minimum(np.arange(ns) + min(hold, ns) - 1, ns - 1)]
    out["hL"][idx] = hold_close - entry
    out["hS"][idx] = entry - hold_close


MNQ_PV, NQ_PV = 2.0, 20.0
MNQ_COMM, NQ_COMM = 0.50, 1.70  # round-trip $/contract


def trade_pnl(
    df: pl.DataFrame,
    *,
    direction: str,
    target: int,
    stop: int,
    margin: int,
    stop_slip_ticks: float = 1.0,
    cost_mult: float = 1.0,
    instrument: str = "MNQ",
) -> np.ndarray:
    """Per-signal-bar $ P&L (1 contract) for a (target, stop, fill-margin, slip, cost) combo.

    Conservative: same-bar target+stop -> STOP. entry slipped 1 tick; stop slipped extra
    ``stop_slip_ticks``; time-exit slipped 1 tick; target = exact limit. ``cost_mult`` scales BOTH
    commission and all slippage (the cost-stress lever).
    """
    pv, comm = (MNQ_PV, MNQ_COMM) if instrument == "MNQ" else (NQ_PV, NQ_COMM)
    is_long = direction == "long"
    tt = df[f"{'tL' if is_long else 'tS'}_{target}_{margin}"].to_numpy()
    st = df[f"{'sL' if is_long else 'sS'}_{stop}"].to_numpy()
    hold_raw = df["hL" if is_long else "hS"].to_numpy()
    entry_slip = TICK * cost_mult
    stop_extra = stop_slip_ticks * TICK * cost_mult
    exit_slip = TICK * cost_mult
    comm_c = comm * cost_mult

    win = (tt > 0) & ((st == 0) | (tt < st))  # target strictly before stop
    loss = (st > 0) & ((tt == 0) | (st <= tt))  # stop first or same bar
    pts = np.where(
        win,
        target - entry_slip,
        np.where(loss, -(stop + stop_extra + entry_slip), hold_raw - entry_slip - exit_slip),
    )
    return pts * pv - comm_c
