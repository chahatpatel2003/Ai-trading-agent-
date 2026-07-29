#!/usr/bin/env python
"""Tick-level microstructure decision dataset from NQ MBP-1.

Per RTH session (08:30-14:30 CT, front-month symbol from the bar frame):
  1. Aggregate events onto a 100ms grid: L1 state (last/min/max of bid/ask,
     last sizes), trade prints (min/max px, buy/sell aggressor volume,
     count), quote-update count, Cont-style order-flow imbalance, L1 cancel
     volume (size decreases at unchanged price without trades).
  2. Emit DECISION ROWS on a 1s grid. Features use ONLY bins <= t
     (trailing windows). An order placed at decision time t becomes
     effective at bin t+1 (structural 100ms latency).
  3. Labels are strictly forward, per side (B = join bid / long,
     S = join ask / short), cancel horizon 10s, outcome horizon 30s:
       {s}_thr_dt   through-fill time (trade strictly beyond P, or
                    opposite best strictly beyond P) — conservative
       {s}_tch_dt   touch-fill time (trade at/beyond P or opposite best
                    at/beyond P) — optimistic bound
       {s}_qpx_dt   queue-proxy fill: cumulative contra-aggressor volume
                    since placement (in bins whose trades reach P)
                    exceeds the displayed queue at placement; or through
       {s}_up{1,2,3,4}   first time the EXIT-side quote (bid for longs,
                    ask for shorts) moves >= k ticks in favor, from fill
       {s}_dn{4,6,8,12}  first time it moves >= k ticks against
       {s}_mark{1,5,15,30}s  exit-quote minus fill px (ticks) at fill+dt
     plus shared forward mid moves fmid_{1,3,5,10}s (ticks, from bin t+1).

Output: outputs/tick_ds/tickds_<file>.parquet (float32).
Usage: build_tick_dataset.py --file 2024_H1.parquet [--max-days N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tslab.config import load_config                      # noqa: E402
from build_fill_table import MBP_DIR, session_windows     # noqa: E402

TICK = 0.25
BIN_NS = 100_000_000            # 100ms
DEC_EVERY = 10                  # decision every 10 bins = 1s
CANCEL_S = 10.0
OUT_S = 30.0
UPS = (1, 2, 3, 4)
DNS = (4, 6, 8, 12)
MARKS = (1, 5, 15, 30)
COLS = ["ts_event", "action", "side", "price", "size",
        "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "symbol"]


def build_grid(ev: pd.DataFrame, t0: int, n_bins: int) -> dict:
    """Aggregate one session's events onto the 100ms grid."""
    ts = ev["ts"].to_numpy()
    b = np.clip((ts - t0) // BIN_NS, 0, n_bins - 1).astype(np.int64)
    is_tr = (ev["action"].to_numpy() == "T")
    px = ev["price"].to_numpy(float)
    sz = ev["size"].to_numpy(float)
    sd = ev["side"].to_numpy()
    bid = ev["bid_px_00"].to_numpy(float)
    ask = ev["ask_px_00"].to_numpy(float)
    bsz = ev["bid_sz_00"].to_numpy(float)
    asz = ev["ask_sz_00"].to_numpy(float)

    g = {}
    nan = np.full(n_bins, np.nan, np.float64)

    def last_of(v, mask=None):
        out = nan.copy()
        bb, vv = (b, v) if mask is None else (b[mask], v[mask])
        out[bb] = vv                      # later rows overwrite: last wins
        return out

    def minmax_of(v, mask, fn):
        bb, vv = b[mask], v[mask]
        ok = ~np.isnan(vv)
        bb, vv = bb[ok], vv[ok]
        init = np.inf if fn == "min" else -np.inf
        out = np.full(n_bins, init)
        if len(bb):
            if fn == "min":
                np.minimum.at(out, bb, vv)
            else:
                np.maximum.at(out, bb, vv)
        out[np.isinf(out)] = np.nan
        return out

    def sum_of(v, mask):
        out = np.zeros(n_bins)
        bb = b[mask]
        if len(bb):
            np.add.at(out, bb, v[mask])
        return out

    q = ~is_tr
    g["bid_last"] = last_of(bid)
    g["ask_last"] = last_of(ask)
    g["bsz_last"] = last_of(bsz)
    g["asz_last"] = last_of(asz)
    # min/max must init from nan: use at-reductions
    g["bid_min"] = minmax_of(bid, slice(None), "min")
    g["bid_max"] = minmax_of(bid, slice(None), "max")
    g["ask_min"] = minmax_of(ask, slice(None), "min")
    g["ask_max"] = minmax_of(ask, slice(None), "max")
    g["tr_min"] = minmax_of(px, is_tr, "min")
    g["tr_max"] = minmax_of(px, is_tr, "max")
    g["buyv"] = sum_of(sz, is_tr & (sd == "B"))
    g["sellv"] = sum_of(sz, is_tr & (sd == "A"))
    g["ntr"] = sum_of(np.ones_like(sz), is_tr)
    g["nq"] = sum_of(np.ones_like(sz), q)

    # Cont OFI on L1 quote states (event-to-event deltas, summed per bin)
    dbid = np.diff(bid, prepend=bid[0])
    dask = np.diff(ask, prepend=ask[0])
    dbsz = np.diff(bsz, prepend=bsz[0])
    dasz = np.diff(asz, prepend=asz[0])
    ofi_e = (np.where(dbid > 0, bsz, np.where(dbid < 0,
                                              -np.roll(bsz, 1), dbsz))
             - np.where(dask < 0, asz, np.where(dask > 0,
                                                -np.roll(asz, 1), dasz)))
    g["ofi"] = sum_of(ofi_e, slice(None))
    # cancel volume: L1 size drop at unchanged price, no trade on event
    cancel_e = (np.where((dbid == 0) & (dbsz < 0) & ~is_tr, -dbsz, 0.0)
                + np.where((dask == 0) & (dasz < 0) & ~is_tr, -dasz, 0.0))
    g["cancelv"] = sum_of(cancel_e, slice(None))

    # forward-fill state columns (book persists between events)
    for k in ("bid_last", "ask_last", "bsz_last", "asz_last"):
        v = g[k]
        m = np.isnan(v)
        idx = np.where(~m, np.arange(n_bins), 0)
        np.maximum.accumulate(idx, out=idx)
        g[k] = v[idx]
    # bin extremes: where no event in bin, extreme = carried last state
    for k, base in (("bid_min", "bid_last"), ("bid_max", "bid_last"),
                    ("ask_min", "ask_last"), ("ask_max", "ask_last")):
        v = g[k]
        g[k] = np.where(np.isnan(v), g[base], v)
    return g


def roll_sum(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = c[w:] - c[:-w]
    return np.concatenate([c[1:w] - c[0], out])   # partial windows at start


def first_passage(path: np.ndarray, start: np.ndarray, ref: np.ndarray,
                  thr_ticks: float, direction: int, horizon_bins: int
                  ) -> np.ndarray:
    """For each start bin i (aligned with ref px), first dt (s) within
    horizon where direction*(path - ref) >= thr_ticks*TICK. NaN if never.
    Vectorized via strided window view."""
    n = len(path)
    pad = np.full(horizon_bins, np.nan)
    padded = np.concatenate([path, pad])
    win = np.lib.stride_tricks.sliding_window_view(
        padded, horizon_bins)[start + 1]           # bins strictly after i
    hit = direction * (win - ref[:, None]) >= thr_ticks * TICK - 1e-9
    any_ = hit.any(axis=1)
    first = hit.argmax(axis=1)
    return np.where(any_, (first + 1) * (BIN_NS / 1e9), np.nan)


def build_session(ev: pd.DataFrame, t0s: np.ndarray) -> pd.DataFrame:
    t0 = int(t0s.min()) - 120_000_000_000          # 2-min feature pre-roll
    t_end = int(t0s.max()) + 60_000_000_000        # last bar + 60s
    n_bins = int((t_end - t0) // BIN_NS) + int(OUT_S * 10) + 20
    g = build_grid(ev, t0, n_bins)
    mid = (g["bid_last"] + g["ask_last"]) / 2.0
    dec = np.arange(0, n_bins - int((CANCEL_S + OUT_S) * 10) - 15,
                    DEC_EVERY)
    dec = dec[dec >= 1200]                         # decisions from RTH open
    d = {}
    bl, al = g["bid_last"][dec], g["ask_last"][dec]
    bs, as_ = g["bsz_last"][dec], g["asz_last"][dec]
    d["bid"] = bl                       # absolute L1 (for the simulator)
    d["ask"] = al
    d["spread_t"] = (al - bl) / TICK
    d["log_bsz"] = np.log1p(bs)
    d["log_asz"] = np.log1p(as_)
    d["qimb"] = (bs - as_) / np.maximum(bs + as_, 1.0)
    d["micro_off"] = ((as_ * bl + bs * al) / np.maximum(bs + as_, 1.0)
                      - mid[dec]) / TICK
    for w, tag in ((10, "1s"), (50, "5s"), (300, "30s")):
        d[f"ofi_{tag}"] = roll_sum(g["ofi"], w)[dec]
        d[f"tp_{tag}"] = roll_sum(g["buyv"] - g["sellv"], w)[dec]
    d["trct_5s"] = roll_sum(g["ntr"], 50)[dec]
    d["qrate_5s"] = roll_sum(g["nq"], 50)[dec]
    d["cancel_5s"] = roll_sum(g["cancelv"], 50)[dec]
    for w, tag in ((10, "1s"), (50, "5s"), (300, "30s")):
        prev = mid[np.maximum(dec - w, 0)]
        d[f"mom_{tag}"] = (mid[dec] - prev) / TICK
    dm = np.diff(mid, prepend=mid[0]) / TICK
    d["rv_30s"] = roll_sum(dm * dm, 300)[dec]
    mx = pd.Series(mid).rolling(300, min_periods=1).max().to_numpy()
    mn = pd.Series(mid).rolling(300, min_periods=1).min().to_numpy()
    d["range_30s"] = (mx[dec] - mn[dec]) / TICK
    tot_v = g["buyv"] + g["sellv"]
    pv = np.cumsum(np.where(np.isnan(g["tr_min"]), 0.0,
                            (g["tr_min"] + g["tr_max"]) / 2) * tot_v)
    cv = np.cumsum(tot_v)
    vwap = pv / np.maximum(cv, 1.0)
    d["vwap_dist"] = np.where(cv[dec] > 0,
                              (mid[dec] - vwap[dec]) / TICK, 0.0)
    d["open_dist"] = (mid[dec] - mid[300]) / TICK
    d["tod"] = dec / n_bins
    last_tr_bin = np.where(g["ntr"] > 0, np.arange(n_bins), -1)
    np.maximum.accumulate(last_tr_bin, out=last_tr_bin)
    d["t_since_tr"] = np.where(last_tr_bin[dec] >= 0,
                               (dec - last_tr_bin[dec]) / 10.0, 60.0)
    sgn = np.sign(g["buyv"] - g["sellv"])
    lastside = sgn.copy()
    for i in range(1, n_bins):                     # ffill nonzero sign
        if lastside[i] == 0:
            lastside[i] = lastside[i - 1]
    d["last_side"] = lastside[dec]
    midchg = (np.abs(np.diff(mid, prepend=mid[0])) > 1e-9).astype(float)
    d["flips_5s"] = roll_sum(midchg, 50)[dec]

    # ---- labels ----
    hz_c = int(CANCEL_S * 10)
    hz_o = int(OUT_S * 10)
    fills = {}
    for side, tag in ((1, "B"), (-1, "S")):
        P = bl if side > 0 else al
        Q = bs if side > 0 else as_
        opp_ext = g["ask_min"] if side > 0 else g["bid_max"]
        tr_ext = g["tr_min"] if side > 0 else g["tr_max"]
        contrav = g["sellv"] if side > 0 else g["buyv"]
        pad = np.full(hz_c, np.nan)
        opp_w = np.lib.stride_tricks.sliding_window_view(
            np.concatenate([opp_ext, pad]), hz_c)[dec + 1]
        tr_w = np.lib.stride_tricks.sliding_window_view(
            np.concatenate([tr_ext, pad]), hz_c)[dec + 1]
        cv_w = np.lib.stride_tricks.sliding_window_view(
            np.concatenate([contrav, np.zeros(hz_c)]), hz_c)[dec + 1]
        Pk = P[:, None]
        thru = (side * (tr_w - Pk) < -1e-9) | (side * (opp_w - Pk) < -1e-9)
        tch = (side * (tr_w - Pk) < 1e-9) | (side * (opp_w - Pk) < 1e-9)
        d[f"{tag}_thr_dt"] = np.where(
            thru.any(1), (thru.argmax(1) + 1) / 10.0, np.nan)
        d[f"{tag}_tch_dt"] = np.where(
            tch.any(1), (tch.argmax(1) + 1) / 10.0, np.nan)
        # queue proxy: cum contra vol in bins whose trades reach P
        reach = side * (tr_w - Pk) < 1e-9
        cumv = np.cumsum(np.where(reach, cv_w, 0.0), axis=1)
        qhit = cumv > Q[:, None]
        q_dt = np.where(qhit.any(1), (qhit.argmax(1) + 1) / 10.0, np.nan)
        d[f"{tag}_qpx_dt"] = np.fmin(q_dt, d[f"{tag}_thr_dt"])
        # passage/marks measured from THROUGH fill
        fdt = d[f"{tag}_thr_dt"]
        fill_bin = np.where(np.isnan(fdt), -1,
                            dec + 1 + (np.nan_to_num(fdt) * 10 - 1)
                            ).astype(np.int64)
        ok = fill_bin >= 0
        fb = np.where(ok, fill_bin, 0)
        exit_path = g["bid_last"] if side > 0 else g["ask_last"]
        exit_lo = g["bid_min"] if side > 0 else g["ask_min"]
        exit_hi = g["bid_max"] if side > 0 else g["ask_max"]
        fav = exit_hi if side > 0 else exit_lo
        adv = exit_lo if side > 0 else exit_hi
        for k in UPS:
            v = first_passage(fav, fb, P, k, side, hz_o)
            d[f"{tag}_up{k}"] = np.where(ok, v, np.nan)
        for k in DNS:
            v = first_passage(adv, fb, P, k, -side, hz_o)
            d[f"{tag}_dn{k}"] = np.where(ok, v, np.nan)
        for m in MARKS:
            j = np.minimum(fb + m * 10, n_bins - 1)
            d[f"{tag}_mark{m}s"] = np.where(
                ok, side * (exit_path[j] - P) / TICK, np.nan)
        fills[tag] = fdt
    for m in (1, 3, 5, 10):
        j = np.minimum(dec + 1 + m * 10, n_bins - 1)
        d[f"fmid_{m}s"] = (mid[j] - mid[dec + 1]) / TICK
    out = pd.DataFrame({k: np.asarray(v, np.float32)
                        for k, v in d.items()})
    out.insert(0, "ts", pd.to_datetime(t0 + dec * BIN_NS, utc=True))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--max-days", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir / "tick_ds"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"tickds_{args.file.replace('.parquet','')}.parquet"
    if out_path.exists():
        print(f"{out_path} exists — skipping")
        return 0
    bars = pd.read_parquet(cfg.paths.outputs_dir
                           / "nq_micro_bars_full.parquet",
                           columns=["session", "symbol"])
    sess_map = session_windows(bars)
    pf = pq.ParquetFile(MBP_DIR / args.file)
    n_rg = pf.metadata.num_row_groups
    rg_min, rg_max = [], []
    for i in range(n_rg):
        st = pf.metadata.row_group(i).column(0).statistics
        rg_min.append(pd.Timestamp(st.min).value)
        rg_max.append(pd.Timestamp(st.max).value)
    rg_min, rg_max = np.array(rg_min), np.array(rg_max)
    f_lo = pd.Timestamp(rg_min.min(), tz="UTC")
    f_hi = pd.Timestamp(rg_max.max(), tz="UTC")
    todo = []
    for sess, (sym, t0s) in sorted(sess_map.items()):
        w_lo = int(t0s.min()) - int(120e9)         # 2-min pre-roll
        w_hi = int(t0s.max()) + int(120e9)
        if pd.Timestamp(w_lo, tz="UTC") >= f_lo and \
                pd.Timestamp(w_hi, tz="UTC") <= f_hi:
            todo.append((sess, sym, t0s, w_lo, w_hi))
    if args.max_days:
        todo = todo[:args.max_days]
    print(f"{args.file}: {len(todo)} sessions", flush=True)
    frames = []
    import time as _time
    t_start = _time.time()
    for k, (sess, sym, t0s, w_lo, w_hi) in enumerate(todo):
        sel = np.nonzero((rg_max >= w_lo) & (rg_min <= w_hi))[0]
        if sel.size == 0:
            continue
        df = pf.read_row_groups(sel.tolist(), columns=COLS).to_pandas()
        df = df[df["symbol"] == sym]
        ts = df["ts_event"].astype("int64")
        df = df[((ts >= w_lo) & (ts <= w_hi)).to_numpy()]
        if not len(df):
            continue
        df = df.assign(ts=df["ts_event"].astype("int64"))
        df = df.sort_values("ts", kind="mergesort")
        out = build_session(df, t0s)
        out["session"] = sess
        frames.append(out)
        if (k + 1) % 25 == 0:
            el = _time.time() - t_start
            print(f"  [{k+1}/{len(todo)}] {sess} rows="
                  f"{sum(len(f) for f in frames)} "
                  f"({el:.0f}s, {el/(k+1):.1f}s/day)", flush=True)
    res = pd.concat(frames, ignore_index=True)
    res.to_parquet(out_path)
    print(f"wrote {out_path} rows={len(res)} "
          f"({(_time.time()-t_start)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
