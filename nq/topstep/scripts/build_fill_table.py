#!/usr/bin/env python
"""Build the per-bar limit-entry fill table from NQ MBP-1 ticks.

For every RTH entry bar (bar_open_utc in the front-month bar frame, CT time
in [08:30, 14:30]), each side (+1 buy / -1 sell) and each limit offset
delta in {-1, 0, 1, 2} ticks (-1 = improve one tick inside the spread when
spread >= 2 ticks, clamped to join-quote otherwise; 0 = join near quote;
1/2 = pullback below/above), adjudicate a hypothetical limit order placed
at the bar's open instant t0 against the actual quote/trade path:

  placement snapshot   last book state with ts_event <= t0 (standing book)
  through fill         first event t > t0 with (trade price beyond P) or
                       (opposite best strictly beyond P): guaranteed fill
                       for ANY queue position — the conservative rule
  queue fill           delta=0: cumulative contra-aggressor traded volume
                       AT P after t0 exceeds the FULL displayed queue at
                       placement (zero-cancel assumption), or through;
                       delta=-1: first contra trade at/beyond P (we are
                       alone at the improved level), or through;
                       delta>=1: through only (depth beyond L1 unknown)
  post-fill extremes   min/max trade price from fill time to the end of
                       the fill's minute (lets the engine avoid crediting
                       or debiting the pre-fill part of the fill bar)

NO FUTURE QUOTES: adjudication uses only events with ts_event > t0;
the placement snapshot is the book standing AT t0. Horizon capped at
t0 + 180s (3 one-minute bars).

Output: outputs/fill_table/filltab_<file>.parquet, one row per
(bar_open_utc, side, delta).

Usage: build_fill_table.py --file 2024_H1.parquet [--max-days N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tslab.config import load_config                      # noqa: E402

MBP_DIR = Path("/Volumes/T9/nq data/raw/mbp1")
TICK = 0.25
HORIZON_S = 180.0
CT = ZoneInfo("America/Chicago")
COLS = ["ts_event", "action", "side", "price", "size",
        "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "symbol"]
DELTAS = (-1, 0, 1, 2)


def session_windows(bars: pd.DataFrame):
    """session -> (symbol, entry-bar opens ndarray[datetime64 UTC ns])."""
    ct_times = bars.index.tz_convert(CT)
    mask = ((ct_times.hour > 8) | ((ct_times.hour == 8)
                                   & (ct_times.minute >= 30)))
    mask &= ((ct_times.hour < 14) | ((ct_times.hour == 14)
                                     & (ct_times.minute <= 30)))
    b = bars[mask]
    out = {}
    for sess, g in b.groupby("session"):
        out[sess] = (g["symbol"].iloc[0],
                     g.index.tz_convert("UTC").asi8)   # ns since epoch
    return out


def adjudicate_day(ev: pd.DataFrame, t0s: np.ndarray) -> list:
    """ev: this session's front-month events sorted by ts_event (UTC ns
    int64 in col 'ts'), t0s: entry-bar opens (ns). Returns row dicts."""
    ts = ev["ts"].to_numpy()
    is_trade = (ev["action"].to_numpy() == "T")
    px = ev["price"].to_numpy(float)
    sz = ev["size"].to_numpy(float)
    tside = ev["side"].to_numpy()
    bid = ev["bid_px_00"].to_numpy(float)
    ask = ev["ask_px_00"].to_numpy(float)
    bsz = ev["bid_sz_00"].to_numpy(float)
    asz = ev["ask_sz_00"].to_numpy(float)
    rows = []
    hor = int(HORIZON_S * 1e9)
    minute = int(60 * 1e9)
    for t0 in t0s:
        i0 = np.searchsorted(ts, t0, side="right")   # events strictly > t0
        ip = i0 - 1                                  # standing book at t0
        if ip < 0:
            continue
        b0, a0, bs0, as0 = bid[ip], ask[ip], bsz[ip], asz[ip]
        if not (np.isfinite(b0) and np.isfinite(a0)) or a0 <= b0:
            continue
        i1 = np.searchsorted(ts, t0 + hor, side="right")
        if i1 <= i0:
            for side in (1, -1):
                for d in DELTAS:
                    rows.append(_row(t0, side, d, b0, a0, bs0, as0,
                                     None, None, None, None, None, None))
            continue
        s_ts = ts[i0:i1]
        s_tr = is_trade[i0:i1]
        s_px = px[i0:i1]
        s_sz = sz[i0:i1]
        s_sd = tside[i0:i1]
        s_bid = bid[i0:i1]
        s_ask = ask[i0:i1]
        spread_ticks = round((a0 - b0) / TICK)
        for side in (1, -1):
            near = b0 if side > 0 else a0
            for d in DELTAS:
                if d == -1:
                    off = 1 if spread_ticks >= 2 else 0
                    P = near + side * off * TICK
                else:
                    P = near - side * d * TICK
                if side > 0:
                    thru = (s_tr & (s_px < P)) | (s_ask < P)
                    contra = s_tr & (s_sd == "A") & (s_px <= P)
                else:
                    thru = (s_tr & (s_px > P)) | (s_bid > P)
                    contra = s_tr & (s_sd == "B") & (s_px >= P)
                i_thru = int(np.argmax(thru)) if thru.any() else -1
                t_thru = s_ts[i_thru] if i_thru >= 0 else None
                # queue rule
                t_q = None
                if d == -1 and P != near:      # truly improved: alone at P
                    iq = int(np.argmax(contra)) if contra.any() else -1
                    t_q = s_ts[iq] if iq >= 0 else None
                elif d == 0 or (d == -1 and P == near):
                    at_p = s_tr & (s_px == P) & (
                        s_sd == ("A" if side > 0 else "B"))
                    if at_p.any():
                        q0 = bs0 if side > 0 else as0
                        cum = np.cumsum(np.where(at_p, s_sz, 0.0))
                        beat = cum > q0
                        iq = int(np.argmax(beat)) if beat.any() else -1
                        t_q = s_ts[iq] if iq >= 0 else None
                if t_thru is not None and (t_q is None or t_thru < t_q):
                    t_q = t_thru if t_q is None else min(t_q, t_thru)
                pf = {}
                for tag, tf in (("thr", t_thru), ("qf", t_q)):
                    if tf is None:
                        pf[tag] = (None, None, None)
                        continue
                    bar_off = int((tf - t0) // minute)
                    bar_end = t0 + (bar_off + 1) * minute
                    j0 = np.searchsorted(s_ts, tf, side="left")
                    j1 = np.searchsorted(s_ts, bar_end, side="right")
                    seg = s_px[j0:j1][s_tr[j0:j1]]
                    pf[tag] = (bar_off,
                               float(seg.min()) if seg.size else None,
                               float(seg.max()) if seg.size else None)
                rows.append(_row(t0, side, d, b0, a0, bs0, as0,
                                 _secs(t_thru, t0), pf["thr"],
                                 _secs(t_q, t0), pf["qf"], P, spread_ticks))
    return rows


def _secs(t, t0):
    return None if t is None else (t - t0) / 1e9


def _row(t0, side, d, b0, a0, bs0, as0, thr_dt, thr_pf, qf_dt, qf_pf,
         P=None, spread=None):
    thr_pf = thr_pf or (None, None, None)
    qf_pf = qf_pf or (None, None, None)
    return {
        "bar_ts": t0, "side": side, "delta": d, "limit_px": P,
        "bid0": b0, "ask0": a0, "bid_sz0": bs0, "ask_sz0": as0,
        "spread0_ticks": spread,
        "thr_dt": thr_dt, "thr_bar_off": thr_pf[0],
        "thr_pf_min": thr_pf[1], "thr_pf_max": thr_pf[2],
        "qf_dt": qf_dt, "qf_bar_off": qf_pf[0],
        "qf_pf_min": qf_pf[1], "qf_pf_max": qf_pf[2],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--max-days", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir / "fill_table"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"filltab_{args.file.replace('.parquet','')}.parquet"
    if out_path.exists():
        print(f"{out_path} exists — skipping")
        return 0

    bars = pd.read_parquet(cfg.paths.outputs_dir
                           / "nq_micro_bars_full.parquet",
                           columns=["session", "symbol"])
    sess_map = session_windows(bars)

    pf = pq.ParquetFile(MBP_DIR / args.file)
    n_rg = pf.metadata.num_row_groups
    # row-group time ranges from metadata (no decode)
    rg_min, rg_max = [], []
    for i in range(n_rg):
        st = pf.metadata.row_group(i).column(0).statistics
        rg_min.append(pd.Timestamp(st.min).value)
        rg_max.append(pd.Timestamp(st.max).value)
    rg_min = np.array(rg_min)
    rg_max = np.array(rg_max)

    # sessions covered by this file
    f_lo = pd.Timestamp(rg_min.min()).tz_localize("UTC")
    f_hi = pd.Timestamp(rg_max.max()).tz_localize("UTC")
    todo = []
    for sess, (sym, t0s) in sorted(sess_map.items()):
        w_lo, w_hi = t0s.min(), t0s.max() + int(HORIZON_S * 1e9)
        if (pd.Timestamp(w_lo, tz="UTC") >= f_lo
                and pd.Timestamp(w_hi, tz="UTC") <= f_hi):
            todo.append((sess, sym, t0s, w_lo, w_hi))
    if args.max_days:
        todo = todo[:args.max_days]
    print(f"{args.file}: {len(todo)} sessions to adjudicate", flush=True)

    all_rows = []
    import time as _time
    t_start = _time.time()
    for k, (sess, sym, t0s, w_lo, w_hi) in enumerate(todo):
        # snapshot needs the standing book at first t0: pull events a bit
        # earlier so the last-event-<=t0 lookup finds the pre-open book
        pre = int(300 * 1e9)
        sel = np.nonzero((rg_max >= w_lo - pre) & (rg_min <= w_hi))[0]
        if sel.size == 0:
            continue
        tbl = pf.read_row_groups(sel.tolist(), columns=COLS)
        df = tbl.to_pandas()
        df = df[df["symbol"] == sym]
        ts = df["ts_event"].astype("int64")
        m = (ts >= w_lo - pre) & (ts <= w_hi)
        df = df[m.to_numpy()]
        if not len(df):
            continue
        df = df.assign(ts=df["ts_event"].astype("int64"))
        df = df.sort_values("ts", kind="mergesort")
        all_rows += adjudicate_day(df, t0s)
        if (k + 1) % 25 == 0:
            el = _time.time() - t_start
            print(f"  [{k+1}/{len(todo)}] {sess} rows={len(all_rows)} "
                  f"({el:.0f}s, {el/(k+1):.1f}s/day)", flush=True)
    out = pd.DataFrame(all_rows)
    out["bar_ts"] = pd.to_datetime(out["bar_ts"], utc=True)
    out.to_parquet(out_path)
    print(f"wrote {out_path} rows={len(out)} "
          f"({(_time.time()-t_start)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
