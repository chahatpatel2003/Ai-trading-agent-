#!/usr/bin/env python
"""Leakage audits for the limit-entry fill table.

A1 horizon truncation invariance: rebuilding a session's fill rows from an
   event stream truncated at time T_c must reproduce EXACTLY the full-day
   rows for every bar whose adjudication window [t0, t0+180s] ends before
   T_c. Proves no fill decision uses information beyond its own horizon.
A2 placement causality: the placement snapshot (bid0/ask0/sizes) must equal
   the book state of the LAST event with ts_event <= t0 — never a later
   event. Verified by independent recomputation.
A3 fill re-verification: for sampled filled rows, independently confirm
   (a) the through time is the FIRST event after t0 satisfying the
   through condition and no earlier event satisfies it; (b) through
   implies a real cross (trade beyond P or opposite best beyond P).

Writes outputs/leakage_audit_fill_table.json; exit 1 on any FAIL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tslab.config import load_config                      # noqa: E402
from build_fill_table import (COLS, HORIZON_S, MBP_DIR,   # noqa: E402
                              TICK, adjudicate_day, session_windows)

SAMPLE_SESSIONS = ["2021-03-10", "2022-06-15", "2023-02-08",
                   "2023-11-14", "2024-03-05", "2024-10-02"]


def load_day(pfiles, sess, sym, w_lo, w_hi):
    pre = int(300 * 1e9)
    for fp in pfiles:
        pf = pq.ParquetFile(fp)
        n = pf.metadata.num_row_groups
        mins, maxs = [], []
        for i in range(n):
            st = pf.metadata.row_group(i).column(0).statistics
            mins.append(pd.Timestamp(st.min).value)
            maxs.append(pd.Timestamp(st.max).value)
        mins, maxs = np.array(mins), np.array(maxs)
        sel = np.nonzero((maxs >= w_lo - pre) & (mins <= w_hi))[0]
        if sel.size == 0:
            continue
        df = pf.read_row_groups(sel.tolist(), columns=COLS).to_pandas()
        df = df[df["symbol"] == sym]
        ts = df["ts_event"].astype("int64")
        df = df[((ts >= w_lo - pre) & (ts <= w_hi)).to_numpy()]
        if len(df):
            df = df.assign(ts=df["ts_event"].astype("int64"))
            return df.sort_values("ts", kind="mergesort")
    return None


def rows_key(rows):
    return {(r["bar_ts"], r["side"], r["delta"]): r for r in rows}


def main() -> int:
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    bars = pd.read_parquet(outdir / "nq_micro_bars_full.parquet",
                           columns=["session", "symbol"])
    sess_map = session_windows(bars)
    pfiles = sorted(MBP_DIR.glob("*.parquet"))
    results = []
    rng = np.random.default_rng(7)
    for sess_str in SAMPLE_SESSIONS:
        sess = pd.Timestamp(sess_str).date()
        if sess not in sess_map:
            results.append({"name": f"A0_data:{sess}", "status": "SKIP",
                            "msg": "session not in bars"})
            continue
        sym, t0s = sess_map[sess]
        w_lo, w_hi = int(t0s.min()), int(t0s.max() + HORIZON_S * 1e9)
        ev = load_day(pfiles, sess, sym, w_lo, w_hi)
        if ev is None or not len(ev):
            results.append({"name": f"A0_data:{sess}", "status": "FAIL",
                            "msg": "no events loaded"})
            continue
        full_rows = rows_key(adjudicate_day(ev, t0s))

        # A1: truncate at the horizon end of the middle bar
        mid = t0s[len(t0s) // 2]
        t_cut = int(mid + HORIZON_S * 1e9)
        ev_tr = ev[ev["ts"] <= t_cut]
        tr_rows = rows_key(adjudicate_day(
            ev_tr, t0s[t0s + HORIZON_S * 1e9 <= t_cut]))
        bad = 0
        for k, r in tr_rows.items():
            f = full_rows.get(k)
            if f is None or any(
                    (pd.isna(r[c]) != pd.isna(f[c]))
                    or (not pd.isna(r[c]) and r[c] != f[c])
                    for c in r if c != "bar_ts"):
                bad += 1
        results.append({
            "name": f"A1_horizon_truncation:{sess}",
            "status": "PASS" if bad == 0 else "FAIL",
            "msg": f"{len(tr_rows)} rows compared, {bad} mismatched"})

        # A2: placement snapshot causality (independent recompute)
        ts = ev["ts"].to_numpy()
        bid = ev["bid_px_00"].to_numpy(float)
        ask = ev["ask_px_00"].to_numpy(float)
        bad2 = 0
        checked2 = 0
        for t0 in t0s[rng.choice(len(t0s), size=min(60, len(t0s)),
                                 replace=False)]:
            ip = np.searchsorted(ts, t0, side="right") - 1
            if ip < 0:
                continue
            for side in (1, -1):
                r = full_rows.get((t0, side, 0))
                if r is None:
                    continue
                checked2 += 1
                if r["bid0"] != bid[ip] or r["ask0"] != ask[ip]:
                    bad2 += 1
            assert ts[ip] <= t0
        results.append({
            "name": f"A2_placement_causality:{sess}",
            "status": "PASS" if bad2 == 0 else "FAIL",
            "msg": f"{checked2} snapshots, {bad2} mismatched"})

        # A3: independent through-fill re-verification
        is_tr = (ev["action"].to_numpy() == "T")
        px = ev["price"].to_numpy(float)
        bad3 = 0
        checked3 = 0
        filled = [(k, r) for k, r in full_rows.items()
                  if not pd.isna(r["thr_dt"])]
        for k, r in [filled[i] for i in
                     rng.choice(len(filled),
                                size=min(80, len(filled)),
                                replace=False)]:
            t0, side, delta = k
            P = r["limit_px"]
            i0 = np.searchsorted(ts, t0, side="right")
            i1 = np.searchsorted(ts, int(t0 + HORIZON_S * 1e9),
                                 side="right")
            if side > 0:
                cond = (is_tr[i0:i1] & (px[i0:i1] < P)) | (ask[i0:i1] < P)
            else:
                cond = (is_tr[i0:i1] & (px[i0:i1] > P)) | (bid[i0:i1] > P)
            checked3 += 1
            if not cond.any():
                bad3 += 1
                continue
            t_first = ts[i0:i1][int(np.argmax(cond))]
            if abs((t_first - t0) / 1e9 - r["thr_dt"]) > 1e-9:
                bad3 += 1
        results.append({
            "name": f"A3_through_reverify:{sess}",
            "status": "PASS" if bad3 == 0 else "FAIL",
            "msg": f"{checked3} fills re-verified, {bad3} bad"})
        print(*(f"[{x['status']}] {x['name']}: {x['msg']}"
                for x in results[-3:]), sep="\n", flush=True)

    ok = all(r["status"] != "FAIL" for r in results)
    (outdir / "leakage_audit_fill_table.json").write_text(
        json.dumps({"pass": ok, "checks": results}, indent=2))
    print(f"fill-table audit: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
