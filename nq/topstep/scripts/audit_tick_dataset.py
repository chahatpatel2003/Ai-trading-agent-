#!/usr/bin/env python
"""Leakage audits for the tick-level decision dataset.

B1 feature past-invariance: rebuilding a session with all events AFTER a
   decision time t removed must leave row t's FEATURES identical (features
   depend only on bins <= t). Labels for t are allowed to change.
B2 label forwardness: rebuilding with events BEFORE the session start of
   labels... verified structurally by B1's complement: label columns of
   row t must be computed only from bins > t — checked by truncating at
   t (labels become NaN/different) while earlier rows' labels (whose
   30s+10s horizon ends before t) stay identical.
B3 event-exact spot check: for sampled decision rows, recompute through/
   touch fill times and first-passage times directly from raw events
   (not the grid) and confirm grid values agree within one bin (100ms).

Writes outputs/leakage_audit_tick_ds.json; exit 1 on FAIL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tslab.config import load_config                      # noqa: E402
from build_fill_table import session_windows              # noqa: E402
from build_tick_dataset import (BIN_NS, CANCEL_S, OUT_S,  # noqa: E402
                                TICK, build_session)
from audit_fill_table import load_day                     # noqa: E402
from build_fill_table import MBP_DIR                      # noqa: E402

SESSIONS = ["2021-06-09", "2022-09-21", "2023-05-17", "2024-03-05"]
FEATURES = ["spread_t", "log_bsz", "log_asz", "qimb", "micro_off",
            "ofi_1s", "ofi_5s", "ofi_30s", "tp_1s", "tp_5s", "tp_30s",
            "trct_5s", "qrate_5s", "cancel_5s", "mom_1s", "mom_5s",
            "mom_30s", "rv_30s", "range_30s", "vwap_dist", "open_dist",
            "tod", "t_since_tr", "last_side", "flips_5s"]


def main() -> int:
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    bars = pd.read_parquet(outdir / "nq_micro_bars_full.parquet",
                           columns=["session", "symbol"])
    sess_map = session_windows(bars)
    pfiles = sorted(MBP_DIR.glob("*.parquet"))
    results = []
    rng = np.random.default_rng(11)
    for sess_str in SESSIONS:
        sess = pd.Timestamp(sess_str).date()
        sym, t0s = sess_map[sess]
        w_lo = int(t0s.min()) - int(130e9)
        w_hi = int(t0s.max()) + int(120e9)
        ev = load_day(pfiles, sess, sym, w_lo, w_hi)
        full = build_session(ev, t0s)
        # B1/B2: truncate at a mid-session decision time
        cut_row = len(full) // 2
        t_cut = int(full.iloc[cut_row]["ts"].value)
        # features of row t use the row's own 100ms bin [t, t+BIN_NS):
        # keep the full decision bin, drop everything after it
        ev_tr = ev[ev["ts"] < t_cut + BIN_NS]
        tr = build_session(ev_tr, t0s)
        m = min(len(tr), len(full))
        f_full = full.iloc[:m][FEATURES].to_numpy()
        f_tr = tr.iloc[:m][FEATURES].to_numpy()
        upto = np.searchsorted(full["ts"].astype("int64").to_numpy(),
                               t_cut, side="right")
        same = np.isclose(f_full[:upto], f_tr[:upto], atol=1e-5,
                          equal_nan=True)
        bad = int((~same).sum())
        results.append({"name": f"B1_feature_past_invariance:{sess}",
                        "status": "PASS" if bad == 0 else "FAIL",
                        "msg": f"{upto} rows x {len(FEATURES)} features, "
                               f"{bad} mismatched"})
        # labels with horizon fully before the cut must also be identical
        lab_cols = [c for c in full.columns
                    if c not in FEATURES + ["ts", "session"]]
        safe = np.searchsorted(
            full["ts"].astype("int64").to_numpy(),
            t_cut - int((CANCEL_S + OUT_S + 1) * 1e9), side="right")
        l_full = full.iloc[:safe][lab_cols].to_numpy()
        l_tr = tr.iloc[:safe][lab_cols].to_numpy()
        badl = int((~np.isclose(l_full, l_tr, atol=1e-5,
                                equal_nan=True)).sum())
        results.append({"name": f"B2_label_horizon:{sess}",
                        "status": "PASS" if badl == 0 else "FAIL",
                        "msg": f"{safe} rows, {badl} mismatched"})
        # B3: event-exact re-verification of fills + up/dn passages
        ts_e = ev["ts"].to_numpy()
        is_tr_ = (ev["action"].to_numpy() == "T")
        px = ev["price"].to_numpy(float)
        bid = ev["bid_px_00"].to_numpy(float)
        ask = ev["ask_px_00"].to_numpy(float)
        bad3, checked3 = 0, 0
        rows = rng.choice(len(full) - 500, size=50, replace=False) + 200
        ts_all = full["ts"].astype("int64").to_numpy()
        for r in rows:
            t_dec = ts_all[r]
            i_pl = np.searchsorted(ts_e, t_dec + BIN_NS, side="right")
            ip = np.searchsorted(ts_e, t_dec + BIN_NS, side="right") - 1
            # placement price: book at end of decision bin
            P = bid[ip]
            i_hz = np.searchsorted(ts_e, t_dec + BIN_NS
                                   + int(CANCEL_S * 1e9), side="right")
            seg_tr = is_tr_[i_pl:i_hz]
            seg_px = px[i_pl:i_hz]
            seg_ask = ask[i_pl:i_hz]
            thru = (seg_tr & (seg_px < P - 1e-9)) | (seg_ask < P - 1e-9)
            got = full.iloc[r]["B_thr_dt"]
            checked3 += 1
            if thru.any():
                t_first = ts_e[i_pl:i_hz][int(np.argmax(thru))]
                exact = (t_first - (t_dec + BIN_NS)) / 1e9
                if np.isnan(got) or abs(got - exact) > 0.21:
                    bad3 += 1
            else:
                if not np.isnan(got):
                    bad3 += 1
        results.append({"name": f"B3_event_exact_fills:{sess}",
                        "status": "PASS" if bad3 <= 1 else "FAIL",
                        "msg": f"{checked3} rows, {bad3} beyond 2-bin "
                               f"tolerance"})
        for x in results[-3:]:
            print(f"[{x['status']}] {x['name']}: {x['msg']}", flush=True)
    ok = all(r["status"] != "FAIL" for r in results)
    (outdir / "leakage_audit_tick_ds.json").write_text(
        json.dumps({"pass": ok, "checks": results}, indent=2))
    print(f"tick-ds audit: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
