"""Phase 57b — enriched PIT-clean GC session features for non-retail discovery.

Adds: roll flags (from raw instrument_id), RTH(pit 08:20-13:30 ET) auction stats,
overnight stats, opening-range features, RTH high/low formation times, normalized
RTH 5-min path matrix. No look-ahead in any forward field. Cached to scratchpad.
"""

from __future__ import annotations

import glob
import os
import tempfile

import numpy as np
import polars as pl

SCR = os.environ.get("NQLAB_SCRATCH", os.path.join(tempfile.gettempdir(), "nqlab-phase57"))
os.makedirs(SCR, exist_ok=True)
CACHE5 = f"{SCR}/gc_5m.parquet"
ROLLMAP = f"{SCR}/gc_rollmap.parquet"
FEAT = f"{SCR}/gc_sessfeat.parquet"
PATHS = f"{SCR}/gc_rthpaths.npz"

NY_OPEN, RTH_END = 860, 1170      # smin: 08:20 ET, 13:30 ET (pit)
ON_START = 60                      # 19:00 ET overnight start


def build_rollmap():
    files = sorted(f for f in glob.glob(os.path.expanduser("~/Desktop/gc/*.parquet"))
                   if "2024_jan" not in os.path.basename(f))
    parts = []
    for f in files:
        d = pl.read_parquet(f, columns=["ts_event", "instrument_id"])
        d = d.with_columns(pl.col("ts_event").dt.convert_time_zone("America/New_York").alias("et"))
        d = d.with_columns((pl.col("et") + pl.duration(hours=6)).dt.date().alias("session_date"))
        parts.append(d.group_by("session_date").agg(
            pl.col("instrument_id").mode().first().alias("iid")))
    m = pl.concat(parts).unique(subset=["session_date"]).sort("session_date")
    m = m.with_columns((pl.col("iid") != pl.col("iid").shift(1)).fill_null(True).alias("roll"))
    m.write_parquet(ROLLMAP)
    print("rollmap sessions:", m.height, "roll days:", int(m["roll"].sum()))
    return m


def build_features():
    g = pl.read_parquet(CACHE5).sort("b5")
    roll = pl.read_parquet(ROLLMAP)
    sess = []
    paths = []
    prev = None
    rows = []
    for (sd,), d in g.group_by(["session_date"], maintain_order=True):
        d = d.sort("smin")
        sm = d["smin"].to_numpy()
        o, h, l, c, v = (d[x].to_numpy() for x in ["open", "high", "low", "close", "volume"])
        v = v.astype(float)
        rth = (sm >= NY_OPEN) & (sm <= RTH_END)
        on = (sm >= ON_START) & (sm < NY_OPEN)
        if rth.sum() < 40 or on.sum() < 30:        # need full pit + overnight
            prev = None
            continue
        rh, rl = float(h[rth].max()), float(l[rth].min())
        ropen = float(o[rth][0])
        rclose = float(c[rth][-1])
        # high/low formation time (smin of RTH extreme)
        rsm = sm[rth]
        hi_t = int(rsm[np.argmax(h[rth])])
        lo_t = int(rsm[np.argmin(l[rth])])
        onh, onl = float(h[on].max()), float(l[on].min())
        # opening ranges from open
        def orr(mins):
            mask = (sm >= NY_OPEN) & (sm < NY_OPEN + mins)
            return (float(h[mask].max()), float(l[mask].min())) if mask.any() else (ropen, ropen)
        or15h, or15l = orr(15)
        or30h, or30l = orr(30)
        or60h, or60l = orr(60)
        # normalized RTH path: close at each 5-min step relative to ropen, on fixed grid
        grid = np.arange(NY_OPEN, RTH_END + 1, 5)
        cl_on_grid = np.interp(grid, sm[rth], c[rth])
        rrow = {
            "session_date": sd,
            "dow": sd.weekday(),
            "ropen": ropen, "rhigh": rh, "rlow": rl, "rclose": rclose,
            "rrange": rh - rl,
            "hi_t": hi_t, "lo_t": lo_t,
            "onh": onh, "onl": onl, "onrange": onh - onl,
            "or15h": or15h, "or15l": or15l, "or15": or15h - or15l,
            "or30h": or30h, "or30l": or30l, "or30": or30h - or30l,
            "or60h": or60h, "or60l": or60l, "or60": or60h - or60l,
            "rvol": float(v[rth].sum()),
            "onvol": float(v[on].sum()),
            "prev_rclose": prev[3] if prev else None,
            "prev_rhigh": prev[1] if prev else None,
            "prev_rlow": prev[2] if prev else None,
            "prev_rrange": (prev[1] - prev[2]) if prev else None,
        }
        rows.append(rrow)
        paths.append(cl_on_grid - ropen)   # $ path vs open
        prev = (ropen, rh, rl, rclose)
    df = pl.DataFrame(rows)
    df = df.join(roll.select(["session_date", "roll", "iid"]), on="session_date", how="left")
    # ATR(10) of rrange, shifted (no lookahead)
    rr = df["rrange"].to_numpy()
    atr = np.array([np.nan if i == 0 else np.mean(rr[max(0, i - 10):i]) for i in range(len(rr))])
    atr = np.where(np.isnan(atr), np.nanmedian(rr), atr)
    df = df.with_columns(pl.Series("atr", np.maximum(atr, 1.0)))
    df = df.with_columns(pl.col("session_date").dt.year().alias("year"))
    df.write_parquet(FEAT)
    P = np.array(paths)
    np.savez_compressed(PATHS, paths=P, dates=np.array([str(x) for x in df["session_date"]]),
                        atr=df["atr"].to_numpy(), grid=np.arange(NY_OPEN, RTH_END + 1, 5))
    print("feature sessions:", df.height, "path matrix:", P.shape)
    print("cols:", df.columns)
    return df


if __name__ == "__main__":
    build_rollmap()
    build_features()
