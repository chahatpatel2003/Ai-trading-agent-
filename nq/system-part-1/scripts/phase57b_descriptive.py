"""Phase 57b — descriptive (unsupervised) discovery on PIT-clean GC. No trading,
just let recurring structure surface. Roll days excluded."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import polars as pl

SCR = os.environ.get("NQLAB_SCRATCH", os.path.join(tempfile.gettempdir(), "nqlab-phase57"))
g = pl.read_parquet(f"{SCR}/gc_5m.parquet").sort("b5")
feat = pl.read_parquet(f"{SCR}/gc_sessfeat.parquet")
clean_dates = set(feat.filter(~pl.col("roll"))["session_date"].to_list())


def smin_to_et(sm):
    etmod = (sm + 1080) % 1440
    return f"{etmod // 60:02d}:{etmod % 60:02d}"


# ---------- Module A: time-of-day vol + drift map ----------
gg = g.filter(pl.col("session_date").is_in(list(clean_dates)))
gg = gg.with_columns(((pl.col("close") - pl.col("open"))).alias("ret"))
tod = gg.group_by("smin").agg(
    pl.col("ret").abs().mean().alias("absret"),
    pl.col("ret").mean().alias("meanret"),
    (pl.col("ret") > 0).mean().alias("upfreq"),
    pl.col("volume").mean().alias("vol"),
    pl.len().alias("n"),
).sort("smin")
print("=== TIME-OF-DAY MAP (5-min, PIT-clean) — top 14 highest-volatility windows ===")
print(f"{'ET':>6s} {'absret$':>8s} {'meanret$':>9s} {'upfreq':>7s} {'avgvol':>8s}")
top = tod.sort("absret", descending=True).head(14).sort("smin")
for r in top.iter_rows(named=True):
    print(f"{smin_to_et(r['smin']):>6s} {r['absret']*10:>8.2f} {r['meanret']*10:>9.3f} "
          f"{r['upfreq']:>7.3f} {r['vol']:>8.0f}")
print("\n=== Lowest-volatility windows (compression zones), RTH only ===")
rthtod = tod.filter((pl.col("smin") >= 860) & (pl.col("smin") <= 1170))
for r in rthtod.sort("absret").head(6).iter_rows(named=True):
    print(f"  {smin_to_et(r['smin'])} absret=${r['absret']*10:.2f}")

# ---------- high/low formation times ----------
fc = feat.filter(~pl.col("roll"))
print("\n=== RTH HIGH/LOW FORMATION TIME (PIT-clean) ===")
for lab, col in [("HIGH", "hi_t"), ("LOW", "lo_t")]:
    vc = fc.group_by(col).agg(pl.len().alias("n")).sort("n", descending=True).head(6)
    s = "  ".join(f"{smin_to_et(r[col])}:{r['n']}" for r in vc.iter_rows(named=True))
    print(f"  {lab:5s} most common: {s}")
# fraction of high/low formed in first hour vs last hour of pit
for lab, col in [("HIGH", "hi_t"), ("LOW", "lo_t")]:
    early = fc.filter(pl.col(col) < 920).height / fc.height     # first ~60min (08:20-09:20)
    late = fc.filter(pl.col(col) > 1110).height / fc.height     # last ~60min (12:30-13:30)
    print(f"  {lab}: formed in first hr {early:.1%}, last hr {late:.1%}")

# ---------- reversal/continuation: open30 dir vs rest-of-day ----------
fc2 = fc.with_columns(
    (pl.col("or30h") - pl.col("ropen")).alias("up30"),
    (pl.col("ropen") - pl.col("or30l")).alias("dn30"),
)
fc2 = fc2.with_columns(
    pl.when((pl.col("or30h") - pl.col("ropen")) >= (pl.col("ropen") - pl.col("or30l")))
    .then(1).otherwise(-1).alias("dir30"),
    (pl.col("rclose") - pl.col("ropen")).alias("dayret"),
)
# does early direction continue?
cont = fc2.with_columns((pl.col("dir30") * pl.col("dayret")).alias("align"))
print("\n=== EARLY(30m) DIRECTION vs FULL-DAY (continuation tendency) ===")
print(f"  P(day closes in 30m-open direction) = {(cont['align']>0).mean():.3f}")
# magnitude: mean dayret when dir30 up vs down
print(f"  mean day$ | up-open : {fc2.filter(pl.col('dir30')==1)['dayret'].mean()*10:.1f}")
print(f"  mean day$ | dn-open : {fc2.filter(pl.col('dir30')==-1)['dayret'].mean()*10:.1f}")

# ---------- overnight inventory unwind: ON move vs RTH move ----------
fc3 = fc.with_columns(
    (pl.col("ropen") - pl.col("prev_rclose")).alias("on_move"),
    (pl.col("rclose") - pl.col("ropen")).alias("rth_move"),
).drop_nulls(["on_move", "rth_move"])
on_up = fc3.filter(pl.col("on_move") > 0)
on_dn = fc3.filter(pl.col("on_move") < 0)
print("\n=== OVERNIGHT GAP -> RTH (inventory unwind / trap) ===")
print(f"  gap UP days  : mean RTH$ = {on_up['rth_move'].mean()*10:>7.1f}  (n={on_up.height})")
print(f"  gap DOWN days: mean RTH$ = {on_dn['rth_move'].mean()*10:>7.1f}  (n={on_dn.height})")
corr = np.corrcoef(fc3["on_move"].to_numpy(), fc3["rth_move"].to_numpy())[0, 1]
print(f"  corr(overnight move, RTH move) = {corr:+.3f}  (neg => RTH fades the gap)")

# ---------- compression -> expansion ----------
fc4 = fc.with_columns(
    (pl.col("or30") / pl.col("atr")).alias("or30_norm"),
    ((pl.col("rhigh") - pl.col("rlow") - pl.col("or30")) / pl.col("atr")).alias("rest_exp"),
)
q = fc4["or30_norm"].quantile(0.25)
comp = fc4.filter(pl.col("or30_norm") <= q)
wide = fc4.filter(pl.col("or30_norm") > fc4["or30_norm"].quantile(0.75))
print("\n=== RANGE COMPRESSION (tight OR30) -> later EXPANSION ===")
print(f"  tight-open days: later range expansion (xATR) = {comp['rest_exp'].mean():.2f}  (n={comp.height})")
print(f"  wide-open  days: later range expansion (xATR) = {wide['rest_exp'].mean():.2f}  (n={wide.height})")

# ---------- day of week ----------
print("\n=== DAY-OF-WEEK (PIT-clean) ===")
dn = ["Mon", "Tue", "Wed", "Thu", "Fri"]
dw = fc.with_columns((pl.col("rclose") - pl.col("ropen")).alias("dr")).group_by("dow").agg(
    (pl.col("dr").mean() * 10).alias("mean_day$"),
    (pl.col("rrange").mean() * 10).alias("mean_range$"),
    pl.len().alias("n")).sort("dow")
for r in dw.iter_rows(named=True):
    if r["dow"] < 5:
        print(f"  {dn[r['dow']]}: mean day ${r['mean_day$']:>6.1f}  range ${r['mean_range$']:>6.1f}  n={r['n']}")
