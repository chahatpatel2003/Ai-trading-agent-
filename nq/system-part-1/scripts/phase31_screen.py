"""Phase 31 — re-screen the full Phase 29 + 30 candidate universe for FREQUENCY + SCALABILITY + edge.

No new strategies; no re-optimization. Reads every stored 1-MNQ trade set and applies the new
business bars: enough trades AND scalable within a 100K/$3,000 account AND >=$500/month at safe size.

Scalability model = FIXED CONTRACTS (N MNQ): at N contracts every trade's net and MAE scale by N, so
the drawdown scales linearly. max_safe_N = floor(3000 / intraday_maxDD_1mnq). Monthly $ uses the FULL
66-month calendar span (dead months count as $0 — the honest business number).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nqlab.discovery import phase29 as P29

DIRS = ["/Volumes/T9/nq data/curated/artifacts/phase29/run1/trades",
        "/Volumes/T9/nq data/curated/artifacts/phase30/run1/trades"]
OUT = Path("docs/phase31")
OUT.mkdir(parents=True, exist_ok=True)
SPAN_MONTHS = 66
DD_LIMIT = 3000.0
MNQ_PV = 2.0


def _intraday_maxdd(nets: np.ndarray, maes: np.ndarray) -> tuple[float, float]:
    """(closed maxDD, intraday maxDD incl. per-trade MAE) for a 1-MNQ sequential path."""
    eq = np.cumsum(nets)
    peak = np.maximum.accumulate(eq)
    closed = float((peak - eq).max()) if eq.size else 0.0
    prev = np.concatenate([[0.0], eq[:-1]])
    lows = prev - maes
    run_peak = np.maximum.accumulate(np.maximum(eq, prev))
    intraday = float((run_peak - lows).max()) if eq.size else 0.0
    return closed, intraday


def _metrics(tf: pl.DataFrame) -> dict[str, Any] | None:
    kept, _ = P29.intraday_only(tf)
    if kept.height == 0:
        return None
    kept = kept.sort("entry_ts")
    nets = kept["net"].to_numpy()
    maes = (kept["mae"].to_numpy()) * MNQ_PV
    n = len(nets)
    wins = nets[nets > 0]
    losses = nets[losses_mask] if (losses_mask := nets < 0).any() else np.array([])
    # frequency
    days = kept["entry_ts"].dt.date()
    first, last = days.min(), days.max()
    span_days = max((last - first).days, 1)
    per_week = n / (span_days / 7)
    years = kept["entry_ts"].dt.year()
    n_years = years.n_unique()
    per_year = n / n_years
    months = kept.with_columns(kept["entry_ts"].dt.strftime("%Y-%m").alias("m"))["m"]
    n_months = months.n_unique()
    # longest no-trade gap (calendar days between consecutive trade days)
    uniq = sorted(set(days.to_list()))
    gap = max(((uniq[i] - uniq[i - 1]).days for i in range(1, len(uniq))), default=0)
    closed_dd, intraday_dd = _intraday_maxdd(nets, maes)
    net = float(nets.sum())
    pf = float(wins.sum() / -losses.sum()) if losses.size else None
    max_safe_n = int(DD_LIMIT // intraday_dd) if intraday_dd > 0 else 0
    monthly_1mnq = net / SPAN_MONTHS
    monthly_safe = monthly_1mnq * max_safe_n
    ml = cur = 0
    for x in nets:
        cur = cur + 1 if x < 0 else 0
        ml = max(ml, cur)
    return {
        "n": n, "per_week": round(per_week, 2), "per_month": round(n / SPAN_MONTHS, 2),
        "per_year": round(per_year, 1), "active_months": n_months, "longest_gap_days": gap,
        "pf": round(pf, 3) if pf else None,
        "win_rate": round(float((nets > 0).mean()), 4), "expectancy": round(net / n, 2),
        "net": round(net, 2), "monthly_1mnq": round(monthly_1mnq, 2),
        "closed_dd": round(closed_dd, 2), "intraday_dd": round(intraday_dd, 2),
        "ret_over_dd": round(net / intraday_dd, 2) if intraday_dd > 0 else None,
        "max_safe_n": max_safe_n, "monthly_at_safe_n": round(monthly_safe, 2),
        "max_loss_streak": ml,
    }


def _classify(m: dict[str, Any]) -> str:
    pf = m.get("pf") or 0
    freq_pass = (m["n"] >= 300) and (m["per_week"] >= 2.0)
    scalable = (m["max_safe_n"] >= 2) and (m["monthly_at_safe_n"] >= 500)
    edge = pf >= 1.4
    if pf < 1.0 or m["net"] <= 0:
        return "REJECT_NO_EDGE"
    if freq_pass and edge and scalable:
        return "SCALABLE_FREQUENT_FORWARD_PAPER"
    if edge and scalable and not freq_pass:
        return "SCALABLE_BUT_TOO_LOW_FREQUENCY"
    if freq_pass and not edge:
        return "FREQUENT_BUT_NO_EDGE"
    if edge and m["max_safe_n"] < 2:
        return "RESEARCH_ONLY_1_MNQ_EDGE"
    if edge and not scalable:
        return "REJECT_NOT_SCALABLE"
    return "REJECT_NO_EDGE"


def main() -> None:
    rows: list[dict[str, Any]] = []
    for d in DIRS:
        src = "p29" if "phase29" in d else "p30"
        for f in glob.glob(d + "/*.parquet"):
            if f.split("/")[-1].startswith("._"):
                continue
            tf = pl.read_parquet(f)
            if tf.height == 0:
                continue
            m = _metrics(tf)
            if m is None:
                continue
            name = tf.row(0, named=True)["candidate"]
            rows.append({"source": src, "candidate": name, **m, "classification": _classify(m)})
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(OUT / "phase31_screen.csv")
    print("universe:", df.height)
    print("\nclassification counts:")
    print(df.group_by("classification").len().sort("len", descending=True))

    # any SCALABLE_FREQUENT?
    sf = df.filter(pl.col("classification") == "SCALABLE_FREQUENT_FORWARD_PAPER")
    print(f"\nSCALABLE_FREQUENT_FORWARD_PAPER: {sf.height}")
    cols = ["candidate", "n", "per_week", "pf", "win_rate", "intraday_dd", "max_safe_n",
            "monthly_at_safe_n", "ret_over_dd"]
    if sf.height:
        with pl.Config(tbl_rows=20, fmt_str_lengths=44, tbl_width_chars=200):
            print(sf.select(cols))

    print("\n=== FREQUENT universe (>=2 trades/week) — ranked by PF ===")
    freq = df.filter(pl.col("per_week") >= 2.0).sort("pf", descending=True)
    print(f"count >=2/week: {freq.height}; with PF>=1.4: {freq.filter(pl.col('pf')>=1.4).height}")
    with pl.Config(tbl_rows=20, fmt_str_lengths=44, tbl_width_chars=200):
        print(freq.select(cols).head(20))

    print("\n=== Best by monthly_at_safe_n (scalable $/month within $3K) ===")
    with pl.Config(tbl_rows=20, fmt_str_lengths=44, tbl_width_chars=200):
        print(df.sort("monthly_at_safe_n", descending=True).select(cols).head(20))

    print("\n=== Highest ret/maxDD (scalability quality) with n>=300 ===")
    with pl.Config(tbl_rows=15, fmt_str_lengths=44, tbl_width_chars=200):
        print(df.filter(pl.col("n") >= 300).sort("ret_over_dd", descending=True).select(cols).head(15))


if __name__ == "__main__":
    main()
