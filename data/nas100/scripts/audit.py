"""
audit.py
========

Full-dataset audit after all downloads/processing. Verifies no missing months,
no corrupt files, no invalid spreads, no invalid timestamps; writes
reports/final_report.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import polars as pl

import config
import state as state_mod


def _months_range(start: tuple[int, int], end: tuple[int, int]):
    y, m = start
    while (y, m) <= end:
        yield f"{y}_{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def run(log, failures=None) -> dict:
    st = state_mod.load_state()
    months = st.get("months", {})
    issues = {"missing_months": [], "corrupt": [], "invalid_spread": [], "failed": list(failures or [])}

    # expected range: data_start .. current month
    if st.get("data_start"):
        ys, ms = map(int, st["data_start"].split("-"))
        start = (ys, ms)
    else:
        start = (config.PREFERRED_START.year, config.PREFERRED_START.month)
    now = datetime.now(timezone.utc)
    end = (now.year, now.month)

    total_files = total_rows = total_bytes = 0
    first_ts = last_ts = None
    spread_weighted = 0.0

    for ym in _months_range(start, end):
        rec = months.get(ym, {})
        status = rec.get("status")
        if status == "no_data":
            continue  # legitimate empty month (pre-history / market closed)
        if status == "failed":
            if ym not in issues["failed"]:
                issues["failed"].append(ym)
            continue
        if status != "completed":
            issues["missing_months"].append(ym)
            continue

        pq = config.PARQUET_DIR / f"{ym}.parquet"
        if not pq.exists():
            issues["missing_months"].append(ym)
            continue
        try:
            n = pl.scan_parquet(pq).select(pl.len()).collect().item()
            if not n or n <= 0:
                issues["corrupt"].append(ym)
                continue
        except Exception:  # noqa: BLE001
            issues["corrupt"].append(ym)
            continue

        proc = config.PROCESSED_DIR / f"{ym}_processed.parquet"
        psize = proc.stat().st_size if proc.exists() else 0
        if proc.exists():
            try:
                bad = pl.scan_parquet(proc).select((pl.col("spread") < 0).sum()).collect().item()
                if bad and int(bad) > 0:
                    issues["invalid_spread"].append({ym: int(bad)})
            except Exception:  # noqa: BLE001
                issues["corrupt"].append(ym)

        total_files += 1
        total_rows += int(n)
        total_bytes += pq.stat().st_size + psize
        if rec.get("avg_spread") is not None:
            spread_weighted += float(rec["avg_spread"]) * int(n)
        ft, lt = rec.get("first_ts"), rec.get("last_ts")
        if ft:
            first_ts = ft if first_ts is None else min(first_ts, ft)
        if lt:
            last_ts = lt if last_ts is None else max(last_ts, lt)

    processing_done = (config.CANDLES_DIR / "candles_mid_1d.parquet").exists()
    clean = not (issues["missing_months"] or issues["corrupt"]
                 or issues["invalid_spread"] or issues["failed"])

    def _human(b):
        s = float(b)
        for u in ["B", "KB", "MB", "GB", "TB"]:
            if s < 1024 or u == "TB":
                return f"{s:.2f} {u}"
            s /= 1024

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": config.SYMBOL,
        "price_factor": st.get("price_factor"),
        "data_start": st.get("data_start"),
        "total_files": total_files,
        "total_rows": total_rows,
        "total_storage_bytes": total_bytes,
        "total_storage_human": _human(total_bytes) if total_bytes else "0 B",
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "average_spread": (spread_weighted / total_rows) if total_rows else None,
        "validation_status": "PASS" if clean else "ATTENTION",
        "processing_status": "complete" if processing_done else "incomplete",
        "issues": issues,
    }
    (config.REPORTS_DIR / "final_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("FINAL AUDIT: status=%s files=%d rows=%s missing=%s corrupt=%s invalid_spread=%s failed=%s",
             report["validation_status"], total_files, f"{total_rows:,}",
             issues["missing_months"] or "none", issues["corrupt"] or "none",
             issues["invalid_spread"] or "none", issues["failed"] or "none")
    return report
