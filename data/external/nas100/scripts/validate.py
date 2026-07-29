"""
validate.py
===========

Validate a month's Parquet tick file against the 18 required checks.

Verdict:
  * "FAIL"               -> a STRUCTURAL problem (missing/empty/corrupt/unsorted/
                            null/non-numeric/wrong-month/incomplete-download or
                            duplicate exact rows). run.py deletes + re-downloads.
  * "PASS_WITH_WARNINGS" -> structurally sound but has data-quality anomalies
                            (e.g. a few crossed bid>ask ticks). NOT re-downloaded
                            (re-downloading cannot fix an inherent feed anomaly).
  * "PASS"               -> all checks clean.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


def _human(n: int | None) -> str | None:
    if n is None:
        return None
    s = float(n)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if s < 1024 or u == "TB":
            return f"{s:.2f} {u}"
        s /= 1024
    return f"{n} B"


def validate_month(parquet_path: Path, year: int, month: int, *,
                   download_complete: bool = True) -> dict[str, Any]:
    rep: dict[str, Any] = {
        "file": parquet_path.name, "year": year, "month": month,
        "checks": {}, "metrics": {}, "quality": {}, "errors": [], "status": "FAIL",
    }
    chk = rep["checks"]
    errs = rep["errors"]

    # 1. exists, 2. size > 0
    chk["file_exists"] = parquet_path.exists()
    if not chk["file_exists"]:
        errs.append("file does not exist")
        return rep
    size = parquet_path.stat().st_size
    rep["metrics"]["file_size_bytes"] = size
    rep["metrics"]["file_size_human"] = _human(size)
    chk["file_size_positive"] = size > 0
    if not chk["file_size_positive"]:
        errs.append("file size is 0")
        return rep

    # 18. parquet opens / 10. no corrupted rows (full read)
    try:
        df = pl.read_parquet(parquet_path)
        chk["parquet_opens"] = True
    except Exception as exc:  # noqa: BLE001
        chk["parquet_opens"] = False
        errs.append(f"parquet failed to open (corrupt): {exc!r}")
        return rep

    cols = df.columns
    n = df.height
    rep["metrics"]["row_count"] = n
    chk["has_rows"] = n > 0
    if n == 0:
        errs.append("zero rows")
        return rep

    # 3/6/7 columns exist
    chk["timestamp_col"] = "timestamp" in cols
    chk["bid_col"] = "bid" in cols
    chk["ask_col"] = "ask" in cols
    if not (chk["timestamp_col"] and chk["bid_col"] and chk["ask_col"]):
        errs.append(f"missing required column(s); have {cols}")
        return rep

    ts, bid, ask = df["timestamp"], df["bid"], df["ask"]

    # 4 timestamps valid (datetime dtype), 8/9 numeric
    chk["timestamp_is_datetime"] = ts.dtype == pl.Datetime or isinstance(ts.dtype, pl.Datetime)
    chk["bid_numeric"] = bid.dtype.is_numeric()
    chk["ask_numeric"] = ask.dtype.is_numeric()
    if not (chk["timestamp_is_datetime"] and chk["bid_numeric"] and chk["ask_numeric"]):
        errs.append("wrong dtype for timestamp/bid/ask")
        return rep

    # 3/10/11 no null/NaN in key columns
    null_ts = ts.null_count()
    null_bid = bid.null_count() + int(bid.is_nan().sum() or 0)
    null_ask = ask.null_count() + int(ask.is_nan().sum() or 0)
    chk["timestamps_exist"] = null_ts == 0
    chk["no_null_bid_ask"] = (null_bid == 0 and null_ask == 0)
    if null_ts or null_bid or null_ask:
        errs.append(f"nulls: ts={null_ts} bid={null_bid} ask={null_ask}")

    # 11 malformed: prices must be positive & finite
    bad_price = int(((bid <= 0) | (ask <= 0) | bid.is_infinite() | ask.is_infinite()).sum())
    chk["prices_positive_finite"] = bad_price == 0
    if bad_price:
        errs.append(f"{bad_price} rows with non-positive/non-finite price")

    # 5 sorted
    chk["timestamps_sorted"] = bool(ts.is_sorted())
    if not chk["timestamps_sorted"]:
        errs.append("timestamps not sorted")

    # 12 no duplicate exact rows (distinct ticks sharing a ms are allowed)
    dup_rows = int(df.is_duplicated().sum())
    chk["no_duplicate_rows"] = dup_rows == 0
    rep["quality"]["duplicate_timestamp_count"] = int(ts.is_duplicated().sum())
    if dup_rows:
        errs.append(f"{dup_rows} exact-duplicate rows")

    # 13/14 bid <= ask  /  spread >= 0  (quality, not structural)
    crossed = int((ask < bid).sum())
    rep["quality"]["crossed_bid_gt_ask"] = crossed
    chk["bid_le_ask"] = crossed == 0

    # 15/16 first/last timestamp in expected month
    first_ts, last_ts = ts.min(), ts.max()
    rep["metrics"]["first_timestamp"] = str(first_ts)
    rep["metrics"]["last_timestamp"] = str(last_ts)

    def _in_month(t) -> bool:
        return isinstance(t, datetime) and t.year == year and t.month == month
    chk["first_in_month"] = _in_month(first_ts)
    chk["last_in_month"] = _in_month(last_ts)
    if not (chk["first_in_month"] and chk["last_in_month"]):
        errs.append(f"timestamp(s) outside {year}-{month:02d}: first={first_ts} last={last_ts}")

    # 17 no gaps from failed downloads (every hour fetched: data or 404)
    chk["download_complete"] = bool(download_complete)
    if not download_complete:
        errs.append("download did not complete all hours (possible gap)")

    # metrics
    rep["metrics"].update({
        "min_bid": float(bid.min()), "max_bid": float(bid.max()),
        "min_ask": float(ask.min()), "max_ask": float(ask.max()),
        "average_spread": float((ask - bid).mean()),
    })

    # ---- verdict ----
    structural = [
        "file_exists", "file_size_positive", "parquet_opens", "has_rows",
        "timestamp_col", "bid_col", "ask_col", "timestamp_is_datetime",
        "bid_numeric", "ask_numeric", "timestamps_exist", "no_null_bid_ask",
        "prices_positive_finite", "timestamps_sorted", "no_duplicate_rows",
        "first_in_month", "last_in_month", "download_complete",
    ]
    structural_ok = all(chk.get(k) for k in structural)
    if not structural_ok:
        rep["status"] = "FAIL"
    elif not chk["bid_le_ask"]:
        rep["status"] = "PASS_WITH_WARNINGS"
    else:
        rep["status"] = "PASS"
    return rep
