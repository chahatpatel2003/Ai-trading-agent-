"""
run.py
======

Autonomous month-by-month NAS100 (Dukascopy) tick downloader.

Per month:  download -> decode -> build -> write parquet+processed -> validate
            -> monthly report -> update state -> next month.
Structural validation failure -> delete + re-download (up to MAX_RETRIES, backoff).
Resumable via download_state.json; completed months are skipped.

Run:
  python scripts/run.py                 # full autonomous run (resumes)
  python scripts/run.py --no-postprocess # download+audit only
  python scripts/run.py --probe-only     # detect symbol/scaling/start and exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import config
import convert
import dukascopy
import state as state_mod
import validate as validate_mod
from logging_setup import setup_logging


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_month() -> tuple[int, int]:
    n = datetime.now(timezone.utc)
    return n.year, n.month


def _next_month(y: int, m: int) -> tuple[int, int]:
    return (y + 1, 1) if m == 12 else (y, m + 1)


def _prev_month(y: int, m: int) -> tuple[int, int]:
    return (y - 1, 12) if m == 1 else (y, m - 1)


def _months(start: tuple[int, int], end: tuple[int, int]):
    y, m = start
    while (y, m) <= end:
        yield y, m
        y, m = _next_month(y, m)


def _backup_state() -> None:
    try:
        if config.STATE_FILE.exists():
            (config.BACKUPS_DIR).mkdir(parents=True, exist_ok=True)
            (config.BACKUPS_DIR / "download_state.backup.json").write_bytes(
                config.STATE_FILE.read_bytes())
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Probing: price scaling factor + first month with data
# --------------------------------------------------------------------------- #
def _probe_hours(y: int, m: int):
    """A few mid-month weekday afternoon UTC hours likely to have ticks."""
    out = []
    for day in (15, 16, 17, 14, 18, 13, 19):
        dt = datetime(y, m, day, tzinfo=timezone.utc)
        if dt.weekday() < 5:
            out += [dt.replace(hour=h) for h in (14, 15, 16)]
            break
    return out


def _month_has_data(session, y, m, dl_log) -> bool:
    for dt in _probe_hours(y, m):
        if dukascopy.decode_hour(dukascopy.fetch_hour(session, dt, max_retries=3, logger=dl_log), dt):
            return True
    return False


def detect_factor(session, dl_log) -> int | None:
    """Auto-detect price scaling using a RECENT (high-priced) active month. A
    recent NAS100 level (~15000-31000) disambiguates the factor because a wrong
    divisor yields an out-of-range price."""
    y, m = _current_month()
    floor = (config.PREFERRED_START.year, config.PREFERRED_START.month)
    for _ in range(36):
        if (y, m) < floor:
            break
        for dt in _probe_hours(y, m):
            recs = dukascopy.decode_hour(dukascopy.fetch_hour(session, dt, max_retries=3, logger=dl_log), dt)
            if recs:
                asks = [r[1] for r in recs[:5000]]
                f = dukascopy.detect_price_factor(asks)
                med = sorted(asks)[len(asks) // 2]
                dl_log.info("Factor probe %d-%02d %02dh: %d ticks, median ask raw=%d -> factor=%s (price~%.2f)",
                            y, m, dt.hour, len(recs), med, f, (med / f) if f else float("nan"))
                if f:
                    return f
        y, m = _prev_month(y, m)
    return None


def find_data_start(session, dl_log) -> tuple[int, int] | None:
    """Coarse (per-year June probe) then fine (per-month) search for the first
    month with tick data."""
    end = _current_month()
    first_year = None
    for yr in range(config.PREFERRED_START.year, end[0] + 1):
        if _month_has_data(session, yr, 6, dl_log) or (yr == end[0] and _month_has_data(session, yr, max(1, end[1] - 1), dl_log)):
            first_year = yr
            break
    if first_year is None:
        return None
    y, m = max(first_year - 1, config.PREFERRED_START.year), 1
    while (y, m) <= end:
        if _month_has_data(session, y, m, dl_log):
            dl_log.info("First data month: %d-%02d", y, m)
            return (y, m)
        y, m = _next_month(y, m)
    return None


# --------------------------------------------------------------------------- #
# Process one month (with repair/retry)
# --------------------------------------------------------------------------- #
def process_month(session, y, m, factor, state, dl_log, val_log, err_log) -> str:
    ym = f"{y}_{m:02d}"
    pq_path = config.PARQUET_DIR / f"{ym}.parquet"
    proc_path = config.PROCESSED_DIR / f"{ym}_processed.parquet"
    raw_dir = config.RAW_DIR / ym

    for attempt in range(config.MAX_RETRIES + 1):
        if attempt > 0:
            delay = min(config.BACKOFF_BASE_SECONDS * (config.BACKOFF_FACTOR ** (attempt - 1)),
                        config.BACKOFF_MAX_SECONDS)
            dl_log.warning("Re-download %s (attempt %d/%d) after %.1fs", ym, attempt, config.MAX_RETRIES, delay)
            for p in (pq_path, proc_path):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            time.sleep(delay)
        try:
            t0 = time.time()
            dl_log.info("START %s", ym)
            results = dukascopy.download_month(session, y, m, logger=dl_log, save_bi5_dir=raw_dir)
            download_complete = True  # download_month raises on real (non-404) errors
            records = dukascopy.decode_month(results)

            if not records:
                dl_log.info("%s: no ticks (pre-history / no data).", ym)
                state_mod.update_month(state, ym, status="no_data", rows=0,
                                       completed_utc=_now())
                return "no_data"

            df = convert.build_frame(records, factor)
            paths = convert.write_month(df, ym)
            rep = validate_mod.validate_month(paths["parquet"], y, m, download_complete=download_complete)
            # write monthly validation report
            (config.REPORTS_DIR / f"validation_{ym}.json").write_text(
                json.dumps(rep, indent=2, default=str), encoding="utf-8")

            if rep["status"] == "FAIL":
                val_log.error("VALIDATION FAIL %s: %s", ym, "; ".join(rep["errors"]))
                err_log.warning("validation FAIL %s attempt %d: %s", ym, attempt, rep["errors"])
                continue  # repair -> retry

            elapsed = time.time() - t0
            mt = rep["metrics"]
            val_log.info("VALIDATION %s %s | rows=%s | %s | spread~%.4f | bid[%.2f,%.2f]",
                         rep["status"], ym, mt.get("row_count"), mt.get("file_size_human"),
                         mt.get("average_spread", 0), mt.get("min_bid", 0), mt.get("max_bid", 0))
            dl_log.info("DONE %s | rows=%s | %.1fs | %s", ym, mt.get("row_count"), elapsed, rep["status"])
            state_mod.update_month(
                state, ym, status="completed", validation=rep["status"],
                rows=mt.get("row_count"), bytes=mt.get("file_size_bytes"),
                first_ts=mt.get("first_timestamp"), last_ts=mt.get("last_timestamp"),
                avg_spread=mt.get("average_spread"), retries=attempt, completed_utc=_now(),
            )
            _backup_state()
            return rep["status"]

        except Exception as exc:  # noqa: BLE001
            err_log.warning("ERROR %s attempt %d: %r", ym, attempt, exc)
            dl_log.warning("error on %s attempt %d: %r", ym, attempt, exc)
            continue

    err_log.error("EXHAUSTED retries for %s", ym)
    state_mod.update_month(state, ym, status="failed", completed_utc=_now())
    return "failed"


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Autonomous NAS100 Dukascopy tick downloader.")
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--only", help="Process only this YYYY-MM (smoke test) and exit.")
    args = ap.parse_args(argv)

    config.ensure_dirs()
    dl_log, val_log, err_log = setup_logging()
    try:
        config.DOWNLOADER_PID_FILE.write_text(str(os.getpid()), encoding="ascii")
    except OSError:
        pass

    dl_log.info("=" * 70)
    dl_log.info("NAS100 DUKASCOPY DOWNLOADER - start %s", _now())
    dl_log.info("Symbol=%s  Root=%s", config.SYMBOL, config.ROOT)
    dl_log.info("=" * 70)

    state = state_mod.load_state()
    session = dukascopy.make_session()

    # price factor
    factor = state.get("price_factor")
    if not factor:
        factor = detect_factor(session, dl_log)
        if not factor:
            err_log.error("Could not detect price scaling factor (no data found for symbol %s).", config.SYMBOL)
            dl_log.error("ABORT: no data / wrong symbol '%s'. Check the Dukascopy symbol.", config.SYMBOL)
            return 3
        state["price_factor"] = factor
        state_mod.save_state(state)
    dl_log.info("Price scaling factor: %d (real_price = raw / %d)", factor, factor)

    if args.only:
        yy, mm = map(int, args.only.split("-"))
        r = process_month(session, yy, mm, factor, state, dl_log, val_log, err_log)
        dl_log.info("--only %s -> %s", args.only, r)
        return 0 if r in ("PASS", "PASS_WITH_WARNINGS", "no_data") else 4

    # data start
    start = None
    if state.get("data_start"):
        ys, ms = map(int, state["data_start"].split("-"))
        start = (ys, ms)
    if start is None:
        start = find_data_start(session, dl_log)
        if start is None:
            dl_log.error("ABORT: no data found anywhere for %s.", config.SYMBOL)
            return 3
        state["data_start"] = f"{start[0]:04d}-{start[1]:02d}"
        state_mod.save_state(state)
    end = _current_month()
    dl_log.info("Downloading months %d-%02d .. %d-%02d", start[0], start[1], end[0], end[1])

    if args.probe_only:
        dl_log.info("--probe-only: symbol=%s factor=%d start=%s. Exiting.", config.SYMBOL, factor, start)
        return 0

    failures = []
    for y, m in _months(start, end):
        ym = f"{y}_{m:02d}"
        rec = state_mod.get_month(state, ym)
        st = rec.get("status")
        is_current = (y, m) == end
        if st == "completed" and not is_current:
            continue
        if st == "no_data" and not is_current:
            continue
        result = process_month(session, y, m, factor, state, dl_log, val_log, err_log)
        if result == "failed":
            failures.append(ym)

    dl_log.info("Download loop done. Failures: %s", failures or "none")

    # post-processing + final audit
    if not args.no_postprocess and not failures:
        try:
            import postprocess
            postprocess.run(dl_log)
        except Exception as exc:  # noqa: BLE001
            err_log.error("postprocess error: %r", exc)
    try:
        import audit
        audit.run(dl_log, failures=failures)
    except Exception as exc:  # noqa: BLE001
        err_log.error("audit error: %r", exc)

    if failures:
        dl_log.error("Finished with %d failed month(s): %s. Re-run to retry.", len(failures), failures)
        return 4
    dl_log.info("ALL MONTHS DONE & VALIDATED. end %s", _now())
    return 0


if __name__ == "__main__":
    sys.exit(main())
