#!/usr/bin/env python3
"""
xauusd_dukascopy.py
===================
Download 5 years of XAU/USD (Gold vs US Dollar) historical data from
Dukascopy and build a cleaned 1-minute OHLCV dataset.

Source
------
Dukascopy historical tick data feed:
    https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM0}/{DD}/{HH}h_ticks.bi5
where MM0 is the MONTH AS A 0-INDEXED NUMBER (January = 00 ... December = 11).
Each file is one UTC hour of tick data, LZMA-compressed.

Tick record format (verified empirically):
    big-endian struct '>3i2f' = 20 bytes per tick:
        int32  ms_into_hour   (milliseconds since the start of the UTC hour)
        int32  ask  (price * 1000)
        int32  bid  (price * 1000)
        float32 ask_volume     (millions of units)
        float32 bid_volume     (millions of units)
Price scale for XAUUSD is 1000 (3 decimal places); confirmed by decoding a
known hour (2024-06-03 14:00 UTC -> bid ~2333.14, matching the gold price).

1-minute OHLCV construction
---------------------------
For each UTC minute, ticks are grouped by (ms_into_hour // 60000):
    open  = bid of first tick in the minute (time order)
    high  = max bid in the minute
    low   = min bid in the minute
    close = bid of last tick in the minute
    volume = sum(ask_volume + bid_volume) over all ticks in the minute
             (Dukascopy native volume, in millions of units)
OHLC uses the BID price (Dukascopy / dukascopy-node default convention).

Output columns (CSV + Parquet):
    timestamp_utc, open, high, low, close, volume, source
All timestamps are UTC. source = "Dukascopy".

Usage
-----
    python xauusd_dukascopy.py download   # fetch raw bi5 files (resumable)
    python xauusd_dukascopy.py build      # decode raw -> CSV/Parquet + reports
    python xauusd_dukascopy.py all        # download then build  (default)
"""

import os
import sys
import time
import lzma
import struct
import logging
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SYMBOL       = "XAUUSD"
PRICE_SCALE  = 1000          # integer price -> real price divisor
SOURCE_LABEL = "Dukascopy"

# Date range (UTC). End is EXCLUSIVE.  2021-06-29 00:00 .. 2026-06-29 00:00
# i.e. full days 2021-06-29 through 2026-06-28 (the end date 2026-06-29 is the
# current date and has no finalized Dukascopy data yet).
START_UTC = dt.datetime(2021, 6, 29, 0, 0, 0, tzinfo=dt.timezone.utc)
END_UTC   = dt.datetime(2026, 6, 29, 0, 0, 0, tzinfo=dt.timezone.utc)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))   # D:\XAUUSD Data
RAW_DIR   = os.path.join(BASE_DIR, "raw")
LOG_FILE  = os.path.join(BASE_DIR, "download.log")

CSV_OUT     = os.path.join(BASE_DIR, "XAUUSD_1min.csv")
PARQUET_OUT = os.path.join(BASE_DIR, "XAUUSD_1min.parquet")
GAPS_OUT    = os.path.join(BASE_DIR, "gaps_report.csv")
DAILY_OUT   = os.path.join(BASE_DIR, "daily_counts.csv")

URL_TMPL = ("https://datafeed.dukascopy.com/datafeed/"
            "{sym}/{y:04d}/{m0:02d}/{d:02d}/{h:02d}h_ticks.bi5")

MAX_WORKERS = 8
TIMEOUT     = 60
MAX_RETRIES = 6
HEADERS     = {"User-Agent": "Mozilla/5.0 (xauusd-dukascopy-downloader)"}


def make_session():
    """Session with a connection pool sized to the worker count so threads
    don't contend for the default 10 pooled connections."""
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS, max_retries=0)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("xauusd")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def iter_hours(start, end):
    """Yield every UTC hour in [start, end), skipping Saturdays (Dukascopy
    gold market is closed all of Saturday, so those files never exist)."""
    cur = start
    one_hour = dt.timedelta(hours=1)
    while cur < end:
        if cur.weekday() != 5:          # 5 == Saturday
            yield cur
        cur += one_hour


def raw_path(hour):
    """Local path mirroring Dukascopy's source URL layout (month 0-indexed)."""
    return os.path.join(RAW_DIR, SYMBOL,
                        f"{hour.year:04d}", f"{hour.month - 1:02d}",
                        f"{hour.day:02d}", f"{hour.hour:02d}h_ticks.bi5")


def url_for(hour):
    return URL_TMPL.format(sym=SYMBOL, y=hour.year, m0=hour.month - 1,
                           d=hour.day, h=hour.hour)


# --------------------------------------------------------------------------- #
# Download phase
# --------------------------------------------------------------------------- #
def fetch_hour(hour, session):
    """Download one hour. Returns a status string.
    Resumable: skips hours that already have a .bi5 or a .nodata marker."""
    path   = raw_path(hour)
    nodata = path + ".nodata"
    if os.path.exists(path) or os.path.exists(nodata):
        return "skip"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = url_for(hour)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 404:
                # No data for this hour (market closed / holiday). Mark it.
                open(nodata, "wb").close()
                return "nodata"
            r.raise_for_status()
            content = r.content
            if len(content) == 0:
                open(nodata, "wb").close()
                return "empty"
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(content)
            os.replace(tmp, path)
            return "ok"
        except Exception as e:                       # network / 5xx -> retry
            if attempt == MAX_RETRIES:
                log.warning("FAILED %s after %d tries: %s", url, attempt, e)
                return "fail"
            time.sleep(2 ** (attempt - 1))
    return "fail"


def download():
    hours = list(iter_hours(START_UTC, END_UTC))
    total = len(hours)
    log.info("Download phase: %d candidate hours (Saturdays skipped) "
             "from %s to %s", total, START_UTC, END_UTC)

    counts = {"ok": 0, "nodata": 0, "skip": 0, "empty": 0, "fail": 0}
    done = 0
    session = make_session()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_hour, h, session): h for h in hours}
        for fut in as_completed(futs):
            res = fut.result()
            counts[res] = counts.get(res, 0) + 1
            done += 1
            if done % 2000 == 0 or done == total:
                log.info("  progress %d/%d  ok=%d nodata=%d skip=%d fail=%d",
                         done, total, counts["ok"], counts["nodata"],
                         counts["skip"], counts["fail"])
    log.info("Download complete: %s", counts)
    if counts["fail"]:
        log.warning("%d hours failed to download; re-run 'download' to retry "
                    "(failures are NOT marked, so they will be retried).",
                    counts["fail"])
    return counts


# --------------------------------------------------------------------------- #
# Build phase
# --------------------------------------------------------------------------- #
def decode_file(path, hour):
    """Decode one bi5 file into a list of per-minute OHLCV tuples for `hour`.
    Returns list of (timestamp_utc(datetime), open, high, low, close, volume)."""
    with open(path, "rb") as f:
        comp = f.read()
    if not comp:
        return []
    data = lzma.decompress(comp)
    n = len(data) // 20
    # bins[minute_in_hour] -> [open, high, low, close, volume, last_ms]
    bins = {}
    unpack = struct.Struct(">3i2f").unpack_from
    for i in range(n):
        ms, ask_i, bid_i, av, bv = unpack(data, i * 20)
        minute = ms // 60000
        if minute > 59:
            minute = 59
        bid = bid_i / PRICE_SCALE
        vol = av + bv
        b = bins.get(minute)
        if b is None:
            bins[minute] = [bid, bid, bid, bid, vol, ms]
        else:
            if bid > b[1]:
                b[1] = bid
            if bid < b[2]:
                b[2] = bid
            if ms >= b[5]:               # keep latest tick as close
                b[3] = bid
                b[5] = ms
            b[4] += vol
    out = []
    for minute, b in bins.items():
        ts = hour + dt.timedelta(minutes=minute)
        out.append((ts, b[0], b[1], b[2], b[3], b[4]))
    return out


def build():
    import pandas as pd

    log.info("Build phase: scanning raw files ...")
    rows = []
    files_used = 0
    hours = list(iter_hours(START_UTC, END_UTC))
    for idx, hour in enumerate(hours):
        path = raw_path(hour)
        if not os.path.exists(path):
            continue
        try:
            recs = decode_file(path, hour)
        except Exception as e:
            log.warning("decode failed %s: %s", path, e)
            continue
        if recs:
            rows.extend(recs)
            files_used += 1
        if (idx + 1) % 5000 == 0:
            log.info("  decoded %d/%d hours, %d bars so far",
                     idx + 1, len(hours), len(rows))

    log.info("Decoded %d hourly files into %d raw minute bars",
             files_used, len(rows))
    if not rows:
        log.error("No data decoded. Did the download phase run?")
        sys.exit(1)

    df = pd.DataFrame(rows, columns=["timestamp_utc", "open", "high",
                                     "low", "close", "volume"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # ---- duplicate verification ------------------------------------------- #
    dup_mask = df["timestamp_utc"].duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup:
        log.warning("Found %d duplicate timestamp rows -> collapsing them.",
                    n_dup)
        # Collapse duplicates: O=first, H=max, L=min, C=last, V=sum
        g = df.groupby("timestamp_utc", as_index=False).agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum"))
        df = g.sort_values("timestamp_utc").reset_index(drop=True)
    final_dups = int(df["timestamp_utc"].duplicated().sum())
    log.info("Duplicate timestamps after cleaning: %d", final_dups)
    assert final_dups == 0, "duplicate timestamps remain!"

    df["source"] = SOURCE_LABEL
    df = df[["timestamp_utc", "open", "high", "low", "close",
             "volume", "source"]]

    # round prices to 3 dp (native XAUUSD precision), volume to 6 dp
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].round(3)
    df["volume"] = df["volume"].round(6)

    # ---- write outputs ---------------------------------------------------- #
    # CSV with ISO-8601 UTC timestamps ("YYYY-MM-DD HH:MM:SS+00:00")
    df.to_csv(CSV_OUT, index=False)
    df.to_parquet(PARQUET_OUT, index=False)
    log.info("Wrote %s and %s  (%d rows)", CSV_OUT, PARQUET_OUT, len(df))

    # ---- gap / missing range analysis ------------------------------------- #
    analyze_gaps(df, pd)

    # ---- README ----------------------------------------------------------- #
    write_readme(df)

    # ---- summary ---------------------------------------------------------- #
    print_summary(df)
    return df


def write_readme(df):
    res = getattr(analyze_gaps, "result", {})
    first = df["timestamp_utc"].min()
    last  = df["timestamp_utc"].max()
    missing_days = res.get("missing_days", [])
    md_preview = "\n".join("    " + d.strftime("%Y-%m-%d (%a)")
                           for d in missing_days[:60])
    if len(missing_days) > 60:
        md_preview += f"\n    ... and {len(missing_days) - 60} more " \
                      f"(see daily_counts.csv for every day's bar count)"
    readme = f"""\
XAU/USD (Gold vs US Dollar) - 1-Minute OHLCV Dataset
====================================================

INSTRUMENT
    XAU/USD  (Gold spot vs US Dollar), Dukascopy symbol "XAUUSD".

SOURCE
    Dukascopy Bank SA historical data feed (tick data).
    Endpoint pattern (one LZMA-compressed file per UTC hour):
        https://datafeed.dukascopy.com/datafeed/XAUUSD/{{YYYY}}/{{MM}}/{{DD}}/{{HH}}h_ticks.bi5
    NOTE: {{MM}} in the URL is the month as a 0-INDEXED number
    (January = 00 ... December = 11). The raw/ folder mirrors this exact
    source layout, so its month sub-folders are also 0-indexed.
    Data source: Dukascopy. (NOT Yahoo Finance.)

DATE RANGE (UTC)
    Requested : 2021-06-29  ->  2026-06-29
    Downloaded: 2021-06-29 00:00 UTC  ->  2026-06-29 00:00 UTC (end exclusive)
                i.e. full days 2021-06-29 through 2026-06-28. The end date
                2026-06-29 is the current date and has no finalized data yet.
    Actual first bar: {first}
    Actual last  bar: {last}

TIMEZONE
    All timestamps are UTC (the Dukascopy feed is natively UTC).

FILES
    XAUUSD_1min.csv        Cleaned 1-minute OHLCV, CSV.
    XAUUSD_1min.parquet    Same data, Apache Parquet (timestamp = UTC datetime).
    raw/                   Original Dukascopy .bi5 tick files (source preserved),
                           mirroring the source URL layout. *.nodata markers flag
                           hours the feed returned no data (market closed/holiday).
    gaps_report.csv        Every gap > 1 minute (start, end, missing minutes,
                           duration, classification: weekend/daily_break/intraday).
    daily_counts.csv       Number of 1-minute bars per UTC calendar day.
    download.log           Full run log.
    xauusd_dukascopy.py    The exact script used to produce this dataset.

COLUMNS (CSV & Parquet)
    timestamp_utc  UTC timestamp of the minute bar's open (minute start).
    open           Bid price at the first tick of the minute.
    high           Maximum bid price during the minute.
    low            Minimum bid price during the minute.
    close          Bid price at the last tick of the minute.
    volume         Sum of (ask_volume + bid_volume) of all ticks in the minute,
                   in Dukascopy native units (millions of units).
    source         Constant string "Dukascopy".

CONSTRUCTION
    Raw bi5 tick records: big-endian '>3i2f' (20 bytes):
        int32 ms_into_hour, int32 ask*1000, int32 bid*1000,
        float32 ask_volume, float32 bid_volume.
    XAUUSD price scale = 1000 (3 decimals); verified by decoding a known hour.
    OHLC uses the BID price (Dukascopy / dukascopy-node default convention).
    Ticks grouped per UTC minute -> O/H/L/C + summed volume as above.

ROW COUNT
    {len(df):,} unique 1-minute bars.

DATA QUALITY
    Duplicate timestamps: {int(df['timestamp_utc'].duplicated().sum())} (verified none remain).
    Gaps > 1 minute: {res.get('n_gaps', 'n/a')} total
        weekend break : {res.get('weekend', 'n/a')}
        daily break   : {res.get('daily_break', 'n/a')}  (~21:00-22:00 UTC daily halt)
        intraday      : {res.get('intraday', 'n/a')}  (see gaps_report.csv)
    Missing trading days (zero bars, excluding Saturdays): {len(missing_days)}
{md_preview if missing_days else "    (none)"}

EXACT COMMANDS USED
    # one-time environment setup
    python -m pip install requests pandas pyarrow

    # download raw tick data + build cleaned dataset + reports + this README
    python xauusd_dukascopy.py all
    #   (equivalently, in two steps:)
    python xauusd_dukascopy.py download
    python xauusd_dukascopy.py build

    Date range and symbol are set at the top of xauusd_dukascopy.py
    (START_UTC, END_UTC, SYMBOL). Re-running 'download' is resumable: it skips
    hours already present in raw/ and retries previously failed hours.

GENERATED
    Folder: {BASE_DIR}
"""
    with open(os.path.join(BASE_DIR, "README.txt"), "w", encoding="utf-8") as f:
        f.write(readme)
    log.info("Wrote README.txt")


def analyze_gaps(df, pd):
    """Report missing date ranges and intraday gaps."""
    ts = df["timestamp_utc"]
    deltas = ts.diff()
    one_min = pd.Timedelta(minutes=1)

    gaps = []
    for i in range(1, len(df)):
        d = deltas.iloc[i]
        if d > one_min:
            start = ts.iloc[i - 1]
            end   = ts.iloc[i]
            missing = int(d / one_min) - 1
            # classify
            dur_h = d.total_seconds() / 3600.0
            wd_start = start.weekday()
            if dur_h >= 12 and (wd_start == 4 or start.weekday() == 5
                                or end.weekday() == 6):
                cls = "weekend"
            elif dur_h <= 3 and 20 <= start.hour <= 22:
                cls = "daily_break"
            else:
                cls = "intraday"
            gaps.append((start, end, missing, round(dur_h, 3), cls))

    gdf = pd.DataFrame(gaps, columns=["gap_start_utc", "gap_end_utc",
                                      "missing_minutes", "duration_hours",
                                      "classification"])
    gdf.to_csv(GAPS_OUT, index=False)

    # per-day bar counts + missing trading days
    daily = (df.set_index("timestamp_utc")
               .groupby(pd.Grouper(freq="D")).size()
               .rename("bars").reset_index())
    daily.columns = ["date_utc", "bars"]
    daily.to_csv(DAILY_OUT, index=False)

    # missing days: calendar days in range (excl Sat) with zero bars
    all_days = pd.date_range(START_UTC.date(),
                             (END_UTC - dt.timedelta(days=1)).date(),
                             freq="D", tz="UTC")
    present = set(df["timestamp_utc"].dt.floor("D"))
    missing_days = [d for d in all_days
                    if d.weekday() != 5 and d not in present]

    intraday = gdf[gdf["classification"] == "intraday"]
    log.info("Gap analysis: %d total gaps (>1min)  | weekend=%d "
             "daily_break=%d intraday=%d",
             len(gdf), int((gdf["classification"] == "weekend").sum()),
             int((gdf["classification"] == "daily_break").sum()),
             len(intraday))
    log.info("Missing trading days (zero bars, excl. Saturdays): %d",
             len(missing_days))

    # stash for summary
    analyze_gaps.result = {
        "n_gaps": len(gdf),
        "weekend": int((gdf["classification"] == "weekend").sum()),
        "daily_break": int((gdf["classification"] == "daily_break").sum()),
        "intraday": len(intraday),
        "missing_days": missing_days,
        "top_intraday": intraday.sort_values("missing_minutes",
                                             ascending=False).head(20),
    }


def print_summary(df):
    res = getattr(analyze_gaps, "result", {})
    print("\n" + "=" * 64)
    print(" XAUUSD 1-MINUTE DATASET — SUMMARY")
    print("=" * 64)
    print(f" Folder         : {BASE_DIR}")
    print(f" Rows (1-min)   : {len(df):,}")
    print(f" First bar (UTC): {df['timestamp_utc'].min()}")
    print(f" Last  bar (UTC): {df['timestamp_utc'].max()}")
    print(f" Duplicate ts   : {int(df['timestamp_utc'].duplicated().sum())}")
    print(f" CSV            : {CSV_OUT}")
    print(f" Parquet        : {PARQUET_OUT}")
    if res:
        print(f" Gaps >1min     : {res['n_gaps']} "
              f"(weekend={res['weekend']}, daily_break={res['daily_break']}, "
              f"intraday={res['intraday']})")
        print(f" Missing days   : {len(res['missing_days'])} "
              f"(see daily_counts.csv / gaps_report.csv)")
    print("=" * 64 + "\n")


# --------------------------------------------------------------------------- #
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if cmd in ("download", "all"):
        download()
    if cmd in ("build", "all"):
        build()
    mins = (time.time() - t0) / 60.0
    log.info("Done '%s' in %.1f min", cmd, mins)
    with open(os.path.join(BASE_DIR, f"_DONE_{cmd}.marker"), "w",
              encoding="utf-8") as f:
        f.write(f"{cmd} finished in {mins:.1f} min\n")


if __name__ == "__main__":
    main()
