XAU/USD (Gold vs US Dollar) - 1-Minute OHLCV Dataset
====================================================

INSTRUMENT
    XAU/USD  (Gold spot vs US Dollar), Dukascopy symbol "XAUUSD".

SOURCE
    Dukascopy Bank SA historical data feed (tick data).
    Endpoint pattern (one LZMA-compressed file per UTC hour):
        https://datafeed.dukascopy.com/datafeed/XAUUSD/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5
    NOTE: {MM} in the URL is the month as a 0-INDEXED number
    (January = 00 ... December = 11). The raw/ folder mirrors this exact
    source layout, so its month sub-folders are also 0-indexed.
    Data source: Dukascopy. (NOT Yahoo Finance.)

DATE RANGE (UTC)
    Requested : 2021-06-29  ->  2026-06-29
    Downloaded: 2021-06-29 00:00 UTC  ->  2026-06-29 00:00 UTC (end exclusive)
                i.e. full days 2021-06-29 through 2026-06-28. The end date
                2026-06-29 is the current date and has no finalized data yet.
    Actual first bar: 2021-06-29 00:00:00+00:00
    Actual last  bar: 2026-06-28 23:59:00+00:00

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
    1,772,172 unique 1-minute bars.

DATA QUALITY
    Duplicate timestamps: 0 (verified none remain).
    Gaps > 1 minute: 1673 total
        weekend break : 261
        daily break   : 1208  (~21:00-22:00 UTC daily halt)
        intraday      : 204  (see gaps_report.csv)
    Missing trading days (zero bars, excluding Saturdays): 10
    2021-12-24 (Fri)
    2022-04-15 (Fri)
    2022-12-25 (Sun)
    2023-01-01 (Sun)
    2023-04-07 (Fri)
    2023-12-24 (Sun)
    2023-12-31 (Sun)
    2024-03-29 (Fri)
    2025-04-18 (Fri)
    2026-04-03 (Fri)

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
    Folder: D:\XAUUSD Data
