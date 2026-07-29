"""
config.py
=========

Central configuration for the NAS100 (Dukascopy) historical tick acquisition
system. Import-safe (no network I/O at import).

Dukascopy serves historical ticks as hourly LZMA-compressed ``.bi5`` files at:
  https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5
where MM is ZERO-BASED (00=January .. 11=December), DD is zero-padded, and HH is
00-23 in UTC. Each record (big-endian, 20 bytes) is:
  uint32 time_ms, uint32 ask, uint32 bid, float32 ask_vol, float32 bid_vol
Prices are integers scaled by 10**point_decimals (auto-detected; see PRICE_FACTOR).
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

SCRIPTS_DIR: Path = Path(__file__).resolve().parent
ROOT: Path = Path(os.environ.get("NAS100_ROOT", r"D:\NAS 100 Data"))

RAW_DIR: Path = ROOT / "raw"
PARQUET_DIR: Path = ROOT / "parquet"
PROCESSED_DIR: Path = ROOT / "processed"
REPORTS_DIR: Path = ROOT / "reports"
LOGS_DIR: Path = ROOT / "logs"
BACKUPS_DIR: Path = ROOT / "backups"          # original .bi5 files, unchanged
STATE_FILE: Path = ROOT / "download_state.json"

# processed/ sub-datasets
MID_PRICE_DIR: Path = PROCESSED_DIR / "mid_price"
SPREAD_DIR: Path = PROCESSED_DIR / "spread"
CANDLES_DIR: Path = PROCESSED_DIR / "candles"
TICKBARS_DIR: Path = PROCESSED_DIR / "tickbars"
STATS_DIR: Path = PROCESSED_DIR / "stats"

DOWNLOAD_LOG: Path = LOGS_DIR / "download.log"
VALIDATION_LOG: Path = LOGS_DIR / "validation.log"
ERROR_LOG: Path = LOGS_DIR / "errors.log"
DOWNLOADER_PID_FILE: Path = LOGS_DIR / "downloader.pid"

ALL_DIRS = (RAW_DIR, PARQUET_DIR, PROCESSED_DIR, REPORTS_DIR, LOGS_DIR, BACKUPS_DIR)


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Instrument / datafeed
# --------------------------------------------------------------------------- #
# Dukascopy symbol for the Nasdaq-100 / US Tech 100 index CFD.
SYMBOL: str = os.environ.get("NAS100_SYMBOL", "USATECHIDXUSD")
DATAFEED_BASE: str = "https://datafeed.dukascopy.com/datafeed"
USER_AGENT: str = "Mozilla/5.0 (compatible; nas100-tick-archiver/1.0)"

# .bi5 record format: big-endian uint32 time_ms, uint32 ask, uint32 bid,
# float32 ask_vol, float32 bid_vol.
BI5_STRUCT: str = ">IIIff"
BI5_RECORD_SIZE: int = 20

# Price scaling: real_price = raw_int / PRICE_FACTOR. Index CFDs commonly use
# 1000 (3 decimals). The decoder AUTO-DETECTS by checking that decoded prices
# land in [SANE_MIN_PRICE, SANE_MAX_PRICE]; this default is the starting guess.
PRICE_FACTOR_DEFAULT: int = 1000
PRICE_FACTOR_CANDIDATES: tuple[int, ...] = (1000, 100, 10, 100000, 10000)
SANE_MIN_PRICE: float = float(os.environ.get("NAS100_MIN_PRICE", "800"))
SANE_MAX_PRICE: float = float(os.environ.get("NAS100_MAX_PRICE", "60000"))

# --------------------------------------------------------------------------- #
# Date range
# --------------------------------------------------------------------------- #
# Preferred earliest month; the downloader probes backward/forward for the real
# first month with data and never errors on empty (weekend/pre-history) hours.
PREFERRED_START: date = date(date.fromisoformat(
    os.environ.get("NAS100_START", "2010-01-01")).year,
    date.fromisoformat(os.environ.get("NAS100_START", "2010-01-01")).month, 1)

# --------------------------------------------------------------------------- #
# Download / retry
# --------------------------------------------------------------------------- #
MAX_RETRIES: int = int(os.environ.get("NAS100_MAX_RETRIES", "10"))
BACKOFF_BASE_SECONDS: float = float(os.environ.get("NAS100_BACKOFF_BASE", "1.5"))
BACKOFF_FACTOR: float = 2.0
BACKOFF_MAX_SECONDS: float = 120.0
# Concurrent hourly-file fetches within a month (polite; Dukascopy tolerates this).
HTTP_CONCURRENCY: int = int(os.environ.get("NAS100_CONCURRENCY", "6"))  # stay under ~10 req/s
HTTP_TIMEOUT: float = 30.0

PARQUET_COMPRESSION: str = "snappy"

# --------------------------------------------------------------------------- #
# Post-processing: candle timeframes + tick-bar sizes
# --------------------------------------------------------------------------- #
# (polars offset alias, label)
TIMEFRAMES: list[tuple[str, str]] = [
    ("1s", "1s"), ("5s", "5s"), ("15s", "15s"), ("30s", "30s"),
    ("1m", "1m"), ("5m", "5m"), ("15m", "15m"), ("30m", "30m"),
    ("1h", "1h"), ("4h", "4h"), ("1d", "1d"),
]
TICK_BAR_SIZES: list[int] = [100, 500, 1000, 2000]
PRICE_KINDS: list[str] = ["bid", "ask", "mid"]
