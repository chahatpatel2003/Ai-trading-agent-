"""
dukascopy.py
============

Download + decode Dukascopy historical tick (.bi5) files.

Each hour is an LZMA-compressed file of 20-byte big-endian records:
  uint32 time_ms (offset within the hour), uint32 ask, uint32 bid,
  float32 ask_volume, float32 bid_volume
Prices are integers scaled by PRICE_FACTOR (auto-detected). Timestamps are UTC.
Hours with no data return HTTP 404 -> treated as empty (weekend/holiday/pre-history).
"""

from __future__ import annotations

import calendar
import lzma
import random
import statistics
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

import config

_REC = struct.Struct(config.BI5_STRUCT)


class NonRetryable(Exception):
    pass


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": config.USER_AGENT})
    return s


def hour_url(dt: datetime) -> str:
    # Dukascopy month is ZERO-BASED (00=Jan .. 11=Dec).
    return (f"{config.DATAFEED_BASE}/{config.SYMBOL}/"
            f"{dt.year:04d}/{dt.month - 1:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5")


def _retryable_status(code: int) -> bool:
    return code == 429 or 500 <= code < 600


def fetch_hour(session: requests.Session, dt: datetime, *,
               max_retries: int = config.MAX_RETRIES, logger=None) -> bytes:
    """Return raw .bi5 bytes for an hour, or b'' if it legitimately has no file (404)."""
    url = hour_url(dt)
    attempt = 0
    while True:
        try:
            r = session.get(url, timeout=config.HTTP_TIMEOUT)
            if r.status_code == 404:
                return b""
            if r.status_code == 200:
                return r.content
            if _retryable_status(r.status_code):
                raise IOError(f"HTTP {r.status_code}")
            raise NonRetryable(f"HTTP {r.status_code} for {url}")
        except NonRetryable:
            raise
        except Exception as exc:  # noqa: BLE001 - network/timeout -> retry
            attempt += 1
            if attempt > max_retries:
                raise
            delay = min(config.BACKOFF_BASE_SECONDS * (config.BACKOFF_FACTOR ** (attempt - 1)),
                        config.BACKOFF_MAX_SECONDS)
            delay += random.uniform(0, delay * 0.25)
            if logger:
                logger.warning("retry %d/%d %s: %r (sleep %.1fs)", attempt, max_retries, url, exc, delay)
            time.sleep(delay)


def decompress_bi5(raw: bytes) -> bytes:
    if not raw:
        return b""
    try:
        return lzma.decompress(raw)                       # FORMAT_AUTO (.xz/.lzma)
    except lzma.LZMAError:
        return lzma.decompress(raw, format=lzma.FORMAT_ALONE)  # legacy alone format


def decode_hour(raw: bytes, hour_dt: datetime) -> list[tuple[int, int, int, float, float]]:
    """Decode one hour to raw (abs_ms, ask_int, bid_int, ask_vol, bid_vol). No scaling."""
    try:
        data = decompress_bi5(raw)
    except (lzma.LZMAError, EOFError, ValueError):
        return []  # zero-byte/garbage 200 response -> treat as no data
    if not data or len(data) < config.BI5_RECORD_SIZE:
        return []
    # Truncate any trailing partial record defensively.
    usable = len(data) - (len(data) % config.BI5_RECORD_SIZE)
    base_ms = int(hour_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    out = []
    for time_ms, ask_i, bid_i, ask_v, bid_v in _REC.iter_unpack(data[:usable]):
        out.append((base_ms + time_ms, ask_i, bid_i, ask_v, bid_v))
    return out


def detect_price_factor(sample_ask_ints: list[int]) -> int | None:
    """Pick the divisor that puts the median price in the sane NAS100 range."""
    if not sample_ask_ints:
        return None
    med = statistics.median(sample_ask_ints)
    for f in config.PRICE_FACTOR_CANDIDATES:
        if config.SANE_MIN_PRICE <= med / f <= config.SANE_MAX_PRICE:
            return f
    return None


def month_hours(year: int, month: int) -> list[datetime]:
    ndays = calendar.monthrange(year, month)[1]
    return [datetime(year, month, d, h, tzinfo=timezone.utc)
            for d in range(1, ndays + 1) for h in range(24)]


def download_month(session: requests.Session, year: int, month: int, *,
                   logger=None, save_bi5_dir: Path | None = None) -> dict[datetime, bytes]:
    """Fetch every hour of a month concurrently. Returns {hour_dt: raw_bytes}.
    Optionally archives the original .bi5 files (unchanged) under save_bi5_dir."""
    hours = month_hours(year, month)
    results: dict[datetime, bytes] = {}

    def work(dt: datetime):
        return dt, fetch_hour(session, dt, logger=logger)

    with ThreadPoolExecutor(max_workers=config.HTTP_CONCURRENCY) as ex:
        for dt, raw in ex.map(work, hours):
            results[dt] = raw

    if save_bi5_dir is not None:
        save_bi5_dir.mkdir(parents=True, exist_ok=True)
        for dt, raw in results.items():
            if raw:
                (save_bi5_dir / f"{dt:%Y%m%d_%H}h_ticks.bi5").write_bytes(raw)
    return results


def decode_month(results: dict[datetime, bytes]) -> list[tuple[int, int, int, float, float]]:
    """Decode all hours, sorted by time. Returns raw (abs_ms, ask_i, bid_i, ask_v, bid_v)."""
    recs: list[tuple[int, int, int, float, float]] = []
    for dt in sorted(results):
        recs.extend(decode_hour(results[dt], dt))
    return recs


def hour_has_data(session: requests.Session, dt: datetime) -> bool:
    """Cheap probe: does this hour have a (non-empty) tick file?"""
    raw = fetch_hour(session, dt, max_retries=3)
    return bool(raw) and bool(decompress_bi5(raw))
