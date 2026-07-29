"""
COT Report Parser — CFTC Commitment of Traders data for gold futures.

Parses the weekly CFTC COT report to extract gold futures positioning
data for commercials (hedgers/smart money), non-commercials (speculators),
and non-reportable (small traders). Computes speculator extremes for
contrarian signals.

Data source: https://www.cftc.gov/dea/newcot/deafut.txt (futures only, short format)
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from logs.logger import get_logger

logger = get_logger("cot_report")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CFTC_URL = "https://www.cftc.gov/dea/newcot/deafut.txt"

# Gold futures on COMEX — filter rows containing both keywords
GOLD_MARKET_KEYWORD = "GOLD"
GOLD_EXCHANGE_KEYWORD = "COMMODITY EXCHANGE"
GOLD_COMMODITY_CODE = "088691"

# Column indices in the CFTC deafut.txt (comma-separated, short format)
# Col 0: Market name, Col 1: YYMMDD, Col 2: YYYY-MM-DD, Col 3: contract code,
# Col 4: exchange initials, Col 5: region, Col 6: commodity code
COL_MARKET_NAME = 0
COL_AS_OF_DATE = 2      # YYYY-MM-DD format (col 1 is YYMMDD)
COL_OPEN_INTEREST = 7
COL_NONCOMM_LONG = 8    # Speculator longs
COL_NONCOMM_SHORT = 9   # Speculator shorts
COL_COMM_LONG = 11      # Commercial longs (hedgers)
COL_COMM_SHORT = 12     # Commercial shorts (hedgers)
COL_NONREPT_LONG = 15   # Small traders long
COL_NONREPT_SHORT = 16  # Small traders short

# Cache refresh interval: 7 days (CFTC publishes weekly, Fridays)
CACHE_MAX_AGE_SECONDS = 7 * 24 * 3600

HISTORY_FILENAME = "cot_gold_history.json"
CACHE_FILENAME = "cot_deafut_cache.txt"


# ---------------------------------------------------------------------------
# COTParser
# ---------------------------------------------------------------------------


class COTParser:
    """
    Downloads and parses the CFTC Commitment of Traders report for
    gold futures (COMEX). Caches locally and maintains a historical
    JSON ledger for lookback analysis.
    """

    def __init__(self, data_dir: str = "./learn/data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.cache_path = self.data_dir / CACHE_FILENAME
        self.history_path = self.data_dir / HISTORY_FILENAME

        # In-memory history (list of weekly positioning dicts)
        self.history: List[Dict[str, Any]] = []
        self.load_history()

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _cache_is_stale(self) -> bool:
        """Return True if the local cache file is missing or older than 7 days."""
        if not self.cache_path.exists():
            return True
        age = time.time() - self.cache_path.stat().st_mtime
        return age > CACHE_MAX_AGE_SECONDS

    def _download_cot(self) -> Optional[str]:
        """Download the current CFTC short-format futures-only COT file."""
        try:
            logger.info("cot_download_start", url=CFTC_URL)
            resp = requests.get(CFTC_URL, timeout=30)
            resp.raise_for_status()
            text = resp.text
            # Persist to cache
            self.cache_path.write_text(text, encoding="utf-8")
            logger.info("cot_download_ok", size_bytes=len(text))
            return text
        except requests.RequestException as exc:
            logger.warning("cot_download_failed", error=str(exc))
            return None

    def _read_cache(self) -> Optional[str]:
        """Read locally cached COT file, if it exists."""
        if self.cache_path.exists():
            return self.cache_path.read_text(encoding="utf-8")
        return None

    def _get_raw_text(self) -> Optional[str]:
        """Get COT text — fresh download if stale, else cached."""
        if self._cache_is_stale():
            text = self._download_cot()
            if text:
                return text
            # Fallback to cache even if stale
            logger.warning("cot_using_stale_cache")
            return self._read_cache()
        return self._read_cache()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_gold_rows(self, raw_text: str) -> List[Dict[str, Any]]:
        """
        Parse the tab-delimited CFTC file, extract all gold/COMEX rows,
        and return a list of positioning dicts sorted by report date.
        """
        results: List[Dict[str, Any]] = []
        lines = raw_text.strip().splitlines()

        for line in lines:
            cols = line.split(",")
            # The short-format file can be comma-separated or mixed;
            # fall back to tab-split if comma gives too few columns.
            if len(cols) < 15:
                cols = line.split("\t")
            if len(cols) < 15:
                continue

            market_name = cols[COL_MARKET_NAME].strip().upper()

            # Filter for gold on COMEX
            if GOLD_MARKET_KEYWORD not in market_name:
                continue
            if GOLD_EXCHANGE_KEYWORD not in market_name:
                continue

            try:
                date_str = cols[COL_AS_OF_DATE].strip()
                # CFTC uses YYMMDD — parse flexibly
                if len(date_str) == 6:
                    report_date = datetime.strptime(date_str, "%y%m%d").strftime(
                        "%Y-%m-%d"
                    )
                elif len(date_str) == 8:
                    report_date = datetime.strptime(date_str, "%Y%m%d").strftime(
                        "%Y-%m-%d"
                    )
                elif "-" in date_str or "/" in date_str:
                    # Try ISO or US formats
                    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
                        try:
                            report_date = datetime.strptime(date_str, fmt).strftime(
                                "%Y-%m-%d"
                            )
                            break
                        except ValueError:
                            continue
                    else:
                        logger.debug("cot_skip_date", raw=date_str)
                        continue
                else:
                    logger.debug("cot_skip_date", raw=date_str)
                    continue

                row = {
                    "report_date": report_date,
                    "non_commercial_long": int(cols[COL_NONCOMM_LONG].strip()),
                    "non_commercial_short": int(cols[COL_NONCOMM_SHORT].strip()),
                    "commercial_long": int(cols[COL_COMM_LONG].strip()),
                    "commercial_short": int(cols[COL_COMM_SHORT].strip()),
                    "non_reportable_long": int(cols[COL_NONREPT_LONG].strip()),
                    "non_reportable_short": int(cols[COL_NONREPT_SHORT].strip()),
                    "open_interest_total": int(cols[COL_OPEN_INTEREST].strip()),
                }
                # Derived net positions
                row["non_commercial_net"] = (
                    row["non_commercial_long"] - row["non_commercial_short"]
                )
                row["commercial_net"] = (
                    row["commercial_long"] - row["commercial_short"]
                )

                results.append(row)

            except (ValueError, IndexError) as exc:
                logger.debug("cot_parse_row_error", error=str(exc), line=line[:120])
                continue

        # Sort by date ascending
        results.sort(key=lambda r: r["report_date"])
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_latest(self) -> Optional[Dict[str, Any]]:
        """
        Download (or read cached) COT data, parse it, merge into
        history, and return the most recent gold positioning dict.
        """
        raw = self._get_raw_text()
        if raw is None:
            logger.error("cot_no_data_available")
            return None

        rows = self._parse_gold_rows(raw)
        if not rows:
            logger.warning("cot_no_gold_rows_found")
            return None

        # Merge new rows into history (deduplicate by report_date)
        existing_dates = {r["report_date"] for r in self.history}
        new_count = 0
        for row in rows:
            if row["report_date"] not in existing_dates:
                self.history.append(row)
                existing_dates.add(row["report_date"])
                new_count += 1

        # Re-sort history
        self.history.sort(key=lambda r: r["report_date"])

        if new_count > 0:
            self.save_history()
            logger.info("cot_history_updated", new_rows=new_count,
                        total=len(self.history))

        latest = rows[-1]
        logger.info(
            "cot_latest",
            date=latest["report_date"],
            spec_net=latest["non_commercial_net"],
            comm_net=latest["commercial_net"],
            oi=latest["open_interest_total"],
        )
        return latest

    def get_gold_positioning(self) -> Optional[Dict[str, Any]]:
        """
        Return the latest gold positioning dict with all fields:
        commercial_{long,short,net}, non_commercial_{long,short,net},
        non_reportable_{long,short}, open_interest_total, report_date.

        Uses cached history first; fetches fresh data if history is empty
        or the latest entry is older than 9 days.
        """
        need_fetch = False
        if not self.history:
            need_fetch = True
        else:
            latest_date = datetime.strptime(
                self.history[-1]["report_date"], "%Y-%m-%d"
            )
            if (datetime.utcnow() - latest_date).days > 9:
                need_fetch = True

        if need_fetch:
            result = self.fetch_latest()
            if result:
                return result
            # Fall back to history
            if self.history:
                return self.history[-1]
            return None

        return self.history[-1]

    def compute_spec_index(self, lookback_weeks: int = 52) -> Optional[float]:
        """
        Speculator (non-commercial) net positioning as a percentile
        of the lookback range (default 52 weeks).

        Returns a float 0-100.  Values > 90 imply extreme speculator
        longs (contrarian bearish). Values < 10 imply extreme speculator
        shorts (contrarian bullish).
        """
        if len(self.history) < 2:
            logger.warning("cot_insufficient_history", count=len(self.history))
            return None

        window = self.history[-lookback_weeks:]
        nets = [r["non_commercial_net"] for r in window]
        current = nets[-1]
        lo, hi = min(nets), max(nets)

        if hi == lo:
            return 50.0  # No range — neutral

        percentile = ((current - lo) / (hi - lo)) * 100.0
        return round(percentile, 2)

    def detect_extremes(self) -> List[str]:
        """
        Detect positioning extremes and return a list of signal strings.

        Possible signals:
        - "EXTREME_SPEC_LONG"        — spec_index > 90%
        - "EXTREME_SPEC_SHORT"       — spec_index < 10%
        - "COMMERCIAL_ACCUMULATING"  — commercial net rising 3+ weeks
        - "COMMERCIAL_DISTRIBUTING"  — commercial net falling 3+ weeks
        """
        signals: List[str] = []

        # --- Speculator extremes ---
        spec_index = self.compute_spec_index()
        if spec_index is not None:
            if spec_index > 90.0:
                signals.append("EXTREME_SPEC_LONG")
                logger.info("cot_extreme_spec_long", spec_index=spec_index)
            elif spec_index < 10.0:
                signals.append("EXTREME_SPEC_SHORT")
                logger.info("cot_extreme_spec_short", spec_index=spec_index)

        # --- Commercial trend (3+ consecutive weeks) ---
        if len(self.history) >= 4:
            recent = self.history[-4:]
            comm_nets = [r["commercial_net"] for r in recent]

            # Check last 3 week-over-week changes
            changes = [
                comm_nets[i + 1] - comm_nets[i] for i in range(len(comm_nets) - 1)
            ]

            if all(c > 0 for c in changes):
                signals.append("COMMERCIAL_ACCUMULATING")
                logger.info(
                    "cot_commercial_accumulating",
                    changes=changes,
                    weeks=len(changes),
                )
            elif all(c < 0 for c in changes):
                signals.append("COMMERCIAL_DISTRIBUTING")
                logger.info(
                    "cot_commercial_distributing",
                    changes=changes,
                    weeks=len(changes),
                )

        if not signals:
            logger.debug("cot_no_extremes_detected")

        return signals

    def get_weekly_changes(self) -> Optional[Dict[str, int]]:
        """
        Return week-over-week changes in key positioning fields.
        Requires at least 2 weeks of history.
        """
        if len(self.history) < 2:
            logger.warning("cot_need_2_weeks_for_changes")
            return None

        curr = self.history[-1]
        prev = self.history[-2]

        fields = [
            "non_commercial_long",
            "non_commercial_short",
            "non_commercial_net",
            "commercial_long",
            "commercial_short",
            "commercial_net",
            "non_reportable_long",
            "non_reportable_short",
            "open_interest_total",
        ]

        changes: Dict[str, int] = {
            "report_date_current": curr["report_date"],
            "report_date_previous": prev["report_date"],
        }
        for f in fields:
            changes[f"chg_{f}"] = curr[f] - prev[f]

        return changes

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_history(self) -> None:
        """Persist historical COT readings to JSON."""
        try:
            with open(self.history_path, "w", encoding="utf-8") as fp:
                json.dump(
                    {
                        "updated_at": datetime.utcnow().isoformat(),
                        "records": self.history,
                    },
                    fp,
                    indent=2,
                )
            logger.debug("cot_history_saved", records=len(self.history))
        except OSError as exc:
            logger.error("cot_history_save_failed", error=str(exc))

    def load_history(self) -> None:
        """Load historical COT readings from JSON."""
        if not self.history_path.exists():
            self.history = []
            return
        try:
            with open(self.history_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            self.history = data.get("records", [])
            # Ensure sorted
            self.history.sort(key=lambda r: r["report_date"])
            logger.info("cot_history_loaded", records=len(self.history))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("cot_history_load_failed", error=str(exc))
            self.history = []

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """
        Return a concise summary dict suitable for logging or
        inclusion in a broader market intelligence report.
        """
        positioning = self.get_gold_positioning()
        if positioning is None:
            return {"status": "no_data"}

        spec_index = self.compute_spec_index()
        extremes = self.detect_extremes()
        changes = self.get_weekly_changes()

        return {
            "report_date": positioning["report_date"],
            "spec_net": positioning["non_commercial_net"],
            "comm_net": positioning["commercial_net"],
            "open_interest": positioning["open_interest_total"],
            "spec_index_52w": spec_index,
            "signals": extremes,
            "weekly_changes": changes,
            "history_weeks": len(self.history),
        }

    def __repr__(self) -> str:
        latest = self.history[-1]["report_date"] if self.history else "N/A"
        return (
            f"COTParser(history={len(self.history)} weeks, "
            f"latest={latest})"
        )


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from logs.logger import setup_logging

    setup_logging(level="DEBUG")

    parser = COTParser()
    print(f"\n{parser}\n")

    positioning = parser.get_gold_positioning()
    if positioning:
        print("=== Gold Positioning ===")
        for k, v in positioning.items():
            print(f"  {k}: {v}")

        spec_idx = parser.compute_spec_index()
        print(f"\n  Spec Index (52w): {spec_idx}")

        extremes = parser.detect_extremes()
        print(f"  Extremes: {extremes or 'None'}")

        changes = parser.get_weekly_changes()
        if changes:
            print("\n=== Weekly Changes ===")
            for k, v in changes.items():
                print(f"  {k}: {v}")

        summary = parser.summary()
        print(f"\n=== Summary ===")
        print(json.dumps(summary, indent=2))
    else:
        print("No COT data available.")
