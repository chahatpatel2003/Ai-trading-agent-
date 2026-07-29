"""
Economic Calendar — tracks high-impact events that move gold.

Uses a static calendar approach with date-based logic for recurring events
(FOMC, NFP, CPI, PPI, Retail Sales, GDP). No fragile web scraping.
"""

import calendar
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from logs.logger import get_logger

logger = get_logger("economic_calendar")

# ---------------------------------------------------------------------------
# Known FOMC meeting dates (announcement day) for 2025-2027
# ---------------------------------------------------------------------------

FOMC_DATES_2025 = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
]

FOMC_DATES_2026 = [
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]

FOMC_DATES_2027 = [
    "2027-01-27", "2027-03-17", "2027-05-05", "2027-06-16",
    "2027-07-28", "2027-09-15", "2027-11-03", "2027-12-15",
]

FOMC_DATES = {
    2025: [datetime.strptime(d, "%Y-%m-%d").date() for d in FOMC_DATES_2025],
    2026: [datetime.strptime(d, "%Y-%m-%d").date() for d in FOMC_DATES_2026],
    2027: [datetime.strptime(d, "%Y-%m-%d").date() for d in FOMC_DATES_2027],
}

# ---------------------------------------------------------------------------
# Event bias descriptions — historical tendencies for gold
# ---------------------------------------------------------------------------

EVENT_BIASES: Dict[str, str] = {
    "FOMC": "volatile, direction depends on hawkish/dovish",
    "NFP": "strong NFP = bearish gold, weak NFP = bullish gold",
    "CPI": (
        "hot CPI = bearish gold short-term (rate hike fears), "
        "bullish long-term (inflation)"
    ),
    "PPI": "hot PPI = bearish gold short-term, mirrors CPI dynamics",
    "RETAIL_SALES": "strong retail = bearish gold (risk-on), weak retail = bullish gold",
    "GDP": "strong GDP = bearish gold (less safe-haven demand), weak GDP = bullish gold",
    "quiet": "no event = trend-following conditions",
}

# Impact weights for risk scoring (higher = bigger gold mover)
EVENT_IMPACT_WEIGHTS: Dict[str, int] = {
    "FOMC": 100,
    "NFP": 85,
    "CPI": 75,
    "PPI": 50,
    "GDP": 60,
    "RETAIL_SALES": 45,
}


def _first_friday(year: int, month: int) -> date:
    """Return the first Friday of a given month (NFP release day)."""
    cal = calendar.monthcalendar(year, month)
    # calendar.monthcalendar returns weeks as lists; Friday is index 4
    for week in cal:
        if week[calendar.FRIDAY] != 0:
            return date(year, month, week[calendar.FRIDAY])
    # Fallback (should never reach here)
    raise ValueError(f"No Friday found in {year}-{month}")


def _estimate_cpi_date(year: int, month: int) -> date:
    """CPI is typically released around the 13th of each month."""
    target = date(year, month, 13)
    # Shift to next weekday if it falls on a weekend
    while target.weekday() >= 5:  # 5=Sat, 6=Sun
        target += timedelta(days=1)
    return target


def _estimate_ppi_date(year: int, month: int) -> date:
    """PPI is typically released around the 14th of each month."""
    target = date(year, month, 14)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


def _estimate_retail_sales_date(year: int, month: int) -> date:
    """Retail Sales is typically released around the 16th of each month."""
    target = date(year, month, 16)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


def _estimate_gdp_dates(year: int) -> List[date]:
    """
    GDP advance estimates come in the last week of Jan, Apr, Jul, Oct.
    Returns estimated dates for a given year.
    """
    gdp_months = [1, 4, 7, 10]
    dates = []
    for month in gdp_months:
        # Last Thursday of the month is the typical GDP release day
        cal = calendar.monthcalendar(year, month)
        last_thursday = None
        for week in reversed(cal):
            if week[calendar.THURSDAY] != 0:
                last_thursday = date(year, month, week[calendar.THURSDAY])
                break
        if last_thursday:
            dates.append(last_thursday)
    return dates


class EconomicCalendar:
    """
    Static economic calendar for tracking high-impact events that move gold.

    Generates dates for FOMC, NFP, CPI, PPI, Retail Sales, and GDP using
    known schedules and date-based heuristics.
    """

    def __init__(self, data_dir: str = "./learn/data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._event_cache: Dict[str, List[Dict]] = {}
        logger.info("economic_calendar.initialized", data_dir=str(self.data_dir))

    # ------------------------------------------------------------------
    # Event generation
    # ------------------------------------------------------------------

    def _generate_events_for_year(self, year: int) -> List[Dict]:
        """Build the full list of high-impact events for a given year."""
        events: List[Dict] = []

        # --- FOMC ---
        fomc_dates = FOMC_DATES.get(year, [])
        if not fomc_dates:
            logger.warning("economic_calendar.no_fomc_dates", year=year)
        for d in fomc_dates:
            events.append({
                "event": "FOMC",
                "date": d.isoformat(),
                "impact": "HIGH",
                "gold_bias": EVENT_BIASES["FOMC"],
            })

        # --- NFP (first Friday of each month) ---
        for month in range(1, 13):
            nfp_date = _first_friday(year, month)
            events.append({
                "event": "NFP",
                "date": nfp_date.isoformat(),
                "impact": "HIGH",
                "gold_bias": EVENT_BIASES["NFP"],
            })

        # --- CPI (~13th of each month) ---
        for month in range(1, 13):
            cpi_date = _estimate_cpi_date(year, month)
            events.append({
                "event": "CPI",
                "date": cpi_date.isoformat(),
                "impact": "HIGH",
                "gold_bias": EVENT_BIASES["CPI"],
            })

        # --- PPI (~14th of each month) ---
        for month in range(1, 13):
            ppi_date = _estimate_ppi_date(year, month)
            events.append({
                "event": "PPI",
                "date": ppi_date.isoformat(),
                "impact": "MEDIUM",
                "gold_bias": EVENT_BIASES["PPI"],
            })

        # --- Retail Sales (~16th of each month) ---
        for month in range(1, 13):
            rs_date = _estimate_retail_sales_date(year, month)
            events.append({
                "event": "RETAIL_SALES",
                "date": rs_date.isoformat(),
                "impact": "MEDIUM",
                "gold_bias": EVENT_BIASES["RETAIL_SALES"],
            })

        # --- GDP (last week of Jan/Apr/Jul/Oct) ---
        for gdp_date in _estimate_gdp_dates(year):
            events.append({
                "event": "GDP",
                "date": gdp_date.isoformat(),
                "impact": "HIGH",
                "gold_bias": EVENT_BIASES["GDP"],
            })

        events.sort(key=lambda e: e["date"])
        return events

    def _get_all_events(self, year: int) -> List[Dict]:
        """Retrieve events for a year, using cache if available."""
        cache_key = str(year)
        if cache_key not in self._event_cache:
            self._event_cache[cache_key] = self._generate_events_for_year(year)
        return self._event_cache[cache_key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_upcoming_events(self, days_ahead: int = 7) -> List[Dict]:
        """
        Return list of upcoming high-impact events within `days_ahead` days.

        Each dict contains: event, date, impact, days_away, gold_bias.
        """
        today = date.today()
        horizon = today + timedelta(days=days_ahead)

        # Collect events from current year and next year if horizon spans Dec-Jan
        years_to_check = {today.year}
        if horizon.year != today.year:
            years_to_check.add(horizon.year)

        upcoming: List[Dict] = []
        for year in years_to_check:
            for event in self._get_all_events(year):
                event_date = date.fromisoformat(event["date"])
                if today <= event_date <= horizon:
                    days_away = (event_date - today).days
                    upcoming.append({
                        "event": event["event"],
                        "date": event["date"],
                        "impact": event["impact"],
                        "days_away": days_away,
                        "gold_bias": event["gold_bias"],
                    })

        upcoming.sort(key=lambda e: e["days_away"])
        logger.debug(
            "economic_calendar.upcoming_events",
            count=len(upcoming),
            days_ahead=days_ahead,
        )
        return upcoming

    def is_event_today(self) -> bool:
        """Return True if a high-impact event is scheduled for today."""
        today = date.today()
        for event in self._get_all_events(today.year):
            if event["impact"] == "HIGH" and event["date"] == today.isoformat():
                logger.info("economic_calendar.event_today", event=event["event"])
                return True
        return False

    def is_nfp_week(self) -> bool:
        """Return True if today is within 2 days of NFP (first Friday)."""
        today = date.today()
        nfp_date = _first_friday(today.year, today.month)

        # Also check next month's NFP if we're at the end of the month
        candidates = [nfp_date]
        if today.month < 12:
            candidates.append(_first_friday(today.year, today.month + 1))
        else:
            candidates.append(_first_friday(today.year + 1, 1))

        # Check previous month's NFP too (e.g., if NFP was yesterday)
        if today.month > 1:
            candidates.append(_first_friday(today.year, today.month - 1))
        else:
            candidates.append(_first_friday(today.year - 1, 12))

        for nfp in candidates:
            if abs((today - nfp).days) <= 2:
                logger.info(
                    "economic_calendar.nfp_week",
                    nfp_date=nfp.isoformat(),
                    days_away=(nfp - today).days,
                )
                return True
        return False

    def is_fomc_week(self) -> bool:
        """Return True if today is within 2 days of an FOMC announcement."""
        today = date.today()
        fomc_dates = FOMC_DATES.get(today.year, [])
        for fomc_date in fomc_dates:
            if abs((today - fomc_date).days) <= 2:
                logger.info(
                    "economic_calendar.fomc_week",
                    fomc_date=fomc_date.isoformat(),
                    days_away=(fomc_date - today).days,
                )
                return True
        return False

    def get_event_risk_score(self) -> int:
        """
        Return a 0-100 risk score based on proximity to high-impact events.

        0 = quiet period, no events nearby
        100 = FOMC day (highest-impact event for gold)

        Scoring: inverse-distance weighting by event impact.
        Same-day events receive full impact weight; events further away
        are discounted exponentially.
        """
        today = date.today()
        events = self.get_upcoming_events(days_ahead=7)

        if not events:
            # Also check past 2 days — a just-passed event still creates volatility
            return 0

        # Also include events from the past 1 day (post-event volatility)
        yesterday = today - timedelta(days=1)
        all_recent: List[Dict] = []
        for year in {today.year, today.year - 1 if today.month == 1 else today.year}:
            for event in self._get_all_events(year):
                event_date = date.fromisoformat(event["date"])
                if yesterday <= event_date <= today + timedelta(days=7):
                    days_away = abs((event_date - today).days)
                    all_recent.append({
                        "event": event["event"],
                        "days_away": days_away,
                        "impact": event["impact"],
                    })

        if not all_recent:
            return 0

        max_score = 0.0
        for event in all_recent:
            base_weight = EVENT_IMPACT_WEIGHTS.get(event["event"], 30)
            days = event["days_away"]

            if days == 0:
                # Event is today — full weight
                score = base_weight
            else:
                # Exponential decay: halves every 2 days
                score = base_weight * (0.5 ** (days / 2.0))

            max_score = max(max_score, score)

        final_score = min(100, int(round(max_score)))
        logger.debug("economic_calendar.risk_score", score=final_score)
        return final_score

    def get_gold_event_bias(self, event_type: str) -> str:
        """
        Return the historical gold bias for a given event type.

        Args:
            event_type: One of FOMC, NFP, CPI, PPI, GDP, RETAIL_SALES, quiet.

        Returns:
            Description of historical gold price reaction.
        """
        event_key = event_type.upper().strip()
        bias = EVENT_BIASES.get(event_key, EVENT_BIASES["quiet"])
        return bias

    def should_reduce_risk(self) -> bool:
        """
        Return True if within 4 hours of a high-impact event.

        High-impact events (FOMC, NFP, CPI, GDP) are typically released at:
          - FOMC: 14:00 ET (18:00 UTC)
          - NFP:  08:30 ET (12:30 UTC)
          - CPI:  08:30 ET (12:30 UTC)
          - GDP:  08:30 ET (12:30 UTC)
          - PPI:  08:30 ET (12:30 UTC)

        If today is an event day, we check whether the current UTC time is
        within 4 hours before or 1 hour after the typical release time.
        """
        now_utc = datetime.utcnow()
        today = now_utc.date()

        # Typical release times in UTC
        release_times: Dict[str, int] = {
            "FOMC": 18,         # 2:00 PM ET
            "NFP": 12,          # 8:30 AM ET (use 12 as approx hour)
            "CPI": 12,          # 8:30 AM ET
            "PPI": 12,          # 8:30 AM ET
            "GDP": 12,          # 8:30 AM ET
            "RETAIL_SALES": 12, # 8:30 AM ET
        }

        for event in self._get_all_events(today.year):
            event_date = date.fromisoformat(event["date"])
            if event_date != today:
                continue

            event_type = event["event"]
            release_hour = release_times.get(event_type, 12)
            release_dt = datetime(
                today.year, today.month, today.day,
                release_hour, 30,  # :30 for NFP/CPI accuracy
            )

            # 4 hours before to 1 hour after
            window_start = release_dt - timedelta(hours=4)
            window_end = release_dt + timedelta(hours=1)

            if window_start <= now_utc <= window_end:
                logger.warning(
                    "economic_calendar.reduce_risk",
                    event=event_type,
                    release_utc=release_dt.isoformat(),
                    minutes_until=int((release_dt - now_utc).total_seconds() / 60),
                )
                return True

        # Also check if tomorrow has a very early event and we're close to midnight
        tomorrow = today + timedelta(days=1)
        years_to_check = {tomorrow.year}
        for year in years_to_check:
            for event in self._get_all_events(year):
                event_date = date.fromisoformat(event["date"])
                if event_date != tomorrow:
                    continue
                event_type = event["event"]
                release_hour = release_times.get(event_type, 12)
                release_dt = datetime(
                    tomorrow.year, tomorrow.month, tomorrow.day,
                    release_hour, 30,
                )
                window_start = release_dt - timedelta(hours=4)
                if now_utc >= window_start:
                    logger.warning(
                        "economic_calendar.reduce_risk_tomorrow",
                        event=event_type,
                        release_utc=release_dt.isoformat(),
                    )
                    return True

        return False

    # ------------------------------------------------------------------
    # Summary / convenience
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict:
        """Return a complete summary of the current economic calendar state."""
        upcoming = self.get_upcoming_events(days_ahead=7)
        return {
            "event_today": self.is_event_today(),
            "nfp_week": self.is_nfp_week(),
            "fomc_week": self.is_fomc_week(),
            "risk_score": self.get_event_risk_score(),
            "should_reduce_risk": self.should_reduce_risk(),
            "upcoming_events_7d": upcoming,
            "nearest_event": upcoming[0] if upcoming else None,
        }

    def __repr__(self) -> str:
        score = self.get_event_risk_score()
        return f"<EconomicCalendar risk_score={score}>"
