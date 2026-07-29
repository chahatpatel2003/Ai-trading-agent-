"""
Seasonal Analyzer — historical seasonal patterns for gold (XAUUSD).

Provides monthly, day-of-week, and hour-of-day biases based on
well-documented gold seasonal tendencies driven by cultural demand
cycles, institutional rebalancing, and trading session dynamics.
"""

from datetime import datetime, date
from typing import Dict, Optional

from logs.logger import get_logger

logger = get_logger("seasonal")

# ---------------------------------------------------------------------------
# Monthly seasonal data — historical gold tendencies
# ---------------------------------------------------------------------------
# Based on 20+ year averages of gold price behaviour.
# avg_return_pct and win_rate_pct are approximate historical values.

MONTHLY_PATTERNS: Dict[int, Dict] = {
    1: {
        "direction": "bullish",
        "avg_return_pct": 2.5,
        "win_rate_pct": 65.0,
        "reason": "Chinese New Year demand, portfolio rebalancing into safe havens",
    },
    2: {
        "direction": "bullish",
        "avg_return_pct": 1.2,
        "win_rate_pct": 58.0,
        "reason": "Continued Chinese New Year buying, seasonal demand carryover",
    },
    3: {
        "direction": "bearish",
        "avg_return_pct": -0.8,
        "win_rate_pct": 42.0,
        "reason": "Tax selling, quarter-end rebalancing, profit-taking",
    },
    4: {
        "direction": "bullish",
        "avg_return_pct": 1.5,
        "win_rate_pct": 60.0,
        "reason": "India Akshaya Tritiya festival — auspicious gold buying",
    },
    5: {
        "direction": "neutral",
        "avg_return_pct": 0.2,
        "win_rate_pct": 48.0,
        "reason": "Summer doldrums begin, reduced physical demand",
    },
    6: {
        "direction": "bearish",
        "avg_return_pct": -0.5,
        "win_rate_pct": 45.0,
        "reason": "Summer doldrums, low physical demand, FOMC volatility",
    },
    7: {
        "direction": "bullish",
        "avg_return_pct": 0.8,
        "win_rate_pct": 55.0,
        "reason": "Indian wedding season starts, early physical demand pickup",
    },
    8: {
        "direction": "bullish",
        "avg_return_pct": 1.8,
        "win_rate_pct": 62.0,
        "reason": "Indian festival buying ramp-up, Raksha Bandhan demand",
    },
    9: {
        "direction": "bullish",
        "avg_return_pct": 1.5,
        "win_rate_pct": 60.0,
        "reason": "Indian festival buying — Ganesh Chaturthi, Diwali preparation",
    },
    10: {
        "direction": "bullish",
        "avg_return_pct": 1.2,
        "win_rate_pct": 58.0,
        "reason": "Diwali and Dhanteras — peak Indian gold buying season",
    },
    11: {
        "direction": "bullish",
        "avg_return_pct": 1.0,
        "win_rate_pct": 56.0,
        "reason": "Continued festival season, year-end institutional positioning",
    },
    12: {
        "direction": "neutral",
        "avg_return_pct": 0.3,
        "win_rate_pct": 50.0,
        "reason": "Year-end, low liquidity, tax-loss harvesting, holiday trading",
    },
}

# ---------------------------------------------------------------------------
# Day-of-week patterns — based on historical intraday gold analysis
# ---------------------------------------------------------------------------

DAY_OF_WEEK_BIAS: Dict[int, Dict] = {
    0: {  # Monday
        "direction": "bullish",
        "strength": 0.15,
        "reason": "Weekend gap trades, safe-haven flows from weekend news",
    },
    1: {  # Tuesday
        "direction": "neutral",
        "strength": 0.0,
        "reason": "Neutral — consolidation day, no strong directional bias",
    },
    2: {  # Wednesday
        "direction": "bearish",
        "strength": -0.1,
        "reason": "Mid-week profit-taking, FOMC announcements often on Wednesday",
    },
    3: {  # Thursday
        "direction": "bullish",
        "strength": 0.2,
        "reason": "Pre-Friday positioning, strongest intraday gold day historically",
    },
    4: {  # Friday
        "direction": "bearish",
        "strength": -0.2,
        "reason": "Risk reduction before weekend, position squaring, NFP volatility",
    },
}

# ---------------------------------------------------------------------------
# Hour-of-day patterns — trading session dynamics (UTC)
# ---------------------------------------------------------------------------
# Represents relative activity/volatility and directional tendency.

HOUR_OF_DAY_BIAS: Dict[str, Dict] = {
    "asian_early": {
        "hours": range(0, 4),
        "volatility": "very_low",
        "direction": "neutral",
        "strength": 0.0,
        "description": "Early Asian session — very low volume, range-bound",
    },
    "asian_late": {
        "hours": range(4, 8),
        "volatility": "low",
        "direction": "neutral",
        "strength": 0.05,
        "description": "Late Asian session — slight Shanghai/physical buying bias",
    },
    "london_open": {
        "hours": range(8, 10),
        "volatility": "highest",
        "direction": "breakout",
        "strength": 0.25,
        "description": "London open — highest volatility, breakout moves, sets daily range",
    },
    "london_session": {
        "hours": range(10, 13),
        "volatility": "high",
        "direction": "trend",
        "strength": 0.15,
        "description": "London mid-session — strong trending, institutional flow",
    },
    "london_ny_overlap": {
        "hours": range(13, 17),
        "volatility": "very_high",
        "direction": "volatile",
        "strength": 0.20,
        "description": "London-NY overlap — second-highest volatility, major data releases",
    },
    "ny_afternoon": {
        "hours": range(17, 20),
        "volatility": "medium",
        "direction": "fading",
        "strength": -0.05,
        "description": "NY afternoon — declining volume, mean-reversion tendencies",
    },
    "ny_close": {
        "hours": range(20, 24),
        "volatility": "low",
        "direction": "neutral",
        "strength": -0.1,
        "description": "NY close — position squaring, low liquidity, avoid new entries",
    },
}


def _get_session_for_hour(hour_utc: int) -> Optional[Dict]:
    """Return the session dict that contains the given UTC hour."""
    for session_data in HOUR_OF_DAY_BIAS.values():
        if hour_utc in session_data["hours"]:
            return session_data
    return None


class SeasonalAnalyzer:
    """
    Analyzes historical seasonal patterns for gold across multiple timeframes:
    monthly, day-of-week, and hour-of-day.

    All biases are based on well-documented historical gold tendencies driven
    by cultural demand (CNY, Indian festivals), institutional flows, and
    trading session dynamics.
    """

    def __init__(self, data_dir: str = "./learn/data") -> None:
        from pathlib import Path

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("seasonal.initialized", data_dir=str(self.data_dir))

    # ------------------------------------------------------------------
    # Monthly bias
    # ------------------------------------------------------------------

    def get_monthly_bias(self) -> Dict[int, Dict]:
        """
        Return dict mapping month number (1-12) to seasonal data.

        Each entry contains:
            direction:      "bullish", "bearish", or "neutral"
            avg_return_pct: Historical average monthly return (%)
            win_rate_pct:   Historical win rate for the month (%)
            reason:         Explanation of the seasonal driver
        """
        return MONTHLY_PATTERNS.copy()

    def get_current_month_bias(self) -> Dict:
        """Return the seasonal data for the current month."""
        month = date.today().month
        return MONTHLY_PATTERNS[month]

    # ------------------------------------------------------------------
    # Seasonal score
    # ------------------------------------------------------------------

    def get_current_seasonal_score(self) -> float:
        """
        Return a score from -1.0 (strongly bearish) to +1.0 (strongly bullish)
        based on the current month and week position within that month.

        The score is derived from the monthly win_rate and avg_return data,
        with minor adjustments for early-month vs late-month positioning.
        """
        today = date.today()
        month = today.month
        day = today.day
        pattern = MONTHLY_PATTERNS[month]

        # Base score from win rate: 50% = 0.0, 65% = +0.6, 42% = -0.32
        win_rate = pattern["win_rate_pct"]
        base_score = (win_rate - 50.0) / 25.0  # Normalise to roughly -1..+1

        # Modulate by average return magnitude
        avg_ret = pattern["avg_return_pct"]
        return_factor = min(1.0, max(-1.0, avg_ret / 3.0))

        # Weighted combination
        score = 0.6 * base_score + 0.4 * return_factor

        # Week-of-month adjustment: first week often has NFP effect,
        # last week may have month-end rebalancing
        if day <= 7:
            # First week — NFP influence, slightly reduce confidence
            score *= 0.9
        elif day >= 25:
            # Last week — month-end flows, can be contrarian
            score *= 0.85

        # Clamp to [-1, +1]
        score = max(-1.0, min(1.0, score))

        logger.debug(
            "seasonal.current_score",
            month=month,
            day=day,
            direction=pattern["direction"],
            score=round(score, 3),
        )
        return round(score, 3)

    # ------------------------------------------------------------------
    # Day-of-week bias
    # ------------------------------------------------------------------

    def get_day_of_week_bias(self) -> Dict[int, Dict]:
        """
        Return dict mapping weekday (0=Monday .. 4=Friday) to bias data.

        Each entry contains:
            direction: "bullish", "bearish", or "neutral"
            strength:  Float bias strength (-0.2 to +0.2)
            reason:    Explanation
        """
        return DAY_OF_WEEK_BIAS.copy()

    def get_current_day_bias(self) -> Dict:
        """Return the day-of-week bias for today."""
        weekday = date.today().weekday()
        if weekday in DAY_OF_WEEK_BIAS:
            return DAY_OF_WEEK_BIAS[weekday]
        # Weekend — market closed
        return {
            "direction": "neutral",
            "strength": 0.0,
            "reason": "Weekend — market closed",
        }

    # ------------------------------------------------------------------
    # Hour-of-day bias
    # ------------------------------------------------------------------

    def get_hour_of_day_bias(self, hour_utc: int) -> Dict:
        """
        Return trading session bias for a given UTC hour.

        Args:
            hour_utc: Hour in UTC (0-23).

        Returns:
            Dict with volatility, direction, strength, and description.
        """
        hour_utc = int(hour_utc) % 24
        session = _get_session_for_hour(hour_utc)

        if session is None:
            return {
                "volatility": "unknown",
                "direction": "neutral",
                "strength": 0.0,
                "description": f"No session data for hour {hour_utc} UTC",
            }

        return {
            "volatility": session["volatility"],
            "direction": session["direction"],
            "strength": session["strength"],
            "description": session["description"],
        }

    # ------------------------------------------------------------------
    # Combined signal
    # ------------------------------------------------------------------

    def get_combined_seasonal_signal(self) -> Dict:
        """
        Weighted combination of month, day-of-week, and hour-of-day biases.

        Weights:
            Monthly:   40% — dominant macro seasonal driver
            Day:       25% — significant intra-week pattern
            Hour:      35% — important for entry/exit timing

        Returns:
            Dict with composite_score (-1 to +1), components breakdown,
            direction label, and confidence level.
        """
        now_utc = datetime.utcnow()

        # --- Monthly component ---
        monthly_score = self.get_current_seasonal_score()

        # --- Day-of-week component ---
        weekday = now_utc.weekday()
        day_data = DAY_OF_WEEK_BIAS.get(weekday)
        if day_data:
            day_score = day_data["strength"]
        else:
            day_score = 0.0

        # Normalise day_score to [-1, +1] range (current max is ~0.2)
        day_score_normalised = day_score / 0.25 if day_score != 0 else 0.0
        day_score_normalised = max(-1.0, min(1.0, day_score_normalised))

        # --- Hour-of-day component ---
        hour_data = self.get_hour_of_day_bias(now_utc.hour)
        hour_strength = hour_data["strength"]

        # Normalise hour strength to [-1, +1]
        hour_score_normalised = hour_strength / 0.25 if hour_strength != 0 else 0.0
        hour_score_normalised = max(-1.0, min(1.0, hour_score_normalised))

        # --- Weighted combination ---
        weight_month = 0.40
        weight_day = 0.25
        weight_hour = 0.35

        composite = (
            weight_month * monthly_score
            + weight_day * day_score_normalised
            + weight_hour * hour_score_normalised
        )
        composite = max(-1.0, min(1.0, round(composite, 3)))

        # Direction label
        if composite > 0.15:
            direction = "bullish"
        elif composite < -0.15:
            direction = "bearish"
        else:
            direction = "neutral"

        # Confidence based on alignment of components
        components = [monthly_score, day_score_normalised, hour_score_normalised]
        all_positive = all(c >= 0 for c in components)
        all_negative = all(c <= 0 for c in components)

        if all_positive or all_negative:
            confidence = "high"
        elif abs(composite) > 0.3:
            confidence = "medium"
        else:
            confidence = "low"

        result = {
            "composite_score": composite,
            "direction": direction,
            "confidence": confidence,
            "components": {
                "monthly": {
                    "score": monthly_score,
                    "weight": weight_month,
                    "month": now_utc.month,
                    "bias": MONTHLY_PATTERNS[now_utc.month]["direction"],
                },
                "day_of_week": {
                    "score": round(day_score_normalised, 3),
                    "weight": weight_day,
                    "weekday": now_utc.strftime("%A"),
                    "bias": day_data["direction"] if day_data else "neutral",
                },
                "hour_of_day": {
                    "score": round(hour_score_normalised, 3),
                    "weight": weight_hour,
                    "hour_utc": now_utc.hour,
                    "session": hour_data["description"],
                },
            },
        }

        logger.info(
            "seasonal.combined_signal",
            composite=composite,
            direction=direction,
            confidence=confidence,
        )
        return result

    # ------------------------------------------------------------------
    # Summary / convenience
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict:
        """Return a complete seasonal analysis summary."""
        now_utc = datetime.utcnow()
        return {
            "current_month": self.get_current_month_bias(),
            "current_day": self.get_current_day_bias(),
            "current_hour": self.get_hour_of_day_bias(now_utc.hour),
            "seasonal_score": self.get_current_seasonal_score(),
            "combined_signal": self.get_combined_seasonal_signal(),
        }

    def __repr__(self) -> str:
        score = self.get_current_seasonal_score()
        return f"<SeasonalAnalyzer score={score}>"
