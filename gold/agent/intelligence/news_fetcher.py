"""
News Fetcher — LIVE gold/market headlines with sentiment scoring.

Fetches from FXStreet RSS (free, no API key, updates in real-time).
Scores each headline for gold impact: bullish, bearish, or neutral.
"""

import re
import time
from datetime import datetime
from typing import Dict, List

import requests

from logs.logger import get_logger

log = get_logger("news_fetcher")

# RSS feeds (free, no key required)
FXSTREET_RSS = "https://www.fxstreet.com/rss/news"
INVESTING_RSS = "https://www.investing.com/rss/news_301.rss"

# Keywords for gold-relevant headlines
GOLD_KEYWORDS = [
    "gold", "xau", "precious", "bullion",
    "fed", "fomc", "rate", "nfp", "payroll", "jobs", "employment",
    "cpi", "inflation", "pce", "ppi",
    "dollar", "dxy", "usd",
    "treasury", "yield", "bond",
    "safe haven", "risk",
]

# Sentiment keywords
BULLISH_GOLD = {
    "gold surge": 0.4, "gold rally": 0.4, "gold rise": 0.3, "gold gain": 0.3,
    "gold jumps": 0.4, "gold climbs": 0.3, "bullish gold": 0.4,
    "safe haven": 0.3, "flight to safety": 0.4,
    "rate cut": 0.4, "dovish": 0.3, "fed cut": 0.4, "pause": 0.2,
    "dollar fall": 0.3, "dollar drop": 0.3, "dollar weak": 0.3, "dollar loses": 0.3,
    "dollar retreat": 0.2, "greenback slip": 0.2,
    "inflation rise": 0.2, "inflation hot": 0.2,
    "geopolitical": 0.2, "war": 0.2, "crisis": 0.2, "tension": 0.2,
    "jobs miss": 0.3, "nfp miss": 0.3, "weak jobs": 0.3, "disappointing": 0.2,
    "yield fall": 0.3, "yields drop": 0.3, "yields decline": 0.3,
    "gold demand": 0.2, "central bank buy": 0.3,
}

BEARISH_GOLD = {
    "gold slide": -0.4, "gold drop": -0.4, "gold fall": -0.4, "gold slump": -0.4,
    "gold tumble": -0.5, "gold plunge": -0.5, "gold decline": -0.3,
    "bearish gold": -0.4, "gold pressure": -0.3,
    "rate hike": -0.4, "hawkish": -0.3, "fed hawk": -0.3,
    "dollar surge": -0.3, "dollar rally": -0.3, "dollar gain": -0.3,
    "dollar strong": -0.3, "greenback rise": -0.2, "dollar boost": -0.3,
    "us dollar gain": -0.3, "usd rally": -0.3,
    "strong jobs": -0.4, "nfp beat": -0.4, "jobs beat": -0.4, "blowout nfp": -0.5,
    "stronger-than-expected": -0.3, "upbeat": -0.2,
    "yield rise": -0.3, "yields surge": -0.3, "treasury yield": -0.2,
    "risk on": -0.2, "equity rally": -0.2,
    "gold outflow": -0.3,
}


class NewsFetcher:
    """Fetches live headlines and scores gold sentiment."""

    def __init__(self, cache_seconds: int = 120):
        self._cache = {}
        self._cache_time = 0
        self._cache_seconds = cache_seconds

    def get_sentiment(self) -> Dict:
        """
        Fetch live news and return sentiment.
        Returns {
            sentiment: float (-1 to +1),
            headlines: [str] (gold-relevant),
            all_headlines: [str],
            scored: [{headline, score, keywords_matched}],
            timestamp: str,
        }
        """
        now = time.time()
        if now - self._cache_time < self._cache_seconds and self._cache:
            return self._cache

        all_headlines = []
        gold_headlines = []
        scored = []
        total_sentiment = 0

        # Fetch FXStreet (best for forex/gold)
        try:
            fxstreet = self._fetch_rss(FXSTREET_RSS)
            all_headlines.extend(fxstreet)
        except Exception as e:
            log.warning("fxstreet_fetch_failed", error=str(e))

        # Fetch Investing.com
        try:
            investing = self._fetch_rss(INVESTING_RSS)
            all_headlines.extend(investing)
        except Exception as e:
            log.warning("investing_fetch_failed", error=str(e))

        # Filter for gold-relevant headlines
        for headline in all_headlines:
            lower = headline.lower()
            if any(k in lower for k in GOLD_KEYWORDS):
                gold_headlines.append(headline)

                # Score sentiment
                score = 0
                matched = []
                for phrase, weight in BULLISH_GOLD.items():
                    if phrase in lower:
                        score += weight
                        matched.append(f"+{phrase}")
                for phrase, weight in BEARISH_GOLD.items():
                    if phrase in lower:
                        score += weight  # weight is already negative
                        matched.append(phrase)

                if score != 0:
                    scored.append({
                        "headline": headline,
                        "score": round(score, 2),
                        "keywords": matched,
                    })
                    total_sentiment += score

        # Normalize sentiment to -1..+1
        if scored:
            avg_sentiment = total_sentiment / len(scored)
            sentiment = max(-1.0, min(1.0, avg_sentiment))
        else:
            sentiment = 0.0

        result = {
            "sentiment": round(sentiment, 3),
            "headlines": gold_headlines[:10],
            "scored": sorted(scored, key=lambda x: abs(x["score"]), reverse=True)[:10],
            "total_headlines": len(all_headlines),
            "gold_relevant": len(gold_headlines),
            "timestamp": datetime.now().isoformat(),
        }

        log.info("news_fetched",
                 total=len(all_headlines),
                 gold_relevant=len(gold_headlines),
                 sentiment=round(sentiment, 3),
                 top_headline=gold_headlines[0][:60] if gold_headlines else "none")

        self._cache = result
        self._cache_time = now
        return result

    def _fetch_rss(self, url: str) -> List[str]:
        """Fetch headlines from an RSS feed."""
        resp = requests.get(url, timeout=10,
                           headers={"User-Agent": "GoldBot/1.0"})
        if resp.status_code != 200:
            return []

        # Parse titles from RSS XML
        # Handle both <title>text</title> and <title><![CDATA[text]]></title>
        titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', resp.text)
        if not titles:
            titles = re.findall(r'<title>(.*?)</title>', resp.text)

        # Clean HTML from titles
        cleaned = []
        for t in titles[1:]:  # skip feed title
            t = re.sub(r'<[^>]+>', '', t).strip()
            if t and len(t) > 10:
                cleaned.append(t)

        return cleaned
