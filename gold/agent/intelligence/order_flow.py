"""
Order Flow Analyzer — OANDA order book & position book intelligence.

Uses OANDA's order book and position book endpoints to reveal where traders
have placed pending orders and where they are currently positioned. This
exposes real buy/sell pressure, stop clusters, and contrarian signals.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import oandapyV20
from oandapyV20.endpoints.instruments import (
    InstrumentsOrderBook,
    InstrumentsPositionBook,
)

from logs.logger import get_logger

logger = get_logger("order_flow")

INSTRUMENT = "XAU_USD"

# Thresholds
EXTREME_POSITIONING_PCT = 70.0  # >70% on one side = extreme
STOP_CLUSTER_TOP_N = 5          # top N stop cluster levels to return
SUPPORT_RESISTANCE_TOP_N = 5    # top N support/resistance zones


class OrderFlowAnalyzer:
    """Analyzes OANDA order book and position book data for XAU/USD."""

    def __init__(
        self,
        token: str,
        account_id: str,
        practice: bool = True,
        data_dir: str = "./learn/data",
    ) -> None:
        self.token = token
        self.account_id = account_id
        self.environment = "practice" if practice else "live"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.client = oandapyV20.API(
            access_token=self.token,
            environment=self.environment,
        )

        # Cached book data (refreshed on each fetch)
        self._order_book: Optional[Dict[str, Any]] = None
        self._position_book: Optional[Dict[str, Any]] = None

        logger.info(
            "order_flow_init",
            environment=self.environment,
            instrument=INSTRUMENT,
        )

    # ------------------------------------------------------------------
    # 1. Fetch Order Book
    # ------------------------------------------------------------------

    def fetch_order_book(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the current order book snapshot from OANDA.

        Returns dict with:
            - buckets: list of {price, long_count_percent, short_count_percent}
            - price: current snapshot price
            - timestamp: ISO timestamp of the snapshot
        """
        try:
            r = InstrumentsOrderBook(instrument=INSTRUMENT)
            self.client.request(r)
            data = r.response

            order_book = data.get("orderBook", {})
            raw_buckets = order_book.get("buckets", [])

            buckets = [
                {
                    "price": float(b["price"]),
                    "long_count_percent": float(b["longCountPercent"]),
                    "short_count_percent": float(b["shortCountPercent"]),
                }
                for b in raw_buckets
            ]

            result = {
                "buckets": buckets,
                "price": float(order_book.get("price", 0)),
                "timestamp": order_book.get("time", datetime.now(timezone.utc).isoformat()),
                "bucket_width": float(order_book.get("bucketWidth", 1.0)),
            }

            self._order_book = result

            logger.info(
                "order_book_fetched",
                price=result["price"],
                bucket_count=len(buckets),
            )
            return result

        except Exception as exc:
            logger.error("order_book_fetch_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # 2. Fetch Position Book
    # ------------------------------------------------------------------

    def fetch_position_book(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the current position book snapshot from OANDA.

        Returns dict with:
            - buckets: list of {price, long_count_percent, short_count_percent}
            - price: current snapshot price
            - timestamp: ISO timestamp of the snapshot
        """
        try:
            r = InstrumentsPositionBook(instrument=INSTRUMENT)
            self.client.request(r)
            data = r.response

            position_book = data.get("positionBook", {})
            raw_buckets = position_book.get("buckets", [])

            buckets = [
                {
                    "price": float(b["price"]),
                    "long_count_percent": float(b["longCountPercent"]),
                    "short_count_percent": float(b["shortCountPercent"]),
                }
                for b in raw_buckets
            ]

            result = {
                "buckets": buckets,
                "price": float(position_book.get("price", 0)),
                "timestamp": position_book.get("time", datetime.now(timezone.utc).isoformat()),
                "bucket_width": float(position_book.get("bucketWidth", 1.0)),
            }

            self._position_book = result

            logger.info(
                "position_book_fetched",
                price=result["price"],
                bucket_count=len(buckets),
            )
            return result

        except Exception as exc:
            logger.error("position_book_fetch_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # 3. Analyze Order Pressure
    # ------------------------------------------------------------------

    def analyze_order_pressure(self) -> Optional[Dict[str, Any]]:
        """
        Compute buy/sell pressure from the order book.

        Returns:
            - buy_pressure: sum of long_count_percent above current price
              (pending buy stops / buy limits above market)
            - sell_pressure: sum of short_count_percent below current price
              (pending sell stops / sell limits below market)
            - net_pressure: buy - sell (positive = more buying interest)
            - resistance_zones: price levels with highest sell order concentration
            - support_zones: price levels with highest buy order concentration
        """
        book = self._order_book
        if book is None:
            book = self.fetch_order_book()
        if book is None:
            return None

        current_price = book["price"]
        buckets = book["buckets"]

        if not buckets:
            logger.warning("order_pressure_no_buckets")
            return None

        # Buy pressure: pending buy orders above current price (buy stops)
        buy_pressure = sum(
            b["long_count_percent"]
            for b in buckets
            if b["price"] > current_price
        )

        # Sell pressure: pending sell orders below current price (sell stops)
        sell_pressure = sum(
            b["short_count_percent"]
            for b in buckets
            if b["price"] < current_price
        )

        net_pressure = buy_pressure - sell_pressure

        # Resistance zones: highest concentration of short (sell) orders
        sorted_by_short = sorted(
            buckets,
            key=lambda b: b["short_count_percent"],
            reverse=True,
        )
        resistance_zones = [
            {"price": b["price"], "short_count_percent": b["short_count_percent"]}
            for b in sorted_by_short[:SUPPORT_RESISTANCE_TOP_N]
            if b["short_count_percent"] > 0
        ]

        # Support zones: highest concentration of long (buy) orders
        sorted_by_long = sorted(
            buckets,
            key=lambda b: b["long_count_percent"],
            reverse=True,
        )
        support_zones = [
            {"price": b["price"], "long_count_percent": b["long_count_percent"]}
            for b in sorted_by_long[:SUPPORT_RESISTANCE_TOP_N]
            if b["long_count_percent"] > 0
        ]

        result = {
            "current_price": current_price,
            "buy_pressure": round(buy_pressure, 4),
            "sell_pressure": round(sell_pressure, 4),
            "net_pressure": round(net_pressure, 4),
            "resistance_zones": resistance_zones,
            "support_zones": support_zones,
        }

        logger.info(
            "order_pressure_analyzed",
            buy=result["buy_pressure"],
            sell=result["sell_pressure"],
            net=result["net_pressure"],
        )
        return result

    # ------------------------------------------------------------------
    # 4. Analyze Positioning
    # ------------------------------------------------------------------

    def analyze_positioning(self) -> Optional[Dict[str, Any]]:
        """
        Analyze position book to determine crowd positioning and
        generate contrarian signals.

        Returns:
            - long_percentage: total % of traders positioned long
            - short_percentage: total % of traders positioned short
            - crowd_direction: "LONG" or "SHORT"
            - contrarian_signal: opposite of crowd direction
            - extreme_positioning: True if >70% on one side
        """
        book = self._position_book
        if book is None:
            book = self.fetch_position_book()
        if book is None:
            return None

        buckets = book["buckets"]
        if not buckets:
            logger.warning("positioning_no_buckets")
            return None

        total_long = sum(b["long_count_percent"] for b in buckets)
        total_short = sum(b["short_count_percent"] for b in buckets)

        # Normalize to percentages (they should sum to ~100 each side,
        # but we compute the ratio between long and short)
        total = total_long + total_short
        if total == 0:
            logger.warning("positioning_zero_total")
            return None

        long_pct = (total_long / total) * 100.0
        short_pct = (total_short / total) * 100.0

        crowd_direction = "LONG" if long_pct > short_pct else "SHORT"
        contrarian_signal = "SHORT" if crowd_direction == "LONG" else "LONG"

        dominant_pct = max(long_pct, short_pct)
        extreme = dominant_pct >= EXTREME_POSITIONING_PCT

        result = {
            "long_percentage": round(long_pct, 2),
            "short_percentage": round(short_pct, 2),
            "crowd_direction": crowd_direction,
            "contrarian_signal": contrarian_signal,
            "extreme_positioning": extreme,
            "dominant_percentage": round(dominant_pct, 2),
        }

        logger.info(
            "positioning_analyzed",
            long_pct=result["long_percentage"],
            short_pct=result["short_percentage"],
            crowd=crowd_direction,
            extreme=extreme,
        )
        return result

    # ------------------------------------------------------------------
    # 5. Get Stop Clusters
    # ------------------------------------------------------------------

    def get_stop_clusters(self) -> Optional[List[Dict[str, Any]]]:
        """
        Find price levels where many stops are clustered.

        Stop hunts target these levels. Stops are inferred from:
        - Long stops: short orders clustered below current price
          (traders placing stop-losses on long positions)
        - Short stops: long orders clustered above current price
          (traders placing stop-losses on short positions)

        Returns list of {price, stop_density, side}.
        """
        book = self._order_book
        if book is None:
            book = self.fetch_order_book()
        if book is None:
            return None

        current_price = book["price"]
        buckets = book["buckets"]

        if not buckets:
            return []

        clusters: List[Dict[str, Any]] = []

        # Long-side stops: sell orders (short_count_percent) below current price
        # These are stop-losses placed by long holders
        below = [b for b in buckets if b["price"] < current_price]
        for b in below:
            if b["short_count_percent"] > 0:
                clusters.append({
                    "price": b["price"],
                    "stop_density": b["short_count_percent"],
                    "side": "LONG_STOPS",
                })

        # Short-side stops: buy orders (long_count_percent) above current price
        # These are stop-losses placed by short holders
        above = [b for b in buckets if b["price"] > current_price]
        for b in above:
            if b["long_count_percent"] > 0:
                clusters.append({
                    "price": b["price"],
                    "stop_density": b["long_count_percent"],
                    "side": "SHORT_STOPS",
                })

        # Sort by density descending, return top N
        clusters.sort(key=lambda c: c["stop_density"], reverse=True)
        top_clusters = clusters[:STOP_CLUSTER_TOP_N]

        logger.info(
            "stop_clusters_found",
            total_clusters=len(clusters),
            top_returned=len(top_clusters),
        )
        return top_clusters

    # ------------------------------------------------------------------
    # 6. Get Flow Summary
    # ------------------------------------------------------------------

    def get_flow_summary(self) -> Optional[Dict[str, Any]]:
        """
        Combined order flow summary with scores and recommendation.

        Returns:
            - order_pressure_score: -1 to +1 (positive = bullish pressure)
            - positioning_score: -1 to +1 (contrarian-adjusted)
            - key_support: strongest support price level
            - key_resistance: strongest resistance price level
            - extreme_alert: bool
            - recommendation: CONTRARIAN_LONG, CONTRARIAN_SHORT, or NEUTRAL
        """
        # Fetch fresh data
        order_book = self.fetch_order_book()
        position_book = self.fetch_position_book()

        if order_book is None and position_book is None:
            logger.error("flow_summary_no_data")
            return None

        # --- Order pressure score ---
        order_pressure_score = 0.0
        key_support = None
        key_resistance = None

        pressure = self.analyze_order_pressure()
        if pressure is not None:
            buy_p = pressure["buy_pressure"]
            sell_p = pressure["sell_pressure"]
            total_p = buy_p + sell_p

            if total_p > 0:
                # Normalize to -1..+1: (buy - sell) / (buy + sell)
                order_pressure_score = (buy_p - sell_p) / total_p
                order_pressure_score = max(-1.0, min(1.0, order_pressure_score))

            if pressure["support_zones"]:
                key_support = pressure["support_zones"][0]["price"]
            if pressure["resistance_zones"]:
                key_resistance = pressure["resistance_zones"][0]["price"]

        # --- Positioning score (contrarian-adjusted) ---
        positioning_score = 0.0
        extreme_alert = False

        positioning = self.analyze_positioning()
        if positioning is not None:
            long_pct = positioning["long_percentage"]
            short_pct = positioning["short_percentage"]
            extreme_alert = positioning["extreme_positioning"]

            # Contrarian: if crowd is long, score goes negative (bearish for
            # the crowd = bullish contrarian). We invert the crowd signal.
            # raw_score = (short_pct - long_pct) / 100 gives contrarian view
            positioning_score = (short_pct - long_pct) / 100.0
            positioning_score = max(-1.0, min(1.0, positioning_score))

        # --- Recommendation ---
        combined_score = (order_pressure_score * 0.4) + (positioning_score * 0.6)

        if extreme_alert:
            # Strong contrarian signal at extremes
            if positioning is not None and positioning["contrarian_signal"] == "LONG":
                recommendation = "CONTRARIAN_LONG"
            elif positioning is not None and positioning["contrarian_signal"] == "SHORT":
                recommendation = "CONTRARIAN_SHORT"
            else:
                recommendation = "NEUTRAL"
        elif combined_score > 0.2:
            recommendation = "CONTRARIAN_LONG"
        elif combined_score < -0.2:
            recommendation = "CONTRARIAN_SHORT"
        else:
            recommendation = "NEUTRAL"

        result = {
            "order_pressure_score": round(order_pressure_score, 4),
            "positioning_score": round(positioning_score, 4),
            "combined_score": round(combined_score, 4),
            "key_support": key_support,
            "key_resistance": key_resistance,
            "extreme_alert": extreme_alert,
            "recommendation": recommendation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "flow_summary",
            order_score=result["order_pressure_score"],
            position_score=result["positioning_score"],
            combined=result["combined_score"],
            recommendation=recommendation,
            extreme=extreme_alert,
        )
        return result

    # ------------------------------------------------------------------
    # 7. Save Snapshot
    # ------------------------------------------------------------------

    def save_snapshot(self) -> Optional[str]:
        """
        Save current order book and position book data to a timestamped
        JSON file for historical analysis.

        Returns the file path of the saved snapshot, or None on failure.
        """
        # Ensure we have data to save
        if self._order_book is None:
            self.fetch_order_book()
        if self._position_book is None:
            self.fetch_position_book()

        if self._order_book is None and self._position_book is None:
            logger.warning("save_snapshot_no_data")
            return None

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        snapshot_dir = self.data_dir / "order_flow_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        filename = f"order_flow_{ts}.json"
        filepath = snapshot_dir / filename

        snapshot = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "instrument": INSTRUMENT,
            "order_book": self._order_book,
            "position_book": self._position_book,
        }

        # Add analysis if data is available
        pressure = self.analyze_order_pressure()
        if pressure is not None:
            snapshot["order_pressure"] = pressure

        positioning = self.analyze_positioning()
        if positioning is not None:
            snapshot["positioning"] = positioning

        stops = self.get_stop_clusters()
        if stops is not None:
            snapshot["stop_clusters"] = stops

        try:
            with open(filepath, "w") as f:
                json.dump(snapshot, f, indent=2, default=str)

            logger.info("snapshot_saved", path=str(filepath))
            return str(filepath)

        except Exception as exc:
            logger.error("snapshot_save_failed", error=str(exc))
            return None
