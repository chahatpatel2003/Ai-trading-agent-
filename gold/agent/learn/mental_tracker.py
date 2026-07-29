"""
Mental Trade Tracker — tracks open mental trades and resolves them.

Every mental trade gets an entry, stop, target, and expiry horizon.
Each cycle, the tracker checks current price against these levels:
  - Price hit TARGET  → WON, compute P&L
  - Price hit STOP    → LOST, compute P&L
  - Bars expired      → EXIT at current price, compute actual P&L
  - Still open        → track max favorable / max adverse excursion

Only ONE trade per strategy allowed at a time (deduplication).
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from logs.logger import get_logger

log = get_logger("mental_tracker")

# Bar durations in seconds for horizon calculation
TF_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400,
}


@dataclass
class MentalTrade:
    """A single tracked mental trade."""
    trade_id: str
    strategy: str
    direction: str          # LONG or SHORT
    entry_price: float
    stop_price: float
    target_price: float
    timeframe: str          # e.g. "1h"
    confidence: float
    reasons: list
    rr_ratio: float
    risk_dollars: float
    lot_size: float
    entry_time: str
    horizon_bars: int = 20  # max bars before force exit
    bars_elapsed: int = 0
    max_favorable: float = 0.0   # best unrealized P&L in price points
    max_adverse: float = 0.0     # worst unrealized P&L in price points
    market_context: dict = field(default_factory=dict)

    # Resolution fields (filled when closed)
    resolved: bool = False
    outcome: str = ""       # WON, LOST, EXPIRED
    exit_price: float = 0.0
    exit_time: str = ""
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    bars_held: int = 0


class MentalTradeTracker:
    """
    Tracks open mental trades and resolves them against live prices.

    Usage:
        tracker = MentalTradeTracker(data_dir)

        # Before taking a trade — check if strategy already has one open
        if tracker.can_open("real_yield"):
            tracker.open_trade(trade)

        # Each cycle — resolve trades against current price
        resolved = tracker.check_outcomes(current_price)
        for trade in resolved:
            # feed to risk manager, log to journal
    """

    def __init__(self, data_dir: str = "./learn/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.open_trades: dict[str, MentalTrade] = {}  # keyed by strategy
        self.resolved_trades: list[dict] = []

        self._state_path = self.data_dir / "mental_trades_state.json"
        self._load_state()

        log.info("mental_tracker_ready", open_trades=len(self.open_trades))

    # ─── Open Trade ───

    def can_open(self, strategy: str) -> bool:
        """Check if strategy has no open trade (deduplication)."""
        return strategy not in self.open_trades

    def open_trade(self, trade: MentalTrade):
        """Register a new mental trade for tracking."""
        if not self.can_open(trade.strategy):
            log.info("mental_trade_skipped_duplicate",
                     strategy=trade.strategy,
                     existing_entry=self.open_trades[trade.strategy].entry_price)
            return False

        self.open_trades[trade.strategy] = trade
        self._save_state()

        log.info("mental_trade_opened",
                 strategy=trade.strategy,
                 direction=trade.direction,
                 entry=trade.entry_price,
                 stop=trade.stop_price,
                 target=trade.target_price,
                 rr=trade.rr_ratio,
                 risk=trade.risk_dollars,
                 horizon=trade.horizon_bars)
        return True

    # ─── Outcome Resolution ───

    def check_outcomes(self, current_price: float) -> list[MentalTrade]:
        """
        Check all open trades against current price.
        Returns list of just-resolved trades.
        """
        resolved = []
        to_remove = []

        for strategy, trade in self.open_trades.items():
            trade.bars_elapsed += 1

            # Track max favorable / adverse excursion
            if trade.direction == "LONG":
                unrealized = current_price - trade.entry_price
            else:
                unrealized = trade.entry_price - current_price

            trade.max_favorable = max(trade.max_favorable, unrealized)
            trade.max_adverse = min(trade.max_adverse, unrealized)

            # Check TARGET hit
            if trade.direction == "LONG" and current_price >= trade.target_price:
                self._resolve(trade, "WON", trade.target_price)
                resolved.append(trade)
                to_remove.append(strategy)

            elif trade.direction == "SHORT" and current_price <= trade.target_price:
                self._resolve(trade, "WON", trade.target_price)
                resolved.append(trade)
                to_remove.append(strategy)

            # Check STOP hit
            elif trade.direction == "LONG" and current_price <= trade.stop_price:
                self._resolve(trade, "LOST", trade.stop_price)
                resolved.append(trade)
                to_remove.append(strategy)

            elif trade.direction == "SHORT" and current_price >= trade.stop_price:
                self._resolve(trade, "LOST", trade.stop_price)
                resolved.append(trade)
                to_remove.append(strategy)

            # Check EXPIRED — exit at current price on last bar
            elif trade.bars_elapsed >= trade.horizon_bars:
                self._resolve(trade, "EXPIRED", current_price)
                resolved.append(trade)
                to_remove.append(strategy)

        for s in to_remove:
            del self.open_trades[s]

        if resolved:
            self._save_state()

        return resolved

    def _resolve(self, trade: MentalTrade, outcome: str, exit_price: float):
        """Fill resolution fields on a trade."""
        trade.resolved = True
        trade.outcome = outcome
        trade.exit_price = exit_price
        trade.exit_time = datetime.now().isoformat()
        trade.bars_held = trade.bars_elapsed

        if trade.direction == "LONG":
            trade.pnl_points = exit_price - trade.entry_price
        else:
            trade.pnl_points = trade.entry_price - exit_price

        # P&L in dollars = lot_size * pnl_points
        trade.pnl_dollars = round(trade.lot_size * trade.pnl_points, 2)

        self.resolved_trades.append(asdict(trade))

        # Keep only last 200 resolved trades
        self.resolved_trades = self.resolved_trades[-200:]

        log.info("mental_trade_resolved",
                 strategy=trade.strategy,
                 outcome=outcome,
                 direction=trade.direction,
                 entry=trade.entry_price,
                 exit=exit_price,
                 pnl_pts=round(trade.pnl_points, 2),
                 pnl_dollars=trade.pnl_dollars,
                 bars=trade.bars_held,
                 mfe=round(trade.max_favorable, 2),
                 mae=round(trade.max_adverse, 2))

    # ─── Status ───

    def get_open_trades(self) -> list[dict]:
        return [asdict(t) for t in self.open_trades.values()]

    def get_resolved_trades(self) -> list[dict]:
        return self.resolved_trades

    def get_stats(self) -> dict:
        """Summary statistics of resolved trades."""
        if not self.resolved_trades:
            return {"total": 0}

        wins = [t for t in self.resolved_trades if t["outcome"] == "WON"]
        losses = [t for t in self.resolved_trades if t["outcome"] == "LOST"]
        expired = [t for t in self.resolved_trades if t["outcome"] == "EXPIRED"]

        total_pnl = sum(t["pnl_dollars"] for t in self.resolved_trades)
        win_rate = len(wins) / len(self.resolved_trades) if self.resolved_trades else 0

        # Per-strategy breakdown
        by_strategy = {}
        for t in self.resolved_trades:
            s = t["strategy"]
            if s not in by_strategy:
                by_strategy[s] = {"wins": 0, "losses": 0, "expired": 0,
                                  "pnl": 0, "trades": 0}
            by_strategy[s]["trades"] += 1
            by_strategy[s]["pnl"] += t["pnl_dollars"]
            if t["outcome"] == "WON":
                by_strategy[s]["wins"] += 1
            elif t["outcome"] == "LOST":
                by_strategy[s]["losses"] += 1
            else:
                by_strategy[s]["expired"] += 1

        return {
            "total": len(self.resolved_trades),
            "wins": len(wins),
            "losses": len(losses),
            "expired": len(expired),
            "win_rate": round(win_rate, 3),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(self.resolved_trades), 2),
            "by_strategy": by_strategy,
            "open_count": len(self.open_trades),
        }

    # ─── Persistence ───

    def _save_state(self):
        state = {
            "open_trades": {k: asdict(v) for k, v in self.open_trades.items()},
            "resolved_trades": self.resolved_trades,
            "last_saved": datetime.now().isoformat(),
        }
        try:
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            log.error("mental_tracker_save_failed", error=str(e))

    def _load_state(self):
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                state = json.load(f)

            self.resolved_trades = state.get("resolved_trades", [])

            for strategy, data in state.get("open_trades", {}).items():
                valid_fields = {k: v for k, v in data.items()
                                if k in MentalTrade.__dataclass_fields__}
                self.open_trades[strategy] = MentalTrade(**valid_fields)

            log.info("mental_tracker_loaded",
                     open=len(self.open_trades),
                     resolved=len(self.resolved_trades))
        except Exception as e:
            log.warning("mental_tracker_load_failed", error=str(e))
