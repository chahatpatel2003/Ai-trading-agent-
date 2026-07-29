"""
Risk Manager — YVPH-style trading discipline for gold.

Enforces position sizing, R:R, loss streaks, daily caps, and compounding
on every trade (mental, paper, and live). This module sits between the
signal generator and trade execution — no trade gets through without
passing all checks.

Rules (from user config):
    R:R minimum           1:2.5
    Max trades / day      4
    Max consecutive losses 3  -> stop trading for the day
    Capital               $1,000
    Max risk / trade      2%  ($20 at start)
    Max daily loss        $60

Compounding logic (from YVPH roadmap):
    Daily target          10%
    Reinvest gains        YES  (risk 2% of CURRENT balance, not initial)
    Withdraw at milestones YES
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from logs.logger import get_logger

log = get_logger("risk_manager")


# ─── Configuration ───

@dataclass
class RiskConfig:
    """All risk parameters in one place."""
    initial_capital: float = 1000.0
    max_risk_pct: float = 0.02           # 2% per trade
    min_rr_ratio: float = 2.5            # 1:2.5
    max_trades_per_day: int = 4
    max_loss_streak: int = 3             # stop after 3 consecutive losses
    max_daily_loss: float = 60.0         # hard dollar stop
    max_concurrent_positions: int = 2
    min_confidence: float = 0.60         # 60% for paper trading (71% for real money)
    daily_target_pct: float = 0.10       # 10% daily target (compounding)
    withdraw_at: list = field(default_factory=lambda: [2000, 3500])  # milestone withdrawals


# ─── Position Sizer ───

@dataclass
class PositionSize:
    """Computed position sizing for a single trade."""
    risk_dollars: float     # how much $ at risk
    stop_distance: float    # distance from entry to stop in price
    lot_size: float         # position size (units)
    entry_price: float
    stop_price: float
    target_price: float
    rr_ratio: float
    risk_pct: float         # actual risk as % of balance


# ─── Daily Ledger ───

@dataclass
class DailyLedger:
    """Tracks today's trading activity."""
    date: str = ""
    trades_taken: int = 0
    consecutive_losses: int = 0
    daily_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    locked_out: bool = False
    lockout_reason: str = ""
    trade_log: list = field(default_factory=list)


class RiskManager:
    """
    Central risk authority. Every trade must pass through here.

    Usage:
        rm = RiskManager(config, data_dir)

        # Before taking a trade:
        ok, reason = rm.can_trade(confidence=0.75, rr_ratio=3.0)

        # Compute position size:
        pos = rm.size_position(entry=4500, stop=4488, direction="LONG")

        # After trade resolves:
        rm.record_outcome(pnl_dollars=25.0, won=True)
    """

    def __init__(self, config: RiskConfig = None, data_dir: str = "./learn/data"):
        self.cfg = config or RiskConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Account state
        self.balance = self.cfg.initial_capital
        self.peak_balance = self.cfg.initial_capital
        self.total_withdrawn = 0.0
        self.day_count = 0

        # Today's ledger
        self.today = DailyLedger(date=str(date.today()))

        # History
        self.daily_history: list[dict] = []

        # Persistence
        self._state_path = self.data_dir / "risk_manager_state.json"
        self._load_state()

        log.info("risk_manager_initialized",
                 balance=self.balance,
                 risk_per_trade=f"${self.cfg.max_risk_pct * self.balance:.2f}",
                 max_trades=self.cfg.max_trades_per_day,
                 max_loss_streak=self.cfg.max_loss_streak)

    # ─── Gate Check ───

    def can_trade(self, confidence: float = 0.0, rr_ratio: float = 0.0) -> tuple:
        """
        Full pre-trade discipline check.

        Returns (allowed: bool, reason: str).
        """
        self._ensure_today()

        # 1. Already locked out?
        if self.today.locked_out:
            return False, f"LOCKED OUT: {self.today.lockout_reason}"

        # 2. Daily trade limit
        if self.today.trades_taken >= self.cfg.max_trades_per_day:
            self._lockout(f"Daily trade limit reached ({self.cfg.max_trades_per_day})")
            return False, self.today.lockout_reason

        # 3. Consecutive loss streak
        if self.today.consecutive_losses >= self.cfg.max_loss_streak:
            self._lockout(f"Loss streak reached ({self.cfg.max_loss_streak} consecutive)")
            return False, self.today.lockout_reason

        # 4. Daily loss cap
        if self.today.daily_pnl <= -self.cfg.max_daily_loss:
            self._lockout(f"Daily loss cap hit (${abs(self.today.daily_pnl):.2f} >= ${self.cfg.max_daily_loss:.2f})")
            return False, self.today.lockout_reason

        # 5. Minimum confidence
        if confidence < self.cfg.min_confidence:
            return False, (f"Confidence {confidence:.0%} below minimum "
                           f"{self.cfg.min_confidence:.0%}")

        # 6. Minimum R:R (with 0.01 tolerance for floating point)
        if rr_ratio < self.cfg.min_rr_ratio - 0.01:
            return False, (f"R:R {rr_ratio:.1f} below minimum "
                           f"{self.cfg.min_rr_ratio:.1f}")

        return True, "All checks passed"

    # ─── Position Sizing ───

    def size_position(self, entry: float, stop: float,
                      direction: str) -> Optional[PositionSize]:
        """
        Compute position size based on 2% risk of current balance.

        Returns PositionSize or None if the trade doesn't meet R:R.
        """
        stop_distance = abs(entry - stop)
        if stop_distance < 0.01:
            return None

        risk_dollars = self.balance * self.cfg.max_risk_pct
        lot_size = risk_dollars / stop_distance

        # Compute target from R:R
        reward_distance = stop_distance * self.cfg.min_rr_ratio
        if direction.upper() == "LONG":
            target = entry + reward_distance
        else:
            target = entry - reward_distance

        rr = reward_distance / stop_distance

        return PositionSize(
            risk_dollars=round(risk_dollars, 2),
            stop_distance=round(stop_distance, 2),
            lot_size=round(lot_size, 4),
            entry_price=entry,
            stop_price=stop,
            target_price=round(target, 2),
            rr_ratio=round(rr, 2),
            risk_pct=round(self.cfg.max_risk_pct * 100, 2),
        )

    # ─── Record Outcomes ───

    def record_outcome(self, pnl_dollars: float, won: bool,
                       strategy: str = "", direction: str = "",
                       entry: float = 0, exit_price: float = 0):
        """Record a trade result and update all counters."""
        self._ensure_today()

        self.today.trades_taken += 1
        self.today.daily_pnl += pnl_dollars

        if won:
            self.today.wins += 1
            self.today.consecutive_losses = 0
        else:
            self.today.losses += 1
            self.today.consecutive_losses += 1

        # Update balance
        self.balance += pnl_dollars
        self.peak_balance = max(self.peak_balance, self.balance)

        # Log the trade
        self.today.trade_log.append({
            "time": datetime.now().isoformat(),
            "strategy": strategy,
            "direction": direction,
            "entry": entry,
            "exit": exit_price,
            "pnl": round(pnl_dollars, 2),
            "won": won,
            "balance_after": round(self.balance, 2),
        })

        # Check if we should lock out
        if self.today.consecutive_losses >= self.cfg.max_loss_streak:
            self._lockout(f"Loss streak: {self.cfg.max_loss_streak} consecutive losses")
        elif self.today.daily_pnl <= -self.cfg.max_daily_loss:
            self._lockout(f"Daily loss cap: ${abs(self.today.daily_pnl):.2f}")

        # Check withdrawal milestones
        self._check_withdrawals()

        self._save_state()

        log.info("trade_recorded_rm",
                 pnl=round(pnl_dollars, 2),
                 won=won,
                 balance=round(self.balance, 2),
                 today_pnl=round(self.today.daily_pnl, 2),
                 streak=self.today.consecutive_losses,
                 trades=self.today.trades_taken)

    # ─── Compounding & Withdrawals ───

    def _check_withdrawals(self):
        """Check if balance hit a withdrawal milestone."""
        for milestone in self.cfg.withdraw_at:
            if self.balance >= milestone and self.total_withdrawn < milestone:
                withdraw = self.cfg.initial_capital  # withdraw initial capital
                self.balance -= withdraw
                self.total_withdrawn += withdraw
                log.info("milestone_withdrawal",
                         milestone=milestone,
                         withdrawn=withdraw,
                         balance=round(self.balance, 2),
                         total_withdrawn=round(self.total_withdrawn, 2))

    # ─── Daily Reset ───

    def daily_reset(self):
        """Archive today's ledger and start fresh."""
        if self.today.trades_taken > 0:
            self.daily_history.append(asdict(self.today))
            self.day_count += 1

        self.today = DailyLedger(date=str(date.today()))
        self._save_state()
        log.info("daily_reset",
                 balance=round(self.balance, 2),
                 day=self.day_count)

    # ─── Status ───

    def get_status(self) -> dict:
        """Current risk manager state for the dashboard."""
        self._ensure_today()
        risk_per_trade = self.balance * self.cfg.max_risk_pct
        return {
            "balance": round(self.balance, 2),
            "initial_capital": self.cfg.initial_capital,
            "peak_balance": round(self.peak_balance, 2),
            "total_withdrawn": round(self.total_withdrawn, 2),
            "day_count": self.day_count,
            "risk_per_trade": round(risk_per_trade, 2),
            "risk_pct": self.cfg.max_risk_pct * 100,
            "min_rr": self.cfg.min_rr_ratio,
            "max_trades": self.cfg.max_trades_per_day,
            "max_loss_streak": self.cfg.max_loss_streak,
            "max_daily_loss": self.cfg.max_daily_loss,
            "today": {
                "date": self.today.date,
                "trades": self.today.trades_taken,
                "wins": self.today.wins,
                "losses": self.today.losses,
                "pnl": round(self.today.daily_pnl, 2),
                "consecutive_losses": self.today.consecutive_losses,
                "locked_out": self.today.locked_out,
                "lockout_reason": self.today.lockout_reason,
            },
            "daily_history": self.daily_history[-10:],
        }

    # ─── Internal ───

    def _lockout(self, reason: str):
        self.today.locked_out = True
        self.today.lockout_reason = reason
        log.warning("trading_locked_out", reason=reason)
        self._save_state()

    def _ensure_today(self):
        """Reset if the date changed."""
        today_str = str(date.today())
        if self.today.date != today_str:
            self.daily_reset()

    def _save_state(self):
        state = {
            "balance": self.balance,
            "peak_balance": self.peak_balance,
            "total_withdrawn": self.total_withdrawn,
            "day_count": self.day_count,
            "today": asdict(self.today),
            "daily_history": self.daily_history[-30:],
            "last_saved": datetime.now().isoformat(),
        }
        try:
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            log.error("risk_state_save_failed", error=str(e))

    def _load_state(self):
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                state = json.load(f)
            self.balance = state.get("balance", self.cfg.initial_capital)
            self.peak_balance = state.get("peak_balance", self.balance)
            self.total_withdrawn = state.get("total_withdrawn", 0)
            self.day_count = state.get("day_count", 0)
            self.daily_history = state.get("daily_history", [])

            today_data = state.get("today", {})
            if today_data.get("date") == str(date.today()):
                self.today = DailyLedger(**{
                    k: v for k, v in today_data.items()
                    if k in DailyLedger.__dataclass_fields__
                })
            log.info("risk_state_loaded",
                     balance=self.balance, day=self.day_count)
        except Exception as e:
            log.warning("risk_state_load_failed", error=str(e))
