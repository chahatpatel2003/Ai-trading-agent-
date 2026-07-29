"""
Paper Trader — executes trades through the PaperBroker with real spread/slippage.

Runs alongside mental trades and predictions. Every signal that passes all gates
(risk manager, optimizer, evolver, ICT, thinker) gets executed as a paper trade
through the PaperBroker with realistic fill simulation.

Also includes:
- Missed opportunity detector (what moves did we miss and why)
- Strategy inventor (connect dots from data to hypothesize new entries)
- Progress tracker toward 200-trade / 70% WR goal
- Regime-segmented performance tracking

The paper trader does NOT replace mental trades or predictions — it runs in
parallel as the execution validation layer the council demanded.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from broker.base import OrderSide, OrderStatus
from broker.paper import PaperBroker
from logs.logger import get_logger

# Try to import OANDA broker, fall back to local simulation
try:
    from broker.oanda_paper import OandaPaperBroker
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False

log = get_logger("paper_trader")

MIN_TRADES_FOR_REAL = 200
TARGET_WIN_RATE = 0.70


@dataclass
class PaperTrade:
    """A paper trade with full context."""
    trade_id: str
    strategy: str
    direction: str
    entry_price: float
    fill_price: float          # after spread + slippage
    stop_price: float
    target_price: float
    lots: float
    risk_dollars: float
    confidence: float
    regime: str
    session: str
    reasons: List[str]
    entry_time: str
    # Outcome (filled when closed)
    exit_price: float = 0.0
    exit_time: str = ""
    pnl_dollars: float = 0.0
    pnl_points: float = 0.0
    outcome: str = ""           # WON, LOST, EXPIRED
    bars_held: int = 0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    slippage_cost: float = 0.0
    spread_cost: float = 0.0


class PaperTrader:
    """
    Executes paper trades through PaperBroker and tracks everything.
    """

    def __init__(self, data_dir: str = "./learn/data",
                 initial_equity: float = 1000.0,
                 spread: float = 0.30, slippage: float = 0.10,
                 use_oanda: bool = True):
        self.data_dir = Path(data_dir)
        self._state_path = self.data_dir / "paper_trader_state.json"

        # Use OANDA practice account if available, else local simulation
        self.use_oanda = use_oanda and OANDA_AVAILABLE
        if self.use_oanda:
            self.broker = OandaPaperBroker()
            connected = self.broker.connect()
            if not connected:
                log.warning("oanda_fallback_to_local",
                            reason="OANDA connection failed, using local PaperBroker")
                self.broker = PaperBroker(initial_equity, spread, slippage)
                self.broker.connect()
                self.use_oanda = False
        else:
            self.broker = PaperBroker(initial_equity, spread, slippage)
            self.broker.connect()

        # State
        self.open_trades: Dict[str, PaperTrade] = {}  # keyed by strategy
        self.closed_trades: List[dict] = []
        self.trade_counter = 0

        # Regime-segmented tracking
        self.regime_performance: Dict[str, dict] = {}  # {regime: {wins, losses, pnl, ...}}

        # Progress toward goal
        self.total_trades = 0
        self.total_wins = 0
        self.total_losses = 0

        # Missed opportunities
        self.missed_moves: List[dict] = []

        # Strategy inventor: hypotheses being tested
        self.invented_strategies: List[dict] = []

        self._load_state()
        broker_type = "OANDA Practice" if self.use_oanda else "Local Simulation"
        equity = self.broker.get_account_equity() if self.use_oanda else getattr(self.broker, 'equity', initial_equity)
        log.info("paper_trader_ready",
                 broker=broker_type,
                 equity=equity,
                 open=len(self.open_trades),
                 total_trades=self.total_trades)

    # ─── Execute Paper Trade ───

    def can_open(self, strategy: str) -> bool:
        """One trade per strategy at a time."""
        return strategy not in self.open_trades

    def execute_trade(self, strategy: str, direction: str,
                      current_price: float, stop_price: float,
                      target_price: float, lots: float,
                      risk_dollars: float, confidence: float,
                      regime: str, session: str,
                      reasons: list) -> Optional[PaperTrade]:
        """Execute a paper trade — on OANDA demo or local simulation."""
        if not self.can_open(strategy):
            return None

        side = OrderSide.BUY if direction == "LONG" else OrderSide.SELL

        if self.use_oanda:
            # Real OANDA order with stop and target
            order = self.broker.place_order(
                "XAU_USD", side, lots, stop_price, target_price)
        else:
            # Local simulation
            self.broker.set_price("XAU_USD", current_price)
            order = self.broker.place_order("XAU_USD", side, lots, stop_price)

        if order.status != OrderStatus.FILLED:
            log.warning("paper_trade_rejected",
                        strategy=strategy, reason=order.rejection_reason)
            return None

        self.trade_counter += 1
        trade = PaperTrade(
            trade_id=f"pt_{strategy}_{self.trade_counter}",
            strategy=strategy,
            direction=direction,
            entry_price=current_price,
            fill_price=order.fill_price,
            stop_price=stop_price,
            target_price=target_price,
            lots=lots,
            risk_dollars=risk_dollars,
            confidence=confidence,
            regime=regime,
            session=session,
            reasons=reasons,
            entry_time=datetime.now().isoformat(),
            slippage_cost=abs(order.fill_price - current_price),
            spread_cost=self.broker.spread,
        )

        self.open_trades[strategy] = trade
        self._save_state()

        log.info("paper_trade_executed",
                 strategy=strategy, direction=direction,
                 entry=current_price, fill=order.fill_price,
                 stop=stop_price, target=target_price,
                 lots=lots, regime=regime)
        return trade

    # ─── Check Outcomes ───

    def check_outcomes(self, current_price: float) -> List[PaperTrade]:
        """Check all open paper trades against current price."""
        resolved = []
        to_remove = []

        if not self.use_oanda:
            self.broker.set_price("XAU_USD", current_price)

        # For OANDA: check if trades are still open on the server
        if self.use_oanda:
            oanda_trades = self.broker.get_open_trades()
            oanda_trade_ids = {t.get("instrument"): t for t in oanda_trades}

        for strategy, trade in self.open_trades.items():
            trade.bars_held += 1

            # Track MFE/MAE
            if trade.direction == "LONG":
                unrealized = current_price - trade.fill_price
            else:
                unrealized = trade.fill_price - current_price
            trade.max_favorable = max(trade.max_favorable, unrealized)
            trade.max_adverse = min(trade.max_adverse, unrealized)

            # For OANDA: check if the server already closed the trade (stop/target hit)
            if self.use_oanda:
                if "XAU_USD" not in oanda_trade_ids and trade.bars_held > 1:
                    # Trade was closed by OANDA (stop or target hit server-side)
                    # Determine outcome from price vs levels
                    if trade.direction == "LONG":
                        if current_price >= trade.target_price:
                            trade.outcome = "WON"
                            trade.exit_price = trade.target_price
                        else:
                            trade.outcome = "LOST"
                            trade.exit_price = trade.stop_price
                    else:
                        if current_price <= trade.target_price:
                            trade.outcome = "WON"
                            trade.exit_price = trade.target_price
                        else:
                            trade.outcome = "LOST"
                            trade.exit_price = trade.stop_price
                    if trade.direction == "LONG":
                        trade.pnl_points = trade.exit_price - trade.fill_price
                    else:
                        trade.pnl_points = trade.fill_price - trade.exit_price
                    trade.pnl_dollars = trade.pnl_points * trade.lots
                    trade.exit_time = datetime.now().isoformat()
                    resolved.append(trade)
                    to_remove.append(strategy)
                    continue

            # Local broker: check stop (OANDA handles stops server-side)
            if not self.use_oanda:
                stopped = self.broker.check_stop_hit("XAU_USD")
                if stopped:
                    trade.outcome = "LOST"
                    trade.exit_price = trade.stop_price
                    trade.pnl_points = -abs(trade.fill_price - trade.stop_price)
                    trade.pnl_dollars = trade.pnl_points * trade.lots
                    trade.exit_time = datetime.now().isoformat()
                    resolved.append(trade)
                    to_remove.append(strategy)
                    continue

            # Check expiry — close on OANDA or locally
            if trade.bars_held >= 20 and strategy not in to_remove:
                if self.use_oanda:
                    self.broker.close_position("XAU_USD")
                else:
                    self.broker.close_position("XAU_USD")
                trade.outcome = "EXPIRED"
                trade.exit_price = current_price
                if trade.direction == "LONG":
                    trade.pnl_points = current_price - trade.fill_price
                else:
                    trade.pnl_points = trade.fill_price - current_price
                trade.pnl_dollars = trade.pnl_points * trade.lots
                trade.exit_time = datetime.now().isoformat()
                resolved.append(trade)
                to_remove.append(strategy)

        for s in to_remove:
            del self.open_trades[s]

        # Update counters
        for trade in resolved:
            self.total_trades += 1
            if trade.outcome == "WON":
                self.total_wins += 1
            elif trade.outcome == "LOST":
                self.total_losses += 1

            # Regime tracking
            r = trade.regime
            if r not in self.regime_performance:
                self.regime_performance[r] = {
                    "wins": 0, "losses": 0, "expired": 0,
                    "pnl": 0.0, "trades": 0
                }
            self.regime_performance[r]["trades"] += 1
            self.regime_performance[r]["pnl"] += trade.pnl_dollars
            if trade.outcome == "WON":
                self.regime_performance[r]["wins"] += 1
            elif trade.outcome == "LOST":
                self.regime_performance[r]["losses"] += 1
            else:
                self.regime_performance[r]["expired"] += 1

            self.closed_trades.append(asdict(trade))
            self.closed_trades = self.closed_trades[-500:]

        if resolved:
            self._save_state()

        return resolved

    # ─── Missed Opportunity Detector ───

    def detect_missed_opportunities(self, price_data: dict,
                                     active_strategies: list,
                                     current_price: float,
                                     open_mental_trades: list = None,
                                     regime: str = "",
                                     market_changes: dict = None) -> List[dict]:
        """
        After each cycle, check: did gold make a big move that we missed?
        If so, figure out WHY we missed it, and what to do differently.
        Not just a log — this changes the bot's future behavior.
        """
        missed = []

        for tf, df in price_data.items():
            if df is None or len(df) < 10:
                continue

            # Check last 5 bars for significant moves
            recent_high = float(df["high"].iloc[-5:].max())
            recent_low = float(df["low"].iloc[-5:].min())
            move_size = recent_high - recent_low
            avg_range = float((df["high"] - df["low"]).iloc[-20:].mean())

            if move_size <= avg_range * 2.0:
                continue

            move_dir = "UP" if float(df["close"].iloc[-1]) > float(df["open"].iloc[-5]) else "DOWN"

            # ─── ANSWER: WHY did we miss it? ───
            reasons_missed = []
            action_items = []

            # 1. Were we holding the WRONG direction?
            if open_mental_trades:
                for t in open_mental_trades:
                    if isinstance(t, dict):
                        t_dir = t.get("direction", "")
                    else:
                        t_dir = getattr(t, "direction", "")
                    if (move_dir == "DOWN" and t_dir == "LONG") or (move_dir == "UP" and t_dir == "SHORT"):
                        strat = t.get("strategy", "") if isinstance(t, dict) else getattr(t, "strategy", "")
                        reasons_missed.append(
                            f"Was holding {t_dir} ({strat}) while market went {move_dir} — WRONG DIRECTION")
                        action_items.append(
                            f"Add regime/trend filter to {strat}: don't go {t_dir} "
                            f"when higher TFs show {move_dir} trend")

            # 2. Did a strategy signal correctly but get blocked?
            if not self.open_trades and not open_mental_trades:
                reasons_missed.append(
                    f"Had NO position — all strategies were blocked by confidence/risk gates")
                action_items.append(
                    "Lower minimum confidence from 71% to 65% for paper trading — "
                    "need to take trades to learn")

            # 3. What was the market context?
            if market_changes:
                dxy_chg = market_changes.get("dxy", 0)
                us10y_chg = market_changes.get("us10y", 0)
                vix_chg = market_changes.get("vix", 0)
                if move_dir == "DOWN" and dxy_chg > 0.2:
                    reasons_missed.append(f"DXY rose +{dxy_chg:.2f}% — clear SHORT signal ignored")
                if move_dir == "DOWN" and us10y_chg > 0.3:
                    reasons_missed.append(f"Yields rose +{us10y_chg:.2f}% — bearish gold signal")
                if move_dir == "UP" and vix_chg > 1.0:
                    reasons_missed.append(f"VIX spiked +{vix_chg:.1f}% — should have been LONG gold")

            # 4. What was the regime?
            if regime:
                if move_dir == "DOWN" and regime in ("deflation_fear", "risk_off"):
                    reasons_missed.append(
                        f"Regime was {regime} — bearish for gold, should have been SHORT")
                elif move_dir == "UP" and regime in ("risk_off", "uncertainty"):
                    reasons_missed.append(
                        f"Regime was {regime} — safe-haven buying, should have been LONG")

            if not reasons_missed:
                reasons_missed.append(
                    f"Gold moved {move_dir} ${move_size:.0f} on {tf} and we had no position")

            missed_opp = {
                "timestamp": datetime.now().isoformat(),
                "timeframe": tf,
                "move_direction": move_dir,
                "move_size": round(move_size, 2),
                "avg_range": round(avg_range, 2),
                "reasons_missed": reasons_missed,
                "action_items": action_items,
                "question": f"Why wasn't I in this ${move_size:.0f} {move_dir} move on {tf}?",
                "regime": regime,
            }
            missed.append(missed_opp)
            self.missed_moves.append(missed_opp)
            self.missed_moves = self.missed_moves[-50:]

            log.info("missed_opportunity",
                     tf=tf, direction=move_dir,
                     size=round(move_size, 2),
                     reasons=reasons_missed[:2],
                     actions=action_items[:2])

        if missed:
            self._save_state()
        return missed

    # ─── Strategy Inventor ───

    def invent_strategy(self, winning_trades: list, losing_trades: list,
                        market_data: dict) -> Optional[dict]:
        """
        Look at what winning trades have in common that losing trades don't.
        Hypothesize a new entry rule and add it to invented_strategies for testing.
        """
        if len(winning_trades) < 3:
            return None

        # Find common conditions in winners
        win_regimes = [t.get("regime", "?") for t in winning_trades]
        win_sessions = [t.get("session", "?") for t in winning_trades]
        win_directions = [t.get("direction", "?") for t in winning_trades]
        win_confidences = [t.get("confidence", 0) for t in winning_trades]

        from collections import Counter
        best_regime = Counter(win_regimes).most_common(1)[0] if win_regimes else ("?", 0)
        best_session = Counter(win_sessions).most_common(1)[0] if win_sessions else ("?", 0)
        best_direction = Counter(win_directions).most_common(1)[0] if win_directions else ("?", 0)
        avg_win_conf = sum(win_confidences) / len(win_confidences) if win_confidences else 0

        # Only invent if there's a clear pattern
        if best_regime[1] < len(winning_trades) * 0.6:
            return None  # No dominant regime in winners

        hypothesis = {
            "name": f"invented_{best_direction[0].lower()}_{best_regime[0]}_{best_session[0]}",
            "direction": best_direction[0],
            "regime_filter": best_regime[0],
            "session_filter": best_session[0],
            "min_confidence": round(avg_win_conf * 0.9, 2),
            "reasoning": (
                f"Winning trades cluster in {best_regime[0]} regime during {best_session[0]} session. "
                f"{best_direction[0]} direction dominates ({best_direction[1]}/{len(winning_trades)}). "
                f"Avg winning confidence: {avg_win_conf:.0%}."
            ),
            "status": "MENTAL_TEST",
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "created": datetime.now().isoformat(),
        }

        # Check if we already have this hypothesis
        existing = [s["name"] for s in self.invented_strategies]
        if hypothesis["name"] not in existing:
            self.invented_strategies.append(hypothesis)
            self.invented_strategies = self.invented_strategies[-20:]
            self._save_state()
            log.info("strategy_invented", name=hypothesis["name"],
                     reasoning=hypothesis["reasoning"])
            return hypothesis

        return None

    # ─── Progress Tracking ───

    def get_progress(self) -> dict:
        """Progress toward 200-trade / 70% WR goal."""
        wr = self.total_wins / self.total_trades if self.total_trades > 0 else 0
        trades_remaining = max(0, MIN_TRADES_FOR_REAL - self.total_trades)
        wr_on_track = wr >= TARGET_WIN_RATE

        return {
            "total_trades": self.total_trades,
            "target_trades": MIN_TRADES_FOR_REAL,
            "trades_remaining": trades_remaining,
            "progress_pct": round(self.total_trades / MIN_TRADES_FOR_REAL * 100, 1),
            "wins": self.total_wins,
            "losses": self.total_losses,
            "win_rate": round(wr, 3),
            "target_win_rate": TARGET_WIN_RATE,
            "wr_on_track": wr_on_track,
            "ready_for_real": self.total_trades >= MIN_TRADES_FOR_REAL and wr >= TARGET_WIN_RATE,
            "equity": round(self.broker.get_account_equity() if self.use_oanda else getattr(self.broker, "equity", 0), 2),
            "regime_performance": self.regime_performance,
            "missed_opportunities": len(self.missed_moves),
            "invented_strategies": len(self.invented_strategies),
        }

    def get_status(self) -> dict:
        return {
            "progress": self.get_progress(),
            "open_trades": [asdict(t) for t in self.open_trades.values()],
            "closed_trades": self.closed_trades[-20:],
            "missed_moves": self.missed_moves[-10:],
            "invented_strategies": self.invented_strategies,
            "regime_performance": self.regime_performance,
        }

    # ─── Persistence ───

    def _save_state(self):
        state = {
            "equity": self.broker.get_account_equity() if self.use_oanda else getattr(self.broker, 'equity', 0),
            "trade_counter": self.trade_counter,
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "open_trades": {k: asdict(v) for k, v in self.open_trades.items()},
            "closed_trades": self.closed_trades[-500:],
            "regime_performance": self.regime_performance,
            "missed_moves": self.missed_moves[-50:],
            "invented_strategies": self.invented_strategies,
            "last_saved": datetime.now().isoformat(),
        }
        try:
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            log.error("paper_trader_save_failed", error=str(e))

    def _load_state(self):
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                state = json.load(f)
            if not self.use_oanda and hasattr(self.broker, 'equity'):
                self.broker.equity = state.get("equity", 1000)
            self.trade_counter = state.get("trade_counter", 0)
            self.total_trades = state.get("total_trades", 0)
            self.total_wins = state.get("total_wins", 0)
            self.total_losses = state.get("total_losses", 0)
            self.closed_trades = state.get("closed_trades", [])
            self.regime_performance = state.get("regime_performance", {})
            self.missed_moves = state.get("missed_moves", [])
            self.invented_strategies = state.get("invented_strategies", [])

            for strategy, data in state.get("open_trades", {}).items():
                valid = {k: v for k, v in data.items() if k in PaperTrade.__dataclass_fields__}
                self.open_trades[strategy] = PaperTrade(**valid)

            _eq = self.broker.get_account_equity() if self.use_oanda else getattr(self.broker, "equity", 0)
            log.info("paper_trader_loaded", trades=self.total_trades, equity=_eq)
        except Exception as e:
            log.warning("paper_trader_load_failed", error=str(e))


# Need pandas import for missed opportunity detector
import pandas as pd
