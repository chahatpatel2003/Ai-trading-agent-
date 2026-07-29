"""
Strategy Lab — Invents new strategy hypotheses and mentally tests them.

The bot:
1. Observes what combinations of signals precede winning moves
2. Generates new strategy rules from those observations
3. Mentally simulates them on rolling data (no real trades)
4. Scores by accuracy, risk-reward, and consistency
5. Promotes strategies that pass N confirmations
"""
import json
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.indicators import rsi, macd, bollinger_bands, atr, ema, sma, adx, stochastic, obv
from logs.logger import get_logger

log = get_logger("learn.strategy_lab")


class StrategyHypothesis:
    """A strategy hypothesis — a set of rules the bot invented."""

    def __init__(self, name: str, rules: list, direction: str,
                 timeframe: str = "1h", origin: str = "observation"):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.rules = rules  # list of condition strings
        self.direction = direction  # "LONG", "SHORT", or "BOTH"
        self.timeframe = timeframe
        self.origin = origin  # what triggered this hypothesis
        self.created = str(datetime.now())

        # Mental testing results
        self.signals_generated = 0
        self.correct = 0
        self.wrong = 0
        self.partial = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.avg_bars_held = 0.0
        self.best_trade = 0.0
        self.worst_trade = 0.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

        # Status
        self.status = "TESTING"  # TESTING, PROMISING, CONFIRMED, REJECTED
        self.confidence_score = 0.0

    def to_dict(self):
        total = self.correct + self.wrong + self.partial
        win_rate = self.correct / total if total > 0 else 0
        return {
            "id": self.id,
            "name": self.name,
            "rules": self.rules,
            "direction": self.direction,
            "timeframe": self.timeframe,
            "origin": self.origin,
            "created": self.created,
            "signals_generated": self.signals_generated,
            "correct": self.correct,
            "wrong": self.wrong,
            "partial": self.partial,
            "win_rate": round(win_rate, 3),
            "total_pnl": round(self.total_pnl, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "best_trade": round(self.best_trade, 2),
            "worst_trade": round(self.worst_trade, 2),
            "max_consecutive_losses": self.max_consecutive_losses,
            "status": self.status,
            "confidence_score": round(self.confidence_score, 3),
        }


class StrategyLab:
    """Invents, tests, and evolves trading strategy hypotheses."""

    def __init__(self, data_dir: str = "./learn/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.hypotheses = []  # all strategy hypotheses
        self.signal_memory = defaultdict(list)  # what signals appeared before wins/losses
        self.min_tests = 10  # minimum mental tests before judging
        self.promotion_win_rate = 0.55  # min win rate to promote
        self.rejection_win_rate = 0.40  # below this = rejected

        self._load_state()
        self._seed_hypotheses()

    def _seed_hypotheses(self):
        """Create initial strategy hypotheses if none exist."""
        if self.hypotheses:
            return

        seeds = [
            StrategyHypothesis(
                "squeeze_momentum_long",
                ["bb_squeeze == True", "rsi > 50", "macd_hist > 0", "ema8 > ema21"],
                "LONG", origin="BB squeeze + bullish momentum alignment"
            ),
            StrategyHypothesis(
                "squeeze_momentum_short",
                ["bb_squeeze == True", "rsi < 50", "macd_hist < 0", "ema8 < ema21"],
                "SHORT", origin="BB squeeze + bearish momentum alignment"
            ),
            StrategyHypothesis(
                "triple_oversold_bounce",
                ["rsi < 30", "stoch_k < 20", "price < bb_lower", "adx > 20"],
                "LONG", origin="Triple oversold in trending market"
            ),
            StrategyHypothesis(
                "triple_overbought_fade",
                ["rsi > 70", "stoch_k > 80", "price > bb_upper", "adx > 20"],
                "SHORT", origin="Triple overbought in trending market"
            ),
            StrategyHypothesis(
                "ema_ribbon_pullback_long",
                ["ema8 > ema21 > ema55", "price > ema21", "price < ema8", "rsi > 40", "rsi < 60"],
                "LONG", origin="Pullback to EMA21 in bullish ribbon"
            ),
            StrategyHypothesis(
                "ema_ribbon_pullback_short",
                ["ema8 < ema21 < ema55", "price < ema21", "price > ema8", "rsi > 40", "rsi < 60"],
                "SHORT", origin="Rally to EMA21 in bearish ribbon"
            ),
            StrategyHypothesis(
                "volume_breakout_long",
                ["volume_ratio > 2.0", "close > bb_upper", "adx > 25", "macd_hist > 0"],
                "LONG", origin="Volume spike above BB with trend"
            ),
            StrategyHypothesis(
                "volume_breakout_short",
                ["volume_ratio > 2.0", "close < bb_lower", "adx > 25", "macd_hist < 0"],
                "SHORT", origin="Volume spike below BB with trend"
            ),
            StrategyHypothesis(
                "macd_divergence_long",
                ["rsi < 40", "macd_hist > macd_hist_prev", "price < sma50", "obv_rising"],
                "LONG", origin="Hidden bullish MACD divergence + OBV accumulation"
            ),
            StrategyHypothesis(
                "adx_trend_surge_long",
                ["adx > 30", "adx_rising", "ema8 > ema21", "rsi > 50", "rsi < 70"],
                "LONG", origin="Strong ADX trend + aligned EMAs"
            ),
            StrategyHypothesis(
                "adx_trend_surge_short",
                ["adx > 30", "adx_rising", "ema8 < ema21", "rsi < 50", "rsi > 30"],
                "SHORT", origin="Strong ADX trend + bearish EMAs"
            ),
            StrategyHypothesis(
                "round_number_bounce",
                ["near_round_50", "rsi < 45", "stoch_k < 30"],
                "LONG", origin="Oversold near round $50 level support"
            ),
        ]
        self.hypotheses = seeds
        log.info("strategy_lab_seeded", hypotheses=len(seeds))

    def compute_signals(self, df: pd.DataFrame) -> dict:
        """Compute all signals needed for hypothesis evaluation."""
        if len(df) < 60:
            return {}

        close = df["close"]
        price = float(close.iloc[-1])
        n = len(df)

        rsi_vals = rsi(close, 14)
        macd_l, sig_l, macd_h = macd(close, 12, 26, 9)
        bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20, 2.0)
        atr_vals = atr(df, 14)
        ema8 = ema(close, min(8, n - 1))
        ema21 = ema(close, min(21, n - 1))
        ema55 = ema(close, min(55, n - 1))
        sma50 = sma(close, min(50, n - 1))
        adx_vals = adx(df, min(14, n // 3)) if n > 45 else pd.Series(0, index=df.index)
        stoch_k, stoch_d = stochastic(df, 14, 3) if n > 20 else (pd.Series(50, index=df.index), pd.Series(50, index=df.index))
        obv_vals = obv(df)

        bb_width = (bb_upper - bb_lower) / bb_mid
        bb_width_avg = bb_width.rolling(50, min_periods=10).mean()

        vol_avg = df["volume"].rolling(20, min_periods=5).mean()
        vol_ratio = df["volume"] / vol_avg.replace(0, 1)

        # Round number proximity
        base_round = round(price / 50) * 50
        near_round_50 = abs(price - base_round) < float(atr_vals.iloc[-1]) * 0.5 if not pd.isna(atr_vals.iloc[-1]) else False

        # ADX rising
        adx_rising = False
        if len(adx_vals) > 3 and not pd.isna(adx_vals.iloc[-1]) and not pd.isna(adx_vals.iloc[-3]):
            adx_rising = float(adx_vals.iloc[-1]) > float(adx_vals.iloc[-3])

        # OBV rising
        obv_ema_val = ema(obv_vals, 20)
        obv_rising = float(obv_vals.iloc[-1]) > float(obv_ema_val.iloc[-1]) if not pd.isna(obv_ema_val.iloc[-1]) else False

        def _f(val):
            return float(val) if not pd.isna(val) else 0

        return {
            "price": price,
            "close": price,
            "rsi": _f(rsi_vals.iloc[-1]),
            "macd_hist": _f(macd_h.iloc[-1]),
            "macd_hist_prev": _f(macd_h.iloc[-2]) if len(macd_h) > 1 else 0,
            "bb_upper": _f(bb_upper.iloc[-1]),
            "bb_lower": _f(bb_lower.iloc[-1]),
            "bb_mid": _f(bb_mid.iloc[-1]),
            "bb_squeeze": bool(not pd.isna(bb_width_avg.iloc[-1]) and bb_width.iloc[-1] < bb_width_avg.iloc[-1] * 0.7),
            "atr": _f(atr_vals.iloc[-1]),
            "ema8": _f(ema8.iloc[-1]),
            "ema21": _f(ema21.iloc[-1]),
            "ema55": _f(ema55.iloc[-1]),
            "sma50": _f(sma50.iloc[-1]),
            "adx": _f(adx_vals.iloc[-1]),
            "adx_rising": adx_rising,
            "stoch_k": _f(stoch_k.iloc[-1]),
            "stoch_d": _f(stoch_d.iloc[-1]),
            "obv_rising": obv_rising,
            "volume_ratio": _f(vol_ratio.iloc[-1]),
            "near_round_50": near_round_50,
        }

    def evaluate_hypotheses(self, df: pd.DataFrame, signals: dict) -> list:
        """Check which hypotheses trigger on current data and mentally simulate."""
        if not signals:
            return []

        triggered = []
        price = signals["price"]

        for hyp in self.hypotheses:
            if hyp.status == "REJECTED":
                continue

            if self._check_rules(hyp.rules, signals):
                hyp.signals_generated += 1
                triggered.append(hyp)
                log.info("hypothesis_triggered",
                         name=hyp.name, direction=hyp.direction,
                         signals_so_far=hyp.signals_generated,
                         price=price)

        return triggered

    def resolve_mental_trades(self, triggered_hypotheses: list, df: pd.DataFrame,
                               entry_prices: dict, bars_since_entry: dict) -> list:
        """Check mental trades for each triggered hypothesis."""
        if len(df) < 2:
            return []

        current_price = float(df["close"].iloc[-1])
        current_high = float(df["high"].iloc[-1])
        current_low = float(df["low"].iloc[-1])
        atr_val = float(atr(df, 14).iloc[-1]) if not pd.isna(atr(df, 14).iloc[-1]) else 10

        resolved = []

        for hyp_id, entry_price in list(entry_prices.items()):
            hyp = next((h for h in self.hypotheses if h.id == hyp_id), None)
            if not hyp:
                continue

            bars = bars_since_entry.get(hyp_id, 0) + 1
            bars_since_entry[hyp_id] = bars

            target = entry_price + (atr_val * 2.5 if hyp.direction == "LONG" else -atr_val * 2.5)
            stop = entry_price + (-atr_val * 1.5 if hyp.direction == "LONG" else atr_val * 1.5)

            resolved_this = False
            pnl = 0

            if hyp.direction == "LONG":
                if current_high >= target:
                    pnl = target - entry_price
                    hyp.correct += 1
                    hyp.consecutive_losses = 0
                    resolved_this = True
                elif current_low <= stop:
                    pnl = stop - entry_price
                    hyp.wrong += 1
                    hyp.consecutive_losses += 1
                    resolved_this = True
                elif bars >= 20:
                    pnl = current_price - entry_price
                    if pnl > 0:
                        hyp.correct += 1
                        hyp.consecutive_losses = 0
                    else:
                        hyp.wrong += 1
                        hyp.consecutive_losses += 1
                    resolved_this = True
            else:
                if current_low <= target:
                    pnl = entry_price - target
                    hyp.correct += 1
                    hyp.consecutive_losses = 0
                    resolved_this = True
                elif current_high >= stop:
                    pnl = entry_price - stop
                    hyp.wrong += 1
                    hyp.consecutive_losses += 1
                    resolved_this = True
                elif bars >= 20:
                    pnl = entry_price - current_price
                    if pnl > 0:
                        hyp.correct += 1
                        hyp.consecutive_losses = 0
                    else:
                        hyp.wrong += 1
                        hyp.consecutive_losses += 1
                    resolved_this = True

            if resolved_this:
                hyp.total_pnl += pnl
                hyp.best_trade = max(hyp.best_trade, pnl)
                hyp.worst_trade = min(hyp.worst_trade, pnl)
                hyp.max_consecutive_losses = max(hyp.max_consecutive_losses, hyp.consecutive_losses)

                del entry_prices[hyp_id]
                if hyp_id in bars_since_entry:
                    del bars_since_entry[hyp_id]

                resolved.append({
                    "hypothesis": hyp.name,
                    "direction": hyp.direction,
                    "entry": entry_price,
                    "exit": current_price,
                    "pnl": round(pnl, 2),
                    "outcome": "CORRECT" if pnl > 0 else "WRONG",
                    "bars_held": bars,
                })

                # Update status
                self._update_status(hyp)

                log.info("mental_trade_resolved",
                         hypothesis=hyp.name,
                         outcome="CORRECT" if pnl > 0 else "WRONG",
                         pnl=round(pnl, 2),
                         status=hyp.status)

        return resolved

    def invent_from_observations(self, winning_signals: list, losing_signals: list):
        """Invent new hypotheses based on observed winning/losing signal combinations."""
        if len(winning_signals) < 5:
            return

        # Find common features in winning signals
        common_win = self._find_common_conditions(winning_signals)
        common_lose = self._find_common_conditions(losing_signals) if losing_signals else []

        # Generate new hypothesis from winning patterns NOT present in losing patterns
        unique_win = [c for c in common_win if c not in common_lose]

        if len(unique_win) >= 3:
            # Determine direction from the signals
            avg_rsi = np.mean([s.get("rsi", 50) for s in winning_signals])
            direction = "LONG" if avg_rsi < 50 else "SHORT"

            name = f"invented_{str(uuid.uuid4())[:6]}"
            hyp = StrategyHypothesis(
                name=name,
                rules=unique_win[:5],
                direction=direction,
                origin=f"Auto-invented from {len(winning_signals)} winning observations",
            )
            self.hypotheses.append(hyp)
            log.info("hypothesis_invented",
                     name=name,
                     rules=unique_win[:5],
                     direction=direction)

    def _find_common_conditions(self, signal_list: list) -> list:
        """Find conditions that are common across signal snapshots."""
        conditions = []

        # Check RSI ranges
        rsi_vals = [s.get("rsi", 50) for s in signal_list]
        avg_rsi = np.mean(rsi_vals)
        if avg_rsi < 35:
            conditions.append("rsi < 35")
        elif avg_rsi > 65:
            conditions.append("rsi > 65")

        # Check MACD
        macd_vals = [s.get("macd_hist", 0) for s in signal_list]
        if np.mean(macd_vals) > 0 and sum(1 for v in macd_vals if v > 0) / len(macd_vals) > 0.7:
            conditions.append("macd_hist > 0")
        elif np.mean(macd_vals) < 0 and sum(1 for v in macd_vals if v < 0) / len(macd_vals) > 0.7:
            conditions.append("macd_hist < 0")

        # Check ADX
        adx_vals = [s.get("adx", 0) for s in signal_list]
        if np.mean(adx_vals) > 25:
            conditions.append("adx > 25")

        # Check squeeze
        squeeze_pct = sum(1 for s in signal_list if s.get("bb_squeeze")) / len(signal_list)
        if squeeze_pct > 0.5:
            conditions.append("bb_squeeze == True")

        # Check volume
        vol_ratios = [s.get("volume_ratio", 1) for s in signal_list]
        if np.mean(vol_ratios) > 1.5:
            conditions.append("volume_ratio > 1.5")

        # Check EMA alignment
        ema_bull_pct = sum(1 for s in signal_list
                          if s.get("ema8", 0) > s.get("ema21", 0) > s.get("ema55", 0)) / len(signal_list)
        if ema_bull_pct > 0.7:
            conditions.append("ema8 > ema21 > ema55")

        ema_bear_pct = sum(1 for s in signal_list
                          if s.get("ema8", 0) < s.get("ema21", 0) < s.get("ema55", 0)) / len(signal_list)
        if ema_bear_pct > 0.7:
            conditions.append("ema8 < ema21 < ema55")

        return conditions

    def _check_rules(self, rules: list, signals: dict) -> bool:
        """Check if all rules are satisfied by current signals."""
        for rule in rules:
            try:
                if not eval(rule, {"__builtins__": {}}, signals):
                    return False
            except Exception:
                return False
        return True

    def _update_status(self, hyp: StrategyHypothesis):
        """Update hypothesis status based on results."""
        total = hyp.correct + hyp.wrong + hyp.partial
        if total < self.min_tests:
            hyp.status = "TESTING"
            hyp.confidence_score = 0
            return

        win_rate = hyp.correct / total

        # Confidence = win rate weighted by sample size and consistency
        sample_factor = min(1.0, total / 30)  # more tests = more confident
        consistency = 1.0 - (hyp.max_consecutive_losses / max(total, 1))
        hyp.confidence_score = win_rate * 0.6 + sample_factor * 0.2 + consistency * 0.2

        if win_rate >= self.promotion_win_rate and total >= self.min_tests:
            if hyp.total_pnl > 0 and hyp.max_consecutive_losses < 5:
                hyp.status = "CONFIRMED"
            else:
                hyp.status = "PROMISING"
        elif win_rate < self.rejection_win_rate and total >= self.min_tests:
            hyp.status = "REJECTED"
        else:
            hyp.status = "TESTING"

    def get_lab_report(self) -> str:
        """Generate the strategy lab report."""
        lines = []
        lines.append("STRATEGY LAB REPORT")
        lines.append("=" * 70)
        lines.append(f"Total hypotheses: {len(self.hypotheses)}")
        lines.append(f"Confirmed: {sum(1 for h in self.hypotheses if h.status == 'CONFIRMED')}")
        lines.append(f"Promising: {sum(1 for h in self.hypotheses if h.status == 'PROMISING')}")
        lines.append(f"Testing: {sum(1 for h in self.hypotheses if h.status == 'TESTING')}")
        lines.append(f"Rejected: {sum(1 for h in self.hypotheses if h.status == 'REJECTED')}")

        # Sort by confidence
        sorted_hyps = sorted(self.hypotheses, key=lambda h: h.confidence_score, reverse=True)

        lines.append(f"\n{'Name':<30s} {'Dir':>5s} {'W':>4s} {'L':>4s} {'WR%':>6s} {'PnL':>8s} {'Status':<12s}")
        lines.append("-" * 70)

        for h in sorted_hyps:
            total = h.correct + h.wrong + h.partial
            wr = h.correct / total * 100 if total > 0 else 0
            lines.append(
                f"  {h.name:<28s} {h.direction:>5s} {h.correct:>4d} {h.wrong:>4d} "
                f"{wr:>5.1f}% {h.total_pnl:>+7.1f} {h.status:<12s}"
            )
            if h.status in ("CONFIRMED", "PROMISING"):
                lines.append(f"    Rules: {' AND '.join(h.rules)}")
                lines.append(f"    Origin: {h.origin}")

        return "\n".join(lines)

    def save_state(self):
        state = {
            "last_updated": str(datetime.now()),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
        }
        path = self.data_dir / "strategy_lab.json"
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self):
        path = self.data_dir / "strategy_lab.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                state = json.load(f)
            for h_data in state.get("hypotheses", []):
                h = StrategyHypothesis(
                    name=h_data["name"],
                    rules=h_data["rules"],
                    direction=h_data["direction"],
                    timeframe=h_data.get("timeframe", "1h"),
                    origin=h_data.get("origin", "loaded"),
                )
                h.id = h_data.get("id", h.id)
                h.created = h_data.get("created", h.created)
                h.signals_generated = h_data.get("signals_generated", 0)
                h.correct = h_data.get("correct", 0)
                h.wrong = h_data.get("wrong", 0)
                h.partial = h_data.get("partial", 0)
                h.total_pnl = h_data.get("total_pnl", 0)
                h.best_trade = h_data.get("best_trade", 0)
                h.worst_trade = h_data.get("worst_trade", 0)
                h.max_consecutive_losses = h_data.get("max_consecutive_losses", 0)
                h.status = h_data.get("status", "TESTING")
                h.confidence_score = h_data.get("confidence_score", 0)
                self.hypotheses.append(h)
            log.info("strategy_lab_loaded", hypotheses=len(self.hypotheses))
        except Exception as e:
            log.warning("strategy_lab_load_failed", error=str(e))
