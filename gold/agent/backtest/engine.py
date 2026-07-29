"""
Backtesting engine.
Simulates strategy execution bar-by-bar with realistic transaction costs,
spread, slippage, and risk management. No lookahead.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.metrics import PerformanceMetrics, compute_metrics
from logs.logger import get_logger
from risk.manager import RiskManager
from strategies.base import BaseStrategy

log = get_logger("backtest.engine")


@dataclass
class BacktestResult:
    strategy_name: str
    equity_curve: pd.Series
    trades: list[dict]
    metrics: PerformanceMetrics
    signals_generated: int
    signals_rejected: int
    rejection_reasons: dict


@dataclass
class OpenPosition:
    direction: int  # +1 long, -1 short
    entry_price: float
    stop_price: float
    lots: float
    entry_bar: int
    trailing_stop_distance: float = None


class BacktestEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.spread = cfg["costs"]["spread_points"]
        self.slippage = cfg["costs"]["slippage_points"]
        self.commission = cfg["costs"]["commission_per_lot"]
        self.lot_size = cfg["costs"]["lot_size_oz"]
        self.initial_equity = cfg["backtest"]["initial_equity"]
        # Bars per year depends on timeframe
        tf = cfg.get("general", {}).get("timeframe", "1d")
        if tf == "1d":
            self.bars_per_year = 252
        elif tf in ("1h", "60m"):
            self.bars_per_year = 252 * 23
        elif tf in ("4h", "240m"):
            self.bars_per_year = 252 * 6
        else:
            self.bars_per_year = 252

    def run(self, strategy: BaseStrategy, df: pd.DataFrame) -> BacktestResult:
        """Run a full backtest of a strategy on the given data."""
        log.info("backtest_start", strategy=strategy.name, bars=len(df))

        # Generate signals
        signals_df = strategy.generate_signals(df.copy())

        # Initialize state
        risk_mgr = RiskManager(self.cfg, self.initial_equity)
        equity = self.initial_equity
        equity_curve = []
        trades = []
        position: OpenPosition = None
        rejection_counts = {}
        signals_generated = 0
        signals_rejected = 0

        for i in range(len(signals_df)):
            bar = signals_df.iloc[i]
            bar_time = signals_df.index[i]
            risk_mgr.on_new_bar(bar_time)

            current_price = bar["close"]

            # Check if we need to close an existing position
            if position is not None:
                closed, pnl = self._check_exit(position, bar, i)
                if closed:
                    trades.append({
                        "entry_price": position.entry_price,
                        "exit_price": current_price,
                        "direction": position.direction,
                        "lots": position.lots,
                        "pnl": pnl,
                        "bars_held": i - position.entry_bar,
                        "entry_time": str(signals_df.index[position.entry_bar]),
                        "exit_time": str(bar_time),
                    })
                    risk_mgr.record_trade_result(pnl)
                    equity = risk_mgr.state.equity
                    position = None
                elif position.trailing_stop_distance:
                    # Update trailing stop
                    position = self._update_trailing_stop(position, bar)

            # Check for new signal if flat
            signal = int(bar.get("signal", 0))
            if signal != 0 and position is None:
                signals_generated += 1
                stop_price = bar.get("stop_price", np.nan)
                if pd.isna(stop_price):
                    signals_rejected += 1
                    rejection_counts["no_stop_price"] = rejection_counts.get("no_stop_price", 0) + 1
                    equity_curve.append(equity)
                    continue

                # Apply spread/slippage to entry
                if signal == 1:
                    entry_price = current_price + self.spread / 2 + self.slippage
                else:
                    entry_price = current_price - self.spread / 2 - self.slippage

                # Calculate position size via risk manager
                sizing = risk_mgr.calculate_position_size(
                    entry_price=entry_price,
                    stop_price=stop_price,
                    current_time=bar_time,
                )

                if sizing.lots <= 0:
                    signals_rejected += 1
                    reason = sizing.rejection.value
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    equity_curve.append(equity)
                    continue

                # Open position
                trailing_dist = bar.get("trailing_stop_distance", None)
                if pd.notna(trailing_dist) if isinstance(trailing_dist, float) else trailing_dist is not None:
                    trailing_dist = float(trailing_dist)
                else:
                    trailing_dist = None

                position = OpenPosition(
                    direction=signal,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    lots=sizing.lots,
                    entry_bar=i,
                    trailing_stop_distance=trailing_dist,
                )
                risk_mgr.on_position_opened()

                # Deduct commission
                commission_cost = self.commission * sizing.lots
                equity -= commission_cost

            equity_curve.append(equity)

        # Close any remaining position at the end
        if position is not None:
            final_price = signals_df.iloc[-1]["close"]
            pnl = self._calculate_pnl(position, final_price)
            trades.append({
                "entry_price": position.entry_price,
                "exit_price": final_price,
                "direction": position.direction,
                "lots": position.lots,
                "pnl": pnl,
                "bars_held": len(signals_df) - 1 - position.entry_bar,
                "entry_time": str(signals_df.index[position.entry_bar]),
                "exit_time": str(signals_df.index[-1]),
            })
            risk_mgr.record_trade_result(pnl)
            equity_curve[-1] = risk_mgr.state.equity

        eq_series = pd.Series(equity_curve, index=signals_df.index[:len(equity_curve)])
        metrics = compute_metrics(eq_series, trades, self.initial_equity, self.bars_per_year)

        log.info(
            "backtest_complete",
            strategy=strategy.name,
            trades=len(trades),
            sharpe=metrics.sharpe_ratio,
            max_dd=metrics.max_drawdown_pct,
            return_pct=metrics.total_return_pct,
        )

        return BacktestResult(
            strategy_name=strategy.name,
            equity_curve=eq_series,
            trades=trades,
            metrics=metrics,
            signals_generated=signals_generated,
            signals_rejected=signals_rejected,
            rejection_reasons=rejection_counts,
        )

    def _check_exit(self, pos: OpenPosition, bar: pd.Series, bar_idx: int) -> tuple[bool, float]:
        """Check if position should be closed (stop hit, signal flip, or target)."""
        if pos.direction == 1:
            # Long: stop hit if low <= stop_price
            if bar["low"] <= pos.stop_price:
                exit_price = pos.stop_price - self.slippage  # slippage on stop
                return True, self._calculate_pnl(pos, exit_price)
        else:
            # Short: stop hit if high >= stop_price
            if bar["high"] >= pos.stop_price:
                exit_price = pos.stop_price + self.slippage
                return True, self._calculate_pnl(pos, exit_price)

        # Check for signal reversal
        signal = int(bar.get("signal", 0))
        if signal != 0 and signal != pos.direction:
            exit_price = bar["close"] + (self.spread / 2 + self.slippage) * (-pos.direction)
            return True, self._calculate_pnl(pos, exit_price)

        # Check target for mean-reversion
        target = bar.get("target_price", None)
        if target is not None and not pd.isna(target):
            if pos.direction == 1 and bar["high"] >= target:
                return True, self._calculate_pnl(pos, float(target))
            elif pos.direction == -1 and bar["low"] <= target:
                return True, self._calculate_pnl(pos, float(target))

        return False, 0.0

    def _update_trailing_stop(self, pos: OpenPosition, bar: pd.Series) -> OpenPosition:
        """Move trailing stop in the direction of profit."""
        if pos.direction == 1:
            new_stop = bar["high"] - pos.trailing_stop_distance
            pos.stop_price = max(pos.stop_price, new_stop)
        else:
            new_stop = bar["low"] + pos.trailing_stop_distance
            pos.stop_price = min(pos.stop_price, new_stop)
        return pos

    def _calculate_pnl(self, pos: OpenPosition, exit_price: float) -> float:
        """PnL = direction * (exit - entry) * lots * lot_size - costs."""
        gross = pos.direction * (exit_price - pos.entry_price) * pos.lots * self.lot_size
        exit_costs = self.commission * pos.lots
        return gross - exit_costs
