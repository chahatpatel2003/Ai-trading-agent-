"""
Paper Trading Runner.
Runs strategies against live-ish data in paper mode.
Logs every signal, fill, rejection, and PnL for comparison against backtest.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from broker.base import OrderSide, OrderStatus
from broker.paper import PaperBroker
from config.loader import load_config
from data.provider import get_data
from logs.logger import get_logger, setup_logging
from risk.manager import RiskManager
from strategies.base import BaseStrategy, Signal
from strategies.breakout import BreakoutStrategy
from strategies.ensemble import EnsembleStrategy
from strategies.ichimoku_cloud import IchimokuCloudStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.momentum_trend import MomentumTrendStrategy
from strategies.multi_timeframe import MultiTimeframeStrategy
from strategies.rsi_divergence import RSIDivergenceStrategy
from strategies.session_momentum import SessionMomentumStrategy
from strategies.smart_money import SmartMoneyStrategy
from strategies.stochastic_rsi import StochasticRSIStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.triple_ema import TripleEMAStrategy
from strategies.volatility_breakout import VolatilityBreakoutStrategy
from strategies.volume_price import VolumePriceStrategy

log = get_logger("paper_trade.runner")


class PaperTradeRunner:
    """
    Runs one or more strategies in paper trading mode.
    Simulates real-time bar-by-bar execution using historical data
    or live data feed (when connected).
    """

    def __init__(self, cfg: dict, strategies: list[BaseStrategy]):
        self.cfg = cfg
        self.strategies = strategies
        pt_cfg = cfg["paper_trade"]
        self.initial_equity = pt_cfg["initial_equity"]
        self.log_signals = pt_cfg.get("log_every_signal", True)
        self.log_rejections = pt_cfg.get("log_rejections", True)

        # Each strategy gets its own broker and risk manager
        self.brokers: dict[str, PaperBroker] = {}
        self.risk_managers: dict[str, RiskManager] = {}
        self.signal_logs: dict[str, list] = {}

        for strat in strategies:
            self.brokers[strat.name] = PaperBroker(
                initial_equity=self.initial_equity,
                spread=cfg["costs"]["spread_points"],
                slippage=cfg["costs"]["slippage_points"],
            )
            self.risk_managers[strat.name] = RiskManager(cfg, self.initial_equity)
            self.signal_logs[strat.name] = []

    def run_on_historical(self, df: pd.DataFrame) -> dict:
        """
        Run paper trading simulation on historical data bar-by-bar.
        Returns performance summary for each strategy.
        """
        log.info("paper_trade_start", strategies=[s.name for s in self.strategies], bars=len(df))

        results = {}

        for strat in self.strategies:
            broker = self.brokers[strat.name]
            risk_mgr = self.risk_managers[strat.name]
            broker.connect()

            # Generate signals for the full dataset
            signals_df = strat.generate_signals(df.copy())

            for i in range(len(signals_df)):
                bar = signals_df.iloc[i]
                bar_time = signals_df.index[i]
                current_price = bar["close"]

                # Update broker price
                broker.set_price("XAUUSD", current_price)
                risk_mgr.on_new_bar(bar_time)

                # Check stops on existing positions
                had_position = len(broker.get_open_positions()) > 0
                broker.check_stop_hit("XAUUSD")
                broker.update_position_prices("XAUUSD")

                # If a stop was hit, sync the risk manager
                if had_position and not broker.get_open_positions():
                    if broker.trade_pnl_log:
                        last_trade = broker.trade_pnl_log[-1]
                        risk_mgr.record_trade_result(last_trade["pnl"])

                signal = int(bar.get("signal", 0))
                stop_price = bar.get("stop_price", None)

                # Log the signal
                signal_entry = {
                    "timestamp": str(bar_time),
                    "signal": signal,
                    "price": current_price,
                    "stop_price": stop_price,
                    "action": "none",
                    "rejection_reason": "",
                }

                if signal != 0 and not broker.get_open_positions():
                    if stop_price is None or pd.isna(stop_price):
                        signal_entry["action"] = "rejected"
                        signal_entry["rejection_reason"] = "no_stop_price"
                    else:
                        # Size the position
                        entry = current_price
                        sizing = risk_mgr.calculate_position_size(
                            entry_price=entry,
                            stop_price=float(stop_price),
                            current_time=bar_time,
                        )

                        if sizing.lots > 0:
                            side = OrderSide.BUY if signal == 1 else OrderSide.SELL
                            order = broker.place_order(
                                "XAUUSD", side, sizing.lots, float(stop_price)
                            )
                            if order.status == OrderStatus.FILLED:
                                signal_entry["action"] = "filled"
                                signal_entry["fill_price"] = order.fill_price
                                signal_entry["lots"] = sizing.lots
                                risk_mgr.on_position_opened()
                            else:
                                signal_entry["action"] = "rejected"
                                signal_entry["rejection_reason"] = order.rejection_reason
                        else:
                            signal_entry["action"] = "rejected"
                            signal_entry["rejection_reason"] = sizing.rejection.value

                # Check for signal reversal — close existing position
                elif signal != 0 and broker.get_open_positions():
                    pos = broker.get_open_positions()[0]
                    current_side = 1 if pos.side == OrderSide.BUY else -1
                    if signal != current_side:
                        broker.close_position("XAUUSD")
                        if broker.trade_pnl_log:
                            last_trade = broker.trade_pnl_log[-1]
                            risk_mgr.record_trade_result(last_trade["pnl"])
                        signal_entry["action"] = "reversal_close"

                self.signal_logs[strat.name].append(signal_entry)

            # Close any remaining position
            if broker.get_open_positions():
                broker.close_position("XAUUSD")
                if broker.trade_pnl_log:
                    last_trade = broker.trade_pnl_log[-1]
                    risk_mgr.record_trade_result(last_trade["pnl"])

            broker.disconnect()

            results[strat.name] = {
                "final_equity": round(broker.equity, 2),
                "total_return_pct": round(
                    (broker.equity - self.initial_equity) / self.initial_equity * 100, 2
                ),
                "total_trades": len(broker.trade_pnl_log),
                "winning_trades": sum(1 for t in broker.trade_pnl_log if t["pnl"] > 0),
                "losing_trades": sum(1 for t in broker.trade_pnl_log if t["pnl"] <= 0),
                "total_pnl": round(sum(t["pnl"] for t in broker.trade_pnl_log), 2),
                "trade_log": broker.trade_pnl_log,
                "signal_count": len([s for s in self.signal_logs[strat.name] if s["signal"] != 0]),
                "rejections": len([s for s in self.signal_logs[strat.name] if s["action"] == "rejected"]),
                "risk_state": risk_mgr.get_state_summary(),
            }

            log.info(
                "paper_trade_complete",
                strategy=strat.name,
                equity=results[strat.name]["final_equity"],
                trades=results[strat.name]["total_trades"],
                return_pct=results[strat.name]["total_return_pct"],
            )

        return results

    def save_logs(self, output_dir: str = "./logs") -> None:
        """Save all signal and trade logs to JSON files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for strat_name, signals in self.signal_logs.items():
            path = out / f"paper_signals_{strat_name}_{timestamp}.json"
            with open(path, "w") as f:
                json.dump(signals, f, indent=2, default=str)
            log.info("logs_saved", strategy=strat_name, path=str(path))


def build_strategies(cfg: dict) -> list[BaseStrategy]:
    """Instantiate all enabled strategies from config."""
    strategies = []
    scfg = cfg["strategies"]

    if scfg["trend_following"]["enabled"]:
        strategies.append(TrendFollowingStrategy(cfg))
    if scfg["mean_reversion"]["enabled"]:
        strategies.append(MeanReversionStrategy(cfg))
    if scfg["breakout"]["enabled"]:
        strategies.append(BreakoutStrategy(cfg))
    if scfg["momentum"]["enabled"]:
        strategies.append(MomentumStrategy(cfg))
    if scfg.get("multi_timeframe", {}).get("enabled"):
        strategies.append(MultiTimeframeStrategy(cfg))
    if scfg.get("ichimoku_cloud", {}).get("enabled"):
        strategies.append(IchimokuCloudStrategy(cfg))
    if scfg.get("rsi_divergence", {}).get("enabled"):
        strategies.append(RSIDivergenceStrategy(cfg))
    if scfg.get("volume_price", {}).get("enabled"):
        strategies.append(VolumePriceStrategy(cfg))
    if scfg.get("stochastic_rsi", {}).get("enabled"):
        strategies.append(StochasticRSIStrategy(cfg))
    if scfg.get("momentum_trend", {}).get("enabled"):
        strategies.append(MomentumTrendStrategy(cfg))
    if scfg.get("session_momentum", {}).get("enabled"):
        strategies.append(SessionMomentumStrategy(cfg))
    if scfg.get("volatility_breakout", {}).get("enabled"):
        strategies.append(VolatilityBreakoutStrategy(cfg))
    if scfg.get("triple_ema", {}).get("enabled"):
        strategies.append(TripleEMAStrategy(cfg))
    if scfg.get("smart_money", {}).get("enabled"):
        strategies.append(SmartMoneyStrategy(cfg))
    if scfg["ensemble"]["enabled"] and len(strategies) >= 2:
        strategies.append(EnsembleStrategy(cfg, strategies.copy()))

    return strategies
