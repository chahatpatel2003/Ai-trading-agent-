"""
Paper Broker — simulates order execution without real money.
Logs every fill, rejection, and PnL update for audit.
"""
import uuid
from datetime import datetime

import numpy as np

from broker.base import BaseBroker, Order, OrderSide, OrderStatus, Position
from logs.logger import get_logger

log = get_logger("broker.paper")


class PaperBroker(BaseBroker):
    """
    Simulated broker for paper trading.
    Tracks positions, equity, and trade history in memory.
    """

    def __init__(self, initial_equity: float, spread: float = 0.30, slippage: float = 0.10):
        self.equity = initial_equity
        self.initial_equity = initial_equity
        self.spread = spread
        self.slippage = slippage
        self.positions: dict[str, Position] = {}
        self.order_history: list[Order] = []
        self.trade_pnl_log: list[dict] = []
        self._current_prices: dict[str, tuple[float, float]] = {}
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        log.info("paper_broker_connected", equity=self.equity)
        return True

    def disconnect(self) -> None:
        self._connected = False
        log.info("paper_broker_disconnected")

    def set_price(self, symbol: str, mid_price: float) -> None:
        """Set the current market price (called by the paper trading loop)."""
        bid = mid_price - self.spread / 2
        ask = mid_price + self.spread / 2
        self._current_prices[symbol] = (bid, ask)

    def get_current_price(self, symbol: str) -> tuple[float, float]:
        if symbol not in self._current_prices:
            raise ValueError(f"No price set for {symbol}")
        return self._current_prices[symbol]

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        lots: float,
        stop_price: float,
    ) -> Order:
        if not self._connected:
            return self._rejected_order(symbol, side, lots, stop_price, "broker_disconnected")

        if symbol in self.positions:
            return self._rejected_order(symbol, side, lots, stop_price, "position_already_open")

        bid, ask = self.get_current_price(symbol)

        # Simulate fill with slippage
        if side == OrderSide.BUY:
            fill_price = ask + self.slippage
        else:
            fill_price = bid - self.slippage

        order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            lots=lots,
            entry_price=fill_price,
            stop_price=stop_price,
            status=OrderStatus.FILLED,
            fill_price=fill_price,
            slippage=self.slippage,
            timestamp=datetime.utcnow().isoformat(),
        )

        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            lots=lots,
            entry_price=fill_price,
            current_price=fill_price,
            unrealized_pnl=0.0,
            stop_price=stop_price,
        )

        self.order_history.append(order)

        log.info(
            "paper_order_filled",
            order_id=order.order_id,
            side=side.value,
            lots=lots,
            fill_price=round(fill_price, 2),
            stop=round(stop_price, 2),
        )

        return order

    def close_position(self, symbol: str) -> Order:
        if symbol not in self.positions:
            return self._rejected_order(
                symbol, OrderSide.SELL, 0, 0, "no_position_to_close"
            )

        pos = self.positions[symbol]
        bid, ask = self.get_current_price(symbol)

        # Close at adverse price + slippage
        if pos.side == OrderSide.BUY:
            close_price = bid - self.slippage
        else:
            close_price = ask + self.slippage

        # Calculate PnL
        direction = 1 if pos.side == OrderSide.BUY else -1
        pnl = direction * (close_price - pos.entry_price) * pos.lots * 100  # 100 oz per lot

        self.equity += pnl

        close_order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY,
            lots=pos.lots,
            entry_price=close_price,
            stop_price=0,
            status=OrderStatus.FILLED,
            fill_price=close_price,
            slippage=self.slippage,
            timestamp=datetime.utcnow().isoformat(),
        )

        self.trade_pnl_log.append({
            "entry_time": "paper",
            "exit_time": datetime.utcnow().isoformat(),
            "side": pos.side.value,
            "lots": pos.lots,
            "entry_price": pos.entry_price,
            "exit_price": close_price,
            "pnl": round(pnl, 2),
            "equity_after": round(self.equity, 2),
        })

        self.order_history.append(close_order)

        log.info(
            "paper_position_closed",
            symbol=symbol,
            pnl=round(pnl, 2),
            equity=round(self.equity, 2),
        )

        del self.positions[symbol]
        return close_order

    def update_position_prices(self, symbol: str) -> None:
        """Update unrealized PnL based on current price."""
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        bid, ask = self.get_current_price(symbol)
        mid = (bid + ask) / 2
        direction = 1 if pos.side == OrderSide.BUY else -1
        pos.current_price = mid
        pos.unrealized_pnl = direction * (mid - pos.entry_price) * pos.lots * 100

    def check_stop_hit(self, symbol: str) -> bool:
        """Check if current price has hit the stop loss. Returns True if stopped out."""
        if symbol not in self.positions:
            return False
        pos = self.positions[symbol]
        bid, ask = self.get_current_price(symbol)

        if pos.side == OrderSide.BUY and bid <= pos.stop_price:
            log.info("stop_hit", symbol=symbol, side="long", stop=pos.stop_price, bid=bid)
            self.close_position(symbol)
            return True
        elif pos.side == OrderSide.SELL and ask >= pos.stop_price:
            log.info("stop_hit", symbol=symbol, side="short", stop=pos.stop_price, ask=ask)
            self.close_position(symbol)
            return True
        return False

    def get_open_positions(self) -> list[Position]:
        return list(self.positions.values())

    def get_account_equity(self) -> float:
        return self.equity

    def get_spread(self, symbol: str) -> float:
        return self.spread

    def _rejected_order(self, symbol, side, lots, stop, reason) -> Order:
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            lots=lots,
            entry_price=0,
            stop_price=stop,
            status=OrderStatus.REJECTED,
            rejection_reason=reason,
            timestamp=datetime.utcnow().isoformat(),
        )
        self.order_history.append(order)
        log.warning("paper_order_rejected", reason=reason, symbol=symbol)
        return order
