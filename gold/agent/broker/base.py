"""
Broker abstraction layer.
Paper and live brokers implement this interface identically.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    lots: float
    entry_price: float
    stop_price: float
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float = 0.0
    slippage: float = 0.0
    timestamp: str = ""
    rejection_reason: str = ""


@dataclass
class Position:
    symbol: str
    side: OrderSide
    lots: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_price: float


class BaseBroker(ABC):
    """Interface that all broker adapters must implement."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the broker."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the broker."""
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> tuple[float, float]:
        """Returns (bid, ask) for the symbol."""
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        lots: float,
        stop_price: float,
    ) -> Order:
        """Place a market order with a stop loss."""
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> Order:
        """Close the current position for a symbol."""
        pass

    @abstractmethod
    def get_open_positions(self) -> list[Position]:
        """Return all open positions."""
        pass

    @abstractmethod
    def get_account_equity(self) -> float:
        """Return current account equity."""
        pass

    @abstractmethod
    def get_spread(self, symbol: str) -> float:
        """Return current spread for the symbol."""
        pass
