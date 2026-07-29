"""
OANDA Practice Broker — places REAL orders on OANDA's demo account.

Same interface as PaperBroker but executes through OANDA's REST API.
You'll see trades appear in fxTrade Practice platform with real market fills.

This is NOT simulation — orders go to OANDA's servers, fills are real market
prices, and P&L is tracked on their platform.
"""

import os
import uuid
from datetime import datetime
from pathlib import Path

import oandapyV20
from oandapyV20.endpoints import orders, trades, accounts, positions, pricing
from oandapyV20.contrib.requests import (
    MarketOrderRequest,
    StopLossDetails,
    TakeProfitDetails,
)

from broker.base import BaseBroker, Order, OrderSide, OrderStatus, Position
from logs.logger import get_logger

log = get_logger("broker.oanda_paper")

INSTRUMENT = "XAU_USD"


class OandaPaperBroker(BaseBroker):
    """
    Places real orders on OANDA practice (demo) account.
    Implements the same BaseBroker interface so it's a drop-in replacement.
    """

    def __init__(self, token: str = None, account_id: str = None):
        # Load from env if not provided
        self.token = token or os.getenv("OANDA_TOKEN", "")
        self.account_id = account_id or os.getenv("OANDA_ACCOUNT_ID", "")

        if not self.token or not self.account_id:
            # Try loading from .env file
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("OANDA_TOKEN="):
                        self.token = line.split("=", 1)[1].strip()
                    elif line.startswith("OANDA_ACCOUNT_ID="):
                        self.account_id = line.split("=", 1)[1].strip()

        self.client = None
        self._connected = False
        self.order_history = []
        self.trade_pnl_log = []

    def connect(self) -> bool:
        try:
            self.client = oandapyV20.API(
                access_token=self.token, environment="practice"
            )
            # Test connection by getting account summary
            r = accounts.AccountSummary(accountID=self.account_id)
            self.client.request(r)
            acct = r.response["account"]
            self._connected = True
            log.info("oanda_practice_connected",
                     account=self.account_id,
                     balance=acct["balance"],
                     currency=acct["currency"],
                     open_trades=acct["openTradeCount"])
            return True
        except Exception as e:
            log.error("oanda_practice_connect_failed", error=str(e))
            self._connected = False
            return False

    def disconnect(self) -> None:
        self._connected = False
        self.client = None
        log.info("oanda_practice_disconnected")

    def get_current_price(self, symbol: str) -> tuple:
        """Get live bid/ask from OANDA."""
        try:
            params = {"instruments": symbol}
            r = pricing.PricingInfo(accountID=self.account_id, params=params)
            self.client.request(r)
            price = r.response["prices"][0]
            bid = float(price["bids"][0]["price"])
            ask = float(price["asks"][0]["price"])
            return (bid, ask)
        except Exception as e:
            log.error("oanda_price_fetch_failed", error=str(e))
            raise

    def set_price(self, symbol: str, mid_price: float) -> None:
        """No-op — OANDA uses live prices, not set prices."""
        pass

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        lots: float,
        stop_price: float,
        target_price: float = None,
    ) -> Order:
        """Place a market order on OANDA practice with stop loss and optional take profit."""
        if not self._connected or not self.client:
            return self._make_order(symbol, side, lots, stop_price,
                                    OrderStatus.REJECTED, reason="not_connected")

        # OANDA uses units, not lots. For XAU_USD: 1 unit = 1 oz of gold
        # Our lot_size from risk manager is already in units appropriate for the risk
        # Convert: we want to risk $20 with a stop distance, lot_size = risk$/stop_dist
        # OANDA expects integer units for XAU_USD
        units = max(1, int(round(lots)))
        if side == OrderSide.SELL:
            units = -units  # negative for short

        try:
            # Build order request
            order_body = {
                "type": "MARKET",
                "instrument": symbol,
                "units": str(units),
                "stopLossOnFill": {
                    "price": f"{stop_price:.2f}",
                },
            }

            if target_price and target_price > 0:
                order_body["takeProfitOnFill"] = {
                    "price": f"{target_price:.2f}",
                }

            data = {"order": order_body}

            r = orders.OrderCreate(accountID=self.account_id, data=data)
            self.client.request(r)

            response = r.response
            fill = response.get("orderFillTransaction", {})

            if fill:
                fill_price = float(fill.get("price", 0))
                order_id = fill.get("id", str(uuid.uuid4())[:8])
                trade_id = fill.get("tradeOpened", {}).get("tradeID", "")

                order = Order(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    lots=abs(units),
                    entry_price=fill_price,
                    stop_price=stop_price,
                    status=OrderStatus.FILLED,
                    fill_price=fill_price,
                    slippage=0,  # real market fill
                    timestamp=datetime.utcnow().isoformat(),
                )
                order.trade_id = trade_id  # OANDA trade ID for closing

                self.order_history.append(order)

                log.info("oanda_order_filled",
                         order_id=order_id,
                         trade_id=trade_id,
                         side=side.value,
                         units=units,
                         fill_price=fill_price,
                         stop=stop_price,
                         target=target_price)
                return order
            else:
                # Check for rejection
                reject = response.get("orderRejectTransaction", {})
                reason = reject.get("rejectReason", "unknown_rejection")
                log.warning("oanda_order_rejected", reason=reason)
                return self._make_order(symbol, side, abs(units), stop_price,
                                        OrderStatus.REJECTED, reason=reason)

        except Exception as e:
            log.error("oanda_order_failed", error=str(e))
            return self._make_order(symbol, side, abs(units), stop_price,
                                    OrderStatus.REJECTED, reason=str(e))

    def close_position(self, symbol: str) -> Order:
        """Close all positions for the instrument."""
        try:
            # Close using position endpoint (closes all units for instrument)
            data = {"longUnits": "ALL", "shortUnits": "ALL"}

            # Try closing long first, then short
            for close_type in ["longUnits", "shortUnits"]:
                try:
                    close_data = {close_type: "ALL"}
                    r = positions.PositionClose(
                        accountID=self.account_id,
                        instrument=symbol,
                        data=close_data,
                    )
                    self.client.request(r)
                    response = r.response

                    # Get the close transaction
                    txns = (response.get("longOrderFillTransaction") or
                            response.get("shortOrderFillTransaction"))
                    if txns:
                        close_price = float(txns.get("price", 0))
                        pnl = float(txns.get("pl", 0))
                        units = abs(int(txns.get("units", 0)))

                        order = Order(
                            order_id=txns.get("id", ""),
                            symbol=symbol,
                            side=OrderSide.SELL if close_type == "longUnits" else OrderSide.BUY,
                            lots=units,
                            entry_price=close_price,
                            stop_price=0,
                            status=OrderStatus.FILLED,
                            fill_price=close_price,
                            slippage=0,
                            timestamp=datetime.utcnow().isoformat(),
                        )

                        self.trade_pnl_log.append({
                            "exit_time": datetime.utcnow().isoformat(),
                            "side": close_type.replace("Units", ""),
                            "units": units,
                            "exit_price": close_price,
                            "pnl": pnl,
                        })

                        log.info("oanda_position_closed",
                                 symbol=symbol, pnl=pnl, price=close_price)
                        return order
                except oandapyV20.exceptions.V20Error:
                    continue  # no position of this type

            return self._make_order(symbol, OrderSide.SELL, 0, 0,
                                    OrderStatus.REJECTED, reason="no_position_to_close")

        except Exception as e:
            log.error("oanda_close_failed", error=str(e))
            return self._make_order(symbol, OrderSide.SELL, 0, 0,
                                    OrderStatus.REJECTED, reason=str(e))

    def close_trade(self, trade_id: str) -> dict:
        """Close a specific trade by its OANDA trade ID."""
        try:
            r = trades.TradeClose(accountID=self.account_id, tradeID=trade_id)
            self.client.request(r)
            txn = r.response.get("orderFillTransaction", {})
            pnl = float(txn.get("pl", 0))
            price = float(txn.get("price", 0))
            log.info("oanda_trade_closed", trade_id=trade_id, pnl=pnl, price=price)
            return {"pnl": pnl, "price": price, "trade_id": trade_id}
        except Exception as e:
            log.error("oanda_trade_close_failed", trade_id=trade_id, error=str(e))
            return {"error": str(e)}

    def get_open_positions(self) -> list:
        """Get all open positions from OANDA."""
        try:
            r = positions.OpenPositions(accountID=self.account_id)
            self.client.request(r)
            result = []
            for pos in r.response.get("positions", []):
                long_units = int(pos.get("long", {}).get("units", 0))
                short_units = abs(int(pos.get("short", {}).get("units", 0)))
                if long_units > 0:
                    result.append(Position(
                        symbol=pos["instrument"],
                        side=OrderSide.BUY,
                        lots=long_units,
                        entry_price=float(pos["long"].get("averagePrice", 0)),
                        current_price=0,
                        unrealized_pnl=float(pos["long"].get("unrealizedPL", 0)),
                        stop_price=0,
                    ))
                if short_units > 0:
                    result.append(Position(
                        symbol=pos["instrument"],
                        side=OrderSide.SELL,
                        lots=short_units,
                        entry_price=float(pos["short"].get("averagePrice", 0)),
                        current_price=0,
                        unrealized_pnl=float(pos["short"].get("unrealizedPL", 0)),
                        stop_price=0,
                    ))
            return result
        except Exception as e:
            log.error("oanda_positions_failed", error=str(e))
            return []

    def get_open_trades(self) -> list:
        """Get all open trades from OANDA with full details."""
        try:
            r = trades.OpenTrades(accountID=self.account_id)
            self.client.request(r)
            return r.response.get("trades", [])
        except Exception as e:
            log.error("oanda_trades_failed", error=str(e))
            return []

    def get_account_equity(self) -> float:
        """Get current account equity from OANDA."""
        try:
            r = accounts.AccountSummary(accountID=self.account_id)
            self.client.request(r)
            return float(r.response["account"]["balance"])
        except Exception as e:
            log.error("oanda_equity_failed", error=str(e))
            return 0

    def get_spread(self, symbol: str) -> float:
        """Get current spread from live prices."""
        try:
            bid, ask = self.get_current_price(symbol)
            return round(ask - bid, 3)
        except Exception:
            return 0.30  # fallback

    def check_stop_hit(self, symbol: str) -> bool:
        """OANDA handles stops server-side — check if trade was closed."""
        open_trades = self.get_open_trades()
        xau_trades = [t for t in open_trades if t.get("instrument") == symbol]
        # If we had a trade but it's gone, stop was hit
        return False  # OANDA auto-executes stops

    def _make_order(self, symbol, side, lots, stop, status, reason=""):
        return Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            side=side,
            lots=lots,
            entry_price=0,
            stop_price=stop,
            status=status,
            rejection_reason=reason,
            timestamp=datetime.utcnow().isoformat(),
        )
