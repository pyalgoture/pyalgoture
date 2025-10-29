from abc import abstractmethod
from collections import OrderedDict, defaultdict
from datetime import datetime
from time import time
from typing import Any

from .base import Base
from .utils.event_engine import (
    EVENT_ACCOUNT,
    EVENT_ORDER,
    EVENT_POSITION,
    EVENT_TICK,
    EVENT_TRADE,
    Event,
)
from .utils.objects import (
    AccountData,
    AssetType,
    Exchange,
    OrderData,
    OrderType,
    PositionData,
    PositionMode,
    PositionSide,
    Side,
    Status,
    TickData,
    TradeData,
)
from .utils.util_suuid import ShortUUID


class Broker(Base):
    """
    Base broker class for handling trading operations, order management, and data updates.

    This class provides the foundation for implementing specific broker connections
    and manages orders, trades, positions, and account data.
    """

    def __init__(self, base_currency: str = "USDT") -> None:
        """
        Initialize the broker with default settings.

        Args:
            base_currency: The base currency for trading operations (default: "USDT")
        """
        self.base_currency = base_currency
        self.order_count = 0
        self.shortuuid = ShortUUID(alphabet="123456789abcdefghijkmnopqrstuvwxyz")

        self.open_order_records: dict[str, OrderData] = OrderedDict()
        self.order_records: dict[str, OrderData] = OrderedDict()
        self.sent_orders: dict[str, Any] = {}

        self.trade_records: dict[str, TradeData] = OrderedDict()
        self.positions: dict[str, PositionData] = defaultdict(PositionData)
        self.accounts: dict[str, AccountData] = defaultdict(AccountData)

        self._print_tick_data: bool = False
        self._print_user_trade_data: bool = False

        self.public_ws_clients: dict[str, Any] = {}
        self.private_ws_clients: dict[str, Any] = {}
        self.private_rest_clients: dict[str, Any] = {}

        self.position_mode: PositionMode | None = None
        self.default_account: AccountData | None = None

    @abstractmethod
    def send_order(self) -> None:
        """Send a new order to server."""
        pass

    @abstractmethod
    def cancel_order(self) -> None:
        """Cancel an existing order."""
        pass

    def connect(self) -> None:
        """Start broker connection."""
        pass

    def close(self) -> None:
        """Close broker connection."""
        pass

    def subscribe(self, req: Any) -> None:
        """Subscribe tick data update."""
        pass

    def query_account(self) -> None:
        """Query account balance."""
        pass

    def query_position(self) -> None:
        """Query holding positions."""
        pass

    @property
    def available_balance(self) -> float:
        """Get the available balance from the default account."""
        return self.default_account.available if self.default_account else 0.0

    def get_default_account(self) -> AccountData | None:
        """Get the default account data."""
        return self.default_account

    def get_open_order(self, order_id: str) -> OrderData | None:
        """
        Get an open order by order ID.

        Args:
            order_id: The order ID to search for

        Returns:
            OrderData if found, None otherwise
        """
        return self.open_order_records.get(order_id)

    def get_order(self, order_id: str | None = None, seq: int | None = None) -> OrderData | None:
        """
        Get an order by order ID or sequence number.

        Args:
            order_id: The order ID to search for
            seq: The sequence number to search for

        Returns:
            OrderData if found, None otherwise
        """
        if order_id:
            return self.order_records.get(order_id)

        if isinstance(seq, int) and self.order_records and seq < len(self.order_records):
            order_keys = list(self.order_records.keys())
            return self.order_records[order_keys[seq]]

        return None

    def get_trade(self, order_id: str | None = None, seq: int | None = None) -> TradeData | None:
        """
        Get a trade by order ID or sequence number.

        Args:
            order_id: The order ID to search for
            seq: The sequence number to search for

        Returns:
            TradeData if found, None otherwise
        """
        if order_id:
            return self.trade_records.get(order_id)

        if isinstance(seq, int) and self.trade_records and seq < len(self.trade_records):
            trade_keys = list(self.trade_records.keys())
            return self.trade_records[trade_keys[seq]]

        return None

    def get_account(self, asset: str, exchange: Exchange) -> AccountData | None:
        """
        Get the selected account.

        Args:
            asset: Asset name
            exchange: Exchange enum

        Returns:
            AccountData if found, None otherwise
        """
        aio_account_id = self.ctx._get_aio_account_id(symbol=asset, exchange=exchange)
        return self.accounts.get(aio_account_id)

    def get_position_size(
        self,
        symbol: str,
        exchange: Exchange,
        position_side: PositionSide = PositionSide.NET,
    ) -> float:
        """
        Get the size of selected position.

        Args:
            symbol: Instrument name
            exchange: Exchange enum
            position_side: Position side (default: NET)

        Returns:
            Position size rounded to 6 decimal places
        """
        pos: PositionData | None = self.get_position(
            symbol=symbol,
            exchange=exchange,
            position_side=position_side,
            force_return=True,
        )
        return round(pos.size, 6) if pos else 0.0

    def get_position(
        self,
        symbol: str,
        exchange: Exchange,
        position_side: PositionSide = PositionSide.NET,
        force_return: bool = False,
    ) -> PositionData | None:
        """
        Get the selected position.

        Args:
            symbol: Instrument name
            exchange: Exchange enum
            position_side: Position side (default: NET)
            force_return: Whether to force return even if position size is 0

        Returns:
            PositionData if found and has size (or force_return is True), None otherwise
        """
        aio_position_id = self.ctx._get_aio_position_id(symbol=symbol, exchange=exchange, position_side=position_side)
        pos = self.positions.get(aio_position_id)

        if not pos:
            return None

        if force_return:
            return pos

        return pos if pos.get("size", 0) else None

    def get_trades(self) -> list[TradeData]:
        """Get all trade records."""
        return list(self.trade_records.values())

    def get_orders(self) -> list[OrderData]:
        """Get all order records."""
        return list(self.order_records.values())

    def get_open_orders(self) -> list[OrderData]:
        """Get all open order records."""
        return list(self.open_order_records.values())

    def get_positions(self) -> list[PositionData]:
        """Get all position records."""
        return list(self.positions.values())

    def get_open_positions(self) -> list[PositionData]:
        """Get all positions with non-zero size."""
        return [pos for pos in self.positions.values() if pos.size]

    def get_accounts(self) -> list[AccountData]:
        """Get all account records."""
        return list(self.accounts.values())

    def set_leverage(
        self,
        leverage: float,
        symbol: str,
        exchange: Exchange,
        position_side: PositionSide | None = None,
    ) -> bool:
        """
        Set leverage for positions.

        Args:
            leverage: Leverage value to set
            symbol: Instrument name
            exchange: Exchange enum
            position_side: Specific position side or None for all sides

        Returns:
            True if successful
        """
        position_sides = [position_side] if position_side else [PositionSide.NET, PositionSide.LONG, PositionSide.SHORT]

        for side in position_sides:
            position = self.ctx.get_position(
                symbol=symbol,
                exchange=exchange,
                position_side=side,
                force_return=True,
            )
            position.leverage = leverage
            position.set_leverage(leverage)

        return True

    def set_margin_mode(
        self,
        is_isolated: bool,
        symbol: str,
        exchange: Exchange,
        position_side: PositionSide | None = None,
    ) -> bool:
        """
        Set margin mode for positions.

        Args:
            is_isolated: Whether to use isolated margin
            symbol: Instrument name
            exchange: Exchange enum
            position_side: Specific position side or None for all sides

        Returns:
            True if successful
        """
        position_sides = [position_side] if position_side else [PositionSide.NET, PositionSide.LONG, PositionSide.SHORT]

        for side in position_sides:
            position = self.ctx.get_position(
                symbol=symbol,
                exchange=exchange,
                position_side=side,
                force_return=True,
            )
            position.is_isolated = is_isolated
            position.set_margin_mode(is_isolated)

        return True

    def add_order(self, order: OrderData) -> None:
        """Add an order to the records."""
        self.order_records[order.order_id] = order

    def add_trade(self, trade: TradeData) -> None:
        """Add a trade to the records."""
        self.trade_records[trade.order_id] = trade

    def add_account(self, account: AccountData) -> None:
        """Add an account to the records."""
        self.accounts[account.aio_account_id] = account

    def add_position(self, position: PositionData) -> None:
        """Add a position to the records."""
        self.positions[position.aio_position_id] = position

    def update_order(self, order: OrderData) -> None:
        """
        Update order records and manage open orders.

        Args:
            order: OrderData to update
        """
        order_id = order.order_id

        if order.is_active():
            self.open_order_records[order_id] = order
        else:
            self.open_order_records.pop(order_id, None)

        self.add_order(order)

    def update_trade(self, trade: TradeData) -> None:
        """
        Update trade and position records.

        Args:
            trade: TradeData to update
        """
        position = self.ctx.get_position(
            symbol=trade.symbol, exchange=trade.exchange, position_side=trade.position_side, force_return=True
        )

        if self._print_user_trade_data:
            self.debug(
                f"[update_trade (before)] aio_position_id:{getattr(trade, 'aio_position_id', '')} "
                f"trade:{trade} | position:{position}"
            )

        available_balance = self.ctx.strategy.available_balance
        position.update_by_tradefeed(trade=trade, available_balance=available_balance)
        position.update_by_marketdata(last_price=trade.price, available_balance=available_balance)

        trade.pnl = position.trade_pnl  # Note: Before fees
        trade.leverage = position.leverage

        self.add_trade(trade)

        if self._print_user_trade_data:
            self.debug(
                f"[update_trade (after)] aio_position_id:{getattr(trade, 'aio_position_id', '')} "
                f"trade:{trade} | position:{position}"
            )

    def update_account(self, account: AccountData) -> None:
        """
        Update account records and set default account if base currency.

        Args:
            account: AccountData to update
        """
        if account.asset == self.base_currency:
            if self.default_account:
                account.initial_capital = self.default_account.initial_capital
                account.capital = self.default_account.capital
            self.default_account = account

        self.accounts[account.aio_account_id] = account

    def check_position(self, position: dict[str, Any]) -> None:
        """
        Check position consistency between exchange and local records.

        Args:
            position: Position data from exchange

        Raises:
            ValueError: If position data is inconsistent
        """
        exchange = Exchange(position["exchange"].upper())
        df = self.ctx.get_datafeed(symbol=position["symbol"], exchange=exchange)
        if not df:
            return

        position_side = PositionSide(position["position_side"].upper())

        current_position = self.get_position(
            symbol=position["symbol"], exchange=exchange, position_side=position_side, force_return=True
        )

        if self._print_user_trade_data:
            self.debug(
                f"[check_position] Exchange Position(WS):{position}; Current Position:{current_position} ({self.ctx.now})"
            )

        if not current_position:
            raise ValueError(
                f"[check_position] Cannot find position for symbol:{position['symbol']}; "
                f"exchange:{position['exchange']}; position_side:{position['position_side']}"
            )

        # Check for inconsistencies
        size_mismatch = abs(current_position.size) != position["size"]
        price_mismatch = position["size"] and abs(current_position.avg_open_price - position["avg_open_price"]) > 0.0001

        if size_mismatch or price_mismatch:
            price_diff_msg = (
                f" ({current_position.avg_open_price} -> {position['avg_open_price']})" if price_mismatch else ""
            )
            size_diff_msg = f" ({current_position.size} -> {position['size']})" if size_mismatch else ""

            self.error(
                f"[check_position] Position data inconsistent. "
                f"Price Diff:{price_mismatch}{price_diff_msg}; Size Diff:{size_mismatch}{size_diff_msg} | "
                f"Current Position:{current_position}; Exchange Position:{position}"
            )

    def on_event(self, type: str, data: Any = None, priority: int = 1) -> None:
        """
        General event push.

        Args:
            type: Event type
            data: Event data
            priority: Event priority
        """
        event = Event(type=type, data=data, priority=priority)
        self.trace(f"-------> put event{type} - tick:{self.ctx.tick} - is_live:{self.ctx.is_live}")

        if self.ctx.is_live:
            self.ctx.event_engine.put(event)
        else:
            # Route events to appropriate strategy handlers
            event_handlers = {
                EVENT_ORDER: self.ctx.strategy.process_order_event,
                EVENT_TRADE: self.ctx.strategy.process_trade_event,
                EVENT_ACCOUNT: self.ctx.strategy.process_account_event,
                EVENT_POSITION: self.ctx.strategy.process_position_event,
            }

            handler = event_handlers.get(type)
            if handler:
                handler(event)

    def on_tick_callback(self, tick_data: dict[str, Any] | TickData | None) -> None:
        """
        Tick event push.

        Args:
            tick_data: Tick data from websocket or TickData object
        """
        if not tick_data:
            return

        if isinstance(tick_data, dict):
            # Data comes from websocket
            tick_data["exchange"] = Exchange(tick_data["exchange"])
            tick_data["asset_type"] = AssetType(tick_data["asset_type"])
            tick_data = TickData(**tick_data)

        if self._print_tick_data:
            self.debug(f"[on_tick_callback] tick_data:{tick_data}")

        self.on_event(type=EVENT_TICK, data=tick_data, priority=1)

    def _process_order_data(self, order: dict[str, Any]) -> OrderData | None:
        """
        Process raw order data into OrderData object.

        Args:
            order: Raw order data from exchange

        Returns:
            OrderData object or None if processing fails
        """
        try:
            exchange = Exchange(order["exchange"].upper())

            # Check if the order is managed by the strategy
            df = self.ctx.get_datafeed(symbol=order["symbol"], exchange=exchange)
            if not df:
                return None

            return OrderData(
                order_id=order["order_id"],
                ori_order_id=order["ori_order_id"],
                exchange=exchange,
                code=order["code"],
                symbol=order["symbol"],
                asset_type=AssetType(order["asset_type"]),
                price=order["price"],
                quantity=order["quantity"],
                side=Side(order["side"].upper()),
                position_side=PositionSide(order["position_side"].upper()),
                executed_quantity=order["executed_quantity"],
                order_type=OrderType(order["order_type"].upper()),
                created_at=order["created_at"],
                updated_at=order["updated_at"],
                status=Status(order["status"].upper()),
                tag=self._get_order_tag(order["order_id"]),
                msg=self._get_order_msg(order["order_id"]),
            )
        except Exception as e:
            self._log_callback_error("on_order_callback", e, order)
            return None

    def _get_order_tag(self, order_id: str) -> str:
        """Get order tag from sent orders or default."""
        return self.sent_orders[order_id]["tag"] if order_id in self.sent_orders else "From Exchange"

    def _get_order_msg(self, order_id: str) -> str:
        """Get order message from sent orders or default."""
        return self.sent_orders[order_id]["msg"] if order_id in self.sent_orders else "From Exchange"

    def on_order_callback(self, order: dict[str, Any] | OrderData) -> None:
        """
        Order event push.

        Args:
            order: Order data from exchange or OrderData object
        """
        if self._print_user_trade_data and hasattr(self, "ctx"):
            self.debug(f"[on_order_callback] order:{order} ({datetime.now()})")

        if not hasattr(self, "ctx"):
            print(f"[on_order_callback] ctx not initialized yet. order:{order}")
            return

        if not isinstance(order, OrderData):
            order = self._process_order_data(order)
            if not order:
                return

        self.on_event(type=EVENT_ORDER, data=order, priority=13)

    def _process_trade_data(self, trade: dict[str, Any]) -> TradeData | None:
        """
        Process raw trade data into TradeData object.

        Args:
            trade: Raw trade data from exchange

        Returns:
            TradeData object or None if processing fails
        """
        try:
            exchange = Exchange(trade["exchange"].upper())

            # Check if the trade is managed by the strategy
            df = self.ctx.get_datafeed(symbol=trade["symbol"], exchange=exchange)
            if not df:
                return None

            asset_type = self.ctx.get_asset_type(symbol=trade["symbol"], exchange=exchange)
            # position_side = (
            #     PositionSide(trade["position_side"].upper()) if trade.get("position_side") else PositionSide.NET
            # )
            if self.position_mode == PositionMode.ONEWAY:
                ## Oneway mode
                position_side = PositionSide.NET
            else:
                ## Hedge mode
                if trade["position_side"].upper() != "NET":
                    position_side = PositionSide(trade["position_side"].upper())
                else:
                    if trade["side"] == "BUY":
                        position_side = PositionSide.LONG if trade["is_open"] else PositionSide.SHORT
                    elif trade["side"] == "SELL":
                        position_side = PositionSide.SHORT if trade["is_open"] else PositionSide.LONG

            print(f">>>>>>>>>> position_side:{position_side}; self.position_mode:{self.position_mode};")

            if self.position_mode == PositionMode.ONEWAY:
                position_side = PositionSide.NET

            if self._print_user_trade_data:
                self.debug(
                    f"[on_trade_callback] position_mode:{self.position_mode}; "
                    f"position_side:{position_side} ({self.ctx.now})"
                )

            return TradeData(
                order_id=trade["order_id"] or trade["ori_order_id"],
                ori_order_id=trade["ori_order_id"],
                symbol=trade["symbol"],
                code=trade["code"],
                exchange=exchange,
                asset_type=asset_type,
                side=Side(trade["side"].upper()),
                quantity=trade["quantity"],
                price=trade["price"],
                traded_at=trade["traded_at"],
                position_side=position_side,
                commission=trade["commission"],
                commission_asset=trade["commission_asset"],
                is_open=trade["is_open"],
                is_maker=trade["is_maker"],
                tag=self._get_order_tag(trade["order_id"]),
                msg=self._get_order_msg(trade["order_id"]),
            )
        except Exception as e:
            self._log_callback_error("on_trade_callback", e, trade)
            return None

    def on_trade_callback(self, trade: dict[str, Any] | TradeData) -> None:
        """
        Trade event push.

        Args:
            trade: Trade data from exchange or TradeData object
        """
        if self._print_user_trade_data and hasattr(self, "ctx"):
            self.debug(f"[on_trade_callback] raw trade data:{trade} ({datetime.now()})")

        if not hasattr(self, "ctx"):
            print(f"[on_trade_callback] ctx not initialized yet. trade:{trade}")
            return

        if not isinstance(trade, TradeData):
            trade = self._process_trade_data(trade)
            if not trade:
                return

        self.on_event(type=EVENT_TRADE, data=trade, priority=12)

    def on_position_callback(self, position: dict[str, Any] | PositionData) -> None:
        """
        Position event push.

        Args:
            position: Position data from exchange or PositionData object
        """
        if self._print_user_trade_data and hasattr(self, "ctx"):
            self.debug(f"[on_position_callback] raw position data:{position} ({datetime.now()})")

        if not hasattr(self, "ctx"):
            print(f"[on_position_callback] ctx not initialized yet. position:{position}")
            return

        self.on_event(type=EVENT_POSITION, data=position, priority=11)

    def _process_account_data(self, account: dict[str, Any]) -> AccountData | None:
        """
        Process raw account data into AccountData object.

        Args:
            account: Raw account data from exchange

        Returns:
            AccountData object or None if processing fails
        """
        try:
            exchange = Exchange(account["exchange"].upper())
            return AccountData(
                asset=account["symbol"],
                exchange=exchange,
                balance=account["balance"],
                frozen=account["frozen"],
                available=account["available"],
            )
        except Exception as e:
            self._log_callback_error("on_account_callback", e, account)
            return None

    def on_account_callback(self, account: dict[str, Any] | AccountData) -> None:
        """
        Account event push.

        Args:
            account: Account data from exchange or AccountData object
        """
        if self._print_user_trade_data and hasattr(self, "ctx"):
            self.debug(f"[on_account_callback] account:{account} ({datetime.now()})")

        if not hasattr(self, "ctx"):
            print(f"[on_account_callback] ctx not initialized yet. account:{account}")
            return

        if not isinstance(account, AccountData):
            account = self._process_account_data(account)
            if not account:
                return

        self.on_event(type=EVENT_ACCOUNT, data=account, priority=14)

    def _log_callback_error(self, callback_name: str, error: Exception, data: Any) -> None:
        """
        Log callback processing errors with detailed information.

        Args:
            callback_name: Name of the callback function
            error: Exception that occurred
            data: Data that caused the error
        """
        tb_lineno = error.__traceback__.tb_lineno if error.__traceback__ else "unknown"
        tb_filename = error.__traceback__.tb_frame.f_code.co_filename if error.__traceback__ else "unknown"
        self.execption(f"Error in {callback_name} - {error} in line {tb_lineno} for file {tb_filename}. data:{data}")

    def _new_order_id(self, prefix: str = "aio") -> str:
        """
        Create new order ID.

        Args:
            prefix: Prefix for the order ID

        Returns:
            Unique order ID string
        """
        self.order_count += 1
        suffix = str(self.order_count).rjust(5, "0")
        prefix_part = f"{prefix}-" if prefix else ""

        return f"{prefix_part}{int(time())}-{self.shortuuid.uuid()[:10]}-{suffix}"
