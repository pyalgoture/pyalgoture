import time
from collections.abc import Callable
from typing import Any

from ..broker import Broker
from ..exchanges.dummy import DummyClient
from ..utils.objects import (
    AssetType,
    Exchange,
    OrderData,
    PositionData,
    PositionMode,
    PositionSide,
    Side,
    Status,
    TickData,
    TradeData,
)
from ..utils.util_connection import get_exchange_connection


class RealtimeBroker(Broker):
    """Real-time broker for handling live trading operations.

    Provides callbacks for connection lifecycle and market events:
    - on_connect, on_disconnect, on_error
    - on_tick, on_account, on_trade, on_order, on_position
    """

    def __init__(
        self,
        base_currency: str = "USDT",
        **kwargs: Any,
    ) -> None:
        """Initialize the realtime broker.

        Args:
            base_currency: Base currency for trading (default: "USDT")
            exchange: Exchange to connect to
            **kwargs: Additional configuration parameters
        """
        super().__init__(base_currency=base_currency)
        self.default_account: Any = None

    def initialize(self) -> None:
        """Initialize broker connections and setup accounts/positions."""
        if len(self.private_rest_clients) > 1:
            self.warning("There are multiple connections, please be careful, the function may not work as expected.")

        for connection_key, rest_client in self.private_rest_clients.items():
            self._setup_client_permissions(rest_client)
            self._setup_default_account()
            self._setup_existing_positions(connection_key, rest_client)

    def _setup_client_permissions(self, rest_client: Any) -> None:
        """Setup client permissions if available."""
        if hasattr(rest_client, "query_permission"):
            permission_result = rest_client.query_permission()
            self.debug(f"Exchange permission: {permission_result}")

    def _setup_default_account(self) -> None:
        """Set up the default account based on base currency."""
        self.query_account()
        time.sleep(1.5)  # Allow time for account setup
        for account_id, account in self.accounts.items():
            if account.asset == self.base_currency:
                self.default_account = account
                break

        if not self.default_account:
            raise ValueError(f"No default account found for {self.base_currency}.")

    def _setup_existing_positions(self, connection_key: str, rest_client: Any) -> None:
        """Setup existing positions for all symbols."""
        # NOTE: option must enable cross margin
        # NOTE: query position and check if match current setup (leverage, position_mode, margin_mode)
        # TODO: fetch vip level and fee rate
        for symbol, exchange in self.ctx.symbols:
            feed = self.ctx.get_datafeed(symbol=symbol, exchange=exchange)
            current_position = self.ctx.get_position(symbol=symbol, exchange=exchange, force_return=True)

            position_result = rest_client.query_position(symbol=feed.code, force_return=True)

            if position_result.success and position_result.data:
                self._sync_position_with_exchange(position_result.data, current_position, symbol, exchange, feed)

    def _sync_position_with_exchange(
        self, exchange_positions: list[dict], current_position: PositionData, symbol: str, exchange: Exchange, feed: Any
    ) -> None:
        """Synchronize position with exchange data."""
        exchange_position = exchange_positions[0]
        self.debug(
            f"[{symbol} - {exchange}] Position: {exchange_position} | "
            f"leverage:{exchange_position['leverage']}({current_position.leverage}) | "
            f"is_isolated:{exchange_position['is_isolated']}({current_position.is_isolated})"
        )

        # Sync leverage if different
        if exchange_position["leverage"] != current_position.leverage:
            self.set_leverage(
                leverage=exchange_position["leverage"],
                symbol=symbol,
                exchange=exchange,
                position_side=current_position.position_side,
            )

        # Sync margin mode if different
        if exchange_position["is_isolated"] != current_position.is_isolated:
            self.set_margin_mode(
                is_isolated=exchange_position["is_isolated"],
                symbol=symbol,
                exchange=exchange,
                position_side=current_position.position_side,
            )

        # Setup position mode
        # self.position_mode = PositionMode.ONEWAY if len(exchange_positions) == 1 else PositionMode.HEDGE
        self.set_position_mode(
            position_mode=self.position_mode,
            symbol=symbol,
            exchange=exchange,
        )

        # Create initial position if size exists
        for exchange_position in exchange_positions:
            if exchange_position["size"]:
                trade = TradeData(
                    order_id="",
                    ori_order_id="",
                    symbol=symbol,
                    code=feed.code,
                    exchange=exchange,
                    asset_type=feed.asset_type,
                    side=Side("SELL" if exchange_position["side"].upper() == "SHORT" else "BUY"),
                    quantity=exchange_position["size"],
                    price=exchange_position["avg_open_price"],
                    traded_at=self.ctx.now,
                    position_side=PositionSide(exchange_position["position_side"].upper()),
                    commission=0.0000001,
                    commission_asset=None,
                    is_open=True,
                    is_maker=False,
                    tag="Initial existing position",
                    msg="",
                )

                self.debug(f"Constructing initial position using trade: {trade}")
                current_position.update_by_tradefeed(trade=trade, available_balance=self.available_balance)

                # Update position with current market data
                tick_data = feed.fetch_tick()
                current_position.update_by_marketdata(
                    last_price=tick_data.last_price, available_balance=self.available_balance
                )

    def add_connection(
        self,
        exchange: str | Exchange,
        asset_type: str | AssetType,
        key: str | None = None,
        secret: str | None = None,
        user_id: str | None = None,
        passphrase: str | None = None,
        is_testnet: bool = False,
        is_demo: bool = False,
        position_mode: PositionMode = PositionMode.ONEWAY,
        on_error_callback: Callable | None = None,
        on_connected_callback: Callable | None = None,
        on_disconnected_callback: Callable | None = None,
        is_dummy: bool = False,
        balance: float | None = None,
        is_overwrite: bool = False,
        username: str | None = None,
        password: str | None = None,
        client_id: str | None = None,
        host: str | None = None,
        port: str | None = None,
        logger: Any = None,
    ) -> tuple[Any, Any, Any]:
        """Add a new exchange connection and return the client connections.

        Args:
            exchange: Exchange name or Exchange enum
            asset_type: Asset type string or AssetType enum
            key: API key for authentication
            secret: API secret for authentication
            user_id: User ID for some exchanges
            passphrase: Passphrase for some exchanges
            is_testnet: Whether to use testnet
            is_demo: Whether to use demo environment
            position_mode: Position mode (ONEWAY or HEDGE)
            on_account_callback: Callback for account events
            on_order_callback: Callback for order events
            on_trade_callback: Callback for trade events
            on_position_callback: Callback for position events
            is_dummy: Whether to use dummy client
            balance: Initial balance for dummy client
            is_overwrite: Whether to overwrite existing connections
            username: Username for RPC connections
            password: Password for RPC connections
            client_id: Client ID for RPC connections
            host: Host for RPC connections
            port: Port for RPC connections
            logger: Logger instance

        Returns:
            Tuple of (private_rest_client, private_ws_client, public_ws_client)
        """
        self.position_mode = position_mode

        exchange_enum = Exchange(exchange.upper()) if isinstance(exchange, str) else exchange
        asset_type_enum = AssetType(asset_type.upper()) if isinstance(asset_type, str) else asset_type
        connection_key = self._get_connection_key(exchange_enum, asset_type_enum)

        # Create all three connection types
        public_ws_client = get_exchange_connection(
            exchange=exchange,
            asset_type=asset_type,
            return_connection_only=True,
            is_data_websocket=True,
            is_testnet=is_testnet,
            is_demo=is_demo,
            logger=logger,
        )
        private_ws_client = get_exchange_connection(
            exchange=exchange,
            asset_type=asset_type,
            return_connection_only=True,
            is_user_websocket=True,
            username=username,
            password=password,
            key=key,
            secret=secret,
            user_id=user_id,
            passphrase=passphrase,
            is_testnet=is_testnet,
            is_demo=is_demo,
            is_dummy=is_dummy,
            balance=balance,
            is_overwrite=is_overwrite,
            on_account_callback=self.on_account_callback,
            on_order_callback=self.on_order_callback,
            on_trade_callback=self.on_trade_callback,
            on_position_callback=self.on_position_callback,
            on_error_callback=on_error_callback,
            on_connected_callback=on_connected_callback,
            on_disconnected_callback=on_disconnected_callback,
            logger=logger,
        )
        if not private_ws_client:
            raise ValueError(
                f"Cannot get private ws connection for {exchange_enum.value} {asset_type_enum.value}{'(Dummy)' if is_dummy else ''}"
            )
        private_rest_client = get_exchange_connection(
            exchange=exchange,
            asset_type=asset_type,
            return_connection_only=True,
            is_dummy=is_dummy,
            username=username,
            password=password,
            balance=balance,
            is_overwrite=is_overwrite,
            key=key,
            secret=secret,
            user_id=user_id,
            passphrase=passphrase,
            is_testnet=is_testnet,
            is_demo=is_demo,
            session=True,
            logger=logger,
        )

        # Store connections with standardized key
        self.public_ws_clients[connection_key] = public_ws_client
        self.private_ws_clients[connection_key] = private_ws_client
        self.private_rest_clients[connection_key] = private_rest_client

        return private_rest_client, private_ws_client, public_ws_client

    def _get_connection_key(self, exchange: Exchange, asset_type: AssetType) -> str:
        """Generate standardized connection key."""
        return f"{exchange.value}|{asset_type.value}"

    def get_connection(
        self, exchange: str | Exchange, asset_type: str | AssetType, is_rest: bool = True, is_public: bool = True
    ) -> Any:
        """Get a connection client for the specified exchange and asset type.

        Args:
            exchange: Exchange name or Exchange enum
            asset_type: Asset type name or AssetType enum
            is_rest: Whether to return REST client (True) or WebSocket client (False)
            is_public: Whether to return public (True) or private (False) client

        Returns:
            The requested client connection or None if not found
        """

        exchange_enum = Exchange(exchange.upper()) if isinstance(exchange, str) else exchange
        asset_type_enum = AssetType(asset_type.upper()) if isinstance(asset_type, str) else asset_type
        connection_key = self._get_connection_key(exchange_enum, asset_type_enum)

        if is_rest:
            return self.private_rest_clients.get(connection_key)
        elif is_public:
            return self.public_ws_clients.get(connection_key)
        else:
            return self.private_ws_clients.get(connection_key)

    def send_order(self, order: Any, **kwargs: Any) -> Any:
        """Send a new order to server.

        Implementation requirements:
        * Create an OrderData from req using OrderRequest.create_order_data
        * Assign a unique (gateway instance scope) id to OrderData.orderid
        * Send request to server
            * If request is sent, OrderData.status should be set to Status.SUBMITTING
            * If request failed to send, OrderData.status should be set to Status.REJECTED
        * Response on_order
        * Return OrderData

        Args:
            order: Order data to send
            **kwargs: Additional order parameters

        Returns:
            OrderData object with updated status
        """
        connection_key = self._get_connection_key(order.exchange, order.asset_type)
        rest_client = self.private_rest_clients[connection_key]
        old_qty = self.ctx.get_position_size(
            symbol=order.symbol, exchange=order.exchange, position_side=order.position_side
        )

        # Determine if this is an opening or closing order
        if order.is_open is None:
            quantity_with_direction = (
                order.quantity if order.side in [Side.BUY, Side.BUYCOVER] else (-1) * order.quantity
            )
            order.is_open = (old_qty * quantity_with_direction) >= 0

        self.sent_orders[order.order_id] = order

        # Build order payload
        payload = self._build_order_payload(order, **kwargs)
        if isinstance(rest_client, DummyClient):
            payload["exchange"] = order.exchange.value
            payload["asset_type"] = order.asset_type.value

        # Send order to exchange
        response = rest_client.send_order(**payload)

        self.debug(
            f"Send order details:\n"
            f"  Exchange: {order.exchange} | Asset Type: {order.asset_type}\n"
            f"  Previous Quantity: {old_qty}\n"
            f"  Payload: {payload}\n"
            f"  Order: {order}\n"
            f"  Response: {response}"
        )

        # Handle failed orders
        if response["error"]:
            order.status = Status.FAILED
            order.updated_at = self.ctx.now if self.ctx.is_live else self.ctx.tick
            order.msg = response["msg"]
            self.on_order_callback(order)

        return order

    def _build_order_payload(self, order: Any, **kwargs: Any) -> dict[str, Any]:
        """Build the order payload for the exchange API."""
        payload = {
            "symbol": order.code,
            "quantity": order.quantity,
            "side": order.side.value,
            "order_id": order.order_id,
            "time_in_force": order.time_in_force.value,
            **kwargs,  # Add additional parameters from kwargs
        }

        # Add optional order parameters
        if order.price:
            payload["price"] = order.price
        if order.stop_price:
            payload["stop_price"] = order.stop_price

        # Add reduce_only flag for futures closing orders
        if (
            order.asset_type
            in [
                AssetType.FUTURE,
                AssetType.INVERSE_FUTURE,
                AssetType.PERPETUAL,
                AssetType.DATED_FUTURE,
                AssetType.INVERSE_PERPETUAL,
                AssetType.INVERSE_DATED_FUTURE,
            ]
            and not order.is_open
        ):
            payload["reduce_only"] = True

        # Add position side handling
        if order.position_side == PositionSide.NET:
            if order.exchange == Exchange.BINGX:
                ### Bingx special handling
                if not order.is_open:  # means close position
                    if payload["side"] == "BUY":
                        position_side = "short"  # buy at short side ==> close short
                    else:
                        position_side = "long"  # sell at long side ==> close long
                else:
                    if payload["side"] == "BUY":
                        position_side = "long"  # buy at long side ==> open long
                    else:
                        position_side = "short"  # sell at short side ==> open short
            else:
                position_side = order.position_side.value
            payload["position_side"] = position_side
        elif order.position_side in (PositionSide.LONG, PositionSide.SHORT):
            payload["position_side"] = order.position_side.value.lower()

        return payload

    def set_leverage(
        self,
        leverage: float,
        symbol: str,
        exchange: Exchange,
        position_side: PositionSide | None = None,
    ) -> bool:
        """Set leverage for a position."""
        super().set_leverage(leverage, symbol, exchange, position_side)

        datafeed = self.ctx.get_datafeed(symbol=symbol, exchange=exchange)
        asset_type_enum = datafeed.asset_type
        exchange_enum = datafeed.exchange
        connection_key = self._get_connection_key(exchange_enum, asset_type_enum)
        rest_client = self.private_rest_clients[connection_key]

        if hasattr(rest_client, "set_leverage"):
            payload = {
                "leverage": leverage,
                "symbol": datafeed.code,
            }

            if "BINGX" in connection_key:
                payload["position_side"] = "short"

            if isinstance(rest_client, DummyClient):
                payload.update({"exchange": datafeed.exchange.value, "asset_type": datafeed.asset_type.value})

            response = rest_client.set_leverage(**payload)

            if response["success"]:
                self.debug(f"Set position leverage res: {response} for {symbol}")
            else:
                self.error(f"Failed to set position leverage - payload:{payload}; {response} for {symbol}")

            return True
        return False

    def set_margin_mode(
        self,
        is_isolated: bool,
        symbol: str,
        exchange: Exchange,
        position_side: PositionSide | None = None,
    ) -> bool:
        """Set margin mode for a position."""
        super().set_margin_mode(is_isolated, symbol, exchange, position_side)

        datafeed = self.ctx.get_datafeed(symbol=symbol, exchange=exchange)
        position = self.ctx.get_position(symbol=symbol, exchange=exchange, force_return=True)
        asset_type_enum = datafeed.asset_type
        exchange_enum = datafeed.exchange
        connection_key = self._get_connection_key(exchange_enum, asset_type_enum)
        rest_client = self.private_rest_clients[connection_key]

        if hasattr(rest_client, "set_margin_mode"):
            payload = {
                "symbol": datafeed.code,
                "is_isolated": is_isolated,
            }

            if "BYBIT" in connection_key:
                payload["leverage"] = position.leverage

            response = rest_client.set_margin_mode(**payload)

            if response["success"]:
                self.debug(f"Set margin mode res: {response} for {symbol}")
            else:
                margin_mode = "ISOLATED" if is_isolated else "CROSS"
                self.error(f"Failed to set margin mode: {margin_mode}; {response} for {symbol}")

            return True
        return False

    def set_position_mode(
        self,
        position_mode: PositionMode,
        symbol: str,
        exchange: Exchange,
    ) -> bool:
        """Set position mode for a symbol."""
        datafeed = self.ctx.get_datafeed(symbol=symbol, exchange=exchange)
        asset_type_enum = datafeed.asset_type
        exchange_enum = datafeed.exchange
        connection_key = self._get_connection_key(exchange_enum, asset_type_enum)
        rest_client = self.private_rest_clients[connection_key]

        if hasattr(rest_client, "set_position_mode"):
            self.debug(f"Setting position mode {position_mode.value} for {symbol}({datafeed.code}) [{connection_key}]")

            payload = {"mode": position_mode.value}
            if "BYBIT" in connection_key:
                payload["symbol"] = datafeed.code

            response = rest_client.set_position_mode(**payload)

            if response["success"]:
                self.debug(f"Set position mode res: {response} for {symbol}")
            else:
                self.error(f"Failed to set position mode - payload:{payload}; {response} for {symbol}")

            return True
        return False

    def cancel_order(
        self,
        order: OrderData | None = None,
        symbol: str | None = None,
        exchange: Exchange | None = None,
        order_id: str | None = None,
    ) -> None:
        """
        Cancel an existing order.

        Implementation tasks:
        * Send request to server
        """
        if order:
            symbol = order.symbol
            exchange = order.exchange
            original_order_id = order.ori_order_id
        else:
            original_order_id = order_id

        datafeed = self.ctx.get_datafeed(symbol=symbol, exchange=exchange)
        asset_type_enum = datafeed.asset_type
        exchange_enum = datafeed.exchange
        connection_key = self._get_connection_key(exchange_enum, asset_type_enum)
        rest_client = self.private_rest_clients[connection_key]

        payload = {"order_id": original_order_id, "symbol": datafeed.code}
        if isinstance(rest_client, DummyClient):
            payload.update({"exchange": exchange_enum.value, "asset_type": asset_type_enum.value})

        response = rest_client.cancel_order(**payload)
        self.debug(
            f"In realtime broker({connection_key}) - Cancel order ==> {order} | "
            f"\ncancel_order payload:{payload}; \nResult: {response}\n"
        )

        if order and response["success"]:
            order.status = Status.CANCELED
            order.updated_at = self.ctx.now if self.ctx.is_live else self.ctx.tick
            order.msg = response["msg"]
            self.on_order_callback(order)

    def connect(self) -> None:
        """Start broker connection.

        Implementation requirements:
        * Connect to server if necessary
        * Log connected if all necessary connections are established
        * Query and respond with corresponding callbacks:
            * contracts: on_contract
            * account asset: on_account
            * account holding: on_position
            * orders of account: on_order
            * trades of account: on_trade
        * If any query fails, write log
        """
        for connection_key, public_ws_client in self.public_ws_clients.items():
            if public_ws_client:
                public_ws_client.connect()

    def finish(self) -> None:
        """Finish broker operations."""
        self.close()

    def close(self) -> None:
        """Close broker connection."""
        # Close all REST clients
        for connection_key, rest_client in self.private_rest_clients.items():
            if rest_client:
                rest_client.stop()

        # Close all public WebSocket clients
        for connection_key, public_ws_client in self.public_ws_clients.items():
            if public_ws_client:
                public_ws_client.stop()

        # Close all private WebSocket clients
        for connection_key, private_ws_client in self.private_ws_clients.items():
            if private_ws_client:
                private_ws_client.stop()

        self.debug("All WebSocket and REST clients are closed in realtime broker.")

    def subscribe(
        self, symbol: str, exchange: str | Exchange | None = None, asset_type: str | AssetType | None = None
    ) -> None:
        """Subscribe to tick data updates.

        Args:
            symbol: Symbol to subscribe to
            exchange: Exchange to subscribe on (optional)
            asset_type: Asset type to subscribe to (optional)
        """
        if exchange and asset_type:
            exchange_enum = Exchange(exchange.upper()) if isinstance(exchange, str) else exchange
            asset_type_enum = AssetType(asset_type.upper()) if isinstance(asset_type, str) else asset_type
            connection_key = self._get_connection_key(exchange_enum, asset_type_enum)

            if not self.public_ws_clients.get(connection_key):
                self.error(
                    f"In realtime broker - no public_ws_client for {connection_key} | "
                    f"All public_ws_clients:{self.public_ws_clients}"
                )
                return

            public_ws_clients = {connection_key: self.public_ws_clients[connection_key]}
        else:
            public_ws_clients = self.public_ws_clients

        for connection_key, public_ws_client in public_ws_clients.items():
            if public_ws_client:
                public_ws_client.subscribe(symbols=symbol, channel="depth", on_tick=self.on_tick_callback)
                public_ws_client.subscribe(symbols=symbol, channel="ticker", on_tick=self.on_tick_callback)

    def query_trades(self, symbol: str, limit: int | None = None) -> None:
        """Query trades."""
        for connection_key, rest_client in self.private_rest_clients.items():
            trades_result = rest_client.query_trades(symbol=symbol)
            if trades_result["success"]:
                for trade in trades_result["data"]:
                    self.debug(f"{connection_key} broker query_trades trade: {trade}")
                    self.on_trade_callback(trade)
            else:
                self.error(f"Something went wrong when query trades - {trades_result} - key:{connection_key}")

    def query_account(self) -> None:
        """Query account balance."""
        for connection_key, rest_client in self.private_rest_clients.items():
            account_result = rest_client.query_account()

            if account_result["success"]:
                for asset, account in account_result["data"].items():
                    self.debug(f"{connection_key} broker query_account account: {account}")
                    self.on_account_callback(account)
            else:
                self.error(f"Something went wrong when query account - {account_result} - key:{connection_key}")

    def query_position(self, symbol: str | None = None) -> None:
        """Query holding positions.

        Args:
            symbol: Specific symbol to query positions for (optional)
        """
        for connection_key, rest_client in self.private_rest_clients.items():
            position_result = (
                rest_client.query_position(symbol=symbol, force_return=True) if symbol else rest_client.query_position()
            )

            if position_result["success"]:
                for position in position_result["data"]:
                    self.debug(f"{connection_key} broker query_position position: {position}")
                    self.on_position_callback(position)
            else:
                self.error(f"Something went wrong when query position - {position_result} - key:{connection_key}")

    def on_tick(self, tick_data: Any) -> None:
        """Handle tick data callback."""
        pass

    def on_bar(self, tick: Any) -> None:
        """Handle bar data callback."""
        pass
