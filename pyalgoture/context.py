from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from .datafeed import DataFeed
from .utils.event_engine import EventEngine
from .utils.objects import (
    AccountData,
    AssetType,
    BarData,
    Exchange,
    OrderData,
    PositionData,
    PositionMode,
    PositionSide,
    TickData,
    TradeData,
)
from .utils.util_dt import tz_manager

if TYPE_CHECKING:
    from .broker import Broker
    from .statistic import Statistic
    from .strategy import Strategy


class Context:
    """The Context class which stores all the shared data for trading operations."""

    def __init__(
        self,
        event_engine: EventEngine,
        feeds: dict[str, DataFeed],
        broker: "Broker",
        strategy: "Strategy",
        statistic: "Statistic",
        logger: Any,
        benchmark: DataFeed | None = None,
        do_print: bool = True,
        path: str | None = None,
        is_live: bool = False,
    ) -> None:
        """Initialize the Context with all necessary components.

        Args:
            event_engine: The event engine for handling events
            feeds: Dictionary of data feeds keyed by symbol
            broker: The broker instance for trading operations
            strategy: The trading strategy instance
            statistic: The statistics tracking instance
            logger: The logging instance
            benchmark: Optional benchmark data feed
            do_print: Whether to enable printing
            path: Optional file path
            is_live: Whether this is a live trading context
        """
        self.event_engine: EventEngine = event_engine
        self.feeds: dict[str, DataFeed] = feeds
        self.tick: datetime | None = None

        self.broker: Broker = broker
        self.strategy: Strategy = strategy
        self.statistic: Statistic = statistic

        self.benchmark: DataFeed | None = benchmark
        self.logger: Any = logger

        self.path: str | None = path

        self.bar_data: dict[str, BarData] = {}
        self.tick_data: dict[str, TickData] = {}
        self.history: dict[str, pd.DataFrame] = defaultdict(pd.DataFrame)
        self.latest_price: dict[str, float] = {}

        self.do_print = do_print
        self.is_live = is_live

        # Create positions for all feeds
        for feed in self.feeds.values():
            self._create_position(feed=feed)

    def get_logger(self) -> Any:
        """Get the logger instance."""
        return self.logger

    def _create_position(self, feed: DataFeed) -> None:
        """Create positions for all position sides for a given feed.

        Args:
            feed: The data feed to create positions for
        """
        position_sides = [PositionSide.NET, PositionSide.LONG, PositionSide.SHORT]

        for position_side in position_sides:
            position = PositionData(
                code=feed.code,
                symbol=feed.symbol,
                exchange=feed.exchange,
                asset_type=feed.asset_type,
                size=0.0,
                position_side=position_side,
                created_at=self.now,
                leverage=1,
                is_isolated=None,
                margin_mode=None,
                is_contract=feed.is_contract,
                consider_price=feed.consider_price,
                is_inverse=feed.is_inverse,
                commission_rate=feed.commission_rate,
                maker_commission_rate=feed.maker_commission_rate,
                taker_commission_rate=feed.taker_commission_rate,
            )

            # Add optional methods if they exist on the feed
            if hasattr(feed, "cal_commission"):
                position.cal_commission = feed.cal_commission
            if hasattr(feed, "cal_interest"):
                position.cal_interest = feed.cal_interest

            self.broker.add_position(position)

    def __getattr__(self, key: str) -> Any:
        """Enable attribute access through getitem."""
        return self[key]

    def __getitem__(self, attr: str) -> Any:
        """Enable dictionary-style access to attributes."""
        return super().__getattribute__(attr)

    def __setitem__(self, key: str, value: Any) -> None:
        """Enable dictionary-style setting of attributes."""
        self.__dict__[key] = value

    @property
    def now(self) -> datetime:
        """Get the current datetime with timezone."""
        return datetime.now(tz_manager.tz).replace(microsecond=0)

    def set_leverage(
        self,
        leverage: float,
        symbol: str,
        exchange: Exchange,
        position_side: PositionSide | None = None,
    ) -> Any:
        """Set leverage for a position.

        Args:
            leverage: The leverage value to set
            symbol: The trading symbol
            exchange: The exchange
            position_side: Optional position side

        Returns:
            Result from broker's set_leverage method
        """
        return self.broker.set_leverage(
            leverage=leverage,
            symbol=symbol,
            exchange=exchange,
            position_side=position_side,
        )

    def set_margin_mode(
        self,
        is_isolated: bool,
        symbol: str,
        exchange: Exchange,
        position_side: PositionSide | None = None,
    ) -> Any:
        """Set margin mode for a position.

        Args:
            is_isolated: Whether to use isolated margin
            symbol: The trading symbol
            exchange: The exchange
            position_side: Optional position side

        Returns:
            Result from broker's set_margin_mode method
        """
        return self.broker.set_margin_mode(
            is_isolated=is_isolated,
            symbol=symbol,
            exchange=exchange,
            position_side=position_side,
        )

    def set_position_mode(
        self,
        position_mode: PositionMode,
        symbol: str,
        exchange: Exchange,
    ) -> bool:
        """Set position mode for a symbol.

        Args:
            position_mode: The position mode to set
            symbol: The trading symbol
            exchange: The exchange

        Returns:
            True if successful, False otherwise
        """
        res: bool = False
        if hasattr(self.broker, "set_position_mode"):
            res = self.broker.set_position_mode(
                position_mode=position_mode,
                symbol=symbol,
                exchange=exchange,
            )
        else:
            self.warning("Only RealtimeBroker supports set_position_mode.")
        return res

    def _get_aio_symbol(self, symbol: str, exchange: Exchange) -> str:
        """Generate AIO symbol string from symbol and exchange.

        Args:
            symbol: The trading symbol
            exchange: The exchange

        Returns:
            Formatted AIO symbol string
        """
        if isinstance(exchange, str):
            exchange = Exchange(exchange.upper())
        return f"{symbol}|{exchange.value}"

    def _get_aio_position_id(
        self,
        symbol: str,
        exchange: Exchange,
        position_side: PositionSide = PositionSide.NET,
    ) -> str:
        """Generate AIO position ID from symbol, exchange, and position side.

        Args:
            symbol: The trading symbol
            exchange: The exchange
            position_side: The position side

        Returns:
            Formatted AIO position ID string
        """
        if isinstance(exchange, str):
            exchange = Exchange(exchange.upper())
        if isinstance(position_side, str):
            position_side = PositionSide(position_side.upper())

        return f"{symbol}|{exchange.value}|{position_side.value}"

    def _parse_aio_symbol(self, aio_symbol: str) -> tuple[str, Exchange]:
        """Parse AIO symbol string into symbol and exchange.

        Args:
            aio_symbol: The AIO symbol string to parse

        Returns:
            Tuple of symbol and exchange
        """
        symbol, exchange = aio_symbol.split("|")
        return symbol, Exchange(exchange)

    def _parse_aio_position_id(self, aio_position_id: str) -> tuple[str, Exchange, PositionSide]:
        """Parse AIO position ID into symbol, exchange, and position side.

        Args:
            aio_position_id: The AIO position ID string to parse

        Returns:
            Tuple of symbol, exchange, and position side
        """
        symbol, exchange, position_side = aio_position_id.split("|")
        return symbol, Exchange(exchange), PositionSide(position_side)

    @property
    def default_account(self) -> AccountData:
        """Get the default cash account.

        Returns:
            The default cash account
        """
        return self.broker.default_account

    @property
    def symbols(self) -> list[tuple[str, Exchange]]:
        """Get list of all symbols and their corresponding exchanges.

        Returns:
            List of tuples containing symbol and exchange pairs
        """
        return [(feed.symbol, feed.exchange) for feed in self.get_feeds()]

    @property
    def holding_value(self) -> float:
        """Calculate the entire portfolio value.

        Returns:
            The total portfolio value
        """
        return sum(float(position.market_value) for position in self.get_positions())

    @property
    def holding_number(self) -> int:
        """Count the number of holdings with non-zero size.

        Returns:
            The number of holdings
        """
        return sum(1 for pos in self.get_positions() if pos.get("size", 0))

    @property
    def assets_value(self) -> float:
        """Calculate the entire asset value (portfolio value + available balance).

        Returns:
            The total asset value
        """
        balance: float = self.default_account.balance
        holding: float = self.holding_value
        return balance + holding

    @property
    def available_balance(self) -> float:
        """Get the available balance from the default account.

        Returns:
            The available balance
        """
        balance: float = self.default_account.balance
        return balance

    def get_asset_type(self, symbol: str, exchange: Exchange) -> AssetType | None:
        """Get the asset type of the selected symbol.

        Args:
            symbol: The instrument name
            exchange: The exchange

        Returns:
            The asset type of the requested position, or None if not found
        """
        datafeed = self.get_datafeed(symbol=symbol, exchange=exchange)
        return datafeed.asset_type if datafeed else None

    def get_exchange(self, symbol: str, exchange: Exchange) -> Exchange | None:
        """Get the exchange of the selected symbol.

        Args:
            symbol: The instrument name
            exchange: The exchange

        Returns:
            The exchange of the requested position, or None if not found
        """
        datafeed = self.get_datafeed(symbol=symbol, exchange=exchange)
        return datafeed.exchange if datafeed else None

    def get_commission_rate(self, symbol: str, exchange: Exchange) -> float:
        """Get the commission rate for a symbol.

        Args:
            symbol: The instrument name
            exchange: The exchange

        Returns:
            The commission rate, or 0.0 if not found or is contract
        """
        position = self.get_position(symbol=symbol, exchange=exchange, force_return=True)
        if position is None or position.is_contract:
            return 0.0

        if position.commission_rate:
            return position.commission_rate
        elif position.maker_commission_rate and position.taker_commission_rate:
            return max(position.maker_commission_rate, position.taker_commission_rate)
        else:
            return 0.0

    def get_market_value(
        self,
        symbol: str,
        exchange: Exchange,
        position_side: PositionSide = PositionSide.NET,
    ) -> float:
        """Get the market value of the selected position.

        Args:
            symbol: The instrument name
            exchange: The exchange
            position_side: The position side

        Returns:
            The market value of the requested position
        """
        position = self.get_position(
            symbol=symbol,
            exchange=exchange,
            position_side=position_side,
            force_return=True,
        )
        return position.market_value if position else 0.0

    def get_datafeed(self, symbol: str, exchange: Exchange) -> DataFeed | None:
        """Get the data feed for a symbol.

        Args:
            symbol: The instrument name
            exchange: The exchange

        Returns:
            The datafeed, or None if not found
        """
        aio_symbol = self._get_aio_symbol(symbol=symbol, exchange=exchange)
        return self.feeds.get(aio_symbol)

    def get_tick_data(self, symbol: str, exchange: Exchange) -> TickData | None:
        """Get the tick data for a symbol.

        Args:
            symbol: The instrument name
            exchange: The exchange

        Returns:
            The tick data as a pandas Series, or None if not found
        """
        aio_symbol = self._get_aio_symbol(symbol=symbol, exchange=exchange)
        return self.tick_data.get(aio_symbol)

    def get_bar_data(self, symbol: str, exchange: Exchange) -> BarData | None:
        """Get the bar data for a symbol.

        Args:
            symbol: The instrument name
            exchange: The exchange

        Returns:
            The bar data as a BarData, or None if not found
        """
        aio_symbol = self._get_aio_symbol(symbol=symbol, exchange=exchange)
        return self.bar_data.get(aio_symbol)

    def get_latest_price(self, symbol: str, exchange: Exchange) -> float:
        """Get the latest price for a symbol.

        Args:
            symbol: The instrument name
            exchange: The exchange

        Returns:
            The latest price, or None if not found
        """
        aio_symbol = self._get_aio_symbol(symbol=symbol, exchange=exchange)
        return self.latest_price.get(aio_symbol, 0.0)

    def get_history(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame | None:
        """Get the historical bar data for a symbol.

        Args:
            symbol: The instrument name
            exchange: The exchange
            start: Optional start time for filtering
            end: Optional end time for filtering

        Returns:
            The historical bar data as a DataFrame, or None if not found
        """
        aio_symbol = self._get_aio_symbol(symbol=symbol, exchange=exchange)
        history = self.history.get(aio_symbol)

        if history is not None and not history.empty:
            try:
                # Apply datetime filtering using boolean indexing
                if start and end:
                    return history[(history.index >= start) & (history.index <= end)]
                elif start:
                    return history[history.index >= start]
                elif end:
                    return history[history.index <= end]
            except (KeyError, TypeError, AttributeError):
                # Return the full history if filtering fails
                pass
        return history

    def get_feeds(self) -> list[DataFeed]:
        """Get all data feeds.

        Returns:
            List of all DataFeed objects
        """
        return list(self.feeds.values())

    def get_bar_datas(self) -> list[Any]:
        """Get all bar data.

        Returns:
            List of all bar data
        """
        return list(self.bar_data.values())

    def get_historys(self) -> list[Any]:
        """Get all historical data.

        Returns:
            List of all historical data
        """
        return list(self.history.values())

    def add_datafeed(self, feed: DataFeed) -> None:
        """Add a data feed to the context.

        Args:
            feed: The DataFeed to add
        """
        self.feeds[feed.aio_symbol] = feed

    # ========================== BROKER DELEGATION METHODS =======================================

    def get_positions(self) -> list[PositionData]:
        """Get the list of all positions.

        Returns:
            List of all positions
        """
        return self.broker.get_positions()

    def get_open_positions(self) -> list[PositionData]:
        """Get the list of all open positions.

        Returns:
            List of all open positions
        """
        return self.broker.get_open_positions()

    def get_accounts(self) -> list[AccountData]:
        """Get the list of all accounts.

        Returns:
            List of all accounts
        """
        return self.broker.get_accounts()

    def get_trades(self) -> list[TradeData]:
        """Get the list of all trades.

        Returns:
            List of all trades
        """
        return self.broker.get_trades()

    def get_orders(self) -> list[OrderData]:
        """Get the list of all orders.

        Returns:
            List of all orders
        """
        return self.broker.get_orders()

    def get_open_orders(self) -> list[OrderData]:
        """Get the list of all open orders.

        Returns:
            List of all open orders
        """
        return self.broker.get_open_orders()

    def get_open_order(self, order_id: str) -> OrderData:
        """Get the selected open order.

        Args:
            order_id: The order ID

        Returns:
            The requested order
        """
        return self.broker.get_open_order(order_id=order_id)

    def get_order(self, order_id: str | None = None, seq: int | None = None) -> OrderData:
        """Get the selected order.

        Args:
            order_id: Optional order ID
            seq: Optional sequence number (e.g., -1 for latest order)

        Returns:
            The requested order
        """
        return self.broker.get_order(order_id=order_id, seq=seq)

    def get_trade(self, order_id: str | None = None, seq: int | None = None) -> TradeData:
        """Get the selected trade.

        Args:
            order_id: Optional order ID
            seq: Optional sequence number (e.g., -1 for latest trade)

        Returns:
            The requested trade
        """
        return self.broker.get_trade(order_id=order_id, seq=seq)

    def get_position_size(
        self,
        symbol: str,
        exchange: Exchange,
        position_side: PositionSide = PositionSide.NET,
    ) -> float:
        """Get the selected position size.

        Args:
            symbol: The instrument name
            exchange: The exchange
            position_side: The position side

        Returns:
            The requested position size
        """
        return self.broker.get_position_size(
            symbol=symbol,
            exchange=exchange,
            position_side=position_side,
        )

    def get_position(
        self,
        symbol: str,
        exchange: Exchange,
        position_side: PositionSide = PositionSide.NET,
        force_return: bool = False,
    ) -> PositionData | None:
        """Get the selected position.

        Args:
            symbol: The instrument name
            exchange: The exchange
            position_side: The position side
            force_return: Whether to force return an object

        Returns:
            The requested position, or None if not found
        """
        position = self.broker.get_position(
            symbol=symbol,
            exchange=exchange,
            position_side=position_side,
            force_return=force_return,
        )
        if force_return and not position:
            print(f"[ERROR] Position({symbol}|{exchange}|{position_side}) cannot be found.")
            return None

        return position

    def get_account(
        self,
        asset: str,
        exchange: Exchange,
        force_return: bool = False,
    ) -> AccountData:
        """Get the selected account.

        Args:
            asset: The asset name
            exchange: The exchange
            force_return: Whether to force return an object

        Returns:
            The requested account
        """
        return self.broker.get_account(
            asset=asset,
            exchange=exchange,
            force_return=force_return,
        )

    def debug(self, content: str) -> None:
        """Log content at debug level.

        Args:
            content: The content to log
        """
        self.broker.debug(content)

    def info(self, content: str) -> None:
        """Log content at info level.

        Args:
            content: The content to log
        """
        self.broker.info(content)

    def error(self, content: str) -> None:
        """Log content at error level.

        Args:
            content: The content to log
        """
        self.broker.error(content)

    def warning(self, content: str) -> None:
        """Log content at warning level.

        Args:
            content: The content to log
        """
        self.broker.warning(content)

    def critical(self, content: str) -> None:
        """Log content at critical level.

        Args:
            content: The content to log
        """
        self.broker.critical(content)
