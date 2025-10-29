import os
from abc import abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from dateutil.parser import parse

from .base import Base
from .broker import Broker
from .brokers.backtest_broker import BacktestBroker
from .context import Context
from .datafeed import DataFeed
from .strategy import Strategy
from .utils.event_engine import EVENT_BAR, Event, EventEngine
from .utils.indicator import BarManager
from .utils.logger import LogData, get_logger
from .utils.models import normalize_name
from .utils.objects import (
    AssetType,
    Exchange,
    ExecutionType,
    PositionSide,
    Side,
    TickData,
    TradeData,
)

# from .utils.util_auth import authentication
from .utils.util_dt import find_closest_datetime, tz_manager
from .utils.util_math import prec_round

# from memory_profiler import profile


class Orchestrator:
    """
    Main orchestrator class that coordinates data feeds, strategies, and brokers for trading operations.

    This class manages the execution of trading strategies, handles data feeds, processes trade history,
    and coordinates between different components of the trading system.
    """

    def __init__(
        self,
        feeds: list[DataFeed] | DataFeed,
        StrategyClass: type[Strategy],
        benchmark: DataFeed | None = None,
        logger: Any = None,
        store_path: str = "",
        broker: Broker | None = None,
        statistic: Any = None,
        trade_history: list[dict[str, Any]] | None = None,
        hist_trade_skip_datafeed: bool = False,
        hist_trade_skip_checking: bool = False,
        debug: bool = False,
        prefix: str = "",
        **kwargs: Any,
    ):
        """
        Initialize the Orchestrator.

        Args:
            feeds: DataFeed instance or list of DataFeed instances
            StrategyClass: The Strategy class to be used
            benchmark: Optional benchmark datafeed for comparison
            logger: Custom logger instance (optional)
            store_path: Base path to store output from backtesting
            broker: Broker instance for trade execution
            statistic: Statistics handler instance
            trade_history: List of historical trade data
            hist_trade_skip_datafeed: Skip datafeed validation for historical trades
            hist_trade_skip_checking: Skip validation checks for historical trades
            debug: Enable debug mode
            prefix: Prefix for log files
            **kwargs: Additional keyword arguments passed to strategy
        """
        # authentication(bypass=kwargs.get("bypass", False), _from="orchestrator")

        # Validate and normalize feeds input
        self.feeds = self._validate_and_normalize_feeds(feeds)

        # Set benchmark
        self.benchmark = self._validate_benchmark(benchmark)

        # Validate trade history requirement
        if not self.feeds and not trade_history:
            raise ValueError("Please specify either data feeds or trade history")

        # Initialize core components
        self._debug = debug
        self.event_engine = EventEngine()
        self.ctx: Context | None = None
        self.context: Context | None = None

        # Set trade history processing flags
        self.hist_trade_skip_checking = hist_trade_skip_checking
        self.hist_trade_skip_datafeed = hist_trade_skip_datafeed

        # Process feeds and determine exchange
        self.exchange = self._process_feeds_and_exchange()

        # Validate and set broker
        self.broker = self._validate_broker(broker)
        self.statistic = statistic

        # Initialize strategy
        self.strategy = self._initialize_strategy(StrategyClass, kwargs)
        self.StrategyClass = StrategyClass

        # Set up paths and logging
        self.path = os.path.abspath(store_path) if store_path else store_path
        self.logger = self._setup_logger(logger, prefix)

        # Initialize hook and runner lists
        self._pre_hook_list: list[Base] = []
        self._post_hook_list: list[Base] = []
        self._runner_list: list[Base] = []
        self._full_runner_list: list[Base] = []

        # Process trade days and trade history
        self.tradedays = self._extract_trade_days()
        self.trade_history = self._process_trade_history(trade_history or [])

        # Initialize bar manager
        self.bars_manager = BarManager(on_bar=self.set_bar_data)

    def _validate_and_normalize_feeds(self, feeds: list[DataFeed] | DataFeed) -> dict[str, DataFeed]:
        """Validate and normalize feeds input into a dictionary."""
        if not isinstance(feeds, list | DataFeed):
            raise TypeError("`feeds` must be DataFeed or list of DataFeed")

        if isinstance(feeds, DataFeed):
            feeds = [feeds]

        for item in feeds:
            if not isinstance(item, DataFeed):
                raise TypeError("`data` must be DataFeed")

        return {feed.aio_symbol: feed for feed in feeds}

    def _validate_benchmark(self, benchmark: DataFeed | None) -> DataFeed | None:
        """Validate benchmark datafeed."""
        if benchmark and not isinstance(benchmark, DataFeed):
            raise ValueError("Please specify the correct benchmark datafeed")
        return benchmark

    def _process_feeds_and_exchange(self) -> Exchange | None:
        """Process feeds and determine the exchange."""
        feeds_exchange = {feed.exchange for feed in self.feeds.values()}

        if len(feeds_exchange) > 1:
            if isinstance(self.broker, BacktestBroker):
                return None
            else:
                raise ValueError("Cannot specify datafeed from multiple exchanges for broker.")

        return next(iter(feeds_exchange)) if feeds_exchange else None

    def _validate_broker(self, broker: Broker | None) -> Broker:
        """Validate broker instance."""
        if not broker:
            raise ValueError("Please specify the broker")
        return broker

    def _initialize_strategy(self, StrategyClass: type[Strategy], kwargs: dict[str, Any]) -> Strategy:
        """Initialize strategy with validation."""
        if not (isinstance(StrategyClass, type) and issubclass(StrategyClass, Strategy)):
            raise TypeError("`strategy` must be a Strategy sub-type")

        # Set additional attributes on strategy class
        for key, value in kwargs.items():
            setattr(StrategyClass, key, value)

        return StrategyClass()

    def _setup_logger(self, logger: Any, prefix: str) -> Any:
        """Set up logger instance."""
        if logger:
            return logger

        log_path = os.path.join(self.path, f"{prefix}trade_hist.log") if self.path else ""
        return get_logger(
            log_path=log_path,
            use_default_dt=False,
            default_level="TRACE" if self._debug else "DEBUG",
        )

    def _extract_trade_days(self) -> list[datetime]:
        """Extract and combine all trade days from feeds."""
        all_trade_days = set()

        for feed in self.feeds.values():
            if feed.data:
                all_trade_days.update(feed.data.keys())

        return sorted(all_trade_days)

    def _process_trade_history(
        self, trade_history: list[dict[str, Any]]
    ) -> defaultdict[datetime, list[dict[str, Any]]]:
        """Process and validate trade history data."""
        processed_history: defaultdict[datetime, list[dict[str, Any]]] = defaultdict(list)

        if not trade_history:
            return processed_history

        # Normalize datetime field
        normalized_history = [
            {**trade, "datetime": trade.get("traded_at", trade.get("datetime", ""))} for trade in trade_history
        ]

        # Sort trades by datetime and order_id if available
        sort_key = (
            lambda d: (d["datetime"], d["order_id"]) if "order_id" in trade_history[0] else lambda d: d["datetime"]
        )

        try:
            sorted_history = sorted(normalized_history, key=sort_key)  # type: ignore
        except Exception:
            sorted_history = sorted(normalized_history, key=lambda d: d["datetime"])

        # Process each trade
        for trade in sorted_history:
            self._validate_trade_fields(trade)
            cleaned_trade = self.clean_history_trade_data(trade)
            closest_tick = self._find_trade_tick(cleaned_trade)

            if closest_tick is not None:
                processed_history[closest_tick].append(cleaned_trade)

        return processed_history

    def _validate_trade_fields(self, trade: dict[str, Any]) -> None:
        """Validate required fields in trade data."""
        required_fields = ["symbol", "quantity", "price", "datetime", "side"]
        missing_fields = [field for field in required_fields if field not in trade]

        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

    def _find_trade_tick(self, trade: dict[str, Any]) -> datetime | None:
        """Find the appropriate tick for a trade based on configuration."""
        dt: datetime = trade["datetime"]

        if self.hist_trade_skip_datafeed:
            if dt not in self.tradedays:
                self.tradedays.append(dt)
            return dt

        # Validate trade against datafeed
        symbol = trade["symbol"]
        exchange = trade["exchange"]
        feed = self.feeds.get(f"{symbol}|{exchange.value}")

        if not feed or not feed.data:
            print(f"[ERROR] Trade skipped due to missing datafeed: {trade}")
            return None

        feed_dates = list(feed.data.keys())
        time_buffer = timedelta(days=1, hours=8)

        if dt < feed_dates[0] - time_buffer or dt > feed_dates[-1] + time_buffer:
            print(f"[ERROR] Trade skipped - datetime outside datafeed period: {trade}")
            return None

        dt = find_closest_datetime(datetime_list=self.tradedays, target_datetime=dt)
        return dt

    def process_tick_event(
        self, event: Event, ctx: Context | None = None, bars_manager: BarManager | None = None
    ) -> None:
        """
        Process tick events to construct bar data from tick data.

        Args:
            event: Event containing tick data
            ctx: Context instance (optional, uses self.ctx if not provided)
            bars_manager: BarManager instance (optional, uses self.bars_manager if not provided)
        """
        bars_manager = bars_manager or self.bars_manager
        ctx = ctx or self.ctx
        tick_data: TickData = event.data

        try:
            bars_manager.update_tick(tick_data)
        except Exception as e:
            error_msg = f"Error processing tick event: {e}. tick_data: {tick_data}"
            print(error_msg)
            self.execption(error_msg)

    def set_bar_data(self, bars: dict[str, Any], tick: datetime, ctx: Context | None = None) -> None:
        """
        Set bar data and create bar event.

        Args:
            bars: Dictionary containing bar data
            tick: Datetime of the tick
            ctx: Context instance (optional, uses self.ctx if not provided)
        """
        ctx = ctx or self.ctx
        event = Event(type=EVENT_BAR, data={"tick": tick, "bars": bars}, priority=3)
        if ctx:
            if ctx.is_live:
                ctx.event_engine.put(event)
            else:
                ctx.strategy.process_bar_event(event)

            self.last_process = tick

    @property
    def runners(self) -> list[Base]:
        """Get all runners."""
        return self._full_runner_list

    def clean_history_trade_data(self, trade: dict[str, Any]) -> dict[str, Any]:
        """
        Clean and normalize historical trade data.

        Args:
            trade: Raw trade data dictionary

        Returns:
            Cleaned and normalized trade data
        """
        # Normalize asset type and symbol
        trade["asset_type"] = AssetType(trade.get("asset_type", "spot").upper())
        trade["code"] = trade["symbol"]
        trade["symbol"] = normalize_name(trade["code"], trade["asset_type"].value)

        # Normalize numeric values
        trade["quantity"] = prec_round(float(trade["quantity"]), 6)
        trade["price"] = prec_round(float(trade["price"]), 6)

        # Normalize enums
        trade["side"] = Side(str(trade["side"]).upper())
        trade["position_side"] = (
            PositionSide(str(trade["position_side"]).upper()) if "position_side" in trade else PositionSide.NET
        )

        # Handle position side for spot/stock assets
        if trade["asset_type"] in [AssetType.SPOT, AssetType.STOCK]:
            trade["position_side"] = PositionSide.NET

        # Set exchange
        trade["exchange"] = Exchange(str(trade["exchange"]).upper()) if "exchange" in trade else self.exchange

        # Parse datetime
        trade["datetime"] = parse(trade["datetime"]) if isinstance(trade["datetime"], str) else trade["datetime"]

        # Handle timezone
        try:
            if trade["datetime"].tzinfo is None:
                trade["datetime"] = tz_manager.tz.localize(trade["datetime"])
        except Exception as e:
            error_msg = (
                f"Datetime error in clean_history_trade_data: {e}. "
                f"Trade: {trade}, datetime: {trade['datetime']} ({type(trade['datetime'])})"
            )
            print(error_msg)
            raise ValueError("Datetime processing error")

        # Set message
        trade["msg"] = trade.get("msg", trade.get("order_id", ""))

        return trade

    def execute_history_trade(self, trade: dict[str, Any], is_adhoc: bool = False, ctx: Context | None = None) -> None:
        """
        Execute a historical trade.

        Args:
            trade: Trade data dictionary
            is_adhoc: Whether this is an ad-hoc trade execution
            ctx: Context instance (optional, uses self.ctx if not provided)
        """
        ctx = ctx or self.ctx

        if is_adhoc:
            trade = self.clean_history_trade_data(trade)

        print(f">>> trade: {trade}")

        trade_data = TradeData(
            order_id=trade["order_id"],
            ori_order_id=trade.get("ori_order_id", trade["order_id"]),
            quantity=trade["quantity"],
            side=trade["side"],
            price=trade["price"],
            code=trade["code"],
            symbol=trade["symbol"],
            exchange=trade["exchange"],
            asset_type=trade["asset_type"],
            position_side=trade["position_side"],
            commission=trade["commission"],
            commission_asset=trade["commission_asset"],
            traded_at=trade["traded_at"],
            msg=trade.get("msg", ""),
            tag=trade.get("tag", ""),
            is_open=trade["is_open"],
            execution_type=(ExecutionType.LIQUIDATION if "liq-" in trade["order_id"] else ExecutionType.TRADE),
        )

        self.broker.on_trade_callback(trade_data)

        if hasattr(self.broker, "_update_available"):
            self.broker._update_available(trade_data, is_frozen=True, is_trade=True)

    def add_runner(self, runner: Base) -> None:
        """
        Add a customized runner.

        All runners will be bound to a Context object for sharing data in the backtest.

        Args:
            runner: Instance that must inherit from Base class
        """
        if not isinstance(runner, Base):
            raise TypeError("Runner must inherit from Base class")

        self._runner_list.append(runner)

    def add_hook(self, HookClass: type[Base], hook_type: str = "post") -> None:
        """
        Add a customized hook.

        Args:
            HookClass: Class that must inherit from Base
            hook_type: Hook type determining execution order ("pre" or "post")
        """
        hook = HookClass()

        if not isinstance(hook, Base):
            raise TypeError("Hook must inherit from Base class")

        if hook_type == "post" and hook not in self._post_hook_list:
            self._post_hook_list.append(hook)
        elif hook_type == "pre" and hook not in self._pre_hook_list:
            self._pre_hook_list.append(hook)

    def update_strategytegy(self, StrategyClass: type[Strategy]) -> None:
        """
        Update the strategy class.

        Args:
            StrategyClass: New strategy class to use
        """
        for i, runner in enumerate(self._runner_list):
            if isinstance(runner, Strategy):
                self._runner_list[i] = StrategyClass()
                break

    def reset(self) -> None:
        """Reset the orchestrator to initial state."""
        self.event_engine = EventEngine()
        self._pre_hook_list = []
        self._post_hook_list = []
        self._runner_list = []
        self._full_runner_list = []
        self.strategy = None
        self.broker.reset()
        if self.statistic:
            self.statistic.reset()

    @abstractmethod
    def start(self) -> None:
        """Start the backtesting process."""

    def log(self, log: LogData) -> None:
        """
        Log a message with the appropriate level.

        Args:
            log: LogData instance containing log information
        """
        if log.level == "EXCEPTION":
            self.logger.bind(tick=log.tick).exception(log.content)
        else:
            self.logger.bind(tick=log.tick).log(log.level, log.content)

    def _create_log_data(self, content: str, level: str) -> LogData:
        """Create LogData instance with current tick."""
        return LogData(
            tick=self.ctx.now if self.ctx and self.ctx.is_live else (self.ctx.tick if self.ctx else None),
            content=content,
            level=level,
        )

    def trace(self, content: str) -> None:
        """Log message at trace level."""
        self.log(self._create_log_data(content, "TRACE"))

    def debug(self, content: str) -> None:
        """Log message at debug level."""
        self.log(self._create_log_data(content, "DEBUG"))

    def info(self, content: str) -> None:
        """Log message at info level."""
        self.log(self._create_log_data(content, "INFO"))

    def notice(self, content: str) -> None:
        """Log message at notice level."""
        self.log(self._create_log_data(content, "NOTICE"))

    def error(self, content: str) -> None:
        """Log message at error level."""
        self.log(self._create_log_data(content, "ERROR"))

    def warning(self, content: str) -> None:
        """Log message at warning level."""
        self.log(self._create_log_data(content, "WARNING"))

    def critical(self, content: str) -> None:
        """Log message at critical level."""
        self.log(self._create_log_data(content, "CRITICAL"))

    def execption(self, content: str) -> None:
        """Log message at exception level."""
        self.log(self._create_log_data(content, "EXCEPTION"))
