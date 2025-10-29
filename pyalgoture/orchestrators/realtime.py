import os
import signal
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta
from functools import partial
from itertools import chain
from threading import Thread
from time import sleep
from typing import Any

from ..broker import Broker
from ..brokers.backtest_broker import BacktestBroker
from ..context import Context
from ..datafeed import DataFeed
from ..orchestra import Orchestrator
from ..statistic import Statistic
from ..strategy import Strategy
from ..utils.event_engine import EVENT_TICK, EVENT_TIMER, Event
from ..utils.models import normalize_name
from ..utils.objects import AssetType, Exchange, State
from ..utils.simple_mq import SignalMQ
from ..utils.util_dt import parse2datetime, tz_manager
from ..utils.util_schedule import Scheduler

# def term_handler(signum, frame):
#     # Raise KeyboardInterrupt - so we can handle it in the same way as Ctrl-C
#     raise KeyboardInterrupt()


# signal.signal(signal.SIGTERM, term_handler)
# # signal.signal(signal.SIGINT, term_handler)


class RealTime(Orchestrator):
    """Real-time trading orchestrator that manages live trading operations.

    This class handles real-time data feeds, strategy execution, and broker interactions
    for live trading scenarios. It supports multiple modes: websocket, tick-based, and bar-based.
    """

    startup_time: datetime | None = None
    state: State = State.NEW
    last_process: datetime | None = None

    def __init__(
        self,
        feeds: list[DataFeed] | DataFeed,
        StrategyClass: type[Strategy],
        broker: Broker,
        statistic: Statistic | None = None,
        logger: Any = None,
        trade_history: list[Any] = [],
        store_path: str = "",
        on_trade_callback_custom: Callable[[Any, Any], None] | None = None,
        benchmark: DataFeed | None = None,
        prefix: str = "",
        hist_trade_skip_datafeed: bool = False,
        hist_trade_skip_checking: bool = False,  # skip checking historical trade record, to prevent insufficient balance/position issue.
        mq_id: str = "",  # message queue identifier
        debug: bool = False,
        do_print: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the RealTime orchestrator.

        Args:
            feeds: DataFeed instances or list of DataFeed instances
            StrategyClass: Strategy class to instantiate
            broker: Broker instance for trade execution
            statistic: Optional statistic tracker
            logger: Optional logger instance
            trade_history: Historical trade records
            store_path: Path for storing data
            on_trade_callback_custom: Custom trade callback function
            benchmark: Optional benchmark datafeed
            prefix: Prefix for file names
            hist_trade_skip_datafeed: Skip datafeed for historical trades
            hist_trade_skip_checking: Skip checking historical trade records
            mq_id: Message queue identifier
            debug: Enable debug mode
            do_print: Enable print statements
            **kwargs: Additional keyword arguments
        """
        self._setup_custom_trade_callback(on_trade_callback_custom)

        super().__init__(
            feeds=feeds,
            StrategyClass=StrategyClass,
            logger=logger,
            store_path=store_path,
            broker=broker,
            statistic=statistic,
            trade_history=trade_history,
            prefix=prefix,
            hist_trade_skip_datafeed=hist_trade_skip_datafeed,
            hist_trade_skip_checking=hist_trade_skip_checking,
            debug=debug,
            benchmark=benchmark,
            **kwargs,
        )
        self.do_print = do_print
        self.mq_id = mq_id

        self.ea_symbol_map: defaultdict[str, set[str]] = defaultdict(set)  # exchange|asset_type: code set
        self.ea_connection_map: defaultdict[str, DataFeed] = defaultdict()  # exchange|asset_type: datafeed

        for aio_symbol, feed in self.feeds.items():
            self.ea_symbol_map[f"{feed.exchange}|{feed.asset_type}"].add(feed.code)
            self.ea_connection_map[f"{feed.exchange}|{feed.asset_type}"] = feed

    def startup(self, rpc_config: dict[str, Any]) -> None:
        """Start the RPC manager with the given configuration.

        Args:
            rpc_config: RPC configuration dictionary
        """
        from ..rpc.rpc_manager import RPCManager

        self.rpc_manager: RPCManager = RPCManager(self, rpc_config)
        self.rpc_manager.start()

    def start(
        self,
        mode: str = "bar",
        is_console: bool = True,
        keep_alive: bool = True,
        tick_interval: int = 5,
    ) -> None:
        """Start the real-time trading system.

        Args:
            mode: Trading mode ("ws", "tick", or "bar")
            is_console: Enable console interaction
            keep_alive: Keep the system running
            tick_interval: Interval for tick processing in seconds

        Raises:
            ValueError: If BacktestBroker is used with realtime bar mode
            AssertionError: If mode is not one of the supported values
        """
        if mode not in {"ws", "tick", "bar"}:
            raise ValueError(f"Mode must be one of 'ws', 'tick', 'bar', got: {mode}")

        self.mode = mode
        self.event_engine.start()

        if self.mode == "bar" and isinstance(self.broker, BacktestBroker):
            raise ValueError("BacktestBroker is not compatible with Realtime `bar` mode")

        self._setup_runners()
        if self.mq_id:
            self._setup_message_queue()

        self._initialize_context(mode)

        ### NOTE: pre-execution E.g. include early start date backtesting & trade history execute
        if self.trade_history and self.tradedays:
            self._execute_historical_trades()

        if self.mode == "bar":
            self._setup_bar_mode()
        elif self.mode == "tick":
            self._setup_tick_mode(tick_interval)
        elif self.mode == "ws":
            self._setup_websocket_mode()

        if keep_alive:
            self._run_main_loop(is_console)

    def _setup_custom_trade_callback(self, custom_trade_callback: Callable[[Any, Any], None]) -> None:
        """Setup custom trade callback."""
        self.on_trade_callback_custom = custom_trade_callback

    def _setup_runners(self) -> None:
        """Setup all runners (broker, strategy, statistics)."""
        self.add_runner(self.broker)
        self.add_runner(self.strategy)
        if self.statistic:
            self.add_runner(self.statistic)

    def _setup_message_queue(self) -> None:
        """Setup message queue if specified."""
        self.notice(f"Building connection with MQ(mq_id:{self.mq_id})...")
        self.mq = SignalMQ(name=self.mq_id)
        execute_signal = partial(self.execute_history_trade, is_adhoc=True)
        self.mq.subscribe(callback=execute_signal)

    def _initialize_context(self, mode: str) -> None:
        """Initialize the trading context."""
        self.ctx = self.context = Context(
            event_engine=self.event_engine,
            feeds=self.feeds,
            broker=self.broker,
            strategy=self.strategy,
            statistic=self.statistic,
            benchmark=self.benchmark,
            do_print=self.do_print,
            path=self.path,
            logger=self.logger,
            is_live=True,
        )

        if self.on_trade_callback_custom:
            self.ctx.on_trade_callback_custom = self.on_trade_callback_custom

        self.ctx.tick = self.ctx.now
        self.ctx.trade_history = self.trade_history

        # Setup runners and initialize them
        self._full_runner_list = list(chain(self._pre_hook_list, self._runner_list, self._post_hook_list))
        self.ctx.runners = self._full_runner_list

        for runner in self._full_runner_list:
            runner.ctx = runner.context = self.ctx
            runner.logger = self.logger

        self.strategy.register_event()

        for runner in self._full_runner_list:
            runner.initialize()

    def _execute_historical_trades(self) -> None:
        """Execute historical trades if available."""
        self.ctx.tick = self.tradedays[0] if self.tradedays else None

        for tick in self.tradedays:
            if tick in self.ctx.trade_history:
                for trade in self.ctx.trade_history[tick]:
                    self.execute_history_trade(trade)

    def _setup_bar_mode(self) -> None:
        """Setup bar mode with scheduler for periodic data fetching."""

        def _get_bar() -> None:
            tick = self.ctx.now.replace(second=0, microsecond=0)
            bar_data = {}

            for symbol, datafeed in self.ctx.feeds.items():
                if datafeed.aio_symbol in bar_data:
                    continue

                if hasattr(datafeed, "fetch_spot"):
                    new_bar = datafeed.fetch_spot(interval="1m", start=tick - timedelta(minutes=1), tz=True)
                    if not new_bar:
                        self.ctx.broker.warning(f"symbol: {symbol} fetch realtime data failed.")
                        continue

                    datafeed.update_data(new_data=new_bar)
                    new_bar = next(iter(new_bar.values()))
                    bar_data[symbol] = new_bar

            self.set_bar_data(bars=bar_data, tick=tick)

        self.scheduler = Scheduler()
        self.scheduler.every().minute.at(":00").do(_get_bar)

        def process_timer_event(event: Event) -> None:
            self.scheduler.run_pending()

        self.ctx.event_engine.register(EVENT_TIMER, process_timer_event)

    def _setup_tick_mode(self, tick_interval: int) -> None:
        """Setup tick mode with timer-based data fetching."""

        def process_timer_event(event: Event) -> None:
            current_second = datetime.now().second
            if current_second not in {0, 59} and current_second % tick_interval != 0:
                return

            for symbol, datafeed in self.ctx.feeds.items():
                if hasattr(datafeed, "fetch_tick"):
                    tick_data = datafeed.fetch_tick(tz=True)
                    self.broker.on_tick_callback(tick_data)

        self.ctx.event_engine.register(EVENT_TICK, self.process_tick_event)
        self.ctx.event_engine.register(EVENT_TIMER, process_timer_event)

    def _setup_websocket_mode(self) -> None:
        """Setup websocket mode for real-time data streaming."""
        self.ctx.event_engine.register(EVENT_TICK, self.process_tick_event)

        for ea, feed in self.ea_connection_map.items():
            feed.connect()
            feed.subscribe(
                symbols=list(self.ea_symbol_map[ea]),
                callback=self.broker.on_tick_callback,
            )

    def _run_main_loop(self, is_console: bool) -> None:
        """Run the main event loop with console interaction."""

        def _toggle_ws_debug(check_private: bool = True) -> None:
            """Toggle WebSocket debug mode."""
            ws_clients = self.broker.private_ws_clients if check_private else self.broker.public_ws_clients
            if ws_clients:
                ws_client = None
                for k, ws_client in ws_clients.items():
                    self.notice(f"{k}'s websocket connection status: {ws_client.is_connected()}")
                    if hasattr(ws_client, "debug"):
                        ws_client.debug = not ws_client.debug

                debug_status = ws_client and hasattr(ws_client, "debug") and ws_client.debug
                self.notice(f"WS Debug mode status: {'ON' if debug_status else 'OFF'}")
            else:
                self.error(
                    f"Failed, Broker {'Private' if check_private else 'Public '} WebSocket client is not specified."
                )

        def _toggle_strategy_debug() -> None:
            """Toggle strategy debug mode."""
            self.logger.filter.level = "DEBUG" if self.logger.filter.level == "TRACE" else "TRACE"
            self.notice(f"Strategy Debug mode status: {'ON' if self.logger.filter.level == 'TRACE' else 'OFF'}")

        def _pause_resume() -> None:
            """Pause/Resume operation."""
            self.notice("Pause / Resume")

        def _terminate() -> str:
            """Terminate the system."""
            self.notice("Terminating...")
            return "terminate"

        def _print_stats() -> None:
            """Print current statistics."""
            if hasattr(self.ctx.statistic, "data"):
                self.notice(f"{self.ctx.statistic.stats_result}")

        def _update_stats() -> None:
            """Update statistics."""
            self.ctx.statistic.stats(
                interval=self.benchmark.interval if self.benchmark else "1h",
                benchmark=self.benchmark,
            )
            self.notice("..completed")

        def _generate_contest_output() -> None:
            """Generate contest output in a separate thread."""
            thread = Thread(
                target=self.ctx.statistic.contest_output,
                kwargs={
                    "interval": self.benchmark.interval if self.benchmark else "1h",
                    "benchmark": self.benchmark,
                    "path": self.path,
                    "prefix": "live_",
                    "is_plot": True,
                    "is_live": True,
                    "open_browser": False,
                    "extra_ohlc": False,
                    "only_ohlc": False,
                },
                daemon=True,
            )
            thread.start()

        def _print_bars() -> None:
            """Print current bar data."""
            self.debug(f"\n{self.ctx.get_bar_datas()}\n")

        def _toggle_print_ticks() -> None:
            """Toggle tick data printing."""
            self.ctx.broker._print_tick_data = not self.ctx.broker._print_tick_data
            status = "ON" if self.ctx.broker._print_tick_data else "OFF"
            self.notice(f"print tick data on_callback status: {status}")

        def _toggle_print_user_trade() -> None:
            """Toggle user trade data printing."""
            self.ctx.broker._print_user_trade_data = not self.ctx.broker._print_user_trade_data
            status = "ON" if self.ctx.broker._print_user_trade_data else "OFF"
            self.notice(f"print user trade data on_callback status: {status}")

        def _print_positions() -> None:
            """Print current positions."""
            for pos in self.ctx.get_positions():
                if pos.size or pos.avg_open_price or pos.total_pnl:
                    self.notice(f">>>> {pos}")

        def _print_position_pnl() -> None:
            """Print position PnL."""
            for pos in self.ctx.get_positions():
                if pos.size or pos.avg_open_price or pos.total_pnl:
                    self.notice(f">>>> {pos.get_last_trade_pnl()}")

        def _print_accounts() -> None:
            """Print account information."""
            for acc in self.ctx.get_accounts():
                self.notice(f">>>> {acc}")

        def _export_datafeeds() -> None:
            """Export all datafeeds to CSV files."""
            for aio_symbol, hist in self.feeds.items():
                hist_data = hist.data
                hist_data.to_csv(f"{aio_symbol}.csv")

        def _print_event_queue_info() -> None:
            """Print event queue information."""
            ee = self.event_engine
            self.notice(f"Event Engine Info - empty:{ee.empty}; size:{ee.size}; full:{ee.full}; debug:{ee.debug}")

        def _toggle_event_engine_debug() -> None:
            """Toggle event engine debug mode."""
            self.event_engine.debug = not self.event_engine.debug
            status = "ON" if self.event_engine.debug else "OFF"
            self.notice(f"Event Engine Debug mode status: {status}")

        def _one_click_exit() -> None:
            """Close all positions with one click."""
            for pos in self.ctx.get_positions():
                if pos.size or pos.avg_open_price or pos.total_pnl:
                    order = self.ctx.strategy.close(
                        symbol=pos.symbol,
                        exchange=pos.exchange,
                        position_side=pos.position_side,
                        msg="One click exit",
                    )
                    self.notice(f"[One click exit] pos: {pos} is closed. \n\t\tRef Order: {order}")

        def _one_click_exit_and_terminate() -> str:
            """Close all positions and terminate."""
            self._one_click_exit()
            return _terminate()

        def _print_trades() -> None:
            """Print recent trades."""
            trades = self.ctx.get_trades()
            recent_trades = trades[:20]

            for trade in recent_trades:
                self.notice(f">>>> {trade}")

            trade_count = len(trades)
            recent_count = len(recent_trades)
            self.notice(f"Recent {recent_count} trades are printed above. Total trades: {trade_count}")

        def _dump_data() -> None:
            """Dump data to file."""
            self.dump(path=os.path.join(self.path, "bt.aio"))

        try:
            while True:
                sleep(1)
                if is_console:
                    menu_options = {
                        "0": ("Enable/Disable WS Debug Mode", _toggle_ws_debug),
                        "1": ("Enable/Disable Strategy Debug Mode", _toggle_strategy_debug),
                        "2": ("Pause/Resume", _pause_resume),
                        "3": ("Terminate", _terminate),
                        "4": ("Print stats", _print_stats),
                        "5": ("Update stats", _update_stats),
                        "6": ("Generate contest output", _generate_contest_output),
                        "7": ("Print bars", _print_bars),
                        "8": ("Enable/Disable print ticks", _toggle_print_ticks),
                        "9": ("Enable/Disable print user trade data", _toggle_print_user_trade),
                        "10": ("Print position", _print_positions),
                        "11": ("Print position pnl", _print_position_pnl),
                        "12": ("Print account", _print_accounts),
                        "13": ("Export all datafeeds", _export_datafeeds),
                        "14": ("Event queue info", _print_event_queue_info),
                        "15": ("Enable/Disable Event Engine Debug Mode", _toggle_event_engine_debug),
                        "16": ("One click exit", _one_click_exit),
                        "17": ("One click exit and terminate", _one_click_exit_and_terminate),
                        "18": ("Print trades", _print_trades),
                        "19": ("Dump", _dump_data),
                    }

                    print("Main Choice: Choose 1 of 16 choices")
                    for key, (description, _) in menu_options.items():
                        print(f"\tEnter {key}: {description}")

                    choice = input("Please make a choice:\n")

                    if choice in menu_options:
                        description, handler = menu_options[choice]
                        result = handler()  # type: ignore
                        if result == "terminate":
                            raise KeyboardInterrupt()
                    else:
                        print("Please enter the correct choice.")

        except (KeyboardInterrupt, SystemExit):
            self.error("\n! Received keyboard interrupt, quitting threads.\n")
        finally:
            for runner in self._full_runner_list:
                runner.finish()

            self.broker.close()
            self.event_engine.stop()

    def add_ticker(
        self,
        code: str,
        datafeedClass: type[DataFeed],
        asset_type: str | AssetType | None = None,
        exchange: str | Exchange | None = None,
        is_subscribe: bool = True,
        **kwargs: Any,
    ) -> tuple[bool, str]:
        """Add a new ticker to the system.

        Args:
            code: Ticker symbol code
            datafeedClass: DataFeed class to instantiate
            asset_type: Asset type (optional)
            exchange: Exchange (optional)
            is_subscribe: Whether to subscribe to real-time data
            **kwargs: Additional arguments for datafeed

        Returns:
            Tuple of (success: bool, message: str)
        """
        symbol = normalize_name(
            code,
            asset_type.value if isinstance(asset_type, AssetType) else asset_type.upper(),
        )
        if symbol in self.ctx.symbols:
            return False, "symbol already existed in your data feeds"
        datafeed = datafeedClass(
            code,
            interval="1m",
            is_live=True,
            asset_type=asset_type,
            exchange=exchange,
            **kwargs,
        )

        if self.ctx.is_live:
            if hasattr(datafeed, "fetch_spot"):
                new_bar = datafeed.fetch_spot(interval="1m", tz=True)
                self.trace(
                    f"--- add_ticker --- new_bar: {new_bar}; code: {code}; symbol: {symbol}; asset_type: {datafeed.asset_type}"
                )
                datafeed.update_data(new_bar)
            if is_subscribe:
                self.ctx.broker.subscribe(symbol=code, exchange=exchange, asset_type=asset_type)

        self.ctx.add_datafeed(feed=datafeed)

        self.ctx.create_position(
            symbol=datafeed.symbol,
            exchange=datafeed.exchange,
            asset_type=datafeed.asset_type,
        )

        return True, ""

    def add_tickdata(
        self,
        symbol: str,
        tick: datetime,
        datafeedClass: type[DataFeed],
        asset_type: str | AssetType | None = None,
        exchange: str | Exchange | None = None,
    ) -> tuple[bool, str]:
        """Add new ticker data on the fly.

        Args:
            symbol: Symbol identifier
            tick: Timestamp for the tick data
            datafeedClass: DataFeed class to use
            asset_type: Asset type (optional)
            exchange: Exchange (optional)

        Returns:
            Tuple of (success: bool, message: str)
        """
        if symbol not in self.ctx.symbols:
            return self.add_ticker(symbol, tick, datafeedClass, asset_type, exchange)
        else:
            exchange = exchange if isinstance(exchange, Exchange) else Exchange(exchange.upper())
            datafeed = self.ctx.get_datafeed(symbol=symbol, exchange=exchange)
            if hasattr(datafeed, "fetch_spot"):
                new_bar = datafeed.fetch_spot(interval="1m", tz=True)
                self.trace("--- add_tickdata --- new_bar:", new_bar, symbol, tick)
                datafeed.update_data(new_bar)

            return True, ""
