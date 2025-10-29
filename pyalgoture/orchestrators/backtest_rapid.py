"""
High-performance backtest engine using Rust core via PyO3 bindings.

This module provides a Rust-accelerated backtest orchestrator that maintains
compatibility with the existing pyalgoture framework while offering significant
performance improvements for strategy backtesting.
"""

from itertools import chain
from typing import Any

try:
    from rapid_engine import BacktestConfig, BacktestEngine
except ImportError:
    raise ImportError(
        "Rust engine not available. Please build the rapid module first:\ncd rapid && maturin develop --release"
    )

from ..broker import Broker
from ..context import Context
from ..datafeed import DataFeed
from ..orchestra import Orchestrator
from ..statistic import Statistic
from ..strategy import Strategy


class BackTestRapid(Orchestrator):
    """
    High-performance backtest orchestrator using Rust core engine.

    This class provides significant performance improvements over the standard
    BackTest class while maintaining full compatibility with pyalgoture strategies.
    """

    def __init__(
        self,
        feeds: list[DataFeed] | DataFeed,
        StrategyClass: Strategy,
        trade_history: list[Any] | None = None,
        store_path: str = "",
        broker: Broker | None = None,
        statistic: Statistic | None = None,
        backtest_mode: str = "bar",
        do_print: bool = True,
        debug: bool = False,
        logger: Any | None = None,
        benchmark: DataFeed | None = None,
        prefix: str = "",
        hist_trade_skip_datafeed: bool = False,
        hist_trade_skip_checking: bool = False,
        # Rust engine specific parameters
        commission_rate: float = 0.0,
        slippage_bps: float = 0.0,
        batch_size: int = 1000,
        initial_cash: float = 100000.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize BackTestRapid orchestrator.

        Args:
            feeds: Data feeds for backtesting
            StrategyClass: Strategy class to be tested
            trade_history: Historical trade data (if any)
            store_path: Path to store results
            broker: Broker instance
            statistic: Statistic instance
            backtest_mode: Mode for backtesting ('bar', 'tick', 'bar_tick')
            do_print: Whether to print output
            debug: Debug mode flag
            logger: Logger instance
            benchmark: Benchmark data feed
            prefix: Prefix for output files
            hist_trade_skip_datafeed: Skip datafeed for historical trades
            hist_trade_skip_checking: Skip checking for historical trades
            commission_rate: Commission rate for trades
            slippage_bps: Slippage in basis points
            batch_size: Batch size for processing
            initial_cash: Initial cash amount
            **kwargs: Additional keyword arguments
        """
        if trade_history is None:
            trade_history = []

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
        if backtest_mode not in ["bar", "tick", "bar_tick"]:
            raise ValueError(f"backtest_mode must be one of ['bar', 'tick', 'bar_tick'], got {backtest_mode}")
        self.backtest_mode = backtest_mode

        # Rust engine configuration
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.batch_size = batch_size
        self.initial_cash = initial_cash

        # Initialize Rust engine components
        self.rust_config = None
        self.rust_engine = None

    def start(self) -> dict[str, Any]:
        """
        Start the high-performance backtesting process.

        Returns:
            Backtest results dictionary
        """
        self.event_engine.start()

        # Set up traditional pyalgoture components for compatibility
        self.add_runner(self.broker)
        self.add_runner(self.strategy)
        if self.statistic:
            self.add_runner(self.statistic)

        # Create context
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
        )
        self.ctx.tick = self.tradedays[0] if self.tradedays else None
        self.ctx.trade_history = self.trade_history
        self.ctx.backtest_mode = self.backtest_mode

        # Set up runners
        self._full_runner_list = list(chain(self._pre_hook_list, self._runner_list, self._post_hook_list))
        self.ctx.runners = self._full_runner_list

        # Initialize runners
        for runner in self._full_runner_list:
            runner.ctx = runner.context = self.ctx
            runner.logger = self.logger

        for runner in self._full_runner_list:
            runner.initialize()

        # Execute historical trades first (if any)
        if self.ctx.trade_history:
            for tick in self.tradedays:
                if tick in self.ctx.trade_history:
                    for trade in self.ctx.trade_history[tick]:
                        self.execute_history_trade(trade)

        # Use Rust engine for the main backtest loop
        results = self._run_rust_backtest()

        # Finish runners
        for runner in self._full_runner_list:
            runner.finish()

        while not self.event_engine.empty:
            pass

        self.event_engine.stop()

        return results

    def _run_rust_backtest(self) -> dict[str, Any]:
        """Run the main backtest using Rust engine."""
        # Determine date range
        start_date = min(self.tradedays).strftime("%Y-%m-%d %H:%M:%S")
        end_date = max(self.tradedays).strftime("%Y-%m-%d %H:%M:%S")

        # Create Rust configuration
        self.rust_config = BacktestConfig(
            start=start_date,
            end=end_date,
            cash=self.initial_cash,
            commission_rate=self.commission_rate,
            slippage_bps=self.slippage_bps,
            batch_size=self.batch_size,
        )

        # Create Rust engine
        self.rust_engine = BacktestEngine(self.rust_config)

        # Convert data to Rust format
        data = self._convert_feed_data()

        # Run backtest with strategy directly
        results = self.rust_engine.run_strategy(self.strategy, self.ctx, data)

        # return self._process_rust_results(results)
        return None

    def _convert_feed_data(self) -> list[dict[str, Any]]:
        """Convert DataFeeds to Rust format."""
        data = []

        # Create a unified timeline from all feeds
        all_bars = {}
        for tick in sorted(self.tradedays):
            tick_bars = {}
            for aio_symbol, feed in self.feeds.items():
                if tick in feed.data:
                    bar = feed.data[tick]
                    if bar:
                        tick_bars[aio_symbol] = bar

            if tick_bars:
                all_bars[tick] = tick_bars

        # Convert to list format for Rust engine
        for tick in sorted(all_bars.keys()):
            data.append({"datetime": tick, "bars": all_bars[tick]})

        return data

    # def _process_rust_results(self, results: dict[str, Any]) -> dict[str, Any]:
    #     """Process results from Rust engine and integrate with pyalgoture statistics."""
    #     # Update broker state with final results
    #     if hasattr(self.ctx.broker, "add_balance"):
    #         final_equity = results.get("equity", self.initial_cash)
    #         balance_change = final_equity - self.initial_cash
    #         if balance_change != 0:
    #             self.ctx.broker.add_balance(balance_change)

    #     # Process trades and update statistics
    #     trades = results.get("trades", [])
    #     for trade in trades:
    #         # Convert to pyalgoture trade format and update statistics
    #         # This would need proper TradeData object creation
    #         pass

    #     # Integrate with pyalgoture statistics
    #     if self.statistic:
    #         # Update statistic with equity curve and trades
    #         equity_curve = results.get("equity_curve", [])
    #         for eq_point in equity_curve:
    #             # Update statistic tracking
    #             pass

    #     return results
