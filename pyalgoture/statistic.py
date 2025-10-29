import json
import os
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import pandas as pd

from .base import Base
from .datafeed import DataFeed
from .utils.portfolio_manager import PortfolioManager
from .utils.util_objects import set_default

# warnings.filterwarnings("ignore", category=RuntimeWarning)

pd.set_option("display.max_rows", 150)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")


def nested_defaultdict() -> defaultdict[Any, dict[Any, Any]]:
    """Create a nested defaultdict with dict as the default factory."""
    return defaultdict(dict)


class Statistic(Base):
    """
    A class for handling trading statistics, portfolio management, and performance analysis.

    This class provides functionality for tracking trades, calculating statistics,
    generating plots, and creating performance reports.
    """

    def __init__(self) -> None:
        """Initialize the Statistic instance with default values."""
        super().__init__()

        self.stats_result = pd.Series(dtype="float64")
        self.data = pd.DataFrame()

        self.custom_plot: defaultdict[Any, defaultdict[Any, dict[Any, Any]]] = defaultdict(nested_defaultdict)
        self.custom_stat: dict[str, Any] = {}

        self.pm: PortfolioManager | None = None

    def reset(self) -> None:
        """Reset all statistics and data to initial state."""
        self.stats_result = pd.Series(dtype="float64")
        self.data = pd.DataFrame()
        self.pm = None

    def initialize(self) -> None:
        """Initialize the portfolio manager with current context data."""
        self.pm = PortfolioManager(aum=self.ctx.default_account.balance, logger=self.logger)
        self.pm.init(
            positions={position.aio_position_id: position for position in self.ctx.get_positions()},
            account=self.ctx.default_account,
            symbols={symbol for symbol, _ in self.ctx.symbols},
        )

    def on_tick(self, tick_data: Any) -> None:
        """Handle tick data events (currently no implementation)."""
        pass

    def upload_trade(self, trade: Any) -> None:
        """
        Upload a trade to the portfolio manager.

        Args:
            trade: Trade object to be uploaded
        """
        if self.pm:
            self.pm.upload_trade(trade)
        else:
            print(f"PM is not initialized yet. trade:{trade}")

    def on_bar(self, tick: Any) -> None:
        """
        Handle bar data events and update positions.

        Args:
            tick: Current tick data
        """
        for position in self.ctx.get_positions():
            bar_data = self.ctx.get_bar_data(symbol=position.symbol, exchange=position.exchange)
            if bar_data:
                position.update_by_marketdata(
                    last_price=bar_data["close"],
                    available_balance=self.ctx.strategy.available_balance,
                )
        if self.pm:
            self.pm.add_snapshot(tick)

    def finish(self) -> None:
        """Finalize statistics processing (only needed for live trading)."""
        if self.ctx.is_live:
            print("****" * 8)
            self.on_bar(self.ctx.tick)

    def add_custom_stat(self, name: str, val: Any = None) -> bool:
        """
        Add a custom statistic.

        Args:
            name: Name of the custom statistic
            val: Value of the custom statistic (can be callable)

        Returns:
            True if successfully added
        """
        self.custom_stat[name] = val
        return True

    def stats(
        self,
        interval: str | None = None,
        benchmark: DataFeed | None = None,
        resample_interval: str | None = None,
        annualization: Any | None = None,
        risk_free: float = 0.0,
        required_return: float = 0.0,
        return_dict: bool = False,
        use_qs: bool = False,
        lite: bool = False,
    ) -> Any:
        """
        Generate comprehensive trading statistics.

        Args:
            interval: Time interval for analysis
            benchmark: Benchmark data feed for comparison
            resample_interval: Interval for resampling data
            annualization: Annualization factor
            risk_free: Risk-free rate for calculations
            required_return: Required return threshold
            return_dict: Whether to return results as dictionary
            use_qs: Whether to use quantstats library
            lite: Whether to generate lite version of stats

        Returns:
            Statistics results as Series or dict
        """
        benchmark = benchmark or self.ctx.benchmark
        if not interval:
            interval = benchmark.interval if benchmark else next(iter(self.ctx.feeds.values())).interval

        if not self.pm:
            return None

        stats = (
            self.pm.genereate_stats_lite(
                interval=interval,
                resample_interval=resample_interval,
                annualization=annualization,
                risk_free=risk_free,
                required_return=required_return,
            )
            if lite
            else self.pm.genereate_stats(
                interval=interval,
                benchmark=benchmark,
                resample_interval=resample_interval,
                annualization=annualization,
                risk_free=risk_free,
                required_return=required_return,
                use_qs=use_qs,
            )
        )

        if self.custom_stat and stats is not None:
            for attr, val in self.custom_stat.items():
                stats[attr] = val() if callable(val) else val

        if stats is not None:
            self.stats_result = stats
            self.data = self.pm.data

            if return_dict:
                return json.loads(json.dumps(self.stats_result.to_dict(), indent=4, default=set_default))

        return stats

    def to_dict(self) -> dict[str, Any]:
        """
        Convert statistics to dictionary format.

        Returns:
            Statistics as dictionary
        """
        res: dict[str, Any] = {}
        if self.stats_result.empty:
            res = self.stats(return_dict=True)
        else:
            res = json.loads(json.dumps(self.stats_result.to_dict(), indent=4, default=set_default))
        return res

    def draw(
        self,
        data_point: float,
        graph_name: str,
        figure_name: str,
        color: str = "",
        is_line: bool = True,
        is_scatter: bool = False,
        is_bar: bool = False,
        plot_height: int = 110,
        tick: Any | None = None,
    ) -> None:
        """
        Draw custom data points and construct graph and figure.

        Args:
            data_point: Data point value to plot
            graph_name: Name of the graph (multiple graphs can exist in same figure)
            figure_name: Name of the figure
            color: Color for the plot (optional)
            is_line: Whether to draw as line plot
            is_scatter: Whether to draw as scatter plot
            is_bar: Whether to draw as bar plot
            plot_height: Height of the plot in pixels
            tick: Tick data (uses current context tick if None)

        Note:
            First layer is figure name, second layer consists of graph_name|type|color,
            last layer is tick and data.
        """
        graph_type = "scatter" if is_scatter else "bar" if is_bar else "line"

        figure_key = f"{figure_name}|{plot_height}"
        graph_key = f"{graph_name}|{graph_type}|{color}"

        self.custom_plot[figure_key][graph_key][tick or self.ctx.tick] = data_point

    def draw_figure(self) -> Any:
        """
        Plot custom bokeh figure into report html.

        Returns:
            Figure object from strategy
        """
        return self.ctx.strategy.draw_figure()

    def plot(
        self,
        path: str | None = None,
        feeds: list[Any] | None = None,
        prefix: str = "",
        suffix: str = "",
        interactive: bool = True,
        open_browser: bool | None = None,
        debug: bool = False,
        extra_ohlc: bool = False,
        only_ohlc: bool = False,
        in_nav: bool = False,
        custom_plot_data: dict[str, Any] | None = None,
        draw_figure: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate interactive plots and reports.

        Args:
            path: Output path for the report
            feeds: List of data feeds to plot
            prefix: Prefix for the report filename
            suffix: Suffix for the report filename
            interactive: Whether to create interactive plots
            open_browser: Whether to open browser automatically
            debug: Whether to enable debug mode
            extra_ohlc: Whether to include extra OHLC data
            only_ohlc: Whether to show only OHLC data
            in_nav: Whether to include in navigation
            custom_plot_data: Custom plot data dictionary
            draw_figure: Custom draw figure function
            **kwargs: Additional keyword arguments

        Returns:
            Plot result from portfolio manager
        """
        feeds = feeds or list(self.ctx.feeds.values())
        custom_plot_data = custom_plot_data or {}

        if not self.pm:
            return None

        return self.pm.plot(
            path=os.path.join(path, f"{prefix}report{suffix}.html") if path else None,
            feeds=feeds,
            interactive=interactive,
            open_browser=open_browser,
            debug=debug,
            extra_ohlc=extra_ohlc,
            only_ohlc=only_ohlc,
            in_nav=in_nav,
            custom_plot_data=custom_plot_data,
            draw_figure=draw_figure,
            **kwargs,
        )

    def contest_output(
        self,
        interval: str | None = None,
        path: str | None = None,
        benchmark: DataFeed | None = None,
        resample_interval: str | None = None,
        prefix: str = "",
        suffix: str = "",
        is_live: bool = False,
        # Below parameters are for plot only
        is_plot: bool = True,
        feeds: list[Any] | None = None,
        extra_ohlc: bool = False,
        only_ohlc: bool = False,
        in_nav: bool = False,
        open_browser: bool | None = None,
        debug: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Generate contest output with statistics and plots.

        Args:
            interval: Time interval for analysis
            path: Output path for files
            benchmark: Benchmark data feed
            resample_interval: Interval for resampling
            prefix: Filename prefix
            suffix: Filename suffix
            is_live: Whether this is live trading
            is_plot: Whether to generate plots
            feeds: List of data feeds
            extra_ohlc: Whether to include extra OHLC data
            only_ohlc: Whether to show only OHLC data
            in_nav: Whether to include in navigation
            open_browser: Whether to open browser
            debug: Whether to enable debug mode
            **kwargs: Additional keyword arguments

        Returns:
            Statistics results
        """
        benchmark = benchmark or self.ctx.benchmark
        if not interval:
            interval = benchmark.interval if benchmark else next(iter(self.ctx.feeds.values())).interval

        path = path or self.ctx.path
        feeds = feeds or list(self.ctx.feeds.values())

        self.trace(f">>>>[statistic - contest_output] interval: {interval}, path: {path}")

        if self.pm:
            self.stats_result = self.pm.contest_output(
                path=path,
                interval=interval,
                benchmark=benchmark,
                resample_interval=resample_interval,
                prefix=prefix,
                suffix=suffix,
                is_live=is_live,
                # Below parameters are for plot only
                is_plot=is_plot,
                feeds=feeds,
                extra_ohlc=extra_ohlc,
                only_ohlc=only_ohlc,
                in_nav=in_nav,
                custom_plot_data=self.custom_plot,
                draw_figure=self.draw_figure,
                open_browser=open_browser,
                debug=debug,
                **kwargs,
            )
        return self.stats_result
