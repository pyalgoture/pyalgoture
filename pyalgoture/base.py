from abc import ABC, abstractmethod
from datetime import datetime

from .context import Context
from .utils.event_engine import EVENT_LOG, Event
from .utils.logger import LogData


class Base(ABC):
    """Abstract base class for trading strategies and components.

    This class provides the core interface and logging functionality
    for trading strategies in both backtesting and live trading environments.
    """

    ctx: Context
    context: Context

    def initialize(self) -> None:
        """Initialize the component before backtesting starts."""

    def finish(self) -> None:
        """Clean up after backtesting completes."""

    @abstractmethod
    def on_tick(self, tick_data: dict) -> None:
        """Process tick data.

        This method must be implemented by subclasses and will be
        invoked for every tick during execution.

        Args:
            tick_data: Dictionary containing tick data
        """

    @abstractmethod
    def on_bar(self, tick: datetime) -> None:
        """Process bar data.

        This method must be implemented by subclasses and will be
        invoked for every bar during execution.

        Args:
            tick: Datetime of the current bar
        """

    def log(self, log_data: LogData) -> None:
        """Send log data to the appropriate logging system.

        Args:
            log_data: Log data to be processed
        """
        event = Event(type=EVENT_LOG, data=log_data, priority=2)
        if self.ctx.is_live:
            self.ctx.event_engine.put(event)
        else:
            self.ctx.strategy.process_log_event(event)

    def _create_log_data(self, content: str, level: str) -> LogData:
        """Create LogData instance with current tick information.

        Args:
            content: Log message content
            level: Log level

        Returns:
            LogData instance with appropriate tick information
        """
        return LogData(
            tick=self.ctx.now if self.ctx.is_live else self.ctx.tick,
            content=content,
            level=level,
        )

    def trace(self, content: str) -> None:
        """Log message at trace level.

        Args:
            content: Message content to log
        """
        self.log(self._create_log_data(content, "TRACE"))

    def debug(self, content: str) -> None:
        """Log message at debug level.

        Args:
            content: Message content to log
        """
        self.log(self._create_log_data(content, "DEBUG"))

    def info(self, content: str) -> None:
        """Log message at info level.

        Args:
            content: Message content to log
        """
        self.log(self._create_log_data(content, "INFO"))

    def notice(self, content: str) -> None:
        """Log message at notice level.

        Args:
            content: Message content to log
        """
        self.log(self._create_log_data(content, "NOTICE"))

    def error(self, content: str) -> None:
        """Log message at error level.

        Args:
            content: Message content to log
        """
        self.log(self._create_log_data(content, "ERROR"))

    def warning(self, content: str) -> None:
        """Log message at warning level.

        Args:
            content: Message content to log
        """
        self.log(self._create_log_data(content, "WARNING"))

    def critical(self, content: str) -> None:
        """Log message at critical level.

        Args:
            content: Message content to log
        """
        self.log(self._create_log_data(content, "CRITICAL"))

    def exception(self, content: str) -> None:
        """Log message at exception level.

        Args:
            content: Message content to log
        """
        self.log(self._create_log_data(content, "EXCEPTION"))
