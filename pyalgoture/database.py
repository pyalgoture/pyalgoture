import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo, available_timezones  # noqa

from .utils.objects import AssetType, AttrDict, BarData, Exchange, TickData  # noqa
from .utils.util_dt import UTC_TZ

DB_TZ = UTC_TZ


def convert_tz(dt: datetime) -> datetime:
    """Convert timezone of datetime object to DB_TZ.

    Args:
        dt: Datetime object to convert

    Returns:
        Datetime object converted to DB_TZ timezone
    """
    return dt.astimezone(DB_TZ)


def convert_tz1(dt: datetime) -> datetime:
    """Convert timezone of datetime object to DB_TZ using timestamp method.

    Args:
        dt: Datetime object to convert

    Returns:
        Datetime object converted to DB_TZ timezone
    """
    return datetime.fromtimestamp(dt.timestamp(), DB_TZ)


@dataclass
class BarOverview:
    """Overview of bar data stored in database.

    Attributes:
        symbol: Trading symbol identifier
        exchange: Exchange where the symbol is traded
        interval: Time interval for the bars (e.g., '1m', '5m', '1h')
        count: Number of bar records available
        start: Earliest datetime of available data
        end: Latest datetime of available data
    """

    symbol: str = ""
    exchange: Exchange | None = None
    interval: str | None = None
    count: int = 0
    start: datetime | None = None
    end: datetime | None = None


@dataclass
class TickOverview:
    """Overview of tick data stored in database.

    Attributes:
        symbol: Trading symbol identifier
        exchange: Exchange where the symbol is traded
        count: Number of tick records available
        start: Earliest datetime of available data
        end: Latest datetime of available data
    """

    symbol: str = ""
    exchange: Exchange | None = None
    count: int = 0
    start: datetime | None = None
    end: datetime | None = None


class BaseDatabase(ABC):
    """Abstract database class for connecting to different database systems.

    This class defines the interface for database operations including
    saving, loading, and managing bar and tick data.
    """

    @abstractmethod
    def save_bar_data(self, bars: list[BarData], stream: bool = False) -> bool:
        """Save bar data into database.

        Args:
            bars: List of BarData objects to save
            stream: Whether to use streaming mode for large datasets

        Returns:
            True if save operation was successful, False otherwise
        """

    @abstractmethod
    def save_tick_data(self, ticks: list[TickData], stream: bool = False) -> bool:
        """Save tick data into database.

        Args:
            ticks: List of TickData objects to save
            stream: Whether to use streaming mode for large datasets

        Returns:
            True if save operation was successful, False otherwise
        """

    @abstractmethod
    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[BarData]:
        """Load bar data from database.

        Args:
            symbol: Trading symbol to load data for
            exchange: Exchange where the symbol is traded
            interval: Time interval for the bars
            start: Start datetime for data range
            end: End datetime for data range

        Returns:
            List of BarData objects within the specified range
        """

    @abstractmethod
    def load_tick_data(self, symbol: str, exchange: Exchange, start: datetime, end: datetime) -> list[TickData]:
        """Load tick data from database.

        Args:
            symbol: Trading symbol to load data for
            exchange: Exchange where the symbol is traded
            start: Start datetime for data range
            end: End datetime for data range

        Returns:
            List of TickData objects within the specified range
        """

    def delete_bar_data(self, symbol: str, exchange: Exchange, interval: str) -> int:
        """Delete all bar data with given symbol + exchange + interval.

        Args:
            symbol: Trading symbol to delete data for
            exchange: Exchange where the symbol is traded
            interval: Time interval for the bars to delete

        Returns:
            Number of records deleted (default implementation returns 0)
        """
        return 0

    def delete_tick_data(self, symbol: str, exchange: Exchange) -> int:
        """Delete all tick data with given symbol + exchange.

        Args:
            symbol: Trading symbol to delete data for
            exchange: Exchange where the symbol is traded

        Returns:
            Number of records deleted (default implementation returns 0)
        """
        return 0

    @abstractmethod
    def get_bar_overview(self) -> list[BarOverview]:
        """Return bar data available in database.

        Returns:
            List of BarOverview objects describing available bar data
        """

    @abstractmethod
    def get_tick_overview(self) -> list[TickOverview]:
        """Return tick data available in database.

        Returns:
            List of TickOverview objects describing available tick data
        """
