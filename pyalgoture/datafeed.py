import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import (
    ALL_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from datetime import datetime, timedelta
from random import randint
from typing import Any

import pandas as pd
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

from .databases.filesystem_db import FilesystemDatabase
from .utils.logger import change_logger_level, get_logger
from .utils.models import normalize_name
from .utils.objects import AssetType, BarData, Exchange

# from .utils.util_auth import authentication
from .utils.util_dt import tz_manager

Num = int | float


class DataFeed(ABC):
    """Abstract base class for data feeds providing market data.

    This class handles fetching, storing, and managing market data for various
    financial instruments across different exchanges and asset types.
    """

    def __init__(
        self,
        code: str,
        asset_type: AssetType | str,
        exchange: Exchange | str | None = None,
        symbol: str | None = None,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
        period: str | None = None,
        interval: str | None = None,
        min_volume: Num | None = None,
        max_volume: Num | None = None,
        min_notional: Num | None = None,
        tick_size: Num | None = None,
        step_size: Num | None = None,
        launch_date: str | datetime | None = None,
        delivery_date: str | datetime | None = None,
        order_ascending: bool | None = True,
        store_path: str | None = None,
        is_live: bool | None = False,
        debug: bool = False,
        jitter: int = 0,
        commission_rate: float | None = None,
        min_commission_rate: float | None = None,
        taker_commission_rate: float | None = None,
        maker_commission_rate: float | None = None,
        is_contract: bool | None = None,
        consider_price: bool | None = None,
        is_inverse: bool | None = None,
        db: Any | None = None,
        logger: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize DataFeed instance.

        Args:
            code: Instrument code (e.g., 'BTCUSDT', '3690.HK')
            asset_type: Type of asset (stock, crypto, future, etc.)
            exchange: Exchange where the instrument is traded
            symbol: Symbol name (defaults to code if not provided)
            start_date: Start date for data collection
            end_date: End date for data collection
            period: Period specification (e.g., '1d', '1mo', '1y')
            interval: Data interval (e.g., '1m', '5m', '1h', '1d')
            min_volume: Minimum volume filter
            max_volume: Maximum volume filter
            min_notional: Minimum notional value
            tick_size: Minimum price movement
            step_size: Minimum quantity increment
            launch_date: Instrument launch date
            delivery_date: Delivery date for futures
            order_ascending: Whether to order data chronologically
            store_path: Path for data storage
            is_live: Whether this is live trading mode
            debug: Enable debug logging
            jitter: Random time jitter in seconds
            commission_rate: General commission rate
            min_commission_rate: Minimum commission rate
            taker_commission_rate: Taker commission rate
            maker_commission_rate: Maker commission rate
            is_contract: Whether this is a contract instrument
            consider_price: Whether to consider price in calculations
            is_inverse: Whether this is an inverse contract
            db: Database instance for data storage

        Raises:
            ValueError: If code is invalid or period specification is incorrect
        """
        # authentication(bypass=kwargs.get("bypass", False), _from="datafeed")

        self.code = code
        self.symbol = symbol or code
        self.asset_type = asset_type
        self.exchange = exchange
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.interval = interval
        self.max_volume = max_volume
        self.min_volume = min_volume
        self.min_notional = min_notional
        self.tick_size = tick_size
        self.step_size = step_size
        self.launch_date = parse(launch_date) if isinstance(launch_date, str) else launch_date
        self.delivery_date = parse(delivery_date) if isinstance(delivery_date, str) else delivery_date
        self.is_trading = True

        self.order_ascending = order_ascending
        self.store_path = store_path
        self.debug = debug
        self.jitter = jitter
        self.commission_rate = commission_rate
        self.min_commission_rate = min_commission_rate
        self.taker_commission_rate = taker_commission_rate
        self.maker_commission_rate = maker_commission_rate
        self.is_contract = is_contract
        self.consider_price = consider_price
        self.is_inverse = is_inverse
        self.is_live = is_live
        self.logger = logger or get_logger()

        if not self.code or not isinstance(self.code, str):
            raise ValueError(f"{self.code} is invalid, the code has to be a str.")

        self.aio_symbol: str = f"{self.symbol}|{self.exchange.value}"

        self.end_date = self.end_date or datetime.now()
        self.end_date: str = (
            parse(self.end_date).strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(self.end_date, str)
            else self.end_date.strftime("%Y-%m-%d %H:%M:%S")
        )

        if self.period and not self.start_date:
            if not self._convert_period():
                raise ValueError(
                    "The specified period is not correct. It has to be one of 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd"
                )
        else:
            self.start_date = parse(self.start_date) if isinstance(self.start_date, str) else self.start_date

        self.start_date: str = self.start_date.strftime("%Y-%m-%d %H:%M:%S") if self.start_date else self.start_date

        if self.debug:
            change_logger_level(self.logger, "TRACE")
            self.logger.trace(
                f"[DEBUG] [DataFeed - init] symbol:{self.symbol}| start_date:{self.start_date}[{type(self.start_date)}] | end_date:{self.end_date}[{type(self.end_date)}] | interval:{self.interval}"
            )

        if self.start_date and self.start_date == self.end_date:
            self.end_date: str = (parse(self.end_date).replace(hour=23, minute=59, second=59)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        data_path = None
        if self.store_path:
            data_path = os.path.join(self.store_path, ".data")
            if not os.path.exists(data_path):
                os.mkdir(data_path)
        self.db = db or FilesystemDatabase(base_path=data_path)

        self.data = OrderedDict()

        if not self.step_size:
            self.fetch_info()

        if not self.is_live:
            # TODO: in thread
            self.data = self.fetch_data()

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return (
            f"<DataFeed code:{self.code}, symbol:{self.symbol}, asset_type:{self.asset_type}, "
            f"exchange:{self.exchange}, start_date:{self.start_date}, end_date:{self.end_date}, "
            f"interval:{self.interval}, max_volume:{self.max_volume}, min_volume:{self.min_volume}, "
            f"min_notional:{self.min_notional}, tick_size:{self.tick_size}, step_size:{self.step_size}, "
            f"launch_date:{getattr(self, 'launch_date', None)}, "
            f"delivery_date:{getattr(self, 'delivery_date', None)}, is_trading:{self.is_trading}; | "
            f"commission_rate:{self.commission_rate}, taker_commission_rate:{self.taker_commission_rate}, "
            f"maker_commission_rate:{self.maker_commission_rate}, is_contract:{self.is_contract}, "
            f"consider_price:{self.consider_price}, is_inverse:{self.is_inverse} >"
        )

    def __hash__(self) -> int:
        return hash((self.symbol, self.asset_type, self.exchange))

    def __eq__(self, other) -> bool:
        if not isinstance(other, DataFeed):
            return False
        return (self.symbol, self.asset_type, self.exchange) == (
            other.symbol,
            other.asset_type,
            other.exchange,
        )

    def __ne__(self, other) -> bool:
        return not (self == other)

    def _convert_period(self) -> bool:
        """Convert period string to start_date.

        Returns:
            True if conversion successful, False otherwise
        """
        end_date = parse(self.end_date)

        period_conversions = {
            "ytd": lambda: end_date.replace(month=1, day=1),
            "d": lambda days: end_date - relativedelta(days=days),
            "mo": lambda months: end_date - relativedelta(months=months),
            "y": lambda years: end_date - relativedelta(years=years),
        }

        if self.period == "ytd":
            self.start_date = period_conversions["ytd"]()
        elif "d" in self.period:
            days = int(self.period.replace("d", ""))
            self.start_date = period_conversions["d"](days)
        elif "mo" in self.period:
            months = int(self.period.replace("mo", ""))
            self.start_date = period_conversions["mo"](months)
        elif "y" in self.period:
            years = int(self.period.replace("y", ""))
            self.start_date = period_conversions["y"](years)
        else:
            return False
        return True

    def _format_date_string(self, date: str | datetime | None) -> str:
        """Format date to standard string format.

        Args:
            date: Date to format

        Returns:
            Formatted date string
        """
        if isinstance(date, str):
            return date
        elif date:
            return date.strftime("%Y-%m-%d %H:%M:%S")
        return ""

    def _calculate_interval_alignment(self, dt: datetime, interval: str) -> dict:
        """Calculate proper alignment for datetime based on interval.

        Args:
            dt: Datetime to align
            interval: Interval specification

        Returns:
            Dictionary with alignment parameters
        """
        if "m" in interval:
            interval_int = int(interval.replace("m", ""))
            return {"minute": (dt.minute // interval_int) * interval_int}
        elif "h" in interval:
            interval_int = int(interval.replace("h", ""))
            return {"hour": (dt.hour // interval_int) * interval_int, "minute": 0}
        elif "d" in interval:
            interval_int = int(interval.replace("d", ""))
            return {"day": (dt.day // interval_int) * interval_int, "hour": 0, "minute": 0}
        return {}

    def fetch_data(
        self,
        code: str | None = None,
        interval: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """Fetch market data for the specified parameters.

        Args:
            code: Instrument code (defaults to self.code)
            interval: Data interval (defaults to self.interval)
            start_date: Start date (defaults to self.start_date)
            end_date: End date (defaults to self.end_date)

        Returns:
            Dictionary of market data keyed by datetime
        """
        start_date = self._format_date_string(start_date) or self.start_date
        end_date = self._format_date_string(end_date) or self.end_date
        code = code or self.code
        interval = interval or self.interval
        exchange = self.exchange
        asset_type = self.asset_type
        symbol = self.symbol if code == self.code else normalize_name(code, asset_type.value)

        self.logger.trace(
            f"[fetch_data - DEBUG] code:{code}; symbol:{symbol}; exchange:{exchange}; "
            f"asset_type:{asset_type}; interval:{interval}; start_date:{start_date}; "
            f"end_date:{end_date}; launch_date:{self.launch_date}; delivery_date:{self.delivery_date}"
        )

        try:
            overview = self.db.get_bar_overview(
                symbol=symbol,
                exchange=exchange.value,
                interval=interval,
                tz=tz_manager.tz,
            )
            self.logger.trace(f"overview:{overview}")

            ori_sd = overview["start"].strftime("%Y-%m-%d %H:%M:%S") if overview.get("start") else ""
            ori_ed = overview["end"].strftime("%Y-%m-%d %H:%M:%S") if overview.get("end") else ""

            # NOTE: if has overview data, req start date add 2 more days in case cannot fetch all data and need to redownload
            req_sd = (
                max(
                    start_date,
                    (self.launch_date + timedelta(days=2 if ori_sd else 0)).strftime("%Y-%m-%d %H:%M:%S"),
                )
                if self.launch_date
                else start_date
            )
            req_sd = parse(req_sd).replace(second=0)
            req_ed = parse(end_date).replace(second=0)

            # Calculate interval alignment
            alignment_params = self._calculate_interval_alignment(req_ed, interval)
            alignment_params_sd = self._calculate_interval_alignment(req_sd, interval)

            req_sd = req_sd.replace(**alignment_params_sd).strftime("%Y-%m-%d %H:%M:%S")
            req_ed = req_ed.replace(**alignment_params).strftime("%Y-%m-%d %H:%M:%S")
            req_ed = min(req_ed, self.delivery_date.strftime("%Y-%m-%d %H:%M:%S")) if self.delivery_date else req_ed

            self.logger.trace(
                f"[DEBUG] \n\tExisting start & end are {ori_sd} <=> {ori_ed}; "
                f"\n\tRequested start & end are {start_date} <=> {end_date}; "
                f"\n\tActual request start & end are {req_sd} <=> {req_ed}"
            )

            # Check if existing data covers the request
            if not start_date or (req_sd < ori_sd) or (req_ed > ori_ed):
                self.logger.trace(
                    f"The newly request date range is not covered for {code}... "
                    f"Existing start & end are {ori_sd} <=> {ori_ed}; "
                    f"Requested start & end are {start_date} <=> {end_date};"
                )

                if not (ori_sd and ori_ed) or (req_sd < ori_sd and req_ed > ori_ed):
                    # No existing data OR request out of current data range
                    data = self._download_data(code=code, interval=interval, start_date=req_sd, end_date=req_ed)
                    if data:
                        self.db.save_bar_data(bars=data)
                else:
                    tmp_sd = req_sd if req_sd < ori_sd else ori_ed
                    tmp_ed = ori_sd if req_sd < ori_sd else req_ed

                    new_data = self.fetch_hist(code=code, interval=interval, start=tmp_sd, end=tmp_ed)
                    self.logger.trace(
                        f"Newly request date range start:{tmp_sd}, end:{tmp_ed}, interval:{interval} "
                        f"for {symbol}. New data length:{len(new_data)}"
                    )
                    if new_data:
                        self.db.save_bar_data(bars=new_data)

            data = self.db.load_bar_data(
                symbol=symbol,
                exchange=exchange.value,
                interval=interval,
                start=start_date,
                end=end_date,
                tz=tz_manager.tz,
            )

            final_sd = next(iter(data)) if data else None
            final_ed = list(data.keys())[-1] if data else None

            if not data:
                new_data = self.fetch_hist(code=code, interval=interval, start=req_sd, end=req_ed)
                self.logger.error(
                    f"Failed to load data, Redownloading.... Newly request date range start:{req_sd}, "
                    f"end:{req_ed}, interval:{interval} for {symbol}. New data length:{len(new_data)}"
                )
                if new_data:
                    self.db.save_bar_data(bars=new_data)

            self.logger.trace(
                f"code:{code} - symbol:{symbol}[{interval} - {exchange.value}] data is up-to-date now. "
                f"Current date range:{final_sd} <=> {final_ed}"
            )

            if not data:
                self.logger.error(
                    f"Data is empty, something went wrong when download {symbol} data. "
                    f"start: {start_date}, end: {end_date}"
                )
                return data

            if not self.start_date:
                ds = list(data.keys())
                self.start_date = ds[0].strftime("%Y-%m-%d %H:%M:%S")
                self.end_date = ds[-1].strftime("%Y-%m-%d %H:%M:%S")

            if self.jitter:
                data = {
                    k + timedelta(seconds=seconds): {**v, "datetime": v["datetime"] + timedelta(seconds=seconds)}
                    for k, v in data.items()
                    for seconds in [randint(-self.jitter, self.jitter)]
                }

            return data

        except Exception as e:
            self.logger.exception(
                f"Something went wrong when download data. Error: {e} in line {e.__traceback__.tb_lineno} "
                f"for file {e.__traceback__.tb_frame.f_code.co_filename}. symbol:{symbol}; "
                f"start: {start_date}; end: {end_date}; interval:{interval}"
            )
            return {}

    @property
    def dataframe(self) -> pd.DataFrame:
        """Convert data to pandas DataFrame.

        Returns:
            DataFrame with datetime index and market data columns
        """
        if not self.data:
            return pd.DataFrame()

        try:
            df = pd.DataFrame.from_records([v.to_dict(return_obj=False, return_aio=False) for v in self.data.values()])
            df["datetime"] = pd.to_datetime(df["datetime"].astype(str))
            df.set_index("datetime", inplace=True)
            df.index = df.index.tz_convert(tz_manager.tz)

            if df.index.freq is None:
                df.index.freq = df.index.inferred_freq
            return df
        except Exception as e:
            self.logger.exception(
                f"Something went wrong when convert data into dataframe. Error: {e} "
                f"in line {e.__traceback__.tb_lineno} for file {e.__traceback__.tb_frame.f_code.co_filename}. "
                f"start_date:{self.start_date}; end_date:{self.end_date}; symbol:{self.symbol}; "
                f"asset_type:{self.asset_type}; exchange:{self.exchange}; interval:{self.interval};"
            )
            return pd.DataFrame()

    @abstractmethod
    def _download_data(self, code: str) -> dict:
        """Download data from external source.

        Args:
            code: Instrument code to download

        Returns:
            Dictionary of downloaded data
        """

    def update_data(
        self,
        new_data: dict[datetime, BarData] | BarData,
        force_update: bool = False,
        old_data: dict | None = None,
    ) -> dict | None:
        """Update the data with new data.

        Args:
            new_data: New data to update (single BarData or dict of BarData)
            force_update: Whether to force update existing data
            old_data: Existing data to update (defaults to self.data)

        Returns:
            Updated data dictionary or None if update failed
        """
        try:
            data = old_data or self.data
            if isinstance(new_data, BarData):
                new_data = {new_data.datetime: new_data}

            for nd, nitem in new_data.items():
                if force_update or nd not in data:
                    data[nd] = nitem

            # Sort data
            return dict(sorted(data.items(), reverse=not self.order_ascending))
        except Exception as e:
            self.logger.exception(f"Something went wrong when update data, {e} - new_data:{new_data}")
            return None

    def set_status(self, is_trading: bool) -> None:
        """Set trading status.

        Args:
            is_trading: Whether the instrument is currently trading
        """
        self.is_trading = is_trading

    def _validate_positive_value(self, value: Num, name: str) -> None:
        """Validate that a value is positive.

        Args:
            value: Value to validate
            name: Name of the parameter for error message

        Raises:
            ValueError: If value is not positive
        """
        if value <= 0:
            raise ValueError(f"The {name} cannot be less than or equal to 0.")

    def set_min_volume(self, min_volume: Num) -> None:
        """Set minimum volume filter.

        Args:
            min_volume: Minimum volume value

        Raises:
            ValueError: If min_volume is not positive
        """
        self._validate_positive_value(min_volume, "min_volume")
        self.min_volume = min_volume

    def set_max_volume(self, max_volume: Num) -> None:
        """Set maximum volume filter.

        Args:
            max_volume: Maximum volume value

        Raises:
            ValueError: If max_volume is not positive
        """
        self._validate_positive_value(max_volume, "max_volume")
        self.max_volume = max_volume

    def set_min_notional(self, min_notional: Num) -> None:
        """Set minimum notional value.

        Args:
            min_notional: Minimum notional value

        Raises:
            ValueError: If min_notional is not positive
        """
        self._validate_positive_value(min_notional, "min_notional")
        self.min_notional = min_notional

    def set_tick_size(self, tick_size: Num) -> None:
        """Set tick size (minimum price movement).

        Args:
            tick_size: Tick size value

        Raises:
            ValueError: If tick_size is not positive
        """
        self._validate_positive_value(tick_size, "tick_size")
        self.tick_size = tick_size

    def set_step_size(self, step_size: Num) -> None:
        """Set step size (minimum quantity increment).

        Args:
            step_size: Step size value

        Raises:
            ValueError: If step_size is not positive
        """
        self._validate_positive_value(step_size, "step_size")
        self.step_size = step_size

    def set_launch_date(self, launch_date: datetime | str) -> None:
        """Set instrument launch date.

        Args:
            launch_date: Launch date as datetime or string
        """
        if launch_date:
            self.launch_date = parse(launch_date) if isinstance(launch_date, str) else launch_date

    def set_delivery_date(self, delivery_date: datetime | str) -> None:
        """Set delivery date for futures contracts.

        Args:
            delivery_date: Delivery date as datetime or string
        """
        if delivery_date:
            self.delivery_date = parse(delivery_date) if isinstance(delivery_date, str) else delivery_date


class DataFeedDownloader:
    """DataFeed downloader for batch processing multiple instruments.

    Example:
        dfd = DataFeedDownloader(
            DataFeedClass=CryptoDataFeed,
            codes=['ETHUSDT', 'BTCUSDT', 'APEUSDT'],
            exchange='bybit',
            asset_type='future',
            start_date='2023-08-27 11:21:23',
            end_date='2023-10-24 11:21:23',
            interval='4h',
            order_ascending=True,
            store_path=base,
            is_fetch_info=True
        )
        feeds = dfd.start()
    """

    def __init__(
        self,
        DataFeedClass: type[DataFeed],
        codes: list[str],
        asset_type: AssetType | dict[str, AssetType],
        exchange: Exchange | dict[str, Exchange] | None = None,
        start_date: str | datetime | dict[str, datetime] | None = None,
        end_date: str | datetime | dict[str, datetime] | None = None,
        period: str | dict[str, str] | None = None,
        interval: str | dict[str, str] | None = None,
        max_volume: Num | dict[str, Num] | None = None,
        min_volume: Num | dict[str, Num] | None = None,
        min_notional: Num | dict[str, Num] | None = None,
        tick_size: Num | dict[str, Num] | None = None,
        step_size: Num | dict[str, Num] | None = None,
        order_ascending: bool | None = True,
        store_path: str | dict[str, str] | None = None,
        is_live: bool | None = False,
        is_fetch_info: bool | None = False,
        debug: bool = False,
        jitter: int = 0,
        db: Any | None = None,
        logger: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize DataFeedDownloader.

        Args:
            DataFeedClass: DataFeed class to instantiate
            codes: List of instrument codes to process
            asset_type: Asset type(s) for instruments
            exchange: Exchange(s) for instruments
            start_date: Start date(s) for data collection
            end_date: End date(s) for data collection
            period: Period specification(s)
            interval: Data interval(s)
            max_volume: Maximum volume filter(s)
            min_volume: Minimum volume filter(s)
            min_notional: Minimum notional value(s)
            tick_size: Tick size(s)
            step_size: Step size(s)
            order_ascending: Whether to order data chronologically
            store_path: Storage path(s)
            is_live: Whether this is live trading mode
            is_fetch_info: Whether to fetch instrument info
            debug: Enable debug logging
            jitter: Random time jitter in seconds
            db: Database instance
        """
        # authentication(bypass=kwargs.get("bypass", False), _from="datafeed")
        self.init_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4)

        self.codes = codes
        self.asset_type = asset_type
        self.exchange = exchange
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.interval = interval
        self.max_volume = max_volume
        self.min_volume = min_volume
        self.min_notional = min_notional
        self.tick_size = tick_size
        self.step_size = step_size
        self.order_ascending = order_ascending
        self.store_path = store_path
        self.is_live = is_live
        self.debug = debug
        self.jitter = jitter
        self.db = db
        self.logger = logger

        self.DataFeedClass = DataFeedClass
        self.is_fetch_info = is_fetch_info

    def _get_param_for_code(self, param: Any, code: str) -> Any:
        """Get parameter value for specific code.

        Args:
            param: Parameter value (can be dict or single value)
            code: Code to get parameter for

        Returns:
            Parameter value for the specified code
        """
        return param[code] if isinstance(param, dict) and code in param else param

    def _instantiate_datafeed(self, code: str) -> DataFeed:
        """Instantiate DataFeed for a specific code.

        Args:
            code: Instrument code

        Returns:
            DataFeed instance
        """
        self.logger.info(f"Start instantiating {self.DataFeedClass} for {code}")

        params = {
            "code": code,
            "asset_type": self._get_param_for_code(self.asset_type, code),
            "exchange": self._get_param_for_code(self.exchange, code),
            "start_date": self._get_param_for_code(self.start_date, code),
            "end_date": self._get_param_for_code(self.end_date, code),
            "period": self._get_param_for_code(self.period, code),
            "interval": self._get_param_for_code(self.interval, code),
            "max_volume": self._get_param_for_code(self.max_volume, code),
            "min_volume": self._get_param_for_code(self.min_volume, code),
            "min_notional": self._get_param_for_code(self.min_notional, code),
            "tick_size": self._get_param_for_code(self.tick_size, code),
            "step_size": self.step_size,
            "store_path": self._get_param_for_code(self.store_path, code),
            "order_ascending": self.order_ascending,
            "is_live": self.is_live,
            "debug": self.debug,
            "jitter": self.jitter,
            "db": self.db,
        }

        return self.DataFeedClass(**params)

    def start(self) -> list[DataFeed]:
        """Start downloading data for all codes.

        Returns:
            List of DataFeed instances
        """
        futures: dict[str, Future] = {
            code: self.init_executor.submit(self._instantiate_datafeed, code) for code in self.codes
        }

        feeds = []
        for future in as_completed(list(futures.values())):
            df = future.result()
            self.logger.info(f"Finish instantiating {df} for {df.code}")
            feeds.append(df)

        if self.is_fetch_info:
            info_futures = [self.init_executor.submit(df.fetch_info) for df in feeds]
            wait(info_futures, return_when=ALL_COMPLETED)

        return feeds
