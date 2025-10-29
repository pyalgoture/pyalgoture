import traceback
from datetime import datetime
from typing import Any

import pandas as pd

from ..datafeed import DataFeed
from ..utils.logger import get_logger
from ..utils.models import normalize_name
from ..utils.objects import AssetType, BarData, Exchange
from ..utils.util_dt import tz_manager

Num = int | float


class CustomDataFeed(DataFeed):
    """
    A DataFeed implementation for custom CSV data or pandas DataFrames.

    Supports reading price data from CSV files or directly from pandas DataFrames
    with automatic data validation and formatting.
    """

    def __init__(
        self,
        code: str,
        p: str | pd.DataFrame,
        symbol: str | None = None,
        asset_type: str | AssetType | None = AssetType.SPOT,
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
        exchange: str | Exchange | None = None,
        jitter: int = 0,
        commission_rate: float | None = None,
        min_commission_rate: float | None = None,
        taker_commission_rate: float | None = None,
        maker_commission_rate: float | None = None,
        is_contract: bool | None = None,
        consider_price: bool | None = None,
        is_inverse: bool | None = None,
        logger: Any = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize CustomDataFeed for CSV files or pandas DataFrames.

        Requirements:
            - CSV format or pandas DataFrame
            - Date/datetime as index
            - Columns must include 'close', optionally 'open', 'high', 'low', 'volume'

        Args:
            code: Symbol code identifier
            p: Path to CSV file or pandas DataFrame containing the data
            symbol: Custom symbol name, defaults to normalized code
            asset_type: Type of asset (SPOT, FUTURE, etc.)
            start_date: Start date for data filtering
            end_date: End date for data filtering
            period: Time period specification
            interval: Data interval specification
            min_volume: Minimum volume filter
            max_volume: Maximum volume filter
            min_notional: Minimum notional filter
            tick_size: Minimum price movement
            step_size: Minimum quantity movement
            launch_date: Asset launch date
            delivery_date: Asset delivery date (for futures)
            order_ascending: Sort order for data
            exchange: Exchange identifier
            jitter: Time jitter in seconds
            commission_rate: Trading commission rate
            min_commission_rate: Minimum commission rate
            taker_commission_rate: Taker commission rate
            maker_commission_rate: Maker commission rate
            is_contract: Whether this is a contract
            consider_price: Whether to consider price in calculations
            is_inverse: Whether this is an inverse contract

        Raises:
            ValueError: If 'p' is not a string path or pandas DataFrame
        """
        if not isinstance(p, str | pd.DataFrame):
            raise ValueError("The 'p' parameter must be either a file path or pandas DataFrame.")

        self.p = p
        self.code = code

        # Parse date parameters
        self.start_date = self._parse_date(start_date)
        self.end_date = self._parse_date(end_date)
        self.period = period
        self.interval = interval

        # Set asset type
        self.asset_type = self._parse_asset_type(asset_type)
        self.symbol = symbol or normalize_name(code, self.asset_type.value)

        self.logger = logger or get_logger()

        # Set trading parameters
        self._set_trading_params(
            min_volume,
            max_volume,
            min_notional,
            tick_size,
            step_size,
            launch_date,
            delivery_date,
            order_ascending,
            jitter,
        )

        # Set commission parameters
        self._set_commission_params(
            commission_rate,
            min_commission_rate,
            taker_commission_rate,
            maker_commission_rate,
            is_contract,
            consider_price,
            is_inverse,
        )

        # Set exchange
        self.exchange = self._parse_exchange(exchange)
        self.aio_symbol = f"{self.symbol}|{self.exchange.value}"

        # Load and process data
        self.data: dict[Any, BarData] = self.fetch_data()
        self.is_trading = True

    def _parse_date(self, date_input: str | datetime | None) -> datetime | None:
        """Parse date input into datetime object."""
        if isinstance(date_input, str):
            from dateutil.parser import parse

            return parse(date_input)
        return date_input

    def _parse_asset_type(self, asset_type: str | AssetType | None) -> AssetType:
        """Parse and validate asset type."""
        if isinstance(asset_type, str) and asset_type.upper() in AssetType._value2member_map_:
            return AssetType(asset_type.upper())
        elif isinstance(asset_type, AssetType):
            return asset_type
        else:
            return AssetType.SPOT

    def _parse_exchange(self, exchange: str | Exchange | None) -> Exchange:
        """Parse exchange parameter."""
        if isinstance(exchange, str):
            return Exchange(exchange.upper())
        return exchange or Exchange.LOCAL

    def _set_trading_params(
        self,
        min_volume: Num | None,
        max_volume: Num | None,
        min_notional: Num | None,
        tick_size: Num | None,
        step_size: Num | None,
        launch_date: str | datetime | None,
        delivery_date: str | datetime | None,
        order_ascending: bool | None,
        jitter: int,
    ) -> None:
        """Set trading-related parameters."""
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.min_notional = min_notional
        self.tick_size = tick_size
        self.step_size = step_size
        self.launch_date = launch_date
        self.delivery_date = delivery_date
        self.order_ascending = order_ascending
        self.jitter = jitter

    def _set_commission_params(
        self,
        commission_rate: float | None,
        min_commission_rate: float | None,
        taker_commission_rate: float | None,
        maker_commission_rate: float | None,
        is_contract: bool | None,
        consider_price: bool | None,
        is_inverse: bool | None,
    ) -> None:
        """Set commission-related parameters."""
        self.commission_rate = commission_rate
        self.min_commission_rate = min_commission_rate
        self.taker_commission_rate = taker_commission_rate
        self.maker_commission_rate = maker_commission_rate
        self.is_contract = is_contract
        self.consider_price = consider_price
        self.is_inverse = is_inverse

        # Set default commission rates based on asset type
        self._set_default_commission_rates()

    def _set_default_commission_rates(self) -> None:
        """Set default commission rates based on asset type."""
        if self.asset_type == AssetType.SPOT and self.commission_rate is None:
            self.commission_rate = 0.001
        elif self.asset_type in {AssetType.FUTURE, AssetType.PERPETUAL, AssetType.DATED_FUTURE}:
            self._set_derivatives_commission_rates()
        elif self.asset_type in {
            AssetType.INVERSE_FUTURE,
            AssetType.INVERSE_PERPETUAL,
            AssetType.INVERSE_DATED_FUTURE,
        }:
            self._set_derivatives_commission_rates(is_inverse=True)

    def _set_derivatives_commission_rates(self, is_inverse: bool = False) -> None:
        """Set commission rates for derivatives contracts."""
        if all(rate is None for rate in [self.commission_rate, self.taker_commission_rate, self.maker_commission_rate]):
            self.maker_commission_rate = 0.0002
            self.taker_commission_rate = 0.00055
            self.consider_price = True
            self.is_contract = True
            if is_inverse:
                self.is_inverse = True

    def _load_dataframe(self) -> pd.DataFrame:
        """Load data from file or use provided DataFrame."""
        if isinstance(self.p, str):
            df = pd.read_csv(self.p, parse_dates=True, index_col=0)
        else:
            df = self.p.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                df.index = df.index.tz_localize(tz=tz_manager.tz)
        return df

    def _filter_dataframe_by_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame by start and end dates."""
        if self.start_date and not self.end_date:
            df = df.loc[pd.to_datetime(self.start_date) :]
        elif not self.start_date and self.end_date:
            df = df.loc[: pd.to_datetime(self.end_date)]
        elif self.start_date and self.end_date:
            df = df.loc[pd.to_datetime(self.start_date) : pd.to_datetime(self.end_date)]
        return df

    def _normalize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame timezone."""
        if df.index.tz:
            df.index = df.index.tz_convert(tz_manager.tz)
        else:
            df.index = df.index.tz_localize(tz_manager.tz)
        return df

    def _fill_missing_ohlc_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing OHLC columns using close price."""
        required_columns = ["open", "high", "low"]
        for col in required_columns:
            if col in df.columns and df[col].isna().all():
                if col == "open":
                    df[col] = df["close"].shift(1)
                else:
                    df[col] = df["close"]
        return df

    def _convert_to_bar_data(self, df: pd.DataFrame) -> dict[Any, BarData]:
        """Convert DataFrame to dictionary of BarData objects."""
        df["datetime"] = df.index
        data = {}

        for timestamp, row_data in df.to_dict("index").items():
            if self.jitter:
                jitter_delta = pd.Timedelta(seconds=self.jitter)
                row_data["datetime"] = row_data["datetime"] + jitter_delta
                timestamp = timestamp + jitter_delta

            row_data["exchange"] = self.exchange
            row_data["asset_type"] = self.asset_type
            data[timestamp] = BarData(**row_data)

        return data

    def fetch_data(self) -> dict[Any, BarData]:
        """
        Load and process data from CSV file or DataFrame.

        Returns:
            Dictionary with datetime keys and BarData values
        """
        try:
            # Load DataFrame
            df = self._load_dataframe()

            # Filter by dates
            df = self._filter_dataframe_by_dates(df)

            # Normalize timezone
            df = self._normalize_timezone(df)

            # Fill missing OHLC columns
            df = self._fill_missing_ohlc_columns(df)

            # Convert to BarData objects
            return self._convert_to_bar_data(df)

        except Exception as e:
            tb = e.__traceback__
            filename = tb.tb_frame.f_code.co_filename if tb else "unknown"
            line_number = tb.tb_lineno if tb else "unknown"
            print(f"[ERROR] Failed to retrieve custom data. Error: {e} in line {line_number} for file {filename}.")
            return {}

    def _download_data(self, symbol: Any, start_date: Any, end_date: Any) -> dict[Any, Any]:
        """
        Placeholder method for consistency with DataFeed interface.

        Returns:
            Empty dictionary as data is loaded in fetch_data()
        """
        return {}
