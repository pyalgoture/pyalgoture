from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from dateutil.parser import parse

from ..datafeed import DataFeed
from ..exchanges.yahoofinance import YahooClient
from ..utils.objects import AssetType, BarData, Exchange

Num = int | float


class StockDataFeed(DataFeed):
    """
    A DataFeed implementation for stock market data using Yahoo Finance.

    Provides methods to fetch historical and real-time stock data
    from Yahoo Finance API with automatic data formatting.
    """

    def __init__(
        self,
        code: str,
        asset_type: str | AssetType | None = AssetType.STOCK,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
        period: str | None = "",
        interval: str | None = "1d",
        min_volume: Num | None = None,
        min_notional: Num | None = None,
        tick_size: Num | None = None,
        step_size: Num | None = None,
        order_ascending: bool | None = True,
        store_path: str | None = None,
        debug: bool | None = False,
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
        Initialize StockDataFeed for Yahoo Finance data.

        Args:
            code: Stock symbol code
            asset_type: Type of asset (defaults to STOCK)
            start_date: Start date for data queries
            end_date: End date for data queries
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd')
                   If start_date and end_date are specified, period won't be effective
            interval: Data interval ('1m', '5m', '15m', '30m', '1h', '2h', '3h', '4h', '6h', '12h', '1d', '1M', '1y')
            min_volume: Minimum volume filter
            min_notional: Minimum notional filter
            tick_size: Minimum price movement
            step_size: Minimum quantity movement
            order_ascending: Sort order for data
            store_path: Path to store downloaded data files
            debug: Enable debug mode
            jitter: Time jitter in seconds
            commission_rate: Trading commission rate
            min_commission_rate: Minimum commission rate
            taker_commission_rate: Taker commission rate
            maker_commission_rate: Maker commission rate
            is_contract: Whether this is a contract
            consider_price: Whether to consider price in calculations
            is_inverse: Whether this is an inverse contract

        Notes:
            Exchange is automatically set to HKEX for stock data.
            Default commission rate for stocks is 0.1385% (0.001385).
        """
        exchange = Exchange.HKEX
        self.bc = YahooClient(actions=False)

        super().__init__(
            code=code,
            exchange=exchange,
            asset_type=AssetType.STOCK,
            start_date=start_date,
            end_date=end_date,
            period=period,
            interval=interval,
            min_volume=min_volume,
            min_notional=min_notional,
            tick_size=tick_size,
            step_size=step_size,
            order_ascending=order_ascending,
            store_path=store_path,
            debug=debug or False,
            jitter=jitter,
            commission_rate=commission_rate,
            min_commission_rate=min_commission_rate,
            taker_commission_rate=taker_commission_rate,
            maker_commission_rate=maker_commission_rate,
            is_contract=is_contract,
            consider_price=consider_price,
            is_inverse=is_inverse,
            logger=logger,
            **kwargs,
        )

        # Set default commission rate for stocks
        if asset_type == AssetType.STOCK and self.commission_rate is None:
            self.commission_rate = 0.001385

    def _prepare_data_for_bar_conversion(self, df: pd.DataFrame, code: str, interval: str) -> pd.DataFrame:
        """Prepare DataFrame for conversion to BarData objects."""
        df["datetime"] = df.index
        df["interval"] = interval
        df["exchange"] = self.exchange
        df["asset_type"] = AssetType.STOCK
        df["symbol"] = code
        df["code"] = code
        return df

    def _convert_dataframe_to_bar_data(self, df: pd.DataFrame, code: str, interval: str) -> dict[Any, BarData]:
        """Convert DataFrame to dictionary of BarData objects."""
        if df.empty:
            return {}

        prepared_df = self._prepare_data_for_bar_conversion(df, code, interval)
        prepared_df.sort_index(ascending=self.order_ascending, inplace=True)
        return {timestamp: BarData(**row_data) for timestamp, row_data in prepared_df.to_dict("index").items()}

    def _download_data(self, code: str, interval: str, start_date: str, end_date: str) -> dict[Any, BarData]:
        """
        Download historical stock data with buffer days.

        Args:
            code: Stock symbol code
            interval: Data interval
            start_date: Start date string
            end_date: End date string

        Returns:
            Dictionary with datetime keys and BarData values
        """
        # Add buffer days to ensure we get enough data
        sd = parse(start_date) - timedelta(days=3)
        ed = parse(end_date) + timedelta(days=3)

        try:
            response = self.bc.query_history(ticker=code, interval=interval, start=sd, end=ed)
            self.logger.info(f"[StockDataFeed - _download_data] data: {response}")

            if not response.get("error") and not response["data"].empty:
                return self._convert_dataframe_to_bar_data(response["data"], code, interval)

        except Exception as e:
            self.logger.exception(f"[ERROR] Failed to download data for {code}: {e}")

        return {}

    def fetch_hist(
        self,
        start: datetime | str,
        end: datetime | str,
        interval: str | None = None,
        code: str | None = None,
    ) -> dict[Any, BarData]:
        """
        Fetch historical stock data for the specified time range.

        Args:
            start: Start datetime or string
            end: End datetime or string
            interval: Data interval, defaults to self.interval
            code: Stock symbol code, defaults to self.code

        Returns:
            Dictionary with datetime keys and BarData values
        """
        code = code or self.code
        interval = interval or self.interval

        # Convert string dates to datetime objects
        start_dt = parse(start) if isinstance(start, str) else start
        end_dt = parse(end) if isinstance(end, str) else end

        try:
            response = self.bc.query_history(ticker=code, interval=interval, start=start_dt, end=end_dt)

            if not response.get("error") and not response["data"].empty:
                return self._convert_dataframe_to_bar_data(response["data"], code, interval)
            else:
                self.logger.error(f"Failed to fetch historical data: {response}")

        except Exception as e:
            self.logger.exception(f"Exception during historical data fetch for {code}: {e}")

        return {}

    def fetch_spot(self, code: str | None = None, interval: str | None = None) -> dict[Any, BarData]:
        """
        Fetch the most recent stock price data.

        Args:
            code: Stock symbol code, defaults to self.code
            interval: Data interval, defaults to self.interval

        Returns:
            Dictionary with datetime keys and BarData values
        """
        code = code or self.code
        interval = interval or self.interval

        try:
            response = self.bc.query_history(code, interval=interval)

            if not response.get("error") and not response["data"].empty:
                df = response["data"].set_index(["datetime"], drop=False)
                return self._convert_dataframe_to_bar_data(df, code, interval)
            else:
                self.logger.error(f"Failed to fetch spot data: {response}")

        except Exception as e:
            self.logger.exception(f"Exception during spot data fetch for {code}: {e}")

        return {}

    def fetch_info(self, code: str | None = None) -> dict[Any, Any]:
        """
        Fetch stock information.

        Args:
            code: Stock symbol code, defaults to self.code

        Returns:
            Empty dictionary (not implemented for stock data)

        Notes:
            This method is a placeholder for consistency with the DataFeed interface.
            Stock information fetching is not currently implemented.
        """
        return {}
