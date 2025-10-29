from collections.abc import Callable
from datetime import datetime
from typing import Any

from dateutil.parser import parse

from ..datafeed import DataFeed
from ..utils.models import normalize_name
from ..utils.objects import AssetType, BarData, Exchange, TickData
from ..utils.util_connection import get_exchange_connection
from ..utils.util_dt import tz_manager

Num = int | float


class CryptoDataFeed(DataFeed):
    """
    A DataFeed implementation for cryptocurrency market data.

    Provides methods to fetch historical and real-time cryptocurrency data
    including price bars, ticks, funding rates, and symbol information.
    """

    def __init__(
        self,
        code: str,
        exchange: str | Exchange,
        asset_type: str | AssetType | None = AssetType.SPOT,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
        period: str | None = "",
        interval: str | None = "1d",
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
        debug: bool | None = False,
        jitter: int = 0,
        commission_rate: float | None = None,
        min_commission_rate: float | None = None,
        taker_commission_rate: float | None = None,
        maker_commission_rate: float | None = None,
        is_contract: bool | None = None,
        consider_price: bool | None = None,
        is_inverse: bool | None = None,
        session: bool = False,
        is_testnet: bool = False,
        logger: Any | None = None,
        **kwargs: Any,
    ) -> None:
        exchange = exchange if isinstance(exchange, Exchange) else Exchange(exchange.upper())

        self.bc, self.exchange, self.asset_type = get_exchange_connection(
            exchange=exchange,
            asset_type=asset_type,
            session=session,
            is_testnet=is_testnet,
        )
        symbol = normalize_name(code, self.asset_type.value)
        super().__init__(
            symbol=symbol,
            code=code,
            exchange=self.exchange,
            asset_type=self.asset_type,
            start_date=start_date,
            end_date=end_date,
            period=period,
            interval=interval,
            min_volume=min_volume,
            max_volume=max_volume,
            min_notional=min_notional,
            tick_size=tick_size,
            step_size=step_size,
            launch_date=launch_date,
            delivery_date=delivery_date,
            order_ascending=order_ascending,
            store_path=store_path,
            is_live=is_live,
            debug=debug,
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

        if self.asset_type in [AssetType.SPOT]:
            if self.commission_rate is None:
                self.commission_rate = 0.001

        elif self.asset_type in [
            AssetType.FUTURE,
            AssetType.PERPETUAL,
            AssetType.DATED_FUTURE,
        ]:
            if (
                self.commission_rate is None
                and self.taker_commission_rate is None
                and self.maker_commission_rate is None
            ):
                self.maker_commission_rate = 0.0002
                self.taker_commission_rate = 0.00055
                self.consider_price = True
                self.is_contract = True

        elif self.asset_type in [
            AssetType.INVERSE_FUTURE,
            AssetType.INVERSE_PERPETUAL,
            AssetType.INVERSE_DATED_FUTURE,
        ]:
            if (
                self.commission_rate is None
                and self.taker_commission_rate is None
                and self.maker_commission_rate is None
            ):
                self.maker_commission_rate = 0.0002
                self.taker_commission_rate = 0.00055
                self.consider_price = True
                self.is_contract = True
                self.is_inverse = True

        self.public_ws_client = None

    def _download_data(self, code: str, interval: str, start_date: str, end_date: str, tz: bool = False) -> dict:
        """
        Download historical price data from the exchange.

        Args:
            code: Symbol code
            interval: Time interval (e.g., '1d', '1h')
            start_date: Start date string
            end_date: End date string
            tz: Whether to convert to local timezone

        Returns:
            Dictionary with datetime keys and BarData values
        """
        if not start_date:
            return {}
        sd = parse(start_date)
        ed = parse(end_date)
        data = {}
        try:
            res = self.bc.query_history(symbol=code, interval=interval, start=sd, end=ed)
            if res.get("error"):
                self.logger.error(
                    f"Something went wrong for {code}[{self.asset_type}] from {self.exchange} when download data. Error:{res}. code:{code}, interval:{interval}, start_date:{start_date}, end_date:{end_date}"
                )
                return {}
            else:
                if res["data"]:
                    for d in res["data"]:
                        if tz:
                            d["datetime"] = d["datetime"].astimezone(tz_manager.tz)
                        d["exchange"] = Exchange(d["exchange"])
                        d["asset_type"] = AssetType(d["asset_type"])
                        data[d["datetime"]] = BarData(**d)
                    ### sorting
                    data = dict(sorted(data.items(), reverse=not self.order_ascending))
                    return data
        except Exception as e:
            self.logger.exception(
                f"Something went wrong for {code}[{self.asset_type}] from {self.exchange} when download data. Error:{str(e)}. code:{code}, interval:{interval}, start_date:{start_date}, end_date:{end_date}, data:{data}"
            )

        return data

    def fetch_hist(
        self,
        start: datetime | str,
        end: datetime | str,
        interval: str | None = None,
        code: str | None = None,
        is_store: bool = False,
        tz: bool = False,
    ) -> dict:
        """
        Fetch historical data for the specified time range.

        Args:
            start: Start datetime or string
            end: End datetime or string
            interval: Time interval, defaults to self.interval
            code: Symbol code, defaults to self.code
            is_store: Whether to use stored data fetch method
            tz: Whether to convert to local timezone

        Returns:
            Dictionary with datetime keys and BarData values
        """
        code = code or self.code
        interval = interval or self.interval
        start_date = start.strftime("%Y-%m-%d %H:%M:%S") if isinstance(start, datetime) else start
        end_date = end.strftime("%Y-%m-%d %H:%M:%S") if isinstance(end, datetime) else end

        if is_store:
            return self.fetch_data(
                code=code,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            return self._download_data(
                code=code,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                tz=tz,
            )

    def fetch_tick(self, code: str | None = None, tz: bool = False) -> TickData | dict:
        """
        Fetch current tick data for the symbol.

        Args:
            code: Symbol code, defaults to self.code
            tz: Whether to convert to local timezone

        Returns:
            TickData object or empty dict on error
        """
        code = code or self.code

        data = {}
        try:
            res = self.bc.query_prices(code)
            if not res.get("error"):
                tick = res["data"]
                if tz:
                    tick["datetime"] = tick["datetime"].astimezone(tz_manager.tz)
                tick["exchange"] = Exchange(tick["exchange"])
                tick["asset_type"] = AssetType(tick["asset_type"])
                return TickData(**tick)
            else:
                self.logger.error(f"fetch tick data failed. code:{code}, data:{data}")
                if not self.fetch_info(code=code):
                    self.logger.error(f"fetch info data failed. code:{code}, data:{data}. Status mark as suspended.")
        except Exception as e:
            self.logger.exception(
                f"Something went wrong for {code}[{self.asset_type}] from {self.exchange} when fetching tick price. Error:{str(e)}."
            )
        return data

    def fetch_spot(
        self,
        code: str | None = None,
        interval: str | None = None,
        start: datetime | None = None,
        tz: bool = False,
    ) -> dict:
        """
        Fetch the most recent spot price data.

        Args:
            code: Symbol code, defaults to self.code
            interval: Time interval, defaults to self.interval
            start: Start time for query
            tz: Whether to convert to local timezone

        Returns:
            Dictionary with datetime keys and BarData values
        """
        code = code or self.code
        interval = interval or self.interval

        data = {}
        try:
            res = self.bc.query_history(code, interval=interval, start=start, limit=1)
            if not res.get("error") and res["data"]:
                for d in res["data"]:
                    if tz:
                        d["datetime"] = d["datetime"].astimezone(tz_manager.tz)
                    d["exchange"] = Exchange(d["exchange"])
                    d["asset_type"] = AssetType(d["asset_type"])
                    data[d["datetime"]] = BarData(**d)
                return data
            else:
                self.logger.error(f"fetch spot data failed. code:{code}, interval:{interval}, data:{data}")
        except Exception as e:
            self.logger.exception(
                f"Something went wrong for {code}(interval:{interval})[{self.asset_type}] from {self.exchange} when fetching spot. Error:{str(e)}."
            )
        return {}

    def fetch_funding_rate(
        self,
        code: str | None = None,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        limit: int = 200,
        tz: bool = False,
    ) -> dict:
        """
        Fetch historical funding rate data for perpetual contracts.

        Args:
            code: Symbol code, defaults to self.code
            start: Start time for the query
            end: End time for the query
            limit: Number of records to return (default 200)
            tz: Whether to convert timestamps to local timezone

        Returns:
            Dictionary with datetime keys and funding rate data values

        Notes:
            - Only works with perpetual contracts (linear, inverse asset types)
            - Automatically handles pagination for large datasets
            - Returns data sorted by timestamp (newest first)
        """
        code = code or self.code

        # Check if the underlying connection supports funding rate queries
        if not hasattr(self.bc, "query_funding_rate_history"):
            self.logger.error(f"Exchange {self.exchange} does not support funding rate history queries")
            return {}

        # Validate asset type for funding rates
        if self.asset_type not in [
            AssetType.PERPETUAL,
            AssetType.INVERSE_PERPETUAL,
            AssetType.FUTURE,
            AssetType.INVERSE_FUTURE,
            AssetType.DATED_FUTURE,
            AssetType.INVERSE_DATED_FUTURE,
        ]:
            self.logger.error(
                f"Funding rates are only available for perpetual and future contracts. Current asset type: {self.asset_type}"
            )
            return {}

        # Convert string dates to datetime objects if needed
        start_dt = parse(start) if isinstance(start, str) else start
        end_dt = parse(end) if isinstance(end, str) else end

        data = {}
        try:
            # Call the underlying exchange's funding rate method
            res = self.bc.query_funding_rate_history(symbol=code, start=start_dt, end=end_dt, limit=limit)

            if not res.get("error") and res.get("success"):
                if res["data"]:
                    for d in res["data"]:
                        # Apply timezone conversion if requested
                        if tz and d.get("datetime"):
                            d["datetime"] = d["datetime"].astimezone(tz_manager.tz)

                        # Ensure exchange and asset_type are proper enum objects
                        if "exchange" in d:
                            d["exchange"] = Exchange(d["exchange"])
                        if "asset_type" in d:
                            d["asset_type"] = AssetType(d["asset_type"])

                        # Use datetime as key for consistency with other fetch methods
                        if "datetime" in d:
                            data[d["datetime"]] = d

                    # Sort data according to order_ascending setting
                    data = dict(sorted(data.items(), reverse=not self.order_ascending))
                    return data
                else:
                    self.logger.error(
                        f"No funding rate data available for {code}[{self.asset_type}] from {self.exchange}"
                    )
            else:
                error_msg = res.get("msg", "Unknown error")
                self.logger.error(
                    f"Fetch funding rate failed for {code}[{self.asset_type}] from {self.exchange}. Error: {error_msg}"
                )

        except Exception as e:
            self.logger.exception(
                f"Something went wrong for {code}[{self.asset_type}] from {self.exchange} when fetching funding rates. Error:{str(e)}."
            )

        return {}

    def fetch_info(self, code: str | None = None) -> dict:
        """
        Fetch symbol information and update internal parameters.

        Args:
            code: Symbol code, defaults to self.code

        Returns:
            Dictionary containing symbol information
        """
        code = code or self.code

        try:
            data = self.bc.query_symbols(code)

            if not data.get("error"):
                symbol_data = data["data"]
                if isinstance(symbol_data, list):
                    symbol_data = symbol_data[0]
                if symbol_data:
                    param_setters = {
                        "min_volume": self.set_min_volume,
                        "max_volume": self.set_max_volume,
                        "min_notional": self.set_min_notional,
                        "tick_size": self.set_tick_size,
                        "step_size": self.set_step_size,
                    }

                    for param, setter in param_setters.items():
                        if symbol_data.get(param):
                            setter(symbol_data[param])

                    extra = symbol_data.get("extra", {})
                    if extra.get("launch_date"):
                        self.set_launch_date(extra["launch_date"])
                    if extra.get("delivery_date"):
                        self.set_delivery_date(extra["delivery_date"])
                else:
                    self.logger.error(
                        f"{code}[{self.asset_type}] from {self.exchange} fetch infomation failed. min_volume & min_notional are set as 0."
                    )

                return symbol_data
            else:
                self.set_status(is_trading=False)
                self.logger.error(
                    f"{code}[{self.asset_type}] from {self.exchange} fetch infomation failed. result:{data}"
                )
        except Exception as e:
            self.logger.exception(
                f"Something went wrong for {code}[{self.asset_type}] from {self.exchange} when fetching info. Error:{str(e)}."
            )
        return {}

    def connect(self) -> None:
        """Establish WebSocket connection for real-time data."""
        if self.public_ws_client:
            self.logger.info("public_ws_client already connected")
        else:
            self.public_ws_client = get_exchange_connection(
                exchange=self.exchange,
                asset_type=self.asset_type,
                return_connection_only=True,
                is_data_websocket=True,
                is_testnet=False,
            )
            self.public_ws_client.connect()
            self.logger.info("public_ws_client is connected")

    def subscribe(self, symbols: str | list, callback: Callable) -> None:
        """
        Subscribe to real-time tick data updates.

        Args:
            symbols: Symbol or list of symbols to subscribe to
            callback: Callback function to handle data updates
        """
        if not self.public_ws_client:
            self.connect()

        self.public_ws_client.subscribe(symbols=symbols, channel="depth", on_tick=callback)
        self.public_ws_client.subscribe(symbols=symbols, channel="ticker", on_tick=callback)

    def fetch_symbols(self, symbol: str | None = None) -> dict | list:
        """
        Fetch available symbols from the exchange.

        Args:
            symbol: Specific symbol to fetch, or None for all symbols

        Returns:
            Dictionary or list containing symbol data
        """
        try:
            data = self.bc.query_symbols(symbol=symbol)

            if not data.get("error"):
                symbol_data = data["data"]
                if not symbol_data:
                    self.logger.error(f"{self.asset_type} from {self.exchange} fetch symbols failed.")
                return symbol_data
            else:
                self.logger.error(f"{self.asset_type} from {self.exchange} fetch symbols failed. result:{data}")

        except Exception as e:
            self.logger.exception(
                f"Something went wrong for {self.asset_type} from {self.exchange} when fetching symbols. Error:{str(e)}."
            )
        return {}

    def close(self) -> None:
        """Close WebSocket connection."""
        if self.public_ws_client:
            self.public_ws_client.close()
