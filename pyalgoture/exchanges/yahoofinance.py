# from dateutil.relativedelta import relativedelta
# from dateutil.parser import parse
# from collections import defaultdict
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from ..utils.client_rest import RestClient

YAHOO_REST_HOST = "https://query1.finance.yahoo.com"


class YahooClient(RestClient):
    def __init__(
        self,
        url_base: str = YAHOO_REST_HOST,
        proxy_host: str = "",
        proxy_port: int = 0,
        actions: bool = True,
    ) -> None:
        super().__init__(url_base, proxy_host, proxy_port)
        self.use_adjust = True
        self.back_adjust = False
        self.prepost = False
        self.actions = actions
        self.proxy = None
        self.rounding = False
        self.tz = None

    def connect(self) -> None:
        """connect exchange server"""

        self.start()

        print("Yahoo Finance REST API srart ")

    # def __init__(
    #     self,
    #     use_adjust: Optional[bool]=True,
    #     back_adjust: Optional[bool]=False,
    #     prepost: Optional[bool]=False,
    #     actions: Optional[bool]=True,
    #     proxy: Optional[str]=None,
    #     rounding: Optional[bool]=False,
    #     tz: Optional[pytz.timezone]=None
    # ):

    #     self.use_adjust = use_adjust
    #     self.back_adjust = back_adjust
    #     self.prepost = prepost
    #     self.actions = actions
    #     self.proxy = proxy
    #     self.rounding = rounding
    #     self.tz = tz

    def query_symbols(self, ticker: str = None, store_path: str = None) -> dict:
        LOTSIZE_URL = "https://webb-site.com/dbpub/mcap.asp"
        save_path = os.path.join(store_path, "securities.csv") if store_path else os.path.join(".", "securities.csv")
        if not os.path.exists(save_path):
            print("Fetch lot size data...")
            df = pd.read_html(LOTSIZE_URL)[0]
            df = df[["StockCode", "Issued shares", "Boardlot"]]
            df.rename(
                columns={
                    "StockCode": "ticker",
                    "Issued shares": "issued_shares",
                    "Boardlot": "board_lot",
                },
                inplace=True,
            )
            df["ticker"] = df["ticker"].astype(str).str.zfill(4) + ".HK"
            df.set_index(["ticker"], inplace=True)

            df.to_csv(save_path, index=True)
        else:
            df = pd.read_csv(save_path, index_col=0)

        return df.loc[df.index == ticker, "board_lot"].get(ticker, 1)

    def query_history(self, ticker: str, start: datetime, end: datetime, interval: str) -> list:
        # start=start - timedelta(days=3)
        # end=end + timedelta(days=3)
        try:
            start_time = int(time.mktime(start.timetuple()))
            end_time = int(time.mktime(end.timetuple()))
            params = {"period1": start_time, "period2": end_time}

            params["interval"] = interval.lower()
            params["includePrePost"] = self.prepost
            params["events"] = "div,splits"

            # 1) fix weired bug with Yahoo! - returning 60m for 30m bars
            if params["interval"] == "30m":
                params["interval"] = "15m"

            # Getting data from json
            resp = self.request(
                method="GET",
                path=f"/v8/finance/chart/{ticker}",
                params=params,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
                },
            )
            # print(resp.status_code,'=======', len(resp.data))
            if resp.status_code // 100 == 2:
                if "Will be right back" in resp.text:
                    raise RuntimeError(
                        "*** YAHOO FINANCE IS CURRENTLY DOWN! ***\nPlease contact our engineers to resolve the issue. Thank you for your patience."
                    )
                data = resp.data

                err_msg = "No data found for this date range, symbol may be delisted"
                if "chart" in data and data["chart"]["error"]:
                    err_msg = data["chart"]["error"]["description"]
                    print(f"- {ticker}: {err_msg}")
                    return self._empty_df()

                elif "chart" not in data or data["chart"]["result"] is None or not data["chart"]["result"]:
                    print(f"- {ticker}: {err_msg}")
                    return self._empty_df()

                # parse quotes
                try:
                    quotes = self.parse_quotes(data["chart"]["result"][0], self.tz)
                except Exception as e:
                    err_msg = e.__str__()
                    print(f"- {ticker}: {err_msg}")
                    return self._empty_df()

                # 2) fix weired bug with Yahoo! - returning 60m for 30m bars
                if interval.lower() == "30m":
                    quotes2 = quotes.resample("30T")
                    quotes = pd.DataFrame(
                        index=quotes2.last().index,
                        data={
                            "open": quotes2["open"].first(),
                            "high": quotes2["high"].max(),
                            "low": quotes2["low"].min(),
                            "close": quotes2["close"].last(),
                            "adj_close": quotes2["adj_close"].last(),
                            "volume": quotes2["volume"].sum(),
                        },
                    )
                    try:
                        quotes["dividends"] = quotes2["dividends"].max()
                    except Exception:
                        pass
                    try:
                        quotes["splits"] = quotes2["dividends"].max()
                    except Exception:
                        pass

                if self.use_adjust:
                    quotes = self._auto_adjust(quotes)
                elif self.back_adjust:
                    quotes = self._back_adjust(quotes)

                if self.rounding:
                    quotes = np.round(quotes, data["chart"]["result"][0]["meta"]["priceHint"])
                quotes["volume"] = quotes["volume"].fillna(0).astype(np.int64)

                quotes.dropna(inplace=True)
                # lot size
                # quotes['lot_size'] = self.fetch_lotsize(ticker)

                # actions
                dividends, splits = self.parse_actions(data["chart"]["result"][0], self.tz)

                # combine
                df = pd.concat([quotes, dividends, splits], axis=1, sort=True)
                df["dividends"].fillna(0, inplace=True)
                df["splits"].fillna(0, inplace=True)
                # print(df,'??', data["chart"]["result"][0]["meta"]["exchangeTimezoneName"])
                # index eod/intraday
                df.index = pd.to_datetime(df.index)
                df.index = df.index.tz_localize("UTC").tz_convert(
                    data["chart"]["result"][0]["meta"]["exchangeTimezoneName"]
                )

                df.index.name = "datetime"
                # if params["interval"][-1] == "m":
                #     df.index.name = "datetime"
                # else:
                #     df.index = pd.to_datetime(df.index.date)
                #     if tz is not None:
                #         df.index = df.index.tz_localize(tz)
                #     df.index.name = "date"

                if not self.actions:
                    df.drop(columns=["dividends", "splits"], inplace=True)

                return {"error": False, "success": True, "data": df, "msg": ""}

            else:
                return {
                    "error": True,
                    "success": False,
                    "data": {"status_code": resp.status_code, "msg": resp.data["msg"]},
                    "msg": resp.data["msg"],
                }
        except Exception:
            import traceback

            traceback.print_exc()

    @staticmethod
    def _adj_prices(ts):
        """
        Back adjust prices relative to adj_close for dividends and splits
        """
        ts["open"] = ts["open"] * ts["adj_close"] / ts["close"]
        ts["high"] = ts["high"] * ts["adj_close"] / ts["close"]
        ts["low"] = ts["low"] * ts["adj_close"] / ts["close"]
        ts["close"] = ts["close"] * ts["adj_close"] / ts["close"]
        return ts

    @staticmethod
    def _empty_df(index=[]):
        empty = pd.DataFrame(
            index=index,
            data={
                "open": np.nan,
                "high": np.nan,
                "low": np.nan,
                "close": np.nan,
                "adj_close": np.nan,
                "volume": np.nan,
            },
        )
        empty.index.name = "date"
        return empty

    @staticmethod
    def _auto_adjust(data):
        df = data.copy()
        ratio = df["close"] / df["adj_close"]
        df["open"] = df["open"] / ratio
        df["high"] = df["high"] / ratio
        df["low"] = df["low"] / ratio
        # df["adj_open"] = df["open"] / ratio
        # df["adj_high"] = df["high"] / ratio
        # df["adj_low"] = df["low"] / ratio

        # df.drop(
        #     ["open", "high", "low", "close"],
        #     axis=1, inplace=True)

        # df.rename(columns={
        #     "adj_open": "open", "adj_high": "high",
        #     "adj_low": "low", "adj_close": "close"
        # }, inplace=True)

        # df = df[["open", "high", "low", "close", "volume"]]
        return df[["open", "high", "low", "close", "volume"]]

    @staticmethod
    def _back_adjust(data):
        """back-adjusted data to mimic true historical prices"""

        df = data.copy()
        ratio = df["adj_close"] / df["close"]
        df["open"] = df["open"] * ratio
        df["high"] = df["high"] * ratio
        df["low"] = df["low"] * ratio
        # df["adj_open"] = df["open"] * ratio
        # df["adj_high"] = df["high"] * ratio
        # df["adj_low"] = df["low"] * ratio

        # df.drop(
        #     ["open", "high", "low", "adj_close"],
        #     axis=1, inplace=True)

        # df.rename(columns={
        #     "adj_open": "open", "adj_high": "high",
        #     "adj_low": "low"
        # }, inplace=True)

        return df[["open", "high", "low", "close", "volume"]]

    @staticmethod
    def parse_quotes(data, tz=None):
        timestamps = data["timestamp"]
        # print(timestamps,'--------')
        ohlc = data["indicators"]["quote"][0]
        volumes = ohlc["volume"]
        opens = ohlc["open"]
        closes = ohlc["close"]
        lows = ohlc["low"]
        highs = ohlc["high"]

        adjclose = closes
        if "adjclose" in data["indicators"]:
            adjclose = data["indicators"]["adjclose"][0]["adjclose"]

        quotes = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "adj_close": adjclose,
                "volume": volumes,
            }
        )

        quotes.index = pd.to_datetime(timestamps, unit="s")
        quotes.sort_index(inplace=True)
        # print(quotes,'?')
        if tz is not None:
            quotes.index = quotes.index.tz_localize(tz)

        return quotes

    @staticmethod
    def parse_actions(data, tz=None):
        dividends = pd.DataFrame(columns=["dividends"])
        splits = pd.DataFrame(columns=["splits"])

        if "events" in data:
            if "dividends" in data["events"]:
                dividends = pd.DataFrame(data=list(data["events"]["dividends"].values()))
                dividends.set_index("date", inplace=True)
                dividends.index = pd.to_datetime(dividends.index, unit="s")
                dividends.sort_index(inplace=True)
                if tz is not None:
                    dividends.index = dividends.index.tz_localize(tz)

                dividends.columns = ["dividends"]

            if "splits" in data["events"]:
                splits = pd.DataFrame(data=list(data["events"]["splits"].values()))
                splits.set_index("date", inplace=True)
                splits.index = pd.to_datetime(splits.index, unit="s")
                splits.sort_index(inplace=True)
                if tz is not None:
                    splits.index = splits.index.tz_localize(tz)
                splits["splits"] = splits["numerator"] / splits["denominator"]
                splits = splits["splits"]

        return dividends, splits
