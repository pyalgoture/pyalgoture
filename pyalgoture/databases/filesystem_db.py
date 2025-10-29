import json
import os
import pickle
from datetime import datetime
from typing import Any

import pandas as pd
from dateutil.parser import parse

from ..database import (
    DB_TZ,
    AssetType,
    AttrDict,
    BarData,
    BaseDatabase,
    Exchange,
    TickData,
)


class FilesystemDatabase(BaseDatabase):
    """Filesystem datebase connecter"""

    try:
        default_path = os.path.expanduser("~/.pyalgoture")
        default_path = os.path.join(default_path, "data")
        if not os.path.exists(default_path):
            os.mkdir(default_path)
    except Exception:
        default_path = ""

    def __init__(self, base_path: str | None = None) -> None:
        """"""
        self.base_path = base_path if base_path else self.default_path

        self.overview_path = os.path.join(self.base_path, "overview.json")
        if not os.path.exists(self.overview_path):
            with open(self.overview_path, "w") as f:
                json.dump({}, f)

    def _to_df(self, bars: list | dict | pd.DataFrame):
        if isinstance(bars, list):
            df = pd.DataFrame([data.to_dict(return_obj=False) for data in bars])

        elif isinstance(bars, dict):
            df = pd.DataFrame([data.to_dict(return_obj=False) for data in bars.values()])
        else:
            df = bars
        df["datetime"] = df["datetime"].astype(str)
        df["datetime"] = pd.to_datetime(df["datetime"])

        df.set_index("datetime", inplace=True)
        df.index = df.index.tz_convert(DB_TZ)
        return df

    def _to_dict(self, df):
        df["datetime"] = df.index
        data = df.to_dict("index")
        d = {}
        for k, v in data.items():
            v["exchange"] = Exchange(v["exchange"])
            v["asset_type"] = AssetType(v["asset_type"])
            d[k] = BarData(**v)
        return d

    def _get_key(self, symbol, exchange, interval):
        return f"{symbol.replace('/', '-')}_{exchange}_{interval}".lower()

    def _get_path(self, symbol, exchange, interval):
        """"""
        ext = "csv"
        key = f"{self._get_key(symbol=symbol, exchange=exchange, interval=interval)}.{ext}"
        path = os.path.join(self.base_path, key)
        return path

    def save_bar_data(self, bars: dict[datetime, BarData]) -> bool:
        """save bar data"""
        # read primary key params
        if bars:
            bar: BarData = bars[next(iter(bars))]
            symbol: str = bar.symbol
            exchange: str = bar.exchange.value
            interval: str = bar.interval
        else:
            print(f"[filesystemdv - save_bar_data] empty input bars:{bars}")
            return False
        try:
            path = self._get_path(symbol=symbol, exchange=exchange, interval=interval)
            df = self._to_df(bars)
            if os.path.exists(path):
                existing_df = self._to_df(pd.read_csv(path))

                df = pd.concat([existing_df, df])

                df = df[~df.index.duplicated()]

                df = df.sort_index()

            df.to_csv(path)

            # update bar overview data
            with open(self.overview_path) as f:
                overviews = json.load(f)
                key = self._get_key(symbol=symbol, exchange=exchange, interval=interval)
                data = overviews.get(key, {})
                data.update(
                    {
                        "symbol": symbol,
                        "exchange": exchange,
                        "interval": interval,
                        "count": len(df),
                        "start": df.index[0].strftime("%Y-%m-%d %H:%M:%S"),
                        "end": df.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                overviews[key] = data
                with open(self.overview_path, "w") as f:
                    json.dump(
                        overviews,
                        f,
                        indent=4,
                    )

            return True
        except Exception as e:
            tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
            tb_filename = e.__traceback__.tb_frame.f_code.co_filename if e.__traceback__ else "unknown"
            print(
                f"[ERROR] Something went wrong on FilesystemDatabase - save_bar_data. Error: {e.__str__()} in line {tb_lineno} for file {tb_filename}. symbol:{symbol}; exchange: {exchange}; interval:{interval}"
            )
            return False

    def save_tick_data(self, ticks: list[TickData]) -> bool:
        """Save tick data - not implemented for filesystem database"""
        return False

    def load_bar_data(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start: str | datetime,
        end: str | datetime,
        tz=None,
    ) -> dict[datetime, BarData]:
        """read bar data"""
        try:
            path = self._get_path(symbol=symbol, exchange=exchange, interval=interval)
            if isinstance(start, str):
                start = parse(start)
            start = start.astimezone(DB_TZ).strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(end, str):
                end = parse(end)
            end = end.astimezone(DB_TZ).strftime("%Y-%m-%d %H:%M:%S")

            bars: dict[datetime, BarData] = {}

            if os.path.exists(path):
                existing_df = self._to_df(pd.read_csv(path))

                existing_df = existing_df.loc[start:end]  # type: ignore

                bars = self._to_dict(existing_df)

            return bars

        except Exception as e:
            tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
            tb_filename = e.__traceback__.tb_frame.f_code.co_filename if e.__traceback__ else "unknown"
            print(
                f"[ERROR] Something went wrong on FilesystemDatabase - load_bar_data. Error: {e.__str__()} in line {tb_lineno} for file {tb_filename}. symbol:{symbol}; exchange: {exchange}; interval:{interval}; start:{start}; end:{end}"
            )
            return {}

    def load_tick_data(self, symbol: str, exchange: str, start: datetime, end: datetime) -> list[TickData]:
        """Load tick data - not implemented for filesystem database"""
        return []

    def delete_bar_data(self, symbol: str, exchange: str, interval: str) -> int:
        """Delete bar data"""
        try:
            path = self._get_path(symbol=symbol, exchange=exchange, interval=interval)
            count = 0

            # Count existing records if file exists
            if os.path.exists(path):
                df = pd.read_csv(path)
                count = len(df)
                # Remove the CSV file
                os.remove(path)

            # Remove from overview
            with open(self.overview_path) as f:
                overviews = json.load(f)

            key = self._get_key(symbol=symbol, exchange=exchange, interval=interval)
            if key in overviews:
                del overviews[key]

                with open(self.overview_path, "w") as f:
                    json.dump(overviews, f, indent=4)

            return count

        except Exception as e:
            tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
            tb_filename = e.__traceback__.tb_frame.f_code.co_filename if e.__traceback__ else "unknown"
            print(
                f"[ERROR] Something went wrong on FilesystemDatabase - delete_bar_data. Error: {e.__str__()} in line {tb_lineno} for file {tb_filename}. symbol:{symbol}; exchange: {exchange}; interval:{interval}"
            )
            return 0

    def get_bar_overview(
        self,
        symbol: str | None = None,
        exchange: str | None = None,
        interval: str | None = None,
        tz=None,
    ) -> list[AttrDict] | AttrDict:
        """get bar overview data"""
        tz = tz if tz else DB_TZ
        try:
            # with self._lock:
            with open(self.overview_path) as f:
                overviews = json.load(f)
            if symbol and exchange and interval:
                overviews = overviews.get(
                    self._get_key(symbol=symbol, exchange=exchange, interval=interval),
                    {},
                )
                if overviews:
                    overviews["start"] = DB_TZ.localize(parse(overviews["start"])).astimezone(tz)
                    overviews["end"] = DB_TZ.localize(parse(overviews["end"])).astimezone(tz)
                return AttrDict(overviews)

            # For list results, convert each dict to AttrDict
            result_list = []
            for k, v in overviews.items():
                v["start"] = DB_TZ.localize(parse(v["start"])).astimezone(tz)
                v["end"] = DB_TZ.localize(parse(v["end"])).astimezone(tz)
                result_list.append(AttrDict(v))
            return result_list

        except Exception as e:
            tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
            tb_filename = e.__traceback__.tb_frame.f_code.co_filename if e.__traceback__ else "unknown"
            print(
                f"[ERROR] Something went wrong on FilesystemDatabase - get_bar_overview. Error: {e.__str__()} in line {tb_lineno} for file {tb_filename}. symbol:{symbol}; exchange: {exchange}; interval:{interval}; "
            )
            return []

    def get_tick_overview(self) -> list[AttrDict]:
        """get tick overview data"""
        return []
