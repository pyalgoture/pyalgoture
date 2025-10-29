import os
import sqlite3
from datetime import datetime
from typing import Any

import pandas as pd
from dateutil.parser import parse

from ..database import DB_TZ, BarOverview, BaseDatabase, TickOverview
from ..utils.objects import AssetType, BarData, Exchange, TickData
from ..utils.util_objects import AttrDict

CREATE_BAR_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS bar_data(
   symbol TEXT,
   exchange TEXT,
   code TEXT,
   asset_type TEXT,
   datetime TIMESTAMP,
   interval TEXT,

   volume REAL,
   turnover REAL,
   open_interest REAL,
   open REAL,
   high REAL,
   low REAL,
   close REAL,
   PRIMARY KEY (symbol, exchange, interval, datetime)
)
"""

CREATE_BAROVERVIEW_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS bar_overview
(
   symbol TEXT,
   exchange TEXT,
   interval TEXT,
   count INTEGER,
   start TIMESTAMP,
   end TIMESTAMP,
   PRIMARY KEY (symbol, exchange, interval)
)
"""

SAVE_BAR_QUERY = """
INSERT OR IGNORE INTO bar_data 
(symbol, exchange, code, asset_type, datetime, interval, volume, turnover, open_interest, open, high, low, close)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

SAVE_BAROVERVIEW_QUERY = """
INSERT OR REPLACE INTO bar_overview 
(symbol, exchange, interval, count, start, end)
VALUES (?, ?, ?, ?, ?, ?)
"""

LOAD_BAR_QUERY = """
SELECT * FROM bar_data
WHERE symbol = ?
AND exchange = ?
AND interval = ?
AND datetime >= ?
AND datetime <= ?
ORDER BY datetime ASC
"""

LOAD_BAROVERVIEW_QUERY = """
SELECT * FROM bar_overview
WHERE symbol = ?
AND exchange = ?
AND interval = ?
"""

COUNT_BAR_QUERY = """
SELECT COUNT(close) FROM bar_data
WHERE symbol = ?
AND exchange = ?
AND interval = ?
"""

LOAD_ALL_BAROVERVIEW_QUERY = "SELECT * FROM bar_overview"

LOAD_ALL_BAROVERVIEW_SELECT_QUERY = """
SELECT * FROM bar_overview
WHERE symbol = ?
AND exchange = ?
AND interval = ?
"""

DELETE_BAR_QUERY = """
DELETE FROM bar_data
WHERE symbol = ?
AND exchange = ?
AND interval = ?
"""

DELETE_BAROVERVIEW_QUERY = """
DELETE FROM bar_overview
WHERE symbol = ?
AND exchange = ?
AND interval = ?
"""


class SqliteDatabase(BaseDatabase):
    """Database adapter for SQLite"""

    try:
        default_path = os.path.expanduser("~/.pyalgoture")
        if not os.path.exists(default_path):
            os.mkdir(default_path)
        default_path = os.path.join(default_path, "pyalgoture.db")
    except Exception:
        default_path = ""

    def __init__(self, base_path: str | None = None, read_only: bool = False) -> None:
        """"""
        self.db_path: str = base_path if base_path and base_path.endswith(".db") else self.default_path
        if not os.path.exists(self.db_path):
            read_only = False

        # Open database connection
        uri_path = f"file:{self.db_path}?mode=ro" if read_only else self.db_path
        self.connection: sqlite3.Connection = sqlite3.connect(
            uri_path if read_only else self.db_path,
            uri=read_only,
            check_same_thread=False,
        )

        # Enable foreign keys and set timezone handling
        self.connection.execute("PRAGMA foreign_keys = ON")

        self.cursor: sqlite3.Cursor = self.connection.cursor()

        # Create tables if necessary
        if not read_only:
            self.cursor.execute(CREATE_BAR_TABLE_QUERY)
            self.cursor.execute(CREATE_BAROVERVIEW_TABLE_QUERY)
            self.connection.commit()

    def save_bar_data(self, bars: dict[datetime, BarData]) -> bool:
        """Save bar data"""
        # Save bars into db
        bars_list: list[BarData] = list(bars.values())

        first_bar: BarData = bars_list[0]
        symbol: str = first_bar.symbol
        exchange: str = first_bar.exchange.value
        interval: str = first_bar.interval
        try:
            records: list[dict] = []
            for bar in bars_list:
                record: dict = {
                    "symbol": bar.symbol,
                    "exchange": bar.exchange.value,
                    "code": bar.code,
                    "asset_type": bar.asset_type.value,
                    "datetime": bar.datetime,
                    "interval": bar.interval,
                    "volume": bar.volume,
                    "turnover": bar.turnover,
                    "open_interest": 0.0,  # bar.open_interest,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                }
                records.append(record)

            # Insert bars using executemany for better performance
            for record in records:
                self.cursor.execute(
                    SAVE_BAR_QUERY,
                    (
                        record["symbol"],
                        record["exchange"],
                        record["code"],
                        record["asset_type"],
                        record["datetime"].replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"),
                        record["interval"],
                        record["volume"],
                        record["turnover"],
                        record["open_interest"],
                        record["open"],
                        record["high"],
                        record["low"],
                        record["close"],
                    ),
                )

            # Query bars overview
            params: tuple = (symbol, exchange, interval)

            self.cursor.execute(LOAD_BAROVERVIEW_QUERY, params)
            row: tuple = self.cursor.fetchone()

            # New contract
            if not row:
                overview_data: tuple = (
                    symbol,
                    exchange,
                    interval,
                    len(bars_list),
                    records[0]["datetime"].replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"),
                    records[-1]["datetime"].replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"),
                )
            # Existing contract
            else:
                self.cursor.execute(COUNT_BAR_QUERY, params)
                count = self.cursor.fetchone()[0]

                start_dt = min(
                    records[0]["datetime"].replace(tzinfo=None),
                    datetime.strptime(row[4], "%Y-%m-%d %H:%M:%S"),
                )
                end_dt = max(
                    records[-1]["datetime"].replace(tzinfo=None),
                    datetime.strptime(row[5], "%Y-%m-%d %H:%M:%S"),
                )

                overview_data: tuple = (
                    symbol,
                    exchange,
                    interval,
                    count,
                    start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                )

            print(f"[save_bar_data] overview data:{overview_data}")

            self.cursor.execute(SAVE_BAROVERVIEW_QUERY, overview_data)
            self.connection.commit()

            return True
        except Exception as e:
            tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
            tb_filename = e.__traceback__.tb_frame.f_code.co_filename if e.__traceback__ else "unknown"
            print(
                f"[ERROR] Something went wrong on SqliteDatabase - save_bar_data. Error: {e.__str__()} in line {tb_lineno} for file {tb_filename}. symbol:{symbol}; exchange: {exchange}; interval:{interval}"
            )
            self.connection.rollback()
            return False

    def save_tick_data(self, ticks: list[TickData]) -> bool:
        """Save tick data"""
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
        """Load bar data"""
        # Load data from db
        tz = tz if tz else DB_TZ
        exchange = exchange.upper()
        if isinstance(start, str):
            start = parse(start)
        start = start.astimezone(DB_TZ).strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(end, str):
            end = parse(end)
        end = end.astimezone(DB_TZ).strftime("%Y-%m-%d %H:%M:%S")
        params = (symbol, exchange, interval, str(start), str(end))

        self.cursor.execute(LOAD_BAR_QUERY, params)
        data: list[tuple] = self.cursor.fetchall()
        bars: dict[datetime, BarData] = {}

        for row in data:
            dt = DB_TZ.localize(datetime.strptime(row[4], "%Y-%m-%d %H:%M:%S")).astimezone(tz)
            bar = {
                "symbol": symbol,
                "exchange": Exchange(exchange),
                "code": row[2],
                "asset_type": AssetType(row[3]),
                "datetime": dt,
                "interval": interval,
                "volume": row[6],
                "turnover": row[7],
                # open_interest:row[8],
                "open": row[9],
                "high": row[10],
                "low": row[11],
                "close": row[12],
            }
            bars[dt] = BarData(**bar)

        return bars

    def load_tick_data(self, symbol: str, exchange: str, start: datetime, end: datetime) -> list[TickData]:
        """Load tick data"""
        return []

    def delete_bar_data(self, symbol: str, exchange: str, interval: str) -> int:
        """Delete bar data"""
        params: tuple = (symbol, exchange, interval)

        # Query data count
        self.cursor.execute(COUNT_BAR_QUERY, params)
        count = self.cursor.fetchone()[0]

        # Remove bars
        self.cursor.execute(DELETE_BAR_QUERY, params)

        # Remove bar overview
        self.cursor.execute(DELETE_BAROVERVIEW_QUERY, params)

        self.connection.commit()

        return count

    def delete_tick_data(
        self,
        symbol: str,
        exchange: str,
    ) -> int:
        """Delete tick data"""
        return 0

    def get_bar_overview(
        self, symbol: str | None = None, exchange: str | None = None, interval: str | None = None, tz=None
    ) -> list[AttrDict] | AttrDict:
        """Get bar overview"""
        tz = tz if tz else DB_TZ
        if symbol and exchange and interval:
            params: tuple = (symbol, exchange, interval)
            self.cursor.execute(LOAD_ALL_BAROVERVIEW_SELECT_QUERY, params)
        else:
            self.cursor.execute(LOAD_ALL_BAROVERVIEW_QUERY)

        data: list[tuple] = self.cursor.fetchall()

        overviews: list[AttrDict] = []

        for row in data:
            overview = AttrDict(
                symbol=row[0],
                exchange=Exchange(row[1]),
                interval=str(row[2]),
                count=row[3],
                start=DB_TZ.localize(datetime.strptime(row[4], "%Y-%m-%d %H:%M:%S")).astimezone(tz),
                end=DB_TZ.localize(datetime.strptime(row[5], "%Y-%m-%d %H:%M:%S")).astimezone(tz),
            )
            overviews.append(overview)
        if symbol and exchange and interval:
            return overviews[0] if overviews else {}
        return overviews

    def get_tick_overview(self) -> list[AttrDict]:
        """Get tick overview"""
        return []

    def execute(self, query: str, data: tuple | None = None) -> None:
        """Execute SQL query"""
        if data:
            self.cursor.execute(query, data)
        else:
            self.cursor.execute(query)
        self.connection.commit()

    def close(self) -> None:
        """Close database connection"""
        self.connection.close()

    def __del__(self) -> None:
        """Destructor to ensure connection is closed"""
        try:
            self.connection.close()
        except Exception:
            pass
