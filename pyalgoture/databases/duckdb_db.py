import os
from datetime import datetime
from typing import Any

import duckdb
import pandas as pd
from dateutil.parser import parse

from ..database import DB_TZ, BarOverview, BaseDatabase, TickOverview
from ..utils.objects import AssetType, BarData, Exchange, TickData
from ..utils.util_objects import AttrDict

CREATE_BAR_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS bar_data(
   "symbol" VARCHAR,
   "exchange" VARCHAR,
   "code" VARCHAR,
   "asset_type" VARCHAR,
   "datetime" TIMESTAMP,
   "interval" VARCHAR,

   "volume" FLOAT,
   "turnover" FLOAT,
   "open_interest" FLOAT,
   "open" FLOAT,
   "high" FLOAT,
   "low" FLOAT,
   "close" FLOAT,
   PRIMARY KEY (symbol, exchange, interval, datetime)
)
"""

CREATE_BAROVERVIEW_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS bar_overview
(
   "symbol" VARCHAR,
   "exchange" VARCHAR,
   "interval" VARCHAR,
   "count" INT,
   "start" TIMESTAMP,
   "end" TIMESTAMP,
   PRIMARY KEY (symbol, exchange, interval)
)
"""

SAVE_BAR_QUERY = "INSERT INTO bar_data SELECT * FROM df ON CONFLICT DO NOTHING"

SAVE_BAROVERVIEW_QUERY = """
INSERT INTO bar_overview VALUES
($symbol, $exchange, $interval, $count, $start, $end)
ON CONFLICT
DO UPDATE SET "count" = EXCLUDED.count, "start" = EXCLUDED.start, "end" = EXCLUDED.end
"""

LOAD_BAR_QUERY = """
SELECT * FROM bar_data
WHERE symbol = $symbol
AND exchange = $exchange
AND interval = $interval
AND datetime >= $start
AND datetime <= $end
ORDER BY datetime ASC
"""

LOAD_BAROVERVIEW_QUERY = """
SELECT * FROM bar_overview
WHERE symbol = $symbol
AND exchange = $exchange
AND interval = $interval
"""

COUNT_BAR_QUERY = """
SELECT COUNT(close) FROM bar_data
WHERE symbol = $symbol
AND exchange = $exchange
AND interval = $interval
"""

LOAD_ALL_BAROVERVIEW_QUERY = "SELECT * FROM bar_overview"

LOAD_ALL_BAROVERVIEW_SELECT_QUERY = """
SELECT * FROM bar_overview
WHERE symbol = $symbol
AND exchange = $exchange
AND interval = $interval
"""

DELETE_BAR_QUERY = """
DELETE FROM bar_data
WHERE symbol = $symbol
AND exchange = $exchange
AND interval = $interval
"""

DELETE_BAROVERVIEW_QUERY = """
DELETE FROM bar_overview
WHERE symbol = $symbol
AND exchange = $exchange
AND interval = $interval
"""


class DuckdbDatabase(BaseDatabase):
    """Database adpater for DuckDB"""

    try:
        default_path = os.path.expanduser("~/.pyalgoture")
        if not os.path.exists(default_path):
            os.mkdir(default_path)
        default_path = os.path.join(default_path, "pyalgoture.duck")
    except Exception:
        default_path = ""

    def __init__(self, base_path: str | None = None, read_only: bool = False) -> None:
        """"""
        self.db_path: str = base_path if base_path and base_path.endswith(".db") else self.default_path
        if not os.path.exists(self.db_path):
            read_only = False

        # Open database connection
        self.connection: duckdb.DuckDBPyConnection = duckdb.connect(
            database=self.db_path,
            read_only=read_only,
            # config={'access_mode': 'read_only'}
        )
        self.connection.execute("SET TimeZone = 'UTC';")

        self.cursor: Any = self.connection.cursor()

        # Create tables if necessary
        if not read_only:
            self.cursor.execute(CREATE_BAR_TABLE_QUERY)
            self.cursor.execute(CREATE_BAROVERVIEW_TABLE_QUERY)

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

            df: pd.DataFrame = pd.DataFrame.from_records(records)  # noqa

            self.connection.execute(SAVE_BAR_QUERY)

            # Query bars overview
            params: dict = {
                "symbol": symbol,
                "exchange": exchange,
                "interval": interval,
            }

            self.execute(LOAD_BAROVERVIEW_QUERY, params)
            row: tuple = self.cursor.fetchone()

            # New contract
            if not row:
                data: dict = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "interval": interval,
                    "start": records[0]["datetime"].replace(tzinfo=None),
                    "end": records[-1]["datetime"].replace(tzinfo=None),
                    "count": len(bars_list),
                }
            # Existing contract
            else:
                self.execute(COUNT_BAR_QUERY, params)
                count = self.cursor.fetchone()[0]

                existing_data: dict = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "interval": interval,
                    "start": min(records[0]["datetime"].replace(tzinfo=None), row[4]),
                    "end": max(records[-1]["datetime"].replace(tzinfo=None), row[5]),
                    "count": count,
                }
            overview_data = data if "data" in locals() else existing_data
            print(f"[save_bar_data] overview data:{overview_data}")

            self.execute(SAVE_BAROVERVIEW_QUERY, overview_data)

            return True
        except Exception as e:
            tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
            tb_filename = e.__traceback__.tb_frame.f_code.co_filename if e.__traceback__ else "unknown"
            print(
                f"[ERROR] Something went wrong on DuckdbDatabase - save_bar_data. Error: {e.__str__()} in line {tb_lineno} for file {tb_filename}. symbol:{symbol}; exchange: {exchange}; interval:{interval}"
            )
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
        params = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "start": str(start),
            "end": str(end),
        }
        self.execute(LOAD_BAR_QUERY, params)
        data: list[tuple] = self.cursor.fetchall()
        bars: dict[datetime, BarData] = {}

        for row in data:
            dt = DB_TZ.localize(row[4]).astimezone(tz)
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
        params: dict = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
        }

        # Query data count
        self.execute(COUNT_BAR_QUERY, params)
        count = self.cursor.fetchone()[0]

        # Remove bars
        self.execute(DELETE_BAR_QUERY, params)

        # Remove bar overview
        self.cursor.execute(DELETE_BAROVERVIEW_QUERY, params)

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
            params: dict = {
                "symbol": symbol,
                "exchange": exchange,
                "interval": interval,
            }
            self.execute(LOAD_ALL_BAROVERVIEW_SELECT_QUERY, params)
        else:
            self.execute(LOAD_ALL_BAROVERVIEW_QUERY)

        data: list[tuple] = self.cursor.fetchall()

        overviews: list[AttrDict] = []

        for row in data:
            overview = AttrDict(
                symbol=row[0],
                exchange=Exchange(row[1]),
                interval=str(row[2]),
                count=row[3],
                start=DB_TZ.localize(row[4]).astimezone(tz),
                end=DB_TZ.localize(row[5]).astimezone(tz),
            )
            overviews.append(overview)
        if symbol and exchange and interval:
            return overviews[0] if overviews else {}
        return overviews

    def get_tick_overview(self) -> list[AttrDict]:
        """Get tick overview"""
        return []

    def execute(self, query: str, data: object = None) -> None:
        """Execute SQL query"""
        self.cursor.execute(query, data)
        self.connection.commit()
