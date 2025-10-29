from datetime import datetime

from dateutil.parser import parse
from peewee import (
    AutoField,
    CharField,
    DatabaseProxy,
    DateTimeField,
    DoubleField,
    IntegerField,
    Model,
    ModelDelete,
    ModelSelect,
    chunked,
    fn,
)
from peewee import MySQLDatabase as PeeweeMySQLDatabase
from playhouse.shortcuts import ReconnectMixin

from ..database import (
    DB_TZ,
    AssetType,
    AttrDict,
    BarData,
    BaseDatabase,
    Exchange,
    TickData,
    convert_tz,
)


class ReconnectMySQLDatabase(ReconnectMixin, PeeweeMySQLDatabase):
    """MysqlDatabase with ReconnectMixin"""

    pass


db = DatabaseProxy()


class DateTimeMillisecondField(DateTimeField):
    """DateTimeField that support Millisecond"""

    def get_modifiers(self):
        return [3]


class DbBarData(Model):
    """Bar data model"""

    id: AutoField = AutoField()

    symbol: str = CharField()
    exchange: str = CharField()
    code: str = CharField()
    asset_type: str = CharField()
    datetime: datetime = DateTimeField()
    ts: float = DoubleField()
    interval: str = CharField()

    volume: float = DoubleField()
    turnover: float = DoubleField()
    open: float = DoubleField()
    high: float = DoubleField()
    low: float = DoubleField()
    close: float = DoubleField()
    open_interest: float = DoubleField(null=True)

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes: tuple = ((("symbol", "exchange", "interval", "datetime"), True),)


class DbTickData(Model):
    """Tick data model"""

    id: AutoField = AutoField()

    symbol: str = CharField()
    exchange: str = CharField()
    code: str = CharField()
    asset_type: str = CharField()
    datetime: datetime = DateTimeMillisecondField()
    ts: float = DoubleField()

    last_price: float = DoubleField()
    prev_volume_24h: float = DoubleField()
    prev_turnover_24h: float = DoubleField()
    prev_open_interest_24h: float = DoubleField(null=True)

    prev_open_24h: float = DoubleField(null=True)
    prev_high_24h: float = DoubleField(null=True)
    prev_low_24h: float = DoubleField(null=True)
    prev_close_24h: float = DoubleField(null=True)
    price_change_pct: float = DoubleField(null=True)
    price_change: float = DoubleField(null=True)

    bid_price_1: float = DoubleField()
    bid_price_2: float = DoubleField(null=True)
    bid_price_3: float = DoubleField(null=True)
    bid_price_4: float = DoubleField(null=True)
    bid_price_5: float = DoubleField(null=True)

    ask_price_1: float = DoubleField()
    ask_price_2: float = DoubleField(null=True)
    ask_price_3: float = DoubleField(null=True)
    ask_price_4: float = DoubleField(null=True)
    ask_price_5: float = DoubleField(null=True)

    bid_volume_1: float = DoubleField()
    bid_volume_2: float = DoubleField(null=True)
    bid_volume_3: float = DoubleField(null=True)
    bid_volume_4: float = DoubleField(null=True)
    bid_volume_5: float = DoubleField(null=True)

    ask_volume_1: float = DoubleField()
    ask_volume_2: float = DoubleField(null=True)
    ask_volume_3: float = DoubleField(null=True)
    ask_volume_4: float = DoubleField(null=True)
    ask_volume_5: float = DoubleField(null=True)

    # localtime: datetime = DateTimeMillisecondField(null=True)

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes: tuple = ((("symbol", "exchange", "datetime"), True),)


class DbBarOverview(Model):
    """Bar data overview model"""

    id: AutoField = AutoField()

    symbol: str = CharField()
    exchange: str = CharField()
    interval: str = CharField()
    count: int = IntegerField()
    start: datetime = DateTimeField()
    end: datetime = DateTimeField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes: tuple = ((("symbol", "exchange", "interval"), True),)


class DbTickOverview(Model):
    """Tick data overview model"""

    id: AutoField = AutoField()

    symbol: str = CharField()
    exchange: str = CharField()
    count: int = IntegerField()
    start: datetime = DateTimeField()
    end: datetime = DateTimeField()

    class Meta:
        database: PeeweeMySQLDatabase = db
        indexes: tuple = ((("symbol", "exchange"), True),)


class MysqlDatabase(BaseDatabase):
    """Mysql datebase connecter"""

    def __init__(self, database, host, port, user=None, password=None) -> None:
        """"""
        global db
        try:
            # Create a direct connection to MySQL server without using the proxy
            import pymysql

            conn = pymysql.connect(host=host, port=port, user=user, password=password)
            cursor = conn.cursor()

            # Create the database if it doesn't exist
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error creating database: {str(e)}")
            raise

        # Connect to the specified database
        database = ReconnectMySQLDatabase(database=database, user=user, password=password, host=host, port=port)
        db.initialize(database)

        self.db: PeeweeMySQLDatabase = db
        self.db.connect()
        self.db.create_tables([DbBarData, DbTickData, DbBarOverview, DbTickOverview])

    def save_bar_data(self, bars: dict[datetime, BarData], stream: bool = False) -> bool:
        """save bar data"""

        # read primary key params
        bars_list = [bar.to_dict(return_obj=False) for bar in bars.values()]
        bar_dict = bars_list[0]
        symbol: str = bar_dict["symbol"]
        exchange: str = bar_dict["exchange"]
        interval: str = bar_dict["interval"]

        # use upsert to update db data
        try:
            with self.db.atomic():
                for c in chunked(bars_list, 50):
                    DbBarData.insert_many(c).on_conflict_replace().execute()

            # update bar overview data
            overview: DbBarOverview = DbBarOverview.get_or_none(
                DbBarOverview.symbol == symbol,
                DbBarOverview.exchange == exchange,
                DbBarOverview.interval == interval,
            )
            if not overview:
                overview: DbBarOverview = DbBarOverview()
                overview.symbol = symbol
                overview.exchange = exchange
                overview.interval = interval

            if stream and overview.id:
                # In stream mode, update overview based on existing values plus new data
                overview.end = bars_list[-1]["datetime"]
                overview.count += len(bars_list)
            else:
                # In non-stream mode or new overview, query database for actual counts
                s: ModelSelect = (
                    DbBarData.select(
                        fn.MIN(DbBarData.datetime).alias("start"),
                        fn.MAX(DbBarData.datetime).alias("end"),
                        fn.COUNT(DbBarData.datetime).alias("count"),
                    )
                    .where(
                        (DbBarData.symbol == symbol)
                        & (DbBarData.exchange == exchange)
                        & (DbBarData.interval == interval)
                    )
                    .order_by(DbBarData.datetime.asc())
                )

                result = s.get()
                first_datetime = result.start
                last_datetime = result.end
                total_count = result.count
                overview.start = first_datetime
                overview.end = last_datetime
                overview.count = total_count

            if overview.start and overview.end:
                overview.save()

            return True
        except Exception as e:
            print(
                f"[ERROR] Something went wrong on MysqlDatabase - save_bar_data. Error: {e.__str__()} in line {e.__traceback__.tb_lineno} for file {e.__traceback__.tb_frame.f_code.co_filename}. symbol:{symbol}; exchange: {exchange}; interval:{interval}"
            )
            return False

    def save_tick_data(self, ticks: list[TickData], stream: bool = False) -> bool:
        """save tick data"""

        # read primary key params
        first_tick: TickData = ticks[0]
        symbol: str = first_tick.symbol
        exchange: str = first_tick.exchange

        data: list = []

        for tick in ticks:
            tick.datetime = convert_tz(tick.datetime)
            tick_dict = tick.to_dict(return_obj=False)

            # Map TickData fields to database field names
            db_tick_dict = {
                "symbol": tick_dict["symbol"],
                "exchange": tick_dict["exchange"],
                "code": tick_dict["code"],
                "asset_type": tick_dict["asset_type"],
                "datetime": tick_dict["datetime"],
                "ts": tick_dict["ts"],
                "last_price": tick_dict["last_price"],
                "prev_volume_24h": tick_dict["volume"],
                "prev_turnover_24h": tick_dict["turnover"],
                "prev_open_interest_24h": tick_dict["open_interest"],
                "prev_open_24h": tick_dict.get("prev_open_24h"),
                "prev_high_24h": tick_dict.get("prev_high_24h"),
                "prev_low_24h": tick_dict.get("prev_low_24h"),
                "prev_close_24h": tick_dict.get("prev_close_24h"),
                "price_change_pct": tick_dict.get("price_change_pct"),
                "price_change": tick_dict.get("price_change"),
                "bid_price_1": tick_dict["bid_price_1"],
                "bid_price_2": tick_dict["bid_price_2"],
                "bid_price_3": tick_dict["bid_price_3"],
                "bid_price_4": tick_dict["bid_price_4"],
                "bid_price_5": tick_dict["bid_price_5"],
                "ask_price_1": tick_dict["ask_price_1"],
                "ask_price_2": tick_dict["ask_price_2"],
                "ask_price_3": tick_dict["ask_price_3"],
                "ask_price_4": tick_dict["ask_price_4"],
                "ask_price_5": tick_dict["ask_price_5"],
                "bid_volume_1": tick_dict["bid_volume_1"],
                "bid_volume_2": tick_dict["bid_volume_2"],
                "bid_volume_3": tick_dict["bid_volume_3"],
                "bid_volume_4": tick_dict["bid_volume_4"],
                "bid_volume_5": tick_dict["bid_volume_5"],
                "ask_volume_1": tick_dict["ask_volume_1"],
                "ask_volume_2": tick_dict["ask_volume_2"],
                "ask_volume_3": tick_dict["ask_volume_3"],
                "ask_volume_4": tick_dict["ask_volume_4"],
                "ask_volume_5": tick_dict["ask_volume_5"],
            }
            data.append(db_tick_dict)

        with self.db.atomic():
            for c in chunked(data, 50):
                result = DbTickData.insert_many(c).on_conflict_ignore().execute()

        # update tick overview data
        overview: DbTickOverview = DbTickOverview.get_or_none(
            DbTickOverview.symbol == symbol,
            DbTickOverview.exchange == exchange,
        )

        if not overview:
            overview: DbTickOverview = DbTickOverview()
            overview.symbol = symbol
            overview.exchange = exchange
            overview.start = ticks[0].datetime
            overview.end = ticks[-1].datetime
            overview.count = len(ticks)
        elif stream:
            overview.end = ticks[-1].datetime
            overview.count += len(ticks)
        else:
            # Convert overview datetimes to ensure timezone compatibility
            s: ModelSelect = (
                DbTickData.select(
                    fn.MIN(DbTickData.datetime).alias("start"),
                    fn.MAX(DbTickData.datetime).alias("end"),
                    fn.COUNT(DbTickData.datetime).alias("count"),
                )
                .where((DbTickData.symbol == symbol) & (DbTickData.exchange == exchange))
                .order_by(DbTickData.datetime.asc())
            )

            result = s.get()
            first_datetime = result.start
            last_datetime = result.end
            total_count = result.count
            overview.start = first_datetime
            overview.end = last_datetime
            overview.count = total_count

        if overview.start and overview.end:
            overview.save()

        return True

    def load_bar_data(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start: str | datetime,
        end: str | datetime,
        tz=None,
    ) -> dict[datetime, BarData]:
        """"""
        tz = tz if tz else DB_TZ
        exchange = exchange.upper()
        if isinstance(start, str):
            start = parse(start)
        start = start.astimezone(DB_TZ).strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(end, str):
            end = parse(end)
        end = end.astimezone(DB_TZ).strftime("%Y-%m-%d %H:%M:%S")

        s: ModelSelect = (
            DbBarData.select()
            .where(
                (DbBarData.symbol == symbol)
                & (DbBarData.exchange == exchange)
                & (DbBarData.interval == interval)
                & (DbBarData.datetime >= start)
                & (DbBarData.datetime <= end)
            )
            .order_by(DbBarData.datetime)
        )
        bars: dict[datetime, BarData] = {}
        for db_bar in s.dicts():
            db_bar.pop("id", None)
            db_bar["datetime"] = DB_TZ.localize(db_bar["datetime"]).astimezone(tz)
            db_bar["exchange"] = Exchange(db_bar["exchange"])
            db_bar["asset_type"] = AssetType(db_bar["asset_type"])
            db_bar = BarData(**db_bar)
            bars[db_bar["datetime"]] = db_bar

        return bars

    def load_tick_data(
        self,
        symbol: str,
        exchange: str,
        start: str | datetime,
        end: str | datetime,
    ) -> list[TickData]:
        """read tick data"""
        s: ModelSelect = (
            DbTickData.select()
            .where(
                (DbTickData.symbol == symbol)
                & (DbTickData.exchange == exchange)
                & (DbTickData.datetime >= start)
                & (DbTickData.datetime <= end)
            )
            .order_by(DbTickData.datetime)
        )

        ticks: list[TickData] = []
        for db_tick in s.dicts():
            db_tick.pop("id", None)

            # Map database field names back to TickData field names
            tick_dict = {
                "symbol": db_tick["symbol"],
                "code": db_tick["code"],
                "exchange": Exchange(db_tick["exchange"]),
                "asset_type": AssetType(db_tick["asset_type"]),
                "datetime": db_tick["datetime"],
                "ts": db_tick["ts"],
                "last_price": db_tick["last_price"],
                "volume": db_tick["prev_volume_24h"],
                "turnover": db_tick["prev_turnover_24h"],
                "open_interest": db_tick["prev_open_interest_24h"],
                "prev_open_24h": db_tick.get("prev_open_24h"),
                "prev_high_24h": db_tick.get("prev_high_24h"),
                "prev_low_24h": db_tick.get("prev_low_24h"),
                "prev_close_24h": db_tick.get("prev_close_24h"),
                "price_change_pct": db_tick.get("price_change_pct"),
                "price_change": db_tick.get("price_change"),
                "bid_price_1": db_tick["bid_price_1"],
                "bid_price_2": db_tick["bid_price_2"],
                "bid_price_3": db_tick["bid_price_3"],
                "bid_price_4": db_tick["bid_price_4"],
                "bid_price_5": db_tick["bid_price_5"],
                "ask_price_1": db_tick["ask_price_1"],
                "ask_price_2": db_tick["ask_price_2"],
                "ask_price_3": db_tick["ask_price_3"],
                "ask_price_4": db_tick["ask_price_4"],
                "ask_price_5": db_tick["ask_price_5"],
                "bid_volume_1": db_tick["bid_volume_1"],
                "bid_volume_2": db_tick["bid_volume_2"],
                "bid_volume_3": db_tick["bid_volume_3"],
                "bid_volume_4": db_tick["bid_volume_4"],
                "bid_volume_5": db_tick["bid_volume_5"],
                "ask_volume_1": db_tick["ask_volume_1"],
                "ask_volume_2": db_tick["ask_volume_2"],
                "ask_volume_3": db_tick["ask_volume_3"],
                "ask_volume_4": db_tick["ask_volume_4"],
                "ask_volume_5": db_tick["ask_volume_5"],
            }

            tick = TickData(**tick_dict)
            ticks.append(tick)

        return ticks

    def delete_bar_data(self, symbol: str, exchange: str, interval: str) -> int:
        """delete bar data"""
        d: ModelDelete = DbBarData.delete().where(
            (DbBarData.symbol == symbol) & (DbBarData.exchange == exchange) & (DbBarData.interval == interval)
        )
        count: int = d.execute()

        # Delete bar overview data
        d2: ModelDelete = DbBarOverview.delete().where(
            (DbBarOverview.symbol == symbol)
            & (DbBarOverview.exchange == exchange)
            & (DbBarOverview.interval == interval)
        )
        d2.execute()
        return count

    def delete_tick_data(self, symbol: str, exchange: str) -> int:
        """delete tick data"""
        d: ModelDelete = DbTickData.delete().where((DbTickData.symbol == symbol) & (DbTickData.exchange == exchange))

        count: int = d.execute()

        # Delete tick overview data
        d2: ModelDelete = DbTickOverview.delete().where(
            (DbTickOverview.symbol == symbol) & (DbTickOverview.exchange == exchange)
        )
        d2.execute()
        return count

    def update_bar_overview(symbol: str, exchange: str, interval: str):
        overview: DbBarOverview = DbBarOverview.get_or_none(
            DbBarOverview.symbol == symbol,
            DbBarOverview.exchange == exchange,
            DbBarOverview.interval == interval,
        )

        if not overview:
            return False

        else:
            s: ModelSelect = (
                DbBarData.select(
                    fn.MIN(DbBarData.datetime).alias("start"),
                    fn.MAX(DbBarData.datetime).alias("end"),
                    fn.COUNT(DbBarData.datetime).alias("count"),
                )
                .where(
                    (DbBarData.symbol == symbol) & (DbBarData.exchange == exchange) & (DbBarData.interval == interval)
                )
                .order_by(DbBarData.datetime.asc())
            )

            result = s.get()

            first_datetime = result.start
            last_datetime = result.end
            total_count = result.count
            print(
                f"First datetime: {first_datetime}({type(first_datetime)}); last_datetime:{last_datetime}; count:{total_count}"
            )
            overview.start = str(first_datetime)
            overview.end = str(last_datetime)
            overview.count = total_count

            overview.save()
            return True

    def get_bar_overview(
        self, symbol: str | None = None, exchange: str | None = None, interval: str | None = None, tz=None
    ) -> list[AttrDict] | dict:
        """get bar overview data"""
        tz = tz if tz else DB_TZ
        if symbol and exchange and interval:
            overview: DbBarOverview = DbBarOverview.get_or_none(
                DbBarOverview.symbol == symbol,
                DbBarOverview.exchange == exchange,
                DbBarOverview.interval == interval,
            )
            if overview:
                return AttrDict(
                    {
                        "symbol": overview.symbol,
                        "exchange": overview.exchange,
                        "interval": overview.interval,
                        "count": overview.count,
                        "start": DB_TZ.localize(overview.start).astimezone(tz),
                        "end": DB_TZ.localize(overview.end).astimezone(tz),
                    }
                )
            return {}

        data_count: int = DbBarData.select().count()
        overview_count: int = DbBarOverview.select().count()
        if data_count and not overview_count:
            self.init_bar_overview()

        s: ModelSelect = DbBarOverview.select()
        overviews: list[AttrDict] = []
        for overview in s:
            overviews.append(
                AttrDict(
                    {
                        "symbol": overview.symbol,
                        "exchange": overview.exchange,
                        "interval": overview.interval,
                        "count": overview.count,
                        "start": DB_TZ.localize(overview.start).astimezone(tz),
                        "end": DB_TZ.localize(overview.end).astimezone(tz),
                    }
                )
            )
        return overviews

    def get_tick_overview(self, tz=None) -> list[AttrDict]:
        """get tick overview data"""
        tz = tz if tz else DB_TZ
        s: ModelSelect = DbTickOverview.select()
        overviews: list = []
        for overview in s:
            overviews.append(
                AttrDict(
                    {
                        "symbol": overview.symbol,
                        "exchange": overview.exchange,
                        "count": overview.count,
                        "start": DB_TZ.localize(overview.start).astimezone(tz),
                        "end": DB_TZ.localize(overview.end).astimezone(tz),
                    }
                )
            )
        return overviews

    def init_bar_overview(self) -> None:
        """initialize db overview data"""
        s: ModelSelect = DbBarData.select(
            DbBarData.symbol,
            DbBarData.exchange,
            DbBarData.interval,
            fn.COUNT(DbBarData.id).alias("count"),
        ).group_by(DbBarData.symbol, DbBarData.exchange, DbBarData.interval)

        for data in s:
            print(f"working on {data.symbol}  - {data.exchange} - {data.interval} - count:{data.count} .....")
            try:
                overview: DbBarOverview = DbBarOverview.get_or_none(
                    DbBarOverview.symbol == data.symbol,
                    DbBarOverview.exchange == data.exchange,
                    DbBarOverview.interval == data.interval,
                )
                if not overview:
                    overview: DbBarOverview = DbBarOverview()

                overview.symbol = data.symbol
                overview.exchange = data.exchange
                overview.interval = data.interval
                overview.count = data.count

                start_bar: DbBarData = (
                    DbBarData.select()
                    .where(
                        (DbBarData.symbol == data.symbol)
                        & (DbBarData.exchange == data.exchange)
                        & (DbBarData.interval == data.interval)
                    )
                    .order_by(DbBarData.datetime.asc())
                    .first()
                )
                # print(f"start_bar: {model_to_dict(start_bar)}")
                overview.start = start_bar.datetime

                end_bar: DbBarData = (
                    DbBarData.select()
                    .where(
                        (DbBarData.symbol == data.symbol)
                        & (DbBarData.exchange == data.exchange)
                        & (DbBarData.interval == data.interval)
                    )
                    .order_by(DbBarData.datetime.desc())
                    .first()
                )
                overview.end = end_bar.datetime

                overview.save()

                print(
                    f"Completed {data.symbol}  - {data.exchange} - {data.interval} - count:{data.count} | start:{start_bar.datetime}; end:{end_bar.datetime}....."
                )

            except Exception as e:
                print(
                    f"Something went wrong when store overview data for {data.symbol}  - {data.exchange} - {data.interval}. Error:{str(e)}"
                )
