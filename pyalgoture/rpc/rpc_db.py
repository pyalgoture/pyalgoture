import os
from datetime import datetime

from peewee import (
    CharField,
    DatabaseProxy,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
)
from peewee import SqliteDatabase as PeeweeSqliteDatabase
from playhouse.shortcuts import model_to_dict

from ..utils.logger import get_logger

# path: str = str(get_file_path("database.db"))
# db: PeeweeSqliteDatabase = PeeweeSqliteDatabase(path)
db = DatabaseProxy()


class DbStrategyData(Model):
    """
    bot_start: first trade time
    bot_startup:  engine start running time
    last_recon: last reconciliation time
    """

    strategy: str = CharField(primary_key=True)
    # symbol: str = CharField()
    # exchange: str = CharField()

    bot_start: datetime = DateTimeField(null=True)
    bot_startup: datetime = DateTimeField()
    last_recon: datetime = DateTimeField(null=True)

    initial_balance: float = FloatField(null=True)

    class Meta:
        database: PeeweeSqliteDatabase = db


class DbTradeData(Model):
    strategy: str = ForeignKeyField(DbStrategyData, field="strategy", backref="trades", on_delete="CASCADE")

    ori_order_id: str = CharField(primary_key=True)
    order_id: str = CharField()

    symbol: str = CharField()
    code: str = CharField()
    asset_type: str = CharField()
    exchange: str = CharField()

    price: float = FloatField()
    quantity: float = FloatField()
    side: str = CharField()
    position_side: str = CharField()
    traded_at: datetime = DateTimeField()

    commission: float = FloatField(default=0.0)
    commission_asset: str = CharField(null=True)
    commission_rate: float = FloatField(default=0.0)
    is_open: bool = IntegerField(null=True)  # Use IntegerField to store bool
    is_maker: bool = IntegerField(null=True)  # Use IntegerField to store bool

    # custom params
    traded_at_ts: int = IntegerField(null=True)
    action: str = CharField(default="")
    # execution_type: str = CharField(default="TRADE")

    leverage: int = IntegerField(null=True)
    msg: str = CharField(default="")
    tag: str = CharField(default="")

    class Meta:
        database: PeeweeSqliteDatabase = db
        indexes: tuple = ((("strategy",), True),)


# class DbTradePnL(Model):

#     id: AutoField = AutoField()
#     strategy: str = CharField()

#     exchange: str = CharField()
#     symbol: str = CharField()
#     code: str = CharField()
#     asset_type: str = CharField()
#     position_side: str = CharField()
#     is_open: bool = IntegerField(default=1)  # Use IntegerField to store bool

#     commission_open_rate: float = FloatField(default=0.0)
#     commission_open_cost: float = FloatField(null=True, default=0.0)
#     commission_open_asset: str = CharField(null=True)
#     commission_close_rate: float = FloatField(default=0.0)
#     commission_close_cost: float = FloatField(null=True, default=0.0)
#     commission_close_asset: str = CharField(null=True)

#     open_price: float = FloatField(default=0.0)
#     open_trade_value: float = FloatField(default=0.0)

#     close_price: float = FloatField(null=True)
#     close_trade_value: float = FloatField(default=0.0)

#     realized_pnl: float = FloatField(default=0.0)
#     close_pnl: float = FloatField(null=True)
#     close_roi: float = FloatField(null=True)
#     size: float = FloatField(default=0.0)
#     size_requested: float = FloatField(default=0.0)

#     open_date: datetime = DateTimeField(null=True)
#     close_date: datetime = DateTimeField(null=True)

#     max_price: float = FloatField(null=True)
#     min_price: float = FloatField(null=True)

#     exit_reason: str = CharField(null=True)
#     enter_reason: str = CharField(null=True)

#     # Leverage trading params
#     leverage: float = FloatField(default=1.0)
#     is_short: bool = IntegerField(default=0)  # Use IntegerField to store bool
#     liquidation_price: float = FloatField(null=True)

#     # Margin Trading params
#     interest_rate: float = FloatField(default=0.0)

#     # Perpetual params
#     funding_fee_total: float = FloatField(null=True, default=0.0)

#     # trades: str = CharField(default="[]")  # store as json string
#     # funding_fees: str = CharField(default="[]")  # store as json string

#     class Meta:
#         database: PeeweeSqliteDatabase = db
#         indexes: tuple = ((("symbol", "exchange", "position_side"), True),)


class RpcDatabase:
    """RPC Database Operations Class"""

    try:
        default_path = os.path.expanduser("~/.pyalgoture")
        default_path = os.path.join(default_path, "pyalgoture.sqlite")
    except Exception:
        default_path = ""

    def __init__(self, base_path: str = None, logger=None):
        """Initialize database"""
        self.base_path = base_path if base_path else self.default_path
        print(f"[RpcDatabase] base_path:{self.base_path}")
        global db
        database = PeeweeSqliteDatabase(self.base_path)
        db.initialize(database)
        self.db: PeeweeSqliteDatabase = db
        db.connect()
        self.create_tables()
        self.logger = logger if logger else get_logger()
        self.logger.notice(f"RpcDatabase db path: {self.base_path}")

    def create_tables(self):
        """Create database tables"""
        self.db.create_tables(
            [
                DbStrategyData,
                DbTradeData,
                # DbTradePnL,
            ]
        )

    def save_strategy(self, stra_data: dict) -> bool:
        """Save stra data"""
        try:
            DbStrategyData.create(
                strategy=stra_data["strategy"],
                bot_start=stra_data.get("bot_start"),
                bot_startup=stra_data["bot_startup"],
            )
            return True
        except Exception as e:
            print(f"Failed to save stra data: {e}")
            return False

    def load_strategy(self, strategy: str) -> list:
        """Load stra data"""
        try:
            query = DbStrategyData.select().where(DbStrategyData.strategy == strategy)
            result = query.first()
            return model_to_dict(result) if result else {}
        except Exception as e:
            print(f"Failed to load stra data: {e}")
            return {}

    def load_strategies(self) -> list:
        """Load all stra data"""
        try:
            query = DbStrategyData.select()
            return [model_to_dict(row) for row in query]
        except Exception as e:
            print(f"Failed to load all stra data: {e}")
            return []

    def update_strategy(self, strategy: str, stra_data: dict) -> bool:
        """Update stra data"""
        try:
            stra = DbStrategyData.get_by_id(strategy)
            for key, value in stra_data.items():
                setattr(stra, key, value)
            stra.save()
            return True
        except Exception as e:
            print(f"Failed to update stra data: {e}")
            return False

    def delete_strategy(self, strategy: str) -> bool:
        """Delete stra data"""
        try:
            stra = DbStrategyData.get_by_id(strategy)
            stra.delete_instance()
            return True
        except Exception as e:
            print(f"Failed to delete stra data: {e}")
            return False

    def save_trade(self, trade_data: dict) -> bool:
        """Save trade data"""
        try:
            # Convert boolean values to integers
            is_open = trade_data.get("is_open")
            if is_open is not None:
                trade_data["is_open"] = 1 if is_open else 0

            is_maker = trade_data.get("is_maker")
            if is_maker is not None:
                trade_data["is_maker"] = 1 if is_maker else 0

            # Ensure all required fields exist
            required_fields = {field.name for field in DbTradeData._meta.fields.values()}
            filtered_data = {key: value for key, value in trade_data.items() if key in required_fields}

            filtered_data = {}
            for field in required_fields:
                if field not in trade_data:
                    print(f"Failed to save trade data: Missing required field {field}")
                    return False
                filtered_data[field] = trade_data[field]

            DbTradeData.create(**filtered_data)
            return True
        except Exception as e:
            print(f"Failed to save trade data: {e}")
            return False

    def load_trades(self, strategy: str, start: datetime = None) -> list:
        """Load trade data"""
        try:
            query = DbTradeData.select().where(DbTradeData.strategy == strategy)

            if start:
                # query = query.where(DbTradeData.traded_at >= start, DbTradeData.strategy == strategy)
                start_ts = int(datetime.timestamp(start) * 1000)
                query = query.where(
                    DbTradeData.traded_at_ts >= start_ts,
                    DbTradeData.strategy == strategy,
                )

            query = query.order_by(DbTradeData.traded_at)

            result = []
            for row in query:
                trade_dict = model_to_dict(row)
                # Convert integers back to boolean values
                if "is_open" in trade_dict:
                    trade_dict["is_open"] = bool(trade_dict["is_open"])
                if "is_maker" in trade_dict:
                    trade_dict["is_maker"] = bool(trade_dict["is_maker"])
                result.append(trade_dict)

            return result
        except Exception as e:
            print(f"Failed to load trade data: {e}")
            return []

    def delete_trade(self, trade_id: int) -> bool:
        """Delete trade data"""
        try:
            trade = DbTradeData.get_by_id(trade_id)
            trade.delete_instance()
            return True
        except Exception as e:
            print(f"Failed to delete trade data: {e}")
            return False

    # def save_tradepnl(self, tradepnl_data: dict) -> bool:
    #     """Save trade PnL data"""
    #     try:
    #         # Convert boolean values to integers
    #         is_open = tradepnl_data.get("is_open")
    #         if is_open is not None:
    #             tradepnl_data["is_open"] = 1 if is_open else 0

    #         is_short = tradepnl_data.get("is_short")
    #         if is_short is not None:
    #             tradepnl_data["is_short"] = 1 if is_short else 0

    #         # Convert lists to JSON strings
    #         # if "trades" in tradepnl_data and isinstance(tradepnl_data["trades"], list):
    #         #     tradepnl_data["trades"] = json.dumps(tradepnl_data["trades"])

    #         # if "funding_fees" in tradepnl_data and isinstance(tradepnl_data["funding_fees"], list):
    #         #     tradepnl_data["funding_fees"] = json.dumps(tradepnl_data["funding_fees"])

    #         # Ensure all required fields exist
    #         required_fields = [
    #             "symbol", "code", "asset_type", "exchange", "position_side"
    #         ]
    #         for field in required_fields:
    #             if field not in tradepnl_data:
    #                 print(f"Failed to save trade PnL data: Missing required field {field}")
    #                 return False

    #         TradePnL.create(**tradepnl_data)
    #         return True
    #     except Exception as e:
    #         print(f"Failed to save trade PnL data: {e}")
    #         return False

    # def load_tradepnls(self, symbol: str = None, exchange: str = None, position_side: str = None) -> list:
    #     """Load trade PnL data"""
    #     try:
    #         query = TradePnL.select()

    #         if symbol:
    #             query = query.where(TradePnL.symbol == symbol)
    #         if exchange:
    #             query = query.where(TradePnL.exchange == exchange)
    #         if position_side:
    #             query = query.where(TradePnL.position_side == position_side)

    #         query = query.order_by(TradePnL.open_date)

    #         result = []
    #         for row in query:
    #             tradepnl_dict = model_to_dict(row)

    #             # Convert integers back to boolean values
    #             if "is_open" in tradepnl_dict:
    #                 tradepnl_dict["is_open"] = bool(tradepnl_dict["is_open"])
    #             if "is_short" in tradepnl_dict:
    #                 tradepnl_dict["is_short"] = bool(tradepnl_dict["is_short"])

    #             # Convert JSON strings back to lists
    #             # if "trades" in tradepnl_dict and tradepnl_dict["trades"]:
    #             #     try:
    #             #         tradepnl_dict["trades"] = json.loads(tradepnl_dict["trades"])
    #             #     except:
    #             #         tradepnl_dict["trades"] = []

    #             # if "funding_fees" in tradepnl_dict and tradepnl_dict["funding_fees"]:
    #             #     try:
    #             #         tradepnl_dict["funding_fees"] = json.loads(tradepnl_dict["funding_fees"])
    #             #     except:
    #             #         tradepnl_dict["funding_fees"] = []

    #             result.append(tradepnl_dict)

    #         return result
    #     except Exception as e:
    #         print(f"Failed to load trade PnL data: {e}")
    #         return []

    # def delete_tradepnl(self, tradepnl_id: int) -> bool:
    #     """Delete trade PnL data"""
    #     try:
    #         tradepnl = TradePnL.get_by_id(tradepnl_id)
    #         tradepnl.delete_instance()
    #         return True
    #     except Exception as e:
    #         print(f"Failed to delete trade PnL data: {e}")
    #         return False
