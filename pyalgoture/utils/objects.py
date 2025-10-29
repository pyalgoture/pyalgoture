import os
from collections import OrderedDict
from collections.abc import Iterator
from datetime import datetime
from datetime import datetime as DateTime
from enum import Enum, unique
from math import floor, inf
from typing import Any

import pytz
from pydantic import Field
from pydantic.dataclasses import dataclass

# from dataclasses import dataclass, field  # , , replace, asdict
# from dataclasses import asdict


def prec_round(value: float, target: int) -> float:
    """Round a float value to specified decimal places with extra precision"""
    k = 1 / (10 ** (target + 1))
    res: float = round(value + k, target)
    return res


UTC_TZ = pytz.utc

SECRET_KEY = "areix-ai"
IS_JUPYTER_NOTEBOOK = "JPY_PARENT_PID" in os.environ


def to_timestamp(dt: DateTime) -> int:
    return int(dt.astimezone(UTC_TZ).timestamp() * 1000)


# NOTE: code, ticker, symbol, security (equity / derivative)


class AttrDict(dict):
    def __getattr__(self, k: str) -> Any:
        # return self[k]
        return self.get(k)

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v


"""
##############
RPC
##############
"""


class State(Enum):
    """
    Bot application states
    """

    NEW = 0
    RUNNING = 1
    STOPPED = 2
    RELOAD_CONFIG = 3

    def __str__(self) -> str:
        return f"{self.name.lower()}"


# Enum for parsing requests from ws consumers
class RPCRequestType(str, Enum):
    SUBSCRIBE = "subscribe"

    WHITELIST = "whitelist"
    ANALYZED_DF = "analyzed_df"

    def __str__(self) -> str:
        return self.value


class RPCMessageType(str, Enum):
    STATUS = "status"
    WARNING = "warning"
    EXCEPTION = "exception"
    STARTUP = "startup"

    ENTRY = "entry"
    ENTRY_FILL = "entry_fill"
    ENTRY_CANCEL = "entry_cancel"

    EXIT = "exit"
    EXIT_FILL = "exit_fill"
    EXIT_CANCEL = "exit_cancel"

    PROTECTION_TRIGGER = "protection_trigger"
    PROTECTION_TRIGGER_GLOBAL = "protection_trigger_global"

    STRATEGY_MSG = "strategy_msg"

    WHITELIST = "whitelist"
    ANALYZED_DF = "analyzed_df"
    NEW_CANDLE = "new_candle"

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return self.value


NO_ECHO_MESSAGES = (
    RPCMessageType.ANALYZED_DF,
    RPCMessageType.WHITELIST,
    RPCMessageType.NEW_CANDLE,
)


"""
##############
Trading
##############
"""


@unique
class Status(Enum):
    """
    Order status.
    """

    NEW = "NEW"
    FILLING = "FILLING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"

    AMENDED = "AMENDED"
    AMENDING = "AMENDING"

    CANCELING = "CANCELING"
    CANCELED = "CANCELED"

    FAILED = "FAILED"  # REJECTED, EXPIRED
    LIQUIDATED = "LIQUIDATED"

    UNRECOGNIZED = "UNRECOGNIZED"


ACTIVE_STATUSES = set(
    [
        Status.NEW,
        Status.FILLING,
        Status.PARTIALLY_FILLED,
        Status.AMENDED,
        Status.AMENDING,
        Status.CANCELING,
    ]
)


@unique
class TimeInForce(Enum):
    """
    Time in force of order/trade/position.
    """

    GTC = "GTC"  # Good Til Canceled
    IOC = "IOC"  # Immediate Or Cancel (will try to fill the order as much as it can)
    FOK = "FOK"  # Fill or Kill (will expire if the full order cannot be filled)
    PostOnly = "PostOnly"


@unique
class Side(Enum):
    """
    Side of order/trade/position.
    """

    BUY = "BUY"
    SELL = "SELL"
    SHORTSELL = "SHORTSELL"
    BUYCOVER = "BUYCOVER"
    UNRECOGNIZED = "UNRECOGNIZED"


@unique
class PositionMode(Enum):
    """
    position mode
    """

    ONEWAY = "ONEWAY"
    HEDGE = "HEDGE"
    UNRECOGNIZED = "UNRECOGNIZED"


@unique
class MarginMode(Enum):
    """
    margin mode
    """

    CROSS = "CROSS"
    ISOLATED = "ISOLATED"
    UNRECOGNIZED = "UNRECOGNIZED"


@unique
class PositionSide(Enum):
    """
    Position side.
    For contracts only

    NET: can hold either a long or a short position of a contract.
    Long/Short: can hold both long and short positions simultaneously of a contract.
    """

    ### position_mode: SingleMerged / oneway
    # FUSE = "FUSE"
    # BOTH = "BOTH"
    NET = "NET"

    ### position_mode: Hedged
    LONG = "LONG"
    SHORT = "SHORT"

    UNRECOGNIZED = "UNRECOGNIZED"

    # in binance: Default BOTH for One-way Mode ; LONG or SHORT for Hedge Mode. It must be sent in Hedge Mode.
    # in bybit: MergedSingle: One-Way Mode; BothSide: Hedge Mode;  Position idx, used to identify positions in different position modes. Required if you are under One-Way Mode; 0-One-Way Mode; 1-Buy side of both side mode; 2-Sell side of both side mode
    # in okx: The default is net in the net mode; It is required in the long/short mode, and can only be long or short.; Only applicable to FUTURES/SWAP.


@unique
class OrderType(Enum):
    """
    Order type.
    """

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOPLIMIT = "STOPLIMIT"
    OCO = "OCO"
    HISTORY = "HISTORY"
    UNRECOGNIZED = "UNRECOGNIZED"
    # TODO: stop trailing


@unique
class ExecutionType(Enum):
    """
    Execution type.
    """

    TRADE = "TRADE"
    LIQUIDATION = "LIQUIDATION"
    FUNDING = "FUNDING"
    SETTLE = "SETTLE"  # for delisting or delivery


class AssetType(Enum):
    """
    Asset type.
    """

    # traditional finance native
    STOCK = "STOCK"
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    FORWARD = "FORWARD"
    INDEX = "INDEX"
    FOREX = "FOREX"
    ETF = "ETF"
    BOND = "BOND"
    WARRANT = "WARRANT"
    SPREAD = "SPREAD"
    FUND = "FUND"
    AMERICAN_OPTION = "AMERICAN_OPTION"
    EUROPEAN_OPTION = "EUROPEAN_OPTION"

    # crypto native
    SPOT = "SPOT"
    PERPETUAL = "PERPETUAL"
    DATED_FUTURE = "DATED_FUTURE"
    INVERSE_PERPETUAL = "INVERSE_PERPETUAL"
    INVERSE_DATED_FUTURE = "INVERSE_DATED_FUTURE"
    FUTURE = "FUTURE"
    INVERSE_FUTURE = "INVERSE_FUTURE"
    LIQUID_SWAP = "LIQUID_SWAP"
    EARN = "EARN"
    OPTION = "OPTION"
    SAVING = "SAVING"
    LINEAR = "LINEAR"
    INVERSE = "INVERSE"

    CUSTOM = "CUSTOM"

    PM = ""


class Exchange(Enum):
    """
    Exchange.
    """

    # CryptoCurrency
    BINANCE = "BINANCE"
    BYBIT = "BYBIT"
    EQONEX = "EQONEX"
    BINGX = "BINGX"
    BITGET = "BITGET"
    FTX = "FTX"
    OKX = "OKX"
    DYDXV3 = "DYDXV3"

    # LocalBacktesting
    LOCAL = "LOCAL"
    DUMMY = "DUMMY"

    # StockMarket
    IBKR = "IBKR"
    FUTU = "FUTU"
    YAHOO = "YAHOO"
    HKEX = "HKEX"

    PM = ""


class OptionType(Enum):
    """
    Option type.
    """

    CALL = "CALL"
    PUT = "PUT"


@dataclass
class BaseData:
    """
    Any data object should inherit base data.
    """

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        """
        For dict() function, when dict() is passed an iterable that produces key-value pairs, a new dictionary object with those key-value pairs is produced.
        """
        # for key, value in self.__dict__.items():
        #     yield (key, value)
        yield from self.__dict__.items()

    def update(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)

    def update_from_dict(self, item: dict[str, Any]) -> None:
        for k, v in item.items():
            self.update(k, v)

    def to_dict(self, display: list[str] = [], return_obj: bool = True, return_aio: bool = False) -> dict[str, Any]:
        if display:
            return {k: v for k, v in self.__dict__.items() if k in display}
        if not return_obj:
            return {
                k: v.value if isinstance(v, Enum) else v
                for k, v in self.__dict__.items()
                if not (not return_aio and k.startswith("aio_"))
            }

        return self.__dict__

    def __setitem__(self, k: str, v: Any) -> None:
        setattr(self, k, v)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def __delitem__(self, k: str) -> None:
        self[k] = None

    def get(self, k: str, default: Any = None) -> Any:
        return getattr(self, k, default)


@dataclass
class OrderData(BaseData):
    """
    Order data contains information for tracking lastest status
    of a specific order.
    """

    order_id: str
    ori_order_id: str | None
    exchange: Exchange
    code: str
    symbol: str
    asset_type: AssetType

    price: float | None
    quantity: float
    side: Side
    position_side: PositionSide
    executed_quantity: float
    created_at: DateTime
    updated_at: DateTime
    order_type: OrderType | None = None
    tag: str = ""
    msg: str = ""
    status: Status = Status.NEW
    time_in_force: TimeInForce = TimeInForce.GTC
    created_at_ts: int | None = None
    updated_at_ts: int | None = None

    # custom params
    requested_price: float | None = 0.0
    stop_price: float | None = 0.0
    trail_price: float | None = 0.0
    trail_percent: float | None = 0.0
    commission: float | None = 0.0
    commission_asset: str | None = None
    requested_quantity: float | None = 0.0
    is_open: bool | None = None

    def __post_init__(self) -> None:
        """"""
        self.requested_quantity = self.quantity
        self.aio_symbol: str = f"{self.symbol}|{self.exchange.value}"
        self.aio_position_id: str = f"{self.aio_symbol}|{self.position_side.value}"
        if self.created_at:
            self.created_at_ts = to_timestamp(self.created_at)
        if self.updated_at:
            self.updated_at_ts = to_timestamp(self.updated_at)

    def is_active(self) -> bool:
        """
        Check if the order is active.
        """
        if self.status in ACTIVE_STATUSES:
            return True
        else:
            return False


@dataclass
class TradeData(BaseData):
    """
    Trade data contains information for tracking lastest status
    of a specific order.
    """

    symbol: str
    code: str
    asset_type: AssetType
    exchange: Exchange

    ori_order_id: str
    order_id: str
    price: float
    quantity: float
    side: Side
    position_side: PositionSide  # NOTE: Some exchange does not return position_side, needs to be updated on callback
    traded_at: DateTime

    commission: float = 0.0
    commission_rate: float = 0.0
    commission_asset: str | None = None
    is_open: bool | None = None
    is_maker: bool | None = None

    # custom params
    traded_at_ts: int | None = None  # traded_at timestamp
    action: str = ""
    execution_type: ExecutionType = ExecutionType.TRADE

    gross_amount: float | None = 0.0
    # needs to be updated on callback
    pnl: float | None = 0.0
    leverage: float | None = 1.0
    duration: int | None = None  # in seconds

    msg: str = ""
    tag: str = ""

    def __post_init__(self) -> None:
        """"""
        self.aio_symbol: str = f"{self.symbol}|{self.exchange.value}"
        self.aio_position_id: str = f"{self.aio_symbol}|{self.position_side.value}"

        self.traded_at_ts = to_timestamp(self.traded_at)
        self._update_action()

        self.gross_amount = prec_round(self.quantity * self.price, 6)
        if self.gross_amount and self.gross_amount > 0:
            self.commission_rate = prec_round(self.commission / self.gross_amount, 5)

    def _update_action(self) -> None:
        if self.is_open is not None:
            self.action = (
                f"OPEN {'LONG' if self.side == Side.BUY else 'SHORT'}"
                if self.is_open
                else f"CLOSE {'SHORT' if self.side == Side.BUY else 'LONG'}"
            )


@dataclass
class AccountData(BaseData):
    """
    Account data contains information about balance, frozen and available.
    """

    asset: str
    exchange: Exchange

    balance: float = 0
    frozen: float = 0
    available: float = 0

    # actual_balance: float = 0
    # actual_frozen: float = 0
    # actual_available: float = 0

    capital: float = 0
    initial_capital: float = 0
    additional_capitals: dict = Field(default_factory=lambda: OrderedDict())

    def __post_init__(self) -> None:
        """"""
        self.initial_capital = self.balance
        self.capital = self.balance
        self.available: float = self.balance - self.frozen
        self.aio_account_id: str = f"{self.asset}|{self.exchange}"

    def add_balance(self, dt: DateTime | str, amount: float) -> None:
        self.available += amount
        self.balance += amount
        self.capital += amount
        if isinstance(dt, DateTime):
            dt = dt.strftime("%Y-%m-%d %H:%M:%S")
        self.additional_capitals[dt] = amount


@dataclass
class TradePnL(BaseData):
    """
    Trade PNL model.
    """

    # trades: list[TradeData] = Field(default_factory=list)
    trades: list[dict] = Field(default_factory=list)

    exchange: Exchange | None = None
    symbol: str = ""
    code: str = ""
    asset_type: AssetType | None = None
    position_side: PositionSide | None = None
    is_open: bool = True
    commission_open_rate: float = 0.0
    commission_open_cost: float | None = 0.0
    commission_open_asset: str | None = None
    commission_close_rate: float = 0.0
    commission_close_cost: float | None = 0.0
    commission_close_asset: str | None = None

    open_price: float = 0.0
    # open_price_requested: float | None = None
    open_trade_value: float = 0.0

    close_price: float | None = None
    # close_price_requested: float | None = None
    close_trade_value: float = 0.0

    realized_pnl: float = 0.0  # cumulative pnl(after fees)
    # unrealized_pnl: float = 0.0
    close_pnl: float | None = None  # close_pnl(after fees)
    close_roi: float | None = None  # close_pnl(before fees) / close_trade_value
    size: float = 0.0
    size_requested: float = 0.0

    open_date: DateTime | None = None
    close_date: DateTime | None = None

    max_price: float | None = -inf
    min_price: float | None = inf

    exit_reason: str | None = None
    enter_reason: str | None = None

    # Leverage trading params
    leverage: float = 1.0
    is_short: bool = False
    liquidation_price: float | None = None

    # Margin Trading params
    interest_rate: float = 0.0

    # Perpetual params
    funding_fees: list[dict] = Field(default_factory=list)
    funding_fee_total: float | None = 0.0

    _remain_commission_open_cost: float = 0.0
    _remain_funding_fee: float = 0.0

    def __repr__(self) -> str:
        return f"<TradePnL symbol:{self.symbol}, leverage: {self.leverage}, is_open:{self.is_open}, is_short:{self.is_short}, open_date:{self.open_date}, close_date:{self.close_date}, open_price:{self.open_price}, close_price:{self.close_price}, commission_open_cost:{self.commission_open_cost}, commission_close_cost:{self.commission_close_cost}, size:{self.size}, realized_pnl:{self.realized_pnl}, close_roi:{self.close_roi}, close_pnl:{self.close_pnl}, funding_fee_total:{self.funding_fee_total}, funding_fees:{self.funding_fees}, max_price:{self.max_price}, min_price:{self.min_price}>"

    def __post_init__(self) -> None:
        """"""
        self.aio_symbol: str = f"{self.symbol}|{self.exchange.value if self.exchange else ''}"
        self.aio_position_id: str = f"{self.aio_symbol}|{self.position_side.value if self.position_side else ''}"

    def update_price(self, price: float) -> None:
        """Update price metrics"""
        if self.max_price is not None:
            self.max_price = max(self.max_price, price)
        else:
            self.max_price = price
        if self.min_price is not None:
            self.min_price = min(self.min_price, price)
        else:
            self.min_price = price

    def update_funding_fee(self, price: float, funding_rate: float, funding_datetime: DateTime) -> None:
        """Update funding fee metrics"""
        funding_fee = price * self.size * funding_rate * (-1 if self.is_short else 1)
        self.funding_fees.append(
            {
                "price": price,
                "funding_rate": funding_rate,
                "funding_fee": funding_fee,
                "funding_datetime": funding_datetime,
            }
        )
        # print(f">>> funding_fee:{self.funding_fees[-1]}")
        if self.funding_fee_total is not None:
            self.funding_fee_total += funding_fee
        else:
            self.funding_fee_total = funding_fee
        self.realized_pnl -= funding_fee
        self._remain_funding_fee += funding_fee

    def update_liquidation_price(self, liq_price: float) -> None:
        """Update liquidation price"""
        self.liquidation_price = liq_price


@dataclass
class PositionData(BaseData):
    """
    Class for tracking position P&L and related metrics.

    Key metrics tracked:
    - Position: Current position size in the instrument
    - Market Value: Position size * current market price
    - Average Price: Total cost basis / total position size
    - P&L: Current profit/loss
    - Unrealized P&L: Paper profit/loss on open positions
    - Realized P&L: Actual profit/loss on closed positions
    - Net Investment: Total cost basis of current position

    Daily P&L calculation:
    PositionNow * PriceNow - positionAtResetTime * priceAtResetTime + NetAmountTraded

    Gross vs Net P&L:
    Buy trades:
    - Gross = quantity * price + fees
    - Net = quantity * price

    Sell trades:
    - Gross = quantity * price
    - Net = quantity * price - fees
    """

    # Input params
    code: str
    symbol: str
    asset_type: AssetType
    exchange: Exchange

    position_side: PositionSide
    size: float = 0.0
    side: PositionSide | None = None
    avg_open_price: float = 0.0

    opened_at: DateTime | None = None
    updated_at: DateTime | None = None
    opened_at_ts: int | None = None  # opened_at timestamp
    updated_at_ts: int | None = None  # updated_at timestamp

    position_margin: float = 0.0
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0

    # Custom params
    multiplier: float = 1
    margin_ratio: float = 0
    leverage: float = 1
    is_contract: bool | None = False
    consider_price: bool | None = True
    is_inverse: bool | None = False
    is_option: bool | None = False
    is_isolated: bool | None = True
    extra_maint_margin: float = 0
    maint_margin_rate: float = 0.5 / 100
    margin_mode: MarginMode | None = None

    # Commission and interest rates
    commission_rate: float | None = None
    maker_commission_rate: float | None = None
    taker_commission_rate: float | None = None
    min_commission: float | None = None
    commission_asset: str | None = None
    interest_rate: float | None = None
    use_trade_days: bool = False

    # Position tracking
    net_investment: float = 0.0
    total_turnover: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    market_value: float = 0.0
    notional_value: float = 0.0
    total_commission: float = 0.0
    total_funding_fee: float = 0.0
    total_credit: float = 0.0
    is_open: bool | None = None
    liquidation_price: float = 0.0

    # Trade metrics
    trade_commission: float = 0.0
    trade_amount: float = 0.0
    trade_pnl: float = 0.0  # no consider fees
    trade_roi: float = 0.0  # no consider fees
    trade_duration: int = 0  # in seconds

    last_price: float | None = None
    debug: bool = False

    # trades: dict[str, TradeData] = Field(default_factory=dict)  # key: trade_id
    trades: dict[str, dict] = Field(default_factory=dict)  # key: trade_id
    tradepnls: list[TradePnL] = Field(default_factory=list)

    def __post_init__(self) -> None:
        """"""
        self.aio_symbol: str = f"{self.symbol}|{self.exchange.value}"
        self.aio_position_id: str = f"{self.aio_symbol}|{self.position_side.value}"
        if self.opened_at:
            self.opened_at_ts = to_timestamp(self.opened_at)
        if self.updated_at:
            self.updated_at_ts = to_timestamp(self.updated_at)

        if self.leverage != 1:
            self.margin_ratio = 1 / self.leverage
        if self.margin_ratio:
            if 0 < self.margin_ratio <= 1:
                self.leverage = 1 / self.margin_ratio

    def reset(self) -> None:
        """
        Reset all position tracking metrics to initial state while preserving
        configuration parameters.
        """
        # Reset position metrics
        self.size = 0.0
        self.side = None
        self.avg_open_price = 0.0
        self.is_open = None

        # Reset financial metrics
        self.net_investment = 0.0
        self.total_turnover = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        self.market_value = 0.0
        self.notional_value = 0.0
        self.total_commission = 0.0
        self.total_funding_fee = 0.0
        self.total_credit = 0.0
        self.liquidation_price = 0.0

        # Reset trade specific metrics
        self.trade_commission = 0.0
        self.trade_amount = 0.0
        self.trade_pnl = 0.0
        self.trade_roi = 0.0
        self.trade_duration = 0
        self.last_price = None

        # Reset trade collections
        self.trades = {}
        self.tradepnls = []

        # Update timestamps if needed
        self.opened_at = None
        self.updated_at = None
        self.opened_at_ts = None
        self.updated_at_ts = None

    def to_dict(self, display: list[str] = [], return_obj: bool = True, return_aio: bool = False) -> dict[str, Any]:
        """Convert key position metrics to dictionary format"""
        if display:
            return {k: v for k, v in self.__dict__.items() if k in display}
        if not return_obj:
            return {
                k: v.value if isinstance(v, Enum) else v
                for k, v in self.__dict__.items()
                if not (not return_aio and k.startswith("aio_"))
            }
        return {
            "symbol": self.symbol,
            "asset_type": self.asset_type.value,
            "exchange": self.exchange.value,
            "position_side": self.position_side.value,
            # 'side': self.side.value,
            "opened_at": self.opened_at,
            "updated_at": self.updated_at,
            "position_margin": self.position_margin,
            "initial_margin": self.initial_margin,
            "maintenance_margin": self.maintenance_margin,
            "leverage": self.leverage,
            "commission_rate": {
                "commission_rate": self.commission_rate,
                "maker_commission_rate": self.maker_commission_rate,
                "taker_commission_rate": self.taker_commission_rate,
            },
            "is_contract": self.is_contract,
            "is_inverse": self.is_inverse,
            "is_isolated": self.is_isolated,
            "liquidation_price": self.liquidation_price,
            "total_commission": self.total_commission,
            "total_pnl": self.total_pnl,
            "net_investment": self.net_investment,
            "size": self.size,
            "avg_open_price": self.avg_open_price,
            "market_value": self.market_value,
            "notional_value": self.notional_value,
            "last_price": self.last_price,
        }

    def __repr__(self) -> str:
        return f"<PositionData symbol:{self.symbol}, asset_type:{self.asset_type}, exchange:{self.exchange}, position_side:{self.position_side}, leverage: {self.leverage}, multiplier:{self.multiplier}, consider_price:{self.consider_price}, is_contract: {self.is_contract}, is_inverse: {self.is_inverse}, is_isolated: {self.is_isolated}, commission_rate: {self.commission_rate}, maker_commission_rate: {self.maker_commission_rate}, taker_commission_rate: {self.taker_commission_rate}; avg_open_price:{self.avg_open_price}, size:{self.size}; market_value:{self.market_value}; notional_value:{self.notional_value}, net_investment:{self.net_investment}, total_pnl:{self.total_pnl}(rlze:{self.realized_pnl} & unrlze:{self.unrealized_pnl}), total_commission:{self.total_commission}, total_turnover:{self.total_turnover}, liquidation_price:{self.liquidation_price}, is_open:{self.is_open}, last_price:{self.last_price}>\n "

    def __setitem__(self, key: str, item: Any) -> None:
        self.__dict__[key] = item

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __len__(self) -> int:
        return len(self.__dict__)

    def __delitem__(self, key: str) -> None:
        del self.__dict__[key]

    def clear(self) -> None:
        return self.__dict__.clear()

    def copy(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def has_key(self, k: str) -> bool:
        return k in self.__dict__

    def update(self, *args: Any, **kwargs: Any) -> None:
        return self.__dict__.update(*args, **kwargs)

    def keys(self) -> list[str]:
        return list(self.__dict__.keys())

    def values(self) -> list[Any]:
        return list(self.__dict__.values())

    def set_leverage(self, leverage: float) -> bool:
        """Update position leverage if valid value provided"""
        if leverage:
            self.leverage = leverage
            self.margin_ratio = 1 / self.leverage
            return True
        return False

    def set_margin_mode(self, is_isolated: bool) -> bool:
        """Update margin mode if valid boolean provided"""
        if isinstance(is_isolated, bool):
            self.is_isolated = is_isolated
            return True
        return False

    def update_by_tradefeed(self, trade: TradeData, available_balance: float = 0.0) -> None:
        """
        Update position metrics based on new trade execution.

        Args:
            trade: TradeData
            available_balance: Account balance (optional)
        """
        commission = trade.commission
        commission_rate = trade.commission_rate
        traded_price = trade.price
        traded_quantity = trade.quantity
        side = trade.side.value
        is_maker = trade.is_maker
        if not commission:
            commission = self.cal_commission(traded_price, traded_quantity, is_maker or False)
            commission_rate = prec_round(commission / (traded_price * traded_quantity), 5)
        self.total_commission = prec_round(self.total_commission + commission, 8)

        trade_direction = 1 if side == "BUY" else -1
        quantity_with_direction = traded_quantity * trade_direction
        position_direction = (abs(self.size) / self.size) if self.size else trade_direction
        self.is_open = (self.size * quantity_with_direction) >= 0

        ### Update tradedata
        if self.is_open:
            self.trade_pnl = 0
            self.trade_roi = 0
            self.trade_duration = 0
            self.opened_at = trade.traded_at
            self.opened_at_ts = to_timestamp(self.opened_at)
        else:
            self.trade_pnl = self.cal_trade_pnl(
                quantity_with_direction=min(abs(quantity_with_direction), abs(self.size))
                * (abs(self.size) / self.size),
                last_price=traded_price,
                is_inverse=self.is_inverse,
                avg_open_price=self.avg_open_price,
                multiplier=self.multiplier,
            )
            self.trade_roi = prec_round(
                ((traded_price - self.avg_open_price) / self.avg_open_price) * position_direction, 4
            )
            # TODO: temporary fix for timezone issue
            self.trade_duration = int(
                (trade.traded_at.replace(tzinfo=None) - self.opened_at.replace(tzinfo=None)).total_seconds()
                if self.opened_at
                else 0
            )
            self.updated_at = trade.traded_at
            self.updated_at_ts = to_timestamp(self.updated_at)

        trade.pnl = self.trade_pnl
        trade.leverage = self.leverage
        trade.duration = self.trade_duration
        if trade.is_open is None:
            trade.is_open = self.is_open
            trade._update_action()
        if not trade.commission:
            trade.commission = commission
        self.trade_amount = abs(self.get_trade_value(quantity=traded_quantity, price=traded_price))

        ### Append / Create / Update Trade and TradePnL
        trade_plain = trade.to_dict(return_obj=False)
        self.trades[trade.ori_order_id] = trade_plain
        current_trade_pnl = None
        if self.size == 0.0 and self.is_open:  # Opening new position
            current_trade_pnl = TradePnL()
            current_trade_pnl.exchange = trade.exchange
            current_trade_pnl.symbol = trade.symbol
            current_trade_pnl.code = trade.code
            current_trade_pnl.asset_type = trade.asset_type
            current_trade_pnl.position_side = trade.position_side
            current_trade_pnl.leverage = self.leverage
            current_trade_pnl.is_short = trade_direction < 0
            current_trade_pnl.open_date = trade.traded_at
            current_trade_pnl.open_price = traded_price
            current_trade_pnl.open_trade_value = traded_quantity * traded_price
            current_trade_pnl.size = traded_quantity
            current_trade_pnl.size_requested = traded_quantity
            current_trade_pnl.commission_open_rate = commission_rate
            current_trade_pnl.commission_open_cost = commission
            current_trade_pnl.commission_open_asset = trade.commission_asset
            current_trade_pnl.enter_reason = trade.tag
            current_trade_pnl._remain_commission_open_cost += commission
            current_trade_pnl.trades.append(trade_plain)
            self.tradepnls.append(current_trade_pnl)
        else:  # Updating existing position
            current_trade_pnl = self.get_last_trade_pnl()
            if current_trade_pnl:
                current_trade_pnl.trades.append(trade_plain)

                # Update price extremes
                current_trade_pnl.update_price(trade.price)

                # Handle closing trade
                if current_trade_pnl.is_open and (
                    (current_trade_pnl.is_short and trade.side == Side.BUY)
                    or (not current_trade_pnl.is_short and trade.side == Side.SELL)
                ):
                    # Update trade info
                    # funding_fee = sum(fee['funding_fee'] for fee in current_trade_pnl.funding_fees if current_trade_pnl.open_date <= fee['funding_datetime'] <= trade.traded_at) if current_trade_pnl.close_date else current_trade_pnl.funding_fee_total
                    current_trade_pnl.close_date = trade.traded_at
                    current_trade_pnl.close_price = trade.price
                    current_trade_pnl.close_trade_value = trade.price * trade.quantity
                    # current_trade_pnl.commission_close_rate = self.maker_commission_rate if is_maker else self.taker_commission_rate
                    current_trade_pnl.commission_close_rate = commission_rate
                    if current_trade_pnl.commission_close_cost is not None:
                        current_trade_pnl.commission_close_cost += commission
                    else:
                        current_trade_pnl.commission_close_cost = commission
                    current_trade_pnl.commission_close_asset = trade.commission_asset

                    # Calculate ROI and total PnL for closed portion
                    closed_quantity = min(current_trade_pnl.size, trade.quantity)
                    closed_trade_value = current_trade_pnl.open_price * closed_quantity
                    current_trade_pnl.close_roi = prec_round(
                        (self.trade_pnl / closed_trade_value) * self.leverage, 6
                    )  # before fee
                    # Calculate commission proportionally based on closed quantity ratio
                    closed_ratio = closed_quantity / current_trade_pnl.size
                    funding_fee_portion = current_trade_pnl._remain_funding_fee * closed_ratio
                    commission_open_portion = current_trade_pnl._remain_commission_open_cost * closed_ratio
                    # print(
                    #     f">>> closed_ratio:{closed_ratio}; trade_pnl:{self.trade_pnl}; commission_open_portion:{commission_open_portion}(raw:{current_trade_pnl._remain_commission_open_cost}); commission_close:{commission}; funding_fee_portion:{funding_fee_portion}(raw:{current_trade_pnl._remain_funding_fee})"
                    # )
                    current_trade_pnl._remain_funding_fee -= funding_fee_portion
                    current_trade_pnl._remain_commission_open_cost -= commission_open_portion
                    current_trade_pnl.close_pnl = prec_round(self.trade_pnl - commission_open_portion - commission, 6)
                    current_trade_pnl.realized_pnl += current_trade_pnl.close_pnl  # NOTE: not consider funding fee in here since it will be automatically add when funding distributed
                    current_trade_pnl.close_pnl -= funding_fee_portion  # after all fees

                    # Update position status
                    current_trade_pnl.size = prec_round(current_trade_pnl.size - closed_quantity, 6)
                    current_trade_pnl.is_open = current_trade_pnl.size > 0
                    current_trade_pnl.exit_reason = trade.tag

                else:
                    # Handle opening trade
                    current_trade_pnl.is_open = True
                    # current_trade_pnl.open_price = trade.price
                    # print(f">>> quantity_with_direction:{quantity_with_direction}; traded_price:{traded_price}; is_open:{True}; is_inverse:{self.is_inverse}; avg_open_price:{current_trade_pnl.open_price}; size:{current_trade_pnl.quantity}")
                    current_trade_pnl.open_price = self.cal_avg_open_price(
                        quantity_with_direction,
                        traded_price,
                        is_open=True,
                        is_inverse=self.is_inverse,
                        avg_open_price=current_trade_pnl.open_price,
                        size=current_trade_pnl.size * (-1 if current_trade_pnl.is_short else 1),
                    )
                    current_trade_pnl.size = prec_round(current_trade_pnl.size + traded_quantity, 6)
                    current_trade_pnl.size_requested = prec_round(current_trade_pnl.size_requested + traded_quantity, 6)
                    current_trade_pnl.open_trade_value = current_trade_pnl.open_price * current_trade_pnl.size
                    # current_trade_pnl.open_date = trade.traded_at
                    if current_trade_pnl.commission_open_cost is not None:
                        current_trade_pnl.commission_open_cost += commission
                    else:
                        current_trade_pnl.commission_open_cost = commission
                    current_trade_pnl._remain_commission_open_cost += commission
                    # current_trade_pnl.commission_open_asset = trade.commission_asset

        ### Update position metrics
        # Update realized P&L
        self.realized_pnl = prec_round(self.trade_pnl + self.realized_pnl, 6)

        # Update opened_at and updated_at
        if self.is_open:
            self.opened_at = trade.traded_at
            self.opened_at_ts = to_timestamp(self.opened_at)
        self.updated_at = trade.traded_at
        self.updated_at_ts = to_timestamp(self.updated_at)

        # Update average open price
        if self.is_open:
            if self.is_inverse:
                self.avg_open_price = prec_round(
                    1
                    / (
                        (
                            (((1 / self.avg_open_price) if self.avg_open_price else 0) * self.size)
                            + ((1 / traded_price) * quantity_with_direction)
                        )
                        / (self.size + quantity_with_direction)
                    ),
                    8,
                )
            else:
                self.avg_open_price = prec_round(
                    ((self.avg_open_price * self.size) + (traded_price * quantity_with_direction))
                    / (self.size + quantity_with_direction),
                    8,
                )
        else:
            if traded_quantity > abs(self.size):
                self.avg_open_price = prec_round(traded_price, 8)

        # Update size
        if not self.is_open and not self.is_contract and traded_quantity > abs(self.size):
            self.size = 0.0
            self.side = None
            self.opened_at = None
            self.opened_at_ts = None
        else:
            self.size = prec_round(quantity_with_direction + self.size, 8)
            self.side = PositionSide.LONG if self.size > 0 else PositionSide.SHORT

        # Update margin
        if self.is_contract:
            self.initial_margin = prec_round(abs(self.size) * self.avg_open_price / self.leverage, 6)
            self.maintenance_margin = prec_round(abs(self.size) * self.avg_open_price * self.maint_margin_rate, 6)
            self.position_margin = prec_round(self.initial_margin + self.maintenance_margin, 6)

        # Update net investment
        investment = abs(self.get_trade_value(quantity=self.size, price=self.avg_open_price))
        self.net_investment = prec_round(max(self.net_investment, investment), 6)
        self.total_turnover = prec_round(self.total_turnover + self.trade_amount, 6)

        # Update P&L, market value, notional value, liquidation price
        self.update_by_marketdata(last_price=traded_price, available_balance=available_balance)

    def get_last_trade_pnl(self) -> TradePnL | None:
        # return self.tradepnls[max(self.tradepnls.keys())]
        return self.tradepnls[-1] if self.tradepnls else None

    def update_by_marketdata(
        self,
        last_price: float,
        available_balance: float = 0.0,
        funding_rate: float = 0.0,
        funding_datetime: DateTime | None = None,
    ) -> None:
        """
        Update position metrics based on new market price.

        Args:
            last_price: Current market price
            available_balance: Account balance (optional)
        """
        self.last_price = last_price
        self.unrealized_pnl = self.cal_trade_pnl(
            quantity_with_direction=self.size,
            last_price=last_price,
            is_inverse=self.is_inverse,
            avg_open_price=self.avg_open_price,
            multiplier=self.multiplier,
        )
        self.total_pnl = prec_round(self.realized_pnl + self.unrealized_pnl, 6)
        if self.is_contract:
            self.market_value = self.unrealized_pnl
            if self.is_isolated:  #  and self.leverage != 1
                self.liquidation_price = self.cal_liq_px_isolated(
                    entry_price=self.avg_open_price,
                    quantity=self.size,
                    leverage=self.leverage,
                    is_inverse=self.is_inverse,
                    maint_margin_rate=self.maint_margin_rate,
                    extra_maint_margin=self.extra_maint_margin,
                    debug=False,
                )
            else:
                self.liquidation_price = self.cal_liq_px_cross(
                    entry_price=self.avg_open_price,
                    quantity=self.size,
                    net_position_size=self.size,
                    mark_price=last_price,
                    available_balance=max(0, available_balance),
                    leverage=self.leverage,
                    is_inverse=self.is_inverse,
                    maint_margin_rate=self.maint_margin_rate,
                    debug=False,
                )
                if self.liquidation_price:
                    self.liquidation_price = prec_round(self.liquidation_price, 6)
        else:
            self.market_value = self.get_trade_value(quantity=self.size, price=last_price) * (
                1 if self.size > 0 else -1
            )

        self.notional_value = (
            abs((self.size / self.avg_open_price * last_price * self.multiplier) if self.avg_open_price else 0.0)
            if self.is_inverse
            else abs(self.size * last_price * self.multiplier)
        )

        current_trade_pnl = self.get_last_trade_pnl()
        if current_trade_pnl and current_trade_pnl.is_open:
            current_trade_pnl.update_price(last_price)
            if self.liquidation_price:
                current_trade_pnl.update_liquidation_price(liq_price=self.liquidation_price)
            if funding_rate and funding_datetime:
                self.total_funding_fee += funding_rate * last_price * self.size
                current_trade_pnl.update_funding_fee(
                    price=last_price,
                    funding_rate=funding_rate,
                    funding_datetime=funding_datetime,
                )

    def get_trade_value(self, quantity: float, price: float, for_comm: bool = False) -> float:
        """
        Calculate trade value based on quantity and price.

        Args:
            quantity: Trade size
            price: Trade price
            for_comm: Whether calculation is for commission (optional)

        Returns:
            float: Trade value
        """
        quantity = abs(quantity)

        if self.is_contract:
            if self.is_inverse:
                if not (self.avg_open_price and price):
                    return 0.0
                if for_comm:
                    mkt = quantity * (1 / (self.avg_open_price if self.avg_open_price else price)) * price
                else:
                    mkt = quantity * self.get_margin(
                        (1 / (self.avg_open_price if self.avg_open_price else price)) * price
                    )
            else:
                if for_comm:
                    mkt = quantity * price
                else:
                    mkt = quantity * self.get_margin(price)
        else:
            mkt = quantity * price

        return prec_round(mkt, 6)

    def get_margin(self, price: float) -> float:
        """
        Calculate required margin for given price.

        Args:
            price: Current price

        Returns:
            float: Required margin amount
        """
        if 0 < self.margin_ratio <= 1:
            return price * self.margin_ratio

        elif self.margin_ratio <= 0:
            return price * self.multiplier

        return self.position_margin

    def cal_commission(self, price: float, quantity: float, is_maker: bool = False) -> float:
        """
        Calculate trade commission.

        Args:
            price: Trade price
            quantity: Trade size
            is_maker: Whether trade is maker (optional)

        Returns:
            float: Commission amount
        """
        commission_rate = (
            self.commission_rate
            if not self.maker_commission_rate and not self.taker_commission_rate
            else self.maker_commission_rate
            if is_maker
            else self.taker_commission_rate
        )
        commission_rate = commission_rate if commission_rate else 0.0
        if self.is_contract and not self.consider_price:
            commission = quantity * commission_rate
        else:
            if self.is_option:
                commission = price * quantity * 0.0002
            else:
                commission = self.get_trade_value(quantity=quantity, price=price, for_comm=True) * commission_rate
                if self.min_commission:
                    commission = max(commission, self.min_commission)

        return prec_round(commission, 8)

    def cal_interest(
        self, price: float, quantity: float, start_date: datetime, end_date: datetime, use_trade_days: bool = False
    ) -> float:
        """
        Calculate interest charges.

        Args:
            price: Position price
            quantity: Position size
            start_date: Interest start date
            end_date: Interest end date
            use_trade_days: Whether to use trading days (optional)

        Returns:
            float: Interest amount
        """
        days = (end_date - start_date).days
        cdays = 252 if self.use_trade_days else 365
        interest_rate = self.interest_rate if self.interest_rate is not None else 0.0
        return prec_round(days * price * quantity * (interest_rate / cdays), 6)

    def cal_max_order_quantity(
        self,
        available_balance: float,
        price: float,
        step_size: float,
        order_quantity: float,
        is_open: bool,
        is_maker: bool = False,
    ) -> float:
        """
        Calculate maximum allowed order size.

        Args:
            available_balance: Account balance
            price: Current price
            step_size: Order size increment
            order_quantity: Desired order size
            is_open: Whether opening new position
            is_maker: Whether order is maker

        Returns:
            float: Maximum allowed order size
        """
        max_order_quantity: float = 0.0
        step_size = step_size if step_size else 1
        commission_rate = (
            self.commission_rate
            if not self.maker_commission_rate and not self.taker_commission_rate
            else self.maker_commission_rate
            if is_maker
            else self.taker_commission_rate
        )
        commission_rate = commission_rate if commission_rate else 0.0
        price = self.get_margin(price=price)
        if not is_open:
            if order_quantity > abs(self.size):
                available_balance += abs(self.size) * self.get_margin(
                    price=self.last_price if self.last_price else price
                )

        if self.is_contract:
            if self.is_inverse:
                max_order_quantity = available_balance
            else:
                max_order_quantity = (
                    floor(available_balance / price / self.multiplier / (commission_rate + 1) / step_size) * step_size
                )
        else:
            max_order_quantity = (
                floor(available_balance / price / self.multiplier / (commission_rate + 1) / step_size) * step_size
            )
        if not is_open:
            if order_quantity > abs(self.size):
                max_order_quantity += abs(self.size)
        return max_order_quantity

    @staticmethod
    def cal_trade_pnl(
        quantity_with_direction: float, last_price: float, is_inverse: bool, avg_open_price: float, multiplier: float
    ) -> float:
        """
        Calculate trade P&L.

        Args:
            quantity_with_direction: Position size with direction
            last_price: Current price
            is_inverse: Whether inverse contract
            avg_open_price: Average entry price
            multiplier: Contract multiplier

        Returns:
            float: Trade P&L
        """
        trade_pnl = 0.0
        if is_inverse:
            trade_pnl = (
                multiplier * ((1 / avg_open_price) - (1 / last_price)) * quantity_with_direction * last_price
                if avg_open_price
                else 0.0
            )
        else:
            trade_pnl = multiplier * (last_price - avg_open_price) * quantity_with_direction

        return prec_round(trade_pnl, 6)

    @staticmethod
    def cal_avg_open_price(
        quantity_with_direction: float,
        traded_price: float,
        is_open: bool,
        is_inverse: bool,
        avg_open_price: float,
        size: float,
    ) -> float:
        """
        Calculate average open price.

        Args:
            traded_quantity: Trade size
            traded_price: Trade price
            is_open: Whether position still open
            is_inverse: Whether inverse contract
            avg_open_price: Current average price
            quantity_with_direction: Position size with direction
            size: Current net position

        Returns:
            float: New average open price
        """
        new_avg_open_price = 0.0
        if is_open:
            if is_inverse:
                new_avg_open_price = 1 / (
                    (
                        (((1 / avg_open_price) if avg_open_price else 0) * size)
                        + ((1 / traded_price) * quantity_with_direction)
                    )
                    / (size + quantity_with_direction)
                )
            else:
                new_avg_open_price = ((avg_open_price * size) + (traded_price * quantity_with_direction)) / (
                    size + quantity_with_direction
                )
        else:
            if abs(quantity_with_direction) > abs(size):
                new_avg_open_price = traded_price

        return prec_round(new_avg_open_price, 7)

    @staticmethod
    def cal_liq_px_isolated(
        entry_price: float,
        quantity: float,
        leverage: float = 1,
        is_inverse: bool = False,
        maint_margin_rate: float = 0.5 / 100,
        extra_maint_margin: float = 0,
        debug: bool = False,
    ) -> float:
        """
        Calculate liquidation price for isolated margin.

        Args:
            entry_price: Entry price
            quantity: Position size
            leverage: Position leverage
            is_inverse: Whether inverse contract
            maint_margin_rate: Maintenance margin rate
            extra_maint_margin: Additional margin
            debug: Enable debug output

        Returns:
            float: Liquidation price
        """
        liquidation_price: float = 0.0
        if not quantity:
            return liquidation_price

        position_direction = 1 if quantity > 0 else -1
        quantity = abs(quantity)
        if is_inverse:
            liquidation_price = entry_price / (
                (1 + position_direction * 1 / leverage - position_direction * maint_margin_rate)
                + position_direction * (extra_maint_margin * (entry_price / quantity))
            )
            if debug:
                print(
                    f"[cal_liq_px_isolated] position_direction:{position_direction} | formula:{entry_price} / ((1 {'+' if position_direction == 1 else '-'} {1 / leverage} {'-' if position_direction == 1 else '+'} {maint_margin_rate}){'+' if position_direction == 1 else '-'} ({extra_maint_margin}*({entry_price}/{quantity})) )"
                )

        else:
            # liquidation_price = entry_price * (
            #     1 - position_direction * (1 / leverage) + position_direction * maint_margin_rate
            # ) - position_direction * (extra_maint_margin / quantity)
            im = (quantity * entry_price) / leverage
            mm = (quantity * entry_price) * maint_margin_rate
            liquidation_price = (
                entry_price
                - position_direction * ((im - mm) / quantity)
                - position_direction * (extra_maint_margin / quantity)
            )
            if debug:
                print(
                    f"[cal_liq_px_isolated] position_direction:{position_direction} | formula:{entry_price} * (1 {'-' if position_direction == 1 else '+'} {1 / leverage} {'+' if position_direction == 1 else '-'} {maint_margin_rate}){'-' if position_direction == 1 else '+'} ({extra_maint_margin}/{quantity}))"
                )

        return liquidation_price

    @staticmethod
    def cal_liq_px_cross(
        entry_price: float,
        quantity: float,
        net_position_size: float,
        mark_price: float | None = None,
        available_balance: float = 0.0,
        leverage: float = 1,
        is_inverse: bool = False,
        maint_margin_rate: float = 0.5 / 100,
        debug: bool = False,
    ) -> float:
        """
        Calculate liquidation price for cross margin.

        Args:
            entry_price: Entry price
            quantity: Position size
            net_position_size: Total position size
            mark_price: Current mark price
            available_balance: Account balance
            leverage: Position leverage
            is_inverse: Whether inverse contract
            maint_margin_rate: Maintenance margin rate
            debug: Enable debug output

        Returns:
            float: Liquidation price
        """
        liquidation_price: float = 0.0
        if not net_position_size or not quantity:
            return liquidation_price

        position_direction = 1 if quantity > 0 else -1
        quantity = abs(quantity)
        net_position_size = abs(net_position_size) if net_position_size else 0.0
        if is_inverse:
            order_margin = 0.0
            fee_to_open = 0.0
            bust_px = (1 + position_direction * 0.0006 * quantity) / (
                quantity / entry_price
                + position_direction * (available_balance / entry_price - order_margin - fee_to_open)
            )
            rhs = -(
                available_balance
                - order_margin
                - maint_margin_rate * quantity / entry_price
                - (quantity * 0.0006) / bust_px
            )
            liquidation_price = 1 / (-position_direction * rhs / quantity + 1 / entry_price)
            if debug:
                print(
                    f"[cal_liq_px_cross] liquidation_price:{liquidation_price} | bust_px:{bust_px}; position_direction:{position_direction} | bust px formula: (1+{position_direction}*0.0006 * {quantity}) / ({quantity / entry_price} {'+' if position_direction == 1 else '-'}({available_balance / entry_price} - {order_margin} - {fee_to_open}))"
                )

        else:
            if mark_price:
                px = (
                    (entry_price if mark_price > entry_price else mark_price)
                    if position_direction == 1
                    else (mark_price if mark_price > entry_price else entry_price)
                )
            else:
                px = entry_price

            im = (quantity * entry_price) / leverage
            mm = (quantity * entry_price) * maint_margin_rate
            liquidation_price = px - position_direction * ((available_balance + im - mm) / net_position_size)
            if debug:
                print(
                    f"[cal_liq_px_cross] liquidation_price:{liquidation_price} | px:{px}; position_direction:{position_direction}; im:{im}; mm:{mm}; available_balance:{available_balance} | formula:[{px} {'-' if position_direction == 1 else '+'} ({available_balance}+{im}-{mm})] / {net_position_size}"
                )

        return liquidation_price


@dataclass
class TickData(BaseData):
    """
    Tick data contains information about:
        * last trade in market
        * orderbook snapshot
        * intraday market statistics.
    """

    symbol: str
    code: str
    exchange: Exchange
    datetime: DateTime
    asset_type: AssetType

    volume: float = 0.0  # past 24 hr
    turnover: float = 0.0  # past 24 hr
    open_interest: float | None = None  # past 24 hr
    last_price: float = 0.0  # past 24 hr

    bid_price_1: float = 0.0
    bid_price_2: float = 0.0
    bid_price_3: float = 0.0
    bid_price_4: float = 0.0
    bid_price_5: float = 0.0

    ask_price_1: float = 0.0
    ask_price_2: float = 0.0
    ask_price_3: float = 0.0
    ask_price_4: float = 0.0
    ask_price_5: float = 0.0

    bid_volume_1: float = 0.0
    bid_volume_2: float = 0.0
    bid_volume_3: float = 0.0
    bid_volume_4: float = 0.0
    bid_volume_5: float = 0.0

    ask_volume_1: float = 0.0
    ask_volume_2: float = 0.0
    ask_volume_3: float = 0.0
    ask_volume_4: float = 0.0
    ask_volume_5: float = 0.0

    localtime: DateTime | None = None
    ts: int | None = None

    def __post_init__(self) -> None:
        """"""
        self.aio_symbol: str = f"{self.symbol}|{self.exchange.value}"
        if self.datetime:
            self.ts = to_timestamp(self.datetime)


@dataclass
class BarData(BaseData):
    """
    Candlestick bar data of a certain trading period.
    """

    symbol: str
    code: str
    exchange: Exchange
    datetime: DateTime
    asset_type: AssetType

    interval: str | None = None
    volume: float = 0.0
    turnover: float = 0.0
    open_interest: float | None = None
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0

    ts: int | None = None

    def __post_init__(self) -> None:
        """"""
        self.aio_symbol: str = f"{self.symbol}|{self.exchange.value}"
        if self.datetime:
            self.ts = to_timestamp(self.datetime)


@dataclass
class InstrumentData(BaseData):
    """
    Symbol data contains basic information about each contract traded.

    TODO: add start datetime, deliveray datetime
    """

    symbol: str
    exchange: Exchange
    code: str
    asset_type: AssetType

    step_size: float
    tick_size: float

    multiplier: float | None = 1.0  # multiplier the contract
    min_volume: float | None = 0.0  # minimum trading volume of the contract
    min_notional: float | None = 0.0  # minimum trading volume of the contract
    max_volume: float | None = 0.0  # maximum trading volume of the contract

    option_strike: float = 0.0
    option_underlying: str = ""  # aio_symbol of underlying contract
    option_type: OptionType | None = None
    option_listed: DateTime | None = None
    option_expiry: DateTime | None = None
    option_portfolio: str = ""
    option_index: str = ""  # for identifying options with same strike price

    def __post_init__(self) -> None:
        """"""
        self.aio_symbol: str = f"{self.symbol}|{self.exchange.value}"
