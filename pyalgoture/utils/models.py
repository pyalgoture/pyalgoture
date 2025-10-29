from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pytz
from dateutil.parser import parse
from dateutil.relativedelta import FR, relativedelta

"""
TODO: pydantic type checking
"""

"""
NOTE - same for binance and bybit:
One-Way Mode
Under one-way mode, you can hold either a long or a short position of a contract.

Hedge Mode
Under hedge mode, you can hold both long and short positions simultaneously of a contract.

Order quantity must be position
Position quantity could be either pos or neg
"""
CHINA_TZ = pytz.timezone("Asia/Hong_Kong")
UTC_TZ = pytz.utc

TZ_MAP = {"hkt": CHINA_TZ, "utc": UTC_TZ}

ACCOUNT_TYPE_SPOT = "SPOT"
ACCOUNT_TYPE_CONTRACT = "CONTRACT"
ACCOUNT_TYPE_UNIFIED = "UNIFIED"

ASSET_TYPE_LIQUID_SWAP = "LIQUID_SWAP"
ASSET_TYPE_EARN = "EARN"
ASSET_TYPE_SPOT = "SPOT"
ASSET_TYPE_FUTURE = "FUTURE"
ASSET_TYPE_INVERSE_FUTURE = "INVERSE_FUTURE"
ASSET_TYPE_INVERSE_PERPETUAL = "INVERSE_PERPETUAL"
ASSET_TYPE_INVERSE_DATED_FUTURE = "INVERSE_DATED_FUTURE"
ASSET_TYPE_PERPETUAL = "PERPETUAL"
ASSET_TYPE_DATED_FUTURE = "DATED_FUTURE"
ASSET_TYPE_OPTION = "OPTION"

FUTURE_MONTH_CODES = {
    "F": "01",
    "G": "02",
    "H": "03",
    "J": "04",
    "K": "05",
    "M": "06",
    "N": "07",
    "Q": "08",
    "U": "09",
    "V": "10",
    "X": "11",
    "Z": "12",
}


def change_scale(num: int | float, scale: int) -> float:
    """Scale number by power of 10 for eqonex.

    Args:
        num: Number to scale
        scale: Power of 10 to scale by (negative scale means division)

    Returns:
        Scaled number
    """
    return float(num * 10 ** (-scale))


def generate_datetime(tz: pytz.BaseTzInfo | None = None) -> datetime:
    """Generate current datetime in specified timezone.

    Args:
        tz: Timezone to use, defaults to UTC

    Returns:
        Current datetime in specified timezone
    """
    if not tz:
        tz = UTC_TZ
    return datetime.now(tz)


def timestamp_to_datetime(timestamp: int | float, tz: pytz.BaseTzInfo | None = None) -> datetime:
    """Convert timestamp to datetime object in specified timezone.

    Args:
        timestamp: Unix timestamp (in seconds or milliseconds)
        tz: Target timezone, defaults to UTC

    Returns:
        Datetime object in specified timezone
    """
    if not tz:
        tz = UTC_TZ
    timestamp_int = int(timestamp)
    if len(str(timestamp_int)) == 13:
        timestamp_int //= 1000
    dt = datetime.fromtimestamp(timestamp_int, tz=UTC_TZ)
    return dt.astimezone(tz)


def isoformat_to_datetime(timestring: str, tz: pytz.BaseTzInfo | None = None) -> datetime:
    """Convert ISO format time string to datetime object.

    Handles formats like:
    - '2019-03-05T09:56:55.728933+00:00'
    - '20220427-07:02:14.668'
    - '20220704-10:32:24.372'

    Args:
        timestring: ISO format time string
        tz: Target timezone, defaults to UTC

    Returns:
        Datetime object in specified timezone
    """
    if not tz:
        tz = UTC_TZ
    if "Z" in timestring:
        timestring = timestring.replace("Z", "")

    dt = parse(timestring)
    dt_utc = UTC_TZ.localize(dt)
    return dt_utc.astimezone(tz)


def future_asset_type(symbol: str, input_type: str) -> str:
    """Determine future asset type based on symbol and input type.

    Args:
        symbol: Trading symbol
        input_type: Type of future ('inverse' or 'future')

    Returns:
        Asset type string
    """
    return (
        (ASSET_TYPE_INVERSE_DATED_FUTURE if symbol[-2:].isdigit() else ASSET_TYPE_INVERSE_PERPETUAL)
        if input_type == "inverse"
        else (ASSET_TYPE_DATED_FUTURE if symbol[-2:].isdigit() else ASSET_TYPE_PERPETUAL)
        if input_type == "future"
        else "unknown"
    )


def binance_asset_type(symbol: str | None, datatype: str | None) -> str:
    """Determine Binance asset type from symbol and datatype.

    Examples:
    - spot: BTCUSDT
    - perpetual: BTCUSDT
    - dated_future: BTCUSDT_240628
    - inverse perpetual: BTCUSD_PERP
    - inverse dated_future: BTCUSD_221230

    Args:
        symbol: Trading symbol
        datatype: Data type ('spot', 'option', 'inverse', 'linear')

    Returns:
        Asset type string
    """
    if symbol is None or datatype is None:
        return "unknown"

    symbol = symbol.upper()
    asset = "unknown"
    if datatype == "spot":
        asset = ASSET_TYPE_SPOT
    elif datatype == "option":
        asset = ASSET_TYPE_OPTION
    elif datatype == "inverse":
        if symbol[-2:].isdigit():
            asset = ASSET_TYPE_INVERSE_DATED_FUTURE
        else:
            asset = ASSET_TYPE_INVERSE_PERPETUAL
    elif datatype == "linear":
        if symbol[-2:].isdigit():
            asset = ASSET_TYPE_DATED_FUTURE
        else:
            asset = ASSET_TYPE_PERPETUAL
    # print(f"symbol:{symbol}; datatype:{datatype}; asset:{asset}")
    return asset


def eqonex_asset_type(symbol: str | None) -> str:
    if symbol is None:
        return "unknown"
    if "[" in symbol and "]" in symbol:
        if "[F]" in symbol:
            return ASSET_TYPE_PERPETUAL
        else:
            return ASSET_TYPE_DATED_FUTURE
    else:
        return ASSET_TYPE_SPOT


def ftx_asset_type(symbol: str | None) -> str:
    if symbol is None:
        return "unknown"
    symbol = symbol.upper()
    asset = "unknown"
    if symbol.split("/")[-1] == "USD" or symbol.split("/")[-1] == "USDT" or symbol.split("/")[-1] == "BTC":
        asset = ASSET_TYPE_SPOT
    elif symbol.split("/")[-1] == "BRZ" or symbol.split("/")[-1] == "EUR" or symbol.split("/")[-1] == "TRYB":
        asset = ASSET_TYPE_SPOT
    elif symbol.split("/")[-1] == "DOGE":
        # TLSA/DOGE - type spot
        asset = ASSET_TYPE_SPOT
    elif "PERP" in symbol:
        asset = ASSET_TYPE_PERPETUAL
    elif symbol[-4:].isdigit():
        asset = ASSET_TYPE_DATED_FUTURE
    elif "MOVE" in symbol and symbol[-1:].isdigit():
        asset = ASSET_TYPE_DATED_FUTURE
    return asset


def okx_asset_type(symbol: str | None) -> str:
    if symbol is None:
        return "unknown"
    symbol = symbol.upper()
    asset = "unknown"
    if "SWAP" in symbol:
        if "-USDT-" in symbol or "-USDC-" in symbol:
            asset = ASSET_TYPE_PERPETUAL
        elif "-USD-" in symbol:
            asset = ASSET_TYPE_INVERSE_PERPETUAL
    elif symbol.count("-") == 4:
        asset = ASSET_TYPE_OPTION
    elif symbol[-5:].isdigit():
        if "-USDT-" in symbol or "-USDC-" in symbol:
            asset = ASSET_TYPE_DATED_FUTURE
        elif "-USD-" in symbol:
            asset = ASSET_TYPE_INVERSE_DATED_FUTURE
    else:
        asset = ASSET_TYPE_SPOT
    return asset


def bingx_asset_type(symbol: str | None, datatype: str | None) -> str:
    if symbol is None or datatype is None:
        return "unknown"
    symbol = symbol.upper()
    asset = "unknown"
    if "-" in symbol:
        if datatype == "spot":
            asset = ASSET_TYPE_SPOT
        elif datatype == "linear":
            asset = ASSET_TYPE_PERPETUAL
    return asset


def bybit_asset_type(symbol: str | None, datatype: str | None) -> str:
    if symbol is None or datatype is None:
        return "unknown"
    symbol = symbol.upper()
    asset = "unknown"
    if datatype == "spot":
        asset = ASSET_TYPE_SPOT
    elif datatype == "option":
        asset = ASSET_TYPE_OPTION
    elif datatype == "inverse":
        if symbol[-2:].isdigit():
            asset = ASSET_TYPE_INVERSE_DATED_FUTURE
        else:
            asset = ASSET_TYPE_INVERSE_PERPETUAL
    elif datatype == "linear":
        if symbol[-2:].isdigit():
            asset = ASSET_TYPE_DATED_FUTURE
        else:
            asset = ASSET_TYPE_PERPETUAL
    # print(f"symbol:{symbol}; datatype:{datatype}; asset:{asset}")
    return asset


def bitget_asset_type(symbol: str | None, datatype: str | None) -> str:
    if symbol is None or datatype is None:
        return "unknown"
    symbol = symbol.upper()
    asset = "unknown"
    if datatype == "spot":
        asset = ASSET_TYPE_SPOT
    elif datatype == "option":
        asset = ASSET_TYPE_OPTION
    elif datatype == "inverse":
        if symbol[-2:].isdigit():
            asset = ASSET_TYPE_INVERSE_DATED_FUTURE
        else:
            asset = ASSET_TYPE_INVERSE_PERPETUAL
    elif datatype == "linear":
        if symbol[-2:].isdigit():
            asset = ASSET_TYPE_DATED_FUTURE
        else:
            asset = ASSET_TYPE_PERPETUAL
    # print(f"symbol:{symbol}; datatype:{datatype}; asset:{asset}")
    return asset


def okx_inst_type(symbol: str | None = None, datatype: str | None = None) -> str:
    symbol = symbol.upper() if symbol else symbol
    datatype = datatype.lower() if datatype else datatype
    asset = "unknown"
    if symbol:
        if "SWAP" in symbol:
            asset = "SWAP"
        elif symbol.count("-") == 4:
            asset = "OPTION"
        elif symbol[-5:].isdigit():
            asset = "FUTURES"
        else:
            asset = "SPOT"
    if datatype:
        if datatype == "linear":
            asset = "SWAP"
        # elif symbol[-5:].isdigit():
        #     asset = "FUTURES"
        else:
            asset = "SPOT"
    return asset


def normalize_name(symbol: str | None, asset_type: str | None) -> str:
    """
    standarized format:
        spot: BTC/USDT
        perpetual: BTCUSDT
        dated future: BTCUSDT230929
        inverse perpetual: BTCUSD
        inverse dated future: BTCUSD230929
        option: BTC-230929-19500-C
    """
    # Handle None values
    if symbol is None or asset_type is None:
        return ""

    # try:
    name = symbol = symbol.upper()
    asset_type = asset_type.upper()

    def parse_spot_symbol(symbol: str, base: str) -> str:
        coin = symbol.split(base)[0]
        if coin:
            name = f"{coin}/{base}"
        else:
            name = f"{base}/{symbol.split(base)[1]}"
        return name

    if asset_type == ASSET_TYPE_SPOT:
        """
        spot
            bybit => BTCUSDT
            binance => BTCUSDT
            okx => BTC-USDT
            eqonex => BTC/USDC
            bingx => BTC-USDT
            bitget => BTCUSDT_SPBL
            bitget v2 => BTCUSDT
        """
        if "_SPBL" in symbol:
            # for bitget
            symbol = symbol.replace("_SPBL", "")
        if "/" in symbol:
            name = symbol
        elif "-" in symbol:
            name = symbol.replace("-", "/")
        elif "USDC" in symbol:
            name = parse_spot_symbol(symbol, "USDC")
        elif "USDT" in symbol:
            name = parse_spot_symbol(symbol, "USDT")
        elif "BUSD" in symbol:
            name = parse_spot_symbol(symbol, "BUSD")
        elif "USD" in symbol:
            name = parse_spot_symbol(symbol, "USD")
        elif "UST" in symbol:
            name = parse_spot_symbol(symbol, "UST")
        elif "TUSD" in symbol:
            name = parse_spot_symbol(symbol, "TUSD")
        elif "BTC" in symbol:
            name = parse_spot_symbol(symbol, "BTC")
        elif "ETH" in symbol:
            name = parse_spot_symbol(symbol, "ETH")
        elif "BNB" in symbol:
            name = parse_spot_symbol(symbol, "BNB")
        elif "DAI" in symbol:
            name = parse_spot_symbol(symbol, "DAI")
        else:
            name = symbol

    elif (
        asset_type == ASSET_TYPE_PERPETUAL
        or asset_type == ASSET_TYPE_DATED_FUTURE
        or asset_type == ASSET_TYPE_INVERSE_PERPETUAL
        or asset_type == ASSET_TYPE_INVERSE_DATED_FUTURE
        or asset_type == ASSET_TYPE_FUTURE
        or asset_type == ASSET_TYPE_INVERSE_FUTURE
    ):
        """
        perpetual
            bybit => BTCUSDT
            binance => BTCUSDT
            okx => BTC-USDT-SWAP
            eqonex => BTC/USDC[F]
            bingx ==> BTC-USDT
            bitget ==> BTCUSDT_UMCBL
            bitget v2 => BTCUSDT
            dydx ==> BTC-USD

        usdc perprtual
            bybit => BTCPERP         (USDC)
            bitget ==> BTCPERP_CMCBL

        dated future
            bybit => BTC-05MAY23    (USDC)
            binance => BTCUSDT_221230
            okx => BTC-USDT-221118
            eqonex => BTC/USDC[220325]

        inverse perpetual
            bybit => BTCUSD
            binance => BTCUSD_PERP
            okx => BTC-USD-SWAP
            bitget => BTCUSD_DMCBL

        inverse dated future
            bybit => BTCUSDH23
            binance => BTCUSD_221230
            okx => BTC-USD-221118
            bitget => BTCUSD_DMCBL_230929
        """
        if "-PERP" in symbol:
            # for bybit
            name = f"{symbol.replace('-PERP', '')}USD"
        elif "_PERP" in symbol:
            # for binance inverse future
            name = symbol.replace("_PERP", "")
        elif "SWAP" in symbol:
            # for okx perpetual
            name = symbol.replace("-SWAP", "").replace("-", "")
        elif "[" in symbol and "]" in symbol:
            # for eqonex
            if "[F]" in symbol:
                # perpetual - BTC/USDC[F]
                name = symbol.replace("[F]", "").replace("/", "")  # .replace('[','').replace(']','').replace('/','')
            else:
                # dated future - BTC/USDC[220325]
                nsplit = symbol.split("[")
                name = nsplit[0].replace("/", "") + nsplit[1][:-1]
                # print(f"{name} EQONEX dated future.................................................")
        elif "-" in symbol:
            # for dydx future (BTC-USD)
            # for okx future (BTC-USDT-SWAP, BTC-USD-230512)
            # for bybit usdc dated future (BTC-05MAY23 ==> BTCUSDC230505)
            if symbol.count("-") == 2:
                name = symbol.replace("-", "")
            else:
                pre, sub = symbol.split("-")
                if sub[:2].isdigit():
                    name = pre + "USDC" + parse(sub).strftime("%y%m%d")
                else:
                    # for bingx perpetual         (BTC-USDT  ==> BTCUSDT)
                    name = symbol.replace("-", "")
                    # for dydx future (BTC-USD ==> BTCUSDT)
                    if name.endswith("USD"):
                        name = name.replace("USD", "USDT")

        elif "_UMCBL" in symbol:
            # for biget perpetual
            name = symbol.replace("_UMCBL", "")
        elif "_SUMCBL" in symbol:
            # for biget perpetual testnet
            name = symbol.replace("_SUMCBL", "")
        elif "_CMCBL" in symbol:
            # for biget usdc perpetual
            name = symbol.replace("_CMCBL", "")
        elif "_SCMCBL" in symbol:
            # for biget usdc perpetual testnet
            name = symbol.replace("_SCMCBL", "")
        elif "_DMCBL" in symbol:
            # for biget inverse perpetual & inverse dated
            if "_DMCBL_" in symbol:
                name = symbol.replace("_DMCBL_", "")
            else:
                name = symbol.replace("_DMCBL", "")
        elif "_SDMCBL" in symbol:
            # for biget inverse perpetual & inverse dated testnet
            if "_SDMCBL_" in symbol:
                name = symbol.replace("_SDMCBL_", "")
            else:
                name = symbol.replace("_SDMCBL", "")
        elif "PERP" in symbol:
            # for bybit usdc perp
            name = symbol.replace("PERP", "USDC")
        else:
            if symbol.count("-") == 2:
                # for okx
                name = symbol.replace("-", "")
            else:
                name = symbol.replace("-", "").replace("_", "").replace("/", "")
                if (
                    asset_type == ASSET_TYPE_INVERSE_DATED_FUTURE
                    or asset_type == ASSET_TYPE_DATED_FUTURE
                    or asset_type == ASSET_TYPE_FUTURE
                    or asset_type == ASSET_TYPE_INVERSE_FUTURE
                ):
                    if symbol[-2:].isdigit() and symbol[-3] in FUTURE_MONTH_CODES:
                        # for bybit inverse future
                        mth_raw_code = symbol[-3]
                        mth = FUTURE_MONTH_CODES[mth_raw_code]
                        year = symbol[-2:]
                        day = (date(int(f"20{year}"), int(mth), 1) + relativedelta(weekday=FR(-1), day=31)).day
                        name = f"{symbol[:-3]}{year}{mth}{day}"
                        # print(f"!!!BYBIT dated future inverse!!! mth: {mth}; year: {year}; day: {day}; name: {name} ................................................")

    elif asset_type == ASSET_TYPE_OPTION:
        """
        option
            bybit => BTC-14JUN22-24500-C
            binance => BTC-221118-19500-C
            okx => BTC-USD-221118-11000-P
        """

        name = symbol.replace("USD-", "")
        splits = name.split("-")
        # print(name,'??????????',splits, splits[1],splits[1].isdigit())
        if len(splits) > 1 and not splits[1].isdigit():
            d = datetime.strptime(splits[1], "%d%b%y").strftime("%y%m%d")
            name = name.replace(splits[1], d)
    # except Exception as e:
    #     print(f"Error in normalize_name: {e}. symbol:{symbol}; asset_type:{asset_type}")

    return name


def denormalize_name(name: str | None, asset_type: str | None) -> str:
    """
    standarized format:
        spot: BTC/USDT
        perpetual: BTCUSDT
        dated future: BTCUSDT230929
        inverse perpetual: BTCUSD
        inverse dated future: BTCUSD230929
        option: BTC-230929-19500-C

    spot
        bybit => BTCUSDT
        binance => BTCUSDT
        okx => BTC-USDT
        eqonex => BTC/USDC
        bingx => BTC-USDT
        bitget => BTCUSDT_SPBL

    perpetual
        bybit => BTCUSDT
        binance => BTCUSDT
        okx => BTC-USDT-SWAP
        eqonex => BTC/USDC[F]
        bingx ==> BTC-USDT
        bitget ==> BTCUSDT_UMCBL
        dydx ==> BTC-USD

    usdc perprtual
        bybit => BTCPERP         (USDC)
        bitget ==> BTCPERP_CMCBL

    dated future
        bybit => BTC-05MAY23    (USDC)
        binance => BTCUSDT_221230
        okx => BTC-USDT-221118
        eqonex => BTC/USDC[220325]

    inverse perpetual
        bybit => BTCUSD
        binance => BTCUSD_PERP
        okx => BTC-USD-SWAP
        bitget => BTCUSD_DMCBL

    inverse dated future
        bybit => BTCUSDH23
        binance => BTCUSD_221230
        okx => BTC-USD-221118
        bitget => BTCUSD_DMCBL_230929

    option
        bybit => BTC-14JUN22-24500-C
        binance => BTC-221118-19500-C
        okx => BTC-USD-221118-11000-P
    """
    # Handle None values
    if name is None:
        return ""

    symbol = name.upper()
    return symbol


def order_status_map(order_status: str | None, exchange: str | None) -> str:
    """
    Status(Enum):
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

    BYBIT:
        Created - order has been accepted by the system but not yet put through the matching engine
        New - order has been placed successfully
        Rejected
        PartiallyFilled
        Filled
        PendingCancel - matching engine has received the cancelation request but it may not be canceled successfully
        Cancelled

        Only for conditional orders:
            Untriggered - order yet to be triggered
            Deactivated - order has been canceled by the user before being triggered
            Triggered - order has been triggered by last traded price
            Active - order has been triggered and the new active order has been successfully placed. Is the final state of a successful conditional order

    BINANCE:
        NEW	The order has been accepted by the engine.
        PARTIALLY_FILLED	A part of the order has been filled.
        FILLED	The order has been completed.
        CANCELED	The order has been canceled by the user.
        PENDING_CANCEL	Currently unused
        REJECTED	The order was not accepted by the engine and not processed.
        EXPIRED	The order was canceled according to the order type's rules (e.g. LIMIT FOK orders with no fill, LIMIT IOC or MARKET orders that partially fill)
        or by the exchange, (e.g. orders canceled during liquidation, orders canceled during maintenance)

    OKX:
        canceled
        live
        partially_filled
        filled

    BINGX
    訂單狀態, NEW新訂單 PENDING委託中 PARTIALLY_FILLED部分成交 FILLED完全成交 CANCELED已撤銷 FAILED失敗

    BITGET
    spot:
        status(订单状态)
        字段	说明
        init	初始化 插入DB
        new	未成交 orderbook中等待撮合
        partial_fill	部分成交
        full_fill	全部成交
        cancelled	已撤销
    future:
        state(订单状态)
        字段	说明
        init	初始订单 插入DB成功
        new	新建订单 orderbook中等待撮合
        partially_filled	部分成交
        filled	全部成交
        canceled	已撤销
    future ws:
        init new partial-fill full-fill cancelled

    """
    bybit_order_status_map = {
        "CREATED": "NEW",
        "NEW": "NEW",
        "REJECTED": "FAILED",
        "PENDINGCANCEL": "CANCELING",
        "CANCELLED": "CANCELED",
        "PARTIALLYFILLED": "FILLING",
        "FILLED": "FILLED",
        # Only for conditional orders:
        "UNTRIGGERED": "NEW",
        "DEACTIVATED": "CANCELED",
        "TRIGGERED": "FILLING",
        "ACTIVE": "FILLING",
    }
    binance_order_status_map = {
        "NEW": "NEW",
        "REJECTED": "FAILED",
        "EXPIRED": "FAILED",
        "PENDING_CANCEL": "CANCELING",
        "CANCELED": "CANCELED",
        "PARTIALLY_FILLED": "FILLING",
        "FILLED": "FILLED",
        # Only for conditional orders: (stop order)
        "Untriggered": "NEW",
        "Deactivated": "CANCELED",
        "Triggered": "FILLING",
        "Active": "FILLING",
    }
    okx_order_status_map = {
        "CANCELED": "CANCELED",
        "LIVE": "FILLING",
        "PARTIALLY_FILLED": "FILLING",
        "FILLED": "FILLED",
        # algo order
        "PAUSE": "CANCELING",
        "EFFECTIVE": "FILLING",
        "ORDER_FAILED": "FAILED",
    }
    bingx_order_status_map = {
        "NEW": "NEW",
        "CANCELED": "CANCELED",
        "PENDING": "FILLING",
        "PARTIALLY_FILLED": "FILLING",
        "FILLED": "FILLED",
        "FAILED": "FAILED",
    }
    bitget_order_status_map = {
        # future
        "INIT": "NEW",
        "NEW": "NEW",
        "CANCELED": "CANCELED",
        "PARTIALLY_FILLED": "FILLING",
        "FILLED": "FILLED",
        # spot
        "PARTIAL_FILL": "FILLING",
        "FULL_FILL": "FILLED",
        "CANCELLED": "CANCELED",
        # FUTURE WS
        "PARTIAL-FILL": "FILLING",
        "FULL-FILL": "FILLED",
    }
    # Handle None values
    if exchange is None or order_status is None:
        return "UNRECOGNIZED"

    exchange = exchange.upper()
    order_status = order_status.upper()
    if exchange == "BYBIT":
        order_status = bybit_order_status_map.get(order_status, "UNRECOGNIZED")
    elif exchange == "BINANCE":
        order_status = binance_order_status_map.get(order_status, "UNRECOGNIZED")
    elif exchange == "OKX":
        order_status = okx_order_status_map.get(order_status, "UNRECOGNIZED")
    elif exchange == "BINGX":
        order_status = bingx_order_status_map.get(order_status, "UNRECOGNIZED")
    elif exchange == "BITGET":
        order_status = bitget_order_status_map.get(order_status, "UNRECOGNIZED")

    return order_status


@dataclass
class BaseData:
    """
    Any data object should inherit base data.
    """

    def update(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)

    def update_from_dict(self, item: dict[str, Any]) -> None:
        for k, v in item.items():
            self.update(k, v)

    def to_dict(self, display: list[str] | None = None) -> dict[str, Any]:
        if display:
            return {k: v for k, v in self.__dict__.items() if k in display}
        return self.__dict__

    def dict(self, display: list[str] | None = None) -> dict[str, Any]:
        return self.__dict__

    def __setitem__(self, k: str, v: Any) -> None:
        setattr(self, k, v)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def __delitem__(self, k: str) -> None:
        self[k] = None

    def get(self, k: str, default: Any | None = None) -> Any:
        return getattr(self, k, default)


@dataclass
class CommonDataSchema(BaseData):
    status_code: int | None = None
    msg: dict | list | str | None = None


@dataclass
class CommonResponseSchema(BaseData):
    success: bool | None = None
    error: bool | None = None
    data: list | dict | str | None = None
    msg: str | dict | list | None = None

    def __setitem__(self, k: str, v: Any) -> None:
        setattr(self, k, v)

    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    def __delitem__(self, k: str) -> None:
        self[k] = None

    def get(self, k: str, default: Any | None = None) -> Any:
        return getattr(self, k, default)

    def replace_msg(self, msg: str | dict | list | None) -> None:
        self.msg = msg
        self.data = msg


@dataclass
class AccountSchema(BaseData):
    symbol: str | None = None
    exchange: str | None = None
    asset_type: str | None = None
    account_type: str | None = None
    available: float | None = None
    frozen: float | None = None
    balance: float | None = None
    market_value: float | None = None

    def to_general_form(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asset": self.symbol,
            "exchange": self.exchange,
            "asset_type": self.asset_type,
            "account_type": self.account_type,
            "available": self.available,
            "frozen": self.frozen,
            "balance": self.balance,
            "market_value": self.market_value,
        }

    def from_bitget_format(self, msg: dict[str, Any], datatype: str, account_type: str) -> None:
        # spot: {'coinId': 704, 'coinName': 'CGPT', 'available': '0.00000000', 'frozen': '0.00000000', 'lock': '0.00000000', 'uTime': '1681196765309'}, {'coinId': 705, 'coinName': 'ZZZ', 'available': '0.00000000', 'frozen': '0.00000000', 'lock': '0.00000000', 'uTime': '1681196765309'}
        # linear: {'marginCoin': 'BTC', 'locked': '0', 'available': '0', 'crossMaxAvailable': '0', 'fixedMaxAvailable': '0', 'maxTransferOut': '0', 'equity': '0', 'usdtEquity': '0', 'btcEquity': '0', 'crossRiskRate': '0', 'unrealizedPL': None, 'bonus': '0'}
        self.symbol = str(msg["coinName"]) if "coinName" in msg else msg["marginCoin"]
        self.asset_type = bitget_asset_type(self.symbol, datatype)
        self.account_type = account_type
        self.available = float(msg["available"])
        self.frozen = float(msg["frozen"]) if "frozen" in msg else float(msg["locked"])
        self.balance = self.available + self.frozen
        self.market_value = float(msg["usdtEquity"]) if "usdtEquity" in msg else None
        self.exchange = "BITGET"

    def from_bitget_to_form(self, msg: dict[str, Any], datatype: str, account_type: str) -> dict[str, Any] | None:
        # if datatype=='spot' and not float(msg["available"]) and not float(msg["frozen"]):
        # print(msg,'????')
        if not float(msg["available"]) and not float(msg["frozen"] if "frozen" in msg else msg["locked"]):
            return None
        self.from_bitget_format(msg, datatype, account_type)
        return self.to_general_form()

    def from_bitget_v2_format(self, msg: dict[str, Any], datatype: str, account_type: str) -> None:
        # print(msg, datatype, account_type) # linear CONTRACT
        # spot: {'coin': 'USDT', 'available': '20.18958000', 'limitAvailable': '0', 'frozen': '0.00000000', 'locked': '0.00000000', 'uTime': '1693125461000'}
        # linear: {'marginCoin': 'USDT', 'locked': '0', 'available': '163.160395', 'crossedMaxAvailable': '163.160395', 'isolatedMaxAvailable': '163.160395', 'maxTransferOut': '163.160395', 'accountEquity': '163.160395', 'usdtEquity': '163.160395', 'btcEquity': '0.002471512327', 'crossedRiskRate': '0', 'unrealizedPL': '0', 'coupon': '0', 'crossedUnrealizedPL': None, 'isolatedUnrealizedPL': None}
        self.symbol = str(msg["coin"]) if datatype == "spot" else msg["marginCoin"]
        self.asset_type = bitget_asset_type(self.symbol, datatype)
        self.account_type = account_type
        self.available = float(msg["available"])
        self.frozen = float(msg["frozen"]) if "frozen" in msg else float(msg["locked"])
        self.balance = self.available + self.frozen
        self.market_value = float(msg["usdtEquity"]) if "usdtEquity" in msg else None
        self.exchange = "BITGET"

    def from_bitget_v2_to_form(self, msg: dict[str, Any], datatype: str, account_type: str) -> dict[str, Any] | None:
        # if datatype=='spot' and not float(msg["available"]) and not float(msg["frozen"]):
        # print(msg,'????')
        # if not float(msg["available"]) and not float(msg["frozen"] if "frozen" in msg else msg["locked"]):
        #     return None
        self.from_bitget_v2_format(msg, datatype, account_type)
        return self.to_general_form()

    def from_bybit_v2_format(self, msg: dict[str, Any], datatype: str, account_type: str) -> None:
        # print(msg,'????account')
        # {'availableToBorrow': '', 'bonus': '0', 'accruedInterest': '0', 'availableToWithdraw': '', 'totalOrderIM': '0', 'equity': '48571.51037223', 'totalPositionMM': '0', 'usdValue': '48587.05325554', 'unrealisedPnl': '0', 'collateralSwitch': True, 'spotHedgingQty': '0', 'borrowAmount': '0', 'totalPositionIM': '0', 'walletBalance': '48571.51037223', 'cumRealisedPnl': '-1428.48962777', 'locked': '0', 'marginCollateral': True, 'coin': 'USDT'} ????account

        # spot: {'coin': 'BTC', 'transferBalance': '0.60357745', 'walletBalance': '0.60357745', 'bonus': ''}
        # linear: {'coin': 'BTC', 'transferBalance': '0.10540102', 'walletBalance': '0.10540102', 'bonus': ''}
        # inverse:
        # option: {'coin': 'USDC', 'transferBalance': '654.584839', 'walletBalance': '654.584839', 'bonus': ''}
        ### unified:
        # {'availableToBorrow': '1500000', 'bonus': '0', 'accruedInterest': '0', 'availableToWithdraw': '0', 'totalOrderIM': '0', 'equity': '11.95631066', 'totalPositionMM': '5.63359235', 'usdValue': '11.95440338', 'unrealisedPnl': '5.677', 'borrowAmount': '0.0', 'totalPositionIM': '75.12619235', 'walletBalance': '6.27931066', 'cumRealisedPnl': '6.27931066', 'coin': 'USDC'} ????account
        # {'availableToBorrow': '3', 'bonus': '0', 'accruedInterest': '0', 'availableToWithdraw': '0.00265733', 'totalOrderIM': '0', 'equity': '0.00265733', 'totalPositionMM': '0', 'usdValue': '69.47760952', 'unrealisedPnl': '0', 'borrowAmount': '0.0', 'totalPositionIM': '0', 'walletBalance': '0.00265733', 'cumRealisedPnl': '0', 'coin': 'BTC'} ????account

        # equity?
        self.symbol = str(msg["coin"])  # .lower()
        self.asset_type = bybit_asset_type(self.symbol, datatype)
        self.account_type = account_type
        self.available = float(msg["free"]) if "free" in msg else float(msg["walletBalance"])
        self.balance = float(msg["walletBalance"])
        self.frozen = float(msg["locked"])
        self.market_value = float(msg["usdValue"]) if "usdValue" in msg and msg["usdValue"] else None
        self.exchange = "BYBIT"

    def from_bybit_v2_to_form(self, msg: dict[str, Any], datatype: str, account_type: str) -> dict[str, Any] | None:
        if msg["walletBalance"] == "0":
            return None
        self.from_bybit_v2_format(msg, datatype, account_type)
        return self.to_general_form()

    def from_dydx_v3_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        # print(f">>> {msg}")
        self.symbol = "USDT"
        self.asset_type = ASSET_TYPE_PERPETUAL
        self.account_type = ""
        self.available = float(msg["freeCollateral"])
        self.balance = float(msg["equity"])
        self.frozen = self.balance - self.available
        self.market_value = float(msg["equity"])
        self.exchange = "DYDX"

        return self.to_general_form()

    def from_okx_format(self, msg: dict[str, Any], datatype: str, account_type: str) -> None:
        # print(f"[okx] account msg:{msg}")
        # [okx] account msg:{'availBal': '3.2960824157473905', 'availEq': '3.2960824157473905', 'borrowFroz': '', 'cashBal': '3.2960824157473905', 'ccy': 'BTC', 'clSpotInUseAmt': '', 'crossLiab': '', 'disEq': '217325.62121355586', 'eq': '3.2968085742347673', 'eqUsd': '217325.62121355586', 'fixedBal': '0', 'frozenBal': '0.0007261584873767', 'imr': '0', 'interest': '', 'isoEq': '0.0007261584873767', 'isoLiab': '', 'isoUpl': '-0.0007800176349804', 'liab': '', 'maxLoan': '', 'maxSpotInUse': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1720771215153', 'upl': '-0.0007800176349804', 'uplLiab': ''}
        self.symbol = str(msg["ccy"])
        self.asset_type = okx_asset_type(self.symbol)
        self.account_type = account_type
        self.available = float(msg["availBal"]) if msg["availBal"] else 0.0
        self.balance = float(msg["cashBal"]) if msg["cashBal"] else 0.0
        self.frozen = float(msg["frozenBal"]) if msg["frozenBal"] else 0.0
        self.market_value = float(msg["eqUsd"]) if msg["eqUsd"] else 0.0

        self.exchange = "OKX"

    def from_okx_to_form(self, msg: dict[str, Any], datatype: str, account_type: str) -> dict[str, Any] | None:
        self.from_okx_format(msg, datatype, account_type)
        return self.to_general_form()

    def from_ftx_format(self, msg: dict[str, Any]) -> None:
        self.created_at = isoformat_to_datetime(msg["createdAt"])
        self.ori_order_id = msg["id"]
        self.symbol = msg["market"]
        if self.symbol is not None:
            self.asset_type = ftx_asset_type(self.symbol)
            self.name = normalize_name(self.symbol, self.asset_type)
        self.price = msg["price"]
        self.size = msg["size"]
        self.status = msg["status"]
        self.order_type = msg["type"]
        self.ioc = msg["ioc"]
        self.post_only = msg["postOnly"]
        self.order_id = msg["clientId"]
        self.filled_size = msg["filledSize"]
        self.exchange = "FTX"

    def from_ftx_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.from_ftx_format(msg)
        return self.to_general_form()

    def from_bybit_spot_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.symbol = msg["coin"]

        self.balance = float(msg["balance"])
        self.available = float(msg["available"])
        self.frozen = float(msg["frozen"])

        self.exchange = "BYBIT"

    def from_bybit_future_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.symbol = msg["coin"]

        self.balance = float(msg["balance"])
        self.available = float(msg["available"])
        self.frozen = float(msg["frozen"])

        self.exchange = "BYBIT"

    def from_bybit_inverse_future_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.symbol = msg["coin"]

        self.balance = float(msg["balance"])
        self.available = float(msg["available"])
        self.frozen = float(msg["frozen"])

        self.exchange = "BYBIT"

    def from_bybit_option_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.symbol = msg["coin"]

        self.balance = float(msg["balance"])
        self.available = float(msg["available"])
        self.frozen = float(msg["frozen"])

        self.exchange = "BYBIT"

    def from_bybit_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if datatype == "spot":
            self.from_bybit_spot_format(msg)
        elif datatype == "linear":
            self.from_bybit_future_format(msg)
        elif datatype == "inverse":
            self.from_bybit_inverse_future_format(msg)
        elif datatype == "option":
            self.from_bybit_option_format(msg)
        else:
            self.from_bybit_spot_format(msg)

        return self.to_general_form()

    def from_binance_spot_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.symbol = msg["coin"]

        self.balance = float(msg["balance"])
        self.available = float(msg["available"])
        self.frozen = float(msg["frozen"])

        self.exchange = "BINANCE"

    def from_binance_future_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.symbol = msg["coin"]

        self.balance = float(msg["balance"])
        self.available = float(msg["available"])
        self.frozen = float(msg["frozen"])

        self.exchange = "BINANCE"

    def from_binance_inverse_future_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.symbol = msg["coin"]

        self.balance = float(msg["balance"])
        self.available = float(msg["available"])
        self.frozen = float(msg["frozen"])

        self.exchange = "BINANCE"

    def from_binance_option_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.symbol = msg["coin"]

        self.balance = float(msg["balance"])
        self.available = float(msg["available"])
        self.frozen = float(msg["frozen"])

        self.exchange = "BINANCE"

    def from_binance_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if datatype == "spot":
            self.from_binance_spot_format(msg)
        elif datatype == "linear":
            self.from_binance_spot_format(msg)
        elif datatype == "inverse":
            self.from_binance_inverse_future_format(msg)
        elif datatype == "option":
            self.from_binance_option_format(msg)
        else:
            self.from_binance_spot_format(msg)

        return self.to_general_form()

    def from_binance_v2_format(self, msg: dict[str, Any], datatype: str, account_type: str) -> None:
        # print(f">>>> msg:{msg} | account_type:{account_type}")
        self.symbol = msg["asset"]
        if self.symbol is not None:
            self.asset_type = binance_asset_type(self.symbol, datatype)
        self.account_type = account_type
        self.available = float(msg["free"]) if "free" in msg else None  # float(msg["availableBalance"])
        self.frozen = float(msg["locked"]) if "locked" in msg else None
        if "balance" in msg:
            self.balance = float(msg["balance"])
        else:
            # Handle None values in addition
            available_val = self.available if self.available is not None else 0.0
            frozen_val = self.frozen if self.frozen is not None else 0.0
            self.balance = available_val + frozen_val
        self.market_value = None
        self.exchange = "BINANCE"

    def from_binance_v2_to_form(self, msg: dict[str, Any], datatype: str, account_type: str) -> dict[str, Any] | None:
        if datatype == "spot":
            # if not (float(msg["free"]) and float(msg["locked"])):
            if float(msg["free"]) == 0.0 and float(msg["locked"]) == 0.0:
                return None
        else:
            if not float(msg["balance"]):
                return None

        self.from_binance_v2_format(msg, datatype, account_type)

        return self.to_general_form()


@dataclass
class ProxySchema(BaseData):
    proxy_host: str = ""
    proxy_port: int = 0


@dataclass
class PositionSchema(BaseData):
    asset_type: str | None = None
    symbol: str | None = None
    name: str | None = None
    price: None | float = None  # market price show None
    side: str | None = None
    position_side: str | None = None
    size: float | None = None
    position_value: float | None = None
    entry_price: float | None = None
    leverage: float | None = None
    position_margin: float | None = None
    initial_margin: float | None = None
    maintenance_margin: float | None = None
    realized_pnl: float | None = None
    unrealized_pnl: float | None = None
    is_isolated: bool | None = None
    auto_add_margin: bool | None = None
    liq_price: float | None = None
    bust_price: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    exchange: str | None = None

    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None

    def to_general_form(self) -> dict[str, Any]:
        data = {
            "code": self.symbol,
            "symbol": self.name,
            "asset_type": self.asset_type,
            "exchange": self.exchange,
            "side": self.side,
            "position_side": self.position_side,
            "size": self.size,
            "avg_open_price": self.entry_price,
            "leverage": self.leverage,
            "is_isolated": self.is_isolated,
            "liquidation_price": self.liq_price,
            "position_value": self.position_value,
            "position_margin": self.position_margin,
            "initial_margin": self.initial_margin,
            "maintenance_margin": self.maintenance_margin,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "auto_add_margin": self.auto_add_margin,
            "bust_price": self.bust_price,
        }
        if self.asset_type == "OPTION":
            data["delta"] = self.delta
            data["gamma"] = self.gamma
            data["vega"] = self.vega
            data["theta"] = self.theta
        return data

    def from_bitget_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print(msg,"????")
        # linear: {'marginCoin': 'USDT', 'symbol': 'BTCUSDT_UMCBL', 'holdSide': 'long', 'openDelegateCount': '0', 'margin': '0', 'available': '0', 'locked': '0', 'total': '0', 'leverage': 20, 'achievedProfits': '0', 'averageOpenPrice': '0', 'marginMode': 'crossed', 'holdMode': 'double_hold', 'unrealizedPL': '0', 'liquidationPrice': '0', 'keepMarginRate': '0.004', 'marketPrice': '30077.41', 'cTime': '1680798362978'}
        # linear ws: {'posId': '1032128976686002178', 'instId': 'SBTCSUSDT_SUMCBL', 'instName': 'SBTCSUSDT', 'marginCoin': 'SUSDT', 'margin': '4.2796', 'marginMode': 'fixed', 'holdSide': 'long', 'holdMode': 'single_hold', 'total': '0.001', 'available': '0.001', 'locked': '0', 'averageOpenPrice': '29734.75', 'leverage': 7, 'achievedProfits': '-0.0382', 'upl': '-0.0404', 'uplRate': '-0.0094', 'liqPx': '25572.72', 'keepMarginRate': '0.004', 'fixedMarginRate': '0.142162945012', 'marginRate': '0.032221246204', 'cTime': '1681804644748', 'uTime': '1681806999979', 'markPrice': '29694.34'} ????

        self.symbol = str(msg["symbol"]) if "symbol" in msg else msg["instId"]
        if self.symbol is not None:
            self.asset_type = bitget_asset_type(self.symbol, datatype=datatype)
            self.name = normalize_name(self.symbol, self.asset_type)
        self.position_side = msg["holdSide"] if msg["holdMode"] == "single_hold" else "net"
        self.size = float(msg["total"])
        self.side = msg["holdSide"].upper()
        self.position_value = float(msg["margin"])
        self.entry_price = float(msg["averageOpenPrice"])
        self.leverage = float(msg["leverage"])
        self.realized_pnl = float(msg["achievedProfits"])
        self.unrealized_pnl = float(msg["unrealizedPL"]) if "unrealizedPL" in msg else float(msg["upl"])
        self.is_isolated = True if msg["marginMode"] == "fixed" else False  # crossed:全倉，fixed：逐倉
        self.auto_add_margin = None
        self.liq_price = float(msg["liqPx"]) if "liqPx" in msg else None
        self.bust_price = None
        self.initial_margin = None
        self.maintenance_margin = None
        self.position_margin = float(msg["margin"])
        self.exchange = "BITGET"

    def from_bitget_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any] | None:
        if not float(msg["total"]):
            return None
        self.from_bitget_format(msg, datatype)
        return self.to_general_form()

    def from_bitget_v2_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print(msg,"????")
        # linear: 'marginCoin': 'USDT', 'symbol': 'BTCUSDT', 'holdSide': 'long', 'openDelegateSize': '0.002', 'marginSize': '6.59711', 'available': '0.002', 'locked': '0', 'total': '0.002', 'leverage': '20', 'achievedProfits': '0', 'openPriceAvg': '65971.1', 'marginMode': 'crossed', 'posMode': 'hedge_mode', 'unrealizedPL': '0.0922', 'liquidationPrice': '-15017.43572', 'keepMarginRate': '0.004', 'markPrice': '66017.2', 'marginRatio': '0.006766777178', 'cTime': '1713782760820'}
        # linear ws:

        self.symbol = str(msg["symbol"]) if "symbol" in msg else msg["instId"]
        if self.symbol is not None:
            self.asset_type = bitget_asset_type(self.symbol, datatype=datatype)
            self.name = normalize_name(self.symbol, self.asset_type)
        self.position_side = "net" if msg["posMode"] == "one_way_mode" else msg["holdSide"]
        self.size = float(msg["total"])
        self.side = msg["holdSide"].upper()
        self.position_value = float(msg["marginSize"])
        self.entry_price = float(msg["openPriceAvg"])
        self.leverage = float(msg["leverage"])
        self.realized_pnl = None
        self.unrealized_pnl = float(msg["unrealizedPL"])
        self.is_isolated = True if msg["marginMode"] == "isolated" else False  # crossed:全倉，fixed：逐倉
        self.auto_add_margin = None
        self.liq_price = float(msg["liquidationPrice"])
        self.bust_price = None
        self.initial_margin = None
        self.maintenance_margin = None
        self.position_margin = float(msg["marginSize"])
        self.exchange = "BITGET"

    def from_bitget_v2_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any] | None:
        if not float(msg["total"]):
            return None
        self.from_bitget_v2_format(msg, datatype)
        return self.to_general_form()

    def from_ftx_format(self, msg: dict[str, Any]) -> None:
        self.symbol = msg["future"]
        if self.symbol is not None:
            self.asset_type = ftx_asset_type(self.symbol)
            self.name = normalize_name(self.symbol, self.asset_type)
        self.price = None
        self.side = msg["side"]
        self.position_side = ""
        self.size = float(msg["size"])
        self.position_value = float(msg["cost"])
        self.entry_price = msg["entryPrice"]
        self.leverage = None
        self.position_margin = None
        self.realized_pnl = float(msg["realizedPnl"])
        self.unrealized_pnl = float(msg["unrealizedPnl"])
        self.is_isolated = None
        self.auto_add_margin = None
        self.liq_price = msg["estimatedLiquidationPrice"]
        self.bust_price = None
        self.initial_margin = float(msg["initialMarginRequirement"])
        self.maintenance_margin = float(msg["maintenanceMarginRequirement"])
        self.exchange = "FTX"

    def from_ftx_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.from_ftx_format(msg)
        return self.to_general_form()

    def from_bingx_format(self, msg: dict[str, Any], datatype: str | None = None) -> None:
        self.symbol = str(msg["symbol"])
        if self.symbol is not None:
            self.asset_type = bingx_asset_type(self.symbol, datatype=datatype)
            self.name = normalize_name(self.symbol, self.asset_type)
        self.price = None
        self.position_side = msg["positionSide"].upper()
        self.size = float(msg["positionAmt"])
        self.side = msg["positionSide"].upper()  # since bingx only has hedge mode #"LONG" if self.size > 0 else "SHORT"
        self.position_value = float(msg["unrealizedProfit"])
        self.entry_price = float(msg["avgPrice"])
        self.leverage = float(msg["leverage"])
        self.position_margin = float(msg["initialMargin"])
        self.realized_pnl = float(msg["realisedProfit"])
        self.unrealized_pnl = float(msg["unrealizedProfit"])
        self.is_isolated = msg["isolated"]
        self.auto_add_margin = None
        self.liq_price = None
        self.bust_price = None
        self.initial_margin = float(msg["initialMargin"])
        self.maintenance_margin = None
        self.exchange = "BingX"

    def from_bingx_to_form(self, msg: dict[str, Any], datatype: str | None = None) -> dict[str, Any]:
        self.from_bingx_format(msg, datatype=datatype)
        return self.to_general_form()

    def from_okx_format(self, msg: dict[str, Any]) -> None:
        # print(f">>>>>>> msg:{msg}")
        # >>>>>>> msg:{'adl': '5', 'availPos': '1', 'avgPx': '0.0905', 'baseBal': '', 'baseBorrowed': '', 'baseInterest': '', 'bePx': '', 'bizRefId': '', 'bizRefType': '', 'cTime': '1720698008696', 'ccy': 'BTC', 'clSpotInUseAmt': '', 'closeOrderAlgo': [], 'deltaBS': '-0.007780689212659095', 'deltaPA': '-0.006334087089049863', 'fee': '-0.000002', 'fundingFee': '0', 'gammaBS': '-2.4882741211319316E-7', 'gammaPA': '-0.003191985539030362', 'idxPx': '63740.0', 'imr': '', 'instId': 'BTC-USD-240830-55000-C', 'instType': 'OPTION', 'interest': '', 'last': '0.131', 'lever': '', 'liab': '', 'liabCcy': '', 'liqPenalty': '0', 'liqPx': '', 'margin': '0.0024111761223571', 'markPx': '0.1446234821387017', 'maxSpotInUseAmt': '', 'mgnMode': 'isolated', 'mgnRatio': '1.378417633171292', 'mmr': '0.001746234821387', 'notionalUsd': '637.4', 'optVal': '-0.001446234821387', 'pendingCloseOrdLiabVal': '', 'pnl': '0', 'pos': '-1', 'posCcy': '', 'posId': '1616748673207209984', 'posSide': 'net', 'quoteBal': '', 'quoteBorrowed': '', 'quoteInterest': '', 'realizedPnl': '-0.000002', 'spotInUseAmt': '', 'spotInUseCcy': '', 'thetaBS': '0.20148199607886155', 'thetaPA': '6.1579400902632065E-6', 'tradeId': '3', 'uTime': '1720698008696', 'upl': '-0.000541234821387', 'uplLastPx': '-0.000405', 'uplRatio': '-0.5980495263944942', 'uplRatioLastPx': '-0.4475138121546962', 'usdPx': '63740', 'vegaBS': '-0.6655536530650441', 'vegaPA': '-1.0441760743165067E-5'}
        self.symbol = msg["instId"]
        if self.symbol is not None:
            self.asset_type = okx_asset_type(self.symbol)
            self.name = normalize_name(self.symbol, self.asset_type)
        self.price = None
        self.position_side = msg["posSide"].upper()
        self.size = float(msg["pos"])
        self.side = "LONG" if self.size > 0 else "SHORT"
        self.position_value = float(msg["notionalUsd"])
        self.entry_price = float(msg["avgPx"])
        self.leverage = float(msg["lever"]) if msg["lever"] else 1.0
        self.position_margin = float(msg["margin"])
        self.realized_pnl = None
        self.unrealized_pnl = float(msg["upl"])
        self.is_isolated = True if msg["mgnMode"] == "isolated" else False
        self.auto_add_margin = None
        self.liq_price = float(msg["liqPx"]) if msg["liqPx"] else None
        self.bust_price = None
        self.initial_margin = float(msg["imr"]) if msg["imr"] else 0.0
        self.maintenance_margin = float(msg["mmr"]) if msg["mmr"] else 0.0
        self.exchange = "OKX"

    def from_okx_to_form(self, msg: dict[str, Any], datatype: str | None = None) -> dict[str, Any]:
        self.from_okx_format(msg)
        return self.to_general_form()

    def from_eqonex_format(self, msg: dict[str, Any]) -> None:
        qty = float(change_scale(msg["quantity"], msg["quantity_scale"]))
        side = "LONG" if qty > 0 else "SHORT"

        self.symbol = str(msg["symbol"])
        if self.symbol is not None:
            self.asset_type = eqonex_asset_type(self.symbol)
            self.name = normalize_name(self.symbol, self.asset_type)
        self.price = None
        self.side = side
        self.size = abs(qty)
        self.position_value = abs(msg["usdValue"])
        self.entry_price = msg["usdAvgCostBasis"]
        self.leverage = msg["accountLeverage"]
        self.position_margin = msg["usdMarginAccountTotal"]
        self.realized_pnl = float(msg["usdRealized"])
        self.unrealized_pnl = float(msg["usdUnrealized"])
        self.is_isolated = False
        self.auto_add_margin = False
        self.liq_price = None
        self.bust_price = None
        self.initial_margin = None
        self.maintenance_margin = None
        self.exchange = "EQONEX"

    def from_eqonex_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_eqonex_format(msg)
        return self.to_general_form()

    def from_bybit_future_format(self, msg: dict[str, Any]) -> None:
        if self.symbol is not None:
            self.asset_type = future_asset_type(self.symbol, "future")
            self.name = normalize_name(self.symbol, self.asset_type)
        self.initial_margin = float(msg["position_margin"])
        self.side = "LONG" if msg["side"] == "Buy" else "SHORT"
        msg["position_idx"] = str(msg["position_idx"])
        self.position_side = "NET" if msg["position_idx"] == "0" else "LONG" if msg["position_idx"] == "1" else "SHORT"
        self.maintenance_margin = None
        self.is_isolated = msg["isolated"] if "isolated" in msg else self.is_isolated
        self.exchange = "BYBIT"

    def from_bybit_inverse_future_format(self, msg: dict[str, Any]) -> None:
        if self.symbol is not None:
            self.asset_type = future_asset_type(self.symbol, "inverse")
            self.name = normalize_name(self.symbol, self.asset_type)
        self.initial_margin = float(msg["position_margin"])
        self.side = "LONG" if msg["side"] == "Buy" else "SHORT"
        msg["position_idx"] = str(msg["position_idx"])
        self.position_side = "NET" if msg["position_idx"] == "0" else "LONG" if msg["position_idx"] == "1" else "SHORT"
        self.maintenance_margin = None
        self.is_isolated = msg["isolated"] if "isolated" in msg else self.is_isolated
        self.exchange = "BYBIT"

    def from_bybit_option_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = ASSET_TYPE_OPTION
        if self.symbol is not None:
            self.name = normalize_name(self.symbol, self.asset_type)
        self.initial_margin = float(msg["positionIM"])
        self.maintenance_margin = float(msg["positionMM"])
        self.entry_price = float(msg["entryPrice"])
        self.side = msg["side"]
        self.size = float(msg["size"])
        self.realized_pnl = float(msg["sessionRPL"]) if msg["sessionRPL"] else 0.0
        self.unrealized_pnl = float(msg["sessionUPL"]) if msg["sessionUPL"] else 0.0
        self.exchange = "BYBIT"

    def from_bybit_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if datatype == "linear":
            self.from_bybit_future_format(msg)
        elif datatype == "inverse":
            self.from_bybit_inverse_future_format(msg)
        elif datatype == "option":
            self.from_bybit_option_format(msg)
        else:
            self.from_bybit_future_format(msg)

        return self.to_general_form()

    def from_bybit_v2_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print(f"[PositionSchema] {msg}")
        # linear: {'positionIdx': 0, 'riskId': 11, 'riskLimitValue': '900000', 'symbol': 'ETHUSDT', 'side': 'Buy', 'size': '0.06', 'avgPrice': '1533.49', 'positionValue': '92.0094', 'tradeMode': 1, 'positionStatus': 'Normal', 'autoAddMargin': 0, 'leverage': '5', 'markPrice': '1498.61', 'liqPrice': '1234.46', 'bustPrice': '1226.80', 'positionMM': '0.36804', 'positionIM': '0.920094', 'tpslMode': 'Full', 'takeProfit': '0.00', 'stopLoss': '0.00', 'trailingStop': '0.00', 'unrealisedPnl': '-2.0928', 'cumRealisedPnl': '-23.83978056', 'createdTime': '1667977098917', 'updatedTime': '1676361600105'}
        # inverse: {'positionIdx': 0, 'riskId': 1, 'riskLimitValue': '150', 'symbol': 'BTCUSD', 'side': 'Buy', 'size': '13', 'avgPrice': '22114.10880142', 'positionValue': '0.00058786', 'tradeMode': 1, 'positionStatus': 'Normal', 'autoAddMargin': 0, 'leverage': '5', 'markPrice': '22110.51', 'liqPrice': '18506.00', 'bustPrice': '18428.50', 'positionMM': '0.00000353', 'positionIM': '0.00000588', 'tpslMode': 'Full', 'takeProfit': '0.00', 'stopLoss': '0.00', 'trailingStop': '0.00', 'unrealisedPnl': '-0.00000009', 'cumRealisedPnl': '-0.01088376', 'createdTime': '1653273368466', 'updatedTime': '1676448979130'}
        # option: {'symbol': 'BTC-14JUL23-30500-C', 'leverage': '', 'autoAddMargin': 0, 'avgPrice': '275', 'liqPrice': '', 'riskLimitValue': '', 'takeProfit': '', 'positionValue': '3.47092598', 'tpslMode': '', 'riskId': 0, 'trailingStop': '', 'unrealisedPnl': '0.72092598', 'markPrice': '347.09259856', 'adlRankIndicator': 0, 'cumRealisedPnl': '-0.09026502', 'positionMM': '0', 'createdTime': '1688970110353', 'positionIdx': 0, 'positionIM': '0', 'updatedTime': '1688970110356', 'side': 'Buy', 'bustPrice': '', 'positionBalance': '', 'size': '0.01', 'positionStatus': 'Normal', 'stopLoss': '', 'tradeMode': 0}
        self.symbol = str(msg["symbol"])
        self.asset_type = bybit_asset_type(self.symbol, datatype=datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.price = None
        self.position_side = (
            "net"
            if not msg["positionIdx"]
            else "long"
            if msg["positionIdx"] == 1
            else "short"
            if msg["positionIdx"] == 2
            else ""
        )
        self.size = float(msg["size"])  ### NOTE: exchange return positive number only
        self.side = ("LONG" if msg["side"].upper() == "BUY" else "SHORT") if self.size else None
        self.position_value = float(msg["positionValue"]) if msg["positionValue"] else None
        self.entry_price = (
            (float(msg["avgPrice"]) if msg["avgPrice"] else 0.0)
            if "avgPrice" in msg
            else (float(msg["entryPrice"]) if msg["entryPrice"] else 0.0)
        )
        self.leverage = float(msg["leverage"]) if msg["leverage"] else 1
        # self.realized_pnl = float(msg["cumRealisedPnl"]) if msg["cumRealisedPnl"] else 0.0
        self.realized_pnl = float(msg["curRealisedPnl"]) if msg["curRealisedPnl"] else 0.0
        self.unrealized_pnl = float(msg["unrealisedPnl"]) if msg["unrealisedPnl"] else 0.0
        self.is_isolated = True if msg["tradeMode"] else False  # 0:全倉，1：逐倉
        self.auto_add_margin = None
        self.liq_price = float(msg["liqPrice"]) if msg["liqPrice"] else None
        self.bust_price = float(msg["bustPrice"]) if msg["bustPrice"] else None
        self.initial_margin = float(msg["positionIM"]) if msg["positionIM"] else 0.0
        self.maintenance_margin = float(msg["positionMM"]) if msg["positionMM"] else 0.0
        self.position_margin = self.maintenance_margin + self.initial_margin
        self.exchange = "BYBIT"
        if datatype == "option":
            self.delta = float(msg["delta"]) if msg.get("delta") else 0.0
            self.gamma = float(msg["gamma"]) if msg.get("gamma") else 0.0
            self.vega = float(msg["vega"]) if msg.get("vega") else 0.0
            self.theta = float(msg["theta"]) if msg.get("theta") else 0.0

    def from_bybit_v2_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        self.from_bybit_v2_format(msg, datatype)
        return self.to_general_form()

    def from_dydx_v3_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        # print(f"[PositionSchema] {msg}")
        # [PositionSchema] {'market': 'ETH-USD', 'status': 'OPEN', 'side': 'LONG', 'size': '2.234', 'maxSize': '2.234', 'entryPrice': '1811.300000', 'exitPrice': '0.000000',
        #           'unrealizedPnl': '26.258436', 'realizedPnl': '-1.361289', 'createdAt': '2023-11-01T13:28:08.546Z', 'closedAt': None, 'sumOpen': '2.234', 'sumClose': '0', 'netFunding': '-1.361289'}
        self.symbol = msg["market"]
        self.asset_type = ASSET_TYPE_PERPETUAL
        self.name = normalize_name(self.symbol, self.asset_type)
        self.price = None
        self.position_side = "net"
        self.size = float(msg["size"])  ### NOTE: exchange return positive number only
        self.side = msg["side"].upper()
        self.position_value = None
        self.entry_price = float(msg["entryPrice"])
        self.leverage = None
        self.realized_pnl = float(msg["realizedPnl"])
        self.unrealized_pnl = float(msg["unrealizedPnl"])
        self.is_isolated = False
        self.auto_add_margin = None
        self.liq_price = None
        self.bust_price = None
        self.initial_margin = None
        self.maintenance_margin = None
        self.position_margin = None
        self.exchange = "DYDX"
        return self.to_general_form()

    def from_binance_future_format(self, msg: dict[str, Any]) -> None:
        # print(msg,'????')
        # {'symbol': 'BTCUSDT', 'initialMargin': '0.87079107', 'maintMargin': '0.17415821', 'unrealizedProfit': '-0.02627850', 'positionInitialMargin': '0.87079107', 'openOrderInitialMargin': '0', 'leverage': '20', 'isolated': False, 'entryPrice': '17442.1', 'maxNotional': '250000', 'positionSide': 'BOTH', 'positionAmt': '0.001', 'notional': '17.41582149', 'isolatedWallet': '0', 'updateTime': 1668151269528, 'bidNotional': '0', 'askNotional': '0'} ????
        size = float(msg["positionAmt"])
        self.symbol = str(msg["symbol"])
        self.asset_type = future_asset_type(self.symbol, "future")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.price = None
        self.side = "LONG" if size >= 0 else "SHORT"
        self.size = abs(size)
        self.position_value = None
        self.entry_price = float(msg["entryPrice"])
        self.leverage = float(msg["leverage"])
        self.position_margin = float(msg["openOrderInitialMargin"])
        self.realized_pnl = None
        self.unrealized_pnl = float(msg["unrealizedProfit"])
        self.is_isolated = msg["isolated"]
        self.auto_add_margin = None
        self.liq_price = float(msg["liquidationPrice"]) if msg.get("liquidationPrice") else None
        self.bust_price = None
        self.initial_margin = float(msg["initialMargin"])
        self.maintenance_margin = float(msg["maintMargin"])
        self.position_side = "NET" if msg["positionSide"] == "BOTH" else msg["positionSide"]
        self.exchange = "BINANCE"

    def from_binance_inverse_future_format(self, msg: dict[str, Any]) -> None:
        # print(msg,'??inverse??')
        size = float(msg["positionAmt"])
        self.symbol = str(msg["symbol"])
        self.asset_type = future_asset_type(self.symbol, "inverse")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.price = None
        self.side = "LONG" if size >= 0 else "SHORT"
        self.size = abs(size)
        self.position_value = None
        self.entry_price = float(msg["entryPrice"])
        self.leverage = float(msg["leverage"])
        self.position_margin = float(msg["openOrderInitialMargin"])
        self.realized_pnl = None
        self.unrealized_pnl = float(msg["unrealizedProfit"])
        self.is_isolated = msg["isolated"]
        self.auto_add_margin = None
        self.liq_price = float(msg["liquidationPrice"]) if msg.get("liquidationPrice") else None
        self.bust_price = None
        self.initial_margin = float(msg["initialMargin"])
        self.maintenance_margin = float(msg["maintMargin"])
        self.position_side = "NET" if msg["positionSide"] == "BOTH" else msg["positionSide"]
        self.exchange = "BINANCE"

    def from_binance_option_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = ASSET_TYPE_OPTION
        self.name = normalize_name(self.symbol, self.asset_type)
        self.price = msg["markPrice"]
        self.side = msg["positionSide"]
        self.size = msg["quantity"]
        self.position_value = None
        self.entry_price = msg["entryPrice"]
        self.leverage = msg["leverage"]
        self.position_margin = msg["openOrderInitialMargin"]
        self.realized_pnl = None
        self.unrealized_pnl = float(msg["unrealizedPNL"])
        self.is_isolated = None
        self.auto_add_margin = None
        self.liq_price = None
        self.bust_price = None
        self.initial_margin = None
        self.maintenance_margin = None
        self.exchange = "BINANCE"

    def from_binance_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if datatype == "linear":
            self.from_binance_future_format(msg)
        elif datatype == "inverse":
            self.from_binance_inverse_future_format(msg)
        elif datatype == "option":
            self.from_binance_option_format(msg)
        else:
            self.from_binance_spot_format(msg)

        return self.to_general_form()

    def from_binance_v2_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print(f">>> msg:{msg}")
        # >>> msg:{'symbol': 'BTCUSDT', 'positionAmt': '0.009', 'entryPrice': '43371.83333334', 'breakEvenPrice': '43389.18206667', 'markPrice': '42814.17954404', 'unRealizedProfit': '-5.01888410', 'liquidationPrice': '0', 'leverage': '20', 'maxNotionalValue': '20000000', 'marginType': 'cross', 'isolatedMargin': '0.00000000', 'isAutoAddMargin': 'false', 'positionSide': 'BOTH', 'notional': '385.32761589', 'isolatedWallet': '0', 'updateTime': 1705545989899, 'isolated': False, 'adlQuantile': 1}
        # >>> msg:{'symbol': 'BTCUSDT', 'initialMargin': '35.71690321', 'maintMargin': '1.54215225', 'unrealizedProfit': '-4.80843565', 'positionInitialMargin': '19.27690321', 'openOrderInitialMargin': '16.44000000', 'leverage': '20', 'isolated': False, 'entryPrice': '43371.83333334', 'breakEvenPrice': '43389.18206667', 'maxNotional': '20000000', 'positionSide': 'BOTH', 'positionAmt': '0.009', 'notional': '385.53806434', 'isolatedWallet': '0', 'updateTime': 1705545989899, 'bidNotional': '328.80000000', 'askNotional': '329.70000000'}
        # inverse >>> msg:{'symbol': 'BTCUSD_PERP', 'positionAmt': '-30', 'entryPrice': '44992.28470984', 'markPrice': '42752.33334524', 'unRealizedProfit': '0.00349350', 'liquidationPrice': '0', 'leverage': '20', 'maxQty': '50', 'marginType': 'cross', 'isolatedMargin': '0.00000000', 'isAutoAddMargin': 'false', 'positionSide': 'BOTH', 'notionalValue': '-0.07017160', 'isolatedWallet': '0', 'updateTime': 1705559442871, 'breakEvenPrice': '44837.47687876'}
        size = float(msg["positionAmt"])
        self.symbol = str(msg["symbol"])
        self.asset_type = binance_asset_type(self.symbol, datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.price = None
        self.side = "LONG" if size >= 0 else "SHORT"
        self.size = abs(size)
        self.position_value = None
        self.entry_price = float(msg["entryPrice"])
        self.leverage = float(msg["leverage"]) if "leverage" in msg else None
        # self.position_margin = float(msg["openOrderInitialMargin"])
        self.position_margin = float(msg["isolatedMargin"]) if "isolatedMargin" in msg else None
        self.realized_pnl = None
        # self.unrealized_pnl = float(msg["unrealizedProfit"])
        self.unrealized_pnl = float(msg["unRealizedProfit"])
        # self.is_isolated = msg["isolated"]
        self.is_isolated = msg["marginType"] == "isolated"
        self.auto_add_margin = None
        self.liq_price = float(msg["liquidationPrice"]) if msg.get("liquidationPrice") else None
        self.bust_price = None
        # self.initial_margin = float(msg["initialMargin"])
        # self.initial_margin = float(msg["notional"]) / self.leverage
        # self.maintenance_margin = float(msg["maintMargin"])
        self.position_side = "NET" if msg["positionSide"] == "BOTH" else msg["positionSide"]
        self.exchange = "BINANCE"

    def from_binance_v2_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        # print(f">>> msg:{msg}")
        if not (float(msg["positionAmt"]) and float(msg["entryPrice"])):
            return {}
        self.from_binance_v2_format(msg, datatype)
        return self.to_general_form()

    def from_binance_spot_format(self, msg: dict[str, Any]) -> None:
        # Placeholder method for spot format - positions typically don't exist for spot trading
        # This method is called by from_binance_to_form but spot positions are rare
        pass


@dataclass
class SendOrderResultSchema(BaseData):
    asset_type: str | None = None
    order_id: int | str | None = None
    created_at: datetime | None = None
    symbol: str | None = None
    name: str | None = None
    price: None | float = None  # market price show None
    side: str | None = None
    size: float | None = None
    status: str | None = None
    type: str | None = None
    ioc: bool | None = None
    post_only: bool | None = None
    ori_order_id: int | str | None = None
    time_in_force: str | None = None
    exchange: str | None = None

    def to_general_form(self) -> dict[str, Any]:
        return {
            "asset_type": self.asset_type,
            "code": self.symbol,
            "symbol": self.name,
            "ori_order_id": self.ori_order_id,
            "order_id": self.order_id,
            "datetime": self.created_at,
            "exchange": self.exchange,
        }

    def from_bitget_format(self, msg: dict[str, Any], symbol: str, datatype: str) -> None:
        self.created_at = generate_datetime()
        self.order_id = msg["clientOrderId"]
        self.symbol = symbol
        self.asset_type = bitget_asset_type(self.symbol, datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.ori_order_id = msg["orderId"]
        self.exchange = "BITGET"

    def from_bitget_to_form(self, msg: dict[str, Any], symbol: str, datatype: str) -> dict[str, Any]:
        self.from_bitget_format(msg, symbol, datatype)
        return self.to_general_form()

    def from_bitget_v2_format(self, msg: dict[str, Any], symbol: str, datatype: str) -> None:
        # {'clientOid': 'x-240422-182108918-00001', 'orderId': '1166248659734134787'}, msg='query ok') | payload:{'symbol': 'BTCUSDT', 'productType': 'SUSDT-FUTURES', 'marginCoin': 'USDT', 'side': 'buy', 'size': 0.002, 'clientOid': 'x-240422-182108918-00001', 'timeInForceValue': 'normal', 'marginMode': 'crossed', 'tradeSide': 'open', 'orderType': 'limit', 'price': 54000}
        self.created_at = generate_datetime()
        self.order_id = msg["clientOid"]
        self.symbol = symbol
        self.asset_type = bitget_asset_type(self.symbol, datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.ori_order_id = msg["orderId"]
        self.exchange = "BITGET"

    def from_bitget_v2_to_form(self, msg: dict[str, Any], symbol: str, datatype: str) -> dict[str, Any]:
        self.from_bitget_v2_format(msg, symbol, datatype)
        return self.to_general_form()

    def from_bybit_v2_format(self, msg: dict[str, Any], symbol: str, datatype: str) -> None:
        self.created_at = generate_datetime()
        self.order_id = msg["orderLinkId"]
        self.symbol = symbol
        self.asset_type = bybit_asset_type(self.symbol, datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.ori_order_id = msg["orderId"]
        self.exchange = "BYBIT"

    def from_bybit_v2_to_form(self, msg: dict[str, Any], symbol: str, datatype: str) -> dict[str, Any]:
        self.from_bybit_v2_format(msg, symbol, datatype)
        return self.to_general_form()

    def from_dydx_v3_to_form(
        self,
        msg: dict[str, Any],
    ) -> dict[str, Any]:
        # {'id': '7cab8a3647b198a77ff969374f41b73c2c8cba613e336932ff6d3102744f634', 'clientId': 'aio-231102-153843684-00001', 'accountId': 'ee7ef06a-da05-5ea9-bb2a-738752d11ad1', 'market': 'ETH-USD', 'side': 'BUY', 'price': '1740.9', 'triggerPrice': None, 'trailingPercent': None, 'size': '0.9', 'reduceOnlySize': None, 'remainingSize': '0.9', 'type': 'LIMIT', 'createdAt': '2023-11-02T07:38:43.861Z', 'unfillableAt': None, 'expiresAt': '2023-11-03T07:38:43.000Z', 'status': 'PENDING', 'timeInForce': 'GTT', 'postOnly': False, 'reduceOnly': False, 'cancelReason': None}}
        self.created_at = generate_datetime()
        self.order_id = msg["clientId"]
        self.symbol = msg["market"]
        self.asset_type = ASSET_TYPE_PERPETUAL
        self.name = normalize_name(self.symbol, self.asset_type)
        self.ori_order_id = msg["id"]
        self.exchange = "DYDX"
        return self.to_general_form()

    def from_bingx_format(self, msg: dict[str, Any], symbol: str, datatype: str) -> None:
        self.created_at = generate_datetime()
        self.order_id = None
        self.symbol = symbol
        self.asset_type = bingx_asset_type(self.symbol, datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.ori_order_id = msg["orderId"]
        self.exchange = "BingX"

    def from_bingx_to_form(self, msg: dict[str, Any], symbol: str, datatype: str) -> dict[str, Any]:
        self.from_bingx_format(msg, symbol, datatype)
        return self.to_general_form()

    def from_okx_format(self, msg: dict[str, Any], symbol: str) -> None:
        self.created_at = generate_datetime()
        self.order_id = msg["clOrdId"]
        self.symbol = symbol
        self.asset_type = okx_asset_type(self.symbol)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.ori_order_id = msg["ordId"]
        self.exchange = "OKX"

    def from_okx_to_form(self, msg: dict[str, Any], symbol: str) -> dict[str, Any]:
        self.from_okx_format(msg, symbol)
        return self.to_general_form()

    def from_ftx_format(self, msg: dict[str, Any]) -> None:
        self.created_at = isoformat_to_datetime(msg["createdAt"])
        self.ori_order_id = msg["id"]
        self.symbol = msg["market"]
        if self.symbol is not None:
            self.asset_type = ftx_asset_type(self.symbol)
            self.name = normalize_name(self.symbol, self.asset_type)
        self.price = msg["price"]
        self.size = msg["size"]
        self.status = msg["status"]
        self.order_type = msg["type"]
        self.ioc = msg["ioc"]
        self.post_only = msg["postOnly"]
        self.order_id = msg["clientId"]
        self.filled_size = msg["filledSize"]
        self.exchange = "FTX"

    def from_ftx_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.from_ftx_format(msg)
        return self.to_general_form()

    def from_eqonex_format(self, msg: dict[str, Any], symbol: str) -> None:
        self.created_at = generate_datetime()
        self.order_id = msg["clOrdId"]
        self.symbol = symbol
        self.asset_type = eqonex_asset_type(self.symbol)
        self.name = normalize_name(self.symbol, self.asset_type)
        # self.price = msg['price']
        # self.size = msg['size']
        # self.status = msg['status']
        # self.order_type = msg['type']
        # self.ioc = msg['ioc']
        # self.post_only = msg['postOnly']
        self.ori_order_id = msg["id"]
        self.exchange = "EQONEX"

    def from_eqonex_to_form(self, msg: dict[str, Any], symbol: str) -> dict[str, Any]:
        self.from_eqonex_format(msg, symbol)
        return self.to_general_form()

    def from_bybit_spot_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = ASSET_TYPE_SPOT
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = timestamp_to_datetime(float(msg["transactTime"]))
        self.order_id = msg["orderLinkId"]
        self.price = msg["price"]
        self.size = msg["origQty"]
        self.side = msg["side"]
        self.status = order_status_map(msg["status"], "BYBIT")
        self.order_type = msg["type"]
        self.ori_order_id = msg["orderId"]
        self.time_in_force = msg["timeInForce"]
        self.exchange = "BYBIT"

    def from_bybit_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = future_asset_type(self.symbol, "future")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = isoformat_to_datetime(msg["created_time"])
        self.order_id = msg["order_link_id"]
        self.price = msg["price"]
        self.side = msg["side"]
        self.size = msg["qty"]
        self.status = order_status_map(msg["order_status"], "BYBIT")
        self.order_type = msg["order_type"]
        self.ori_order_id = msg["order_id"]
        self.time_in_force = msg["time_in_force"]
        self.exchange = "BYBIT"

    def from_bybit_inverse_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = future_asset_type(self.symbol, "inverse")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = isoformat_to_datetime(msg["created_at"])
        self.order_id = msg["order_link_id"]
        self.price = msg["price"]
        self.side = msg["side"]
        self.size = msg["qty"]
        self.status = order_status_map(msg["order_status"], "BYBIT")
        self.order_type = msg["order_type"]
        self.ori_order_id = msg["order_id"]
        self.time_in_force = msg["time_in_force"]

    def from_bybit_option_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = ASSET_TYPE_OPTION
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = isoformat_to_datetime(msg["created_at"])
        self.order_id = msg["order_link_id"]
        self.price = msg["price"]
        self.side = msg["side"]
        self.size = msg["qty"]
        self.status = order_status_map(msg["order_status"], "BYBIT")
        self.order_type = msg["order_type"]
        self.ori_order_id = msg["order_id"]
        self.time_in_force = msg["time_in_force"]
        self.exchange = "BYBIT"

    def from_bybit_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if datatype == "spot":
            self.from_bybit_spot_format(msg)
        elif datatype == "linear":
            self.from_bybit_future_format(msg)
        elif datatype == "inverse":
            self.from_bybit_inverse_future_format(msg)
        elif datatype == "option":
            self.from_bybit_option_format(msg)
        else:
            self.from_bybit_spot_format(msg)

        return self.to_general_form()

    def from_binance_spot_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.symbol = str(msg["symbol"])
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = timestamp_to_datetime(float(msg["transactTime"]))
        self.ori_order_id = msg["clientOrderId"]
        self.price = msg["price"]
        self.size = msg["origQty"]
        self.side = msg["side"]
        self.status = order_status_map(msg["status"], "BINANCE")
        self.order_type = msg["type"]
        self.ori_order_id = msg["orderId"]
        self.time_in_force = msg["timeInForce"]
        self.exchange = "BINANCE"

    def from_binance_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = future_asset_type(self.symbol, "future")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = timestamp_to_datetime(float(msg["updateTime"]))
        self.order_id = msg["clientOrderId"]
        self.price = msg["price"]
        self.size = msg["origQty"]
        self.side = msg["side"]
        self.status = order_status_map(msg["status"], "BINANCE")
        self.order_type = msg["type"]
        self.ori_order_id = msg["orderId"]
        self.time_in_force = msg["timeInForce"]
        self.exchange = "BINANCE"

    def from_binance_inverse_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = future_asset_type(self.symbol, "inverse")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = timestamp_to_datetime(float(msg["updateTime"]))
        self.order_id = msg["clientOrderId"]
        self.price = msg["price"]
        self.size = msg["origQty"]
        self.side = msg["side"]
        self.status = order_status_map(msg["status"], "BINANCE")
        self.order_type = msg["type"]
        self.ori_order_id = msg["orderId"]
        self.time_in_force = msg["timeInForce"]
        self.exchange = "BINANCE"

    def from_binance_option_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = ASSET_TYPE_OPTION
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = timestamp_to_datetime(float(msg["time"]))
        self.order_id = msg["id"]
        self.price = msg["price"]
        self.size = msg["quantity"]
        self.side = msg["side"]
        self.status = order_status_map(msg["status"], "BINANCE")
        self.order_type = msg["type"]
        self.ori_order_id = msg["clientOrderId"]
        self.time_in_force = msg["timeInForce"]
        self.exchange = "BINANCE"

    def from_binance_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if datatype == "spot":
            self.from_binance_spot_format(msg)
        elif datatype == "linear":
            self.from_binance_future_format(msg)
        elif datatype == "inverse":
            self.from_binance_inverse_future_format(msg)
        elif datatype == "option":
            self.from_binance_option_format(msg)
        else:
            self.from_binance_spot_format(msg)

        return self.to_general_form()

    def from_binance_v2_format(self, msg: dict[str, Any], datatype: str) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = binance_asset_type(self.symbol, datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = timestamp_to_datetime(float(msg["time"]))
        self.updated_at = timestamp_to_datetime(float(msg["updateTime"]))
        self.ori_order_id = msg["orderId"]
        self.price = float(msg["price"])
        self.size = float(msg["origQty"])
        self.status = msg["status"]
        self.order_type = msg["type"]
        self.side = msg["side"]
        self.order_id = msg["clientOrderId"]
        self.filled_size = float(msg["executedQty"])
        self.position_side = (
            ("NET" if msg["positionSide"] == "BOTH" else msg["positionSide"]) if "positionSide" in msg else "NET"
        )
        self.exchange = "BINANCE"

    def from_binance_v2_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_binance_v2_format(msg, datatype)

        return self.to_general_form()


@dataclass
class OrderSchema(BaseData):
    asset_type: str | None = None
    ori_order_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    symbol: str | None = None
    name: str | None = None
    price: None | float | int | str = None  # market price show None
    side: str | None = None
    position_side: str | None = None
    size: float | None = None
    status: str | None = None
    order_type: str | None = None
    ioc: bool | None = None
    post_only: bool | None = None
    order_id: str | None = None
    exchange: str | None = None
    filled_size: float | None = None

    def to_general_form(self) -> dict[str, Any]:
        return {
            "asset_type": self.asset_type,
            "code": self.symbol,
            "symbol": self.name,
            "exchange": self.exchange,
            "ori_order_id": str(self.ori_order_id),
            "order_id": str(self.order_id) if self.order_id else self.order_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "price": self.price,
            "quantity": self.size,
            "order_type": self.order_type,
            "side": self.side,
            "position_side": self.position_side,
            "status": order_status_map(self.status, self.exchange),
            "executed_quantity": self.filled_size,
        }

    def from_bitget_format(self, msg: dict[str, Any], datatype: str) -> None:
        # spot: {'accountId': '2825475201', 'symbol': 'ETHUSDT_SPBL', 'orderId': '1029583357171109888', 'clientOrderId': 'x-230411-152201619-00001', 'price': '1700', 'quantity': '0.005', 'orderType': 'limit', 'side': 'buy', 'status': 'cancelled', 'fillPrice': '', 'fillQuantity': '0', 'fillTotalAmount': '0', 'enterPointSource': 'API', 'feeDetail': '', 'cTime': '1681197721760'}
        #       {'accountId': '2825475201', 'symbol': 'ETHUSDT_SPBL', 'orderId': '1029590326820917248', 'clientOrderId': 'x-230411-154943236-00001', 'price': '', 'quantity': '5.7677', 'orderType': 'market', 'side': 'buy', 'status': 'full_fill', 'fillPrice': '1922.54', 'fillQuantity': '0.003', 'fillTotalAmount': '5.76762', 'enterPointSource': 'API', 'feeDetail': '{"ETH":{"deduction":false,"feeCoinCode":"ETH","totalDeductionFee":0,"totalFee":-0.000003000000}
        # linear: {'symbol': 'BTCUSDT_UMCBL', 'size': 0.001, 'orderId': '1030293002620350466', 'clientOid': '1030293002620350469', 'filledQty': 0.0, 'fee': 0.0, 'price': 29000.0, 'state': 'new', 'side': 'open_long', 'timeInForce': 'normal', 'totalProfits': 0.0, 'posSide': 'long', 'marginCoin': 'USDT', 'filledAmount': 0.0, 'orderType': 'limit', 'leverage': '20', 'marginMode': 'crossed', 'reduceOnly': False, 'enterPointSource': 'WEB', 'tradeSide': 'open_long', 'holdMode': 'double_hold', 'cTime': '1681366914420', 'uTime': '1681366914420'}]
        # {'symbol': 'BTCUSDT_UMCBL', 'size': 0.001, 'orderId': '1030292933431111683', 'clientOid': '1030292933468860416', 'filledQty': 0.001, 'fee': -0.0180501, 'price': None, 'priceAvg': 30083.5, 'state': 'filled', 'side': 'open_long', 'timeInForce': 'normal', 'totalProfits': 0.0, 'posSide': 'long', 'marginCoin': 'USDT', 'filledAmount': 30.0835, 'orderType': 'market', 'leverage': '20', 'marginMode': 'crossed', 'reduceOnly': False, 'enterPointSource': 'WEB', 'tradeSide': 'open_long', 'holdMode': 'double_hold', 'cTime': '1681366897933', 'uTime': '1681366898068'}
        # linear ws: {"accFillSz":"0","cTime":1681809268419,"clOrdId":"1032148369767772160","eps":"WEB","force":"normal","hM":"single_hold","instId":"SBTCSUSDT_SUMCBL","lever":"7","low":false,"notionalUsd":"290.04","ordId":"1032148369730023425","ordType":"limit","orderFee":[{"feeCcy":"SUSDT","fee":"0"}],"posSide":"net","px":"29004","side":"buy","status":"new","sz":"0.01","tS":"buy_single","tdMode":"isolated","tgtCcy":"SUSDT","uTime":1681809268419}

        self.created_at = timestamp_to_datetime(int(msg["cTime"]))
        self.ori_order_id = msg["orderId"] if "orderId" in msg else msg["ordId"]
        self.symbol = str(msg["symbol"]) if "symbol" in msg else msg["instId"]
        self.asset_type = bitget_asset_type(self.symbol, datatype=datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.price = (
            float(msg["price"])
            if "price" in msg and msg["price"]
            else float(msg["fillPrice"])
            if "fillPrice" in msg
            else float(msg["priceAvg"])
            if "priceAvg" in msg
            else float(msg["px"])
        )
        self.order_type = msg["orderType"] if "orderType" in msg else msg["ordType"]
        self.size = (
            (float(msg["quantity"]) if self.order_type == "limit" else float(msg["quantity"]) / self.price)
            if datatype == "spot"
            else float(msg["size"])
            if "size" in msg
            else float(msg["sz"])
        )  # Order quantity (base coin when orderType=limit; quote coin when orderType=market)
        self.status = order_status_map(msg["status"] if "status" in msg else msg["state"], exchange="bitget")
        self.side = (
            msg["side"].upper().replace("_SINGLE", "")
            if msg["side"].upper() in ["BUY", "SELL", "BUY_SINGLE", "SELL_SINGLE"]
            else (
                "BUY"
                if msg["side"] == "open_long"
                else "SELL"
                if msg["side"] == "open_short"
                else "SELL"
                if msg["side"] == "close_long"
                else "BUY"
            )
        )  #  linear 'side': 'open_long'
        self.ioc = None
        self.post_only = None
        self.order_id = (
            msg["clientOrderId"]
            if "clientOrderId" in msg
            else msg["clientOid"]
            if "clientOid" in msg
            else msg["clOrdId"]
        )
        self.filled_size = (
            float(msg["fillQuantity"])
            if "fillQuantity" in msg
            else float(msg["filledQty"])
            if "filledQty" in msg
            else float(msg["accFillSz"])
        )
        self.exchange = "BITGET"

    def from_bitget_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_bitget_format(msg, datatype)
        return self.to_general_form()

    def from_bitget_v2_format(self, msg: dict[str, Any], datatype: str) -> None:
        # linear : {'symbol': 'BTCUSDT', 'size': '0.002', 'orderId': '1166248854945431557', 'clientOid': 'x-240422-182155454-00001', 'baseVolume': '0', 'fee': '0', 'price': '54000', 'priceAvg': '', 'state': 'canceled', 'side': 'buy', 'force': 'gtc', 'totalProfits': '0', 'posSide': 'long', 'marginCoin': 'USDT', 'presetStopSurplusPrice': '', 'presetStopLossPrice': '', 'quoteVolume': '0', 'orderType': 'limit', 'leverage': '20', 'marginMode': 'crossed', 'reduceOnly': 'NO', 'enterPointSource': 'API', 'tradeSide': 'open', 'posMode': 'hedge_mode', 'orderSource': 'normal', 'newTradeSide': 'open', 'cTime': '1713781315553', 'uTime': '1713782419463'}
        self.created_at = timestamp_to_datetime(int(msg["cTime"]))
        self.ori_order_id = msg["orderId"] if "orderId" in msg else msg["ordId"]
        self.symbol = str(msg["symbol"]) if "symbol" in msg else msg["instId"]
        self.asset_type = bitget_asset_type(self.symbol, datatype=datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.price = (
            float(msg["price"])
            if "price" in msg and msg["price"]
            else float(msg["fillPrice"])
            if "fillPrice" in msg
            else float(msg["priceAvg"])
            if "priceAvg" in msg
            else float(msg["px"])
        )
        self.order_type = msg["orderType"] if "orderType" in msg else msg["ordType"]
        self.size = (
            (float(msg["quantity"]) if self.order_type == "limit" else float(msg["quantity"]) / self.price)
            if datatype == "spot"
            else float(msg["size"])
            if "size" in msg
            else float(msg["sz"])
        )  # Order quantity (base coin when orderType=limit; quote coin when orderType=market)
        self.status = order_status_map(msg["status"] if "status" in msg else msg["state"], exchange="bitget")
        self.side = (
            msg["side"].upper().replace("_SINGLE", "")
            if msg["side"].upper() in ["BUY", "SELL", "BUY_SINGLE", "SELL_SINGLE"]
            else (
                "BUY"
                if msg["side"] == "open_long"
                else "SELL"
                if msg["side"] == "open_short"
                else "SELL"
                if msg["side"] == "close_long"
                else "BUY"
            )
        )  #  linear 'side': 'open_long'
        self.ioc = None
        self.post_only = None
        self.order_id = (
            msg["clientOrderId"]
            if "clientOrderId" in msg
            else msg["clientOid"]
            if "clientOid" in msg
            else msg["clOrdId"]
        )
        # self.filled_size = float(msg["fillQuantity"]) if "fillQuantity" in msg else float(msg["filledQty"]) if "filledQty" in msg else float(msg["accFillSz"])
        self.filled_size = 0.0
        self.exchange = "BITGET"

    def from_bitget_v2_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_bitget_v2_format(msg, datatype)
        return self.to_general_form()

    def from_bybit_v2_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print(f"[OrderSchema] {msg}")
        # option: {'symbol': 'BTC-14JUL23-30500-C', 'orderType': 'Limit', 'orderLinkId': 'x-230710-142143598-00001', 'orderId': '9173cc37-b8f0-4d5b-b241-c790b654362a', 'cancelType': 'UNKNOWN', 'avgPrice': '', 'stopOrderType': '', 'lastPriceOnCreated': '', 'orderStatus': 'New', 'takeProfit': '', 'cumExecValue': '0', 'smpType': 'None', 'triggerDirection': 0, 'blockTradeId': '', 'isLeverage': '', 'rejectReason': 'EC_NoError', 'price': '200', 'orderIv': '0.27974322', 'createdTime': '1688970103785', 'tpTriggerBy': '', 'positionIdx': 0, 'timeInForce': 'GTC', 'leavesValue': '2', 'updatedTime': '1688970103794', 'side': 'Buy', 'smpGroup': 0, 'triggerPrice': '', 'cumExecFee': '0', 'leavesQty': '0.01', 'slTriggerBy': '', 'closeOnTrigger': False, 'placeType': 'price', 'cumExecQty': '0', 'reduceOnly': False, 'qty': '0.01', 'stopLoss': '', 'smpOrderId': '', 'triggerBy': ''}
        # option ws: {'category': 'option', 'symbol': 'BTC-14JUL23-30500-C', 'orderId': '294abbf1-396a-498a-9957-cca6bdc7c28c', 'orderLinkId': '', 'blockTradeId': '', 'side': 'Buy', 'positionIdx': 0, 'orderStatus': 'Filled', 'cancelType': 'UNKNOWN', 'rejectReason': 'EC_NoError', 'timeInForce': 'IOC', 'isLeverage': '', 'price': '1295', 'qty': '0.01', 'avgPrice': '280', 'leavesQty': '0', 'leavesValue': '0', 'cumExecQty': '0.01', 'cumExecValue': '2.8', 'cumExecFee': '0.09034445', 'orderType': 'Market', 'stopOrderType': '', 'orderIv': '', 'triggerPrice': '', 'takeProfit': '', 'stopLoss': '', 'triggerBy': '', 'tpTriggerBy': '', 'slTriggerBy': '', 'triggerDirection': 0, 'placeType': 'price', 'lastPriceOnCreated': '', 'closeOnTrigger': False, 'reduceOnly': False, 'smpGroup': 0, 'smpType': 'CancelTaker', 'smpOrderId': '', 'createdTime': '1688973912717', 'updatedTime': '1688973912726'}
        self.created_at = timestamp_to_datetime(int(msg["createdTime"]))
        self.updated_at = timestamp_to_datetime(int(msg["updatedTime"]))
        self.ori_order_id = msg["orderId"]
        self.symbol = str(msg["symbol"])
        self.asset_type = bybit_asset_type(self.symbol, datatype=datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        if "avgPrice" in msg and (msg["avgPrice"] and msg["avgPrice"] != "0"):
            self.price = float(msg["avgPrice"])
        else:
            self.price = float(msg["price"])
        self.side = msg["side"]
        self.size = float(msg["qty"])
        self.status = msg["orderStatus"]
        self.order_type = msg["orderType"]
        self.ioc = None
        self.post_only = None
        self.order_id = msg["orderLinkId"]
        self.filled_size = float(msg["cumExecQty"])
        self.position_side = "NET" if msg["positionIdx"] == 0 else "LONG" if msg["positionIdx"] == 1 else "SHORT"
        self.exchange = "BYBIT"

    def from_bybit_v2_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_bybit_v2_format(msg, datatype)
        return self.to_general_form()

    def from_dydx_v3_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        # print(f"[OrderSchema] {msg}")
        # {'id': '7e8df1d40d947a14d698da8795dfc083fdd4e776cf005bc4b4d07a27b2a171c', 'clientId': 'aio-231102-155747639-00001', 'accountId': 'ee7ef06a-da05-5ea9-bb2a-738752d11ad1', 'market': 'ETH-USD',
        # 'side': 'BUY', 'price': '1740.9', 'triggerPrice': None, 'trailingPercent': None, 'size': '0.9', 'reduceOnlySize': None, 'remainingSize': '0.9', 'type': 'LIMIT', 'createdAt': '2023-11-02T07:57:47.779Z',
        # 'unfillableAt': None, 'expiresAt': '2023-11-03T07:57:47.000Z', 'status': 'OPEN', 'timeInForce': 'GTT', 'postOnly': False, 'reduceOnly': False, 'cancelReason': None}
        self.created_at = isoformat_to_datetime(msg["createdAt"])
        self.ori_order_id = msg["id"]
        self.symbol = msg["market"]
        self.asset_type = ASSET_TYPE_PERPETUAL
        self.name = normalize_name(self.symbol, self.asset_type)
        self.price = float(msg["price"])
        self.side = msg["side"]
        self.size = float(msg["size"])
        self.status = msg["status"]
        self.order_type = msg["type"]
        self.ioc = None
        self.post_only = None
        self.order_id = msg["clientId"]
        self.filled_size = self.size - float(msg["remainingSize"])
        self.exchange = "DYDX"

        return self.to_general_form()

    def from_bingx_format(self, msg: dict[str, Any], datatype: str) -> None:
        self.created_at = timestamp_to_datetime(int(msg["time"]))
        self.ori_order_id = msg["orderId"]
        self.symbol = str(msg["symbol"])
        self.asset_type = bingx_asset_type(self.symbol, datatype=datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.price = float(msg["price"])
        self.size = float(msg["origQty"])
        self.status = msg["status"]
        self.order_type = msg["type"]
        self.ioc = None
        self.post_only = None
        self.order_id = None
        self.filled_size = float(msg["executedQty"])
        self.exchange = "BingX"

    def from_bingx_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_bingx_format(msg, datatype)
        return self.to_general_form()

    def from_okx_format(self, msg: dict[str, Any]) -> None:
        self.created_at = timestamp_to_datetime(int(msg["cTime"]))
        self.updated_at = timestamp_to_datetime(int(msg["uTime"]))
        self.ori_order_id = msg["ordId"]
        self.symbol = msg["instId"]
        self.asset_type = okx_asset_type(self.symbol)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.price = float(msg["px"]) if msg["px"] else float(msg["fillPx"]) if msg["fillPx"] else None
        self.size = float(msg["sz"])
        self.status = msg["state"]  # TODO:
        self.order_type = msg["ordType"]
        self.ioc = None
        self.post_only = None
        self.order_id = msg["clOrdId"]
        self.filled_size = float(msg["fillSz"])
        self.position_side = msg["posSide"].upper()
        self.exchange = "OKX"

    def from_okx_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.from_okx_format(msg)
        return self.to_general_form()

    def from_ftx_format(self, msg: dict[str, Any]) -> None:
        self.created_at = isoformat_to_datetime(msg["createdAt"])
        self.ori_order_id = msg["id"]
        self.symbol = msg["market"]
        if self.symbol is not None:
            self.asset_type = ftx_asset_type(self.symbol)
            self.name = normalize_name(self.symbol, self.asset_type)
        self.price = msg["price"]
        self.size = msg["size"]
        self.status = msg["status"]
        self.order_type = msg["type"]
        self.ioc = msg["ioc"]
        self.post_only = msg["postOnly"]
        self.order_id = msg["clientId"]
        self.filled_size = msg["filledSize"]
        self.exchange = "FTX"

    def from_ftx_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.from_ftx_format(msg)
        return self.to_general_form()

    def from_eqonex_format(self, msg: dict[str, Any]) -> None:
        timestamp = msg["timeStamp"] if "timeStamp" in msg else msg["timestamp"]
        price = msg["price"] if "price" in msg else msg["getPrice"]
        quantity = msg["quantity"] if "quantity" in msg else msg["oderQty"]
        quantity_scale = msg["quantity_scale"] if "quantity_scale" in msg else msg["orderQty_scale"]
        self.created_at = isoformat_to_datetime(timestamp)
        self.ori_order_id = msg["orderId"]
        self.symbol = str(msg["symbol"])
        self.asset_type = eqonex_asset_type(self.symbol)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.price = float(change_scale(price, msg["price_scale"]))
        self.size = float(change_scale(quantity, quantity_scale))
        self.status = msg["ordStatus"]
        self.order_type = msg["ordType"]
        # self.ioc =
        # self.post_only =
        self.order_id = msg["clOrdId"]
        self.filled_size = float(change_scale(msg["cumQty"], msg["cumQty_scale"]))
        self.exchange = "EQONEX"

    def from_eqonex_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        symbol = msg["symbol"]
        if datatype == "spot":
            if "[" in symbol and "]" in symbol:
                return {}
        elif datatype == "linear":
            if not ("[" in symbol and "]" in symbol):
                return {}
        self.from_eqonex_format(msg)
        return self.to_general_form()

    def from_bybit_spot_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.created_at = timestamp_to_datetime(float(msg["time"]))
        self.ori_order_id = msg["orderId"]
        self.symbol = str(msg["symbol"])
        self.name = normalize_name(self.symbol, self.asset_type)
        self.price = float(msg["price"])
        self.size = float(msg["origQty"])
        self.status = msg["status"]
        self.order_type = msg["type"]
        self.order_id = msg["orderLinkId"]
        self.filled_size = msg["executedQty"]
        self.exchange = "BYBIT"

    def from_bybit_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = future_asset_type(self.symbol, "future")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = (
            isoformat_to_datetime(msg["created_time"])
            if "created_time" in msg
            else isoformat_to_datetime(msg["create_time"])
        )
        self.ori_order_id = msg["order_id"]
        self.price = float(msg["price"])
        self.size = float(msg["qty"])
        self.status = msg["order_status"]
        self.order_type = msg["order_type"]
        self.order_id = msg["order_link_id"]
        self.filled_size = msg["cum_exec_qty"]
        self.exchange = "BYBIT"

    def from_bybit_inverse_future_format(self, msg: dict[str, Any]) -> None:
        # print(msg,'????')
        self.symbol = str(msg["symbol"])
        self.asset_type = future_asset_type(self.symbol, "inverse")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = (
            isoformat_to_datetime(msg["created_at"]) if "created_at" in msg else isoformat_to_datetime(msg["timestamp"])
        )
        self.ori_order_id = msg["order_id"]
        self.price = float(msg["price"])
        self.size = float(msg["qty"])
        self.status = msg["order_status"]
        self.order_type = msg["order_type"]
        self.order_id = msg["order_link_id"]
        self.filled_size = msg["cum_exec_qty"]
        self.exchange = "BYBIT"

    def from_bybit_option_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = ASSET_TYPE_OPTION
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = timestamp_to_datetime(
            msg["updateTimeStamp"]
        )  # if "createdAt" in msg else timestamp_to_datetime(msg["createTimeStamp"])
        self.ori_order_id = msg["orderId"]
        self.price = float(msg["price"]) if "price" in msg else float(msg["orderPrice"])
        self.side = msg["side"]
        self.size = float(msg["qty"]) if "qty" in msg else float(msg["orderAllSize"])
        self.status = msg["orderStatus"]
        self.order_type = msg["orderType"]
        self.order_id = msg["orderLinkId"] if "orderLinkId" in msg else ""
        self.filled_size = msg["cumExecQty"] if "cumExecQty" in msg else float(msg["orderFilledSize"])
        self.exchange = "BYBIT"

    def from_bybit_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if datatype == "spot":
            self.from_bybit_spot_format(msg)
        elif datatype == "linear":
            self.from_bybit_future_format(msg)
        elif datatype == "inverse":
            self.from_bybit_inverse_future_format(msg)
        elif datatype == "option":
            self.from_bybit_option_format(msg)
        else:
            self.from_bybit_spot_format(msg)

        return self.to_general_form()

    def from_binance_spot_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.symbol = str(msg["symbol"])
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = timestamp_to_datetime(float(msg["time"]))
        self.ori_order_id = msg["clientOrderId"]
        self.price = float(msg["price"])
        self.size = float(msg["origQty"])
        self.side = msg["side"]
        self.status = msg["status"]
        self.order_type = msg["type"]
        self.order_id = msg["clientOrderId"]
        self.filled_size = msg["executedQty"]
        self.exchange = "BINANCE"

    def from_binance_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = future_asset_type(self.symbol, "future")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = timestamp_to_datetime(float(msg["time"]))
        self.ori_order_id = msg["orderId"]
        self.price = float(msg["price"])
        self.size = float(msg["origQty"])
        self.status = msg["status"]
        self.order_type = msg["type"]
        self.side = msg["side"]
        self.order_id = msg["clientOrderId"]
        self.filled_size = msg["executedQty"]
        self.exchange = "BINANCE"

    def from_binance_inverse_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = future_asset_type(self.symbol, "inverse")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = timestamp_to_datetime(float(msg["time"]))
        self.ori_order_id = msg["orderId"]
        self.price = float(msg["price"])
        self.size = float(msg["origQty"])
        self.status = msg["status"]
        self.order_type = msg["type"]
        self.side = msg["side"]
        self.order_id = msg["clientOrderId"]
        self.filled_size = msg["executedQty"]
        self.exchange = "BINANCE"

    def from_binance_option_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = ASSET_TYPE_OPTION
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = timestamp_to_datetime(float(msg["createDate"]))
        self.ori_order_id = msg["id"]
        self.price = float(msg["price"])
        self.size = float(msg["quantity"])
        self.status = msg["status"]
        self.order_type = msg["type"]
        self.side = msg["side"]
        self.order_id = msg["clientOrderId"]
        self.filled_size = msg["executedQty"]
        self.exchange = "BINANCE"

    def from_binance_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if datatype == "spot":
            self.from_binance_spot_format(msg)
        elif datatype == "linear":
            self.from_binance_future_format(msg)
        elif datatype == "inverse":
            self.from_binance_inverse_future_format(msg)
        elif datatype == "option":
            self.from_binance_option_format(msg)
        else:
            self.from_binance_spot_format(msg)

        return self.to_general_form()

    def from_binance_v2_format(self, msg: dict[str, Any], datatype: str) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = binance_asset_type(self.symbol, datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.created_at = timestamp_to_datetime(float(msg["time"]))
        self.updated_at = timestamp_to_datetime(float(msg["updateTime"]))
        self.ori_order_id = msg["orderId"]
        self.price = float(msg["price"])
        self.size = float(msg["origQty"])
        self.status = msg["status"]
        self.order_type = msg["type"]
        self.side = msg["side"]
        self.order_id = msg["clientOrderId"]
        self.filled_size = float(msg["executedQty"])
        self.position_side = (
            ("NET" if msg["positionSide"] == "BOTH" else msg["positionSide"]) if "positionSide" in msg else "NET"
        )
        self.exchange = "BINANCE"

    def from_binance_v2_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_binance_v2_format(msg, datatype)

        return self.to_general_form()


@dataclass
class PermissionSchema(BaseData):
    user_id: str | None = None
    created_at: datetime | None = None
    expired_at: datetime | None = None
    permissions: dict | list | None = None
    referrer: str | None = None
    referee: str | None = None
    vip_level: str | None = None
    unified: bool | None = None

    def to_general_form(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "expired_at": self.expired_at,
            "permissions": self.permissions,
            "referee": self.referee,
            "referrer": self.referrer,
            "vip_level": self.vip_level,
        }

    def from_bitget_format(self, msg: dict[str, Any]) -> None:
        """
        >>>> api_info:success=True error=False data={'user_id': '2825475201', 'inviter_id': '5402859283', 'agent_inviter_code': None, 'channel': 'stfz', 'ips': '', 'authorities': ['trade', 'transfer', 'readonly'], 'parentId': 2825475201, 'trader': False} msg='query ok'
        """
        # print(f"permission msg:{msg}")
        self.user_id = msg["user_id"]
        self.referrer = msg["inviter_id"]
        self.created_at = None
        self.expired_at = None
        authorities = msg["authorities"]
        self.permissions = {
            "enable_withdrawals": None,
            "enable_internaltransfer": None,
            "enable_universaltransfer": "transfer" in authorities,
            "enable_reading": "readonly" in authorities,
            "enable_options": None,
            "enable_futures": "trade" in authorities,
            "enable_spot_and_margintrading": "trade" in authorities,
            "enable_margin": None,
        }

    def from_bitget_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.from_bitget_format(msg)
        return self.to_general_form()

    def from_bitget_v2_format(self, msg: dict[str, Any]) -> None:
        """
        >>>> api_info:success=True error=False data={'user_id': '2825475201', 'inviter_id': '5402859283', 'agent_inviter_code': None, 'channel': 'stfz', 'ips': '', 'authorities': ['trade', 'transfer', 'readonly'], 'parentId': 2825475201, 'trader': False} msg='query ok'
        """
        # print(f"permission msg:{msg}")
        self.user_id = msg["userId"]
        self.referrer = msg["inviterId"]
        self.created_at = None
        self.expired_at = None
        authorities = msg["authorities"]
        self.permissions = {
            "enable_withdrawals": None,
            "enable_internaltransfer": None,
            "enable_universaltransfer": "transfer" in authorities,
            "enable_reading": "readonly" in authorities,
            "enable_options": None,
            "enable_futures": "trade" in authorities,
            "enable_spot_and_margintrading": "trade" in authorities,
            "enable_margin": None,
        }

    def from_bitget_v2_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.from_bitget_v2_format(msg)
        return self.to_general_form()

    def from_bingx_format(self, msg: dict[str, Any]) -> None:
        # ms
        print(f"permission msg:{msg}")
        self.user_id = None
        self.created_at = timestamp_to_datetime(float(msg["createTime"]))
        self.expired_at = None  # timestamp_to_datetime(float(expired_ts)) if expired_ts else self.created_at + relativedelta(months=10)
        self.permissions = {
            "enable_withdrawals": None,
            "enable_internaltransfer": None,
            "enable_universaltransfer": msg["permitsUniversalTransfer"],
            "enable_reading": msg["enableReading"],
            "enable_options": None,
            "enable_futures": msg["enableFutures"],
            "enable_spot_and_margintrading": msg["enableSpotAndMarginTrading"],
            "enable_margin": None,
        }  ### master <-> subaccount  ### mastet spot <-> master future, etc.

    def from_bingx_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.from_bingx_format(msg)
        return self.to_general_form()

    def from_binance_format(self, msg: dict[str, Any]) -> None:
        # ms
        self.user_id = None
        self.created_at = timestamp_to_datetime(float(msg["createTime"]))
        expired_ts = msg.get("tradingAuthorityExpirationTime", "")
        self.expired_at = (
            timestamp_to_datetime(float(expired_ts)) if expired_ts else self.created_at + relativedelta(months=10)
        )
        self.permissions = {
            "enable_withdrawals": msg["enableWithdrawals"],
            "enable_internaltransfer": msg["enableInternalTransfer"],
            "enable_universaltransfer": msg["permitsUniversalTransfer"],
            "enable_reading": msg["enableReading"],
            "enable_options": msg["enableVanillaOptions"],
            "enable_futures": msg["enableFutures"],
            "enable_spot_and_margintrading": msg["enableSpotAndMarginTrading"],
            "enable_margin": msg["enableMargin"],
        }  ### master <-> subaccount  ### mastet spot <-> master future, etc.

    def from_binance_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.from_binance_format(msg)
        return self.to_general_form()

    def from_bybit_format(self, msg: dict[str, Any]) -> None:
        print(f"permission raw msg:{msg}")
        self.user_id = msg["user_id"]
        self.created_at = isoformat_to_datetime(msg["created_at"])
        self.expired_at = (
            isoformat_to_datetime(msg["expired_at"])
            if msg["expired_at"]
            else generate_datetime() + timedelta(days=3650)
        )
        read_only = msg["read_only"]
        """
        {'api_key': 'LQsoX9QppatOkBeec3', 'type': 'personal', 'user_id': 7242843, 'inviter_id': 6721911, 'ips': ['103.109.103.49'], 'note': 'test_20221117', 'permissions': ['NFTQueryProductList', 'SpotTrade', 'OptionsTrade', 'CopyTrading', 'ExchangeHistory', 'Order', 'Position', 'AccountTransfer', 'SubMemberTransfer', 'Withdraw', 'DerivativesTrade'], 'created_at': '2022-11-16T03:45:16Z', 'expired_at': '', 'read_only': False, 'vip_level': 'No VIP', 'mkt_maker_level': '0', 'affiliate_id': 21753} ....
        """

        # print(msg,'....')
        self.permissions = {
            "enable_withdrawals": False if read_only else True if "Withdraw" in msg["permissions"] else False,
            "enable_internaltransfer": False
            if read_only
            else True
            if "SubMemberTransfer" in msg["permissions"]
            else False,  ### master <-> subaccount
            "enable_universaltransfer": False
            if read_only
            else True
            if "AccountTransfer" in msg["permissions"]
            else False,  ### mastet spot <-> master future, etc.
            "enable_reading": True,
            "enable_options": False if read_only else True if "OptionsTrade" in msg["permissions"] else False,
            "enable_futures": False
            if read_only
            else True
            if "Position" in msg["permissions"]
            and "Order" in msg["permissions"]
            and "DerivativesTrade" in msg["permissions"]
            else False,
            "enable_spot_and_margintrading": False
            if read_only
            else True
            if "SpotTrade" in msg["permissions"]
            else False,
            "enable_margin": None,
        }
        self.referee = str(msg["affiliate_id"])

    def from_bybit_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.from_bybit_format(msg)
        return self.to_general_form()

    def from_bybit_v2_format(self, msg: dict[str, Any]) -> None:
        # print(f"permission raw msg:{msg}")
        # permission raw msg:{'id': '391675', 'note': 'test_20230217', 'apiKey': 'MQLRJGFAADZNHNGZFC', 'readOnly': 0, 'secret': '', 'permissions': {'ContractTrade': ['Order', 'Position'], 'Spot': ['SpotTrade'], 'Wallet': ['AccountTransfer'], 'Options': ['OptionsTrade'], 'Derivatives': ['DerivativesTrade'], 'CopyTrading': [], 'BlockTrade': [], 'Exchange': ['ExchangeHistory'], 'NFT': []}, 'ips': ['*'], 'type': 1, 'deadlineDay': 85, 'expiredAt': '2023-05-17T09:49:24Z', 'createdAt': '2023-02-17T09:49:24Z', 'unified': 0, 'uta': 1, 'userID': 1193481, 'inviterID': 0, 'vipLevel': 'No VIP', 'mktMakerLevel': '0', 'affiliateID': 0, 'rsaPublicKey': ''}
        self.user_id = str(msg["userID"])
        self.created_at = isoformat_to_datetime(msg["createdAt"])
        self.expired_at = (
            isoformat_to_datetime(msg["expiredAt"]) if msg["expiredAt"] else generate_datetime() + timedelta(days=3650)
        )
        read_only = msg["readOnly"]
        self.permissions = {
            "enable_withdrawals": False if read_only else True if "Withdraw" in msg["permissions"]["Wallet"] else False,
            "enable_internaltransfer": False
            if read_only
            else True
            if "SubMemberTransfer" in msg["permissions"]["Wallet"]
            else False,  ### master <-> subaccount
            "enable_universaltransfer": False
            if read_only
            else True
            if "AccountTransfer" in msg["permissions"]["Wallet"]
            else False,  ### mastet spot <-> master future, etc.
            "enable_reading": True,
            "enable_options": False
            if read_only
            else True
            if "OptionsTrade" in msg["permissions"]["Options"]
            else False,
            "enable_futures": False
            if read_only
            else True
            if "Position" in msg["permissions"]["ContractTrade"]
            and "Order" in msg["permissions"]["ContractTrade"]
            and "DerivativesTrade" in msg["permissions"]["Derivatives"]
            else False,
            "enable_spot_and_margintrading": False
            if read_only
            else True
            if "SpotTrade" in msg["permissions"]["Spot"]
            else False,
            "enable_margin": None,
        }

        self.unified = True if msg["uta"] else False
        self.vip_level = str(msg["vipLevel"])
        self.referrer = str(msg["inviterID"])
        self.referee = str(msg["affiliateID"])

    def from_bybit_v2_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.from_bybit_v2_format(msg)
        return self.to_general_form()

    def from_binance_v2_format(self, msg: dict[str, Any]) -> None:
        # ms
        self.user_id = None
        self.created_at = timestamp_to_datetime(float(msg["createTime"]))
        expired_ts = msg.get("tradingAuthorityExpirationTime", "")
        self.expired_at = (
            timestamp_to_datetime(float(expired_ts)) if expired_ts else self.created_at + relativedelta(months=10)
        )
        self.permissions = {
            "enable_withdrawals": msg["enableWithdrawals"],
            "enable_internaltransfer": msg["enableInternalTransfer"],
            "enable_universaltransfer": msg["permitsUniversalTransfer"],
            "enable_reading": msg["enableReading"],
            "enable_options": msg["enableVanillaOptions"],
            "enable_futures": msg["enableFutures"],
            "enable_spot_and_margintrading": msg["enableSpotAndMarginTrading"],
            "enable_margin": msg["enableMargin"],
        }  ### master <-> subaccount  ### mastet spot <-> master future, etc.

    def from_binance_v2_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.from_binance_v2_format(msg)
        return self.to_general_form()


@dataclass
class TradeSchema(BaseData):
    asset_type: str | None = None
    symbol: str | None = None
    name: str | None = None
    order_id: str | None = None
    trade_id: str | None = None
    ori_order_id: str | None = None
    price: float | None = None
    quantity: float | None = None
    side: str | None = None
    commission: float | None = None
    commission_asset: str | None = None
    traded_at: datetime | None = None
    exchange: str | None = None
    position_side: str | None = "net"
    is_open: bool | None = None
    is_maker: bool | None = None

    def to_general_form(self) -> dict[str, Any]:
        return {
            "code": self.symbol,
            "symbol": self.name,
            "asset_type": self.asset_type,
            "exchange": self.exchange,
            "ori_order_id": self.ori_order_id,
            "order_id": self.order_id,
            "price": self.price,
            "quantity": self.quantity,
            "side": self.side,
            "position_side": self.position_side,
            "traded_at": self.traded_at,
            "commission": self.commission,
            "commission_asset": self.commission_asset,
            "is_open": self.is_open,
            "is_maker": self.is_maker,
        }

    def from_bitget_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print(msg)
        # spot: {'symbol': 'BTCUSDT', 'orderId': '1266298531902219520', 'orderLinkId': '', 'side': 'Buy', 'orderPrice': '', 'orderQty': '', 'leavesQty': '', 'orderType': 'Market', 'stopOrderType': '', 'execFee': '0', 'execId': '2100000000003413568', 'execPrice': '19210.52', 'execQty': '0.000312', 'execType': '', 'execValue': '', 'execTime': '1665690548362', 'isMaker': False, 'feeRate': '', 'tradeIv': '', 'markIv': '', 'markPrice': '', 'indexPrice': '', 'underlyingPrice': '', 'blockTradeId': ''}
        # linear: {'tradeId': '1030292933602631682', 'symbol': 'BTCUSDT_UMCBL', 'orderId': '1030292933431111683', 'price': '30083.50', 'sizeQty': '0.001', 'fee': '-0.0180501', 'side': 'open_long', 'fillAmount': '30.0835', 'profit': '0', 'enterPointSource': 'WEB', 'tradeSide': 'open_long', 'holdMode': 'double_hold', 'takerMakerFlag': 'taker', 'cTime': '1681366897964'},
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = bitget_asset_type(self.symbol, datatype=datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.order_id = None
        self.ori_order_id = msg["orderId"]
        self.price = float(msg["fillPrice"]) if "fillPrice" in msg else float(msg["price"])
        self.quantity = float(msg["fillQuantity"]) if "fillQuantity" in msg else float(msg["sizeQty"])
        self.side = (
            msg["side"].upper().replace("_SINGLE", "")
            if msg["side"].upper() in ["BUY", "SELL", "BUY_SINGLE", "SELL_SINGLE"]
            else (
                "BUY"
                if msg["side"] == "open_long"
                else "SELL"
                if msg["side"] == "open_short"
                else "SELL"
                if msg["side"] == "close_long"
                else "BUY"
            )
        )  #  linear 'side': 'open_long'
        self.commission = float(msg["fees"]) if "fees" in msg else float(msg["fee"])
        self.commission_asset = (
            msg["feeCcy"] if "feeCcy" in msg else "USDT" if datatype == "linear" else None
        )  # TODO: handle inverse
        self.traded_at = timestamp_to_datetime(float(msg["cTime"]))
        self.exchange = "BITGET"

    def from_bitget_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_bitget_format(msg, datatype=datatype)
        return self.to_general_form()

    def from_bitget_v2_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print(msg)
        # linear: {'tradeId': '1166254916684197889', 'symbol': 'BTCUSDT', 'orderId': '1166254916360634388', 'price': '65971.1', 'baseVolume': '0.002', 'feeDetail': [{'deduction': 'no', 'feeCoin': 'USDT', 'totalDeductionFee': '0', 'totalFee': '-0.07916532'}], 'side': 'buy', 'quoteVolume': '131.9422', 'profit': '0', 'enterPointSource': 'api', 'tradeSide': 'open', 'posMode': 'hedge_mode', 'tradeScope': 'taker', 'cTime': '1713782760783'}
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = bitget_asset_type(self.symbol, datatype=datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.order_id = None
        self.ori_order_id = msg["orderId"]
        self.price = float(msg["price"])
        self.quantity = float(msg["baseVolume"])
        self.side = msg["side"].upper()
        self.commission = float(msg["feeDetail"][0]["totalFee"])
        self.commission_asset = msg["feeDetail"][0]["feeCoin"]
        self.traded_at = timestamp_to_datetime(float(msg["cTime"]))
        self.exchange = "BITGET"

    def from_bitget_v2_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_bitget_v2_format(msg, datatype=datatype)
        return self.to_general_form()

    def from_ftx_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["market"]).upper()
        self.asset_type = ftx_asset_type(self.symbol)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.order_id = msg["id"]
        self.ori_order_id = msg["orderId"]
        self.price = msg["price"]
        self.quantity = msg["size"]
        self.side = str(msg["side"]).upper()
        self.commission = float(msg["feeRate"]) + float(msg["fee"])
        self.commission_asset = msg["feeCurrency"]
        self.traded_at = isoformat_to_datetime(msg["time"])
        self.exchange = "FTX"

    def from_ftx_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.from_ftx_format(msg)
        return self.to_general_form()

    def from_bybit_v2_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print('>>>[TradeSchema - from_bybit_v2_format]',msg)
        # {'symbol': 'BTCUSDT', 'orderId': '1266298531902219520', 'orderLinkId': '', 'side': 'Buy', 'orderPrice': '', 'orderQty': '', 'leavesQty': '', 'orderType': 'Market', 'stopOrderType': '', 'execFee': '0', 'execId': '2100000000003413568', 'execPrice': '19210.52', 'execQty': '0.000312', 'execType': '', 'execValue': '', 'execTime': '1665690548362', 'isMaker': False, 'feeRate': '', 'tradeIv': '', 'markIv': '', 'markPrice': '', 'indexPrice': '', 'underlyingPrice': '', 'blockTradeId': ''}
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = bybit_asset_type(self.symbol, datatype=datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.order_id = msg["orderLinkId"]
        self.ori_order_id = msg["orderId"]
        self.price = float(msg["execPrice"])
        self.quantity = float(msg["execQty"])
        self.side = str(msg["side"]).upper()
        self.commission = float(msg["execFee"]) if msg["execFee"] else 0.0
        self.commission_asset = None
        self.traded_at = timestamp_to_datetime(float(msg["execTime"]))
        ### NOTE: still cannot correctly identify whether the position mode (oneway/hedge)
        close_qty = float(msg["closedSize"]) if "closedSize" in msg and msg["closedSize"] else None
        # print(f"[DEBUG] close_qty:{close_qty}; closedSize:{msg.get('closedSize')}|")
        if msg.get("closedSize"):
            self.is_open = False if close_qty else True
        # self.position_side = ("short" if self.side == "BUY" else "long") if close_qty else ("long" if self.side == "BUY" else "short")
        self.position_side = "NET"  # only usdt-perp has hedge mode(long-short)  NOTE:  there is not field to identify the position side yet
        self.is_maker = msg["isMaker"]
        self.exchange = "BYBIT"

    def from_bybit_v2_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_bybit_v2_format(msg, datatype=datatype)
        if msg.get("execType", "") == "Funding":
            return {}
        return self.to_general_form()

    def from_bingx_format(self, msg: dict[str, Any], datatype: str) -> None:
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = bingx_asset_type(self.symbol, datatype=datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.order_id = None
        self.ori_order_id = msg["orderId"]
        self.price = float(msg["price"])
        self.quantity = float(msg["executedQty"])
        self.side = str(msg["side"]).upper()
        self.commission = float(msg["commission"]) if "commission" in msg else None
        self.commission_asset = None
        self.traded_at = timestamp_to_datetime(float(msg["updateTime"]))
        self.exchange = "BingX"

    def from_bingx_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_bingx_format(msg, datatype=datatype)
        if msg["status"] != "FILLED":
            return {}
        return self.to_general_form()

    def from_okx_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["instId"]).upper()
        self.asset_type = okx_asset_type(self.symbol)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.order_id = msg["ordId"]
        self.ori_order_id = msg["ordId"]
        self.price = float(msg["fillPx"])
        self.quantity = float(msg["fillSz"])
        self.side = str(msg["side"]).upper()
        self.commission = float(msg["fee"])
        self.commission_asset = msg["feeCcy"]
        self.traded_at = timestamp_to_datetime(float(msg["ts" if "ts" in msg else "fillTime"]))
        self.is_open = msg["reduceOnly"] == "true" if "reduceOnly" in msg else None
        self.is_maker = str(msg.get("execType", "")) == "M"  # M=maker, T=taker
        self.position_side = msg["posSide"]
        self.exchange = "OKX"

    def from_okx_to_form(self, msg: dict[str, Any], datatype: str | None = None) -> dict[str, Any]:
        self.from_okx_format(msg)
        return self.to_general_form()

    def from_eqonex_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = eqonex_asset_type(self.symbol)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.order_id = msg["clOrdId"]
        self.ori_order_id = str(msg["orderId"])
        self.price = float(msg["price"])
        self.quantity = float(msg["qty"])
        self.side = str(msg["side"]).upper()
        self.commission = float(msg["commission"]) if "commission" in msg else float(msg["fee"])
        self.commission_asset = msg["commCurrency"] if "commCurrency" in msg else msg["feeAsset"]
        self.traded_at = (
            isoformat_to_datetime(msg["date"]) if "commCurrency" in msg else isoformat_to_datetime(msg["time"])
        )
        self.exchange = "EQONEX"

    def from_eqonex_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        symbol = msg["symbol"]
        if datatype == "spot":
            if "[" in symbol and "]" in symbol:
                return {}
        elif datatype == "linear":
            if not ("[" in symbol and "]" in symbol):
                return {}
        self.from_eqonex_format(msg)
        return self.to_general_form()

    def from_bybit_spot_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.symbol = str(msg["symbol"]).upper()
        self.name = normalize_name(self.symbol, self.asset_type)
        self.order_id = msg["id"]
        self.ori_order_id = msg["orderId"]
        self.price = float(msg["price"])
        self.quantity = float(msg["qty"])
        self.side = "BUY" if msg["isBuyer"] else "SELL"
        self.commission = float(msg["feeAmount"]) + float(msg["commission"])
        self.commission_asset = msg["commissionAsset"]
        self.traded_at = timestamp_to_datetime(float(msg["time"]))
        self.exchange = "BYBIT"

    def from_bybit_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = future_asset_type(self.symbol, "future")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.order_id = msg["order_link_id"]
        self.ori_order_id = msg["order_id"]
        self.price = float(msg["exec_price"]) if "exec_price" in msg else float(msg["price"])
        self.quantity = float(msg["exec_qty"])
        self.side = msg["side"]
        self.commission = float(msg["exec_fee"])  # float(msg["fee_rate"]) +
        self.commission_asset = "USDT"
        self.traded_at = (
            timestamp_to_datetime(float(msg["trade_time_ms"]))
            if "trade_time_ms" in msg
            else isoformat_to_datetime(msg["trade_time"])
        )
        self.exchange = "BYBIT"

    def from_bybit_inverse_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = future_asset_type(self.symbol, "inverse")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.order_id = msg["order_link_id"]
        self.ori_order_id = msg["order_id"]
        self.price = float(msg["exec_price"]) if "exec_price" in msg else float(msg["price"])
        self.quantity = float(msg["exec_qty"])
        self.side = msg["side"]
        self.commission = float(msg["exec_fee"])  # float(msg["fee_rate"]) +
        # inverse future commission_asset should be coin
        self.commission_asset = self.symbol.split("USD")[0]
        self.traded_at = (
            timestamp_to_datetime(float(msg["trade_time_ms"]))
            if "trade_time_ms" in msg
            else isoformat_to_datetime(msg["trade_time"])
        )
        self.exchange = "BYBIT"

    def from_bybit_option_format(self, msg: dict[str, Any]) -> None:
        ## TODO: need iv or vol?
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = ASSET_TYPE_OPTION
        self.name = normalize_name(self.symbol, self.asset_type)
        self.order_id = msg["orderLinkId"]
        self.ori_order_id = msg["orderId"]
        self.trade_id = msg["tradeId"]
        self.price = float(msg["execPrice"])
        self.quantity = float(msg["execQty"])
        self.side = msg["side"]
        self.commission = float(msg["execFee"])  # float(msg["fee_rate"]) +
        self.commission_asset = "USDC"
        self.traded_at = timestamp_to_datetime(float(msg["tradeTime"]))
        self.exchange = "BYBIT"

    def from_bybit_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if datatype == "spot":
            self.from_bybit_spot_format(msg)
        elif datatype == "linear":
            if msg["exec_type"] == "Funding":
                return {}
            self.from_bybit_future_format(msg)
        elif datatype == "inverse":
            if msg["exec_type"] == "Funding":
                return {}
            self.from_bybit_inverse_future_format(msg)
        elif datatype == "option":
            self.from_bybit_option_format(msg)
        else:
            self.from_bybit_spot_format(msg)

        return self.to_general_form()

    def from_binance_spot_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.symbol = str(msg["symbol"]).upper()
        self.name = normalize_name(self.symbol, self.asset_type)
        # self.order_id = msg['clientOrderId']
        self.order_id = str(msg["id"])
        self.ori_order_id = str(msg["orderId"])
        self.price = float(msg["price"])
        self.quantity = float(msg["qty"])
        self.side = "BUY" if msg["isBuyer"] else "SELL"
        self.commission = float(msg["commission"])
        self.commission_asset = msg["commissionAsset"]
        self.traded_at = timestamp_to_datetime(float(msg["time"]))
        self.exchange = "BINANCE"

    def from_binance_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = future_asset_type(self.symbol, "future")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.order_id = str(msg["id"])
        self.ori_order_id = str(msg["orderId"])
        self.price = float(msg["price"])
        self.quantity = float(msg["qty"])
        self.side = "BUY" if msg["buyer"] else "SELL"
        self.commission = float(msg["commission"])
        self.commission_asset = msg["commissionAsset"]
        self.traded_at = timestamp_to_datetime(float(msg["time"]))
        self.exchange = "BINANCE"

    def from_binance_inverse_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = future_asset_type(self.symbol, "inverse")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.order_id = str(msg["id"])
        self.ori_order_id = str(msg["orderId"])
        self.price = float(msg["price"])
        self.quantity = float(msg["qty"])
        self.side = "BUY" if msg["buyer"] else "SELL"
        self.commission = float(msg["commission"])
        self.commission_asset = msg["commissionAsset"]
        self.traded_at = timestamp_to_datetime(float(msg["time"]))
        self.exchange = "BINANCE"

    def from_binance_option_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = ASSET_TYPE_OPTION
        self.name = normalize_name(self.symbol, self.asset_type)
        self.order_id = msg["id"]
        self.ori_order_id = msg["orderId"]
        self.price = msg["price"]
        self.quantity = msg["quantity"]
        self.side = msg["side"]
        self.commission = float(msg["fee"])
        self.commission_asset = msg["quoteAsset"]
        self.traded_at = timestamp_to_datetime(float(msg["createDate"]))
        self.exchange = "BINANCE"

    def from_binance_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if datatype == "spot":
            self.from_binance_spot_format(msg)
        elif datatype == "linear":
            self.from_binance_future_format(msg)
        elif datatype == "inverse":
            self.from_binance_future_format(msg)
        elif datatype == "option":
            self.from_binance_option_format(msg)
        else:
            self.from_binance_spot_format(msg)

        return self.to_general_form()

    def from_binance_v2_format(self, msg: dict[str, Any], datatype: str) -> None:
        try:
            self.symbol = str(msg["symbol"]).upper()
            self.asset_type = binance_asset_type(self.symbol, datatype=datatype)
            self.name = normalize_name(self.symbol, self.asset_type)
            self.order_id = msg["clientOrderId"] if "clientOrderId" in msg else None
            self.ori_order_id = str(msg["orderId"])
            self.price = float(msg["price"])
            self.quantity = float(msg["qty"])
            self.side = "BUY" if msg["isBuyer" if "isBuyer" in msg else "buyer"] else "SELL"
            self.commission = float(msg["commission"])
            self.commission_asset = msg["commissionAsset"]
            self.traded_at = timestamp_to_datetime(float(msg["time"]))
            self.position_side = (
                ("NET" if msg["positionSide"] == "BOTH" else msg["positionSide"]) if "positionSide" in msg else "NET"
            )
            self.is_maker = msg["maker"]
            # self.is_open = msg["reduceOnly"] == 'true' if 'reduceOnly' in msg else None
            self.is_open = (self.side == "BUY" and self.position_side == "NET") or (
                self.side == "SELL" and self.position_side == "SHORT"
            )
            self.exchange = "BINANCE"
        except Exception as e:
            print(f"Something went wrong when parse binance {datatype} trade data. Error:{str(e)}")
            print(f"[TradeSchema - from_binance_v2_format] msg:{msg}; datatype:{datatype}")

    def from_binance_v2_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_binance_v2_format(msg, datatype)

        return self.to_general_form()


@dataclass
class SymbolSchema(BaseData):
    asset_type: str | None = None
    symbol: str | None = None
    name: str | None = None
    tick_size: float | None = None
    step_size: float | None = None
    max_volume: float | None = None
    min_volume: float | None = None
    min_notional: float | None = None
    base_currency: str | None = None
    quote_currency: str | None = None
    product: str | None = None
    extra: dict | None = None
    exchange: str | None = None
    # extra for option will contains maturity, callput, strike, delivery_fee,

    def to_general_form(self) -> dict[str, Any]:
        return {
            "asset_type": self.asset_type,
            "code": self.symbol,
            "exchange": self.exchange,
            "symbol": self.name,
            "tick_size": self.tick_size,
            "step_size": self.step_size,
            "min_volume": self.min_volume,
            "max_volume": self.max_volume,
            "min_notional": self.min_notional,
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "product": self.product,
            "extra": self.extra,
        }

    def from_bitget_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print(msg)
        self.symbol = str(msg["symbol"])  # .lower()
        self.asset_type = bitget_asset_type(self.symbol, datatype)
        self.product = datatype if datatype != "linear" else "future"
        self.name = normalize_name(self.symbol, self.asset_type)
        self.tick_size = (
            change_scale(1, int(float(msg["priceScale"]))) if "priceScale" in msg else float(msg["priceEndStep"])
        )
        self.min_volume = float(msg["minTradeAmount"]) if "minTradeAmount" in msg else float(msg["sizeMultiplier"])
        self.min_notional = float(msg["minTradeUSDT"]) if "minTradeUSDT" in msg else float(msg["minTradeNum"])
        self.base_currency = msg["baseCoin"]
        self.quote_currency = msg["quoteCoin"]
        # self.size = 1
        # self.history_data = msg["enabled"]
        self.extra = {}
        self.exchange = "BITGET"

    def from_bitget_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        if "symbolStatus" in msg and msg["symbolStatus"] != "normal":
            return {}
        self.from_bitget_format(msg, datatype)
        return self.to_general_form()

    def from_bitget_v2_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print(msg)
        self.symbol = str(msg["symbol"])  # .lower()
        self.asset_type = bitget_asset_type(self.symbol, datatype)
        self.product = datatype if datatype != "linear" else "future"
        self.name = normalize_name(self.symbol, self.asset_type)
        self.tick_size = change_scale(1, int(float(msg["pricePrecision"])))
        self.step_size = change_scale(1, int(float(msg["quantityPrecision"])))
        self.min_volume = None
        self.min_notional = float(msg["minTradeUSDT"])
        self.base_currency = msg["baseCoin"]
        self.quote_currency = msg["quoteCoin"]
        self.extra = {}
        self.exchange = "BITGET"

    def from_bitget_v2_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        if msg["status"] != "online":
            return {}
        self.from_bitget_v2_format(msg, datatype=datatype)
        return self.to_general_form()

    def from_ftx_format(self, msg: dict[str, Any]) -> None:
        # print(msg)
        self.symbol = str(msg["name"])  # .lower()
        self.asset_type = ftx_asset_type(self.symbol)
        self.product = msg["type"]
        self.name = normalize_name(self.symbol, self.asset_type)
        # Only sport market has these two items in FTX
        # if msg['baseCurrency'] and msg['quoteCurrency']:
        #     self.name = str(msg['baseCurrency']).upper() + '/' + str(msg['quoteCurrency']).upper()
        # else:
        #     self.name = str(msg['name']).upper()

        self.tick_size = float(msg["priceIncrement"])
        self.min_volume = float(msg["sizeIncrement"])
        self.min_notional = float(msg["minProvideSize"])
        # quoteCurrency for spot / underlying for future
        self.quote_currency = msg["quoteCurrency"] or msg["underlying"]
        # self.size = 1
        # self.history_data = msg["enabled"]
        self.extra = {"enabled": msg["enabled"], "volume_24h": msg["volumeUsd24h"]}
        self.exchange = "FTX"

    def from_ftx_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.from_ftx_format(msg)
        return self.to_general_form()

    def from_bingx_spot_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print(msg,'????????')
        self.symbol = str(msg["symbol"])
        self.asset_type = bingx_asset_type(self.symbol, datatype)
        self.product = datatype
        self.name = normalize_name(self.symbol, self.asset_type)

        self.tick_size = float(msg["tickSize"])
        self.min_volume = float(msg["minQty"])
        self.min_notional = float(msg["minNotional"])
        # quoteCurrency for spot / underlying for future
        self.quote_currency = self.symbol.split("-")[1] if self.symbol else None
        # self.size = 1
        # self.history_data = True
        self.extra = {}
        self.exchange = "BINGX"

    def from_bingx_future_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print(msg,'????????')
        self.symbol = str(msg["symbol"])
        self.asset_type = bingx_asset_type(self.symbol, datatype)
        self.product = datatype
        self.name = normalize_name(self.symbol, self.asset_type)

        self.tick_size = 10 ** (-float(msg["pricePrecision"]))
        self.min_volume = float(msg["size"])
        self.min_notional = float(msg["tradeMinLimit"])
        # quoteCurrency for spot / underlying for future
        self.quote_currency = msg["currency"]
        # self.size = 1
        # self.history_data = True
        self.extra = {}
        self.exchange = "BINGX"

    def from_bingx_to_form(self, msg: dict[str, Any], datatype: str | None = None) -> dict[str, Any]:
        if msg["status"] != 1:
            return {}
        if datatype == "spot":
            self.from_bingx_spot_format(msg, datatype)
        elif datatype == "linear":
            self.from_bingx_future_format(msg, datatype)

        return self.to_general_form()

    def from_okx_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print(msg,'????????')
        # {'alias': '', 'baseCcy': '', 'category': '1', 'ctMult': '0.01', 'ctType': '', 'ctVal': '1', 'ctValCcy': 'BTC', 'expTime': '1725004800000', 'instFamily': 'BTC-USD', 'instId': 'BTC-USD-240830-54000-P', 'instType': 'OPTION', 'lever': '', 'listTime': '1715934600000', 'lotSz': '1', 'maxIcebergSz': '1000000.0000000000000000', 'maxLmtAmt': '20000000', 'maxLmtSz': '1000000', 'maxMktAmt': '', 'maxMktSz': '5000', 'maxStopSz': '5000', 'maxTriggerSz': '1000000.0000000000000000', 'maxTwapSz': '1000000.0000000000000000', 'minSz': '1', 'optType': 'P', 'quoteCcy': '', 'settleCcy': 'BTC', 'state': 'live', 'stk': '54000', 'tickSz': '0.0001', 'uly': 'BTC-USD'} ????????
        self.symbol = str(msg["instId"])  # .lower()
        self.asset_type = okx_asset_type(self.symbol)
        self.product = datatype  # msg['instType']
        self.name = normalize_name(self.symbol, self.asset_type)
        # Only sport market has these two items in FTX
        # if msg['baseCurrency'] and msg['quoteCurrency']:
        #     self.name = str(msg['baseCurrency']).upper() + '/' + str(msg['quoteCurrency']).upper()
        # else:
        #     self.name = str(msg['name']).upper()

        self.step_size = float(msg["lotSz"])
        self.tick_size = float(msg["tickSz"])
        self.min_volume = float(msg["minSz"])
        self.max_volume = float(msg["maxMktSz"])
        self.min_notional = float(msg["lotSz"])  # FIXME: cant find field
        # quoteCurrency for spot / underlying for future
        self.quote_currency = msg["quoteCcy"] if datatype == "spot" else msg["settleCcy"]
        # self.size = 1
        # self.history_data = True
        self.extra = {}
        if datatype == "option":
            # self.extra = {
            # "listing": timestamp_to_datetime(float(msg["listTime"])),
            # "maturity": timestamp_to_datetime(float(msg["expTime"])),
            # "cp": msg["optType"].lower(),
            # "status": msg["state"],
            # "strike": float(msg["stk"])}
            self.extra = {
                "launch_date": timestamp_to_datetime(float(msg["listTime"]))
                if msg.get("listTime") and float(msg["listTime"])
                else None,
                "delivery_date": timestamp_to_datetime(float(msg["expTime"]))
                if msg.get("expTime") and float(msg["expTime"])
                else None,
                "option_type": msg["optType"].lower() if msg.get("optType") else None,
                # "delivery_fee_rate": float(msg["deliveryFeeRate"]) if msg.get("deliveryFeeRate") else None,
                "status": msg["state"],
            }
            self.extra["strike"] = float(self.name.split("-")[2]) if "-" in self.name else None
            self.extra["underlying"] = self.name[:10]
        self.exchange = "OKX"

    def from_okx_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        if msg["state"] != "live":
            return {}
        if datatype == "linear" and msg["ctType"] != "linear":
            return {}
        elif datatype == "inverse" and msg["ctType"] != "inverse":
            return {}

        self.from_okx_format(msg, datatype=datatype)
        return self.to_general_form()

    def from_eqonex_format(self, msg: dict[str, Any]) -> None:
        # print(msg,'!!!')
        """
        {'instrumentId': 52, 'symbol': 'BTC/USDC', 'quoteId': 1, 'baseId': 3, 'price_scale': 2, 'quantity_scale': 6, 'securityStatus': 1, 'securityDesc': 'BTC/USDC', 'assetType': 'PAIR', 'currency': 'BTC', 'contAmtCurr': 'USDC', 'settlCurrency': 'USDC', 'commCurrency': 'USDC', 'cfiCode': 'IFXXXP', 'securityExchange': 'EQOS', 'micCode': 'EQOC', 'instrumentPricePrecision': 2, 'minPriceIncrement': 1.0, 'minPriceIncrementAmount': 1.0, 'roundLot': 100, 'minTradeVol': 0.0001, 'maxTradeVol': 0.0, 'qtyType': 0, 'contractMultiplier': 1.0, 'auctionStartTime': 0, 'auctionDuration': 0, 'auctionFrequency': 0, 'auctionPrice': 0, 'auctionVolume': 0, 'marketStatus': 'OPEN'} !!!
        """
        self.symbol = str(msg["symbol"])  # .lower()
        self.asset_type = eqonex_asset_type(self.symbol)
        self.product = "spot" if msg["assetType"] == "PAIR" else "future"
        self.name = normalize_name(self.symbol, self.asset_type)

        self.tick_size = float(msg["minPriceIncrement"])
        self.min_volume = float(msg["minTradeVol"])
        self.min_notional = float(msg["minPriceIncrementAmount"])  # *  float(msg['minTradeVol'])

        self.quote_currency = msg["contAmtCurr"]
        # self.size = 1
        # self.history_data = True
        self.extra = {
            "instrument_id": msg["instrumentId"],
            "price_scale": msg["price_scale"],
            "quantity_scale": msg["quantity_scale"],
        }

        self.exchange = "EQONEX"

    def from_eqonex_to_form(self, msg: dict[str, Any], datatype: str, return_all: bool = False) -> dict[str, Any]:
        if not return_all:
            symbol = msg["symbol"]
            if datatype == "spot":
                if "[" in symbol and "]" in symbol:
                    return {}
            elif datatype == "linear":
                if not ("[" in symbol and "]" in symbol):
                    return {}
        self.from_eqonex_format(msg)
        return self.to_general_form()

    def from_bybit_spot_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.symbol = str(msg["name"])  # .lower()
        self.name = normalize_name(self.symbol, self.asset_type)
        # Only sport market has these two items in FTX
        # self.name = str(msg['name']).upper()
        self.tick_size = float(msg["minPricePrecision"])
        self.min_volume = float(msg["basePrecision"])  # 'minTradeQuantity
        self.min_notional = float(msg["minTradeAmount"])
        # quoteCurrency for spot / underlying for future
        self.quote_currency = msg["quoteCurrency"]
        # self.size = 1
        self.product = "spot"
        # self.history_data = None
        self.extra = None
        self.exchange = "BYBIT"

    def from_bybit_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["name"])  # .lower()
        self.asset_type = future_asset_type(self.symbol, "future")
        self.name = normalize_name(self.symbol, self.asset_type)
        # Only sport market has these two items in FTX
        self.name = str(msg["name"]).upper()
        self.tick_size = float(msg["price_filter"]["tick_size"])
        self.min_volume = float(msg["lot_size_filter"]["qty_step"])
        self.min_notional = float(msg["price_filter"]["min_price"])
        # quoteCurrency for spot / underlying for future
        self.quote_currency = msg["quote_currency"]
        # self.size = 1
        self.product = "future"
        # self.history_data = None
        self.extra = None
        self.exchange = "BYBIT"

    def from_bybit_inverse_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["name"])  # .lower()
        self.asset_type = future_asset_type(self.symbol, "inverse")
        self.name = normalize_name(self.symbol, self.asset_type)
        # # Only sport market has these two items in FTX
        # self.name = str(msg['name']).upper()
        self.tick_size = float(msg["price_filter"]["tick_size"])
        self.min_volume = float(msg["lot_size_filter"]["qty_step"])
        self.min_notional = float(msg["price_filter"]["min_price"])
        # quoteCurrency for spot / underlying for future
        self.quote_currency = msg["quote_currency"]
        # self.size = 1
        self.product = "inverse_future"
        # self.history_data = None
        # self.extra = {'maturity': } if self.asset_type == ASSET_TYPE_INVERSE_DATED_FUTURE else None
        self.extra = None
        self.exchange = "BYBIT"

    def from_bybit_option_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])  # .upper()
        self.asset_type = ASSET_TYPE_OPTION
        # print(msg,'....')
        self.name = normalize_name(self.symbol, self.asset_type)
        # Only sport market has these two items in FTX
        # self.name = str(msg['symbol']).upper()
        self.tick_size = float(msg["tickSize"])
        self.min_volume = float(msg["minOrderSize"])
        self.min_notional = float(msg["minOrderPrice"])
        # quoteCurrency for spot / underlying for future
        self.quote_currency = msg["quoteCoin"]
        # self.size = 1
        self.product = "option"
        # self.history_data = None
        splits = self.name.split("-")
        self.extra = {
            "maturity": timestamp_to_datetime(int(msg["deliveryTime"])),
            "callput": splits[-1],
            "strike": float(splits[-2]),
            "delivery_fee": float(msg["basicDeliveryFeeRate"]),
        }
        self.exchange = "BYBIT"

    def from_bybit_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if datatype == "spot":
            self.from_bybit_spot_format(msg)
        elif datatype == "linear":
            if msg["quote_currency"] == "USD":
                return {}
            self.from_bybit_future_format(msg)
        elif datatype == "inverse":
            if msg["quote_currency"] != "USD":
                return {}
            self.from_bybit_inverse_future_format(msg)
        elif datatype == "option":
            if msg["status"] != "ONLINE":
                # Status, can be WAITING_ONLINE, ONLINE, DELIVERING, or OFFLINE
                return {}
            self.from_bybit_option_format(msg)
        else:
            self.from_bybit_spot_format(msg)

        return self.to_general_form()

    def from_bybit_v2_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print(msg,'????')
        # spot: {'symbol': 'BTCUSDT', 'baseCoin': 'BTC', 'quoteCoin': 'USDT', 'innovation': '0', 'status': '1', 'lotSizeFilter': {'basePrecision': '0.000001', 'quotePrecision': '0.00000001', 'minOrderQty': '0.00004', 'maxOrderQty': '63.01197227', 'minOrderAmt': '1', 'maxOrderAmt': '100000'}, 'priceFilter': {'tickSize': '0.01'}}
        # linear: {'symbol': '10000NFTUSDT', 'contractType': 'LinearPerpetual', 'status': 'Trading', 'baseCoin': '10000NFT', 'quoteCoin': 'USDT', 'launchTime': '1643007175000', 'deliveryTime': '0', 'deliveryFeeRate': '', 'priceScale': '6', 'leverageFilter': {'minLeverage': '1', 'maxLeverage': '12.50', 'leverageStep': '0.01'}, 'priceFilter': {'minPrice': '0.000005', 'maxPrice': '9.999990', 'tickSize': '0.000005'}, 'lotSizeFilter': {'maxOrderQty': '370000', 'minOrderQty': '10', 'qtyStep': '10', 'postOnlyMaxOrderQty': '3700000'}, 'unifiedMarginTrade': True, 'fundingInterval': 480, 'settleCoin': 'USDT'}
        # inverse: {'symbol': 'ADAUSD', 'contractType': 'InversePerpetual', 'status': 'Trading', 'baseCoin': 'ADA', 'quoteCoin': 'USD', 'launchTime': '1647302400000', 'deliveryTime': '0', 'deliveryFeeRate': '', 'priceScale': '4', 'leverageFilter': {'minLeverage': '1', 'maxLeverage': '50.00', 'leverageStep': '0.01'}, 'priceFilter': {'minPrice': '0.0001', 'maxPrice': '199.9998', 'tickSize': '0.0001'}, 'lotSizeFilter': {'maxOrderQty': '150000', 'minOrderQty': '1', 'qtyStep': '1', 'postOnlyMaxOrderQty': '450000'}, 'unifiedMarginTrade': False, 'fundingInterval': 480, 'settleCoin': 'ADA'}
        # option: {'category': 'option', 'symbol': 'BTC-30JUN23-100000-C', 'status': 'ONLINE', 'baseCoin': 'BTC', 'quoteCoin': 'USD', 'settleCoin': 'USDC', 'optionsType': 'Call', 'launchTime': '1672905600000', 'deliveryTime': '1688112000000', 'deliveryFeeRate': '0.00015', 'priceFilter': {'minPrice': '5', 'maxPrice': '10000000', 'tickSize': '5'}, 'lotSizeFilter': {'maxOrderQty': '10000', 'minOrderQty': '0.01', 'qtyStep': '0.01'}}
        # option: {'symbol': 'BTC-8JUL23-30250-C', 'status': 'Trading', 'baseCoin': 'BTC', 'quoteCoin': 'USD', 'settleCoin': 'USDC', 'optionsType': 'Call', 'launchTime': '1688544000000', 'deliveryTime': '1688803200000', 'deliveryFeeRate': '0.00015', 'priceFilter': {'minPrice': '5', 'maxPrice': '10000000', 'tickSize': '5'}, 'lotSizeFilter': {'maxOrderQty': '10000', 'minOrderQty': '0.01', 'qtyStep': '0.01'}} ????
        self.symbol = str(msg["symbol"])
        self.asset_type = bybit_asset_type(self.symbol, datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.tick_size = float(msg["priceFilter"]["tickSize"])
        self.step_size = float(msg["lotSizeFilter"]["basePrecision" if datatype == "spot" else "qtyStep"])
        self.min_volume = float(msg["lotSizeFilter"]["minOrderQty"])
        self.max_volume = float(msg["lotSizeFilter"]["maxOrderQty"])
        self.min_notional = msg["lotSizeFilter"].get("minOrderAmt")
        self.min_notional = float(self.min_notional) if self.min_notional else self.min_notional
        self.base_currency = msg["baseCoin"]
        self.quote_currency = msg["quoteCoin"]
        self.product = datatype if datatype != "linear" else "future"
        self.extra = {
            "launch_date": timestamp_to_datetime(float(msg["launchTime"]))
            if msg.get("launchTime") and float(msg["launchTime"])
            else None,
            "delivery_date": timestamp_to_datetime(float(msg["deliveryTime"]))
            if msg.get("deliveryTime") and float(msg["deliveryTime"])
            else None,
            "option_type": msg["optionsType"].lower() if msg.get("optionsType") else None,
            "delivery_fee_rate": float(msg["deliveryFeeRate"]) if msg.get("deliveryFeeRate") else None,
            "status": msg["status"],
        }
        if datatype == "option":
            self.extra["strike"] = float(self.name.split("-")[2]) if "-" in self.name else None
            self.extra["underlying"] = self.name[:10]
        self.exchange = "BYBIT"

    def from_bybit_v2_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        self.from_bybit_v2_format(msg, datatype)
        return self.to_general_form()

    def from_dydx_v3_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        # print(f">>> {msg}")
        # {'market': '1INCH-USD', 'status': 'ONLINE', 'baseAsset': '1INCH', 'quoteAsset': 'USD', 'stepSize': '1', 'tickSize': '0.001', 'indexPrice': '0.2870', 'oraclePrice': '0.2872', 'priceChange24H': '-0.009253', 'nextFundingRate': '0.0000125000', 'nextFundingAt': '2023-11-01T11:00:00.000Z', 'minOrderSize': '10', 'type': 'PERPETUAL', 'initialMarginFraction': '0.10', 'maintenanceMarginFraction': '0.05', 'transferMarginFraction': '0.007595', 'volume24H': '4937456.682000', 'trades24H': '111', 'openInterest': '7430491', 'incrementalInitialMarginFraction': '0.02', 'incrementalPositionSize': '70000', 'maxPositionSize': '3500000', 'baselinePositionSize': '350000', 'assetResolution': '10000000', 'syntheticAssetId': '0x31494e43482d370000000000000000'}
        self.symbol = str(msg["market"])
        self.asset_type = ASSET_TYPE_PERPETUAL
        self.name = normalize_name(self.symbol, self.asset_type)
        self.tick_size = float(msg["tickSize"])
        self.min_volume = float(msg["minOrderSize"])
        self.min_notional = None
        self.base_currency = msg["baseAsset"]
        self.quote_currency = msg["quoteAsset"]
        self.product = "future"
        self.exchange = "DYDX"

        return self.to_general_form()

    def from_binance_spot_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.symbol = str(msg["symbol"]).upper()
        # self.name = normalize_name(self.symbol, self.asset_type)
        self.name = f"{msg['baseAsset'].upper()}/{msg['quoteAsset'].upper()}"
        for f in msg["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                self.tick_size = float(f["tickSize"])
            elif f["filterType"] == "LOT_SIZE":
                self.min_volume = float(f["stepSize"])
            elif f["filterType"] == "MIN_NOTIONAL":
                self.min_notional = float(f["minNotional"])
        # quoteCurrency for spot / underlying for future
        self.quote_currency = msg["quoteAsset"]
        # self.size = 1
        self.product = "spot"
        # self.history_data = True
        self.extra = {}
        self.exchange = "BINANCE"

    def from_binance_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = future_asset_type(self.symbol, "future")
        self.name = normalize_name(self.symbol, self.asset_type)
        # self.name = f"{msg['baseAsset'].upper()}/{msg['quoteAsset'].upper()}"
        for f in msg["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                self.tick_size = float(f["tickSize"])
            elif f["filterType"] == "LOT_SIZE":
                self.min_volume = float(f["stepSize"])
                self.min_notional = float(f["minQty"])
        # quoteCurrency for spot / underlying for future
        self.quote_currency = msg["quoteAsset"]
        # self.size = 1
        self.product = "future"
        # self.history_data = True
        self.extra = {}
        self.exchange = "BINANCE"

    def from_binance_inverse_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = future_asset_type(self.symbol, "inverse")
        self.name = normalize_name(self.symbol, self.asset_type)
        # self.name = f"{msg['baseAsset'].upper()}/{msg['quoteAsset'].upper()}"
        for f in msg["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                self.tick_size = float(f["tickSize"])
            elif f["filterType"] == "LOT_SIZE":
                self.min_volume = float(f["stepSize"])
                self.min_notional = float(f["minQty"])
        # quoteCurrency for spot / underlying for future
        self.quote_currency = msg["quoteAsset"]
        # self.size = 1
        self.product = "inverse_future"
        # self.history_data = True
        self.extra = {}
        self.exchange = "BINANCE"

    def from_binance_option_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = ASSET_TYPE_OPTION
        self.name = normalize_name(self.symbol, self.asset_type)
        # self.name = f"{msg['symbol'].upper()}/{msg['quoteAsset'].upper()}"
        for f in msg["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                self.tick_size = float(f["tickSize"])
                self.min_notional = float(f["minPrice"])
            elif f["filterType"] == "LOT_SIZE":
                self.min_volume = float(f["stepSize"])
                self.min_notional = float(f["minQty"])
        # quoteCurrency for spot / underlying for future
        self.quote_currency = msg["quoteAsset"]
        # self.size = float(msg['unit'])
        self.product = "option"
        # self.history_data = True
        self.extra = {}
        self.exchange = "BINANCE"

    def from_binance_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if datatype == "spot":
            if msg["status"] != "TRADING":
                # print(f'=====no valid===> {msg}')
                return {}
            self.from_binance_spot_format(msg)
        elif datatype == "linear":
            if msg["status"] != "TRADING":
                # print(f'=====no valid===> {msg}')
                return {}
            self.from_binance_future_format(msg)
        elif datatype == "inverse":
            if msg["contractStatus"] != "TRADING":
                # print(f'=====no valid===> {msg}')
                return {}
            self.from_binance_inverse_future_format(msg)
        elif datatype == "option":
            self.from_binance_option_format(msg)
        else:
            self.from_binance_spot_format(msg)

        return self.to_general_form()

    def from_binance_v2_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print(f">>> msg:{msg}")
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = binance_asset_type(self.symbol, datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        for f in msg["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                self.tick_size = float(f["tickSize"])
            elif f["filterType"] == "LOT_SIZE":
                self.min_volume = float(f["minQty"])
            # elif f["filterType"] in ["LOT_SIZE", "MARKET_LOT_SIZE"]:
                self.step_size = float(f["stepSize"])
            elif f["filterType"] == "MIN_NOTIONAL":
                self.min_notional = float(f["minNotional"] if "minNotional" in f else f["notional"])
        self.base_currency = msg["baseAsset"]
        self.quote_currency = msg["quoteAsset"]
        self.product = datatype if datatype != "linear" else "future"
        self.extra = {
            "launch_date": timestamp_to_datetime(float(msg["onboardDate"]))
            if msg.get("onboardDate") and float(msg["onboardDate"])
            else None,
            "delivery_date": timestamp_to_datetime(float(msg["deliveryDate"]))
            if msg.get("deliveryDate") and float(msg["deliveryDate"])
            else None,
            # "option_type": msg["optionsType"].lower() if msg.get("optionsType") else None,
            # "delivery_fee_rate": float(msg["deliveryFeeRate"]) if msg.get("deliveryFeeRate") else None,
            "status": msg["status" if datatype != "inverse" else "contractStatus"],
        }
        self.exchange = "BINANCE"
            
    def from_binance_v2_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if msg["status" if datatype != "inverse" else "contractStatus"] != "TRADING":
            # print(f'=====no valid===> {msg}')
            return {}
        self.from_binance_v2_format(msg, datatype)

        return self.to_general_form()


@dataclass
class HistoryOHLCSchema(BaseData):
    asset_type: str | None = None
    symbol: str | None = None
    name: str | None = None
    dt: datetime | None = None
    interval: str | None = None
    volume: str | float | None = None
    turnover: str | float | None = None
    open: str | float | None = None
    high: str | float | None = None
    low: str | float | None = None
    close: str | float | None = None
    exchange: str | None = None

    def to_general_form(self) -> dict[str, Any]:
        return {
            "asset_type": self.asset_type,
            "code": self.symbol,
            "exchange": self.exchange,
            "symbol": self.name,
            "interval": self.interval,
            "turnover": self.turnover,
            "volume": self.volume,
            "close": self.close,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "datetime": self.dt,
        }

    def from_bitget_format(self, msg: dict[str, Any], extra: dict[str, Any], datatype: str) -> None:
        # print(f"msg:{msg}<<")
        if extra["symbol"]:
            self.symbol = str(extra["symbol"]).upper()
            self.asset_type = bitget_asset_type(self.symbol, datatype)
        if extra["interval"]:
            self.interval = extra["interval"]

        self.name = normalize_name(self.symbol, self.asset_type)
        self.dt = timestamp_to_datetime(msg["ts"])
        self.turnover = float(msg["usdtVol"]) if "usdtVol" in msg else float(msg["quoteVol"])
        self.volume = float(msg["baseVol"])
        self.open = float(msg["open"])
        self.high = float(msg["high"])
        self.low = float(msg["low"])
        self.close = float(msg["close"])
        self.exchange = "BITGET"

    def from_bitget_to_form(self, msg: dict[str, Any], extra: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_bitget_format(msg, extra, datatype)
        return self.to_general_form()

    def from_bitget_v2_format(self, msg: dict[str, Any], extra: dict[str, Any], datatype: str) -> None:
        # print(f"msg:{msg}<<")
        if extra["symbol"]:
            self.symbol = str(extra["symbol"]).upper()
            self.asset_type = bitget_asset_type(self.symbol, datatype)
        if extra["interval"]:
            self.interval = extra["interval"]

        self.name = normalize_name(self.symbol, self.asset_type)
        self.dt = timestamp_to_datetime(msg["ts"])
        self.turnover = float(msg["usdtVol"]) if "usdtVol" in msg else float(msg["quoteVol"])
        self.volume = float(msg["baseVol"])
        self.open = float(msg["open"])
        self.high = float(msg["high"])
        self.low = float(msg["low"])
        self.close = float(msg["close"])
        self.exchange = "BITGET"

    def from_bitget_v2_to_form(self, msg: dict[str, Any], extra: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_bitget_v2_format(msg, extra, datatype)
        return self.to_general_form()

    def from_bybit_v2_format(self, msg: dict[str, Any], extra: dict[str, Any], datatype: str) -> None:
        if extra["symbol"]:
            self.symbol = str(extra["symbol"]).upper()
            self.asset_type = bybit_asset_type(self.symbol, datatype)
        if extra["interval"]:
            self.interval = extra["interval"]

        self.name = normalize_name(self.symbol, self.asset_type)
        self.dt = timestamp_to_datetime(msg["start"])
        self.turnover = float(msg["turnover"])
        self.volume = float(msg["volume"])
        self.open = float(msg["open"])
        self.high = float(msg["high"])
        self.low = float(msg["low"])
        self.close = float(msg["close"])
        self.exchange = "BYBIT"

    def from_bybit_v2_to_form(self, msg: dict[str, Any], extra: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_bybit_v2_format(msg, extra, datatype)
        return self.to_general_form()

    def from_ftx_format(self, msg: dict[str, Any], extra: dict[str, Any] = {}) -> None:
        if extra["symbol"]:
            self.symbol = str(extra["symbol"]).upper()
            self.asset_type = ftx_asset_type(self.symbol)
        if extra["interval"]:
            self.interval = extra["interval"]
        self.name = normalize_name(self.symbol, self.asset_type)
        self.dt = isoformat_to_datetime(msg["startTime"])
        self.turnover = None
        self.volume = float(msg["volume"])
        self.open = float(msg["open"])
        self.high = float(msg["high"])
        self.low = float(msg["low"])
        self.close = float(msg["close"])
        self.exchange = "FTX"

    def from_ftx_to_form(self, msg: dict[str, Any], extra: dict[str, Any] = {}) -> dict[str, Any]:
        self.from_ftx_format(msg, extra)
        return self.to_general_form()

    def from_bingx_format(self, msg: dict[str, Any], datatype: str, extra: dict[str, Any] = {}) -> None:
        if extra["symbol"]:
            self.symbol = str(extra["symbol"]).upper()
            self.asset_type = bingx_asset_type(self.symbol, datatype)
        if extra["interval"]:
            self.interval = extra["interval"]
        self.name = normalize_name(self.symbol, self.asset_type)
        self.dt = timestamp_to_datetime(float(msg["time"]))
        self.volume = float(msg["volume"])
        self.open = float(msg["open"])
        self.high = float(msg["high"])
        self.low = float(msg["low"])
        self.close = float(msg["close"])
        self.turnover = self.close * self.volume
        self.exchange = "BingX"

    def from_bingx_to_form(self, msg: dict[str, Any], datatype: str, extra: dict[str, Any] = {}) -> dict[str, Any]:
        self.from_bingx_format(msg, datatype, extra)
        return self.to_general_form()

    def from_okx_format(self, msg: dict[str, Any], extra: dict[str, Any] = {}) -> None:
        # print(msg, extra)
        if extra["symbol"]:
            self.symbol = str(extra["symbol"]).upper()
            self.asset_type = okx_asset_type(self.symbol)
        if extra["interval"]:
            self.interval = extra["interval"]
        self.name = normalize_name(self.symbol, self.asset_type)
        self.dt = timestamp_to_datetime(float(msg["starttime"]))
        self.turnover = float(msg["turnover"])
        self.volume = float(msg["volume"])
        self.open = float(msg["open"])
        self.high = float(msg["high"])
        self.low = float(msg["low"])
        self.close = float(msg["close"])
        self.exchange = "OKX"

    def from_okx_to_form(
        self, msg: dict[str, Any], extra: dict[str, Any] = {}, datatype: str | None = None
    ) -> dict[str, Any]:
        self.from_okx_format(msg, extra)
        return self.to_general_form()

    def from_eqonex_format(self, msg: dict[str, Any], extra: dict[str, Any] = {}) -> None:
        if extra["symbol"]:
            self.symbol = str(extra["symbol"])
            self.asset_type = eqonex_asset_type(self.symbol)
        if extra["interval"]:
            self.interval = extra["interval"]

        price_scale = extra["price_scale"]
        self.name = normalize_name(self.symbol, self.asset_type)
        self.dt = timestamp_to_datetime(msg["starttime"])
        self.volume = float(msg["volume"])
        self.open = change_scale(float(msg["open"]), price_scale)
        self.high = change_scale(float(msg["high"]), price_scale)
        self.low = change_scale(float(msg["low"]), price_scale)
        self.close = change_scale(float(msg["close"]), price_scale)
        self.turnover = (self.volume * self.close) if (self.volume is not None and self.close is not None) else None
        self.exchange = "EQONEX"

    def from_eqonex_to_form(
        self, msg: dict[str, Any], extra: dict[str, Any] = {}, datatype: str | None = None
    ) -> dict[str, Any]:
        self.from_eqonex_format(msg, extra)
        return self.to_general_form()

    def from_bybit_spot_format(self, msg: dict[str, Any], extra: dict[str, Any] = {}) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        if extra["symbol"]:
            self.symbol = str(extra["symbol"]).upper()
        if extra["interval"]:
            self.interval = extra["interval"]
        self.name = normalize_name(self.symbol, self.asset_type)
        self.dt = timestamp_to_datetime(float(msg["startTime"]))
        self.volume = float(msg["volume"])
        self.open = float(msg["open"])
        self.high = float(msg["high"])
        self.low = float(msg["low"])
        self.close = float(msg["close"])
        self.turnover = float(msg["quoteAssetVolume"]) if msg["quoteAssetVolume"] else self.close * self.volume
        self.exchange = "BYBIT"

    def from_bybit_future_format(self, msg: dict[str, Any], extra: dict[str, Any] = {}) -> None:
        if extra["symbol"]:
            self.symbol = str(extra["symbol"]).upper()
        self.asset_type = future_asset_type(str(self.symbol), "future")
        if extra["interval"]:
            self.interval = extra["interval"]
        self.name = normalize_name(self.symbol, self.asset_type)
        self.dt = timestamp_to_datetime(float(msg["open_time"]) * 1000)
        self.turnover = float(msg["turnover"])
        self.volume = float(msg["volume"])
        self.open = float(msg["open"])
        self.high = float(msg["high"])
        self.low = float(msg["low"])
        self.close = float(msg["close"])
        self.exchange = "BYBIT"

    def from_bybit_inverse_future_format(self, msg: dict[str, Any], extra: dict[str, Any] = {}) -> None:
        if extra["symbol"]:
            self.symbol = str(extra["symbol"]).upper()
        self.asset_type = future_asset_type(str(self.symbol), "inverse")
        if extra["interval"]:
            self.interval = extra["interval"]
        self.name = normalize_name(self.symbol, self.asset_type)
        self.dt = timestamp_to_datetime(float(msg["open_time"]) * 1000)
        self.turnover = float(msg["turnover"])
        self.volume = float(msg["volume"])
        self.open = float(msg["open"])
        self.high = float(msg["high"])
        self.low = float(msg["low"])
        self.close = float(msg["close"])
        self.exchange = "BYBIT"

    def from_bybit_option_format(self, msg: dict[str, Any], extra: dict[str, Any] = {}) -> None:
        self.asset_type = ASSET_TYPE_OPTION
        if extra["symbol"]:
            self.symbol = str(extra["symbol"]).upper()
        if extra["interval"]:
            self.interval = extra["interval"]
        self.name = normalize_name(self.symbol, self.asset_type)
        self.dt = timestamp_to_datetime(float(msg["open_time"]))
        self.turnover = float(msg["turnover"])
        self.volume = float(msg["volume"])
        self.open = float(msg["open"])
        self.high = float(msg["high"])
        self.low = float(msg["low"])
        self.close = float(msg["close"])
        self.exchange = "BYBIT"

    def from_bybit_to_form(
        self, msg: dict[str, Any], extra: dict[str, Any] = {}, datatype: str = "spot"
    ) -> dict[str, Any]:
        if datatype == "spot":
            self.from_bybit_spot_format(msg, extra)
        elif datatype == "linear":
            self.from_bybit_future_format(msg, extra)
        elif datatype == "inverse":
            self.from_bybit_inverse_future_format(msg, extra)
        elif datatype == "option":
            self.from_bybit_option_format(msg, extra)
        else:
            self.from_bybit_spot_format(msg, extra)

        return self.to_general_form()

    def from_binance_spot_format(self, msg: dict[str, Any], extra: dict[str, Any] = {}, tz: Any | None = None) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        if extra["symbol"]:
            self.symbol = str(extra["symbol"]).upper()
        if extra["interval"]:
            self.interval = extra["interval"]
        self.name = normalize_name(self.symbol, self.asset_type)
        self.dt = timestamp_to_datetime(float(msg["startTime"]), tz=tz)
        self.turnover = float(msg["quoteAssetVolume"])
        self.volume = float(msg["volume"])
        self.open = float(msg["open"])
        self.high = float(msg["high"])
        self.low = float(msg["low"])
        self.close = float(msg["close"])
        self.exchange = "BINANCE"

    def from_binance_future_format(
        self, msg: dict[str, Any], extra: dict[str, Any] = {}, tz: Any | None = None
    ) -> None:
        if extra["symbol"]:
            self.symbol = str(extra["symbol"]).upper()
        self.asset_type = future_asset_type(str(self.symbol), "future")
        if extra["interval"]:
            self.interval = extra["interval"]
        self.name = normalize_name(self.symbol, self.asset_type)
        # print('????',self.asset_type, self.name,'????')
        self.dt = timestamp_to_datetime(float(msg["startTime"]), tz=tz)
        self.turnover = float(msg["quoteAssetVolume"])
        self.volume = float(msg["volume"])
        self.open = float(msg["open"])
        self.high = float(msg["high"])
        self.low = float(msg["low"])
        self.close = float(msg["close"])
        self.exchange = "BINANCE"

    def from_binance_inverse_future_format(
        self, msg: dict[str, Any], extra: dict[str, Any] = {}, tz: Any | None = None
    ) -> None:
        if extra["symbol"]:
            self.symbol = str(extra["symbol"]).upper()
        self.asset_type = future_asset_type(str(self.symbol), "inverse")
        if extra["interval"]:
            self.interval = extra["interval"]
        self.name = normalize_name(self.symbol, self.asset_type)
        self.dt = timestamp_to_datetime(float(msg["startTime"]), tz=tz)
        self.turnover = float(msg["quoteAssetVolume"])
        self.volume = float(msg["volume"])
        self.open = float(msg["open"])
        self.high = float(msg["high"])
        self.low = float(msg["low"])
        self.close = float(msg["close"])
        self.exchange = "BINANCE"

    def from_binance_option_format(
        self, msg: dict[str, Any], extra: dict[str, Any] = {}, tz: Any | None = None
    ) -> None:
        self.asset_type = ASSET_TYPE_OPTION
        if extra["symbol"]:
            self.symbol = str(extra["symbol"]).upper()
        if extra["interval"]:
            self.interval = extra["interval"]
        self.name = normalize_name(self.symbol, self.asset_type)
        self.dt = timestamp_to_datetime(float(msg["openTime"]), tz=tz)
        self.turnover = msg["takerVolume"]
        self.volume = float(msg["volume"])
        self.open = float(msg["open"])
        self.high = float(msg["high"])
        self.low = float(msg["low"])
        self.close = float(msg["close"])
        self.exchange = "BINANCE"

    def from_binance_to_form(
        self, msg: dict[str, Any], extra: dict[str, Any] = {}, datatype: str = "spot", tz: Any | None = None
    ) -> dict[str, Any]:
        if datatype == "spot":
            self.from_binance_spot_format(msg, extra, tz)
        elif datatype == "linear":
            self.from_binance_future_format(msg, extra, tz)
        elif datatype == "inverse":
            self.from_binance_inverse_future_format(msg, extra, tz)
        elif datatype == "option":
            self.from_binance_option_format(msg, extra, tz)
        else:
            return {}

        return self.to_general_form()

    def from_binance_v2_format(
        self,
        msg: dict[str, Any],
        datatype: str,
        extra: dict[str, Any] = {},
        tz: Any | None = None,
    ) -> None:
        if extra["symbol"]:
            self.symbol = str(extra["symbol"]).upper()
        if extra["interval"]:
            self.interval = extra["interval"]

        self.asset_type = binance_asset_type(self.symbol, datatype)

        self.name = normalize_name(self.symbol, self.asset_type)
        self.dt = timestamp_to_datetime(float(msg["startTime"]))  # , tz=tz
        self.turnover = float(msg["quoteAssetVolume"])
        self.volume = float(msg["volume"])
        self.open = float(msg["open"])
        self.high = float(msg["high"])
        self.low = float(msg["low"])
        self.close = float(msg["close"])
        self.exchange = "BINANCE"

    def from_binance_v2_to_form(
        self, msg: dict[str, Any], extra: dict[str, Any] = {}, datatype: str = "spot", tz: Any | None = None
    ) -> dict[str, Any]:
        self.from_binance_v2_format(msg, datatype, extra, tz)

        return self.to_general_form()


@dataclass
class TickerSchema(BaseData):
    asset_type: str | None = None
    symbol: str | None = None
    name: str | None = None
    price: float | None = None
    prev_volume_24h: float | None = None
    prev_turnover_24h: float | None = None
    open_interest: float | None = None
    ask_price_1: float | None = None
    bid_price_1: float | None = None
    ask_volume_1: float | None = None
    bid_volume_1: float | None = None
    exchange: str | None = None
    dt: datetime | None = None

    bid_iv: float | None = None
    ask_iv: float | None = None
    volume: float | None = None
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None
    underlying_price: float | None = None
    index_price: float | None = None
    mark_price: float | None = None
    mark_iv: float | None = None

    def to_general_form(self) -> dict[str, Any]:
        data = {
            "asset_type": self.asset_type,
            "code": self.symbol,
            "exchange": self.exchange,
            "symbol": self.name,
            "last_price": self.price,
            "prev_volume_24h": self.prev_volume_24h,
            "prev_turnover_24h": self.prev_turnover_24h,
            "open_interest": self.open_interest,
            "ask_price_1": self.ask_price_1,
            "bid_price_1": self.bid_price_1,
            "ask_volume_1": self.ask_volume_1,
            "bid_volume_1": self.bid_volume_1,
            "datetime": self.dt,
        }
        if self.asset_type == "OPTION":
            data["bid_iv"] = self.bid_iv
            data["ask_iv"] = self.ask_iv
            data["volume"] = self.volume
            data["delta"] = self.delta
            data["gamma"] = self.gamma
            data["vega"] = self.vega
            data["theta"] = self.theta
            data["underlying_price"] = self.underlying_price
            data["index_price"] = self.index_price
            data["mark_price"] = self.mark_price
            data["mark_iv"] = self.mark_iv

        return data

    def from_bitget_format(self, msg: dict[str, Any], symbol: str, datatype: str) -> None:
        self.symbol = symbol
        self.asset_type = bitget_asset_type(self.symbol, datatype)
        self.price = float(msg["close"]) if "close" in msg else float(msg["last"])
        self.exchange = "BITGET"
        self.name = normalize_name(self.symbol, self.asset_type)

    def from_bitget_to_form(self, msg: dict[str, Any], symbol: str, datatype: str) -> dict[str, Any]:
        self.from_bitget_format(msg, symbol, datatype)
        return self.to_general_form()

    def from_bitget_v2_format(self, msg: dict[str, Any], symbol: str, datatype: str) -> None:
        self.symbol = symbol
        self.asset_type = bitget_asset_type(self.symbol, datatype)
        self.price = float(msg["lastPr"])
        self.prev_volume_24h = float(msg["baseVolume"])
        self.prev_turnover_24h = float(msg["quoteVolume"])
        self.bid_price_1 = float(msg["bidPr"])
        self.bid_volume_1 = float(msg["bidSz"])
        self.ask_price_1 = float(msg["askPr"])
        self.ask_volume_1 = float(msg["askSz"])
        self.dt = timestamp_to_datetime(int(msg["ts"]))
        self.exchange = "BITGET"
        self.name = normalize_name(self.symbol, self.asset_type)

    def from_bitget_v2_to_form(self, msg: dict[str, Any], symbol: str, datatype: str) -> dict[str, Any]:
        self.from_bitget_v2_format(msg, symbol, datatype)
        return self.to_general_form()

    def from_bingx_format(self, msg: dict[str, Any], symbol: str, datatype: str) -> None:
        self.symbol = symbol
        self.asset_type = bingx_asset_type(self.symbol, datatype)
        self.price = float(msg["price"])
        self.exchange = "BingX"
        self.name = normalize_name(self.symbol, self.asset_type)

    def from_bingx_to_form(self, msg: dict[str, Any], symbol: str, datatype: str) -> dict[str, Any]:
        self.from_bingx_format(msg, symbol, datatype)
        return self.to_general_form()

    def from_bybit_v2_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print(msg)
        # option: {'symbol': 'BTC-8JUL23-30250-C', 'bid1Price': '0', 'bid1Size': '0', 'bid1Iv': '0', 'ask1Price': '0', 'ask1Size': '0', 'ask1Iv': '0', 'lastPrice': '0', 'highPrice24h': '0', 'lowPrice24h': '0', 'markPrice': '267.32394146', 'indexPrice': '30116.02', 'markIv': '0.4717', 'underlyingPrice': '30118.6729', 'openInterest': '0', 'turnover24h': '3633.260714', 'volume24h': '0.12', 'totalVolume': '1', 'totalTurnover': '6074', 'delta': '0.44208968', 'gamma': '0.00047994', 'vega': '6.88290974', 'theta': '-132.68767251', 'predictedDeliveryPrice': '0', 'change24h': '0'}
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = bybit_asset_type(self.symbol, datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.price = float(msg["lastPrice"])
        self.prev_volume_24h = float(msg["volume24h"])
        self.prev_turnover_24h = float(msg["turnover24h"])
        self.open_interest = float(msg["openInterest"])
        self.bid_price_1 = float(msg["bid1Price"]) if msg["bid1Price"] else None
        self.bid_volume_1 = float(msg["bid1Size"]) if msg["bid1Size"] else None
        self.ask_price_1 = float(msg["ask1Price"]) if msg["ask1Price"] else None
        self.ask_volume_1 = float(msg["ask1Size"]) if msg["ask1Size"] else None
        self.dt = generate_datetime()
        self.exchange = "BYBIT"
        if datatype == "option":
            self.bid_iv = float(msg["bid1Iv"]) if msg["bid1Iv"] else float(msg["bidIv"]) if msg["bidIv"] else None
            self.ask_iv = float(msg["ask1Iv"]) if msg["ask1Iv"] else float(msg["askIv"]) if msg["askIv"] else None
            self.volume = float(msg["totalVolume"])
            self.delta = float(msg["delta"])
            self.gamma = float(msg["gamma"])
            self.vega = float(msg["vega"])
            self.theta = float(msg["theta"])
            self.underlying_price = float(msg["underlyingPrice"])
            self.index_price = float(msg["indexPrice"])
            self.mark_price = float(msg["markPrice"])
            self.mark_iv = (
                float(msg["markIv"]) if msg["markIv"] else float(msg["markPriceIv"]) if msg["markPriceIv"] else None
            )

    def from_bybit_v2_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_bybit_v2_format(msg, datatype)
        return self.to_general_form()

    def from_dydx_v3_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        # print(msg)
        # option: {'symbol': 'BTC-8JUL23-30250-C', 'bid1Price': '0', 'bid1Size': '0', 'bid1Iv': '0', 'ask1Price': '0', 'ask1Size': '0', 'ask1Iv': '0', 'lastPrice': '0', 'highPrice24h': '0', 'lowPrice24h': '0', 'markPrice': '267.32394146', 'indexPrice': '30116.02', 'markIv': '0.4717', 'underlyingPrice': '30118.6729', 'openInterest': '0', 'turnover24h': '3633.260714', 'volume24h': '0.12', 'totalVolume': '1', 'totalTurnover': '6074', 'delta': '0.44208968', 'gamma': '0.00047994', 'vega': '6.88290974', 'theta': '-132.68767251', 'predictedDeliveryPrice': '0', 'change24h': '0'}
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = ASSET_TYPE_PERPETUAL
        self.price = float(msg["lastPrice"])
        self.bid_price_1 = float(msg["bid1Price"]) if msg["bid1Price"] else None
        self.bid_volume_1 = float(msg["bid1Size"]) if msg["bid1Size"] else None
        self.ask_price_1 = float(msg["ask1Price"]) if msg["ask1Price"] else None
        self.ask_volume_1 = float(msg["ask1Size"]) if msg["ask1Size"] else None
        self.exchange = "DYDX"
        self.name = normalize_name(self.symbol, self.asset_type)

        return self.to_general_form()

    def from_ftx_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["name"]).upper()
        self.asset_type = ftx_asset_type(self.symbol)
        self.price = float(msg["price"])
        self.exchange = "FTX"
        self.name = normalize_name(self.symbol, self.asset_type)

    def from_ftx_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.from_ftx_format(msg)
        return self.to_general_form()

    def from_okx_format(self, msg: dict[str, Any]) -> None:
        # print('>>> ',msg)
        # option: {'instType': 'OPTION', 'instId': 'BTC-USD-230714-27000-P', 'last': '0.0005', 'lastSz': '40', 'askPx': '0.0005', 'askSz': '4010', 'bidPx': '', 'bidSz': '0', 'open24h': '0.0005', 'high24h': '0.0005', 'low24h': '0.0005', 'volCcy24h': '0', 'vol24h': '0', 'ts': '1689301260012', 'sodUtc0': '0.0005', 'sodUtc8': '0.0005'}
        self.symbol = str(msg["instId"]).upper()
        self.asset_type = okx_asset_type(self.symbol)
        self.price = float(msg["last"]) if msg["last"] else None
        self.exchange = "OKX"
        self.name = normalize_name(self.symbol, self.asset_type)
        self.bid_price_1 = float(msg["bidPx"]) if msg["bidPx"] else None
        self.bid_volume_1 = float(msg["bidSz"]) if msg["bidSz"] else None
        self.ask_price_1 = float(msg["askPx"]) if msg["askPx"] else None
        self.ask_volume_1 = float(msg["askSz"]) if msg["askSz"] else None
        self.dt = timestamp_to_datetime(int(msg["ts"]))

    def from_okx_to_form(self, msg: dict[str, Any], datatype: str | None = None) -> dict[str, Any]:
        self.from_okx_format(msg)
        return self.to_general_form()

    def from_eqonex_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["s"]).upper()
        self.asset_type = eqonex_asset_type(self.symbol)
        self.price = float(msg["c"])
        self.exchange = "EQONEX"
        self.name = normalize_name(self.symbol, self.asset_type)

    def from_eqonex_to_form(self, msg: dict[str, Any], datatype: str | None = None) -> dict[str, Any]:
        self.from_eqonex_format(msg)
        return self.to_general_form()

    def from_bybit_spot_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.symbol = str(msg["symbol"]).upper()
        self.price = float(msg["price"])
        self.exchange = "BYBIT"
        self.name = normalize_name(self.symbol, self.asset_type)

    def from_bybit_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = future_asset_type(self.symbol, "future")
        self.price = msg["close"]
        self.exchange = "BYBIT"
        self.name = normalize_name(self.symbol, self.asset_type)

    def from_bybit_inverse_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = future_asset_type(self.symbol, "inverse")
        self.price = msg["close"]
        self.exchange = "BYBIT"
        self.name = normalize_name(self.symbol, self.asset_type)

    def from_bybit_option_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_OPTION
        self.symbol = str(msg["symbol"]).upper()
        self.price = msg["close"]
        self.exchange = "BYBIT"
        self.name = normalize_name(self.symbol, self.asset_type)

    def from_bybit_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if datatype == "spot":
            self.from_bybit_spot_format(msg)
        elif datatype == "linear":
            self.from_bybit_future_format(msg)
        elif datatype == "inverse":
            self.from_bybit_inverse_future_format(msg)
        elif datatype == "option":
            self.from_bybit_option_format(msg)
        else:
            self.from_bybit_spot_format(msg)

        return self.to_general_form()

    def from_binance_spot_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_SPOT
        self.symbol = str(msg["symbol"]).upper()
        self.price = float(msg["price"])
        self.exchange = "BINANCE"
        self.name = normalize_name(self.symbol, self.asset_type)

    def from_binance_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = future_asset_type(self.symbol, "future")
        self.price = float(msg["price"])
        self.exchange = "BINANCE"
        self.name = normalize_name(self.symbol, self.asset_type)

    def from_binance_inverse_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = future_asset_type(self.symbol, "inverse")
        self.price = msg["price"]
        self.exchange = "BINANCE"
        self.name = normalize_name(self.symbol, self.asset_type)

    def from_binance_option_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_OPTION
        self.symbol = str(msg["symbol"]).upper()
        self.price = float(msg["lastPrice"])
        self.exchange = "BINANCE"
        self.name = normalize_name(self.symbol, self.asset_type)

    def from_binance_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if datatype == "spot":
            self.from_binance_spot_format(msg)
        elif datatype == "linear":
            self.from_binance_future_format(msg)
        elif datatype == "inverse":
            self.from_binance_future_format(msg)
        elif datatype == "option":
            self.from_binance_option_format(msg)
        else:
            self.from_binance_spot_format(msg)
        return self.to_general_form()

    def from_binance_v2_format(self, msg: dict[str, Any], datatype: str) -> None:
        # print(f">> msg:{msg}")
        # >> msg:{'symbol': 'BTCUSDT', 'priceChange': '-232.66000000', 'priceChangePercent': '-0.542', 'weightedAvgPrice': '43005.05032794', 'prevClosePrice': '42897.11000000', 'lastPrice': '42664.44000000', 'lastQty': '0.00049000', 'bidPrice': '42662.19000000', 'bidQty': '0.00100000', 'askPrice': '42663.80000000', 'askQty': '0.00088000', 'openPrice': '42897.10000000', 'highPrice': '43772.00000000', 'lowPrice': '10000.01000000', 'volume': '1022.41397000', 'quoteVolume': '43968964.23583660', 'openTime': 1705394559452, 'closeTime': 1705480959452, 'firstId': 1206561, 'lastId': 1401132, 'count': 194572}
        # inverse >> msg:{'symbol': 'BTCUSD_PERP', 'pair': 'BTCUSD', 'priceChange': '2.0', 'priceChangePercent': '0.004', 'weightedAvgPrice': '44004.00034793', 'lastPrice': '45399.0', 'lastQty': '4', 'openPrice': '45397.0', 'highPrice': '45399.9', 'lowPrice': '42850.0', 'volume': '2507143', 'baseVolume': '5697.53427001', 'openTime': 1705472040000, 'closeTime': 1705558479469, 'firstId': 98107420, 'lastId': 98112738, 'count': 5316}
        # self.symbol = str(msg["symbol"]).upper()
        # self.asset_type = binance_asset_type(self.symbol, datatype=datatype)
        # self.price = float(msg["price"])
        # self.exchange = "BINANCE"
        # self.name = normalize_name(self.symbol, self.asset_type)

        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = binance_asset_type(self.symbol, datatype)
        self.name = normalize_name(self.symbol, self.asset_type)

        self.price = float(msg["lastPrice"])
        self.prev_volume_24h = float(msg["volume"])
        self.prev_turnover_24h = float(msg["quoteVolume"]) if "quoteVolume" in msg else float(msg["baseVolume"])
        self.bid_price_1 = float(msg["bidPrice"]) if msg.get("bidPrice") else None
        self.bid_volume_1 = float(msg["bidQty"]) if msg.get("bidQty") else None
        self.ask_price_1 = float(msg["askPrice"]) if msg.get("askPrice") else None
        self.ask_volume_1 = float(msg["askQty"]) if msg.get("askQty") else None
        self.dt = timestamp_to_datetime(int(msg["closeTime"]))
        self.exchange = "BINANCE"

    def from_binance_v2_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_binance_v2_format(msg, datatype)
        return self.to_general_form()


@dataclass
class LeverageSchema(BaseData):
    symbol: str | None = None
    leverage: str | None = None


@dataclass
class PositionModeSchema(BaseData):
    coin: str | None = None
    symbol: str | None = None
    position_mode: str | None = None


@dataclass
class MarginModeSchema(BaseData):
    coin: str | None = None
    symbol: str | None = None
    margin_mode: str | None = None


@dataclass
class SavingProductSchema(BaseData):
    asset_type: str | None = None
    symbol: str | None = None
    avg_apy: float | None = None
    exchange: str | None = None
    product_id: str | None = None
    purchased_amount: float | None = None
    can_purchase: bool | None = None
    can_redeem: bool | None = None
    featured: bool | None = None
    status: str | None = None
    up_limit: float | None = None
    up_limit_per_user: float | None = None

    def to_general_form(self) -> dict[str, Any]:
        return {
            "asset_type": self.asset_type,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "avg_apy": self.avg_apy,
            "product_id": self.product_id,
            "purchased_amount": self.purchased_amount,
            "can_purchase": self.can_purchase,
            "can_redeem": self.can_redeem,
            "featured": self.featured,
            "status": self.status,
            "up_limit": self.up_limit,
            "up_limit_per_user": self.up_limit_per_user,
        }

    def from_binance_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = ASSET_TYPE_EARN
        self.symbol = msg["asset"]
        self.avg_apy = float(msg["avgAnnualInterestRate"])
        self.exchange = "BINANCE"
        self.product_id = msg["productId"]
        self.purchased_amount = msg["purchasedAmount"]
        self.can_purchase = msg["canPurchase"]
        self.can_redeem = msg["canRedeem"]
        self.featured = msg["featured"]
        self.status = msg["status"]
        self.up_limit = msg["upLimit"]
        self.up_limit_per_user = msg["upLimitPerUser"]

    def from_binance_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        self.from_binance_format(msg)
        return self.to_general_form()

    def from_binance_v2_format(self, msg: dict[str, Any], datatype: str = "spot") -> None:
        self.asset_type = ASSET_TYPE_EARN
        self.symbol = msg["asset"]
        self.avg_apy = float(msg["avgAnnualInterestRate"])
        self.exchange = "BINANCE"
        self.product_id = msg["productId"]
        self.purchased_amount = msg["purchasedAmount"]
        self.can_purchase = msg["canPurchase"]
        self.can_redeem = msg["canRedeem"]
        self.featured = msg["featured"]
        self.status = msg["status"]
        self.up_limit = msg["upLimit"]
        self.up_limit_per_user = msg["upLimitPerUser"]

    def from_binance_v2_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        self.from_binance_v2_format(msg, datatype=datatype)
        return self.to_general_form()


@dataclass
class IncomeSchema(BaseData):
    asset_type: str | None = None
    symbol: str | None = None
    name: str | None = None
    income_type: str | None = None
    income: float | None = None
    asset: str | None = None
    info: str | None = None
    time: datetime | None = None

    def to_general_form(self) -> dict[str, Any]:
        return {
            "asset_type": self.asset_type,
            "code": self.symbol,
            "symbol": self.name,
            "income_type": self.income_type,
            "income": self.income,
            "asset": self.asset,
            "info": self.info,
            "time": self.time,
        }

    def from_ftx_format(self, msg: dict[str, Any]) -> None:
        self.symbol = msg["future"]
        self.asset_type = ftx_asset_type(self.symbol)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.income = msg["payment"]
        self.income_type = "Funding"
        self.asset = "USD"
        self.info = "TRANSFER"
        self.time = isoformat_to_datetime(msg["time"])

    def from_ftx_to_form(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.from_ftx_format(msg)
        return self.to_general_form()

    def from_bybit_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = future_asset_type(self.symbol, "future")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.income = float(msg["exec_fee"])
        self.income_type = "Funding"
        self.asset = "USD"
        self.info = "TRANSFER"
        self.time = isoformat_to_datetime(msg["exec_time"])

    def from_bybit_inverse_future_format(self, msg: dict[str, Any]) -> None:
        self.symbol = str(msg["symbol"])
        self.asset_type = future_asset_type(self.symbol, "inverse")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.income = msg["income"]
        self.income_type = msg["incomeType"]
        self.asset = msg["asset"]
        self.info = msg["info"]
        self.time = timestamp_to_datetime(float(msg["time"]))

    def from_bybit_spot_format(self, msg: dict[str, Any]) -> None:
        # Placeholder for spot format - spot income is rare
        self.symbol = msg.get("symbol", "")
        self.asset_type = ASSET_TYPE_SPOT
        self.name = normalize_name(self.symbol, self.asset_type)
        self.income = 0.0
        self.income_type = "Unknown"
        self.asset = "USD"
        self.info = "SPOT"
        self.time = generate_datetime()

    def from_bybit_option_format(self, msg: dict[str, Any]) -> None:
        # Placeholder for option format - option income is rare
        self.symbol = msg.get("symbol", "")
        self.asset_type = ASSET_TYPE_OPTION
        self.name = normalize_name(self.symbol, self.asset_type)
        self.income = 0.0
        self.income_type = "Unknown"
        self.asset = "USD"
        self.info = "OPTION"
        self.time = generate_datetime()

    def from_bybit_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if datatype == "spot":
            self.from_bybit_spot_format(msg)
        elif datatype == "linear":
            self.from_bybit_future_format(msg)
        elif datatype == "inverse":
            self.from_bybit_inverse_future_format(msg)
        elif datatype == "option":
            self.from_bybit_option_format(msg)
        else:
            self.from_bybit_spot_format(msg)

        return self.to_general_form()

    def from_binance_future_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = future_asset_type(str(self.symbol), "future")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.income_type = msg["incomeType"]
        self.time = timestamp_to_datetime(float(msg["time"]))

    def from_binance_inverse_future_format(self, msg: dict[str, Any]) -> None:
        self.asset_type = future_asset_type(str(self.symbol), "inverse")
        self.name = normalize_name(self.symbol, self.asset_type)
        self.income_type = msg["incomeType"]
        self.time = timestamp_to_datetime(float(msg["time"]))

    def from_binance_to_form(self, msg: dict[str, Any], datatype: str = "spot") -> dict[str, Any]:
        if datatype == "linear":
            self.from_binance_future_format(msg)
        elif datatype == "inverse":
            self.from_binance_inverse_future_format(msg)
        else:
            self.from_binance_future_format(msg)
        return self.to_general_form()


@dataclass
class TransferSchema(BaseData):
    """
    binance deposti
    {
        "id": "769800519366885376",
        "amount": "0.001",
        "coin": "BNB",
        "network": "BNB",
        "status": 0,
        "address": "bnb136ns6lfw4zs5hg4n85vdthaad7hq5m4gtkgf23",
        "addressTag": "101764890",
        "txId": "98A3EA560C6B3336D348B6C83F0F95ECE4F1F5919E94BD006E5BF3BF264FACFC",
        "insertTime": 1661493146000,
        "transferType": 0,
        "confirmTimes": "1/1",
        "unlockConfirm": 0,
        "walletType": 0
    },
    binance withdrawl
    {
        "address": "0x94df8b352de7f46f64b01d3666bf6e936e44ce60",
        "amount": "8.91000000",
        "applyTime": "2019-10-12 11:12:02",
        "coin": "USDT",
        "id": "b6ae22b3aa844210a7041aee7589627c",
        "withdrawOrderId": "WITHDRAWtest123", // will not be returned if there's no withdrawOrderId for this withdraw.
        "network": "ETH",
        "transferType": 0,   // 1 for internal transfer, 0 for external transfer
        "status": 6,
        "transactionFee": "0.004",
        "confirmNo":3,  // confirm times for withdraw
        "info":"The address is not valid. Please confirm with the recipient", // reason for withdrawal failure
        "txId": "0xb5ef8c13b968a406cc62a93a8bd80f9e9a906ef1b3fcf20a2e48573c17659268"
    },
    status:
        0(0:Email Sent,1:Cancelled 2:Awaiting Approval 3:Rejected 4:Processing 5:Failure 6:Completed)

    bybit despoit
            {
                "coin": "LTC",
                "chain": "LTC",
                "amount": "0.156",
                "tx_id": "XXXXXXXXXXXXXXXXXXXXXXXXXX",
                "status": 3,
                "to_address": "XXXXXXXXXXXXXXXXXXXXXXXXXX",
                "tag": "",
                "deposit_fee": "",
                "success_at": "1631697910",
                "confirmations": "0",
                "tx_index": "",
                "block_hash": ""
            },

    bybit withdrawl
            {
                "coin": "LTC",
                "chain": "LTC",
                "amount": "0.157",
                "tx_id": "XXXXXXXXXXXXXXXXXXXXXXXXXX",
                "status": "success",
                "to_address": "XXXXXXXXXXXXXXXXXXXXXXXXXX",
                "tag": "",
                "withdraw_fee": "0.001",
                "create_time": "1631694166",
                "update_time": "1631694775",
                "withdraw_id":"301121231312"
            },
    status:
        0=unknown
        1=ToBeConfirmed
        2=processing
        3=success
        4=deposit failed

    okx deposit
    {
        "actualDepBlkConfirm": "17",
        "amt": "135.705808",
        "ccy": "USDT",
        "chain": "USDT-TRC20",
        "depId": "34579090",
        "from": "",
        "state": "2",
        "to": "TN4hxxxxxxxxxxxizqs",
        "ts": "1655251200000",
        "txId": "16f36383292f451xxxxxxxxxxxxxxx584f3391642d988f97"
    }

    okx withdrawl
    {
        "chain": "ETH-Ethereum",
        "fee": "0.007",
        "ccy": "ETH",
        "clientId": "",
        "amt": "0.029809",
        "txId": "0x35c******b360a174d",
        "from": "156****359",
        "to": "0xa30d1fab********7CF18C7B6C579",
        "state": "2",
        "ts": "1655251200000",
        "wdId": "15447421"
    }

    status:
        0: pending 1:sending 2: sent
        3: awaiting email verification
        4: awaiting manual verification
        5: awaiting identity verification

    """

    coin: str | None = None
    network: str | None = None
    status: str | None = None  # pending, success, failed
    transaction_id: str | None = None
    amount: float | None = None
    fee: float | None = None
    dt: datetime | None = None
    transfer_type: str | None = None  # internal, external
    inout: str | None = None  # in, out
    address: str | None = None
    exchange: str | None = None

    def to_general_form(self) -> dict[str, Any]:
        return {
            "asset": self.coin,
            "network": self.network,
            "status": self.status,
            "address": self.address,
            "transaction_id": self.transaction_id,
            "amount": self.amount,
            "fee": self.fee,
            "datetime": self.dt,
            "transfer_type": self.transfer_type,
            "inout": self.inout,
            "exchange": self.exchange,
        }

    def from_bybit_v2_format(self, msg: dict[str, Any], transfer_type: str, inout: str) -> None:
        """
        external deposit: {'coin': 'ID', 'chain': 'BSC', 'amount': '301', 'txID': '0xe587df02ffea7b83ddf06812dd57d550328b0e3c3061c15e8d80e9dbb66b2e0a', 'status': 3, 'toAddress': '0xc7bda6ca0179653d3fb08a8933806732e6a83e99', 'tag': '', 'depositFee': '', 'successAt': '1679585297000', 'confirmations': '106', 'txIndex': '402', 'blockHash': '0x5d6486075d81e3ba3afbc93f409badeb68ef5717698b73aa271d2e991831ad79'}
        external withdraw: {'coin': 'USDT', 'chain': 'ARBI', 'amount': '51', 'txID': '0x8004221a670ee503c13aa737b55636a14c3b1c35db4427afbdbac0b28ce22955', 'status': 'success', 'toAddress': '0xe9c33326a3b83ce7905ed9fc54445d5f79cbca42', 'tag': '', 'withdrawFee': '0.3', 'createTime': '1680744215000', 'updateTime': '1680744254000', 'withdrawId': '14650317', 'withdrawType': 0}, {'coin': 'ETH', 'chain': 'ARBI', 'amount': '0.01414', 'txID': '0x4fdca8c1194f529d7498b04140c60000b8edc33f3d72ffa08ed228329434b753', 'status': 'success', 'toAddress': '0xF5e66Fc1dd20d411263ED0B0fFab359Be47fd829', 'tag': '', 'withdrawFee': '0.0003', 'createTime': '1679366935000', 'updateTime': '1679366990000', 'withdrawId': '13871748', 'withdrawType': 0}, {'coin': 'MATIC', 'chain': 'MATIC', 'amount': '9.65', 'txID': '0x720454872b1e1f1a350407d393a2d95ca18525f77e579c47b3c98e6a6e1d51b6', 'status': 'success', 'toAddress': '0xF5e66Fc1dd20d411263ED0B0fFab359Be47fd829', 'tag': '', 'withdrawFee': '0.1', 'createTime': '1679321623000', 'updateTime': '1679321745000', 'withdrawId': '13851997', 'withdrawType': 0}
        """
        self.coin = msg["coin"]
        self.network = msg["chain"]
        self.status = (
            "success"
            if msg["status"] in [3, "success"]
            else "failed"
            if msg["status"] in [4, "CancelByUser", "Reject", "Fail"]
            else "pending"
        )
        self.address = msg["toAddress"]
        self.transaction_id = msg["txID"]
        self.amount = float(msg["amount"])
        self.fee = (
            (float(msg["depositFee"]) if msg.get("depositFee") else 0.0) if inout == "in" else float(msg["withdrawFee"])
        )
        self.transfer_type = transfer_type
        self.inout = inout
        self.dt = (
            timestamp_to_datetime(int(msg["successAt"]))
            if inout == "in"
            else timestamp_to_datetime(int(msg["updateTime"]))
        )
        self.exchange = "BYBIT"

    def from_bybit_v2_to_form(self, msg: dict[str, Any], transfer_type: str, inout: str) -> dict[str, Any]:
        self.from_bybit_v2_format(msg, transfer_type=transfer_type, inout=inout)
        return self.to_general_form()

    def from_ftx_format(self, msg: dict[str, Any], transfer_type: str, inout: str) -> None:
        self.coin = msg["ccy"]
        self.network = msg["chain"]
        self.status = "success" if msg["status"] == 6 else "failed" if msg["status"] == 5 else "pending"
        self.address = msg["addr"]
        self.transaction_id = str(msg["txId"])
        self.amount = float(msg["amt"])
        self.fee = float(msg["fee"]) if msg.get("fee") else 0.0
        self.transfer_type = transfer_type
        self.inout = inout
        self.dt = timestamp_to_datetime(int(msg["ts"]))
        self.exchange = "FTX"

    def from_ftx_to_form(self, msg: dict[str, Any], transfer_type: str, inout: str) -> dict[str, Any]:
        self.from_ftx_format(msg, transfer_type=transfer_type, inout=inout)
        return self.to_general_form()

    def from_bingx_format(self, msg: dict[str, Any], transfer_type: str, inout: str) -> None:
        self.coin = msg["coin"]
        self.network = msg["network"]
        self.status = "success" if int(msg["status"]) == 0 else "pending"
        self.address = msg["address"]
        self.transaction_id = str(msg["txId"])
        self.amount = float(msg["amount"])
        self.fee = float(msg["transactionFee"]) if "transactionFee" in msg else None
        self.transfer_type = transfer_type
        self.inout = inout
        self.dt = (
            timestamp_to_datetime(int(msg["insertTime"]))
            if "insertTime" in msg
            else isoformat_to_datetime(msg["applyTime"])
        )
        self.exchange = "BINGX"

    def from_bingx_to_form(self, msg: dict[str, Any], transfer_type: str, inout: str) -> dict[str, Any]:
        self.from_bingx_format(msg, transfer_type=transfer_type, inout=inout)
        return self.to_general_form()

    def from_okx_format(self, msg: dict[str, Any], transfer_type: str, inout: str) -> None:
        self.coin = msg["ccy"]
        self.network = msg["chain"]
        self.status = "success" if int(msg["state"]) == 2 else "pending"
        self.address = msg["to"]
        self.transaction_id = str(msg["txId"])
        self.amount = float(msg["amt"])
        self.fee = float(msg["fee"]) if msg.get("fee") else 0.0
        self.transfer_type = transfer_type
        self.inout = inout
        self.dt = timestamp_to_datetime(int(msg["ts"]))
        self.exchange = "OKX"

    def from_okx_to_form(self, msg: dict[str, Any], transfer_type: str, inout: str) -> dict[str, Any]:
        self.from_okx_format(msg, transfer_type=transfer_type, inout=inout)
        return self.to_general_form()

    def from_bybit_format(self, msg: dict[str, Any], transfer_type: str, inout: str) -> None:
        self.coin = msg["coin"]
        self.network = msg["chain"]
        self.status = "success" if msg["status"] == 3 else "failed" if msg["status"] == 4 else "pending"
        self.address = msg["to_address"]
        self.transaction_id = msg["tx_id"]
        self.amount = float(msg["amount"])
        self.fee = (
            (float(msg["deposit_fee"]) if msg.get("deposit_fee") else 0.0)
            if inout == "in"
            else float(msg["withdraw_fee"])
        )
        self.transfer_type = transfer_type
        self.inout = inout
        self.dt = (
            timestamp_to_datetime(int(msg["success_at"]))
            if inout == "in"
            else timestamp_to_datetime(int(msg["update_time"]))
        )
        self.exchange = "BYBIT"

    def from_bybit_to_form(self, msg: dict[str, Any], transfer_type: str, inout: str) -> dict[str, Any]:
        self.from_bybit_format(msg, transfer_type=transfer_type, inout=inout)
        return self.to_general_form()

    def from_binance_format(self, msg: dict[str, Any], transfer_type: str, inout: str) -> None:
        self.coin = msg["coin"]
        self.network = msg["network"]
        self.status = (
            ("success" if msg["status"] == 6 else "failed" if msg["status"] == 5 else "pending")
            if inout == "out"
            else ("success" if msg["status"] == 1 else "failed" if msg["status"] in [6, 7] else "pending")
        )
        self.address = msg["address"]
        self.transaction_id = str(msg["id"])
        self.amount = float(msg["amount"])
        self.fee = float(msg["transactionFee"]) if msg.get("transactionFee") else 0.0
        self.transfer_type = transfer_type
        self.inout = inout
        self.dt = (
            isoformat_to_datetime(msg["completeTime"])
            if "completeTime" in msg
            else timestamp_to_datetime(float(msg["insertTime"]))
            if "insertTime" in msg
            else isoformat_to_datetime(msg["applyTime"])
        )
        self.exchange = "BINANCE"

    def from_binance_to_form(self, msg: dict[str, Any], transfer_type: str, inout: str) -> dict[str, Any]:
        self.from_binance_format(msg, transfer_type=transfer_type, inout=inout)
        return self.to_general_form()

    def from_binance_v2_format(self, msg: dict[str, Any], transfer_type: str, inout: str) -> None:
        self.coin = msg["coin"]
        self.network = msg["network"]
        self.status = (
            ("success" if msg["status"] == 6 else "failed" if msg["status"] == 5 else "pending")
            if inout == "out"
            else ("success" if msg["status"] == 1 else "failed" if msg["status"] in [6, 7] else "pending")
        )
        self.address = msg["address"]
        self.transaction_id = str(msg["id"])
        self.amount = float(msg["amount"])
        self.fee = float(msg["transactionFee"]) if msg.get("transactionFee") else 0.0
        self.transfer_type = transfer_type
        self.inout = inout
        self.dt = (
            isoformat_to_datetime(msg["completeTime"])
            if "completeTime" in msg
            else timestamp_to_datetime(float(msg["insertTime"]))
            if "insertTime" in msg
            else isoformat_to_datetime(msg["applyTime"])
        )
        self.exchange = "BINANCE"

    def from_binance_v2_to_form(self, msg: dict[str, Any], transfer_type: str, inout: str) -> dict[str, Any]:
        self.from_binance_v2_format(msg, transfer_type=transfer_type, inout=inout)
        return self.to_general_form()


@dataclass
class FundingRateSchema(BaseData):
    """Schema for funding rate historical data"""

    symbol: str | None = None
    name: str | None = None
    asset_type: str | None = None
    funding_rate: float | None = None
    funding_dt: datetime | None = None
    exchange: str | None = None

    def to_general_form(self) -> dict[str, Any]:
        return {
            "code": self.symbol,
            "symbol": self.name,
            "asset_type": self.asset_type,
            "funding_rate": self.funding_rate,
            "datetime": self.funding_dt,
            "exchange": self.exchange,
        }

    def from_bybit_v2_format(self, msg: dict[str, Any], datatype: str) -> None:
        """
        Parse Bybit V5 funding rate history format
        Example msg: {
            "symbol": "ETHPERP",
            "fundingRate": "0.0001",
            "fundingRateTimestamp": "1672041600000"
        }
        """
        self.symbol = str(msg["symbol"]).upper()
        self.asset_type = bybit_asset_type(self.symbol, datatype=datatype)
        self.name = normalize_name(self.symbol, self.asset_type)
        self.funding_rate = float(msg["fundingRate"])
        self.funding_dt = timestamp_to_datetime(int(msg["fundingRateTimestamp"]))
        self.exchange = "BYBIT"

    def from_bybit_v2_to_form(self, msg: dict[str, Any], datatype: str) -> dict[str, Any]:
        self.from_bybit_v2_format(msg, datatype=datatype)
        return self.to_general_form()
