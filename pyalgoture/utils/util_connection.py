from collections.abc import Callable
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from time import sleep, time
from traceback import print_exc
from typing import Union

from dateutil.parser import parse

# from ..exchanges.bitget import BitgetClient, BitgetDataWebsocket, BitgetTradeWebsocket
# from ..exchanges.bybit import BybitFutureClient, BybitFutureDataWebsocket, BybitFutureTradeWebsocket, BybitInverseFutureClient, BybitInverseFutureDataWebsocket, BybitInverseFutureTradeWebsocket, BybitSpotClient, BybitSpotDataWebsocket, BybitSpotTradeWebsocket
from ..exchanges.binance_v2 import (
    BinanceClient,
    BinanceDataWebsocket,
    BinanceTradeWebsocket,
)

# from ..exchanges.binance import BinanceFutureClient, BinanceFutureDataWebsocket, BinanceFutureTradeWebsocket, BinanceInverseFutureClient, BinanceInverseFutureDataWebsocket, BinanceInverseFutureTradeWebsocket, BinanceSpotClient, BinanceSpotDataWebsocket, BinanceSpotTradeWebsocket
# from ..exchanges.bingx import BingxClient, BingxDataWebsocket, BingxTradeWebsocket
from ..exchanges.bitget_v2 import (
    BitgetClient,
    BitgetDataWebsocket,
    BitgetTradeWebsocket,
)
from ..exchanges.bybit_v2 import BybitClient, BybitDataWebsocket, BybitTradeWebsocket
from ..exchanges.dummy import DummyClient, DummyTradeWebsocket

# from ..exchanges.eqonex import EqonexClient, EqonexDataWebsocket, EqonexTradeWebsocket
from ..exchanges.okx import OKXClient, OKXDataWebsocket, OKXTradeWebsocket
from .objects import AssetType, Exchange
from .util_dt import tz_manager

# ============================= Exchange utils ============================


PERMISSION_ERROR_MSG = {
    "binance": [
        "Invalid API-key, IP, or permissions for action."
    ],  # binance # 'Invalid API-key, IP, or permissions for action, request ip: 18.162.143.83'
    "bybit": [
        "Invalid API-key, IP, or permissions for action.",
        "This operation is not supported.",
        "unmatched ip.",
        "api_key expire",
        "Permission denied!",
    ],  # bybit
}
RETURN_PERMISSION_ERROR_MSG = "Your API key has no corresponding permission. Please update and renew your API key."
RETURN_PERMISSION_ERROR_MSG_ZH = "您的 API 密鑰沒有相應的權限。 請更新並更新您的 API 密鑰。"

# BINANCE_DEFAULT_SPOT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT", "TRXUSDT", "LINKUSDT", "UNIUSDT", "AXSUSDT", "SANDUSDT", "MANAUSDT"]
# BINANCE_DEFAULT_FUTURE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT", "GMTUSDT", "DOGEUSDT", "APEUSDT", "LINKUSDT", "UNIUSDT", "AXSUSDT", "SANDUSDT", "MANAUSDT"]  # , 'BTCUSDT_220624'
# BINANCE_DEFAULT_INVERSE_FUTURE_SYMBOLS = ["BTCUSD_PERP", "ETHUSD_PERP", "BNBUSD_PERP", "SOLUSD_PERP", "XRPUSD_PERP", "DOGEUSD_PERP", "BNBUSD_PERP", "APEUSD_PERP", "MATICUSD_PERP"]  #'BTCUSD_220930'] # ,'BTCUSD_220624'

# BYBIT_DEFAULT_SPOT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT", "TRXUSDT", "LINKUSDT", "UNIUSDT", "AXSUSDT", "SANDUSDT", "MANAUSDT", "BITUSDT"]
# BYBIT_DEFAULT_FUTURE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT", "LINKUSDT", "UNIUSDT", "AXSUSDT", "SANDUSDT", "MANAUSDT", "BITUSDT", "GMTUSDT", "APEUSDT"]  # 'TRXUSDT',
# BYBIT_DEFAULT_INVERSE_FUTURE_SYMBOLS = ["BTCUSD", "ETHUSD", "ADAUSD", "XRPUSD", "DOTUSD", "LTCUSD", "BITUSD", "EOSUSD", "MANAUSD"]  # "ETHUSDU22", "BTCUSDU22", "SOLUSD", 'BTCUSDM22',  'ETHUSDM22',

BINANCE_DEFAULT_SPOT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
BINANCE_DEFAULT_FUTURE_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
BINANCE_DEFAULT_INVERSE_FUTURE_SYMBOLS = ["BTCUSD_PERP", "ETHUSD_PERP"]

BYBIT_DEFAULT_SPOT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
BYBIT_DEFAULT_FUTURE_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
BYBIT_DEFAULT_INVERSE_FUTURE_SYMBOLS = ["BTCUSD", "ETHUSD"]

EQONEX_DEFAULT_SPOT_SYMBOLS = ["BTC/USDC", "ETH/USDC"]
EQONEX_DEFAULT_FUTURE_SYMBOLS = ["BTC/USDC[F]", "ETH/USDC[F]"]

BINGX_DEFAULT_SPOT_SYMBOLS = ["BTC-USDT", "ETH-USDT"]
BINGX_DEFAULT_FUTURE_SYMBOLS = ["BTC-USDT", "ETH-USDT"]

BITGET_DEFAULT_SPOT_SYMBOLS = ["BTCUSDT_SPBL", "ETHUSDT_SPBL"]
BITGET_DEFAULT_FUTURE_SYMBOLS = ["BTCUSDT_UMCBL", "ETHUSDT_UMCBL"]
BITGET_DEFAULT_INVERSE_FUTURE_SYMBOLS = ["BTCUSD_DMCBL", "ETHUSD_DMCBL"]

DEFAULT_SYMBOL_MAP = {
    "binance": {
        "spot": ["BTCUSDT", "ETHUSDT"],
        "future": ["BTCUSDT", "ETHUSDT"],
        "inverse_future": ["BTCUSD_PERP", "ETHUSD_PERP"],
    },
    "bybit": {
        "spot": ["BTCUSDT", "ETHUSDT"],
        "future": ["BTCUSDT", "ETHUSDT"],
        "inverse_future": ["BTCUSD", "ETHUSD"],
    },
    "eqonex": {
        "spot": ["BTC/USDC", "ETH/USDC"],
        "future": ["BTC/USDC[F]", "ETH/USDC[F]"],
    },
    "bingx": {
        "spot": ["BTC-USDT", "ETH-USDT"],
        "future": ["BTC-USDT", "ETH-USDT"],
    },
    "bitget": {
        "spot": ["BTCUSDT", "ETHUSDT"],
        "future": ["BTCUSDT", "ETHUSDT"],
        "inverse_future": ["BTCUSD", "ETHUSD"],
    },
    # "bitget": {
    #     "spot": ["BTCUSDT_SPBL", "ETHUSDT_SPBL"],
    #     "future": ["BTCUSDT_UMCBL", "ETHUSDT_UMCBL"],
    #     "inverse_future": ["BTCUSD_DMCBL", "ETHUSD_DMCBL"],
    # },
}


def _check_permission_error_msg(msg: str, exchange: str | None = None, ret_tf: bool = False) -> str | tuple[bool, str]:
    """
    Check if error message indicates permission issues and return standardized message.

    Args:
        msg: Error message to check
        exchange: Exchange name for specific error patterns
        ret_tf: Whether to return tuple with boolean flag

    Returns:
        Processed error message or tuple of (is_permission_error, message)
    """
    is_per_err = False

    if exchange:
        err_msgs = PERMISSION_ERROR_MSG.get(exchange, [])
    else:
        err_msgs = [item for sublist in PERMISSION_ERROR_MSG.values() for item in sublist]

    for per_err_msg in err_msgs:
        similarity = SequenceMatcher(None, per_err_msg.lower(), msg.lower()).ratio()
        if similarity > 0.88:
            is_per_err = True
            msg = RETURN_PERMISSION_ERROR_MSG
            break

    return (is_per_err, msg) if ret_tf else msg


def get_exchange_connection(
    exchange: str | Exchange,
    asset_type: str | AssetType,
    return_connection_only: bool = False,
    return_class_only: bool = False,
    is_connect: bool = True,
    is_testnet: bool = False,
    is_demo: bool = False,
    username: str | None = None,
    password: str | None = None,
    key: str | None = None,
    secret: str | None = None,
    user_id: str | None = None,
    passphrase: str | None = None,
    is_data_websocket: bool = False,
    is_user_websocket: bool = False,
    on_account_callback: Callable | None = None,
    on_order_callback: Callable | None = None,
    on_trade_callback: Callable | None = None,
    on_position_callback: Callable | None = None,
    on_error_callback: Callable | None = None,
    on_connected_callback: Callable | None = None,
    on_disconnected_callback: Callable | None = None,
    debug: bool = False,
    session: bool = False,
    is_dummy: bool = False,
    is_overwrite: bool = False,
    balance: float | None = None,
    logger=None,
    **kwargs,
):
    connection_class = None
    connection = None
    payload = {}
    if isinstance(exchange, str):
        exchange = Exchange(exchange.upper())
    if asset_type and isinstance(asset_type, str):
        asset_type = AssetType(asset_type.upper())

    if not asset_type:
        payload["datatype"] = ""
    elif asset_type in [AssetType.SPOT, AssetType.SAVING, AssetType.EARN]:
        payload["datatype"] = "spot"
    elif asset_type in [
        AssetType.PERPETUAL,
        AssetType.DATED_FUTURE,
        AssetType.FUTURE,
        AssetType.LINEAR,
    ]:
        payload["datatype"] = "linear"
    elif asset_type in [
        AssetType.INVERSE_PERPETUAL,
        AssetType.INVERSE_DATED_FUTURE,
        AssetType.INVERSE_FUTURE,
        AssetType.INVERSE,
    ]:
        payload["datatype"] = "inverse"
    elif asset_type == AssetType.OPTION:
        payload["datatype"] = "option"

    if is_dummy:
        if is_user_websocket:
            connection_class = DummyTradeWebsocket
        else:
            connection_class = DummyClient
            payload["exchange"] = exchange.value
            payload["asset_type"] = asset_type.value
            payload["overwrite"] = is_overwrite
            payload["balance"] = balance
    elif exchange in [Exchange.BINANCE]:
        if is_data_websocket:
            connection_class = BinanceDataWebsocket
        elif is_user_websocket:
            connection_class = BinanceTradeWebsocket
        else:
            connection_class = BinanceClient
    elif exchange in [Exchange.BYBIT]:
        if is_data_websocket:
            connection_class = BybitDataWebsocket
        elif is_user_websocket:
            connection_class = BybitTradeWebsocket
        else:
            connection_class = BybitClient
    # elif exchange in [Exchange.EQONEX]:
    #     if is_data_websocket:
    #         connection_class = EqonexDataWebsocket
    #     elif is_user_websocket:
    #         connection_class = EqonexTradeWebsocket
    #     else:
    #         connection_class = EqonexClient
    elif exchange in [Exchange.OKX]:
        if is_data_websocket:
            connection_class = OKXDataWebsocket
        elif is_user_websocket:
            connection_class = OKXTradeWebsocket
        else:
            connection_class = OKXClient
    # elif exchange in [Exchange.BINGX]:
    #     if is_data_websocket:
    #         connection_class = BingxDataWebsocket
    #     elif is_user_websocket:
    #         connection_class = BingxTradeWebsocket
    #     else:
    #         connection_class = BingxClient
    elif exchange in [Exchange.BITGET]:
        if is_data_websocket:
            connection_class = BitgetDataWebsocket
        elif is_user_websocket:
            connection_class = BitgetTradeWebsocket
        else:
            connection_class = BitgetClient

    if not connection_class:
        raise ValueError("Unknown asset type or exchange")

    if return_class_only:
        return connection_class

    if debug:
        payload["debug"] = debug
    if is_testnet:
        payload["is_testnet"] = is_testnet
    if is_demo:
        payload["is_demo"] = is_demo
    if on_account_callback:
        payload["on_account_callback"] = on_account_callback
    if on_order_callback:
        payload["on_order_callback"] = on_order_callback
    if on_trade_callback:
        payload["on_trade_callback"] = on_trade_callback
    if on_position_callback:
        payload["on_position_callback"] = on_position_callback
    if on_error_callback:
        payload["on_error_callback"] = on_error_callback
    if on_connected_callback:
        payload["on_connected_callback"] = on_connected_callback
    if on_disconnected_callback:
        payload["on_disconnected_callback"] = on_disconnected_callback
    if session:
        payload["session"] = session
    if logger:
        payload["logger"] = logger

    try:
        connection = connection_class(**payload)

        if is_connect:
            connection_payload = {}
            if key and secret:
                connection_payload["key"] = key
                connection_payload["secret"] = secret
                if exchange in [Exchange.EQONEX]:
                    connection_payload["user_id"] = user_id
                elif exchange in [Exchange.OKX, Exchange.BITGET]:
                    connection_payload["passphrase"] = passphrase
                # print(f"connection: {connection}; connection_payload: {connection_payload}")
                conn_res = connection.connect(**connection_payload)
                if not conn_res:
                    connection = None

            if is_dummy and username and password:
                connection.register(username=username, password=password)
                connection_payload["username"] = username
                connection_payload["password"] = password

                conn_res = connection.connect(**connection_payload)
                if not conn_res:
                    connection = None

    except Exception as e:
        print_exc()
        print(
            f"[ERROR] Something went wrong when get_exchange_connection. Error:{str(e)}. exchange:{exchange}; asset_type:{asset_type}; key:{key}; secret:{secret}; user_id:{user_id}; passphrase:{passphrase} ; is_testnet:{is_testnet}; is_demo:{is_demo}; is_data_websocket:{is_data_websocket}; is_user_websocket:{is_user_websocket}; "
        )
        connection = None

    if return_connection_only:
        return connection

    return connection, exchange, asset_type


def fetch_trades(
    exchange: str,
    asset_type: str,
    key: str | None = None,
    secret: str | None = None,
    user_id: str | None = None,
    passphrase: str | None = None,
    connection=None,
    is_testnet: bool = False,
    is_demo: bool = False,
    symbols: list = [],
    additional_symbols: list = [],
    logger=None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    is_debug: bool = False,
    sleep_time: float | None = None,
    wait_rate_limit: bool = False,
    **kwargs,
) -> tuple[bool, str | dict | list]:
    """
    by symbol
        - bybit future
        - bybit inverse future
        - binance spot
        - binance future
        - binance inverse future
    """
    assert asset_type in ["spot", "future", "inverse_future", "option"]
    # assert exchange in ["eqonex", "binance", "bybit", "okx", "bingx", 'bitget']
    std_err_msg = f"Trade fetch failed. === key: {key} -- exchange: {exchange} -- asset_type: {asset_type} -- symbols: {symbols} -- additional_symbols: {additional_symbols} -- start_time: {start_time} -- end_time:{end_time} -- extra info:{kwargs} "
    try:
        if symbols and isinstance(symbols, str):
            symbols = [symbols]
        if start_time and isinstance(start_time, str):
            start_time = parse(start_time)
        if end_time and isinstance(end_time, str):
            end_time = parse(end_time)
        if not connection:
            connection = get_exchange_connection(
                exchange=exchange,
                asset_type=asset_type,
                is_testnet=is_testnet,
                is_demo=is_demo,
                key=key,
                secret=secret,
                user_id=user_id,
                passphrase=passphrase,
                return_connection_only=True,
            )
        if not sleep_time:
            if len(symbols) > 200:
                if exchange == "binance" and asset_type == "spot":
                    sleep_time = 0.8
                else:
                    sleep_time = 0.4
            else:
                sleep_time = 0.1

        res = {}
        if symbols:
            ret = []
            for symbol in symbols:
                st = time()
                # if is_debug:
                #     print(f'asset_type: {asset_type}; symbol: {symbol}; exchange: {exchange} - extra info:{kwargs}')
                res = connection.query_trades(
                    symbol=symbol.upper(),
                    start=start_time,
                    end=end_time if end_time else datetime.now(),
                )  # start=datetime(2019,1,1))
                sleep(sleep_time)
                if res.get("error", True):
                    if wait_rate_limit:
                        print(f"res:{res}???")
                        status_code = res.get("data", {}).get("status_code")
                        if status_code and status_code == 429:
                            print("Rate limit exceeded, stop for 60s.....")
                            # TODO: retry the same symbol
                            sleep(60)
                    if_per_err, msg = _check_permission_error_msg(
                        msg=res.get("msg", ""), exchange=exchange, ret_tf=True
                    )
                    if if_per_err:
                        return False, f"Trade fetch failed. {msg}"

                    if logger:
                        logger.error(f"[{symbol}] " + std_err_msg % res)
                    else:
                        print(f"[{symbol}] " + std_err_msg % res)
                else:
                    # print(res, '!!!!')
                    if res.get("data"):
                        if is_debug:
                            print(
                                f">>> [fetch_trades] asset_type: {asset_type}; symbol: {symbol}; exchange: {exchange} - sd:{res['data'][0]['datetime']} ; ed:{res['data'][-1]['datetime']} ; extra info:{kwargs} - trades length: {len(res['data'])} || time consumed:{time() - st}s"
                            )
                        ret += res["data"]
            return True, ret
        else:
            if exchange in ["bybit", "binance", "bingx", "bitget"]:
                ret = []
                if not symbols:
                    if exchange == "bybit" and asset_type in [
                        "future",
                        "inverse_future",
                    ]:
                        # if asset_type == "future":
                        #     symbols = BYBIT_DEFAULT_FUTURE_SYMBOLS
                        # else:
                        #     symbols = BYBIT_DEFAULT_INVERSE_FUTURE_SYMBOLS
                        symbols = DEFAULT_SYMBOL_MAP[exchange][asset_type]
                    elif exchange == "binance" and asset_type in [
                        "spot",
                        "future",
                        "inverse_future",
                    ]:
                        # if asset_type == "future":
                        #     symbols = BINANCE_DEFAULT_FUTURE_SYMBOLS
                        # elif asset_type == "inverse_future":
                        #     symbols = BINANCE_DEFAULT_INVERSE_FUTURE_SYMBOLS
                        # else:
                        #     symbols = BINANCE_DEFAULT_SPOT_SYMBOLS
                        symbols = DEFAULT_SYMBOL_MAP[exchange][asset_type]
                    elif exchange == "bitget" and asset_type in [
                        "spot",
                        "future",
                        "inverse_future",
                    ]:
                        symbols = DEFAULT_SYMBOL_MAP[exchange][asset_type]
                    elif exchange == "bingx" and asset_type in ["spot", "future"]:
                        symbols = DEFAULT_SYMBOL_MAP[exchange][asset_type]
                if additional_symbols:
                    symbols += additional_symbols
                    symbols = list(set(symbols))
                # if exchange == "bybit" and asset_type == "future" and start_time: # NOTE: this is no longer needed in v2
                #     # Special handling - if specify start time, bybit willl only return less 7days data
                #     past7days = datetime.now() - timedelta(days=7)
                #     start_time = start_time.replace(tzinfo=None)
                #     if start_time < past7days:
                #         start_time = None
                if exchange == "binance" and asset_type == "future" and start_time:
                    # Special handling - if specify start time, binance willl only return less 7days data
                    past7days = datetime.now() - timedelta(days=6)
                    start_time = start_time.replace(tzinfo=None)
                    if start_time < past7days:
                        start_time = None
                for symbol in symbols:
                    # if is_debug:
                    #     print(f'asset_type: {asset_type}; symbol: {symbol}; exchange: {exchange} - extra info:{kwargs}')
                    res = connection.query_trades(
                        symbol=symbol.upper(),
                        start=start_time,
                        end=end_time if end_time else datetime.now(),
                    )  # start=datetime(2019,1,1))
                    sleep(sleep_time)
                    if res.get("error", True):
                        if wait_rate_limit:
                            status_code = res.get("data", {}).get("status_code")
                            if status_code and status_code == 429:
                                print("Rate limit exceeded, stop for 60s.....")
                                # TODO: retry the same symbol
                                sleep(60)
                        if_per_err, msg = _check_permission_error_msg(
                            msg=res.get("msg", ""), exchange=exchange, ret_tf=True
                        )
                        if if_per_err:
                            return False, f"Trade fetch failed. {msg}"

                        if logger:
                            logger.error(f"[{symbol}] " + std_err_msg % res)
                        else:
                            print(f"[{symbol}] " + std_err_msg % res)
                    else:
                        # print(res, '!!!!')
                        if res.get("data"):
                            if is_debug:
                                print(
                                    f"asset_type: {asset_type}; symbol: {symbol}; exchange: {exchange} - sd:{res['data'][0]['datetime']} ; ed:{res['data'][-1]['datetime']} ; extra info:{kwargs} - trades length: {len(res['data'])}"
                                )
                            ret += res["data"]
                return True, ret

            res = connection.query_trades(start=start_time)  # start=datetime(2019,1,1))

            connection.close()

            if res.get("error", True):
                if_per_err, msg = _check_permission_error_msg(msg=res.get("msg", ""), exchange=exchange, ret_tf=True)
                if logger:
                    if if_per_err:
                        logger.warning(std_err_msg % res)
                    else:
                        logger.error(std_err_msg % res)
                else:
                    print(std_err_msg % res)
                return (
                    False,
                    f"Trade fetch failed. {_check_permission_error_msg(res.get('msg', ''))}",
                )
            else:
                if res.get("data"):
                    if is_debug:
                        print(
                            f"asset_type: {asset_type}; exchange: {exchange} - extra info:{kwargs} - trades length: {len(res['data'])}"
                        )
                    return True, res.get("data", [])
                else:
                    return True, []

    except Exception as e:
        excp_msg = f"{str(e)} ++ fetch_trades - {asset_type} - {exchange} - {key} -  extra info:{kwargs}"
        if logger:
            logger.exception(excp_msg)
        else:
            print(excp_msg)
            print_exc()
        return False, "Trade fetch failed. Something went wrong."


def fetch_open_orders(
    exchange: str,
    asset_type: str,
    key: str | None = None,
    secret: str | None = None,
    user_id: str | None = None,
    passphrase: str | None = None,
    connection=None,
    is_testnet: bool = False,
    is_demo: bool = False,
    symbols: list = [],
    additional_symbols: list = [],
    logger=None,
    **kwargs,
) -> tuple[bool, str | dict | list]:
    """
    by symbol
        - bybit future
        - bybit inverse future
    """
    assert asset_type in ["spot", "future", "inverse_future", "option"]
    # assert exchange in ["eqonex", "binance", "bybit", "okx", "bingx"]
    std_err_msg = f"Order fetch failed. === key: {key} -- exchange: {exchange} -- asset_type: {asset_type} -- symbols: {symbols} -- additional_symbols: {additional_symbols} -- extra info:{kwargs} "
    try:
        if not connection:
            connection = get_exchange_connection(
                exchange=exchange,
                asset_type=asset_type,
                is_testnet=is_testnet,
                is_demo=is_demo,
                key=key,
                secret=secret,
                user_id=user_id,
                passphrase=passphrase,
                return_connection_only=True,
            )

        if symbols:
            ret = []
            for symbol in symbols:
                # print(f'asset_type: {asset_type}; symbol: {symbol}; exchange: {exchange}')
                res = connection.query_open_orders(symbol=symbol.upper())
                sleep(0.2)
                if res.get("error", True):
                    if_per_err, msg = _check_permission_error_msg(
                        msg=res.get("msg", ""), exchange=exchange, ret_tf=True
                    )
                    if if_per_err:
                        return False, f"Order fetch failed. {msg}"
                    if logger:
                        if if_per_err:
                            logger.warning(f"[{symbol}] " + std_err_msg % res)
                        else:
                            logger.error(f"[{symbol}] " + std_err_msg % res)
                    else:
                        print(std_err_msg % res)
                else:
                    # print(res, '!!!!')
                    if res.get("data"):
                        ret += res["data"]
            # print(ret, '!!!!')
            return True, ret

        if exchange in "bingx" and asset_type in ["spot"]:
            ret = []
            symbols = DEFAULT_SYMBOL_MAP[exchange][asset_type]
            # if exchange == "bybit" and asset_type in ["future", "inverse_future"]:
            #     ret = []
            #     if not symbols:
            #         if asset_type == "future":
            #             symbols = BYBIT_DEFAULT_FUTURE_SYMBOLS
            #         else:
            #             symbols = BYBIT_DEFAULT_INVERSE_FUTURE_SYMBOLS
            if additional_symbols:
                symbols += additional_symbols
                symbols = list(set(symbols))
            for symbol in symbols:
                # print(f'asset_type: {asset_type}; symbol: {symbol}; exchange: {exchange}')
                res = connection.query_open_orders(symbol=symbol.upper())
                sleep(0.2)
                if res.get("error", True):
                    if_per_err, msg = _check_permission_error_msg(
                        msg=res.get("msg", ""), exchange=exchange, ret_tf=True
                    )
                    if if_per_err:
                        return False, f"Order fetch failed. {msg}"
                    if logger:
                        if if_per_err:
                            logger.warning(f"[{symbol}] " + std_err_msg % res)
                        else:
                            logger.error(f"[{symbol}] " + std_err_msg % res)
                    else:
                        print(std_err_msg % res)
                else:
                    # print(res, '!!!!')
                    if res.get("data"):
                        ret += res["data"]
            # print(ret, '!!!!')
            return True, ret

        res = connection.query_open_orders()

        connection.close()

        if res.get("error", True):
            if_per_err, msg = _check_permission_error_msg(msg=res.get("msg", ""), exchange=exchange, ret_tf=True)
            if logger:
                if if_per_err:
                    logger.warning(std_err_msg % res)
                else:
                    logger.error(std_err_msg % res)
            else:
                print(std_err_msg % res)
            return False, f"Order fetch failed. {msg}"
        else:
            # print(res,'!!!!')
            if res.get("data"):
                return True, res.get("data", [])
            else:
                return True, []
    except Exception as e:
        excp_msg = f"{str(e)} ++ fetch_open_orders - {asset_type} - {exchange} - {key} -  extra info:{kwargs}"
        if logger:
            logger.exception(excp_msg)
        else:
            print(excp_msg)
            print_exc()
        return False, "Order fetch failed. Something went wrong."


def fetch_account(
    exchange: str,
    asset_type: str,
    key: str | None = None,
    secret: str | None = None,
    user_id: str | None = None,
    passphrase: str | None = None,
    connection=None,
    is_testnet: bool = False,
    is_demo: bool = False,
    logger=None,
    **kwargs,
) -> tuple[bool, str | list]:
    """
    eqonex account - skip future
    """
    assert asset_type in ["spot", "future", "inverse_future", "option", "saving"]
    # assert exchange in ["eqonex", "binance", "bybit", "okx", "bingx"]
    std_err_msg = f"Account fetch failed. === key: {key} -- exchange: {exchange} -- asset_type: {asset_type} -- extra info:{kwargs}"
    try:
        if exchange == "eqonex" and asset_type in ["future", "inverse_future"]:
            return True, []
        if not connection:
            connection = get_exchange_connection(
                exchange=exchange,
                asset_type=asset_type,
                is_testnet=is_testnet,
                is_demo=is_demo,
                key=key,
                secret=secret,
                user_id=user_id,
                passphrase=passphrase,
                return_connection_only=True,
            )
        if asset_type == "saving":
            res = connection.query_saving_account()
        else:
            res = connection.query_account()

        connection.close()
        if res.get("error", True):
            if_per_err, msg = _check_permission_error_msg(msg=res.get("msg", ""), exchange=exchange, ret_tf=True)

            if logger:
                if if_per_err:
                    logger.warning(std_err_msg % res)
                else:
                    logger.error(std_err_msg % res)
            else:
                print(std_err_msg % res)
            return False, f"Account fetch failed. {msg}"
        else:
            return True, list(res.get("data", {}).values())
    except Exception as e:
        excp_msg = f"{str(e)} ++ fetch_account - {asset_type} - {exchange} - {key} -  extra info:{kwargs}"
        if logger:
            logger.exception(excp_msg)
        else:
            print(excp_msg)
            print_exc()
        return False, "Account fetch failed. Something went wrong."


def fetch_position(
    exchange: str,
    asset_type: str,
    symbol: str | None = None,
    key: str | None = None,
    secret: str | None = None,
    user_id: str | None = None,
    passphrase: str | None = None,
    connection=None,
    is_testnet: bool = False,
    is_demo: bool = False,
    logger=None,
    force_return: bool = False,
    **kwargs,
) -> tuple[bool, str | dict]:
    """
    eqonex account - skip future
    """
    assert asset_type in ["future", "inverse_future", "option"]
    # assert exchange in ["eqonex", "binance", "bybit", "okx", "bingx"]
    std_err_msg = f"Position fetch failed. === key: {key} -- exchange: {exchange} -- asset_type: {asset_type} -- extra info:{kwargs}"

    try:
        if not connection:
            connection = get_exchange_connection(
                exchange=exchange,
                asset_type=asset_type,
                is_testnet=is_testnet,
                is_demo=is_demo,
                key=key,
                secret=secret,
                user_id=user_id,
                passphrase=passphrase,
                return_connection_only=True,
            )
        if symbol:
            res = connection.query_position(symbol=symbol, force_return=force_return)
        else:
            res = connection.query_position(force_return=force_return)

        connection.close()

        if res.get("error", True):
            if_per_err, msg = _check_permission_error_msg(msg=res.get("msg", ""), exchange=exchange, ret_tf=True)
            if logger:
                if if_per_err:
                    logger.warning(std_err_msg % res)
                else:
                    logger.error(std_err_msg % res)
            else:
                print(std_err_msg % res)
            return False, f"Position fetch failed. {msg}"
        else:
            return True, res.get("data", [])
    except Exception as e:
        excp_msg = f"{str(e)} ++ fetch_position - {asset_type} - {exchange} - {key} -  extra info:{kwargs}"
        if logger:
            logger.exception(excp_msg)
        else:
            print(excp_msg)
            print_exc()
        return False, "Position fetch failed. Something went wrong."


def fetch_permission(
    exchange: str,
    key: str | None = None,
    secret: str | None = None,
    user_id: str | None = None,
    passphrase: str | None = None,
    connection=None,
    is_testnet: bool = False,
    is_demo: bool = False,
    logger=None,
    **kwargs,
) -> tuple[bool, str | dict]:
    """
    {'created_at': datetime.datetime(2021, 9, 1, 21, 33, 14, tzinfo=<DstTzInfo 'Asia/Hong_Kong' HKT+8:00:00 STD>),
    'expired_at': datetime.datetime(2022, 7, 11, 8, 0, tzinfo=<DstTzInfo 'Asia/Hong_Kong' HKT+8:00:00 STD>),
    'permissions':
        {'enable_withdrawals': False, 'enable_internaltransfer': False, 'enable_universaltransfer': True, 'enable_options': True, 'enable_reading': True, 'enable_futures': True, 'enable_margin': False, 'enable_spot_and_margintrading': True}}

    {'created_at': datetime.datetime(2022, 4, 27, 3, 45, 27, tzinfo=<DstTzInfo 'Asia/Hong_Kong' HKT+8:00:00 STD>),
    'expired_at': datetime.datetime(2022, 7, 27, 3, 45, 27, tzinfo=<DstTzInfo 'Asia/Hong_Kong' HKT+8:00:00 STD>),
    'permissions':
        {'enable_withdrawals': None, 'enable_internaltransfer': False, 'enable_universaltransfer': False, 'enable_options': True, 'enable_reading': True, 'enable_futures': True, 'enable_margin': None, 'enable_spot_and_margintrading': True}}
    """
    # assert asset_type in ["spot"]
    # assert exchange in ["eqonex", "binance", "bybit", "okx", "bingx"]
    std_err_msg = f"Permission fetch failed. === key: {key} -- exchange: {exchange} -- extra info:{kwargs}"

    try:
        ret = {}
        if exchange == "eqonex":
            ret["expired_at"] = tz_manager.tz.localize(datetime(2100, 1, 1, 0, 0, 0))
            ret["enable_view"] = True
            ret["enable_spot_trade"] = True
            ret["enable_future_trade"] = True
            ret["enable_option_trade"] = True
            ret["enable_withdrawls"] = True
            ret["enable_transfer"] = True
        else:
            # data = connection.apikey_info()['data']
            # expired_at = data.get('expired_at')
            # enable_view = data.get('enable_view')
            # enable_spot_trade = data.get('enable_spot_trade')
            # enable_future_trade = data.get('enable_future_trade')
            # enable_option_trade = data.get('enable_option_trade')
            # enable_withdrawls = data.get('enable_withdrawls')
            # enable_transfer = data.get('enable_transfer')
            if not connection:
                connection = get_exchange_connection(
                    exchange=exchange,
                    asset_type="spot",
                    is_testnet=is_testnet,
                    is_demo=is_demo,
                    key=key,
                    secret=secret,
                    user_id=user_id,
                    passphrase=passphrase,
                    return_connection_only=True,
                )

            res = connection.query_permission()

            if res.get("error", True):
                if_per_err, msg = _check_permission_error_msg(msg=res.get("msg", ""), exchange=exchange, ret_tf=True)
                if logger:
                    if if_per_err:
                        logger.warning(std_err_msg % res)
                    else:
                        logger.error(std_err_msg % res)
                else:
                    print(std_err_msg % res)
                return False, f"Permission fetch failed. {msg}"

            data = connection.query_permission()["data"]
            ret["expired_at"] = data.get("expired_at")
            ret["enable_view"] = data.get("permissions", {}).get("enable_reading")
            ret["enable_spot_trade"] = data.get("permissions", {}).get("enable_spot_and_margintrading")
            ret["enable_future_trade"] = data.get("permissions", {}).get("enable_futures")
            ret["enable_option_trade"] = data.get("permissions", {}).get("enable_options")
            ret["enable_withdrawls"] = data.get("permissions", {}).get("enable_withdrawals")
            ret["enable_transfer"] = data.get("permissions", {}).get("enable_internaltransfer")

            connection.close()

        return True, ret
    except Exception as e:
        excp_msg = f"{str(e)} ++ fetch_permission - {exchange} - {key} -  extra info:{kwargs}"
        if logger:
            logger.exception(excp_msg)
        else:
            print(excp_msg)
            print_exc()
        return False, "Permission fetch failed. Something went wrong."


def fetch_symbols(
    exchange: str,
    asset_type: str,
    connection=None,
    is_testnet: bool = False,
    is_demo: bool = False,
    logger=None,
    **kwargs,
) -> tuple[bool, list | str]:
    """
    Fetch available trading symbols from exchange.

    Args:
        exchange: Exchange name
        asset_type: Asset type (spot, future, inverse_future, option)
        connection: Existing exchange connection (optional)
        is_testnet: Use testnet environment
        is_demo: Use demo mode
        logger: Logger instance for error reporting
        **kwargs: Additional parameters

    Returns:
        Tuple of (success_flag, symbols_list_or_error_message)
    """
    assert asset_type in ["spot", "future", "inverse_future", "option"], f"Invalid asset_type: {asset_type}"

    try:
        if not connection:
            connection = get_exchange_connection(
                exchange=exchange,
                asset_type=asset_type,
                is_testnet=is_testnet,
                is_demo=is_demo,
                return_connection_only=True,
            )

        res = connection.query_symbols()
        connection.close()

        if res.get("error", True):
            error_msg = res.get("msg", "Unknown error")
            if logger:
                logger.error(f"Symbol fetch failed for {exchange}:{asset_type} - {error_msg}")
            else:
                print(f"Symbol fetch failed for {exchange}:{asset_type} - {error_msg}")
            return False, f"Symbol fetch failed. {error_msg}"

        return True, res.get("data", [])

    except Exception as e:
        context_info = f"exchange={exchange}, asset_type={asset_type}"
        exc_msg = f"{str(e)} ++ fetch_symbols - {context_info}"

        if logger:
            logger.exception(exc_msg)
        else:
            print_exc()
            print(exc_msg)

        return False, "Symbol fetch failed. Something went wrong."


def fetch_history(
    exchange: str,
    asset_type: str,
    symbol: str,
    interval: str = "1d",
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    connection=None,
    logger=None,
    **kwargs,
) -> tuple[bool, list | str]:
    """
    Fetch historical OHLC data for a trading symbol.

    Args:
        exchange: Exchange name
        asset_type: Asset type (spot, future, inverse_future, option)
        symbol: Trading symbol (e.g., 'BTCUSDT')
        interval: Time interval (default: '1d')
        start: Start time for historical data
        end: End time for historical data
        connection: Existing exchange connection (optional)
        logger: Logger instance for error reporting
        **kwargs: Additional parameters

    Returns:
        Tuple of (success_flag, ohlc_data_or_error_message)
    """
    assert asset_type in ["spot", "future", "inverse_future", "option"], f"Invalid asset_type: {asset_type}"

    # Parse datetime parameters
    if start and isinstance(start, str):
        start = parse(start)
    if end and isinstance(end, str):
        end = parse(end)

    symbol = symbol.upper()

    try:
        connection = get_exchange_connection(exchange=exchange, asset_type=asset_type, return_connection_only=True)

        res = connection.query_history(symbol=symbol, interval=interval, start=start, end=end)
        connection.close()

        if res.get("error", True):
            error_msg = res.get("msg", "Unknown error")
            if logger:
                logger.error(f"OHLC fetch failed for {symbol}@{exchange}: {error_msg}")
            else:
                print(f"OHLC fetch failed for {symbol}@{exchange}: {error_msg}")
            return False, f"OHLC fetch failed. {error_msg}"

        return True, res.get("data", [])

    except Exception as e:
        context_info = f"symbol={symbol}, exchange={exchange}, asset_type={asset_type}, interval={interval}"
        exc_msg = f"{str(e)} ++ fetch_history - {context_info}"

        if logger:
            logger.exception(exc_msg)
        else:
            print(exc_msg)
            print_exc()

        return False, "OHLC fetch failed. Something went wrong."


def fetch_price(
    symbol: str,
    exchange: str | None = None,
    asset_type: str | None = None,
    connection=None,
    logger=None,
    **kwargs,
) -> tuple[bool, list | str]:
    """
    Fetch current price for a trading symbol from exchange.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        exchange: Exchange name (required if no connection provided)
        asset_type: Asset type (required if no connection provided)
        connection: Existing exchange connection (optional)
        logger: Logger instance for error reporting
        **kwargs: Additional parameters

    Returns:
        Tuple of (success_flag, price_data_or_error_message)

    Example:
        >>> success, data = fetch_price("BTCUSDT", "binance", "spot")
        >>> if success:
        ...     print(f"Current price: {data}")
    """
    symbol = symbol.upper()

    try:
        if not connection:
            assert asset_type in ["spot", "future", "inverse_future", "option"], (
                f"Invalid asset_type: {asset_type}. Must be one of: spot, future, inverse_future, option"
            )
            assert exchange, "exchange parameter is required when connection is not provided"

            connection = get_exchange_connection(exchange=exchange, asset_type=asset_type, return_connection_only=True)

        res = connection.query_prices(symbol=symbol)
        connection.close()

        if res.get("error", True):
            error_msg = res.get("msg", "Unknown error")
            return False, f"Price fetch failed. {error_msg}"

        return True, res.get("data", [])

    except Exception as e:
        context_info = f"symbol={symbol}, exchange={exchange}, asset_type={asset_type}"
        exc_msg = f"{str(e)} ++ fetch_price - {context_info}"

        if logger:
            logger.exception(exc_msg)
        else:
            print(exc_msg)
            print_exc()

        return False, "Price fetch failed. Something went wrong."
