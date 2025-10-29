import base64
import hashlib
import hmac
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Optional
from urllib.parse import urlencode

from ..utils import models
from ..utils.client_rest import Response, RestClient
from ..utils.client_ws import WebsocketClient
from ..utils.logger import get_logger
from ..utils.models import UTC_TZ, okx_asset_type

logger = get_logger(default_level="warning")


EXCHANGE = "OKX"

TIMEDELTA_MAP: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}

"""
The Production Trading URL:
    REST: https://www.okx.com
    Public WebSocket: wss://ws.okx.com:8443/ws/v5/public
    Private WebSocket: wss://ws.okx.com:8443/ws/v5/private
    Business WebSocket: wss://ws.okx.com:8443/ws/v5/business

AWS URL:
    REST: https://aws.okx.com
    Public WebSocket: wss://wsaws.okx.com:8443/ws/v5/public
    Private WebSocket: wss://wsaws.okx.com:8443/ws/v5/private
    Business WebSocket: wss://wsaws.okx.com:8443/ws/v5/business
"""
OKX_API_HOST = "https://www.okx.com/api/v5"
OKX_PUBLIC_WEBSOCKET_HOST = "wss://ws.okx.com:8443/ws/v5/public"
OKX_PRIVATE_WEBSOCKET_HOST = "wss://ws.okx.com:8443/ws/v5/private"

# NOTE: `x-simulated-trading: 1` needs to be added to the header of the Demo Trading request.
OKX_TESTNET_REST_HOST = "https://www.okx.com/api/v5"
OKX_TESTNET_PUBLIC_WEBSOCKET_HOST = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
OKX_TESTNET_PRIVATE_WEBSOCKET_HOST = "wss://wspap.okx.com:8443/ws/v5/private?brokerId=9999"

# e.g. [1m/3m/5m/15m/30m/1H/2H/4H/6H/12H/1D/1W/1M/3M/6M/1Y]
INTERVAL_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "12h": "12H",
    "1d": "1D",
    "1w": "1W",
    "1M": "1M",
    "3M": "3M",
    "6M": "6M",
    "1y": "1Y",
}

TIMEDELTA_MAPPING_SEC: dict[str, int] = {
    "1m": 60,
    "3m": 60 * 3,
    "5m": 60 * 5,
    "15m": 60 * 15,
    "30m": 60 * 15,
    "1h": 60 * 60 * 1,
    "2h": 60 * 60 * 2,
    "4h": 60 * 60 * 4,
    "6h": 60 * 60 * 6,
    "8h": 60 * 60 * 8,
    "12h": 60 * 60 * 12,
    "1d": 60 * 60 * 24 * 1,
    "3d": 60 * 60 * 24 * 3,
    "1w": 60 * 60 * 24 * 7,
    "1M": 60 * 60 * 24 * 30,
}

ASSET_TYPE_MAP = {
    "spot": "SPOT",
    "margin": "MARGIN",
    "future": ["SWAP", "FUTURES"],
    "inverse_future": ["SWAP", "FUTURES"],
    "option": "OPTION",
}

MAPPING_CHANNEL: dict[str, str] = {
    ### public
    "depth": "books5",
    "kline": "candle",
    "ticker": "tickers",
    ### private
    "orders": "orders",
    "account": "account",
    "position": "positions",
    "greeks": "greeks",
}


def generate_signature(msg: str, secret_key: str) -> bytes:
    """"""
    return base64.b64encode(hmac.new(secret_key.encode(), msg.encode(), hashlib.sha256).digest())


def generate_timestamp() -> str:
    """"""
    now: datetime = datetime.utcnow()
    # now: datetime = datetime.now(UTC_TZ).replace(tzinfo=None)
    timestamp: str = now.isoformat("T", "milliseconds")
    # print(timestamp,'???????')
    return timestamp + "Z"


class OKXClient(RestClient):
    """
    api docs: https://developer.okx.com/#introduction
    demo trading: https://www.okx.com/demo-trading-explorer/v5/en

    Instrument type
    SPOT
    MARGIN
    SWAP
    FUTURES
    OPTION

    in this class:
        spot
        future           (SWAP + FUTURES)
        inverse-future   (SWAP + FUTURES)
        option

    """

    DATA_TYPE = "spot"  # spot, future, inverse_future, option # only applied to symbols, orders, and trades
    FD_BROKER_ID = "6cb9d8a417b4BCDE"  # rest api / oauth
    ND_BROKER_ID = "6c11bf7603f6SCDE"  # subaccount

    def __init__(
        self,
        proxy_host: str = "",
        proxy_port: int = 0,
        datatype: str = "spot",
        is_testnet: bool = False,
        is_live: bool = False,
        logger=None,
        session: bool = False,
        **kwargs,
    ) -> None:
        # url_base: str = OKX_API_HOST,
        super().__init__(
            OKX_TESTNET_REST_HOST if is_testnet else OKX_API_HOST,
            proxy_host,
            proxy_port,
            logger=logger,
        )
        if datatype not in ["spot", "linear", "inverse", "option"]:
            raise ValueError("Invalid data type.")

        self.DATA_TYPE = datatype  # for symbol, orders, trades, account

        self.order_count: int = 0
        self.order_count_lock: Lock = Lock()
        self.connect_time: int = 0
        self.key: str = ""
        self.secret: str = ""
        self.passphrase: str = ""
        self.is_testnet = is_testnet
        self.is_live = is_live
        self.session = session

    def info(self):
        return "OKX REST API start"

    def new_order_id(self) -> str:
        """new_order_id"""
        # prefix: str = datetime.now().strftime("%Y%m%d-%H%M%S-")
        prefix: str = models.generate_datetime().strftime("%y%m%d-%H%M%S%f")[:-3] + "-"  #  # datetime.now()
        with self.order_count_lock:
            self.order_count += 1
            suffix: str = str(self.order_count).rjust(5, "0")

            order_id: str = f"{self.FD_BROKER_ID}" + prefix + suffix
            return order_id

    def _remove_none(self, payload):
        return {k: v for k, v in payload.items() if v is not None}

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if self.session:
            response = self.add_request(method="GET", path=path, params=params)
        else:
            response = self.request(method="GET", path=path, params=params)
        return self._process_response(response)

    def _post(
        self,
        path: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        if self.session:
            response = self.add_request(method="POST", path=path, data=data, params=params)
        else:
            response = self.request(method="POST", path=path, data=data, params=params)
        return self._process_response(response, is_post=True)

    def connect(self, key: str, secret: str, passphrase: str, **kwargs) -> None:
        """connect exchange server"""
        self.key = key
        self.secret = secret
        self.passphrase = passphrase

        # print(self.key, secret, passphrase,'.....')
        if self.is_live:
            self.start()
        logger.debug(self.info())
        return True

    def sign(self, request):
        """
        https://github.com/vn-crypto/vnpy_okex/blob/main/vnpy_okex/okex_gateway.py
        """
        if "public" in request.path or "market" in request.path:
            return request

        timestamp: str = generate_timestamp()
        if request.data:
            request.data = json.dumps(request.data)

        if request.params:
            # path: str = self.url_base + request.path + "?" + urlencode(request.params)
            path: str = "/api/v5" + request.path + "?" + urlencode(request.params)
        else:
            # path: str = self.url_base + request.path
            path: str = "/api/v5" + request.path

        # print(f"path:{path};   method:{request.method};")
        if request.data:
            msg: str = timestamp + request.method + path + request.data
        else:
            msg: str = timestamp + request.method + path
        # print(
        #     msg,
        #     "<====msg",
        #     type(msg),
        #     type(request.data),
        # )
        signature: bytes = generate_signature(msg, self.secret)

        # add header
        request.headers = {
            "OK-ACCESS-KEY": self.key,
            "OK-ACCESS-SIGN": signature.decode(),
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }
        # print(request.headers, "<=====header")
        if self.is_testnet:
            request.headers["x-simulated-trading"] = "1"

        # print(request,'request!')
        return request

    def _process_response(self, response: Response, is_post: bool = False) -> dict:
        try:
            data = response.json()
            # print(data,'!!!!', response.ok)
        except ValueError:
            print("!!okx - _process_response !!", response.text)
            logger.debug(response.text)
            raise
        else:
            # logger.debug(data)

            if not response.ok or (response.ok and data.get("code", "") != "0"):
                payload_data = models.CommonDataSchema(status_code=response.status_code, msg=dict(data))
                if not data.get("msg"):
                    print("!!okx - return raw data !!", response.text)
                    ### TODO post, use sMsg
                    # {
                    # "code": "0",
                    # "msg": "",
                    # "data": [
                    #     {
                    #     "clOrdId": "oktswap6",
                    #     "ordId": "312269865356374016",
                    #     "tag": "",
                    #     "sCode": "0",
                    #     "sMsg": ""
                    #     }
                    # ]
                    # }

                # print('??????', payload_data.dict(), '|||',data,'????', response, is_post)
                msg = data.get("msg", "Something went wrong when fetching data from OKX server.")
                if is_post:
                    d = data.get("data", [])
                    # print(d,'.......')
                    if d and len(d):
                        msg = d[0].get(
                            "sMsg",
                            "Something went wrong when fetching data from OKX server.",
                        )
                # print(f"msg:{msg}")
                return models.CommonResponseSchema(success=False, error=True, data=payload_data.dict(), msg=msg)
            elif response.ok and not data:
                # print('data',data, '?????')
                payload_data = models.CommonDataSchema(status_code=response.status_code, msg=dict(data))
                return models.CommonResponseSchema(
                    success=True,
                    error=False,
                    data=payload_data.dict(),
                    msg="query ok, but result is None",
                )
            else:
                # print('data',data,'!!!!!')
                return models.CommonResponseSchema(success=True, error=False, data=data, msg="query ok")

    def _regular_account_payload(self, common_response: models.CommonResponseSchema, account_type: str) -> dict:
        if isinstance(common_response.data, list):
            # payload = list()
            payload = dict()
            for msg in common_response.data:
                data = models.AccountSchema()
                result = data.from_okx_to_form(msg, datatype=self.DATA_TYPE, account_type=account_type)
                if result:
                    # payload.append(result)
                    payload[result["symbol"]] = result
        elif isinstance(common_response.data, dict):
            payload = models.AccountSchema()
            payload = payload.from_okx_to_form(common_response.data, datatype=self.DATA_TYPE, account_type=account_type)
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _regular_order_payload(self, common_response: models.CommonResponseSchema) -> dict:
        if isinstance(common_response.data, list):
            payload = list()
            for msg in common_response.data:
                data = models.OrderSchema()
                result = data.from_okx_to_form(msg)
                if result:
                    payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.OrderSchema()
            payload = payload.from_okx_to_form(common_response.data)
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _regular_trades_payload(self, common_response: models.CommonResponseSchema) -> dict:
        if isinstance(common_response.data, list):
            payload = list()
            for msg in common_response.data:
                data = models.TradeSchema()
                result = data.from_okx_to_form(msg, datatype=self.DATA_TYPE)
                if result:
                    payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.TradeSchema()
            payload = payload.from_okx_to_form(common_response.data, datatype=self.DATA_TYPE)
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _regular_symbols_payload(self, common_response: models.CommonResponseSchema) -> dict:
        if isinstance(common_response.data, list):
            # payload = list()
            payload = dict()
            for msg in common_response.data:
                data = models.SymbolSchema()
                result = data.from_okx_to_form(msg, self.DATA_TYPE)
                if result:
                    # payload.append(result)
                    if len(common_response.data) == 1:
                        payload = result
                    else:
                        payload[result["symbol"]] = result
        elif isinstance(common_response.data, dict):
            payload = models.SymbolSchema()
            payload = payload.from_okx_to_form(common_response.data, self.DATA_TYPE)
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _regular_historical_prices_payload(
        self, common_response: models.CommonResponseSchema, extra: dict = None
    ) -> dict:
        def key_data(data):
            key = ["starttime", "open", "high", "low", "close", "volume", "turnover"]
            return dict(zip(key, data))

        if isinstance(common_response.data, list):
            payload = list()
            for msg in common_response.data:
                msg = key_data(msg)
                data = models.HistoryOHLCSchema()
                result = data.from_okx_to_form(msg, extra, datatype=self.DATA_TYPE)
                payload.append(result)
        elif isinstance(common_response.data, dict):
            common_response.data = key_data(common_response.data)
            payload = (
                models.HistoryOHLCSchema()
            )  # .model_construct(common_response.data, extra, datatype=self.DATA_TYPE)
            payload = payload.from_okx_to_form(common_response.data, extra)
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _regular_symbol_payload(self, common_response: models.CommonResponseSchema) -> dict:
        if isinstance(common_response.data, list):
            payload = list()
            for msg in common_response.data:
                data = models.TickerSchema()
                result = data.from_okx_to_form(msg, datatype=self.DATA_TYPE)
                payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.TickerSchema()
            payload = payload.from_okx_to_form(common_response.data, datatype=self.DATA_TYPE)
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _regular_position_payload(self, common_response: models.CommonResponseSchema) -> dict:
        if isinstance(common_response.data, list):
            payload = list()
            for msg in common_response.data:
                data = models.PositionSchema()
                result = data.from_okx_to_form(msg, datatype=self.DATA_TYPE)
                if result:
                    payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.PositionSchema()
            payload = payload.from_okx_to_form(common_response.data, datatype=self.DATA_TYPE)
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _regular_transfer_payload(
        self,
        common_response: models.CommonResponseSchema,
        transfer_type: str,
        inout: str,
    ) -> dict:
        if isinstance(common_response.data, list):
            payload = list()
            for msg in common_response.data:
                data = models.TransferSchema()
                result = data.from_okx_to_form(msg, transfer_type=transfer_type, inout=inout)
                payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.TransferSchema()
            payload = payload.from_okx_to_form(common_response.data, transfer_type=transfer_type, inout=inout)
        else:
            common_response.msg = "payload error in fetch transfer data"
            Exception("payload error in fetch transfer data")

        common_response.data = payload
        return common_response

    def query_permission(self) -> dict:
        """
        requirement 9 - api key info(permission & expiry date)[restful]
        FTX don't provide this api
        """
        api_info = models.PermissionSchema().dict()
        # api_info['expired_at'] = datetime(2100,1,1)
        api_info["permissions"] = {
            "enable_withdrawals": False,
            "enable_internaltransfer": True,
            "enable_universaltransfer": True,
            "enable_options": True,
            "enable_reading": True,
            "enable_futures": True,
            "enable_margin": True,
            "enable_spot_and_margintrading": True,
        }
        return models.CommonResponseSchema(
            success=True,
            error=False,
            data=api_info,
            msg="OKX don't suport this feature",
        )

    def query_account(self) -> dict:
        """query_account
        {
            "code": "0",
            "data": [
                {
                    "adjEq": "10679688.0460531643092577",
                    "details": [
                        {
                            "availBal": "",
                            "availEq": "9930359.9998",
                            "cashBal": "9930359.9998",
                            "ccy": "USDT",
                            "crossLiab": "0",
                            "disEq": "9439737.0772999514",
                            "eq": "9930359.9998",
                            "eqUsd": "9933041.196999946",
                            "frozenBal": "0",
                            "interest": "0",
                            "isoEq": "0",
                            "isoLiab": "0",
                            "isoUpl":"0",
                            "liab": "0",
                            "maxLoan": "10000",
                            "mgnRatio": "",
                            "notionalLever": "",
                            "ordFrozen": "0",
                            "twap": "0",
                            "uTime": "1620722938250",
                            "upl": "0",
                            "uplLiab": "0",
                            "stgyEq":"0",
                            "spotInUseAmt":""
                        },
                        {
                            "availBal": "",
                            "availEq": "33.6799714158199414",
                            "cashBal": "33.2009985",
                            "ccy": "BTC",
                            "crossLiab": "0",
                            "disEq": "1239950.9687532129092577",
                            "eq": "33.771820625136023",
                            "eqUsd": "1239950.9687532129092577",
                            "frozenBal": "0.0918492093160816",
                            "interest": "0",
                            "isoEq": "0",
                            "isoLiab": "0",
                            "isoUpl":"0",
                            "liab": "0",
                            "maxLoan": "1453.92289531493594",
                            "mgnRatio": "",
                            "notionalLever": "",
                            "ordFrozen": "0",
                            "twap": "0",
                            "uTime": "1620722938250",
                            "upl": "0.570822125136023",
                            "uplLiab": "0",
                            "stgyEq":"0",
                            "spotInUseAmt":""
                        }
                    ],
                    "imr": "3372.2942371050594217",
                    "isoEq": "0",
                    "mgnRatio": "70375.35408747017",
                    "mmr": "134.8917694842024",
                    "notionalUsd": "33722.9423710505978888",
                    "ordFroz": "0",
                    "totalEq": "11172992.1657531589092577",
                    "uTime": "1623392334718"
                }
            ],
            "msg": ""
        }

        """

        if self.DATA_TYPE == "spot":
            account_type = "SPOT"
        else:
            account_type = "CONTRACT"

        balance = self._get("/account/balance")
        account = defaultdict(dict)
        # logger.debug(balance)
        # print(f">>> balance:{balance}")
        if balance.success and balance.data:
            if not balance.data["data"]:
                balance.msg = "account wallet is empty"
                return balance
            # datatype = self.DATA_TYPE.upper()
            # for account_data_out in balance.data.get("data", []):
            #     for account_data in account_data_out.get("details", []):
            #         key = account_data["ccy"]
            #         free = float(account_data["availBal"]) if account_data["availBal"] else 0.0
            #         frozen = float(account_data["frozenBal"])
            #         total = float(account_data["cashBal"])
            #         if total:
            #             account[key]["symbol"] = key
            #             account[key]["available"] = free
            #             account[key]["frozen"] = frozen
            #             account[key]["balance"] = total
            #             account[key]["market_value"] = float(account_data["eqUsd"])
            #             account[key]["exchange"] = EXCHANGE
            #             account[key]["asset_type"] = datatype
            # balance.data = account
            # return balance
            coin_list = []
            for account_data_out in balance.data.get("data", []):
                coin_list += account_data_out.get("details", [])
            balance.data = coin_list

            return self._regular_account_payload(balance, account_type)
        else:
            return balance

    def send_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type=None,
        price=None,
        stop_price=None,
        reduce_only: bool = False,
        **kwargs,
    ) -> dict:
        """
        place order for SPOT
        POST /api/v5/trade/order
        body
        {
            "instId":"BTC-USDT",
            "tdMode":"cash",
            "clOrdId":"b15",
            "side":"buy",
            "ordType":"limit",
            "px":"2.15",
            "sz":"2"
        }
        response:
            {
            "code": "0",
            "msg": "",
            "data": [
                {
                "clOrdId": "oktswap6",
                "ordId": "312269865356374016",
                "tag": "",
                "sCode": "0",
                "sMsg": ""
                }
            ]
            }



        tgtCcy	String	No	Quantity type
            base_ccy: Base currency ,quote_ccy: Quote currency
            Only applicable to SPOT Market Orders
            Default is quote_ccy for buy, base_ccy for sell
        """
        symbol: str = symbol.upper()
        side: str = side.lower()  # Order side, buy sell
        if self.DATA_TYPE == "spot":
            mode = "cash"
        else:
            mode = "isolated"
            # mode = "cross"

        order_id: str = self.new_order_id()  # generate_new_order_id
        # print(len(order_id), 'order_id len')
        if not order_type:
            """
            Order type
                market: Market order
                limit: Limit order
                post_only: Post-only order
                fok: Fill-or-kill order
                ioc: Immediate-or-cancel order
                optimal_limit_ioc: Market order with immediate-or-cancel order (applicable only to Futures and Perpetual swap).
            """
            if not price:
                order_type = "market"
                if stop_price:
                    # TODO: use place algo order api
                    order_type = "stop"
            else:
                order_type = "limit"
                if stop_price:
                    order_type = "stop_limit"

        payload: dict = {
            "instId": symbol,
            "tdMode": mode,
            "side": side,
            "ordType": order_type,
            "sz": str(quantity),
            "clOrdId": order_id,
            "tag": self.FD_BROKER_ID,
            # "posSide":"long"
        }
        if order_type == "limit":
            payload["px"] = str(price)
        # if order_type in [3, 4]:
        #     payload["stopPx"] = stop_price

        if reduce_only:
            payload["reduceOnly"] = reduce_only

        asset_type = okx_asset_type(symbol)
        if "px" not in payload and asset_type == "SPOT":
            """
            tgtCcy
                base_ccy: Base currency ,quote_ccy: Quote currency
                    Only applicable to SPOT Market Orders
                    Default is quote_ccy for buy, base_ccy for sell
                Base currency, e.g. BTC in BTC-USDT
                Quote currency, e.g. USDT in BTC-USDT
                    Only applicable to SPOT/MARGIN
            """
            payload["tgtCcy"] = "base_ccy"

        # print(f"send order payload:{payload}")
        orders_response = self._post("/trade/order", payload)

        if orders_response.success:
            if orders_response.data:
                data = orders_response.data["data"]
                if data:
                    data = data[0]
                result = models.SendOrderResultSchema()  # .model_construct(data)
                orders_response.data = result.from_okx_to_form(data, symbol=symbol)
            else:
                orders_response.msg = "query sender order is empty"

        return orders_response

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """
        # order_id: refer to custom client order id
        order_id: refer to original order id
        POST /api/v5/trade/cancel-order
        body
        {
            "ordId":"2510789768709120",
            "instId":"BTC-USD-190927"
        }


        {
            "code":"0",
            "msg":"",
            "data":[
                {
                    "clOrdId":"oktswap6",
                    "ordId":"12345689",
                    "sCode":"0",
                    "sMsg":""
                }
            ]
        }

        """
        symbol = symbol.upper()
        payload: dict = {"instId": symbol, "ordId": order_id}

        return self._post("/trade/cancel-order", payload)

    def cancel_orders(self, symbol: str = None) -> dict:
        """ """
        return models.CommonResponseSchema(success=False, error=True, data={}, msg="OKX don't suport this feature")

    def query_order_status(self, order_id: str, symbol: str) -> dict:
        """
        {
            "code": "0",
            "msg": "",
            "data": [
                {
                "instType": "FUTURES",
                "instId": "BTC-USD-200329",
                "ccy": "",
                "ordId": "312269865356374016",
                "clOrdId": "b1",
                "tag": "",
                "px": "999",
                "sz": "3",
                "pnl": "5",
                "ordType": "limit",
                "side": "buy",
                "posSide": "long",
                "tdMode": "isolated",
                "accFillSz": "0",
                "fillPx": "0",
                "tradeId": "0",
                "fillSz": "0",
                "fillTime": "0",
                "state": "live",
                "avgPx": "0",
                "lever": "20",
                "tpTriggerPx": "",
                "tpTriggerPxType": "last",
                "tpOrdPx": "",
                "slTriggerPx": "",
                "slTriggerPxType": "last",
                "slOrdPx": "",
                "feeCcy": "",
                "fee": "",
                "rebateCcy": "",
                "rebate": "",
                "tgtCcy":"",
                "category": "",
                "uTime": "1597026383085",
                "cTime": "1597026383085"
                }
            ]
            }

        """
        payload = {"ordId": order_id, "instId": symbol}  #
        order_status = self._get("/trade/order", payload)
        # print(f"order_status: {order_status}")

        if order_status.success:
            if order_status.data:
                # return self._regular_order_payload(order_status)
                data = order_status.data["data"]
                if data:
                    order_status.data = data[0]
                    return self._regular_order_payload(order_status)
            else:
                order_status.msg = "query order is empty"

        return order_status

    def query_open_orders(self, symbol: str = None) -> dict:
        """query open order
        {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "accFillSz": "0",
                    "avgPx": "",
                    "cTime": "1618235248028",
                    "category": "normal",
                    "ccy": "",
                    "clOrdId": "",
                    "fee": "0",
                    "feeCcy": "BTC",
                    "fillPx": "",
                    "fillSz": "0",
                    "fillTime": "",
                    "instId": "BTC-USDT",
                    "instType": "SPOT",
                    "lever": "5.6",
                    "ordId": "301835739059335168",
                    "ordType": "limit",
                    "pnl": "0",
                    "posSide": "net",
                    "px": "59200",
                    "rebate": "0",
                    "rebateCcy": "USDT",
                    "side": "buy",
                    "slOrdPx": "",
                    "slTriggerPx": "",
                    "slTriggerPxType": "last",
                    "state": "live",
                    "sz": "1",
                    "tag": "",
                    "tgtCcy": "",
                    "tdMode": "cross",
                    "source":"",
                    "tpOrdPx": "",
                    "tpTriggerPx": "",
                    "tpTriggerPxType": "last",
                    "tradeId": "",
                    "uTime": "1618235248028"
                }
            ]
        }


        """

        if symbol:
            payload = {"instId": symbol}
        else:
            payload = None

        order_status = self._get("/trade/orders-pending", payload)
        if order_status.success:
            if order_status.data:
                order_status.data = order_status.data.get("data", [])
                return self._regular_order_payload(order_status)
            else:
                order_status.msg = "query order is empty"

        return order_status

    def query_all_orders(self) -> dict:
        """
        TODO: low priority to implement
        """
        all_order = self._post("/getUserHistory?type=order&format=json", {"account": self.user_id})
        if all_order.success:
            if all_order.data:
                all_order.data = all_order.data.get("orderHistory", [])
                return self._regular_order_payload(all_order)
            else:
                all_order.msg = "query order is empty"

        return all_order

    def query_symbols(self, symbol: str = None, base_coin="BTC") -> dict:
        """query_symbols
        https://www.okx.com/api/v5/public/instruments?instType=SPOT
        GET /api/v5/public/instruments?instType=SPOT

        For option, Either `uly` or `instFamily` is required
        """
        params = {}
        if self.DATA_TYPE == "spot":
            params = {"instType": "SPOT"}
        elif self.DATA_TYPE == "option":
            params = {"instType": "OPTION", "instFamily": f"{base_coin}-USD"}
        else:
            if not symbol:
                params = {"instType": "SWAP"}
                symbols_perp = self._get("/public/instruments", params=params)
                if symbols_perp.success:
                    if symbols_perp.data:
                        params = {"instType": "FUTURES"}
                        symbols_fut = self._get("/public/instruments", params=params)
                        symbols_perp.data = symbols_perp.data.get("data", []) + symbols_fut.data.get("data", [])
                        # print(symbols_perp.data,'???')
                        return self._regular_symbols_payload(symbols_perp)
                    else:
                        symbols_perp.msg = "query symbol is empty"

                return symbols_perp

        if symbol:
            params["instId"] = symbol
            params["instType"] = models.okx_inst_type(symbol)

        symbols = self._get("/public/instruments", params=params)

        if symbols.success:
            if symbols.data:
                symbols.data = symbols.data.get("data", [])
                return self._regular_symbols_payload(symbols)
            else:
                symbols.msg = "query symbol is empty"

        return symbols

    def query_trades(
        self,
        symbol: str = None,
        start: datetime = None,
        end: datetime = None,
        limit=100,
    ) -> dict:
        """
        TODO:
        1. pagenation... since currently doesnt hv many datas....
        2. inst type.... (need multiple invokes)
        """
        payload = {"limit": limit}
        is_old: bool = False
        uri = "/trade/fills"
        if start:
            if (datetime.now(UTC_TZ) - start.replace(tzinfo=UTC_TZ)).days > 3:
                is_old = True

            payload["begin"] = int(datetime.timestamp(start) * 1000)
        if end:
            payload["end"] = int(datetime.timestamp(end) * 1000)
        if symbol:
            payload["instId"] = symbol

        if is_old:
            # if self.DATA_TYPE == 'spot':
            #     payload={"instType": "SPOT"}
            # else:
            #     if not symbol:
            #         payload={"instType": "SWAP"}
            #         trades_perp = self._get('/trade/fills', params=payload)
            #         if trades_perp.success:
            #             if trades_perp.data:
            #                 params={"instType": "FUTURES"}
            #                 trades_fut = self._get('/trade/fills', params=payload)
            #                 trades_perp.data = trades_perp.data.get('data', []) + trades_fut.data.get('data', [])
            #                 return self._regular_trades_payload(trades_perp)
            #             else:
            #                 trades_perp.msg = "query symbol is empty"

            #         return trades_perp
            if symbol:
                payload["instType"] = models.okx_inst_type(symbol=symbol)
            else:
                payload["instType"] = models.okx_inst_type(
                    datatype=self.DATA_TYPE
                )  ### TODO: multiple invoke to swap and futures
            uri = "/trade/fills-history"

        # print(f"trade hist payload:{payload}, is_old:{is_old}")
        trades = self._get(uri, params=payload)

        if trades.success:
            if trades.data:
                trades.data = trades.data.get("data", [])
                return self._regular_trades_payload(trades)
            else:
                trades.msg = "query symbol is empty"

        return trades

    def query_history(
        self,
        symbol: str,
        interval: str = "1h",
        start: datetime = None,
        end: datetime = None,
        limit: int = 100,
    ) -> dict:
        """
        # e.g. [1m/3m/5m/15m/30m/1H/2H/4H/6H/12H/1D/1W/1M/3M/6M/1Y]

        Bar size, the default is 1m
        e.g. [1m/3m/5m/15m/30m/1H/2H/4H]
        Hong Kong time opening price k-line: [6H/12H/1D/2D/3D/1W/1M/3M/6M/1Y]
        UTC time opening price k-line: [6Hutc/12Hutc/1Dutc/2Dutc/3Dutc/1Wutc/1Mutc/3Mutc/6Mutc/1Yutc]

        """

        def iteration_history(symbol, interval, data, payload, retry=1):
            # give data from start time with limit
            retry_counter = 0
            historical_prices_datalist = data.data.get("data", [])
            historical_prices_datalist.sort(key=lambda k: k[0])
            # logger.debug(interval)
            if not historical_prices_datalist:
                data.data = historical_prices_datalist
                return data
            last_timestamp = int(historical_prices_datalist[1][0])
            start_timestamp = payload["before"]
            # end_timestamp = payload["after"]
            # interval_timestamp = end_timestamp - last_timestamp
            interval_timestamp = last_timestamp - start_timestamp
            # print(f"init last_timestamp {last_timestamp} {datetime.fromtimestamp(last_timestamp/1000)}")
            # print(f"init start_timestamp {start_timestamp} {datetime.fromtimestamp(start_timestamp/1000)}")
            # print(f"init interval_timestamp {interval_timestamp} - interval_timestamp > interval:{interval_timestamp > interval}")

            while interval_timestamp > interval:
                last_timestamp = int(historical_prices_datalist[1][0])
                # interval_timestamp = end_timestamp - last_timestamp
                interval_timestamp = last_timestamp - start_timestamp

                # print(f"last_timestamp {last_timestamp} {datetime.fromtimestamp(last_timestamp/1000)}")
                # # print(f"end_timestamp {end_timestamp} {datetime.fromtimestamp(end_timestamp/1000)}")
                # print(f"interval_timestamp {interval_timestamp}")
                # # print(historical_prices_datalist[0:5])

                payload["after"] = last_timestamp

                prices = self._get("/market/candles", self._remove_none(payload))  # history-
                if prices.error:
                    print(prices, "ERROR!")
                    break
                # print(prices.data,'!!!!!!!!!!!')
                prices_data = prices.data.get("data", [])
                # print(prices_data[:2],f'???len{len(prices_data)}???', prices_data[-2:])

                historical_prices_datalist.extend(prices_data)
                historical_prices_datalist = [list(item) for item in set(tuple(x) for x in historical_prices_datalist)]
                historical_prices_datalist.sort(key=lambda k: k[0])
                time.sleep(0.1)

                # print(f"payload: {payload} start:{datetime.fromtimestamp(payload['before']/1000)}; end:{datetime.fromtimestamp(payload['after']/1000)}")
                # print(f"retry_counter: {retry_counter}")

                if len(prices_data) != 100:
                    retry_counter += 1

                if retry_counter >= retry:
                    break

            data.data = historical_prices_datalist
            # print(f"data length: {len(historical_prices_datalist)}")
            return data

        symbol = symbol.upper()

        inp_interval = INTERVAL_MAP.get(interval)
        if not inp_interval:
            return models.CommonResponseSchema(success=False, error=True, data={}, msg="Invalid interval.")

        payload = {"instId": symbol, "bar": inp_interval}
        if limit:
            payload["limit"] = limit

        if start:
            start = UTC_TZ.localize(start)  # NOTE: okx start and end timestamp default regard it as utc timestamp
            payload["before"] = int(datetime.timestamp(start)) * 1000  # convert to mm
            if not end:
                end = datetime.now()
        if end:
            end = UTC_TZ.localize(end)
            payload["after"] = int(datetime.timestamp(end)) * 1000  # convert to mm

        interval_ts = TIMEDELTA_MAPPING_SEC[interval] * 1000
        # print(f"ohlc payload:{payload}; start:{start} end:{end}; interval_ts:{interval_ts}")
        # if (payload['after'] - payload['before']) > interval_ts:
        #     ### it is due the okx fetch hist start from enddate (& return in desc )
        #     payload['after'] = None

        historical_prices = self._get("/market/candles", self._remove_none(payload))  # history-
        # if end:
        #     payload['after'] = int(datetime.timestamp(end)) * 1000
        # print(historical_prices,'???')
        extra = {"symbol": symbol, "interval": interval}

        if historical_prices.success:
            if historical_prices.data:
                # print(historical_prices.data,'!!!!!!1!!!!!')
                if "before" in payload:
                    historical_prices = iteration_history(
                        symbol=symbol,
                        interval=interval_ts,
                        data=historical_prices,
                        payload=payload,
                    )
                else:
                    # historical_prices.data = historical_prices.data.get('data')
                    historical_prices.data = historical_prices.data.get("data", [])

                # print(f"------->:{historical_prices} - len:{len(historical_prices.data)}")
                return self._regular_historical_prices_payload(historical_prices, extra=extra)
            else:
                historical_prices.msg = "query historical ohlc is empty"
                historical_prices.data = []

        return historical_prices

    def query_last_price(self, symbol: str, interval: str = "1m", limit: int = 1) -> dict:
        """ """
        return self.query_history(symbol, interval, limit=1)

    def query_prices(self, symbol: str, base_coin="BTC") -> dict:
        """
        GET /api/v5/market/ticker?instId=BTC-USD-SWAP

        {
            "code":"0",
            "msg":"",
            "data":[
            {
                "instType":"SWAP",
                "instId":"BTC-USD-SWAP",
                "last":"9999.99",
                "lastSz":"0.1",
                "askPx":"9999.99",
                "askSz":"11",
                "bidPx":"8888.88",
                "bidSz":"5",
                "open24h":"9000",
                "high24h":"10000",
                "low24h":"8888.88",
                "volCcy24h":"2222",
                "vol24h":"2222",
                "sodUtc0":"2222",
                "sodUtc8":"2222",
                "ts":"1597026383085"
            }
        ]
        }

        """
        symbol = symbol.upper()

        payload = {"instId": symbol}

        # /api/v5/public/opt-summary
        spots = self._get("/market/ticker", self._remove_none(payload))

        # logger.debug(spots)
        if spots.success:
            if spots.data:
                d = spots.data.get("data", [])
                spots.data = d[0] if d else {}
                return self._regular_symbol_payload(spots)
            else:
                spots.msg = "query spots is empty"

        return spots

    def set_leverage(
        self,
        leverage: float,
        symbol: str,
        is_isolated: bool = True,
        position_side: str = None,
    ) -> dict:
        """
        set leverage for isolated `MARGIN` at pairs level
        POST /api/v5/account/set-leverage
        body
        {
            "instId":"BTC-USDT",
            "lever":"5",
            "mgnMode":"isolated"
        }

        set leverage for cross `MARGIN` in Single-currency margin at pairs level
        POST /api/v5/account/set-leverage
        body
        {
            "instId":"BTC-USDT",
            "lever":"5",
            "mgnMode":"cross"
        }

        set leverage for cross `MARGIN` in Multi-currency margin at currency level
        POST /api/v5/account/set-leverage
        body
        {
            "ccy":"BTC",
            "lever":"5",
            "mgnMode":"cross"
        }

        set leverage on long BTC-USDT-200802 for isolated `FUTURES`
        POST /api/v5/account/set-leverage
        body
        {
            "instId":"BTC-USDT-200802",
            "lever":"5",
            "posSide":"long",
            "mgnMode":"isolated"
        }

        set leverage for cross `FUTURES/SWAP` at underlying level
        POST /api/v5/account/set-leverage
        body
        {
            "instId":"BTC-USDT-200802",
            "lever":"5",
            "mgnMode":"cross"
        }
        Response:
        {
        "code": "0",
        "msg": "",
        "data": [
            {
            "lever": "30",
            "mgnMode": "isolated",
            "instId": "BTC-USDT-SWAP",
            "posSide": "long"
            }
        ]
        }

        """
        payload = {
            "instId": symbol.upper(),
            "mgnMode": "isolated" if is_isolated else "cross",
            "lever": str(leverage),
        }
        if position_side:
            assert position_side in ["long", "short"]
            payload["posSide"] = position_side
        leverage_payload = self._post("/account/set-leverage", self._remove_none(payload))

        if leverage_payload.success:
            if leverage_payload.data:
                leverage_payload.data = models.LeverageSchema(symbol=symbol, leverage=leverage).dict()
                return leverage_payload
            else:
                leverage_payload.msg = "fail to config leverage_payload"

        return leverage_payload

    def query_position(self, force_return: bool = False) -> dict:
        """query_position
        {
        "code": "0",
        "msg": "",
        "data": [
            {
            "adl":"1",
            "availPos":"1",
            "avgPx":"2566.31",
            "cTime":"1619507758793",
            "ccy":"ETH",
            "deltaBS":"",
            "deltaPA":"",
            "gammaBS":"",
            "gammaPA":"",
            "imr":"",
            "instId":"ETH-USD-210430",
            "instType":"FUTURES",
            "interest":"0",
            "usdPx":"",
            "last":"2566.22",
            "lever":"10",
            "liab":"",
            "liabCcy":"",
            "liqPx":"2352.8496681818233",
            "markPx":"2353.849",
            "margin":"0.0003896645377994",
            "mgnMode":"isolated",
            "mgnRatio":"11.731726509588816",
            "mmr":"0.0000311811092368",
            "notionalUsd":"2276.2546609009605",
            "optVal":"",
            "pTime":"1619507761462",
            "pos":"1",
            "posCcy":"",
            "posId":"307173036051017730",
            "posSide":"long",
            "spotInUseAmt": "",
            "spotInUseCcy": "",
            "thetaBS":"",
            "thetaPA":"",
            "tradeId":"109844",
            "uTime":"1619507761462",
            "upl":"-0.0000009932766034",
            "uplRatio":"-0.0025490556801078",
            "vegaBS":"",
            "vegaPA":""
            }
        ]
        }

        """

        payload = {
            # 'instType':'SWAP'
        }

        position = self._get("/account/positions", self._remove_none(payload))
        # print(f"position:{position}")
        """
        position:success=True error=False data={'code': '0', 'data': [{'adl': '1', 'availPos': '1', 'avgPx': '1333', 'baseBal': '', 'cTime': '1664201788281', 'ccy': 'USDT', 'deltaBS': '', 'deltaPA': '', 'gammaBS': '', 'gammaPA': '', 'imr': '', 'instId': 'ETH-USDT-SWAP', 'instType': 'SWAP', 'interest': '0', 'last': '1274.12', 'lever': '16', 'liab': '', 'liabCcy': '', 'liqPx': '1409.9576455948234', 'margin': '8.33125', 'markPx': '1274.09', 'mgnMode': 'isolated', 'mgnRatio': '24.805939925750945', 'mmr': '0.509636', 'notionalUsd': '127.40772591', 'optVal': '', 'pendingCloseOrdLiabVal': '', 'pos': '1', 'posCcy': '', 'posId': '494170398842572847', 'posSide': 'short', 'quoteBal': '', 'spotInUseAmt': '', 'spotInUseCcy': '', 'thetaBS': '', 'thetaPA': '', 'tradeId': '661054265', 'uTime': '1664201788281', 'upl': '5.891000000000009', 'uplRatio': '0.7070967741935501', 'usdPx': '', 'vegaBS': '', 'vegaPA': ''}], 'msg': ''} msg='query ok'
        success=True error=False data=[{'asset_type': 'PERPETUAL', 'code': 'ETH-USDT-SWAP', 'symbol': 'ETHUSDT', 'side': 'SHORT', 'size': 1.0, 'position_value': 127.40772591, 'entry_price': None, 'leverage': 16.0, 'position_margin': 8.33125, 'initial_margin': 0.0, 'maintenance_margin': 0.509636, 'realised_pnl': None, 'unrealised_pnl': 5.891000000000009, 'is_isolated': True, 'auto_add_margin': None, 'liq_price': '1409.9576455948234', 'bust_price': None}] msg='query ok' =======query_position 1
        """

        if position.success:
            if position.data:
                position.data = position.data.get("data", [])
                return self._regular_position_payload(position)
            else:
                position.msg = "query future position is empty"

        return position

    def query_transfer(self, start: datetime = None, end: datetime = None) -> dict:
        """ """
        payload = {}
        # ms - epoch time
        if start:
            start = UTC_TZ.localize(start)  # NOTE: start and end timestamp default regard it as utc timestamp
            payload["after"] = int(datetime.timestamp(start)) * 1000
        if end:
            end = UTC_TZ.localize(end)
            payload["before"] = int(datetime.timestamp(end)) * 1000
        # if payload:
        #     if not payload.get('start_time'):
        #         payload['start_time'] = payload['end_time'] - 2592000
        #     if not payload.get('end_time'):
        #         payload['end_time'] = payload['start_time'] + 2592000
        # # print(f"query_transfer payload:{payload}")

        # raw_end_time = payload.get("before")
        deposit_list = []
        withdraw_list = []

        deposit = self._get("/asset/deposit-history", params=payload)
        # print(deposit,'..deposit...')
        if deposit.success and deposit.data:
            deposit_list += deposit.data.get("data", [])
        else:
            return deposit

        withdraw = self._get("/asset/withdrawal-history", params=payload)
        # print(withdraw,'...withdraw..')
        if withdraw.success and withdraw.data:
            withdraw_list += withdraw.data.get("data", [])
        else:
            return withdraw

        # print(f'deposit_list:{deposit_list}; len:{len(deposit_list)}')
        deposit.data = deposit_list
        deposit_ret = self._regular_transfer_payload(deposit, transfer_type="external", inout="in")
        # print(f'withdraw_list:{withdraw_list}; len:{len(withdraw_list)}')
        withdraw.data = withdraw_list
        withdraw_ret = self._regular_transfer_payload(withdraw, transfer_type="external", inout="out")

        withdraw_ret.data = withdraw_ret.data + deposit_ret.data

        return withdraw_ret

    def query_nd_info(self):
        nd_info = self._get("/broker/nd/info")
        # print(f"nd_info:{nd_info}")
        # # success=True error=False data={'code': '0', 'data': [{'level': 'lv1', 'maxSubAcctQty': '10000', 'subAcctQty': '0', 'ts': '1665995142708'}], 'msg': ''} msg='query ok'
        return nd_info

    def query_sub_accounts(self):
        sub_accs_info = self._get("/broker/nd/subaccount-info")
        # print(f"nd_info:{nd_info}")
        # # success=True error=False data={'code': '0', 'data': [{'level': 'lv1', 'maxSubAcctQty': '10000', 'subAcctQty': '0', 'ts': '1665995142708'}], 'msg': ''} msg='query ok'
        return sub_accs_info


class OKXDataWebsocket(WebsocketClient):
    """ """

    DATA_TYPE = "spot"

    def __init__(
        self,
        proxy_host: str = "",
        proxy_port: int = 0,
        ping_interval: int = 18,
        is_testnet: bool = False,
        datatype: str = "spot",
        speed: float = 100,
        debug: bool = False,
        logger=None,
        **kwargs,
    ) -> None:
        super().__init__(
            debug=debug,
            logger=logger,
        )

        assert datatype in ["spot", "linear", "inverse", "option"]
        self.DATA_TYPE = datatype

        self.is_testnet: bool = is_testnet
        self.proxy_host: str = proxy_host
        self.proxy_port: int = proxy_port
        self.ping_interval: int = ping_interval

        self.ticks: dict[str, dict] = defaultdict(dict)
        self.reqid: int = 0

        self.kbar_time: int = int(time.time()) - (int(time.time()) % 60)
        self.bids_asks_data: dict[str, dict] = defaultdict(dict)
        self.ticker_data: dict[str, dict] = defaultdict(dict)

        self.on_tick = None
        self.on_ticker_callback = None
        self.on_depth_callback = None
        self.on_kline_callback = None
        self.on_connected_callback = None
        self.on_disconnected_callback = None
        self.on_error_callback = None

        self.last_time = time.monotonic()
        self.speed = speed

        # self.waitfor_connection()


class OKXTradeWebsocket(WebsocketClient):
    """ """

    DATA_TYPE = None

    def __init__(
        self,
        on_account_callback: Callable = None,
        on_order_callback: Callable = None,
        on_trade_callback: Callable = None,
        on_position_callback: Callable = None,
        on_connected_callback: Callable = None,
        on_disconnected_callback: Callable = None,
        on_error_callback: Callable = None,
        proxy_host: str = "",
        proxy_port: int = 0,
        datatype: str = None,
        is_testnet: bool = False,
        is_demo: bool = False,
        ping_interval: int = 15,
        debug: bool = False,
        logger=None,
        **kwargs,
    ) -> None:
        super().__init__(debug=debug, logger=logger)
        if datatype:
            assert datatype in ["spot", "linear", "inverse", "option", ""]
        self.DATA_TYPE = datatype
        self.on_account_callback = on_account_callback
        self.on_order_callback = on_order_callback
        self.on_trade_callback = on_trade_callback
        self.on_position_callback = on_position_callback
        self.on_connected_callback = on_connected_callback
        self.on_disconnected_callback = on_disconnected_callback
        self.on_error_callback = on_error_callback

        self.ping_interval = ping_interval
        self.is_testnet = is_testnet
        self.is_demo = is_demo
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self._logged_in = False
        self.key = None
        self.secret = None
        self.passphrase = None
        self.debug = debug

    def info(self):
        return f"OKX {self.DATA_TYPE} Trade Websocket Start"

    # def custom_ping(self):
    #     return json.dumps({"op": "ping"})

    def connect(
        self,
        key: str,
        secret: str,
        passphrase: str,
        is_testnet: bool = None,
        is_demo: bool = None,
        **kwargs,
    ):
        """
        Give api key and secret for Authorization.
        """
        self.key = key
        self.secret = secret
        self.passphrase = passphrase
        if is_testnet is not None:
            self.is_testnet = is_testnet
        if is_demo is not None:
            self.is_demo = is_demo

        host = OKX_TESTNET_PRIVATE_WEBSOCKET_HOST if self.is_testnet else OKX_PRIVATE_WEBSOCKET_HOST
        self.init(
            host=host,
            proxy_host=self.proxy_host,
            proxy_port=self.proxy_port,
            ping_interval=self.ping_interval,
        )
        # self.init(host=host+"?max_alive_time=10m", proxy_host=self.proxy_host, proxy_port=self.proxy_port, ping_interval=self.ping_interval)
        self.start()

        # self.waitfor_connection()
        if self.waitfor_connection():
            logger.info(self.info() + "......")
        else:
            logger.info(self.info() + " failed")
            return False

        # if not self._logged_in:
        #     self._login()
        # logger.info(self.info())

        return True

    def _login(self):
        """
        Authorize websocket connection.
        """
        # Generate expires.
        expires = int((time.time() + 1) * 1000)

        if self.key is None or self.secret is None:
            raise PermissionError(f"Authenticated endpoints require keys. api_key:{self.key}; api_secret:{self.secret}")

        timestamp: str = str(time.time())
        msg: str = timestamp + "GET" + "/users/self/verify"
        # signature: bytes = generate_signature(msg, self.secret)
        signature: bytes = base64.b64encode(hmac.new(self.secret.encode(), msg.encode(), hashlib.sha256).digest())

        req: dict = {
            "op": "login",
            "args": [
                {
                    "apiKey": self.key,
                    "passphrase": self.passphrase,
                    "timestamp": timestamp,
                    "sign": signature.decode("utf-8"),
                }
            ],
        }
        self.send_packet(req)
        if self.debug:
            print(f"auth req:{req}")
        self._logged_in = True

    def on_disconnected(self) -> None:
        print(f"OKX {self.DATA_TYPE} user trade data Websocket disconnected, now try to connect @ {datetime.now()}")
        self._logged_in = False
        if self.on_disconnected_callback:
            self.on_disconnected_callback()

    def on_connected(self) -> None:
        if self.waitfor_connection():
            print(f"OKX {self.DATA_TYPE} user trade data websocket connect success")
        else:
            print(f"OKX {self.DATA_TYPE} user trade data websocket connect fail")

        if not self._logged_in:
            self._login()
            # self._logged_in = True
        # logger.debug("OKX Future user trade data Websocket connect success")

        # req: dict = self.sub_stream(channel=[ "orders", "account", "position"])
        # if self.debug:
        #     print(f"OKX {self.DATA_TYPE} user trade data websocket on_connected req:{req} - self._active:{self._active}")

        # self.send_packet(req)
        # if self.on_connected_callback:
        #     self.on_connected_callback()

    def on_error(self, exception_type, exception_value, traceback_obj):
        super().on_error(exception_type, exception_value, traceback_obj)
        if self.on_error_callback:
            self.on_error_callback(exception_value)

    def inject_callback(self, channel: str, callbackfn: Callable):
        if channel == "account":
            self.on_account_callback = callbackfn
        elif channel == "orders":
            self.on_order_callback = callbackfn
        elif channel == "trades":
            self.on_trade_callback = callbackfn
        elif channel == "position":
            self.on_position_callback = callbackfn
        elif channel == "connect":
            self.on_connected_callback = callbackfn
        elif channel == "disconnect":
            self.on_disconnected_callback = callbackfn
        elif channel == "error":
            self.on_error_callback = callbackfn
        else:
            Exception("invalid callback function")

    def sub_stream(self, channel="orders"):
        """ """
        sub = {
            "op": "subscribe",
        }
        if isinstance(channel, list):
            sub["args"] = [{"channel": MAPPING_CHANNEL[c], "instType": "ANY"} for c in channel]
        else:
            mapping_channel = MAPPING_CHANNEL[channel]
            sub["args"] = [{"channel": f"{mapping_channel}", "instType": "ANY"}]
        # okx_req: dict = {
        #     "op": "subscribe",
        #     "args": [
        #         {
        #             "channel": "orders",
        #             "instType": "ANY"
        #         },
        #         {
        #             "channel": "account"
        #         },
        #         {
        #             "channel": "positions",
        #             "instType": "ANY"
        #         },
        #     ]
        # }
        return sub

    def subscribe(self, channel: str = None) -> None:
        """ """
        if channel and channel not in ["trades", "orders", "account", "position"]:
            Exception("invalid subscription")
            return

        if not self._logged_in:
            self._login()

        ### Default sub all private channels if not specified
        self.channel = channel
        req: dict = self.sub_stream(channel=channel)
        if self.debug:
            print(f"subscribe - req: {req}")
        self.send_packet(req)

    def on_packet(self, packet: dict) -> None:
        """
        [DEBUG] OKX option Trade Websocket - topic:account; on_packet event {'arg': {'channel': 'account', 'uid': '315131188874494398'}, 'data': [{'adjEq': '', 'borrowFroz': '', 'details': [{'availBal': '3.2960824157473905', 'availEq': '3.2960824157473905', 'borrowFroz': '', 'cashBal': '3.2960824157473905', 'ccy': 'BTC', 'clSpotInUseAmt': '', 'coinUsdPrice': '64435', 'crossLiab': '', 'disEq': '212450.70277686208', 'eq': '3.297132036577358', 'eqUsd': '212450.70277686208', 'fixedBal': '0', 'frozenBal': '0.0010496208299675', 'imr': '0', 'interest': '', 'isoEq': '0.0010496208299675', 'isoLiab': '', 'isoUpl': '-0.0004565552923896', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1720771215153', 'upl': '-0.0004565552923896', 'uplLiab': ''}, {'availBal': '9000', 'availEq': '9000', 'borrowFroz': '', 'cashBal': '9000', 'ccy': 'TUSD', 'clSpotInUseAmt': '', 'coinUsdPrice': '6.9115226481', 'crossLiab': '', 'disEq': '62203.7038329', 'eq': '9000', 'eqUsd': '62203.7038329', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1654528914706', 'upl': '0', 'uplLiab': ''}, {'availBal': '15.3766562', 'availEq': '15.3766562', 'borrowFroz': '', 'cashBal': '15.3766562', 'ccy': 'ETH', 'clSpotInUseAmt': '', 'coinUsdPrice': '3179.23', 'crossLiab': '', 'disEq': '48885.926690725995', 'eq': '15.3766562', 'eqUsd': '48885.926690725995', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1668614453628', 'upl': '0', 'uplLiab': ''}, {'availBal': '300', 'availEq': '300', 'borrowFroz': '', 'cashBal': '300', 'ccy': 'OKB', 'clSpotInUseAmt': '', 'coinUsdPrice': '39.166', 'crossLiab': '', 'disEq': '9399.84', 'eq': '300', 'eqUsd': '11749.8', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1654528914654', 'upl': '0', 'uplLiab': ''}, {'availBal': '1500', 'availEq': '1500', 'borrowFroz': '', 'cashBal': '1500', 'ccy': 'UNI', 'clSpotInUseAmt': '', 'coinUsdPrice': '7.092', 'crossLiab': '', 'disEq': '7446.599999999999', 'eq': '1500', 'eqUsd': '10638', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1654528914706', 'upl': '0', 'uplLiab': ''}, {'availBal': '9000', 'availEq': '9000', 'borrowFroz': '', 'cashBal': '9000', 'ccy': 'USDK', 'clSpotInUseAmt': '', 'coinUsdPrice': '1', 'crossLiab': '', 'disEq': '0', 'eq': '9000', 'eqUsd': '9000', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1654528914603', 'upl': '0', 'uplLiab': ''}, {'availBal': '8923.55158475', 'availEq': '8923.55158475', 'borrowFroz': '', 'cashBal': '8923.55158475', 'ccy': 'USDC', 'clSpotInUseAmt': '', 'coinUsdPrice': '1.0003', 'crossLiab': '', 'disEq': '8361.256098359645', 'eq': '8987.90158475', 'eqUsd': '8990.597955225425', 'fixedBal': '0', 'frozenBal': '64.35000000000001', 'imr': '0', 'interest': '', 'isoEq': '64.35000000000001', 'isoLiab': '', 'isoUpl': '-0.1275', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1722506419167', 'upl': '-0.1275', 'uplLiab': ''}, {'availBal': '9000', 'availEq': '9000', 'borrowFroz': '', 'cashBal': '9000', 'ccy': 'PAX', 'clSpotInUseAmt': '', 'coinUsdPrice': '0.998', 'crossLiab': '', 'disEq': '0', 'eq': '9000', 'eqUsd': '8982', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1654528914654', 'upl': '0', 'uplLiab': ''}, {'availBal': '300', 'availEq': '300', 'borrowFroz': '', 'cashBal': '300', 'ccy': 'JFI', 'clSpotInUseAmt': '', 'coinUsdPrice': '19.7955412', 'crossLiab': '', 'disEq': '0', 'eq': '300', 'eqUsd': '5938.662359999999', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1654528914654', 'upl': '0', 'uplLiab': ''}, {'availBal': '5288.213214513', 'availEq': '5288.213214513', 'borrowFroz': '', 'cashBal': '5288.213214513', 'ccy': 'USDT', 'clSpotInUseAmt': '', 'coinUsdPrice': '0.99997', 'crossLiab': '', 'disEq': '5295.663947766844', 'eq': '5295.822822451518', 'eqUsd': '5295.663947766844', 'fixedBal': '0', 'frozenBal': '7.6096079385178195', 'imr': '0', 'interest': '', 'isoEq': '7.6096079385178195', 'isoLiab': '', 'isoUpl': '1.6269299999999995', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1720685164051', 'upl': '1.6269299999999995', 'uplLiab': ''}, {'availBal': '30000', 'availEq': '30000', 'borrowFroz': '', 'cashBal': '30000', 'ccy': 'TRX', 'clSpotInUseAmt': '', 'coinUsdPrice': '0.129', 'crossLiab': '', 'disEq': '2709', 'eq': '30000', 'eqUsd': '3870', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1654528914758', 'upl': '0', 'uplLiab': ''}, {'availBal': '30', 'availEq': '30', 'borrowFroz': '', 'cashBal': '30', 'ccy': 'LTC', 'clSpotInUseAmt': '', 'coinUsdPrice': '70.39', 'crossLiab': '', 'disEq': '2006.1149999999998', 'eq': '30', 'eqUsd': '2111.7', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1654528914706', 'upl': '0', 'uplLiab': ''}, {'availBal': '3043.956', 'availEq': '3043.956', 'borrowFroz': '', 'cashBal': '3043.956', 'ccy': 'ADA', 'clSpotInUseAmt': '', 'coinUsdPrice': '0.38561', 'crossLiab': '', 'disEq': '939.0238985280001', 'eq': '3043.956', 'eqUsd': '1173.77987316', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1664613439903', 'upl': '0', 'uplLiab': ''}], 'imr': '', 'isoEq': '139.61100282923684', 'mgnRatio': '', 'mmr': '', 'notionalUsd': '', 'ordFroz': '', 'totalEq': '391290.53743664036', 'uTime': '1722506420487', 'upl': ''}]}
        [DEBUG] OKX option Trade Websocket - topic:positions; on_packet event {'arg': {'channel': 'positions', 'instType': 'ANY', 'uid': '315131188874494398'}, 'data': [{'adl': '1', 'availPos': '3', 'avgPx': '64477.5', 'baseBal': '', 'baseBorrowed': '', 'baseInterest': '', 'bePx': '64542.00975487743', 'bizRefId': '', 'bizRefType': '', 'cTime': '1722505680519', 'ccy': 'USDC', 'clSpotInUseAmt': '', 'closeOrderAlgo': [], 'deltaBS': '', 'deltaPA': '', 'fee': '-0.09671625', 'fundingFee': '0', 'gammaBS': '', 'gammaPA': '', 'idxPx': '64430.0', 'imr': '', 'instId': 'BTC-USDC-SWAP', 'instType': 'SWAP', 'interest': '', 'last': '64435.4', 'lever': '3', 'liab': '', 'liabCcy': '', 'liqPenalty': '0', 'liqPx': '43288.116112789525', 'margin': '64.4775', 'markPx': '64435', 'maxSpotInUseAmt': '', 'mgnMode': 'isolated', 'mgnRatio': '47.55623053132172', 'mmr': '1.2564825', 'notionalUsd': '193.3629915', 'optVal': '', 'pTime': '1722506420487', 'pendingCloseOrdLiabVal': '', 'pnl': '0', 'pos': '3', 'posCcy': '', 'posId': '1677831675978555392', 'posSide': 'long', 'quoteBal': '', 'quoteBorrowed': '', 'quoteInterest': '', 'realizedPnl': '-0.09671625', 'spotInUseAmt': '', 'spotInUseCcy': '', 'thetaBS': '', 'thetaPA': '', 'tradeId': '43738140', 'uTime': '1722506419167', 'upl': '-0.1275', 'uplLastPx': '-0.1262999999999956', 'uplRatio': '-0.0019774339886006', 'uplRatioLastPx': '-0.0019588228451786', 'usdPx': '', 'vegaBS': '', 'vegaPA': ''}, {'adl': '5', 'availPos': '1', 'avgPx': '0.0905', 'baseBal': '', 'baseBorrowed': '', 'baseInterest': '', 'bePx': '', 'bizRefId': '', 'bizRefType': '', 'cTime': '1720698008696', 'ccy': 'BTC', 'clSpotInUseAmt': '', 'closeOrderAlgo': [], 'deltaBS': '-0.008182987697401478', 'deltaPA': '-0.006820814217141863', 'fee': '-0.000002', 'fundingFee': '0', 'gammaBS': '-2.6669874733980323E-7', 'gammaPA': '-0.0035439854564227147', 'idxPx': '64435.0', 'imr': '', 'instId': 'BTC-USD-240830-55000-C', 'instType': 'OPTION', 'interest': '', 'last': '0.131', 'lever': '', 'liab': '', 'liabCcy': '', 'liqPenalty': '0', 'liqPx': '', 'margin': '0.0024111761223571', 'markPx': '0.136155529238958', 'maxSpotInUseAmt': '', 'mgnMode': 'isolated', 'mgnRatio': '1.448540720384059', 'mmr': '0.0016615552923896', 'notionalUsd': '644.35', 'optVal': '-0.0013615552923896', 'pTime': '1722506420487', 'pendingCloseOrdLiabVal': '', 'pnl': '0', 'pos': '-1', 'posCcy': '', 'posId': '1616748673207209984', 'posSide': 'net', 'quoteBal': '', 'quoteBorrowed': '', 'quoteInterest': '', 'realizedPnl': '-0.000002', 'spotInUseAmt': '', 'spotInUseCcy': '', 'thetaBS': '-0.014940259680188014', 'thetaPA': '6.977125795067151E-6', 'tradeId': '3', 'uTime': '1720698008696', 'upl': '-0.0004565552923896', 'uplLastPx': '-0.000405', 'uplRatio': '-0.5044809860658341', 'uplRatioLastPx': '-0.4475138121546962', 'usdPx': '64435', 'vegaBS': '-0.4781402851557994', 'vegaPA': '-7.420125688539259E-6'}, {'adl': '1', 'availPos': '0.3', 'avgPx': '59029.5', 'baseBal': '', 'baseBorrowed': '', 'baseInterest': '', 'bePx': '58822.66628808464', 'bizRefId': '', 'bizRefType': '', 'cTime': '1720685164051', 'ccy': 'USDT', 'clSpotInUseAmt': '', 'closeOrderAlgo': [], 'deltaBS': '', 'deltaPA': '', 'fee': '-0.008854425', 'fundingFee': '0.0797279385178201', 'gammaBS': '', 'gammaPA': '', 'idxPx': '64460.0', 'imr': '', 'instId': 'BTC-USDT-SWAP', 'instType': 'SWAP', 'interest': '', 'last': '64451.1', 'lever': '3', 'liab': '', 'liabCcy': '', 'liqPenalty': '0', 'liqPx': '39264.027880402406', 'margin': '5.9826779385178201', 'markPx': '64452.6', 'maxSpotInUseAmt': '', 'mgnMode': 'isolated', 'mgnRatio': '87.45569024561166', 'mmr': '0.07734312', 'notionalUsd': '19.335199926599998', 'optVal': '', 'pTime': '1722506420487', 'pendingCloseOrdLiabVal': '', 'pnl': '0', 'pos': '0.3', 'posCcy': '', 'posId': '1616745279948165120', 'posSide': 'long', 'quoteBal': '', 'quoteBorrowed': '', 'quoteInterest': '', 'realizedPnl': '0.0708735135178201', 'spotInUseAmt': '', 'spotInUseCcy': '', 'thetaBS': '', 'thetaPA': '', 'tradeId': '1081619170', 'uTime': '1722499200598', 'upl': '1.6269299999999995', 'uplLastPx': '1.6264799999999997', 'uplRatio': '0.2756130409371584', 'uplRatioLastPx': '0.2755368078672524', 'usdPx': '', 'vegaBS': '', 'vegaPA': ''}]}
        [DEBUG] OKX option Trade Websocket - topic:orders; on_packet event {'arg': {'channel': 'orders', 'instType': 'ANY', 'uid': '315131188874494398'}, 'data': [{'instType': 'SWAP', 'instId': 'BTC-USDC-SWAP', 'tgtCcy': '', 'ccy': '', 'ordId': '1677856460825534464', 'clOrdId': '', 'algoClOrdId': '', 'algoId': '', 'tag': '', 'px': '', 'sz': '1', 'notionalUsd': '64.45473061999999', 'ordType': 'market', 'side': 'buy', 'posSide': 'long', 'tdMode': 'isolated', 'accFillSz': '1', 'fillNotionalUsd': '64.45473061999999', 'avgPx': '64435.4', 'state': 'filled', 'lever': '3', 'pnl': '0', 'feeCcy': 'USDC', 'fee': '-0.0322177', 'rebateCcy': 'USDC', 'rebate': '0', 'category': 'normal', 'uTime': '1722506419167', 'cTime': '1722506419165', 'source': '', 'reduceOnly': 'false', 'cancelSource': '', 'quickMgnType': '', 'stpId': '', 'stpMode': 'cancel_maker', 'attachAlgoClOrdId': '', 'lastPx': '64475', 'isTpLimit': 'false', 'slTriggerPx': '', 'slTriggerPxType': '', 'tpOrdPx': '', 'tpTriggerPx': '', 'tpTriggerPxType': '', 'slOrdPx': '', 'fillPx': '64435.4', 'tradeId': '43738140', 'fillSz': '1', 'fillTime': '1722506419166', 'fillPnl': '0', 'fillFee': '-0.0322177', 'fillFeeCcy': 'USDC', 'execType': 'T', 'fillPxVol': '', 'fillPxUsd': '', 'fillMarkVol': '', 'fillFwdPx': '', 'fillMarkPx': '64434.9', 'amendSource': '', 'reqId': '', 'amendResult': '', 'code': '0', 'msg': '', 'pxType': '', 'pxUsd': '', 'pxVol': '', 'linkedAlgoOrd': {'algoId': ''}, 'attachAlgoOrds': []}]}
        """
        if "event" in packet:
            topic: str = packet["event"]
        elif "op" in packet:
            topic: str = packet["op"]
        else:
            topic: str = packet["arg"]["channel"]

        if topic == "login":
            self.on_login(packet)

        if self.debug:
            print(f"[DEBUG] OKX {self.DATA_TYPE} Trade Websocket - topic:{topic}; on_packet event {packet}")
        # return

        if not topic:
            logger.error(f"[OKX] the other packet without topic: {packet}")
            return

        if topic == "account":
            self.on_account(packet)
        elif topic == "orders":
            self.on_order(packet)
        elif topic == "positions":
            self.on_position(packet)
        else:
            logger.info(f"the other packet type: {packet}")

    def on_login(self, packet: dict) -> None:
        if packet["code"] == "0":
            req: dict = self.sub_stream(channel=["orders", "account", "position"])
            if self.debug:
                print(
                    f"OKX {self.DATA_TYPE} user trade data websocket on_connected req:{req} - self._active:{self._active}"
                )

            self.send_packet(req)
            self._logged_in = True
        else:
            logger.error("Login Failed.")

    def on_account(self, packet: dict) -> None:
        """
        [DEBUG] OKX option Trade Websocket - topic:account; on_packet event {'arg': {'channel': 'account', 'uid': '315131188874494398'}, 'data': [{'adjEq': '', 'borrowFroz': '', 'details': [{'availBal': '3.2960824157473905', 'availEq': '3.2960824157473905', 'borrowFroz': '', 'cashBal': '3.2960824157473905', 'ccy': 'BTC', 'clSpotInUseAmt': '', 'coinUsdPrice': '64435', 'crossLiab': '', 'disEq': '212450.70277686208', 'eq': '3.297132036577358', 'eqUsd': '212450.70277686208', 'fixedBal': '0', 'frozenBal': '0.0010496208299675', 'imr': '0', 'interest': '', 'isoEq': '0.0010496208299675', 'isoLiab': '', 'isoUpl': '-0.0004565552923896', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1720771215153', 'upl': '-0.0004565552923896', 'uplLiab': ''}, {'availBal': '9000', 'availEq': '9000', 'borrowFroz': '', 'cashBal': '9000', 'ccy': 'TUSD', 'clSpotInUseAmt': '', 'coinUsdPrice': '6.9115226481', 'crossLiab': '', 'disEq': '62203.7038329', 'eq': '9000', 'eqUsd': '62203.7038329', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1654528914706', 'upl': '0', 'uplLiab': ''}, {'availBal': '15.3766562', 'availEq': '15.3766562', 'borrowFroz': '', 'cashBal': '15.3766562', 'ccy': 'ETH', 'clSpotInUseAmt': '', 'coinUsdPrice': '3179.23', 'crossLiab': '', 'disEq': '48885.926690725995', 'eq': '15.3766562', 'eqUsd': '48885.926690725995', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1668614453628', 'upl': '0', 'uplLiab': ''}, {'availBal': '300', 'availEq': '300', 'borrowFroz': '', 'cashBal': '300', 'ccy': 'OKB', 'clSpotInUseAmt': '', 'coinUsdPrice': '39.166', 'crossLiab': '', 'disEq': '9399.84', 'eq': '300', 'eqUsd': '11749.8', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1654528914654', 'upl': '0', 'uplLiab': ''}, {'availBal': '1500', 'availEq': '1500', 'borrowFroz': '', 'cashBal': '1500', 'ccy': 'UNI', 'clSpotInUseAmt': '', 'coinUsdPrice': '7.092', 'crossLiab': '', 'disEq': '7446.599999999999', 'eq': '1500', 'eqUsd': '10638', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1654528914706', 'upl': '0', 'uplLiab': ''}, {'availBal': '9000', 'availEq': '9000', 'borrowFroz': '', 'cashBal': '9000', 'ccy': 'USDK', 'clSpotInUseAmt': '', 'coinUsdPrice': '1', 'crossLiab': '', 'disEq': '0', 'eq': '9000', 'eqUsd': '9000', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1654528914603', 'upl': '0', 'uplLiab': ''}, {'availBal': '8923.55158475', 'availEq': '8923.55158475', 'borrowFroz': '', 'cashBal': '8923.55158475', 'ccy': 'USDC', 'clSpotInUseAmt': '', 'coinUsdPrice': '1.0003', 'crossLiab': '', 'disEq': '8361.256098359645', 'eq': '8987.90158475', 'eqUsd': '8990.597955225425', 'fixedBal': '0', 'frozenBal': '64.35000000000001', 'imr': '0', 'interest': '', 'isoEq': '64.35000000000001', 'isoLiab': '', 'isoUpl': '-0.1275', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1722506419167', 'upl': '-0.1275', 'uplLiab': ''}, {'availBal': '9000', 'availEq': '9000', 'borrowFroz': '', 'cashBal': '9000', 'ccy': 'PAX', 'clSpotInUseAmt': '', 'coinUsdPrice': '0.998', 'crossLiab': '', 'disEq': '0', 'eq': '9000', 'eqUsd': '8982', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1654528914654', 'upl': '0', 'uplLiab': ''}, {'availBal': '300', 'availEq': '300', 'borrowFroz': '', 'cashBal': '300', 'ccy': 'JFI', 'clSpotInUseAmt': '', 'coinUsdPrice': '19.7955412', 'crossLiab': '', 'disEq': '0', 'eq': '300', 'eqUsd': '5938.662359999999', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1654528914654', 'upl': '0', 'uplLiab': ''}, {'availBal': '5288.213214513', 'availEq': '5288.213214513', 'borrowFroz': '', 'cashBal': '5288.213214513', 'ccy': 'USDT', 'clSpotInUseAmt': '', 'coinUsdPrice': '0.99997', 'crossLiab': '', 'disEq': '5295.663947766844', 'eq': '5295.822822451518', 'eqUsd': '5295.663947766844', 'fixedBal': '0', 'frozenBal': '7.6096079385178195', 'imr': '0', 'interest': '', 'isoEq': '7.6096079385178195', 'isoLiab': '', 'isoUpl': '1.6269299999999995', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1720685164051', 'upl': '1.6269299999999995', 'uplLiab': ''}, {'availBal': '30000', 'availEq': '30000', 'borrowFroz': '', 'cashBal': '30000', 'ccy': 'TRX', 'clSpotInUseAmt': '', 'coinUsdPrice': '0.129', 'crossLiab': '', 'disEq': '2709', 'eq': '30000', 'eqUsd': '3870', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1654528914758', 'upl': '0', 'uplLiab': ''}, {'availBal': '30', 'availEq': '30', 'borrowFroz': '', 'cashBal': '30', 'ccy': 'LTC', 'clSpotInUseAmt': '', 'coinUsdPrice': '70.39', 'crossLiab': '', 'disEq': '2006.1149999999998', 'eq': '30', 'eqUsd': '2111.7', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1654528914706', 'upl': '0', 'uplLiab': ''}, {'availBal': '3043.956', 'availEq': '3043.956', 'borrowFroz': '', 'cashBal': '3043.956', 'ccy': 'ADA', 'clSpotInUseAmt': '', 'coinUsdPrice': '0.38561', 'crossLiab': '', 'disEq': '939.0238985280001', 'eq': '3043.956', 'eqUsd': '1173.77987316', 'fixedBal': '0', 'frozenBal': '0', 'imr': '0', 'interest': '', 'isoEq': '0', 'isoLiab': '', 'isoUpl': '0', 'liab': '', 'maxLoan': '', 'maxSpotInUseAmt': '', 'mgnRatio': '', 'mmr': '0', 'notionalLever': '0', 'ordFrozen': '0', 'rewardBal': '0', 'smtSyncEq': '0', 'spotInUseAmt': '', 'spotIsoBal': '0', 'stgyEq': '0', 'twap': '0', 'uTime': '1664613439903', 'upl': '0', 'uplLiab': ''}], 'imr': '', 'isoEq': '139.61100282923684', 'mgnRatio': '', 'mmr': '', 'notionalUsd': '', 'ordFroz': '', 'totalEq': '391290.53743664036', 'uTime': '1722506420487', 'upl': ''}]}
        """
        logger.debug(f"packet on_account: {packet}")
        try:
            if self.DATA_TYPE == "spot":
                account_type = "SPOT"
            else:
                account_type = "CONTRACT"
            for acc_data in packet["data"]:
                for data in acc_data.get("details", []):
                    account = models.AccountSchema()  # .model_construct(data)
                    account = account.from_okx_to_form(data, datatype=self.DATA_TYPE, account_type=account_type)
                    if account and account["symbol"] and self.on_account_callback:
                        self.on_account_callback(account)
        except Exception:
            et, ev, tb = sys.exc_info()
            self.on_error(et, ev, tb)

    def on_order(self, packet: dict) -> None:
        """
        [DEBUG] OKX linear Trade Websocket - on_packet event {'arg': {'channel': 'orders', 'instType': 'ANY', 'uid': '315131188874494398'}, 'data': [{'instType': 'SWAP', 'instId': 'BTC-USDC-SWAP', 'tgtCcy': '', 'ccy': '', 'ordId': '1677856460825534464', 'clOrdId': '', 'algoClOrdId': '', 'algoId': '', 'tag': '', 'px': '', 'sz': '1', 'notionalUsd': '64.45473061999999', 'ordType': 'market', 'side': 'buy', 'posSide': 'long', 'tdMode': 'isolated', 'accFillSz': '1', 'fillNotionalUsd': '64.45473061999999', 'avgPx': '64435.4', 'state': 'filled', 'lever': '3', 'pnl': '0', 'feeCcy': 'USDC', 'fee': '-0.0322177', 'rebateCcy': 'USDC', 'rebate': '0', 'category': 'normal', 'uTime': '1722506419167', 'cTime': '1722506419165', 'source': '', 'reduceOnly': 'false', 'cancelSource': '', 'quickMgnType': '', 'stpId': '', 'stpMode': 'cancel_maker', 'attachAlgoClOrdId': '', 'lastPx': '64475', 'isTpLimit': 'false', 'slTriggerPx': '', 'slTriggerPxType': '', 'tpOrdPx': '', 'tpTriggerPx': '', 'tpTriggerPxType': '', 'slOrdPx': '', 'fillPx': '64435.4', 'tradeId': '43738140', 'fillSz': '1', 'fillTime': '1722506419166', 'fillPnl': '0', 'fillFee': '-0.0322177', 'fillFeeCcy': 'USDC', 'execType': 'T', 'fillPxVol': '', 'fillPxUsd': '', 'fillMarkVol': '', 'fillFwdPx': '', 'fillMarkPx': '64434.9', 'amendSource': '', 'reqId': '', 'amendResult': '', 'code': '0', 'msg': '', 'pxType': '', 'pxUsd': '', 'pxVol': '', 'linkedAlgoOrd': {'algoId': ''}, 'attachAlgoOrds': []}]}
        """
        try:
            logger.debug(f"packet on_order: {packet}")
            datas: dict = packet["data"]

            # if datas["order_type"] not in ['LIMIT', 'Market']:
            #     return

            for data in datas:
                order = models.OrderSchema()
                order = order.from_okx_to_form(data)  # , datatype=self.DATA_TYPE
                if self.on_order_callback:  # and order_list
                    self.on_order_callback(order)
                if order["status"] == "FILLED":
                    trade = models.TradeSchema()  # .model_construct(data)
                    trade = trade.from_okx_to_form(data, datatype=self.DATA_TYPE)
                    if trade and self.on_trade_callback:
                        # print(trade,'....')
                        self.on_trade_callback(trade)

        except Exception:
            et, ev, tb = sys.exc_info()
            self.on_error(et, ev, tb)

    def on_trade(self, packet: dict) -> None:
        """
        [DEBUG] OKX linear Trade Websocket - on_packet event {'id': '1193481299665b2-56be-40da-8a29-f488cb47190b', 'topic': 'execution', 'creationTime': 1678888392300, 'data': [{'category': 'linear', 'symbol': 'BTCUSDT', 'execFee': '0.01490148', 'execId': 'dae9c7a8-ceb9-51fa-8881-d5157ecc2c51', 'execPrice': '24835.8', 'execQty': '0.001', 'execType': 'Trade', 'execValue': '24.8358', 'isMaker': False, 'feeRate': '0.0006', 'tradeIv': '', 'markIv': '', 'blockTradeId': '', 'markPrice': '24883.13', 'indexPrice': '', 'underlyingPrice': '', 'leavesQty': '0', 'orderId': 'e9b9e551-081b-414c-a109-0e56d225ecd2', 'orderLinkId': '', 'orderPrice': '26116.9', 'orderQty': '0.001', 'orderType': 'Market', 'stopOrderType': 'UNKNOWN', 'side': 'Buy', 'execTime': '1678888392289', 'isLeverage': '0'}]}
        """
        try:
            logger.debug(f"packet on_trade: {packet}")
            datas: dict = packet["data"]

            for data in datas:
                trade = models.TradeSchema()  # .model_construct(data)
                trade = trade.from_okx_to_form(data)  # , datatype=self.DATA_TYPE
                if trade and self.on_trade_callback:
                    # print(trade,'....')
                    self.on_trade_callback(trade)
        except Exception:
            et, ev, tb = sys.exc_info()
            self.on_error(et, ev, tb)

    def on_position(self, packet: dict) -> None:
        """
        [DEBUG] OKX linear Trade Websocket - on_packet event {'arg': {'channel': 'positions', 'instType': 'ANY', 'uid': '315131188874494398'}, 'data': [{'adl': '1', 'availPos': '3', 'avgPx': '64477.5', 'baseBal': '', 'baseBorrowed': '', 'baseInterest': '', 'bePx': '64543.92892333837', 'bizRefId': '', 'bizRefType': '', 'cTime': '1722505680519', 'ccy': 'USDC', 'clSpotInUseAmt': '', 'closeOrderAlgo': [], 'deltaBS': '', 'deltaPA': '', 'fee': '-0.09671625', 'fundingFee': '-0.0057546266301038', 'gammaBS': '', 'gammaPA': '', 'idxPx': '64533.2', 'imr': '', 'instId': 'BTC-USDC-SWAP', 'instType': 'SWAP', 'interest': '', 'last': '64555', 'lever': '3', 'liab': '', 'liabCcy': '', 'liqPenalty': '0', 'liqPx': '43290.04784378318', 'margin': '64.4717453733698962', 'markPx': '64550.6', 'maxSpotInUseAmt': '', 'mgnMode': 'isolated', 'mgnRatio': '47.72265432328238', 'mmr': '1.2587367', 'notionalUsd': '193.69053036', 'optVal': '', 'pTime': '1722591938341', 'pendingCloseOrdLiabVal': '', 'pnl': '0', 'pos': '3', 'posCcy': '', 'posId': '1677831675978555392', 'posSide': 'long', 'quoteBal': '', 'quoteBorrowed': '', 'quoteInterest': '', 'realizedPnl': '-0.1024708766301038', 'spotInUseAmt': '', 'spotInUseCcy': '', 'thetaBS': '', 'thetaPA': '', 'tradeId': '43738140', 'uTime': '1722585600399', 'upl': '0.2192999999999956', 'uplLastPx': '0.2325', 'uplRatio': '0.0034011864603929', 'uplRatioLastPx': '0.0036059090380365', 'usdPx': '', 'vegaBS': '', 'vegaPA': ''}, {'adl': '5', 'availPos': '1', 'avgPx': '0.0905', 'baseBal': '', 'baseBorrowed': '', 'baseInterest': '', 'bePx': '', 'bizRefId': '', 'bizRefType': '', 'cTime': '1720698008696', 'ccy': 'BTC', 'clSpotInUseAmt': '', 'closeOrderAlgo': [], 'deltaBS': '-0.008230372994509206', 'deltaPA': '-0.006866524685722093', 'fee': '-0.000002', 'fundingFee': '0', 'gammaBS': '-2.657095664662894E-7', 'gammaPA': '-0.003413932939411576', 'idxPx': '64535.4', 'imr': '', 'instId': 'BTC-USD-240830-55000-C', 'instType': 'OPTION', 'interest': '', 'last': '0.131', 'lever': '', 'liab': '', 'liabCcy': '', 'liqPenalty': '0', 'liqPx': '', 'margin': '0.0024111761223571', 'markPx': '0.1364046840590669', 'maxSpotInUseAmt': '', 'mgnMode': 'isolated', 'mgnRatio': '1.4463757488078566', 'mmr': '0.0016640468405907', 'notionalUsd': '645.354', 'optVal': '-0.0013640468405907', 'pTime': '1722591938341', 'pendingCloseOrdLiabVal': '', 'pnl': '0', 'pos': '-1', 'posCcy': '', 'posId': '1616748673207209984', 'posSide': 'net', 'quoteBal': '', 'quoteBorrowed': '', 'quoteInterest': '', 'realizedPnl': '-0.000002', 'spotInUseAmt': '', 'spotInUseCcy': '', 'thetaBS': '-0.039081738259309405', 'thetaPA': '7.004118291776267E-6', 'tradeId': '3', 'uTime': '1720698008696', 'upl': '-0.0004590468405907', 'uplLastPx': '-0.000405', 'uplRatio': '-0.5072340780007388', 'uplRatioLastPx': '-0.4475138121546962', 'usdPx': '64535.4', 'vegaBS': '-0.462822543065028', 'vegaPA': '-7.171896199529975E-6'}, {'adl': '1', 'availPos': '0.3', 'avgPx': '59029.5', 'baseBal': '', 'baseBorrowed': '', 'baseInterest': '', 'bePx': '58815.15235383589', 'bizRefId': '', 'bizRefType': '', 'cTime': '1720685164051', 'ccy': 'USDT', 'clSpotInUseAmt': '', 'closeOrderAlgo': [], 'deltaBS': '', 'deltaPA': '', 'fee': '-0.008854425', 'fundingFee': '0.0819809917023081', 'gammaBS': '', 'gammaPA': '', 'idxPx': '64593.0', 'imr': '', 'instId': 'BTC-USDT-SWAP', 'instType': 'SWAP', 'interest': '', 'last': '64594', 'lever': '3', 'liab': '', 'liabCcy': '', 'liqPenalty': '0', 'liqPx': '39256.48375455446', 'margin': '5.9849309917023081', 'markPx': '64594.7', 'maxSpotInUseAmt': '', 'mgnMode': 'isolated', 'mgnRatio': '87.77799613879922', 'mmr': '0.07751364', 'notionalUsd': '19.371239988299997', 'optVal': '', 'pTime': '1722591938341', 'pendingCloseOrdLiabVal': '', 'pnl': '0', 'pos': '0.3', 'posCcy': '', 'posId': '1616745279948165120', 'posSide': 'long', 'quoteBal': '', 'quoteBorrowed': '', 'quoteInterest': '', 'realizedPnl': '0.0731265667023081', 'spotInUseAmt': '', 'spotInUseCcy': '', 'thetaBS': '', 'thetaPA': '', 'tradeId': '1081619170', 'uTime': '1722585600082', 'upl': '1.669559999999999', 'uplLastPx': '1.66935', 'uplRatio': '0.2828348537595606', 'uplRatioLastPx': '0.2827992783269382', 'usdPx': '', 'vegaBS': '', 'vegaPA': ''}]}
        """
        try:
            logger.debug(f"packet on_position: {packet}")
            datas: dict = packet["data"]
            for data in datas:
                position = models.PositionSchema()
                position = position.from_okx_to_form(data, datatype=self.DATA_TYPE)
                if position and self.on_position_callback:
                    self.on_position_callback(position)
        except Exception:
            et, ev, tb = sys.exc_info()
            self.on_error(et, ev, tb)


"""
work without api key & secret
    query_symbols
    query_last_price
    query_prices
    query_history
    OKXDataWebsocket

instType - compulsory
- order history - can use a func to identify type based on symbol
- trade history
- tickers
- symbols
- OKXTradeWebsocket (but can use `ANY`)

instType - optional
- position & pos risk
"""
