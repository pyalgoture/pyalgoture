import hashlib
import hmac
import sys
import time
import urllib
from collections import defaultdict
from collections.abc import Callable
from copy import copy
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock, Timer
from typing import Any

from dateutil.relativedelta import relativedelta

from ..utils import models
from ..utils.client_rest import Response, RestClient
from ..utils.client_ws import WebsocketClient
from ..utils.logger import get_logger
from ..utils.models import TZ_MAP, UTC_TZ

logger = get_logger(default_level="warning")

BINANCE_TESTNET_SPOT_HOST: str = "https://testnet.binance.vision"
BINANCE_TESTNET_FUTURE_REST_HOST: str = "https://testnet.binancefuture.com"
BINANCE_TESTNET_INVERSE_FUTURE_REST_HOST: str = "https://testnet.binancefuture.com"
BINANCE_TESTNET_SPOT_WEBSOCKET_TRADE_HOST: str = "wss://testnet.binance.vision/ws"
BINANCE_TESTNET_SPOT_WEBSOCKET_DATA_HOST: str = "wss://testnet.binance.vision/stream"
BINANCE_TESTNET_FUTURE_WEBSOCKET_TRADE_HOST: str = "wss://stream.binancefuture.com/ws"
BINANCE_TESTNET_FUTURE_WEBSOCKET_DATA_HOST: str = "wss://stream.binancefuture.com/stream"
BINANCE_TESTNET_INVERSE_FUTURE_WEBSOCKET_TRADE_HOST: str = "wss://dstream.binancefuture.com/ws"
BINANCE_TESTNET_INVERSE_FUTURE_WEBSOCKET_DATA_HOST: str = "wss://dstream.binancefuture.com/stream"


BINANCE_SPOT_HOST: str = "https://api.binance.com"
BINANCE_FUTURE_REST_HOST: str = "https://fapi.binance.com"
BINANCE_INVERSE_FUTURE_REST_HOST: str = "https://dapi.binance.com"
BINANCE_OPTION_REST_HOST: str = "https://vapi.binance.com"
BINANCE_SPOT_WEBSOCKET_TRADE_HOST: str = "wss://stream.binance.com:9443/ws"
BINANCE_SPOT_WEBSOCKET_DATA_HOST: str = "wss://stream.binance.com:9443/stream"
BINANCE_FUTURE_WEBSOCKET_TRADE_HOST: str = "wss://fstream.binance.com/ws"
BINANCE_FUTURE_WEBSOCKET_DATA_HOST: str = "wss://fstream.binance.com/stream"
BINANCE_INVERSE_FUTURE_WEBSOCKET_TRADE_HOST: str = "wss://dstream.binance.com/ws"
BINANCE_INVERSE_FUTURE_WEBSOCKET_DATA_HOST: str = "wss://dstream.binance.com/stream"
BINANCE_OPTION_WEBSOCKET_TRADE_HOST: str = "wss://vstream.binance.com/ws"
BINANCE_OPTION_WEBSOCKET_DATA_HOST: str = "wss://vstream.binance.com/stream"

EXCHANGE = "BINANCE"

TIMEDELTA: dict[str, timedelta] = {
    "1m": relativedelta(minutes=1),
    "3m": relativedelta(minutes=3),
    "5m": relativedelta(minutes=5),
    "15m": relativedelta(minutes=15),
    "30m": relativedelta(minutes=30),
    "1h": relativedelta(hours=1),
    "2h": relativedelta(hours=2),
    "4h": relativedelta(hours=4),
    "6h": relativedelta(hours=6),
    "8h": relativedelta(hours=8),
    "12h": relativedelta(hours=12),
    "1d": relativedelta(days=1),
    "3d": relativedelta(days=3),
    "1w": relativedelta(weeks=1),
    "1M": relativedelta(months=1),
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

KBAR_INTERVAL: dict[str, str] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
    "1M": "1M",
}

MAPPING_CHANNEL: dict[str, str] = {
    "depth": "depth5",
    "kline": "kline",
    "ticker": "ticker",
}


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class Security(Enum):
    """Security Auth Type"""

    NONE = 0
    SIGNED = 1
    API_KEY = 2


class BinanceClient(RestClient):
    BROKER_ID = "FP7KJ7RN"
    DATA_TYPE = "spot"

    def __init__(
        self,
        proxy_host: str = "",
        datatype: str = "spot",
        proxy_port: int = 0,
        is_testnet: bool = False,
        debug: bool = False,
        session: bool = False,
        **kwargs,
    ) -> None:
        assert datatype in ["spot", "linear", "inverse", "option"]
        self.DATA_TYPE = datatype
        if self.DATA_TYPE == "spot":
            url_base = BINANCE_TESTNET_SPOT_HOST if is_testnet else BINANCE_SPOT_HOST
        elif self.DATA_TYPE == "linear":
            url_base = BINANCE_TESTNET_FUTURE_REST_HOST if is_testnet else BINANCE_FUTURE_REST_HOST
        elif self.DATA_TYPE == "inverse":
            url_base = BINANCE_TESTNET_INVERSE_FUTURE_REST_HOST if is_testnet else BINANCE_INVERSE_FUTURE_REST_HOST
        elif self.DATA_TYPE == "option":
            url_base = BINANCE_OPTION_REST_HOST if is_testnet else BINANCE_OPTION_REST_HOST

        super().__init__(url_base, proxy_host, proxy_port)

        self.is_testnet = is_testnet
        self.key: str = ""
        self.secret: str = ""
        self.time_offset: int = 0
        self.order_count: int = 0
        self.order_count_lock: Lock = Lock()
        self.connect_time: int = 0
        self.debug = debug
        self.session = session

    def info(self):
        return f"Binance {self.DATA_TYPE} REST API start"

    def new_order_id(self) -> str:
        """new_order_id"""
        # prefix: str = datetime.now().strftime("%Y%m%d-%H%M%S-")
        prefix: str = datetime.now().strftime("%y%m%d-%H%M%S%f")[:-3] + "-"
        with self.order_count_lock:
            self.order_count += 1
            suffix: str = str(self.order_count).rjust(5, "0")

            order_id: str = f"x-{self.BROKER_ID}" + prefix + suffix
            return order_id

    def _remove_none(self, payload):
        return {k: v for k, v in payload.items() if v is not None}

    def sign(self, request):
        """generate binance signature"""
        security: Security = request.data["security"]
        if security == Security.NONE:
            request.data = None
            return request

        if request.params:
            path: str = request.path + "?" + urllib.parse.urlencode(request.params)
        else:
            request.params = dict()
            path: str = request.path

        if security == Security.SIGNED:
            timestamp: int = int(time.time() * 1000)

            if self.time_offset > 0:
                timestamp -= abs(self.time_offset)
            elif self.time_offset < 0:
                timestamp += abs(self.time_offset)

            request.params["timestamp"] = timestamp

            query: str = urllib.parse.urlencode(sorted(request.params.items()))
            signature: bytes = hmac.new(self.secret, query.encode("utf-8"), hashlib.sha256).hexdigest()

            query += f"&signature={signature}"
            path: str = request.path + "?" + query

        request.path = path
        request.params = {}
        request.data = {}

        # add req header
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "X-MBX-APIKEY": self.key,
            "Connection": "close",
        }

        if security in [Security.SIGNED, Security.API_KEY]:
            request.headers = headers
        # logger.debug(request)
        # print(request)
        return request

    def connect(self, key: str, secret: str, **kwargs) -> None:
        """connect exchange server"""
        self.key = key
        self.secret = secret.encode()
        self.connect_time = int(datetime.now(UTC_TZ).strftime("%y%m%d%H%M%S")) * self.order_count
        self.start()
        self.query_time()
        logger.info(self.info())
        return True

    def _common_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        security: Security = Security.API_KEY,
    ):
        data: dict = {"security": security}
        params = {} if params is None else params
        if self.session:
            response = self.add_request(method=method, path=path, params=params, data=data)
        else:
            response = self.request(method=method, path=path, params=params, data=data)

        # print(f'raw response:{response.text}')
        # logger.error(f'raw response:{response.text}')
        return self._process_response(response)

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        security: Security = Security.API_KEY,
    ) -> Any:
        return self._common_request(method="GET", path=path, params=params, security=security)

    def _post(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        security: Security = Security.API_KEY,
    ) -> Any:
        return self._common_request(method="POST", path=path, params=params, security=security)

    def _put(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        security: Security = Security.API_KEY,
    ) -> Any:
        return self._common_request(method="PUT", path=path, params=params, security=security)

    def _delete(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        security: Security = Security.API_KEY,
    ) -> Any:
        return self._common_request(method="DELETE", path=path, params=params, security=security)

    def _process_response(self, response: Response) -> dict:
        try:
            data = response.json()
        except ValueError:
            print("!!binance - _process_response !!", response.text)
            logger.error(response.text)
            raise
        else:
            # print('????',data)

            if not response.ok:
                if not data.get("msg"):
                    print("!!binance - return raw data !!", response.text)
                payload_data = models.CommonDataSchema(
                    status_code=response.status_code,
                    msg=data.get("msg", "No return msg from binance exchange"),
                )
                return models.CommonResponseSchema(
                    success=False,
                    error=True,
                    data=payload_data.dict(),
                    msg=data.get("msg", "No return msg from binance exchange"),
                )
            else:
                return models.CommonResponseSchema(success=True, error=False, data=data, msg="query ok")

    def query_time(self):
        """query_time"""
        if self.DATA_TYPE == "spot":
            url = "/api/v3/time"
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/time"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/time"
        resp = self._get(url, security=Security.NONE)
        # print(resp, '<==== binance spot')
        local_time = int(time.time() * 1000)
        server_time = resp.data.get("serverTime")
        if server_time:
            server_time = int(server_time)
            self.time_offset = local_time - server_time
        else:
            print(f"something went wrong in binance when query time. {resp}")
            self.time_offset = 1

        return server_time

    def _regular_saving_products_payload(self, common_response: models.CommonResponseSchema) -> dict:
        # logger.debug(order_response)

        if isinstance(common_response.data, list):
            payload = list()
            for msg in common_response.data:
                data = models.SavingProductSchema()
                result = data.from_binance_v2_to_form(msg, datatype=self.DATA_TYPE)
                payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.SavingProductSchema()
            payload = payload.from_binance_v2_to_form(common_response.data, datatype=self.DATA_TYPE)
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _regular_symbols_payload(self, common_response: models.CommonResponseSchema) -> dict:
        # logger.debug(order_response)

        if isinstance(common_response.data, list):
            payload = dict()
            # payload = list()
            for msg in common_response.data:
                data = models.SymbolSchema()
                result = data.from_binance_v2_to_form(msg, datatype=self.DATA_TYPE)
                if result:
                    # payload.append(result)
                    payload[result["symbol"]] = result

        elif isinstance(common_response.data, dict):
            payload = models.SymbolSchema()
            payload = payload.from_binance_v2_to_form(common_response.data, datatype=self.DATA_TYPE)
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
                result = data.from_binance_v2_to_form(msg, datatype=self.DATA_TYPE)
                payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.OrderSchema()
            payload = payload.from_binance_v2_to_form(common_response.data, datatype=self.DATA_TYPE)
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
                result = data.from_binance_v2_to_form(msg, datatype=self.DATA_TYPE)
                payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.TradeSchema()
            payload = payload.from_binance_v2_to_form(common_response.data, datatype=self.DATA_TYPE)
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _regular_historical_prices_payload(
        self, common_response: models.CommonResponseSchema, extra: dict = None, tz=None
    ) -> dict:
        def key_data(data):
            key = [
                "startTime",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "endTime",
                "quoteAssetVolume",
                "trades",
                "takerBaseVolume",
                "takerQuoteVolume",
                "ignore",
            ]
            return dict(zip(key, data))

        if isinstance(common_response.data, list):
            payload = list()
            for msg in common_response.data:
                msg = key_data(msg)
                data = models.HistoryOHLCSchema()
                result = data.from_binance_v2_to_form(msg, extra, datatype=self.DATA_TYPE, tz=tz)
                payload.append(result)
        elif isinstance(common_response.data, dict):
            common_response.data = key_data(common_response.data)
            payload = models.HistoryOHLCSchema()
            result = payload.from_binance_v2_to_form(common_response.data, extra, datatype=self.DATA_TYPE, tz=tz)
            payload = result
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _regular_ticker_payload(self, common_response: models.CommonResponseSchema) -> dict:
        if isinstance(common_response.data, list):
            # payload = list()
            for msg in common_response.data:
                data = models.TickerSchema()
                payload = data.from_binance_v2_to_form(msg, datatype=self.DATA_TYPE)
                # payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.TickerSchema()
            payload = payload.from_binance_v2_to_form(common_response.data, datatype=self.DATA_TYPE)
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
                result = data.from_binance_v2_to_form(msg, transfer_type=transfer_type, inout=inout)
                payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.TransferSchema()
            payload = payload.from_binance_v2_to_form(common_response.data, transfer_type=transfer_type, inout=inout)
        else:
            common_response.msg = "payload error in fetch transfer data"
            Exception("payload error in fetch transfer data")

        common_response.data = payload
        return common_response

    def _regular_account_payload(self, common_response: models.CommonResponseSchema, account_type: str) -> dict:
        if isinstance(common_response.data, list):
            # payload = list()
            payload = dict()
            for msg in common_response.data:
                data = models.AccountSchema()
                result = data.from_binance_v2_to_form(msg, datatype=self.DATA_TYPE, account_type=account_type)
                if result:
                    # payload.append(result)
                    payload[result["symbol"]] = result
        elif isinstance(common_response.data, dict):
            payload = models.AccountSchema()
            payload = payload.from_binance_v2_to_form(
                common_response.data, datatype=self.DATA_TYPE, account_type=account_type
            )
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
                result = data.from_binance_v2_to_form(msg, datatype=self.DATA_TYPE)
                if result:
                    payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.PositionSchema()
            payload = payload.from_binance_v2_to_form(common_response.data, datatype=self.DATA_TYPE)
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _aggregate_trades(self, trades):
        ret = defaultdict(dict)
        for i in trades:
            ori_order_id = i["ori_order_id"]
            if ori_order_id in ret:
                ret[ori_order_id]["quantity"] += i["quantity"]
                ret[ori_order_id]["commission"] += i["commission"]
            else:
                ret[ori_order_id] = i
        return list(ret.values())

    def query_permission(self):
        """
        {'ipRestrict': False, 'createTime': 1630503194000, 'enableFutures': True, 'enableSpotAndMarginTrading': False,
        'enableWithdrawals': False, 'enableInternalTransfer': False, 'enableMargin': False,
        'permitsUniversalTransfer': False, 'enableVanillaOptions': True, 'enableReading': True}
        """
        api_info = self._get("/sapi/v1/account/apiRestrictions", security=Security.SIGNED)

        # logger.info(api_info)
        # print(api_info, "======")
        if api_info.success:
            if api_info.data:
                data = models.PermissionSchema()  # .model_construct(api_info.data)
                data.from_binance_v2_to_form(api_info.data)
                api_info.data = data.dict()
            else:
                api_info.msg = "query api info is empty"

        return api_info

    def query_account(self) -> dict:
        """query_account
        spot:
            {
                "makerCommission": 15,
                "takerCommission": 15,
                "buyerCommission": 0,
                "sellerCommission": 0,
                "canTrade": true,
                "canWithdraw": true,
                "canDeposit": true,
                "updateTime": 123456789,
                "accountType": "SPOT",
                "balances": [
                    {
                    "asset": "BTC",
                    "free": "4723846.89208129",
                    "locked": "0.00000000"
                    },
                    {
                    "asset": "LTC",
                    "free": "4763368.68006011",
                    "locked": "0.00000000"
                    }
                ],
                    "permissions": [
                    "SPOT"
                ]
            }
        future:
            [
                {
                    "accountAlias": "SgsR",    // unique account code
                    "asset": "USDT",    // asset name
                    "balance": "122607.35137903", // wallet balance
                    "crossWalletBalance": "23.72469206", // crossed wallet balance
                    "crossUnPnl": "0.00000000"  // unrealized profit of crossed positions
                    "availableBalance": "23.72469206",       // available balance
                    "maxWithdrawAmount": "23.72469206",     // maximum amount for transfer out
                    "marginAvailable": true,    // whether the asset can be used as margin in Multi-Assets mode
                    "updateTime": 1617939110373
                }
            ]
        """

        if self.DATA_TYPE == "spot":
            url = "/api/v3/account"
        elif self.DATA_TYPE == "linear":
            # url = "/fapi/v2/account"
            url = "/fapi/v2/balance"
        elif self.DATA_TYPE == "inverse":
            # url = "/dapi/v1/account"
            url = "/dapi/v1/balance"

        if self.DATA_TYPE == "spot":
            account_type = "SPOT"
        else:
            account_type = "CONTRACT"

        balance = self._get(url, security=Security.SIGNED)
        # print(f"query_account res:{balance}")
        if balance.success:
            if balance.data:
                if self.DATA_TYPE == "spot":
                    balance.data = balance.data["balances"]
                return self._regular_account_payload(balance, account_type=account_type)

            else:
                balance.msg = "query data is empty"

        return balance

    def send_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type=None,
        price=None,
        stop_price=None,
        position_side: str = "",
        reduce_only: bool = False,
        **kwargs,
    ) -> dict:
        """send_order
        {
            "symbol": "BTCUSDT",
            "orderId": 28,
            "orderListId": -1, //Unless OCO, value will be -1
            "clientOrderId": "6gCrw2kRUAF9CvJDGP16IP",
            "transactTime": 1507725176595,
            "price": "0.00000000",
            "origQty": "10.00000000",
            "executedQty": "10.00000000",
            "cummulativeQuoteQty": "10.00000000",
            "status": "FILLED",
            "timeInForce": "GTC",
            "type": "MARKET",
            "side": "SELL"
        }
        """
        if self.DATA_TYPE == "spot":
            url = "/api/v3/order"
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/order"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/order"
        payload = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            # 'price': price,
            "quantity": quantity,
            # 'type': order_type,
            "newClientOrderId": self.new_order_id(),
            "newOrderRespType": "RESULT",  # ACK, RESULT, FULL
        }
        if not order_type:
            if not price:
                order_type = "MARKET"
                if stop_price:
                    order_type = "STOP_LOSS"
            else:
                order_type = "LIMIT"
                if stop_price:
                    order_type = "STOP_LOSS_LIMIT"
            payload["type"] = order_type

        if order_type == "MARKET":
            pass
        elif order_type == "LIMIT":
            payload["price"] = price
            payload["timeInForce"] = "GTC"
        elif order_type in ["STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"]:
            payload["timeInForce"] = "GTC"
            payload["stopPrice"] = stop_price
        elif order_type in ["STOP_LOSS", "TAKE_PROFIT"]:
            payload["stopPrice"] = stop_price

        if self.DATA_TYPE != "spot":
            if position_side:
                position_side = position_side.upper()
                assert position_side in ["LONG", "SHORT", "NET"], "position_side error"
                payload["positionSide"] = "BOTH" if position_side == "NET" else position_side
            if reduce_only:
                payload["reduceOnly"] = "true"

        orders_response = self._post(url, self._remove_none(payload), security=Security.SIGNED)

        if orders_response.success:
            if orders_response.data:
                result = models.SendOrderResultSchema()  # .model_construct(orders_response.data)
                orders_response.data = result.from_binance_v2_to_form(orders_response.data, datatype=self.DATA_TYPE)
            else:
                orders_response.msg = "query sender order is empty"

        return orders_response

    def cancel_order(self, order_id, symbol) -> None:
        """cancel_order
        {
            "symbol": "LTCBTC",
            "orderId": 1,
            "orderListId": -1 //Unless part of an OCO, the value will always be -1.
            "clientOrderId": "myOrder1",
            "price": "0.1",
            "origQty": "1.0",
            "executedQty": "0.0",
            "cummulativeQuoteQty": "0.0",
            "status": "NEW",
            "timeInForce": "GTC",
            "type": "LIMIT",
            "side": "BUY",
            "stopPrice": "0.0",
            "icebergQty": "0.0",
            "time": 1499827319559,
            "updateTime": 1499827319559,
            "isWorking": true,
            "origQuoteOrderQty": "0.000000"
        }

        """
        if self.DATA_TYPE == "spot":
            url = "/api/v3/order"
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/order"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/order"
        payload = {
            "symbol": symbol.upper(),
            # "origClientOrderId": order_id,
            "orderId": order_id,
        }
        return self._delete(url, payload, security=Security.SIGNED)

    def cancel_orders(self, symbol: str) -> dict:
        """cancel_orders
        [
            {
                "symbol": "BTCUSDT",
                "origClientOrderId": "E6APeyTJvkMvLMYMqu1KQ4",
                "orderId": 11,
                "orderListId": -1,
                "clientOrderId": "pXLV6Hz6mprAcVYpVMTGgx",
                "price": "0.089853",
                "origQty": "0.178622",
                "executedQty": "0.000000",
                "cummulativeQuoteQty": "0.000000",
                "status": "CANCELED",
                "timeInForce": "GTC",
                "type": "LIMIT",
                "side": "BUY"
            },
            {
                "symbol": "BTCUSDT",
                "origClientOrderId": "A3EF2HCwxgZPFMrfwbgrhv",
                "orderId": 13,
                "orderListId": -1,
                "clientOrderId": "pXLV6Hz6mprAcVYpVMTGgx",
                "price": "0.090430",
                "origQty": "0.178622",
                "executedQty": "0.000000",
                "cummulativeQuoteQty": "0.000000",
                "status": "CANCELED",
                "timeInForce": "GTC",
                "type": "LIMIT",
                "side": "BUY"
            },
            {
                "orderListId": 1929,
                "contingencyType": "OCO",
                "listStatusType": "ALL_DONE",
                "listOrderStatus": "ALL_DONE",
                "listClientOrderId": "2inzWQdDvZLHbbAmAozX2N",
                "transactionTime": 1585230948299,
                "symbol": "BTCUSDT",
                "orders": [
                {
                    "symbol": "BTCUSDT",
                    "orderId": 20,
                    "clientOrderId": "CwOOIPHSmYywx6jZX77TdL"
                },
                {
                    "symbol": "BTCUSDT",
                    "orderId": 21,
                    "clientOrderId": "461cPg51vQjV3zIMOXNz39"
                }
                ],
                "orderReports": [
                {
                    "symbol": "BTCUSDT",
                    "origClientOrderId": "CwOOIPHSmYywx6jZX77TdL",
                    "orderId": 20,
                    "orderListId": 1929,
                    "clientOrderId": "pXLV6Hz6mprAcVYpVMTGgx",
                    "price": "0.668611",
                    "origQty": "0.690354",
                    "executedQty": "0.000000",
                    "cummulativeQuoteQty": "0.000000",
                    "status": "CANCELED",
                    "timeInForce": "GTC",
                    "type": "STOP_LOSS_LIMIT",
                    "side": "BUY",
                    "stopPrice": "0.378131",
                    "icebergQty": "0.017083"
                },
                {
                    "symbol": "BTCUSDT",
                    "origClientOrderId": "461cPg51vQjV3zIMOXNz39",
                    "orderId": 21,
                    "orderListId": 1929,
                    "clientOrderId": "pXLV6Hz6mprAcVYpVMTGgx",
                    "price": "0.008791",
                    "origQty": "0.690354",
                    "executedQty": "0.000000",
                    "cummulativeQuoteQty": "0.000000",
                    "status": "CANCELED",
                    "timeInForce": "GTC",
                    "type": "LIMIT_MAKER",
                    "side": "BUY",
                    "icebergQty": "0.639962"
                }
                ]
            }
            ]
        """
        if self.DATA_TYPE == "spot":
            url = "/api/v3/openOrders"
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/openOrders"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/openOrders"
        payload = {"symbol": symbol}
        return self._delete(url, self._remove_none(payload), Security.SIGNED)

    def query_order_status(self, symbol, order_id) -> dict:
        """
        query_order_status
        {
            "symbol": "LTCBTC",
            "orderId": 1,
            "orderListId": -1 //Unless part of an OCO, the value will always be -1.
            "clientOrderId": "myOrder1",
            "price": "0.1",
            "origQty": "1.0",
            "executedQty": "0.0",
            "cummulativeQuoteQty": "0.0",
            "status": "NEW",
            "timeInForce": "GTC",
            "type": "LIMIT",
            "side": "BUY",
            "stopPrice": "0.0",
            "icebergQty": "0.0",
            "time": 1499827319559,
            "updateTime": 1499827319559,
            "isWorking": true,
            "origQuoteOrderQty": "0.000000"
        }
        """
        if self.DATA_TYPE == "spot":
            url = "/api/v3/order"
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/order"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/order"
        # payload = {"symbol": symbol.upper(), "origClientOrderId": order_id}
        payload = {"symbol": symbol.upper(), "orderId": order_id}

        order_status = self._get(url, self._remove_none(payload), security=Security.SIGNED)
        if order_status.success:
            if order_status.data:
                return self._regular_order_payload(order_status)
            else:
                order_status.msg = "query order is empty"

        return order_status

    def query_open_orders(self, symbol: str = None) -> dict:
        """query open order
        [
            {
                "symbol": "BTCUSDT",
                "origClientOrderId": "E6APeyTJvkMvLMYMqu1KQ4",
                "orderId": 11,
                "orderListId": -1,
                "clientOrderId": "pXLV6Hz6mprAcVYpVMTGgx",
                "price": "0.089853",
                "origQty": "0.178622",
                "executedQty": "0.000000",
                "cummulativeQuoteQty": "0.000000",
                "status": "CANCELED",
                "timeInForce": "GTC",
                "type": "LIMIT",
                "side": "BUY"
            },
            {
                "symbol": "BTCUSDT",
                "origClientOrderId": "A3EF2HCwxgZPFMrfwbgrhv",
                "orderId": 13,
                "orderListId": -1,
                "clientOrderId": "pXLV6Hz6mprAcVYpVMTGgx",
                "price": "0.090430",
                "origQty": "0.178622",
                "executedQty": "0.000000",
                "cummulativeQuoteQty": "0.000000",
                "status": "CANCELED",
                "timeInForce": "GTC",
                "type": "LIMIT",
                "side": "BUY"
            },
            {
                "orderListId": 1929,
                "contingencyType": "OCO",
                "listStatus": "ALL_DONE",
                "listOrderStatus": "ALL_DONE",
                "listClientOrderId": "2inzWQdDvZLHbbAmAozX2N",
                "transactionTime": 1585230948299,
                "symbol": "BTCUSDT",
                "orders": [
                {
                    "symbol": "BTCUSDT",
                    "orderId": 20,
                    "clientOrderId": "CwOOIPHSmYywx6jZX77TdL"
                },
                {
                    "symbol": "BTCUSDT",
                    "orderId": 21,
                    "clientOrderId": "461cPg51vQjV3zIMOXNz39"
                }
                ],
                "orderReports": [
                {
                    "symbol": "BTCUSDT",
                    "origClientOrderId": "CwOOIPHSmYywx6jZX77TdL",
                    "orderId": 20,
                    "orderListId": 1929,
                    "clientOrderId": "pXLV6Hz6mprAcVYpVMTGgx",
                    "price": "0.668611",
                    "origQty": "0.690354",
                    "executedQty": "0.000000",
                    "cummulativeQuoteQty": "0.000000",
                    "status": "CANCELED",
                    "timeInForce": "GTC",
                    "type": "STOP_LOSS_LIMIT",
                    "side": "BUY",
                    "stopPrice": "0.378131",
                    "icebergQty": "0.017083"
                },
                {
                    "symbol": "BTCUSDT",
                    "origClientOrderId": "461cPg51vQjV3zIMOXNz39",
                    "orderId": 21,
                    "orderListId": 1929,
                    "clientOrderId": "pXLV6Hz6mprAcVYpVMTGgx",
                    "price": "0.008791",
                    "origQty": "0.690354",
                    "executedQty": "0.000000",
                    "cummulativeQuoteQty": "0.000000",
                    "status": "CANCELED",
                    "timeInForce": "GTC",
                    "type": "LIMIT_MAKER",
                    "side": "BUY",
                    "icebergQty": "0.639962"
                }
                ]
            }
        ]

        """
        if self.DATA_TYPE == "spot":
            url = "/api/v3/openOrders"
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/openOrders"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/openOrders"

        # if self.DATA_TYPE != "spot" and not symbol:
        #     return models.CommonResponseSchema(success=False, error=True, data={}, msg=f"`symbol` field must be specified for query binance open orders.")

        payload = {}
        if symbol:
            payload["symbol"] = symbol

        order_status = self._get(url, self._remove_none(payload), security=Security.SIGNED)

        if order_status.success:
            if order_status.data:
                return self._regular_order_payload(order_status)
            else:
                order_status.msg = "query order is empty"

        return order_status

    def query_all_orders(self, symbol: str, limit=None) -> dict:
        """query all orders
        [
            {
                "symbol": "LTCBTC",
                "orderId": 1,
                "orderListId": -1, //Unless OCO, the value will always be -1
                "clientOrderId": "myOrder1",
                "price": "0.1",
                "origQty": "1.0",
                "executedQty": "0.0",
                "cummulativeQuoteQty": "0.0",
                "status": "NEW",
                "timeInForce": "GTC",
                "type": "LIMIT",
                "side": "BUY",
                "stopPrice": "0.0",
                "icebergQty": "0.0",
                "time": 1499827319559,
                "updateTime": 1499827319559,
                "isWorking": true,
                "origQuoteOrderQty": "0.000000"
            }
        ]
        """
        if self.DATA_TYPE == "spot":
            url = "/api/v3/allOrders"
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/allOrders"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/allOrders"
            if not limit:
                limit = 100
        if not limit:
            limit = 1000

        payload = {"symbol": symbol.upper(), "limit": limit}
        all_order = self._get(url, self._remove_none(payload), security=Security.SIGNED)

        if all_order.success:
            if all_order.data:
                return self._regular_order_payload(all_order)
            else:
                all_order.msg = "query order is empty"

        return all_order

    def query_trades(self, symbol: str, start: datetime = None, end: datetime = None, limit=1000) -> dict:
        """query trades
        spot:
            The time between startTime and endTime can't be longer than 24 hours.
        future:
            If startTime and endTime are both not sent, then the last 7 days' data will be returned.
            The time between startTime and endTime cannot be longer than 7 days.
            The parameter fromId cannot be sent with startTime or endTime.
        Response:
        [
            {
                "symbol": "BNBBTC",
                "id": 28457,
                "orderId": 100234,
                "orderListId": -1, //Unless OCO, the value will always be -1
                "price": "4.00000100",
                "qty": "12.00000000",
                "quoteQty": "48.000012",
                "commission": "10.10000000",
                "commissionAsset": "BNB",
                "time": 1499865549590,
                "isBuyer": true,
                "isMaker": false,
                "isBestMatch": true
            }
        ]
        """

        def _generate_window_dates(start_date, end_date, window):
            current_date = start_date

            while current_date <= end_date:
                window_end_date = current_date + timedelta(days=window)  # - timedelta(days=1)

                if window_end_date > end_date:
                    window_end_date = end_date

                yield current_date, window_end_date

                current_date = window_end_date + timedelta(days=1)
            else:
                if window_end_date != end_date:
                    yield window_end_date, end_date

        if self.DATA_TYPE == "spot":
            url = "/api/v3/myTrades"
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/userTrades"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/userTrades"

        payload = {"symbol": symbol.upper(), "limit": limit}
        if start:
            payload["startTime"] = int(datetime.timestamp(start) * 1000)
        if end:
            payload["endTime"] = int(datetime.timestamp(end) * 1000)

        # trades = self._get(url, self._remove_none(payload), security=Security.SIGNED)
        # print(f"[query_trades] trades:{trades}")
        if start and end:
            trade_list = []
            window = 1 if self.DATA_TYPE == "spot" else 6

            for start_dt, end_dt in _generate_window_dates(start, end, window):
                payload["startTime"] = int(datetime.timestamp(start_dt) * 1000)
                payload["endTime"] = int(datetime.timestamp(end_dt) * 1000)
                trades = self._get(url, self._remove_none(payload), security=Security.SIGNED)
                # print(f"[DEBUG] start:{start_dt}; end:{end_dt}; tradelength:{len(trades.data)}")
                if trades.success:
                    trade_list += trades.data
                # else:
                #     print(f"[ERROR] Binance {self.DATA_TYPE} query trades error: {trades}")

            trades.data = trade_list
        else:
            trades = self._get("/api/v3/myTrades", self._remove_none(payload), security=Security.SIGNED)

        if trades.success:
            if trades.data:
                ret_trades = self._regular_trades_payload(trades)
                # aggregate by ori_order_id
                ret_trades.data = self._aggregate_trades(ret_trades["data"])
                return ret_trades
            else:
                trades.msg = "query trades is empty"

        return trades

    def query_position(self, symbol: str = None, force_return: bool = False) -> dict:
        """
        {
            "feeTier": 0,       // account commission tier
            "canTrade": true,   // if can trade
            "canDeposit": true,     // if can transfer in asset
            "canWithdraw": true,    // if can transfer out asset
            "updateTime": 0,        // reserved property, please ignore
            "totalInitialMargin": "0.00000000",
            // total initial margin required with current mark price (useless with isolated positions), only for USDT asset
            "totalMaintMargin": "0.00000000",     // total maintenance margin required, only for USDT asset
            "totalWalletBalance": "23.72469206",     // total wallet balance, only for USDT asset
            "totalUnrealizedProfit": "0.00000000",   // total unrealized profit, only for USDT asset
            "totalMarginBalance": "23.72469206",     // total margin balance, only for USDT asset
            "totalPositionInitialMargin": "0.00000000",
            // initial margin required for positions with current mark price, only for USDT asset
            "totalOpenOrderInitialMargin": "0.00000000",
            // initial margin required for open orders with current mark price, only for USDT asset
            "totalCrossWalletBalance": "23.72469206",      // crossed wallet balance, only for USDT asset
            "totalCrossUnPnl": "0.00000000",      // unrealized profit of crossed positions, only for USDT asset
            "availableBalance": "23.72469206",       // available balance, only for USDT asset
            "maxWithdrawAmount": "23.72469206"     // maximum amount for transfer out, only for USDT asset
            "assets": [
                {
                    "asset": "USDT",            // asset name
                    "walletBalance": "23.72469206",      // wallet balance
                    "unrealizedProfit": "0.00000000",    // unrealized profit
                    "marginBalance": "23.72469206",      // margin balance
                    "maintMargin": "0.00000000",        // maintenance margin required
                    "initialMargin": "0.00000000",    // total initial margin required with current mark price
                    "positionInitialMargin": "0.00000000",    //initial margin required for positions with current mark price
                    "openOrderInitialMargin": "0.00000000",   // initial margin required for open orders with current mark price
                    "crossWalletBalance": "23.72469206",      // crossed wallet balance
                    "crossUnPnl": "0.00000000"       // unrealized profit of crossed positions
                    "availableBalance": "23.72469206",       // available balance
                    "maxWithdrawAmount": "23.72469206",     // maximum amount for transfer out
                    "marginAvailable": true,    // whether the asset can be used as margin in Multi-Assets mode
                    "updateTime": 1625474304765 // last update time
                },
                {
                    "asset": "BUSD",            // asset name
                    "walletBalance": "103.12345678",      // wallet balance
                    "unrealizedProfit": "0.00000000",    // unrealized profit
                    "marginBalance": "103.12345678",      // margin balance
                    "maintMargin": "0.00000000",        // maintenance margin required
                    "initialMargin": "0.00000000",    // total initial margin required with current mark price
                    "positionInitialMargin": "0.00000000",    //initial margin required for positions with current mark price
                    "openOrderInitialMargin": "0.00000000",   // initial margin required for open orders with current mark price
                    "crossWalletBalance": "103.12345678",      // crossed wallet balance
                    "crossUnPnl": "0.00000000"       // unrealized profit of crossed positions
                    "availableBalance": "103.12345678",       // available balance
                    "maxWithdrawAmount": "103.12345678",     // maximum amount for transfer out
                    "marginAvailable": true,    // whether the asset can be used as margin in Multi-Assets mode
                    "updateTime": 1625474304765 // last update time
                }
            ],
            "positions": [  // positions of all symbols in the market are returned
                // only "BOTH" positions will be returned with One-way mode
                // only "LONG" and "SHORT" positions will be returned with Hedge mode
                {
                    "symbol": "BTCUSDT",    // symbol name
                    "initialMargin": "0",   // initial margin required with current mark price
                    "maintMargin": "0",     // maintenance margin required
                    "unrealizedProfit": "0.00000000",  // unrealized profit
                    "positionInitialMargin": "0",      // initial margin required for positions with current mark price
                    "openOrderInitialMargin": "0",     // initial margin required for open orders with current mark price
                    "leverage": "100",      // current initial leverage
                    "isolated": true,       // if the position is isolated
                    "entryPrice": "0.00000",    // average entry price
                    "maxNotional": "250000",    // maximum available notional with current leverage
                    "bidNotional": "0",  // bids notional, ignore
                    "askNotional": "0",  // ask notional, ignore
                    "positionSide": "BOTH",     // position side
                    "positionAmt": "0",         // position amount
                    "updateTime": 0           // last update time
                }
            ]
        }
        """

        if self.DATA_TYPE in ["spot"]:
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg="Spot is not support to query position.",
            )

        if self.DATA_TYPE == "linear":
            # url = "/fapi/v2/account"
            url = "/fapi/v2/positionRisk"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/positionRisk"

        payload = {}
        if symbol:
            payload["symbol"] = symbol

        position = self._get(url, params=self._remove_none(payload), security=Security.SIGNED)
        # print(f"[query_position] position:{position}")
        if position.success:
            if position.data:
                # position.data = position.data['positions']
                return self._regular_position_payload(position)
            else:
                position.msg = "query future position is empty"

        return position

    def query_history(
        self,
        symbol: str,
        interval: str = "1h",
        start: datetime = None,
        end: datetime = None,
        limit: int = 1000,
        tz: str = "hkt",
    ) -> list:
        """query_history
        [
            [
                1499040000000,      // Open time
                "0.01634790",       // Open
                "0.80000000",       // High
                "0.01575800",       // Low
                "0.01577100",       // Close
                "148976.11427815",  // Volume
                1499644799999,      // Close time
                "2434.19055334",    // Quote asset volume
                308,                // Number of trades
                "1756.87402397",    // Taker buy base asset volume
                "28.46694368",      // Taker buy quote asset volume
                "17928899.62484339" // Ignore.
            ]
        ]
        """
        if self.DATA_TYPE == "spot":
            url = "/api/v3/klines"
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/klines"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/klines"

        def iteration_history(symbol, interval, data, payload, retry=3):
            # give data from start time with limit
            retry_counter = 0
            historical_prices_datalist = data.data
            # logger.debug(interval)
            last_timestamp = int(historical_prices_datalist[-1][0])
            end_timestamp = payload["endTime"]
            interval_timestamp = end_timestamp - last_timestamp

            while interval_timestamp > interval:
                last_timestamp = int(historical_prices_datalist[-1][0])
                interval_timestamp = end_timestamp - last_timestamp

                logger.debug(f"last_timestamp {last_timestamp}")
                logger.debug(f"end_timestamp {end_timestamp}")
                logger.debug(f"interval_timestamp {interval_timestamp}")
                # logger.debug(historical_prices_datalist[0:5])

                payload["startTime"] = last_timestamp

                prices = self._get(url, self._remove_none(payload), security=Security.NONE)
                if prices.error:
                    break
                logger.debug(prices.data[0:2])

                historical_prices_datalist.extend(prices.data)
                historical_prices_datalist = [list(item) for item in set(tuple(x) for x in historical_prices_datalist)]
                historical_prices_datalist.sort(key=lambda k: k[0])
                time.sleep(0.1)

                logger.debug(f"payload: {payload}")
                logger.debug(f"retry_counter: {retry_counter}")

                if len(prices.data) != 1000:
                    retry_counter += 1

                if retry_counter >= 3:
                    break

            data.data = historical_prices_datalist
            logger.debug(f"data length: {len(historical_prices_datalist)}")
            return data

        payload: dict = {
            "symbol": symbol.upper(),
            "interval": KBAR_INTERVAL[interval],
            "limit": limit,
        }  # 1m, 1h, 1d

        if start:
            # start = UTC_TZ.localize(start)
            start = start.astimezone(UTC_TZ)  # NOTE: binance start and end timestamp default regard it as utc timestamp
            payload["startTime"] = int(datetime.timestamp(start)) * 1000  # convert to mm
            if not end:
                end = datetime.now()
        if end:
            # end = UTC_TZ.localize(end)
            end = end.astimezone(UTC_TZ)
            payload["endTime"] = int(datetime.timestamp(end)) * 1000  # convert to mm
        historical_prices = self._get(url, self._remove_none(payload), security=Security.NONE)
        # print(f'[query_history] ==> payload: {payload}; historical_prices:{historical_prices};')

        extra = {"interval": interval, "symbol": symbol}
        tz = TZ_MAP.get(tz.lower(), "hkt")
        if historical_prices.success:
            if historical_prices.data:
                # handle query time
                if "startTime" in payload:
                    # ms
                    interval = TIMEDELTA_MAPPING_SEC[interval] * 1000
                    historical_prices = iteration_history(
                        symbol=symbol,
                        interval=interval,
                        data=historical_prices,
                        payload=payload,
                    )
                return self._regular_historical_prices_payload(historical_prices, extra, tz=tz)
            else:
                historical_prices.msg = "query historical prices is empty"

        return historical_prices

    def query_last_price(self, symbol: str, interval: str = "1m", limit: int = 1) -> dict:
        if self.DATA_TYPE == "spot":
            url = "/api/v3/ticker/klines"
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/klines"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/klines"

        payload = {
            "symbol": symbol.upper(),
            "interval": KBAR_INTERVAL[interval],
            "limit": limit,
        }  # 1m, 1h, 1d
        last_historical_prices = self._get(url, self._remove_none(payload), security=Security.NONE)
        extra = {"interval": interval, "symbol": symbol}

        if last_historical_prices.success:
            if last_historical_prices.data:
                last_data = self._regular_historical_prices_payload(last_historical_prices, extra)
                if isinstance(last_data.data, list):
                    last_data.data = last_data.data[0]
                return last_data
            else:
                last_historical_prices.msg = "query latest historical prices is empty"

        return last_historical_prices

    def query_symbols(self, symbol: str = None) -> dict:
        """query_symbols"""
        if self.DATA_TYPE == "spot":
            url = "/api/v3/exchangeInfo"
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/exchangeInfo"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/exchangeInfo"

        payload = {}
        if symbol:
            payload = {"symbol": symbol.upper()}

        symbols = self._get(url, self._remove_none(payload), security=Security.NONE)
        # print(f"[query_symbols] symbols:{symbols}")
        if symbols.success:
            if symbols.data:
                symbols.data = symbols.data["symbols"]
                data = self._regular_symbols_payload(symbols)
                if symbol:
                    if self.DATA_TYPE != "spot":
                        data.data = data.data[symbol.upper()] if symbol.upper() in data.data else {}
                        if not data.data:
                            symbols.msg = "query symbol is empty"
                    else:
                        symbol_data = list(data.data.keys())
                        if symbol_data:
                            data.data = data.data[symbol_data[0]]
                        else:
                            symbols.msg = "query symbol is empty"
                return data

            else:
                symbols.msg = "query symbol is empty"

        return symbols

    def query_prices(self, symbol: str = None) -> dict:
        """query_prices
        Response:

        {
            "symbol": "LTCBTC",
            "price": "4.00000200"
        }
        OR

        [
            {
                "symbol": "LTCBTC",
                "price": "4.00000200"
            },
            {
                "symbol": "ETHBTC",
                "price": "0.07946600"
            }
        ]
        """

        if self.DATA_TYPE == "spot":
            # url = "/api/v3/ticker/price"
            url = "/api/v3/ticker/24hr"
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/ticker/24hr"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/ticker/24hr"

        payload = {}
        if symbol:
            payload["symbol"] = symbol.upper()

        spots = self._get(url, payload, security=Security.NONE)

        logger.debug(spots)
        if spots.success:
            if spots.data:
                return self._regular_ticker_payload(spots)
            else:
                spots.msg = "query spots is empty"

        return spots

    def set_position_mode(
        self,
        mode: str,
    ) -> dict:
        """ """
        if mode not in ["oneway", "hedge"]:
            # raise ValueError("Invalid position_mode")
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg="Please specify the correct position mode.",
            )
        if self.DATA_TYPE == "spot":
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg="In normal account, only inverse & linear support to switch margin mode.",
            )
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/positionSide/dual"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/positionSide/dual"

        ### NOTE: since the trade data cannot identify the position side, so will disable the hedge mode from now on (2025-03-14)
        if mode in ["hedge"]:
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg="Since the trade data cannot identify the position side, so will disable the hedge mode from now on (2025-03-14).",
            )
        payload = {
            "dualSidePosition": "false" if mode == "oneway" else "true",
        }

        mode_payload = self._post(url, self._remove_none(payload), security=Security.SIGNED)
        # print(f"[bybit === set_position_mode]:{mode_payload}")
        if mode_payload.success:
            mode_payload.data = models.PositionModeSchema(position_mode=mode).dict()
            return mode_payload

        return mode_payload

    def set_leverage(
        self,
        leverage: float,
        symbol: str,
    ) -> dict:
        """ """
        if self.DATA_TYPE == "spot":
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg="In normal account, only inverse & linear support to switch margin mode.",
            )
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/leverage"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/leverage"

        payload = {
            "symbol": symbol.upper(),
            "leverage": int(leverage),
        }

        leverage_payload = self._post(url, self._remove_none(payload), security=Security.SIGNED)
        # print(leverage_payload,'!!!!set-leverage')
        if leverage_payload.success:
            if leverage_payload.data:
                leverage_payload.data = models.LeverageSchema(symbol=symbol, leverage=leverage).dict()
                return leverage_payload
        return leverage_payload

    def set_margin_mode(self, is_isolated: bool, symbol: str) -> dict:
        """ """
        if self.DATA_TYPE == "spot":
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg="In normal account, only inverse & linear support to switch margin mode.",
            )
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/marginType"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/marginType"

        payload = {
            "symbol": symbol.upper(),
            "marginType": "ISOLATED" if is_isolated else "CROSSED",
        }

        margin_payload = self._post(url, self._remove_none(payload), security=Security.SIGNED)
        # print(margin_payload,'!!!!switch-isolated')

        if margin_payload.success:
            if margin_payload.data:
                margin_payload.data = models.MarginModeSchema(
                    symbol=symbol, margin_mode="isolated" if is_isolated else "crossed"
                ).dict()
                return margin_payload

        return margin_payload

    def query_liquidswap_account(self) -> dict:
        """query_saving_products
        Response:
        [
        {
            "poolId":45,
            "poolName":"AAVE/BNB",
            "updateTime":1650556521000,
            "liquidity":{
                "AAVE":"6214.87000361",
                "BNB":"2814.18265965"
            },
            "share":{
                "shareAmount":"1.46623158",
                "sharePercentage":"0.00035305",
                "asset":{
                    "BNB":"0.99356287",
                    "AAVE":"2.19419449"
                }
            }
        },
        {
            "poolId":32,
            "poolName":"BNB/USDT",
            "updateTime":1650556544000,
            "liquidity":{
                "BNB":"6591.24471935",
                "USDT":"2773735.49878665"
            },
            "share":{
                "shareAmount":"47.4013724",
                "sharePercentage":"0.00036373",
                "asset":{
                    "BNB":"2.39747584",
                    "USDT":"1008.90865738"
                }
            }
        },
        {
            "poolId":17,
            "poolName":"BNB/BUSD",
            "updateTime":1650556562000,
            "liquidity":{
                "BNB":"27950.43894738",
                "BUSD":"11761457.46203985"
            },
            "share":{
                "shareAmount":"21.56837072",
                "sharePercentage":"0.00003961",
                "asset":{
                    "BNB":"1.10735983",
                    "BUSD":"465.97356371"
                }
            }
        },
        {
            "poolId":36,
            "poolName":"DOT/USDT",
            "updateTime":1650556502000,
            "liquidity":{
                "DOT":"82803.06520713",
                "USDT":"1599337.65055312"
            },
            "share":{
                "shareAmount":"57.34280437",
                "sharePercentage":"0.00016407",
                "asset":{
                    "USDT":"262.40748796",
                    "DOT":"13.58571426"
                }
            }
        },
        {
            "poolId":31,
            "poolName":"SUSHI/BTC",
            "updateTime":1650556430000,
            "liquidity":{
                "BTC":"21.26471817",
                "SUSHI":"253540.82562287"
            },
            "share":{
                "shareAmount":"0.25888281",
                "sharePercentage":"0.00011388",
                "asset":{
                    "SUSHI":"28.87396577",
                    "BTC":"0.00242168"
                }
            }
        },
        {
            "poolId":13,
            "poolName":"BTC/ETH",
            "updateTime":1650556530000,
            "liquidity":{
                "BTC":"1357.25881619",
                "ETH":"18388.46524205"
            },
            "share":{
                "shareAmount":"0.26277889",
                "sharePercentage":"0.00005463",
                "asset":{
                    "ETH":"1.00473936",
                    "BTC":"0.07416015"
                }
            }
        }
        ]
        """
        # saving_account = self._get('/sapi/v1/bswap/poolConfigure', security=Security.SIGNED)
        liquidswap_account = self._get("/sapi/v1/bswap/liquidity", security=Security.SIGNED)
        liquidswap_account.data = [i for i in liquidswap_account.data if i["share"]["shareAmount"] != "0"]

        # account = defaultdict(dict)
        # if liquidswap_account.success:
        #     if liquidswap_account.data:
        #         datatype = 'LIQUID_SWAP'
        #         for account_data in liquidswap_account.data:
        #             share_amount = float(account_data['share']['shareAmount'])
        #             if not share_amount:
        #                 continue

        #             free = 0
        #             locked = float(account_data["principal"])
        #             apy = float(account_data["interestRate"])
        #             if free != 0 or locked != 0:
        #                 key = account_data["asset"]
        #                 account[key]["symbol"] = key
        #                 account[key]["available"] = free
        #                 account[key]["frozen"] = locked
        #                 account[key]["balance"] = free + locked
        #                 account[key]["exchange"] = EXCHANGE
        #                 account[key]["asset_type"] = datatype
        #                 account[key]["apy"] = apy

        #         liquidswap_account.data = account

        #     else:
        #         liquidswap_account.msg = "query liquidswap account is empty"

        return liquidswap_account

    def query_saving_products(self, symbol: str = None, limit=100) -> dict:
        """query_saving_products
        Response:
        [
            {
                "asset": "BTC",
                "avgAnnualInterestRate": "0.00250025",
                "canPurchase": true,
                "canRedeem": true,
                "dailyInterestPerThousand": "0.00685000",
                "featured": true,
                "minPurchaseAmount": "0.01000000",
                "productId": "BTC001",
                "purchasedAmount": "16.32467016",
                "status": "PURCHASING",
                "upLimit": "200.00000000",
                "upLimitPerUser": "5.00000000"
            },
            {
                "asset": "BUSD",
                "avgAnnualInterestRate": "0.01228590",
                "canPurchase": true,
                "canRedeem": true,
                "dailyInterestPerThousand": "0.03836000",
                "featured": true,
                "minPurchaseAmount": "0.10000000",
                "productId": "BUSD001",
                "purchasedAmount": "10.38932339",
                "status": "PURCHASING",
                "upLimit": "100000.00000000",
                "upLimitPerUser": "50000.00000000"
            }
        ]
        """
        payload = {"size": limit}
        saving_products = self._get(
            "/sapi/v1/lending/daily/product/list",
            self._remove_none(payload),
            security=Security.SIGNED,
        )

        if saving_products.success:
            if saving_products.data:
                data = self._regular_saving_products_payload(saving_products)
                if symbol:
                    symbol_data = [x for x in data.data if x["symbol"].lower() == symbol.lower()]
                    if symbol_data:
                        data.data = symbol_data
                    # symbol_data = list(data.data.keys())
                    # if symbol_data:
                    #     data.data = data.data[symbol_data[0]]
                    else:
                        saving_products.msg = "query saving product is empty"
                return data

            else:
                saving_products.msg = "query saving products is empty"

        return saving_products

    def query_saving_account(self) -> dict:
        """query_saving_account
        Response:
        {
        "totalAmountInBTC":"2.07153753",
        "totalAmountInUSDT":"88301.64894031",
        "totalFixedAmountInBTC":"0",
        "totalFixedAmountInUSDT":"0",
        "totalFlexibleInBTC":"2.07153753",
        "totalFlexibleInUSDT":"88301.64894031",
        "positionAmountVos":[
            {
                "asset":"BNB",
                "amount":"81.94558092",
                "amountInBTC":"0.81396546",
                "amountInUSDT":"34695.75896153"
            },
            {
                "asset":"BTC",
                "amount":"0.69158321",
                "amountInBTC":"0.69158321",
                "amountInUSDT":"29480.54627426"
            },
            {
                "asset":"ETH",
                "amount":"6.85452685",
                "amountInBTC":"0.50722813",
                "amountInUSDT":"21620.41149973"
            },
            {
                "asset":"LINK",
                "amount":"54.28702119",
                "amountInBTC":"0.01809929",
                "amountInUSDT":"771.41857111"
            }]
        }
        """
        # saving_account = self._get('/sapi/v1/lending/union/account', security=Security.SIGNED)
        # account = defaultdict(dict)
        # if saving_account.success:
        #     if saving_account.data:
        #         datatype = 'EARN'
        #         for account_data in saving_account.data["positionAmountVos"]:
        #             free = float(account_data["amount"])
        #             locked = float(account_data["locked"])
        #             if free != 0 or locked != 0:
        #                 key = account_data["asset"]
        #                 account[key]["symbol"] = key
        #                 account[key]["available"] = free
        #                 account[key]["frozen"] = locked
        #                 account[key]["balance"] = free + locked
        #                 account[key]["exchange"] = EXCHANGE
        #                 account[key]["asset_type"] = datatype
        #         saving_account.data = account
        #         return saving_account
        #     else:
        #         saving_account.msg = 'query saving account is empty'

        # return saving_account
        if self.is_testnet:
            return models.CommonResponseSchema(success=True, error=False, data={}, msg="")

        fixed_saving_account = self._get("/sapi/v1/lending/project/position/list", security=Security.SIGNED)  # Fixed
        flexible_saving_account = self._get(
            "/sapi/v1/lending/daily/token/position", security=Security.SIGNED
        )  # Flexible

        # print(f'!!!flexible_saving_account:{flexible_saving_account}!!!!!flexible_saving_account:{flexible_saving_account}!!')
        account = defaultdict(dict)
        if fixed_saving_account.success:
            if fixed_saving_account.data:
                datatype = "FIXED_SAVING"
                for account_data in fixed_saving_account.data:
                    free = 0
                    locked = float(account_data["principal"])
                    apy = float(account_data["interestRate"])
                    if free != 0 or locked != 0:
                        key = account_data["asset"]
                        account[key]["symbol"] = key
                        account[key]["available"] = free
                        account[key]["frozen"] = locked
                        account[key]["balance"] = free + locked
                        account[key]["exchange"] = EXCHANGE
                        account[key]["asset_type"] = datatype
                        account[key]["apy"] = apy

                fixed_saving_account.data = account
            else:
                fixed_saving_account.msg = "query saving account is empty"
        # print(flexible_saving_account.data,'!!!flexible_saving_account!!')
        if flexible_saving_account.success:
            if flexible_saving_account.data:
                datatype = "FLEXIBLE_SAVING"
                for account_data in flexible_saving_account.data:
                    # print(f">>> account_data: {account_data}")
                    # {'asset': 'AAVE', 'productId': 'AAVE001', 'productName': 'AAVE', 'annualInterestRate': '0', 'totalAmount': '0.90934045', 'lockedAmount': '0', 'freeAmount': '0.90934045', 'freezeAmount': '0', 'totalInterest': '0.00566972', 'canRedeem': True, 'redeemingAmount': '0'}

                    free = float(account_data["totalAmount"])
                    locked = 0
                    apy = float(account_data.get("avgAnnualInterestRate", account_data["annualInterestRate"]))
                    if free != 0 or locked != 0:
                        key = account_data["asset"]
                        account[key]["symbol"] = key
                        account[key]["available"] = free
                        account[key]["frozen"] = locked
                        account[key]["balance"] = free + locked
                        account[key]["exchange"] = EXCHANGE
                        account[key]["asset_type"] = datatype
                        account[key]["apy"] = apy
                fixed_saving_account.data = account

        if not fixed_saving_account.data:
            fixed_saving_account.data = {}

        return fixed_saving_account

    def send_order_saving_products(self, symbol, quantity) -> dict:
        """send_order_saving_products
        Response:

        {
            "purchaseId": 40607
        }
        """
        products = self.query_saving_products(symbol=symbol)
        logger.debug(products.data)
        if products.error:
            Exception("query saving products fail")
            return products

        payload = {"productId": products.data[0]["product_id"], "amount": quantity}
        return self._post(
            "/sapi/v1/lending/daily/purchase",
            self._remove_none(payload),
            security=Security.SIGNED,
        )

    def cancel_order_saving_products(self, symbol, quantity) -> None:
        """cancel_order_saving_products
        Response:

        {}
        """
        products = self.query_saving_products(symbol=symbol)
        if products.error:
            Exception("query saving products fail")
            return products

        payload = {
            "productId": products.data[0]["product_id"],
            "amount": quantity,
            "type": "NORMAL",
        }
        return self._post(
            "/sapi/v1/lending/daily/redeem",
            self._remove_none(payload),
            security=Security.SIGNED,
        )

    def query_transfer(self, start: datetime = None, end: datetime = None) -> dict:
        """ """
        if self.is_testnet:
            return models.CommonResponseSchema(success=True, error=False, data=[], msg="")

        payload = {}
        # ms - epoch time
        ts_limit = 24 * 60 * 60 * 90 * 1000
        if start:
            start = UTC_TZ.localize(start)  # NOTE: bybit start and end timestamp default regard it as utc timestamp
            payload["startTime"] = int(datetime.timestamp(start) * 1000)
        if end:
            end = UTC_TZ.localize(end)
            payload["endTime"] = int(datetime.timestamp(end) * 1000)
        if payload:
            if not payload.get("startTime"):
                payload["start_time"] = payload["end_time"] - ts_limit
            if not payload.get("endTime"):
                payload["end_time"] = payload["start_time"] + ts_limit
        # print(f"query_transfer payload:{payload}")

        raw_end_time = payload.get("endTime")
        deposit_list = []
        withdraw_list = []
        while True:
            if payload:
                if raw_end_time - payload["startTime"] > ts_limit:  # 90day
                    payload["endTime"] = payload["startTime"] + ts_limit
                else:
                    payload["endTime"] = raw_end_time

            deposit = self._get(
                "/sapi/v1/capital/deposit/hisrec",
                params=payload,
                security=Security.SIGNED,
            )
            # print(deposit,'..deposit...')
            if deposit.success:  #  and deposit.data
                deposit_list += deposit.data
            else:
                return deposit

            withdraw = self._get(
                "/sapi/v1/capital/withdraw/history",
                params=payload,
                security=Security.SIGNED,
            )
            # print(withdraw,'...withdraw..')
            if withdraw.success:  #  and withdraw.data
                withdraw_list += withdraw.data
            else:
                return withdraw
            # print(f'payload:{payload}....')
            if not payload or (raw_end_time - payload["startTime"] <= ts_limit):  # 90day
                break
            else:
                payload["startTime"] = payload["endTime"]

        # print(f'deposit_list:{deposit_list}; len:{len(deposit_list)}')
        deposit.data = deposit_list
        deposit_ret = self._regular_transfer_payload(deposit, transfer_type="external", inout="in")
        # print(f'withdraw_list:{withdraw_list}; len:{len(withdraw_list)}')
        withdraw.data = withdraw_list
        withdraw_ret = self._regular_transfer_payload(withdraw, transfer_type="external", inout="out")

        withdraw_ret.data = withdraw_ret.data + deposit_ret.data

        return withdraw_ret

    # User Stream Endpoints

    def stream_get_listen_key(self):
        """Start a new user data stream and return the listen key
        If a stream already exists it should return the same key.
        If the stream becomes invalid a new key is returned.

        https://binance-docs.github.io/apidocs/spot/en/#listen-key-spot

        response
        {
            "listenKey": "pqia91ma19a5s61cv6a81va65sdf19v8a65a1a5s61cv6a81va65sdf19v8a65a1"
        }
        """
        if self.DATA_TYPE == "spot":
            url = "/api/v3/userDataStream"
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/listenKey"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/listenKey"

        if self.DATA_TYPE == "spot":
            return self._post(url)
        else:
            return self._post(url, security=Security.SIGNED)

    def stream_keepalive(self, listenKey):
        """PING a user data stream to prevent a time out.

        https://binance-docs.github.io/apidocs/spot/en/#listen-key-spot

        response   {}
        """
        if self.DATA_TYPE == "spot":
            url = "/api/v3/userDataStream"
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/listenKey"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/listenKey"

        params = {"listenKey": listenKey}
        if self.DATA_TYPE == "spot":
            return self._post(
                url,
                params,
            )
        else:
            return self._put(url, params, security=Security.SIGNED)

    def stream_close(self, listenKey):
        """Close out a user data stream.
        https://binance-docs.github.io/apidocs/spot/en/#listen-key-spot
        response  {}

        """
        if self.DATA_TYPE == "spot":
            url = "/api/v3/userDataStream"
        elif self.DATA_TYPE == "linear":
            url = "/fapi/v1/listenKey"
        elif self.DATA_TYPE == "inverse":
            url = "/dapi/v1/listenKey"

        params = {"listenKey": listenKey}
        if self.DATA_TYPE == "spot":
            return self._post(
                url,
                params,
            )
        else:
            return self._delete(url, params, security=Security.SIGNED)

    # Cross-margin
    def margin_stream_get_listen_key(self):
        """Start a new cross-margin data stream and return the listen key
        If a stream already exists it should return the same key.
        If the stream becomes invalid a new key is returned.

        Can be used to keep the stream alive.

        https://binance-docs.github.io/apidocs/spot/en/#listen-key-margin

        :returns: API response

        .. code-block:: python

            {
                "listenKey": "pqia91ma19a5s61cv6a81va65sdf19v8a65a1a5s61cv6a81va65sdf19v8a65a1"
            }

        :raises: BinanceRequestException, BinanceAPIException

        """
        return self._post("/sapi/v1/userDataStream")

    def margin_stream_keepalive(self, listenKey):
        """PING a cross-margin data stream to prevent a time out.

        https://binance-docs.github.io/apidocs/spot/en/#listen-key-margin

        :param listenKey: required
        :type listenKey: str

        :returns: API response

        .. code-block:: python

            {}

        :raises: BinanceRequestException, BinanceAPIException

        """
        payload = {"listenKey": listenKey}
        return self._post("/sapi/v1/userDataStream", payload)

    def margin_stream_close(self, listenKey):
        """Close out a cross-margin data stream.

        https://binance-docs.github.io/apidocs/spot/en/#listen-key-margin

        :param listenKey: required
        :type listenKey: str

        :returns: API response

        .. code-block:: python

            {}

        :raises: BinanceRequestException, BinanceAPIException

        """
        params = {"listenKey": listenKey}
        return self._delete("/sapi/v1/userDataStream", params)

    # Isolated margin
    def isolated_margin_stream_get_listen_key(self, symbol):
        """Start a new isolated margin data stream and return the listen key
        If a stream already exists it should return the same key.
        If the stream becomes invalid a new key is returned.

        Can be used to keep the stream alive.

        https://binance-docs.github.io/apidocs/spot/en/#listen-key-isolated-margin

        :param symbol: required - symbol for the isolated margin account
        :type symbol: str

        :returns: API response

        .. code-block:: python

            {
                "listenKey":  "T3ee22BIYuWqmvne0HNq2A2WsFlEtLhvWCtItw6ffhhdmjifQ2tRbuKkTHhr"
            }

        :raises: BinanceRequestException, BinanceAPIException

        """
        params = {"symbol": symbol}
        return self._post("/sapi/v1/userDataStream/isolated", params)

    def isolated_margin_stream_keepalive(self, symbol, listenKey):
        """PING an isolated margin data stream to prevent a time out.

        https://binance-docs.github.io/apidocs/spot/en/#listen-key-isolated-margin

        :param symbol: required - symbol for the isolated margin account
        :type symbol: str
        :param listenKey: required
        :type listenKey: str

        :returns: API response

        .. code-block:: python

            {}

        :raises: BinanceRequestException, BinanceAPIException

        """
        params = {"symbol": symbol.upper(), "listenKey": listenKey}
        return self._put("/sapi/v1/userDataStream/isolated", params)

    def isolated_margin_stream_close(self, symbol, listenKey):
        """Close out an isolated margin data stream.

        https://binance-docs.github.io/apidocs/spot/en/#listen-key-isolated-margin

        :param symbol: required - symbol for the isolated margin account
        :type symbol: str
        :param listenKey: required
        :type listenKey: str

        :returns: API response

        .. code-block:: python

            {}

        :raises: BinanceRequestException, BinanceAPIException

        """
        params = {"symbol": symbol.upper(), "listenKey": listenKey}
        return self._delete("/sapi/v1/userDataStream/isolated", params)


class BinanceDataWebsocket(WebsocketClient):
    DATA_TYPE = "spot"

    def __init__(
        self,
        proxy_host: str = "",
        proxy_port: int = 0,
        ping_interval: int = 60,
        is_testnet: bool = False,
        datatype: str = "spot",
        **kwargs,
    ) -> None:
        super().__init__()

        assert datatype in ["spot", "linear", "inverse", "option"]
        self.DATA_TYPE = datatype

        self.is_testnet: bool = is_testnet
        self.proxy_host: str = proxy_host
        self.proxy_port: int = proxy_port
        self.ping_interval: int = ping_interval

        self.ticks: dict[str, dict] = {}
        self.reqid: int = 0

        self.on_tick = None
        self.on_ticker_callback = None
        self.on_depth_callback = None
        self.on_kline_callback = None
        self.on_connected_callback = None
        self.on_disconnected_callback = None
        self.on_error_callback = None

    def info(self):
        return f"Binance {self.DATA_TYPE} Data Websocket start"

    # def custom_ping(self):
    #     # return json.dumps({"op": "ping"})
    #     pass

    def connect(self):
        """
        Give api key and secret for Authorization.
        """
        if self.DATA_TYPE == "spot":
            host = BINANCE_TESTNET_SPOT_WEBSOCKET_DATA_HOST if self.is_testnet else BINANCE_SPOT_WEBSOCKET_DATA_HOST
        elif self.DATA_TYPE == "linear":
            host = BINANCE_TESTNET_FUTURE_WEBSOCKET_DATA_HOST if self.is_testnet else BINANCE_FUTURE_WEBSOCKET_DATA_HOST
        elif self.DATA_TYPE == "inverse":
            host = (
                BINANCE_TESTNET_INVERSE_FUTURE_WEBSOCKET_DATA_HOST
                if self.is_testnet
                else BINANCE_INVERSE_FUTURE_WEBSOCKET_DATA_HOST
            )
        elif self.DATA_TYPE == "option":
            host = BINANCE_OPTION_WEBSOCKET_DATA_HOST

        self.init(
            host=host,
            proxy_host=self.proxy_host,
            proxy_port=self.proxy_port,
            ping_interval=self.ping_interval,
        )
        self.start()

        if self.waitfor_connection():
            logger.info(self.info())
            return True
        else:
            logger.info(self.info() + " failed")
            return False

    def on_connected(self) -> None:
        """"""
        if self.waitfor_connection():
            print(f"Binance {self.DATA_TYPE} Market data websocket connect success")
        else:
            print(f"Binance {self.DATA_TYPE} Market data websocket connect fail")

        print(f"===> [DEBUG] Binance {self.DATA_TYPE} Market data on_connected - self.ticks: {self.ticks}")

        # resubscribe
        if self.ticks:
            ### not working
            # print(self.ticks,len(self.ticks),'!!!!!!!!')
            # for symbol, detail in self.ticks.items():
            #     req: dict = self.sub_stream(symbol=symbol, channel=detail['channel'], interval=detail['interval'])
            #     logger.debug(req)
            #     print(f"!!!!! rea: {req}")
            #     self.send_packet(req)

            channels = []
            for key, detail in self.ticks.items():
                req: dict = self.sub_stream(
                    symbol=detail["symbol"],
                    channel=detail["channel"],
                    interval=detail.get("interval", "1m"),
                )
                channels += req["params"]

            req: dict = {"method": "SUBSCRIBE", "params": channels, "id": self.reqid}
            self.send_packet(req)

        if self.on_connected_callback:
            self.on_connected_callback()

    def on_disconnected(self):
        print(f"Binance {self.DATA_TYPE} market data Websocket disconnected, now try to connect @ {datetime.now()}")
        if self.on_disconnected_callback:
            self.on_disconnected_callback()

    def inject_callback(self, channel: str, callbackfn: Callable):
        if channel == "ticker":
            self.on_ticker_callback = callbackfn
        elif channel == "depth":
            self.on_depth_callback = callbackfn
        elif channel == "kline":
            self.on_kline_callback = callbackfn
        elif channel == "connect":
            self.on_connected_callback = callbackfn
        elif channel == "disconnect":
            self.on_disconnected_callback = callbackfn
        elif channel == "error":
            self.on_error_callback = callbackfn
        else:
            Exception("invalid callback function")

    def sub_stream(self, symbol, channel="depth", interval="1m"):
        """
        implement v1 public topic, due v2 kline don't return any data.
        """
        self.reqid += 1
        sub = {"method": "SUBSCRIBE", "id": self.reqid}

        symbol = symbol.lower()
        if channel == "kline":
            # <symbol>@kline_<interval>
            sub["params"] = [f"{symbol}@{MAPPING_CHANNEL[channel]}_{KBAR_INTERVAL[interval]}"]
        elif channel == "depth":
            # <symbol>@depth<interval> use 5 bidask
            sub["params"] = [f"{symbol}@{MAPPING_CHANNEL[channel]}"]
        elif channel == "ticker":
            sub["params"] = [f"{symbol}@{MAPPING_CHANNEL[channel]}"]

        return sub

    def subscribe(
        self,
        symbols: list | str,
        on_tick: Callable = None,
        channel="depth",
        interval="1m",
        depth=1,
    ) -> None:
        """"""
        if channel not in ["depth", "kline", "ticker"]:
            raise Exception("invalid subscription")

        if on_tick:
            self.on_tick = on_tick
            self.inject_callback(channel=channel, callbackfn=on_tick)

        if isinstance(symbols, str):
            symbols = [symbols]
        elif isinstance(symbols, list):
            pass
        else:
            logger.debug("invalid input symbols.")
            return

        for symbol in symbols:
            if symbol in self.ticks:
                return
            # for websocket - all show lower symbol in binance
            # symbol = symbol.lower()

            # tick data dict
            asset_type = models.binance_asset_type(symbol=symbol, datatype=self.DATA_TYPE)
            tick = {
                "code": symbol,
                "exchange": EXCHANGE,
                "channel": channel,
                "asset_type": asset_type,
                "product": self.DATA_TYPE,
                "symbol": models.normalize_name(symbol, asset_type),
            }
            key = ""
            if channel == "kline":
                key = f"{symbol}|{interval}"
                tick["interval"] = interval
            else:
                key = f"{symbol}|{channel}"

            self.ticks[key] = tick
            req: dict = self.sub_stream(symbol=symbol, channel=channel, interval=interval)
            logger.debug(req)
            self.send_packet(req)

    def on_packet(self, packet: dict) -> None:
        """on_packet"""
        # logger.debug(f"on_packet event {packet}")
        # print(f"on_packet event {packet}")

        stream: str = packet.get("stream", None)
        if not stream:
            print(f"[binance] the other packet without topic: {packet}")
            return

        symbol, channel = stream.split("@")
        symbol = symbol.upper()
        channel = channel.split("_")[0]
        is_fire: bool = False

        if channel == "kline":
            interval = packet["data"]["k"]["i"]
            tick: dict = self.ticks[f"{symbol}|{interval}"]
            data: dict = packet["data"]["k"]
            tick["volume"] = float(data["v"])
            tick["turnover"] = float(data["q"])
            tick["open"] = float(data["o"])
            tick["high"] = float(data["h"])
            tick["low"] = float(data["l"])
            tick["close"] = float(data["c"])
            tick["start"] = models.timestamp_to_datetime(float(data["t"]))  # bar start time
            tick["datetime"] = models.generate_datetime()
            if self.on_kline_callback:
                is_fire = True
                self.on_kline_callback(copy(tick))

        elif channel == "ticker":
            tick: dict = self.ticks[f"{symbol}|ticker"]
            data: dict = packet["data"]
            tick["prev_open_24h"] = float(data["o"])
            tick["prev_high_24h"] = float(data["h"])
            tick["prev_low_24h"] = float(data["l"])
            tick["prev_volume_24h"] = float(data["v"])
            tick["prev_turnover_24h"] = float(data["q"])

            tick["last_price"] = float(data["c"])
            tick["price_change_pct"] = float(data["P"]) / 100
            tick["price_change"] = float(data["p"])
            tick["prev_close_24h"] = tick["last_price"] + tick["price_change"]

            tick["datetime"] = models.generate_datetime()
            if self.on_ticker_callback:
                is_fire = True
                self.on_ticker_callback(copy(tick))
        else:
            tick: dict = self.ticks[f"{symbol}|depth"]
            data: dict = packet["data"]
            bids: list = data["bids" if "bids" in data else "b"]

            for n in range(min(5, len(bids))):
                price, volume = bids[n]

                tick["bid_price_" + str(n + 1)] = float(price)
                tick["bid_volume_" + str(n + 1)] = float(volume)

            asks: list = data["asks" if "asks" in data else "a"]
            for n in range(min(5, len(asks))):
                price, volume = asks[n]
                tick["ask_price_" + str(n + 1)] = float(price)
                tick["ask_volume_" + str(n + 1)] = float(volume)

            tick["datetime"] = models.generate_datetime()
            if self.on_depth_callback:
                is_fire = True
                self.on_depth_callback(copy(tick))

        if not is_fire and self.on_tick and tick:
            self.on_tick(copy(tick))


class BinanceTradeWebsocket(WebsocketClient):
    """
    Binance trade Websocket
    3 type account: spot, margin, isolated_margin

    spot user data ws doesnt need to subscribe any stream
    """

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
        ping_interval: int = 60,
        debug: bool = False,
        # logger = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if datatype:
            assert datatype in ["spot", "linear", "inverse", "option"]
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
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self._logged_in = False
        self.key = None
        self.secret = None
        self.debug = debug

        self.client = BinanceClient(datatype=datatype, is_testnet=is_testnet)
        self.account_type = "spot"
        self.margin_symbol = "BTCUSDT"
        self.listen_key: str = ""
        self.refresh_key_time: float = time.time()
        self.timer_listen_key = RepeatTimer(30 * 60, self._keep_alive)

    def __del__(self):
        if self._logged_in:
            self.timer_listen_key.cancel()
            if self.account_type == "spot":
                self.client.stream_close(self.listen_key)
            elif self.account_type == "margin":
                self.client.margin_stream_close(self.listen_key)
            elif self.account_type == "isolated_margin":
                self.client.isolated_margin_stream_close(self.margin_symbol, self.listen_key)

    def info(self):
        return f"Binance {self.DATA_TYPE} Trade Websocket Start"

    # def custom_ping(self):
    #     # return json.dumps({"op": "ping"})
    #     pass

    def connect(
        self,
        key: str,
        secret: str,
        is_testnet: bool = None,
        account_type="spot",
        symbol="BTCUSDT",
    ):
        """
        Give api key and secret for Authorization.
        """
        self.key = key
        self.secret = secret
        if is_testnet is not None:
            self.is_testnet = is_testnet
        self.account_type = account_type
        self.margin_symbol = symbol

        if self.DATA_TYPE == "spot":
            host = BINANCE_TESTNET_SPOT_WEBSOCKET_TRADE_HOST if self.is_testnet else BINANCE_SPOT_WEBSOCKET_TRADE_HOST
        elif self.DATA_TYPE == "linear":
            host = (
                BINANCE_TESTNET_FUTURE_WEBSOCKET_TRADE_HOST if self.is_testnet else BINANCE_FUTURE_WEBSOCKET_TRADE_HOST
            )
        elif self.DATA_TYPE == "inverse":
            host = (
                BINANCE_TESTNET_INVERSE_FUTURE_WEBSOCKET_DATA_HOST
                if self.is_testnet
                else BINANCE_INVERSE_FUTURE_WEBSOCKET_TRADE_HOST
            )
        elif self.DATA_TYPE == "option":
            host = BINANCE_OPTION_WEBSOCKET_TRADE_HOST

        if not self._logged_in:
            self._login()

        self.init(
            host=f"{host}/{self.listen_key}",
            proxy_host=self.proxy_host,
            proxy_port=self.proxy_port,
            ping_interval=self.ping_interval,
        )
        self.start()

        if self.waitfor_connection():
            logger.info(self.info())
        else:
            logger.info(self.info() + " failed")
            return False

        return True

    def _login(self):
        """
        docs: https://binance-docs.github.io/apidocs/spot/en/#listen-key-spot
        Authorize websocket connection.
        """
        api_key = self.key
        api_secret = self.secret

        if api_key is None or api_secret is None:
            raise PermissionError("Authenticated endpoints require keys.")

        self.client.connect(api_key, api_secret)

        if self.account_type == "spot":
            key_payload = self.client.stream_get_listen_key()
        elif self.account_type == "margin":
            key_payload = self.client.margin_stream_get_listen_key()
        elif self.account_type == "isolated_margin":
            key_payload = self.client.isolated_margin_stream_get_listen_key(self.margin_symbol)

        # print(f'auth res: {key_payload}')

        if key_payload.success:
            self.listen_key = key_payload.data["listenKey"]
            # self.host = f"{self.url_base}/{self.listen_key}"

            # self.init(f"{self.url_base}/{self.listen_key}", self.proxy_host, self.proxy_port)
            # self.start()

            self.timer_listen_key.daemon = True
            self.timer_listen_key.start()
            self._logged_in = True
            # print(f"login with listenKey {self.listen_key}, url: {self.url_base}/{self.listen_key}")
            print(f"login with listenKey {self.listen_key}")
        else:
            print(f"auth res: {key_payload}")
            raise PermissionError("Authenticated endpoints require listenKey for subscription.")

    def _keep_alive(self):
        """
        https://binance-docs.github.io/apidocs/spot/en/#listen-key-spot
        Keepalive a user data stream to prevent a time out.
        User data streams will close after 60 minutes.
        It's recommended to send a ping about every 30 minutes.
        """
        self.refresh_key_time = time.time()
        logger.debug(f"keep_alive: {self.refresh_key_time}")

        if self.account_type == "spot":
            self.client.stream_keepalive(self.listen_key)
        elif self.account_type == "margin":
            self.client.margin_stream_keepalive(self.listen_key)
        elif self.account_type == "isolated_margin":
            self.client.isolated_margin_stream_keepalive(self.margin_symbol, self.listen_key)

    def on_disconnected(self) -> None:
        print(f"Binance {self.DATA_TYPE} user trade data Websocket disconnected, now try to connect @ {datetime.now()}")
        self._logged_in = False
        # self._login()
        if self.on_disconnected_callback:
            self.on_disconnected_callback()

    def on_connected(self) -> None:
        print(f"Binance {self.DATA_TYPE} user trade data Websocket connect success")

    def on_packet(self, packet: dict) -> None:
        """keep alive check on packet is comming"""
        # print(f"Binance spot user data ws, on_packet event {packet}")

        if self.DATA_TYPE == "spot":
            if packet["e"] == "outboundAccountPosition":
                self.on_account_spot(packet)
            elif packet["e"] == "executionReport":
                self.on_order_spot(packet)
        else:
            if packet["e"] == "ACCOUNT_UPDATE":
                self.on_account_contract(packet)
            elif packet["e"] == "ORDER_TRADE_UPDATE":
                self.on_order_contract(packet)

    def on_account_contract(self, packet: dict) -> None:
        """on account balance update"""
        packet = packet["a"]
        # print(f"[on_account_contract] packet >>> {packet}")

        for d in packet["B"]:
            data = {
                "asset": d["a"],
                "balance": d["wb"],
            }
            account = models.AccountSchema()
            account = account.from_binance_v2_to_form(
                data,
                datatype=self.DATA_TYPE,
                account_type="spot" if self.DATA_TYPE == "spot" else "contract",
            )
            if account and account["symbol"] and self.on_account_callback:
                self.on_account_callback(account)

        for d in packet.get("P", []):
            data = {
                "symbol": d["s"],
                "positionAmt": d["pa"],
                "entryPrice": d["ep"],
                # 'leverage': d["wb"],
                # 'isolatedMargin': d["wb"],
                "unRealizedProfit": d["up"],
                "marginType": d["mt"],
                # 'liquidationPrice': d["wb"],
                "positionSide": d["ps"],
            }
            position = models.PositionSchema()
            position = position.from_binance_v2_to_form(
                data,
                datatype=self.DATA_TYPE,
            )
            if position and position["symbol"] and self.on_position_callback:
                self.on_position_callback(position)

    def on_order_contract(self, packet: dict) -> None:
        """on order place/cancel"""
        # filter not support trade type
        packet = packet["o"]
        # print(f"[on_order_contract] packet >>> {packet}")

        if packet["o"] not in ["LIMIT", "MARKET"]:
            return
        # print(f"packet >>> {packet}")
        data = {
            "symbol": packet["s"],
            "time": packet["T"],
            "orderId": packet["i"],
            "price": packet["L"],  #  packet["p"],
            "origQty": packet["q"],
            "status": packet["X"],
            "type": packet["o"],
            "side": packet["S"],
            "clientOrderId": packet["c"],
            "executedQty": packet["z"],
            "positionSide": packet["ps"],
        }
        if packet["X"] != "FILLED":
            order = models.OrderSchema()
            order = order.from_binance_v2_to_form(data, datatype=self.DATA_TYPE)
            if self.on_order_callback:  # and order_list
                self.on_order_callback(order)
            return

        data["time"] = packet["T"]
        data["commission"] = packet["n"]
        data["commissionAsset"] = packet["N"]
        data["isBuyer"] = packet["S"] == "BUY"
        data["qty"] = packet["z"]
        data["price"] = packet["L"]
        data["positionSide"] = packet["ps"]
        trade = models.TradeSchema()
        trade = trade.from_binance_v2_to_form(data, datatype=self.DATA_TYPE)
        if trade and self.on_trade_callback:
            self.on_trade_callback(trade)

    def on_account_spot(self, packet: dict) -> None:
        """on account balance update"""
        for d in packet["B"]:
            data = {
                "asset": d["a"],
                "free": d["f"],
                "locked": d["l"],
            }
            account = models.AccountSchema()
            account = account.from_binance_v2_to_form(
                data,
                datatype=self.DATA_TYPE,
                account_type="spot" if self.DATA_TYPE == "spot" else "contract",
            )
            if account and account["symbol"] and self.on_account_callback:
                self.on_account_callback(account)

    def on_order_spot(self, packet: dict) -> None:
        """on order place/cancel


        packet['x'] - Execution types:

        NEW - The order has been accepted into the engine.
        CANCELED - The order has been canceled by the user.
        REPLACED (currently unused)
        REJECTED - The order has been rejected and was not processed. (This is never pushed into the User Data Stream)
        TRADE - Part of the order or all of the order's quantity has filled.
        EXPIRED - The order was canceled according to the order type's rules (e.g. LIMIT FOK orders with no fill, LIMIT IOC or MARKET orders that partially fill) or by the exchange, (e.g. orders canceled during liquidation, orders canceled during maintenance)
        """
        # filter not support trade type
        if packet["o"] not in ["LIMIT", "MARKET"]:
            return
        # print(f"packet >>> {packet}")
        data = {
            "symbol": packet["s"],
            "time": packet["O"],
            "orderId": packet["i"],
            "price": packet["L"],  #  packet["p"],
            "origQty": packet["q"],
            "status": packet["X"],
            "type": packet["o"],
            "side": packet["S"],
            "clientOrderId": packet["c"] if packet["C"] == "" else packet["C"],
            "executedQty": packet["z"],
        }
        if packet["X"] != "FILLED":
            order = models.OrderSchema()
            order = order.from_binance_v2_to_form(data, datatype=self.DATA_TYPE)
            if self.on_order_callback:  # and order_list
                self.on_order_callback(order)
            return

        data["time"] = packet["T"]
        data["commission"] = packet["n"]
        data["commissionAsset"] = packet["N"]
        data["isBuyer"] = packet["S"] == "BUY"
        data["qty"] = packet["z"]
        data["price"] = packet["L"]
        trade = models.TradeSchema()
        trade = trade.from_binance_v2_to_form(data, datatype=self.DATA_TYPE)
        if trade and self.on_trade_callback:
            self.on_trade_callback(trade)
