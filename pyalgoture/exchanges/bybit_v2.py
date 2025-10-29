import hashlib
import hmac
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from copy import copy
from datetime import datetime, timedelta
from threading import Lock
from typing import Any

from ..utils import models
from ..utils.client_rest import Request, Response, RestClient
from ..utils.client_ws import WebsocketClient
from ..utils.logger import get_logger
from ..utils.models import UTC_TZ
from .bybit_ws import WebSocket

logger = get_logger(default_level="warning")

BYBIT_API_HOST = "https://api.bybit.com"
BYBIT_TESTNET_API_HOST = "https://api-testnet.bybit.com"
BYBIT_DEMO_API_HOST = "https://api-demo.bybit.com"

BYBIT_WS_SPOT_PUBLIC_HOST = "wss://stream.bybit.com/v5/public/spot"
BYBIT_WS_FUTURE_PUBLIC_HOST = "wss://stream.bybit.com/v5/public/linear"
BYBIT_WS_INVERSE_PUBLIC_HOST = "wss://stream.bybit.com/v5/public/inverse"
BYBIT_WS_OPTION_PUBLIC_HOST = "wss://stream.bybit.com/v5/public/option"
BYBIT_WS_PRIVATE_HOST = "wss://stream.bybit.com/v5/private"

BYBIT_TESTNET_WS_SPOT_PUBLIC_HOST = "wss://stream-testnet.bybit.com/v5/public/spot"
BYBIT_TESTNET_WS_FUTURE_PUBLIC_HOST = "wss://stream-testnet.bybit.com/v5/public/linear"
BYBIT_TESTNET_WS_INVERSE_PUBLIC_HOST = "wss://stream-testnet.bybit.com/v5/public/inverse"
BYBIT_TESTNET_WS_OPTION_PUBLIC_HOST = "wss://stream-testnet.bybit.com/v5/public/option"
BYBIT_TESTNET_WS_PRIVATE_HOST = "wss://stream-testnet.bybit.com/v5/private"

BYBIT_DEMO_WS_SPOT_PUBLIC_HOST = "wss://stream-demo.bybit.com/v5/public/spot"
BYBIT_DEMO_WS_FUTURE_PUBLIC_HOST = "wss://stream-demo.bybit.com/v5/public/linear"
BYBIT_DEMO_WS_INVERSE_PUBLIC_HOST = "wss://stream-demo.bybit.com/v5/public/inverse"
BYBIT_DEMO_WS_OPTION_PUBLIC_HOST = "wss://stream-demo.bybit.com/v5/public/option"
BYBIT_DEMO_WS_PRIVATE_HOST = "wss://stream-demo.bybit.com/v5/private"

EXCHANGE = "BYBIT"
INVERSE_SETTLECOINS = ["ADA", "XRP", "BIT", "BTC", "DOT", "ETH", "LTC", "MANA", "EOS"]
LINEAR_SETTLECOINS = ["USDT", "USDC"]

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
    # 1,3,5,15,30,60,120,240,360,720,D,M,W
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
    "1M": "M",
}
MAPPING_CHANNEL: dict[str, str] = {
    ### public
    "depth": "orderbook",
    "kline": "kline",
    "ticker": "tickers",
    ### private
    "trades": "execution",
    "orders": "order",
    "account": "wallet",
    "position": "position",
    "greeks": "greeks",
}


KBAR_INTERVAL_REV: dict[str, str] = {
    "1": "1m",
    "3": "3m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "60": "1h",
    "120": "2h",
    "240": "4h",
    "360": "6h",
    "D": "1d",
    "W": "1w",
    "M": "1M",
}


RETRY_CODES = {10002, 10006, 30034, 30035, 130035, 130150}

"""
"""


def remove_none(payload):
    """
    Remove None value from payload
    """
    return {k: v for k, v in payload.items() if v is not None}


def find_index(source, target, key):
    """
    Find the index in source list of the targeted ID.
    """
    return next(i for i, j in enumerate(source) if j[key] == target[key])


class BybitClient(RestClient):
    """ """

    BROKER_ID = ""
    DATA_TYPE = "spot"

    def __init__(
        self,
        proxy_host: str = "",
        proxy_port: int = 0,
        datatype: str = "spot",
        is_testnet: bool = False,
        is_demo: bool = False,
        debug: bool = False,
        logger=None,
        session: bool = False,
        **kwargs,
    ) -> None:
        # url_base: str = None,
        super().__init__(
            url_base=BYBIT_TESTNET_API_HOST if is_testnet else BYBIT_DEMO_API_HOST if is_demo else BYBIT_API_HOST,
            proxy_host=proxy_host,
            proxy_port=proxy_port,
            logger=logger,
        )

        assert datatype in ["spot", "linear", "inverse", "option"]
        self.DATA_TYPE = datatype
        self.order_count: int = 0
        self.order_count_lock: Lock = Lock()
        self.connect_time: int = 0
        self.key: str = ""
        self.secret: str = ""
        self.is_testnet: bool = is_testnet
        self.is_demo: bool = is_demo
        self.time_offset: int = 0
        self.recv_window: int = 5000
        self.unified = None
        self.debug = debug

        self.session = session

    def set_datatype(self, datatype: str) -> None:
        self.DATA_TYPE = datatype

    def info(self):
        return f"Bybit {self.DATA_TYPE} REST API start"

    def new_order_id(self) -> str:
        """new_order_id"""
        # prefix: str = datetime.now().strftime("%Y%m%d-%H%M%S-")
        prefix: str = models.generate_datetime().strftime("%y%m%d-%H%M%S%f")[:-3] + "-"  #  # datetime.now()
        with self.order_count_lock:
            self.order_count += 1
            suffix: str = str(self.order_count).rjust(5, "0")

            order_id: str = f"x-{self.BROKER_ID}" + prefix + suffix
            return order_id

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if self.session:
            response = self.add_request(method="GET", path=path, params=params)
        else:
            response = self.request(method="GET", path=path, params=params)
        if self.debug:
            print(f"[GET]({path}) - params:{params} | response: {response.text}({response.status_code})")
        # TODO: based on statu code and force retry
        return self._process_response(response)

    def _post(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if self.session:
            response = self.add_request(method="POST", path=path, data=params, headers={"Referer": "AREIX"})
        else:
            response = self.request(method="POST", path=path, data=params, headers={"Referer": "AREIX"})
        if self.debug:
            print(f"[POST]({path}) - params:{params} | response: {response.text}({response.status_code})")
        return self._process_response(response)

    def _delete(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if self.session:
            response = self.add_request(method="DELETE", path=path, data=params)
        else:
            response = self.request(method="DELETE", path=path, data=params)
        if self.debug:
            print(f"[DELETE]({path}) - params:{params} | response: {response.text}({response.status_code})")
        return self._process_response(response)

    def retry(self, response: Response, request: Request):
        path = request.path
        data = response.json()
        if "usdc" in path:
            ret_code = "retCode"
            ret_msg = "retMsg"
        else:
            ret_code = "ret_code"
            ret_msg = "ret_msg"

        error_msg = f"{data.get(ret_msg)} (ErrCode: {data.get(ret_code)})"
        err_delay = self._retry_delay
        if data.get(ret_code) in RETRY_CODES:
            # 10002, recv_window error; add 2.5 seconds and retry.
            if data[ret_code] == 10002:
                error_msg += ". Added 2.5 seconds to recv_window"
                self.recv_window += 2500

            # 10006, ratelimit error; wait until rate_limit_reset_ms and retry.
            elif data[ret_code] == 10006:
                print(f"{error_msg}. Ratelimited on current request. Sleeping, then trying again. Request: {path}")

                # Calculate how long we need to wait.
                limit_reset = data["rate_limit_reset_ms"] / 1000
                reset_str = time.strftime("%X", time.localtime(limit_reset))
                err_delay = int(limit_reset) - int(time.time())
                error_msg = f"Ratelimit will reset at {reset_str}. Sleeping for {err_delay} seconds"
            print(error_msg)
            time.sleep(err_delay)
            return True
        return False

    def connect(self, key: str, secret: str, **kwargs) -> None:
        """connect exchange server"""
        self.key = key
        self.secret = secret
        self.connect_time = int(datetime.now(UTC_TZ).strftime("%y%m%d%H%M%S"))
        # self.query_time()
        self.start()
        if self.is_demo:
            self.unified = True
        else:
            self.query_permission()
        self.logger.debug(f"[DEBUG] >> unified: {self.unified} | datatype:{self.DATA_TYPE} | {self.info()}")
        return True

    @staticmethod
    def prepare_payload(method, parameters):
        """
        Prepares the request payload and validates parameter value types.
        """

        def cast_values():
            string_params = [
                "qty",
                "price",
                "triggerPrice",
                "takeProfit",
                "stopLoss",
            ]
            integer_params = ["positionIdx"]
            for key, value in parameters.items():
                if key in string_params:
                    if not isinstance(value, str):
                        parameters[key] = str(value)
                elif key in integer_params:
                    if not isinstance(value, int):
                        parameters[key] = int(value)

        if method == "GET":
            payload = "&".join([str(k) + "=" + str(v) for k, v in sorted(parameters.items()) if v is not None])
            return payload
        else:
            cast_values()
            return json.dumps(parameters)

    def _auth(self, payload, recv_window, timestamp):
        """
        Generates authentication signature per Bybit API specifications.
        """

        api_key = self.key
        api_secret = self.secret

        if api_key is None or api_secret is None:
            raise PermissionError("Authenticated endpoints require keys.")

        payload = payload if payload else ""
        param_str = str(timestamp) + api_key + str(recv_window) + payload
        # print(f"param_str: {param_str}")
        hash = hmac.new(
            bytes(api_secret, "utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256,
        )

        # print(f"[>>> bybit_v2 - auth] param_str:{param_str}; signature:{hash.hexdigest()} | api_key:{api_key}; api_secret:{api_secret}")

        return hash.hexdigest()

    def sign(self, request, recv_window=5000):
        """generate bybit signature"""
        query = request.params if request.params else {}
        query_data = request.data if request.data else {}
        # query.update(query_data)
        method = request.method
        path = request.path
        if "/market" in path:
            return request

        req_params = None
        if query:
            req_params = self.prepare_payload(method, query)
        if query_data:
            req_params = self.prepare_payload(method, query_data)

        timestamp = str(int(time.time() * 10**3))
        signature = self._auth(
            payload=req_params,
            recv_window=recv_window,
            timestamp=timestamp,
        )
        headers = {
            "Content-Type": "application/json",
            "X-BAPI-API-KEY": self.key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": str(recv_window),
        }
        request.headers = headers

        if method == "GET":
            if req_params:
                request.params = {}
                request.path = path + f"?{req_params}"

        else:
            request.data = req_params

        # print(f"[DEBUG], path:{request.path}; method:{method}; params:{request.params}; data:{request.data}; headers:{request.headers}; signature:{signature}")

        return request

    def _process_response(self, response: Response) -> dict:
        """
        {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
            },
            "retExtInfo": {},
            "time": 1671017382656
        }
        """
        try:
            data = response.json()
            # print(data,'!!!!', response.ok)
        except ValueError:
            print(f"[BYBIT] in _process_response, something went wrong when parsing data. Raw data:{response.text}")
            raise
        else:
            """
            DEBUG:{'retCode': 10024, 'retMsg': 'The product or service you are seeking to access is not available to you due to regulatory restrictions.If you believe you are a permitted customer of this product or service,please reach out to us at support@bybit.com', 'result': {'config': {'endpointExec': 'KYC_PROMPT_TOAST', 'endpointArgs': {}}}, 'retExtInfo': {}, 'time': 1683684352478} | response.ok:True | data["result"]:{'config': {'endpointExec': 'KYC_PROMPT_TOAST', 'endpointArgs': {}}}

            """
            # print(f'DEBUG:{data} | response.ok:{response.ok} | data["result"]:{data["result"]}')
            if not response.ok or (response.ok and data["retCode"] != 0):  # and not data["result"]
                payload_data = models.CommonDataSchema(status_code=response.status_code, msg=dict(data))
                if not data.get("retMsg"):
                    print(f"[BYBIT] request without retMsg msg. Raw data::{response.text}")
                return models.CommonResponseSchema(
                    success=False,
                    error=True,
                    data=payload_data.dict(),
                    msg=data.get("retMsg", "No return msg from bybit exchange."),
                )
            elif response.ok and data["result"] is None and data["retCode"] == 0:
                payload_data = models.CommonDataSchema(status_code=response.status_code, msg=dict(data))
                return models.CommonResponseSchema(
                    success=False,
                    error=True,
                    data=payload_data.dict(),
                    msg="query ok, but result is None",
                )
            else:
                return models.CommonResponseSchema(success=True, error=False, data=data["result"], msg="query ok")

    def _regular_order_payload(self, common_response: models.CommonResponseSchema) -> dict:
        if isinstance(common_response.data, list):
            payload = list()
            for msg in common_response.data:
                # msg['price'] = float(msg['price'])
                # print(f"debug: {msg}")
                data = models.OrderSchema()
                result = data.from_bybit_v2_to_form(msg, datatype=self.DATA_TYPE)
                payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.OrderSchema()
            payload = payload.from_bybit_v2_to_form(common_response.data, datatype=self.DATA_TYPE)
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
                result = data.from_bybit_v2_to_form(msg, datatype=self.DATA_TYPE)
                if result:
                    payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.TradeSchema()
            payload = payload.from_bybit_v2_to_form(common_response.data, datatype=self.DATA_TYPE)
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _regular_symbols_payload(
        self,
        common_response: models.CommonResponseSchema,
    ) -> dict:
        if isinstance(common_response.data, list):
            # payload = list()
            payload = dict()
            for msg in common_response.data:
                data = models.SymbolSchema()
                result = data.from_bybit_v2_to_form(msg, datatype=self.DATA_TYPE)
                if result:
                    # payload.append(result)
                    payload[result["symbol"]] = result
        elif isinstance(common_response.data, dict):
            payload = models.SymbolSchema()
            payload = payload.from_bybit_v2_to_form(common_response.data, datatype=self.DATA_TYPE)
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _regular_historical_prices_payload(
        self, common_response: models.CommonResponseSchema, extra: dict = None
    ) -> dict:
        def key_data(data):
            key = [
                "start",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
            ]
            return dict(zip(key, data))

        if isinstance(common_response.data, list):
            payload = list()
            for msg in common_response.data:
                msg = key_data(msg)
                data = models.HistoryOHLCSchema()
                result = data.from_bybit_v2_to_form(msg, extra, datatype=self.DATA_TYPE)
                payload.append(result)
        elif isinstance(common_response.data, dict):
            common_response.data = key_data(common_response.data)
            payload = models.HistoryOHLCSchema()
            payload = payload.from_bybit_v2_to_form(common_response.data, extra, datatype=self.DATA_TYPE)
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _regular_ticker_payload(
        self,
        common_response: models.CommonResponseSchema,
    ) -> dict:
        payload = list()
        if isinstance(common_response.data, list):
            for msg in common_response.data:
                data = models.TickerSchema()
                payload = data.from_bybit_v2_to_form(msg, datatype=self.DATA_TYPE)
                # payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.TickerSchema()
            payload = payload.from_bybit_v2_to_form(common_response.data, datatype=self.DATA_TYPE)
        else:
            common_response.msg = "payload error in fetch symbol data"
            Exception("payload error in fetch symbol data")
        # print(f"payload: {payload}")
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
                result = data.from_bybit_v2_to_form(
                    msg,
                    transfer_type=transfer_type,
                    inout=inout,
                )
                payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.TransferSchema()
            payload = payload.from_bybit_v2_to_form(
                common_response.data,
                transfer_type=transfer_type,
                inout=inout,
            )
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
                result = data.from_bybit_v2_to_form(msg, datatype=self.DATA_TYPE, account_type=account_type)
                if result:
                    # payload.append(result)
                    payload[result["symbol"]] = result
        elif isinstance(common_response.data, dict):
            payload = models.AccountSchema()
            payload = payload.from_bybit_v2_to_form(
                common_response.data, datatype=self.DATA_TYPE, account_type=account_type
            )
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _regular_position_payload(
        self,
        common_response: models.CommonResponseSchema,
    ) -> dict:
        if isinstance(common_response.data, list):
            payload = list()
            for msg in common_response.data:
                data = models.PositionSchema()
                result = data.from_bybit_v2_to_form(msg, datatype=self.DATA_TYPE)
                payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.PositionSchema()
            payload = payload.from_bybit_v2_to_form(common_response.data, datatype=self.DATA_TYPE)
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _regular_incomes_payload(
        self,
        common_response: models.CommonResponseSchema,
    ) -> dict:
        if isinstance(common_response.data, list):
            payload = list()
            for msg in common_response.data:
                data = models.IncomeSchema()
                result = data.from_bybit_v2_to_form(msg, datatype=self.DATA_TYPE)
                payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.IncomeSchema()
            payload = payload.from_bybit_v2_to_form(common_response.data, datatype=self.DATA_TYPE)
        else:
            common_response.msg = "payload error in order"
            Exception("payload error in order")

        common_response.data = payload
        return common_response

    def _regular_funding_rate_payload(
        self,
        common_response: models.CommonResponseSchema,
    ) -> dict:
        if isinstance(common_response.data, list):
            payload = list()
            for msg in common_response.data:
                data = models.FundingRateSchema()
                result = data.from_bybit_v2_to_form(msg, datatype=self.DATA_TYPE)
                payload.append(result)
        elif isinstance(common_response.data, dict):
            payload = models.FundingRateSchema()
            payload = payload.from_bybit_v2_to_form(common_response.data, datatype=self.DATA_TYPE)
        else:
            common_response.msg = "payload error in funding rate data"
            Exception("payload error in funding rate data")

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

    def query_permission(self) -> dict:
        """ """
        api_info = self._get("/v5/user/query-api")
        # print(f">>>> api_info:{api_info}")
        if api_info.success:
            payload = models.PermissionSchema()  # .model_construct(api_info.data)
            api_info.data = payload.from_bybit_v2_to_form(api_info.data)
            self.unified = payload.unified

        return api_info

    def query_account(self) -> dict:
        """
        Account type
        Unified account: UNIFIED (trade spot/linear/options), CONTRACT(trade inverse)
        Normal account: CONTRACT, SPOT

        CONTRACT
        SPOT
        INVESTMENT
        OPTION
        UNIFIED
        FUND
        """
        # account_type = 'UNIFIED' if self.unified else self.DATA_TYPE.upper() if self.DATA_TYPE in ['spot', 'option','investment', 'fund'] else 'CONTRACT'
        # payload = {
        #     'accountType': account_type
        # }
        # balance = self._get("/v5/asset/transfer/query-account-coins-balance", payload)
        # account_type = "UNIFIED" if self.unified else "SPOT" if self.DATA_TYPE == "spot" else "CONTRACT"
        if self.unified:
            if self.DATA_TYPE == "inverse":
                account_type = "CONTRACT"
            else:
                account_type = "UNIFIED"
        else:
            if self.DATA_TYPE == "spot":
                account_type = "SPOT"
            else:
                account_type = "CONTRACT"

        payload = {"accountType": account_type}
        balance = self._get("/v5/account/wallet-balance", payload)
        # print(f">>> response:{balance} | payload:{payload}")

        if balance.success:
            if balance.data:
                # balance.data = balance.data.get('balance', [])
                account_list = balance.data.get("list", [])
                coin_list = []
                for acc in account_list:
                    coin_list += acc["coin"]
                balance.data = coin_list

                return self._regular_account_payload(balance, account_type)
            else:
                balance.msg = "query account is empty"

        return balance

    def send_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type=None,
        price=None,
        position_side: str = "",
        reduce_only: bool = False,
        order_id: str = "",
        is_margin_trading: bool = False,
        time_in_force: str = "GTC",
        **kwargs,
    ) -> dict:
        """
        positionIdx: Used to identify positions in different position modes. Under hedge-mode, this param is required
        0: one-way mode
        1: hedge-mode Buy side
        2: hedge-mode Sell side

        ### if not bid/ask price, the order will be cancel automatically
        """

        payload = {
            "symbol": symbol.upper(),
            "side": str(side).lower().capitalize(),
            # 'price': price,
            "qty": quantity,
            "orderLinkId": order_id if order_id else self.new_order_id(),
            "timeInForce": time_in_force,
            "agentSource": "AREIX",  # 0.35%
            "category": self.DATA_TYPE,
        }
        if is_margin_trading:
            # isLeverage	false	integer	Whether to borrow. Valid for Unified spot only. 0(default): false then spot trading, 1: true then margin trading
            # Not all the spot support margin trading, need to check instrurment api => "marginTrading":"none" / "both"
            if self.unified and self.DATA_TYPE.lower() == "spot":
                payload["isLeverage"] = 1
            else:
                return models.CommonResponseSchema(
                    success=False,
                    error=True,
                    data={},
                    msg="Spot Margin Trading is only available for UNIFIED account.",
                )
        if not order_type:
            if not price:
                order_type = "Market"
            else:
                order_type = "Limit"

            payload["orderType"] = order_type
        else:
            order_type = order_type.lower().capitalize()

        if order_type.upper() == "LIMIT":
            payload["price"] = price
        elif order_type.upper() == "MARKET":
            if (
                self.DATA_TYPE.lower() == "spot" and payload["side"] == "Buy"
            ):  # if it is market buy, it is qty needs to be in USDT
                px = self.query_prices(symbol)
                last_price = self.query_prices(symbol)["data"]["last_price"]
                payload["qty"] = round(last_price * quantity, 4)
                print(f"payload: {payload} --- in bybit spot send_order --- last_price: {last_price}")
        if position_side:
            if position_side == "long":
                payload["positionIdx"] = 1
            elif position_side == "short":
                payload["positionIdx"] = 2
        if reduce_only:
            payload["reduceOnly"] = reduce_only

        orders_response = self._post("/v5/order/create", payload)
        # print(f">>> send_order_response :{orders_response}")
        # success=True error=False data={'config': {'endpointExec': 'KYC_PROMPT_TOAST', 'endpointArgs': {}}} msg='query ok'

        if orders_response.success:
            if orders_response.data:
                result = models.SendOrderResultSchema()  # .model_construct(orders_response.data)
                orders_response.data = result.from_bybit_v2_to_form(
                    orders_response.data, symbol=symbol, datatype=self.DATA_TYPE
                )
            else:
                orders_response.msg = "query sender order is empty"

        return orders_response

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """ """
        payload = {"category": self.DATA_TYPE, "symbol": symbol, "orderId": order_id}
        return self._post("/v5/order/cancel", payload)

    def cancel_orders(self, symbol: str) -> dict:
        """ """
        payload = {"category": self.DATA_TYPE, "symbol": symbol}
        return self._post("/v5/order/cancel-all", payload)

    def query_order_status(self, order_id: str = None, symbol: str = None) -> dict:
        """ """
        payload = {"category": self.DATA_TYPE}
        if order_id:
            payload["orderId"] = order_id
        if symbol:
            payload["symbol"] = symbol

        order_status = self._get("/v5/order/realtime", payload)
        # print(f"query_order_status: {order_status}")
        if order_status.success:
            if order_status.data:
                order_status.data = order_status.data.get("list", [])
                d = self._regular_order_payload(order_status)
                if d["data"]:
                    d["data"] = d["data"][0]
                return d
            else:
                order_status.msg = "query order is empty"
                order_status.data = []

        return order_status

    def query_open_orders(
        self,
        symbol: str = None,
        settle_coin: str = "USDT",
        order_id: str = None,
        limit: int = 50,
    ) -> dict:
        """
        For linear & inverse, either symbol or settleCoin is required

        baseCoin. For option only. Return all option open orders if not passed
        """
        payload = {"category": self.DATA_TYPE}
        if limit:
            payload["limit"] = limit
        if order_id:
            payload["orderId"] = order_id
        if symbol:
            payload["symbol"] = symbol
        else:
            # if (self.DATA_TYPE == "linear" and settle_coin.upper() in LINEAR_SETTLECOINS) or (self.DATA_TYPE == "inverse" and settle_coin.upper() in INVERSE_SETTLECOINS):
            #     payload["settleCoin"] = settle_coin.upper()
            # elif self.DATA_TYPE == "inverse":
            if self.DATA_TYPE in ["linear", "inverse"]:
                settle_coins = LINEAR_SETTLECOINS if self.DATA_TYPE == "linear" else INVERSE_SETTLECOINS
                all_open_orders = []
                order_status = None
                for _settle_coin in settle_coins:
                    payload["settleCoin"] = _settle_coin
                    order_status = self._get("/v5/order/realtime", payload)
                    if order_status.success:
                        if order_status.data:
                            all_open_orders += order_status.data.get("list", [])
                if order_status and order_status.success:
                    order_status.data = all_open_orders
                    if all_open_orders:
                        return self._regular_order_payload(order_status)
                    else:
                        order_status.msg = "query inverse open orders is empty"

                return order_status

        order_status = self._get("/v5/order/realtime", payload)
        # print(f"query_open_orders: {order_status} | payload:{payload}")
        if order_status.success:
            if order_status.data:
                order_status.data = order_status.data.get("list", [])
                return self._regular_order_payload(order_status)
            else:
                order_status.msg = "query order is empty"
                order_status.data = []

        return order_status

    def query_all_orders(
        self,
        symbol: str = None,
        order_id: str = None,
        settle_coin: str = "USDT",
        limit=50,
    ) -> dict:
        """ """
        payload = {
            "category": self.DATA_TYPE,
            "openOnly": 0,
        }  # 0 means query open orders only
        if limit:
            payload["limit"] = limit
        if order_id:
            payload["orderId"] = order_id
        if symbol:
            payload["symbol"] = symbol
        else:
            payload["settleCoin"] = settle_coin

        order_status = self._get("/v5/order/history", payload)
        # print(f"query_open_orders: {order_status}")
        if order_status.success:
            if order_status.data:
                order_status.data = order_status.data.get("list", [])
                return self._regular_order_payload(order_status)
            else:
                order_status.msg = "query order is empty"
                order_status.data = []

        return order_status

    def query_symbols(
        self,
        symbol: str = None,
        category: str = None,
        base_coin: str = None,
        limit: int = 1000,
    ) -> dict:
        """ """
        payload = {"category": category if category else self.DATA_TYPE, "limit": limit}
        if symbol:
            payload["symbol"] = symbol
        if base_coin:
            payload["baseCoin"] = base_coin
        symbols = self._get("/v5/market/instruments-info", payload)
        if symbols.success:
            if symbols.data:
                if symbol and symbols.data.get("list", []):
                    symbols.data = symbols.data["list"][0]
                else:
                    symbols.data = symbols.data["list"]
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
        The start timestamp (ms)
            startTime and endTime are not passed, return 7 days by default
            Only startTime is passed, return range between startTime and startTime+7 days
            Only endTime is passed, return range between endTime-7 days and endTime
            If both are passed, the rule is endTime - startTime <= 7 days
        """

        def _generate_window_dates(start_date, end_date, window):
            current_date = start_date
            window_end_date = end_date

            while current_date <= end_date:
                window_end_date = current_date + timedelta(days=window)  # - timedelta(days=1)

                if window_end_date > end_date:
                    window_end_date = end_date

                yield current_date, window_end_date

                current_date = window_end_date + timedelta(days=1)
            else:
                if window_end_date != end_date:
                    yield window_end_date, end_date

        payload = {
            "category": self.DATA_TYPE,
            "limit": limit,
        }
        if start:
            start = start.astimezone(UTC_TZ) if start.tzinfo else UTC_TZ.localize(start)
            payload["startTime"] = int(datetime.timestamp(start) * 1000)
            if not end:
                end = datetime.now(UTC_TZ)
        if end:
            end = end.astimezone(UTC_TZ) if end.tzinfo else UTC_TZ.localize(end)
            payload["endTime"] = int(datetime.timestamp(end) * 1000)
        if symbol:
            payload["symbol"] = symbol
        else:
            if self.DATA_TYPE in ["linear", "inverse"]:
                return models.CommonResponseSchema(success=False, error=True, data={}, msg="Please specify the symbol.")

        print(f"[bybitv2 - query_trades] payload:{payload} | start:{start} | end:{end}")
        if start and end:
            trade_list = []
            window = 1 if self.DATA_TYPE == "spot" else 6

            for start_dt, end_dt in _generate_window_dates(start, end, window):
                payload["startTime"] = int(datetime.timestamp(start_dt) * 1000)
                payload["endTime"] = int(datetime.timestamp(end_dt) * 1000)
                payload.pop("cursor", None)
                trades_next = self._get("/v5/execution/list", remove_none(payload))
                if trades_next.success:
                    trade_list += trades_next.data["list"]
                else:
                    self.logger.error(f">>>>> trades_next error:{trades_next} | payload:{payload}")
                # print(f">> start_dt:{start_dt} | end_dt:{end_dt} | payload:{payload} | trades_next:{len(trades_next.data['list'])}")
                while trades_next.data.get("nextPageCursor"):
                    payload["cursor"] = trades_next.data.get("nextPageCursor")
                    trades_next = self._get("/v5/execution/list", remove_none(payload))
                    if trades_next["success"]:
                        trade_list += trades_next.data["list"]
                    else:
                        self.logger.error(f">>>>> trades_next error:{trades_next} | payload:{payload}")
                    # print(f"??? cursor:{payload['cursor']} | payload:{payload} | len:{len(trades_next.data['list'])}")

            # Create trades response object with the collected trade_list
            trades = models.CommonResponseSchema(success=True, error=False, data={"list": trade_list}, msg="")
        else:
            # if start or end or both None:
            trade_list = []
            trades = self._get("/v5/execution/list", remove_none(payload))
            # print(f"[DEBUG] len trades:{len(trades.data['list'])}; nextPageCursor:{trades.data.get('nextPageCursor')} | first new trade:{trades.data['list'][0]}; last new trade:{trades.data['list'][-1]};")
            if trades.success:
                trade_list = trades.data["list"]
            else:
                self.logger.error(f">>>>> trades error:{trades} | payload:{payload}")
            # print(f"trades:{trades}")

            trades_next = copy(trades)
            while trades_next.data.get("nextPageCursor"):
                payload["cursor"] = trades_next.data.get("nextPageCursor")
                trades_next = self._get("/v5/execution/list", remove_none(payload))
                # print(f">>> trades_next:{trades_next}")
                if trades_next["success"]:
                    trade_list += trades_next.data["list"]
                else:
                    self.logger.error(f">>>>> trades_next error:{trades_next} | payload:{payload}")
                # print(f"    [DEBUG] len new trades:{len(new_trades)}; first new trade:{new_trades[0]['execTime']}; last new trade:{new_trades[-1]['execTime']}; nextPageCursor:{trades_next.data.get('nextPageCursor')} ...")
            trades = models.CommonResponseSchema(success=True, error=False, data={"list": trade_list}, msg="")

        if trades.success:
            # print(f"[DEBUG] trades>>>>: {trades}")
            if trades.data:
                trades.data = trades.data.get("list", [])
                ret_trades = self._regular_trades_payload(trades)
                # aggregate by ori_order_id
                ret_trades.data = self._aggregate_trades(ret_trades["data"])
                return ret_trades
            else:
                trades.msg = "query trades is empty"
                trades.data = []

        return trades

    def query_historical_vol(
        self,
        symbol: str,
        period: str | int,
        start: datetime = None,
        end: datetime = None,
        limit: int = 1000,
    ) -> dict:
        """
        The data is hourly.
        BTC: 7,14,21,30,60,90,180,270days
        ETH: 7,14,21,30,60,90,180,270days
        SOL: 7,14,21,30,60,90        days
        Period. If not passed, it returns 7 days by default. SOL does not have 180 and 270


        """

        def iteration_history(symbol, period, data, payload, limit, retry=3):
            retry_counter = 0
            historical_prices_datalist = data.data
            # self.logger.debug(interval)
            if not historical_prices_datalist:
                return data
            first_timestamp = int(historical_prices_datalist[0][0])
            start_timestamp = payload["start"]
            interval_timestamp = first_timestamp - start_timestamp
            # self.logger.debug(payload)

            while interval_timestamp > period:
                first_timestamp = int(historical_prices_datalist[0][0])
                interval_timestamp = first_timestamp - start_timestamp

                # self.logger.debug(f"first_timestamp {first_timestamp}")
                # self.logger.debug(f"interval_timestamp {interval_timestamp}")
                # self.logger.debug(historical_prices_datalist[0:5])

                payload["end"] = first_timestamp

                prices = self._get("/v5/market/historical-volatility", remove_none(payload))
                if prices.error:
                    break
                prices.data = prices.data.get("list", [])
                # self.logger.debug(prices.data[0:2])

                historical_prices_datalist.extend(prices.data)
                historical_prices_datalist = [list(item) for item in set(tuple(x) for x in historical_prices_datalist)]
                historical_prices_datalist.sort(key=lambda k: k[0])
                time.sleep(0.1)

                # self.logger.debug(f"payload: {payload}")
                # self.logger.debug(f"retry_counter: {retry_counter}")

                if len(prices.data) != limit:
                    retry_counter += 1

                if retry_counter >= retry:
                    break

            data.data = historical_prices_datalist
            # self.logger.debug(f"data length: {len(historical_prices_datalist)}")
            return data

        # main
        if self.DATA_TYPE not in ["option"]:
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg=f"{self.DATA_TYPE} is not support to fetch historical volatility.",
            )
        if symbol not in ["BTC", "ETH", "SOL"]:
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg=f"{symbol} is not support to fetch historical volatility.",
            )

        if int(period) not in [7, 14, 21, 30, 60, 90, 180, 270]:
            return models.CommonResponseSchema(success=False, error=True, data={}, msg="Invalid period.")

        payload = {"symbol": symbol.upper(), "period": period, "limit": limit}

        # ms - epoch time
        if start:
            start = UTC_TZ.localize(start)  # NOTE: bybit start and end timestamp default regard it as utc timestamp
            payload["startTime"] = int(datetime.timestamp(start) * 1000)
        if end:
            end = UTC_TZ.localize(end)
            payload["endTime"] = int(datetime.timestamp(end) * 1000)

        historical_vols = self._get("/v5/market/historical-volatility", remove_none(payload))
        extra = {"period": period, "symbol": symbol}
        print(f"historical_vols: {historical_vols}")
        if historical_vols.success:
            if historical_vols.data:
                historical_vols.data = historical_vols.data.get("list", [])
                historical_vols.data.sort(key=lambda k: k[0])
                # handle query time
                if "startTime" in payload:
                    period = TIMEDELTA_MAPPING_SEC[period] * 1000
                    historical_vols = iteration_history(
                        symbol=symbol,
                        period=period,
                        data=historical_vols,
                        payload=payload,
                        limit=limit,
                    )
                # return self._regular_historical_prices_payload(historical_vols, extra)
            else:
                historical_vols.msg = "query historical prices is empty"

        return historical_vols

    def query_history(
        self,
        symbol: str,
        interval: str = "1h",
        start: datetime = None,
        end: datetime = None,
        limit: int = 1000,
    ) -> dict:
        """
        Kline interval. 1,3,5,15,30,60,120,240,360,720,D,M,W
        """

        # print(f">>>>>>>>symbol:{symbol}; interval:{interval}; start:{start}; end:{end}")
        def iteration_history(symbol, interval, data, payload, limit, retry=3):
            retry_counter = 0
            historical_prices_datalist = data.data
            if not historical_prices_datalist:
                return data
            # self.logger.debug(interval)
            # print(f"[DEBUG] iteration_history - historical_prices_datalist:{historical_prices_datalist}")
            first_timestamp = int(historical_prices_datalist[0][0])
            start_timestamp = payload["start"]
            interval_timestamp = first_timestamp - start_timestamp
            # self.logger.debug(payload)

            while interval_timestamp > interval:
                first_timestamp = int(historical_prices_datalist[0][0])
                interval_timestamp = first_timestamp - start_timestamp

                # self.logger.debug(f"first_timestamp {first_timestamp}")
                # self.logger.debug(f"interval_timestamp {interval_timestamp}")
                # self.logger.debug(historical_prices_datalist[0:5])

                payload["end"] = first_timestamp

                prices = self._get("/v5/market/kline", remove_none(payload))
                if prices.error:
                    break
                prices.data = prices.data.get("list", [])
                # self.logger.debug(prices.data[0:2])

                historical_prices_datalist.extend(prices.data)
                historical_prices_datalist = [list(item) for item in set(tuple(x) for x in historical_prices_datalist)]
                historical_prices_datalist.sort(key=lambda k: k[0])
                time.sleep(0.1)

                # self.logger.debug(f"payload: {payload}")
                # self.logger.debug(f"retry_counter: {retry_counter}")

                if len(prices.data) != limit:
                    retry_counter += 1

                if retry_counter >= retry:
                    break

            data.data = historical_prices_datalist
            # self.logger.debug(f"data length: {len(historical_prices_datalist)}")
            return data

        # main
        if self.DATA_TYPE not in ["spot", "linear", "inverse"]:
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg=f"{self.DATA_TYPE} is not support to fetch historical klines.",
            )

        if interval not in KBAR_INTERVAL:
            return models.CommonResponseSchema(success=False, error=True, data={}, msg="Invalid interval.")

        payload = {
            "symbol": symbol.upper(),
            "interval": KBAR_INTERVAL[interval],
            "limit": limit,
        }
        # ms - epoch time
        if start:
            start = start.astimezone(UTC_TZ)  # NOTE: bybit start and end timestamp default regard it as utc timestamp
            payload["start"] = int(datetime.timestamp(start) * 1000)
        if end:
            end = end.astimezone(UTC_TZ)
            payload["end"] = int(datetime.timestamp(end) * 1000)

        historical_prices = self._get("/v5/market/kline", remove_none(payload))
        extra = {"interval": interval, "symbol": symbol}

        # print(f"[query_history] historical_prices: {historical_prices} | payload:{payload}")

        if historical_prices.success:
            if historical_prices.data:
                historical_prices.data = historical_prices.data.get("list", [])
                historical_prices.data.sort(key=lambda k: k[0])
                # handle query time
                if "start" in payload:
                    interval = TIMEDELTA_MAPPING_SEC[interval] * 1000
                    historical_prices = iteration_history(
                        symbol=symbol,
                        interval=interval,
                        data=historical_prices,
                        payload=payload,
                        limit=limit,
                    )

                return self._regular_historical_prices_payload(historical_prices, extra)
            else:
                historical_prices.msg = "query historical prices is empty"

        return historical_prices

    def query_last_price(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 1,
    ) -> dict:
        """
        interval: 1,3,5,15,30,60,120,240,360,720,D,M,W
        """
        if self.DATA_TYPE in ["option"]:
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg="Option is not support to fetch kline data.",
            )

        payload = {
            "category": self.DATA_TYPE,
            "symbol": symbol,
            "interval": KBAR_INTERVAL[interval],
            "limit": limit,
        }
        last_historical_prices = self._get("/v5/market/kline", payload)
        extra = {"interval": interval, "symbol": symbol}
        # print(f"last_historical_prices: {last_historical_prices}")

        if last_historical_prices.success:
            if last_historical_prices.data:
                last_historical_prices.data = last_historical_prices.data.get("list", [])
                last_data = self._regular_historical_prices_payload(last_historical_prices, extra)
                if last_data.data and isinstance(last_data.data, list):
                    last_data.data = last_data.data[0]
                return last_data
            else:
                last_historical_prices.msg = "query latest historical prices is empty"

        return last_historical_prices

    def query_prices(self, symbol: str = None, base_coin: str = None, exp_date: str = None) -> dict:
        """ """
        payload = {"category": self.DATA_TYPE}

        if symbol:
            payload["symbol"] = symbol.upper()
        if base_coin:
            payload["baseCoin"] = base_coin
        if exp_date:
            payload["expDate"] = exp_date
        spots = self._get("/v5/market/tickers", remove_none(payload))

        # self.logger.debug(spots)
        # print(f">>>>>>>>> symbol:{symbol}???? DATA_TYPE:{self.DATA_TYPE} price:{spots} | payload:{payload}")
        """
        option:
        success=True error=False data={'category': 'option', 'list': [{'symbol': 'BTC-31MAR23-20000-C', 'bid1Price': '0', 'bid1Size': '0', 'bid1Iv': '0', 'ask1Price': '4460', 'ask1Size': '0.04', 'ask1Iv': '0', 'lastPrice': '910', 'highPrice24h': '0', 'lowPrice24h': '0', 'markPrice': '5057.88495797', 'indexPrice': '24902.23', 'markIv': '0.7673', 'underlyingPrice': '24916.74', 'openInterest': '19.82', 'turnover24h': '2431.3', 'volume24h': '0.1', 'totalVolume': '2540', 'totalTurnover': '49881242', 'delta': '0.92604985', 'gamma': '0.00003495', 'vega': '7.31457284', 'theta': '-17.49516262', 'predictedDeliveryPrice': '0', 'change24h': '0'}]} msg='query ok'
        CommonResponseSchema(success=True, error=False, data={'category': 'option', 'list': [{'symbol': 'BTC-8JUL23-30250-C', 'bid1Price': '20', 'bid1Size': '12.6', 'bid1Iv': '0.1036', 'ask1Price': '225', 'ask1Size': '1040.4', 'ask1Iv': '0.4225', 'lastPrice': '0', 'highPrice24h': '0', 'lowPrice24h': '0', 'markPrice': '260.4705185', 'indexPrice': '30091.73', 'markIv': '0.4742', 'underlyingPrice': '30095.1557', 'openInterest': '0', 'turnover24h': '3633.260714', 'volume24h': '0.12', 'totalVolume': '1', 'totalTurnover': '6074', 'delta': '0.43165204', 'gamma': '0.0004733', 'vega': '6.88482449', 'theta': '-132.02226823', 'predictedDeliveryPrice': '0', 'change24h': '0'}]}, msg='query ok')

        """
        if spots.success:
            if spots.data:
                spots.data = spots.data.get("list", [])
                return self._regular_ticker_payload(spots)
            else:
                spots.msg = "query spots is empty"
        return spots

    def query_transfer(self, start: datetime = None, end: datetime = None, inout=None) -> dict:
        """
        https://bybit-exchange.github.io/docs/v5/asset/deposit-record
        https://bybit-exchange.github.io/docs/v5/asset/withdraw-record
        endTime - startTime should be less than 30 days. Query last 30 days records by default.

        """
        # return models.CommonResponseSchema(success=False, error=True, data={}, msg=f"Please waiting for the update")
        payload = {}
        # ms - epoch time
        ts_limit = 24 * 60 * 60 * 30 * 1000
        if start:
            start = UTC_TZ.localize(start)  # NOTE: bybit start and end timestamp default regard it as utc timestamp
            payload["startTime"] = int(datetime.timestamp(start)) * 1000
        if end:
            end = UTC_TZ.localize(end)
            payload["endTime"] = int(datetime.timestamp(end)) * 1000
        if payload:
            if not payload.get("startTime"):
                payload["startTime"] = payload["endTime"] - ts_limit
            if not payload.get("endTime"):
                payload["endTime"] = payload["startTime"] + ts_limit
        # print(f"query_transfer payload:{payload}")

        raw_end_time = payload.get("endTime")
        deposit_list = []
        withdraw_list = []
        deposit = None
        withdraw = None
        while True:
            if payload:
                if raw_end_time - payload["startTime"] > ts_limit:  # 30day
                    payload["endTime"] = payload["startTime"] + ts_limit
                else:
                    payload["endTime"] = raw_end_time
            if inout is None or inout == "in":
                deposit = self._get("/v5/asset/deposit/query-record", params=payload)
                # print(f">>> deposit: {deposit} | payload:{payload}")
                """
                >>> deposit: success=True error=False data={'rows': [{'coin': 'ID', 'chain': 'BSC', 'amount': '301', 'txID': '0xe587df02ffea7b83ddf06812dd57d550328b0e3c3061c15e8d80e9dbb66b2e0a', 'status': 3, 'toAddress': '0xc7bda6ca0179653d3fb08a8933806732e6a83e99', 'tag': '', 'depositFee': '', 'successAt': '1679585297000', 'confirmations': '106', 'txIndex': '402', 'blockHash': '0x5d6486075d81e3ba3afbc93f409badeb68ef5717698b73aa271d2e991831ad79'}], 'nextPageCursor': 'eyJtaW5JRCI6MjA3NDg0OTEsIm1heElEIjoyMDc0ODQ5MX0='} msg='query ok' | payload:{}
                """
                if deposit.success and deposit.data:
                    deposit_list += deposit.data.get("rows", [])
                else:
                    return deposit

            if inout is None or inout == "out":
                withdraw = self._get("/v5/asset/withdraw/query-record", params=payload)
                # print(f">>> withdraw: {withdraw} | payload:{payload}")
                """
                >>> withdraw: success=True error=False data={'rows': [{'coin': 'USDT', 'chain': 'ARBI', 'amount': '51', 'txID': '0x8004221a670ee503c13aa737b55636a14c3b1c35db4427afbdbac0b28ce22955', 'status': 'success', 'toAddress': '0xe9c33326a3b83ce7905ed9fc54445d5f79cbca42', 'tag': '', 'withdrawFee': '0.3', 'createTime': '1680744215000', 'updateTime': '1680744254000', 'withdrawId': '14650317', 'withdrawType': 0}, {'coin': 'ETH', 'chain': 'ARBI', 'amount': '0.01414', 'txID': '0x4fdca8c1194f529d7498b04140c60000b8edc33f3d72ffa08ed228329434b753', 'status': 'success', 'toAddress': '0xF5e66Fc1dd20d411263ED0B0fFab359Be47fd829', 'tag': '', 'withdrawFee': '0.0003', 'createTime': '1679366935000', 'updateTime': '1679366990000', 'withdrawId': '13871748', 'withdrawType': 0}, {'coin': 'MATIC', 'chain': 'MATIC', 'amount': '9.65', 'txID': '0x720454872b1e1f1a350407d393a2d95ca18525f77e579c47b3c98e6a6e1d51b6', 'status': 'success', 'toAddress': '0xF5e66Fc1dd20d411263ED0B0fFab359Be47fd829', 'tag': '', 'withdrawFee': '0.1', 'createTime': '1679321623000', 'updateTime': '1679321745000', 'withdrawId': '13851997', 'withdrawType': 0}], 'nextPageCursor': 'eyJtaW5JRCI6MTM4NTE5OTcsIm1heElEIjoxNDY1MDMxN30='} msg='query ok' | payload:{}
                """
                if withdraw.success and withdraw.data:
                    withdraw_list += withdraw.data.get("rows", [])
                else:
                    return withdraw

            # print(f'payload:{payload}....')
            if not payload or (raw_end_time - payload["startTime"] <= ts_limit):  # 30day
                break
            else:
                payload["startTime"] = payload["endTime"]

        # print(f'deposit_list:{deposit_list}; len:{len(deposit_list)}')
        data = []
        if deposit:
            deposit.data = deposit_list
            deposit_ret = self._regular_transfer_payload(deposit, transfer_type="external", inout="in")
            data += deposit_ret.data
            ret = deposit

        # print(f'withdraw_list:{withdraw_list}; len:{len(withdraw_list)}')
        if withdraw:
            withdraw.data = withdraw_list
            withdraw_ret = self._regular_transfer_payload(withdraw, transfer_type="external", inout="out")
            data += withdraw_ret.data
            ret = withdraw

        ret.data = data
        if ret.data:
            ret.data = sorted(ret.data, key=lambda k: k["datetime"])

        return ret

    def query_internal_transfer(self) -> dict:
        """
        https://bybit-exchange.github.io/docs/v5/asset/internal-deposit-record
        """
        # transfer = self._get("/asset/v1/private/transfer/list")
        # transfers = list()
        # # self.logger.debug(balance)
        # if transfer.success and transfer.data:
        #     if not transfer.data["list"]:
        #         transfer.msg = "transfer list is empty"
        #         return transfer
        #     for transfer_data in transfer.data["list"]:
        #         transfers.append(transfer_data)
        #     transfer.data = transfers
        #     return transfer
        # else:
        #     return transfer
        return models.CommonResponseSchema(success=False, error=True, data={}, msg="Please waiting for the update")

    def query_position(
        self,
        symbol: str = None,
        settle_coin: str = "USDT",
        limit: int = 200,
        force_return: bool = False,
    ) -> dict:
        """ """
        if self.DATA_TYPE in ["spot"]:
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg="Spot is not support to query position.",
            )
        if self.DATA_TYPE in ["option"] and not self.unified:
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg="Only unified account support to query option position.",
            )

        payload = {"category": self.DATA_TYPE}

        # if self.DATA_TYPE == 'inverse':
        #     for settle_coin
        #     position = self._get("/v5/position/list", payload)
        #     # print(f"position: {position} | payload:{payload}")

        #     if position.success:
        #         if position.data:
        #             # position = handle_valid_data(position, symbol)
        #             # print(position,'!!!!!')
        #             position.data = position.data["list"]
        #             return self._regular_position_payload(position)
        #         else:
        #             position.msg = "query future position is empty"

        #     return position
        if limit:
            payload["limit"] = limit
        if symbol:
            payload["symbol"] = symbol
        else:
            # if (self.DATA_TYPE == "linear" and settle_coin in LINEAR_SETTLECOINS) or (self.DATA_TYPE == "inverse" and settle_coin in INVERSE_SETTLECOINS):
            #     payload["settleCoin"] = settle_coin
            # elif self.DATA_TYPE == "inverse":
            if self.DATA_TYPE in ["linear", "inverse"]:
                settle_coins = LINEAR_SETTLECOINS if self.DATA_TYPE == "linear" else INVERSE_SETTLECOINS
                all_positions = []
                position = None
                for _settle_coin in settle_coins:
                    payload["settleCoin"] = _settle_coin
                    position = self._get("/v5/position/list", payload)
                    if position.success:
                        if position.data:
                            all_positions += position.data.get("list", [])
                if position and position.success:
                    position.data = all_positions
                    if all_positions:
                        return self._regular_position_payload(position)
                    else:
                        position.msg = "query position is empty"

                return position

        # if self.DATA_TYPE == 'option':
        #     payload['baseCoin'] =  'USD'

        position = self._get("/v5/position/list", payload)
        # print(f"position: {position} | payload:{payload}")

        if position.success:
            if position.data:
                # position = handle_valid_data(position, symbol)
                # print(position,'!!!!!')
                position.data = position.data["list"]
                return self._regular_position_payload(position)
            else:
                position.msg = "query linear position is empty"

        return position

    def set_position_mode(self, mode: str, coin: str = "USDT", symbol: str = "") -> dict:
        """
            普通帳戶	統一帳戶
        USDT 永續	支持單雙向持倉	僅支持單向持倉
        USDC 永續	僅支持單向持倉	僅支持單向持倉
        反向永續	僅支持單向持倉	N/A
        反向交割	支持單雙向持倉	N/A

        統一帳戶: inverse
        普通帳戶: linear, inverse
        """
        mode = mode.lower()
        if mode not in ["oneway", "hedge"]:
            # raise ValueError("Invalid position_mode")
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg="Please specify the correct position mode.",
            )
        # if self.unified and self.DATA_TYPE not in ["inverse"]:
        #     return models.CommonResponseSchema(success=False, error=True, data={}, msg=f"In unified account, only inverse support to switch position mode.")
        if not self.unified and self.DATA_TYPE not in ["linear", "inverse"]:
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg="In normal account, only inverse & linear support to switch position mode.",
            )
        # ### NOTE: since the trade data cannot identify the position side, so will disable the hedge mode from now on (2025-03-14)
        # if mode in ["hedge"]:
        #     return models.CommonResponseSchema(
        #         success=False,
        #         error=True,
        #         data={},
        #         msg="Since the trade data cannot identify the position side, so will disable the hedge mode from now on (2025-03-14).",
        #     )
        payload = {
            "category": self.DATA_TYPE,
            "mode": 0 if mode == "oneway" else 3,
        }
        if symbol:
            payload["symbol"] = symbol
        else:
            payload["coin"] = coin

        mode_payload = self._post("/v5/position/switch-mode", remove_none(payload))
        # print(f"[bybit === set_position_mode]:{mode_payload}")
        if mode_payload.success:
            mode_payload.data = models.PositionModeSchema(position_mode=mode, coin=coin, symbol=symbol).dict()
            return mode_payload

        return mode_payload

    def set_leverage(
        self,
        leverage: float,
        symbol: str,
    ) -> dict:
        """
        統一帳戶: inverse
        普通帳戶: linear, inverse
        """
        # if self.unified and self.DATA_TYPE not in ["inverse"]:
        #     return models.CommonResponseSchema(success=False, error=True, data={}, msg=f"In unified account, only inverse support to switch margin mode.")
        if not self.unified and self.DATA_TYPE not in ["linear", "inverse"]:
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg="In normal account, only inverse & linear support to switch margin mode.",
            )

        payload = {
            "category": self.DATA_TYPE,
            "symbol": symbol.upper(),
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }

        leverage_payload = self._post("/v5/position/set-leverage", remove_none(payload))
        # print(leverage_payload,'!!!!set-leverage')
        if leverage_payload.success:
            if leverage_payload.data:
                leverage_payload.data = models.LeverageSchema(symbol=symbol, leverage=leverage).dict()
                return leverage_payload
        return leverage_payload

    def set_margin_mode(self, is_isolated: bool, leverage: float, symbol: str) -> dict:
        """
        統一帳戶: inverse
        普通帳戶: linear, inverse
        """
        # if self.unified and self.DATA_TYPE not in ["inverse"]:
        #     return models.CommonResponseSchema(success=False, error=True, data={}, msg=f"In unified account, only inverse support to switch margin mode.")
        if not self.unified and self.DATA_TYPE not in ["linear", "inverse"]:
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg="In normal account, only inverse & linear support to switch margin mode.",
            )

        payload = {
            "category": self.DATA_TYPE,
            "symbol": symbol.upper(),
            "tradeMode": 1 if is_isolated else 0,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }

        leverage_payload = self._post("/v5/position/switch-isolated", remove_none(payload))
        # print(leverage_payload,'!!!!switch-isolated')

        if leverage_payload.success:
            if leverage_payload.data:
                leverage_payload.data = models.MarginModeSchema(
                    symbol=symbol, margin_mode="isolated" if is_isolated else "crossed"
                ).dict()
                return leverage_payload

        return leverage_payload

    def set_collateral(self, coin: str, switch_on: bool = True) -> dict:
        """ """

        payload = {
            "coin": coin,
            "collateralSwitch": "ON" if switch_on else "OFF",
        }

        collateral_payload = self._post("/v5/account/set-collateral-switch", remove_none(payload))
        # print(f"??? set_position_mode:{mode_payload}")
        if collateral_payload.success:
            # mode_payload.data = models.PositionModeSchema(mode=mode, coin=coin, symbol=symbol).dict()
            return collateral_payload

        return collateral_payload

    def query_custom(self, url, *args, **kwargs):
        payload = {
            "category": self.DATA_TYPE,
        }
        for k, v in kwargs.items():
            payload[k] = v
        # print(f"payload:{payload}| args:{args} | kwargs:{kwargs}")

        api_ret = self._get(url, payload)

        return api_ret

    def query_funding_rate_history(
        self,
        symbol: str,
        start: datetime = None,
        end: datetime = None,
        limit: int = 200,
    ) -> dict:
        """
        Query historical funding rates for USDT and USDC perpetual / Inverse perpetual.

        Parameters
        ----------
        symbol : str
            Symbol name (e.g., "BTCUSDT", "ETHPERP"), case insensitive
        start : datetime, optional
            Start time for the query. If not provided, returns the latest records.
        end : datetime, optional
            End time for the query. If not provided, returns records up to current time.
        limit : int, optional
            Total number of records to return. Default 200. Will automatically paginate if needed.

        Returns
        -------
        dict
            CommonResponseSchema with funding rate history data

        Notes
        -----
        - Only supports "linear" and "inverse" data types (perpetual contracts)
        - Automatically handles pagination for long time periods
        - Each symbol has different funding intervals (typically 8 hours)
        - API returns max 200 records per request, so pagination is handled internally
        """

        def fetch_funding_page(payload, retry=3):
            """Fetch a single page of funding rate data with retry logic."""
            retry_counter = 0

            while retry_counter < retry:
                funding_rates = self._get("/v5/market/funding/history", remove_none(payload))

                if funding_rates.success:
                    return funding_rates
                else:
                    retry_counter += 1
                    if retry_counter >= retry:
                        break
                    time.sleep(0.1)  # Brief pause before retry

            return funding_rates

        # Validate data type
        if self.DATA_TYPE not in ["linear", "inverse"]:
            return models.CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg=f"Funding rate history is only available for linear and inverse perpetual contracts. Current data type: {self.DATA_TYPE}",
            )

        # Build the initial payload
        payload = {
            "category": self.DATA_TYPE,
            "symbol": symbol.upper(),
            "limit": min(200, limit),  # API max is 200 per request
        }

        # Convert datetime parameters to timestamps
        start_timestamp = None
        end_timestamp = None

        if start:
            start = UTC_TZ.localize(start) if start.tzinfo is None else start
            start_timestamp = int(datetime.timestamp(start) * 1000)
            payload["startTime"] = start_timestamp

        if end:
            end = UTC_TZ.localize(end) if end.tzinfo is None else end
            end_timestamp = int(datetime.timestamp(end) * 1000)
            payload["endTime"] = end_timestamp

        # Fetch the first page
        funding_rates = fetch_funding_page(payload)

        if not funding_rates.success:
            return funding_rates

        if not funding_rates.data:
            funding_rates.msg = "No funding rate history found"
            funding_rates.data = []
            return funding_rates

        # Extract the list from response
        all_funding_data = funding_rates.data.get("list", [])

        if not all_funding_data:
            funding_rates.data = []
            return funding_rates

        # Check if we need to paginate (when we have start time and got 200 records)
        if start_timestamp and len(all_funding_data) == 200 and len(all_funding_data) < limit:
            # Sort by timestamp to get the earliest timestamp for pagination
            all_funding_data.sort(key=lambda x: int(x["fundingRateTimestamp"]))

            # Continue fetching pages until we have enough data or reach the start time
            while len(all_funding_data) < limit:
                # Get the timestamp of the earliest record for the next page
                earliest_timestamp = int(all_funding_data[0]["fundingRateTimestamp"])

                # Check if we've reached or passed the start time
                if earliest_timestamp <= start_timestamp:
                    break

                # Update payload for the next page
                pagination_payload = payload.copy()
                pagination_payload["endTime"] = earliest_timestamp

                # Fetch the next page
                next_page = fetch_funding_page(pagination_payload)

                if not next_page.success or not next_page.data:
                    break

                next_data = next_page.data.get("list", [])
                if not next_data:
                    break

                # Add new data to the beginning (older records)
                all_funding_data = next_data + all_funding_data

                # Remove duplicates (same timestamp)
                seen_timestamps = set()
                unique_data = []
                for record in all_funding_data:
                    timestamp = record["fundingRateTimestamp"]
                    if timestamp not in seen_timestamps:
                        seen_timestamps.add(timestamp)
                        unique_data.append(record)

                all_funding_data = unique_data

                # If we got less than 200 records, we've reached the end
                if len(next_data) < 200:
                    break

                # Add a small delay to respect rate limits
                time.sleep(0.1)

        # Filter records by time range if both start and end are specified
        if start_timestamp and end_timestamp:
            all_funding_data = [
                record
                for record in all_funding_data
                if start_timestamp <= int(record["fundingRateTimestamp"]) <= end_timestamp
            ]
        elif start_timestamp:
            all_funding_data = [
                record for record in all_funding_data if int(record["fundingRateTimestamp"]) >= start_timestamp
            ]
        elif end_timestamp:
            all_funding_data = [
                record for record in all_funding_data if int(record["fundingRateTimestamp"]) <= end_timestamp
            ]

        # Limit the results to the requested amount
        if len(all_funding_data) > limit:
            # Sort by timestamp (newest first) and take the requested limit
            all_funding_data.sort(key=lambda x: int(x["fundingRateTimestamp"]), reverse=True)
            all_funding_data = all_funding_data[:limit]

        # Sort by timestamp (newest first as returned by API)
        all_funding_data.sort(key=lambda x: int(x["fundingRateTimestamp"]), reverse=True)

        # Update the response with all collected data
        funding_rates.data = all_funding_data

        # Process through the regular payload handler
        return self._regular_funding_rate_payload(funding_rates)


'''
class BybitTradeWebsocket(WebsocketClient):
    """
    Implement private topics
    doc: https://bybit-exchange.github.io/docs/linear/#t-privatetopics
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
        is_demo: bool = False,
        ping_interval: int = 15,
        debug: bool = False,
        logger=None,
        **kwargs,
    ) -> None:
        super().__init__(
            debug=debug,
            logger=logger,
        )
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
        self.debug = debug

    def info(self):
        return f"Bybit {self.DATA_TYPE} Trade Websocket Start"

    def custom_ping(self):
        return json.dumps({"op": "ping"})

    def connect(
        self,
        key: str,
        secret: str,
        is_testnet: bool = None,
        is_demo: bool = None,
        **kwargs,
    ):
        """
        Give api key and secret for Authorization.
        """
        self.key = key
        self.secret = secret
        if is_testnet is not None:
            self.is_testnet = is_testnet
        if is_demo is not None:
            self.is_demo = is_demo

        host = (
            BYBIT_TESTNET_WS_PRIVATE_HOST
            if self.is_testnet
            else BYBIT_DEMO_WS_PRIVATE_HOST
            if self.is_demo
            else BYBIT_WS_PRIVATE_HOST
        )
        self.init(
            host=host,
            proxy_host=self.proxy_host,
            proxy_port=self.proxy_port,
            ping_interval=self.ping_interval,
        )
        # self.init(host=host+"?max_alive_time=30s", proxy_host=self.proxy_host, proxy_port=self.proxy_port, ping_interval=self.ping_interval)
        self.start()

        # self.waitfor_connection()
        if self.waitfor_connection():
            self.logger.info(self.info())
        else:
            self.logger.info(self.info() + " failed")
            return False

        if not self._logged_in:
            self._login()
        self.logger.info(self.info())

        return True

    def _login(self):
        """
        Authorize websocket connection.
        """
        # Generate expires.
        expires = int((time.time() + 10) * 1000)
        api_key = self.key
        api_secret = self.secret

        if api_key is None or api_secret is None:
            self.logger.error(
                f"Bybit {self.DATA_TYPE} user trade data websocket - _login - api_key:{api_key}; api_secret:{api_secret}"
            )
            raise PermissionError(f"Authenticated endpoints require keys. api_key:{api_key}; api_secret:{api_secret}")

        # Generate signature.
        signature = str(
            hmac.new(
                bytes(api_secret, "utf-8"),
                bytes(f"GET/realtime{expires}", "utf-8"),
                digestmod="sha256",
            ).hexdigest()
        )
        """
        api_key = ""
        api_secret = ""

        # Generate expires.
        expires = int((time.time() + 1) * 1000)

        # Generate signature.
        signature = str(hmac.new(
            bytes(api_secret, "utf-8"),
            bytes(f"GET/realtime{expires}", "utf-8"), digestmod="sha256"
        ).hexdigest())

        # Authenticate with API.
        ws.send(
            json.dumps({
                "op": "auth",
                "args": [api_key, expires, signature]
            })
        )
        """
        req: dict = {"op": "auth", "args": [api_key, expires, signature]}
        # if self.debug:
        self.logger.info(f"Bybit {self.DATA_TYPE} user trade data websocket sending auth req:{req}")
        self.send_packet(req)
        self._logged_in = True

    def on_disconnected(self) -> None:
        self.logger.info(
            f"Bybit {self.DATA_TYPE} user trade data Websocket disconnected, now try to connect @ {datetime.now()}"
        )
        self._logged_in = False
        if self.on_disconnected_callback:
            self.on_disconnected_callback()

    def on_connected(self) -> None:
        if self.waitfor_connection():
            self.logger.info(f"Bybit {self.DATA_TYPE} user trade data websocket connect success")
        else:
            self.logger.error(f"Bybit {self.DATA_TYPE} user trade data websocket connect fail")

        if not self._logged_in:
            self._login()
            # self._logged_in = True
        # self.logger.debug("Bybit Future user trade data Websocket connect success")

        req: dict = self.sub_stream(channel=["trades", "orders", "account", "position"])
        # if self.debug:
        self.logger.debug(
            f"Bybit {self.DATA_TYPE} user trade data websocket on_connected sending subscription req:{req}"
        )

        self.send_packet(req)
        if self.on_connected_callback:
            self.on_connected_callback()

    def on_error(self, exception_value):
        super().on_error(exception_value)
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
            sub["args"] = [MAPPING_CHANNEL[c] for c in channel]
        else:
            mapping_channel = MAPPING_CHANNEL[channel]
            sub["args"] = [f"{mapping_channel}"]

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
        req: dict = self.sub_stream(channel=channel if channel else ["trades", "orders", "account", "position"])
        # if self.debug:
        self.logger.debug(f"Bybit {self.DATA_TYPE} user trade data websocket - subscribe - req: {req}")
        self.send_packet(req)

    def on_packet(self, packet: dict) -> None:
        """
        [DEBUG] Bybit linear Trade Websocket - on_packet event {'success': True, 'ret_msg': '', 'op': 'auth', 'conn_id': 'cg0qkmkpqfh3e1o3gm70-19i2'}
        [DEBUG] Bybit linear Trade Websocket - on_packet event {'success': True, 'ret_msg': '', 'op': 'subscribe', 'conn_id': 'cg0qkmkpqfh3e1o3gm70-19i2'}
        [DEBUG] Bybit linear Trade Websocket - on_packet event {'id': '11934815ff8f8d1-5afd-4d64-81ea-ccfa6f3a300d', 'topic': 'wallet', 'creationTime': 1678875089932, 'data': [{'accountIMRate': '0.0076', 'accountMMRate': '0.0004', 'totalEquity': '1643.81858535', 'totalWalletBalance': '1640.07424313', 'totalMarginBalance': '1643.81858535', 'totalAvailableBalance': '1631.16913997', 'totalPerpUPL': '3.74434221', 'totalInitialMargin': '12.64944538', 'totalMaintenanceMargin': '0.75116291', 'coin': [{'coin': 'USDC', 'equity': '0.92804784', 'usdValue': '0.92736525', 'walletBalance': '1.54616784', 'availableToWithdraw': '0', 'availableToBorrow': '1500000', 'borrowAmount': '0', 'accruedInterest': '0', 'totalOrderIM': '0', 'totalPositionIM': '3.14195033', 'totalPositionMM': '0.22731653', 'unrealisedPnl': '-0.61812', 'cumRealisedPnl': '1.54616784', 'bonus': '0'}, {'coin': 'USDT', 'equity': '1572.49398574', 'usdValue': '1580.50431727', 'walletBalance': '1568.15408574', 'availableToWithdraw': '1520.72009747', 'availableToBorrow': '2500000', 'borrowAmount': '0', 'accruedInterest': '0', 'totalOrderIM': '0', 'totalPositionIM': '9.46160827', 'totalPositionMM': '0.52135777', 'unrealisedPnl': '4.3399', 'cumRealisedPnl': '0.14828172', 'bonus': '0'}], 'accountType': 'UNIFIED'}]}

        """
        # if self.debug:
        #     self.logger.debug(f"Bybit {self.DATA_TYPE} Trade Websocket - on_packet event {packet}")
        if "auth" == packet.get("op", ""):
            self._logged_in = True if packet["success"] else False
            if self._logged_in:
                self.logger.info(f"Bybit {self.DATA_TYPE} Trade Websocket - successfully logined - {packet}")
            else:
                self.logger.error(f"Bybit {self.DATA_TYPE} Trade Websocket auth failed - on_packet event {packet}")
            return
        try:
            if "topic" not in packet:
                # self.logger.debug(f"[bybit] the other packet without topic: {packet}")
                return

            if not packet:
                self.logger.debug(f"Bybit {self.DATA_TYPE} Trade Websocket - unknown packet event {packet}")
                return

            if packet["topic"] == "wallet":
                self.on_account(packet)
            elif packet["topic"] == "order":
                self.on_order(packet)
            elif packet["topic"] == "execution":
                self.on_trade(packet)
            elif packet["topic"] == "position":
                self.on_position(packet)
            else:
                self.logger.info(f"Bybit {self.DATA_TYPE} Trade Websocket - the other packet type: {packet}")
        except Exception as e:
            self.logger.error(f"Bybit {self.DATA_TYPE} Trade Websocket - Error in on_packet: {e}. packet: {packet}")

    def on_account(self, packet: dict) -> None:
        """
        [DEBUG] Bybit linear Trade Websocket - on_packet event {'id': '11934812a537b05-b575-4ea5-8585-f731d62c455b', 'topic': 'wallet', 'creationTime': 1678887797421, 'data': [{'accountIMRate': '0.0092', 'accountMMRate': '0.0005', 'totalEquity': '1646.13810243', 'totalWalletBalance': '1640.20715187', 'totalMarginBalance': '1646.13810243', 'totalAvailableBalance': '1630.97132641', 'totalPerpUPL': '5.93095056', 'totalInitialMargin': '15.16677602', 'totalMaintenanceMargin': '0.88992747', 'coin': [{'coin': 'USDC', 'equity': '1.67796784', 'usdValue': '1.67838974', 'walletBalance': '1.54616784', 'availableToWithdraw': '0', 'availableToBorrow': '1500000', 'borrowAmount': '0', 'accruedInterest': '0', 'totalOrderIM': '0', 'totalPositionIM': '3.14195033', 'totalPositionMM': '0.22731653', 'unrealisedPnl': '0.1318', 'cumRealisedPnl': '1.54616784', 'bonus': '0'}, {'coin': 'USDT', 'equity': '1573.91127744', 'usdValue': '1581.26457518', 'walletBalance': '1568.13912744', 'availableToWithdraw': '1518.1986267', 'availableToBorrow': '2500000', 'borrowAmount': '0', 'accruedInterest': '0', 'totalOrderIM': '0', 'totalPositionIM': '11.96812074', 'totalPositionMM': '0.65947274', 'unrealisedPnl': '5.77215', 'cumRealisedPnl': '0.13332342', 'bonus': '0'}], 'accountType': 'UNIFIED'}]}
        """
        try:
            # self.logger.debug(f"packet on_account: {packet}")

            for acc_data in packet["data"]:
                account_type = acc_data["accountType"]
                datatype = "linear" if account_type == "UNIFIED" else "linear" if account_type == "CONTRACT" else "spot"
                if self.DATA_TYPE and (
                    (account_type == "UNIFIED" and self.DATA_TYPE not in ["linear", "spot", "option"])
                    or (account_type == "CONTRACT" and self.DATA_TYPE not in ["linear", "inverse"])
                    or (account_type == "SPOT" and self.DATA_TYPE not in ["spot"])
                ):
                    continue
                for data in acc_data.get("coin", []):
                    account = models.AccountSchema()  # .model_construct(data)
                    account = account.from_bybit_v2_to_form(data, datatype=datatype, account_type=account_type)
                    if account and account["symbol"] and self.on_account_callback:
                        self.on_account_callback(account)
        except Exception as e:
            self.on_error(e)

    def on_order(self, packet: dict) -> None:
        """
        [DEBUG] Bybit linear Trade Websocket - on_packet event {'id': '1193481acf42bfa-cbca-4232-93a2-dcf9dead9510', 'topic': 'order', 'creationTime': 1678887821206, 'data': [{'symbol': 'BTCUSDT', 'orderId': 'c2153424-5ab3-456c-b8e8-98bc266dad0e', 'side': 'Buy', 'orderType': 'Market', 'cancelType': 'UNKNOWN', 'price': '26140.4', 'qty': '0.001', 'orderIv': '', 'timeInForce': 'IOC', 'orderStatus': 'Filled', 'orderLinkId': '', 'lastPriceOnCreated': '24895.7', 'reduceOnly': False, 'leavesQty': '0', 'leavesValue': '0', 'cumExecQty': '0.001', 'cumExecValue': '24.8993', 'avgPrice': '24899.3', 'blockTradeId': '', 'positionIdx': 0, 'cumExecFee': '0.01493958', 'createdTime': '1678887821191', 'updatedTime': '1678887821200', 'rejectReason': 'EC_NoError', 'stopOrderType': '', 'triggerPrice': '', 'takeProfit': '', 'stopLoss': '', 'tpTriggerBy': '', 'slTriggerBy': '', 'triggerDirection': 0, 'triggerBy': '', 'closeOnTrigger': False, 'category': 'linear', 'isLeverage': ''}]}
        """
        try:
            # self.logger.debug(f"packet on_order: {packet}")
            datas: dict = packet["data"]

            # if datas["order_type"] not in ['LIMIT', 'Market']:
            #     return

            for data in datas:
                datatype = data["category"]
                if self.DATA_TYPE and datatype != self.DATA_TYPE:
                    continue
                order = models.OrderSchema()  # .model_construct(data)
                order = order.from_bybit_v2_to_form(data, datatype=datatype)
                if self.on_order_callback:  # and order_list
                    self.on_order_callback(order)

        except Exception as e:
            self.on_error(e)

    def on_trade(self, packet: dict) -> None:
        """
        [DEBUG] Bybit linear Trade Websocket - on_packet event {'id': '1193481299665b2-56be-40da-8a29-f488cb47190b', 'topic': 'execution', 'creationTime': 1678888392300, 'data': [{'category': 'linear', 'symbol': 'BTCUSDT', 'execFee': '0.01490148', 'execId': 'dae9c7a8-ceb9-51fa-8881-d5157ecc2c51', 'execPrice': '24835.8', 'execQty': '0.001', 'execType': 'Trade', 'execValue': '24.8358', 'isMaker': False, 'feeRate': '0.0006', 'tradeIv': '', 'markIv': '', 'blockTradeId': '', 'markPrice': '24883.13', 'indexPrice': '', 'underlyingPrice': '', 'leavesQty': '0', 'orderId': 'e9b9e551-081b-414c-a109-0e56d225ecd2', 'orderLinkId': '', 'orderPrice': '26116.9', 'orderQty': '0.001', 'orderType': 'Market', 'stopOrderType': 'UNKNOWN', 'side': 'Buy', 'execTime': '1678888392289', 'isLeverage': '0'}]}
        """
        try:
            # self.logger.debug(f"packet on_trade: {packet}")
            datas: dict = packet["data"]

            is_all_filled = False
            for data in datas:
                datatype = data["category"]
                if self.DATA_TYPE and datatype != self.DATA_TYPE:
                    continue
                trade = models.TradeSchema()  # .model_construct(data)
                trade = trade.from_bybit_v2_to_form(data, datatype=datatype)
                if trade and self.on_trade_callback:  # and trade_list:
                    # print(trade,'....')
                    self.on_trade_callback(trade)
        except Exception as e:
            self.on_error(e)

    def on_position(self, packet: dict) -> None:
        """
        [DEBUG] Bybit linear Trade Websocket - on_packet event {'id': '119348107a024b9-ed00-400d-b81c-105c5eb57e7e', 'topic': 'position', 'creationTime': 1678887821206, 'data': [{'positionIdx': 0, 'tradeMode': 0, 'riskId': 1, 'riskLimitValue': '2000000', 'symbol': 'BTCUSDT', 'side': 'Buy', 'size': '0.006', 'entryPrice': '23989.61666667', 'leverage': '10', 'positionValue': '143.9377', 'markPrice': '24922.99', 'positionIM': '14.47149636', 'positionMM': '0.79741486', 'takeProfit': '0', 'stopLoss': '0', 'trailingStop': '0', 'unrealisedPnl': '5.60024', 'cumRealisedPnl': '0.11838384', 'createdTime': '1676887474897', 'updatedTime': '1678887821202', 'tpslMode': 'Full', 'liqPrice': '', 'bustPrice': '', 'positionStatus': 'Normal'}]}
        """
        try:
            # self.logger.debug(f"packet on_position: {packet}")
            datas: dict = packet["data"]
            for data in datas:
                # position_value = float(data["position_value"])
                # if not position_value:
                #     continue
                datatype = data.get("category", "linear")
                if not datatype:
                    datatype = "linear"
                if self.DATA_TYPE and datatype != self.DATA_TYPE:
                    continue
                """
                Unified account: does not have this field
                Unified account - inverse & Normal account: linear, inverse.
                """
                position = models.PositionSchema()  # .model_construct(data)
                position = position.from_bybit_v2_to_form(data, datatype=datatype)
                if position and self.on_position_callback:
                    self.on_position_callback(position)
        except Exception as e:
            self.on_error(e)

'''


class BybitDataWebsocketTesting(WebsocketClient):
    """
    IP Limits
        Do not frequently connect and disconnect the connection.
        Do not build over 500 connections in 5 minutes. This is counted separately per WebSocket endpoint.

    Public channel - Args limits
        Spot can input up to 10 args for a single subscription connection
        Options can input up to 2000 args for a single subscription connection
        No args limit for Derivatives for now

    To avoid network or program issues, we recommend that you send the ping heartbeat packet every 20 seconds to maintain the WebSocket connection.
    """

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
        self.logger = get_logger() if logger is None else logger

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

    def info(self):
        return f"Bybit {self.DATA_TYPE} Data Websocket start"

    def connect(self):
        """ """
        self.ws_public = WebSocket(
            testnet=self.is_testnet,
            channel_type=self.DATA_TYPE,
        )

        self.logger.info(self.info())

    def on_connected(self) -> None:
        """"""
        self.logger.info(f"Bybit {self.DATA_TYPE} Market data websocket connect success")

        if self.on_connected_callback:
            self.on_connected_callback()

    def on_disconnected(self) -> None:
        self.logger.info(
            f"Bybit {self.DATA_TYPE} market data Websocket disconnected, now try to connect @ {datetime.now()}"
        )
        if self.on_disconnected_callback:
            self.on_disconnected_callback()

    def subscribe(
        self,
        symbols: list | str,
        on_tick: Callable = None,
        channel="depth",
        interval="1m",
        depth=1,
    ) -> None:
        """
        channel: 'depth', 'kline'
        """
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
            self.logger.error("invalid input symbols.")
            return

        for symbol in symbols:
            if symbol in self.ticks:
                return

            # tick data dict
            asset_type = models.bybit_asset_type(symbol=symbol, datatype=self.DATA_TYPE)
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
            req: dict = self.sub_stream(symbol=symbol, channel=channel, interval=interval, depth=depth)
            self.send_packet(req)

    def on_packet(self, packet: dict) -> None:
        """on_packet
        >> packet: {'success': True, 'ret_msg': '', 'conn_id': 'f02b82c1-be41-4fae-b69c-6c4a57ed6443', 'req_id': '', 'op': 'subscribe'}
        >> packet: {'success': False, 'ret_msg': 'error:already subscribed,topic:kline.1.BTCUSDT', 'conn_id': 'f02b82c1-be41-4fae-b69c-6c4a57ed6443', 'req_id': '', 'op': 'subscribe'}
        kline:
        >> packet: {'topic': 'kline.1.BTCUSDT', 'data': [{'start': 1678867020000, 'end': 1678867079999, 'interval': '1', 'open': '24858', 'close': '24887.6', 'high': '24899', 'low': '24857.9', 'volume': '296.899', 'turnover': '7387562.6728', 'confirm': False, 'timestamp': 1678867067260}], 'ts': 1678867067260, 'type': 'snapshot'}
        >> packet: {'topic': 'kline.1.BTCUSDT', 'data': [{'start': 1678867020000, 'end': 1678867079999, 'interval': '1', 'open': '24858', 'close': '24889.1', 'high': '24899', 'low': '24857.9', 'volume': '296.343', 'turnover': '7373724.7014', 'confirm': False, 'timestamp': 1678867066259}], 'ts': 1678867066259, 'type': 'snapshot'}

        orderbook:
        >> packet: {'topic': 'orderbook.1.BTCUSDT', 'type': 'snapshot', 'ts': 1678867506163, 'data': {'s': 'BTCUSDT', 'b': [['24896.80', '7.134']], 'a': [['24896.90', '6.155']], 'u': 21331291, 'seq': 46077635974}}
        >> packet: {'topic': 'orderbook.1.BTCUSDT', 'type': 'delta', 'ts': 1678867506171, 'data': {'s': 'BTCUSDT', 'b': [['24896.80', '9.634']], 'a': [['24896.90', '6.055']], 'u': 21331292, 'seq': 46077636001}}
        >> packet: {'topic': 'orderbook.1.BTCUSDT', 'type': 'delta', 'ts': 1678867506181, 'data': {'s': 'BTCUSDT', 'b': [['24896.80', '7.134']], 'a': [['24896.90', '6.035']], 'u': 21331293, 'seq': 46077636011}}
        >> packet: {'topic': 'orderbook.1.BTCUSDT', 'type': 'delta', 'ts': 1678867506191, 'data': {'s': 'BTCUSDT', 'b': [['24896.80', '7.154']], 'a': [], 'u': 21331294, 'seq': 46077636017}}
        >> packet: {'topic': 'orderbook.1.BTCUSDT', 'type': 'delta', 'ts': 1678867506201, 'data': {'s': 'BTCUSDT', 'b': [], 'a': [['24896.90', '5.715']], 'u': 21331295, 'seq': 46077636029}}
        >> packet: {'topic': 'orderbook.1.BTCUSDT', 'type': 'delta', 'ts': 1678867506211, 'data': {'s': 'BTCUSDT', 'b': [['24896.80', '7.174']], 'a': [['24896.90', '5.615']], 'u': 21331296, 'seq': 46077636038}}
        >> packet: {'topic': 'orderbook.1.BTCUSDT', 'type': 'delta', 'ts': 1678867506231, 'data': {'s': 'BTCUSDT', 'b': [['24896.80', '7.182']], 'a': [], 'u': 21331297, 'seq': 46077636052}}
        >> packet: {'topic': 'orderbook.1.BTCUSDT', 'type': 'delta', 'ts': 1678867506242, 'data': {'s': 'BTCUSDT', 'b': [['24896.80', '7.186']], 'a': [], 'u': 21331298, 'seq': 46077636055}}

        tickers:
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'snapshot', 'data': {'symbol': 'BTCUSDT', 'tickDirection': 'ZeroPlusTick', 'price24hPcnt': '0.02524', 'lastPrice': '24883.40', 'prevPrice24h': '24270.80', 'highPrice24h': '26500.00', 'lowPrice24h': '23966.00', 'prevPrice1h': '24803.20', 'markPrice': '24882.60', 'indexPrice': '24865.13', 'openInterest': '46517.745', 'openInterestValue': '1157482441.74', 'turnover24h': '11395433060.876307', 'volume24h': '453267.97999999', 'nextFundingTime': '1678896000000', 'fundingRate': '0.000131', 'bid1Price': '24883.40', 'bid1Size': '18.452', 'ask1Price': '24883.50', 'ask1Size': '3.989'}, 'cs': 46077820483, 'ts': 1678867659469}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'tickDirection': 'PlusTick', 'price24hPcnt': '0.025244', 'lastPrice': '24883.50', 'turnover24h': '11395433508.779005', 'volume24h': '453267.99799999', 'fundingRate': '0.000131', 'bid1Price': '24883.40', 'bid1Size': '18.310', 'ask1Price': '24883.50', 'ask1Size': '4.090'}, 'cs': 46077820675, 'ts': 1678867659669}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'bid1Price': '24883.40', 'bid1Size': '18.310', 'ask1Price': '24883.50', 'ask1Size': '3.974'}, 'cs': 46077820723, 'ts': 1678867659769}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'bid1Price': '24883.40', 'bid1Size': '18.310', 'ask1Price': '24883.50', 'ask1Size': '3.974'}, 'cs': 46077820780, 'ts': 1678867659869}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'bid1Price': '24883.40', 'bid1Size': '18.310', 'ask1Price': '24883.50', 'ask1Size': '3.974'}, 'cs': 46077820818, 'ts': 1678867659968}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'bid1Price': '24883.40', 'bid1Size': '16.665', 'ask1Price': '24883.50', 'ask1Size': '5.494'}, 'cs': 46077820870, 'ts': 1678867660069}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'tickDirection': 'MinusTick', 'price24hPcnt': '0.02524', 'lastPrice': '24883.40', 'turnover24h': '11395435126.123806', 'volume24h': '453268.06299999', 'fundingRate': '0.000131', 'bid1Price': '24883.40', 'bid1Size': '19.635', 'ask1Price': '24883.50', 'ask1Size': '5.194'}, 'cs': 46077820977, 'ts': 1678867660168}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'bid1Price': '24883.40', 'bid1Size': '19.635', 'ask1Price': '24883.50', 'ask1Size': '5.194'}, 'cs': 46077821019, 'ts': 1678867660268}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'bid1Price': '24883.40', 'bid1Size': '19.668', 'ask1Price': '24883.50', 'ask1Size': '4.891'}, 'cs': 46077821082, 'ts': 1678867660368}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'price24hPcnt': '0.02524', 'markPrice': '24883.40', 'indexPrice': '24866.13', 'openInterestValue': '1157519655.93', 'fundingRate': '0.000131', 'bid1Price': '24883.40', 'bid1Size': '19.655', 'ask1Price': '24883.50', 'ask1Size': '4.891'}, 'cs': 46077821141, 'ts': 1678867660469}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'tickDirection': 'ZeroPlusTick', 'price24hPcnt': '0.025396', 'lastPrice': '24887.20', 'turnover24h': '11395447793.695705', 'volume24h': '453268.57199999', 'fundingRate': '0.000131', 'bid1Price': '24887.10', 'bid1Size': '23.211', 'ask1Price': '24887.20', 'ask1Size': '9.824'}, 'cs': 46077822776, 'ts': 1678867660868}
        """
        # self.logger.debug(f'packet: {packet}')
        # print(f'>> packet: {packet}')

        if "topic" not in packet:
            # print(f"[bybit] the other packet without topic: {packet}")
            return

        channel: str = packet["topic"].split(".")[0]
        symbol: str = packet["topic"].split(".")[-1]
        # params: dict = packet['params']
        # tick: dict = self.ticks[symbol]
        # print(packet, tick, symbol, channel,'~~~')

        is_fire: bool = False
        # kline = candle
        tick = None
        if channel == "kline":
            interval = KBAR_INTERVAL_REV[packet["topic"].split(".")[1]]
            tick: dict = self.ticks[f"{symbol}|{interval}"]

            data: dict = packet["data"][0]
            tick["volume"] = float(data["volume"])
            tick["turnover"] = float(data["turnover"])
            tick["open"] = float(data["open"])
            tick["high"] = float(data["high"])
            tick["low"] = float(data["low"])
            tick["close"] = float(data["close"])
            tick["start"] = models.timestamp_to_datetime(float(data["start"]))
            tick["datetime"] = models.generate_datetime()  # datetime.now()

            if self.on_kline_callback:
                is_fire = True
                self.on_kline_callback(copy(tick))

        elif channel == "tickers":
            tick: dict = self.ticks[f"{symbol}|ticker"]
            tick.update(self.handle_tickerdata(packet, symbol=symbol))

            if self.on_ticker_callback:
                is_fire = True
                self.on_ticker_callback(copy(tick))

        elif channel == "orderbook":
            current_time = time.monotonic()
            elapsed_time = current_time - self.last_time

            # save asks / bids in tick for next iteration
            tick: dict = self.ticks[f"{symbol}|depth"]
            tick.update(self.handle_orderbook(packet, symbol=symbol))

            if elapsed_time >= self.speed / 100:
                if self.on_depth_callback:
                    is_fire = True
                    self.on_depth_callback(copy(tick))

                self.last_time = current_time

        if not is_fire and self.on_tick and tick:
            self.on_tick(copy(tick))

    def handle_kline(self, packet, symbol):
        pass

    def handle_tickerdata(self, packet, symbol):
        # Record the initial snapshot.
        if "snapshot" in packet["type"]:
            self.ticker_data[symbol] = packet["data"]

        # Make updates according to delta response.
        elif "delta" in packet["type"]:
            for key, value in packet["data"].items():
                self.ticker_data[symbol][key] = value
        ts: int = packet["ts"]

        """
        >>>> CONTRACT:
        {
            "topic": "tickers.BTCUSDT",
            "type": "snapshot",
            "data": {
                "symbol": "BTCUSDT",
                "tickDirection": "PlusTick",
                "price24hPcnt": "0.017103",
                "lastPrice": "17216.00",
                "prevPrice24h": "16926.50",
                "highPrice24h": "17281.50",
                "lowPrice24h": "16915.00",
                "prevPrice1h": "17238.00",
                "markPrice": "17217.33",
                "indexPrice": "17227.36",
                "openInterest": "68744.761",
                "openInterestValue": "1183601235.91",
                "turnover24h": "1570383121.943499",
                "volume24h": "91705.276",
                "nextFundingTime": "1673280000000",
                "fundingRate": "-0.000212",
                "bid1Price": "17215.50",
                "bid1Size": "84.489",
                "ask1Price": "17216.00",
                "ask1Size": "83.020"
            },
            "cs": 24987956059,
            "ts": 1673272861686
        }

        >>>> OPTION:
        {
            "id": "tickers.BTC-6JAN23-17500-C-2480334983-1672917511074",
            "topic": "tickers.BTC-6JAN23-17500-C",
            "ts": 1672917511074,
            "data": {
                "symbol": "BTC-6JAN23-17500-C",
                "bidPrice": "0",
                "bidSize": "0",
                "bidIv": "0",
                "askPrice": "10",
                "askSize": "5.1",
                "askIv": "0.514",
                "lastPrice": "10",
                "highPrice24h": "25",
                "lowPrice24h": "5",
                "markPrice": "7.86976724",
                "indexPrice": "16823.73",
                "markPriceIv": "0.4896",
                "underlyingPrice": "16815.1",
                "openInterest": "49.85",
                "turnover24h": "446802.8473",
                "volume24h": "26.55",
                "totalVolume": "86",
                "totalTurnover": "1437431",
                "delta": "0.047831",
                "gamma": "0.00021453",
                "vega": "0.81351067",
                "theta": "-19.9115368",
                "predictedDeliveryPrice": "0",
                "change24h": "-0.33333334"
            },
            "type": "snapshot"
        }
        >>>> SPOT:
        {
            "topic": "tickers.BTCUSDT",
            "ts": 1673853746003,
            "type": "snapshot",
            "cs": 2588407389,
            "data": {
                "symbol": "BTCUSDT",
                "lastPrice": "21109.77",
                "highPrice24h": "21426.99",
                "lowPrice24h": "20575",
                "prevPrice24h": "20704.93",
                "volume24h": "6780.866843",
                "turnover24h": "141946527.22907118",
                "price24hPcnt": "0.0196",
                "usdIndexPrice": "21120.2400136"
            }
        }
        """
        data = copy(self.ticker_data[symbol])
        result = {}
        if self.DATA_TYPE in ["linear", "inverse", "spot"]:
            result["prev_open_24h"] = None
            result["prev_high_24h"] = float(data["highPrice24h"])
            result["prev_low_24h"] = float(data["lowPrice24h"])
            result["prev_close_24h"] = float(data["prevPrice24h"])
            result["prev_volume_24h"] = float(data["volume24h"])
            result["prev_turnover_24h"] = float(data["turnover24h"])
            result["open_interest"] = float(data["openInterest"])

            result["last_price"] = float(data["lastPrice"])
            result["price_change_pct"] = float(data["price24hPcnt"])
            result["price_change"] = result["last_price"] - result["prev_close_24h"]
            result["bid_price_1"] = float(data["bid1Price"]) if "bid1Price" in data else None
            result["bid_volume_1"] = float(data["bid1Size"]) if "bid1Size" in data else None
            result["ask_price_1"] = float(data["ask1Price"]) if "ask1Price" in data else None
            result["ask_volume_1"] = float(data["ask1Size"]) if "ask1Size" in data else None
        elif self.DATA_TYPE in ["option"]:
            result["prev_open_24h"] = None
            result["prev_high_24h"] = float(data["highPrice24h"])
            result["prev_low_24h"] = float(data["lowPrice24h"])
            result["prev_close_24h"] = None
            result["prev_volume_24h"] = float(data["volume24h"])
            result["prev_turnover_24h"] = float(data["turnover24h"])
            result["open_interest"] = float(data["openInterest"])

            result["bid_price_1"] = float(data["bidPrice"]) if "bidPrice" in data else None
            result["bid_volume_1"] = float(data["bidSize"]) if "bidSize" in data else None
            result["bid_iv"] = float(data["bidIv"])
            result["ask_price_1"] = float(data["askPrice"]) if "askPrice" in data else None
            result["ask_volume_1"] = float(data["askSize"]) if "askSize" in data else None
            result["ask_iv"] = float(data["askIv"])

            result["volume"] = float(data["totalVolume"])
            result["last_price"] = float(data["lastPrice"])
            result["price_change_pct"] = float(data["change24h"])  ## TODO: TBC, price change or pct change?
            result["price_change"] = None
            result["delta"] = float(data["delta"])
            result["gamma"] = float(data["gamma"])
            result["vega"] = float(data["vega"])
            result["theta"] = float(data["theta"])
            result["underlying_price"] = float(data["underlyingPrice"])
            result["index_price"] = float(data["indexPrice"])
            result["mark_price"] = float(data["markPrice"])
            result["mark_iv"] = float(data["markPriceIv"])

        result["datetime_now"] = models.generate_datetime()
        result["datetime"] = models.timestamp_to_datetime(ts)
        result["ts"] = ts
        return result

    def handle_orderbook(self, packet, symbol):
        # call by reference
        data: dict = packet[
            "data"
        ]  # The data is ordered by price, starting with the lowest buys and ending with the highest sells.
        ts: int = packet["ts"]
        # Record the initial snapshot.
        if packet["type"] == "snapshot":
            self.bids_asks_data[symbol]["bids"] = [
                d for d in data["b"]
            ]  # BUY side - the element is sorted by price in descending order
            self.bids_asks_data[symbol]["asks"] = [
                d for d in data["a"]
            ]  # SELL3 side - the element is sorted by price in ascending order

        # Make updates according to delta response.
        elif packet["type"] == "delta":
            book_sides = {"bids": data["b"], "asks": data["a"]}
            # print(f">>> data: {data}")
            for side, entries in book_sides.items():
                for entry in entries:
                    # Delete.
                    if float(entry[1]) == 0:
                        index = find_index(self.bids_asks_data[symbol][side], entry, 0)
                        # print(f"{side} DELETE - entry:{entry}; index:{index}; len:{len(self.bids_asks_data[symbol][side])}")
                        self.bids_asks_data[symbol][side].pop(index)
                        continue

                    # Insert.
                    price_level_exists = entry[0] in [level[0] for level in self.bids_asks_data[symbol][side]]
                    if not price_level_exists:
                        # print(f"{side} INSERT - entry:{entry}; len:{len(self.bids_asks_data[symbol][side])}")
                        self.bids_asks_data[symbol][side].append(entry)
                        continue

                    # Update.
                    qty_changed = entry[1] != next(
                        level[1] for level in self.bids_asks_data[symbol][side] if level[0] == entry[0]
                    )
                    if price_level_exists and qty_changed:
                        index = find_index(self.bids_asks_data[symbol][side], entry, 0)
                        # print(f"{side} UPDATE - entry:{entry}; index:{index}; len:{len(self.bids_asks_data[symbol][side])}")
                        self.bids_asks_data[symbol][side][index] = entry
                        continue

        # print('len of asks', len(self.bids_asks_data[symbol]['asks']),'len of bids', len(self.bids_asks_data[symbol]['bids']),'||||asks:',self.bids_asks_data[symbol]['asks'],'||||bids:',self.bids_asks_data[symbol]['bids'],'|||raw:',data)
        # print(self.bids_asks_data[symbol]['bids'][:5], packet["type"], self.bids_asks_data[symbol]['asks'][:5])

        # computer the bids and asks with current data
        bids: list = self.bids_asks_data[symbol]["bids"]
        for n in range(min(5, len(bids))):
            price = bids[n][0]
            volume = bids[n][1]
            # price = bids[n]["price"]
            # volume = bids[n]["size"]
            self.bids_asks_data[symbol]["bid_price_" + str(n + 1)] = float(price)
            self.bids_asks_data[symbol]["bid_volume_" + str(n + 1)] = float(volume)

        asks: list = self.bids_asks_data[symbol]["asks"]
        for n in range(min(5, len(asks))):
            price = asks[n][0]
            volume = asks[n][1]
            # price = asks[n]["price"]
            # volume = asks[n]["size"]
            self.bids_asks_data[symbol]["ask_price_" + str(n + 1)] = float(price)
            self.bids_asks_data[symbol]["ask_volume_" + str(n + 1)] = float(volume)

        self.bids_asks_data[symbol]["datetime_now"] = models.generate_datetime()
        self.bids_asks_data[symbol]["datetime"] = models.timestamp_to_datetime(ts)
        self.bids_asks_data[symbol]["ts"] = ts

        result = copy(self.bids_asks_data[symbol])
        result.pop("bids")
        result.pop("asks")
        return result


class BybitDataWebsocket(WebsocketClient):
    """
    IP Limits
        Do not frequently connect and disconnect the connection.
        Do not build over 500 connections in 5 minutes. This is counted separately per WebSocket endpoint.

    Public channel - Args limits
        Spot can input up to 10 args for a single subscription connection
        Options can input up to 2000 args for a single subscription connection
        No args limit for Derivatives for now

    To avoid network or program issues, we recommend that you send the ping heartbeat packet every 20 seconds to maintain the WebSocket connection.
    """

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

    def info(self):
        return f"Bybit {self.DATA_TYPE} Data Websocket start"

    def custom_ping(self):
        return json.dumps({"op": "ping", "req_id": str(int(time.time()))})

    def connect(self):
        """ """
        if self.DATA_TYPE == "spot":
            host = BYBIT_TESTNET_WS_SPOT_PUBLIC_HOST if self.is_testnet else BYBIT_WS_SPOT_PUBLIC_HOST
        elif self.DATA_TYPE == "linear":
            host = BYBIT_TESTNET_WS_FUTURE_PUBLIC_HOST if self.is_testnet else BYBIT_WS_FUTURE_PUBLIC_HOST
        elif self.DATA_TYPE == "inverse":
            host = BYBIT_TESTNET_WS_INVERSE_PUBLIC_HOST if self.is_testnet else BYBIT_WS_INVERSE_PUBLIC_HOST
        elif self.DATA_TYPE == "option":
            host = BYBIT_TESTNET_WS_OPTION_PUBLIC_HOST if self.is_testnet else BYBIT_WS_OPTION_PUBLIC_HOST

        self.init(
            host=host,
            proxy_host=self.proxy_host,
            proxy_port=self.proxy_port,
            ping_interval=self.ping_interval,
        )
        self.start()

        if self.waitfor_connection():
            self.logger.info(self.info())
            return True
        else:
            self.logger.info(self.info() + " failed")
            return False

    def on_connected(self) -> None:
        """"""
        if self.waitfor_connection():
            self.logger.info(f"Bybit {self.DATA_TYPE} Market data websocket connect success")
        else:
            self.logger.error(f"Bybit {self.DATA_TYPE} Market data websocket connect fail")

        print(f"===> [DEBUG] Bybit {self.DATA_TYPE} Market data on_connected - self.ticks: {self.ticks}")

        # resubscribe
        if self.ticks:
            channels = []
            for key, detail in self.ticks.items():
                req: dict = self.sub_stream(
                    symbol=detail["symbol"],
                    channel=detail["channel"],
                    interval=detail.get("interval"),
                )
                channels += req["args"]

            req: dict = {
                "op": "subscribe",
                "args": channels,
            }
            self.logger.info(f"on_connected, resubscribing the channels - req:{req}")
            self.send_packet(req)

        if self.on_connected_callback:
            self.on_connected_callback()

    def on_disconnected(self) -> None:
        self.logger.info(
            f"Bybit {self.DATA_TYPE} market data Websocket disconnected, now try to connect @ {datetime.now()}"
        )
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

    def sub_stream(self, symbol, channel="depth", interval="1m", depth=1):
        """ """
        self.reqid += 1
        mapping_channel = MAPPING_CHANNEL[channel]
        sub = {
            "op": "subscribe",
        }

        if channel == "kline":
            sub["args"] = [f"{mapping_channel}.{KBAR_INTERVAL[interval]}.{symbol}"]
        elif channel == "depth":
            """
            Linear & inverse:
            Level 1 data, push frequency: 10ms
            Level 50 data, push frequency: 20ms
            Level 200 data, push frequency: 100ms
            Level 500 data, push frequency: 100ms

            Spot:
            Level 1 data, push frequency: 10ms
            Level 50 data, push frequency: 20ms

            Option:
            Level 25 data, push frequency: 20ms
            Level 100 data, push frequency: 100ms
            """
            depth = 25 if self.DATA_TYPE == "option" else depth
            sub["args"] = [f"{mapping_channel}.{depth}.{symbol}"]
            # sub["args"] = [f"{mapping_channel}.1.{symbol}"]
            # sub["args"] = [f"{mapping_channel}.50.{symbol}"]
            # sub["args"] = [f"{mapping_channel}.200.{symbol}"]
        elif channel == "ticker":
            sub["args"] = [f"{mapping_channel}.{symbol}"]

        # print(f'req_id:{self.reqid}!!!!!!!!!!symbol:{symbol}; channel:{channel}')
        return sub

    def subscribe(
        self,
        symbols: list | str,
        on_tick: Callable = None,
        channel="depth",
        interval="1m",
        depth=1,
    ) -> None:
        """
        channel: 'depth', 'kline'
        """
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
            self.logger.error("invalid input symbols.")
            return

        for symbol in symbols:
            if symbol in self.ticks:
                return

            # tick data dict
            asset_type = models.bybit_asset_type(symbol=symbol, datatype=self.DATA_TYPE)
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
            req: dict = self.sub_stream(symbol=symbol, channel=channel, interval=interval, depth=depth)
            self.send_packet(req)

    def on_packet(self, packet: dict) -> None:
        """on_packet
        >> packet: {'success': True, 'ret_msg': '', 'conn_id': 'f02b82c1-be41-4fae-b69c-6c4a57ed6443', 'req_id': '', 'op': 'subscribe'}
        >> packet: {'success': False, 'ret_msg': 'error:already subscribed,topic:kline.1.BTCUSDT', 'conn_id': 'f02b82c1-be41-4fae-b69c-6c4a57ed6443', 'req_id': '', 'op': 'subscribe'}
        kline:
        >> packet: {'topic': 'kline.1.BTCUSDT', 'data': [{'start': 1678867020000, 'end': 1678867079999, 'interval': '1', 'open': '24858', 'close': '24887.6', 'high': '24899', 'low': '24857.9', 'volume': '296.899', 'turnover': '7387562.6728', 'confirm': False, 'timestamp': 1678867067260}], 'ts': 1678867067260, 'type': 'snapshot'}
        >> packet: {'topic': 'kline.1.BTCUSDT', 'data': [{'start': 1678867020000, 'end': 1678867079999, 'interval': '1', 'open': '24858', 'close': '24889.1', 'high': '24899', 'low': '24857.9', 'volume': '296.343', 'turnover': '7373724.7014', 'confirm': False, 'timestamp': 1678867066259}], 'ts': 1678867066259, 'type': 'snapshot'}

        orderbook:
        >> packet: {'topic': 'orderbook.1.BTCUSDT', 'type': 'snapshot', 'ts': 1678867506163, 'data': {'s': 'BTCUSDT', 'b': [['24896.80', '7.134']], 'a': [['24896.90', '6.155']], 'u': 21331291, 'seq': 46077635974}}
        >> packet: {'topic': 'orderbook.1.BTCUSDT', 'type': 'delta', 'ts': 1678867506171, 'data': {'s': 'BTCUSDT', 'b': [['24896.80', '9.634']], 'a': [['24896.90', '6.055']], 'u': 21331292, 'seq': 46077636001}}
        >> packet: {'topic': 'orderbook.1.BTCUSDT', 'type': 'delta', 'ts': 1678867506181, 'data': {'s': 'BTCUSDT', 'b': [['24896.80', '7.134']], 'a': [['24896.90', '6.035']], 'u': 21331293, 'seq': 46077636011}}
        >> packet: {'topic': 'orderbook.1.BTCUSDT', 'type': 'delta', 'ts': 1678867506191, 'data': {'s': 'BTCUSDT', 'b': [['24896.80', '7.154']], 'a': [], 'u': 21331294, 'seq': 46077636017}}
        >> packet: {'topic': 'orderbook.1.BTCUSDT', 'type': 'delta', 'ts': 1678867506201, 'data': {'s': 'BTCUSDT', 'b': [], 'a': [['24896.90', '5.715']], 'u': 21331295, 'seq': 46077636029}}
        >> packet: {'topic': 'orderbook.1.BTCUSDT', 'type': 'delta', 'ts': 1678867506211, 'data': {'s': 'BTCUSDT', 'b': [['24896.80', '7.174']], 'a': [['24896.90', '5.615']], 'u': 21331296, 'seq': 46077636038}}
        >> packet: {'topic': 'orderbook.1.BTCUSDT', 'type': 'delta', 'ts': 1678867506231, 'data': {'s': 'BTCUSDT', 'b': [['24896.80', '7.182']], 'a': [], 'u': 21331297, 'seq': 46077636052}}
        >> packet: {'topic': 'orderbook.1.BTCUSDT', 'type': 'delta', 'ts': 1678867506242, 'data': {'s': 'BTCUSDT', 'b': [['24896.80', '7.186']], 'a': [], 'u': 21331298, 'seq': 46077636055}}

        tickers:
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'snapshot', 'data': {'symbol': 'BTCUSDT', 'tickDirection': 'ZeroPlusTick', 'price24hPcnt': '0.02524', 'lastPrice': '24883.40', 'prevPrice24h': '24270.80', 'highPrice24h': '26500.00', 'lowPrice24h': '23966.00', 'prevPrice1h': '24803.20', 'markPrice': '24882.60', 'indexPrice': '24865.13', 'openInterest': '46517.745', 'openInterestValue': '1157482441.74', 'turnover24h': '11395433060.876307', 'volume24h': '453267.97999999', 'nextFundingTime': '1678896000000', 'fundingRate': '0.000131', 'bid1Price': '24883.40', 'bid1Size': '18.452', 'ask1Price': '24883.50', 'ask1Size': '3.989'}, 'cs': 46077820483, 'ts': 1678867659469}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'tickDirection': 'PlusTick', 'price24hPcnt': '0.025244', 'lastPrice': '24883.50', 'turnover24h': '11395433508.779005', 'volume24h': '453267.99799999', 'fundingRate': '0.000131', 'bid1Price': '24883.40', 'bid1Size': '18.310', 'ask1Price': '24883.50', 'ask1Size': '4.090'}, 'cs': 46077820675, 'ts': 1678867659669}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'bid1Price': '24883.40', 'bid1Size': '18.310', 'ask1Price': '24883.50', 'ask1Size': '3.974'}, 'cs': 46077820723, 'ts': 1678867659769}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'bid1Price': '24883.40', 'bid1Size': '18.310', 'ask1Price': '24883.50', 'ask1Size': '3.974'}, 'cs': 46077820780, 'ts': 1678867659869}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'bid1Price': '24883.40', 'bid1Size': '18.310', 'ask1Price': '24883.50', 'ask1Size': '3.974'}, 'cs': 46077820818, 'ts': 1678867659968}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'bid1Price': '24883.40', 'bid1Size': '16.665', 'ask1Price': '24883.50', 'ask1Size': '5.494'}, 'cs': 46077820870, 'ts': 1678867660069}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'tickDirection': 'MinusTick', 'price24hPcnt': '0.02524', 'lastPrice': '24883.40', 'turnover24h': '11395435126.123806', 'volume24h': '453268.06299999', 'fundingRate': '0.000131', 'bid1Price': '24883.40', 'bid1Size': '19.635', 'ask1Price': '24883.50', 'ask1Size': '5.194'}, 'cs': 46077820977, 'ts': 1678867660168}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'bid1Price': '24883.40', 'bid1Size': '19.635', 'ask1Price': '24883.50', 'ask1Size': '5.194'}, 'cs': 46077821019, 'ts': 1678867660268}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'bid1Price': '24883.40', 'bid1Size': '19.668', 'ask1Price': '24883.50', 'ask1Size': '4.891'}, 'cs': 46077821082, 'ts': 1678867660368}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'price24hPcnt': '0.02524', 'markPrice': '24883.40', 'indexPrice': '24866.13', 'openInterestValue': '1157519655.93', 'fundingRate': '0.000131', 'bid1Price': '24883.40', 'bid1Size': '19.655', 'ask1Price': '24883.50', 'ask1Size': '4.891'}, 'cs': 46077821141, 'ts': 1678867660469}
        >> packet: {'topic': 'tickers.BTCUSDT', 'type': 'delta', 'data': {'symbol': 'BTCUSDT', 'tickDirection': 'ZeroPlusTick', 'price24hPcnt': '0.025396', 'lastPrice': '24887.20', 'turnover24h': '11395447793.695705', 'volume24h': '453268.57199999', 'fundingRate': '0.000131', 'bid1Price': '24887.10', 'bid1Size': '23.211', 'ask1Price': '24887.20', 'ask1Size': '9.824'}, 'cs': 46077822776, 'ts': 1678867660868}
        """
        # self.logger.debug(f'packet: {packet}')
        # print(f'>> packet: {packet}')

        if "topic" not in packet:
            # print(f"[bybit] the other packet without topic: {packet}")
            return

        channel: str = packet["topic"].split(".")[0]
        symbol: str = packet["topic"].split(".")[-1]
        # params: dict = packet['params']
        # tick: dict = self.ticks[symbol]
        # print(packet, tick, symbol, channel,'~~~')

        is_fire: bool = False
        # kline = candle
        tick = None
        if channel == "kline":
            interval = KBAR_INTERVAL_REV[packet["topic"].split(".")[1]]
            tick: dict = self.ticks[f"{symbol}|{interval}"]

            data: dict = packet["data"][0]
            tick["volume"] = float(data["volume"])
            tick["turnover"] = float(data["turnover"])
            tick["open"] = float(data["open"])
            tick["high"] = float(data["high"])
            tick["low"] = float(data["low"])
            tick["close"] = float(data["close"])
            tick["start"] = models.timestamp_to_datetime(float(data["start"]))
            tick["datetime"] = models.generate_datetime()  # datetime.now()

            if self.on_kline_callback:
                is_fire = True
                self.on_kline_callback(copy(tick))

        elif channel == "tickers":
            tick: dict = self.ticks[f"{symbol}|ticker"]
            tick.update(self.handle_tickerdata(packet, symbol=symbol))

            if self.on_ticker_callback:
                is_fire = True
                self.on_ticker_callback(copy(tick))

        elif channel == "orderbook":
            current_time = time.monotonic()
            elapsed_time = current_time - self.last_time

            # save asks / bids in tick for next iteration
            tick: dict = self.ticks[f"{symbol}|depth"]
            tick.update(self.handle_orderbook(packet, symbol=symbol))

            if elapsed_time >= self.speed / 100:
                if self.on_depth_callback:
                    is_fire = True
                    self.on_depth_callback(copy(tick))

                self.last_time = current_time

        if not is_fire and self.on_tick and tick:
            self.on_tick(copy(tick))

    def handle_tickerdata(self, packet, symbol):
        # Record the initial snapshot.
        if "snapshot" in packet["type"]:
            self.ticker_data[symbol] = packet["data"]

        # Make updates according to delta response.
        elif "delta" in packet["type"]:
            for key, value in packet["data"].items():
                self.ticker_data[symbol][key] = value
        ts: int = packet["ts"]

        """
        >>>> CONTRACT:
        {
            "topic": "tickers.BTCUSDT",
            "type": "snapshot",
            "data": {
                "symbol": "BTCUSDT",
                "tickDirection": "PlusTick",
                "price24hPcnt": "0.017103",
                "lastPrice": "17216.00",
                "prevPrice24h": "16926.50",
                "highPrice24h": "17281.50",
                "lowPrice24h": "16915.00",
                "prevPrice1h": "17238.00",
                "markPrice": "17217.33",
                "indexPrice": "17227.36",
                "openInterest": "68744.761",
                "openInterestValue": "1183601235.91",
                "turnover24h": "1570383121.943499",
                "volume24h": "91705.276",
                "nextFundingTime": "1673280000000",
                "fundingRate": "-0.000212",
                "bid1Price": "17215.50",
                "bid1Size": "84.489",
                "ask1Price": "17216.00",
                "ask1Size": "83.020"
            },
            "cs": 24987956059,
            "ts": 1673272861686
        }

        >>>> OPTION:
        {
            "id": "tickers.BTC-6JAN23-17500-C-2480334983-1672917511074",
            "topic": "tickers.BTC-6JAN23-17500-C",
            "ts": 1672917511074,
            "data": {
                "symbol": "BTC-6JAN23-17500-C",
                "bidPrice": "0",
                "bidSize": "0",
                "bidIv": "0",
                "askPrice": "10",
                "askSize": "5.1",
                "askIv": "0.514",
                "lastPrice": "10",
                "highPrice24h": "25",
                "lowPrice24h": "5",
                "markPrice": "7.86976724",
                "indexPrice": "16823.73",
                "markPriceIv": "0.4896",
                "underlyingPrice": "16815.1",
                "openInterest": "49.85",
                "turnover24h": "446802.8473",
                "volume24h": "26.55",
                "totalVolume": "86",
                "totalTurnover": "1437431",
                "delta": "0.047831",
                "gamma": "0.00021453",
                "vega": "0.81351067",
                "theta": "-19.9115368",
                "predictedDeliveryPrice": "0",
                "change24h": "-0.33333334"
            },
            "type": "snapshot"
        }
        >>>> SPOT:
        {
            "topic": "tickers.BTCUSDT",
            "ts": 1673853746003,
            "type": "snapshot",
            "cs": 2588407389,
            "data": {
                "symbol": "BTCUSDT",
                "lastPrice": "21109.77",
                "highPrice24h": "21426.99",
                "lowPrice24h": "20575",
                "prevPrice24h": "20704.93",
                "volume24h": "6780.866843",
                "turnover24h": "141946527.22907118",
                "price24hPcnt": "0.0196",
                "usdIndexPrice": "21120.2400136"
            }
        }
        """
        data = copy(self.ticker_data[symbol])
        result = {}
        if self.DATA_TYPE in ["linear", "inverse", "spot"]:
            result["prev_open_24h"] = None
            result["prev_high_24h"] = float(data["highPrice24h"])
            result["prev_low_24h"] = float(data["lowPrice24h"])
            result["prev_close_24h"] = float(data["prevPrice24h"])
            result["prev_volume_24h"] = float(data["volume24h"])
            result["prev_turnover_24h"] = float(data["turnover24h"])
            result["open_interest"] = float(data["openInterest"])

            result["last_price"] = float(data["lastPrice"])
            result["price_change_pct"] = float(data["price24hPcnt"])
            result["price_change"] = result["last_price"] - result["prev_close_24h"]
            result["bid_price_1"] = float(data["bid1Price"]) if "bid1Price" in data else None
            result["bid_volume_1"] = float(data["bid1Size"]) if "bid1Size" in data else None
            result["ask_price_1"] = float(data["ask1Price"]) if "ask1Price" in data else None
            result["ask_volume_1"] = float(data["ask1Size"]) if "ask1Size" in data else None
        elif self.DATA_TYPE in ["option"]:
            result["prev_open_24h"] = None
            result["prev_high_24h"] = float(data["highPrice24h"])
            result["prev_low_24h"] = float(data["lowPrice24h"])
            result["prev_close_24h"] = None
            result["prev_volume_24h"] = float(data["volume24h"])
            result["prev_turnover_24h"] = float(data["turnover24h"])
            result["open_interest"] = float(data["openInterest"])

            result["bid_price_1"] = float(data["bidPrice"]) if "bidPrice" in data else None
            result["bid_volume_1"] = float(data["bidSize"]) if "bidSize" in data else None
            result["bid_iv"] = float(data["bidIv"])
            result["ask_price_1"] = float(data["askPrice"]) if "askPrice" in data else None
            result["ask_volume_1"] = float(data["askSize"]) if "askSize" in data else None
            result["ask_iv"] = float(data["askIv"])

            result["volume"] = float(data["totalVolume"])
            result["last_price"] = float(data["lastPrice"])
            result["price_change_pct"] = float(data["change24h"])  ## TODO: TBC, price change or pct change?
            result["price_change"] = None
            result["delta"] = float(data["delta"])
            result["gamma"] = float(data["gamma"])
            result["vega"] = float(data["vega"])
            result["theta"] = float(data["theta"])
            result["underlying_price"] = float(data["underlyingPrice"])
            result["index_price"] = float(data["indexPrice"])
            result["mark_price"] = float(data["markPrice"])
            result["mark_iv"] = float(data["markPriceIv"])

        result["datetime_now"] = models.generate_datetime()
        result["datetime"] = models.timestamp_to_datetime(ts)
        result["ts"] = ts
        return result

    def handle_orderbook(self, packet, symbol):
        # call by reference
        data: dict = packet[
            "data"
        ]  # The data is ordered by price, starting with the lowest buys and ending with the highest sells.
        ts: int = packet["ts"]
        # Record the initial snapshot.
        if packet["type"] == "snapshot":
            self.bids_asks_data[symbol]["bids"] = [
                d for d in data["b"]
            ]  # BUY side - the element is sorted by price in descending order
            self.bids_asks_data[symbol]["asks"] = [
                d for d in data["a"]
            ]  # SELL3 side - the element is sorted by price in ascending order

        # Make updates according to delta response.
        elif packet["type"] == "delta":
            book_sides = {"bids": data["b"], "asks": data["a"]}
            # print(f">>> data: {data}")
            for side, entries in book_sides.items():
                for entry in entries:
                    # Delete.
                    if float(entry[1]) == 0:
                        index = find_index(self.bids_asks_data[symbol][side], entry, 0)
                        # print(f"{side} DELETE - entry:{entry}; index:{index}; len:{len(self.bids_asks_data[symbol][side])}")
                        self.bids_asks_data[symbol][side].pop(index)
                        continue

                    # Insert.
                    price_level_exists = entry[0] in [level[0] for level in self.bids_asks_data[symbol][side]]
                    if not price_level_exists:
                        # print(f"{side} INSERT - entry:{entry}; len:{len(self.bids_asks_data[symbol][side])}")
                        self.bids_asks_data[symbol][side].append(entry)
                        continue

                    # Update.
                    qty_changed = entry[1] != next(
                        level[1] for level in self.bids_asks_data[symbol][side] if level[0] == entry[0]
                    )
                    if price_level_exists and qty_changed:
                        index = find_index(self.bids_asks_data[symbol][side], entry, 0)
                        # print(f"{side} UPDATE - entry:{entry}; index:{index}; len:{len(self.bids_asks_data[symbol][side])}")
                        self.bids_asks_data[symbol][side][index] = entry
                        continue

        # print('len of asks', len(self.bids_asks_data[symbol]['asks']),'len of bids', len(self.bids_asks_data[symbol]['bids']),'||||asks:',self.bids_asks_data[symbol]['asks'],'||||bids:',self.bids_asks_data[symbol]['bids'],'|||raw:',data)
        # print(self.bids_asks_data[symbol]['bids'][:5], packet["type"], self.bids_asks_data[symbol]['asks'][:5])

        # computer the bids and asks with current data
        bids: list = self.bids_asks_data[symbol]["bids"]
        for n in range(min(5, len(bids))):
            price = bids[n][0]
            volume = bids[n][1]
            # price = bids[n]["price"]
            # volume = bids[n]["size"]
            self.bids_asks_data[symbol]["bid_price_" + str(n + 1)] = float(price)
            self.bids_asks_data[symbol]["bid_volume_" + str(n + 1)] = float(volume)

        asks: list = self.bids_asks_data[symbol]["asks"]
        for n in range(min(5, len(asks))):
            price = asks[n][0]
            volume = asks[n][1]
            # price = asks[n]["price"]
            # volume = asks[n]["size"]
            self.bids_asks_data[symbol]["ask_price_" + str(n + 1)] = float(price)
            self.bids_asks_data[symbol]["ask_volume_" + str(n + 1)] = float(volume)

        self.bids_asks_data[symbol]["datetime_now"] = models.generate_datetime()
        self.bids_asks_data[symbol]["datetime"] = models.timestamp_to_datetime(ts)
        self.bids_asks_data[symbol]["ts"] = ts

        result = copy(self.bids_asks_data[symbol])
        result.pop("bids")
        result.pop("asks")
        return result


class BybitTradeWebsocket:
    """
    Implement private topics
    doc: https://bybit-exchange.github.io/docs/linear/#t-privatetopics
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
        is_demo: bool = False,
        ping_interval: int = 15,
        debug: bool = False,
        logger=None,
        **kwargs,
    ) -> None:
        self.logger = get_logger() if logger is None else logger

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
        self.key = None
        self.secret = None
        self.debug = debug

    def info(self):
        return f"Bybit {self.DATA_TYPE} Trade Websocket Start"

    def connect(
        self,
        key: str,
        secret: str,
        is_testnet: bool = None,
        is_demo: bool = None,
        **kwargs,
    ):
        """
        Give api key and secret for Authorization.
        """
        self.key = key
        self.secret = secret
        if is_testnet is not None:
            self.is_testnet = is_testnet
        if is_demo is not None:
            self.is_demo = is_demo

        self.ws_private = WebSocket(
            testnet=self.is_testnet,
            demo=self.is_demo,
            channel_type="private",
            api_key=self.key,
            api_secret=self.secret,
            trace_logging=False,
        )
        self.ws_private.position_stream(self.on_position)
        self.ws_private.order_stream(self.on_order)
        self.ws_private.execution_stream(self.on_trade)
        self.ws_private.wallet_stream(self.on_account)

        self.logger.info(self.info())

        return True

    def on_disconnected(self) -> None:
        self.logger.info(
            f"Bybit {self.DATA_TYPE} user trade data Websocket disconnected, now try to connect @ {datetime.now()}"
        )
        if self.on_disconnected_callback:
            self.on_disconnected_callback()

    def on_connected(self) -> None:
        # if self.debug:
        self.logger.debug(f"Bybit {self.DATA_TYPE} user trade data websocket on_connected")

        if self.on_connected_callback:
            self.on_connected_callback()

    def on_error(self, exception_value):
        super().on_error(exception_value)
        if self.on_error_callback:
            self.on_error_callback(exception_value)

    def on_account(self, packet: dict) -> None:
        """
        [DEBUG] Bybit linear Trade Websocket - on_packet event {'id': '11934812a537b05-b575-4ea5-8585-f731d62c455b', 'topic': 'wallet', 'creationTime': 1678887797421, 'data': [{'accountIMRate': '0.0092', 'accountMMRate': '0.0005', 'totalEquity': '1646.13810243', 'totalWalletBalance': '1640.20715187', 'totalMarginBalance': '1646.13810243', 'totalAvailableBalance': '1630.97132641', 'totalPerpUPL': '5.93095056', 'totalInitialMargin': '15.16677602', 'totalMaintenanceMargin': '0.88992747', 'coin': [{'coin': 'USDC', 'equity': '1.67796784', 'usdValue': '1.67838974', 'walletBalance': '1.54616784', 'availableToWithdraw': '0', 'availableToBorrow': '1500000', 'borrowAmount': '0', 'accruedInterest': '0', 'totalOrderIM': '0', 'totalPositionIM': '3.14195033', 'totalPositionMM': '0.22731653', 'unrealisedPnl': '0.1318', 'cumRealisedPnl': '1.54616784', 'bonus': '0'}, {'coin': 'USDT', 'equity': '1573.91127744', 'usdValue': '1581.26457518', 'walletBalance': '1568.13912744', 'availableToWithdraw': '1518.1986267', 'availableToBorrow': '2500000', 'borrowAmount': '0', 'accruedInterest': '0', 'totalOrderIM': '0', 'totalPositionIM': '11.96812074', 'totalPositionMM': '0.65947274', 'unrealisedPnl': '5.77215', 'cumRealisedPnl': '0.13332342', 'bonus': '0'}], 'accountType': 'UNIFIED'}]}
        """
        try:
            # self.logger.debug(f"packet on_account: {packet}")

            for acc_data in packet["data"]:
                account_type = acc_data["accountType"]
                datatype = "linear" if account_type == "UNIFIED" else "linear" if account_type == "CONTRACT" else "spot"
                if self.DATA_TYPE and (
                    (account_type == "UNIFIED" and self.DATA_TYPE not in ["linear", "spot", "option"])
                    or (account_type == "CONTRACT" and self.DATA_TYPE not in ["linear", "inverse"])
                    or (account_type == "SPOT" and self.DATA_TYPE not in ["spot"])
                ):
                    continue
                for data in acc_data.get("coin", []):
                    account = models.AccountSchema()  # .model_construct(data)
                    account = account.from_bybit_v2_to_form(data, datatype=datatype, account_type=account_type)
                    if account and account["symbol"] and self.on_account_callback:
                        self.on_account_callback(account)
        except Exception as e:
            self.on_error(e)

    def on_order(self, packet: dict) -> None:
        """
        [DEBUG] Bybit linear Trade Websocket - on_packet event {'id': '1193481acf42bfa-cbca-4232-93a2-dcf9dead9510', 'topic': 'order', 'creationTime': 1678887821206, 'data': [{'symbol': 'BTCUSDT', 'orderId': 'c2153424-5ab3-456c-b8e8-98bc266dad0e', 'side': 'Buy', 'orderType': 'Market', 'cancelType': 'UNKNOWN', 'price': '26140.4', 'qty': '0.001', 'orderIv': '', 'timeInForce': 'IOC', 'orderStatus': 'Filled', 'orderLinkId': '', 'lastPriceOnCreated': '24895.7', 'reduceOnly': False, 'leavesQty': '0', 'leavesValue': '0', 'cumExecQty': '0.001', 'cumExecValue': '24.8993', 'avgPrice': '24899.3', 'blockTradeId': '', 'positionIdx': 0, 'cumExecFee': '0.01493958', 'createdTime': '1678887821191', 'updatedTime': '1678887821200', 'rejectReason': 'EC_NoError', 'stopOrderType': '', 'triggerPrice': '', 'takeProfit': '', 'stopLoss': '', 'tpTriggerBy': '', 'slTriggerBy': '', 'triggerDirection': 0, 'triggerBy': '', 'closeOnTrigger': False, 'category': 'linear', 'isLeverage': ''}]}
        """
        try:
            # self.logger.debug(f"packet on_order: {packet}")
            datas: dict = packet["data"]

            # if datas["order_type"] not in ['LIMIT', 'Market']:
            #     return

            for data in datas:
                datatype = data["category"]
                if self.DATA_TYPE and datatype != self.DATA_TYPE:
                    continue
                order = models.OrderSchema()  # .model_construct(data)
                order = order.from_bybit_v2_to_form(data, datatype=datatype)
                if self.on_order_callback:  # and order_list
                    self.on_order_callback(order)

        except Exception as e:
            self.on_error(e)

    def on_trade(self, packet: dict) -> None:
        """
        [DEBUG] Bybit linear Trade Websocket - on_packet event {'id': '1193481299665b2-56be-40da-8a29-f488cb47190b', 'topic': 'execution', 'creationTime': 1678888392300, 'data': [{'category': 'linear', 'symbol': 'BTCUSDT', 'execFee': '0.01490148', 'execId': 'dae9c7a8-ceb9-51fa-8881-d5157ecc2c51', 'execPrice': '24835.8', 'execQty': '0.001', 'execType': 'Trade', 'execValue': '24.8358', 'isMaker': False, 'feeRate': '0.0006', 'tradeIv': '', 'markIv': '', 'blockTradeId': '', 'markPrice': '24883.13', 'indexPrice': '', 'underlyingPrice': '', 'leavesQty': '0', 'orderId': 'e9b9e551-081b-414c-a109-0e56d225ecd2', 'orderLinkId': '', 'orderPrice': '26116.9', 'orderQty': '0.001', 'orderType': 'Market', 'stopOrderType': 'UNKNOWN', 'side': 'Buy', 'execTime': '1678888392289', 'isLeverage': '0'}]}
        """
        try:
            # self.logger.debug(f"packet on_trade: {packet}")
            datas: dict = packet["data"]

            is_all_filled = False
            for data in datas:
                datatype = data["category"]
                if self.DATA_TYPE and datatype != self.DATA_TYPE:
                    continue
                trade = models.TradeSchema()  # .model_construct(data)
                trade = trade.from_bybit_v2_to_form(data, datatype=datatype)
                if trade and self.on_trade_callback:  # and trade_list:
                    # print(trade,'....')
                    self.on_trade_callback(trade)
        except Exception as e:
            self.on_error(e)

    def on_position(self, packet: dict) -> None:
        """
        [DEBUG] Bybit linear Trade Websocket - on_packet event {'id': '119348107a024b9-ed00-400d-b81c-105c5eb57e7e', 'topic': 'position', 'creationTime': 1678887821206, 'data': [{'positionIdx': 0, 'tradeMode': 0, 'riskId': 1, 'riskLimitValue': '2000000', 'symbol': 'BTCUSDT', 'side': 'Buy', 'size': '0.006', 'entryPrice': '23989.61666667', 'leverage': '10', 'positionValue': '143.9377', 'markPrice': '24922.99', 'positionIM': '14.47149636', 'positionMM': '0.79741486', 'takeProfit': '0', 'stopLoss': '0', 'trailingStop': '0', 'unrealisedPnl': '5.60024', 'cumRealisedPnl': '0.11838384', 'createdTime': '1676887474897', 'updatedTime': '1678887821202', 'tpslMode': 'Full', 'liqPrice': '', 'bustPrice': '', 'positionStatus': 'Normal'}]}
        """
        try:
            # self.logger.debug(f"packet on_position: {packet}")
            datas: dict = packet["data"]
            for data in datas:
                # position_value = float(data["position_value"])
                # if not position_value:
                #     continue
                datatype = data.get("category", "linear")
                if not datatype:
                    datatype = "linear"
                if self.DATA_TYPE and datatype != self.DATA_TYPE:
                    continue
                """
                Unified account: does not have this field
                Unified account - inverse & Normal account: linear, inverse.
                """
                position = models.PositionSchema()  # .model_construct(data)
                position = position.from_bybit_v2_to_form(data, datatype=datatype)
                if position and self.on_position_callback:
                    self.on_position_callback(position)
        except Exception as e:
            self.on_error(e)
