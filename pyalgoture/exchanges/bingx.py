# import base64
# import gzip
# import hashlib
# import hmac
# import json
# import sys
# import time
# from collections import defaultdict
# from copy import copy
# from datetime import datetime, timedelta

# # from threading import Lock, Timer
# from threading import Timer
# from typing import Any, Callable, Optional, dict

# # from loguru import logger
# # from requests import Session
# from urllib.parse import urlencode

# from ..utils import models
# from ..utils.client_rest import Response, RestClient
# from ..utils.client_ws import WebsocketClient
# from ..utils.logger import get_logger
# from ..utils.models import ASSET_TYPE_SPOT, UTC_TZ, bingx_asset_type

# logger = get_logger(default_level="warning")

# EXCHANGE = "BINGX"

# TIMEDELTA_MAP: dict[str, timedelta] = {
#     "1m": timedelta(minutes=1),
#     "1h": timedelta(hours=1),
#     "1d": timedelta(days=1),
# }

# BINGX_API_HOST = "https://open-api.bingx.com/openApi"
# BINGX_SPOT_WEBSOCKET_HOST = "wss://open-api-ws.bingx.com/market"
# BINGX_FUTURE_WEBSOCKET_HOST = "wss://open-api-swap.bingx.com/swap-market"

# TIMEDELTA_MAPPING_SEC: dict[str, int] = {
#     "1m": 60,
#     "3m": 60 * 3,
#     "5m": 60 * 5,
#     "15m": 60 * 15,
#     "30m": 60 * 15,
#     "1h": 60 * 60 * 1,
#     "2h": 60 * 60 * 2,
#     "4h": 60 * 60 * 4,
#     "6h": 60 * 60 * 6,
#     "8h": 60 * 60 * 8,
#     "12h": 60 * 60 * 12,
#     "1d": 60 * 60 * 24 * 1,
#     "3d": 60 * 60 * 24 * 3,
#     "1w": 60 * 60 * 24 * 7,
#     "1M": 60 * 60 * 24 * 30,
# }

# ASSET_TYPE_MAP = {
#     "spot": "SPOT",
#     "margin": "MARGIN",
#     "future": ["SWAP", "FUTURES"],
#     "inverse_future": ["SWAP", "FUTURES"],
#     "option": "OPTION",
# }


# def generate_signature(msg: str, secret_key: str) -> bytes:
#     """"""
#     return base64.b64encode(
#         hmac.new(secret_key.encode(), msg.encode(), hashlib.sha256).digest()
#     )


# def generate_timestamp() -> str:
#     """"""
#     now: datetime = datetime.utcnow()
#     timestamp: str = now.isoformat("T", "milliseconds")
#     return timestamp + "Z"


# def parse_timestamp(timestamp: str) -> datetime:
#     """"""
#     dt: datetime = datetime.fromtimestamp(int(timestamp) / 1000)
#     return UTC_TZ.localize(dt)


# def get_float_value(data: dict, key: str) -> float:
#     """"""
#     data_str: str = data.get(key, "")
#     if not data_str:
#         return 0.0
#     return float(data_str)


# class RepeatTimer(Timer):
#     def run(self):
#         while not self.finished.wait(self.interval):
#             self.function(*self.args, **self.kwargs)


# class BingxClient(RestClient):
#     """
#     api docs: https://developer.bingx.com/#introduction

#     """

#     BROKER_ID = "Uvn2GfZrNsg7JeX6RxWI53LgHDatUc7v"
#     DATA_TYPE = "spot"

#     def __init__(
#         self,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         datatype: str = "spot",
#         is_live: bool = False,
#         **kwargs,
#     ) -> None:
#         # url_base: str = BINGX_API_HOST,
#         super().__init__(BINGX_API_HOST, proxy_host, proxy_port)
#         if datatype not in ["spot", "linear"]:
#             raise ValueError("Invalid data type.")

#         self.DATA_TYPE = datatype  # for symbol, orders, trades, account

#         self.order_count: int = 0
#         # self.order_count_lock: Lock = Lock()
#         self.connect_time: int = 0
#         self.key: str = ""
#         self.secret: str = ""
#         # self.is_testnet = is_testnet
#         self.is_live = is_live

#     def info(self):
#         return "BINGX REST API start"

#     @staticmethod
#     def generate_timestamp(expire_after: float = 30) -> int:
#         """generate_timestamp"""
#         return int(time.time() * 1000 + expire_after * 1000)

#     @staticmethod
#     def generate_datetime(datetime_str: str = None) -> datetime:
#         """generate_datetime"""
#         if datetime_str:
#             dt: datetime = datetime.strptime(datetime_str, "%Y%m%d-%H:%M:%S.%f")
#         else:
#             dt: datetime = datetime.now()
#         dt: datetime = UTC_TZ.localize(dt)
#         return dt

#     @staticmethod
#     def generate_datetime_ts(timestamp: float) -> datetime:
#         """generate_datetime"""
#         dt: datetime = datetime.fromtimestamp(timestamp / 1000)
#         # dt: datetime = UTC_TZ.localize(dt)
#         dt: datetime = dt.astimezone(UTC_TZ)
#         return dt

#     def new_order_id(self) -> str:
#         """new_order_id"""
#         # prefix: str = datetime.now().strftime("%Y%m%d-%H%M%S-")
#         # prefix: str = datetime.now().strftime("%y%m%d-%H%M%S%f")[:-3]+'-'
#         prefix: str = datetime.now().strftime("%y%m%d%H%M%S%f")[:-7]
#         # with self.order_count_lock:
#         self.order_count += 1
#         suffix: str = str(self.order_count).rjust(5, "0")

#         order_id: str = prefix + suffix
#         return order_id

#     def _remove_none(self, payload):
#         return {k: v for k, v in payload.items() if v is not None}

#     def _get(
#         self, path: str, params: Optional[dict[str, Any]] = None, ignore: bool = False
#     ) -> Any:
#         # response = self.request(method="GET", path=path,  params=params)
#         response = self.query(method="GET", path=path, params=params)
#         return self._process_response(response, ignore=ignore)

#     def _post(
#         self,
#         path: str,
#         data: Optional[dict[str, Any]] = None,
#         params: Optional[dict[str, Any]] = None,
#         ignore: bool = False,
#     ) -> Any:
#         response = self.query(method="POST", path=path, data=data, params=params)
#         return self._process_response(response, ignore)

#     def _delete(
#         self,
#         path: str,
#         data: Optional[dict[str, Any]] = None,
#         params: Optional[dict[str, Any]] = None,
#         ignore: bool = False,
#     ) -> Any:
#         response = self.query(method="DELETE", path=path, data=data, params=params)
#         return self._process_response(response, ignore)

#     def _put(
#         self,
#         path: str,
#         data: Optional[dict[str, Any]] = None,
#         params: Optional[dict[str, Any]] = None,
#         ignore: bool = False,
#     ) -> Any:
#         response = self.query(method="PUT", path=path, data=data, params=params)
#         return self._process_response(response, ignore)

#     def connect(self, key: str, secret: str) -> None:
#         """connect exchange server"""
#         self.key = key
#         self.secret = secret

#         # print(self.key, secret,'.....')
#         if self.is_live:
#             self.start()
#         logger.debug(self.info())
#         return True

#     def sign(self, request):
#         """
#         https://github.com/vn-crypto/vnpy_okex/blob/main/vnpy_okex/okex_gateway.py
#         """
#         if (
#             "common" in request.path
#             or "market" in request.path
#             or "quote" in request.path
#         ):
#             return request

#         if not request.params:
#             request.params = dict()
#         if not request.data:
#             request.data = dict()

#         timestamp: int = int(time.time() * 1000)
#         request.params["timestamp"] = timestamp

#         # data_query: str = urlencode(sorted(request.data.items()))
#         # query: str = urlencode(sorted(request.params.items()))
#         # if data_query:
#         #     request.data = data_query
#         #     data_query = query + data_query
#         # # print(f"query: {query}; data_query:{data_query}")

#         # signature: bytes = hmac.new(self.secret.encode("utf-8"), data_query.encode("utf-8") if data_query else query.encode("utf-8"), hashlib.sha256).hexdigest()
#         # # print(f"signature: {signature}")
#         data_query: str = urlencode(sorted(request.data.items()))
#         query: str = urlencode(sorted(request.params.items()))
#         if data_query:
#             request.data = {}
#             query = query + "&" + data_query
#         # print(f"query: {query}; data_query:{data_query}")

#         signature: bytes = hmac.new(
#             self.secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256
#         ).hexdigest()
#         # print(f"signature: {signature}")

#         query += f"&signature={signature}"
#         path: str = request.path + "?" + query

#         request.params = {}
#         request.path = path

#         # add header
#         request.headers = {
#             "X-SOURCE-KEY": self.BROKER_ID,
#             "X-BX-APIKEY": self.key,
#             "Content-Type": "application/x-www-form-urlencoded",
#         }
#         # print(f"method:{request.method}; path: {request.path}; params: {request.params}; data: {request.data}; headers:{request.headers}\n")
#         # print(request.headers, "<=====header")

#         # print(request,'request!')
#         return request

#     def _process_response(self, response: Response, ignore: bool = False) -> dict:
#         try:
#             data = response.data
#             # print(data,'!!!!', response.ok)
#         except ValueError:
#             print("!!bingx - _process_response !!", response.text)
#             logger.debug(response.data())
#             raise
#         else:
#             # logger.debug(data)
#             # print('data >> ',data)
#             if response.ok and isinstance(data, list) or ignore:
#                 ### transfer schema is diff from other apis
#                 return models.CommonResponseSchema(
#                     success=True, error=False, data=data, msg="query ok"
#                 )

#             if not response.ok or (response.ok and data.get("code", 1111) != 0):
#                 payload_data = models.CommonDataSchema(
#                     status_code=response.status_code, msg=dict(data)
#                 )

#                 msg = data.get(
#                     "msg", "Something went wrong when fetching data from BingX server."
#                 )
#                 error_code = data.get("code", 1)
#                 if error_code == 100001:
#                     msg = "Signature authentication failed."
#                 elif error_code == 100202:
#                     msg = "Insufficient balance."
#                 elif error_code == 100400:
#                     msg = "Invalid parameter."
#                 elif error_code == 100440:
#                     msg = "Order price deviates greatly from the market price."
#                 elif error_code == 100500:
#                     msg = "Internal server error."
#                 elif error_code == 100503:
#                     msg = "Server busy."
#                 return models.CommonResponseSchema(
#                     success=False, error=True, data=payload_data.dict(), msg=msg
#                 )
#             elif response.ok and not data:
#                 # print('data',data, '?????')
#                 payload_data = models.CommonDataSchema(
#                     status_code=response.status_code, msg=dict(data)
#                 )
#                 return models.CommonResponseSchema(
#                     success=True,
#                     error=False,
#                     data=payload_data.dict(),
#                     msg="query ok, but result is None",
#                 )
#             else:
#                 # print('data',data,'!!!!!')
#                 return models.CommonResponseSchema(
#                     success=True, error=False, data=data, msg="query ok"
#                 )

#     def _regular_order_payload(
#         self, common_response: models.CommonResponseSchema
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 data = models.OrderSchema()
#                 result = data.from_bingx_to_form(msg, datatype=self.DATA_TYPE)
#                 if result:
#                     payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.OrderSchema()
#             payload = payload.from_bingx_to_form(
#                 common_response.data, datatype=self.DATA_TYPE
#             )
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def _regular_trades_payload(
#         self, common_response: models.CommonResponseSchema
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 data = models.TradeSchema()
#                 result = data.from_bingx_to_form(msg, datatype=self.DATA_TYPE)
#                 if result:
#                     payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.TradeSchema()
#             payload = payload.from_bingx_to_form(
#                 common_response.data, datatype=self.DATA_TYPE
#             )
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def _regular_symbols_payload(
#         self, common_response: models.CommonResponseSchema
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             # payload = list()
#             payload = dict()
#             for msg in common_response.data:
#                 data = models.SymbolSchema()
#                 result = data.from_bingx_to_form(msg, self.DATA_TYPE)
#                 if result:
#                     # payload.append(result)
#                     if len(common_response.data) == 1:
#                         payload = result
#                     else:
#                         payload[result["symbol"]] = result
#         elif isinstance(common_response.data, dict):
#             payload = models.SymbolSchema()
#             payload = payload.from_bingx_to_form(common_response.data, self.DATA_TYPE)
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def _regular_historical_prices_payload(
#         self, common_response: models.CommonResponseSchema, extra: dict = None
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 data = models.HistoryOHLCSchema()
#                 result = data.from_bingx_to_form(
#                     msg, datatype=self.DATA_TYPE, extra=extra
#                 )
#                 payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = (
#                 models.HistoryOHLCSchema()
#             )  # .model_construct(common_response.data, extra, datatype=self.DATA_TYPE)
#             payload = payload.from_bingx_to_form(common_response.data, extra)
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def _regular_symbol_payload(
#         self, common_response: models.CommonResponseSchema
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 data = models.TickerSchema()
#                 result = data.from_bingx_to_form(msg, datatype=self.DATA_TYPE)
#                 payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.TickerSchema()
#             payload = payload.from_bingx_to_form(
#                 common_response.data, datatype=self.DATA_TYPE
#             )
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def _regular_position_payload(
#         self, common_response: models.CommonResponseSchema
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 data = models.PositionSchema()
#                 result = data.from_bingx_to_form(msg, datatype=self.DATA_TYPE)
#                 if result:
#                     payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.PositionSchema()
#             payload = payload.from_bingx_to_form(
#                 common_response.data, datatype=self.DATA_TYPE
#             )
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def _regular_transfer_payload(
#         self,
#         common_response: models.CommonResponseSchema,
#         transfer_type: str,
#         inout: str,
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 data = models.TransferSchema()
#                 result = data.from_bingx_to_form(
#                     msg, transfer_type=transfer_type, inout=inout
#                 )
#                 payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.TransferSchema()
#             payload = payload.from_bingx_to_form(
#                 common_response.data, transfer_type=transfer_type, inout=inout
#             )
#         else:
#             common_response.msg = "payload error in fetch transfer data"
#             Exception("payload error in fetch transfer data")

#         common_response.data = payload
#         return common_response

#     def query_permission(self) -> dict:
#         """ """
#         # return models.CommonResponseSchema(success=False, error=True, data={}, msg=f"BingX doesnt support query historical data yet.")
#         api_info = self._get("/v1/account/apiRestrictions", ignore=True)

#         # logger.info(api_info)
#         print(api_info, "======")
#         if api_info.success:
#             if api_info.data:
#                 data = models.PermissionSchema()  # .model_construct(api_info.data)
#                 data.from_bingx_to_form(api_info.data)
#                 api_info.data = data.dict()
#             else:
#                 api_info.msg = "query api info is empty"

#         return api_info

#     def query_account(self) -> dict:
#         """query_account
#         {
#             "code": 0,
#             "msg": "",
#             "ttl": 1,
#             "data": {
#                 "balances": [
#                     {
#                         "asset": "USDT",
#                         "free": "16.73971130673954",
#                         "locked": "0"
#                     }
#                 ]
#             }
#         }

#         """

#         if self.DATA_TYPE == "spot":
#             uri = "/spot/v1/account/balance"
#         else:
#             uri = "/swap/v2/user/balance"
#         balance = self._get(uri)
#         account = defaultdict(dict)
#         # print(f">>> balance:{balance} | datatype:{self.DATA_TYPE}")

#         if balance.success and balance.data:
#             if self.DATA_TYPE == "spot":
#                 if not balance.data.get("data", {}).get("balances", []):
#                     balance.msg = "account wallet is empty"
#                     return balance
#                 for account_data in balance.data.get("data", {}).get("balances", []):
#                     key = account_data["asset"]
#                     free = float(account_data["free"])
#                     frozen = float(account_data["locked"])
#                     total = free + frozen
#                     if total:
#                         account[key]["symbol"] = key
#                         account[key]["available"] = free
#                         account[key]["frozen"] = frozen
#                         account[key]["balance"] = total
#                         account[key]["market_value"] = None
#                         account[key]["exchange"] = EXCHANGE
#                         account[key]["asset_type"] = self.DATA_TYPE.upper()
#             else:
#                 account_data = balance.data.get("data", {}).get("balance", {})
#                 if account_data:
#                     key = account_data["asset"]
#                     frozen = float(account_data["freezedMargin"])
#                     total = float(account_data["balance"])
#                     free = total - frozen - float(account_data["usedMargin"])
#                     if total:
#                         account[key]["symbol"] = key
#                         account[key]["available"] = free
#                         account[key]["frozen"] = frozen
#                         account[key]["balance"] = total
#                         account[key]["market_value"] = float(account_data["equity"])
#                         account[key]["exchange"] = EXCHANGE
#                         account[key]["asset_type"] = self.DATA_TYPE.upper()

#             balance.data = account
#             return balance
#         else:
#             return balance

#     def send_order(
#         self,
#         symbol: str,
#         quantity: float,
#         side: str,
#         order_type=None,
#         price=None,
#         stop_price=None,
#         position_side: str = "long",
#         reduce_only: bool = False,
#         **kwargs,
#     ) -> dict:
#         """
#         For limit orders, price is required.
#         For limit orders, either quantity or quoteOrderQty is required. When two parameters are passed at the same time, the server uses the parameter quantity first.
#         * For buy-side market orders, quoteOrderQty is required.
#         * For sell-side market orders, quantity is required.

#         response:
#         {
#             "code": 0,
#             "msg": "",
#             "data": {
#                 "symbol": "XRP-USDT",
#                 "orderId": 1514090846268424192,
#                 "transactTime": 1649822362855,
#                 "price": "0.5",
#                 "origQty": "10",
#                 "executedQty": "0",
#                 "cummulativeQuoteQty": "0",
#                 "status": "PENDING",
#                 "type": "LIMIT",
#                 "side": "BUY"
#             }
#         }
#         position_side long, side buy -> open long
#         position_side long, side sell -> close long
#         position_side short, side sell -> open short
#         position_side short, side buy -> close short

#         """
#         symbol: str = symbol.upper()
#         side: str = side.upper()  # Order side, buy / sell

#         order_id: str = self.new_order_id()  # generate_new_order_id

#         if not order_type:
#             if not price:
#                 order_type = "MARKET"
#             else:
#                 order_type = "LIMIT"

#         payload: dict = {
#             "symbol": symbol,
#             "side": side,
#             "type": order_type,
#             "quantity": quantity,  # E.g. 0.1BTC
#             # "quoteOrderQty": quoteOrderQty,   # E.g. 100USDT (needa specify if it is BUY MARKET ORDER)
#         }
#         # if reduce_only:
#         #     payload["closePosition"] = True

#         if not price and self.DATA_TYPE == "spot" and side == "BUY":
#             last_price = self.query_prices(symbol)["data"]["price"]
#             quote_qty = quantity * last_price
#             payload["quoteOrderQty"] = quote_qty
#             payload.pop("quantity")
#             print(
#                 f">> Market Spot BUY payload:payload:{payload}; last_price;{last_price}"
#             )
#         if order_type == "LIMIT":
#             payload["price"] = price
#         if position_side:
#             position_side = position_side.upper()
#             payload["positionSide"] = position_side

#         # if reduce_only:
#         #     payload["reduceOnly"] = reduce_only

#         # asset_type = bingx_asset_type(symbol, self.DATA_TYPE)

#         if self.DATA_TYPE == "spot":
#             uri = "/spot/v1/trade/order"
#         else:
#             uri = "/swap/v2/trade/order"

#         orders_response = self._post(uri, data=payload)
#         print(
#             f"[DEBUG] orders_response:{orders_response};  send order payload:{payload}"
#         )

#         if orders_response.success:
#             if orders_response.data:
#                 if self.DATA_TYPE == "spot":
#                     data = orders_response.data["data"]
#                 else:
#                     data = orders_response.data.get("data", {}).get("order", {})
#                 result = models.SendOrderResultSchema()  # .model_construct(data)
#                 orders_response.data = result.from_bingx_to_form(
#                     data, symbol=symbol, datatype=self.DATA_TYPE
#                 )
#             else:
#                 orders_response.msg = "query sender order is empty"

#         return orders_response

#     def cancel_order(self, order_id: str, symbol: str, **kwargs) -> dict:
#         """ """
#         symbol = symbol.upper()
#         payload: dict = {"symbol": symbol, "orderId": order_id}
#         if self.DATA_TYPE == "spot":
#             uri = "/spot/v1/trade/cancel"
#             return self._post(uri, payload)
#         else:
#             uri = "/swap/v2/trade/order"
#             return self._delete(uri, payload)

#     def cancel_orders(self, symbol: str = None) -> dict:
#         """ """
#         if self.DATA_TYPE == "spot":
#             return models.CommonResponseSchema(
#                 success=False,
#                 error=True,
#                 data={},
#                 msg="BINGX doesn't support this feature",
#             )
#         payload: dict = {"symbol": symbol}
#         return self._delete("/swap/v2/trade/allOpenOrders", payload)

#     def query_order_status(self, order_id: str, symbol: str) -> dict:
#         """ """
#         payload = {"orderId": order_id, "symbol": symbol}  #
#         if self.DATA_TYPE == "spot":
#             uri = "/spot/v1/trade/query"
#         else:
#             uri = "/swap/v2/trade/order"
#         order_status = self._get(uri, payload)
#         # print(f"order_status: {order_status}")

#         if order_status.success:
#             if order_status.data:
#                 # return self._regular_order_payload(order_status)
#                 if self.DATA_TYPE == "spot":
#                     order_status.data = order_status.data.get("data", {})
#                 else:
#                     order_status.data = order_status.data.get("data", {}).get(
#                         "order", {}
#                     )
#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"

#         return order_status

#     def query_open_orders(self, symbol: str = None) -> dict:
#         """"""
#         payload = {}
#         if self.DATA_TYPE == "spot":
#             uri = "/spot/v1/trade/openOrders"
#             if not symbol:
#                 return models.CommonResponseSchema(
#                     success=False, error=True, data={}, msg="Please specify the symbol."
#                 )
#             payload["symbol"] = symbol
#         else:
#             uri = "/swap/v2/trade/openOrders"

#         order_status = self._get(uri, payload)
#         if order_status.success:
#             if order_status.data:
#                 order_status.data = order_status.data.get("data", {}).get("orders", [])
#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"

#         return order_status

#     def query_all_orders(self, symbol: str = None) -> dict:
#         """ """

#         if self.DATA_TYPE == "spot":
#             uri = "/spot/v1/trade/historyOrders"
#             payload = {"symbol": symbol, "pageIndex": 1, "pageSize": 100}
#         else:
#             uri = "/swap/v2/trade/allOrders"
#             payload = {"symbol": symbol, "limit": 1000}
#         all_order = self._get(uri, payload=payload)
#         if all_order.success:
#             if all_order.data:
#                 all_order.data = all_order.data.get("data", {}).get("orders", [])
#                 return self._regular_order_payload(all_order)
#             else:
#                 all_order.msg = "query order is empty"

#         return all_order

#     def query_symbols(self, symbol: str = None) -> dict:
#         """query_symbols"""
#         params = {}

#         if symbol:
#             params["symbol"] = symbol

#         if self.DATA_TYPE == "spot":
#             url = "/spot/v1/common/symbols"
#         elif self.DATA_TYPE == "linear":
#             url = "/swap/v2/quote/contracts"

#         symbols = self._get(url, params=params)

#         if symbols.success:
#             if symbols.data:
#                 if self.DATA_TYPE == "spot":
#                     symbols.data = symbols.data.get("data", {}).get("symbols", [])
#                 else:
#                     symbols.data = symbols.data.get("data", [])

#                 symbols = self._regular_symbols_payload(symbols)
#                 if symbol:
#                     if symbol in symbols.data:
#                         # symbols.data = {symbol: symbols.data[symbol]}
#                         symbols.data = symbols.data[symbol]

#                 return symbols
#             else:
#                 symbols.msg = "query symbol is empty"

#         return symbols

#     def query_trades(
#         self,
#         symbol: str = None,
#         start: datetime = None,
#         end: datetime = None,
#         limit=100,
#     ) -> dict:
#         """ """
#         if not symbol:
#             return models.CommonResponseSchema(
#                 success=False, error=True, data={}, msg="Please specify the symbol."
#             )

#         if self.DATA_TYPE == "spot":
#             uri = "/spot/v1/trade/historyOrders"
#             payload = {"symbol": symbol, "pageIndex": 1, "pageSize": 100}
#         else:
#             """
#             TODO: loop to fetch trade if over 7days
#             The maximum query time range shall not exceed 7 days
#             Query data within the last 7 days by default
#             """
#             uri = "/swap/v2/trade/allOrders"
#             payload = {"symbol": symbol, "limit": 1000}

#         if start:
#             payload["startTime"] = int(datetime.timestamp(start) * 1000)
#         if end:
#             payload["endTime"] = int(datetime.timestamp(end) * 1000)

#         trades = self._get(uri, params=payload)

#         if trades.success:
#             if trades.data:
#                 trades.data = trades.data.get("data", {}).get("orders", [])
#                 return self._regular_trades_payload(trades)
#             else:
#                 trades.msg = "query symbol is empty"

#         return trades

#     def query_history(
#         self,
#         symbol: str,
#         interval: str = "1h",
#         start: datetime = None,
#         end: datetime = None,
#         limit: int = 1440,
#     ) -> dict:
#         """ """
#         if self.DATA_TYPE == "spot":
#             return models.CommonResponseSchema(
#                 success=False,
#                 error=True,
#                 data={},
#                 msg="BingX doesnt support query historical data yet.",
#             )

#         def iteration_history(symbol, interval, data, payload, limit, retry=1):
#             # give data from start time with limit
#             retry_counter = 0
#             historical_prices_datalist = data.data.get("data", [])
#             # print(historical_prices_datalist,'???', len(historical_prices_datalist))
#             # historical_prices_datalist.sort(key=lambda k: k['time'])
#             # logger.debug(interval)
#             first_timestamp = int(historical_prices_datalist[0]["time"])
#             last_timestamp = int(historical_prices_datalist[-1]["time"])
#             start_timestamp = payload["startTime"]
#             end_timestamp = payload["endTime"]
#             # interval_timestamp = end_timestamp - last_timestamp
#             interval_timestamp = last_timestamp - start_timestamp
#             # print(f"init first_timestamp {first_timestamp} {datetime.fromtimestamp(first_timestamp/1000)}")
#             # print(f"init last_timestamp {last_timestamp} {datetime.fromtimestamp(last_timestamp/1000)}")
#             # print(f"init start_timestamp {start_timestamp} {datetime.fromtimestamp(start_timestamp/1000)}")
#             # print(f"init interval_timestamp {interval_timestamp} - interval_timestamp > interval:{interval_timestamp > interval} | interval:{interval}")

#             while interval_timestamp > interval:
#                 first_timestamp = int(historical_prices_datalist[0]["time"])
#                 last_timestamp = int(historical_prices_datalist[-1]["time"])
#                 # interval_timestamp = end_timestamp - last_timestamp
#                 interval_timestamp = last_timestamp - start_timestamp

#                 # print(f"first_timestamp {first_timestamp} {datetime.fromtimestamp(first_timestamp/1000)}")
#                 # print(f"last_timestamp {last_timestamp} {datetime.fromtimestamp(last_timestamp/1000)}")
#                 # # print(f"end_timestamp {end_timestamp} {datetime.fromtimestamp(end_timestamp/1000)}")
#                 # print(f"interval_timestamp {interval_timestamp} ")
#                 # # # print(historical_prices_datalist[0:5])

#                 payload["endTime"] = first_timestamp

#                 prices = self._get(
#                     "/swap/v2/quote/klines", self._remove_none(payload)
#                 )  # history-
#                 if prices.error:
#                     print(prices, "ERROR!")
#                     break
#                 # print(prices.data,'!!!!!!!!!!!')
#                 prices_data = prices.data.get("data", [])
#                 # print(prices_data[:2],f'???len{len(prices_data)}???', prices_data[-2:])
#                 # print(f'???len{len(prices_data)}???')

#                 historical_prices_datalist.extend(prices_data)
#                 # print(f">> 111historical_prices_datalist:{historical_prices_datalist}")
#                 historical_prices_datalist = list(
#                     {v["time"]: v for v in historical_prices_datalist}.values()
#                 )
#                 # print(f">> 222historical_prices_datalist:{historical_prices_datalist}")
#                 historical_prices_datalist.sort(key=lambda k: k["time"])
#                 time.sleep(0.1)

#                 # print(f"payload: {payload} start:{datetime.fromtimestamp(payload['before']/1000)}; end:{datetime.fromtimestamp(payload['after']/1000)}")
#                 # print(f"retry_counter: {retry_counter}")

#                 if len(prices_data) != limit:
#                     retry_counter += 1

#                 if retry_counter >= retry:
#                     break

#             data.data = historical_prices_datalist
#             # print(f"data length: {len(historical_prices_datalist)}")
#             return data

#         symbol = symbol.upper()

#         # inp_interval = INTERVAL_MAP.get(interval)
#         if not interval:
#             return models.CommonResponseSchema(
#                 success=False, error=True, data={}, msg="Invalid interval."
#             )

#         payload = {"symbol": symbol, "interval": interval}
#         if limit:
#             payload["limit"] = limit

#         if start:
#             start = UTC_TZ.localize(start)
#             payload["startTime"] = (
#                 int(datetime.timestamp(start)) * 1000
#             )  # convert to mm
#             if not end:
#                 end = datetime.now()
#         if end:
#             end = UTC_TZ.localize(end)
#             payload["endTime"] = int(datetime.timestamp(end)) * 1000  # convert to mm

#         interval_ts = TIMEDELTA_MAPPING_SEC[interval] * 1000

#         # print(f"payload:{payload}")
#         historical_prices = self._get(
#             "/swap/v2/quote/klines", self._remove_none(payload)
#         )  # history-

#         extra = {"symbol": symbol, "interval": interval}

#         if historical_prices.success:
#             if historical_prices.data:
#                 # print(historical_prices.data,'!!!!!!1!!!!!')
#                 if "startTime" in payload:
#                     historical_prices = iteration_history(
#                         symbol=symbol,
#                         interval=interval_ts,
#                         data=historical_prices,
#                         payload=payload,
#                         limit=limit,
#                     )
#                 else:
#                     historical_prices.data = historical_prices.data.get("data", [])
#                 # historical_prices.data = historical_prices.data.get("data", [])

#                 # print(f"------->:{historical_prices} - len:{len(historical_prices.data)}")
#                 return self._regular_historical_prices_payload(
#                     historical_prices, extra=extra
#                 )
#             else:
#                 historical_prices.msg = "query historical ohlc is empty"
#                 historical_prices.data = []

#         return historical_prices

#     def query_last_price(
#         self, symbol: str, interval: str = "1m", limit: int = 1
#     ) -> dict:
#         """ """
#         return models.CommonResponseSchema(
#             success=False,
#             error=True,
#             data={},
#             msg="BingX doesnt support query historical data yet.",
#         )

#     def query_prices(self, symbol: str) -> dict:
#         """
#         spot:
#             https://open-api.bingx.com/openApi/spot/v1/market/trades?symbol=BTC-USDT&limit=1
#             https://open-api.bingx.com/openApi/spot/v1/market/depth?symbol=BTC-USDT&limit=1
#         perp:
#             https://open-api.bingx.com/openApi/swap/v2/quote/trades?symbol=BTC-USDT&limit=1
#             https://open-api.bingx.com/openApi/swap/v2/quote/depth?symbol=BTC-USDT&limit=5
#         """
#         # return models.CommonResponseSchema(success=False, error=True, data={}, msg=f"BingX doesnt support query historical data yet.")
#         url = (
#             "/spot/v1/market/trades"
#             if self.DATA_TYPE == "spot"
#             else "/swap/v2/quote/trades"
#         )
#         payload = {"symbol": symbol, "limit": 1}
#         spots = self._get(url, payload)

#         if spots.success:
#             if spots.data:
#                 spots.data = spots.data.get("data", [])
#                 if spots.data:
#                     spots.data = spots.data[0]
#                 # print(f"!!!spots:{spots}")
#                 payload = models.TickerSchema()
#                 spots.data = payload.from_bingx_to_form(
#                     spots.data, symbol=symbol, datatype=self.DATA_TYPE
#                 )

#                 return spots
#             else:
#                 spots.msg = "query spots is empty"

#         return spots

#     def set_leverage(
#         self,
#         leverage: float,
#         symbol: str,
#         is_isolated: bool = True,
#         position_side: str = "long",
#     ) -> dict:
#         """ """

#         payload = {"symbol": symbol.upper(), "leverage": leverage}
#         if position_side:
#             position_side = position_side.upper()
#             assert position_side in ["LONG", "SHORT"]
#             payload["side"] = position_side
#         leverage_payload = self._post(
#             "/swap/v2/trade/leverage", self._remove_none(payload)
#         )

#         payload = {"symbol": symbol.upper()}
#         if is_isolated:
#             margin_type = "ISOLATED"
#         else:
#             margin_type = "CROSSED"
#         payload["marginType"] = margin_type
#         margin_type_payload = self._post(
#             "/swap/v2/trade/marginType", self._remove_none(payload)
#         )
#         if margin_type_payload.error:
#             return leverage_payload
#         # print(f"margin_type_payload:{margin_type_payload}")

#         if leverage_payload.success:
#             if leverage_payload.data:
#                 leverage_payload.data = models.LeverageSchema(
#                     symbol=symbol, leverage=leverage
#                 ).dict()
#                 return leverage_payload
#             else:
#                 leverage_payload.msg = "fail to config leverage_payload"

#         return leverage_payload

#     def query_position(self, symbol: str = None, force_return: bool = False) -> dict:
#         """query_position
#         {
#         "code": 0,
#         "msg": "",
#         "data": [
#                 {
#                     "symbol": "BTC-USDT",
#                     "positionId": "12345678",
#                     "positionSide": "LONG",
#                     "isolated": true,
#                     "positionAmt": "123.33",
#                     "availableAmt": "128.99",
#                     "unrealizedProfit": "1.22",
#                     "realisedProfit": "8.1",
#                     "initialMargin": "123.33",
#                     "avgPrice": "2.2",
#                     "leverage": 10}
#             ]
#         }

#         """

#         payload = {}
#         if symbol:
#             payload["symbol"] = symbol

#         position = self._get("/swap/v2/user/positions", self._remove_none(payload))
#         # print(f"position:{position}")
#         """
#         position:success=True error=False data={'code': '0', 'data': [{'adl': '1', 'availPos': '1', 'avgPx': '1333', 'baseBal': '', 'cTime': '1664201788281', 'ccy': 'USDT', 'deltaBS': '', 'deltaPA': '', 'gammaBS': '', 'gammaPA': '', 'imr': '', 'instId': 'ETH-USDT-SWAP', 'instType': 'SWAP', 'interest': '0', 'last': '1274.12', 'lever': '16', 'liab': '', 'liabCcy': '', 'liqPx': '1409.9576455948234', 'margin': '8.33125', 'markPx': '1274.09', 'mgnMode': 'isolated', 'mgnRatio': '24.805939925750945', 'mmr': '0.509636', 'notionalUsd': '127.40772591', 'optVal': '', 'pendingCloseOrdLiabVal': '', 'pos': '1', 'posCcy': '', 'posId': '494170398842572847', 'posSide': 'short', 'quoteBal': '', 'spotInUseAmt': '', 'spotInUseCcy': '', 'thetaBS': '', 'thetaPA': '', 'tradeId': '661054265', 'uTime': '1664201788281', 'upl': '5.891000000000009', 'uplRatio': '0.7070967741935501', 'usdPx': '', 'vegaBS': '', 'vegaPA': ''}], 'msg': ''} msg='query ok'
#         success=True error=False data=[{'asset_type': 'PERPETUAL', 'symbol': 'ETH-USDT-SWAP', 'name': 'ETHUSDT', 'side': 'SHORT', 'size': 1.0, 'position_value': 127.40772591, 'entry_price': None, 'leverage': 16.0, 'position_margin': 8.33125, 'initial_margin': 0.0, 'maintenance_margin': 0.509636, 'realised_pnl': None, 'unrealised_pnl': 5.891000000000009, 'is_isolated': True, 'auto_add_margin': None, 'liq_price': '1409.9576455948234', 'bust_price': None}] msg='query ok' =======query_position 1
#         """

#         if position.success:
#             if position.data:
#                 position.data = position.data.get("data", [])
#                 return self._regular_position_payload(position)
#             else:
#                 position.msg = "query future position is empty"

#         return position

#     def query_transfer(self, start: datetime = None, end: datetime = None) -> dict:
#         """ """
#         payload = {}
#         # ms - epoch time
#         if start:
#             start = UTC_TZ.localize(
#                 start
#             )  # NOTE: bybit start and end timestamp default regard it as utc timestamp
#             payload["startTime"] = int(datetime.timestamp(start)) * 1000
#         if end:
#             end = UTC_TZ.localize(end)
#             payload["endTime"] = int(datetime.timestamp(end)) * 1000
#         # if payload:
#         #     if not payload.get('start_time'):
#         #         payload['start_time'] = payload['end_time'] - 2592000
#         #     if not payload.get('end_time'):
#         #         payload['end_time'] = payload['start_time'] + 2592000
#         # # print(f"query_transfer payload:{payload}")

#         deposit_list = []
#         withdraw_list = []

#         deposit = self._get("/api/v3/capital/deposit/hisrec", params=payload)
#         # print(deposit,'..deposit...')
#         if deposit.success and deposit.data:
#             deposit_list += deposit.data
#         else:
#             return deposit

#         withdraw = self._get("/api/v3/capital/withdraw/history", params=payload)
#         # print(withdraw,'...withdraw..')
#         if withdraw.success and withdraw.data:
#             withdraw_list += withdraw.data
#         else:
#             return withdraw

#         # print(f'deposit_list:{deposit_list}; len:{len(deposit_list)}')

#         deposit.data = deposit_list
#         deposit_ret = self._regular_transfer_payload(
#             deposit, transfer_type="external", inout="in"
#         )
#         # print(f'withdraw_list:{withdraw_list}; len:{len(withdraw_list)}')
#         withdraw.data = withdraw_list
#         withdraw_ret = self._regular_transfer_payload(
#             withdraw, transfer_type="external", inout="out"
#         )

#         withdraw_ret.data = withdraw_ret.data + deposit_ret.data

#         return withdraw_ret

#     def universal_transfer(self, out: str, to: str, asset: str, amount: float) -> dict:
#         """ """
#         if out == "spot" and to == "future":
#             type_ = "FUND_PFUTURES"
#         elif out == "future" and to == "spot":
#             type_ = "PFUTURES_FUND"

#         asset = asset.upper()
#         payload: dict = {"asset": asset, "amount": amount, "type": type_}

#         return self._post("/api/v3/asset/transfer", payload)

#     # User Stream Endpoints
#     def stream_get_listen_key(self):
#         """
#         https://bingx-api.github.io/docs/spot/other-interface.html#generate-listen-key

#         response
#         {
#             "listenKey": "pqia91ma19a5s61cv6a81va65sdf19v8a65a1a5s61cv6a81va65sdf19v8a65a1"
#         }
#         """
#         return self._post("/user/auth/userDataStream", ignore=True)

#     def stream_keepalive(self, listen_key):
#         """PING a user data stream to prevent a time out.

#         https://binance-docs.github.io/apidocs/spot/en/#listen-key-spot

#         response   {}
#         """
#         params = {"listenKey": listen_key}
#         return self._put("/user/auth/userDataStream", params, ignore=True)

#     def stream_close(self, listen_key):
#         """Close out a user data stream.
#         https://binance-docs.github.io/apidocs/spot/en/#listen-key-spot
#         response  {}

#         """
#         params = {"listenKey": listen_key}
#         return self._delete("/user/auth/userDataStream", params, ignore=True)


# class BingxDataWebsocket(WebsocketClient):
#     DATA_TYPE = "linear"

#     def __init__(
#         self,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         ping_interval: int = 20,
#         datatype: str = "spot",
#         speed: float = 100,
#         debug: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         assert datatype in ["spot", "linear", "inverse", "option"]
#         self.DATA_TYPE = datatype

#         self.proxy_host: str = proxy_host
#         self.proxy_port: int = proxy_port
#         self.ping_interval: int = ping_interval

#         self.ticks: dict[str, dict] = defaultdict(dict)
#         self.reqid: int = 0

#         self._logged_in = False
#         self.key = None
#         self.secret = None
#         self.kbar_time: int = int(time.time()) - (int(time.time()) % 60)
#         self.bids_asks_data: dict[str, dict] = defaultdict(dict)
#         self.ticker_data: dict[str, dict] = defaultdict(dict)

#         self.on_tick = None
#         self.on_ticker_callback = None
#         self.on_depth_callback = None
#         self.on_kline_callback = None
#         self.on_connected_callback = None
#         self.on_disconnected_callback = None
#         self.on_error_callback = None

#         self.last_time = time.monotonic()
#         self.speed = speed
#         self.debug = debug

#     def unpack_data(self, data: str):
#         """
#         unpack the json data (Overload the func if data is not json)
#             ping: b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\n\xc8\xccK\x07\x04\x00\x00\xff\xff\xc3\x92\xe7\x85\x04\x00\x00\x00'
#         """
#         data = gzip.decompress(data)
#         try:
#             return json.loads(data)
#         except Exception as e:
#             d = data.decode("utf-8")
#             if d == "Ping":
#                 self.send_packet("Pong")
#                 return d
#             else:
#                 print(f"unpacked data: {data} - Error:{e}")
#                 return {}

#     def info(self):
#         return f"BingX {self.DATA_TYPE} Data Websocket start"

#     def connect(self):
#         """ """
#         if self.DATA_TYPE == "spot":
#             host = BINGX_SPOT_WEBSOCKET_HOST
#         elif self.DATA_TYPE == "linear":
#             host = BINGX_FUTURE_WEBSOCKET_HOST
#         else:
#             raise Exception("Unknown data type")

#         self.init(
#             host=host,
#             proxy_host=self.proxy_host,
#             proxy_port=self.proxy_port,
#             ping_interval=self.ping_interval,
#         )
#         self.start()

#         if self.waitfor_connection():
#             logger.info(self.info())
#             return True
#         else:
#             logger.info(self.info() + " failed")
#             return False

#     def on_connected(self) -> None:
#         """"""
#         if self.waitfor_connection():
#             print(f"BingX {self.DATA_TYPE} Market data websocket connect success")
#         else:
#             print(f"BingX {self.DATA_TYPE} Market data websocket connect fail")

#         if self.debug:
#             print(
#                 f"===> debug BingX {self.DATA_TYPE} Market data on_connected - self.ticks: {self.ticks}"
#             )

#         # resubscribe
#         if self.ticks:
#             for key, detail in self.ticks.items():
#                 req: dict = self.sub_stream(
#                     symbol=detail["symbol"],
#                     channel=detail["channel"],
#                     interval=detail.get("interval", "1m"),
#                 )

#                 if self.debug:
#                     print(
#                         f"[DEBUG]  on_connected, resubscribing the channels - req:{req} - detail:{detail}"
#                     )
#                 if req:
#                     self.send_packet(req)

#         if self.on_connected_callback:
#             self.on_connected_callback()

#     def on_disconnected(self) -> None:
#         print(
#             f"BingX {self.DATA_TYPE} market data Websocket disconnected, now try to connect @ {datetime.now()}"
#         )
#         if self.on_disconnected_callback:
#             self.on_disconnected_callback()

#     def inject_callback(self, channel: str, callbackfn: Callable):
#         if channel == "ticker":
#             self.on_ticker_callback = callbackfn
#         elif channel == "depth":
#             self.on_depth_callback = callbackfn
#         elif channel == "kline":
#             self.on_kline_callback = callbackfn
#         elif channel == "connect":
#             self.on_connected_callback = callbackfn
#         elif channel == "disconnect":
#             self.on_disconnected_callback = callbackfn
#         elif channel == "error":
#             self.on_error_callback = callbackfn
#         else:
#             Exception("invalid callback function")

#     def sub_stream(self, symbol, channel="depth", interval="1m"):
#         """
#         implement v1 public topic, due v2 kline don't return any data.
#         """
#         self.reqid += 1
#         # SPOT: { "id": "1212", "dataType":"BTC-USDT@depth"}
#         # PERP: { "id": "1212", "reqType": "sub", "dataType":"BTC-USDT@depth20"}
#         if self.DATA_TYPE == "spot":
#             sub = {"id": str(self.reqid)}
#         else:
#             sub = {"id": str(self.reqid), "reqType": "sub"}
#         # sub = {"id": str(self.reqid)}

#         if channel == "kline":
#             # <symbol>@kline_<interval>; interval: 1m, 3m. 5m. 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
#             if self.DATA_TYPE == "spot":
#                 sub["dataType"] = f"{symbol}@kline_1min"
#             else:
#                 sub["dataType"] = f"{symbol}@kline_{interval}"
#         elif channel == "depth":
#             # <symbol>@depth<level>; level: 5, 10, 20, 50, 100
#             if self.DATA_TYPE == "spot":
#                 sub["dataType"] = f"{symbol}@depth"
#             else:
#                 sub["dataType"] = f"{symbol}@depth20"
#         elif channel == "ticker":
#             # sub["params"] = [f"{symbol}@{SPOT_MAPPING_CHANNEL[channel]}"]
#             if self.DATA_TYPE == "spot":
#                 sub["dataType"] = f"{symbol}@kline_1min"
#             else:
#                 sub["dataType"] = f"{symbol}@kline_1m"

#         return sub

#     def subscribe(self, symbols, on_tick=None, channel="depth", interval="1m") -> None:
#         """"""
#         # self.channel = channel
#         # self.interval = interval
#         if channel not in ["depth", "kline", "ticker"]:
#             raise Exception("invalid subscription")
#         if self.DATA_TYPE == "spot" and interval != "1m":
#             raise Exception(
#                 "invalid interval, currently only support 1 minute interval of BingX Spot websocket data."
#             )

#         if on_tick:
#             self.on_tick = on_tick
#             self.inject_callback(channel=channel, callbackfn=on_tick)

#         if isinstance(symbols, str):
#             symbols = [symbols]
#         elif isinstance(symbols, list):
#             pass
#         else:
#             logger.debug("invalid input symbols.")
#             return
#         if self.debug:
#             print(f"[BEBUG] symbols:{symbols}; channel:{channel}; interval:{interval}")
#         for symbol in symbols:
#             if symbol in self.ticks:
#                 return
#             # for websocket - all show lower symbol in binance
#             symbol = symbol.upper()

#             # tick data dict
#             asset_type = models.bingx_asset_type(symbol=symbol, datatype=self.DATA_TYPE)
#             tick = {
#                 "symbol": symbol,
#                 "exchange": EXCHANGE,
#                 "channel": channel,
#                 "asset_type": asset_type,
#                 "name": models.normalize_name(symbol, asset_type),
#             }
#             key = ""
#             if channel == "kline":
#                 key = f"{symbol}|{interval}"
#                 tick["interval"] = interval
#             else:
#                 key = f"{symbol}|{channel}"

#             self.ticks[key] = tick
#             req: dict = self.sub_stream(
#                 symbol=symbol, channel=channel, interval=interval
#             )
#             if self.debug:
#                 print(
#                     f"[BEBUG] symbols:{symbols}; channel:{channel}; interval:{interval} -- self.ticks:{self.ticks} | send packet req:{req}"
#                 )
#             if req:
#                 self.send_packet(req)

#     def on_packet(self, packet: dict) -> None:
#         """on_packet"""
#         # logger.debug(f"on_packet event {packet}")
#         """
#         spot:
#         >>> on_packet: {'id': '1', 'code': 0, 'msg': 'SUCCESS', 'timestamp': 1679468278558}
#         >>> on_packet: {'id': '2', 'code': 0, 'msg': 'SUCCESS', 'timestamp': 1679468278558}
#         >>> on_packet: {'code': 0, 'data': {'asks': [['28286.15', '0.00005'], ['28285.60', '0.00024'], ['28285.50', '0.00002'], ['28284.08', '0.00005'], ['28283.92', '0.00002'], ['28283.80', '0.00012'], ['28281.64', '0.00033'], ['28281.57', '0.00002'], ['28281.45', '0.00002'], ['28281.44', '0.00006'], ['28281.04', '0.00002'], ['28280.27', '0.00020'], ['28279.66', '0.00010'], ['28272.72', '0.00010'], ['28250.00', '0.00202'], ['28248.00', '0.00365'], ['28245.44', '0.00261'], ['28245.32', '0.00029'], ['28234.10', '9.57725'], ['28232.47', '41.64864'], ['28226.35', '13.47278'], ['28226.24', '9.83260'], ['28221.45', '14.01364'], ['28221.25', '9.47351'], ['28218.02', '6.35341'], ['28217.22', '5.40671'], ['28216.84', '5.50851'], ['28212.33', '6.24262'], ['28211.52', '5.72443'], ['28211.39', '7.04893'], ['28209.41', '6.78676'], ['28207.35', '6.71760'], ['28207.33', '6.35385'], ['28205.00', '0.01027'], ['28203.53', '2.03309'], ['28203.20', '1.76045'], ['28203.10', '2.58162'], ['28200.59', '0.30672'], ['28200.30', '0.31827'], ['28199.86', '2.08352'], ['28199.53', '0.29449'], ['28199.23', '0.31126'], ['28199.04', '1.57219'], ['28198.78', '0.33883'], ['28198.61', '0.26982'], ['28198.52', '1.77642'], ['28198.15', '0.27642'], ['28197.92', '0.30250'], ['28197.75', '0.30886'], ['28197.69', '0.34543']], 'bids': [['28191.81', '0.24555'], ['28191.80', '0.27383'], ['28191.78', '0.32127'], ['28191.76', '0.29343'], ['28191.74', '0.24450'], ['28191.72', '0.31279'], ['28191.71', '0.28270'], ['28191.69', '0.26916'], ['28191.67', '0.27642'], ['28191.65', '0.35061'], ['28191.63', '0.26641'], ['28191.62', '0.29109'], ['28191.38', '0.30196'], ['28189.32', '2.00226'], ['28187.00', '1.47288'], ['28185.94', '3.22978'], ['28184.53', '3.12347'], ['28183.38', '5.74076'], ['28183.15', '6.97501'], ['28180.93', '6.22355'], ['28180.30', '4.85447'], ['28179.30', '6.95728'], ['28175.97', '5.26969'], ['28174.72', '6.55910'], ['28171.97', '7.04695'], ['28169.26', '15.01806'], ['28159.78', '0.00042'], ['28157.89', '11.88534'], ['28154.38', '0.00035'], ['28150.00', '0.00008'], ['28149.82', '0.00016'], ['28145.32', '12.42028'], ['28143.24', '12.66966'], ['28138.90', '13.67412'], ['28133.18', '0.00004'], ['28126.13', '0.00020'], ['28122.08', '0.00525'], ['28119.26', '0.00015'], ['28118.03', '0.00009'], ['28117.25', '0.00003'], ['28117.08', '0.00017'], ['28115.55', '0.00003'], ['28115.53', '0.00014'], ['28112.72', '0.00023'], ['28110.25', '0.00047'], ['28110.00', '0.00017'], ['28108.92', '0.00030'], ['28105.10', '0.00262'], ['28104.45', '0.00002'], ['28103.83', '0.00031']]}, 'dataType': 'BTC-USDT@depth', 'success': True}
#         >>> on_packet: {'ping': '42b678e01f8b4f0d99e8a8ddb76798a7', 'time': '2023-03-22T14:58:18.136+0800'}
#         >>> on_packet: {'code': 0, 'data': {'E': 1679468493688, 'K': {'T': 1679468519999, 'c': '28203.53', 'h': '28204.58', 'i': '1min', 'l': '28194.03', 'n': 24, 'o': '28195.05', 'q': '20080.353547', 's': 'BTC-USDT', 't': 1679468460000, 'v': '0.712130'}, 'e': 'kline', 's': 'BTC-USDT'}, 'dataType': 'BTC-USDT@kline_1min', 'success': True}

#         linear:
#         >>> on_packet: {'code': 0, 'dataType': 'BTC-USDT@kline_1m', 's': 'BTC-USDT', 'data': [{'c': '27592.5', 'o': '27610.0', 'h': '27610.0', 'l': '27592.0', 'v': '4.4903', 'T': 1679564700000}]}
#         >>> on_packet: {'code': 0, 'dataType': 'BTC-USDT@depth20', 'data': {'bids': [['27592.0', '26.1350'], ['27591.5', '28.9701'], ['27591.0', '41.8478'], ['27590.5', '17.8037'], ['27590.0', '4.0989'], ['27589.5', '113.8348'], ['27589.0', '38.1398'], ['27588.5', '65.0626'], ['27588.0', '51.3085'], ['27587.5', '55.5173'], ['27587.0', '18.0973'], ['27586.5', '58.3275'], ['27586.0', '88.5882'], ['27585.5', '99.5280'], ['27585.0', '90.1937'], ['27584.5', '162.3834'], ['27584.0', '113.0469'], ['27583.5', '141.1507'], ['27583.0', '102.4179'], ['27582.5', '126.3345']], 'asks': [['27602.0', '84.0728'], ['27601.5', '99.1484'], ['27601.0', '74.1201'], ['27600.5', '108.5532'], ['27600.0', '141.4938'], ['27599.5', '129.0975'], ['27599.0', '87.8726'], ['27598.5', '108.6223'], ['27598.0', '56.9824'], ['27597.5', '92.9177'], ['27597.0', '23.2437'], ['27596.5', '59.3911'], ['27596.0', '57.5477'], ['27595.5', '67.5911'], ['27595.0', '128.6192'], ['27594.5', '4.2620'], ['27594.0', '17.7061'], ['27593.5', '24.0720'], ['27593.0', '43.2912'], ['27592.5', '33.2474']]}}

#         """
#         if self.debug:
#             print(f">>> on_packet: {packet}")
#         if isinstance(packet, str):
#             return
#         stream: str = packet.get("dataType", None)
#         if not stream:
#             if "ping" in packet or ("code" in packet and not packet["code"]):
#                 # successfully subscribed {'id': '2', 'code': 0, 'msg': 'SUCCESS', 'timestamp': 1675755354559}
#                 return
#             print(f"Cannot find dataType on packet: {packet}")
#             return

#         symbol, raw_channel = stream.split("@")
#         channel = raw_channel.split("_")[0]
#         # tick: dict = self.ticks[symbol]

#         if channel == "kline":
#             if self.ticks.get(f"{symbol}|ticker"):
#                 tick: dict = self.ticks[f"{symbol}|ticker"]
#                 if self.DATA_TYPE == "spot":
#                     data: dict = packet["data"]["K"]
#                     tick["volume"] = float(data["v"])
#                     tick["last_price"] = float(data["c"])
#                     tick["turnover"] = float(data["q"])
#                 else:
#                     data: dict = packet["data"][0]
#                     tick["volume"] = float(data["v"])
#                     tick["last_price"] = float(data["c"])
#                     tick["turnover"] = tick["last_price"] * tick["volume"]
#                 tick["datetime"] = BingxClient.generate_datetime()

#                 self.on_tick(copy(tick))

#             if self.DATA_TYPE == "spot":
#                 interval = "1m"  # packet["data"]["k"]["i"]
#                 tick: dict = self.ticks.get(f"{symbol}|{interval}")
#                 if not tick:
#                     return
#                 data: dict = packet["data"]["K"]
#                 tick["volume"] = float(data["v"])
#                 tick["open"] = float(data["o"])
#                 tick["high"] = float(data["h"])
#                 tick["low"] = float(data["l"])
#                 tick["close"] = float(data["c"])
#                 tick["turnover"] = float(data["q"])
#                 tick["start"] = BingxClient.generate_datetime_ts(
#                     float(data["t"])
#                 )  # bar start time
#                 tick["datetime"] = BingxClient.generate_datetime()
#             else:
#                 interval = raw_channel.split("_")[1]
#                 tick: dict = self.ticks.get(f"{symbol}|{interval}")
#                 if not tick:
#                     return
#                 data: dict = packet["data"][0]
#                 tick["volume"] = float(data["v"])
#                 tick["open"] = float(data["o"])
#                 tick["high"] = float(data["h"])
#                 tick["low"] = float(data["l"])
#                 tick["close"] = float(data["c"])
#                 tick["turnover"] = tick["close"] * tick["volume"]
#                 tick["start"] = BingxClient.generate_datetime_ts(
#                     float(data["T"])
#                 )  # bar start time
#                 tick["datetime"] = BingxClient.generate_datetime()

#         # elif channel == "ticker":
#         #     tick: dict = self.ticks[f"{symbol}|ticker"]
#         #     data: dict = packet["data"]
#         #     tick["prev_open_24h"] = float(data["o"])
#         #     tick["prev_high_24h"] = float(data["h"])
#         #     tick["prev_low_24h"] = float(data["l"])
#         #     tick["prev_volume_24h"] = float(data["v"])
#         #     tick["prev_turnover_24h"] = float(data["q"])

#         #     tick["last_price"] = float(data["c"])
#         #     tick["price_change_pct"] = float(data["P"]) / 100
#         #     tick["price_change"] = float(data["p"])
#         #     tick["prev_close_24h"] = tick["last_price"] + tick["price_change"]

#         #     tick["datetime"] = datetime.now()

#         else:
#             tick: dict = self.ticks[f"{symbol}|depth"]
#             data: dict = packet["data"]
#             bids: list = data["bids"]

#             for n in range(min(5, len(bids))):
#                 price, volume = bids[n]

#                 tick["bid_price_" + str(n + 1)] = float(price)
#                 tick["bid_volume_" + str(n + 1)] = float(volume)

#             asks: list = data["asks"]
#             for n in range(min(5, len(asks))):
#                 price, volume = asks[n]
#                 tick["ask_price_" + str(n + 1)] = float(price)
#                 tick["ask_volume_" + str(n + 1)] = float(volume)

#             tick["datetime"] = BingxClient.generate_datetime()

#         self.on_tick(copy(tick))


# class BingxTradeWebsocket(WebsocketClient):
#     """
#     BingX trade Websocket
#     """

#     DATA_TYPE = "linear"

#     def __init__(
#         self,
#         on_account_callback=None,
#         on_order_callback=None,
#         on_trade_callback=None,
#         on_position_callback=None,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         datatype: str = "spot",
#         debug: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         assert datatype in ["spot", "linear"]
#         self.DATA_TYPE = datatype
#         self.key = None
#         self.secret = None
#         self._logged_in = False
#         self.on_account_callback = on_account_callback
#         self.on_order_callback = on_order_callback
#         self.on_trade_callback = on_trade_callback
#         self.on_position_callback = on_position_callback
#         self.client = BingxClient(datatype=datatype)
#         self.listen_key: str = ""
#         self.refresh_key_time: float = time.time()
#         self.proxy_host = proxy_host
#         self.proxy_port = proxy_port
#         self.timer_listen_key = RepeatTimer(30 * 60, self._keep_alive)
#         self.url_base = None
#         self.reqid: int = 0

#         self.debug = debug

#     def unpack_data(self, data: str):
#         """
#         unpack the json data (Overload the func if data is not json)
#         """
#         data = gzip.decompress(data)
#         try:
#             return json.loads(data)
#         except Exception as e:
#             d = data.decode("utf-8")
#             if d == "Ping":
#                 self.send_packet("Pong")
#                 return d
#             else:
#                 print(f"unpacked data: {data} - Error:{e}")
#                 return {}

#     def __del__(self):
#         if self._logged_in:
#             self.timer_listen_key.cancel()
#             self.client.stream_close(self.listen_key)

#     # def inject_callback(self, callbackfn, channel='orders', ):
#     #     if channel == 'account':
#     #         self.on_account_callback = callbackfn
#     #     elif channel == 'orders':
#     #         self.on_order_callback = callbackfn
#     #     elif channel == 'trades':
#     #         self.on_trade_callback = callbackfn
#     #     else:
#     #         Exception("invalid callback function")

#     def info(self):
#         return f"BingX {self.DATA_TYPE} Trade Websocket Start"

#     def connect(self, key: str, secret: str):
#         """
#         Give api key and secret for Authorization.
#         """
#         if self.DATA_TYPE == "spot":
#             host = BINGX_SPOT_WEBSOCKET_HOST
#         elif self.DATA_TYPE == "linear":
#             host = BINGX_FUTURE_WEBSOCKET_HOST
#         else:
#             raise Exception("Unknown data type")

#         self.url_base = host
#         self.key = key
#         self.secret = secret
#         if not self._logged_in:
#             self._login()
#         logger.info(self.info())
#         return True

#     def _login(self):
#         """
#         Authorize websocket connection.
#         """
#         api_key = self.key
#         api_secret = self.secret

#         if api_key is None or api_secret is None:
#             raise PermissionError("Authenticated endpoints require keys.")

#         self.client.connect(api_key, api_secret)

#         key_payload = self.client.stream_get_listen_key()
#         if self.debug:
#             print(f"[DEBUG] auth res: {key_payload}")
#         if key_payload.success:
#             self.listen_key = key_payload.data["listenKey"]
#             # self.host = f"{self.url_base}/{self.listen_key}"

#             self.init(
#                 f"{self.url_base}?listenKey={self.listen_key}",
#                 self.proxy_host,
#                 self.proxy_port,
#             )
#             self.start()

#             self.timer_listen_key.daemon = True
#             self.timer_listen_key.start()
#             self._logged_in = True
#             if self.debug:
#                 print(
#                     f"login with listenKey {self.listen_key}, url: {self.url_base}?listenKey={self.listen_key}"
#                 )
#         else:
#             raise PermissionError(
#                 "Authenticated endpoints require listenKey for subscription."
#             )

#     def _keep_alive(self):
#         """
#         Keepalive a user data stream to prevent a time out.
#         User data streams will close after 60 minutes.
#         It's recommended to send a ping about every 30 minutes.
#         """
#         self.refresh_key_time = time.time()
#         if self.debug:
#             print(f"keep_alive: {self.refresh_key_time}")

#         self.client.stream_keepalive(self.listen_key)

#     def on_disconnected(self) -> None:
#         print(
#             f"BingX {self.DATA_TYPE} user trade data Websocket disconnected, now try to connect @ {datetime.now()}"
#         )
#         # self._logged_in = False
#         # self._login()

#     def on_connected(self) -> None:
#         print(f"BingX {self.DATA_TYPE} user trade data Websocket connect success")
#         if self.DATA_TYPE == "spot":
#             self.reqid += 1
#             req = {"id": str(self.reqid), "dataType": "ACCOUNT_UPDATE"}
#             if self.DATA_TYPE != "spot":
#                 req["reqType"] = "sub"
#             if self.debug:
#                 print(f"[DEBUG] first req: {req}")
#             self.send_packet(req)

#             self.reqid += 1
#             req = {
#                 "id": str(self.reqid),
#                 "dataType": "spot.executionReport"
#                 if self.DATA_TYPE == "spot"
#                 else "ORDER_TRADE_UPDATE",
#             }
#             if self.DATA_TYPE != "spot":
#                 req["reqType"] = "sub"
#             if self.debug:
#                 print(f"[DEBUG] second req: {req}")
#             self.send_packet(req)

#     def on_packet(self, packet: dict) -> None:
#         """keep alive check on packet is comming"""
#         # print(f"BingX Future user data ws, on_packet event {packet}")
#         """
#         linear:
#         >>> on_packet: {'e': 'ORDER_TRADE_UPDATE', 'E': 1679566778035, 'o': {'s': 'BTC-USDT', 'c': '', 'i': 1638459780701884416, 'S': 'BUY', 'o': 'LIMIT', 'q': '0.00080000', 'p': '25000.00000000', 'ap': '0.00000000', 'x': 'TRADE', 'X': 'CANCELED', 'N': 'USDT', 'n': '0.00000000', 'T': 0, 'wt': 'MARK_PRICE', 'ps': 'LONG', 'rp': '0.00000000'}}

#         >>> on_packet: {'e': 'ORDER_TRADE_UPDATE', 'E': 1680094496622, 'o': {'s': 'BTC-USDT', 'c': '', 'i': 1641061364371820544, 'S': 'BUY', 'o': 'LIMIT', 'q': '0.00040000', 'p': '25000.00000000', 'ap': '0.00000000', 'x': 'TRADE', 'X': 'CANCELED', 'N': 'USDT', 'n': '0.00000000', 'T': 0, 'wt': 'MARK_PRICE', 'ps': 'LONG', 'rp': '0.00000000'}}
#         packet on_order: {'s': 'BTC-USDT', 'c': '', 'i': 1641061364371820544, 'S': 'BUY', 'o': 'LIMIT', 'q': '0.00040000', 'p': '25000.00000000', 'ap': '0.00000000', 'x': 'TRADE', 'X': 'CANCELED', 'N': 'USDT', 'n': '0.00000000', 'T': 0, 'wt': 'MARK_PRICE', 'ps': 'LONG', 'rp': '0.00000000'}
#         >>> on_order_callback - {'symbol': 'BTC-USDT', 'asset_type': 'SPOT', 'name': 'BTC/USDT', 'exchange': 'BINGX', 'ori_order_id': '1641061364371820544', 'order_id': None, 'type': 'LIMIT', 'side': 'BUY', 'price': 25000.0, 'quantity': 0.0004, 'executed_quantity': None, 'status': 'CANCELED', 'datetime': datetime.datetime(2023, 3, 29, 20, 54, 56, 622000, tzinfo=<DstTzInfo 'Asia/Hong_Kong' HKT+8:00:00 STD>)}
#         >>> on_packet: Ping
#         >>> on_packet: Ping
#         >>> on_packet: Ping
#         >>> on_packet: {'e': 'ACCOUNT_UPDATE', 'E': 1680094514116, 'a': {'m': 'ORDER', 'B': [{'a': 'USDT', 'wb': '3.99556190', 'cw': '2.29197141', 'bc': '0'}], 'P': [{'s': 'BTC-USDT', 'pa': '0.00030000', 'ep': '28396.50000000', 'up': '-0.00010000', 'mt': 'isolated', 'iw': '1.70359049', 'ps': 'SHORT'}]}}
#         on_account raw packet: {'m': 'ORDER', 'B': [{'a': 'USDT', 'wb': '3.99556190', 'cw': '2.29197141', 'bc': '0'}], 'P': [{'s': 'BTC-USDT', 'pa': '0.00030000', 'ep': '28396.50000000', 'up': '-0.00010000', 'mt': 'isolated', 'iw': '1.70359049', 'ps': 'SHORT'}]}
#         >>> on_account_callback - {'symbol': 'USDT', 'asset_type': 'SPOT', 'balance': 3.9955619, 'available': 3.9955619, 'frozen': None, 'exchange': 'BINGX'}
#         packet on_position: {'m': 'ORDER', 'B': [{'a': 'USDT', 'wb': '3.99556190', 'cw': '2.29197141', 'bc': '0'}], 'P': [{'s': 'BTC-USDT', 'pa': '0.00030000', 'ep': '28396.50000000', 'up': '-0.00010000', 'mt': 'isolated', 'iw': '1.70359049', 'ps': 'SHORT'}]}
#         >>> on_position_callback - {'symbol': 'BTC-USDT', 'asset_type': 'PERPETUAL', 'name': 'BTCUSDT', 'size': 0.0003, 'entry_price': 28396.5, 'unrealised_pnl': -0.0001, 'is_isolated': True, 'position_margin': 1.70359049, 'position_side': 'SHORT', 'exchange': 'BINGX'}
#         >>> on_packet: {'e': 'ORDER_TRADE_UPDATE', 'E': 1680094514124, 'o': {'s': 'BTC-USDT', 'c': '', 'i': 1641061451231662080, 'S': 'SELL', 'o': 'MARKET', 'q': '0.00030000', 'p': '28397.20000000', 'ap': '28396.50000000', 'x': 'TRADE', 'X': 'FILLED', 'N': 'USDT', 'n': '-0.00425947', 'T': 1680094514120, 'wt': 'MARK_PRICE', 'ps': 'SHORT', 'rp': '0.00000000'}}
#         packet on_trade: {'e': 'ORDER_TRADE_UPDATE', 'E': 1680094514124, 'o': {'s': 'BTC-USDT', 'c': '', 'i': 1641061451231662080, 'S': 'SELL', 'o': 'MARKET', 'q': '0.00030000', 'p': '28397.20000000', 'ap': '28396.50000000', 'x': 'TRADE', 'X': 'FILLED', 'N': 'USDT', 'n': '-0.00425947', 'T': 1680094514120, 'wt': 'MARK_PRICE', 'ps': 'SHORT', 'rp': '0.00000000'}}

#         =====================================
#         >>> on_packet: {'e': 'ORDER_TRADE_UPDATE', 'E': 1680095496286, 'o': {'s': 'BTC-USDT', 'c': '', 'i': 1641065570621198336, 'S': 'SELL', 'o': 'MARKET', 'q': '0.00010000', 'p': '28495.40000000', 'ap': '28495.00000000', 'x': 'TRADE', 'X': 'FILLED', 'N': 'USDT', 'n': '-0.00142475', 'T': 1680095496283, 'wt': 'MARK_PRICE', 'ps': 'SHORT', 'rp': '0.00000000'}}
#         packet on_order: {'s': 'BTC-USDT', 'c': '', 'i': 1641065570621198336, 'S': 'SELL', 'o': 'MARKET', 'q': '0.00010000', 'p': '28495.40000000', 'ap': '28495.00000000', 'x': 'TRADE', 'X': 'FILLED', 'N': 'USDT', 'n': '-0.00142475', 'T': 1680095496283, 'wt': 'MARK_PRICE', 'ps': 'SHORT', 'rp': '0.00000000'}
#         >>> on_trade_callback - {'symbol': 'BTC-USDT', 'asset_type': 'SPOT', 'name': 'BTC/USDT', 'exchange': 'BINGX', 'ori_order_id': '1641065570621198336', 'order_id': None, 'side': 'SELL', 'price': 28495.4, 'quantity': 0.0001, 'commission': -0.00142475, 'commission_asset': 'USDT', 'datetime': datetime.datetime(2023, 3, 29, 21, 11, 36, 283000, tzinfo=<DstTzInfo 'Asia/Hong_Kong' HKT+8:00:00 STD>)}
#         >>> on_packet: {'e': 'ACCOUNT_UPDATE', 'E': 1680095496279, 'a': {'m': 'ORDER', 'B': [{'a': 'USDT', 'wb': '3.99129163', 'cw': '0.61714240', 'bc': '0'}], 'P': [{'s': 'BTC-USDT', 'pa': '0.00060000', 'ep': '28432.50000000', 'up': '-0.03770000', 'mt': 'isolated', 'iw': '3.37414923', 'ps': 'SHORT'}]}}
#         on_account raw packet: {'m': 'ORDER', 'B': [{'a': 'USDT', 'wb': '3.99129163', 'cw': '0.61714240', 'bc': '0'}], 'P': [{'s': 'BTC-USDT', 'pa': '0.00060000', 'ep': '28432.50000000', 'up': '-0.03770000', 'mt': 'isolated', 'iw': '3.37414923', 'ps': 'SHORT'}]}
#         >>> on_account_callback - {'symbol': 'USDT', 'asset_type': 'SPOT', 'balance': 3.99129163, 'available': 3.99129163, 'frozen': None, 'exchange': 'BINGX'}
#         packet on_position: {'m': 'ORDER', 'B': [{'a': 'USDT', 'wb': '3.99129163', 'cw': '0.61714240', 'bc': '0'}], 'P': [{'s': 'BTC-USDT', 'pa': '0.00060000', 'ep': '28432.50000000', 'up': '-0.03770000', 'mt': 'isolated', 'iw': '3.37414923', 'ps': 'SHORT'}]}
#         >>> on_position_callback - {'symbol': 'BTC-USDT', 'asset_type': 'PERPETUAL', 'name': 'BTCUSDT', 'size': 0.0006, 'entry_price': 28432.5, 'unrealised_pnl': -0.0377, 'is_isolated': True, 'position_margin': 3.37414923, 'position_side': 'SHORT', 'exchange': 'BINGX'}

#         """
#         if self.debug:
#             print(f">>> on_packet: {packet}")
#         if isinstance(packet, str):
#             return
#         stream: str = packet.get("dataType", None)
#         if not stream:
#             if "e" in packet:
#                 pass
#             else:
#                 if "ping" in packet or ("code" in packet and not packet["code"]):
#                     # successfully subscribed {'id': '2', 'code': 0, 'msg': 'SUCCESS', 'timestamp': 1675755354559}
#                     return
#                 print(f"Cannot find dataType on packet: {packet}")
#                 return

#         if packet.get("e", "") == "ACCOUNT_UPDATE":
#             self.on_account(packet)
#             if self.DATA_TYPE != "spot":
#                 self.on_position(packet)
#         elif (
#             self.DATA_TYPE == "spot" and packet.get("e", "") == "ORDER_TRADE_UPDATE"
#         ) or (packet.get("e", "") == "ORDER_TRADE_UPDATE"):
#             # if packet["o"]["x"] == "TRADE" and packet["o"]["X"] != 'CANCELED' :
#             #     self.on_trade(packet)
#             # else:
#             #     self.on_order(packet)
#             self.on_order(packet)

#     def on_account(self, packet: dict) -> None:
#         """on account balance update"""
#         packet = packet["a"]
#         if self.debug:
#             print(f"on_account raw packet: {packet}")
#         for d in packet["B"]:
#             account = {
#                 "symbol": d["a"],
#                 "asset_type": ASSET_TYPE_SPOT,
#                 # 'name':  models.normalize_name(d["a"], ASSET_TYPE_SPOT),
#                 "balance": float(d["wb"]),
#                 "available": float(d["wb"]),
#                 "frozen": 0.0,
#                 "exchange": EXCHANGE,
#             }

#             if account and account["symbol"] and self.on_account_callback:
#                 self.on_account_callback(account)

#     def on_order(self, packet: dict) -> None:
#         """on order place/cancel"""
#         # filter not support trade type
#         if self.DATA_TYPE == "spot":
#             datas = packet["data"]
#         else:
#             datas = packet["o"]
#         if self.debug:
#             print(f"packet on_order: {datas}")
#         if datas["o"] not in ["LIMIT", "MARKET"]:
#             return
#         symbol = datas["s"]
#         asset_type = bingx_asset_type(symbol, datatype=self.DATA_TYPE)
#         order = {
#             "symbol": symbol,
#             "asset_type": asset_type,
#             "name": models.normalize_name(symbol, asset_type),
#             "exchange": EXCHANGE,
#             "ori_order_id": str(datas["i"]),
#             "order_id": None,
#             "type": datas["o"],  # LIMIT, MARKET
#             "side": datas["S"],  # BUY, SELL
#             "price": float(datas["p"]),
#             "quantity": float(datas["q"]),
#             "executed_quantity": float(datas["z"]) if "z" in datas else None,
#             "status": datas[
#                 "X"
#             ],  # NEW, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED, EXPIRED(CANCELED)
#             "position_side": datas["ps"] if "ps" in datas else None,
#             "datetime": BingxClient.generate_datetime_ts(
#                 float(datas["O"]) if "O" in datas else float(datas["T"])
#             ),
#         }
#         if datas["X"] != "FILLED":
#             if order and self.on_order_callback:
#                 self.on_order_callback(order)
#             return

#         # round the number
#         # trade_volume = float(packet["l"])

#         # if not trade_volume:
#         #     return

#         trade = {
#             "symbol": order["symbol"],
#             "asset_type": order["asset_type"],
#             "name": order["name"],
#             "exchange": order["exchange"],
#             "ori_order_id": order["ori_order_id"],
#             "order_id": order["order_id"],
#             "side": order["side"],
#             "price": order["price"],  # float(packet["L"]),
#             "quantity": order["quantity"],  # float(packet["q"]),
#             "position_side": order["position_side"],
#             "commission": float(datas["n"]),
#             "commission_asset": datas["N"],
#             "datetime": BingxClient.generate_datetime_ts(
#                 float(datas["T"])
#             ),  # user E -> eventtime?
#         }
#         if trade and self.on_trade_callback:
#             self.on_trade_callback(trade)

#     # def on_trade(self, packet: dict) -> None:
#     #     """
#     #     [DEBUG] Bybit linear Trade Websocket - on_packet event {'id': '1193481299665b2-56be-40da-8a29-f488cb47190b', 'topic': 'execution', 'creationTime': 1678888392300, 'data': [{'category': 'linear', 'symbol': 'BTCUSDT', 'execFee': '0.01490148', 'execId': 'dae9c7a8-ceb9-51fa-8881-d5157ecc2c51', 'execPrice': '24835.8', 'execQty': '0.001', 'execType': 'Trade', 'execValue': '24.8358', 'isMaker': False, 'feeRate': '0.0006', 'tradeIv': '', 'markIv': '', 'blockTradeId': '', 'markPrice': '24883.13', 'indexPrice': '', 'underlyingPrice': '', 'leavesQty': '0', 'orderId': 'e9b9e551-081b-414c-a109-0e56d225ecd2', 'orderLinkId': '', 'orderPrice': '26116.9', 'orderQty': '0.001', 'orderType': 'Market', 'stopOrderType': 'UNKNOWN', 'side': 'Buy', 'execTime': '1678888392289', 'isLeverage': '0'}]}
#     #     """
#     #     if self.debug:
#     #         print(f"packet on_trade: {packet}")
#     #     datas: dict = packet["data"]
#     #     # if self.DATA_TYPE == "spot":
#     #     #     datas = packet["data"]
#     #     # else:
#     #     #     dt = packet['E']
#     #     #     datas = packet["o"]

#     #     trade_list = list()
#     #     for data in datas:
#     #         trade = models.TradeSchema()#.model_construct(data)
#     #         trade = trade.from_bingx_to_form(data, datatype=self.DATA_TYPE)
#     #         trade_list.append(trade)

#     #         if self.on_trade_callback:  # and trade_list:
#     #             # self.on_trade_callback(trade_list)
#     #             # print(trade,'....')
#     #             self.on_trade_callback(trade)

#     def on_position(self, packet: dict) -> None:
#         """
#         [DEBUG] Bybit linear Trade Websocket - on_packet event {'id': '119348107a024b9-ed00-400d-b81c-105c5eb57e7e', 'topic': 'position', 'creationTime': 1678887821206, 'data': [{'positionIdx': 0, 'tradeMode': 0, 'riskId': 1, 'riskLimitValue': '2000000', 'symbol': 'BTCUSDT', 'side': 'Buy', 'size': '0.006', 'entryPrice': '23989.61666667', 'leverage': '10', 'positionValue': '143.9377', 'markPrice': '24922.99', 'positionIM': '14.47149636', 'positionMM': '0.79741486', 'takeProfit': '0', 'stopLoss': '0', 'trailingStop': '0', 'unrealisedPnl': '5.60024', 'cumRealisedPnl': '0.11838384', 'createdTime': '1676887474897', 'updatedTime': '1678887821202', 'tpslMode': 'Full', 'liqPrice': '', 'bustPrice': '', 'positionStatus': 'Normal'}]}
#         """
#         print(
#             f"[DEBUG] bingx on_position packet: {packet}"
#         )  # TODO: for FUNDING_FEE event couldnt catch symbol
#         packet = packet["a"]
#         if self.debug:
#             print(f"packet on_position: {packet}")
#         datas: dict = packet.get("P", [])
#         for data in datas:
#             # position = models.PositionSchema()#.model_construct(data)
#             # position = position.from_bingx_to_form(data, datatype=self.DATA_TYPE)
#             symbol = data["s"]
#             if not symbol:
#                 print(f"[ERROR] bingx on_position symbol is None - datas:{datas}")
#                 return
#             asset_type = bingx_asset_type(symbol=symbol, datatype=self.DATA_TYPE)
#             position = {
#                 "symbol": symbol,
#                 "asset_type": asset_type,
#                 "name": models.normalize_name(data["s"], asset_type),
#                 "size": float(data["pa"]),
#                 "entry_price": float(data["ep"]),
#                 "unrealised_pnl": float(data["up"]),
#                 "realised_pnl": None,
#                 "is_isolated": data["mt"] == "isolated",
#                 "position_margin": float(data["iw"]),
#                 "position_side": data["ps"],
#                 "side": data["ps"],
#                 "exchange": EXCHANGE,
#             }

#             if position and self.on_position_callback:
#                 self.on_position_callback(position)
