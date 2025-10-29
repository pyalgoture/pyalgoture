# import hashlib
# import hmac
# import json
# import logging
# import operator
# import time
# from collections import defaultdict
# from copy import copy
# from datetime import datetime

# # from threading import Lock
# from typing import Any, Callable, Optional, Unio

# from ..utils import models
# from ..utils.client_rest import Request, Response, RestClient
# from ..utils.client_ws import WebsocketClient
# from ..utils.models import ASSET_TYPE_SPOT, UTC_TZ

# logger = logging.getLogger("bybit")

# BYBIT_API_HOST = "https://api.bybit.com"
# BYBIT_TESTNET_API_HOST = "https://api-testnet.bybit.com"

# # spot v2 can't get data from kline
# BYBIT_WS_SPOT_PUBLIC_HOST = "wss://stream.bybit.com/spot/quote/ws/v1"
# BYBIT_WS_SPOT_PRIVATE_HOST = "wss://stream.bybit.com/spot/ws"
# BYBIT_WS_FUTURE_PUBLIC_HOST = "wss://stream.bybit.com/realtime_public"
# BYBIT_WS_FUTURE_PRIVATE_HOST = "wss://stream.bybit.com/realtime_private"
# BYBIT_WS_INVERSE_HOST = "wss://stream.bybit.com/realtime"
# BYBIT_WS_OPTION_PUBLIC_HOST = "wss://stream.bybit.com/trade/option/usdc/public/v1"
# BYBIT_WS_OPTION_PRIVATE_HOST = "wss://stream.bybit.com/trade/option/usdc/private/v1"

# BYBIT_TESTNET_WS_SPOT_PUBLIC_V2_HOST = "wss://stream-testnet.bybit.com/spot/quote/ws/v2"
# BYBIT_TESTNET_WS_SPOT_PUBLIC_HOST = "wss://stream-testnet.bybit.com/spot/quote/ws/v1"
# BYBIT_TESTNET_WS_SPOT_PRIVATE_HOST = "wss://stream-testnet.bybit.com/spot/ws"
# BYBIT_TESTNET_WS_FUTURE_PUBLIC_HOST = "wss://stream-testnet.bybit.com/realtime_public"
# BYBIT_TESTNET_WS_FUTURE_PRIVATE_HOST = "wss://stream-testnet.bybit.com/realtime_private"
# BYBIT_TESTNET_WS_INVERSE_HOST = "wss://stream-testnet.bybit.com/realtime"
# BYBIT_TESTNET_WS_OPTION_PUBLIC_HOST = (
#     "wss://stream-testnet.bybit.com/trade/option/usdc/public/v1"
# )
# BYBIT_TESTNET_WS_OPTION_PRIVATE_HOST = (
#     "wss://stream-testnet.bybit.com/trade/option/usdc/private/v1"
# )

# EXCHANGE = "BYBIT"

# TIMEDELTA: dict[str, str] = {
#     "1m": "1",
#     "3m": "3",
#     "5m": "5",
#     "15m": "15",
#     "30m": "30",
#     "1h": "60",
#     "2h": "120",
#     "4h": "240",
#     "6h": "360",
#     # "8h":  "1", don't support
#     "12h": "720",
#     "1d": "D",
#     # "3d":  "1", don't support
#     "1w": "W",
#     "1M": "M",
# }

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

# SPOT_KBAR_INTERVAL: dict[str, str] = {
#     "1m": "1m",
#     "3m": "3m",
#     "5m": "5m",
#     "15m": "15m",
#     "30m": "30m",
#     "1h": "1h",
#     "2h": "2h",
#     "4h": "4h",
#     "6h": "6h",
#     "12h": "12h",
#     "1d": "1d",
#     "1w": "1w",
#     "1M": "1M",
# }

# SPOT_MAPPING_CHANNEL: dict[str, str] = {
#     # public
#     "depth": "depth",
#     "kline": "kline",
#     "ticker": "realtimes",
#     # private
#     "trades": "ticketInfo",
#     "orders": "executionReport",
#     "account": "outboundAccountInfo",
# }

# FUTURE_KBAR_INTERVAL: dict[str, str] = {
#     "1m": "1",
#     "3m": "3",
#     "5m": "5",
#     "15m": "15",
#     "30m": "30",
#     "1h": "60",
#     "2h": "120",
#     "4h": "240",
#     "6h": "360",
#     # "8h":  "1", don't support
#     # "12h": "720", don't support
#     "1d": "D",
#     # "3d":  "1", don't support
#     "1w": "W",
#     "1M": "M",
# }
# FUTURE_KBAR_INTERVAL_REV: dict[str, str] = {
#     "1": "1m",
#     "3": "3m",
#     "5": "5m",
#     "15": "15m",
#     "30": "30m",
#     "60": "1h",
#     "120": "2h",
#     "240": "4h",
#     "360": "6h",
#     "D": "1d",
#     "W": "1w",
#     "M": "1M",
# }

# FUTURE_MAPPING_CHANNEL: dict[str, str] = {
#     ### public
#     "depth": "orderBookL2_25",
#     "kline": "candle",
#     "ticker": "instrument_info",
#     ### private
#     "trades": "execution",
#     "orders": "order",
#     "account": "wallet",
#     "position": "position",
# }

# INVERSE_FUTURE_KBAR_INTERVAL: dict[str, str] = {
#     "1m": "1",
#     "3m": "3",
#     "5m": "5",
#     "15m": "15",
#     "30m": "30",
#     "1h": "60",
#     "2h": "120",
#     "4h": "240",
#     "6h": "360",
#     # "8h":  "1", don't support
#     # "12h": "720", don't support
#     "1d": "D",
#     # "3d":  "1", don't support
#     "1w": "W",
#     "1M": "M",
# }
# INVERSE_FUTURE_MAPPING_CHANNEL: dict[str, str] = {
#     ### public
#     "depth": "orderBookL2_25",
#     "kline": "klineV2",
#     "ticker": "instrument_info",
#     ### private
#     "trades": "execution",
#     "orders": "order",
#     "position": "position",
#     "account": "wallet",
# }
# OPTION_MAPPING_CHANNEL: dict[str, str] = {
#     ### public
#     "depth": "delta.orderbook100",
#     # "depth": "orderbook100",
#     "ticker": "instrument_info",
#     ### private
#     "trades": "user.openapi.option.trade",
#     "orders": "user.order",
#     "position": "user.openapi.option.position",
#     "greeks": "user.openapi.greeks",
# }

# RETRY_CODES = {10002, 10006, 30034, 30035, 130035, 130150}


# class BybitSpotClient(RestClient):
#     """
#     api docs: https://bybit-exchange.github.io/docs/spot/#t-introduction
#     """

#     BROKER_ID = ""
#     DATA_TYPE = "spot"

#     def __init__(
#         self,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         is_testnet: bool = False,
#         **kwargs,
#     ) -> None:
#         # url_base: str = None,
#         super().__init__(
#             BYBIT_TESTNET_API_HOST if is_testnet else BYBIT_API_HOST,
#             proxy_host,
#             proxy_port,
#         )

#         self.order_count: int = 0
#         # self.order_count_lock: Lock = Lock()
#         self.connect_time: int = 0
#         self.key: str = ""
#         self.secret: str = ""
#         self.time_offset: int = 0
#         self.recv_window: int = 5000

#     def info(self):
#         return "Bybit Spot REST API start"

#     @staticmethod
#     def generate_datetime(timestamp: float) -> datetime:
#         """generate_datetime"""
#         dt: datetime = datetime.fromtimestamp(timestamp, tz=UTC_TZ)
#         # dt: datetime = UTC_TZ.localize(dt)
#         # dt: datetime = UTC_TZ.localize(dt)
#         dt: datetime = dt.astimezone(UTC_TZ)
#         return dt

#     @staticmethod
#     def isoformat_to_datetime(timestring: str) -> datetime:
#         """generate datetime from time string (string)
#         2019-03-05T09:56:55.728933+00:00'

#         '2022-08-22T16:08:25.020641014Z'
#         """
#         # timestring=timestring.replace('Z', '')
#         # dt: datetime = datetime.fromisoformat(timestring)
#         # dt: datetime = UTC_TZ.localize(dt)
#         # dt_local: datetime = dt.astimezone(UTC_TZ)

#         # # dt: datetime = parse(timestring)
#         # # # dt_local: datetime = UTC_TZ.localize(dt)
#         # # dt: datetime = UTC_TZ.localize(dt)
#         # # dt_local: datetime = dt.astimezone(UTC_TZ)

#         # return dt_local

#         if "Z" in timestring:
#             timestring = timestring.replace("Z", "")

#         # # dt: datetime = datetime.fromisoformat(timestring)
#         # dt: datetime = parse(timestring)
#         # # dt_local: datetime = UTC_TZ.localize(dt)
#         # dt: datetime = UTC_TZ.localize(dt)
#         # dt_local: datetime = dt.astimezone(UTC_TZ)

#         dt: datetime = parse(timestring)
#         # dt: datetime = datetime.fromisoformat(timestring)
#         dt: datetime = UTC_TZ.localize(dt)
#         dt_local: datetime = dt.astimezone(UTC_TZ)

#         return dt_local

#     def query_time(self):
#         """query_time
#         {
#             "ret_code": 0,
#             "ret_msg": "",
#             "ext_code": null,
#             "ext_info": null,
#             "result": {
#                 "serverTime": 1625799317787
#             }
#         }
#         {'ret_code': 10002, 'ret_msg': 'invalid request, please check your timestamp and recv_window param. req_timestamp: 1651031897761 server_timestamp: 1651031901356 recv_window: 1000', 'ext_code': '', 'ext_info': '', 'result': None, 'time_now': '1651031901.356939'}}

#         """
#         # resp = self.request("GET", "/spot/v1/time")
#         resp = self.query("GET", "/spot/v1/time")
#         local_time = int(time.time() * 1000)
#         server_time = int(float(resp.data["result"]["serverTime"]))
#         self.time_offset = local_time - server_time
#         # print(f"local_time {local_time}")
#         # print(f"server_time {server_time}")
#         # print(f"timeoffset {self.time_offset}")
#         return server_time

#     def new_order_id(self) -> str:
#         """new_order_id"""
#         # prefix: str = datetime.now().strftime("%Y%m%d-%H%M%S-")
#         prefix: str = (
#             models.generate_datetime().strftime("%y%m%d-%H%M%S%f")[:-3] + "-"
#         )  #  # datetime.now()
#         # with self.order_count_lock:
#         self.order_count += 1
#         suffix: str = str(self.order_count).rjust(5, "0")

#         order_id: str = f"x-{self.BROKER_ID}" + prefix + suffix
#         return order_id

#     def _remove_none(self, payload):
#         return {k: v for k, v in payload.items() if v is not None}

#     def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
#         # response = self.request(method="GET", path=path,  params=params)
#         response = self.query(method="GET", path=path, params=params)
#         # TODO: based on statu code and force retry
#         return self._process_response(response)

#     def _post(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
#         # response = self.request(method="POST", path=path, data=params, headers={'Referer': 'PYALGOTURE'})
#         response = self.query(
#             method="POST", path=path, data=params, headers={"Referer": "PYALGOTURE"}
#         )
#         return self._process_response(response)

#     def _delete(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
#         # response = self.request(method="DELETE", path=path, data=params)
#         response = self.query(method="DELETE", path=path, data=params)
#         return self._process_response(response)

#     def retry(self, response: Response, request: Request):
#         path = request.path
#         data = response.data
#         if "usdc" in path:
#             ret_code = "retCode"
#             ret_msg = "retMsg"
#         else:
#             ret_code = "ret_code"
#             ret_msg = "ret_msg"

#         error_msg = f"{data.get(ret_msg)} (ErrCode: {data.get(ret_code)})"
#         err_delay = self._retry_delay
#         if data.get(ret_code) in RETRY_CODES:
#             # 10002, recv_window error; add 2.5 seconds and retry.
#             if data[ret_code] == 10002:
#                 error_msg += ". Added 2.5 seconds to recv_window"
#                 self.recv_window += 2500

#             # 10006, ratelimit error; wait until rate_limit_reset_ms and retry.
#             elif data[ret_code] == 10006:
#                 print(
#                     f"{error_msg}. Ratelimited on current request. Sleeping, then trying again. Request: {path}"
#                 )

#                 # Calculate how long we need to wait.
#                 limit_reset = data["rate_limit_reset_ms"] / 1000
#                 reset_str = time.strftime("%X", time.localtime(limit_reset))
#                 err_delay = int(limit_reset) - int(time.time())
#                 error_msg = f"Ratelimit will reset at {reset_str}. Sleeping for {err_delay} seconds"
#             print(error_msg)
#             time.sleep(err_delay)
#             return True
#         return False

#     def connect(self, key: str, secret: str) -> None:
#         """connect exchange server"""
#         self.key = key
#         self.secret = secret
#         self.connect_time = int(datetime.now(UTC_TZ).strftime("%y%m%d%H%M%S"))
#         self.query_time()
#         self.start()
#         logger.debug(self.info())
#         return True

#     def _auth(self, method, params, recv_window=5000):  # 1000
#         """
#         Generates authentication signature per Bybit API specifications.

#         Notes
#         -------------------
#         Since the POST method requires a JSONified dict, we need to ensure
#         the signature uses lowercase booleans instead of Python's
#         capitalized booleans. This is done in the bug fix below.

#         """

#         api_key = self.key
#         api_secret = self.secret

#         if api_key is None or api_secret is None:
#             raise PermissionError("Authenticated endpoints require keys.")

#         params["api_key"] = api_key
#         params["recv_window"] = recv_window
#         # bybit server have timeoffset issue - need to correct by querying time api
#         # params['timestamp'] = int((time.time() - self.time_offset) * 10 ** 3)
#         timestamp: int = int(time.time() * 1000)

#         if self.time_offset > 0:
#             timestamp -= abs(self.time_offset)
#         elif self.time_offset < 0:
#             timestamp += abs(self.time_offset)

#         params["timestamp"] = timestamp

#         # Sort dictionary alphabetically to create querystring.
#         _val = "&".join(
#             [
#                 str(k) + "=" + str(v)
#                 for k, v in sorted(params.items())
#                 if (k != "sign") and (v is not None)
#             ]
#         )

#         # Bug fix. Replaces all capitalized booleans with lowercase.
#         if method == "POST":
#             _val = _val.replace("True", "true").replace("False", "false")

#         # Return signature.
#         return str(
#             hmac.new(
#                 bytes(api_secret, "utf-8"), bytes(_val, "utf-8"), digestmod="sha256"
#             ).hexdigest()
#         )

#     def _usdc_auth(self, params, recv_window=5000):
#         """
#         Generates authentication signature per Bybit API specifications.
#         """

#         api_key = self.key
#         api_secret = self.secret

#         if api_key is None or api_secret is None:
#             raise PermissionError("Authenticated endpoints require keys.")
#         payload = json.dumps(params)
#         timestamp = int(time.time() * 10**3)
#         param_str = str(timestamp) + api_key + str(recv_window) + payload
#         hash = hmac.new(
#             bytes(api_secret, "utf-8"), param_str.encode("utf-8"), hashlib.sha256
#         )
#         return hash.hexdigest()

#     def sign(self, request):
#         """generate bybit signature"""
#         query = request.params if request.params else {}
#         query_data = request.data if request.data else {}
#         query.update(query_data)
#         method = request.method
#         path = request.path

#         # Bug fix: change floating whole numbers to integers to prevent
#         # auth signature errors.
#         if query is None:
#             query = {}
#         else:
#             for i in query.keys():
#                 if isinstance(query[i], float) and query[i] == int(query[i]):
#                     query[i] = int(query[i])

#         if "usdc" in path:
#             # Prepare signature.
#             signature = self._usdc_auth(params=query, recv_window=self.recv_window)
#         else:
#             signature = self._auth(
#                 method=method, params=query, recv_window=self.recv_window
#             )
#         # Sort the dictionary alphabetically.
#         query = dict(sorted(query.items(), key=lambda x: x))
#         query["sign"] = signature

#         if query is not None:
#             req_params = {k: v for k, v in query.items() if v is not None}
#         else:
#             req_params = {}

#         if method == "GET":
#             request.params = req_params
#             request.headers = {"Content-Type": "application/x-www-form-urlencoded"}
#         else:
#             if "spot" in path:
#                 full_param_str = "&".join(
#                     [
#                         str(k) + "=" + str(v)
#                         for k, v in sorted(query.items())
#                         if v is not None
#                     ]
#                 )
#                 request.params = {}
#                 request.data = {}
#                 request.headers = {"Content-Type": "application/x-www-form-urlencoded"}
#                 request.path = path + f"?{full_param_str}"
#             elif "usdc" in path:
#                 request.headers = {
#                     "Content-Type": "application/json",
#                     "X-BAPI-API-KEY": self.key,
#                     "X-BAPI-SIGN": signature,
#                     "X-BAPI-SIGN-TYPE": "2",
#                     "X-BAPI-TIMESTAMP": str(int(time.time() * 10**3)),
#                     "X-BAPI-RECV-WINDOW": "5000",
#                 }
#                 request.data = json.dumps(req_params)

#             else:
#                 # request.params = req_params # FIXME?
#                 request.data = req_params  # FIXME?

#         # logger.debug(request)
#         return request

#     def _process_response(self, response: Response) -> dict:
#         try:
#             data = response.data
#             # print(data,'!!!!', response.ok)
#         except ValueError:
#             print("!!bybit - _process_response !!", response.text)
#             logger.debug(response.data())
#             raise
#         else:
#             # logger.debug(data)
#             # print(f'DEBUG:{data} | response.ok:{response.ok} | data["result"]:{data["result"]}')

#             # if not response.ok or (response.ok and data["result"] is None or data["ret_code"] != 0):
#             if not response.ok or (
#                 response.ok and not data["result"] and data["ret_code"] != 0
#             ):
#                 payload_data = models.CommonDataSchema(
#                     status_code=response.status_code, msg=dict(data)
#                 )
#                 if not data.get("ret_msg"):
#                     print("!!bybit - return raw data !!", response.text)
#                 # print('??????', payload_data.dict(), data,'????')
#                 return models.CommonResponseSchema(
#                     success=False,
#                     error=True,
#                     data=payload_data.dict(),
#                     msg=data.get("ret_msg", "No return msg from bybit exchange."),
#                 )
#             elif response.ok and data["result"] is None and data["ret_code"] == 0:
#                 """send cancel order status"""
#                 # print('??????', payload_data.dict(), data,'????')
#                 payload_data = models.CommonDataSchema(
#                     status_code=response.status_code, msg=dict(data)
#                 )
#                 return models.CommonResponseSchema(
#                     success=False,
#                     error=True,
#                     data=payload_data.dict(),
#                     msg="query ok, but result is None",
#                 )
#             else:
#                 return models.CommonResponseSchema(
#                     success=True, error=False, data=data["result"], msg="query ok"
#                 )

#     def _regular_order_payload(
#         self, common_response: models.CommonResponseSchema
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 # msg['price'] = float(msg['price'])
#                 # print(f"debug: {msg}")
#                 data = models.OrderSchema()
#                 result = data.from_bybit_to_form(msg, datatype=self.DATA_TYPE)
#                 payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.OrderSchema()
#             payload = payload.from_bybit_to_form(
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
#                 result = data.from_bybit_to_form(msg, datatype=self.DATA_TYPE)
#                 if result:
#                     payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.TradeSchema()
#             payload = payload.from_bybit_to_form(
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
#                 result = data.from_bybit_to_form(msg, datatype=self.DATA_TYPE)
#                 if result:
#                     # payload.append(result)
#                     payload[result["symbol"]] = result
#         elif isinstance(common_response.data, dict):
#             payload = models.SymbolSchema()
#             payload = payload.from_bybit_to_form(
#                 common_response.data, datatype=self.DATA_TYPE
#             )
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def _regular_historical_prices_payload(
#         self, common_response: models.CommonResponseSchema, extra: dict = None
#     ) -> dict:
#         def key_data(data):
#             key = [
#                 "startTime",
#                 "open",
#                 "high",
#                 "low",
#                 "close",
#                 "volume",
#                 "endTime",
#                 "quoteAssetVolume",
#                 "trades",
#                 "takerBaseVolume",
#                 "takerQuoteVolume",
#             ]
#             return dict(zip(key, data))

#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 msg = key_data(msg)
#                 data = models.HistoryOHLCSchema()
#                 result = data.from_bybit_to_form(msg, extra, datatype=self.DATA_TYPE)
#                 payload.append(result)
#         elif isinstance(common_response.data, dict):
#             common_response.data = key_data(common_response.data)
#             payload = (
#                 models.HistoryOHLCSchema()
#             )  # .model_construct(common_response.data, extra, datatype=self.DATA_TYPE)
#             payload = payload.from_bybit_to_form(common_response.data, extra)
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def _regular_symbol_payload(
#         self, common_response: models.CommonResponseSchema
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             # payload = list()
#             for msg in common_response.data:
#                 data = models.TickerSchema()
#                 payload = data.from_bybit_to_form(msg, datatype=self.DATA_TYPE)
#                 # payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.TickerSchema()
#             payload = payload.from_bybit_to_form(
#                 common_response.data, datatype=self.DATA_TYPE
#             )
#         else:
#             common_response.msg = "payload error in fetch symbol data"
#             Exception("payload error in fetch symbol data")

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
#                 result = data.from_bybit_to_form(
#                     msg, transfer_type=transfer_type, inout=inout
#                 )
#                 payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.TransferSchema()
#             payload = payload.from_bybit_to_form(
#                 common_response.data, transfer_type=transfer_type, inout=inout
#             )
#         else:
#             common_response.msg = "payload error in fetch transfer data"
#             Exception("payload error in fetch transfer data")

#         common_response.data = payload
#         return common_response

#     def _aggregate_trades(self, trades):
#         ret = defaultdict(dict)
#         for i in trades:
#             ori_order_id = i["ori_order_id"]
#             if ori_order_id in ret:
#                 ret[ori_order_id]["quantity"] += i["quantity"]
#                 ret[ori_order_id]["commission"] += i["commission"]
#             else:
#                 ret[ori_order_id] = i
#         return list(ret.values())

#     def query_permission(self) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.api_key_info())
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "ok",
#             "ext_code": "",
#             "result": [
#                 {
#                     "api_key": "7GkMBBLTbGRfa0Nuh1",
#                     "type": "personal",
#                     "user_id": 1,
#                     "inviter_id": 3,
#                     "ips": [
#                         "*"
#                     ],
#                     "note": "scalping_bot",
#                     "permissions": [
#                         "Order",
#                         "Position"
#                     ],
#                     "created_at": "2019-10-28T13:22:39.000Z",
#                     "expired_at": "2020-01-28T13:22:39.000Z",
#                     "read_only": false,
#                     "vip_level":"",
#                     "mkt_maker_level":""
#                 }
#             ],
#             "ext_info": null,
#             "time_now": "1577445138.790150",
#             "rate_limit_status": 99,
#             "rate_limit_reset_ms": 1577445138812,
#             "rate_limit": 100
#         }
#         {'ret_code': 0, 'ret_msg': 'OK', 'ext_code': '', 'ext_info': '', 'result': [{'api_key': 'f98ehmeoqqKW5qctkP', 'type': 'personal', 'user_id': 6949322, 'inviter_id': 6721911, 'ips': ['*'], 'note': 'OLA_01', 'permissions': ['Order', 'Position', 'SpotTrade', 'OptionsTrade'], 'created_at': '2022-04-27T03:45:27Z', 'expired_at': '2022-07-27T03:45:27Z', 'read_only': False, 'vip_level': 'No VIP', 'mkt_maker_level': '0'}], 'time_now': '1657525794.219057', 'rate_limit_status': 599, 'rate_limit_reset_ms': 1657525794201, 'rate_limit': 600}
#         {'ret_code': 0, 'ret_msg': 'OK', 'ext_code': '', 'ext_info': '', 'result': [{'api_key': 'KKqaq9on9fBtZ1yuUV', 'type': 'PYALGOTURE', 'user_id': 6949322, 'inviter_id': 6721911, 'ips': ['18.167.29.57', '18.163.87.75', '16.163.152.219', '16.162.73.178'], 'note': 'PYALGOTURE', 'permissions': ['SpotTrade', 'OptionsTrade', 'DerivativesTrade', 'Position', 'Order'], 'created_at': '2022-07-10T09:04:13Z', 'expired_at': '', 'read_only': False, 'vip_level': 'No VIP', 'mkt_maker_level': '0'}], 'time_now': '1657525874.431047', 'rate_limit_status': 599, 'rate_limit_reset_ms': 1657525874414, 'rate_limit': 600}

#         """
#         api_info = self._get("/v2/private/account/api-key")
#         # print(f">>>> api_info:{api_info}")
#         if api_info.success:
#             payload = dict()
#             # print('debug!!!!',api_info.data,'!!!!debug')
#             for msg in api_info.data:
#                 # print('debug!!!!',msg,'!!!!debug')
#                 data = models.PermissionSchema()
#                 data.from_bybit_to_form(msg)
#                 payload = data.dict()

#             api_info.data = payload

#         return api_info

#     def query_account(self) -> dict:
#         """
#         {
#             "ret_code": 0,
#             "ret_msg": "",
#             "ext_code": null,
#             "ext_info": null,
#             "result": {
#                 "balances": [
#                     {
#                         "coin": "USDT",
#                         "coinId": "USDT",
#                         "coinName": "USDT",
#                         "total": "10",
#                         "free": "10",
#                         "locked": "0"
#                     }
#                 ]
#             }
#         }
#         """
#         balance = self._get("/spot/v1/account")
#         account = defaultdict(dict)
#         # logger.debug(balance)
#         if balance.success and balance.data:
#             if not balance.data["balances"]:
#                 balance.msg = "account wallet is empty"
#                 balance.data = {}
#                 return balance
#             datatype = self.DATA_TYPE.upper()
#             for account_data in balance.data["balances"]:
#                 free = float(account_data["free"])
#                 locked = float(account_data["locked"])
#                 if free != 0 or locked != 0:
#                     key = account_data["coin"]
#                     account[key]["symbol"] = key
#                     account[key]["available"] = free
#                     account[key]["frozen"] = locked
#                     account[key]["balance"] = free + locked
#                     account[key]["exchange"] = EXCHANGE
#                     account[key]["asset_type"] = datatype

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
#         **kwargs,
#     ) -> dict:
#         """
#         NOTE: bybit spot cannot place stop order
#             Request Example

#             from pybit import HTTP
#             session = HTTP("https://api-testnet.bybit.com",
#                         api_key="", api_secret="",
#                         spot=True)
#             logger.debug(session.place_active_order(
#                 symbol="ETHUSDT",
#                 side="Buy",
#                 type="MARKET",
#                 qty=10,
#                 timeInForce="GTC"
#             ))
#             Response Example

#             {
#                 "ret_code": 0,
#                 "ret_msg": "",
#                 "ext_code": null,
#                 "ext_info": null,
#                 "result": {
#                     "accountId": "1",
#                     "symbol": "ETHUSDT",
#                     "symbolName": "ETHUSDT",
#                     "orderLinkId": "162073788655749",
#                     "orderId": "889208273689997824",
#                     "transactTime": "1620737886573",
#                     "price": "20000",
#                     "origQty": "10",
#                     "executedQty": "0",
#                     "status": "NEW",
#                     "timeInForce": "GTC",
#                     "type": "LIMIT",
#                     "side": "BUY"
#                 }
#             }
#         """
#         payload = {
#             "symbol": symbol.upper().replace("/", ""),
#             "side": str(side).lower().capitalize(),
#             # 'price': price,
#             "qty": quantity,
#             # 'type': order_type,   # LIMIT_MAKER, LIMIT, MARKET
#             "orderLinkId": self.new_order_id(),
#             "timeInForce": "GTC",
#             "agentSource": "PYALGOTURE",  # 0.35%
#         }
#         if not order_type:
#             if not price:
#                 order_type = "MARKET"
#             else:
#                 order_type = "LIMIT"

#             payload["type"] = order_type
#         else:
#             order_type = order_type.upper()

#         if order_type == "LIMIT":
#             payload["price"] = price
#         elif order_type == "MARKET":
#             if payload["side"] == "Buy":
#                 last_price = self.query_prices(symbol)["data"]["price"]
#                 payload["qty"] = round(last_price * quantity, 4)
#                 print(
#                     f"payload: {payload} ????in bybit spot send_order???? last_price: {last_price}"
#                 )

#         orders_response = self._post("/spot/v1/order", payload)

#         if orders_response.success:
#             if orders_response.data:
#                 result = (
#                     models.SendOrderResultSchema()
#                 )  # .model_construct(orders_response.data)
#                 orders_response.data = result.from_bybit_to_form(
#                     orders_response.data, datatype=self.DATA_TYPE
#                 )
#             else:
#                 orders_response.msg = "query sender order is empty"

#         return orders_response

#     def cancel_order(self, order_id: str) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="",
#                     spot=True)
#         logger.debug(session.cancel_active_order(
#             orderId="889826641228952064"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "",
#             "ext_code": null,
#             "ext_info": null,
#             "result": {
#                 "accountId": "10054",
#                 "symbol": "ETHUSDT",
#                 "orderLinkId": "162081160171552",
#                 "orderId": "889826641228952064",
#                 "transactTime": "1620811601728",
#                 "price": "20000",
#                 "origQty": "10",
#                 "executedQty": "0",
#                 "status": "CANCELED",
#                 "timeInForce": "GTC",
#                 "type": "LIMIT",
#                 "side": "BUY"
#             }
#         }
#         """
#         payload = {
#             # "orderLinkId": order_id
#             "orderId": order_id
#         }
#         return self._delete("/spot/v1/order", payload)

#     def cancel_orders(self, symbol: str) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="",
#                     spot=True)
#         logger.debug(session.batch_cancel_active_order(
#             symbol="ETHUSDT",
#             orderTypes="LIMIT,LIMIT_MAKER"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "",
#             "ext_code": null,
#             "ext_info": null,
#             "result": {
#                 "success": true
#             }
#         }
#         """
#         payload = {"symbol": symbol}
#         return self._delete("/spot/order/batch-cancel", payload)

#     def query_order_status(self, order_id: str = None) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="",
#                     spot=True)
#         logger.debug(session.get_active_order(
#             orderId="889826641228952064"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "",
#             "ext_code": null,
#             "ext_info": null,
#             "result": {
#                 "accountId": "1054",
#                 "exchangeId": "301",
#                 "symbol": "ETHUSDT",
#                 "symbolName": "ETHUSDT",
#                 "orderLinkId": "162081160171552",
#                 "orderId": "889826641228952064",
#                 "price": "20000",
#                 "origQty": "10",
#                 "executedQty": "0",
#                 "cummulativeQuoteQty": "0",
#                 "avgPrice": "0",
#                 "status": "NEW",
#                 "timeInForce": "GTC",
#                 "type": "LIMIT",
#                 "side": "BUY",
#                 "stopPrice": "0.0",
#                 "icebergQty": "0.0",
#                 "time": "1620811601728",
#                 "updateTime": "1620811601743",
#                 "isWorking": true
#             }
#         }
#         """
#         if order_id:
#             payload = {"orderLinkId": order_id}
#             order_status = self._get("/spot/v1/order", self._remove_none(payload))
#         else:
#             order_status = self._get("/spot/v1/order")

#         if order_status.success:
#             if order_status.data:
#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"

#         return order_status

#     def query_open_orders(self, symbol: str = None) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="",
#                     spot=True)
#         logger.debug(session.query_active_order(
#             symbol="ETHUSDT"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "",
#             "ext_code": null,
#             "ext_info": null,
#             "result": [
#                 {
#                     "accountId": "10054",
#                     "exchangeId": "301",
#                     "symbol": "ETHUSDT",
#                     "symbolName": "ETHUSDT",
#                     "orderLinkId": "162080709527252",
#                     "orderId": "889788838461927936",
#                     "price": "20000",
#                     "origQty": "10",
#                     "executedQty": "0",
#                     "cummulativeQuoteQty": "0",
#                     "avgPrice": "0",
#                     "status": "NEW",
#                     "timeInForce": "GTC",
#                     "type": "LIMIT",
#                     "side": "BUY",
#                     "stopPrice": "0.0",
#                     "icebergQty": "0.0",
#                     "time": "1620807095287",
#                     "updateTime": "1620807095307",
#                     "isWorking": true
#                 },
#                 {
#                     "accountId": "10054",
#                     "exchangeId": "301",
#                     "symbol": "ETHUSDT",
#                     "symbolName": "ETHUSDT",
#                     "orderLinkId": "162063873503148",
#                     "orderId": "888376530389004800",
#                     "price": "20000",
#                     "origQty": "10",
#                     "executedQty": "0",
#                     "cummulativeQuoteQty": "0",
#                     "avgPrice": "0",
#                     "status": "NEW",
#                     "timeInForce": "GTC",
#                     "type": "LIMIT",
#                     "side": "BUY",
#                     "stopPrice": "0.0",
#                     "icebergQty": "0.0",
#                     "time": "1620638735044",
#                     "updateTime": "1620638735062",
#                     "isWorking": true
#                 }
#             ]
#         }
#         """
#         if symbol:
#             payload = {"symbol": symbol}
#         else:
#             payload = None
#         order_status = self._get("/spot/v1/open-orders", payload)
#         if order_status.success:
#             if order_status.data:
#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"
#                 order_status.data = []

#         return order_status

#     def query_all_orders(self, symbol: str = None, limit=500) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="",
#                     spot=True)
#         logger.debug(session.query_active_order(
#             symbol="ETHUSDT"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "",
#             "ext_code": null,
#             "ext_info": null,
#             "result": [
#                 {
#                     "accountId": "10054",
#                     "exchangeId": "301",
#                     "symbol": "ETHUSDT",
#                     "symbolName": "ETHUSDT",
#                     "orderLinkId": "1620615771764",
#                     "orderId": "888183901021893120",
#                     "price": "5000",
#                     "origQty": "1",
#                     "executedQty": "0",
#                     "cummulativeQuoteQty": "0",
#                     "avgPrice": "0",
#                     "status": "CANCELED",
#                     "timeInForce": "GTC",
#                     "type": "LIMIT",
#                     "side": "BUY",
#                     "stopPrice": "0.0",
#                     "icebergQty": "0.0",
#                     "time": "1620615771836",
#                     "updateTime": "1620617056334",
#                     "isWorking": true
#                 }
#             ]
#         }
#         """
#         payload = {"limit": limit}
#         if symbol:
#             payload = {"symbol": symbol}
#             all_order = self._get("/spot/v1/history-orders", self._remove_none(payload))
#         else:
#             all_order = self._get("/spot/v1/history-orders")

#         if all_order.success:
#             if all_order.data:
#                 return self._regular_order_payload(all_order)
#             else:
#                 all_order.msg = "query order is empty"

#         return all_order

#     def query_symbols(self, symbol: str = None) -> dict:
#         """
#         {
#             "ret_code": 0,
#             "ret_msg": "",
#             "ext_code": null,
#             "ext_info": null,
#             "result": [
#                 {
#                     "name": "BTCUSDT",
#                     "alias": "BTCUSDT",
#                     "baseCurrency": "BTC",
#                     "quoteCurrency": "USDT",
#                     "basePrecision": "0.000001",
#                     "quotePrecision": "0.01",
#                     "minTradeQuantity": "0.0001",
#                     "minTradeAmount": "10",
#                     "minPricePrecision": "0.01",
#                     "maxTradeQuantity": "2",
#                     "maxTradeAmount": "200",
#                     "category": 1
#                 },
#                 {
#                     "name": "ETHUSDT",
#                     "alias": "ETHUSDT",
#                     "baseCurrency": "ETH",
#                     "quoteCurrency": "USDT",
#                     "basePrecision": "0.0001",
#                     "quotePrecision": "0.01",
#                     "minTradeQuantity": "0.0001",
#                     "minTradeAmount": "10",
#                     "minPricePrecision": "0.01",
#                     "maxTradeQuantity": "2",
#                     "maxTradeAmount": "200",
#                     "category": 1
#                 }
#             ]
#         }

#         {
#             "ret_code":0,
#             "ret_msg":"OK",
#             "ext_code":"",
#             "ext_info":"",
#             "result":[
#                 {
#                     "name":"BTCUSD",
#                     "alias":"BTCUSD",
#                     "status":"Trading",
#                     "base_currency":"BTC",
#                     "quote_currency":"USD",
#                     "price_scale":2,
#                     "taker_fee":"0.00075",
#                     "maker_fee":"-0.00025",
#                     "leverage_filter":{
#                         "min_leverage":1,
#                         "max_leverage":100,
#                         "leverage_step":"0.01"
#                     },
#                     "price_filter":{
#                         "min_price":"0.5",
#                         "max_price":"999999.5",
#                         "tick_size":"0.5"
#                     },
#                     "lot_size_filter":{
#                         "max_trading_qty":1000000,
#                         "min_trading_qty":1,
#                         "qty_step":1
#                     }
#                 },
#                 {
#                     "name":"EOSUSD",
#                     "alias":"EOSUSD",
#                     "status":"Trading",
#                     "base_currency":"EOS",
#                     "quote_currency":"USD",
#                     "price_scale":3,
#                     "taker_fee":"0.00075",
#                     "maker_fee":"-0.00025",
#                     "leverage_filter":{
#                         "min_leverage":1,
#                         "max_leverage":50,
#                         "leverage_step":"0.01"
#                     },
#                     "price_filter":{
#                         "min_price":"0.001",
#                         "max_price":"1999.999",
#                         "tick_size":"0.001"
#                     },
#                     "lot_size_filter":{
#                         "max_trading_qty":1000000,
#                         "min_trading_qty":1,
#                         "qty_step":1
#                     }
#                 },
#                 {
#                     "name":"BTCUSDT",
#                     "alias":"BTCUSDT",
#                     "status":"Trading",
#                     "base_currency":"BTC",
#                     "quote_currency":"USDT",
#                     "price_scale":2,
#                     "taker_fee":"0.00075",
#                     "maker_fee":"-0.00025",
#                     "leverage_filter":{
#                         "min_leverage":1,
#                         "max_leverage":100,
#                         "leverage_step":"0.01"
#                     },
#                     "price_filter":{
#                         "min_price":"0.5",
#                         "max_price":"999999.5",
#                         "tick_size":"0.5"
#                     },
#                     "lot_size_filter":{
#                         "max_trading_qty":100,
#                         "min_trading_qty":0.001,
#                         "qty_step":0.001
#                     }
#                 },
#                 {
#                     "name":"BTCUSDM21",
#                     "alias":"BTCUSD0625",
#                     "status":"Trading",
#                     "base_currency":"BTC",
#                     "quote_currency":"USD",
#                     "price_scale":2,
#                     "taker_fee":"0.00075",
#                     "maker_fee":"-0.00025",
#                     "leverage_filter":{
#                         "min_leverage":1,
#                         "max_leverage":100,
#                         "leverage_step":"0.01"
#                     },
#                     "price_filter":{
#                         "min_price":"0.5",
#                         "max_price":"999999.5",
#                         "tick_size":"0.5"
#                     },
#                     "lot_size_filter":{
#                         "max_trading_qty":1000000,
#                         "min_trading_qty":1,
#                         "qty_step":1
#                     }
#                 }
#             ],
#             "time_now":"1615801223.589808"
#         }
#         """

#         symbols = self._get("/spot/v1/symbols")
#         if symbols.success:
#             if symbols.data:
#                 if symbol:
#                     symbol = symbol.upper().replace("/", "")
#                     data = self._regular_symbols_payload(symbols)
#                     symbol_data = [
#                         x for x in data.data.keys() if x.lower() == symbol.lower()
#                     ]
#                     # logger.debug(symbol_data)
#                     if symbol_data:
#                         symbols.data = data.data[symbol_data[0]]
#                     else:
#                         symbols.msg = "query symbol is empty"
#                         symbols.data = {}
#                 else:
#                     return self._regular_symbols_payload(symbols)
#             else:
#                 symbols.msg = "query symbol is empty"

#         return symbols

#     def query_trades(
#         self,
#         symbol: str = None,
#         start: datetime = None,
#         end: datetime = None,
#         limit=1000,
#     ) -> dict:
#         """
#         start expect datetime in UTC
#         Request Example
#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="",
#                     spot=True)
#         logger.debug(session.user_trade_records(
#             symbol="BTCUSDT"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "",
#             "ext_code": null,
#             "ext_info": null,
#             "result": [
#             {
#                     "id": "931975237315196160",
#                     "symbol": "BTCUSDT",
#                     "symbolName": "BTCUSDT",
#                     "orderId": "931975236946097408",
#                     "ticketId": "1057753175328833537",
#                     "matchOrderId": "931975113180558592",
#                     "price": "20000.00001",
#                     "qty": "0.01",
#                     "commission": "0.02000000001",
#                     "commissionAsset": "USDT",
#                     "time": "1625836105890",
#                     "isBuyer": false,
#                     "isMaker": false,
#                     "fee": {
#                         "feeTokenId": "USDT",
#                         "feeTokenName": "USDT",
#                         "fee": "0.02000000001"
#                     },
#                     "feeTokenId": "USDT",
#                     "feeAmount": "0.02000000001",
#                     "makerRebate": "0"
#             }
#             ]
#         }
#         """
#         payload = {"limit": limit}
#         if start:
#             payload["startTime"] = int(datetime.timestamp(start) * 1000)
#         if end:
#             payload["endTime"] = int(datetime.timestamp(end) * 1000)
#         if symbol:
#             payload["symbol"] = symbol

#         trades = self._get("/spot/v1/myTrades", self._remove_none(payload))
#         if trades.success:
#             if trades.data:
#                 ret_trades = self._regular_trades_payload(trades)
#                 # aggregate by ori_order_id
#                 ret_trades.data = self._aggregate_trades(ret_trades["data"])
#                 return ret_trades
#             else:
#                 trades.msg = "query trades is empty"
#                 trades.data = []

#         return trades

#     def query_history(
#         self,
#         symbol: str,
#         interval: str = "1h",
#         start: datetime = None,
#         end: datetime = None,
#         limit: int = 1000,
#     ) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     spot=True)
#         print(session.query_kline(
#             symbol="BTCUSDT",
#             interval="1m"
#         ))

#         Kline interval (interval)
#             1m - 1 minute
#             3m - 3 minutes
#             5m - 5 minutes
#             15m - 15 minutes
#             30m - 30 minutes
#             1h - 1 hour
#             2h - 2 hours
#             4h - 4 hours
#             6h - 6 hours
#             12h - 12 hours
#             1d - 1 day
#             1w - 1 week
#             1M - 1 month

#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": null,
#             "result": [
#                 [
#                     1620917160000,
#                     "50008",
#                     "50008",
#                     "50008",
#                     "50008",
#                     "0",
#                     0,
#                     "0",
#                     0,
#                     "0",
#                     "0"
#                 ]
#             ],
#             "ext_code": null,
#             "ext_info": null
#         }

#         Parameter	        Type	Comment
#         startTime	        long	Start time, unit in millisecond
#         open	            float	Open price
#         high	            float	High price
#         low	                float   Low price
#         close	            float	Close price
#         volume	            float	Trading volume
#         endTime	            long	End time, unit in millisecond
#         quoteAssetVolume	float	Quote asset volume
#         trades	            integer	Number of trades
#         takerBaseVolume	    float	Taker buy volume in base asset
#         takerQuoteVolume	float	Taker buy volume in quote asset
#         """

#         def iteration_history(symbol, interval, data, payload, retry=3):
#             retry_counter = 0
#             historical_prices_datalist = data.data
#             # logger.debug(interval)

#             first_timestamp = int(historical_prices_datalist[0][0])
#             start_timestamp = payload["startTime"]
#             interval_timestamp = first_timestamp - start_timestamp
#             # logger.debug(payload)

#             while interval_timestamp > interval:
#                 first_timestamp = int(historical_prices_datalist[0][0])
#                 interval_timestamp = first_timestamp - start_timestamp

#                 logger.debug(f"first_timestamp {first_timestamp}")
#                 logger.debug(f"interval_timestamp {interval_timestamp}")
#                 # logger.debug(historical_prices_datalist[0:5])

#                 payload["endTime"] = first_timestamp

#                 prices = self._get("/spot/quote/v1/kline", self._remove_none(payload))
#                 if prices.error:
#                     break
#                 logger.debug(prices.data[0:2])

#                 historical_prices_datalist.extend(prices.data)
#                 historical_prices_datalist = [
#                     list(item)
#                     for item in set(tuple(x) for x in historical_prices_datalist)
#                 ]
#                 historical_prices_datalist.sort(key=lambda k: k[0])
#                 time.sleep(0.1)

#                 logger.debug(f"payload: {payload}")
#                 logger.debug(f"retry_counter: {retry_counter}")

#                 if len(prices.data) != 1000:
#                     retry_counter += 1

#                 if retry_counter >= 3:
#                     break

#             data.data = historical_prices_datalist
#             logger.debug(f"data length: {len(historical_prices_datalist)}")
#             return data

#         # main
#         payload = {
#             "symbol": symbol.upper().replace("/", ""),
#             "interval": SPOT_KBAR_INTERVAL[interval],
#             "limit": limit,
#         }

#         # ms - epoch time
#         if start:
#             start = UTC_TZ.localize(
#                 start
#             )  # NOTE: bybit start and end timestamp default regard it as utc timestamp
#             payload["startTime"] = int(datetime.timestamp(start) * 1000)
#         if end:
#             end = UTC_TZ.localize(end)
#             payload["endTime"] = int(datetime.timestamp(end) * 1000)
#         historical_prices = self._get(
#             "/spot/quote/v1/kline", self._remove_none(payload)
#         )
#         extra = {"interval": interval, "symbol": symbol}

#         if historical_prices.success:
#             if historical_prices.data:
#                 # handle query time
#                 if "startTime" in payload:
#                     # ms
#                     interval = TIMEDELTA_MAPPING_SEC[interval] * 1000
#                     historical_prices = iteration_history(
#                         symbol=symbol,
#                         interval=interval,
#                         data=historical_prices,
#                         payload=payload,
#                     )
#                 return self._regular_historical_prices_payload(historical_prices, extra)
#             else:
#                 historical_prices.msg = "query historical prices is empty"

#         return historical_prices

#     def query_last_price(
#         self, symbol: str, interval: str = "1m", limit: int = 1
#     ) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     spot=True)
#         print(session.query_kline(
#             symbol="BTCUSDT",
#             interval="1m"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": null,
#             "result": [
#                 [
#                     1620917160000,
#                     "50008",
#                     "50008",
#                     "50008",
#                     "50008",
#                     "0",
#                     0,
#                     "0",
#                     0,
#                     "0",
#                     "0"
#                 ]
#             ],
#             "ext_code": null,
#             "ext_info": null
#         }
#         """
#         payload = {"symbol": symbol, "interval": interval, "limit": limit}
#         last_historical_prices = self._get("/spot/quote/v1/kline", payload)
#         extra = {"interval": interval, "symbol": symbol}

#         if last_historical_prices.success:
#             if last_historical_prices.data:
#                 last_data = self._regular_historical_prices_payload(
#                     last_historical_prices, extra
#                 )
#                 if type(last_data.data) == list:
#                     last_data.data = last_data.data[0]
#                 return last_data
#             else:
#                 last_historical_prices.msg = "query latest historical prices is empty"

#         return last_historical_prices

#     def query_prices(self, symbol: str = None) -> dict:
#         """
#         {
#             "ret_code": 0,
#             "ret_msg": null,
#             "result": {
#                 "symbol": "BTCUSDT",
#                 "price": "50008"
#             },
#             "ext_code": null,
#             "ext_info": null
#         }
#         """

#         if symbol:
#             payload = {"symbol": symbol.upper().replace("/", "")}
#             spots = self._get("/spot/quote/v1/ticker/price", payload)
#         else:
#             spots = self._get("/spot/quote/v1/ticker/price")

#         logger.debug(spots)
#         if spots.success:
#             if spots.data:
#                 # print('spot raw', spots)
#                 return self._regular_symbol_payload(spots)
#             else:
#                 spots.msg = "query spots is empty"

#         return spots

#     def query_transfer(self, start: datetime = None, end: datetime = None) -> dict:
#         """ """
#         payload = {}
#         # ms - epoch time
#         ts_limit = 24 * 60 * 60 * 30
#         if start:
#             start = UTC_TZ.localize(
#                 start
#             )  # NOTE: bybit start and end timestamp default regard it as utc timestamp
#             payload["start_time"] = int(datetime.timestamp(start))
#         if end:
#             end = UTC_TZ.localize(end)
#             payload["end_time"] = int(datetime.timestamp(end))
#         if payload:
#             if not payload.get("start_time"):
#                 payload["start_time"] = payload["end_time"] - ts_limit
#             if not payload.get("end_time"):
#                 payload["end_time"] = payload["start_time"] + ts_limit
#         # print(f"query_transfer payload:{payload}")

#         raw_end_time = payload.get("end_time")
#         deposit_list = []
#         withdraw_list = []
#         while True:
#             if payload:
#                 if raw_end_time - payload["start_time"] > ts_limit:  # 30day
#                     payload["end_time"] = payload["start_time"] + ts_limit
#                 else:
#                     payload["end_time"] = raw_end_time

#             deposit = self._get(
#                 "/asset/v1/private/deposit/record/query", params=payload
#             )
#             if deposit.success and deposit.data:
#                 deposit_list += deposit.data.get("rows", [])
#             else:
#                 return deposit

#             withdraw = self._get(
#                 "/asset/v1/private/withdraw/record/query", params=payload
#             )
#             if withdraw.success and withdraw.data:
#                 withdraw_list += withdraw.data.get("rows", [])
#             else:
#                 return withdraw
#             # print(f'payload:{payload}....')
#             if not payload or (
#                 raw_end_time - payload["start_time"] <= ts_limit
#             ):  # 30day
#                 break
#             else:
#                 payload["start_time"] = payload["end_time"]

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

#     def query_internal_transfer(self) -> dict:
#         """ """
#         transfer = self._get("/asset/v1/private/transfer/list")
#         transfers = list()
#         # logger.debug(balance)
#         if transfer.success and transfer.data:
#             if not transfer.data["list"]:
#                 transfer.msg = "transfer list is empty"
#                 return transfer
#             for transfer_data in transfer.data["list"]:
#                 transfers.append(transfer_data)
#             transfer.data = transfers
#             return transfer
#         else:
#             return transfer


# class BybitFutureClient(BybitSpotClient):
#     BROKER_ID = ""
#     DATA_TYPE = "linear"

#     def info(self):
#         return "Bybit Future REST API start"

#     def query_time(self):
#         """query_time

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": {},
#             "time_now": "1577444332.192859"
#         }
#         """
#         # resp = self.request("GET", "/v2/public/time")
#         resp = self.query("GET", "/v2/public/time")
#         local_time = int(time.time())
#         server_time = int(float(resp.data["time_now"]))
#         self.time_offset = local_time - server_time
#         # print(f"local_time {local_time}")
#         # print(f"server_time {server_time}")
#         # print(f"timeoffset {self.time_offset}")
#         return server_time

#     def query_account(self) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.get_wallet_balance(coin="BTC"))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": {
#                 "BTC": {
#                     "equity": 1002,                         //equity = wallet_balance + unrealised_pnl
#                     "available_balance": 999.99987471,      //available_balance

#                     "used_margin": 0.00012529,              //used_margin = wallet_balance - available_balance
#                     "order_margin": 0.00012529,             //Used margin by order
#                     "position_margin": 0,                   //position margin
#                     "occ_closing_fee": 0,                   //position closing fee
#                     "occ_funding_fee": 0,                   //funding fee
#                     "wallet_balance": 1000,
#                     //wallet balance. When in Cross Margin mod, the number minus your unclosed loss is your real wallet balance.
#                     "realised_pnl": 0,                      //daily realized profit and loss
#                     "unrealised_pnl": 2,                    //unrealised profit and loss
#                         //when side is sell:
#                         // unrealised_pnl = size * (1.0 / mark_price -  1.0 / entry_price）
#                         //when side is buy:
#                         // unrealised_pnl = size * (1.0 / entry_price -  1.0 / mark_price）
#                     "cum_realised_pnl": 0,                  //total relised profit and loss
#                     "given_cash": 0,                        //given_cash
#                     "service_cash": 0                       //service_cash
#                 }
#             },
#             "time_now": "1578284274.816029",
#             "rate_limit_status": 98,
#             "rate_limit_reset_ms": 1580885703683,
#             "rate_limit": 100
#         }
#         """
#         balance = self._get("/v2/private/wallet/balance")
#         account = defaultdict(dict)
#         if balance.success and balance.data:
#             if balance.success and balance.data:
#                 datatype = self.DATA_TYPE.upper()
#                 for account_key, account_value in balance.data.items():
#                     free = float(account_value["available_balance"])
#                     locked = float(account_value["used_margin"])
#                     if free != 0 or locked != 0:
#                         key = account_key
#                         account[key]["symbol"] = key
#                         account[key]["available"] = free
#                         account[key]["frozen"] = locked
#                         account[key]["balance"] = free + locked
#                         account[key]["exchange"] = EXCHANGE
#                         account[key]["asset_type"] = datatype

#                 balance.data = account
#                 return balance
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
#         reduce_only: bool = False,
#         position_side: str = "net",
#         **kwargs,
#     ) -> dict:
#         """
#         Request Example
#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.place_active_order(
#             symbol="BTCUSDT",
#             side="Sell",
#             order_type="Limit",
#             qty=0.01,
#             price=8083,
#             time_in_force="GoodTillCancel",
#             reduce_only=False,
#             close_on_trigger=False
#         ))

#         Response Example
#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": {
#                 "order_id":"bd1844f-f3c0-4e10-8c25-10fea03763f6",
#                 "user_id": 1,
#                 "symbol": "BTCUSDT",
#                 "side": "Sell",
#                 "order_type": "Limit",
#                 "price": 8083,
#                 "qty": 0.01,
#                 "time_in_force": "GoodTillCancel",
#                 "order_status": "New",
#                 "last_exec_price": 8083,    //Last execution price
#                 "cum_exec_qty": 0,          //Cumulative qty of trading
#                 "cum_exec_value": 0,        //Cumulative value of trading
#                 "cum_exec_fee": 0,          //Cumulative trading fees
#                 "reduce_only": false,       //true means close order, false means open position
#                 "close_on_trigger": false
#                 "order_link_id": "",
#                 "created_time": "2019-10-21T07:28:19.396246Z",
#                 "updated_time": "2019-10-21T07:28:19.396246Z"},
#             "time_now": "1575111823.458705",
#             "rate_limit_status": 98,
#             "rate_limit_reset_ms": 1580885703683,
#             "rate_limit": 100
#         }
#         """

#         def bool_to_str_for_payload(input: bool):
#             return str(input).lower()

#         position_idx = (
#             0
#             if position_side.lower() == "net"
#             else 1
#             if position_side.lower() == "long"
#             else 2
#         )  ### 0 oneway mode; 1 hedge mode long; 2 hedge mode short;
#         payload = {
#             "symbol": symbol.upper().replace("/", ""),
#             "side": str(side).lower().capitalize(),
#             # 'price': price,
#             "qty": quantity,
#             # 'order_type': str(order_type).lower().capitalize(),
#             "order_link_id": self.new_order_id(),
#             "time_in_force": "GoodTillCancel",
#             "close_on_trigger": False,
#             "reduce_only": reduce_only,
#             # 'close_on_trigger':  bool_to_str_for_payload(False),  # NOTE: uncomment this if using
#             # 'reduce_only': bool_to_str_for_payload(reduce_only),
#             "stop_loss": stop_price,
#         }
#         # if not position_idx:
#         if not position_idx:
#             """
#             Requiredif you are under One-Way Mode:
#                 0-One-Way Mode
#                 1-Buy side of both side mode
#                 2-Sell side of both side mode
#             """
#             payload["position_idx"] = position_idx

#         if not order_type:
#             if not price:
#                 order_type = "Market"
#             else:
#                 order_type = "Limit"

#             payload["order_type"] = order_type
#         else:
#             payload["order_type"] = order_type.lower().capitalize()

#         if order_type == "Limit":
#             payload["price"] = price

#         # print(f">>> payload:{self._remove_none(payload)}")
#         orders_response = self._post(
#             "/private/linear/order/create", self._remove_none(payload)
#         )

#         if orders_response.success:
#             if orders_response.data:
#                 # print(f"debug: {orders_response.data}")
#                 result = (
#                     models.SendOrderResultSchema()
#                 )  # .model_construct(orders_response.data)
#                 orders_response.data = result.from_bybit_to_form(
#                     orders_response.data, datatype=self.DATA_TYPE
#                 )
#             else:
#                 orders_response.msg = "query sender order is empty"

#         # ERROR MSG: oc_diff[*****], new_oc[*****] with ob[0]+AB[0]
#         # This message means that you have insufficient balance to place the order, you should make a deposit or reduce the traded amount of the selected order.
#         return orders_response

#     def cancel_order(self, order_id: str, symbol: str) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.cancel_active_order(
#             symbol="BTCUSDT",
#             order_id=""
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": {
#                 "order_id":"bd1844f-f3c0-4e10-8c25-10fea03763f6"},
#             "time_now": "1575112681.814760",
#             "rate_limit_status": 98,
#             "rate_limit_reset_ms": 1580885703683,
#             "rate_limit": 100
#         }
#         """
#         payload = {"symbol": symbol.upper().replace("/", "")}
#         if order_id:
#             # payload['order_link_id'] = order_id
#             payload["order_id"] = order_id

#         return self._post("/private/linear/order/cancel", payload)

#     def cancel_orders(self, symbol: str) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.cancel_all_active_orders(
#             symbol="BTCUSDT"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": [
#                 "89a38056-80f1-45b2-89d3-4d8e3a203a79",
#                 "89a38056-80f1-45b2-89d3-4d8e3a203a79"],
#             "time_now": "1575110339.105675",
#             "rate_limit_status": 98,
#             "rate_limit_reset_ms": 1580885703683,
#             "rate_limit": 100
#         }
#         """
#         payload = {"symbol": symbol}
#         return self._post("/private/linear/order/cancel-all", payload)

#     def query_order_status(self, symbol: str, order_id: str = None) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="",
#                     spot=True)
#         logger.debug(session.get_active_order(
#             orderId="889826641228952064"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "",
#             "ext_code": null,
#             "ext_info": null,
#             "result": {
#                 "accountId": "1054",
#                 "exchangeId": "301",
#                 "symbol": "ETHUSDT",
#                 "symbolName": "ETHUSDT",
#                 "orderLinkId": "162081160171552",
#                 "orderId": "889826641228952064",
#                 "price": "20000",
#                 "origQty": "10",
#                 "executedQty": "0",
#                 "cummulativeQuoteQty": "0",
#                 "avgPrice": "0",
#                 "status": "NEW",
#                 "timeInForce": "GTC",
#                 "type": "LIMIT",
#                 "side": "BUY",
#                 "stopPrice": "0.0",
#                 "icebergQty": "0.0",
#                 "time": "1620811601728",
#                 "updateTime": "1620811601743",
#                 "isWorking": true
#             }
#         }
#         """
#         payload = {"symbol": symbol, "limit": 50}
#         if order_id:
#             payload["order_link_id"] = order_id

#         order_status = self._get("/private/linear/order/list", payload)
#         if order_status.success:
#             if order_status.data["data"]:
#                 if "last_page" in order_status.data:
#                     pages = order_status.data["last_page"]
#                     order_status_list = list
#                     # query each pages
#                     for i in range(pages):
#                         payload["page"] = i + 1
#                         order_page = self._get("/private/linear/order/list", payload)
#                         order_status_list.append(order_page.data["data"])
#                     order_status.data = order_status_list

#                 else:
#                     order_status.data = order_status.data["data"]

#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"

#         return order_status

#     def query_all_orders(self, symbol: str, limit=50) -> dict:
#         """
#         {
#             "ret_code": 0,
#             "ret_msg": "ok",
#             "ext_code": "",
#             "result": {
#                 "current_page": 1,
#                 "last_page": 6,
#                 "data": [
#                     {
#                         "order_id":"bd1844f-f3c0-4e10-8c25-10fea03763f6",
#                         "user_id": 1,
#                         "symbol": "BTCUSDT",
#                         "side": "Sell",
#                         "order_type": "Limit",
#                         "price": 8083,
#                         "qty": 10,
#                         "time_in_force": "GoodTillCancel",
#                         "order_status": "New",
#                         "last_exec_price": 8083,
#                         "cum_exec_qty": 0,
#                         "cum_exec_value": 0,
#                         "cum_exec_fee": 0,
#                         "order_link_id": "",
#                         "reduce_only": false,
#                         "close_on_trigger": false,
#                         "created_time": "2019-10-21T07:28:19.396246Z",
#                         "updated_time": "2019-10-21T07:28:19.396246Z"}
#                 ]
#             },
#             "ext_info": null,
#             "time_now": "1577448922.437871",
#             "rate_limit_status": 98,
#             "rate_limit_reset_ms": 1580885703683,
#             "rate_limit": 100
#         }
#         """
#         payload = {"symbol": symbol, "limit": limit}
#         order_status = self._get(
#             "/private/linear/order/list", self._remove_none(payload)
#         )
#         if order_status.success:
#             if order_status.data["data"]:
#                 if "last_page" in order_status.data:
#                     pages = order_status.data["last_page"]
#                     order_status_list = list
#                     # query each pages
#                     for i in range(pages):
#                         payload["page"] = i + 1
#                         order_page = self._get("/private/linear/order/list", payload)
#                         order_status_list.append(order_page.data["data"])
#                     order_status.data = order_status_list

#                 else:
#                     order_status.data = order_status.data["data"]

#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"

#         return order_status

#     def query_open_orders(self, symbol: str) -> dict:  #  = None
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.query_active_order(
#             symbol="BTCUSDT",
#             order_id=""
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": {
#                 "order_id": "e3f7662b-8b94-42e2-8d46-dead09dd2a52",
#                 "user_id": 106958,
#                 "symbol": "BTCUSDT",
#                 "side": "Sell",
#                 "order_type": "Market",
#                 "price": 11775,
#                 "qty": 0.001,
#                 "time_in_force": "ImmediateOrCancel",
#                 "order_status": "Filled",
#                 "last_exec_price": 11874.5,
#                 "cum_exec_qty": 0.001,
#                 "cum_exec_value": 11.8745,
#                 "cum_exec_fee": 0.00890588,
#                 "order_link_id": "",
#                 "reduce_only": false,
#                 "close_on_trigger": false,
#                 "created_time": "2020-08-10T19:28:56Z",
#                 "updated_time": "2020-08-10T19:28:57Z"
#             },
#             "time_now": "1597171508.869341",
#             "rate_limit_status": 598,
#             "rate_limit_reset_ms": 1597171508867,
#             "rate_limit": 600
#         }

#         //When only symbol is passed, the response uses a different structure:

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": [
#                 {
#                     "order_id": "6449d89e-6ef5-4f54-a065-12b2744de0ae",
#                     "user_id": 118921,
#                     "symbol": "LINKUSDT",
#                     "side": "Buy",
#                     "order_type": "Limit",
#                     "price": 9,
#                     "qty": 0.1,
#                     "time_in_force": "GoodTillCancel",
#                     "order_status": "New",
#                     "last_exec_price": 11874.5,
#                     "cum_exec_qty": 0.005,
#                     "cum_exec_value": 11.8745,
#                     "cum_exec_fee": 0.00890588,
#                     "order_link_id": "",
#                     "reduce_only": false,
#                     "created_time": "2020-11-27T08:25:44Z",
#                     "updated_time": "2020-11-27T08:25:44Z",
#                     "take_profit": 0,
#                     "stop_loss": 0,
#                     "tp_trigger_by": "UNKNOWN",
#                     "sl_trigger_by": "UNKNOWN"
#                 },
#                 ...
#                 {
#                     "order_id": "6d4dc4e0-b4e3-4fc5-a92d-3d693bdff4a5",
#                     "user_id": 118921,
#                     "symbol": "LINKUSDT",
#                     "side": "Buy",
#                     "order_type": "Limit",
#                     "price": 8.2,
#                     "qty": 9999,
#                     "time_in_force": "GoodTillCancel",
#                     "order_status": "New",
#                     "last_exec_price": 11888.5,
#                     "cum_exec_qty": 0.004,
#                     "cum_exec_value": 11.8745,
#                     "cum_exec_fee": 0.00890588,
#                     "order_link_id": "",
#                     "reduce_only": false,
#                     "created_time": "2020-11-23T09:19:49Z",
#                     "updated_time": "2020-11-23T09:20:31Z",
#                     "take_profit": 0,
#                     "stop_loss": 0,
#                     "tp_trigger_by": "UNKNOWN",
#                     "sl_trigger_by": "UNKNOWN"
#                 }
#             ],
#             "time_now": "1606465563.551193",
#             "rate_limit_status": 599,
#             "rate_limit_reset_ms": 1606465563547,
#             "rate_limit": 600
#         }

#         """
#         if symbol:
#             payload = {"symbol": symbol}
#         else:
#             payload = None
#         order_status = self._get("/private/linear/order/search", payload)

#         if order_status.success:
#             if order_status.data:
#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"
#                 order_status.data = []

#         return order_status

#     def query_symbols(self, symbol: str = None) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com")
#         print(session.query_symbol())
#         Response Example

#         {
#             "ret_code":0,
#             "ret_msg":"OK",
#             "ext_code":"",
#             "ext_info":"",
#             "result":[
#                 {
#                     "name":"BTCUSD",
#                     "alias":"BTCUSD",
#                     "status":"Trading",
#                     "base_currency":"BTC",
#                     "quote_currency":"USD",
#                     "price_scale":2,
#                     "taker_fee":"0.00075",
#                     "maker_fee":"-0.00025",
#                     "funding_interval":480,
#                     "leverage_filter":{
#                         "min_leverage":1,
#                         "max_leverage":100,
#                         "leverage_step":"0.01"
#                     },
#                     "price_filter":{
#                         "min_price":"0.5",
#                         "max_price":"999999.5",
#                         "tick_size":"0.5"
#                     },
#                     "lot_size_filter":{
#                         "max_trading_qty":1000000,
#                         "min_trading_qty":1,
#                         "qty_step":1
#                     }
#                 },
#                 {
#                     "name":"EOSUSD",
#                     "alias":"EOSUSD",
#                     "status":"Trading",
#                     "base_currency":"EOS",
#                     "quote_currency":"USD",
#                     "price_scale":3,
#                     "taker_fee":"0.00075",
#                     "maker_fee":"-0.00025",
#                     "funding_interval":480,
#                     "leverage_filter":{
#                         "min_leverage":1,
#                         "max_leverage":50,
#                         "leverage_step":"0.01"
#                     },
#                     "price_filter":{
#                         "min_price":"0.001",
#                         "max_price":"1999.999",
#                         "tick_size":"0.001"
#                     },
#                     "lot_size_filter":{
#                         "max_trading_qty":1000000,
#                         "min_trading_qty":1,
#                         "qty_step":1
#                     }
#                 },
#                 {
#                     "name":"BTCUSDT",
#                     "alias":"BTCUSDT",
#                     "status":"Trading",
#                     "base_currency":"BTC",
#                     "quote_currency":"USDT",
#                     "price_scale":2,
#                     "taker_fee":"0.00075",
#                     "maker_fee":"-0.00025",
#                     "funding_interval":480,
#                     "leverage_filter":{
#                         "min_leverage":1,
#                         "max_leverage":100,
#                         "leverage_step":"0.01"
#                     },
#                     "price_filter":{
#                         "min_price":"0.5",
#                         "max_price":"999999.5",
#                         "tick_size":"0.5"
#                     },
#                     "lot_size_filter":{
#                         "max_trading_qty":100,
#                         "min_trading_qty":0.001,
#                         "qty_step":0.001
#                     }
#                 },
#                 {
#                     "name":"BTCUSDM21",
#                     "alias":"BTCUSD0625",
#                     "status":"Trading",
#                     "base_currency":"BTC",
#                     "quote_currency":"USD",
#                     "price_scale":2,
#                     "taker_fee":"0.00075",
#                     "maker_fee":"-0.00025",
#                     "funding_interval":480,
#                     "leverage_filter":{
#                         "min_leverage":1,
#                         "max_leverage":100,
#                         "leverage_step":"0.01"
#                     },
#                     "price_filter":{
#                         "min_price":"0.5",
#                         "max_price":"999999.5",
#                         "tick_size":"0.5"
#                     },
#                     "lot_size_filter":{
#                         "max_trading_qty":1000000,
#                         "min_trading_qty":1,
#                         "qty_step":1
#                     }
#                 }
#             ],
#             "time_now":"1615801223.589808"
#         }
#         """

#         symbols = self._get("/v2/public/symbols")

#         if symbols.success:
#             if symbols.data:
#                 if symbol:
#                     symbol = symbol.upper().replace("/", "")
#                     data = self._regular_symbols_payload(symbols)
#                     symbol_data = [
#                         x for x in data.data.keys() if x.lower() == symbol.lower()
#                     ]
#                     # logger.debug(symbol_data)
#                     if symbol_data:
#                         symbols.data = data.data[symbol_data[0]]
#                     else:
#                         symbols.msg = "query symbol is empty"
#                         symbols.data = {}
#                 else:
#                     return self._regular_symbols_payload(symbols)

#             else:
#                 symbols.msg = "query symbol is empty"

#         return symbols

#     def query_trades(
#         self, symbol: str, start: datetime = None, end: datetime = None, limit=200
#     ) -> dict:
#         """
#         start expect datetime in UTC
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.user_trade_records(
#             symbol="BTCUSDT"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#                 "ret_msg": "OK",
#                 "ext_code": "",
#                 "ext_info": "",
#                 "result": {
#                     "current_page": 1,
#                     "data": [
#                         {
#                             "order_id": "7369b2f4-52f1-4698-abf7-368e4ba9aefa",
#                             "order_link_id": "",
#                             "side": "Buy",  //Side Enum
#                             "symbol": "BTCUSDT", //Symbol Enum
#                             "exec_id": "9b8216fa-98d7-55c0-b5fa-279db5727996",
#                             "price": 5894,//Abandoned!!
#                             "order_price": 5894,
#                             "order_qty": 0.001,
#                             "order_type": "Limit", //Order Type Enum
#                             "fee_rate": 0.00075,
#                             "exec_price": 5894,
#                             "exec_type": "Trade", //Exec Type Enum
#                             "exec_qty": 0.001,
#                             "exec_fee": 0.0044205,
#                             "exec_value": 5.894,
#                             "leaves_qty": 0,
#                             "closed_size": 0, // The corresponding closing size of the closing order
#                             "last_liquidity_ind": "RemovedLiquidity",
#                             //Liquidity Enum, only valid while exec_type is Trade, AdlTrade, BustTrade
#                             "trade_time": 1585547384,//Abandoned!!
#                             "trade_time_ms": 1585547384847
#                         }
#                     ]
#                 },
#                 "time_now": "1577480599.097287",
#                 "rate_limit_status": 119,
#                 "rate_limit_reset_ms": 1580885703683,
#                 "rate_limit": 120
#             }

#         }
#         {'ret_code': 0, 'ret_msg': 'OK', 'ext_code': '', 'ext_info': '', 'result': {'current_page': 1, 'data': None}, 'time_now': '1658569281.352596', 'rate_limit_status': 119, 'rate_limit_reset_ms': 1658569281347, 'rate_limit': 120}
#         """
#         payload = {"symbol": symbol, "limit": limit}
#         if start:
#             payload["start_time"] = int(datetime.timestamp(start) * 1000)
#         if end:
#             payload["end_time"] = int(datetime.timestamp(end) * 1000)
#         # print(payload,'query_trades')
#         page = 1
#         trade_list = []
#         while True:
#             payload["page"] = page
#             trades = self._get(
#                 "/private/linear/trade/execution/list", self._remove_none(payload)
#             )
#             # print(trades,'?????????????', symbol)
#             if trades.success:
#                 if trades.data["data"]:
#                     trade_list += trades.data["data"]
#                     page += 1
#                 else:
#                     break
#             else:
#                 return trades

#         trades.data["data"] = trade_list
#         # print(f"DEBUG - len:{len(trade_list)}; page:{page}")

#         if trades.success:
#             if trades.data["data"]:
#                 trades.data = trades.data["data"]
#                 ret_trades = self._regular_trades_payload(trades)
#                 # aggregate by ori_order_id
#                 ret_trades.data = self._aggregate_trades(ret_trades["data"])
#                 return ret_trades
#             else:
#                 trades.msg = "query trades is empty"
#                 trades.data = []

#         return trades

#     def _regular_historical_prices_payload(
#         self,
#         common_response: models.CommonResponseSchema,
#         extra: dict = None,
#         limit: int = None,
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 data = models.HistoryOHLCSchema()
#                 result = data.from_bybit_to_form(msg, extra, datatype=self.DATA_TYPE)
#                 payload.append(result)
#             # print(f"original lenght: {len(payload)}")
#             if limit:
#                 payload = payload[-limit:]
#         elif isinstance(common_response.data, dict):
#             payload = models.HistoryOHLCSchema()
#             payload = payload.from_bybit_to_form(
#                 common_response.data, extra, datatype=self.DATA_TYPE
#             )
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def query_history(
#         self,
#         symbol: str,
#         interval: str = "1h",
#         start: datetime = None,
#         end: datetime = None,
#         limit: int = 200,
#     ) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com")
#         print(session.query_kline(
#             symbol="BTCUSDT",
#             interval=1,
#             limit=2,
#             from_time=1581231260
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": [
#             {
#                 "id": 3866948,
#                 "symbol": "BTCUSDT",
#                 "period": "1",
#                 "start_at": 1577836800, // Abandoned!!
#                 "volume": 1451.59,
#                 "open": 7700,           // Abandoned!!
#                 "high": 999999,
#                 "low": 0.5,
#                 "close": 6000,
#                 "interval": 1,
#                 "open_time": 1577836800,
#                 "turnover": 2.4343353100000003}],
#             "time_now": "1581928016.558522"
#         }
#         """

#         def iteration_history(symbol, interval, data, payload, retry=3):
#             retry_counter = 0
#             historical_prices_datalist = data.data
#             logger.debug(interval)
#             # logger.debug(historical_prices_datalist)

#             end_timestamp = payload["to"]
#             start_timestamp = payload["from"]
#             # print(historical_prices_datalist,'?????')
#             first_timestamp = int(historical_prices_datalist[-1]["open_time"])
#             interval_timestamp = end_timestamp - first_timestamp
#             logger.debug(f"first_timestamp {first_timestamp}")
#             logger.debug(f"start_timestamp {start_timestamp}")
#             logger.debug(f"interval_timestamp {interval_timestamp}")

#             while interval_timestamp > interval:
#                 first_timestamp = int(historical_prices_datalist[-1]["open_time"])
#                 interval_timestamp = end_timestamp - first_timestamp

#                 logger.debug(f"first_timestamp {first_timestamp}")
#                 logger.debug(f"start_timestamp {start_timestamp}")
#                 logger.debug(f"interval_timestamp {interval_timestamp}")
#                 # logger.debug(historical_prices_datalist[0:5])

#                 payload["from"] = first_timestamp

#                 prices = self._get("/public/linear/kline", self._remove_none(payload))
#                 # print(interval,interval_timestamp, limit, '????',prices)

#                 if prices.error:
#                     break
#                 logger.debug(prices.data[0:5])
#                 # print(prices.data[0], '-------------',prices.data[-1],"<------",len(prices.data),interval_timestamp , interval, interval_timestamp > interval)
#                 historical_prices_datalist.extend(prices.data[1:])
#                 historical_prices_datalist.sort(key=lambda k: k["open_time"])
#                 time.sleep(0.1)

#                 logger.debug(f"payload: {payload}")
#                 logger.debug(f"retry_counter: {retry_counter}")
#                 logger.debug(f"data length: {len(historical_prices_datalist)}")

#                 if len(prices.data) != 200:
#                     retry_counter += 1

#                 if retry_counter >= 3:
#                     break

#             data.data = historical_prices_datalist
#             logger.debug(f"data length: {len(historical_prices_datalist)}")
#             return data

#         # main
#         payload = {
#             "symbol": symbol.upper().replace("/", ""),
#             "interval": TIMEDELTA[interval],
#             # 'limit': limit
#         }

#         # seconds - default from 7 D
#         if start:
#             payload["from"] = int(datetime.timestamp(start))
#         else:
#             if limit < 200:
#                 if "m" in interval:
#                     d = 1
#                 elif "h" in interval:
#                     d = 9
#                 else:
#                     d = 200
#             else:
#                 if "m" in interval:
#                     d = 1
#                 elif "h" in interval:
#                     d = 200
#                 else:
#                     d = 950
#             payload["from"] = int(datetime.now().timestamp()) - (3600 * 24 * d)

#         if end:
#             payload["to"] = int(datetime.timestamp(end))
#         else:
#             payload["to"] = int(datetime.timestamp(datetime.now()))

#         historical_prices = self._get(
#             "/public/linear/kline", self._remove_none(payload)
#         )
#         # print('????',historical_prices)
#         extra = {"interval": interval, "symbol": symbol}

#         # logger.debug(historical_prices)
#         # logger.debug(payload)
#         if historical_prices.success:
#             # if 'result' in historical_prices.data:
#             if historical_prices.data:
#                 interval = TIMEDELTA_MAPPING_SEC[interval]
#                 historical_prices = iteration_history(
#                     symbol=symbol,
#                     interval=interval,
#                     data=historical_prices,
#                     payload=payload,
#                 )
#                 # print(historical_prices)
#                 return self._regular_historical_prices_payload(
#                     historical_prices, extra, limit=limit if limit < 200 else None
#                 )
#             else:
#                 historical_prices.msg = "query historical prices is empty"

#         return historical_prices

#     def query_last_price(
#         self, symbol: str, interval: str = "1m", limit: int = 1
#     ) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com")
#         print(session.latest_information_for_symbol(
#             symbol="BTCUSD"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": [
#                 {
#                     "symbol": "BTCUSD",
#                     "bid_price": "7230",
#                     "ask_price": "7230.5",
#                     "last_price": "7230.00",
#                     "last_tick_direction": "ZeroMinusTick",
#                     "prev_price_24h": "7163.00",
#                     "price_24h_pcnt": "0.009353",
#                     "high_price_24h": "7267.50",
#                     "low_price_24h": "7067.00",
#                     "prev_price_1h": "7209.50",
#                     "price_1h_pcnt": "0.002843",
#                     "mark_price": "7230.31",
#                     "index_price": "7230.14",
#                     "open_interest": 117860186,
#                     "open_value": "16157.26",
#                     "total_turnover": "3412874.21",
#                     "turnover_24h": "10864.63",
#                     "total_volume": 28291403954,
#                     "volume_24h": 78053288,
#                     "funding_rate": "0.0001",
#                     "predicted_funding_rate": "0.0001",
#                     "next_funding_time": "2019-12-28T00:00:00Z",
#                     "countdown_hour": 2,
#                     "delivery_fee_rate": "0",
#                     "predicted_delivery_price": "0.00",
#                     "delivery_time": ""
#                 },
#                 {
#                     "symbol": "BTCUSDM21",
#                     "bid_price": "67895",
#                     "ask_price": "67895.5",
#                     "last_price": "67895.50",
#                     "last_tick_direction": "ZeroPlusTick",
#                     "prev_price_24h": "68073.00",
#                     "price_24h_pcnt": "-0.002607",
#                     "high_price_24h": "69286.50",
#                     "low_price_24h": "66156.00",
#                     "prev_price_1h": "68028.50",
#                     "price_1h_pcnt": "-0.001955",
#                     "mark_price": "67883.88",
#                     "index_price": "62687.89",
#                     "open_interest": 531821673,
#                     "open_value": "0.00",
#                     "total_turnover": "85713.36",
#                     "turnover_24h": "2371.47",
#                     "total_volume": 5352251354,
#                     "volume_24h": 160716002,
#                     "funding_rate": "0",
#                     "predicted_funding_rate": "0",
#                     "next_funding_time": "",
#                     "countdown_hour": 0,
#                     "delivery_fee_rate": "0.0005",
#                     "predicted_delivery_price": "0.00",
#                     "delivery_time": "2021-06-25T08:00:00Z"
#                 }
#             ],
#             "time_now": "1577484619.817968"
#         }
#         """
#         payload = {
#             "symbol": symbol,
#             "interval": TIMEDELTA[interval],
#             "limit": limit,
#             "from": int(datetime.now().timestamp()) - (3600 * 24 * 1),
#         }

#         last_historical_prices = self._get(
#             "/public/linear/kline", self._remove_none(payload)
#         )
#         extra = {"interval": interval, "symbol": symbol}

#         # logger.debug(historical_prices)
#         if last_historical_prices.success:
#             if last_historical_prices.data:
#                 last_data = self._regular_historical_prices_payload(
#                     last_historical_prices, extra
#                 )
#                 if type(last_data.data) == list:
#                     last_data.data = last_data.data[0]
#                 return last_data
#             else:
#                 last_historical_prices.msg = "query historical prices is empty"

#         return last_historical_prices

#     def query_prices(self, symbol: str) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.query_index_price_kline(
#             symbol="BTCUSDT",
#             interval=1,
#             limit=2,
#             from_time=1581231260
#         )
#         Response Example

#         {
#             "ret_code":0,
#             "ret_msg":"OK",
#             "ext_code":"",
#             "ext_info":"",
#             "result":[{
#                 "symbol":"BTCUSDT",
#                 "period":"1",
#                 "open_time":1582231260,
#                 "open":"10106.09",
#                 "high":"10108.75",
#                 "low":"10104.66",
#                 "close":"10108.73"
#             }],
#             "time_now":"1591263582.601795"
#         }
#         """
#         payload = {
#             "symbol": symbol.upper().replace("/", ""),
#             "interval": "1",
#             "limit": 1,
#             "from": int(datetime.now().timestamp()) - 3600 * 24 * 1,
#         }
#         future = self._get("/public/linear/index-price-kline", payload)

#         if future.success:
#             if future.data:
#                 # print('linear raw', future)
#                 return self._regular_symbol_payload(future)
#             else:
#                 future.msg = "query future is empty"

#         return future

#     def _regular_position_payload(
#         self, common_response: models.CommonResponseSchema
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 data = models.PositionSchema()
#                 result = data.from_bybit_to_form(msg, datatype=self.DATA_TYPE)
#                 payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.PositionSchema()
#             payload = payload.from_bybit_to_form(
#                 common_response.data, datatype=self.DATA_TYPE
#             )
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def query_position(self, symbol: str = None, force_return: bool = False) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.my_position(
#             symbol="BTCUSDT"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": [
#             {
#                     "user_id":100004,
#                     "symbol":"BTCUSDT",
#                     "side":"Buy",
#                     "size":0,
#                     "position_value":0,      //Current position value
#                     "entry_price":0,         //Average opening price
#                     "liq_price":1,           //Liquidation price
#                     "bust_price":100,        //Bust price
#                     "leverage":0,
#                     "is_isolated":true,
#                     "auto_add_margin": 0,
#                     "position_margin":0,     //Position margin
#                     "occ_closing_fee":0,     //Pre-occupancy closing fee
#                     "realised_pnl":0,        //Today's realised Profit and Loss
#                     "cum_realised_pnl":0,    //Cumulative realised Profit and Loss
#                     "free_qty": 30,          //Qty which can be closed
#                     "tp_sl_mode": "Full",
#                     "unrealised_pnl": 0,
#                     "deleverage_indicator": 0,
#                     "risk_id": 0,
#                     "stop_loss": 0,
#                     "take_profit": 0,
#                     "trailing_stop": 0
#                 },
#                 {
#                     "user_id":100004,
#                     "symbol":"BTCUSDT",
#                     "side":"Sell",
#                     "size":0,
#                     "position_value":0,
#                     "entry_price":0,
#                     "liq_price":1,
#                     "bust_price":100,
#                     "leverage":0,
#                     "is_isolated":true,
#                     "auto_add_margin": 0,
#                     "position_margin":0,
#                     "occ_closing_fee":0,
#                     "realised_pnl":0,
#                     "cum_realised_pnl":0,
#                     "free_qty": 30,
#                     "tp_sl_mode": "Full",
#                     "unrealised_pnl": 0,
#                     "deleverage_indicator": 0,
#                     "risk_id": 0,
#                     "stop_loss": 0,
#                     "take_profit": 0,
#                     "trailing_stop": 0
#                 }
#             ],
#             "time_now": "1577480599.097287",
#             "rate_limit_status": 119,
#             "rate_limit_reset_ms": 1580885703683,
#             "rate_limit": 120
#         }

#         //When symbol is not passed, the response uses a different structure:

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": [
#                 {
#                     "is_valid": true, //represents whether the current data is valid
#                     "data": { //the data can only be used when is_valid is true
#                         "user_id": 118921,
#                         "symbol": "BTCUSDT",
#                         "side": "Buy",
#                         "size": 0.009,
#                         "position_value": 117.6845,
#                         "entry_price": 13076.05555555,
#                         "liq_price": 11834,
#                         "bust_price": 11768.5,
#                         "leverage": 10,
#                         "is_isolated":true,
#                         "auto_add_margin": 0,
#                         "position_margin": 11.84788704,
#                         "occ_closing_fee": 0.07943738,
#                         "realised_pnl": 0,
#                         "cum_realised_pnl": -1.50755354,
#                         "free_qty": 0.009,
#                         "tp_sl_mode": "Full",
#                         "unrealised_pnl": 0,
#                         "deleverage_indicator": 0,
#                         "risk_id": 0,
#                         "stop_loss": 0,
#                         "take_profit": 0,
#                         "trailing_stop": 0
#                     }
#                 },
#                 {
#                     "is_valid": true, //represents whether the current data is valid
#                     "data": { //the data can only be used when is_valid is true
#                         "user_id": 118921,
#                         "symbol": "BTCUSDT",
#                         "side": "Sell",
#                         "size": 0.001,
#                         "position_value": 13.078,
#                         "entry_price": 13078,
#                         "liq_price": 14320,
#                         "bust_price": 14385.5,
#                         "leverage": 10,
#                         "is_isolated":true,
#                         "auto_add_margin": 0,
#                         "position_margin": 1.31858935,
#                         "occ_closing_fee": 0.01078913,
#                         "realised_pnl": 0,
#                         "cum_realised_pnl": 164.30402588,
#                         "free_qty": 0.001,
#                         "tp_sl_mode": "Full",
#                         "unrealised_pnl": 0,
#                         "deleverage_indicator": 0,
#                         "risk_id": 0,
#                         "stop_loss": 0,
#                         "take_profit": 0,
#                         "trailing_stop": 0
#                     }
#                 },
#                 ...
#                 {
#                     "is_valid": true, //represents whether the current data is valid
#                     "data": { //the data can only be used when is_valid is true
#                         "user_id": 118921,
#                         "symbol": "XTZUSDT",
#                         "side": "Buy",
#                         "size": 0,
#                         "position_value": 0,
#                         "entry_price": 0,
#                         "liq_price": 0,
#                         "bust_price": 0,
#                         "leverage": 25,
#                         "is_isolated":true,
#                         "auto_add_margin": 0,
#                         "position_margin": 0,
#                         "occ_closing_fee": 0,
#                         "realised_pnl": 0,
#                         "cum_realised_pnl": 0,
#                         "free_qty": 0,
#                         "tp_sl_mode": "Full",
#                         "unrealised_pnl": 0,
#                         "deleverage_indicator": 0,
#                         "risk_id": 0,
#                         "stop_loss": 0,
#                         "take_profit": 0,
#                         "trailing_stop": 0
#                     }
#                 },
#                 {
#                     "is_valid": true, //represents whether the current data is valid
#                     "data": { //the data can only be used when is_valid is true
#                         "user_id": 118921,
#                         "symbol": "XTZUSDT",
#                         "side": "Sell",
#                         "size": 0,
#                         "position_value": 0,
#                         "entry_price": 0,
#                         "liq_price": 0,
#                         "bust_price": 0,
#                         "leverage": 25,
#                         "is_isolated":true,
#                         "auto_add_margin": 0,
#                         "position_margin": 0,
#                         "occ_closing_fee": 0,
#                         "realised_pnl": 0,
#                         "cum_realised_pnl": 0,
#                         "free_qty": 0,
#                         "tp_sl_mode": "Full",
#                         "unrealised_pnl": 0,
#                         "deleverage_indicator": 0,
#                         "risk_id": 0,
#                         "stop_loss": 0,
#                         "take_profit": 0,
#                         "trailing_stop": 0
#                     }
#                 }
#             ],
#             "time_now": "1604302080.356538",
#             "rate_limit_status": 119,
#             "rate_limit_reset_ms": 1604302080353,
#             "rate_limit": 120
#         }
#         """

#         def handle_valid_data(data, symbol: str = None):
#             if symbol:
#                 update_result = data["data"]
#             else:
#                 update_result = [
#                     result["data"] for result in data.data if result.get("is_valid")
#                 ]
#             # print(f"update_result: {update_result}??")
#             if not force_return:
#                 results = [
#                     result
#                     for result in update_result
#                     if float(result["position_value"]) != 0
#                 ]
#             else:
#                 results = update_result
#             # print(f"results: {results}??")

#             logger.debug(results)
#             if symbol:
#                 data.data = [result for result in results if result["symbol"] == symbol]
#             else:
#                 data.data = results

#             if not data.data:
#                 data.msg = "query future position is empty"
#             return data

#         position = self._get(
#             "/private/linear/position/list",
#             params=self._remove_none({"symbol": symbol}),
#         )
#         # print(f"position ret: {position}")
#         if position.success:
#             if position.data:
#                 position = handle_valid_data(position, symbol)
#                 print(position, "!!!!!")
#                 return self._regular_position_payload(position)
#             else:
#                 position.msg = "query future position is empty"

#         return position

#     def set_position_mode(
#         self, mode: str, coin: str = "USDT", symbol: str = ""
#     ) -> dict:
#         """
#         {
#             "ret_code": 0,
#             "ret_msg": "ok",
#             "ext_code": "",
#             "result": 2,
#             "ext_info": null,
#             "time_now": "1577477968.175013",
#             "rate_limit_status": 74,
#             "rate_limit_reset_ms": 1577477968183,
#             "rate_limit": 75
#         }
#         """
#         if mode not in ["oneway", "hedge"]:
#             raise ValueError("Invalid position_mode")
#         payload = {"mode": "MergedSingle" if mode == "oneway" else "BothSide"}
#         if symbol:
#             payload["symbol"] = symbol
#         else:
#             payload["coin"] = coin

#         mode_payload = self._post(
#             "/private/linear/position/switch-mode", self._remove_none(payload)
#         )
#         # print(f"??? set_position_mode:{mode_payload}")
#         if mode_payload.success:
#             mode_payload.data = models.PositionModeSchema(
#                 mode=mode, coin=coin, symbol=symbol
#             ).dict()
#             return mode_payload

#         return mode_payload

#     def set_leverage(
#         self, leverage: float, symbol: str, is_isolated: bool = True
#     ) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.set_leverage(
#             symbol="BTCUSDT",
#             buy_leverage=10,
#             sell_leverage=10
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": null,
#             "time_now": "1585881527.650138",
#             "rate_limit_status": 74,
#             "rate_limit_reset_ms": 1585881527648,
#             "rate_limit": 75
#         }
#         """
#         payload = {
#             "symbol": symbol.upper().replace("/", ""),
#             "is_isolated": is_isolated,
#             "buy_leverage": leverage,
#             "sell_leverage": leverage,
#         }
#         leverage_payload = self._post(
#             "/private/linear/position/switch-isolated", self._remove_none(payload)
#         )

#         if leverage_payload.success:
#             if leverage_payload.data:
#                 leverage_payload.data = models.LeverageSchema(
#                     symbol=symbol, leverage=leverage
#                 ).dict()
#                 return leverage_payload
#             else:
#                 leverage_payload.msg = "fail to config leverage_payload"
#         else:
#             if (
#                 leverage_payload.data.get("msg", {}).get("ret_code", 0) == 130056
#             ):  # Isolated not modified
#                 # print(leverage_payload,'!!!!')
#                 leverage_payload = self._post(
#                     "/private/linear/position/set-leverage", self._remove_none(payload)
#                 )
#                 if leverage_payload.success:
#                     if leverage_payload.data:
#                         leverage_payload.data = models.LeverageSchema(
#                             symbol=symbol, leverage=leverage
#                         ).dict()
#                         return leverage_payload
#                     else:
#                         leverage_payload.msg = "fail to config leverage_payload"

#         return leverage_payload

#     def _regular_incomes_payload(
#         self, common_response: models.CommonResponseSchema
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 data = models.IncomeSchema()
#                 result = data.from_bybit_to_form(msg, datatype=self.DATA_TYPE)
#                 payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.IncomeSchema()
#             payload = payload.from_bybit_to_form(
#                 common_response.data, datatype=self.DATA_TYPE
#             )
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def query_funding_fee(
#         self, symbol: str, start: datetime = None, end: datetime = None, limit=1000
#     ) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.my_last_funding_fee(
#             symbol="BTCUSDT"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": {
#                 "symbol": "BTCUSDT",
#                 "side": "Buy",
#                 "size": 3.13,
#                 "funding_rate": 0.0001,
#                 "exec_fee": 1.868923,
#                 "exec_time": "2020-04-13T08:00:00.000Z"
#             },
#             "time_now": "1586780352.867171",
#             "rate_limit_status": 119,
#             "rate_limit_reset_ms": 1586780352864,
#             "rate_limit": 120
#         }
#         """
#         payload = {"symbol": symbol}
#         incomes = self._get(
#             "/private/linear/funding/prev-funding", self._remove_none(payload)
#         )
#         logger.debug(incomes)
#         if incomes.success:
#             if incomes.data and "msg" not in incomes.data:
#                 return self._regular_incomes_payload(incomes)
#             elif incomes.data and "msg" in incomes.data:
#                 incomes.data = []
#                 incomes.msg = "query future incomes is empty"
#             else:
#                 incomes.msg = "query future incomes is empty"

#         return incomes


# class BybitInverseFutureClient(BybitFutureClient):
#     BROKER_ID = ""
#     DATA_TYPE = "inverse"

#     def info(self):
#         return "Bybit Inverse Future REST API start"

#     def query_time(self):
#         """query_time

#         {'ret_code': 10002, 'ret_msg': 'invalid request, please check your timestamp and recv_window param. req_timestamp: 1651031897761 server_timestamp: 1651031901356 recv_window: 1000', 'ext_code': '', 'ext_info': '', 'result': None, 'time_now': '1651031901.356939'}}

#         """
#         # resp = self.request("GET", "/v2/public/time")
#         resp = self.query("GET", "/v2/public/time")
#         local_time = int(time.time())
#         server_time = int(float(resp.data["time_now"]))
#         self.time_offset = local_time - server_time
#         # logger.debug(f"local_time {local_time}")
#         # logger.debug(f"server_time {server_time}")
#         # logger.debug(f"timeoffset {self.time_offset}")
#         return server_time

#     def send_order(
#         self,
#         symbol: str,
#         quantity: float,
#         side: str,
#         order_type=None,
#         price=None,
#         stop_price=None,
#         reduce_only: bool = False,
#         position_side: str = "net",
#         **kwargs,
#     ) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.place_active_order(
#             symbol="BTCUSD",
#             side="Buy",
#             order_type="Market",
#             qty=1,
#             time_in_force="GoodTillCancel"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": {
#                 "user_id": 1,
#                 "order_id": "335fd977-e5a5-4781-b6d0-c772d5bfb95b",
#                 "symbol": "BTCUSD",
#                 "side": "Buy",
#                 "order_type": "Market",
#                 "price": 8800,
#                 "qty": 1,
#                 "timeInForce": "GoodTillCancel", # FIXME: different as docs
#                 "order_status": "Created",
#                 "last_exec_time": 0,
#                 "last_exec_price": 0,
#                 "leaves_qty": 1,
#                 "cum_exec_qty": 0,
#                 "cum_exec_value": 0,
#                 "cum_exec_fee": 0,
#                 "reject_reason": "",
#                 "order_link_id": "",
#                 "created_at": "2019-11-30T11:03:43.452Z",
#                 "updated_at": "2019-11-30T11:03:43.455Z"
#             },
#             "time_now": "1575111823.458705",
#             "rate_limit_status": 98,
#             "rate_limit_reset_ms": 1580885703683,
#             "rate_limit": 100
#         }
#         """
#         payload = {
#             "symbol": symbol.upper().replace("/", ""),
#             "side": str(side).lower().capitalize(),
#             # 'price': price,
#             "qty": int(quantity),
#             # 'order_type': str(order_type).capitalize(),
#             "order_link_id": self.new_order_id(),
#             "time_in_force": "GoodTillCancel",
#             "stop_loss": stop_price,
#             # 'reduce_only': str(reduce_only).lower()  # NOTE: uncomment if using self.request
#             "reduce_only": reduce_only,
#         }
#         if order_type is None:
#             if price is None:
#                 order_type = "Market"
#             else:
#                 order_type = "Limit"

#             payload["order_type"] = order_type
#         else:
#             payload["order_type"] = order_type.lower().capitalize()

#         if order_type == "Limit":
#             payload["price"] = price

#         if symbol[-2:].isdigit():
#             ### NOTE: in inverse contract, only future needs to specify the position side
#             suffix = "/futures/private/order/create"
#             position_idx = (
#                 0
#                 if position_side.lower() == "net"
#                 else 1
#                 if position_side.lower() == "long"
#                 else 2
#             )  ### 0 oneway mode; 1 hedge mode long; 2 hedge mode short;
#             if position_idx:
#                 """
#                 Requiredif you are under Hedge Mode:
#                     0-One-Way Mode
#                     1-Buy side of both side mode
#                     2-Sell side of both side mode
#                 """
#                 payload["position_idx"] = position_idx
#         else:
#             ### NOTE: in inverse contract, pertatual doenst need to specify the position side
#             suffix = "/v2/private/order/create"
#         orders_response = self._post(suffix, self._remove_none(payload))

#         if orders_response.success:
#             if orders_response.data:
#                 result = (
#                     models.SendOrderResultSchema()
#                 )  # .model_construct(orders_response.data)
#                 orders_response.data = result.from_bybit_to_form(
#                     orders_response.data, datatype=self.DATA_TYPE
#                 )
#             else:
#                 orders_response.msg = "placed order is empty"

#         return orders_response

#     def cancel_order(self, order_id: str, symbol: str) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.cancel_active_order(
#             symbol="BTCUSD",
#             order_id="3bd1844f-f3c0-4e10-8c25-10fea03763f6"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": {
#                 "user_id": 1,
#                 "order_id": "3bd1844f-f3c0-4e10-8c25-10fea03763f6",
#                 "symbol": "BTCUSD",
#                 "side": "Buy",
#                 "order_type": "Limit",
#                 "price": 8800,
#                 "qty": 1,
#                 "time_in_force": "GoodTillCancel",
#                 "order_status": "New",
#                 "last_exec_time": 0,
#                 "last_exec_price": 0,
#                 "leaves_qty": 1,
#                 "cum_exec_qty": 0,
#                 "cum_exec_value": 0,
#                 "cum_exec_fee": 0,
#                 "reject_reason": "",
#                 "order_link_id": "",
#                 "created_at": "2019-11-30T11:17:18.396Z",
#                 "updated_at": "2019-11-30T11:18:01.811Z"
#             },
#             "time_now": "1575112681.814760",
#             "rate_limit_status": 98,
#             "rate_limit_reset_ms": 1580885703683,
#             "rate_limit": 100
#         }
#         """
#         payload = {"symbol": symbol.upper().replace("/", "")}
#         if order_id:
#             # payload['order_link_id'] = order_id
#             payload["order_id"] = order_id

#         if symbol[-2:].isdigit():
#             suffix = "/futures/private/order/cancel"
#         else:
#             suffix = "/v2/private/order/cancel"

#         return self._post(suffix, payload)

#     def cancel_orders(self, symbol: str) -> dict:
#         """
#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.cancel_active_order(
#             symbol="BTCUSD",
#             order_id="3bd1844f-f3c0-4e10-8c25-10fea03763f6"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": {
#                 "user_id": 1,
#                 "order_id": "3bd1844f-f3c0-4e10-8c25-10fea03763f6",
#                 "symbol": "BTCUSD",
#                 "side": "Buy",
#                 "order_type": "Limit",
#                 "price": 8800,
#                 "qty": 1,
#                 "time_in_force": "GoodTillCancel",
#                 "order_status": "New",
#                 "last_exec_time": 0,
#                 "last_exec_price": 0,
#                 "leaves_qty": 1,
#                 "cum_exec_qty": 0,
#                 "cum_exec_value": 0,
#                 "cum_exec_fee": 0,
#                 "reject_reason": "",
#                 "order_link_id": "",
#                 "created_at": "2019-11-30T11:17:18.396Z",
#                 "updated_at": "2019-11-30T11:18:01.811Z"
#             },
#             "time_now": "1575112681.814760",
#             "rate_limit_status": 98,
#             "rate_limit_reset_ms": 1580885703683,
#             "rate_limit": 100
#         }
#         """
#         payload = {"symbol": symbol}

#         if symbol[-2:].isdigit():
#             suffix = "/futures/private/order/cancelAll"
#         else:
#             suffix = "/v2/private/order/cancelAll"

#         return self._post(suffix, payload)

#     def query_order_status(self, symbol: str, order_id: str = None) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.query_active_order(
#             symbol="BTCUSD",
#             order_id="e66b101a-ef3f-4647-83b5-28e0f38dcae0"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": {
#                 "user_id": 106958,
#                 "symbol": "BTCUSD",
#                 "side": "Buy",
#                 "order_type": "Limit",
#                 "price": "11756.5",
#                 "qty": 1,
#                 "time_in_force": "PostOnly",
#                 "order_status": "Filled",
#                 "ext_fields": {
#                     "o_req_num": -68948112492,
#                     "xreq_type": "x_create"
#                 },
#                 "last_exec_time": "1596304897.847944",
#                 "last_exec_price": "11756.5",
#                 "leaves_qty": 0,
#                 "leaves_value": "0",
#                 "cum_exec_qty": 1,
#                 "cum_exec_value": "0.00008505",
#                 "cum_exec_fee": "-0.00000002",
#                 "reject_reason": "",
#                 "cancel_type": "",
#                 "order_link_id": "",
#                 "created_at": "2020-08-01T18:00:26Z",
#                 "updated_at": "2020-08-01T18:01:37Z",
#                 "order_id": "e66b101a-ef3f-4647-83b5-28e0f38dcae0"
#             },
#             "time_now": "1597171013.867068",
#             "rate_limit_status": 599,
#             "rate_limit_reset_ms": 1597171013861,
#             "rate_limit": 600
#         }

#         //When only symbol is passed, the response uses a different structure:

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": [
#                 {
#                     "user_id": 100228,
#                     "symbol": "BTCUSD",
#                     "side": "Sell",
#                     "order_type": "Limit",
#                     "price": "17740",
#                     "qty": 10,
#                     "time_in_force": "GoodTillCancel",
#                     "order_status": "New",
#                     "ext_fields": {
#                         "o_req_num": 434743,
#                         "xreq_type": "x_create"
#                     },
#                     "last_exec_time": "1608193181.827761",
#                     "leaves_qty": 10,
#                     "leaves_value": "0.00056369",
#                     "cum_exec_qty": 0,
#                     "cum_exec_value": "0.00008505",
#                     "cum_exec_fee": "-0.00000002",
#                     "reject_reason": "EC_NoError",
#                     "cancel_type": "UNKNOWN",
#                     "order_link_id": "",
#                     "created_at": "2020-12-17T08:19:41.827637283Z",
#                     "updated_at": "2020-12-17T08:19:41.827761Z",
#                     "order_id": "d570d931-771e-4911-a24e-cdeddedb5b0e"
#                 },
#                 ...
#                 {
#                     "user_id": 100228,
#                     "symbol": "BTCUSD",
#                     "side": "Sell",
#                     "order_type": "Limit",
#                     "price": "17740",
#                     "qty": 10,
#                     "time_in_force": "GoodTillCancel",
#                     "order_status": "New",
#                     "ext_fields": {
#                         "o_req_num": 434728,
#                         "xreq_type": "x_create"
#                     },
#                     "last_exec_time": "1608193178.955412",
#                     "leaves_qty": 10,
#                     "leaves_value": "0.00056369",
#                     "cum_exec_qty": 0,
#                     "cum_exec_value": "0.00008505",
#                     "cum_exec_fee": "-0.00000002",
#                     "reject_reason": "EC_NoError",
#                     "cancel_type": "UNKNOWN",
#                     "order_link_id": "",
#                     "created_at": "2020-12-17T08:19:38.955297869Z",
#                     "updated_at": "2020-12-17T08:19:38.955412Z",
#                     "order_id": "88b91101-7ac1-40af-90b8-72d53fe23622"
#                 }
#             ],
#             "time_now": "1608193190.911073",
#             "rate_limit_status": 599,
#             "rate_limit_reset_ms": 1608193190909,
#             "rate_limit": 600
#         }

#         """
#         if symbol[-2:].isdigit():
#             suffix = "/futures/private/order"
#         else:
#             suffix = "/v2/private/order"

#         if symbol:
#             payload = {"symbol": symbol}
#         else:
#             payload = None

#         if order_id:
#             payload["order_link_id"] = order_id

#         order_status = self._get(suffix, payload)

#         if order_status.success:
#             if order_status.data:
#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"

#         return order_status

#     def query_all_orders(self, symbol: str = None, limit=50) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.get_active_order(
#             symbol="BTCUSD",
#             order_status="New"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": {
#                 "data": [
#                     {
#                         "user_id": 160861,
#                         "order_status": "Cancelled",
#                         "symbol": "BTCUSD",
#                         "side": "Buy",
#                         "order_type": "Market",
#                         "price": "9800",
#                         "qty": "16737",
#                         "time_in_force": "ImmediateOrCancel",
#                         "order_link_id": "",
#                         "order_id": "fead08d7-47c0-4d6a-b9e7-5c71d5df8ba1",
#                         "created_at": "2020-07-24T08:22:30Z",
#                         "updated_at": "2020-07-24T08:22:30Z",
#                         "leaves_qty": "0",
#                         "leaves_value": "0",
#                         "cum_exec_qty": "0",
#                         "cum_exec_value": "0",
#                         "cum_exec_fee": "0",
#                         "reject_reason": "EC_NoImmediateQtyToFill"
#                     }
#                 ],
#                 "cursor": "w01XFyyZc8lhtCLl6NgAaYBRfsN9Qtpp1f2AUy3AS4+fFDzNSlVKa0od8DKCqgAn"
#             },
#             "time_now": "1604653633.173848",
#             "rate_limit_status": 599,
#             "rate_limit_reset_ms": 1604653633171,
#             "rate_limit": 600
#         }
#         """
#         payload = {"symbol": symbol, "limit": limit}

#         if symbol[-2:].isdigit():
#             suffix = "/futures/private/order/list"
#         else:
#             suffix = "/v2/private/order/list"

#         order_status = self._get(suffix, payload)
#         print(f"????: {order_status}")
#         if order_status.success:
#             if order_status.data["data"]:
#                 order_status.data = order_status.data["data"]
#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"

#         return order_status

#     def query_open_orders(self, symbol: str) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.query_active_order(
#             symbol="BTCUSD",
#             order_id="e66b101a-ef3f-4647-83b5-28e0f38dcae0"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": {
#                 "user_id": 106958,
#                 "symbol": "BTCUSD",
#                 "side": "Buy",
#                 "order_type": "Limit",
#                 "price": "11756.5",
#                 "qty": 1,
#                 "time_in_force": "PostOnly",
#                 "order_status": "Filled",
#                 "ext_fields": {
#                     "o_req_num": -68948112492,
#                     "xreq_type": "x_create"
#                 },
#                 "last_exec_time": "1596304897.847944",
#                 "last_exec_price": "11756.5",
#                 "leaves_qty": 0,
#                 "leaves_value": "0",
#                 "cum_exec_qty": 1,
#                 "cum_exec_value": "0.00008505",
#                 "cum_exec_fee": "-0.00000002",
#                 "reject_reason": "",
#                 "cancel_type": "",
#                 "order_link_id": "",
#                 "created_at": "2020-08-01T18:00:26Z",
#                 "updated_at": "2020-08-01T18:01:37Z",
#                 "order_id": "e66b101a-ef3f-4647-83b5-28e0f38dcae0"
#             },
#             "time_now": "1597171013.867068",
#             "rate_limit_status": 599,
#             "rate_limit_reset_ms": 1597171013861,
#             "rate_limit": 600
#         }

#         //When only symbol is passed, the response uses a different structure:

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": [
#                 {
#                     "user_id": 100228,
#                     "symbol": "BTCUSD",
#                     "side": "Sell",
#                     "order_type": "Limit",
#                     "price": "17740",
#                     "qty": 10,
#                     "time_in_force": "GoodTillCancel",
#                     "order_status": "New",
#                     "ext_fields": {
#                         "o_req_num": 434743,
#                         "xreq_type": "x_create"
#                     },
#                     "last_exec_time": "1608193181.827761",
#                     "leaves_qty": 10,
#                     "leaves_value": "0.00056369",
#                     "cum_exec_qty": 0,
#                     "cum_exec_value": "0.00008505",
#                     "cum_exec_fee": "-0.00000002",
#                     "reject_reason": "EC_NoError",
#                     "cancel_type": "UNKNOWN",
#                     "order_link_id": "",
#                     "created_at": "2020-12-17T08:19:41.827637283Z",
#                     "updated_at": "2020-12-17T08:19:41.827761Z",
#                     "order_id": "d570d931-771e-4911-a24e-cdeddedb5b0e"
#                 },
#                 ...
#                 {
#                     "user_id": 100228,
#                     "symbol": "BTCUSD",
#                     "side": "Sell",
#                     "order_type": "Limit",
#                     "price": "17740",
#                     "qty": 10,
#                     "time_in_force": "GoodTillCancel",
#                     "order_status": "New",
#                     "ext_fields": {
#                         "o_req_num": 434728,
#                         "xreq_type": "x_create"
#                     },
#                     "last_exec_time": "1608193178.955412",
#                     "leaves_qty": 10,
#                     "leaves_value": "0.00056369",
#                     "cum_exec_qty": 0,
#                     "cum_exec_value": "0.00008505",
#                     "cum_exec_fee": "-0.00000002",
#                     "reject_reason": "EC_NoError",
#                     "cancel_type": "UNKNOWN",
#                     "order_link_id": "",
#                     "created_at": "2020-12-17T08:19:38.955297869Z",
#                     "updated_at": "2020-12-17T08:19:38.955412Z",
#                     "order_id": "88b91101-7ac1-40af-90b8-72d53fe23622"
#                 }
#             ],
#             "time_now": "1608193190.911073",
#             "rate_limit_status": 599,
#             "rate_limit_reset_ms": 1608193190909,
#             "rate_limit": 600
#         }

#         """
#         if symbol[-2:].isdigit():
#             suffix = "/futures/private/order"
#         else:
#             suffix = "/v2/private/order"

#         if symbol:
#             payload = {"symbol": symbol}
#         else:
#             payload = None
#         order_status = self._get(suffix, payload)

#         if order_status.success:
#             if order_status.data:
#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"
#                 order_status.data = []
#         return order_status

#     def query_trades(
#         self, symbol: str, start: datetime = None, end: datetime = None, limit=200
#     ) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.user_trade_records(
#             symbol="BTCUSD"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": {
#                 "order_id": "Abandoned!!", // Abandoned!!
#                 "trade_list": [
#                     {
#                         "closed_size": 0, // The corresponding closing size of the closing order
#                         "cross_seq": 277136382,
#                         "exec_fee": "0.0000001",
#                         "exec_id": "256e5ef8-abfe-5772-971b-f944e15e0d68",
#                         "exec_price": "8178.5",
#                         "exec_qty": 1,
#                         "exec_time": "1571676941.70682",    //Abandoned!!
#                         "exec_type": "Trade", //Exec Type Enum
#                         "exec_value": "0.00012227",
#                         "fee_rate": "0.00075",
#                         "last_liquidity_ind": "RemovedLiquidity",
#                         //Liquidity Enum, only valid while exec_type is Trade, AdlTrade, BustTrade
#                         "leaves_qty": 0,
#                         "nth_fill": 2,
#                         "order_id": "7ad50cb1-9ad0-4f74-804b-d82a516e1029",
#                         "order_link_id": "",
#                         "order_price": "8178",
#                         "order_qty": 1,
#                         "order_type": "Market", //Order Type Enum
#                         "side": "Buy", //Side Enum
#                         "symbol": "BTCUSD", //Symbol Enum
#                         "user_id": 1,
#                         "trade_time_ms": 1577480599000
#                     }
#                 ]
#             },
#             "time_now": "1577483699.281488",
#             "rate_limit_status": 118,
#             "rate_limit_reset_ms": 1577483699244737,
#             "rate_limit": 120
#         }
#         """
#         # don't suport end_time
#         payload = {"symbol": symbol, "limit": limit}
#         if start:
#             payload["start_time"] = int(datetime.timestamp(start) * 1000)

#         if symbol[-2:].isdigit():
#             suffix = "/futures/private/execution/list"
#         else:
#             suffix = "/v2/private/execution/list"

#         # trades = self._get(suffix, self._remove_none(payload))

#         page = 1
#         trade_list = []
#         while True:
#             payload["page"] = page
#             trades = self._get(suffix, self._remove_none(payload))
#             # print(trades,'?????')
#             if trades.success:
#                 if trades.data.get("trade_list", []):
#                     trade_list += trades.data["trade_list"]
#                     page += 1
#                 else:
#                     break
#             else:
#                 return trades

#         trades.data["trade_list"] = trade_list
#         print(f"DEBUG - len:{len(trade_list)}; page:{page}")

#         if trades.success:
#             if trades.data.get("trade_list", []):
#                 trades.data = trades.data["trade_list"]
#                 ret_trades = self._regular_trades_payload(trades)
#                 # aggregate by ori_order_id
#                 ret_trades.data = self._aggregate_trades(ret_trades["data"])
#                 return ret_trades
#             else:
#                 trades.msg = "query trades is empty"
#                 trades.data = []

#         return trades

#     def query_history(
#         self,
#         symbol: str,
#         interval: str = "1h",
#         start: datetime = None,
#         end: datetime = None,
#         limit: int = 200,
#     ) -> dict:
#         """
#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com")
#         print(session.query_kline(
#             symbol="BTCUSD",
#             interval="m",
#             from_time=1581231260
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": [{
#                 "symbol": "BTCUSD",
#                 "interval": "1",
#                 "open_time": 1581231300,
#                 "open": "10112.5",
#                 "high": "10112.5",
#                 "low": "10112",
#                 "close": "10112",
#                 "volume": "75981",
#                 "turnover": "7.51394369"
#             }, {
#                 "symbol": "BTCUSD",
#                 "interval": "1",
#                 "open_time": 1581231360,
#                 "open": "10112",
#                 "high": "10112.5",
#                 "low": "10112",
#                 "close": "10112",
#                 "volume": "24616",
#                 "turnover": "2.4343353100000003"
#             }],
#             "time_now": "1581928016.558522"
#         }
#         """

#         def iteration_history(symbol, interval, data, payload, retry=3):
#             retry_counter = 0
#             historical_prices_datalist = data.data
#             logger.debug(interval)
#             # logger.debug(historical_prices_datalist)

#             end_timestamp = payload["to"]
#             start_timestamp = payload["from"]
#             first_timestamp = int(historical_prices_datalist[-1]["open_time"])
#             interval_timestamp = end_timestamp - first_timestamp
#             logger.debug(f"first_timestamp {first_timestamp}")
#             logger.debug(f"start_timestamp {start_timestamp}")
#             logger.debug(f"interval_timestamp {interval_timestamp}")

#             while interval_timestamp > interval:
#                 first_timestamp = int(historical_prices_datalist[-1]["open_time"])
#                 interval_timestamp = end_timestamp - first_timestamp

#                 logger.debug(f"first_timestamp {first_timestamp}")
#                 logger.debug(f"start_timestamp {start_timestamp}")
#                 logger.debug(f"interval_timestamp {interval_timestamp}")
#                 # logger.debug(historical_prices_datalist[0:5])

#                 payload["from"] = first_timestamp

#                 prices = self._get("/v2/public/kline/list", self._remove_none(payload))
#                 if prices.error:
#                     break
#                 logger.debug(prices.data[0:5])

#                 historical_prices_datalist.extend(prices.data[1:])
#                 historical_prices_datalist.sort(key=lambda k: k["open_time"])
#                 time.sleep(0.1)

#                 if len(prices.data) != 200:
#                     retry_counter += 1

#                 logger.debug(f"payload: {payload}")
#                 logger.debug(f"retry_counter: {retry_counter}")
#                 logger.debug(f"data length: {len(historical_prices_datalist)}")

#                 if retry_counter >= 3:
#                     break

#             data.data = historical_prices_datalist
#             logger.debug(f"data length: {len(historical_prices_datalist)}")
#             return data

#         # main
#         payload = {
#             "symbol": symbol.upper().replace("/", ""),
#             "interval": TIMEDELTA[interval],
#             # 'limit': limit
#         }

#         # seconds - default from 7 D
#         if start:
#             payload["from"] = int(datetime.timestamp(start))
#         else:
#             if limit < 200:
#                 if "m" in interval:
#                     d = 1
#                 elif "h" in interval:
#                     d = 9
#                 else:
#                     d = 200
#             else:
#                 if "m" in interval:
#                     d = 1
#                 elif "h" in interval:
#                     d = 200
#                 else:
#                     d = 900
#             payload["from"] = int(datetime.now().timestamp()) - (3600 * 24 * d)

#         if end:
#             payload["to"] = int(datetime.timestamp(end))
#         else:
#             payload["to"] = int(datetime.timestamp(datetime.now()))

#         historical_prices = self._get(
#             "/v2/public/kline/list", self._remove_none(payload)
#         )
#         extra = {"interval": interval, "symbol": symbol}

#         # logger.info(payload)
#         # logger.debug(historical_prices)
#         if historical_prices.success:
#             # if 'result' in historical_prices.data:
#             if historical_prices.data:
#                 # seconds (spot use ms, future use sec)
#                 interval = TIMEDELTA_MAPPING_SEC[interval]
#                 historical_prices = iteration_history(
#                     symbol=symbol,
#                     interval=interval,
#                     data=historical_prices,
#                     payload=payload,
#                 )
#                 return self._regular_historical_prices_payload(
#                     historical_prices, extra, limit=limit if limit < 200 else None
#                 )
#             else:
#                 historical_prices.msg = "query historical prices is empty"

#         return historical_prices

#     def query_last_price(
#         self, symbol: str, interval: str = "1m", limit: int = 1
#     ) -> dict:
#         """
#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com")
#         print(session.query_kline(
#             symbol="BTCUSD",
#             interval="m",
#             from_time=1581231260
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": [{
#                 "symbol": "BTCUSD",
#                 "interval": "1",
#                 "open_time": 1581231300,
#                 "open": "10112.5",
#                 "high": "10112.5",
#                 "low": "10112",
#                 "close": "10112",
#                 "volume": "75981",
#                 "turnover": "7.51394369"
#             }, {
#                 "symbol": "BTCUSD",
#                 "interval": "1",
#                 "open_time": 1581231360,
#                 "open": "10112",
#                 "high": "10112.5",
#                 "low": "10112",
#                 "close": "10112",
#                 "volume": "24616",
#                 "turnover": "2.4343353100000003"
#             }],
#             "time_now": "1581928016.558522"
#         }
#         """
#         payload = {
#             "symbol": symbol,
#             "interval": TIMEDELTA[interval],
#             "limit": limit,
#             "from": int(datetime.now().timestamp()) - (3600 * 24 * 1),
#         }

#         last_historical_prices = self._get(
#             "/v2/public/kline/list", self._remove_none(payload)
#         )
#         extra = {"interval": interval, "symbol": symbol}

#         logger.debug(last_historical_prices)
#         if last_historical_prices.success:
#             if last_historical_prices.data:
#                 last_data = self._regular_historical_prices_payload(
#                     last_historical_prices, extra
#                 )
#                 if type(last_data.data) == list:
#                     last_data.data = last_data.data[0]
#                 return last_data
#             else:
#                 last_historical_prices.msg = "query historical prices is empty"

#         return last_historical_prices

#     def query_prices(self, symbol: str) -> dict:
#         """
#         Request Example
#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com")
#         print(session.query_index_price_kline(
#             symbol="BTCUSD",
#             interval=30,
#             limit=200,
#             from_time=1600544880
#         ))
#         Response Example

#         {
#             "ret_code":0,
#             "ret_msg":"OK",
#             "ext_code":"",
#             "ext_info":"",
#             "result":[
#                 {
#                     "symbol":"BTCUSD",
#                     "period":"1",
#                     "open_time":1582231260,
#                     "open":"10106.09",
#                     "high":"10108.75",
#                     "low":"10104.66",
#                     "close":"10108.73"
#                 }
#             ],
#             "time_now":"1591263582.601795"
#         }
#         """
#         payload = {
#             "symbol": symbol.upper().replace("/", ""),
#             "interval": "1",
#             "limit": 1,
#             "from": int(datetime.now().timestamp()) - 3600 * 24 * 1,
#         }
#         inverse_future = self._get("/v2/public/index-price-kline", payload)

#         if inverse_future.success:
#             if inverse_future.data:
#                 # print('inverse_future raw', inverse_future)
#                 return self._regular_symbol_payload(inverse_future)
#             else:
#                 inverse_future.msg = "query inverse future is empty"

#         return inverse_future

#     def query_position(
#         self, symbol: str = None, asset_type=None, force_return: bool = False
#     ) -> dict:
#         """
#         asset_type: dated, perpetual
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.my_position(
#             symbol="BTCUSD"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": {
#                 "id": 27913,
#                 "user_id": 1,
#                 "risk_id": 1,
#                 "symbol": "BTCUSD",
#                 "side": "Buy",
#                 "size": 5,
#                 "position_value": "0.0006947",
#                 "entry_price": "7197.35137469",
#                 "is_isolated":true,
#                 "auto_add_margin": 0,
#                 "leverage": "1",
#                 "effective_leverage": "1",
#                 "position_margin": "0.0006947",
#                 "liq_price": "3608",
#                 "bust_price": "3599",
#                 "occ_closing_fee": "0.00000105",
#                 "occ_funding_fee": "0",
#                 "take_profit": "0",
#                 "stop_loss": "0",
#                 "trailing_stop": "0",
#                 "position_status": "Normal",
#                 "deleverage_indicator": 4,
#                 "oc_calc_data":
#                 "{\"blq\":2,\"blv\":\"0.0002941\",\"slq\":0,\"bmp\":6800.408,\"smp\":0,\"fq\":-5,\"fc\":-0.00029477,\"bv2c\":1.00225,\"sv2c\":1.0007575}",
#                 "order_margin": "0.00029477",
#                 "wallet_balance": "0.03000227",
#                 "realised_pnl": "-0.00000126",
#                 "unrealised_pnl": 0,
#                 "cum_realised_pnl": "-0.00001306",
#                 "cross_seq": 444081383,
#                 "position_seq": 287141589,
#                 "created_at": "2019-10-19T17:04:55Z",
#                 "updated_at": "2019-12-27T20:25:45.158767Z"
#                 "tp_sl_mode": "Partial"
#             },
#             "time_now": "1577480599.097287",
#             "rate_limit_status": 119,
#             "rate_limit_reset_ms": 1580885703683,
#             "rate_limit": 120
#         }

#         //When symbol is not passed, the response uses a different structure:

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": [
#                 {
#                     "is_valid": true, //represents whether the current data is valid
#                     "data": { //the data can only be used when is_valid is true
#                         "id": 0,
#                         "position_idx": 0,
#                         "mode": 0,
#                         "user_id": 118921,
#                         "risk_id": 1,
#                         "symbol": "BTCUSD",
#                         "side": "Buy",
#                         "size": 10,
#                         "position_value": "0.00076448",
#                         "entry_price": "13080.78694014",
#                         "is_isolated": false,
#                         "auto_add_margin": 1,
#                         "leverage": "100",
#                         "effective_leverage": "0.01",
#                         "position_margin": "0.40111704",
#                         "liq_price": "25",
#                         "bust_price": "25",
#                         "occ_closing_fee": "0.0003",
#                         "occ_funding_fee": "0",
#                         "take_profit": "0",
#                         "stop_loss": "0",
#                         "trailing_stop": "0",
#                         "position_status": "Normal",
#                         "deleverage_indicator": 1,
#                         "oc_calc_data": "{\"blq\":0,\"slq\":0,\"bmp\":0,\"smp\":0,\"fq\":-10,\"bv2c\":0.0115075,\"sv2c\":0.0114925}",
#                         "order_margin": "0",
#                         "wallet_balance": "0.40141704",
#                         "realised_pnl": "-0.00000008",
#                         "unrealised_pnl": 0.00003797,
#                         "cum_realised_pnl": "-0.090626",
#                         "cross_seq": 764786721,
#                         "position_seq": 581513847,
#                         "created_at": "2020-08-10T07:04:32Z",
#                         "updated_at": "2020-11-02T00:00:11.943371457Z"
#                         "tp_sl_mode": "Partial"
#                     }
#                 },
#                 ...
#                 {
#                     "is_valid": true, //represents whether the current data is valid
#                     "data": { //the data can only be used when is_valid is true
#                         "id": 0,
#                         "position_idx": 0,
#                         "mode": 0,
#                         "user_id": 118921,
#                         "risk_id": 35,
#                         "symbol": "XRPUSD",
#                         "side": "None",
#                         "size": 0,
#                         "position_value": "0",
#                         "entry_price": "0",
#                         "is_isolated": false,
#                         "auto_add_margin": 1,
#                         "leverage": "16.67",
#                         "effective_leverage": "16.67",
#                         "position_margin": "0",
#                         "liq_price": "0",
#                         "bust_price": "0",
#                         "occ_closing_fee": "0",
#                         "occ_funding_fee": "0",
#                         "take_profit": "0",
#                         "stop_loss": "0",
#                         "trailing_stop": "0",
#                         "position_status": "Normal",
#                         "deleverage_indicator": 0,
#                         "oc_calc_data": "{\"blq\":0,\"slq\":0,\"bmp\":0,\"smp\":0,\"bv2c\":0.06153301,\"sv2c\":0.06144302}",
#                         "order_margin": "0",
#                         "wallet_balance": "0",
#                         "realised_pnl": "0",
#                         "unrealised_pnl": 0,
#                         "cum_realised_pnl": "0",
#                         "cross_seq": -1,
#                         "position_seq": 352149441,
#                         "created_at": "2020-08-10T07:04:32Z",
#                         "updated_at": "2020-08-22T08:06:32Z"
#                         "tp_sl_mode": "Partial"
#                     }
#                 }
#             ],
#             "time_now": "1604302124.031104",
#             "rate_limit_status": 118,
#             "rate_limit_reset_ms": 1604302124020,
#             "rate_limit": 120
#         }
#         """

#         def handle_valid_data(data, symbol: str = None):
#             """
#             success=True error=False data=[{'data': {'id': 0, 'position_idx': 0, 'mode': 0, 'user_id': 540325, 'risk_id': 1, 'symbol': 'BTCUSDH23', 'side': 'None', 'size': 0, 'position_value': '0', 'entry_price': '0', 'is_isolated': True, 'auto_add_margin': 0, 'leverage': '10', 'effective_leverage': '10', 'position_margin': '0', 'liq_price': '0', 'bust_price': '0', 'occ_closing_fee': '0', 'occ_funding_fee': '0', 'take_profit': '0', 'stop_loss': '0', 'trailing_stop': '0', 'position_status': 'Normal', 'deleverage_indicator': 0, 'oc_calc_data': '{"blq":0,"slq":0,"bmp":0,"smp":0,"bv2c":0.10126,"sv2c":0.10114}', 'order_margin': '0', 'wallet_balance': '0.10546089', 'realised_pnl': '0', 'unrealised_pnl': 0, 'cum_realised_pnl': '-0.00290844', 'cross_seq': 11797767234, 'position_seq': 0, 'created_at': '2022-11-15T07:41:59.480004285Z', 'updated_at': '2023-01-13T00:00:00.048865577Z', 'tp_sl_mode': 'Full'}, 'is_valid': True}] msg='query ok' ??
#             success=True error=False data={'id': 0, 'position_idx': 0, 'mode': 0, 'user_id': 540325, 'risk_id': 31, 'symbol': 'XRPUSD', 'side': 'None', 'size': 0, 'position_value': '0', 'entry_price': '0', 'is_isolated': False, 'auto_add_margin': 1, 'leverage': '44', 'effective_leverage': '44', 'position_margin': '0', 'liq_price': '0', 'bust_price': '0', 'occ_closing_fee': '0', 'occ_funding_fee': '0', 'take_profit': '0', 'stop_loss': '0', 'trailing_stop': '0', 'position_status': 'Normal', 'deleverage_indicator': 0, 'oc_calc_data': '{"blq":0,"slq":0,"bmp":0,"smp":0,"bv2c":0.02394092,"sv2c":0.02391365}', 'order_margin': '0', 'wallet_balance': '0', 'realised_pnl': '0', 'unrealised_pnl': 0, 'cum_realised_pnl': '0', 'cross_seq': -1, 'position_seq': 0, 'created_at': '2022-08-24T15:08:44.558849069Z', 'updated_at': '2022-08-24T15:09:40.759467235Z', 'tp_sl_mode': 'Full'} msg='query ok' ??
#             """
#             if symbol:
#                 update_result = (
#                     [data["data"]]
#                     if isinstance(data["data"], dict)
#                     else [
#                         result["data"] for result in data.data if result.get("is_valid")
#                     ]
#                 )
#             else:
#                 update_result = [
#                     result["data"] for result in data.data if result.get("is_valid")
#                 ]
#             # print(f"update_result: {update_result}??")
#             if not force_return:
#                 results = [
#                     result
#                     for result in update_result
#                     if float(result["position_value"]) != 0
#                 ]
#             else:
#                 results = update_result
#             # print(f"results: {results}??")

#             logger.debug(results)
#             if symbol:
#                 data.data = [result for result in results if result["symbol"] == symbol]
#             else:
#                 data.data = results

#             if not data.data:
#                 data.msg = "query future position is empty"
#             return data

#         if symbol:
#             asset_type = "dated" if symbol[-2:].isdigit() else "perpetual"

#         if asset_type == "dated":
#             suffix = "/futures/private/position/list"
#             position = self._get(suffix, params=self._remove_none({"symbol": symbol}))
#         elif asset_type == "perpetual":
#             suffix = "/v2/private/position/list"
#             position = self._get(suffix, params=self._remove_none({"symbol": symbol}))
#         else:
#             # raise ValueError('query account type error.')
#             position = position_perp = self._get(
#                 "/v2/private/position/list",
#                 params=self._remove_none({"symbol": symbol}),
#             )
#             position_dated = self._get(
#                 "/futures/private/position/list",
#                 params=self._remove_none({"symbol": symbol}),
#             )
#             if position_perp.success:
#                 position.data += position_dated.data

#         # print(position,'??')

#         # if (not symbol and asset_type == 'perpetual') or asset_type == 'future':
#         #     position = handle_valid_data(position)

#         if position.success:
#             if position.data:
#                 position = handle_valid_data(position, symbol)
#                 return self._regular_position_payload(position)
#             else:
#                 position.msg = "query future position is empty"

#         return position

#     def set_leverage(
#         self, leverage: float, symbol: str, is_isolated: bool = True
#     ) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com",
#                     api_key="", api_secret="")
#         print(session.set_leverage(
#             symbol="BTCUSDT",
#             buy_leverage=10,
#             sell_leverage=10
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "OK",
#             "ext_code": "",
#             "ext_info": "",
#             "result": null,
#             "time_now": "1585881527.650138",
#             "rate_limit_status": 74,
#             "rate_limit_reset_ms": 1585881527648,
#             "rate_limit": 75
#         }
#         """
#         payload = {"symbol": symbol.upper().replace("/", "")}

#         if symbol[-2:].isdigit():
#             # inverse dated future
#             # suffix = '/futures/private/position/leverage/save'
#             suffix = "/futures/private/position/switch-isolated"
#             payload["buy_leverage"] = leverage
#             payload["sell_leverage"] = leverage
#             payload["is_isolated"] = is_isolated
#         else:
#             # suffix = '/v2/private/position/leverage/save'
#             suffix = "/v2/private/position/switch-isolated"
#             # payload['leverage'] = leverage
#             payload["buy_leverage"] = leverage
#             payload["sell_leverage"] = leverage
#             payload["is_isolated"] = is_isolated

#         leverage_payload = self._post(suffix, self._remove_none(payload))

#         if leverage_payload.success:
#             if leverage_payload.data:
#                 leverage_payload.data = models.LeverageSchema(
#                     symbol=symbol, leverage=leverage
#                 ).dict()
#                 return leverage_payload
#             else:
#                 leverage_payload.msg = "fail to config leverage_payload"
#         else:
#             if (
#                 leverage_payload.data.get("msg", {}).get("ret_code", 0) == 30084
#             ):  # Isolated not modified
#                 payload = {"symbol": symbol.upper().replace("/", "")}
#                 if symbol[-2:].isdigit():
#                     # inverse dated future
#                     suffix = "/futures/private/position/leverage/save"
#                     payload["buy_leverage"] = leverage
#                     payload["sell_leverage"] = leverage
#                 else:
#                     # inverse perpetual
#                     suffix = "/v2/private/position/leverage/save"
#                     payload["leverage"] = leverage
#                 leverage_payload = self._post(suffix, self._remove_none(payload))
#                 if leverage_payload.success:
#                     if leverage_payload.data:
#                         leverage_payload.data = models.LeverageSchema(
#                             symbol=symbol, leverage=leverage
#                         ).dict()
#                         return leverage_payload
#                     else:
#                         leverage_payload.msg = "fail to config leverage_payload"

#         return leverage_payload

#     def query_funding_fee(
#         self, symbol: str, start: datetime = None, end: datetime = None, limit=1000
#     ) -> dict:
#         """
#         Request Example

#         from pybit import HTTP
#         session = HTTP("https://api-testnet.bybit.com")
#         print(session.get_the_last_funding_rate(
#             symbol="BTCUSD"
#         ))
#         Response Example

#         {
#             "ret_code": 0,
#             "ret_msg": "ok",
#             "ext_code": "",
#             "result": {
#                 "symbol": "BTCUSD",
#                 "side": "Buy",  // Your position side at the time of settlement
#                 "size": 1,      // Your position size at the time of settlement
#                 "funding_rate": 0.0001,
#                 // Funding rate for settlement. When the funding rate is positive, longs pay shorts. When it is negative, shorts pay longs.
#                 "exec_fee": 0.00000002,  // Funding fee.
#                 "exec_timestamp": 1575907200  // The time of funding settlement occurred, UTC timestamp
#             },
#             "ext_info": null,
#             "time_now": "1577446900.717204",
#             "rate_limit_status": 119,
#             "rate_limit_reset_ms": 1577446900724,
#             "rate_limit": 120
#         }
#         """
#         # not support inverse future
#         payload = {"symbol": symbol}
#         incomes = self._get(
#             "/v2/private/funding/prev-funding", self._remove_none(payload)
#         )
#         logger.debug(incomes)
#         if incomes.success:
#             if incomes.data and "msg" not in incomes.data:
#                 return self._regular_incomes_payload(incomes)
#             elif incomes.data and "msg" in incomes.data:
#                 incomes.data = []
#                 incomes.msg = "query future incomes is empty"
#             else:
#                 incomes.msg = "query future incomes is empty"

#         return incomes


# class BybitOptionClient(BybitInverseFutureClient):
#     BROKER_ID = ""
#     DATA_TYPE = "option"

#     def info(self):
#         return "Bybit Option REST API start"

#     def _post(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
#         # response = self.request(method="POST", path=path, params=params)
#         response = self.query(method="POST", path=path, params=params)
#         return self._process_response(response)

#     def _auth(self, method, params, recv_window=1000):
#         """
#         Generates authentication signature per Bybit API specifications.

#         Notes
#         -------------------
#         Since the POST method requires a JSONified dict, we need to ensure
#         the signature uses lowercase booleans instead of Python's
#         capitalized booleans. This is done in the bug fix below.

#         """

#         api_key = self.key
#         api_secret = self.secret

#         if api_key is None or api_secret is None:
#             raise PermissionError("Authenticated endpoints require keys.")

#         # Append required parameters.
#         params["api_key"] = api_key
#         # bybit server have timeoffset issue - need to correct by querying time api
#         params["timestamp"] = int((time.time() - self.time_offset) * 10**3)

#         # Sort dictionary alphabetically to create querystring.
#         _val = "&".join(
#             [
#                 str(k) + "=" + str(v)
#                 for k, v in sorted(params.items())
#                 if (k != "sign") and (v is not None)
#             ]
#         )

#         # Bug fix. Replaces all capitalized booleans with lowercase.
#         if method == "POST":
#             _val = _val.replace("True", "true").replace("False", "false")

#         # Return signature.
#         return str(
#             hmac.new(
#                 bytes(api_secret, "utf-8"), bytes(_val, "utf-8"), digestmod="sha256"
#             ).hexdigest()
#         )

#     def sign(self, request):
#         """generate bybit signature"""
#         query = request.params if request.params else {}
#         query_data = request.data if request.data else {}
#         query.update(query_data)
#         method = request.method
#         request.path

#         # Bug fix: change floating whole numbers to integers to prevent
#         # auth signature errors.
#         if query is None:
#             query = {}
#         else:
#             for i in query.keys():
#                 if isinstance(query[i], float) and query[i] == int(query[i]):
#                     query[i] = int(query[i])

#         signature = self._auth(method=method, params=query)
#         # Sort the dictionary alphabetically.
#         query = dict(sorted(query.items(), key=lambda x: x))
#         query["sign"] = signature

#         if query is not None:
#             req_params = {k: v for k, v in query.items() if v is not None}
#         else:
#             req_params = {}

#         if method == "GET":
#             request.params = req_params
#             request.headers = {"Content-Type": "application/x-www-form-urlencoded"}
#         else:
#             request.headers = {"Content-Type": "application/json"}
#             request.params = req_params
#             logger.debug(request)

#         return request

#     def _process_response(self, response: Response) -> dict:
#         try:
#             data = response.data
#         except ValueError:
#             logger.debug(response.data())
#             raise
#         else:
#             # logger.debug(data)
#             # print(f"_process_response data:{data}")
#             if not response.ok or (
#                 response.ok and data.get("result") is None and data["retCode"] != 0
#             ):
#                 payload_data = models.CommonDataSchema(
#                     status_code=response.status_code, msg=dict(data)
#                 )
#                 return models.CommonResponseSchema(
#                     success=False,
#                     error=True,
#                     data=payload_data.dict(),
#                     msg=data["retMsg"],
#                 )
#             elif response.ok and data["result"] is None and data["retCode"] == 0:
#                 """send cancel order status"""
#                 payload_data = models.CommonDataSchema(
#                     status_code=response.status_code, msg=dict(data)
#                 )
#                 return models.CommonResponseSchema(
#                     success=True, error=False, data=payload_data.dict(), msg="query ok"
#                 )
#             else:
#                 return models.CommonResponseSchema(
#                     success=True, error=False, data=data["result"], msg="query ok"
#                 )

#     def query_trades(self) -> dict:
#         """
#         TRANSFER_IN
#         TRANSFER_OUT
#         TRADE
#         SETTLEMENT
#         DELIVERY
#         LIQUIDATION
#         """
#         payload = {"type": "TRADE"}
#         trades = self._post("/option/usdc/openapi/private/v1/execution-list", payload)
#         # account = defaultdict(dict)
#         return trades
#         if balance.success and balance.data:
#             if not balance.data["dataList"]:
#                 balance.msg = "account wallet is empty"
#                 return balance

#             for account_data in balance.data["dataList"]:
#                 free = float(account_data["unrealisedPnl"])
#                 locked = float(account_data["orderMargin"])
#                 if free != 0 or locked != 0:
#                     key = account_data["coin"]
#                     account[key]["symbol"] = key
#                     account[key]["available"] = free
#                     account[key]["frozen"] = locked
#                     account[key]["balance"] = free + locked
#                     account[key]["exchange"] = EXCHANGE
#                     account[key]["asset_type"] = "Option"

#             balance.data = account
#             return balance
#         else:
#             return balance

#     def query_account(self) -> dict:
#         """
#         {
#         "retCode":0,
#             "retMsg":"OK",
#             "result":{
#             "resultTotalSize":1,
#             "dataList":[
#             {
#                 "baseCoin":"BTC",
#                 "totalDelta":"-0.1093",
#                 "totalGamma":"0.00000",
#                 "totalVega":"1.8799",
#                 "totalTheta":"-19.2038",
#                 "totalRPL":"-3773.8879",
#                 "sessionUPL":"",
#                 "sessionRPL":"",
#                 "im":"28940.8205",
#                 "mm":"14997.4532"
#             }
#             ]
#         }
#         }
#         """
#         balance = self._post("/option/usdc/openapi/private/v1/query-asset-info")
#         print(balance, "<================balance")
#         wallet = self._post("/option/usdc/openapi/private/v1/query-wallet-balance")
#         print(wallet, "<================wallet")
#         # return balance
#         account = defaultdict(dict)
#         logger.debug(balance)
#         if balance.success and balance.data:
#             if not balance.data["dataList"]:
#                 balance.msg = "account wallet is empty"
#                 return balance

#             for account_data in balance.data["dataList"]:
#                 key = account_data["baseCoin"]
#                 account[key]["symbol"] = key
#                 # account[key]["available"] = free
#                 # account[key]["frozen"] = locked
#                 # account[key]["balance"] = free + locked
#                 account[key]["exchange"] = EXCHANGE
#                 account[key]["asset_type"] = self.DATA_TYPE

#             balance.data = account
#             return balance
#         else:
#             return balance

#     def query_position(self, force_return: bool = False) -> dict:
#         """
#         Request Example


#         Response Example

#         {
#         "result": {
#             "cursor": "BTCPERP%3A1640856463080%2CBTCPERP%3A1640856463080",
#             "resultTotalSize": 1,
#             "dataList": [
#             {
#                 "symbol": "BTCPERP",
#                 "leverage": "10.00",
#                 "occClosingFee": "0.0000",
#                 "liqPrice": "-",
#                 "positionValue": "1403.8050",
#                 "takeProfit": "0.0",
#                 "riskId": "10001",
#                 "trailingStop": "0.0000",
#                 "unrealisedPnl": "0.0000",
#                 "createdAt": "1640856463080",
#                 "markPrice": "46891.81",
#                 "cumRealisedPnl": "0.0000",
#                 "positionMM": "7.9666",
#                 "positionIM": "141.3281",
#                 "updatedAt": "1640856463080",
#                 "tpSLMode": "UNKNOWN",
#                 "side": "Buy",
#                 "bustPrice": "-",
#                 "deleverageIndicator": 0,
#                 "entryPrice": "46793.5",
#                 "size": "0.030",
#                 "sessionRPL": "0.0000",
#                 "positionStatus": "NORMAL",
#                 "sessionUPL": "0.3391",
#                 "stopLoss": "0.0",
#                 "orderMargin": "38.5415",
#                 "sessionAvgPrice": "46880.5"
#             }
#             ]
#         },
#         "retCode": 0,
#             "retMsg": "Success."
#         }
#         """
#         payload = {"category": "OPTION", "limit": 50}
#         balance = self._post("/option/usdc/openapi/private/v1/query-position", payload)
#         account = defaultdict(dict)
#         logger.debug(balance)
#         if balance.success and balance.data:
#             if not balance.data["dataList"]:
#                 balance.msg = "account wallet is empty"
#                 return balance

#             for account_data in balance.data["dataList"]:
#                 free = float(account_data["unrealisedPnl"])
#                 locked = float(account_data["orderMargin"])
#                 if free != 0 or locked != 0:
#                     key = account_data["coin"]
#                     account[key]["symbol"] = key
#                     account[key]["available"] = free
#                     account[key]["frozen"] = locked
#                     account[key]["balance"] = free + locked
#                     account[key]["exchange"] = EXCHANGE
#                     account[key]["asset_type"] = "Option"

#             balance.data = account
#             return balance
#         else:
#             return balance

#     def query_symbols(self, symbol: str = None) -> dict:
#         """
#         {
#             "retCode": 0,
#             "retMsg": "success",
#             "result": {
#                 "resultTotalSize": 1,
#                 "cursor": "",
#                 "dataList": [
#                     {
#                         "symbol": "BTC-30SEP22-400000-C",
#                         "status": "ONLINE",
#                         "baseCoin": "BTC",
#                         "quoteCoin": "USD",
#                         "settleCoin": "USDC",
#                         "takerFee": "0.0003",
#                         "makerFee": "0.0003",
#                         "minLeverage": "",
#                         "maxLeverage": "",
#                         "leverageStep": "",
#                         "minOrderPrice": "5",
#                         "maxOrderPrice": "10000000",
#                         "minOrderSize": "0.01",
#                         "maxOrderSize": "200",
#                         "tickSize": "5",
#                         "minOrderSizeIncrement": "0.01",
#                         "basicDeliveryFeeRate": "0.00015",
#                         "deliveryTime": "1664524800000"
#                     }
#                 ]
#             }
#         }
#         """
#         payload = {"status": "ONLINE"}
#         if symbol:
#             payload["symbol"] = symbol

#         symbols = self._get("/option/usdc/openapi/public/v1/symbols", payload)

#         if symbols.success:
#             if symbols.data:
#                 if symbol:
#                     # print(symbols,'???')
#                     if symbols.data["dataList"]:
#                         symbols.data = symbols.data["dataList"][0]
#                         return self._regular_symbols_payload(symbols)
#                     symbols.msg = "query symbol is empty"
#                     symbols.data = {}
#                 else:
#                     symbols.data = symbols.data["dataList"]
#                     return self._regular_symbols_payload(symbols)

#             else:
#                 symbols.msg = "query symbol is empty"

#         return symbols

#     def query_history(
#         self,
#         symbol: str,
#         interval: str = "1h",
#         start: datetime = None,
#         end: datetime = None,
#         limit: int = 1000,
#     ) -> dict:
#         """ """

#         def iteration_history(symbol, interval, data, payload, retry=3):
#             retry_counter = 0
#             historical_prices_datalist = data.data
#             # logger.debug(interval)

#             first_timestamp = int(historical_prices_datalist[0][0])
#             start_timestamp = payload["startTime"]
#             interval_timestamp = first_timestamp - start_timestamp
#             # logger.debug(payload)

#             while interval_timestamp > interval:
#                 first_timestamp = int(historical_prices_datalist[0][0])
#                 interval_timestamp = first_timestamp - start_timestamp

#                 logger.debug(f"first_timestamp {first_timestamp}")
#                 logger.debug(f"interval_timestamp {interval_timestamp}")
#                 # logger.debug(historical_prices_datalist[0:5])

#                 payload["endTime"] = first_timestamp

#                 prices = self._get("/spot/quote/v1/kline", self._remove_none(payload))
#                 if prices.error:
#                     break
#                 logger.debug(prices.data[0:2])

#                 historical_prices_datalist.extend(prices.data)
#                 historical_prices_datalist = [
#                     list(item)
#                     for item in set(tuple(x) for x in historical_prices_datalist)
#                 ]
#                 historical_prices_datalist.sort(key=lambda k: k[0])
#                 time.sleep(0.1)

#                 logger.debug(f"payload: {payload}")
#                 logger.debug(f"retry_counter: {retry_counter}")

#                 if len(prices.data) != 1000:
#                     retry_counter += 1

#                 if retry_counter >= 3:
#                     break

#             data.data = historical_prices_datalist
#             logger.debug(f"data length: {len(historical_prices_datalist)}")
#             return data

#         # main
#         payload = {
#             "symbol": symbol.upper().replace("/", ""),
#             "interval": SPOT_KBAR_INTERVAL[interval],
#             "limit": limit,
#         }

#         # ms - epoch time
#         if start:
#             start = UTC_TZ.localize(
#                 start
#             )  # NOTE: bybit start and end timestamp default regard it as utc timestamp
#             payload["startTime"] = int(datetime.timestamp(start) * 1000)
#         if end:
#             end = UTC_TZ.localize(end)
#             payload["endTime"] = int(datetime.timestamp(end) * 1000)
#         historical_prices = self._get(
#             "/spot/quote/v1/kline", self._remove_none(payload)
#         )
#         extra = {"interval": interval, "symbol": symbol}

#         if historical_prices.success:
#             if historical_prices.data:
#                 # handle query time
#                 if "startTime" in payload:
#                     # ms
#                     interval = TIMEDELTA_MAPPING_SEC[interval] * 1000
#                     historical_prices = iteration_history(
#                         symbol=symbol,
#                         interval=interval,
#                         data=historical_prices,
#                         payload=payload,
#                     )
#                 return self._regular_historical_prices_payload(historical_prices, extra)
#             else:
#                 historical_prices.msg = "query historical prices is empty"

#         return historical_prices


# class BybitSpotDataWebsocket(WebsocketClient):
#     """
#     Implement public topics
#     doc: https://bybit-exchange.github.io/docs/spot/#t-publictopics
#     """

#     def __init__(
#         self,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         ping_interval: int = 29,
#         is_testnet: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         self.is_testnet: bool = is_testnet
#         self.proxy_host: str = proxy_host
#         self.proxy_port: int = proxy_port
#         self.ping_interval: int = ping_interval

#         self.ticks: dict[str, dict] = {}
#         self.reqid: int = 0
#         self._logged_in = False
#         self.key = None
#         self.secret = None
#         # self.channel = 'kline'
#         self.subscribed = []
#         self.kbar_time: int = int(time.time()) - (int(time.time()) % 60)

#         self.on_tick = None
#         self.on_ticker_callback = None
#         self.on_depth_callback = None
#         self.on_kline_callback = None
#         self.on_connected_callback = None
#         self.on_disconnected_callback = None
#         self.on_error_callback = None

#     def info(self):
#         return "Bybit Spot Data Websocket start"

#     def connect(self):
#         """ """
#         self.init(
#             host=BYBIT_TESTNET_WS_SPOT_PUBLIC_HOST
#             if self.is_testnet
#             else BYBIT_WS_SPOT_PUBLIC_HOST,
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
#             print(
#                 f"Bybit spot market data websocket connect success ({datetime.now()})"
#             )
#         else:
#             print(f"Bybit spot market data websocket connect fail ({datetime.now()})")

#         # resubscribe
#         if self.ticks:
#             # for symbol in self.ticks.keys():
#             #     req: dict = self.sub_stream(symbol=symbol, channel=self.channel)
#             #     logger.debug(req)
#             #     self.send_packet(req)

#             for key, detail in self.ticks.items():
#                 req: dict = self.sub_stream(
#                     symbol=detail["symbol"],
#                     channel=detail["channel"],
#                     interval=detail.get("interval", "1m"),
#                 )
#                 # logger.debug(req)
#                 # print(f"!!!!! rea: {req}")
#                 print(f"on_connected, resubscribing the channels - req:{req}")
#                 self.send_packet(req)

#         if self.on_connected_callback:
#             self.on_connected_callback()

#     def on_disconnected(self) -> None:
#         print(
#             "Bybit Spot market data Websocket disconnected, now try to connect @ {datetime.now()}"
#         )
#         if self.on_disconnected_callback:
#             self.on_disconnected_callback()

#     def on_error(self, exception_type, exception_value, traceback_obj):
#         super().on_error(exception_type, exception_value, traceback_obj)
#         if self.on_error_callback:
#             self.on_error_callback(exception_value)

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
#         sub = {
#             "topic": SPOT_MAPPING_CHANNEL[channel],
#             "event": "sub",
#             "symbol": symbol,
#             "params": {"binary": False},
#         }

#         if channel == "kline":
#             sub["topic"] = f"kline_{SPOT_KBAR_INTERVAL[interval]}"
#         # print(self.reqid,'!!!!!!!!!!', symbol, '------',channel)
#         return sub

#     def subscribe(
#         self,
#         symbols: list | str,
#         on_tick: Callable = None,
#         channel="depth",
#         interval="1m",
#     ) -> None:
#         """
#         channel: 'depth', 'kline', 'ticker'
#         """

#         if channel not in ["depth", "kline", "ticker", "trades", "orders", "account"]:
#             Exception("invalid subscription")
#             return

#         # self.channel = channel
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

#         for symbol in symbols:
#             # symbol = symbol.replace('/', '')
#             if symbol in self.ticks:
#                 return

#             # tick data dict
#             tick = {
#                 "symbol": symbol,
#                 "exchange": EXCHANGE,
#                 "channel": channel,
#                 "interval": interval,
#                 "asset_type": ASSET_TYPE_SPOT,
#                 "name": models.normalize_name(symbol, ASSET_TYPE_SPOT),
#             }
#             key = ""
#             if channel == "kline":
#                 key = f"{symbol}|{interval}"
#             else:
#                 key = f"{symbol}|{channel}"

#             self.ticks[key] = tick
#             req: dict = self.sub_stream(
#                 symbol=symbol, channel=channel, interval=interval
#             )
#             self.send_packet(req)

#     def on_packet(self, packet: dict) -> None:
#         """on_packet
#         docs: https://bybit-exchange.github.io/docs/spot/#t-websocketv2depth
#         channel: depth, kline

#         depth

#         How to Subscribe

#         {
#             "topic": "depth",
#             "event": "sub",
#             "symbol": "BTCUSDT",
#             "params": {
#                 "binary": false
#             }
#         }

#         Response Example - format of all responses
#         {
#             "symbol": "BTCUSDT",
#             "symbolName": "BTCUSDT",
#             "topic": "depth",
#             "params": {
#                 "realtimeInterval": "24h",
#                 "binary": "false"
#             },
#             "data": [{
#                 "e": 301,
#                 "s": "BTCUSDT",
#                 "t": 1565600357643,
#                 "v": "112801745_18",
#                 "b": [
#                 ["11371.49", "0.0014"],
#                 ["11371.12", "0.2"],
#                 ["11369.97", "0.3523"],
#                 ["11369.96", "0.5"],
#                 ["11369.95", "0.0934"],
#                 ["11369.94", "1.6809"],
#                 ["11369.6", "0.0047"],
#                 ["11369.17", "0.3"],
#                 ["11369.16", "0.2"],
#                 ["11369.04", "1.3203"]]
#                 "a": [
#                 ["11375.41", "0.0053"],
#                 ["11375.42", "0.0043"],
#                 ["11375.48", "0.0052"],
#                 ["11375.58", "0.0541"],
#                 ["11375.7", "0.0386"],
#                 ["11375.71", "2"],
#                 ["11377", "2.0691"],
#                 ["11377.01", "0.0167"],
#                 ["11377.12", "1.5"],
#                 ["11377.61", "0.3"]
#                 ],
#                 "o": 0
#             }],
#             "f": true,
#             "sendTime": 1626253839401,
#             "shared": false
#         }

#         kline

#         How to Subscribe

#         {
#             "topic": "kline_1m",
#             "event": "sub",
#             "symbol": "BTCUSDT",
#             "params": {
#                 "binary": false
#             }
#         }

#         {
#         "symbol": "BTCUSDT",
#         "symbolName": "BTCUSDT",
#         "topic": "kline",
#         "params": {
#             "realtimeInterval": "24h",
#             "klineType": "15m",
#             "binary": "false"
#         },
#         "data": [{
#             "t": 1565595900000,
#             "s": "BTCUSDT",
#             "sn": "BTCUSDT",
#             "c": "11436.14",
#             "h": "11437",
#             "l": "11381.89",
#             "o": "11381.89",
#             "v": "16.3306"
#         }],
#         "f": true,
#         "sendTime": 1626252389284,
#         "shared": false
#         }
#         """
#         # logger.debug(f"on_packet event {packet}")
#         # print(f"on_packet event {packet} --------- ori --------")
#         if "topic" not in packet:
#             print(packet)
#             return

#         channel: str = packet["topic"]
#         # params: dict = packet['params']
#         data: dict = packet["data"][0]
#         symbol: str = packet["symbol"]

#         is_fire: bool = False
#         if channel == "kline":
#             interval = packet["params"]["klineType"]
#             tick: dict = self.ticks[f"{symbol}|{interval}"]

#             tick["volume"] = float(data["v"])
#             tick["open"] = float(data["o"])
#             tick["high"] = float(data["h"])
#             tick["low"] = float(data["l"])
#             tick["close"] = float(data["c"])
#             tick["turnover"] = tick["close"] * tick["volume"]
#             tick["start"] = BybitSpotClient.generate_datetime(float(data["t"] / 1000))
#             tick["datetime"] = models.generate_datetime()  # datetime.now()
#             if self.on_kline_callback:
#                 is_fire = True
#                 self.on_kline_callback(copy(tick))

#         elif channel == "realtimes":
#             tick: dict = self.ticks[f"{symbol}|ticker"]
#             tick["prev_open_24h"] = float(data["o"])
#             tick["prev_high_24h"] = float(data["h"])
#             tick["prev_low_24h"] = float(data["l"])
#             tick["prev_volume_24h"] = float(data["v"])
#             tick["prev_turnover_24h"] = float(data["qv"])

#             tick["last_price"] = float(data["c"])
#             tick["price_change_pct"] = float(data["m"])
#             tick["prev_close_24h"] = tick["last_price"] / (1 + tick["price_change_pct"])
#             tick["price_change"] = tick["last_price"] - tick["prev_close_24h"]

#             tick["datetime"] = models.generate_datetime()  # datetime.now()

#             if self.on_ticker_callback:
#                 is_fire = True
#                 self.on_ticker_callback(copy(tick))

#         elif channel == "depth":
#             tick: dict = self.ticks[f"{symbol}|depth"]
#             bids: list = data["b"]
#             for n in range(min(5, len(bids))):
#                 price, volume = bids[n]
#                 tick["bid_price_" + str(n + 1)] = float(price)
#                 tick["bid_volume_" + str(n + 1)] = float(volume)

#             asks: list = data["a"]
#             for n in range(min(5, len(asks))):
#                 price, volume = asks[n]
#                 tick["ask_price_" + str(n + 1)] = float(price)
#                 tick["ask_volume_" + str(n + 1)] = float(volume)
#             tick["datetime"] = models.generate_datetime()  # datetime.now()

#             if self.on_depth_callback:
#                 is_fire = True
#                 self.on_depth_callback(copy(tick))

#         if not is_fire and self.on_tick:
#             self.on_tick(copy(tick))


# class BybitSpotTradeWebsocket(WebsocketClient):
#     """
#     docs: https://bybit-exchange.github.io/docs/spot/#t-privatetopics
#     """

#     def __init__(
#         self,
#         on_account_callback=None,
#         on_order_callback=None,
#         on_trade_callback=None,
#         on_connected_callback: Callable = None,
#         on_disconnected_callback: Callable = None,
#         on_error_callback: Callable = None,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         is_testnet: bool = False,
#         ping_interval: int = 20,
#         debug: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         self.on_account_callback = on_account_callback
#         self.on_order_callback = on_order_callback
#         self.on_trade_callback = on_trade_callback
#         self.on_connected_callback = on_connected_callback
#         self.on_disconnected_callback = on_disconnected_callback
#         self.on_error_callback = on_error_callback

#         self.ping_interval = ping_interval
#         self.is_testnet = is_testnet
#         self.proxy_host = proxy_host
#         self.proxy_port = proxy_port
#         self._logged_in = False
#         self.key = None
#         self.secret = None
#         # self.channel = "orders"
#         self.debug = debug

#     def info(self):
#         return "Bybit Spot Trade Websocket Start"

#     def connect(self, key: str, secret: str):
#         """
#         Give api key and secret for Authorization.
#         """
#         self.key = key
#         self.secret = secret

#         self.init(
#             host=BYBIT_TESTNET_WS_SPOT_PRIVATE_HOST
#             if self.is_testnet
#             else BYBIT_WS_SPOT_PRIVATE_HOST,
#             proxy_host=self.proxy_host,
#             proxy_port=self.proxy_port,
#             ping_interval=self.ping_interval,
#         )
#         self.start()

#         self.waitfor_connection()

#         if not self._logged_in:
#             self._login()
#         # logger.info(self.info())
#         print(self.info())
#         return True

#     def _login(self):
#         """
#         Authorize websocket connection.
#         """

#         # Generate expires.
#         expires = int((time.time() + 10) * 1000)
#         api_key = self.key
#         api_secret = self.secret

#         if api_key is None or api_secret is None:
#             raise PermissionError("Authenticated endpoints require keys.")

#         # Generate signature.
#         _val = f"GET/realtime{expires}"
#         signature = str(
#             hmac.new(
#                 bytes(api_secret, "utf-8"), bytes(_val, "utf-8"), digestmod="sha256"
#             ).hexdigest()
#         )

#         req: dict = {"op": "auth", "args": [api_key, expires, signature]}
#         self.send_packet(req)
#         self._logged_in = True

#     def on_disconnected(self) -> None:
#         print(
#             "Bybit Spot user trade data Websocket disconnected, now try to connect @ {datetime.now()}"
#         )
#         self._logged_in = False
#         if self.on_disconnected_callback:
#             self.on_disconnected_callback()

#     def on_connected(self) -> None:
#         if self.waitfor_connection():
#             print(f"Bybit spot trade data websocket connect success ({datetime.now()})")
#         else:
#             print(f"Bybit spot trade data websocket connect fail ({datetime.now()})")

#         if not self._logged_in:
#             self._login()

#         if self.on_connected_callback:
#             self.on_connected_callback()

#     def on_error(self, exception_type, exception_value, traceback_obj):
#         super().on_error(exception_type, exception_value, traceback_obj)
#         if self.on_error_callback:
#             self.on_error_callback(exception_value)

#     def inject_callback(self, channel: str, callbackfn: Callable):
#         if channel == "account":
#             self.on_account_callback = callbackfn
#         elif channel == "orders":
#             self.on_order_callback = callbackfn
#         elif channel == "trades":
#             self.on_trade_callback = callbackfn
#         elif channel == "connect":
#             self.on_connected_callback = callbackfn
#         elif channel == "disconnect":
#             self.on_disconnected_callback = callbackfn
#         elif channel == "error":
#             self.on_error_callback = callbackfn
#         else:
#             Exception("invalid callback function")

#     # def subscribe(self, channel:str=None) -> None:
#     #     """
#     #         keep the interface:
#     #         channel: 'trades', 'orders', 'account'
#     #         mapping: 'ticketInfo', 'executionReport', 'outboundAccountInfo'
#     #     """

#     #     if channel and not (channel in ['trades', 'orders', 'account']):
#     #         Exception("invalid subscription")
#     #         return

#     #     if not self._logged_in:
#     #         self._login()

#     #     # self.channel = channel
#     #     # NOTE: Sending the authentication message automatically subscribes you to all 3 private topics

#     def on_packet(self, packet: dict) -> None:
#         # logger.debug(f"on_packet event {packet}")
#         if self.debug:
#             print(f"[DEBUG] Bybit Spot Trade Websocket - on_packet event {packet}")
#         if "auth" in packet:
#             self._logged_in = True if packet["auth"] == "success" else False
#             if self._logged_in:
#                 print(f"Bybit Spot Trade Websocket - successfully logined - {packet}")
#             else:
#                 print(
#                     f"Bybit Spot Trade Websocket auth failed - on_packet event {packet}"
#                 )
#             return

#         if "ping" in packet:
#             logger.info(f"healthcheck message {packet}")
#             return

#         if not packet:
#             logger.debug(f"unknown packet event {packet}")
#             return
#         for packet_data in packet:
#             if packet_data["e"] == "outboundAccountInfo":
#                 self.on_account(packet_data)
#             elif packet_data["e"] == "executionReport":
#                 self.on_order(packet_data)
#             elif packet_data["e"] == "ticketInfo":
#                 self.on_trade(packet_data)
#             else:
#                 logger.info(f"the other packet type: {packet_data}")

#         # if packet[0]["e"] == "outboundAccountInfo":
#         #     self.on_account(packet[0])
#         # elif packet[0]["e"] == "executionReport":
#         #     self.on_order(packet[0])
#         # elif packet[0]["e"] == "ticketInfo":
#         #     self.on_trade(packet[0])
#         # else:
#         #     logger.info(f"the other packet type: {self.on_trade(packet[0])}")

#     def on_account(self, packet: dict) -> None:
#         """
#         Response Example - format of all responses

#         [
#             {
#                 "e":"outboundAccountInfo",
#                 "E":"1629969654753",
#                 "T":True,
#                 "W":True,
#                 "D":True,
#                 "B":[
#                     {
#                         "a":"BTC",
#                         "f":"10000000097.1982823144",
#                         "l":"0"
#                     }
#                 ]
#             }
#         ]
#         """
#         logger.debug(f"packet on_account: {packet}")
#         # print(f'packet on_account: {packet}')

#         for d in packet["B"]:
#             account = {
#                 "symbol": d["a"],
#                 "asset_type": ASSET_TYPE_SPOT,
#                 "balance": float(d["f"]) + float(d["l"]),
#                 "available": float(d["f"]),
#                 "frozen": float(d["l"]),
#                 "exchange": EXCHANGE,
#             }

#             if account["symbol"] and self.on_account_callback:
#                 self.on_account_callback(account)

#     def on_order(self, packet: dict) -> None:
#         """
#         Orders
#         [
#             {
#                 "e": "executionReport",
#                 "E": "1499405658658",
#                 "s": "ETHBTC",
#                 "c": "1000087761",
#                 "S": "BUY",
#                 "o": "LIMIT",
#                 "f": "GTC",
#                 "q": "1.00000000",
#                 "p": "0.10264410",
#                 "X": "NEW",
#                 "i": "4293153",
#                 "M": "0",
#                 "l": "0.00000000",
#                 "z": "0.00000000",
#                 "L": "0.00000000",
#                 "n": "0",
#                 "N": "BTC",
#                 "u": true,
#                 "w": true,
#                 "m": false,
#                 "O": "1499405658657",
#                 "Z": "473.199",
#                 "A": "0",
#                 "C": false,
#                 "v": "0"
#             }
#         ]
#         """
#         # filter non-support trade type
#         logger.debug(f"packet on_order: {packet}")

#         if packet["o"] not in ["LIMIT", "MARKET_OF_QUOTE", "MARKET_OF_BASE"]:
#             return

#         status = packet["X"]
#         if status == "FILLED":
#             print(f"on_order callback filled ignore: {packet}")
#             return
#             # trade = {
#             #     "symbol": packet["s"],
#             #     "asset_type": ASSET_TYPE_SPOT,
#             #     "name": models.normalize_name(packet["s"], ASSET_TYPE_SPOT),
#             #     "exchange": EXCHANGE,
#             #     "ori_order_id": packet["i"],
#             #     "order_id": packet["c"],
#             #     "side": packet["S"],
#             #     "price": float(packet["p"]),
#             #     "quantity": float(packet["q"]),
#             #     "commission": float(packet["n"]),
#             #     "commission_asset": packet["N"],
#             #     "datetime": BybitSpotClient.generate_datetime(float(packet["E"]) / 1000),
#             # }

#             # if self.on_trade_callback:
#             #     self.on_trade_callback(trade)

#         else:
#             order = {
#                 "symbol": packet["s"],
#                 "asset_type": ASSET_TYPE_SPOT,
#                 "name": models.normalize_name(packet["s"], ASSET_TYPE_SPOT),
#                 "exchange": EXCHANGE,
#                 "ori_order_id": packet["i"],
#                 "order_id": packet["c"],
#                 "type": "MARKET"
#                 if "MARKET" in packet["o"]
#                 else packet["o"],  # LIMIT, MARKET
#                 "side": packet["S"],  # BUY, SELL
#                 "price": float(packet["p"]),
#                 "quantity": float(packet["q"]),
#                 "executed_quantity": float(packet["z"]),
#                 "status": status,  # NEW, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED, PENDING_CANCEL, PENDING_NEW
#                 "datetime": BybitSpotClient.generate_datetime(
#                     float(packet["O"]) / 1000
#                 ),
#             }

#             if self.on_order_callback:
#                 self.on_order_callback(order)

#     def on_trade(self, packet: dict) -> None:
#         """
#         trades
#         [
#             {
#                 "e":"ticketInfo",
#                 "E":"1621912542359",
#                 "s":"BTCUSDT",
#                 "q":"0.001639",
#                 "t":"1621912542314",
#                 "p":"61000.0",
#                 "T":"899062000267837441",
#                 "o":"899048013515737344",
#                 "c":"1621910874883",
#                 "O":"899062000118679808",
#                 "a":"10043",
#                 "A":"10024",
#                 "m":true
#             }
#         ]
#         """
#         # filter not support trade type
#         logger.debug(f"packet on_trade: {packet}")

#         # Is MAKER. true=maker, false=taker
#         # side = "Buy" if packet['m'] else "Sell"
#         trade = {
#             "symbol": packet["s"],
#             "asset_type": ASSET_TYPE_SPOT,
#             "name": models.normalize_name(packet["s"], ASSET_TYPE_SPOT),
#             "exchange": EXCHANGE,
#             "ori_order_id": packet["o"],
#             "order_id": packet["c"],
#             "side": packet["S"],
#             "price": float(packet["p"]),
#             "quantity": float(packet["q"]),
#             "commission": None,
#             "commission_asset": None,
#             "datetime": BybitSpotClient.generate_datetime(float(packet["E"]) / 1000),
#         }

#         if self.on_trade_callback:
#             self.on_trade_callback(trade)


# class BybitFutureDataWebsocket(WebsocketClient):
#     """
#     Implement public topics
#     doc: https://bybit-exchange.github.io/docs/linear/#t-publictopics
#     """

#     def __init__(
#         self,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         ping_interval: int = 20,
#         is_testnet: bool = False,
#         debug: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__()

#         self.is_testnet: bool = is_testnet
#         self.proxy_host: str = proxy_host
#         self.proxy_port: int = proxy_port
#         self.ping_interval: int = ping_interval

#         self.ticks: dict[str, dict] = defaultdict(dict)
#         self.reqid: int = 0
#         # self.init(host=BYBIT_WS_FUTURE_PUBLIC_HOST, proxy_host=proxy_host, proxy_port=proxy_port, ping_interval=self.ping_interval1)
#         # self.start()
#         self._logged_in = False
#         self.key = None
#         self.secret = None
#         # self.channel = 'kline'
#         self.kbar_time: int = int(time.time()) - (int(time.time()) % 60)
#         # self.bids_asks_data: list = []
#         self.bids_asks_data: dict[str, dict] = defaultdict(dict)
#         self.ticker_data: dict[str, dict] = defaultdict(dict)

#         self.on_tick = None
#         self.on_ticker_callback = None
#         self.on_depth_callback = None
#         self.on_kline_callback = None
#         self.on_connected_callback = None
#         self.on_disconnected_callback = None
#         self.on_error_callback = None

#         self.debug = debug

#         # self.waitfor_connection()

#     def info(self):
#         return "Bybit Future Data Websocket start"

#     def connect(self):
#         """ """
#         self.init(
#             host=BYBIT_TESTNET_WS_FUTURE_PUBLIC_HOST
#             if self.is_testnet
#             else BYBIT_WS_FUTURE_PUBLIC_HOST,
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
#             print(
#                 f"Bybit Future Market data websocket connect success ({datetime.now()})"
#             )
#         else:
#             print(f"Bybit Future Market data websocket connect fail ({datetime.now()})")

#         print(
#             f"===> debug Bybit Future Market data on_connected - self.ticks: {self.ticks} | on_ticker_callback:{self.on_ticker_callback}; on_depth_callback:{self.on_depth_callback}; on_kline_callback:{self.on_kline_callback}"
#         )

#         # resubscribe
#         if self.ticks:
#             # for symbol in self.ticks.keys():
#             #     req: dict = self.sub_stream(symbol=symbol, channel=self.channel)
#             #     logger.debug(req)
#             #     self.send_packet(req)
#             channels = []
#             for key, detail in self.ticks.items():
#                 req: dict = self.sub_stream(
#                     symbol=detail["symbol"],
#                     channel=detail["channel"],
#                     interval=detail.get("interval"),
#                 )
#                 channels += req["args"]
#                 # logger.debug(req)
#                 # print(f"!!!!! rea: {req}")
#                 # self.send_packet(req)
#             req: dict = {"op": "subscribe", "args": channels}
#             print(f"on_connected, resubscribing the channels - req:{req}")
#             self.send_packet(req)

#         if self.on_connected_callback:
#             self.on_connected_callback()

#     def on_disconnected(self) -> None:
#         print(
#             "Bybit Future market data Websocket disconnected, now try to connect @ {datetime.now()}"
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
#         https://bybit-exchange.github.io/docs/linear/#t-websocketkline

#         "depth": "orderBookL2_25",
#         "kline": "candle",
#         """
#         self.reqid += 1
#         mapping_channel = FUTURE_MAPPING_CHANNEL[channel]
#         sub = {"op": "subscribe"}

#         if channel == "kline":
#             sub["args"] = [
#                 f"{mapping_channel}.{FUTURE_KBAR_INTERVAL[interval]}.{symbol}"
#             ]
#         elif channel == "depth":
#             sub["args"] = [f"{mapping_channel}.{symbol}"]
#         elif channel == "ticker":
#             sub["args"] = [f"{mapping_channel}.100ms.{symbol}"]

#         # print(self.reqid,'!!!!!!!!!!', symbol, '------',channel)
#         return sub

#     def subscribe(
#         self,
#         symbols: list | str,
#         on_tick: Callable = None,
#         channel="depth",
#         interval="1m",
#     ) -> None:
#         """
#         channel: 'depth', 'kline'
#         """
#         if channel not in ["depth", "kline", "ticker", "trades", "orders", "account"]:
#             Exception("invalid subscription")
#             return

#         # self.channel = channel
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

#         for symbol in symbols:
#             if symbol in self.ticks:
#                 return

#             # tick data dict
#             asset_type = models.future_asset_type(symbol, "future")
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
#             self.send_packet(req)

#     def on_packet(self, packet: dict) -> None:
#         """on_packet
#         docs: https://bybit-exchange.github.io/docs/linear/#t-websocketkline
#         channel: depth, kline
#         mapping: orderBookL2_25, candle

#         depth

#         How to Subscribe

#         ws.send('{"op": "subscribe", "args": ["orderBookL2_25.BTCUSDT"]}');
#         Snapshot Response Example - format of the first response

#         {
#             "topic": "orderBookL2_25.BTCUSDT",
#             "type": "snapshot",
#             "data": {
#                 "order_book":[
#                     {
#                         "price": "2999.00",
#                         "symbol": "BTCUSDT",
#                         "id": 29990000,
#                         "side": "Buy",
#                         "size": 9
#                     },
#                     {
#                         "price": "3001.00",
#                         "symbol": "BTCUSDT",
#                         "id": 30010000,
#                         "side": "Sell",
#                         "size": 10
#                     }
#                 ]
#             },
#             "cross_seq": 11518,
#             "timestamp_e6": 1555647164875373
#         }
#         Delta Response Example - format of the responses following the snapshot response

#         {
#             "topic": "orderBookL2_25.BTCUSDT",
#             "type": "delta",
#             "data": {
#                 "delete": [
#                     {
#                         "price": "3001.00",
#                         "symbol": "BTCUSDT",
#                         "id": 30010000,
#                         "side": "Sell"
#                     }
#                 ],
#                 "update": [
#                     {
#                         "price": "2999.00",
#                         "symbol": "BTCUSDT",
#                         "id": 29990000,
#                         "side": "Buy",
#                         "size": 8
#                     }
#                 ],
#                 "insert": [
#                     {
#                         "price": "2998.00",
#                         "symbol": "BTCUSDT",
#                         "id": 29980000,
#                         "side": "Buy",
#                         "size": 8
#                     }
#                 ],
#                 "transactTimeE6": 0
#             },
#             "cross_seq": 11519,
#             "timestamp_e6": 1555647221331673
#         }

#         kline

#         How to Subscribe

#         ws.send('{"op":"subscribe","args":["candle.1.BTCUSDT"]}')
#         Response Example - format of all responses

#         {
#             "topic":"candle.1.BTCUSDT",
#             "data":[
#                 {
#                     "start":1588071660,
#                     "end":1588071720,
#                     "open":7744.5,
#                     "close":7745,
#                     "high":7745,
#                     "low":7744,
#                     "volume":"33.051",
#                     "turnover":"255979.995",
#                     "confirm":true,
#                     "cross_seq":71947372,
#                     "timestamp":1588071717325846
#                 }
#             ],
#             "timestamp_e6":1588071721163975
#         }
#         """
#         # logger.debug(f'packet: {packet}')
#         if self.debug:
#             print(f"packet: {packet}")

#         if "topic" not in packet:
#             print(f"the other packet without topic: {packet}")
#             return

#         channel: str = packet["topic"].split(".")[0]
#         symbol: str = packet["topic"].split(".")[-1]
#         # params: dict = packet['params']
#         # tick: dict = self.ticks[symbol]
#         # print(packet, tick, symbol, channel,'~~~')

#         is_fire: bool = False
#         # kline = candle
#         if channel == "candle":
#             interval = FUTURE_KBAR_INTERVAL_REV[packet["topic"].split(".")[1]]
#             tick: dict = self.ticks[f"{symbol}|{interval}"]

#             data: dict = packet["data"][0]
#             tick["volume"] = float(data["volume"])
#             tick["turnover"] = float(data["turnover"])
#             tick["open"] = float(data["open"])
#             tick["high"] = float(data["high"])
#             tick["low"] = float(data["low"])
#             tick["close"] = float(data["close"])
#             # tick['start'] = BybitSpotClient.generate_datetime(float(data['end']))
#             tick["start"] = BybitSpotClient.generate_datetime(float(data["start"]))
#             tick["datetime"] = models.generate_datetime()  # datetime.now()

#             # tick['asset_type'] = models.future_asset_type(symbol, "future")
#             # tick['name'] = models.normalize_name(symbol, tick['asset_type'])
#             # tick['exchange'] = EXCHANGE
#             if self.on_kline_callback:
#                 is_fire = True
#                 self.on_kline_callback(copy(tick))

#         elif channel == "instrument_info":
#             tick: dict = self.ticks[f"{symbol}|ticker"]
#             tick.update(self.handle_tickerdata(packet, symbol=symbol))

#             if self.on_ticker_callback:
#                 is_fire = True
#                 self.on_ticker_callback(copy(tick))

#         # depth = orderBookL2_25
#         elif channel == "orderBookL2_25":
#             # save asks / bids in tick for next iteration
#             tick: dict = self.ticks[f"{symbol}|depth"]
#             tick.update(self.handle_orderbook(packet, symbol=symbol))

#             if self.on_depth_callback:
#                 is_fire = True
#                 self.on_depth_callback(copy(tick))

#         if not is_fire and self.on_tick:
#             self.on_tick(copy(tick))

#     def handle_tickerdata(self, packet, symbol):
#         if "snapshot" in packet["type"]:
#             self.ticker_data[symbol] = packet["data"]

#         elif "delta" in packet["type"]:
#             for key, value in packet["data"]["update"][0].items():
#                 self.ticker_data[symbol][key] = value

#         data = copy(self.ticker_data[symbol])
#         result = {}
#         result["prev_open_24h"] = None
#         result["prev_high_24h"] = float(data["high_price_24h"])
#         result["prev_low_24h"] = float(data["low_price_24h"])
#         result["prev_close_24h"] = float(data["prev_price_24h"])
#         result["prev_volume_24h"] = float(data["volume_24h_e8"]) / 10**8
#         result["prev_turnover_24h"] = float(data["turnover_24h_e8"]) / 10**8

#         result["last_price"] = float(data["last_price"])
#         result["price_change_pct"] = float(data["price_24h_pcnt_e6"]) / 10**6
#         result["price_change"] = result["last_price"] - result["prev_close_24h"]

#         result["datetime"] = models.generate_datetime()  # datetime.now()

#         return result

#     def handle_orderbook(self, packet, symbol):
#         def bids_asks_process(
#             datas, delete_data, update_data, insert_data, is_bit: bool = False
#         ):
#             remove_list = list()
#             add_list = list()

#             for i, data in enumerate(datas):
#                 # delete data
#                 if delete_data:
#                     # for i in range(len(delete_data)):
#                     #     if data['id'] == delete_data[i]['id']:
#                     #         remove_list.append(i)
#                     for entry in delete_data:
#                         if data["id"] == entry["id"]:
#                             remove_list.append(i)

#                 # update new data
#                 if update_data:
#                     # for i in range(len(update_data)):
#                     #     if data['id'] == update_data[i]['id']:
#                     #         remove_list.append(i)
#                     #         add_list.append(update_data[i])
#                     for entry in update_data:
#                         if data["id"] == entry["id"]:
#                             remove_list.append(i)
#                             add_list.append(entry)

#             if remove_list:
#                 # pop from decending list
#                 remove_list.sort(reverse=True)
#                 # print(datas,'?????', remove_list)

#                 [datas.pop(i) for i in remove_list]
#                 # try:
#                 #     for i in remove_list:
#                 #         datas.pop(i)
#                 # except Exception as e:
#                 #     print(str(e), i, datas, len(datas), delete_data, update_data)

#             if add_list:
#                 datas.extend(add_list)
#             # incert new data
#             if insert_data:
#                 datas.extend(insert_data)

#             datas.sort(key=operator.itemgetter("price"), reverse=is_bit)
#             return datas

#         def bids_asks_category(data, category):
#             bids = [d for d in data[category] if d["side"] == "Buy"]
#             asks = [d for d in data[category] if d["side"] == "Sell"]
#             return bids, asks

#         # call by reference
#         data: dict = packet[
#             "data"
#         ]  # The data is ordered by price, starting with the lowest buys and ending with the highest sells.

#         if packet["type"] == "snapshot":
#             self.bids_asks_data[symbol]["bids"] = [
#                 d for d in data["order_book"] if d["side"] == "Buy"
#             ]
#             self.bids_asks_data[symbol]["asks"] = [
#                 d for d in data["order_book"] if d["side"] == "Sell"
#             ]
#         elif packet["type"] == "delta":
#             bids_delete_data, asks_delete_data = bids_asks_category(
#                 data=data, category="delete"
#             )
#             bids_update_data, asks_update_data = bids_asks_category(
#                 data=data, category="update"
#             )
#             bids_insert_data, asks_insert_data = bids_asks_category(
#                 data=data, category="insert"
#             )

#             tmp_bids = copy(self.bids_asks_data[symbol]["bids"])
#             self.bids_asks_data[symbol]["bids"] = bids_asks_process(
#                 datas=tmp_bids,
#                 delete_data=bids_delete_data,
#                 update_data=bids_update_data,
#                 insert_data=bids_insert_data,
#             )

#             tmp_asks = copy(self.bids_asks_data[symbol]["asks"])
#             self.bids_asks_data[symbol]["asks"] = bids_asks_process(
#                 datas=tmp_asks,
#                 delete_data=asks_delete_data,
#                 update_data=asks_update_data,
#                 insert_data=asks_insert_data,
#             )
#         # print('len of asks', len(self.bids_asks_data['asks']),'len of bids', len(self.bids_asks_data[symbol]['bids']))
#         # computer the bids and asks with current data
#         bids: list = self.bids_asks_data[symbol]["bids"]
#         for n in range(min(5, len(bids))):
#             price = bids[n]["price"]
#             volume = bids[n]["size"]
#             self.bids_asks_data[symbol]["bid_price_" + str(n + 1)] = float(price)
#             self.bids_asks_data[symbol]["bid_volume_" + str(n + 1)] = float(volume)

#         asks: list = self.bids_asks_data[symbol]["asks"]
#         for n in range(min(5, len(asks))):
#             price = asks[n]["price"]
#             volume = asks[n]["size"]
#             self.bids_asks_data[symbol]["ask_price_" + str(n + 1)] = float(price)
#             self.bids_asks_data[symbol]["ask_volume_" + str(n + 1)] = float(volume)

#         self.bids_asks_data[symbol]["datetime"] = (
#             models.generate_datetime()
#         )  # datetime.now()

#         result = copy(self.bids_asks_data[symbol])
#         result.pop("bids")
#         result.pop("asks")
#         return result


# class BybitFutureTradeWebsocket(WebsocketClient):
#     """
#     Implement private topics
#     doc: https://bybit-exchange.github.io/docs/linear/#t-privatetopics
#     """

#     DATA_TYPE = "linear"

#     def __init__(
#         self,
#         on_account_callback: Callable = None,
#         on_order_callback: Callable = None,
#         on_trade_callback: Callable = None,
#         on_position_callback: Callable = None,
#         on_connected_callback: Callable = None,
#         on_disconnected_callback: Callable = None,
#         on_error_callback: Callable = None,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         is_testnet: bool = False,
#         ping_interval: int = 20,
#         debug: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         self.on_account_callback = on_account_callback
#         self.on_order_callback = on_order_callback
#         self.on_trade_callback = on_trade_callback
#         self.on_position_callback = on_position_callback
#         self.on_connected_callback = on_connected_callback
#         self.on_disconnected_callback = on_disconnected_callback
#         self.on_error_callback = on_error_callback

#         self.ping_interval = ping_interval
#         self.is_testnet = is_testnet
#         self.proxy_host = proxy_host
#         self.proxy_port = proxy_port
#         self._logged_in = False
#         self.key = None
#         self.secret = None
#         # self.channel = "orders"
#         self.debug = debug

#     def info(self):
#         return "Bybit Future Trade Websocket Start"

#     def connect(self, key: str, secret: str):
#         """
#         Give api key and secret for Authorization.
#         """
#         self.key = key
#         self.secret = secret

#         self.init(
#             host=BYBIT_TESTNET_WS_FUTURE_PRIVATE_HOST
#             if self.is_testnet
#             else BYBIT_WS_FUTURE_PRIVATE_HOST,
#             proxy_host=self.proxy_host,
#             proxy_port=self.proxy_port,
#             ping_interval=self.ping_interval,
#         )
#         self.start()

#         self.waitfor_connection()

#         if not self._logged_in:
#             self._login()
#         logger.info(self.info())
#         return True

#     def _login(self):
#         """
#         Authorize websocket connection.
#         """

#         # Generate expires.
#         expires = int((time.time() + 1) * 1000)
#         api_key = self.key
#         api_secret = self.secret

#         if api_key is None or api_secret is None:
#             raise PermissionError(
#                 f"Authenticated endpoints require keys. api_key:{api_key}; api_secret:{api_secret}"
#             )

#         # Generate signature.
#         _val = f"GET/realtime{expires}"
#         signature = str(
#             hmac.new(
#                 bytes(api_secret, "utf-8"), bytes(_val, "utf-8"), digestmod="sha256"
#             ).hexdigest()
#         )

#         req: dict = {"op": "auth", "args": [api_key, expires, signature]}
#         self.send_packet(req)
#         print(f"auth req:{req}")
#         self._logged_in = True

#     def on_disconnected(self) -> None:
#         print(
#             "Bybit Future user trade data Websocket disconnected, now try to connect @ {datetime.now()}"
#         )
#         self._logged_in = False
#         if self.on_disconnected_callback:
#             self.on_disconnected_callback()

#     def on_connected(self) -> None:
#         if self.waitfor_connection():
#             print(
#                 f"Bybit Future user trade data websocket connect success ({datetime.now()})"
#             )
#         else:
#             print(
#                 f"Bybit Future user trade data websocket connect fail ({datetime.now()})"
#             )

#         if not self._logged_in:
#             self._login()
#             # self._logged_in = True
#         # logger.debug("Bybit Future user trade data Websocket connect success")

#         req: dict = self.sub_stream(channel=["trades", "orders", "account", "position"])
#         print(f"Bybit Future on_connected req:{req} - self._active:{self._active}")
#         self.send_packet(req)
#         if self.on_connected_callback:
#             self.on_connected_callback()

#     def on_error(self, exception_type, exception_value, traceback_obj):
#         super().on_error(exception_type, exception_value, traceback_obj)
#         if self.on_error_callback:
#             self.on_error_callback(exception_value)

#     def inject_callback(self, channel: str, callbackfn: Callable):
#         if channel == "account":
#             self.on_account_callback = callbackfn
#         elif channel == "orders":
#             self.on_order_callback = callbackfn
#         elif channel == "trades":
#             self.on_trade_callback = callbackfn
#         elif channel == "position":
#             self.on_position_callback = callbackfn
#         elif channel == "connect":
#             self.on_connected_callback = callbackfn
#         elif channel == "disconnect":
#             self.on_disconnected_callback = callbackfn
#         elif channel == "error":
#             self.on_error_callback = callbackfn
#         else:
#             Exception("invalid callback function")

#     def sub_stream(self, channel="orders"):
#         """ """
#         sub = {"op": "subscribe"}
#         if isinstance(channel, list):
#             sub["args"] = [FUTURE_MAPPING_CHANNEL[c] for c in channel]
#         else:
#             mapping_channel = FUTURE_MAPPING_CHANNEL[channel]
#             sub["args"] = [f"{mapping_channel}"]

#         return sub

#     def subscribe(self, channel: str = None) -> None:
#         """
#         keep the interface:
#         channel: 'trades', 'orders', 'account'
#         mapping: 'execution', 'order', 'wallet'
#         """
#         if channel and channel not in ["trades", "orders", "account", "position"]:
#             Exception("invalid subscription")
#             return

#         if not self._logged_in:
#             self._login()

#         ### Default sub all private channels if not specified
#         self.channel = channel
#         req: dict = self.sub_stream(channel=channel)
#         print(req)
#         self.send_packet(req)

#     def on_packet(self, packet: dict) -> None:
#         if self.debug:
#             print(f"[DEBUG] Bybit Future Trade Websocket - on_packet event {packet}")
#         # if 'auth' in packet:
#         #     self._logged_in = True if packet['auth'] == "success" else False
#         #     if self._logged_in:
#         #         print(f"Bybit Future Trade Websocket - successfully logined - {packet}")
#         #     else:
#         #         print(f"Bybit Future Trade Websocket auth failed - on_packet event {packet}")
#         #     return

#         if "topic" not in packet:
#             logger.info(f"the other packet without topic: {packet}")
#             return

#         if "ping" in packet:
#             logger.info(f"healthcheck message {packet}")
#             return

#         if not packet:
#             logger.debug(f"unknown packet event {packet}")
#             return

#         if packet["topic"] == "wallet":
#             self.on_account(packet)
#         elif packet["topic"] == "order":
#             self.on_order(packet)
#         elif packet["topic"] == "execution":
#             self.on_trade(packet)
#         elif packet["topic"] == "position":
#             self.on_position(packet)
#         else:
#             logger.info(f"the other packet type: {packet}")

#     def on_account(self, packet: dict) -> None:
#         """
#         How to Subscribe

#         ws.send('{"op": "subscribe", "args": ["wallet"]}')
#         Response Example - format of all responses

#         {
#             "topic": "wallet",
#             "data": [
#                 {
#                     "wallet_balance":429.80713,
#                     "available_balance":429.67322
#                 }
#             ]
#         }
#         """
#         logger.debug(f"packet on_account: {packet}")

#         for d in packet["data"]:
#             account = {
#                 "symbol": "USDT",
#                 "balance": float(d["wallet_balance"]),
#                 "available": float(d["available_balance"]),
#                 "frozen": float(d["wallet_balance"]) - float(d["available_balance"]),
#                 "exchange": EXCHANGE,
#                 "asset_type": self.DATA_TYPE.upper(),
#             }

#             if account["symbol"] and self.on_account_callback:
#                 self.on_account_callback(account)

#     def on_order(self, packet: dict) -> None:
#         """
#             How to Subscribe

#             ws.send('{"op": "subscribe", "args": ["order"]}')
#             Response Example - format of all responses

#             {
#                 "topic": "order",
#                 "action": "",
#                 "data": [
#                     {
#                         "order_id": "xxxxxxxx-xxxx-xxxx-9a8f-4a973eb5c418",
#                         "order_link_id": "",
#                         "symbol": "BTCUSDT",
#                         "side": "Buy",
#                         "order_type": "Limit",
#                         "price": 11000,
#                         "qty": 0.001,
#                         "leaves_qty": 0.001,
#                         "last_exec_price": 0,
#                         "cum_exec_qty": 0,
#                         "cum_exec_value": 0,
#                         "cum_exec_fee": 0,
#                         "time_in_force": "GoodTillCancel",
#                         "create_type": "CreateByUser",
#                         "cancel_type": "UNKNOWN",
#                         "order_status": "New",
#                         "take_profit": 0,
#                         "stop_loss": 0,
#                         "trailing_stop": 0,
#                         "reduce_only": false,
#                         "close_on_trigger": false,
#                         "position_idx": 1,
#                         "create_time": "2020-08-12T21:18:40.780039678Z",
#                         "update_time": "2020-08-12T21:18:40.787986415Z"
#                     }
#                 ]
#             }

#         {
#             "asset_type": self.asset_type,
#             "symbol": self.symbol,
#             "name": self.name,
#             "ori_order_id": str(self.id),
#             "order_id": str(self.client_id),
#             "datetime": self.create_at,
#             "exchange": self.exchange,
#             "price": self.price,
#             "quantity": self.size,
#             "type": self.type,
#             "side": self.side,
#             "status": self.status,
#             "executed_quantity": self.filled_size}
#         """
#         logger.debug(f"packet on_order: {packet}")
#         datas: dict = packet["data"]

#         # if datas["order_type"] not in ['LIMIT', 'Market']:
#         #     return

#         order_list = list()
#         for data in datas:
#             order = models.OrderSchema()  # .model_construct(data)
#             order = order.from_bybit_to_form(data, datatype=self.DATA_TYPE)

#             order_list.append(order)

#             if self.on_order_callback:  # and order_list
#                 # self.on_order_callback(order_list)
#                 self.on_order_callback(order)

#     def on_trade(self, packet: dict) -> None:
#         """
#             How to Subscribe

#             ws.send('{"op": "subscribe", "args": ["execution"]}')
#             Response Example - format of all responses

#             {
#                 "topic": "execution",
#                 "data": [
#                     {
#                         "symbol": "BTCUSDT",
#                         "side": "Sell",
#                         "order_id": "xxxxxxxx-xxxx-xxxx-9a8f-4a973eb5c418",
#                         "exec_id": "xxxxxxxx-xxxx-xxxx-8b66-c3d2fcd352f6",
#                         "order_link_id": "",
#                         "price": 11527.5,
#                         "order_qty": 0.001,
#                         "exec_type": "Trade",
#                         "exec_qty": 0.001,
#                         "exec_fee": 0.00864563,
#                         "leaves_qty": 0,
#                         "is_maker": false,
#                         "trade_time": "2020-08-12T21:16:18.142746Z"
#                     }
#                 ]
#             }


#         {
#             "asset_type": self.asset_type,
#             "symbol": self.symbol,
#             "exchange": self.exchange,
#             "name": self.name,
#             "ori_order_id": self.ori_order_id,
#             "order_id": self.order_id,
#             "price": self.price,
#             "quantity": self.quantity,
#             "side": self.side,
#             "commission": self.commission,
#             "commission_asset": self.commission_asset,
#             "datetime": self.datetime}
#         """
#         logger.debug(f"packet on_trade: {packet}")
#         datas: dict = packet["data"]

#         trade_list = list()
#         for data in datas:
#             trade = models.TradeSchema()  # .model_construct(data)
#             trade = trade.from_bybit_to_form(data, datatype=self.DATA_TYPE)
#             trade_list.append(trade)
#             # print(trade,'....1')

#             if self.on_trade_callback:  # and trade_list:
#                 # self.on_trade_callback(trade_list)
#                 # print(trade,'....')
#                 self.on_trade_callback(trade)

#     def on_position(self, packet: dict) -> None:
#         """
#         {
#         "topic": "position",
#         "action": "update",
#         "data": [
#         {
#                     "user_id": "533285",
#                     "symbol": "BTCUSDT",
#                     "size": 0.01,
#                     "side": "Buy",
#                     "position_value": 202.195,
#                     "entry_price": 20219.5,
#                     "liq_price": 0.5,
#                     "bust_price": 0.5,
#                     "leverage": 99,
#                     "order_margin": 0,
#                     "position_margin": 1959.6383,
#                     "occ_closing_fee": 3e-06,
#                     "take_profit": 25000,
#                     "tp_trigger_by": "LastPrice",
#                     "stop_loss": 18000,
#                     "sl_trigger_by": "LastPrice",
#                     "trailing_stop": 0,
#                     "realised_pnl": -4.189762,
#                     "auto_add_margin": "0",
#                     "cum_realised_pnl": -13.640625,
#                     "position_status": "Normal",
#                     "position_id": "0",
#                     "position_seq": "92962",
#                     "adl_rank_indicator": "2",
#                     "free_qty": 0.01,
#                     "tp_sl_mode": "Full",
#                     "risk_id": "1",
#                     "isolated": false,
#                     "mode": "BothSide",
#                     "position_idx": "1"
#                 }
#         ]
#         }
#         """
#         logger.debug(f"packet on_position: {packet}")
#         # print(f"packet on_position: {packet}")
#         datas: dict = packet["data"]
#         position_list = list()
#         for data in datas:
#             # position_value = float(data["position_value"])
#             # if not position_value:
#             #     continue

#             position = models.PositionSchema()  # .model_construct(data)
#             position = position.from_bybit_to_form(data, datatype=self.DATA_TYPE)

#             position_list.append(position)

#             if self.on_position_callback:  # and position_list:
#                 # self.on_position_callback(position_list)
#                 self.on_position_callback(position)


# class BybitInverseFutureDataWebsocket(WebsocketClient):
#     """
#     Implement public topics
#     doc:
#     https://bybit-exchange.github.io/docs/inverse/#t-introduction
#     https://bybit-exchange.github.io/docs/inverse_futures/#t-websocket
#     """

#     def __init__(
#         self,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         ping_interval: int = 29,
#         is_testnet: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__()

#         self.is_testnet: bool = is_testnet
#         self.proxy_host: str = proxy_host
#         self.proxy_port: int = proxy_port
#         self.ping_interval: int = ping_interval

#         self.ticks: dict[str, dict] = defaultdict(dict)
#         self.reqid: int = 0
#         # self.init(host=BYBIT_WS_INVERSE_HOST, proxy_host=proxy_host, proxy_port=proxy_port, ping_interval=29)
#         # self.start()
#         self._logged_in = False
#         self.key = None
#         self.secret = None
#         # self.channel = 'klineV2'
#         self.kbar_time: int = int(time.time()) - (int(time.time()) % 60)
#         # self.bids_asks_data: list = []
#         self.bids_asks_data: dict[str, dict] = defaultdict(dict)
#         self.ticker_data: dict[str, dict] = defaultdict(dict)

#         self.on_tick = None
#         self.on_ticker_callback = None
#         self.on_depth_callback = None
#         self.on_kline_callback = None
#         self.on_connected_callback = None
#         self.on_disconnected_callback = None
#         self.on_error_callback = None
#         # self.waitfor_connection()

#     def info(self):
#         return "Bybit Inverse Future Data Websocket start"

#     def connect(self):
#         """ """
#         self.init(
#             host=BYBIT_TESTNET_WS_INVERSE_HOST
#             if self.is_testnet
#             else BYBIT_WS_INVERSE_HOST,
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
#             print(
#                 f"Bybit Inverse Future market data websocket connect success ({datetime.now()})"
#             )
#         else:
#             print(
#                 f"Bybit Inverse Future market data websocket connect fail ({datetime.now()})"
#             )

#         # resubscribe
#         print(
#             f"===> debug Bybit Inverse Future Market data on_connected - self.ticks: {self.ticks} | on_ticker_callback:{self.on_ticker_callback}; on_depth_callback:{self.on_depth_callback}; on_kline_callback:{self.on_kline_callback}"
#         )
#         if self.ticks:
#             # for symbol in self.ticks.keys():
#             #     req: dict = self.sub_stream(symbol=symbol, channel=self.channel)
#             #     logger.debug(req)
#             #     self.send_packet(req)
#             channels = []
#             for key, detail in self.ticks.items():
#                 req: dict = self.sub_stream(
#                     symbol=detail["symbol"],
#                     channel=detail["channel"],
#                     interval=detail.get("interval", "1m"),
#                 )
#                 channels += req["args"]
#                 # logger.debug(req)
#                 # print(f"!!!!! rea: {req}")
#                 # self.send_packet(req)
#             req: dict = {"op": "subscribe", "args": channels}
#             print(f"on_connected, resubscribing the channels - req:{req}")
#             self.send_packet(req)

#     def on_disconnected(self) -> None:
#         print(
#             "Bybit Inverse Future market data Websocket disconnected, now try to connect @ {datetime.now()}"
#         )

#     def on_error(self, exception_type, exception_value, traceback_obj):
#         super().on_error(exception_type, exception_value, traceback_obj)
#         if self.on_error_callback:
#             self.on_error_callback(exception_value)

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
#         https://bybit-exchange.github.io/docs/linear/#t-websocketkline

#         "depth": "orderBookL2_25",
#         "kline": "klineV2",
#         """
#         self.reqid += 1
#         mapping_channel = INVERSE_FUTURE_MAPPING_CHANNEL[channel]
#         sub = {"op": "subscribe"}

#         if channel == "kline":
#             sub["args"] = [
#                 f"{mapping_channel}.{INVERSE_FUTURE_KBAR_INTERVAL[interval]}.{symbol}"
#             ]
#         elif channel == "depth":
#             sub["args"] = [f"{mapping_channel}.{symbol}"]
#         elif channel == "ticker":
#             sub["args"] = [f"{mapping_channel}.100ms.{symbol}"]

#         # print(sub,self.reqid,'!!!!!!!!!!', symbol, '------',channel, self.is_connected)
#         return sub

#     def subscribe(self, symbols, on_tick=None, channel="depth", interval="1m") -> None:
#         """
#         channel: "depth", "kline", "ticker"
#         """
#         if channel not in ["depth", "kline", "ticker", "trades", "orders", "account"]:
#             Exception("invalid subscription")
#             return

#         # self.channel = channel
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

#         for symbol in symbols:
#             if symbol in self.ticks:
#                 return

#             # tick data dict
#             asset_type = models.future_asset_type(symbol, "inverse")
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
#             # print(f"req: {req}")
#             self.send_packet(req)

#     def on_packet(self, packet: dict) -> None:
#         """on_packet
#         docs: https://bybit-exchange.github.io/docs/linear/#t-websocketkline
#         channel: depth, kline
#         mapping: orderBookL2_25, candle

#         depth

#         How to Subscribe

#         ws.send('{"op": "subscribe", "args": ["orderBookL2_25.BTCUSD"]}');
#         Snapshot Response Example - format of the first response

#         {
#             "topic": "orderBookL2_25.BTCUSD",
#             "type": "snapshot",
#             "data": [
#                 {
#                     "price": "2999.00",
#                     "symbol": "BTCUSD",
#                     "id": 29990000,
#                     "side": "Buy",
#                     "size": 9
#                 },
#                 {
#                     "price": "3001.00",
#                     "symbol": "BTCUSD",
#                     "id": 30010000,
#                     "side": "Sell",
#                     "size": 10
#                 }
#             ],
#             "cross_seq": 11518,
#             "timestamp_e6": 1555647164875373
#         }
#         Delta Response Example - format of the responses following the snapshot response

#         {
#             "topic": "orderBookL2_25.BTCUSD",
#             "type": "delta",
#             "data": {
#                 "delete": [
#                     {
#                         "price": "3001.00",
#                         "symbol": "BTCUSD",
#                         "id": 30010000,
#                         "side": "Sell"
#                     }
#                 ],
#                 "update": [
#                     {
#                         "price": "2999.00",
#                         "symbol": "BTCUSD",
#                         "id": 29990000,
#                         "side": "Buy",
#                         "size": 8
#                     }
#                 ],
#                 "insert": [
#                     {
#                         "price": "2998.00",
#                         "symbol": "BTCUSD",
#                         "id": 29980000,
#                         "side": "Buy",
#                         "size": 8
#                     }
#                 ],
#                 "transactTimeE6": 0
#             },
#             "cross_seq": 11519,
#             "timestamp_e6": 1555647221331673
#         }

#         kline

#         How to Subscribe

#         ws.send('{"op":"subscribe","args":["klineV2.1.BTCUSD"]}')
#         Response Example - format of all responses

#         {
#             "topic": "klineV2.1.BTCUSD",                //topic name
#             "data": [{
#                 "start": 1572425640,                    //start time of the candle
#                 "end": 1572425700,                      //end time of the candle
#                 "open": 9200,                           //open price
#                 "close": 9202.5,                        //close price
#                 "high": 9202.5,                         //max price
#                 "low": 9196,                            //min price
#                 "volume": 81790,                        //volume
#                 "turnover": 8.889247899999999,          //turnover
#                 "confirm": False,                       //snapshot flag
#                 "cross_seq": 297503466,
#                 "timestamp": 1572425676958323           //cross time
#             }],
#             "timestamp_e6": 1572425677047994            //server time
#         }
#         """
#         # logger.debug(f'packet: {packet}')
#         # print(f'packet: {packet}')

#         if "topic" not in packet:
#             print(f"the other packet without topic: {packet}")
#             return

#         channel: str = packet["topic"].split(".")[0]
#         symbol: str = packet["topic"].split(".")[-1]
#         # params: dict = packet['params']
#         # tick: dict = self.ticks[symbol]

#         is_fire: bool = False
#         # kline = candle
#         if channel == "klineV2":
#             interval = FUTURE_KBAR_INTERVAL_REV[packet["topic"].split(".")[1]]
#             tick: dict = self.ticks[f"{symbol}|{interval}"]

#             data: dict = packet["data"][0]
#             tick["volume"] = float(data["volume"])
#             tick["turnover"] = float(data["turnover"])
#             tick["open"] = float(data["open"])
#             tick["high"] = float(data["high"])
#             tick["low"] = float(data["low"])
#             tick["close"] = float(data["close"])
#             # tick['start'] = BybitSpotClient.generate_datetime(float(data['end']))
#             tick["start"] = BybitSpotClient.generate_datetime(float(data["start"]))
#             tick["datetime"] = models.generate_datetime()  # datetime.now()

#             # tick['asset_type'] = models.future_asset_type(symbol, "inverse")
#             # tick['name'] = models.normalize_name(symbol, tick['asset_type'])
#             # tick['exchange'] = EXCHANGE
#             if self.on_kline_callback:
#                 is_fire = True
#                 self.on_kline_callback(copy(tick))

#         elif channel == "instrument_info":
#             tick: dict = self.ticks[f"{symbol}|ticker"]
#             tick.update(self.handle_tickerdata(packet, symbol=symbol))

#             if self.on_ticker_callback:
#                 is_fire = True
#                 self.on_ticker_callback(copy(tick))

#         # depth = orderBookL2_25
#         elif channel == "orderBookL2_25":
#             # save asks / bids in tick for next iteration
#             tick: dict = self.ticks[f"{symbol}|depth"]
#             tick.update(self.handle_orderbook(packet, symbol=symbol))

#             if self.on_depth_callback:
#                 is_fire = True
#                 self.on_depth_callback(copy(tick))

#         if not is_fire and self.on_tick:
#             self.on_tick(copy(tick))

#     def handle_tickerdata(self, packet, symbol):
#         if "snapshot" in packet["type"]:
#             self.ticker_data[symbol] = packet["data"]

#         elif "delta" in packet["type"]:
#             for key, value in packet["data"]["update"][0].items():
#                 self.ticker_data[symbol][key] = value

#         data = copy(self.ticker_data[symbol])
#         result = {}
#         result["prev_open_24h"] = None
#         result["prev_high_24h"] = float(data["high_price_24h"])
#         result["prev_low_24h"] = float(data["low_price_24h"])
#         result["prev_close_24h"] = float(data["prev_price_24h"])
#         result["prev_volume_24h"] = float(data["volume_24h"])
#         result["prev_turnover_24h"] = float(data["turnover_24h_e8"]) / 10**8

#         result["last_price"] = float(data["last_price"])
#         result["price_change_pct"] = float(data["price_24h_pcnt_e6"]) / 10**6
#         result["price_change"] = result["last_price"] - result["prev_close_24h"]

#         result["datetime"] = models.generate_datetime()  # datetime.now()

#         return result

#     def handle_orderbook(self, packet, symbol):
#         def bids_asks_process(
#             datas, delete_data, update_data, insert_data, is_bit: bool = False
#         ):
#             remove_list = list()
#             add_list = list()

#             for i, data in enumerate(datas):
#                 # delete data
#                 if delete_data:
#                     # for i in range(len(delete_data)):
#                     #     if data['id'] == delete_data[i]['id']:
#                     #         remove_list.append(i)
#                     for entry in delete_data:
#                         if data["id"] == entry["id"]:
#                             remove_list.append(i)

#                 # update new data
#                 if update_data:
#                     # for i in range(len(update_data)):
#                     #     if data['id'] == update_data[i]['id']:
#                     #         remove_list.append(i)
#                     #         add_list.append(update_data[i])
#                     for entry in update_data:
#                         if data["id"] == entry["id"]:
#                             remove_list.append(i)
#                             add_list.append(entry)

#             if remove_list:
#                 # pop from decending list
#                 remove_list.sort(reverse=True)
#                 # print(datas,'?????', remove_list)

#                 [datas.pop(i) for i in remove_list]
#                 # try:
#                 #     for i in remove_list:
#                 #         datas.pop(i)
#                 # except Exception as e:
#                 #     print(str(e), i, datas, len(datas), delete_data, update_data)

#             if add_list:
#                 datas.extend(add_list)
#             # incert new data
#             if insert_data:
#                 datas.extend(insert_data)

#             datas.sort(key=operator.itemgetter("price"), reverse=is_bit)
#             return datas

#         def bids_asks_category(data, category):
#             bids = [d for d in data[category] if d["side"] == "Buy"]
#             asks = [d for d in data[category] if d["side"] == "Sell"]
#             return bids, asks

#         # call by reference
#         data: dict = packet[
#             "data"
#         ]  # The data is ordered by price, starting with the lowest buys and ending with the highest sells.

#         if packet["type"] == "snapshot":
#             self.bids_asks_data[symbol]["bids"] = [
#                 d for d in data if d["side"] == "Buy"
#             ]
#             self.bids_asks_data[symbol]["asks"] = [
#                 d for d in data if d["side"] == "Sell"
#             ]
#         elif packet["type"] == "delta":
#             bids_delete_data, asks_delete_data = bids_asks_category(
#                 data=data, category="delete"
#             )
#             bids_update_data, asks_update_data = bids_asks_category(
#                 data=data, category="update"
#             )
#             bids_insert_data, asks_insert_data = bids_asks_category(
#                 data=data, category="insert"
#             )

#             tmp_bids = copy(self.bids_asks_data[symbol]["bids"])
#             self.bids_asks_data[symbol]["bids"] = bids_asks_process(
#                 datas=tmp_bids,
#                 delete_data=bids_delete_data,
#                 update_data=bids_update_data,
#                 insert_data=bids_insert_data,
#                 is_bit=True,
#             )

#             tmp_asks = copy(self.bids_asks_data[symbol]["asks"])
#             self.bids_asks_data[symbol]["asks"] = bids_asks_process(
#                 datas=tmp_asks,
#                 delete_data=asks_delete_data,
#                 update_data=asks_update_data,
#                 insert_data=asks_insert_data,
#             )
#         # print('len of asks', len(self.bids_asks_data[symbol]['asks']),'len of bids', len(self.bids_asks_data['bids']))
#         # computer the bids and asks with current data
#         bids: list = self.bids_asks_data[symbol]["bids"]
#         for n in range(min(5, len(bids))):
#             price = bids[n]["price"]
#             volume = bids[n]["size"]
#             self.bids_asks_data[symbol]["bid_price_" + str(n + 1)] = float(price)
#             self.bids_asks_data[symbol]["bid_volume_" + str(n + 1)] = float(volume)

#         asks: list = self.bids_asks_data[symbol]["asks"]
#         for n in range(min(5, len(asks))):
#             price = asks[n]["price"]
#             volume = asks[n]["size"]
#             self.bids_asks_data[symbol]["ask_price_" + str(n + 1)] = float(price)
#             self.bids_asks_data[symbol]["ask_volume_" + str(n + 1)] = float(volume)

#         # self.bids_asks_data[symbol]['asset_type'] = models.future_asset_type(symbol, "future")
#         # self.bids_asks_data[symbol]['name'] = models.normalize_name(symbol, self.bids_asks_data[symbol]['asset_type'])
#         # self.bids_asks_data[symbol]['exchange'] = EXCHANGE

#         self.bids_asks_data[symbol]["datetime"] = (
#             models.generate_datetime()
#         )  # datetime.now()

#         result = copy(self.bids_asks_data[symbol])
#         result.pop("bids")
#         result.pop("asks")
#         return result


# class BybitInverseFutureTradeWebsocket(WebsocketClient):
#     """
#     Implement private topics
#     doc: https://bybit-exchange.github.io/docs/linear/#t-privatetopics
#     """

#     DATA_TYPE = "inverse"

#     def __init__(
#         self,
#         on_account_callback: Callable = None,
#         on_order_callback: Callable = None,
#         on_trade_callback: Callable = None,
#         on_position_callback: Callable = None,
#         on_connected_callback: Callable = None,
#         on_disconnected_callback: Callable = None,
#         on_error_callback: Callable = None,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         is_testnet: bool = False,
#         ping_interval: int = 20,
#         debug: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         self.on_account_callback = on_account_callback
#         self.on_order_callback = on_order_callback
#         self.on_trade_callback = on_trade_callback
#         self.on_position_callback = on_position_callback
#         self.on_connected_callback = on_connected_callback
#         self.on_disconnected_callback = on_disconnected_callback
#         self.on_error_callback = on_error_callback

#         self.ping_interval = ping_interval
#         self.is_testnet = is_testnet
#         self.proxy_host = proxy_host
#         self.proxy_port = proxy_port
#         self._logged_in = False
#         self.key = None
#         self.secret = None
#         # self.channel = "orders"
#         self.debug = debug

#     def info(self):
#         return "Bybit Future Trade Websocket Start"

#     def connect(self, key: str, secret: str):
#         """
#         Give api key and secret for Authorization.
#         """
#         self.key = key
#         self.secret = secret

#         self.init(
#             host=BYBIT_TESTNET_WS_INVERSE_HOST
#             if self.is_testnet
#             else BYBIT_WS_INVERSE_HOST,
#             proxy_host=self.proxy_host,
#             proxy_port=self.proxy_port,
#             ping_interval=self.ping_interval,
#         )
#         self.start()

#         self.waitfor_connection()

#         if not self._logged_in:
#             self._login()
#         logger.info(self.info())
#         return True

#     def _login(self):
#         """
#         Authorize websocket connection.
#         """

#         # Generate expires.
#         expires = int((time.time() + 1) * 1000)
#         api_key = self.key
#         api_secret = self.secret

#         if api_key is None or api_secret is None:
#             raise PermissionError("Authenticated endpoints require keys.")

#         # Generate signature.
#         _val = f"GET/realtime{expires}"
#         signature = str(
#             hmac.new(
#                 bytes(api_secret, "utf-8"), bytes(_val, "utf-8"), digestmod="sha256"
#             ).hexdigest()
#         )

#         req: dict = {"op": "auth", "args": [api_key, expires, signature]}

#         logger.debug(req)
#         self.send_packet(req)

#     def on_disconnected(self) -> None:
#         print(
#             "Bybit Inverse Future user trade data Websocket disconnected, now try to connect @ {datetime.now()}"
#         )
#         self._logged_in = False
#         if self.on_disconnected_callback:
#             self.on_disconnected_callback()

#     def on_connected(self) -> None:
#         """"""
#         if self.waitfor_connection():
#             print(
#                 f"Bybit Inverse Future user trade data websocket connect success ({datetime.now()})"
#             )
#         else:
#             print(
#                 f"Bybit Inverse Future user trade data websocket connect fail ({datetime.now()})"
#             )

#         if not self._logged_in:
#             self._login()
#             self._logged_in = True
#         # logger.debug("Bybit Inverse Future user trade data Websocket connect success")

#         req: dict = self.sub_stream(channel=["trades", "orders", "position", "account"])
#         print(f"Bybit Inverse Future on_connected req:{req}")
#         self.send_packet(req)
#         if self.on_connected_callback:
#             self.on_connected_callback()

#     def on_error(self, exception_type, exception_value, traceback_obj):
#         super().on_error(exception_type, exception_value, traceback_obj)
#         if self.on_error_callback:
#             self.on_error_callback(exception_value)

#     def inject_callback(self, channel: str, callbackfn: Callable):
#         if channel == "account":
#             self.on_account_callback = callbackfn
#         elif channel == "orders":
#             self.on_order_callback = callbackfn
#         elif channel == "trades":
#             self.on_trade_callback = callbackfn
#         elif channel == "position":
#             self.on_position_callback = callbackfn
#         elif channel == "connect":
#             self.on_connected_callback = callbackfn
#         elif channel == "disconnect":
#             self.on_disconnected_callback = callbackfn
#         elif channel == "error":
#             self.on_error_callback = callbackfn
#         else:
#             Exception("invalid callback function")

#     def sub_stream(self, channel="orders"):
#         """ """
#         sub = {"op": "subscribe"}
#         if isinstance(channel, list):
#             sub["args"] = [INVERSE_FUTURE_MAPPING_CHANNEL[c] for c in channel]
#         else:
#             mapping_channel = INVERSE_FUTURE_MAPPING_CHANNEL[channel]
#             sub["args"] = [f"{mapping_channel}"]

#         return sub

#     def subscribe(self, channel="orders") -> None:
#         """
#         keep the interface:
#         channel: 'trades', 'orders', 'account'
#         mapping: 'execution', 'order', 'wallet'
#         """
#         if channel not in ["trades", "orders", "position"]:
#             Exception("invalid subscription")
#             return

#         if not self._logged_in:
#             self._login()

#         self.channel = channel
#         req: dict = self.sub_stream(channel=channel)
#         # logger.debug(req)
#         # print(req,'???')
#         self.send_packet(req)

#     def on_packet(self, packet: dict) -> None:
#         # logger.debug(f"on_packet event {packet}")
#         if self.debug:
#             print(
#                 f"[DEBUG] Bybit Inverse Future Trade Websocket - on_packet event {packet}"
#             )
#         # if 'auth' in packet:
#         #     self._logged_in = True if packet['auth'] == "success" else False
#         #     if self._logged_in:
#         #         print(f"Bybit Future Trade Websocket - successfully logined - {packet}")
#         #     else:
#         #         print(f"Bybit Future Trade Websocket auth failed - on_packet event {packet}")
#         #     return

#         if "topic" not in packet:
#             logger.info(f"the other packet without topic: {packet}")
#             return

#         if "ping" in packet:
#             logger.info(f"healthcheck message {packet}")
#             return

#         if not packet:
#             logger.debug(f"unknown packet event {packet}")
#             return

#         if packet["topic"] == "wallet":
#             self.on_account(packet)
#         elif packet["topic"] == "position":
#             self.on_position(packet)
#         elif packet["topic"] == "order":
#             self.on_order(packet)
#         elif packet["topic"] == "execution":
#             self.on_trade(packet)
#         else:
#             logger.info(f"the other packet type: {packet}")

#     def on_account(self, packet: dict) -> None:
#         """
#         How to Subscribe

#         ws.send('{"op": "subscribe", "args": ["wallet"]}')
#         Response Example - format of all responses

#         {
#             "topic": "wallet",
#             "data": [
#                 {
#                     "wallet_balance":429.80713,
#                     "available_balance":429.67322
#                 }
#             ]
#         }
#         """
#         logger.debug(f"packet on_account: {packet}")

#         for d in packet["data"]:
#             account = {
#                 "symbol": d["coin"],
#                 "balance": float(d["wallet_balance"]),
#                 "available": float(d["available_balance"]),
#                 "frozen": float(d["wallet_balance"]) - float(d["available_balance"]),
#                 "exchange": EXCHANGE,
#                 "asset_type": self.DATA_TYPE.upper(),
#             }

#             if account["symbol"] and self.on_account_callback:
#                 self.on_account_callback(account)

#     def on_order(self, packet: dict) -> None:
#         """
#         How to Subscribe

#         ws.send('{"op": "subscribe", "args": ["order"]}')
#         Response Example - format of all responses

#         {
#             "topic": "order",
#             "action": "",
#             "data": [
#                 {
#                     "order_id": "xxxxxxxx-xxxx-xxxx-9a8f-4a973eb5c418",
#                     "order_link_id": "",
#                     "symbol": "BTCUSD",
#                     "side": "Sell",
#                     "order_type": "Market",
#                     "price": "8579.5",
#                     "qty": 1,
#                     "time_in_force": "ImmediateOrCancel",
#                     "create_type": "CreateByClosing",
#                     "cancel_type": "",
#                     "order_status": "Filled",
#                     "leaves_qty": 0,
#                     "cum_exec_qty": 1,
#                     "cum_exec_value": "0.00011655",
#                     "cum_exec_fee": "0.00000009",
#                     "timestamp": "2020-01-14T14:09:31.778Z",
#                     "take_profit": "0",
#                     "stop_loss": "0",
#                     "trailing_stop": "0",
#                     "trailing_active": "0",
#                     "last_exec_price": "8580",
#                     "reduce_only": false,
#                     "close_on_trigger": false
#                 }
#             ]
#         }
#         """
#         logger.debug(f"packet on_order: {packet}")
#         datas: dict = packet["data"]

#         # if datas["order_type"] not in ['LIMIT', 'Market']:
#         #     return

#         order_list = list()
#         for data in datas:
#             order = models.OrderSchema()  # .model_construct(data)
#             order = order.from_bybit_to_form(data, datatype=self.DATA_TYPE)

#             order_list.append(order)

#             if self.on_order_callback:  # and order_list:
#                 # self.on_order_callback(order_list)
#                 self.on_order_callback(order)

#     def on_trade(self, packet: dict) -> None:
#         """
#         How to Subscribe

#         ws.send('{"op": "subscribe", "args": ["execution"]}')
#         Response Example - format of all responses

#         {
#             "topic": "execution",
#             "data": [
#                 {
#                     "symbol": "BTCUSD",
#                     "side": "Buy",
#                     "order_id": "xxxxxxxx-xxxx-xxxx-9a8f-4a973eb5c418",
#                     "exec_id": "xxxxxxxx-xxxx-xxxx-8b66-c3d2fcd352f6",
#                     "order_link_id": "",
#                     "price": "8300",
#                     "order_qty": 1,
#                     "exec_type": "Trade",
#                     "exec_qty": 1,
#                     "exec_fee": "0.00000009",
#                     "leaves_qty": 0,
#                     "is_maker": false,
#                     "trade_time": "2020-01-14T14:07:23.629Z" // trade time
#                 }
#             ]
#         }
#         """
#         logger.debug(f"packet on_trade: {packet}")
#         datas: dict = packet["data"]

#         trade_list = list()
#         for data in datas:
#             trade = models.TradeSchema()  # .model_construct(data)
#             trade = trade.from_bybit_to_form(data, datatype=self.DATA_TYPE)
#             trade_list.append(trade)

#             if self.on_trade_callback:  # and trade_list:
#                 # self.on_trade_callback(trade_list)
#                 self.on_trade_callback(trade)

#     def on_position(self, packet: dict) -> None:
#         """
#         {
#         "topic": "position",
#         "action": "update",
#         "data": [
#         {
#                     "user_id": "533285",
#                     "symbol": "BTCUSDT",
#                     "size": 0.01,
#                     "side": "Buy",
#                     "position_value": 202.195,
#                     "entry_price": 20219.5,
#                     "liq_price": 0.5,
#                     "bust_price": 0.5,
#                     "leverage": 99,
#                     "order_margin": 0,
#                     "position_margin": 1959.6383,
#                     "occ_closing_fee": 3e-06,
#                     "take_profit": 25000,
#                     "tp_trigger_by": "LastPrice",
#                     "stop_loss": 18000,
#                     "sl_trigger_by": "LastPrice",
#                     "trailing_stop": 0,
#                     "realised_pnl": -4.189762,
#                     "auto_add_margin": "0",
#                     "cum_realised_pnl": -13.640625,
#                     "position_status": "Normal",
#                     "position_id": "0",
#                     "position_seq": "92962",
#                     "adl_rank_indicator": "2",
#                     "free_qty": 0.01,
#                     "tp_sl_mode": "Full",
#                     "risk_id": "1",
#                     "isolated": false,
#                     "mode": "BothSide",
#                     "position_idx": "1"
#                 }
#         ]
#         }
#         """
#         logger.debug(f"packet on_position: {packet}")
#         datas: dict = packet["data"]
#         position_list = list()
#         for data in datas:
#             # position_value = float(data["position_value"])
#             # if not position_value:
#             #     continue
#             position = models.PositionSchema()  # .model_construct(data)
#             position = position.from_bybit_to_form(data, datatype=self.DATA_TYPE)

#             position_list.append(position)
#             if self.on_position_callback:  # and position_list:
#                 # self.on_position_callback(position_list)
#                 self.on_position_callback(position)


# class BybitOptionDataWebsocket(WebsocketClient):
#     """
#     Implement public topics
#     doc:
#         https://bybit-exchange.github.io/docs/usdc/option/#t-websocket
#     """

#     def __init__(
#         self,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         ping_interval: int = 20,
#         is_testnet: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__()

#         self.is_testnet: bool = is_testnet
#         self.proxy_host: str = proxy_host
#         self.proxy_port: int = proxy_port
#         self.ping_interval: int = ping_interval

#         self.ticks: dict[str, dict] = defaultdict(dict)
#         self.reqid: int = 0
#         # self.init(host=BYBIT_WS_INVERSE_HOST, proxy_host=proxy_host, proxy_port=proxy_port, ping_interval=29)
#         # self.start()
#         self._logged_in = False
#         self.key = None
#         self.secret = None
#         # self.channel = 'klineV2'
#         self.kbar_time: int = int(time.time()) - (int(time.time()) % 60)
#         # self.bids_asks_data: list = []
#         self.bids_asks_data: dict[str, dict] = defaultdict(dict)
#         self.ticker_data: dict[str, dict] = defaultdict(dict)

#         self.on_tick = None
#         self.on_ticker_callback = None
#         self.on_depth_callback = None
#         self.on_kline_callback = None
#         self.on_connected_callback = None
#         self.on_disconnected_callback = None
#         self.on_error_callback = None
#         # self.waitfor_connection()

#     def info(self):
#         return "Bybit Option Data Websocket start"

#     def connect(self):
#         """ """
#         self.init(
#             host=BYBIT_TESTNET_WS_OPTION_PUBLIC_HOST
#             if self.is_testnet
#             else BYBIT_WS_OPTION_PUBLIC_HOST,
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
#             print(
#                 f"Bybit Option market data websocket connect success ({datetime.now()})"
#             )
#         else:
#             print(f"Bybit Option market data websocket connect fail ({datetime.now()})")

#         # resubscribe
#         print(
#             f"===> debug Bybit Option Market data on_connected - self.ticks: {self.ticks} | on_ticker_callback:{self.on_ticker_callback}; on_depth_callback:{self.on_depth_callback}; on_kline_callback:{self.on_kline_callback}"
#         )
#         if self.ticks:
#             # for symbol in self.ticks.keys():
#             #     req: dict = self.sub_stream(symbol=symbol, channel=self.channel)
#             #     logger.debug(req)
#             #     self.send_packet(req)
#             channels = []
#             for key, detail in self.ticks.items():
#                 req: dict = self.sub_stream(
#                     symbol=detail["symbol"], channel=detail["channel"]
#                 )
#                 channels += req["args"]
#                 # logger.debug(req)
#                 # print(f"!!!!! rea: {req}")
#                 # self.send_packet(req)
#             req: dict = {"op": "subscribe", "args": channels}
#             print(f"on_connected, resubscribing the channels - req:{req}")
#             self.send_packet(req)

#     def on_disconnected(self) -> None:
#         print(
#             "Bybit Option market data Websocket disconnected, now try to connect @ {datetime.now()}"
#         )

#     def on_error(self, exception_type, exception_value, traceback_obj):
#         super().on_error(exception_type, exception_value, traceback_obj)
#         if self.on_error_callback:
#             self.on_error_callback(exception_value)

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

#     def sub_stream(self, symbol, channel="depth"):
#         """
#         https://bybit-exchange.github.io/docs/linear/#t-websocketkline

#         "depth": "orderBookL2_25",
#         "kline": "klineV2",
#         """
#         self.reqid += 1
#         mapping_channel = OPTION_MAPPING_CHANNEL[channel]
#         sub = {"op": "subscribe"}

#         if channel == "depth":
#             sub["args"] = [f"{mapping_channel}.{symbol}"]
#         elif channel == "ticker":
#             sub["args"] = [f"{mapping_channel}.{symbol}"]

#         # print(sub,self.reqid,'!!!!!!!!!!', symbol, '------',channel, self.is_connected)
#         return sub

#     def subscribe(self, symbols, on_tick=None, channel="depth") -> None:
#         """
#         channel: "depth", "kline", "ticker"
#         """
#         if channel not in ["depth", "ticker", "trades", "orders", "account"]:
#             Exception("invalid subscription")
#             return

#         # self.channel = channel
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

#         for symbol in symbols:
#             if symbol in self.ticks:
#                 return

#             # tick data dict
#             asset_type = models.ASSET_TYPE_OPTION
#             tick = {
#                 "symbol": symbol,
#                 "exchange": EXCHANGE,
#                 "channel": channel,
#                 "asset_type": asset_type,
#                 "name": models.normalize_name(symbol, asset_type),
#             }
#             key = f"{symbol}|{channel}"

#             self.ticks[key] = tick
#             req: dict = self.sub_stream(symbol=symbol, channel=channel)
#             # print(f"req: {req}")
#             self.send_packet(req)

#     def on_packet(self, packet: dict) -> None:
#         # print(f'packet: {packet}')

#         if "topic" not in packet:
#             print(f"the other packet without topic: {packet}")
#             return

#         channel: str = packet["topic"].split(".")[0]
#         symbol: str = packet["topic"].split(".")[-1]
#         # params: dict = packet['params']
#         # tick: dict = self.ticks[symbol]
#         # return

#         is_fire: bool = False
#         if channel == "instrument_info":
#             """
#             {'id': 'instrument_info.BTC-31MAR23-5000-P-1412823371-1669706949758', 'topic': 'instrument_info.BTC-31MAR23-5000-P', 'creationTime': 1669706949758, 'data':
#                 {'symbol': 'BTC-31MAR23-5000-P', 'bidPrice': '140', 'askPrice': '0', 'bidIv': '1.2694', 'askIv': '0',
#                 'bidSize': '0.01', 'askSize': '0', 'markPrice': '129.45572072', 'markPriceIv': '1.2489',
#                 'indexPrice': '16469.01', 'underlyingPrice': '16428.62', 'lastPrice': '95', 'delta': '-0.02229503',
#                 'gamma': '0.00000448', 'theta': '-2.58009784', 'vega': '5.04189797', 'change24h': '0', 'volume24h': '0', 'turnover24h': '0',
#                 'high24h': '0', 'low24h': '0', 'totalVolume': '3', 'totalTurnover': '56530', 'openInterest': '2.85', 'predictedDeliveryPrice': '0'}}
#             """
#             data: dict = packet["data"]

#             tick: dict = self.ticks[f"{symbol}|ticker"]
#             tick["bid_price"] = float(data["bidPrice"])
#             tick["ask_price"] = float(data["askPrice"])
#             tick["bid_volume"] = float(data["bidSize"])
#             tick["ask_volume"] = float(data["askSize"])
#             tick["bid_iv"] = float(data["bidIv"])
#             tick["ask_iv"] = float(data["askIv"])
#             tick["mark_price"] = float(data["markPriceIv"])
#             tick["mark_price_iv"] = float(data["markPriceIv"])

#             tick["prev_open_24h"] = None
#             tick["prev_high_24h"] = float(data["high24h"])
#             tick["prev_low_24h"] = float(data["low24h"])
#             tick["prev_volume_24h"] = float(data["volume24h"])
#             tick["prev_turnover_24h"] = float(data["turnover24h"]) / 10**8
#             # tick["prev_openinterest_24h"] = float(data["openInterest"])
#             tick["predicted_delivery_price"] = float(data["predictedDeliveryPrice"])

#             tick["last_price"] = float(data["lastPrice"])
#             tick["price_change_pct"] = None
#             tick["price_change"] = float(data["change24h"])
#             tick["prev_close_24h"] = tick["last_price"] / (tick["price_change"] + 1)

#             tick["delta"] = float(data["delta"])
#             tick["gamma"] = float(data["gamma"])
#             tick["theta"] = float(data["theta"])
#             tick["vega"] = float(data["vega"])
#             tick["underlying_price"] = float(data["underlyingPrice"])

#             tick["datetime"] = models.generate_datetime()  # datetime.now()

#             if self.on_ticker_callback:
#                 is_fire = True
#                 self.on_ticker_callback(copy(tick))

#         elif channel == "delta":  # .orderbook100":
#             """
#             {"id":"delta.orderbook100.BTC-31MAR23-5000-P-2160938609-1669709642715","topic":"delta.orderbook100.BTC-31MAR23-5000-P","creationTime":1669709642715,"data":
#                 {"version":"4650","dataType":"NEW","orderBook":[{"price":"5","size":"0.04","side":"Buy"},{"price":"130","size":"41","side":"Sell"},{"price":"170","size":"0.01","side":"Sell"},{"price":"175","size":"41.7","side":"Sell"},{"price":"215","size":"44.9","side":"Sell"},{"price":"230","size":"0.01","side":"Sell"},{"price":"235","size":"44.9","side":"Sell"},{"price":"340","size":"0.01","side":"Sell"},{"price":"985","size":"0.09","side":"Sell"},{"price":"5000","size":"0.01","side":"Sell"}]}}

#             {'id': 'delta.orderbook100.BTC-31MAR23-15000-P-1415452402-1669710465638', 'topic': 'delta.orderbook100.BTC-31MAR23-15000-P', 'creationTime': 1669710465638, 'data':
#                 {'version': '117909', 'dataType': 'CHANGE', 'delete': [{'price': '1400', 'side': 'Buy'}], 'update': [], 'insert': []}}

#             """
#             # save asks / bids in tick for next iteration
#             # symbol = '-'.join(symbol.split('-')[:4])
#             tick: dict = self.ticks[f"{symbol}|depth"]
#             tick.update(self.handle_orderbook(packet, symbol=symbol))

#             if self.on_depth_callback:
#                 is_fire = True
#                 self.on_depth_callback(copy(tick))

#         if not is_fire and self.on_tick:
#             self.on_tick(copy(tick))

#     def handle_orderbook(self, packet, symbol):
#         def bids_asks_process(
#             datas, delete_data, update_data, insert_data, is_bit: bool = False
#         ):
#             remove_list = list()
#             add_list = list()

#             for i, data in enumerate(datas):
#                 # delete data
#                 if delete_data:
#                     # for i in range(len(delete_data)):
#                     #     if data['id'] == delete_data[i]['id']:
#                     #         remove_list.append(i)
#                     for entry in delete_data:
#                         if data["price"] == entry["price"]:
#                             remove_list.append(i)

#                 # update new data
#                 if update_data:
#                     # for i in range(len(update_data)):
#                     #     if data['id'] == update_data[i]['id']:
#                     #         remove_list.append(i)
#                     #         add_list.append(update_data[i])
#                     for entry in update_data:
#                         if data["price"] == entry["price"]:
#                             remove_list.append(i)
#                             add_list.append(entry)

#             if remove_list:
#                 # pop from decending list
#                 remove_list.sort(reverse=True)
#                 # print(datas,'?????', remove_list)

#                 [datas.pop(i) for i in remove_list]
#                 # try:
#                 #     for i in remove_list:
#                 #         datas.pop(i)
#                 # except Exception as e:
#                 #     print(str(e), i, datas, len(datas), delete_data, update_data)

#             if add_list:
#                 datas.extend(add_list)
#             # incert new data
#             if insert_data:
#                 datas.extend(insert_data)

#             datas.sort(key=operator.itemgetter("price"), reverse=is_bit)
#             return datas

#         def bids_asks_category(data, category):
#             bids = [d for d in data[category] if d["side"] == "Buy"]
#             asks = [d for d in data[category] if d["side"] == "Sell"]
#             return bids, asks

#         # call by reference
#         data: dict = packet[
#             "data"
#         ]  # The data is ordered by price, starting with the lowest buys and ending with the highest sells.

#         if data["dataType"] == "NEW":
#             self.bids_asks_data[symbol]["bids"] = [
#                 d for d in data["orderBook"] if d["side"] == "Buy"
#             ]
#             self.bids_asks_data[symbol]["asks"] = [
#                 d for d in data["orderBook"] if d["side"] == "Sell"
#             ]
#         elif data["dataType"] == "CHANGE":
#             bids_delete_data, asks_delete_data = bids_asks_category(
#                 data=data, category="delete"
#             )
#             bids_update_data, asks_update_data = bids_asks_category(
#                 data=data, category="update"
#             )
#             bids_insert_data, asks_insert_data = bids_asks_category(
#                 data=data, category="insert"
#             )

#             tmp_bids = copy(self.bids_asks_data[symbol]["bids"])
#             self.bids_asks_data[symbol]["bids"] = bids_asks_process(
#                 datas=tmp_bids,
#                 delete_data=bids_delete_data,
#                 update_data=bids_update_data,
#                 insert_data=bids_insert_data,
#                 is_bit=True,
#             )

#             tmp_asks = copy(self.bids_asks_data[symbol]["asks"])
#             self.bids_asks_data[symbol]["asks"] = bids_asks_process(
#                 datas=tmp_asks,
#                 delete_data=asks_delete_data,
#                 update_data=asks_update_data,
#                 insert_data=asks_insert_data,
#             )
#         # print('len of asks', len(self.bids_asks_data[symbol]['asks']),'len of bids', len(self.bids_asks_data['bids']))
#         # computer the bids and asks with current data
#         bids: list = self.bids_asks_data[symbol]["bids"]
#         for n in range(min(5, len(bids))):
#             price = bids[n]["price"]
#             volume = bids[n]["size"]
#             self.bids_asks_data[symbol]["bid_price_" + str(n + 1)] = float(price)
#             self.bids_asks_data[symbol]["bid_volume_" + str(n + 1)] = float(volume)

#         asks: list = self.bids_asks_data[symbol]["asks"]
#         for n in range(min(5, len(asks))):
#             price = asks[n]["price"]
#             volume = asks[n]["size"]
#             self.bids_asks_data[symbol]["ask_price_" + str(n + 1)] = float(price)
#             self.bids_asks_data[symbol]["ask_volume_" + str(n + 1)] = float(volume)

#         # self.bids_asks_data[symbol]['asset_type'] = models.future_asset_type(symbol, "future")
#         # self.bids_asks_data[symbol]['name'] = models.normalize_name(symbol, self.bids_asks_data[symbol]['asset_type'])
#         # self.bids_asks_data[symbol]['exchange'] = EXCHANGE

#         self.bids_asks_data[symbol]["datetime"] = (
#             models.generate_datetime()
#         )  # datetime.now()

#         result = copy(self.bids_asks_data[symbol])
#         result.pop("bids")
#         result.pop("asks")
#         return result


# class BybitOptionTradeWebsocket(WebsocketClient):
#     """
#     Implement private topics
#     doc: https://bybit-exchange.github.io/docs/linear/#t-privatetopics
#     """

#     DATA_TYPE = "option"

#     def __init__(
#         self,
#         on_greeks_callback: Callable = None,
#         on_order_callback: Callable = None,
#         on_trade_callback: Callable = None,
#         on_position_callback: Callable = None,
#         on_connected_callback: Callable = None,
#         on_disconnected_callback: Callable = None,
#         on_error_callback: Callable = None,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         is_testnet: bool = False,
#         ping_interval: int = 20,
#         debug: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         self.on_greeks_callback = on_greeks_callback
#         self.on_order_callback = on_order_callback
#         self.on_trade_callback = on_trade_callback
#         self.on_position_callback = on_position_callback
#         self.on_connected_callback = on_connected_callback
#         self.on_disconnected_callback = on_disconnected_callback
#         self.on_error_callback = on_error_callback

#         self.ping_interval = ping_interval
#         self.is_testnet = is_testnet
#         self.proxy_host = proxy_host
#         self.proxy_port = proxy_port
#         self._logged_in = False
#         self.key = None
#         self.secret = None
#         # self.channel = "orders"
#         self.debug = debug

#     def info(self):
#         return "Bybit Future Trade Websocket Start"

#     def connect(self, key: str, secret: str):
#         """
#         Give api key and secret for Authorization.
#         """
#         self.key = key
#         self.secret = secret

#         self.init(
#             host=BYBIT_TESTNET_WS_OPTION_PRIVATE_HOST
#             if self.is_testnet
#             else BYBIT_WS_OPTION_PRIVATE_HOST,
#             proxy_host=self.proxy_host,
#             proxy_port=self.proxy_port,
#             ping_interval=self.ping_interval,
#         )
#         self.start()

#         self.waitfor_connection()

#         if not self._logged_in:
#             self._login()
#         logger.info(self.info())
#         return True

#     def _login(self):
#         """
#         Authorize websocket connection.
#         """

#         # Generate expires.
#         expires = int((time.time() + 1) * 1000)
#         api_key = self.key
#         api_secret = self.secret

#         if api_key is None or api_secret is None:
#             raise PermissionError("Authenticated endpoints require keys.")

#         # Generate signature.
#         _val = f"GET/realtime{expires}"
#         signature = str(
#             hmac.new(
#                 bytes(api_secret, "utf-8"), bytes(_val, "utf-8"), digestmod="sha256"
#             ).hexdigest()
#         )

#         req: dict = {"op": "auth", "args": [api_key, expires, signature]}

#         # logger.debug(req)
#         if self.debug:
#             print(req)
#         self.send_packet(req)

#     def on_disconnected(self) -> None:
#         print(
#             "Bybit Option user trade data Websocket disconnected, now try to connect @ {datetime.now()}"
#         )
#         self._logged_in = False
#         if self.on_disconnected_callback:
#             self.on_disconnected_callback()

#     def on_connected(self) -> None:
#         """"""
#         if self.waitfor_connection():
#             print(
#                 f"Bybit Option user trade data websocket connect success ({datetime.now()})"
#             )
#         else:
#             print(
#                 f"Bybit Option user trade data websocket connect fail ({datetime.now()})"
#             )

#         if not self._logged_in:
#             self._login()
#             self._logged_in = True
#         # logger.debug("Bybit Option user trade data Websocket connect success")

#         req: dict = self.sub_stream(channel=["trades", "orders", "position", "greeks"])
#         print(f"Bybit Option on_connected req:{req}")
#         self.send_packet(req)
#         if self.on_connected_callback:
#             self.on_connected_callback()

#     def on_error(self, exception_type, exception_value, traceback_obj):
#         super().on_error(exception_type, exception_value, traceback_obj)
#         if self.on_error_callback:
#             self.on_error_callback(exception_value)

#     def inject_callback(self, channel: str, callbackfn: Callable):
#         if channel == "greeks":
#             self.on_greeks_callback = callbackfn
#         elif channel == "orders":
#             self.on_order_callback = callbackfn
#         elif channel == "trades":
#             self.on_trade_callback = callbackfn
#         elif channel == "position":
#             self.on_position_callback = callbackfn
#         elif channel == "connect":
#             self.on_connected_callback = callbackfn
#         elif channel == "disconnect":
#             self.on_disconnected_callback = callbackfn
#         elif channel == "error":
#             self.on_error_callback = callbackfn
#         else:
#             Exception("invalid callback function")

#     def sub_stream(self, channel="orders"):
#         """ """
#         sub = {"op": "subscribe"}
#         if isinstance(channel, list):
#             sub["args"] = [OPTION_MAPPING_CHANNEL[c] for c in channel]
#         else:
#             mapping_channel = OPTION_MAPPING_CHANNEL[channel]
#             sub["args"] = [f"{mapping_channel}"]

#         return sub

#     def subscribe(self, channel="orders") -> None:
#         """
#         keep the interface:
#         channel: 'trades', 'orders', 'account'
#         mapping: 'execution', 'order', 'wallet'
#         """
#         if channel not in ["trades", "orders", "position", "greeks"]:
#             Exception("invalid subscription")
#             return

#         if not self._logged_in:
#             self._login()

#         self.channel = channel
#         req: dict = self.sub_stream(channel=channel)
#         # logger.debug(req)
#         # print(req,'???')
#         self.send_packet(req)

#     def on_packet(self, packet: dict) -> None:
#         # logger.debug(f"on_packet event {packet}")
#         if self.debug:
#             print(f"[DEBUG] Bybit Option Trade Websocket - on_packet event {packet}")

#         """
#         on_packet event {'id': '1b86d79a-d4be-4885-95ed-15df1ea656323c202ae8-5860-496f-97b1-33546c5cf4b0', 'topic': 'user.order', 'creationTime': 1669864668474, 'data': {'orderId': '1b86d79a-d4be-4885-95ed-15df1ea65632', 'symbol': 'BTC-27JAN23-17500-P', 'orderStatus': 'New', 'side': 'Buy', 'orderPrice': '1400', 'orderAllSize': '0.02', 'orderFilledSize': '0', 'orderRemainingSize': '0.02', 'orderType': 'Limit', 'orderTime': '1669864668467', 'timeInForce': 'GoodTillCancel', 'createTimeStamp': '1669864668467', 'updateTimeStamp': '1669864668471', 'execType': 'newed', 'version': '1416240863', 'placeMode': 'advanced', 'placeType': 'price', 'userId': '540325'}}

#         on_packet event {'id': 'd4c00e1b-12f9-4ef9-8001-523ecb777244f0f687ea-f9c8-4659-978c-26187d73335d', 'topic': 'user.order', 'creationTime': 1669865158050, 'data': {'orderId': 'd4c00e1b-12f9-4ef9-8001-523ecb777244', 'symbol': 'BTC-27JAN23-17500-P', 'orderStatus': 'New', 'side': 'Buy', 'orderPrice': '2080', 'orderAllSize': '0.01', 'orderFilledSize': '0', 'orderRemainingSize': '0.01', 'orderType': 'Market', 'orderTime': '1669865158044', 'timeInForce': 'ImmediateOrCancel', 'createTimeStamp': '1669865158044', 'updateTimeStamp': '1669865158048', 'execType': 'newed', 'version': '1416241750', 'placeMode': 'advanced', 'placeType': 'price', 'userId': '540325'}}
#         on_packet event {'id': 'd4c00e1b-12f9-4ef9-8001-523ecb777244bf65c62a-b1e4-431c-9456-ab92b11a8434', 'topic': 'user.order', 'creationTime': 1669865158051, 'data': {'orderId': 'd4c00e1b-12f9-4ef9-8001-523ecb777244', 'symbol': 'BTC-27JAN23-17500-P', 'orderStatus': 'Filled', 'side': 'Buy', 'orderPrice': '2000', 'orderAllSize': '0.01', 'orderFilledSize': '0.01', 'orderRemainingSize': '0', 'orderType': 'Market', 'orderTime': '1669865158044', 'timeInForce': 'ImmediateOrCancel', 'createTimeStamp': '1669865158044', 'updateTimeStamp': '1669865158048', 'execType': 'filled', 'version': '1416241750', 'placeMode': 'advanced', 'placeType': 'price', 'userId': '540325'}}
#         on_packet event {'id': 'ed41a24e-0879-4ced-9e08-2c6ca991faff', 'topic': 'user.openapi.option.trade', 'creationTime': 1669865158090, 'data': {'result': [{'orderId': 'd4c00e1b-12f9-4ef9-8001-523ecb777244', 'orderLinkId': '', 'tradeId': 'f532e092-b61c-5863-a63a-6a7e68d2a62f', 'symbol': 'BTC-27JAN23-17500-P', 'side': 'Buy', 'execPrice': '2000.00000000', 'execQty': '0.01', 'execFee': '0.05147145', 'feeRate': '0.00030000000000000000', 'tradeTime': 1669865158048, 'lastLiquidityInd': 'TAKER', 'execType': 'TRADE', 'blockTradeId': '', 'iv': '0.6655134290900975', 'markIv': '0.5762', 'underlyingPrice': '17141.27000000', 'indexPrice': '17157.15000000', 'markPrice': '1758.39194436'}], 'version': 2, 'baseLine': 3}}
#         on_packet event {'id': 'a2681f82-9f64-43c1-a2cb-62905bb1759e', 'topic': 'user.openapi.option.position', 'creationTime': 1669865158091, 'data': {'result': [{'symbol': 'BTC-27JAN23-17500-P', 'positionStatus': None, 'side': 'Buy', 'size': '0.04', 'entryPrice': '2000', 'sessionAvgPrice': '2000', 'markPrice': '1758.40110473', 'positionIM': '0', 'positionMM': '0', 'sessionUPL': '-9.663955810800', 'sessionRPL': '-0.20583513', 'createdAt': 1669864714684, 'updatedAt': 1669865158057}], 'version': 3, 'baseLine': 3, 'dataType': 'CHANGE'}}
#         on_packet event {'id': '8408cb56-8da8-4273-b528-77241a5c2e06', 'topic': 'user.openapi.greeks', 'creationTime': 1669865158096, 'data': {'result': [{'coin': 'BTC', 'totalDelta': '-0.028931040800', 'totalGamma': '0.000005459400', 'totalVega': '2.276921393700', 'totalTheta': '-0.793758436900'}], 'version': 53, 'baseLine': 1}}


#         on_packet event {'id': 'c8ec756b-0a97-4b52-bee9-e6513580563cab969ad2-7482-4ee6-9672-396bb1ce5bb1', 'topic': 'user.order', 'creationTime': 1669865215667, 'data': {'orderId': 'c8ec756b-0a97-4b52-bee9-e6513580563c', 'symbol': 'BTC-27JAN23-17500-P', 'orderStatus': 'New', 'side': 'Sell', 'orderPrice': '1460', 'orderAllSize': '0.02', 'orderFilledSize': '0', 'orderRemainingSize': '0.02', 'orderType': 'Market', 'orderTime': '1669865215660', 'timeInForce': 'ImmediateOrCancel', 'createTimeStamp': '1669865215660', 'updateTimeStamp': '1669865215664', 'execType': 'newed', 'version': '1416241817', 'placeMode': 'advanced', 'placeType': 'price', 'userId': '540325'}}
#         on_packet event {'id': 'c8ec756b-0a97-4b52-bee9-e6513580563ce46f50dc-eab3-4eb8-b868-9b56a5ff871b', 'topic': 'user.order', 'creationTime': 1669865215667, 'data': {'orderId': 'c8ec756b-0a97-4b52-bee9-e6513580563c', 'symbol': 'BTC-27JAN23-17500-P', 'orderStatus': 'Filled', 'side': 'Sell', 'orderPrice': '1540', 'orderAllSize': '0.02', 'orderFilledSize': '0.02', 'orderRemainingSize': '0', 'orderType': 'Market', 'orderTime': '1669865215660', 'timeInForce': 'ImmediateOrCancel', 'createTimeStamp': '1669865215660', 'updateTimeStamp': '1669865215664', 'execType': 'filled', 'version': '1416241817', 'placeMode': 'advanced', 'placeType': 'price', 'userId': '540325'}}
#         on_packet event {'id': 'b5db0364-1f20-4f89-bd94-304aa4bf8c1c', 'topic': 'user.openapi.option.position', 'creationTime': 1669865215695, 'data': {'result': [{'symbol': 'BTC-27JAN23-17500-P', 'positionStatus': None, 'side': 'Buy', 'size': '0.02', 'entryPrice': '2000', 'sessionAvgPrice': '2000', 'markPrice': '1760.18819688', 'positionIM': '0', 'positionMM': '0', 'sessionUPL': '-4.796236062400', 'sessionRPL': '-9.50875331', 'createdAt': 1669864714684, 'updatedAt': 1669865215675}], 'version': 4, 'baseLine': 3, 'dataType': 'CHANGE'}}
#         on_packet event {'id': '3401a0ce-74f1-466c-9e75-b16f17cd45f4', 'topic': 'user.openapi.option.trade', 'creationTime': 1669865215697, 'data': {'result': [{'orderId': 'c8ec756b-0a97-4b52-bee9-e6513580563c', 'orderLinkId': '', 'tradeId': 'cee5fd08-5c44-529c-83ca-5cdc20c5d168', 'symbol': 'BTC-27JAN23-17500-P', 'side': 'Sell', 'execPrice': '1540.00000000', 'execQty': '0.02', 'execFee': '0.10291818', 'feeRate': '0.00030000000000000000', 'tradeTime': 1669865215664, 'lastLiquidityInd': 'TAKER', 'execType': 'TRADE', 'blockTradeId': '', 'iv': '0.49504963026445203', 'markIv': '0.5764', 'underlyingPrice': '17138.83000000', 'indexPrice': '17153.03000000', 'markPrice': '1759.95885364'}], 'version': 3, 'baseLine': 3}}
#         on_packet event {'id': 'ab104faa-933c-4655-91c0-c04f71acc6d3', 'topic': 'user.openapi.greeks', 'creationTime': 1669865215702, 'data': {'result': [{'coin': 'BTC', 'totalDelta': '-0.033851076800', 'totalGamma': '0.000006479200', 'totalVega': '2.547389429600', 'totalTheta': '-0.930225645200'}], 'version': 55, 'baseLine': 1}}
#         on_packet event {'id': 'eab815f5-3def-4058-a38f-482f0f4bfcbb', 'topic': 'user.openapi.option.position', 'creationTime': 1669865229717, 'data': {'result': [{'symbol': 'BTC-27JAN23-17500-P', 'positionStatus': None, 'side': 'Buy', 'size': '0.0200', 'entryPrice': '2000.00000000', 'sessionAvgPrice':
#         '2000.00000000', 'markPrice': '1758.28185068', 'positionIM': '0', 'positionMM': '0', 'sessionUPL': '-4.834362986400', 'sessionRPL': '-9.50875331', 'createdAt': 1669864714684, 'updatedAt': 1669865215675}, {'symbol': 'BTC-31MAR23-16000-P', 'positionStatus': None, 'side': 'Buy', 'size': '0.0400', 'entryPrice': '2740.00000000', 'sessionAvgPrice': '1865.60914436', 'markPrice': '1854.78018521', 'positionIM': '0', 'positionMM': '0', 'sessionUPL':
#         '-35.408792591600', 'sessionRPL': '-0.19858476', 'createdAt': 1668679783703, 'updatedAt': 1668686424724}], 'version': 4, 'baseLine': 3, 'dataType':
#         'NEW'}}
#         """
#         # print(f"Bybit Option Trade Websocket - on_packet event {packet}")
#         # if 'auth' in packet:
#         #     self._logged_in = True if packet['auth'] == "success" else False
#         #     if self._logged_in:
#         #         print(f"Bybit Future Trade Websocket - successfully logined - {packet}")
#         #     else:
#         #         print(f"Bybit Future Trade Websocket auth failed - on_packet event {packet}")
#         #     return

#         if "topic" not in packet:
#             logger.info(f"the other packet without topic: {packet}")
#             return

#         if "ping" in packet:
#             logger.info(f"healthcheck message {packet}")
#             return

#         if not packet:
#             logger.debug(f"unknown packet event {packet}")
#             return

#         # return

#         if packet["topic"] == "user.openapi.greeks":
#             # self.on_greeks(packet)
#             pass

#         elif packet["topic"] == "user.openapi.option.position":
#             self.on_position(packet)
#         elif packet["topic"] == "user.order":
#             self.on_order(packet)
#         elif packet["topic"] == "user.openapi.option.trade":
#             self.on_trade(packet)
#         else:
#             logger.info(f"the other packet type: {packet}")

#     def on_greeks(self, packet: dict) -> None:
#         """
#         on_packet event {'id': '716d8af8-35d5-4e2f-91e3-db5de7a543f4', 'topic': 'user.openapi.greeks', 'creationTime': 1669864714711, 'data': {'result': [{'coin': 'BTC', 'totalDelta': '-0.014237964000', 'totalGamma': '0.000002402000', 'totalVega': '1.465132609600', 'totalTheta': '-0.384793846800'}], 'version': 51, 'baseLine': 1}}

#         """
#         logger.debug(f"packet on_account: {packet}")

#         for d in packet["data"]:
#             account = {
#                 "symbol": d["coin"],
#                 "balance": float(d["wallet_balance"]),
#                 "available": float(d["available_balance"]),
#                 "frozen": float(d["wallet_balance"]) - float(d["available_balance"]),
#                 "exchange": EXCHANGE,
#                 "asset_type": self.DATA_TYPE.upper(),
#             }

#             if account["symbol"] and self.on_account_callback:
#                 self.on_account_callback(account)

#     def on_order(self, packet: dict) -> None:
#         """
#         on_packet event {'id': '1b86d79a-d4be-4885-95ed-15df1ea656323c202ae8-5860-496f-97b1-33546c5cf4b0', 'topic': 'user.order', 'creationTime': 1669864668474, 'data': {'orderId': '1b86d79a-d4be-4885-95ed-15df1ea65632', 'symbol': 'BTC-27JAN23-17500-P', 'orderStatus': 'New', 'side': 'Buy', 'orderPrice': '1400', 'orderAllSize': '0.02', 'orderFilledSize': '0', 'orderRemainingSize': '0.02', 'orderType': 'Limit', 'orderTime': '1669864668467', 'timeInForce': 'GoodTillCancel', 'createTimeStamp': '1669864668467', 'updateTimeStamp': '1669864668471', 'execType': 'newed', 'version': '1416240863', 'placeMode': 'advanced', 'placeType': 'price', 'userId': '540325'}}
#         on_packet event {'id': '359f054e-6a99-4e0f-984e-6e1d1ad2514a85d99426-17eb-47f1-938b-4524a4a38834', 'topic': 'user.order', 'creationTime': 1669864714686, 'data': {'orderId': '359f054e-6a99-4e0f-984e-6e1d1ad2514a', 'symbol': 'BTC-27JAN23-17500-P', 'orderStatus': 'Filled', 'side': 'Buy', 'orderPrice': '2000', 'orderAllSize': '0.03', 'orderFilledSize': '0.03', 'orderRemainingSize': '0', 'orderType': 'Market', 'orderTime': '1669864714678', 'timeInForce': 'ImmediateOrCancel', 'createTimeStamp': '1669864714678', 'updateTimeStamp': '1669864714682', 'execType': 'filled', 'version': '1416240947', 'placeMode': 'advanced', 'placeType': 'price', 'userId': '540325'}}
#         """
#         logger.debug(f"packet on_order: {packet}")
#         data: dict = packet["data"]

#         order = models.OrderSchema()  # .model_construct(data)
#         order = order.from_bybit_to_form(data, datatype=self.DATA_TYPE)

#         if self.on_order_callback:  # and order_list:
#             # self.on_order_callback(order_list)
#             self.on_order_callback(order)

#     def on_trade(self, packet: dict) -> None:
#         """
#         on_packet event {'id': '19764687-e019-4266-8e6f-26d004ef8ef1', 'topic': 'user.openapi.option.trade', 'creationTime': 1669864714717, 'data': {'result': [{'orderId': '359f054e-6a99-4e0f-984e-6e1d1ad2514a', 'orderLinkId': '', 'tradeId': 'a7f14616-dc08-59da-bbfc-bf20d687553e', 'symbol': 'BTC-27JAN23-17500-P', 'side': 'Buy', 'execPrice': '2000.00000000', 'execQty': '0.03', 'execFee': '0.15436368', 'feeRate': '0.00030000000000000000', 'tradeTime': 1669864714682, 'lastLiquidityInd': 'TAKER', 'execType': 'TRADE', 'blockTradeId': '', 'iv': '0.6647320404763883', 'markIv': '0.5763', 'underlyingPrice': '17137.03000000', 'indexPrice': '17151.52000000', 'markPrice': '1760.71756271'}], 'version': 1, 'baseLine': 3}}

#         """
#         logger.debug(f"packet on_trade: {packet}")
#         datas: dict = packet["data"]["result"]

#         trade_list = list()
#         for data in datas:
#             trade = models.TradeSchema()  # .model_construct(data)
#             trade = trade.from_bybit_to_form(data, datatype=self.DATA_TYPE)
#             trade_list.append(trade)

#             if self.on_trade_callback:  # and trade_list:
#                 # self.on_trade_callback(trade_list)
#                 self.on_trade_callback(trade)

#     def on_position(self, packet: dict) -> None:
#         """
#         on_packet event {'id': 'a7ec90b6-440c-4452-8686-6e2171dce3d1', 'topic': 'user.openapi.option.position', 'creationTime': 1669864347108, 'data': {'result': [{'symbol': 'BTC-31MAR23-16000-P', 'positionStatus': None, 'side': 'Buy', 'size': '0.0400', 'entryPrice': '2740.00000000', 'sessionAvgPrice':
#         '1865.60914436', 'markPrice': '1862.62334484', 'positionIM': '0', 'positionMM': '0', 'sessionUPL': '-35.095066206400', 'sessionRPL': '-0.19858476',
#         'createdAt': 1668679783703, 'updatedAt': 1668686424724}], 'version': 1, 'baseLine': 3, 'dataType': 'NEW'}}

#         on_packet event {'id': '514da1e8-49e4-452b-ba28-a6a7e0ed550b', 'topic': 'user.openapi.option.position', 'creationTime': 1669864714705, 'data': {'result': [{'symbol': 'BTC-27JAN23-17500-P', 'positionStatus': None, 'side': 'Buy', 'size': '0.03', 'entryPrice': '2000.00000000', 'sessionAvgPrice': '2000.00000000', 'markPrice': '1760.65341487', 'positionIM': '0', 'positionMM': '0', 'sessionUPL': '-7.180397553900', 'sessionRPL': '-0.15436368', 'createdAt': 1669864714684, 'updatedAt': 1669864714692}], 'version': 2, 'baseLine': 3, 'dataType': 'CHANGE'}}

#         """
#         logger.debug(f"packet on_position: {packet}")
#         datas: dict = packet["data"]["result"]
#         datatype = packet["data"]["dataType"]
#         position_list = list()
#         for data in datas:
#             # position_value = float(data["position_value"])
#             # if not position_value:
#             #     continue
#             position = models.PositionSchema()  # .model_construct(data)
#             position = position.from_bybit_to_form(data, datatype=self.DATA_TYPE)

#             position_list.append(position)
#             if self.on_position_callback:  # and position_list:
#                 # self.on_position_callback(position_list)
#                 # print(f">>>>>> [{datatype}]")
#                 self.on_position_callback(position)
