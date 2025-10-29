# import base64
# import hmac
# import json
# import time
# import urllib
# from collections import defaultdict
# from copy import copy
# from datetime import datetime

# # from threading import Lock
# from typing import Any, Callable

# from ..utils import models
# from ..utils.client_rest import Response, RestClient
# from ..utils.client_ws import WebsocketClient
# from ..utils.logger import get_logger
# from ..utils.models import UTC_TZ

# logger = get_logger(default_level="warning")

# BITGET_API_HOST = "https://api.bitget.com"

# BITGET_WS_SPOT_HOST = "wss://ws.bitget.com/spot/v1/stream"
# BITGET_WS_CONTRACT_HOST = "wss://ws.bitget.com/mix/v1/stream"

# EXCHANGE = "BITGET"

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

# KBAR_SPOT_INTERVAL: dict[str, str] = {
#     # 1min (1 minute)
#     # 5min (5 minutes)
#     # 15min (15 minutes)
#     # 30min (30 minutes)
#     # 1h (1 hour)
#     # 4h (4 hours)
#     # 6h (6 hours)
#     # 12h (12 hours)
#     # 1day (1 day)
#     # 3day (3 days)
#     # 1week (1 week)
#     # 1M (monthly line)
#     # 6Hutc (UTC0 6 hour)
#     # 12Hutc (UTC0 12 hour)
#     # 1Dutc (UTC0 1day)
#     # 3Dutc (UTC0 3rd)
#     # 1Wutc (UTC0 weekly)
#     # 1Mutc (UTC0 monthly)
#     "1m": "1min",
#     "5m": "5min",
#     "15m": "15min",
#     "30m": "30min",
#     "1h": "1h",
#     "4h": "4h",
#     "6h": "6h",
#     "12h": "12h",
#     "1d": "1day",
#     "3d": "3day",
#     "1w": "1week",
#     "1M": "1M",
# }
# KBAR_CONTRACT_INTERVAL: dict[str, str] = {
#     # 1m(1minute)
#     # 3m(3minute)
#     # 5m(5minute)
#     # 15m(15minute)
#     # 30m(30minute)
#     # 1H(1hour)
#     # 2H (2hour)
#     # 4H (4hour)
#     # 6H (6hour)
#     # 12H (12hour)
#     # 1D (1day)
#     # 3D (3day)
#     # 1W(1week)
#     # 1M (1month)
#     # 6Hutc (UTC0 6hour)
#     # 12Hutc (UTC0 12hour)
#     # 1Dutc (UTC0 1day)
#     # 3Dutc (UTC0 3day)
#     # 1Wutc (UTC0 1 week)
#     # 1Mutc (UTC0 1 month)
#     "1m": "1m",
#     "3m": "3m",
#     "5m": "5m",
#     "15m": "15m",
#     "30m": "30m",
#     "1h": "1H",
#     "2h": "2H",
#     "4h": "4H",
#     "6h": "6H",
#     "12h": "12H",
#     "1d": "1D",
#     "3d": "3D",
#     "1w": "1W",
#     "1M": "1M",
# }


# MAPPING_CHANNEL: dict[str, str] = {
#     ### public
#     "depth": "books",
#     "kline": "candle",
#     "ticker": "ticker",
#     ### private
#     # "trades": "orders",
#     "orders": "orders",
#     "account": "account",
#     "position": "positions",
# }
# KBAR_INTERVAL_REV: dict[str, str] = {
#     "1m": "1m",
#     "3m": "3m",
#     "5m": "5m",
#     "15m": "15m",
#     "30m": "30m",
#     "1H": "1h",
#     "2H": "2h",
#     "4H": "4h",
#     "6H": "6h",
#     "12H": "12h",
#     "1D": "1d",
#     "3D": "3d",
#     "1W": "1w",
#     "1M": "1M",
# }


# def remove_none(payload):
#     """
#     Remove None value from payload
#     """
#     return {k: v for k, v in payload.items() if v is not None}


# class BitgetClient(RestClient):
#     """
#     spot => BTCUSDT_SPBL
#     linear  ==> BTCUSDT_UMCBL & BTCPERP_CMCBL || BTCUSDT_SUMCBL & BTCPERP_SCMCBL
#     inverse ==> BTCUSD_DMCBL & BTCUSD_DMCBL_230929 || BTCUSD_SDMCBL & BTCUSD_SDMCBL_230929

#     productType
#         umcbl USDT perpetual contract
#         dmcbl Universal margin perpetual contract
#         cmcbl USDC perpetual contract
#         sumcbl USDT simulation perpetual contract
#         sdmcbl Universal margin simulation perpetual contract
#         scmcbl USDC simulation perpetual contract

#     """

#     BROKER_ID = ""
#     DATA_TYPE = "spot"

#     def __init__(
#         self,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         datatype: str = "spot",
#         is_testnet: bool = False,
#         **kwargs,
#     ) -> None:
#         # url_base: str = None,
#         super().__init__(BITGET_API_HOST, proxy_host, proxy_port)

#         assert datatype in ["spot", "linear", "inverse"]
#         self.is_testnet = is_testnet
#         self.DATA_TYPE = datatype
#         self.product_type = None
#         if self.DATA_TYPE == "linear":
#             if self.is_testnet:
#                 self.product_type = "sumcbl"
#             else:
#                 self.product_type = "umcbl"
#         elif self.DATA_TYPE == "inverse":
#             if self.is_testnet:
#                 self.product_type = "sdmcbl"
#             else:
#                 self.product_type = "dmcbl"
#         # print(self.product_type)
#         # sys.exit()
#         self.order_count: int = 0
#         # self.order_count_lock: Lock = Lock()
#         self.connect_time: int = 0
#         self.key: str = ""
#         self.secret: str = ""
#         self.time_offset: int = 0
#         self.recv_window: int = 5000
#         self.unified = False

#     def _get_margin_coin(self, symbol):
#         margin_coin = None
#         if self.DATA_TYPE == "linear":
#             if self.product_type in ["cmcbl", "scmcbl"]:
#                 margin_coin = f"{'S' if self.product_type.startswith('s') else ''}USDC"
#             else:
#                 margin_coin = f"{'S' if self.product_type.startswith('s') else ''}USDT"
#         elif self.DATA_TYPE == "inverse":
#             margin_coin = symbol.split("_")[0].replace("USD", "")
#         return margin_coin

#     def info(self):
#         return "Bitget REST API start"

#     """
#     def query_time(self):
#         # resp = self.request("GET", "/spot/v1/time")
#         resp = self.query("GET", "/spot/v1/time")
#         local_time = int(time.time() * 1000)
#         server_time = int(float(resp.data["result"]["serverTime"]))
#         self.time_offset = local_time - server_time
#         # print(f"local_time {local_time}")
#         # print(f"server_time {server_time}")
#         # print(f"timeoffset {self.time_offset}")
#         return server_time
#     """

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

#     def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
#         # response = self.request(method="GET", path=path,  params=params)
#         response = self.query(method="GET", path=path, params=params)
#         # print(f"[GET] response: {response.text}({response.status_code})")
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

#     # def retry(self, response: Response, request: Request):
#     #     path = request.path
#     #     data = response.data
#     #     if "usdc" in path:
#     #         ret_code = "retCode"
#     #         ret_msg = "retMsg"
#     #     else:
#     #         ret_code = "ret_code"
#     #         ret_msg = "ret_msg"

#     #     error_msg = f"{data.get(ret_msg)} (ErrCode: {data.get(ret_code)})"
#     #     err_delay = self._retry_delay
#     #     if data.get(ret_code) in RETRY_CODES:

#     #         # 10002, recv_window error; add 2.5 seconds and retry.
#     #         if data[ret_code] == 10002:
#     #             error_msg += ". Added 2.5 seconds to recv_window"
#     #             self.recv_window += 2500

#     #         # 10006, ratelimit error; wait until rate_limit_reset_ms and retry.
#     #         elif data[ret_code] == 10006:
#     #             print(f"{error_msg}. Ratelimited on current request. Sleeping, then trying again. Request: {path}")

#     #             # Calculate how long we need to wait.
#     #             limit_reset = data["rate_limit_reset_ms"] / 1000
#     #             reset_str = time.strftime("%X", time.localtime(limit_reset))
#     #             err_delay = int(limit_reset) - int(time.time())
#     #             error_msg = f"Ratelimit will reset at {reset_str}. Sleeping for {err_delay} seconds"
#     #         print(error_msg)
#     #         time.sleep(err_delay)
#     #         return True
#     #     return False

#     def connect(self, key: str, secret: str, passphrase: str) -> None:
#         """connect exchange server"""
#         self.key = key
#         self.secret = secret
#         self.passphrase = passphrase
#         self.connect_time = int(datetime.now(UTC_TZ).strftime("%y%m%d%H%M%S"))
#         # self.query_time()
#         self.start()
#         # self.query_permission()
#         logger.debug(self.info())
#         return True

#     @staticmethod
#     def _sign(message, secret_key):
#         mac = hmac.new(
#             bytes(secret_key, encoding="utf8"),
#             bytes(message, encoding="utf-8"),
#             digestmod="sha256",
#         )
#         d = mac.digest()
#         return base64.b64encode(d)

#     @staticmethod
#     def pre_hash(timestamp, method, request_path, body):
#         if body:
#             return str(timestamp) + str.upper(method) + request_path + body
#         else:
#             return str(timestamp) + str.upper(method) + request_path

#     def sign(self, request):
#         """ """
#         if "market" in request.path or "public" in request.path:
#             return request

#         if not request.params:
#             request.params = dict()
#         if not request.data:
#             request.data = dict()

#         timestamp: int = int(time.time() * 1000)
#         # request.params["timestamp"] = timestamp

#         query: str = urllib.parse.urlencode(sorted(request.params.items()))

#         # print(f"query: {query}; data_query:{data_query}")
#         if query:
#             path: str = request.path + "?" + query
#             request.path = path
#         request.params = {}

#         # signature: bytes = hmac.new(self.secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
#         sign_message = self.pre_hash(
#             timestamp=str(timestamp),
#             method=request.method,
#             request_path=request.path,
#             body=json.dumps(request.data) if request.data else None,
#         )
#         signature = self._sign(message=sign_message, secret_key=self.secret)
#         # print(f"signature: {signature}")

#         # add header
#         headers = {
#             "Content-Type": "application/json",
#             "ACCESS-KEY": self.key,
#             "ACCESS-SIGN": signature,
#             "ACCESS-PASSPHRASE": self.passphrase,
#             "ACCESS-TIMESTAMP": str(timestamp),
#             "locale": "en-US",  # zh-CN
#         }
#         request.headers = headers
#         # print(f">>> [DEBUG] sign_message:{sign_message}; signature:{signature} || method:{request.method}; path: {request.path}; params: {request.params}; data: {request.data}; headers:{request.headers}\n")

#         return request

#     def _process_response(self, response: Response) -> dict:
#         """
#         {
#             "retCode": 0,
#             "retMsg": "OK",
#             "result": {
#             },
#             "retExtInfo": {},
#             "time": 1671017382656
#         }
#         """
#         try:
#             data = response.data
#             # print(data,'!!!!', response.ok)
#         except ValueError:
#             print(
#                 f"[BITGET] in _process_response, something went wrong when parsing data. Raw data:{response.text}"
#             )
#             logger.debug(response.data())
#             raise
#         else:
#             # logger.debug(data)
#             # print(f'DEBUG:{data} | response.ok:{response.ok} | data["data"]:{data["data"]}')
#             if response.ok and isinstance(data, list):
#                 ### future - candle data
#                 return models.CommonResponseSchema(
#                     success=True, error=False, data=data, msg="query ok"
#                 )

#             if not response.ok or (
#                 response.ok and not data["data"] and data["code"] != "00000"
#             ):
#                 payload_data = models.CommonDataSchema(
#                     status_code=response.status_code, msg=dict(data)
#                 )
#                 if not data.get("msg"):
#                     print(
#                         f"[BITGET] request without msg msg. Raw data::{response.text}"
#                     )
#                 return models.CommonResponseSchema(
#                     success=False,
#                     error=True,
#                     data=payload_data.dict(),
#                     msg=data.get("msg", "No return msg from bitget exchange."),
#                 )
#             elif response.ok and data["data"] is None and data["code"] == "00000":
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
#                     success=True, error=False, data=data["data"], msg="query ok"
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
#                 result = data.from_bitget_to_form(msg, datatype=self.DATA_TYPE)
#                 payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.OrderSchema()
#             payload = payload.from_bitget_to_form(
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
#                 result = data.from_bitget_to_form(msg, datatype=self.DATA_TYPE)
#                 if result:
#                     payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.TradeSchema()
#             payload = payload.from_bitget_to_form(
#                 common_response.data, datatype=self.DATA_TYPE
#             )
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def _regular_symbols_payload(
#         self,
#         common_response: models.CommonResponseSchema,
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             # payload = list()
#             payload = dict()
#             for msg in common_response.data:
#                 data = models.SymbolSchema()
#                 result = data.from_bitget_to_form(msg, datatype=self.DATA_TYPE)
#                 if result:
#                     # payload.append(result)
#                     payload[result["symbol"]] = result
#         elif isinstance(common_response.data, dict):
#             payload = models.SymbolSchema()
#             payload = payload.from_bitget_to_form(
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
#                 "ts",
#                 "open",
#                 "high",
#                 "low",
#                 "close",
#                 "baseVol",
#                 "quoteVol",
#             ]
#             return dict(zip(key, data))

#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 if self.product_type:
#                     msg = key_data(msg)
#                 data = models.HistoryOHLCSchema()
#                 result = data.from_bitget_to_form(msg, extra, datatype=self.DATA_TYPE)
#                 payload.append(result)
#         elif isinstance(common_response.data, dict):
#             if self.product_type:
#                 common_response.data = key_data(common_response.data)
#             payload = models.HistoryOHLCSchema()
#             payload = payload.from_bitget_to_form(
#                 common_response.data, extra, datatype=self.DATA_TYPE
#             )
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def _regular_ticker_payload(
#         self, common_response: models.CommonResponseSchema, symbol: str
#     ) -> dict:
#         payload = list()
#         if isinstance(common_response.data, list):
#             for msg in common_response.data:
#                 data = models.TickerSchema()
#                 payload = data.from_bitget_to_form(
#                     msg, symbol=symbol, datatype=self.DATA_TYPE
#                 )
#                 # payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.TickerSchema()
#             payload = payload.from_bitget_to_form(
#                 common_response.data, symbol=symbol, datatype=self.DATA_TYPE
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
#                 result = data.from_bitget_to_form(
#                     msg,
#                     transfer_type=transfer_type,
#                     inout=inout,
#                 )
#                 payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.TransferSchema()
#             payload = payload.from_bitget_to_form(
#                 common_response.data,
#                 transfer_type=transfer_type,
#                 inout=inout,
#             )
#         else:
#             common_response.msg = "payload error in fetch transfer data"
#             Exception("payload error in fetch transfer data")

#         common_response.data = payload
#         return common_response

#     def _regular_account_payload(
#         self, common_response: models.CommonResponseSchema, account_type: str
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             # payload = list()
#             payload = dict()
#             for msg in common_response.data:
#                 data = models.AccountSchema()
#                 result = data.from_bitget_to_form(
#                     msg, datatype=self.DATA_TYPE, account_type=account_type
#                 )
#                 if result:
#                     # payload.append(result)
#                     payload[result["symbol"]] = result
#         elif isinstance(common_response.data, dict):
#             payload = models.AccountSchema()
#             payload = payload.from_bitget_to_form(
#                 common_response.data, datatype=self.DATA_TYPE, account_type=account_type
#             )
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def _regular_position_payload(
#         self,
#         common_response: models.CommonResponseSchema,
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 data = models.PositionSchema()
#                 result = data.from_bitget_to_form(msg, datatype=self.DATA_TYPE)
#                 if result:
#                     payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.PositionSchema()
#             payload = payload.from_bitget_to_form(
#                 common_response.data, datatype=self.DATA_TYPE
#             )
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def _regular_incomes_payload(
#         self,
#         common_response: models.CommonResponseSchema,
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 data = models.IncomeSchema()
#                 result = data.from_bitget_to_form(msg, datatype=self.DATA_TYPE)
#                 payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.IncomeSchema()
#             payload = payload.from_bitget_to_form(
#                 common_response.data, datatype=self.DATA_TYPE
#             )
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     # def _aggregate_trades(self, trades):
#     #     ret = defaultdict(dict)
#     #     for i in trades:
#     #         ori_order_id = i["ori_order_id"]
#     #         if ori_order_id in ret:
#     #             ret[ori_order_id]["quantity"] += i["quantity"]
#     #             ret[ori_order_id]["commission"] += i["commission"]
#     #         else:
#     #             ret[ori_order_id] = i
#     #     return list(ret.values())

#     def query_permission(self) -> dict:
#         """ """
#         api_info = self._get("/api/spot/v1/account/getInfo")
#         # print(f">>>> api_info:{api_info}")
#         if api_info.success:
#             payload = models.PermissionSchema()  # .model_construct(api_info.data)
#             api_info.data = payload.from_bitget_to_form(api_info.data)
#         return api_info

#     def query_account(self) -> dict:
#         """ """
#         payload = {}
#         if self.product_type:
#             payload["productType"] = self.product_type
#         balance = self._get(
#             "/api/mix/v1/account/accounts"
#             if self.product_type
#             else "/api/spot/v1/account/assets",
#             payload,
#         )
#         # print(f">>> balance:{balance}")
#         if self.product_type:
#             account_type = "CONTRACT"
#         else:
#             account_type = "SPOT"
#         # print(f">>> product_type:{self.product_type}; account_type:{account_type}; ")
#         if balance.success:
#             if balance.data:
#                 return self._regular_account_payload(balance, account_type)
#             else:
#                 balance.msg = "query account is empty"

#         return balance

#     def send_order(
#         self,
#         symbol: str,
#         quantity: float,
#         side: str,
#         order_type=None,
#         price=None,
#         position_side: str = "net",
#         reduce_only: bool = False,
#         **kwargs,
#     ) -> dict:
#         """
#         contract
#         side:
#                 open_long
#                 open_short
#                 close_long
#                 close_short
#                 buy_single buy under single_hold mode
#                 sell_single sell under single_hold mode
#         timeInForceValue
#             normal 成交为止 订单会一直有效，直到被成交或者取消。
#             post_only 只做marker
#             fok 无法全部立即成交就撤销
#             ioc 无法立即成交的部分就撤销
#         orderType
#             limit
#             market

#         Error:
#             {'code': '400172', 'msg': 'Position direction api.orderType.mismatch', 'requestTime': 1683863766486, 'data': None}
#             {'status_code': 400, 'msg': {'code': '400172', 'msg': 'Position direction api.orderType.mismatch', 'requestTime': 1683863355676, 'data': None}}

#         """
#         if self.product_type:
#             payload = {
#                 "symbol": symbol.upper(),
#                 "marginCoin": self._get_margin_coin(symbol),
#                 "side": str(side).lower() + "_single"
#                 if position_side == "net"
#                 else ("close_short" if str(side).lower() == "buy" else "close_long")
#                 if reduce_only
#                 else ("open_long" if str(side).lower() == "buy" else "open_short"),
#                 "size": quantity,  # contract: base coin
#                 "clientOid": self.new_order_id(),
#                 "timeInForceValue": "normal",
#             }
#             if reduce_only:
#                 payload["reduceOnly"] = reduce_only
#             url = "/api/mix/v1/order/placeOrder"
#         else:
#             payload = {
#                 "symbol": symbol.upper(),
#                 "side": str(side).lower().capitalize(),
#                 "quantity": quantity,  # SPOT: Order quantity, base coin(BTC) when orderType is limit; quote coin(USDT) when orderType is market
#                 "clientOrderId": self.new_order_id(),
#                 "force": "normal",
#             }

#         if not order_type:
#             if not price:
#                 order_type = "market"  # market
#             else:
#                 order_type = "limit"
#                 # order_type = "limited"

#             payload["orderType"] = order_type
#         else:
#             order_type = order_type.lower()

#         if order_type.upper() == "LIMIT":
#             payload["price"] = price
#         elif order_type.upper() == "MARKET":
#             if (
#                 self.DATA_TYPE.lower() == "spot"
#             ):  # if it is market order, it is qty needs to be in USDT
#                 last_price = self.query_prices(symbol)["data"]["price"]
#                 payload["quantity"] = round(last_price * quantity, 4)
#                 print(
#                     f"payload: {payload} ????in bitget spot send_order???? last_price: {last_price}"
#                 )

#         url = "/api/spot/v1/trade/orders"
#         orders_response = self._post(url, payload)

#         print(f">>> palce_order_response: {orders_response} | payload:{payload}")
#         if orders_response.success:
#             if orders_response.data:
#                 result = (
#                     models.SendOrderResultSchema()
#                 )  # .model_construct(orders_response.data)
#                 orders_response.data = result.from_bitget_to_form(
#                     orders_response.data, symbol=symbol, datatype=self.DATA_TYPE
#                 )
#             else:
#                 orders_response.msg = "query sender order is empty"

#         return orders_response

#     def cancel_order(self, order_id: str, symbol: str) -> dict:
#         """ """
#         payload = {
#             "symbol": symbol,
#             "orderId": order_id,
#         }
#         if self.product_type:
#             payload["marginCoin"] = ""
#         # return self._post("/api/spot/v1/trade/cancel-order-v2", payload)
#         return self._post(
#             "/api/mix/v1/order/cancel-order"
#             if self.product_type
#             else "/api/spot/v1/trade/cancel-order",
#             payload,
#         )

#     def cancel_orders(self, symbol: str) -> dict:
#         """ """
#         payload = {"symbol": symbol}
#         return self._post("/api/spot/v1/trade/cancel-symbol-order", payload)

#     def query_order_status(self, order_id: str, symbol: str) -> dict:
#         """ """
#         payload = {}
#         if order_id:
#             payload["orderId"] = order_id
#         if symbol:
#             payload["symbol"] = symbol
#         if self.product_type:
#             order_status = self._get("/api/mix/v1/order/detail", payload)
#         else:
#             order_status = self._post("/api/spot/v1/trade/orderInfo", payload)
#         # print(f"query_open_orders: {order_status}")

#         if order_status.success:
#             if order_status.data:
#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"
#                 order_status.data = []

#         return order_status

#     def query_open_orders(self, symbol: str = None) -> dict:
#         """ """
#         payload = {}
#         if symbol:
#             payload["symbol"] = symbol
#         else:
#             if self.product_type:
#                 payload["productType"] = self.product_type
#                 ### TODO: inverse
#                 payload["marginCoin"] = (
#                     "SUSDT" if self.product_type.startswith("s") else "USDT"
#                 )
#             else:
#                 payload["symbol"] = ""  # for spot only

#         if self.product_type:
#             if symbol:
#                 order_status = self._get("/api/mix/v1/order/current", payload)
#             else:
#                 order_status = self._get("/api/mix/v1/order/marginCoinCurrent", payload)
#         else:
#             order_status = self._post("/api/spot/v1/trade/open-orders", payload)

#         # print(f"query_open_orders: {order_status} | payload:{payload}")
#         if order_status.success:
#             if order_status.data:
#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"
#                 order_status.data = []

#         return order_status

#     def query_all_orders(self, symbol: str = None, limit=500) -> dict:
#         """ """
#         payload = {}
#         if limit:
#             payload["limit"] = limit
#         if symbol:
#             payload["symbol"] = symbol

#         if self.product_type:
#             if limit:
#                 payload["pageSize"] = 100
#             payload["endTime"] = int(time.time() * 1000)
#             payload["startTime"] = payload["endTime"] - 2 * 365 * 24 * 60 * 60
#             if symbol:
#                 url = "/api/mix/v1/order/history"
#             else:
#                 payload["productType"] = self.product_type
#                 url = "/api/mix/v1/order/historyProductType"
#             order_list = []
#             order_status = self._get(url, payload)
#             # print(f">>> query_all_orders: {order_status}")
#             if order_status.success:
#                 if order_status.data["orderList"]:
#                     order_list += order_status.data["orderList"]
#                     # while order_status.data['nextFlag']:
#                     #     payload["lastEndId"] = order_status.data['endId']
#                     #     order_status = self._get(url, payload)
#                 order_status.data = order_list
#         else:
#             order_status = self._post("/api/spot/v1/trade/history", payload)

#         # print(f"query_all_orders: {order_status}")
#         # query_all_orders: success=True error=False data={'nextFlag': False, 'endId': '1030292933431111683', 'orderList': [{'symbol': 'BTCUSDT_UMCBL', 'size': 0.001, 'orderId': '1030292933431111683', 'clientOid': '1030292933468860416', 'filledQty': 0.001, 'fee': -0.0180501, 'price': None, 'priceAvg': 30083.5, 'state': 'filled', 'side': 'open_long', 'timeInForce': 'normal', 'totalProfits': 0.0, 'posSide': 'long', 'marginCoin': 'USDT', 'filledAmount': 30.0835, 'orderType': 'market', 'leverage': '20', 'marginMode': 'crossed', 'reduceOnly': False, 'enterPointSource': 'WEB', 'tradeSide': 'open_long', 'holdMode': 'double_hold', 'cTime': '1681366897933', 'uTime': '1681366898068'}]} msg='query ok'
#         # history:            {'nextFlag': False, 'endId': '1030292933431111683', 'orderList': [{'symbol': 'BTCUSDT_UMCBL', 'size': 0.001, 'orderId': '1030292933431111683', 'clientOid': '1030292933468860416', 'filledQty': 0.001, 'fee': -0.0180501, 'price': None, 'priceAvg': 30083.5, 'state': 'filled', 'side': 'open_long', 'timeInForce': 'normal', 'totalProfits': 0.0, 'posSide': 'long', 'marginCoin': 'USDT', 'filledAmount': 30.0835, 'orderType': 'market', 'leverage': '20', 'marginMode': 'crossed', 'reduceOnly': False, 'enterPointSource': 'WEB', 'tradeSide': 'open_long', 'holdMode': 'double_hold', 'cTime': '1681366897933', 'uTime': '1681366898068'}]}
#         # historyProductType: {'nextFlag': False, 'endId': '1030292933431111683', 'orderList': [{'symbol': 'BTCUSDT_UMCBL', 'size': 0.001, 'orderId': '1030292933431111683', 'clientOid': '1030292933468860416', 'filledQty': 0.001, 'fee': -0.0180501, 'price': None, 'priceAvg': 30083.5, 'state': 'filled', 'side': 'open_long', 'timeInForce': 'normal', 'totalProfits': 0.0, 'posSide': 'long', 'marginCoin': 'USDT', 'filledAmount': 30.0835, 'orderType': 'market', 'leverage': '20', 'marginMode': 'crossed', 'reduceOnly': False, 'enterPointSource': 'WEB', 'tradeSide': 'open_long', 'holdMode': 'double_hold', 'cTime': '1681366897933', 'uTime': '1681366898068'}]}
#         if order_status.success:
#             if order_status.data:
#                 # if 'orderList' in order_status.data:
#                 #     order_status.data = order_status.data['orderList']
#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"
#                 order_status.data = []

#         return order_status

#     def query_symbols(self, symbol: str = None) -> dict:
#         """ """
#         payload = {}
#         if symbol:
#             payload["symbol"] = symbol
#         if self.product_type:
#             payload["productType"] = self.product_type
#             symbols = self._get("/api/mix/v1/market/contracts", payload)
#         else:
#             if symbol:
#                 symbols = self._get("/api/spot/v1/public/product", payload)
#             else:
#                 symbols = self._get("/api/spot/v1/public/products", payload)

#         # print(f">>> symbols:{symbols}")
#         if symbols.success:
#             if symbols.data:
#                 symbols.data = symbols.data
#                 ret = self._regular_symbols_payload(symbols)
#                 if self.product_type and symbol:
#                     ret.data = ret.data.get(symbol, {})

#                 return ret
#             else:
#                 symbols.msg = "query symbol is empty"

#         return symbols

#     def query_trades(
#         self,
#         symbol: str = None,
#         start: datetime = None,
#         end: datetime = None,
#         limit=500,
#     ) -> dict:
#         """ """
#         payload = {
#             "limit": limit,
#         }
#         if symbol:
#             payload["symbol"] = symbol
#         else:
#             return models.CommonResponseSchema(
#                 success=False, error=True, data={}, msg="Please specify the symbol."
#             )
#         if self.product_type:
#             if start:
#                 payload["startTime"] = int(datetime.timestamp(start) * 1000)
#             if end:
#                 payload["endTime"] = int(datetime.timestamp(end) * 1000)
#             # payload['endTime'] = int(time.time()*1000)
#             # payload['startTime'] = payload['endTime']-2*365*24*60*60
#             if symbol:
#                 url = "/api/mix/v1/order/fills"
#             else:
#                 payload["productType"] = self.product_type
#                 url = "/api/mix/v1/order/allFills"
#             trades = self._get(url, remove_none(payload))
#             # trade_list = []
#             # if trades.success:
#             #     trade_list += trades.data
#             #     # while trades.data['nextFlag']:
#             #     #     payload["lastEndId"] = trades.data['endId']
#             #     #     trades = self._get(url, payload)
#             #     trades.data = trade_list
#         else:
#             trades = self._post("/api/spot/v1/trade/fills", remove_none(payload))

#         # print(f"[DEBUG] len trades:{len(trades.data)}; | first new trade:{trades.data[0]}; last new trade:{trades.data[-1]};")
#         # print(f"[DEBUG] len trades:{len(trades.data)}; | trades:{trades};")

#         # if start or end:
#         #     trades_next = copy(trades)
#         #     while trades_next.data.get("nextPageCursor"):
#         #         payload["cursor"] = trades_next.data.get("nextPageCursor")
#         #         trades_next = self._get("/v5/execution/list", remove_none(payload))
#         #         new_trades = trades_next.data["list"]
#         #         trades.data["list"] += new_trades
#         #         # print(f"    [DEBUG] len new trades:{len(new_trades)}; first new trade:{new_trades[0]['execTime']}; last new trade:{new_trades[-1]['execTime']}; nextPageCursor:{trades_next.data.get('nextPageCursor')} ...")

#         if trades.success:
#             if trades.data:
#                 ret_trades = self._regular_trades_payload(trades)
#                 # aggregate by ori_order_id
#                 # ret_trades.data = self._aggregate_trades(ret_trades["data"])
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
#         Kline interval. 1,3,5,15,30,60,120,240,360,720,D,M,W
#         """

#         def iteration_history(url, symbol, interval, data, payload, limit, retry=3):
#             retry_counter = 0
#             historical_prices_datalist = data.data
#             # logger.debug(interval)

#             if self.product_type:
#                 first_timestamp = int(historical_prices_datalist[0][0])
#                 start_timestamp = payload["endTime"]
#             else:
#                 first_timestamp = int(historical_prices_datalist[0]["ts"])
#                 start_timestamp = payload["after"]
#             interval_timestamp = first_timestamp - start_timestamp
#             # print(f"first_timestamp:{first_timestamp}({timestamp_to_datetime(first_timestamp)}) || start_timestamp:{start_timestamp}({timestamp_to_datetime(start_timestamp)}) ")
#             # logger.debug(payload)

#             while interval_timestamp > interval:
#                 if self.product_type:
#                     first_timestamp = int(historical_prices_datalist[0][0])
#                 else:
#                     first_timestamp = int(historical_prices_datalist[0]["ts"])
#                 interval_timestamp = first_timestamp - start_timestamp
#                 # print(f"first_timestamp {first_timestamp}({timestamp_to_datetime(first_timestamp)})")
#                 logger.debug(f"first_timestamp {first_timestamp}")
#                 logger.debug(f"interval_timestamp {interval_timestamp}")
#                 # logger.debug(historical_prices_datalist[0:5])

#                 payload["before"] = first_timestamp

#                 prices = self._get(url, remove_none(payload))
#                 if prices.error:
#                     break
#                 # prices.data = prices.data.get("list", [])
#                 logger.debug(prices.data[0:2])
#                 # print(f"[DEBUG] first 2: {prices.data[0:2]}")

#                 historical_prices_datalist.extend(prices.data)
#                 if self.product_type:
#                     historical_prices_datalist = [
#                         list(item)
#                         for item in set(
#                             tuple(x.items()) for x in historical_prices_datalist
#                         )
#                     ]
#                     historical_prices_datalist.sort(key=lambda k: k[0][0])
#                 else:
#                     historical_prices_datalist = [
#                         dict(item)
#                         for item in set(
#                             tuple(x.items()) for x in historical_prices_datalist
#                         )
#                     ]
#                     historical_prices_datalist.sort(key=lambda k: k["ts"])
#                 time.sleep(0.1)

#                 logger.debug(f"payload: {payload}")
#                 logger.debug(f"retry_counter: {retry_counter}")

#                 if len(prices.data) != limit:
#                     retry_counter += 1

#                 if retry_counter >= retry:
#                     break

#             data.data = historical_prices_datalist
#             logger.debug(f"data length: {len(historical_prices_datalist)}")
#             return data

#         # main
#         if self.DATA_TYPE not in ["spot", "linear", "inverse"]:
#             return models.CommonResponseSchema(
#                 success=False,
#                 error=True,
#                 data={},
#                 msg=f"{self.DATA_TYPE} is not support to fetch historical klines.",
#             )

#         if interval not in KBAR_SPOT_INTERVAL:
#             return models.CommonResponseSchema(
#                 success=False, error=True, data={}, msg="Invalid interval."
#             )

#         if self.product_type:
#             payload = {
#                 "symbol": symbol,
#                 "granularity": KBAR_CONTRACT_INTERVAL[interval],
#                 "limit": limit,
#                 "startTime": int(time.time() * 1000),  # - 2*365*24*60*60
#                 "endTime": int(time.time() * 1000),
#             }
#             if start:
#                 start = start.replace(tzinfo=None)
#                 start = UTC_TZ.localize(
#                     start
#                 )  # NOTE: bitget start and end timestamp default regard it as utc timestamp
#                 payload["startTime"] = int(
#                     datetime.timestamp(start) * 1000
#                 )  # NOTE: start

#             if end:
#                 end = end.replace(tzinfo=None)
#                 end = UTC_TZ.localize(end)
#                 payload["endTime"] = int(datetime.timestamp(end) * 1000)  # NOTE: end

#             url = "/api/mix/v1/market/candles"
#         else:
#             payload = {
#                 "symbol": symbol.upper(),
#                 "period": KBAR_SPOT_INTERVAL[interval],
#                 "limit": limit,
#             }
#             # ms - epoch time
#             if start:
#                 start = start.replace(tzinfo=None)
#                 start = UTC_TZ.localize(
#                     start
#                 )  # NOTE: bitget start and end timestamp default regard it as utc timestamp
#                 payload["after"] = int(datetime.timestamp(start) * 1000)  # NOTE: start
#             if end:
#                 end = end.replace(tzinfo=None)
#                 end = UTC_TZ.localize(end)
#                 payload["before"] = int(datetime.timestamp(end) * 1000)  # NOTE: end
#             url = "/api/spot/v1/market/candles"

#         historical_prices = self._get(url, remove_none(payload))

#         extra = {"interval": interval, "symbol": symbol}
#         # print(f"??? historical_prices: {historical_prices}")

#         if historical_prices.success:
#             if historical_prices.data:
#                 if self.product_type:
#                     historical_prices.data.sort(key=lambda k: k[0][0])
#                 else:
#                     historical_prices.data.sort(key=lambda k: k["ts"])
#                 # handle query time
#                 if "after" in payload or "endTime" in payload:
#                     interval = TIMEDELTA_MAPPING_SEC[interval] * 1000
#                     historical_prices = iteration_history(
#                         url=url,
#                         symbol=symbol,
#                         interval=interval,
#                         data=historical_prices,
#                         payload=payload,
#                         limit=limit,
#                     )
#                 return self._regular_historical_prices_payload(historical_prices, extra)
#             else:
#                 historical_prices.msg = "query historical prices is empty"

#         return historical_prices

#     def query_last_price(
#         self,
#         symbol: str,
#         interval: str = "1m",
#         limit: int = 1,
#     ) -> dict:
#         """
#         interval: 1,3,5,15,30,60,120,240,360,720,D,M,W
#         """
#         if self.product_type:
#             payload = {
#                 "symbol": symbol,
#                 "granularity": KBAR_CONTRACT_INTERVAL[interval],
#                 "limit": limit,
#                 "startTime": int(time.time() * 1000),
#                 "endTime": int(time.time() * 1000),
#             }
#             last_historical_prices = self._get("/api/mix/v1/market/candles", payload)
#         else:
#             payload = {
#                 "symbol": symbol,
#                 "period": KBAR_SPOT_INTERVAL[interval],
#                 "limit": limit,
#             }
#             last_historical_prices = self._get("/api/spot/v1/market/candles", payload)

#         extra = {"interval": interval, "symbol": symbol}
#         # print(f"last_historical_prices: {last_historical_prices}")

#         if last_historical_prices.success:
#             if last_historical_prices.data:
#                 # last_historical_prices.data = last_historical_prices.data.get("list", [])
#                 last_data = self._regular_historical_prices_payload(
#                     last_historical_prices, extra
#                 )
#                 if last_data.data and isinstance(last_data.data, list):
#                     last_data.data = last_data.data[0]
#                 return last_data
#             else:
#                 last_historical_prices.msg = "query latest historical prices is empty"

#         return last_historical_prices

#     def query_prices(self, symbol: str = None) -> dict:
#         """ """
#         payload = {}

#         if symbol:
#             payload["symbol"] = symbol.upper()
#         spots = self._get(
#             "/api/mix/v1/market/ticker"
#             if self.product_type
#             else "/api/spot/v1/market/ticker",
#             remove_none(payload),
#         )

#         # logger.debug(spots)
#         # print(f"spots:{spots}")
#         """
#         spots:success=True error=False data={'category': 'option', 'list': [{'symbol': 'BTC-31MAR23-20000-C', 'bid1Price': '0', 'bid1Size': '0', 'bid1Iv': '0', 'ask1Price': '4460', 'ask1Size': '0.04', 'ask1Iv': '0', 'lastPrice': '910', 'highPrice24h': '0', 'lowPrice24h': '0', 'markPrice': '5057.88495797', 'indexPrice': '24902.23', 'markIv': '0.7673', 'underlyingPrice': '24916.74', 'openInterest': '19.82', 'turnover24h': '2431.3', 'volume24h': '0.1', 'totalVolume': '2540', 'totalTurnover': '49881242', 'delta': '0.92604985', 'gamma': '0.00003495', 'vega': '7.31457284', 'theta': '-17.49516262', 'predictedDeliveryPrice': '0', 'change24h': '0'}]} msg='query ok'
#         """
#         if spots.success:
#             if spots.data:
#                 return self._regular_ticker_payload(spots, symbol)
#             else:
#                 spots.msg = "query spots is empty"

#         return spots

#     def query_transfer(
#         self,
#         start: datetime = None,
#         end: datetime = None,
#     ) -> dict:
#         """ """
#         return models.CommonResponseSchema(
#             success=False, error=True, data={}, msg="Please waiting for the update"
#         )

#     def query_internal_transfer(self) -> dict:
#         """ """
#         return models.CommonResponseSchema(
#             success=False, error=True, data={}, msg="Please waiting for the update"
#         )

#     def query_position(
#         self,
#         symbol: str = None,
#         settle_coin: str = None,
#         force_return: bool = False,
#         **kwargs,
#     ) -> dict:
#         """ """
#         if self.DATA_TYPE in ["spot"]:
#             return models.CommonResponseSchema(
#                 success=False,
#                 error=True,
#                 data={},
#                 msg="Spot is not support to query position.",
#             )

#         payload = {}

#         if symbol:
#             payload["symbol"] = symbol.upper()
#         elif settle_coin:
#             payload["marginCoin"] = settle_coin

#         if symbol:
#             position = self._get("/api/mix/v1/position/singlePosition", payload)
#         else:
#             payload["productType"] = self.product_type
#             position = self._get("/api/mix/v1/position/allPosition", payload)
#         # print(f"position: {position} | payload:{payload}")

#         if position.success:
#             if position.data:
#                 return self._regular_position_payload(position)
#             else:
#                 position.msg = "query linear position is empty"

#         return position

#     def set_position_mode(self, mode: str) -> dict:
#         """ """
#         if mode not in ["oneway", "hedge"]:
#             # raise ValueError("Invalid position_mode")
#             return models.CommonResponseSchema(
#                 success=False,
#                 error=True,
#                 data={},
#                 msg="Please specify the correct position mode.",
#             )

#         payload = {
#             "productType": self.product_type,
#             "holdMode": "single_hold"
#             if mode == "oneway"
#             else "double_hold",  # "single_hold" or "double_hold"
#         }

#         mode_payload = self._post(
#             "/api/mix/v1/account/setPositionMode", remove_none(payload)
#         )
#         print(f">>> set_position_mode:{mode_payload}")
#         # >>> set_position_mode:success=True error=False data={'marginCoin': 'SUSDT', 'symbol': 'SBTCSUSDT_SUMCBL', 'dualSidePosition': True} msg='query ok'

#         if mode_payload.success:
#             coin = None
#             symbol = None
#             mode_payload.data = models.PositionModeSchema(
#                 mode=mode, coin=coin, symbol=symbol
#             ).dict()
#             return mode_payload

#         return mode_payload

#     def set_leverage(
#         self, leverage: float, symbol: str, is_isolated: bool = True
#     ) -> dict:
#         """ """
#         if self.unified and self.DATA_TYPE not in ["inverse"]:
#             return models.CommonResponseSchema(
#                 success=False,
#                 error=True,
#                 data={},
#                 msg="In unified account, only inverse support to switch position mode.",
#             )
#         if not self.unified and self.DATA_TYPE not in ["linear", "inverse"]:
#             return models.CommonResponseSchema(
#                 success=False,
#                 error=True,
#                 data={},
#                 msg="In normal account, only inverse & linear support to switch position mode.",
#             )

#         m_payload = {
#             "symbol": symbol.upper(),
#             "marginCoin": self._get_margin_coin(symbol),
#             "marginMode": "fixed" if is_isolated else "crossed",
#         }
#         marginmode_payload = self._post(
#             "/api/mix/v1/account/setMarginMode", remove_none(m_payload)
#         )
#         print(f"{marginmode_payload} !!!!setMarginMode")
#         # success=True error=False data={'symbol': 'SBTCSUSDT_SUMCBL', 'marginCoin': 'SUSDT', 'longLeverage': 20, 'shortLeverage': 20, 'crossMarginLeverage': None, 'marginMode': 'fixed'} msg='query ok' !!!!setMarginMode

#         payload = {
#             "symbol": symbol.upper(),
#             "marginCoin": self._get_margin_coin(symbol),
#             # "holdSide": 'long' # 'short',
#             "leverage": str(leverage),
#         }
#         if is_isolated:
#             for s in ["long", "short"]:
#                 payload["holdSide"] = s
#                 leverage_payload = self._post(
#                     "/api/mix/v1/account/setLeverage", remove_none(payload)
#                 )
#                 print(f"{leverage_payload} !!!!setLeverage({s})")
#         else:
#             leverage_payload = self._post(
#                 "/api/mix/v1/account/setLeverage", remove_none(payload)
#             )
#             print(f"{leverage_payload} !!!!setLeverage")
#         # success=True error=False data={'symbol': 'SBTCSUSDT_SUMCBL', 'marginCoin': 'SUSDT', 'longLeverage': 7, 'shortLeverage': 7, 'crossMarginLeverage': 7, 'marginMode': 'crossed'} msg='query ok' !!!!setLeverage

#         if leverage_payload.success:
#             if leverage_payload.data:
#                 leverage_payload.data = models.LeverageSchema(
#                     symbol=symbol, leverage=leverage
#                 ).dict()
#                 return leverage_payload
#         # else:
#         #     if leverage_payload.data.get("msg", {}).get("retCode", 0) == 110026:  # 'Cross/isolated margin mode is not modified
#         #         leverage_payload = self._post("/api/mix/v1/account/setLeverage", remove_none(payload))
#         #         # print(leverage_payload,'!!!!set-leverage')
#         #         if leverage_payload.success:
#         #             if leverage_payload.data:
#         #                 leverage_payload.data = models.LeverageSchema(symbol=symbol, leverage=leverage).dict()
#         #                 return leverage_payload
#         return leverage_payload

#     def query_custom(self, url, *args, **kwargs):
#         payload = {
#             "productType": self.product_type,
#         }
#         for k, v in kwargs.items():
#             payload[k] = v
#         print(f"payload:{payload}| args:{args} | kwargs:{kwargs}")

#         api_ret = self._get(url, payload)

#         return api_ret


# class BitgetDataWebsocket(WebsocketClient):
#     """ """

#     DATA_TYPE = "spot"

#     def __init__(
#         self,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         ping_interval: int = 25,
#         is_testnet: bool = False,
#         datatype: str = "spot",
#         debug: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__()

#         assert datatype in ["spot", "linear", "inverse"]
#         self.DATA_TYPE = datatype
#         self.product_type = None
#         self.is_testnet: bool = is_testnet

#         if self.DATA_TYPE == "linear":
#             if self.is_testnet:
#                 self.product_type = "sumcbl"
#             else:
#                 self.product_type = "umcbl"
#         elif self.DATA_TYPE == "inverse":
#             if self.is_testnet:
#                 self.product_type = "sdmcbl"
#             else:
#                 self.product_type = "dmcbl"

#         self.proxy_host: str = proxy_host
#         self.proxy_port: int = proxy_port
#         self.ping_interval: int = ping_interval

#         self.ticks: dict[str, dict] = defaultdict(dict)
#         self.reqid: int = 0

#         self._logged_in = False
#         self.key = None
#         self.secret = None
#         self.passphrase = None
#         self._logged_in = False

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
#         return f"Bitget {self.DATA_TYPE} Data Websocket start"

#     def connect(self):
#         """ """
#         if self.DATA_TYPE == "spot":
#             host = BITGET_WS_SPOT_HOST
#         elif self.DATA_TYPE in ["linear", "inverse"]:
#             host = BITGET_WS_CONTRACT_HOST

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

#     def custom_ping(self):
#         return "ping"

#     def on_connected(self) -> None:
#         """"""
#         if self.waitfor_connection():
#             print(f"Bitget {self.DATA_TYPE} Market data websocket connect success")
#         else:
#             print(f"Bitget {self.DATA_TYPE} Market data websocket connect fail")

#         print(
#             f"===> debug Bitget {self.DATA_TYPE} Market data on_connected - self.ticks: {self.ticks}"
#         )

#         # resubscribe
#         if self.ticks:
#             channels = []
#             for key, detail in self.ticks.items():
#                 symbol = detail["symbol"]
#                 instid = symbol.split("_")[0] if "_" in symbol else symbol

#                 req: dict = self.sub_stream(
#                     symbol=instid,
#                     channel=detail["channel"],
#                     interval=detail.get("interval"),
#                 )
#                 channels += req["args"]

#             req: dict = {
#                 "op": "subscribe",
#                 "args": channels,
#             }
#             print(f"on_connected, resubscribing the channels - req:{req}")
#             self.send_packet(req)

#         if self.on_connected_callback:
#             self.on_connected_callback()

#     def on_disconnected(self) -> None:
#         print(
#             f"Bitget {self.DATA_TYPE} market data Websocket disconnected, now try to connect @ {datetime.now()}"
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
#         spot: instId refer to Get Symbols in response field: symbolName
#         contract: instId from Get All Symbols ==> baseCoin + quoteCoin
#         """
#         self.reqid += 1
#         mapping_channel = MAPPING_CHANNEL[channel]
#         sub = {
#             "op": "subscribe",
#         }

#         if channel == "kline":
#             # if self.product_type:
#             #     _interval = KBAR_CONTRACT_INTERVAL[interval]
#             # else:
#             #     _interval = KBAR_SPOT_INTERVAL[interval]
#             _interval = KBAR_CONTRACT_INTERVAL[interval]

#             sub["args"] = [
#                 {
#                     "instType": "mc" if self.product_type else "SP",
#                     "channel": f"{mapping_channel}{_interval}",
#                     "instId": symbol,
#                 }
#             ]  # candle1W candle1D candle12H candle4H candle1H candle30m candle15m candle5m candle1m
#         elif channel == "depth":
#             """ """
#             # sub["args"] = [f"{mapping_channel}.1.{symbol}"]
#             # sub["args"] = [f"{mapping_channel}.50.{symbol}"]
#             # sub["args"] = [f"{mapping_channel}.200.{symbol}"]
#             sub["args"] = [
#                 {
#                     "instType": "mc" if self.product_type else "SP",
#                     # "channel":f"{mapping_channel}1", # f"{mapping_channel}5", f"{mapping_channel}15",
#                     "channel": f"{mapping_channel}5",
#                     "instId": symbol,
#                 }
#             ]
#         elif channel == "ticker":
#             # sub["args"] = [f"{mapping_channel}.{symbol}"]
#             sub["args"] = [
#                 {
#                     "instType": "mc" if self.product_type else "SP",
#                     "channel": mapping_channel,
#                     "instId": symbol,
#                 }
#             ]

#         # print(f'req_id:{self.reqid}!!!!!!!!!!symbol:{symbol}; channel:{channel}')
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
#         if channel not in ["depth", "kline", "ticker"]:
#             raise Exception("invalid subscription")

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
#             instid = symbol.split("_")[0] if "_" in symbol else symbol

#             if instid in self.ticks:
#                 return

#             # tick data dict
#             asset_type = models.bitget_asset_type(
#                 symbol=symbol, datatype=self.DATA_TYPE
#             )
#             tick = {
#                 "symbol": symbol,
#                 "exchange": EXCHANGE,
#                 "channel": channel,
#                 "asset_type": asset_type,
#                 "name": models.normalize_name(symbol, asset_type),
#             }
#             key = ""
#             if channel == "kline":
#                 key = f"{instid}|{interval}"
#                 tick["interval"] = interval
#             else:
#                 key = f"{instid}|{channel}"

#             self.ticks[key] = tick
#             req: dict = self.sub_stream(
#                 symbol=instid, channel=channel, interval=interval
#             )
#             self.send_packet(req)

#     def on_packet(self, packet: dict) -> None:
#         """on_packet
#         >> packet: {'event': 'error', 'code': 30001, 'msg': "instType:mc,channel:tickers,instId:BTCUSDT doesn't exist"}
#         >> packet: {'event': 'subscribe', 'arg': {'instType': 'mc', 'channel': 'books1', 'instId': 'BTCUSDT'}}
#         >> packet: {'event': 'error', 'code': 30001, 'msg': "instType:mc,channel:tickers,instId:BTCUSDT doesn't exist"}

#         kline:
#         >> packet: {'event': 'subscribe', 'arg': {'instType': 'mc', 'channel': 'candle1m', 'instId': 'BTCUSDT'}}
#         >> packet: {'action': 'snapshot', 'arg': {'instType': 'mc', 'channel': 'candle1m', 'instId': 'BTCUSDT'}, 'data': [['1681751820000', '29498', '29512.5', '29497', '29505.5', '197.642'], ['1681751880000', '29505.5', '29509', '29497', '29499.5', '96.622'], ['1681751940000', '29499.5', '29499.5', '29488', '29491.5', '96.16'], ['1681752000000', '29491.5', '29491.5', '29484', '29484.5', '118.876'], ['1681752060000', '29484.5', '29485', '29481', '29481.5', '73.247'], ['1681752120000', '29481.5', '29481.5', '29461.5', '29465', '106.4'], ['1681752180000', '29465', '29480', '29465', '29479.5', '47.946'], ['1681752240000', '29479.5', '29482', '29478.5', '29482', '51.15'], ['1681752300000', '29482', '29482', '29451', '29462.5', '44.57'], ['1681752360000', '29462.5', '29479', '29460.5', '29468', '48.099'],
#         ['1681752420000', '29468', '29472.5', '29463.5', '29470', '64.013'], ['1681752480000', '29470', '29470', '29467.5', '29469.5', '50.667'], ['1681752540000', '29469.5', '29469.5', '29467.5', '29467.5', '29'], ['1681752600000', '29467.5', '29468', '29457', '29468', '60.517'], ['1681752660000', '29468', '29479.5', '29467.5', '29479.5', '102.446'], ['1681752720000', '29479.5', '29488.5', '29479', '29487.5', '37.801'], ['1681752780000', '29487.5', '29491', '29484.5', '29489.5', '52.591'], ['1681752840000', '29489.5', '29490', '29488.5', '29489', '77.934'], ['1681752900000', '29489', '29489', '29481', '29481', '96.214'], ['1681752960000', '29481', '29482.5', '29481', '29482.5', '45.524'], ['1681753020000', '29482.5', '29502.5', '29482.5', '29502.5', '148.218'], ['1681753080000', '29502.5', '29504.5', '29491', '29492', '47.147'], ['1681753140000', '29492', '29492', '29481', '29481', '53.377'], ['1681753200000', '29481', '29481.5', '29462.5', '29463', '55.139'], ['1681753260000', '29463', '29463', '29453.5', '29454', '104.499'], ['1681753320000', '29454', '29471.5', '29453', '29454', '81.508'], ['1681753380000', '29454', '29471', '29453', '29470', '65.099'], ['1681753440000', '29470', '29470', '29465', '29465', '97.737'], ['1681753500000', '29465', '29465.5', '29463.5', '29463.5', '68.065'], ['1681753560000', '29463.5', '29464', '29463.5', '29464', '84.671'], ['1681753620000', '29464', '29464', '29463.5', '29464', '86.467'], ['1681753680000', '29464', '29483', '29463', '29480.5', '47.835'], ['1681753740000', '29480.5', '29494.5', '29480.5', '29493.5', '59.252'], ['1681753800000', '29493.5', '29494', '29479', '29479.5', '54.284'], ['1681753860000', '29479.5', '29495', '29479', '29485', '145.572'], ['1681753920000', '29485', '29487.5', '29484.5', '29487.5', '80.911'], ['1681753980000', '29487.5', '29487.5', '29476', '29476.5', '53.317'], ['1681754040000', '29476.5', '29476.5', '29451.5', '29454', '80.522'], ['1681754100000', '29454', '29454', '29443', '29444.5', '79.03'], ['1681754160000', '29444.5', '29455.5', '29444.5', '29455', '79.828'], ['1681754220000', '29455', '29474', '29455', '29469.5', '44.035'], ['1681754280000', '29469.5', '29476', '29469.5', '29469.5', '53.029'], ['1681754340000', '29469.5', '29470', '29444', '29444', '101.772'], ['1681754400000', '29444', '29448', '29439', '29439.5', '125.338'], ['1681754460000', '29439.5', '29465', '29439', '29461', '77.557'], ['1681754520000', '29461', '29462', '29448', '29448', '58.929'], ['1681754580000', '29448', '29460.5', '29443', '29455', '65.726'], ['1681754640000', '29455', '29471.5', '29451', '29465', '76.013'], ['1681754700000', '29465', '29470.5', '29463', '29465.5', '79.177'], ['1681754760000', '29465.5', '29473', '29465', '29472', '47.513'], ['1681754820000', '29472', '29479', '29472', '29479', '97.547'], ['1681754880000', '29479', '29479.5', '29476.5', '29478.5', '37.836'], ['1681754940000', '29478.5', '29478.5', '29472.5', '29472.5', '34.887'], ['1681755000000', '29472.5', '29481', '29472.5', '29476.5', '50.507'], ['1681755060000', '29476.5', '29480', '29475.5', '29479.5', '132.443'], ['1681755120000', '29479.5', '29480.5', '29479.5', '29480.5', '89.11'], ['1681755180000', '29480.5', '29489.5', '29480',
#         '29488.5', '91.577'], ['1681755240000', '29488.5', '29489.5', '29454.5', '29458', '274.178'], ['1681755300000', '29458', '29474', '29457', '29469.5', '40.992'], ['1681755360000', '29469.5', '29482', '29469.5', '29482', '65.846'], ['1681755420000', '29482', '29482', '29478.5', '29478.5', '91.463'], ['1681755480000', '29478.5', '29478.5', '29473', '29473.5', '50.584'], ['1681755540000', '29473.5', '29476.5', '29473', '29474', '70.277'], ['1681755600000', '29474', '29487', '29473.5', '29486.5', '47.711'], ['1681755660000', '29486.5', '29489.5', '29485', '29487.5', '37.072'], ['1681755720000', '29487.5', '29487.5', '29483.5', '29483.5', '28.788'], ['1681755780000', '29483.5', '29483.5', '29472', '29472', '84.762'], ['1681755840000', '29472', '29473', '29470.5', '29472', '94.353'], ['1681755900000', '29472', '29474.5', '29472', '29474', '36.06'], ['1681755960000', '29474', '29489.5', '29474', '29489.5', '82.429'], ['1681756020000', '29489.5', '29498', '29489', '29496.5', '260.211'], ['1681756080000', '29496.5', '29496.5', '29490', '29495.5', '120.521'], ['1681756140000', '29495.5', '29515', '29495.5', '29511', '163.083'], ['1681756200000', '29511', '29511', '29498.5', '29502.5', '71.544'], ['1681756260000', '29502.5', '29508.5', '29502.5', '29508.5', '43.072'], ['1681756320000', '29508.5', '29528', '29493', '29493.5', '117.855'], ['1681756380000', '29493.5', '29494', '29490', '29491', '25.519'], ['1681756440000', '29491', '29492', '29487.5', '29488', '105.253'], ['1681756500000', '29488', '29488.5', '29487', '29487', '73.872'], ['1681756560000', '29487', '29488', '29475', '29476', '49.106'], ['1681756620000', '29476', '29502', '29475.5', '29495.5', '122.71'], ['1681756680000', '29495.5', '29498', '29494.5', '29497', '93.672'], ['1681756740000', '29497', '29498', '29497', '29497.5', '75.061'], ['1681756800000', '29497.5', '29503', '29497.5', '29498', '54.673'], ['1681756860000', '29498', '29536.5', '29497.5', '29534.5', '176.228'], ['1681756920000', '29534.5', '29536.5', '29522.5', '29533', '100.25'], ['1681756980000', '29533', '29533.5', '29531', '29533', '49.393'], ['1681757040000', '29533', '29533.5', '29522.5', '29523', '65.999'], ['1681757100000', '29523', '29523', '29517', '29520', '70.559'], ['1681757160000', '29520', '29532.5', '29520', '29532.5', '38.703'], ['1681757220000', '29532.5', '29533', '29530', '29531.5', '83.943'], ['1681757280000', '29531.5', '29561.5', '29531.5', '29538.5', '238.13'], ['1681757340000', '29538.5', '29538.5', '29516', '29516', '91.979'], ['1681757400000', '29516', '29517', '29511', '29512', '64.961'], ['1681757460000', '29512', '29512', '29498', '29498.5', '77.62'], ['1681757520000', '29498.5', '29509.5', '29498', '29500', '46.643'], ['1681757580000', '29500', '29514', '29499.5', '29512', '63.347'], ['1681757640000', '29512', '29512', '29503', '29503.5', '78.363'], ['1681757700000', '29503.5', '29504', '29501.5', '29502.5', '82.356'], ['1681757760000', '29502.5', '29511', '29502', '29508', '58.16'], ['1681757820000', '29508', '29508.5', '29506', '29506', '54.903'], ['1681757880000', '29506', '29506.5', '29506', '29506', '66.059'], ['1681757940000', '29506', '29516', '29506', '29515.5', '33.164'], ['1681758000000', '29515.5', '29515.5', '29502.5', '29502.5', '34.365'], ['1681758060000', '29502.5', '29503', '29501.5', '29503', '113.201'], ['1681758120000', '29503', '29518.5', '29503', '29504.5', '43.316'], ['1681758180000', '29504.5', '29512', '29504.5', '29506', '15.959'], ['1681758240000', '29506', '29506', '29504', '29504', '24.052'], ['1681758300000', '29504', '29504', '29492', '29492.5', '50.112'], ['1681758360000', '29492.5', '29517', '29492.5', '29517', '34.753'], ['1681758420000', '29517', '29532', '29517', '29531', '78.186'], ['1681758480000', '29531', '29534', '29529.5', '29531.5', '51.075'], ['1681758540000', '29531.5', '29532', '29527', '29527', '67.035'], ['1681758600000', '29527', '29528.5', '29519.5', '29519.5', '87.161'], ['1681758660000', '29519.5', '29519.5', '29511', '29511.5', '65.003'], ['1681758720000', '29511.5', '29512.5', '29503', '29511', '65.674'], ['1681758780000', '29511', '29516.5', '29510', '29511.5', '61.067'], ['1681758840000', '29511.5', '29512.5', '29511', '29512.5', '73.229'], ['1681758900000', '29512.5', '29512.5', '29500', '29501.5', '68.725'], ['1681758960000', '29501.5', '29508.5', '29501', '29504.5', '35.928'], ['1681759020000', '29504.5', '29505.5', '29504', '29504', '50.649'], ['1681759080000', '29504', '29504.5', '29503', '29504.5', '54.933'], ['1681759140000', '29504.5', '29506.5', '29504', '29506.5', '38.364'], ['1681759200000', '29506.5', '29529.5', '29506.5', '29529', '68.811'], ['1681759260000', '29529', '29544.5', '29528.5', '29538.5', '128.418'], ['1681759320000', '29538.5', '29544.5', '29533', '29538.5', '128.53'], ['1681759380000', '29538.5', '29546.5', '29537', '29543.5', '66.494'], ['1681759440000', '29543.5', '29546', '29540.5', '29540.5', '126.674'], ['1681759500000', '29540.5', '29540.5', '29536', '29536', '67.253'], ['1681759560000', '29536', '29551', '29534', '29551', '77.849'], ['1681759620000', '29551', '29551', '29541.5', '29545.5', '24.05'], ['1681759680000', '29545.5', '29546', '29540.5', '29545', '32.8'], ['1681759740000', '29545', '29547.5', '29544', '29544.5', '69.615'], ['1681759800000', '29544.5', '29544.5', '29542.5', '29542.5', '64.002'], ['1681759860000', '29542.5', '29563.5', '29542.5', '29563', '174.278'], ['1681759920000', '29563', '29572', '29560.5', '29561', '124.472'], ['1681759980000', '29561', '29561', '29527.5', '29527.5', '83.343'], ['1681760040000', '29527.5', '29539', '29520', '29520', '138.894'], ['1681760100000', '29520', '29520.5', '29510', '29517', '137.321'], ['1681760160000', '29517', '29521', '29503.5', '29505', '138.332'], ['1681760220000', '29505', '29509.5', '29504', '29507.5', '100.46'], ['1681760280000', '29507.5', '29507.5', '29504', '29504', '50.655'], ['1681760340000', '29504', '29505', '29490.5', '29495.5', '56.615'], ['1681760400000', '29495.5', '29499.5', '29473', '29493', '85.932'], ['1681760460000', '29493', '29493', '29465.5', '29480', '108.603'], ['1681760520000', '29480', '29484', '29479.5', '29480.5', '45.171'], ['1681760580000', '29480.5', '29482', '29480.5', '29480.5', '23.537'], ['1681760640000', '29480.5', '29481', '29479.5', '29481', '21.206'], ['1681760700000', '29481', '29481', '29471.5', '29472', '45.47'], ['1681760760000', '29472', '29473', '29446.5', '29473', '93.599'], ['1681760820000', '29473', '29475', '29460.5', '29469', '46.3'], ['1681760880000', '29469', '29480', '29466.5', '29479.5', '69.393'], ['1681760940000', '29479.5', '29489.5', '29479', '29486.5', '123.841'], ['1681761000000', '29486.5', '29497', '29486.5', '29496.5', '98.965'], ['1681761060000', '29496.5', '29509.5', '29496.5', '29502.5', '40.041'], ['1681761120000', '29502.5', '29510', '29494', '29496.5', '46.563'], ['1681761180000', '29496.5', '29496.5', '29491', '29493.5', '67.343'], ['1681761240000', '29493.5', '29497.5', '29487', '29488', '87.565'], ['1681761300000', '29488', '29488', '29466', '29468', '71.776'], ['1681761360000', '29468', '29468', '29450.5', '29466', '75.658'], ['1681761420000', '29466', '29466', '29459.5', '29461.5', '87.964'], ['1681761480000', '29461.5', '29463', '29459.5', '29462', '60.558'], ['1681761540000', '29462', '29463', '29459.5', '29459.5', '51.484'], ['1681761600000', '29459.5', '29466', '29457.5', '29463.5', '82.887'], ['1681761660000', '29463.5', '29465.5', '29459', '29463.5', '51.575'], ['1681761720000', '29463.5', '29476.5', '29463.5', '29476', '83.869'], ['1681761780000', '29476', '29481', '29476', '29478.5', '68.357'], ['1681761840000', '29478.5', '29486', '29478.5', '29482.5', '75.711'], ['1681761900000', '29482.5', '29483.5', '29479', '29479.5', '73.327'], ['1681761960000', '29479.5', '29479.5', '29473', '29475', '50.862'], ['1681762020000', '29475', '29476.5', '29473.5', '29476.5', '63.662'], ['1681762080000', '29476.5', '29490', '29475.5', '29487.5', '37.323'], ['1681762140000', '29487.5', '29487.5', '29474', '29479.5', '85.24'], ['1681762200000', '29479.5', '29480', '29472.5', '29473.5', '79.88'], ['1681762260000', '29473.5', '29479', '29457', '29461', '203.741'], ['1681762320000', '29461', '29476', '29455', '29476', '87.158'], ['1681762380000', '29476', '29476', '29461', '29462', '68.63'], ['1681762440000', '29462', '29465.5', '29460', '29461.5', '29.644'], ['1681762500000', '29461.5', '29464.5', '29458.5', '29462', '53.061'], ['1681762560000', '29462', '29478', '29461.5', '29478', '95.528'], ['1681762620000', '29478', '29478', '29461.5', '29464', '38.497'], ['1681762680000', '29464', '29466', '29461.5', '29466', '62.858'], ['1681762740000', '29466', '29469.5', '29460.5', '29465', '67.465'], ['1681762800000', '29465', '29466', '29458', '29463', '45.821'], ['1681762860000', '29463', '29463', '29457.5', '29459', '61.657'], ['1681762920000', '29459', '29460', '29456.5', '29457', '87.198'], ['1681762980000', '29457', '29463', '29457', '29463', '96.647'], ['1681763040000', '29463', '29464.5', '29462', '29462', '81.351'], ['1681763100000', '29462', '29462', '29450', '29450', '43.302'], ['1681763160000', '29450', '29450.5', '29426.5', '29428.5', '176.022'], ['1681763220000', '29428.5', '29430', '29412.5', '29424', '179.496'], ['1681763280000', '29424', '29424', '29415', '29416.5', '108.517'], ['1681763340000', '29416.5', '29433.5', '29416', '29433.5',
#         '202.374'], ['1681763400000', '29433.5', '29444.5', '29431.5', '29437.5', '274.176'], ['1681763460000', '29437.5', '29445', '29434', '29445', '52.019'], ['1681763520000', '29445', '29446.5', '29444.5', '29446.5', '64.149'], ['1681763580000', '29446.5', '29457.5', '29446.5', '29455', '101.403'], ['1681763640000', '29455', '29456.5', '29455', '29455.5', '73.282'], ['1681763700000', '29455.5', '29456', '29450', '29451', '68.986'], ['1681763760000', '29451', '29464.5', '29451', '29464', '96.015'], ['1681763820000', '29464', '29464.5', '29463', '29463.5', '83.174'], ['1681763880000', '29463.5', '29476', '29463.5', '29465.5', '81.518'], ['1681763940000', '29465.5', '29469.5', '29465.5', '29467', '51.921'], ['1681764000000', '29467', '29467.5', '29467', '29467', '26.59'], ['1681764060000', '29467', '29467.5', '29452.5', '29455', '98.95'], ['1681764120000', '29455', '29455.5', '29442', '29442', '77.319'], ['1681764180000', '29442', '29443.5', '29437', '29441', '66.395'], ['1681764240000', '29441', '29446.5', '29441', '29444', '106.187'], ['1681764300000', '29444', '29444.5', '29443', '29443', '85'], ['1681764360000', '29443', '29443.5', '29433.5', '29434', '68.577'], ['1681764420000', '29434', '29434', '29425', '29425.5', '104.079'], ['1681764480000', '29425.5', '29426.5', '29419.5', '29419.5', '95.605'], ['1681764540000', '29419.5', '29425', '29408.5', '29425', '61.624'], ['1681764600000', '29425', '29437', '29425', '29436.5', '68.802'], ['1681764660000', '29436.5', '29440.5', '29427.5', '29428', '92.708'], ['1681764720000', '29428', '29428', '29421.5', '29422', '173.976'], ['1681764780000', '29422', '29423', '29420', '29422', '87.716'], ['1681764840000', '29422', '29422.5', '29420.5', '29421', '39.293'], ['1681764900000', '29421', '29421', '29418.5', '29419', '44.059'], ['1681764960000', '29419', '29419.5', '29410', '29410', '77.325'], ['1681765020000', '29410', '29437.5', '29409.5', '29432.5', '271.603'], ['1681765080000', '29432.5', '29451', '29432.5', '29436', '86.825'], ['1681765140000', '29436', '29436.5', '29420', '29431', '87.205'], ['1681765200000', '29431', '29436', '29430', '29435.5', '47.51'], ['1681765260000', '29435.5', '29436.5', '29434', '29436.5', '53.512'], ['1681765320000', '29436.5', '29437.5', '29436', '29437.5', '56.097'], ['1681765380000', '29437.5', '29437.5', '29418.5', '29419.5', '67.989'], ['1681765440000', '29419.5', '29424', '29416', '29422', '48.569'], ['1681765500000', '29422', '29436.5', '29422', '29436', '66.188'], ['1681765560000', '29436', '29448.5', '29436', '29442', '26.006'], ['1681765620000', '29442', '29442', '29433.5', '29438', '50.502'], ['1681765680000', '29438', '29440.5', '29432', '29432.5', '34.019'], ['1681765740000', '29432.5', '29432.5', '29425.5', '29426.5', '43.984'], ['1681765800000', '29426.5', '29426.5', '29418.5', '29418.5', '60.078'], ['1681765860000', '29418.5', '29418.5', '29410', '29410', '90.789'], ['1681765920000', '29410', '29414', '29398.5', '29414', '91.263'], ['1681765980000', '29414', '29414', '29413.5', '29413.5', '56.395'], ['1681766040000', '29413.5', '29423', '29413.5', '29423', '84.302'], ['1681766100000', '29423', '29426', '29422.5', '29425', '90.212'], ['1681766160000', '29425', '29426.5', '29424.5', '29425.5', '72.198'], ['1681766220000', '29425.5', '29428.5', '29421.5', '29422.5', '56.82'], ['1681766280000', '29422.5', '29422.5', '29415.5', '29415.5', '42.785'], ['1681766340000', '29415.5', '29416', '29408.5', '29408.5', '52.435'], ['1681766400000', '29408.5', '29408.5', '29391', '29400.5', '97.763'], ['1681766460000', '29400.5', '29401', '29398', '29399.5', '127.276'], ['1681766520000', '29399.5', '29407', '29371.5', '29405.5', '195.371'], ['1681766580000', '29405.5', '29424', '29405.5', '29423.5', '88.199'], ['1681766640000', '29423.5', '29425', '29421', '29423', '67.945'], ['1681766700000', '29423', '29434', '29423', '29434', '49.999'], ['1681766760000', '29434', '29434', '29429', '29430.5', '79.871'], ['1681766820000', '29430.5', '29431', '29414.5', '29415', '90.792'], ['1681766880000', '29415', '29420.5', '29415', '29419.5', '97.066'], ['1681766940000', '29419.5', '29420.5', '29419', '29420', '55.814'], ['1681767000000', '29420', '29427.5', '29420', '29426.5', '39.089'], ['1681767060000', '29426.5', '29431.5', '29426.5', '29430.5', '51.069'], ['1681767120000', '29430.5', '29433', '29430.5', '29432.5', '45.797'], ['1681767180000', '29432.5', '29433.5', '29432', '29433.5', '60.4'], ['1681767240000', '29433.5', '29438.5', '29433.5', '29438.5', '75.095'], ['1681767300000', '29438.5', '29443', '29438', '29442', '60.279'], ['1681767360000', '29442', '29443.5', '29442', '29443.5', '60.065'], ['1681767420000', '29443.5', '29445.5', '29443', '29445', '37.931'], ['1681767480000', '29445', '29450', '29445', '29447.5', '22.413'], ['1681767540000', '29447.5', '29454.5', '29447', '29454', '56.925'], ['1681767600000', '29454', '29456.5', '29446', '29446', '39.13'], ['1681767660000', '29446', '29446.5', '29446', '29446', '39.441'], ['1681767720000', '29446', '29446.5', '29443', '29443', '29.721'], ['1681767780000', '29443', '29443.5', '29440.5', '29442', '66.884'], ['1681767840000', '29442', '29445.5', '29442', '29445.5', '49.935'], ['1681767900000', '29445.5', '29445.5', '29445', '29445', '22.328'], ['1681767960000', '29445', '29445.5', '29445', '29445.5', '25.061'], ['1681768020000', '29445.5', '29445.5', '29439.5', '29440', '46.485'], ['1681768080000', '29440', '29441', '29440', '29441', '31.206'], ['1681768140000', '29441', '29441', '29440', '29440.5', '28.405'], ['1681768200000', '29440.5', '29440.5', '29439.5', '29439.5', '34.408'], ['1681768260000', '29439.5', '29440', '29438', '29438', '23.227'], ['1681768320000', '29438', '29438.5', '29437.5', '29437.5', '47.371'], ['1681768380000', '29437.5', '29438.5', '29437.5', '29437.5', '21.164'], ['1681768440000', '29437.5', '29442', '29437.5', '29440.5', '22.271'], ['1681768500000', '29440.5', '29442.5', '29440.5', '29442', '13.963'], ['1681768560000', '29442', '29442.5', '29439', '29439', '39.248'], ['1681768620000', '29439', '29439.5', '29439', '29439', '33.542'], ['1681768680000', '29439', '29439.5', '29439', '29439.5', '20.931'], ['1681768740000', '29439.5', '29439.5', '29435', '29437.5', '53.92'], ['1681768800000', '29437.5', '29441', '29437', '29440.5', '25.171'], ['1681768860000', '29440.5', '29441', '29440', '29441', '38.645'], ['1681768920000', '29441', '29478', '29441', '29466', '86.386'], ['1681768980000', '29466', '29475', '29462.5', '29467', '44.42'], ['1681769040000', '29467', '29467.5', '29461.5', '29463.5', '19.277'], ['1681769100000', '29463.5', '29470.5', '29460.5', '29460.5', '29.195'], ['1681769160000', '29460.5', '29460.5', '29457.5', '29460.5', '54.78'], ['1681769220000', '29460.5', '29460.5', '29460', '29460', '20.775'], ['1681769280000', '29460', '29460.5', '29454.5', '29454.5', '39.507'], ['1681769340000', '29454.5', '29455', '29454.5', '29455', '31.624'], ['1681769400000', '29455', '29466.5', '29455', '29466', '50.614'], ['1681769460000', '29466', '29467', '29465.5', '29467', '41.847'], ['1681769520000', '29467', '29467', '29465.5', '29467', '38.01'], ['1681769580000', '29467', '29467.5', '29466.5', '29467.5', '24.43'], ['1681769640000', '29467.5', '29467.5', '29467', '29467', '82.384'], ['1681769700000', '29467', '29467.5', '29466.5', '29467', '25.196'], ['1681769760000', '29467', '29467', '29465.5', '29466', '36.607'], ['1681769820000', '29466', '29466.5', '29464.5', '29465', '55.012'], ['1681769880000', '29465', '29465', '29457.5', '29458', '43.057'], ['1681769940000', '29458', '29458', '29457.5', '29457.5', '47.751'], ['1681770000000', '29457.5', '29458', '29457.5', '29457.5', '45.564'], ['1681770060000', '29457.5', '29458', '29457.5', '29458', '18.069'], ['1681770120000', '29458', '29458', '29457.5', '29458', '16.959'], ['1681770180000', '29458', '29458', '29457.5', '29458', '13.534'], ['1681770240000', '29458', '29458.5', '29457.5', '29458', '55.583'], ['1681770300000', '29458', '29458.5', '29457.5', '29457.5', '31.543'], ['1681770360000', '29457.5', '29489.5', '29457.5', '29488.5', '85.591'], ['1681770420000', '29488.5', '29494.5', '29484', '29488', '73.177'], ['1681770480000', '29488', '29490', '29485', '29486', '23.172'], ['1681770540000', '29486', '29487.5', '29485', '29487', '162.17'], ['1681770600000', '29487', '29487', '29485.5', '29486', '49.057'], ['1681770660000', '29486', '29486', '29485.5', '29485.5', '47.148'], ['1681770720000', '29485.5', '29495', '29485.5', '29492', '95.03'], ['1681770780000', '29492', '29492', '29491.5', '29492', '64.132'], ['1681770840000', '29492', '29495', '29491.5', '29492', '40.164'], ['1681770900000', '29492', '29495', '29489.5', '29495', '65.859'], ['1681770960000', '29495', '29515', '29495', '29513', '159.834'], ['1681771020000', '29513', '29522', '29505', '29505', '114.484'], ['1681771080000', '29505', '29510.5', '29505', '29509', '61.072'], ['1681771140000', '29509', '29509.5', '29494.5', '29500', '37.866'], ['1681771200000', '29500', '29500', '29499', '29500', '70.321'], ['1681771260000', '29500', '29508.5', '29500', '29508.5', '56.101'], ['1681771320000', '29508.5', '29510', '29507.5', '29509', '65.801'], ['1681771380000', '29509', '29511.5', '29508', '29511.5', '63.429'], ['1681771440000', '29511.5', '29518.5', '29499.5', '29500', '61.096'], ['1681771500000', '29500', '29510.5', '29499.5', '29510', '44.422'], ['1681771560000', '29510', '29510.5', '29507', '29507', '36.32'], ['1681771620000', '29507', '29507.5', '29496', '29496.5', '62.214'], ['1681771680000', '29496.5', '29496.5', '29495', '29495.5', '71.265'], ['1681771740000', '29495.5', '29495.5', '29488', '29488', '65.007'], ['1681771800000', '29488', '29488.5', '29486.5', '29486.5', '50.873'], ['1681771860000', '29486.5', '29488', '29485', '29486', '31.853'], ['1681771920000', '29486', '29496.5', '29486', '29495.5', '48.507'], ['1681771980000', '29495.5', '29497', '29495.5', '29496.5', '93.174'], ['1681772040000', '29496.5', '29497.5', '29496.5', '29497', '74.807'], ['1681772100000', '29497', '29497', '29492', '29492.5', '66.918'], ['1681772160000', '29492.5', '29493', '29490', '29490.5', '79.718'], ['1681772220000', '29490.5', '29491.5', '29466.5', '29468.5', '109.267'], ['1681772280000', '29468.5', '29477.5', '29468', '29477.5', '129.245'], ['1681772340000', '29477.5', '29487', '29477', '29487', '52.544'], ['1681772400000', '29487', '29490.5', '29487', '29490.5', '37.344'], ['1681772460000', '29490.5', '29491.5', '29489.5', '29490', '68.231'], ['1681772520000', '29490', '29490', '29489.5', '29489.5', '49.538'], ['1681772580000', '29489.5', '29489.5', '29477.5', '29478', '62.255'], ['1681772640000', '29478', '29478', '29477', '29477.5', '53.429'], ['1681772700000', '29477.5',
#         '29477.5', '29469', '29469', '90.355'], ['1681772760000', '29469', '29470.5', '29469', '29470', '37.162'], ['1681772820000', '29470', '29471', '29470', '29471', '47.632'], ['1681772880000', '29471', '29471', '29457.5', '29465.5', '77.339'], ['1681772940000', '29465.5', '29466.5', '29465', '29465.5', '63.648'], ['1681773000000', '29465.5', '29469', '29465', '29468.5', '57.781'], ['1681773060000', '29468.5', '29469', '29460.5', '29460.5', '33.626'], ['1681773120000', '29460.5', '29469.5', '29460.5', '29469.5', '47.047'], ['1681773180000', '29469.5', '29469.5', '29469', '29469', '30.599'], ['1681773240000', '29469', '29469', '29460', '29460', '59.005'], ['1681773300000', '29460', '29465', '29459.5', '29459.5', '52.158'], ['1681773360000', '29459.5', '29462.5', '29459.5', '29462.5', '34.753'], ['1681773420000', '29462.5', '29462.5', '29462', '29462', '60.855'], ['1681773480000', '29462', '29462.5', '29462', '29462', '27.924'], ['1681773540000', '29462', '29462.5', '29447.5', '29447.5', '116.414'], ['1681773600000', '29447.5', '29459.5', '29447.5', '29459.5', '27.682'], ['1681773660000', '29459.5', '29465.5', '29459', '29465.5', '45.472'], ['1681773720000', '29465.5', '29465.5', '29463.5', '29464', '56.554'], ['1681773780000', '29464', '29464', '29456', '29456', '76.401'], ['1681773840000', '29456', '29456.5', '29454.5', '29455.5', '72.237'], ['1681773900000', '29455.5', '29460', '29455.5', '29460', '32.368'], ['1681773960000', '29460', '29460', '29459', '29459.5', '76.606'], ['1681774020000', '29459.5', '29460', '29451.5', '29453.5', '47.696'], ['1681774080000', '29453.5', '29454.5', '29452.5', '29453.5', '48.487'], ['1681774140000', '29453.5', '29453.5', '29452.5', '29452.5', '58.416'], ['1681774200000', '29452.5', '29460', '29449.5', '29460', '74.147'], ['1681774260000', '29460', '29460', '29458.5', '29458.5', '37.069'], ['1681774320000', '29458.5', '29459', '29452', '29452.5', '32.106'], ['1681774380000', '29452.5', '29452.5', '29452', '29452.5', '35.63'], ['1681774440000', '29452.5', '29452.5', '29432', '29438', '80.878'], ['1681774500000', '29438', '29443', '29436.5', '29441', '46.172'], ['1681774560000',
#         '29441', '29450', '29441', '29449', '79.545'], ['1681774620000', '29449', '29449.5', '29434.5', '29436.5', '27.224'], ['1681774680000', '29436.5', '29437', '29435', '29435', '38.249'], ['1681774740000', '29435', '29435.5', '29428.5', '29429.5', '68.876'], ['1681774800000', '29429.5', '29434', '29428', '29433.5', '20.185'], ['1681774860000', '29433.5', '29433.5', '29426', '29426', '18.756'], ['1681774920000', '29426', '29426.5', '29420', '29421', '42.79'], ['1681774980000', '29421', '29421', '29406', '29406', '78.867'], ['1681775040000', '29406', '29415.5', '29402', '29415', '88.148'], ['1681775100000', '29415', '29416', '29413.5', '29416', '46.933'], ['1681775160000', '29416', '29416', '29404', '29405', '345.736'], ['1681775220000', '29405', '29405', '29399', '29399.5', '90.595'], ['1681775280000', '29399.5', '29399.5', '29392', '29392.5', '77.907'], ['1681775340000', '29392.5', '29408.5', '29392', '29408.5', '74.701'], ['1681775400000', '29408.5', '29425.5', '29408.5', '29425.5', '55.618'], ['1681775460000', '29425.5', '29435', '29425', '29425.5', '59.024'], ['1681775520000', '29425.5', '29425.5', '29425', '29425', '30.711'], ['1681775580000', '29425', '29425', '29396', '29396.5', '52.816'], ['1681775640000', '29396.5', '29397', '29396', '29397', '39.542'], ['1681775700000', '29397', '29399', '29396.5', '29399', '39.604'], ['1681775760000', '29399', '29400', '29389', '29389', '53.209'], ['1681775820000', '29389', '29401', '29380', '29401', '89.591'], ['1681775880000', '29401', '29414.5', '29401', '29414.5', '38.947'], ['1681775940000', '29414.5', '29414.5', '29412', '29414.5', '41.027'], ['1681776000000', '29414.5', '29414.5', '29413.5', '29413.5', '38.919'], ['1681776060000', '29413.5', '29421.5', '29413.5', '29421', '50.981'], ['1681776120000', '29421', '29421', '29411', '29411', '38.794'], ['1681776180000', '29411', '29411', '29408.5', '29408.5', '41.906'], ['1681776240000', '29408.5', '29410.5', '29404', '29410.5', '21.816'], ['1681776300000', '29410.5', '29418.5', '29410', '29418', '46.586'], ['1681776360000', '29418', '29422', '29418', '29422', '21.892'], ['1681776420000', '29422', '29426', '29421.5', '29426', '43.053'], ['1681776480000', '29426', '29426', '29425.5', '29426', '11.614'], ['1681776540000', '29426', '29431', '29425.5', '29431', '38.416'], ['1681776600000', '29431', '29431', '29427.5', '29428', '32.109'], ['1681776660000', '29428', '29428.5', '29427.5', '29428', '13.284'], ['1681776720000', '29428', '29433.5', '29428', '29433.5', '20.793'], ['1681776780000', '29433.5', '29435.5', '29433', '29435', '49.349'], ['1681776840000', '29435', '29435', '29434.5', '29435', '37.98'], ['1681776900000', '29435', '29435', '29425', '29425', '35.099'], ['1681776960000', '29425', '29425.5', '29425', '29425', '21.832'], ['1681777020000', '29425', '29425.5', '29397', '29397', '73.957'], ['1681777080000', '29397', '29397.5', '29380', '29380', '67.693'], ['1681777140000', '29380', '29380.5', '29376', '29376.5', '82.471'], ['1681777200000', '29376.5', '29383.5', '29376.5', '29383', '45.585'], ['1681777260000', '29383', '29384.5', '29382', '29384.5', '47.076'], ['1681777320000', '29384.5', '29415', '29384', '29413', '62.992'], ['1681777380000', '29413', '29420.5', '29400', '29410.5', '86.605'], ['1681777440000', '29410.5', '29415', '29408.5', '29415', '39.61'], ['1681777500000', '29415', '29415', '29405', '29405.5', '35.431'], ['1681777560000', '29405.5', '29405.5', '29387.5', '29388.5', '31.274'], ['1681777620000', '29388.5', '29388.5', '29383.5', '29388.5', '20.85'], ['1681777680000', '29388.5', '29389.5', '29385', '29388', '60.995'], ['1681777740000', '29388', '29388', '29384.5', '29384.5', '33.572'], ['1681777800000', '29384.5', '29408', '29384.5', '29399.5', '41.426'], ['1681777860000', '29399.5', '29401.5', '29398.5', '29401', '51.267'], ['1681777920000', '29401', '29404.5', '29400.5', '29402.5', '40.247'], ['1681777980000', '29402.5', '29405.5', '29402', '29405', '30.061'], ['1681778040000', '29405', '29412', '29405', '29411', '22.01'], ['1681778100000', '29411', '29411', '29406.5', '29406.5', '23.637'], ['1681778160000', '29406.5', '29406.5', '29381.5', '29382.5', '60.054'], ['1681778220000', '29382.5', '29382.5', '29345', '29361', '273.802'], ['1681778280000', '29361', '29361.5', '29355', '29355', '79.971'], ['1681778340000', '29355', '29355.5', '29320', '29333.5', '227.792'], ['1681778400000', '29333.5', '29333.5', '29307.5', '29320.5', '143.54'], ['1681778460000', '29320.5', '29341', '29320.5', '29330.5', '86.399'], ['1681778520000', '29330.5', '29330.5', '29270', '29309', '500.116'], ['1681778580000', '29309', '29328', '29300', '29309', '143.178'], ['1681778640000', '29309', '29313.5', '29307.5', '29313.5', '49.155'], ['1681778700000', '29313.5', '29320.5', '29311.5', '29317.5', '99.398'], ['1681778760000', '29317.5', '29332.5', '29317.5', '29332', '102.359'], ['1681778820000', '29332', '29332', '29321', '29324', '51.949'], ['1681778880000', '29324', '29326.5', '29313.5', '29325.5', '68.42'], ['1681778940000', '29325.5', '29331', '29324.5', '29325.5', '66.122'], ['1681779000000', '29325.5', '29328', '29325', '29328', '68.443'], ['1681779060000', '29328', '29349.5', '29325', '29349.5', '83.983'], ['1681779120000', '29349.5', '29355', '29337.5', '29355', '113.387'], ['1681779180000', '29355', '29356', '29343.5', '29354', '61.827'], ['1681779240000', '29354', '29354.5', '29348', '29350', '78.928'], ['1681779300000', '29350', '29356', '29350', '29355', '46.897'], ['1681779360000', '29355', '29356', '29355', '29355.5', '46.799'], ['1681779420000', '29355.5', '29355.5', '29351.5', '29352', '51.772'], ['1681779480000', '29352', '29352', '29351', '29351.5', '66.207'], ['1681779540000', '29351.5', '29352', '29347.5', '29347.5', '26.742'], ['1681779600000', '29347.5', '29361', '29347.5', '29359', '122.542'], ['1681779660000', '29359', '29361', '29349', '29352', '43.343'], ['1681779720000', '29352', '29356', '29352', '29356', '52.13'], ['1681779780000', '29356', '29358', '29356', '29357.5', '43.831'], ['1681779840000', '29357.5', '29361', '29352.5', '29353', '28.823'], ['1681779900000', '29353', '29353', '29349.5', '29350', '31.842'], ['1681779960000', '29350', '29350.5', '29321', '29321.5', '58.945'], ['1681780020000', '29321.5', '29321.5', '29310', '29319.5', '166.352'], ['1681780080000', '29319.5', '29338.5', '29319.5', '29338.5', '75.27'], ['1681780140000', '29338.5', '29338.5', '29318', '29318', '40.645'], ['1681780200000', '29318', '29318', '29305', '29307.5', '125.612'], ['1681780260000', '29307.5', '29307.5', '29253.5', '29266.5', '439.685'], ['1681780320000', '29266.5', '29289', '29266.5', '29286', '176.374'], ['1681780380000', '29286', '29292', '29270', '29270.5', '122.545'], ['1681780440000', '29270.5', '29272', '29130', '29194', '2226.045'], ['1681780500000', '29194', '29197.5', '29085', '29140.5', '1032.321'], ['1681780560000', '29140.5', '29211.5', '29125', '29211', '330.297'], ['1681780620000', '29211', '29239.5', '29208',
#         '29230', '294.434'], ['1681780680000', '29230', '29251.5', '29216', '29248.5', '215.78'], ['1681780740000', '29248.5', '29255', '29239', '29246', '172.681'], ['1681780800000', '29246', '29248', '29200.5', '29228.5', '227.435'], ['1681780860000', '29228.5', '29264', '29215.5', '29247', '140.428'], ['1681780920000', '29247', '29247', '29233', '29241', '101.004'], ['1681780980000', '29241', '29243', '29218', '29218', '99.075'], ['1681781040000', '29218', '29230', '29215.5', '29230', '119.604'], ['1681781100000', '29230', '29236', '29188.5', '29198.5', '142.295'], ['1681781160000', '29198.5', '29233.5', '29198', '29226.5', '97.057'], ['1681781220000', '29226.5', '29227', '29219', '29224', '79.171'], ['1681781280000', '29224', '29256.5', '29223', '29253', '121.356'], ['1681781340000', '29253', '29281.5', '29253', '29273', '185.435'], ['1681781400000', '29273', '29326.5', '29269.5', '29309', '292.342'], ['1681781460000', '29309', '29309', '29291', '29297.5', '185.016'], ['1681781520000', '29297.5', '29300', '29291', '29291', '127.803'], ['1681781580000', '29291', '29330', '29291', '29317', '181.906'], ['1681781640000', '29317', '29322.5', '29307', '29322.5', '88.024'], ['1681781700000', '29322.5', '29322.5', '29313', '29314', '82.21'], ['1681781760000', '29314', '29314', '29308.5', '29310', '38.096']]}
#         >> packet: {'action': 'update', 'arg': {'instType': 'mc', 'channel': 'candle1m', 'instId': 'BTCUSDT'}, 'data': [['1681781760000', '29314', '29314', '29308.5', '29309.5', '38.124']]}
#         >> packet: {'action': 'update', 'arg': {'instType': 'mc', 'channel': 'candle1m', 'instId': 'BTCUSDT'}, 'data': [['1681781760000', '29314', '29314', '29308.5', '29309.5', '38.162']]}
#         >> packet: {'action': 'update', 'arg': {'instType': 'mc', 'channel': 'candle1m', 'instId': 'BTCUSDT'}, 'data': [['1681781760000', '29314', '29314', '29308.5', '29310', '49.102']]}
#         >> packet: {'action': 'update', 'arg': {'instType': 'mc', 'channel': 'candle1m', 'instId': 'BTCUSDT'}, 'data': [['1681781760000', '29314', '29314', '29308.5', '29309.5', '49.108']]}
#         >> packet: {'action': 'update', 'arg': {'instType': 'mc', 'channel': 'candle1m', 'instId': 'BTCUSDT'}, 'data': [['1681781760000', '29314', '29314', '29308.5', '29309.5', '49.109']]}

#         orderbook:
#         >> packet: {'event': 'subscribe', 'arg': {'instType': 'mc', 'channel': 'books1', 'instId': 'BTCUSDT'}}
#         >> packet: {'action': 'snapshot', 'arg': {'instType': 'mc', 'channel': 'books1', 'instId': 'BTCUSDT'}, 'data': [{'asks': [['29221.0', '3.440']], 'bids': [['29220.5', '3.048']], 'ts': '1681781051015'}]}
#         >> packet: {'action': 'snapshot', 'arg': {'instType': 'mc', 'channel': 'books1', 'instId': 'BTCUSDT'}, 'data': [{'asks': [['29221.0', '3.440']], 'bids': [['29220.5', '3.048']], 'ts': '1681781051015'}]}
#         >> packet: {'action': 'snapshot', 'arg': {'instType': 'mc', 'channel': 'books1', 'instId': 'BTCUSDT'}, 'data': [{'asks': [['29221.0', '4.884']], 'bids': [['29220.5', '3.023']], 'ts': '1681781051020'}]}
#         >> packet: {'action': 'snapshot', 'arg': {'instType': 'mc', 'channel': 'books1', 'instId': 'BTCUSDT'}, 'data': [{'asks': [['29221.0', '5.421']], 'bids': [['29220.5', '4.000']], 'ts': '1681781051122'}]}

#         tickers:
#             contract:
#             >> packet: {'event': 'subscribe', 'arg': {'instType': 'mc', 'channel': 'ticker', 'instId': 'BTCUSDT'}}
#             >> packet: {'action': 'snapshot', 'arg': {'instType': 'mc', 'channel': 'ticker', 'instId': 'BTCUSDT'}, 'data': [{'instId': 'BTCUSDT', 'last': '29300.00', 'bestAsk': '29302', 'bestBid': '29301.5', 'high24h': '30078.50', 'low24h': '29085.00', 'priceChangePercent': '-0.00201', 'capitalRate': '0.000078', 'nextSettleTime': 1681801200000, 'systemTime': 1681781589757, 'markPrice': '29296.02', 'indexPrice': '29320.59', 'holding': '66003.348', 'baseVolume': '135437.635', 'quoteVolume': '4007385066.147', 'openUtc': '29414.5000000000000000', 'chgUTC': '-0.00418', 'symbolType': 1, 'symbolId': 'BTCUSDT_UMCBL', 'deliveryPrice': '0', 'bidSz': '79.777', 'askSz': '3.001'}]}
#             >> packet: {'action': 'snapshot', 'arg': {'instType': 'mc', 'channel': 'ticker', 'instId': 'BTCUSDT'}, 'data': [{'instId': 'BTCUSDT', 'last': '29300.00', 'bestAsk': '29302', 'bestBid': '29301.5', 'high24h': '30078.50', 'low24h': '29085.00', 'priceChangePercent': '-0.00201', 'capitalRate': '0.000078', 'nextSettleTime': 1681801200000, 'systemTime': 1681781589757, 'markPrice': '29296.02', 'indexPrice': '29320.59', 'holding': '66003.348', 'baseVolume': '135437.635', 'quoteVolume': '4007385066.147', 'openUtc': '29414.5000000000000000', 'chgUTC': '-0.00418', 'symbolType': 1, 'symbolId': 'BTCUSDT_UMCBL', 'deliveryPrice': '0', 'bidSz': '79.777', 'askSz': '3.001'}]}

#             spot:
#             >> packet: {'event': 'subscribe', 'arg': {'instType': 'sp', 'channel': 'ticker', 'instId': 'BTCUSDT'}}
#             >> packet: {'action': 'snapshot', 'arg': {'instType': 'sp', 'channel': 'ticker', 'instId': 'BTCUSDT'}, 'data': [{'instId': 'BTCUSDT', 'last': '29477.55', 'open24h': '29358.59', 'high24h': '30025.04', 'low24h': '29108.62', 'bestBid': '29477.41', 'bestAsk': '29478.09', 'baseVolume': '6296.5400', 'quoteVolume': '186487524.6870', 'ts': 1681787857705, 'labeId': 0, 'openUtc': '29429.9600000000000000', 'chgUTC': '0.00161', 'bidSz': '1.9504', 'askSz': '0.8091'}]}
#             >> packet: {'action': 'snapshot', 'arg': {'instType': 'sp', 'channel': 'ticker', 'instId': 'BTCUSDT'}, 'data': [{'instId': 'BTCUSDT', 'last': '29477.55', 'open24h': '29358.59', 'high24h': '30025.04', 'low24h': '29108.62', 'bestBid': '29477.41', 'bestAsk': '29478.09', 'baseVolume': '6296.5400', 'quoteVolume': '186487524.6870', 'ts': 1681787857705, 'labeId': 0, 'openUtc': '29429.9600000000000000', 'chgUTC': '0.00161', 'bidSz': '1.9504', 'askSz': '0.8091'}]}
#             >> packet: {'action': 'snapshot', 'arg': {'instType': 'sp', 'channel': 'ticker', 'instId': 'BTCUSDT'}, 'data': [{'instId': 'BTCUSDT', 'last': '29477.55', 'open24h': '29358.59', 'high24h': '30025.04', 'low24h': '29108.62', 'bestBid': '29477.6', 'bestAsk': '29478.09', 'baseVolume': '6296.6982', 'quoteVolume': '186492188.0354', 'ts': 1681787858015, 'labeId': 0, 'openUtc': '29429.9600000000000000', 'chgUTC': '0.00162', 'bidSz': '0.0169', 'askSz': '2.0133'}]}

#         """
#         # logger.debug(f'packet: {packet}')
#         # print(f'>> packet: {packet}')
#         # # return

#         event = packet.get("event", "")
#         if event:
#             if event == "error":
#                 print(f"Something went wrong after subscribe topic: {packet}")
#             else:
#                 print(packet)
#             return

#         instid: str = packet["arg"]["instId"]
#         channel: str = packet["arg"]["channel"]
#         # print(f"channel:{channel}; instid:{instid}; 'candle' in channel:{'candle' in channel}")

#         is_fire: bool = False
#         tick = None
#         if "candle" in channel:
#             if packet["action"] == "snapshot":  ### NOTE: ignore the snapshot
#                 return
#             interval = KBAR_INTERVAL_REV[channel.replace("candle", "")]
#             tick: dict = self.ticks[f"{instid}|{interval}"]

#             data: dict = packet["data"][0]
#             tick["volume"] = float(data[5])
#             tick["open"] = float(data[1])
#             tick["high"] = float(data[2])
#             tick["low"] = float(data[3])
#             tick["close"] = float(data[4])
#             tick["turnover"] = tick["volume"] * tick["close"]
#             tick["start"] = models.timestamp_to_datetime(float(data[0]))
#             tick["datetime"] = models.generate_datetime()  # datetime.now()

#             if self.on_kline_callback:
#                 is_fire = True
#                 self.on_kline_callback(copy(tick))

#         elif channel == "ticker":
#             # if packet['action'] == 'snapshot': ### NOTE: ignore the snapshot
#             #     # print(packet)
#             #     # sys.exit()
#             #     return
#             tick: dict = self.ticks[f"{instid}|ticker"]
#             data: dict = packet["data"][0]

#             tick["prev_open_24h"] = (
#                 float(data["open24h"]) if "open24h" in data else None
#             )
#             tick["prev_high_24h"] = float(data["high24h"])
#             tick["prev_low_24h"] = float(data["low24h"])
#             tick["prev_volume_24h"] = float(data["baseVolume"])
#             tick["prev_turnover_24h"] = float(data["quoteVolume"])

#             tick["last_price"] = float(data["last"])
#             tick["price_change_pct"] = (
#                 float(data["priceChangePercent"])
#                 if "priceChangePercent" in data
#                 else None
#             )
#             tick["prev_close_24h"] = (
#                 (tick["last_price"] / (1 + tick["price_change_pct"]))
#                 if tick["price_change_pct"]
#                 else None
#             )
#             tick["price_change"] = (
#                 (tick["last_price"] - tick["prev_close_24h"])
#                 if tick["price_change_pct"]
#                 else None
#             )

#             tick["bid_price_1"] = float(data["bestBid"]) if "bestBid" in data else None
#             tick["bid_volume_1"] = float(data["bidSz"]) if "bidSz" in data else None
#             tick["ask_price_1"] = float(data["bestAsk"]) if "bestAsk" in data else None
#             tick["ask_volume_1"] = float(data["askSz"]) if "askSz" in data else None
#             tick["datetime_now"] = models.generate_datetime()
#             tick["datetime"] = models.timestamp_to_datetime(
#                 float(data["systemTime"] if "systemTime" in data else data["ts"])
#             )

#             if self.on_ticker_callback:
#                 is_fire = True
#                 self.on_ticker_callback(copy(tick))

#         elif "books" in channel:
#             tick: dict = self.ticks[f"{instid}|depth"]
#             data: dict = packet["data"][0]

#             bids: list = data["bids"]
#             for n in range(min(5, len(bids))):
#                 price = bids[n][0]
#                 volume = bids[n][1]
#                 tick["bid_price_" + str(n + 1)] = float(price)
#                 tick["bid_volume_" + str(n + 1)] = float(volume)

#             asks: list = data["asks"]
#             for n in range(min(5, len(asks))):
#                 price = asks[n][0]
#                 volume = asks[n][1]
#                 tick["ask_price_" + str(n + 1)] = float(price)
#                 tick["ask_volume_" + str(n + 1)] = float(volume)

#             tick["datetime_now"] = models.generate_datetime()
#             tick["datetime"] = models.timestamp_to_datetime(float(data["ts"]))

#             if self.on_ticker_callback:
#                 is_fire = True
#                 self.on_ticker_callback(copy(tick))

#         if not is_fire and self.on_tick and tick:
#             self.on_tick(copy(tick))


# class BitgetTradeWebsocket(BitgetDataWebsocket):
#     """
#     Implement private topics
#     doc: https://bitget-exchange.github.io/docs/linear/#t-privatetopics
#     """

#     DATA_TYPE = None

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
#         datatype: str = None,
#         is_testnet: bool = False,
#         ping_interval: int = 25,
#         debug: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__(
#             proxy_host=proxy_host,
#             proxy_port=proxy_port,
#             ping_interval=ping_interval,
#             is_testnet=is_testnet,
#             datatype=datatype,
#             debug=debug,
#             **kwargs,
#         )

#         self.on_account_callback = on_account_callback
#         self.on_order_callback = on_order_callback
#         self.on_trade_callback = on_trade_callback
#         self.on_position_callback = on_position_callback
#         self.on_connected_callback = on_connected_callback
#         self.on_disconnected_callback = on_disconnected_callback
#         self.on_error_callback = on_error_callback

#         self.debug = debug

#     def info(self):
#         return f"Bitget {self.DATA_TYPE} Trade Websocket Start"

#     def connect(self, key: str, secret: str, passphrase: str, is_testnet: bool = None):
#         """ """
#         self.key = key
#         self.secret = secret
#         self.passphrase = passphrase

#         if self.DATA_TYPE == "spot":
#             host = BITGET_WS_SPOT_HOST
#         elif self.DATA_TYPE in ["linear", "inverse"]:
#             host = BITGET_WS_CONTRACT_HOST

#         self.init(
#             host=host,
#             proxy_host=self.proxy_host,
#             proxy_port=self.proxy_port,
#             ping_interval=self.ping_interval,
#         )
#         self.start()

#         self.waitfor_connection()

#         if not self._logged_in:
#             if self.debug:
#                 print(
#                     f">>> in [connect] send_auth_packet [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
#                 )
#             self._login()
#         return True

#     def _login(self):
#         """
#         Authorize websocket connection.
#         """
#         # Generate expires.
#         ts = int(round(time.time()))
#         api_key = self.key
#         api_secret = self.secret
#         api_passphrase = self.passphrase

#         if api_key is None or api_secret is None or api_passphrase is None:
#             raise PermissionError(
#                 f"Authenticated endpoints require keys. api_key:{api_key}; api_secret:{api_secret}"
#             )

#         sign_message = BitgetClient.pre_hash(
#             timestamp=str(ts), method="GET", request_path="/user/verify", body=None
#         )
#         signature = BitgetClient._sign(message=sign_message, secret_key=api_secret)

#         req: dict = {
#             "op": "login",
#             "args": [
#                 {
#                     "apiKey": api_key,
#                     "passphrase": api_passphrase,
#                     "timestamp": str(ts),
#                     "sign": signature.decode("utf8"),
#                 }
#             ],
#         }
#         if self.debug:
#             print(f"auth req:{req}({type(req)}) | sign_message:{sign_message}| ts:{ts}")

#         self.send_packet(req)

#         # self._logged_in = True

#     def on_disconnected(self) -> None:
#         print(
#             f"Bitget user trade data Websocket disconnected, now try to connect @ {datetime.now()}"
#         )
#         self._logged_in = False
#         if self.on_disconnected_callback:
#             self.on_disconnected_callback()

#     def on_connected(self) -> None:
#         if self.waitfor_connection():
#             print("Bitget user trade data websocket connect success")
#         else:
#             print("Bitget user trade data websocket connect fail")

#         if not self._logged_in:
#             if self.debug:
#                 print(
#                     f">>> in [on_connected] send_auth_packet [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
#                 )
#             self._login()

#     def _subscribe(self):
#         req: dict = self.sub_stream(channel=["trades", "orders", "account", "position"])
#         if self.debug:
#             print(f"Bitget on_connected req:{req} - self._active:{self._active}")

#         if self.debug:
#             print(
#                 f">>> in [on_connected] send_sub_packet [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
#             )
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
#         sub = {
#             "op": "subscribe",
#         }
#         if isinstance(channel, list):
#             sub["args"] = [
#                 {
#                     "channel": MAPPING_CHANNEL[c],
#                     "instId": "default",
#                     "instType": self.product_type.upper(),
#                 }
#                 for c in channel
#                 if c != "trades"
#             ]
#         else:
#             sub["args"] = [
#                 {
#                     "channel": MAPPING_CHANNEL[channel]
#                     if channel != "trades"
#                     else "orders",
#                     "instId": "default",
#                     "instType": self.product_type.upper(),
#                 }
#             ]

#         return sub

#     def subscribe(self, channel: str = None) -> None:
#         """ """
#         if channel and channel not in ["trades", "orders", "account", "position"]:
#             Exception("invalid subscription")
#             return

#         if not self._logged_in:
#             self._login()

#         ### Default sub all private channels if not specified
#         self.channel = channel
#         req: dict = self.sub_stream(channel=channel)
#         if self.debug:
#             print(f"subscribe - req: {req}")
#         self.send_packet(req)

#     def on_packet(self, packet: dict) -> None:
#         """

#         contract:
#         positions:
#             [DEBUG] Bitget Trade Websocket - on_packet event {'action': 'snapshot', 'arg': {'instType': 'sumcbl', 'channel': 'positions', 'instId': 'default'}, 'data': [{'posId': '1032128976686002178', 'instId': 'SBTCSUSDT_SUMCBL', 'instName': 'SBTCSUSDT', 'marginCoin': 'SUSDT', 'margin': '4.2515', 'marginMode': 'fixed', 'holdSide': 'long', 'holdMode': 'single_hold', 'total': '0.001', 'available': '0.001', 'locked': '0', 'averageOpenPrice': '29760.5', 'leverage': 7, 'achievedProfits': '0', 'upl': '-0.0617', 'uplRate': '-0.0145', 'liqPx': '25626.88', 'keepMarginRate': '0.004', 'fixedMarginRate': '0.140474382642', 'marginRate': '0.032606912140', 'cTime': '1681804644748', 'uTime': '1681804675548', 'markPrice': '29698.73'}]}
#         account:
#             [DEBUG] Bitget Trade Websocket - on_packet event {'action': 'snapshot', 'arg': {'instType': 'sumcbl', 'channel': 'account', 'instId': 'default'}, 'data': [{'marginCoin': 'SUSDT', 'locked': '4.57628285', 'available': '2995.73064370', 'maxOpenPosAvailable': '2991.15436084', 'maxTransferOut': '2991.15436084', 'equity': '2999.92037370', 'usdtEquity': '2999.920373700000'}]}
#         orders
#             [DEBUG] Bitget Trade Websocket - on_packet event {'action': 'snapshot', 'arg': {'instType': 'sumcbl', 'channel': 'orders', 'instId': 'default'}, 'data': [{'accFillSz': '0', 'cTime': 1681806901976, 'clOrdId': '1032138444186431488', 'eps': 'WEB', 'force': 'normal', 'hM': 'single_hold', 'instId': 'SBTCSUSDT_SUMCBL', 'lever': '7', 'low': False, 'notionalUsd': '58.004', 'ordId': '1032138444165459974', 'ordType': 'limit', 'orderFee': [{'feeCcy': 'SUSDT', 'fee': '0'}], 'posSide': 'net', 'px': '29002', 'side': 'buy', 'status': 'new', 'sz': '0.002', 'tS': 'buy_single', 'tdMode': 'isolated', 'tgtCcy': 'SUSDT', 'uTime': 1681806901976}]}


#         [DEBUG] Bitget Trade Websocket - on_packet event {'action': 'snapshot', 'arg': {'instType': 'sumcbl', 'channel': 'orders', 'instId': 'default'}, 'data': [{'accFillSz': '0.001', 'avgPx': '29709', 'cTime': 1681806984585, 'clOrdId': '1032138790673690625', 'eps': 'WEB', 'execType': 'T', 'fillFee': '-0.0178254', 'fillFeeCcy': 'SUSDT', 'fillNotionalUsd': '29.709', 'fillPx': '29709', 'fillSz': '0.001', 'fillTime': '1681806984623', 'force': 'normal', 'hM': 'single_hold',
#         'instId': 'SBTCSUSDT_SUMCBL', 'lever': '7', 'low': False, 'notionalUsd': '29.70743', 'ordId': '1032138790669496323', 'ordType': 'market', 'orderFee': [{'feeCcy': 'SUSDT', 'fee': '-0.0178254'}], 'pnl': '0', 'posSide': 'net', 'px': '0', 'side': 'buy', 'status': 'full-fill', 'sz': '0.001', 'tS': 'buy_single', 'tdMode': 'isolated', 'tgtCcy': 'SUSDT', 'tradeId': '1032138790836842497', 'uTime': 1681806984623}]}
#         [DEBUG] Bitget Trade Websocket - on_packet event {'action': 'snapshot', 'arg': {'instType': 'sumcbl', 'channel': 'positions', 'instId': 'default'}, 'data': [{'posId': '1032128976686002178', 'instId': 'SBTCSUSDT_SUMCBL', 'instName': 'SBTCSUSDT', 'marginCoin': 'SUSDT', 'margin': '8.4956', 'marginMode': 'fixed', 'holdSide': 'long', 'holdMode': 'single_hold', 'total': '0.002', 'available': '0.002', 'locked': '0', 'averageOpenPrice': '29734.75', 'leverage': 7, 'achievedProfits': '0', 'upl': '-0.0636', 'uplRate': '-0.0074', 'liqPx': '25604.71', 'keepMarginRate': '0.004', 'fixedMarginRate': '0.141338329564', 'marginRate': '0.032408441146', 'cTime': '1681804644748', 'uTime': '1681806984645', 'markPrice': '29702.91'}]}
#         [DEBUG] Bitget Trade Websocket - on_packet event {'action': 'snapshot', 'arg': {'instType': 'sumcbl', 'channel': 'orders', 'instId': 'default'}, 'data': [{'accFillSz': '0', 'cTime': 1681806999910, 'clOrdId': '1032138854951399424', 'eps': 'WEB', 'force': 'normal', 'hM': 'single_hold', 'instId': 'SBTCSUSDT_SUMCBL', 'lever': '7', 'low': False, 'notionalUsd': '29.70291', 'ordId': '1032138854947205123', 'ordType': 'market', 'orderFee': [{'feeCcy': 'SUSDT', 'fee': '0'}], 'posSide': 'net', 'px': '0', 'side': 'sell', 'status': 'new', 'sz': '0.001', 'tS': 'sell_single', 'tdMode': 'isolated', 'tgtCcy': 'SUSDT', 'uTime': 1681806999910}]}
#         [DEBUG] Bitget Trade Websocket - on_packet event {'action': 'snapshot', 'arg': {'instType': 'sumcbl', 'channel': 'account', 'instId': 'default'}, 'data': [{'marginCoin': 'SUSDT', 'locked': '13.72947978', 'available': '2995.62858897','maxOpenPosAvailable': '2981.89910918', 'maxTransferOut': '2981.89910918', 'equity': '2999.87641040', 'usdtEquity': '2999.876410400000'}]}
#         [DEBUG] Bitget Trade Websocket - on_packet event {'action': 'snapshot', 'arg': {'instType': 'sumcbl', 'channel': 'positions', 'instId': 'default'}, 'data': [{'posId': '1032128976686002178', 'instId': 'SBTCSUSDT_SUMCBL', 'instName': 'SBTCSUSDT', 'marginCoin': 'SUSDT', 'margin': '4.2796', 'marginMode': 'fixed', 'holdSide': 'long', 'holdMode': 'single_hold', 'total': '0.001', 'available': '0.001', 'locked': '0', 'averageOpenPrice': '29734.75', 'leverage': 7, 'achievedProfits': '-0.0382', 'upl': '-0.0318', 'uplRate': '-0.0074', 'liqPx': '25572.72', 'keepMarginRate': '0.004', 'fixedMarginRate': '0.142410278407', 'marginRate': '0.032165520208', 'cTime': '1681804644748', 'uTime': '1681806999979', 'markPrice': '29702.91'}]}
#         [DEBUG] Bitget Trade Websocket - on_packet event {'action': 'snapshot', 'arg': {'instType': 'sumcbl', 'channel': 'orders', 'instId': 'default'}, 'data': [{'accFillSz': '0.001', 'avgPx': '29696.5', 'cTime': 1681806999910, 'clOrdId': '1032138854951399424', 'eps': 'WEB', 'execType': 'T', 'fillFee': '-0.0178179', 'fillFeeCcy': 'SUSDT', 'fillNotionalUsd': '29.6965', 'fillPx': '29696.5', 'fillSz': '0.001', 'fillTime': '1681806999957', 'force': 'normal', 'hM': 'single_hold', 'instId': 'SBTCSUSDT_SUMCBL', 'lever': '7', 'low': False, 'notionalUsd': '29.70291', 'ordId': '1032138854947205123', 'ordType': 'market', 'orderFee': [{'feeCcy': 'SUSDT', 'fee': '-0.0178179'}], 'pnl': '-0.03825', 'posSide': 'net', 'px': '0', 'side': 'sell', 'status': 'full-fill', 'sz': '0.001', 'tS': 'sell_single', 'tdMode': 'isolated', 'tgtCcy': 'SUSDT', 'tradeId': '1032138855152300033', 'uTime': 1681806999957}]}

#         """
#         if self.debug:
#             print(f"[DEBUG] Bitget Trade Websocket - on_packet event {packet}")
#             # return

#         event = packet.get("event", "")
#         if event:
#             if event == "login" and packet["code"] == 0:
#                 self._logged_in = True
#                 self._subscribe()
#             elif event == "error":
#                 print(f"Something went wrong after subscribe topic: {packet}")
#             else:
#                 print(packet)
#             return

#         if not packet:
#             logger.debug(f"unknown packet event {packet}")
#             return

#         channel: str = packet["arg"]["channel"]

#         if channel == "account":
#             self.on_account(packet)
#         elif channel == "orders":
#             self.on_order(packet)
#         # elif channel == "trades":
#         #     self.on_trade(packet)
#         elif channel == "positions":
#             self.on_position(packet)
#         else:
#             logger.info(f"the other packet type: {packet}")

#     def on_account(self, packet: dict) -> None:
#         """ """
#         # logger.debug(f"packet on_account: {packet}")

#         for data in packet["data"]:
#             if self.product_type:
#                 account_type = "CONTRACT"
#             else:
#                 account_type = "SPOT"

#             account = models.AccountSchema()  # .model_construct(data)
#             account = account.from_bitget_to_form(
#                 data, datatype=self.DATA_TYPE, account_type=account_type
#             )

#             if account and account["symbol"] and self.on_account_callback:
#                 self.on_account_callback(account)

#     def on_order(self, packet: dict) -> None:
#         """
#         [DEBUG] Bitget linear Trade Websocket - on_packet event {'id': '1193481acf42bfa-cbca-4232-93a2-dcf9dead9510', 'topic': 'order', 'creationTime': 1678887821206, 'data': [{'symbol': 'BTCUSDT', 'orderId': 'c2153424-5ab3-456c-b8e8-98bc266dad0e', 'side': 'Buy', 'orderType': 'Market', 'cancelType': 'UNKNOWN', 'price': '26140.4', 'qty': '0.001', 'orderIv': '', 'timeInForce': 'IOC', 'orderStatus': 'Filled', 'orderLinkId': '', 'lastPriceOnCreated': '24895.7', 'reduceOnly': False, 'leavesQty': '0', 'leavesValue': '0', 'cumExecQty': '0.001', 'cumExecValue': '24.8993', 'avgPrice': '24899.3', 'blockTradeId': '', 'positionIdx': 0, 'cumExecFee': '0.01493958', 'createdTime': '1678887821191', 'updatedTime': '1678887821200', 'rejectReason': 'EC_NoError', 'stopOrderType': '', 'triggerPrice': '', 'takeProfit': '', 'stopLoss': '', 'tpTriggerBy': '', 'slTriggerBy': '', 'triggerDirection': 0, 'triggerBy': '', 'closeOnTrigger': False, 'category': 'linear', 'isLeverage': ''}]}
#         """
#         # logger.debug(f"packet on_order: {packet}")
#         datas: dict = packet["data"]

#         # order_list = list()
#         for data in datas:
#             order = models.OrderSchema()  # .model_construct(data)
#             order = order.from_bitget_to_form(data, datatype=self.DATA_TYPE)
#             # print(f"order: {order}?????") #TODO: price=0??
#             if order:
#                 if order["status"] != "FILLED":
#                     if self.on_order_callback:  # and order_list
#                         self.on_order_callback(order)
#                 else:
#                     self.on_trade(
#                         {
#                             "symbol": data["instId"],
#                             "orderId": data["ordId"],
#                             "fillPrice": data["avgPx"],
#                             "fillQuantity": data["accFillSz"],
#                             "side": data["side"],
#                             "fees": sum(
#                                 [float(i["fee"]) for i in data["orderFee"]]
#                             ),  # NOTE: FeeNegative number represents the user transaction fee charged by the platform.Positive number represents rebate.
#                             "feeCcy": data["orderFee"][-1]["feeCcy"],
#                             "cTime": data["fillTime"],
#                         }
#                     )

#     def on_trade(self, data: dict) -> None:
#         """
#         [DEBUG] Bitget linear Trade Websocket - on_packet event {'id': '1193481299665b2-56be-40da-8a29-f488cb47190b', 'topic': 'execution', 'creationTime': 1678888392300, 'data': [{'category': 'linear', 'symbol': 'BTCUSDT', 'execFee': '0.01490148', 'execId': 'dae9c7a8-ceb9-51fa-8881-d5157ecc2c51', 'execPrice': '24835.8', 'execQty': '0.001', 'execType': 'Trade', 'execValue': '24.8358', 'isMaker': False, 'feeRate': '0.0006', 'tradeIv': '', 'markIv': '', 'blockTradeId': '', 'markPrice': '24883.13', 'indexPrice': '', 'underlyingPrice': '', 'leavesQty': '0', 'orderId': 'e9b9e551-081b-414c-a109-0e56d225ecd2', 'orderLinkId': '', 'orderPrice': '26116.9', 'orderQty': '0.001', 'orderType': 'Market', 'stopOrderType': 'UNKNOWN', 'side': 'Buy', 'execTime': '1678888392289', 'isLeverage': '0'}]}
#         """
#         logger.debug(f"data on_trade: {data}")

#         trade = models.TradeSchema()  # .model_construct(data)
#         trade = trade.from_bitget_to_form(data, datatype=self.DATA_TYPE)
#         if trade and self.on_trade_callback:
#             self.on_trade_callback(trade)

#     def on_position(self, packet: dict) -> None:
#         """ """
#         # logger.debug(f"packet on_position: {packet}")
#         # print(f"packet on_position: {packet}")
#         datas: dict = packet["data"]
#         for data in datas:
#             position = models.PositionSchema()  # .model_construct(data)
#             position = position.from_bitget_to_form(data, datatype=self.DATA_TYPE)

#             if position and self.on_position_callback:
#                 self.on_position_callback(position)

#         # print(tick.get('last_price', '0'), '~~~~~~~')
#         print(tick, "~~~~~~~")

#     market_ws_api = BitgetDataWebsocket(datatype="linear", is_testnet=False)
#     # market_ws_api = BitgetDataWebsocket(datatype="spot", is_testnet=False)
#     # market_ws_api = BitgetDataWebsocket(datatype='linear', is_testnet=False)
#     # market_ws_api = BitgetFutureDataWebsocket()
#     # market_ws_api = BitgetInverseFutureDataWebsocket(is_testnet=True)
#     # market_ws_api = BitgetOptionDataWebsocket(is_testnet=True)
#     market_ws_api.connect()

#     market_ws_api.subscribe(["BTCUSDT_SPBL"], test_on_tick, channel="ticker")
#     market_ws_api.subscribe(["BTCUSDT_SPBL"], test_on_tick, channel="depth")
#     market_ws_api.subscribe(
#         ["BTCUSDT_SPBL"], test_on_tick, channel="kline", interval="5m"
#     )

#     # market_ws_api.subscribe(["SBTCSUSDT_SUMCBL"], test_on_tick, channel="ticker")

#     # market_ws_api.subscribe(["BTCUSDT_SUMCBL"], test_on_tick, channel="ticker")
#     # market_ws_api.subscribe(["BTCUSDT_SUMCBL"], test_on_tick, channel="kline", interval='1m')
#     # market_ws_api.subscribe(["BTCUSDT_SUMCBL"], test_on_tick, channel="kline", interval='1d')

#     import time

#     time.sleep(400)
#     print("!!!!!!!!" * 10)
#     # market_ws_api.subscribe(['ETHUSDT', 'APEUSDT'], test_on_tick, channel='depth')
#     # time.sleep(3)

#     market_ws_api.close()
