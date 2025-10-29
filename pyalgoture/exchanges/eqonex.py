# import hashlib
# import hmac
# import json
# import logging
# import time
# from collections import defaultdict
# from copy import copy, deepcopy
# from datetime import datetime, timedelta

# # from threading import Lock
# from typing import Any, Optional, dict

# from ..utils import models
# from ..utils.client_rest import Response, RestClient
# from ..utils.client_ws import WebsocketClient
# from ..utils.models import ASSET_TYPE_SPOT, UTC_TZ

# logger = logging.getLogger("eqonex")

# EXCHANGE = "EQONEX"

# TIMEDELTA_MAP: dict[str, timedelta] = {
#     "1m": timedelta(minutes=1),
#     "1h": timedelta(hours=1),
#     "1d": timedelta(days=1),
# }

# EQONEX_API_HOST = "https://eqonex.com/api"
# EQONEX_TESTNET_REST_HOST = "https://testnet.eqonex.com/api"

# EQONEX_WEBSOCKET_HOST = "wss://eqonex.com/wsapi"
# EQONEX_TESTNET_WEBSOCKET_HOST = "wss://testnet.eqonex.com/wsapi"

# ORDER_STATUS_MAP = {
#     "0": "NEW",
#     "1": "PARTIALLY_FILLED",
#     "2": "FILLED",
#     "4": "CANCELED",
#     "8": "REJECTED",
#     "C": "EXPIRED",
# }

# ORDER_TYPE_MAP = {"1": "MARKET", "2": "LIMIT", "3": "STOP", "4": "STOP LIMIT"}

# SIDE_MAP = {"1": "BUY", "2": "SELL"}
# INTERVAL_MAP = {"1m": 1, "5m": 2, "15m": 3, "1h": 4, "6h": 5, "1d": 6, "1w": 7}
# INTERVAL_MAP_REV = {1: "1m", 2: "5m", 3: "15m", 4: "1h", 5: "6h", 6: "1d", 7: "1w"}

# ORDER_TYPE_MAP_INV = {"MARKET": 1, "LIMIT": 2, "STOP": 3, "STOP LIMIT": 4}
# SIDE_MAP_INV = {"BUY": 1, "SELL": 2}


# class EqonexClient(RestClient):
#     """
#     api docs: https://developer.eqonex.com/#introduction
#     """

#     DATA_TYPE = "spot"  # only applied to symbols, orders, and trades
#     BROKER_ID = ""

#     def __init__(
#         self,
#         url_base: str = EQONEX_API_HOST,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         data_type: str = "spot",
#         is_testnet: bool = False,
#     ) -> None:
#         super().__init__(url_base, proxy_host, proxy_port)

#         self.DATA_TYPE = data_type  # for symbol, orders, trades, account

#         self.order_count: int = 0
#         # self.order_count_lock: Lock = Lock()
#         self.connect_time: int = 0
#         self.key: str = ""
#         self.secret: str = ""
#         self.account_id: str = ""
#         self.user_id: str = ""

#         self.symbols_info_by_id = {}
#         self.symbols_info = {}
#         self.symbols_info_raw = {}

#         self.symbols_info_raw = self.query_symbols(return_all=True)
#         if self.symbols_info_raw.get("success"):
#             self.symbols_info = self.symbols_info_raw["data"]
#             # print(self.symbols_info,'!!!')
#             if not self.symbols_info.get("error"):
#                 self.symbols_info_by_id = {
#                     v["extra"]["instrument_id"]: v for k, v in self.symbols_info.items()
#                 }
#             # print('Length of crypto list', len(self.symbols_info))
#         else:
#             print("Something went wrong when initialize Eqonex (query symbols failed).")

#     def info(self):
#         return "Eqonex REST API start"

#     @staticmethod
#     def generate_timestamp(expire_after: float = 30) -> int:
#         """generate_timestamp"""
#         return int(time.time() * 1000 + expire_after * 1000)

#     @staticmethod
#     def generate_datetime(datetime_str: str = None) -> datetime:
#         """generate_datetime"""
#         if datetime_str:
#             dt: datetime = datetime.strptime(datetime_str, "%Y%m%d-%H:%M:%S.%f")
#             dt: datetime = UTC_TZ.localize(dt)
#             dt: datetime = dt.astimezone(UTC_TZ)
#         else:
#             dt: datetime = datetime.now(UTC_TZ)
#         return dt

#     @staticmethod
#     def generate_datetime_ts(timestamp: float) -> datetime:
#         """generate_datetime"""
#         dt: datetime = datetime.fromtimestamp(timestamp / 1000, tz=UTC_TZ)
#         dt: datetime = dt.astimezone(UTC_TZ)
#         return dt

#     @staticmethod
#     def change_scale(num: int, scale: int):
#         return num * 10 ** (-scale)

#     def new_order_id(self) -> str:
#         """new_order_id"""
#         # prefix: str = datetime.now().strftime("%Y%m%d-%H%M%S-")
#         prefix: str = datetime.now().strftime("%y%m%d-%H%M%S%f")[:-3] + "-"
#         # with self.order_count_lock:
#         self.order_count += 1
#         suffix: str = str(self.order_count).rjust(5, "0")

#         order_id: str = f"x-{self.BROKER_ID}" + prefix + suffix
#         return order_id

#     def _get_extra(self, symbol: str):
#         symbol = symbol.upper()
#         symbol_info = self.symbols_info.get(symbol, {})
#         extra = symbol_info.get("extra", {})
#         if not extra:
#             return False, models.CommonResponseSchema(
#                 success=False, error=True, data={}, msg="Invalid Symbol."
#             )
#         else:
#             return True, extra

#     def _remove_none(self, payload):
#         return {k: v for k, v in payload.items() if v is not None}

#     def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
#         # response = self.request(method="GET", path=path,  params=params)
#         response = self.query(method="GET", path=path, params=params)
#         return self._process_response(response)

#     def _post(
#         self,
#         path: str,
#         data: Optional[dict[str, Any]] = None,
#         params: Optional[dict[str, Any]] = None,
#     ) -> Any:
#         # response = self.request(method="POST", path=path, data=params)
#         response = self.query(method="POST", path=path, data=data, params=params)
#         return self._process_response(response)

#     def connect(
#         self, key: str, secret: str, user_id: str, account_id: str = None
#     ) -> None:
#         """connect exchange server"""
#         self.user_id = str(user_id)
#         self.key = key
#         self.secret = secret.encode()
#         self.account_id = str(account_id) if account_id else None

#         # print(self.key, secret, user_id,'.....')

#         self.start()
#         logger.debug(self.info())

#     def sign(self, request):
#         if request.method == "POST":
#             if request.data:
#                 api_params: dict = {**{"userId": self.user_id}, **request.data}
#             else:
#                 api_params: dict = {"userId": self.user_id}
#             if self.account_id:
#                 api_params["account"] = self.account_id

#             request.data = api_params.copy()
#             data2sign = json.dumps(api_params)  # separators=(",", ":"))
#             # print(type(request.data), request.data, '||||',type(data2sign),data2sign)

#             signature: str = hmac.new(
#                 self.secret, data2sign.encode(), hashlib.sha384
#             ).hexdigest()
#             headers: dict = {"requestToken": self.key, "signature": signature}
#             # , 'Content-Type': 'application/json'
#             # , 'Content-Type': 'application/x-www-form-urlencoded'

#             request.headers = headers

#         return request

#     def _process_response(self, response: Response) -> dict:
#         try:
#             data = response.data
#             # print(data,'!!!!', response.ok)
#         except ValueError:
#             print("!!eqonex - _process_response !!", response.text)
#             logger.debug(response.data())
#             raise
#         else:
#             # logger.debug(data)

#             if not response.ok or (response.ok and data.get("error")):
#                 payload_data = models.CommonDataSchema(
#                     status_code=response.status_code, msg=dict(data)
#                 )
#                 if not data.get("error"):
#                     print("!!eqonex - return raw data !!", response.text)
#                 # print('??????', payload_data.dict(), data,'????')
#                 return models.CommonResponseSchema(
#                     success=False,
#                     error=True,
#                     data=payload_data.dict(),
#                     msg=data.get("error", "Eqonex server under maintenance."),
#                 )
#             elif response.ok and not data:
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
#                 return models.CommonResponseSchema(
#                     success=True, error=False, data=data, msg="query ok"
#                 )

#     def _regular_order_payload(
#         self, common_response: models.CommonResponseSchema
#     ) -> dict:
#         def convert_msg(msg):
#             msg["ordType"] = ORDER_TYPE_MAP.get(msg["ordType"], msg["ordType"])
#             msg["side"] = SIDE_MAP.get(msg["side"], msg["side"])
#             msg["ordStatus"] = ORDER_STATUS_MAP.get(msg["ordStatus"], msg["ordStatus"])
#             return msg

#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 msg = convert_msg(msg)
#                 data = models.OrderSchema()
#                 result = data.from_eqonex_to_form(msg, datatype=self.DATA_TYPE)
#                 if result:
#                     payload.append(result)
#         elif isinstance(common_response.data, dict):
#             msg = convert_msg(common_response.data)
#             payload = models.OrderSchema()
#             payload = payload.from_eqonex_to_form(msg, datatype=self.DATA_TYPE)
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
#                 result = data.from_eqonex_to_form(msg, datatype=self.DATA_TYPE)
#                 if result:
#                     payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.TradeSchema()
#             payload = payload.from_eqonex_to_form(
#                 common_response.data, datatype=self.DATA_TYPE
#             )
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def _regular_symbols_payload(
#         self, common_response: models.CommonResponseSchema, return_all: bool = False
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             # payload = list()
#             payload = dict()
#             for msg in common_response.data:
#                 data = models.SymbolSchema()
#                 # print(msg,"<--------")
#                 result = data.from_eqonex_to_form(msg, self.DATA_TYPE, return_all)
#                 if result:
#                     # payload.append(result)
#                     payload[result["symbol"]] = result
#         elif isinstance(common_response.data, dict):
#             payload = models.SymbolSchema()
#             payload = payload.from_eqonex_to_form(
#                 common_response.data, self.DATA_TYPE, return_all
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
#             key = ["starttime", "open", "high", "low", "close", "volume", "seq_num"]
#             return dict(zip(key, data))

#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 msg = key_data(msg)
#                 data = models.HistoryOHLCSchema()
#                 result = data.from_eqonex_to_form(msg, extra, datatype=self.DATA_TYPE)
#                 payload.append(result)
#             # print(payload)
#             payload = sorted(payload, key=lambda x: x["datetime"])
#         elif isinstance(common_response.data, dict):
#             common_response.data = key_data(common_response.data)
#             payload = (
#                 models.HistoryOHLCSchema()
#             )  # .model_construct(common_response.data, extra, datatype=self.DATA_TYPE)
#             payload = payload.from_eqonex_to_form(common_response.data, extra)
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
#                 payload = data.from_eqonex_to_form(msg, datatype=self.DATA_TYPE)
#                 # payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.TickerSchema()
#             payload = payload.from_eqonex_to_form(
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
#                 result = data.from_eqonex_to_form(msg, datatype=self.DATA_TYPE)
#                 payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.PositionSchema()
#             payload = payload.from_eqonex_to_form(
#                 common_response.data, datatype=self.DATA_TYPE
#             )
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def query_permission(self) -> dict:
#         """
#         requirement 9 - api key info(permission & expiry date)[restful]
#         FTX don't provide this api
#         """
#         api_info = models.PermissionSchema().dict()
#         # api_info['expired_at'] = datetime(2100,1,1)
#         api_info["permissions"] = {
#             "enable_withdrawals": True,
#             "enable_internaltransfer": True,
#             "enable_universaltransfer": True,
#             "enable_options": True,
#             "enable_reading": True,
#             "enable_futures": True,
#             "enable_margin": True,
#             "enable_spot_and_margintrading": True,
#         }
#         return models.CommonResponseSchema(
#             success=True,
#             error=False,
#             data=api_info,
#             msg="Eqonex don't suport this feature",
#         )

#     def query_account(self) -> dict:
#         """query_account
#         {
#         "positions": [
#             {
#             "instrumentId": 2,
#             "userId": "your uid",
#             "quantity": 1567000,
#             "availableQuantity": 1566900,
#             "availableTransferQuantity": 1566900,
#             "quantity_scale": 6,
#             "symbol": "ETH",
#             "assetType": "ASSET",
#             "usdCostBasis": 0,
#             "usdAvgCostBasis": 0,
#             "usdValue": 207,
#             "usdUnrealized": 0,
#             "usdRealized": 0,
#             "baseUsdMark": 132.1,
#             "settleCoinUsdMark": 0,
#             "settleCoinUnrealized": 0,
#             "settleCoinRealized": 0,
#             "tamContribution": 9999742387169,
#             "spotOrderAdjustedQuantity": 9999742387169
#             },
#             {
#             "instrumentId": 3,
#             "userId": "your uid",
#             "quantity": 0,
#             "availableQuantity": 0,
#             "availableTransferQuantity": 1566900,
#             "quantity_scale": 6,
#             "symbol": "BTC",
#             "assetType": "ASSET",
#             "usdCostBasis": 0,
#             "usdAvgCostBasis": 0,
#             "usdValue": 0,
#             "usdUnrealized": 0,
#             "usdRealized": 0,
#             "baseUsdMark": 0,
#             "settleCoinUsdMark": 0,
#             "settleCoinUnrealized": 0,
#             "settleCoinRealized": 0,
#             "tamContribution": 9999742387169,
#             "spotOrderAdjustedQuantity": 9999742387169
#             }
#             #...
#         }
#         """

#         balance = self._post("/getPositions")
#         account = defaultdict(dict)
#         # logger.debug(balance)
#         if balance.success and balance.data:
#             if not balance.data["positions"]:
#                 balance.msg = "account wallet is empty"
#                 return balance
#             # if self.DATA_TYPE == 'spot':
#             data_type = self.DATA_TYPE.upper()
#             for account_data in balance.data.get("positions", []):
#                 key = account_data["symbol"]
#                 if "[" in key and "]" in key:
#                     continue
#                 free = float(
#                     self.change_scale(
#                         account_data["availableQuantity"],
#                         account_data["quantity_scale"],
#                     )
#                 )
#                 total = float(
#                     self.change_scale(
#                         account_data["quantity"], account_data["quantity_scale"]
#                     )
#                 )
#                 if total:
#                     account[key]["symbol"] = key
#                     account[key]["available"] = free
#                     account[key]["frozen"] = total - free
#                     account[key]["balance"] = total
#                     account[key]["exchange"] = EXCHANGE
#                     account[key]["asset_type"] = data_type

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
#         **kwargs,
#     ) -> dict:
#         """
#         response: Response
#         {
#             "status":"sent",
#             "id":385617863,
#             "instrumentId":53,
#             "clOrdId":"1613037510849637345",
#             "userId":3583,
#             "price":2000,
#             "quantity":200,
#             "ordType":2
#         }
#         """
#         symbol = symbol.upper()
#         res, symbol_info = self._get_extra(symbol)
#         if not res:
#             return symbol_info

#         instrument_id = symbol_info.get("instrument_id")
#         price_scale = symbol_info.get("price_scale")
#         quantity_scale = symbol_info.get("quantity_scale")

#         order_id: str = self.new_order_id()  # generate_new_order_id
#         side: int = SIDE_MAP_INV.get(side.upper(), None)
#         if not order_type:
#             if not price:
#                 order_type = "MARKET"
#                 if stop_price:
#                     order_type = "STOP"
#             else:
#                 order_type = "LIMIT"
#                 if stop_price:
#                     order_type = "STOP_LIMIT"

#         order_type: int = ORDER_TYPE_MAP_INV.get(order_type.upper(), None)

#         quantity: int = int(self.change_scale(quantity, -quantity_scale))

#         payload: dict = {
#             "instrumentId": instrument_id,  ## TODO: get mapping
#             "symbol": symbol,
#             "side": side,  # 1 = buy, 2 = sell
#             "ordType": order_type,  # 1 = market, 2 = limit, 3 = stop market, 4 = stop limit
#             "quantity": quantity,
#             "clOrdId": order_id,
#             "quantity_scale": quantity_scale,
#             "blockWaitAck": 1,  # 1 = wait for order acknowledgement, when set, response will include the matching engine "orderId" field
#         }
#         if order_type in [2, 4]:
#             payload["timeInForce"] = 1  # GTC/FOK/IOC
#             price: int = int(self.change_scale(price, -price_scale))
#             payload["price"] = price
#             payload["price_scale"] = price_scale
#         if order_type in [3, 4]:
#             payload["stopPx"] = stop_price
#             payload["stopPx_scale"] = price_scale

#         orders_response = self._post("/order", payload)

#         if orders_response.success:
#             # print(orders_response,'!!!!')
#             if orders_response.data:
#                 result = (
#                     models.SendOrderResultSchema()
#                 )  # .model_construct(orders_response.data)
#                 orders_response.data = result.from_eqonex_to_form(
#                     orders_response.data, symbol=symbol
#                 )
#             else:
#                 orders_response.msg = "placed order is empty"

#         return orders_response

#     def cancel_order(self, order_id: str, symbol: str) -> dict:
#         """
#         # order_id: refer to custom client order id
#         order_id: refer to original order id

#         resp: {'status': 'sent', 'id': 0, 'origOrderId': 0, 'clOrdId': '22316-225623-211-00001', 'instrumentId': 25, 'userId': 31174, 'price': 0, 'quantity': 0, 'ordType': 0}

#         """
#         symbol = symbol.upper()
#         res, symbol_info = self._get_extra(symbol)
#         if not res:
#             return symbol_info
#         instrument_id = symbol_info["instrument_id"]

#         # print(symbol, instrument_id)
#         payload: dict = {
#             "instrumentId": instrument_id,
#             # "clOrdId": order_id,
#             "origOrderId": order_id,
#         }

#         return self._post("/cancelOrder", payload)

#     def cancel_orders(self, symbol: str = None) -> dict:
#         """
#         {
#             "status": "sent",
#             "id": 0,
#             "origOrderId": 0,
#             "instrumentId": 52,
#             "userId": 23750,
#             "price": 0,
#             "quantity": 0,
#             "ordType": 0
#         }
#         """
#         payload = {}
#         if symbol:
#             symbol = symbol.upper()
#             res, symbol_info = self._get_extra(symbol)
#             if not res:
#                 return symbol_info

#             instrument_id = symbol_info.get("instrument_id")
#             payload = {"instrumentId": instrument_id}

#         return self._post("/cancelAll", payload)

#     def query_order_status(self, order_id: str) -> dict:
#         """
#         {
#             "orderId": 10959164717,
#             "clOrdId": "1629964559332606823",
#             "symbol": "EQO/USDC",
#             "instrumentId": 452,
#             "side": "1",
#             "userId": 3489,
#             "account": 3489,
#             "execType": "F",
#             "ordType": "1",
#             "ordStatus": "2",
#             "timeInForce": "4",
#             "timeStamp": "20210826-07:55:59.334",
#             "execId": 89157704,
#             "targetStrategy": 0,
#             "isHidden": False,
#             "isReduceOnly": False,
#             "isLiquidation": False,
#             "fee": 0,
#             "fee_scale": 6,
#             "feeInstrumentId": 1,
#             "feeTotal": 0,
#             "price": 0,
#             "price_scale": 4,
#             "quantity": 1000000,
#             "quantity_scale": 6,
#             "leavesQty": 0,
#             "leavesQty_scale": 6,
#             "cumQty": 1000000,
#             "cumQty_scale": 6,
#             "lastPx": 6276,
#             "lastPx_scale": 4,
#             "avgPx": 6276,
#             "avgPx_scale": 4,
#             "lastQty": 1000000,
#             "lastQty_scale": 6
#         }
#         """
#         payload = {"orderId": order_id}  # ori_order_id
#         order_status = self._post("/getOrderStatus", payload)

#         if order_status.success:
#             if order_status.data:
#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"

#         return order_status

#     def query_open_orders(self, symbol: str = None) -> dict:
#         """query open order
#         {
#             "isInitialSnap": false,
#             "orders": [
#                 {
#                     "orderId": 2878793691,
#                     "orderUpdateSeq": 1,
#                     "clOrdId": "1622090693078868486",
#                     "symbol": "EQO/USDC",
#                     "instrumentId": 452,
#                     "side": "1",
#                     "userId": 5477,
#                     "account": 5477,
#                     "execType": "F",
#                     "ordType": "2",
#                     "ordStatus": "1",
#                     "timeInForce": "1",
#                     "timeStamp": "20210527-04:44:53.080",
#                     "execId": 17260550,
#                     "targetStrategy": 0,
#                     "fee": 0,
#                     "feeTotal": -7040,
#                     "fee_scale": 6,
#                     "feeInstrumentId": 1,
#                     "price": 80000,
#                     "price_scale": 4,
#                     "stopPx": 0,
#                     "stopPx_scale": 0,
#                     "quantity": 2000000,
#                     "quantity_scale": 6,
#                     "leavesQty": 0,
#                     "leavesQty_scale": 6,
#                     "cumQty": 1100000,
#                     "cumQty_scale": 6,
#                     "lastPx": 80000,
#                     "lastPx_scale": 4,
#                     "avgPx": 80000,
#                     "avgPx_scale": 4,
#                     "lastQty": 1100000,
#                     "lastQty_scale": 6,
#                     "price2": 0,
#                     "price2_scale": 0,
#                     "hidden": false,
#                     "reduceOnly": false,
#                     "liquidation": false
#                 }
#             ]
#         }

#         """

#         if symbol:
#             payload = {"symbol": symbol}
#         else:
#             payload = None
#         order_status = self._post("/getOpenOrders", payload)
#         if order_status.success:
#             if order_status.data:
#                 order_status.data = order_status.data.get("orders", [])
#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"

#         return order_status

#     def query_all_orders(self) -> dict:
#         """
#         {
#             "instrumentId": 29,
#             "account": 6285,
#             "clOrdId": "1598317184238899164",
#             "symbol": "ETH/USD[F]",
#             "side": "BUY",
#             "ordType": "MARKET",
#             "execType": "TRADE",
#             "ordStatus": "FILLED",
#             "orderId": 3459746267,
#             "secondaryOrderId": 176613,
#             "execId": 15759356,
#             "secondaryExecId": 755,
#             "targetStrategy": 0,
#             "oderQty": 1000000,
#             "orderQty_scale": 6,
#             "leavesQty": 0,
#             "leavesQty_scale": 6,
#             "cumQty": 1000000,
#             "cumQty_scale": 6,
#             "getPrice": 0,
#             "price_scale": 2,
#             "avgPx": 49000,
#             "avgPx_scale": 2,
#             "lastPx": 49000,
#             "lastPx_scale": 2,
#             "lastQty": 1000000,
#             "lastQty_scale": 6,
#             "stopPx": 0,
#             "stopPx_scale": 0,
#             "timeInForce": "1",
#             "expireTime": 0,
#             "timestampMillis": 1598317184238,
#             "expireTimeMillis": 0,
#             "aggressorSide": "BUY",
#             "price2": 0,
#             "price2Scale": 0,
#             "sourceSeqNum": 97,
#             "sourceSendTime": 0,
#             "snapId": 0,
#             "kafkaRecordOffset": 43981771,
#             "transactionId": 1598317184238344975,
#             "isLastMessageInTransaction": false,
#             "timestamp": "20200825-00:59:44.238",
#             "feePositionQuantityChange": -0.19600000000000004,
#             "settlePositionQuantityChange": 0.0
#         },
#         """
#         all_order = self._post(
#             "/getUserHistory?type=order&format=json", {"account": self.user_id}
#         )
#         if all_order.success:
#             if all_order.data:
#                 all_order.data = all_order.data.get("orderHistory", [])
#                 return self._regular_order_payload(all_order)
#             else:
#                 all_order.msg = "query order is empty"

#         return all_order

#     def query_symbols(self, symbol: str = None, return_all=False) -> dict:
#         """query_symbols
#         return_type
#             all
#             spot
#             future
#         {
#             "instrumentPairs":[
#                 {
#                     "instrumentId":512,
#                     "symbol":"XLM/USDC",
#                     "quoteId":1,
#                     "baseId":511,
#                     "price_scale":4,
#                     "quantity_scale":6,
#                     "securityStatus":1,
#                     "securityDesc":"XLM/USDC",
#                     "assetType":"PAIR",
#                     "currency":"XLM",
#                     "contAmtCurr":"USDC",
#                     "settlCurrency":"USDC",
#                     "commCurrency":"USDC",
#                     "cfiCode":"IFXXXP",
#                     "securityExchange":"EQOS",
#                     "micCode":"EQOC",
#                     "instrumentPricePrecision":4,
#                     "minPriceIncrement":1.0E-4,
#                     "minPriceIncrementAmount":1.0,
#                     "roundLot":1,
#                     "minTradeVol":0.010000,
#                     "maxTradeVol":0.000000,
#                     "qtyType":0,
#                     "contractMultiplier":1.0,
#                     "auctionStartTime":0,
#                     "auctionDuration":0,
#                     "auctionFrequency":0,
#                     "auctionPrice":0,
#                     "auctionVolume":0,
#                     "marketStatus":"OPEN"
#                 },
#                 {
#                     "instrumentId":509,
#                     "symbol":"DOT/USDC",
#                     "quoteId":1,
#                     "baseId":508,
#                     "price_scale":4,
#                     "quantity_scale":6,
#                     "securityStatus":1,
#                     "securityDesc":"DOT/USDC",
#                     "assetType":"PAIR",
#                     "currency":"DOT",
#                     "contAmtCurr":"USDC",
#                     "settlCurrency":"USDC",
#                     "commCurrency":"USDC",
#                     "cfiCode":"IFXXXP",
#                     "securityExchange":"EQOS",
#                     "micCode":"EQOC",
#                     "instrumentPricePrecision":4,
#                     "minPriceIncrement":0.01,
#                     "minPriceIncrementAmount":1.0,
#                     "roundLot":1,
#                     "minTradeVol":0.100000,
#                     "maxTradeVol":0.000000,
#                     "qtyType":0,
#                     "contractMultiplier":1.0,
#                     "auctionStartTime":0,
#                     "auctionDuration":0,
#                     "auctionFrequency":0,
#                     "auctionPrice":0,
#                     "auctionVolume":0,
#                     "marketStatus":"OPEN"
#                 },
#                 ........
#             ]
#         }
#         """

#         # if not symbol and self.symbols_info_raw:
#         #     if return_all:
#         #         return self.symbols_info_raw
#         #     ret = {}
#         #     for symbol, data in self.symbols_info_raw.data.items():
#         #         if self.DATA_TYPE == 'spot':
#         #             if '[' in symbol and ']' in symbol:
#         #                 continue
#         #         elif self.DATA_TYPE == 'future':
#         #             if not ('[' in symbol and ']' in symbol):
#         #                 continue
#         #         ret[symbol] = data
#         #     self.symbols_info_raw.data = ret
#         #     return self.symbols_info_raw
#         if self.symbols_info_raw:
#             symbols_info_raw = deepcopy(self.symbols_info_raw)
#             if return_all:
#                 return symbols_info_raw
#             if symbol:
#                 # print('???',   symbol in symbols_info_raw.data)
#                 if symbol in symbols_info_raw.data:
#                     symbols_info_raw.data = symbols_info_raw.data[symbol]
#                 else:
#                     symbols_info_raw.data = {}
#             else:
#                 ret = {}
#                 for symbol, data in symbols_info_raw.data.items():
#                     if self.DATA_TYPE == "spot":
#                         if "[" in symbol and "]" in symbol:
#                             continue
#                     elif self.DATA_TYPE == "future":
#                         if not ("[" in symbol and "]" in symbol):
#                             continue
#                     ret[symbol] = data
#                 symbols_info_raw.data = ret
#             return symbols_info_raw

#         symbols = self._get("/getInstrumentPairs", params={"verbose": "true"})

#         if symbols.success:
#             if symbols.data:
#                 symbols.data = symbols.data.get("instrumentPairs", [])
#                 if symbol:
#                     data = self._regular_symbols_payload(symbols, return_all)
#                     symbol_data = [
#                         x for x in data.data.keys() if x.lower() == symbol.lower()
#                     ]
#                     # logger.debug(symbol_data)
#                     if symbol_data:
#                         symbols.data = data.data[symbol_data[0]]
#                     else:
#                         symbols.msg = "query symbol is empty"
#                 else:
#                     return self._regular_symbols_payload(symbols, return_all)
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
#         """
#         {
#             "trades": [
#                 {
#                     "account": 23750,
#                     "commission": "-0.051214",
#                     "commCurrency": "USDC",
#                     "execId": 16761385,
#                     "ordType": "2",
#                     "ordStatus": "2",
#                     "execType": "F",
#                     "aggressorIndicator": true,
#                     "orderId": 2877532158,
#                     "price": "35619.28",
#                     "qty": "0.001000",
#                     "lastPx": "34143.19",
#                     "avgPx": "34143.19",
#                     "cumQty": "0.001000",
#                     "quoteQty": "0.001000",
#                     "side": "BUY",
#                     "symbol": "BTC/USDC",
#                     "clOrdId": "1612242132259520719",
#                     "submitterId": 23715,
#                     "targetStrategy": "0",
#                     "time": 1612242132260,
#                     "date": "20210202-05:02:12.260"
#                 }
#             ]
#         }
#             ...
#             ]
#         }
#         """
#         payload = {"limit": limit}
#         if start:
#             payload["startTime"] = int(datetime.timestamp(start) * 1000)
#         if end:
#             payload["endTime"] = int(datetime.timestamp(end) * 1000)
#         if symbol:
#             res, instrument_id = self._get_instrumentid(symbol)
#             if not res:
#                 return instrument_id
#             payload["instrumentId"] = instrument_id

#         trades = self._post("/userTrades", self._remove_none(payload))

#         if trades.success:
#             if trades.data:
#                 trades.data = trades.data.get("trades", [])
#                 return self._regular_trades_payload(trades)
#             else:
#                 trades.msg = "query trades is empty"

#         return trades

#     def query_trades1(
#         self,
#         symbol: str = None,
#         start: datetime = None,
#         end: datetime = None,
#         limit=100,
#     ) -> dict:
#         """
#         {
#             "accountId": 6285,
#             "time": 20200825-00: 59: 44.238,
#             "symbol": "ETH/USD[F]",
#             "side": "BUY",
#             "price": "0.0",
#             "qty": "1.0",
#             "fee": "-0.19600000000000004",
#             "feeAsset": "USDC",
#             "ordType": "1",
#             "ordStatus": "2",
#             "execType": "F",
#             "maker": false,
#             "orderId": 3459746267,
#             "tradeId": 15759356,
#             "quoteQty": "1.0",
#             "realizedPnl": 0.0,
#             "clOrdId": "1598317184238899164",
#             "targetStrategy": "0",
#             "timeInForce": "1",
#             "expireTimeMillis": "0",
#             "leavesQty": "0.0",
#             "cumQty": "1.0",
#             "avgPx": "490.0",
#             "lastPx": "490.0",
#             "lastQty": "1.0",
#             "stopPx": "0.0",
#             "price2": "0.0"
#         }

#         """
#         payload = {"limit": limit}
#         if start:
#             payload["startTime"] = int(datetime.timestamp(start) * 1000)
#         if end:
#             payload["endTime"] = int(datetime.timestamp(end) * 1000)
#         else:
#             payload["endTime"] = int(time.time() * 1000)

#         if symbol:
#             res, instrument_id = self._get_instrumentid(symbol)
#             if not res:
#                 return instrument_id
#             payload["instrumentId"] = instrument_id

#         trades = self._post(
#             "/getUserHistory?type=trade&format=json",
#             data={"account": self.user_id},
#             params=self._remove_none(payload),
#         )
#         # print(trades,'???')

#         if trades.success:
#             if trades.data:
#                 trades.data = trades.data.get("tradeHistory", [])
#                 return self._regular_trades_payload(trades)
#             else:
#                 trades.msg = "query trades is empty"

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
#         {
#             "pairId": 52,
#             "t": 1,
#             "s": "BTC/USDC",
#             "lastPx": 3893338,
#             "lastQty": 4000,
#             "o": 43174.80000000001,
#             "h": 43499.280000000006,
#             "l": 37698.740000000005,
#             "c": 38933.380000000005,
#             "v": 20076253.81716401,
#             "q": 500.7522000000002,
#             "chart": [
#                 [
#                     1642780380000,
#                     3893123,
#                     3893338,
#                     3893123,
#                     3893338,
#                     0,
#                     4673
#                 ],
#                 [
#                     1642780320000,
#                     3893123,
#                     3893123,
#                     3893123,
#                     3893123,
#                     0,
#                     4672
#                 ],
#                 ...
#             ]
#         }
#         """

#         symbol = symbol.upper()
#         res, symbol_info = self._get_extra(symbol)
#         if not res:
#             return symbol_info
#         instrument_id = symbol_info.get("instrument_id")
#         price_scale = symbol_info.get("price_scale")
#         # print(symbol_info, instrument_id, price_scale)

#         inp_interval = INTERVAL_MAP.get(interval)
#         if not inp_interval:
#             return models.CommonResponseSchema(
#                 success=False, error=True, data={}, msg="Invalid interval."
#             )

#         payload = {"pairId": instrument_id, "timespan": inp_interval, "limit": limit}

#         # ms - epoch time
#         if start:
#             payload["startDate"] = start.strftime("%Y-%m-%d")
#         if end:
#             payload["endDate"] = end.strftime("%Y-%m-%d")

#         historical_prices = self._get("/getChart", self._remove_none(payload))
#         # print(historical_prices,'???')
#         extra = {"symbol": symbol, "interval": interval, "price_scale": price_scale}

#         if historical_prices.success:
#             if historical_prices.data:
#                 historical_prices.data = historical_prices.data.get("chart", [])
#                 return self._regular_historical_prices_payload(historical_prices, extra)
#             else:
#                 historical_prices.msg = "query historical ohlc is empty"
#                 historical_prices.data = []

#         return historical_prices

#     def query_last_price(
#         self, symbol: str, interval: str = "1m", limit: int = 1
#     ) -> dict:
#         symbol = symbol.upper()
#         res, symbol_info = self._get_extra(symbol)
#         if not res:
#             return symbol_info

#         instrument_id = symbol_info.get("instrument_id")
#         price_scale = symbol_info.get("price_scale")
#         # print(symbol_info, instrument_id, price_scale)

#         inp_interval = INTERVAL_MAP.get(interval)
#         if not inp_interval:
#             return models.CommonResponseSchema(
#                 success=False, error=True, data={}, msg="Invalid interval."
#             )

#         payload = {"pairId": instrument_id, "timespan": inp_interval, "limit": limit}

#         historical_prices = self._get("/getChart", self._remove_none(payload))
#         extra = {"symbol": symbol, "interval": interval, "price_scale": price_scale}

#         if historical_prices.success:
#             if historical_prices.data:
#                 historical_prices.data = historical_prices.data.get("chart", [])
#                 last_data = self._regular_historical_prices_payload(
#                     historical_prices, extra
#                 )
#                 if isinstance(last_data.data, list) and last_data:
#                     last_data.data = last_data.data[0]

#                 return last_data
#             else:
#                 historical_prices.msg = "query historical ohlc is empty"
#                 historical_prices.data = []

#         return historical_prices

#     def query_prices(self, symbol: str) -> dict:
#         """ """
#         symbol = symbol.upper()
#         symbol_info = self.symbols_info.get(symbol, {})
#         if not symbol_info:
#             return models.CommonResponseSchema(
#                 success=False, error=True, data={}, msg="Invalid symbol."
#             )

#         instrument_id = symbol_info.get("extra", {}).get("instrument_id")
#         # price_scale = symbol_info.get("extra", {}).get("price_scale")
#         # print(symbol_info, instrument_id, price_scale)

#         inp_interval = INTERVAL_MAP["1m"]

#         payload = {"pairId": instrument_id, "timespan": inp_interval, "limit": 1}

#         spots = self._get("/getChart", self._remove_none(payload))
#         # extra = {'symbol':symbol,'interval': inp_interval, 'price_scale': price_scale}

#         logger.debug(spots)
#         if spots.success:
#             if spots.data:
#                 return self._regular_symbol_payload(spots)
#             else:
#                 spots.msg = "query spots is empty"

#         return spots

#     def query_position(self, force_return: bool = False) -> dict:
#         """query_position
#         account:
#         {
#             "userId": 7966,
#             "usdValue": 7398245.37,
#             "usdPositionValue": 5863434.96,
#             "usdOpenOrdersValue": 0.0,
#             "usdOpenOrdersRequiredValue": 0.0,
#             "usdMarginValue": 0.0,
#             "usdMarginRequiredValue": 5863434.965279438,
#             "usdMarginMaintValue": 2931717.48,
#             "leverageRatio": 0.79,
#             "usdUnrealized": 40623.17,
#             "riskUpdateDate": "20210813-03:58:58.161",
#             "usdMarginableValue": 7335923.78,
#             "usdOpenOrdersNotional": 0.0,
#             "usdMarginReservedBuys": 0.0,
#             "usdMarginReservedSells": 0.0,
#             "usdMarginAvailable": 1472488.82,
#             "usdPositionsValue": 5863434.96,
#             "usdMarginReserve": 0.0,
#             "usdMarginInitial": 5863434.96,
#             "usdMarginTrigger": 2931717.48,
#             "usdMarginAccountTotal": 7335923.78,
#             "accountLeverage": 0.79
#         }

#         GET RISK:
#         {
#         "userId":31174,
#         "usdValue":855.07,
#         "usdPositionValue":23.03,
#         "usdOpenOrdersValue":0.0,
#         "usdOpenOrdersRequiredValue":0.0,
#         "usdMarginValue":0.0,
#         "usdMarginRequiredValue":0.18429926400000002,
#         "usdMarginMaintValue":0.09,
#         "leverageRatio":0.29,
#         "usdUnrealized":1.54,
#         "riskUpdateDate":"20220427-06:13:37.598",
#         "usdMarginableValue":78.44,
#         "usdOpenOrdersNotional":0.0,
#         "usdMarginReservedBuys":0.0,
#         "usdMarginReservedSells":0.0,
#         "usdMarginAvailable":78.26,
#         "usdPositionsValue":23.03,
#         "usdMarginReserve":0.0,
#         "usdMarginInitial":0.18,
#         "usdMarginTrigger":0.09,
#         "usdMarginAccountTotal":78.44,
#         "accountLeverage":0.29
#         }
#         --------
#         {
#         "userId":31174,
#         "usdValue":855.63,
#         "usdPositionValue":35.15,
#         "usdOpenOrdersValue":0.08,
#         "usdOpenOrdersRequiredValue":0.08000000000000002,
#         "usdMarginValue":0.0,
#         "usdMarginRequiredValue":0.36524181600000005,
#         "usdMarginMaintValue":0.14,
#         "leverageRatio":0.11,
#         "usdUnrealized":1.34,
#         "riskUpdateDate":"20220427-07:04:04.294",
#         "usdMarginableValue":307.42,
#         "usdOpenOrdersNotional":0.08,
#         "usdMarginReservedBuys":0.08,
#         "usdMarginReservedSells":0.0,
#         "usdMarginAvailable":307.05,
#         "usdPositionsValue":35.15,
#         "usdMarginReserve":0.08000000000000002,
#         "usdMarginInitial":0.37,
#         "usdMarginTrigger":0.14,
#         "usdMarginAccountTotal":307.42,
#         "accountLeverage":0.11
#         }
#         return {
#             "asset_type": self.asset_type,
#             "symbol": self.symbol,
#             "side": self.side,
#             "size": self.size,
#             "position_value": self.position_value,
#             "entry_price": self.entry_price,
#             "leverage": self.leverage,
#             "position_margin": self.position_margin,
#             "initial_margin": self.initial_margin,
#             "maintenance_margin": self.maintenance_margin,

#             "realised_pnl": self.realised_pnl,
#             "unrealised_pnl": self.unrealised_pnl,
#             "is_isolated": self.is_isolated,
#             "auto_add_margin": self.auto_add_margin,
#             "liq_price": self.liq_price,
#             "bust_price": self.bust_price,

#         }
#         """

#         def combine_pos_risk(data, risk):
#             ret = []
#             for pos in data.get("positions", []):
#                 if "[" in pos["symbol"] and "]" in pos["symbol"]:
#                     pos.update(risk)
#                     ret.append(pos)
#             return ret

#         position = self._post("/getPositions")
#         risk = self._post("/getRisk")
#         # print(position,'!!1!!!')
#         # print(risk,'!!!2!!')

#         if position.success:
#             if position.data:
#                 position.data = combine_pos_risk(position.data, risk.data)
#                 # print(position.data,'!!!!!')
#                 return self._regular_position_payload(position)
#             else:
#                 position.msg = "query future position is empty"

#         return position


# class EqonexDataWebsocket(WebsocketClient):
#     def __init__(
#         self, proxy_host: str = "", proxy_port: int = 0, is_testnet: bool = False
#     ) -> None:
#         super().__init__()

#         self.eqo = EqonexClient()
#         self.symbols_info = self.eqo.symbols_info
#         self.symbols_info_by_id = self.eqo.symbols_info_by_id

#         self.ticks: dict[str, dict] = {}
#         self.reqid: int = 0

#         # self.interval = '1m'
#         # self.channel = 'kline'

#         # def connect(self, proxy_host: str='', proxy_port: int=0):
#         #     """connect ws channel"""
#         self.init(
#             host=EQONEX_WEBSOCKET_HOST, proxy_host=proxy_host, proxy_port=proxy_port
#         )
#         self.start()

#     def on_connected(self) -> None:
#         """"""
#         print("Eqonex market data Websocket connect success")
#         # resubscribe
#         # print('!!!!', self.ticks,'!!!')
#         if self.ticks:
#             # # for symbol in self.ticks.keys():
#             # req: dict = {
#             #     "symbols":list(self.ticks.keys()),
#             #     # "symbols":[symbol],
#             #     "event":"S",
#             #     "types":[4],   # type 4 == kline, 1 == orderbook
#             #     "requestId": self.reqid,
#             #     "timespan": 1,
#             #     # "level": 2, # forr type=1
#             # }

#             # req:dict = self.sub_stream(list(self.ticks.keys()), channel=channel, )
#             # # print(self.reqid,'<==resubscribe====', req)
#             # self.send_packet(req)

#             for key, detail in self.ticks.items():
#                 req: dict = self.sub_stream(
#                     symbols=[detail["symbol"]],
#                     channel=detail["channel"],
#                     interval=detail["interval"],
#                 )
#                 # logger.debug(req)
#                 self.send_packet(req)

#     def sub_stream(
#         self, symbols: list, channel: str = "depth", interval: str = "1m"
#     ) -> dict:
#         """
#         "depth": 1,
#         "kline": 4,
#         """
#         self.reqid += 1
#         req: dict = {
#             "symbols": symbols,
#             # "symbols":list(self.ticks.keys()),
#             # "symbols":[symbol],
#             "event": "S",
#             # "types":[4],   # type 4 == kline, 1 == orderbook
#             "requestId": str(self.reqid),
#             # "timespan": INTERVAL_MAP[interval],
#             # "level": 2, # forr type=1
#         }

#         if channel == "kline":
#             req["types"] = [4]
#             req["timespan"] = INTERVAL_MAP[interval]
#         elif channel == "depth":
#             req["types"] = [1]
#             req["level"] = 2
#         elif channel == "ticker":
#             req["types"] = [4]
#             # req['timespan'] = 6 # 1d
#             req["timespan"] = 1  # 1m

#         return req

#     def subscribe(
#         self, symbols, on_tick=None, channel="depth", interval="1m"
#     ) -> None:  # , interval='1m'
#         """"""
#         if on_tick:
#             self.on_tick = on_tick

#         if channel not in ["depth", "kline", "ticker", "trades", "orders", "account"]:
#             Exception("invalid subscription")
#             return

#         # self.channel = channel
#         # self.interval = interval

#         if isinstance(symbols, str):
#             symbols = [symbols]
#         elif isinstance(symbols, list):
#             pass
#         else:
#             print("invalid input symbols.")
#             return

#         u_symbols = []
#         for symbol in symbols:
#             symbol = symbol.upper()
#             u_symbols.append(symbol)
#             # if symbol in self.ticks:
#             #     return

#             # tick data dict
#             tick = {
#                 "symbol": symbol,
#                 "exchange": "EQONEX",
#                 "channel": channel,
#                 "interval": interval,
#                 "asset_type": "SPOT",
#                 "name": symbol,
#             }
#             key = ""
#             if channel == "kline":
#                 key = f"{symbol}|{interval}"
#             else:
#                 key = f"{symbol}|{channel}"

#             self.ticks[key] = tick

#         req: dict = self.sub_stream(u_symbols, channel=channel, interval=interval)

#         # print(self.reqid,'<======', req)
#         self.send_packet(req)

#     def on_packet(self, packet: dict) -> None:
#         """on_packet"""
#         # print('------on_packet-----',packet, '--------------ori--------------')

#         response: str = packet.get(
#             "response", None
#         )  # {'requestId': '1', 'sequenceNumber': 10000000201478, 'response': 'Subscribed Successfully', 'event': 'S', 'types': [4], 'symbol': ['BTC/USDC'], 'level': 0, 'timespan': 1, 'errorMessage': None}
#         if response:
#             print(packet)
#             return

#         channel: int = packet["type"]
#         pair_id: int = packet["pairId"]

#         symbol_info = self.symbols_info_by_id.get(pair_id, {})
#         # symbol_info = self.symbols_info.get(symbol, {})
#         symbol = symbol_info["symbol"]
#         price_scale = symbol_info.get("extra", {}).get("price_scale")
#         quantity_scale = symbol_info.get("extra", {}).get("quantity_scale")

#         if channel == 4:
#             interval: int = INTERVAL_MAP_REV[packet["t"]]
#             has_ticker_channel = f"{symbol}|ticker" in self.ticks
#             has_kline_channel = f"{symbol}|{interval}" in self.ticks

#         if channel == 4:
#             """
#             {
#                 "type": 4,
#                 "pairId": 53,
#                 "lastPx": 210100,
#                 "lastQty": 10000,
#                 "chart": [
#                     [
#                         1621468800000,
#                         200242,
#                         220736,
#                         184358,
#                         210100,
#                         1790488,
#                         121
#                     ],
#                     [
#                         1620864000000,
#                         184406,
#                         262862,
#                         184358,
#                         203790,
#                         7970089,
#                         120
#                     ],
#                     ...
#                 ],
#                 "s": "ETH/USDC",
#                 "t": 7,
#                 "o": 2122.7300000000005,
#                 "h": 2207.3600000000006,
#                 "l": 1959.0300000000004,
#                 "c": 2101.0000000000005,
#                 "v": 57614216.86550003,
#                 "q": 28245.33000000001
#             }

#             """

#             # print(f"========>symbol: {symbol}; channel: {channel}; tick_channel: {tick_channel}; chart: {packet['chart']}") # data: {data}
#             if has_ticker_channel and packet["t"] == 1:
#                 data: dict = packet
#                 tick: dict = self.ticks[f"{symbol}|ticker"]

#                 tick["prev_open_24h"] = float(data["o"])
#                 tick["prev_high_24h"] = float(data["h"])
#                 tick["prev_low_24h"] = float(data["l"])
#                 tick["prev_volume_24h"] = float(data["q"])
#                 tick["prev_turnover_24h"] = float(data["v"])
#                 # tick['last_price'] = float(self.eqo.change_scale(data['lastPx'], price_scale))
#                 tick["last_price"] = float(data["c"])
#                 tick["prev_close_24h"] = tick[
#                     "prev_open_24h"
#                 ]  # FIXME: no other choices
#                 tick["price_change"] = tick["last_price"] - tick["prev_close_24h"]
#                 # print(tick['last_price'], tick['prev_close_24h'], tick['price_change'],'<----')
#                 tick["price_change_pct"] = (
#                     (tick["price_change"] / tick["prev_close_24h"])
#                     if tick["prev_close_24h"]
#                     else 0.0
#                 )

#                 tick["datetime"] = datetime.now()
#                 self.on_tick(copy(tick))

#             if has_kline_channel:
#                 data: dict = packet["chart"][0]
#                 tick: dict = self.ticks[f"{symbol}|{interval}"]

#                 tick["volume"] = float(data[5])
#                 tick["open"] = float(
#                     self.eqo.change_scale(data[1], price_scale)
#                 )  # float(data[1])
#                 tick["high"] = float(
#                     self.eqo.change_scale(data[2], price_scale)
#                 )  # float(data[2])
#                 tick["low"] = float(
#                     self.eqo.change_scale(data[3], price_scale)
#                 )  # float(data[3])
#                 tick["close"] = float(
#                     self.eqo.change_scale(data[4], price_scale)
#                 )  # float(data[4])
#                 tick["turnover"] = tick["close"] * tick["volume"]
#                 tick["start"] = self.eqo.generate_datetime_ts(float(data[0]))
#                 tick["datetime"] = datetime.now()
#             else:
#                 return
#         else:
#             """
#             {
#                 "type": 1,
#                 "pairId": 53,
#                 "bids": [
#                     [
#                         207000,
#                         970000,
#                         1622027405860
#                     ],
#                     [
#                         205000,
#                         1000000,
#                         1622027405860
#                     ],
#                     ...
#                 ],
#                 "asks": [
#                     [
#                         210100,
#                         990000,
#                         1622027405860
#                     ],
#                     [
#                         210225,
#                         246860000,
#                         1622027405860
#                     ],
#                     ...
#                 ],
#                 "sequence": 0,
#                 "usdMark": 2101.0,
#                 "marketStatus": 0,
#                 "estFundingRate": 0.0,
#                 "fundingRateTime": 0,
#                 "auctionPrice": 0.0,
#                 "auctionVolume": 0.0
#             }
#             """
#             tick: dict = self.ticks[f"{symbol}|depth"]
#             # print('------on_packet-----',packet, '--------------ori--------------', tick)

#             bids: list = packet["bids"]
#             for n in range(min(5, len(bids))):
#                 price, volume, ts = bids[n]
#                 # tick.__setattr__("bid_price_" + str(n + 1), float(price))
#                 # tick.__setattr__("bid_volume_" + str(n + 1), float(volume))
#                 # tss = self.eqo.generate_datetime_ts(ts)
#                 tick["bid_price_" + str(n + 1)] = float(
#                     self.eqo.change_scale(price, price_scale)
#                 )  # float(price)
#                 tick["bid_volume_" + str(n + 1)] = float(
#                     self.eqo.change_scale(volume, quantity_scale)
#                 )  # float(volume)

#             asks: list = packet["asks"]
#             for n in range(min(5, len(asks))):
#                 price, volume, ts = asks[n]
#                 # print(f"In ASK - {n}: pricei: {price}; volume:{volume}; ts: {ts}")
#                 tick["ask_price_" + str(n + 1)] = float(
#                     self.eqo.change_scale(price, price_scale)
#                 )  # float(price)
#                 tick["ask_volume_" + str(n + 1)] = float(
#                     self.eqo.change_scale(volume, quantity_scale)
#                 )  # float(volume)
#             tick["datetime"] = datetime.now()
#             # tick['datetime'] = tss

#         self.on_tick(copy(tick))


# class EqonexTradeWebsocket(WebsocketClient):
#     """ """

#     def __init__(
#         self,
#         on_account_callback=None,
#         on_order_callback=None,
#         on_trade_callback=None,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         is_testnet: bool = False,
#     ) -> None:
#         super().__init__()
#         self.on_account_callback = on_account_callback
#         self.on_order_callback = on_order_callback
#         self.on_trade_callback = on_trade_callback
#         self.init(
#             host=EQONEX_WEBSOCKET_HOST,
#             proxy_host=proxy_host,
#             proxy_port=proxy_port,
#             ping_interval=30,
#         )
#         self.start()
#         self._logged_in = False
#         self.key = None
#         self.secret = None
#         self.channel = "orders"

#     def info(self):
#         return "Eqonex Trade Websocket Start"

#     def connect(self, key: str, secret: str):
#         """
#         Give api key and secret for Authorization.
#         """
#         self.key = key
#         self.secret = secret
#         if not self._logged_in:
#             self._login()
#         logger.info(self.info())

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
#         self.send_packet(req)

#     def on_connected(self) -> None:
#         if not self._logged_in:
#             self._login()
#         logger.debug("Spot user trade data Websocket connect success")

#     def inject_callback(self, callbackfn, channel="orders"):
#         if channel == "account":
#             self.on_account_callback = callbackfn
#         elif channel == "orders":
#             self.on_order_callback = callbackfn
#         elif channel == "trades":
#             self.on_trade_callback = callbackfn
#         else:
#             Exception("invalid callback function")

#     def subscribe(self, channel="orders") -> None:
#         """
#         keep the interface:
#         channel: 'trades', 'orders', 'account'
#         mapping: 'ticketInfo', 'executionReport', 'outboundAccountInfo'
#         """
#         if channel not in ["depth", "kline", "trades", "orders", "account"]:
#             Exception("invalid subscription")
#             return

#         if not self._logged_in:
#             self._login()

#         self.channel = channel
#         # Sending the authentication message automatically subscribes you to all 3 private topics.

#     def on_packet(self, packet: dict) -> None:
#         logger.debug(f"on_packet event {packet}")
#         if "auth" in packet:
#             self._logged_in = True if packet["auth"] == "success" else False
#             return

#         if "ping" in packet:
#             logger.info(f"healthcheck message {packet}")
#             return

#         if not packet:
#             logger.debug(f"unknown packet event {packet}")
#             return

#         if packet[0]["e"] == "outboundAccountInfo":
#             self.on_account(packet[0])
#         elif packet[0]["e"] == "executionReport":
#             self.on_order(packet[0])
#         elif packet[0]["e"] == "ticketInfo":
#             self.on_trade(packet[0])
#         else:
#             logger.info(f"the other packet type: {self.on_trade(packet[0])}")

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

#         for d in packet["B"]:
#             account = {
#                 "symbol": d["a"],
#                 "asset_type": ASSET_TYPE_SPOT,
#                 "name": models.normalize_name(d["a"], ASSET_TYPE_SPOT),
#                 "balance": float(d["f"]) + float(d["l"]),
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

#         order = {
#             "symbol": packet["s"].lower(),
#             "asset_type": ASSET_TYPE_SPOT,
#             "name": models.normalize_name(packet["s"], ASSET_TYPE_SPOT),
#             "exchange": EXCHANGE,
#             "order_id": packet["C"],
#             "type": packet["o"],  # LIMIT, MARKET
#             "side": packet["S"],  # BUY, SELL
#             "price": float(packet["p"]),
#             "quantity": float(packet["q"]),
#             "traded": float(packet["z"]),
#             "status": packet[
#                 "X"
#             ],  # NEW, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED, PENDING_CANCEL, PENDING_NEW
#             "datetime": EqonexClient.generate_datetime(float(packet["O"]) / 1000),
#         }

#         if self.on_order_callback:
#             self.on_order_callback(order)

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
#         side = "Buy" if packet["m"] else "Sell"
#         trade = {
#             "symbol": packet["s"],
#             "asset_type": ASSET_TYPE_SPOT,
#             "name": models.normalize_name(packet["s"], ASSET_TYPE_SPOT),
#             "exchange": EXCHANGE,
#             "order_id": packet["o"],
#             "trade_id": packet["T"],
#             "side": side,
#             "price": float(packet["p"]),
#             "quantity": float(packet["q"]),
#             "datetime": EqonexClient.generate_datetime(float(packet["E"]) / 1000),
#         }

#         if self.on_trade_callback:
#             self.on_trade_callback(trade)
