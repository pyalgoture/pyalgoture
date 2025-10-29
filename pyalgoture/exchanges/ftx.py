# import hmac
# import logging
# import time
# import urllib
# from collections import defaultdict
# from copy import copy
# from datetime import datetime

# # from threading import Lock
# from typing import Any, Optional, dict

# # from loguru import logger
# from requests import Request  # , Session

# from ..utils import models
# from ..utils.client_rest import Response, RestClient
# from ..utils.client_ws import WebsocketClient
# from ..utils.models import UTC_TZ

# logger = logging.getLogger("ftx")

# ACCOUNT_API = "/api/account"
# SUBACCOUNT_API = "/api/subaccounts"
# LOGIN_API = "/api/login_status"

# # SPOT / FUTURE / OPTION in your account
# BALANCE_API = "/api/wallet/balances"
# POSITION_API = "/api/positions"
# OPTION_API = "/api/options/account_info"

# # orders
# ORDERS_API = "/api/orders"
# ORDERS_CONDITION_API = "/api/conditional_orders"
# ORDERS_HISTORY_API = "/api/orders/history"
# TRADE_API = "/api/fills"

# # SPOT / FUTURE / OPTION infomation in your exchange
# MARKET_API = "/api/markets"
# FUTURE_API = "/api/futures"

# # funding
# FUNDING_PAYMENTS_API = "/api/funding_payments"

# EXCHANGE = "FTX"

# FTX_API_URL = "https://ftx.com"
# FTX_WS_URL = "wss://ftx.com/ws/"

# TIMEDELTA: dict[str, int] = {
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

# MAPPING_CHANNEL: dict[str, str] = {
#     "depth": "orderbook",
#     "kline": "ticker",
#     "trades": "fills",
#     "orders": "orders",
#     "account": "account",
# }  # don't support


# class FTXClient(RestClient):
#     """
#     api docs: https://docs.ftx.com/#rest-api
#     """

#     BROKER_ID = ""

#     def __init__(
#         self,
#         url_base: str = FTX_API_URL,
#         proxy_host: str = "",
#         proxy_port: int = 0,
#         subaccount: str = None,
#         is_testnet: bool = False,
#     ) -> None:
#         super().__init__(url_base, proxy_host, proxy_port)

#         self.order_count: int = 0
#         # self.order_count_lock: Lock = Lock()
#         self.connect_time: int = 0
#         self.key: str = ""
#         self.secret: str = ""
#         self.subaccount_name: str = subaccount if subaccount else ""

#     @staticmethod
#     def generate_datetime(timestamp: float) -> datetime:
#         """generate_datetime"""
#         dt: datetime = datetime.fromtimestamp(timestamp)
#         dt: datetime = UTC_TZ.localize(dt)
#         return dt

#     def _remove_none(self, payload):
#         return {k: v for k, v in payload.items() if v is not None}

#     def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
#         response = self.request(method="GET", path=path, params=params)
#         return self._process_response(response)

#     def _post(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
#         response = self.request(method="POST", path=path, json=params)
#         return self._process_response(response)

#     def _delete(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
#         response = self.request(method="DELETE", path=path, json=params)
#         return self._process_response(response)

#     def _regular_position_payload(
#         self, common_response: models.CommonResponseSchema
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 data = models.PositionSchema()
#                 data.from_ftx_to_form(msg)
#                 payload.append(data.dict())
#         elif isinstance(common_response.data, dict):
#             payload = models.PositionSchema()
#             payload.from_ftx_to_form(common_response.data)
#             payload = payload.dict()
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def new_order_id(self) -> str:
#         """new_order_id"""
#         prefix: str = datetime.now().strftime("%Y%m%d-%H%M%S-")
#         # with self.order_count_lock:
#         self.order_count += 1
#         suffix: str = str(self.order_count).rjust(5, "0")

#         order_id: str = f"x-{self.BROKER_ID}" + prefix + suffix
#         return order_id

#     def connect(self, key: str, secret: str) -> None:
#         """connect exchange server"""
#         self.key = key
#         self.secret = secret.encode()
#         self.connect_time = int(datetime.now(UTC_TZ).strftime("%y%m%d%H%M%S"))
#         self.start()
#         logger.debug("FTX REST API start ")

#     def _request_prepare(self, req):
#         request = Request(
#             method=req.method,
#             url=f"{FTX_API_URL}{req.path}",
#             headers=req.headers,
#             data=req.data,
#             params=req.params,
#             json=req.json,
#         )
#         prepared = request.prepare()
#         return prepared

#     def _process_response(self, response: Response) -> dict:
#         try:
#             data = response.data
#         except ValueError:
#             logger.debug(response.data())
#             raise
#         else:
#             # logger.debug(data)
#             if not data["success"] or not response.ok:
#                 payload_data = models.CommonDataSchema(
#                     status_code=response.status_code, msg=data["error"]
#                 )
#                 return models.CommonResponseSchema(
#                     success=False,
#                     error=True,
#                     data=payload_data.dict(),
#                     msg=data["error"],
#                 )
#             elif response.ok and data["success"] and data["result"] is None:
#                 return models.CommonResponseSchema(
#                     success=True,
#                     error=False,
#                     data={"success": True},
#                     msg="query ok and server return http code only",
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
#                 data = models.OrderSchema()
#                 result = data.from_ftx_to_form(msg)
#                 payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.OrderSchema()
#             payload = payload.from_ftx_to_form(common_response.data)
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def _regular_symbols_payload(
#         self, common_response: models.CommonResponseSchema
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 data = models.SymbolSchema()
#                 result = data.from_ftx_to_form(msg)
#                 payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.SymbolSchema()
#             payload = payload.from_ftx_to_form(common_response.data)
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
#                 result = data.from_ftx_to_form(msg)
#                 payload.append(result)
#         elif isinstance(common_response.data, dict):
#             payload = models.TradeSchema()
#             payload = payload.from_ftx_to_form(common_response.data)
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
#                 data.from_ftx_to_form(msg, extra)
#                 payload.append(data.dict())
#         elif isinstance(common_response.data, dict):
#             payload = models.HistoryOHLCSchema()
#             payload.from_ftx_to_form(common_response.data, extra)
#             payload = payload.dict()
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
#                 data.from_ftx_to_form(msg)
#                 payload.append(data.dict())
#         elif isinstance(common_response.data, dict):
#             payload = models.TickerSchema()
#             payload.from_ftx_to_form(common_response.data)
#             payload = payload.dict()
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def _regular_incomes_payload(
#         self, common_response: models.CommonResponseSchema
#     ) -> dict:
#         if isinstance(common_response.data, list):
#             payload = list()
#             for msg in common_response.data:
#                 data = models.IncomeSchema()
#                 data.from_ftx_to_form(msg)
#                 payload.append(data.dict())
#         elif isinstance(common_response.data, dict):
#             payload = models.IncomeSchema()
#             payload.from_ftx_to_form(common_response.data)
#             payload = payload.dict()
#         else:
#             common_response.msg = "payload error in order"
#             Exception("payload error in order")

#         common_response.data = payload
#         return common_response

#     def login(self) -> dict:
#         return self._get(LOGIN_API)

#     def sign(self, request):
#         """generate ftx signature"""
#         prepared = self._request_prepare(request)
#         ts = int(time.time() * 1000)
#         signature_payload = f"{ts}{prepared.method}{prepared.path_url}".encode()
#         if prepared.body:
#             signature_payload += prepared.body

#         signature = hmac.new(self.secret, signature_payload, "sha256").hexdigest()
#         request.headers = prepared.headers
#         request.headers["FTX-KEY"] = self.key
#         request.headers["FTX-SIGN"] = signature
#         request.headers["FTX-TS"] = str(ts)

#         if self.subaccount_name:
#             request.headers["FTX-SUBACCOUNT"] = urllib.parse.quote(self.subaccount_name)

#         return request

#     def query_permission(self) -> dict:
#         """
#         requirement 9 - api key info(permission & expiry date)[restful]
#         FTX don't provide this api
#         """
#         api_info = models.PermissionSchema()
#         return models.CommonResponseSchema(
#             success=True,
#             error=False,
#             data=api_info.dict(),
#             msg="FTX don't suport this feature",
#         )

#     def query_account(self) -> dict:
#         """
#         Request
#         GET /wallet/balances
#         Response
#         {
#             "success": true,
#             "result": [
#                 {
#                     "coin": "USDTBEAR",
#                     "free": 2320.2,
#                     "spotBorrow": 0.0,
#                     "total": 2340.2,
#                     "usdValue": 2340.2,
#                     "availableWithoutBorrow": 2320.2
#                 }
#             ]
#         }
#         """
#         balance = self._get(BALANCE_API)
#         account = defaultdict(dict)

#         if balance.success and balance.data:
#             for account_data in balance.data:
#                 free = float(account_data["free"])
#                 locked = float(account_data["spotBorrow"])
#                 if free != 0 or locked != 0:
#                     key = account_data["coin"]
#                     account[key]["available"] = free
#                     account[key]["frozen"] = locked
#                     account[key]["balance"] = free + locked
#                     account[key]["exchange"] = EXCHANGE
#                     account[key]["asset_type"] = "SPOT"

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
#         requirement 3 - Send order[restful]

#         Request
#         POST /orders
#         {
#             "market": "XRP-PERP",
#             "side": "sell",
#             "price": 0.306525,
#             "type": "limit",
#             "size": 31431.0,
#             "reduceOnly": false,
#             "ioc": false,
#             "postOnly": false,
#             "clientId": null
#             }

#         Response
#         {
#             "success": true,
#             "result": {
#                 "createdAt": "2019-03-05T09:56:55.728933+00:00",
#                 "filledSize": 0,
#                 "future": "XRP-PERP",
#                 "id": 9596912,
#                 "market": "XRP-PERP",
#                 "price": 0.306525,
#                 "remainingSize": 31431,
#                 "side": "sell",
#                 "size": 31431,
#                 "status": "open",
#                 "type": "limit",
#                 "reduceOnly": false,
#                 "ioc": false,
#                 "postOnly": false,
#                 "clientId": null}
#         }
#         """
#         # assert order_type in ('limit', 'stop', 'trailing_stop', 'trailing_stop')
#         # assert side in ('buy', 'sell')
#         api_endpoint = ORDERS_API

#         payload = {
#             "market": symbol,
#             "side": side.lower(),
#             "price": price,
#             "size": quantity,
#             "type": order_type,
#             "clientId": self.new_order_id(),
#         }
#         if order_type is None:
#             if price is None:
#                 order_type == "market"
#             else:
#                 order_type == "limit"

#             payload["type"] = order_type

#         if order_type == "market":
#             pass
#         elif order_type == "limit":
#             pass
#         elif order_type in ["stop", "takeProfit"]:
#             api_endpoint = ORDERS_CONDITION_API
#             payload["triggerPrice"] = str(stop_price)
#         elif order_type in ["trailingStop"]:
#             api_endpoint = ORDERS_CONDITION_API
#             payload["trailValue"] = str(stop_price)

#         orders_response = self._post(api_endpoint, payload)

#         if orders_response.success:
#             if orders_response.data:
#                 result = (
#                     models.SendOrderResultSchema()
#                 )  # .model_construct(orders_response.data)
#                 orders_response.data = result.from_ftx_to_form(orders_response.data)
#             else:
#                 orders_response.msg = "query sender order is empty"

#         return orders_response

#     def cancel_order(self, order_id: str) -> dict:
#         """
#         requirement 4 - Cancel order[restful]

#         Request
#         DELETE /orders/{order_id}

#         Response
#         {
#             "success": true,
#             "result": "Order queued for cancellation"
#         }
#         """
#         return self._delete(f"{ORDERS_API}/{order_id}")

#     def cancel_orders(self, symbol: str = None) -> dict:
#         """
#         requirement 4 - Cancel order[restful]

#         Request
#         DELETE /orders
#         {
#             "market": "BTC-PERP"}

#         Response
#         {
#             "success": true,
#             "result": "Orders queued for cancelation"
#         }
#         """
#         payload = {"market": symbol}
#         return self._delete(ORDERS_API, payload)

#     def query_order_status(self, order_id: str = None) -> dict:
#         """
#         Request

#         GET /orders/{order_id}
#         Response

#         {
#             "success": true,
#             "result": {
#                 "createdAt": "2019-03-05T09:56:55.728933+00:00",
#                 "filledSize": 10,
#                 "future": "XRP-PERP",
#                 "id": 9596912,
#                 "market": "XRP-PERP",
#                 "price": 0.306525,
#                 "avgFillPrice": 0.306526,
#                 "remainingSize": 31421,
#                 "side": "sell",
#                 "size": 31431,
#                 "status": "open",
#                 "type": "limit",
#                 "reduceOnly": false,
#                 "ioc": false,
#                 "postOnly": false,
#                 "clientId": null,
#                 "liquidation": False
#             }
#         }
#         """
#         order_status = self._get(f"ORDERS_API/{order_id}")
#         if order_status.success:
#             if order_status.data:
#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"

#         return order_status

#     def query_open_orders(self, symbol: str = None) -> dict:
#         """
#         Request

#         GET /orders?market={market}
#         Response

#         {
#             "success": true,
#             "result": [
#                 {
#                     "createdAt": "2019-03-05T09:56:55.728933+00:00",
#                     "filledSize": 10,
#                     "future": "XRP-PERP",
#                     "id": 9596912,
#                     "market": "XRP-PERP",
#                     "price": 0.306525,
#                     "avgFillPrice": 0.306526,
#                     "remainingSize": 31421,
#                     "side": "sell",
#                     "size": 31431,
#                     "status": "open",
#                     "type": "limit",
#                     "reduceOnly": false,
#                     "ioc": false,
#                     "postOnly": false,
#                     "clientId": null
#                 }
#             ]
#         }
#         """
#         if symbol:
#             payload = {"market": symbol}
#         else:
#             payload = None
#         order_status = self._get(ORDERS_API, payload)
#         if order_status.success:
#             if order_status.data:
#                 return self._regular_order_payload(order_status)
#             else:
#                 order_status.msg = "query order is empty"

#         return order_status

#     def query_all_orders(self, symbol: str = None) -> dict:
#         """
#         Request
#         GET /orders/history?market={market}

#         Response
#         {
#             "success": true,
#             "result": [
#                 {
#                     "avgFillPrice": 10135.25,
#                     "clientId": null,
#                     "createdAt": "2019-06-27T15:24:03.101197+00:00",
#                     "filledSize": 0.001,
#                     "future": "BTC-PERP",
#                     "id": 257132591,
#                     "ioc": false,
#                     "market": "BTC-PERP",
#                     "postOnly": false,
#                     "price": 10135.25,
#                     "reduceOnly": false,
#                     "remainingSize": 0.0,
#                     "side": "buy",
#                     "size": 0.001,
#                     "status": "closed",
#                     "type": "limit"
#                 }],
#             "hasMoreData": false}
#         """
#         payload = {"market": symbol}
#         if symbol:
#             all_order = self._get(ORDERS_HISTORY_API, payload)
#         else:
#             all_order = self._get(ORDERS_HISTORY_API)

#         if all_order.success:
#             if all_order.data:
#                 return self._regular_order_payload(all_order)
#             else:
#                 all_order.msg = "query order is empty"

#         return all_order

#     def query_symbols(self, symbol: str = None) -> dict:
#         """
#         Request
#         GET /markets

#         Response
#         {
#             "success": true,
#             "result": [
#                 {
#                 "name": "BTC-0628",
#                 "baseCurrency": null,
#                 "quoteCurrency": null,
#                 "quoteVolume24h": 28914.76,
#                 "change1h": 0.012,
#                 "change24h": 0.0299,
#                 "changeBod": 0.0156,
#                 "highLeverageFeeExempt": false,
#                 "minProvideSize": 0.001,
#                 "type": "future",
#                 "underlying": "BTC",
#                 "enabled": true,
#                 "ask": 3949.25,
#                 "bid": 3949,
#                 "last": 10579.52,
#                 "postOnly": false,
#                 "price": 10579.52,
#                 "priceIncrement": 0.25,
#                 "sizeIncrement": 0.0001,
#                 "restricted": false,
#                 "volumeUsd24h": 28914.76
#                 }
#             ]
#         }
#         """
#         if symbol:
#             symbols = self._get(f"{MARKET_API}/{symbol}")
#         else:
#             symbols = self._get(MARKET_API)

#         if symbols.success:
#             if symbols.data:
#                 return self._regular_symbols_payload(symbols)
#             else:
#                 symbols.msg = "query symbol is empty"

#         return symbols

#     def query_trades(
#         self, start_time: float = None, end_time: float = None, limit=100
#     ) -> dict:
#         """
#         Request
#         GET /fills

#         Response
#         {
#             "success": true,
#             "result": [
#                 {
#                     "id": 6806716374,
#                     "market": "FTT/USD",
#                     "future": null,
#                     "baseCurrency": "FTT",
#                     "quoteCurrency": "USD",
#                     "type": "order",
#                     "side": "buy",
#                     "price": 40.995,
#                     "size": 3.7,
#                     "orderId": 122886081263,
#                     "time": "2022-02-20T08:13:12.292745+00:00",
#                     "tradeId": 3376367982,
#                     "feeRate": 0.0001843,
#                     "fee": 0.00068191,
#                     "feeCurrency": "FTT",
#                     "liquidity": "maker"
#                 }]
#         }
#         """
#         payload = {
#             "include_order_details": 1,
#             "start_time": start_time,
#             "end_time": end_time,
#             "limit": limit,
#         }
#         trades = self._get(TRADE_API, self._remove_none(payload))

#         if trades.success:
#             if trades.data:
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
#     ) -> dict:
#         """
#         Request

#         GET /markets/{market_name}/candles?resolution={resolution}&start_time={start_time}&end_time={end_time}
#         Response

#         {
#             "success": true,
#             "result": [
#                 {
#                     "close": 11055.25,
#                     "high": 11089.0,
#                     "low": 11043.5,
#                     "open": 11059.25,
#                     "startTime": "2019-06-24T17:15:00+00:00",
#                     "volume": 464193.95725
#                 }
#             ]
#         }
#         """

#         def iteration_history(symbol, interval, data, payload):
#             historical_prices_datalist = data.data

#             first_timestamp = int(historical_prices_datalist[0]["time"] / 1000)
#             start_timestamp = payload["start_time"]
#             interval_timestamp = first_timestamp - start_timestamp

#             while interval_timestamp > interval:
#                 first_timestamp = int(historical_prices_datalist[0]["time"] / 1000)
#                 interval_timestamp = first_timestamp - start_timestamp
#                 # logger.debug(first_timestamp)
#                 # logger.debug(interval_timestamp)

#                 payload["end_time"] = first_timestamp
#                 prices = self._get(
#                     f"{MARKET_API}/{symbol}/candles", self._remove_none(payload)
#                 )
#                 if prices.error:
#                     break
#                 # logger.debug(prices.data)

#                 historical_prices_datalist.extend(prices.data)
#                 historical_prices_datalist = [
#                     dict(tupleized)
#                     for tupleized in set(
#                         tuple(item.items()) for item in historical_prices_datalist
#                     )
#                 ]
#                 historical_prices_datalist.sort(key=lambda k: k["time"])
#                 time.sleep(0.1)

#             data.data = historical_prices_datalist
#             logger.debug(f"data length: {len(historical_prices_datalist)}")
#             return data

#         # main
#         payload = {"resolution": TIMEDELTA[interval]}
#         if start:
#             payload["start_time"] = int(datetime.timestamp(start))
#         if end:
#             payload["end_time"] = int(datetime.timestamp(end))

#         logger.debug(payload)
#         historical_prices = self._get(
#             f"{MARKET_API}/{symbol}/candles", self._remove_none(payload)
#         )
#         extra = {"interval": interval, "symbol": symbol}

#         if historical_prices.success:
#             if historical_prices.data:
#                 # handle query time
#                 if "start_time" in payload:
#                     historical_prices = iteration_history(
#                         symbol=symbol,
#                         interval=payload["resolution"],
#                         data=historical_prices,
#                         payload=payload,
#                     )

#                 return self._regular_historical_prices_payload(historical_prices, extra)
#             else:
#                 historical_prices.msg = "query historical prices is empty"

#         return historical_prices

#     def query_last_price(self, symbol: str, interval: str = "1h") -> dict:
#         """
#         Request

#         GET /markets/{market_name}/candles?resolution={resolution}&start_time={start_time}&end_time={end_time}
#         Response

#         {
#             "success": true,
#             "result": [
#                 {
#                     "close": 11055.25,
#                     "high": 11089.0,
#                     "low": 11043.5,
#                     "open": 11059.25,
#                     "startTime": "2019-06-24T17:15:00+00:00",
#                     "volume": 464193.95725
#                 }
#             ]
#         }
#         """
#         payload = {"resolution": TIMEDELTA[interval]}
#         last_historical_prices = self._get(
#             f"{MARKET_API}/{symbol}/candles/last", payload
#         )
#         extra = {"interval": interval, "symbol": symbol}

#         if last_historical_prices.success:
#             if last_historical_prices.data:
#                 return self._regular_historical_prices_payload(
#                     last_historical_prices, extra
#                 )
#             else:
#                 last_historical_prices.msg = "query latest historical prices is empty"

#         return last_historical_prices

#     def query_prices(self, symbol: str = None) -> dict:
#         """
#         Request
#         GET /markets

#         Response
#         {
#             "success": true,
#             "result": [
#                 {
#                     "name": "BTC-0628",
#                     "baseCurrency": null,
#                     "quoteCurrency": null,
#                     "quoteVolume24h": 28914.76,
#                     "change1h": 0.012,
#                     "change24h": 0.0299,
#                     "changeBod": 0.0156,
#                     "highLeverageFeeExempt": false,
#                     "minProvideSize": 0.001,
#                     "type": "future",
#                     "underlying": "BTC",
#                     "enabled": true,
#                     "ask": 3949.25,
#                     "bid": 3949,
#                     "last": 10579.52,
#                     "postOnly": false,
#                     "price": 10579.52,
#                     "priceIncrement": 0.25,
#                     "sizeIncrement": 0.0001,
#                     "restricted": false,
#                     "volumeUsd24h": 28914.76
#                 }
#             ]
#         }
#         """
#         if symbol:
#             spots = self._get(f"{MARKET_API}/{symbol}")
#         else:
#             spots = self._get(f"{MARKET_API}")

#         if spots.success:
#             if spots.data:
#                 return self._regular_symbol_payload(spots)
#             else:
#                 spots.msg = "query spots is empty"

#         return spots

#     def set_leverage(self, leverage: float = 20, symbol: str = None) -> dict:
#         """
#         Request

#         POST /account/leverage
#         {
#             "leverage": 10}
#         """
#         # for FTX - not support single symbol leverage
#         payload = {"leverage": leverage}
#         leverage_payload = self._post(
#             f"{ACCOUNT_API}/leverage", self._remove_none(payload)
#         )
#         logger.debug(leverage_payload)
#         if leverage_payload.success:
#             if leverage_payload.data:
#                 # for FTX - not support single symbol leverage - for general: reutrn symbol
#                 leverage_payload.data = models.LeverageSchema(
#                     symbol=symbol, leverage=leverage
#                 ).dict()
#                 return leverage_payload
#             else:
#                 leverage_payload.msg = "fail to config leverage_payload"

#         return leverage_payload

#     def query_position(
#         self, symbol: str = None, account_type="future", force_return: bool = False
#     ) -> dict:
#         """
#         Request
#         GET /positions

#         Response

#         {
#             "success": true,
#             "result": [
#                 {
#                 "cost": -31.7906,
#                 "cumulativeBuySize": 1.2,
#                 "cumulativeSellSize": 0.0,
#                 "entryPrice": 138.22,
#                 "estimatedLiquidationPrice": 152.1,
#                 "future": "ETH-PERP",
#                 "initialMarginRequirement": 0.1,
#                 "longOrderSize": 1744.55,
#                 "maintenanceMarginRequirement": 0.04,
#                 "netSize": -0.23,
#                 "openSize": 1744.32,
#                 "realizedPnl": 3.39441714,
#                 "recentAverageOpenPrice": 135.31,
#                 "recentBreakEvenPrice": 135.31,
#                 "recentPnl": 3.1134,
#                 "shortOrderSize": 1732.09,
#                 "side": "sell",
#                 "size": 0.23,
#                 "unrealizedPnl": 0,
#                 "collateralUsed": 3.17906
#                 }
#             ]
#         }
#         size, px, unrealizedPnl, symbol.... -- /api/positions
#         """

#         def handle_valid_data(data, symbol: str = None):
#             results = [result for result in data.data if float(result["size"]) != 0]
#             if symbol:
#                 data.data = [result for result in results if result["future"] == symbol]
#             else:
#                 data.data = results

#             if not data.data:
#                 data.msg = "query future position is empty"
#             return data

#         position = self._get(POSITION_API)

#         if position.success:
#             if position.data:
#                 position = handle_valid_data(position, symbol)
#                 return self._regular_position_payload(position)
#             else:
#                 position.msg = "query future position is empty"

#         return position

#     def query_funding_fee(
#         self,
#         symbol: str = None,
#         start: datetime = None,
#         end: datetime = None,
#         limit=1000,
#     ) -> dict:
#         """
#         Request
#         GET /funding_payments

#         Response
#         {
#             "success": true,
#             "result": [
#                 {
#                 "future": "ETH-PERP",
#                 "id": 33830,
#                 "payment": 0.0441342,
#                 "time": "2019-05-15T18:00:00+00:00",
#                 "rate": 0.0001
#                 }
#             ]
#         }
#         """
#         payload = {}
#         if symbol:
#             payload["future"] = symbol.upper()
#         # ftx use sec data
#         if start:
#             payload["start_time"] = int(datetime.timestamp(start))
#         if end:
#             payload["end_time"] = int(datetime.timestamp(end))

#         incomes = self._get(FUNDING_PAYMENTS_API, self._remove_none(payload))
#         if incomes.success:
#             if incomes.data:
#                 return self._regular_incomes_payload(incomes)
#             else:
#                 incomes.msg = "query future incomes is empty"

#         return incomes


# class FTXSpotDataWebsocket(WebsocketClient):
#     """"""

#     def __init__(
#         self, proxy_host: str = "", proxy_port: int = 0, is_testnet: bool = False
#     ) -> None:
#         super().__init__()

#         self.ticks: dict[str, dict] = {}
#         self.reqid: int = 0
#         self.init(
#             host=FTX_WS_URL,
#             proxy_host=proxy_host,
#             proxy_port=proxy_port,
#             ping_interval=15,
#         )
#         self.start()
#         self._logged_in = False
#         self.key = None
#         self.secret = None
#         self.channel = "orderbook"
#         self.kbar_time: int = int(time.time()) - (int(time.time()) % 60)

#     def info(self):
#         return "FTX Spot Data Websocket Start"

#     def connect(self, key: str, secret: str):
#         self.key = key
#         self.secret = secret.encode()
#         if not self._logged_in:
#             self._login()
#         logger.info(self.info())

#     def _login(self) -> None:
#         ts = int(time.time() * 1000)
#         sign = hmac.new(
#             self.secret, f"{ts}websocket_login".encode(), "sha256"
#         ).hexdigest()
#         req = {"op": "login", "args": {"key": self.key, "sign": sign, "time": ts}}
#         self.send_packet(req)
#         self._logged_in = True

#     def on_connected(self) -> None:
#         """"""
#         logger.debug("market data Websocket connect success")
#         if not self._logged_in:
#             self._login()

#         # resubscribe
#         if self.ticks:
#             for symbol in self.ticks.keys():
#                 req: dict = {
#                     "op": "subscribe",
#                     "channel": self.channel,
#                     "market": symbol,
#                 }
#                 self.send_packet(req)

#     def subscribe(self, symbols, on_tick=None, channel="depth") -> None:
#         """
#         general channel: depth, kline
#         mapping channel: orderbooks, customized ticker
#         """
#         if channel not in ["depth", "kline", "trades", "orders"]:
#             Exception("invalid subscription")
#             return

#         self.channel = MAPPING_CHANNEL[channel]
#         if not self._logged_in:
#             self._login()

#         if on_tick:
#             self.on_tick = on_tick

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
#             tick = {
#                 "symbol": symbol,
#                 "exchange": EXCHANGE,
#                 "datetime": datetime.now(UTC_TZ),
#             }
#             self.ticks[symbol] = tick

#             req: dict = {"op": "subscribe", "channel": self.channel, "market": symbol}
#             self.send_packet(req)

#     def customer_olhc(self, symbol, tick_data, data):
#         """generate olhc 1min k by ticker bid ask
#         {'bid': 37776.0, 'ask': 37777.0, 'bidSize': 1.9851, 'askSize': 0.0323, 'last': 37775.0, 'time': 1645587911.8739877}
#         """
#         # call by reference
#         last_price = float(data["last"])
#         volume = float(data["bidSize"] + data["askSize"])
#         tick_data["turnover"] = None
#         tick_data["last_price"] = last_price
#         tick_data["datetime"] = FTXClient.generate_datetime(float(data["time"]))

#         tick_data["asset_type"] = models.ftx_asset_type(symbol)
#         tick_data["name"] = models.normalize_name(symbol, tick_data["asset_type"])
#         tick_data["exchange"] = EXCHANGE

#         if "volume" not in tick_data.keys():
#             tick_data["volume"] = volume
#             tick_data["open_price"] = last_price
#             tick_data["low_price"] = last_price
#             tick_data["high_price"] = last_price

#         if data["time"] > self.kbar_time + 60:
#             self.kbar_time += 60
#             tick_data["volume"] = volume
#             tick_data["open_price"] = last_price
#             tick_data["low_price"] = last_price
#             tick_data["high_price"] = last_price
#         else:
#             tick_data["volume"] += volume
#             tick_data["high_price"] = (
#                 last_price
#                 if last_price > tick_data["high_price"]
#                 else tick_data["high_price"]
#             )
#             tick_data["low_price"] = (
#                 last_price
#                 if last_price < tick_data["low_price"]
#                 else tick_data["low_price"]
#             )

#     def on_packet(self, packet: dict) -> None:
#         """on_packet
#             general channel: depth, kline
#             mapping channel: Orderbooks, customized ticker (only support 1 min K)
#             orderbook = \
#                 'data': {'time': 1645589952.0267098, 'checksum': 1403585516, 'bids': [], 'asks': [[37992.0, 0.0], [38045.0, 0.0102]]
#             ticker = \
#                 'data' : {'bid': 37776.0, 'ask': 37777.0, 'bidSize': 1.9851, 'askSize': 0.0323, 'last': 37775.0, 'time': 1645587911.8739877}
#             trades = \
#                 'data' : {"id": 3396067929, "price": 37996.0, "size": 0.0024, "side": "buy", \
#                     "liquidation": false, "time": "2022-02-23T05:57:22.908810+00:00"},
#         """
#         logger.debug(f"packet: {packet}")

#         message_type = packet["type"]
#         if message_type in {"subscribed", "unsubscribed"}:
#             return
#         elif message_type == "info":
#             if packet["code"] == 20001:
#                 return self.on_connected()
#         elif message_type == "error":
#             raise Exception(packet)

#         channel: str = packet["channel"]
#         data: dict = packet["data"]
#         symbol: str = packet["market"]
#         tick: dict = self.ticks[symbol]

#         if channel == "ticker":
#             self.customer_olhc(symbol, tick, data)
#         elif channel == "orderbook":
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

#             tick["asset_type"] = models.ftx_asset_type(symbol)
#             tick["name"] = models.normalize_name(symbol, tick["asset_type"])
#             tick["exchange"] = EXCHANGE
#         if tick.get("last_price"):
#             tick["localtime"] = datetime.now()

#         self.on_tick(copy(tick))


# class FTXSpotTradeWebsocket(WebsocketClient):
#     """
#     Fills
#     {'op': 'subscribe', 'channel': 'fills'}
#     {
#         "channel": "fills",
#         "data": {
#             "fee": 78.05799225,
#             "feeRate": 0.0014,
#             "future": "BTC-PERP",
#             "id": 7828307,
#             "liquidity": "taker",
#             "market": "BTC-PERP",
#             "orderId": 38065410,
#             "tradeId": 19129310,
#             "price": 3723.75,
#             "side": "buy",
#             "size": 14.973,
#             "time": "2019-05-07T16:40:58.358438+00:00",
#             "type": "order"
#         },
#         "type": "update"
#     }

#     Orders
#     {'op': 'subscribe', 'channel': 'orders'}.
#     {
#         "channel": "orders",
#         "data": {
#             "id": 24852229,
#             "clientId": null,
#             "market": "XRP-PERP",
#             "type": "limit",
#             "side": "buy",
#             "size": 42353.0,
#             "price": 0.2977,
#             "reduceOnly": false,
#             "ioc": false,
#             "postOnly": false,
#             "status": "closed",
#             "filledSize": 42353.0,
#             "remainingSize": 0.0,
#             "avgFillPrice": 0.2978
#         },
#         "type": "update"
#     }
#     """

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
#         self.init(host=FTX_WS_URL, proxy_host=proxy_host, proxy_port=proxy_port)
#         self.start()
#         self._logged_in = False
#         self.key = None
#         self.secret = None
#         self.channel = "orders"

#     def info(self):
#         return "FTX Spot Trade Websocket Start"

#     def connect(self, key: str, secret: str):
#         self.key = key
#         self.secret = secret.encode()
#         if not self._logged_in:
#             self._login()
#         logger.info(self.info())

#     def _login(self) -> None:
#         ts = int(time.time() * 1000)
#         sign = hmac.new(
#             self.secret, f"{ts}websocket_login".encode(), "sha256"
#         ).hexdigest()
#         self.send_packet(
#             {"op": "login", "args": {"key": self.key, "sign": sign, "time": ts}}
#         )
#         self._logged_in = True

#     def on_connected(self) -> None:
#         logger.debug("spot user trade data Websocket connect success")

#     def subscribe(self, channel="orders") -> None:
#         """
#         General channel: trades / orders
#         Mapping channel: Fills / Orders
#         """
#         if channel not in ["depth", "kline", "trades", "orders"]:
#             Exception("invalid subscription")
#             return

#         self.channel = MAPPING_CHANNEL[channel]
#         if not self._logged_in:
#             self._login()

#         req: dict = {"op": "subscribe", "channel": self.channel}
#         self.send_packet(req)

#     def on_packet(self, packet: dict) -> None:
#         if packet["channel"] == "account":
#             self.on_account(packet)
#         elif packet["channel"] == "orders":
#             self.on_order(packet)
#         elif packet["channel"] == "fills":
#             self.on_trade(packet)

#     def on_account(self, packet: dict) -> None:
#         """FTX don't support"""

#     def on_order(self, packet: dict) -> None:
#         """on order place/cancel
#         Orders
#         {'op': 'subscribe', 'channel': 'orders'}.
#         {
#             "channel": "orders",
#             "data": {
#                 "id": 24852229,
#                 "clientId": null,
#                 "market": "XRP-PERP",
#                 "type": "limit",
#                 "side": "buy",
#                 "size": 42353.0,
#                 "price": 0.2977,
#                 "reduceOnly": false,
#                 "ioc": false,
#                 "postOnly": false,
#                 "status": "closed",
#                 "filledSize": 42353.0,
#                 "remainingSize": 0.0,
#                 "avgFillPrice": 0.2978
#             },
#             "type": "update"
#         }
#         """
#         # filter not support trade type
#         logger.debug(f"packet: {packet}")
#         data = packet["data"]

#         if data["type"] not in ["limit", "market"]:
#             return

#         if data["id"] == "":
#             order_id: str = data["clientId"]
#         else:
#             order_id: str = data["id"]

#         order = {
#             "symbol": data["market"].lower(),
#             "order_id": order_id,
#             "type": data["type"],  # LIMIT, MARKET
#             "side": data["side"],  # BUY, SELL
#             "price": float(data["price"]),
#             "quantity": float(data["size"]),
#             "traded": float(data["remainingSize"]),
#             "status": data[
#                 "status"
#             ],  # NEW, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED, EXPIRED(CANCELED)
#             "datetime": datetime.now(),  # payload don't provide this one
#         }
#         order["asset_type"] = models.ftx_asset_type(order["symbol"])
#         order["name"] = models.normalize_name(order["symbol"], order["asset_type"])
#         order["exchange"] = EXCHANGE

#         if self.on_order_callback:
#             self.on_order_callback(order)

#     def on_trade(self, packet: dict) -> None:
#         """on order place/cancel
#         Fill
#         "fee": 78.05799225,
#         "feeRate": 0.0014,
#         "future": "BTC-PERP",
#         "id": 7828307,
#         "liquidity": "taker",
#         "market": "BTC-PERP",
#         "orderId": 38065410,
#         "tradeId": 19129310,
#         "price": 3723.75,
#         "side": "buy",
#         "size": 14.973,
#         "time": "2019-05-07T16:40:58.358438+00:00",
#         "type": "order"
#         """
#         # filter not support trade type
#         logger.debug(f"packet: {packet}")
#         data = packet["data"]

#         trade = {
#             "symbol": data["market"],
#             "order_id": data["orderId"],
#             "trade_id": data["tradeId"],
#             "side": data["buy"],
#             "price": float(data["side"]),
#             "quantity": float(data["size"]),
#             "datetime": datetime.fromtimestamp(time),
#         }
#         trade["asset_type"] = models.ftx_asset_type(trade["symbol"])
#         trade["name"] = models.normalize_name(trade["symbol"], trade["asset_type"])
#         trade["exchange"] = EXCHANGE

#         if self.on_trade_callback:
#             self.on_trade_callback(trade)
