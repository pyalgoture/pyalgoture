import json
import sys
import time
from collections.abc import Callable
from datetime import datetime
from threading import Lock
from typing import Any, Optional

import requests
from dateutil.parser import parse

from ..utils import models
from ..utils.client_rest import Response, RestClient
from ..utils.client_ws import WebsocketClient
from ..utils.logger import change_logger_level, get_logger

logger = get_logger()

DUMMY_API_HOST = "http://localhost:2023"

DUMMY_WS_PUBLIC_HOST = "ws://localhost:2023/v1/public/ws"
DUMMY_WS_PRIVATE_HOST = "ws://localhost:2023/v1/ws"


EXCHANGE = "DUMMY"

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


class DummyClient(RestClient):
    """ """

    def __init__(
        self,
        balance,
        overwrite: bool = False,
        debug: bool = False,
        session: bool = True,
        proxy_host: str = "",
        proxy_port: int = 0,
        **kwargs,
    ) -> None:
        # url_base: str = None,
        super().__init__(url_base=DUMMY_API_HOST, proxy_host=proxy_host, proxy_port=proxy_port)

        self.order_count: int = 0
        self.order_count_lock: Lock = Lock()

        self.overwrite = overwrite
        self.balance = balance

        self.jwt_token: str = ""
        self.debug = debug
        if self.debug:
            global logger
            logger = change_logger_level(logger, new_level="TRACE")
        self.session = session

    def info(self):
        return "Dummy REST API start"

    def new_order_id(self) -> str:
        """new_order_id"""
        # prefix: str = datetime.now().strftime("%Y%m%d-%H%M%S-")
        prefix: str = models.generate_datetime().strftime("%y%m%d-%H%M%S%f")[:-3] + "-"  #  # datetime.now()
        with self.order_count_lock:
            self.order_count += 1
            suffix: str = str(self.order_count).rjust(5, "0")

            order_id: str = "x-" + prefix + suffix
            return order_id

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if self.session:
            response = self.add_request(method="GET", path=path, params=params)
        else:
            response = self.request(method="GET", path=path, params=params)
        logger.trace(f"[GET]({path}) - params:{params} | response: {response.text}({response.status_code})")
        # TODO: based on statu code and force retry
        return self._process_response(response)

    def _post(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if self.session:
            response = self.add_request(method="POST", path=path, data=params, headers={"Referer": "PYALGOTURE"})
        else:
            response = self.query(
                method="POST",
                path=path,
                data=params,
                headers={"Referer": "PYALGOTURE"},
                timeout=(30, 60),
            )
        logger.trace(f"[POST]({path}) - params:{params} | response: {response.text}({response.status_code})")
        return self._process_response(response)

    def _delete(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if self.session:
            response = self.add_request(method="DELETE", path=path, data=params)
        else:
            response = self.request(method="DELETE", path=path, data=params)
        logger.trace(f"[DELETE]({path}) - params:{params} | response: {response.text}({response.status_code})")
        return self._process_response(response)

    def sign(self, request):
        """generate bybit signature"""

        ### TODO: check if jwt_token is expired and relogin to gen the new jwt_token
        if self.jwt_token:
            headers = {
                "Authorization": f"Bearer {self.jwt_token}",
            }

            request.headers = headers

        # print(f"[DEBUG], path:{request.path}; method:{method}; params:{request.params}; data:{request.data}; headers:{request.headers}; signature:{signature}")

        return request

    def register(
        self,
        username: str,
        password: str,
    ):
        res = requests.post(
            DUMMY_API_HOST + "/v1/users/registration/",
            json={"username": username, "password": password},
        ).json()
        print(f"Dummy broker account register return:{res}")

    def connect(self, username: str, password: str, **kwargs) -> None:
        self.username = username
        self.password = password
        res = self._post("/v1/users/login/", params={"username": username, "password": password})
        # print(f"[connect - login] res:{res}")
        if res["error"]:
            return False
        self.jwt_token = res["data"]["token"]

        res = self._post(
            "/v1/connect/",
            params={
                "balance": self.balance,
                "overwrite": self.overwrite,
            },
        )
        # print(f"[connect] res:{res}")

        return True

    def create_position(
        self,
        code: str,
        exchange: str,
        asset_type: str,
        commission_rate=None,
        taker_commission_rate=None,
        maker_commission_rate=None,
        leverage=None,
        is_isolated=None,
    ):
        # print(f"[dummy create_position input] code:{code}, exchange:{exchange}, asset_type:{asset_type}")
        res = self._post(
            "/v1/position/",
            params={
                "exchange": exchange,
                "asset_type": asset_type,
                "code": code,
                "commission_rate": commission_rate,
                "taker_commission_rate": taker_commission_rate,
                "maker_commission_rate": maker_commission_rate,
                "is_isolated": is_isolated,
                "leverage": leverage,
            },
        )
        # print(f"[create_position] res:{res}")
        return True

    def _process_response(self, response: Response) -> dict:
        """ """
        try:
            data = response.json()
            # print(data,'!!!!', response.ok)
        except ValueError:
            print(f"[DUMMY] in _process_response, something went wrong when parsing data. Raw data:{response.text}")
            logger.error(response.text)
            raise
        else:
            if not response.ok or (response.ok and data["error"]):
                payload_data = models.CommonDataSchema(status_code=response.status_code, msg=dict(data))
                return models.CommonResponseSchema(
                    success=False,
                    error=True,
                    data=payload_data.dict(),
                    msg=data.get("msg", "No return msg from dummy exchange."),
                )
            else:
                return models.CommonResponseSchema(success=True, error=False, data=data["data"], msg="query ok")

    def query_account(self) -> dict:
        """ """
        payload = {}
        balance = self._get("/v1/account/", payload)
        # print(f">>> balance:{balance}")

        if balance.success:
            if balance.data:
                data = balance.data
                balance.data = {
                    data["asset"]: models.AccountSchema(
                        symbol=data["asset"],
                        exchange=data["exchange"],
                        balance=data["balance"],
                        available=data["available"],
                        frozen=data["frozen"],
                    ).to_general_form()
                }
            else:
                balance.msg = "query account is empty"

        return balance

    def send_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        exchange: str,
        asset_type: str,
        price=None,
        position_side: str = "",
        reduce_only: bool = False,
        order_id: str = "",
        time_in_force: str = "GTC",
        **kwargs,
    ) -> dict:
        """ """

        payload = {
            "code": symbol.upper(),
            "side": side,
            "quantity": quantity,
            "order_id": order_id if order_id else self.new_order_id(),
            "exchange": exchange,
            "asset_type": asset_type,
            "time_in_force": time_in_force,
        }
        if price:
            payload["price"] = price
        if position_side:
            payload["position_side"] = position_side
        if reduce_only:
            payload["reduce_only"] = reduce_only

        orders_response = self._post("/v1/order/", payload)
        # print(f">>> send_order_response :{orders_response}")

        if orders_response.success:
            if orders_response.data:
                data = orders_response.data
                orders_response.data = models.SendOrderResultSchema(
                    asset_type=asset_type,
                    exchange=exchange,
                    symbol=symbol,
                    name=symbol,
                    ori_order_id=data["order_id"],
                    order_id=order_id,
                    created_at=datetime.now(),
                ).to_general_form()
            else:
                orders_response.msg = "query sender order is empty"

        return orders_response

    def cancel_order(
        self,
        order_id: str,
        exchange: str,
        asset_type: str,
    ) -> dict:
        """ """
        payload = {"exchange": exchange, "asset_type": asset_type, "order_id": order_id}
        return self._delete("/v1/order/", payload)

    def query_order_status(self, order_id: str = None) -> dict:
        """ """
        return self.query_all_orders(order_id=order_id)

    def query_open_orders(
        self,
        symbol: str = None,
    ) -> dict:
        return self.query_all_orders(symbol=symbol, open_order_only=True)

    def query_all_orders(self, symbol: str = None, order_id: str = None, open_order_only=False) -> dict:
        """ """
        payload = {}
        if order_id:
            payload["order_id"] = order_id
        if symbol:
            payload["code"] = symbol
        if open_order_only:
            payload["open_order_only"] = open_order_only

        orders = self._get("/v1/orders/", payload)
        # print(f">>> orders:{orders}")

        if orders.success:
            if orders.data:
                trade_list = []
                for data in orders.data:
                    trade_list.append(
                        models.OrderSchema(
                            symbol=data["code"],
                            name=data["symbol"],
                            exchange=data["exchange"],
                            asset_type=data["asset_type"],
                            ori_order_id=data["order_id"],
                            order_id=data["order_id"],
                            price=data["price"],
                            type=data["order_type"],
                            size=data["quantity"],
                            side=data["side"],
                            status=data["status"],
                            created_at=parse(data["created_at"]),
                        ).to_general_form()
                    )
                orders.data = trade_list
            else:
                orders.msg = "query trade is empty"

        return orders

    def query_trades(
        self,
        symbol: str = None,
        start: datetime = None,
        end: datetime = None,
        limit=100,
    ) -> dict:
        """ """
        payload = {}
        trades = self._get("/v1/trades/", payload)
        # print(f">>> trades:{trades}")

        if trades.success:
            if trades.data:
                trade_list = []
                for data in trades.data:
                    trade_list.append(
                        models.TradeSchema(
                            symbol=data["code"],
                            name=data["symbol"],
                            exchange=data["exchange"],
                            asset_type=data["asset_type"],
                            ori_order_id=data["order_id"],
                            order_id=data["order_id"],
                            price=data["price"],
                            quantity=data["quantity"],
                            side=data["side"],
                            commission=data["commission"],
                            commission_asset=data["commission_asset"],
                            position_side=data["position_side"],
                            datetime=parse(data["traded_at"]),
                        ).to_general_form()
                    )
                trades.data = trade_list
            else:
                trades.msg = "query trade is empty"

        return trades

    def query_position(
        self,
        symbol: str = None,
        settle_coin: str = "USDT",
        limit: int = 200,
        force_return: bool = False,
    ) -> dict:
        """ """
        payload = {}
        position = self._get("/v1/position/", payload)
        # print(f">>> position:{position}")

        if position.success:
            if position.data:
                trade_list = []
                for data in position.data:
                    trade_list.append(
                        models.PositionSchema(
                            symbol=data["code"],
                            name=data["symbol"],
                            exchange=data["exchange"],
                            asset_type=data["asset_type"],
                            entry_price=data["avg_open_price"],
                            size=data["quantity"],
                            side=data["side"],
                            position_side=data["position_side"],
                            leverage=data["leverage"],
                            is_isolated=data["is_isolated"],
                            position_margin=data["position_margin"],
                            initial_margin=data["initial_margin"],
                            maintenance_margin=data["maintenance_margin"],
                            realized_pnl=data["realized_pnl"],
                            unrealized_pnl=data["unrealized_pnl"],
                        ).to_general_form()
                    )
                position.data = trade_list
            else:
                position.msg = "query trade is empty"

        return position

    def set_leverage(
        self,
        symbol: str,
        exchange: str,
        asset_type: str,
        leverage: float = None,
        is_isolated: bool = None,
    ) -> dict:
        """ """
        if leverage:
            payload = {
                "exchange": exchange,
                "asset_type": asset_type,
                "code": symbol,
                "leverage": leverage,
            }
            response = self._post("/v1/leverage/", remove_none(payload))
            # print(response,'!!!!leverage')
        if is_isolated:
            payload = {
                "exchange": exchange,
                "asset_type": asset_type,
                "code": symbol,
                "is_isolated": is_isolated,
            }
            response = self._post("/v1/margin_mode/", remove_none(payload))
            # print(response,'!!!!margin_mode')

        if response.success:
            response.data = {
                "symbol": symbol,
                "leverage": leverage,
                "is_isolated": is_isolated,
                "exchange": exchange,
                "asset_type": asset_type,
            }
        return response


class DummyTradeWebsocket(WebsocketClient):
    """ """

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
        # exchange=None,
        # asset_type: str = None,
        ping_interval: int = 59,
        debug: bool = False,
        # logger = None,
        **kwargs,
    ) -> None:
        super().__init__(debug=debug)  # , logger=logger)

        self.on_account_callback = on_account_callback
        self.on_order_callback = on_order_callback
        self.on_trade_callback = on_trade_callback
        self.on_position_callback = on_position_callback
        self.on_connected_callback = on_connected_callback
        self.on_disconnected_callback = on_disconnected_callback
        self.on_error_callback = on_error_callback

        self.ping_interval = ping_interval
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self._logged_in = False
        self.key = None
        self.secret = None
        self.debug = debug
        if self.debug:
            global logger
            logger = change_logger_level(logger, new_level="TRACE")

    def info(self):
        return "Dummy Trade Websocket Start"

    def custom_ping(self):
        return json.dumps({"op": "ping"})

    def register(
        self,
        username: str,
        password: str,
    ):
        res = requests.post(
            DUMMY_API_HOST + "/v1/users/registration/",
            json={"username": username, "password": password},
        ).json()
        # print(f"Dummy broker account register return:{res}")

    def connect(self, username: str, password: str):
        """
        Give api key and secret for Authorization.
        """
        self.username = username
        self.password = password

        if not self._logged_in:
            self._login()

        self.init(
            host=DUMMY_WS_PRIVATE_HOST,
            proxy_host=self.proxy_host,
            proxy_port=self.proxy_port,
            ping_interval=self.ping_interval,
            headers={
                "Authorization": f"Bearer {self.jwt_token}",
            },
        )
        self.start()

        if self.waitfor_connection():
            logger.info(self.info())
        else:
            logger.info(self.info() + " failed")
            return False

        logger.info(self.info())

        return True

    def _login(self):
        """
        Authorize websocket connection.
        """
        res = requests.post(
            DUMMY_API_HOST + "/v1/users/login/",
            json={"username": self.username, "password": self.password},
        ).json()
        # res = self._post('/v1/users/login/', params={'username':self.username, 'password':self.password})

        if res["error"]:
            logger.error(f"Dummy user trade data Websocket login failed. Return:{res}")
            return False
        self.jwt_token = res["data"]["token"]

        self._logged_in = True

    def on_disconnected(self) -> None:
        logger.info(f"Dummy user trade data Websocket disconnected, now try to connect @ {datetime.now()}")
        self._logged_in = False
        if self.on_disconnected_callback:
            self.on_disconnected_callback()

    def on_connected(self) -> None:
        if self.waitfor_connection():
            logger.info("Dummy user trade data websocket connect success")
        else:
            logger.error("Dummy user trade data websocket connect fail")

        if not self._logged_in:
            self._login()
            # self._logged_in = True
        # logger.debug("Dummy Future user trade data Websocket connect success")

        req: dict = self.sub_stream(channel=["trades", "orders", "account", "position"])
        logger.trace(f"Dummy user trade data websocket on_connected req:{req} - self._active:{self._active}")

        self.send_packet(req)
        if self.on_connected_callback:
            self.on_connected_callback()

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
            "action": "subscribe",
        }
        sub["topic"] = channel

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
        logger.trace(f"subscribe - req: {req}")
        self.send_packet(req)

    def on_packet(self, packet: dict) -> None:
        """ """
        logger.trace(f"[DEBUG] Dummy Trade Websocket - on_packet event {packet}")

        if not packet:
            logger.debug(f"unknown packet event {packet}")
            return

        if "op" in packet and packet["op"] == "ping":
            logger.trace(f"healthcheck message {packet}")
            return

        if "topic" not in packet:
            logger.info(f"[dummy] the other packet without topic: {packet}")
            return

        if packet["topic"] == "account":
            self.on_account(packet)
        elif packet["topic"] == "orders":
            self.on_order(packet)
        elif packet["topic"] == "trades":
            self.on_trade(packet)
        elif packet["topic"] == "position":
            self.on_position(packet)
        else:
            logger.info(f"[dummy] the other packet type: {packet}")

    def on_account(self, packet: dict) -> None:
        """ """
        try:
            logger.debug(f"[dummy] packet on_account: {packet}")
            data = packet["data"]
            if not data:
                logger.error("[dummy] packet on_account, data is None...")
                return
            account = models.AccountSchema(
                symbol=data["asset"],
                exchange=data["exchange"],
                balance=data["balance"],
                available=data["available"],
                frozen=data["frozen"],
            ).to_general_form()
            if account and account["symbol"] and self.on_account_callback:
                self.on_account_callback(account)
        except Exception:
            et, ev, tb = sys.exc_info()
            self.on_error(et, ev, tb)

    def on_order(self, packet: dict) -> None:
        """ """
        try:
            logger.debug(f"[dummy] packet on_order: {packet}")
            data: dict = packet["data"]
            if not data:
                logger.error("[dummy] packet on_order, data is None...")
                return
            order = models.OrderSchema(
                symbol=data["code"],
                name=data["symbol"],
                exchange=data["exchange"],
                asset_type=data["asset_type"],
                ori_order_id=data["order_id"],
                order_id=data["order_id"],
                price=data["price"],
                type=data["order_type"],
                size=data["quantity"],
                side=data["side"],
                status=data["status"],
                created_at=parse(data["created_at"]),
            ).to_general_form()

            if self.on_order_callback:
                self.on_order_callback(order)

        except Exception:
            et, ev, tb = sys.exc_info()
            self.on_error(et, ev, tb)

    def on_trade(self, packet: dict) -> None:
        """ """
        try:
            logger.debug(f"[dummy] packet on_trade: {packet}")
            data: dict = packet["data"]
            if not data:
                logger.error("[dummy] packet on_trade, data is None...")
                return

            trade = models.TradeSchema(
                symbol=data["code"],
                name=data["symbol"],
                exchange=data["exchange"],
                asset_type=data["asset_type"],
                ori_order_id=data["order_id"],
                order_id=data["order_id"],
                price=data["price"],
                quantity=data["quantity"],
                side=data["side"],
                commission=data["commission"],
                commission_asset=data["commission_asset"],
                position_side=data["position_side"],
                datetime=parse(data["traded_at"]),
            ).to_general_form()
            if trade and self.on_trade_callback:
                self.on_trade_callback(trade)
        except Exception:
            et, ev, tb = sys.exc_info()
            self.on_error(et, ev, tb)

    def on_position(self, packet: dict) -> None:
        """ """
        try:
            logger.debug(f"[dummy] packet on_position: {packet}")
            data: dict = packet["data"]
            if not data:
                logger.error("[dummy] packet on_position, data is None...")
                return
            position = models.PositionSchema(
                symbol=data["code"],
                name=data["symbol"],
                exchange=data["exchange"],
                asset_type=data["asset_type"],
                entry_price=data["avg_open_price"],
                size=data["quantity"],
                side=data["side"],
                position_side=data["position_side"],
                leverage=data["leverage"],
                is_isolated=data["is_isolated"],
                position_margin=data["position_margin"],
                initial_margin=data["initial_margin"],
                maintenance_margin=data["maintenance_margin"],
                realized_pnl=data["realized_pnl"],
                unrealized_pnl=data["unrealized_pnl"],
            ).to_general_form()
            if position and self.on_position_callback:
                self.on_position_callback(position)

        except Exception:
            et, ev, tb = sys.exc_info()
            self.on_error(et, ev, tb)
