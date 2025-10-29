import base64
import copy
import hashlib
import hmac
import json
import threading
import time
import uuid
from uuid import uuid4

import websocket

# from Crypto.Hash import SHA256
# from Crypto.PublicKey import RSA
# from Crypto.Signature import PKCS1_v1_5
from ..utils.logger import get_logger

logger = get_logger()


class UnauthorizedExceptionError(Exception):
    pass


class InvalidChannelTypeError(Exception):
    pass


class TopicMismatchError(Exception):
    pass


class FailedRequestError(Exception):
    """
    Exception raised for failed requests.

    Attributes:
        request -- The original request that caused the error.
        message -- Explanation of the error.
        status_code -- The code number returned.
        time -- The time of the error.
        resp_headers -- The response headers from API. None, if the request caused an error locally.
    """

    def __init__(self, request, message, status_code, time, resp_headers):
        self.request = request
        self.message = message
        self.status_code = status_code
        self.time = time
        self.resp_headers = resp_headers
        super().__init__(f"{message.capitalize()} (ErrCode: {status_code}) (ErrTime: {time}).\nRequest → {request}.")


class InvalidRequestError(Exception):
    """
    Exception raised for returned Bybit errors.

    Attributes:
        request -- The original request that caused the error.
        message -- Explanation of the error.
        status_code -- The code number returned.
        time -- The time of the error.
        resp_headers -- The response headers from API. None, if the request caused an error locally.
    """

    def __init__(self, request, message, status_code, time, resp_headers):
        self.request = request
        self.message = message
        self.status_code = status_code
        self.time = time
        self.resp_headers = resp_headers
        super().__init__(f"{message} (ErrCode: {status_code}) (ErrTime: {time}).\nRequest → {request}.")


def generate_timestamp():
    """
    Return a millisecond integer timestamp.
    """
    return int(time.time() * 10**3)


def find_index(source, target, key):
    """
    Find the index in source list of the targeted ID.
    """
    return next(i for i, j in enumerate(source) if j[key] == target[key])


def generate_signature(use_rsa_authentication, secret, param_str):
    def generate_hmac():
        hash = hmac.new(
            bytes(secret, "utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256,
        )
        return hash.hexdigest()

    # def generate_rsa():
    #     hash = SHA256.new(param_str.encode("utf-8"))
    #     encoded_signature = base64.b64encode(PKCS1_v1_5.new(RSA.importKey(secret)).sign(hash))
    #     return encoded_signature.decode()

    # if not use_rsa_authentication:
    #     return generate_hmac()
    # else:
    #     return generate_rsa()

    return generate_hmac()


WSS_NAME = "Unified V5"
PRIVATE_WSS = "wss://{SUBDOMAIN}.{DOMAIN}.com/v5/private"
PUBLIC_WSS = "wss://{SUBDOMAIN}.{DOMAIN}.com/v5/public/{CHANNEL_TYPE}"
AVAILABLE_CHANNEL_TYPES = [
    "inverse",
    "linear",
    "spot",
    "option",
    "private",
]


SUBDOMAIN_TESTNET = "stream-testnet"
SUBDOMAIN_MAINNET = "stream"
DEMO_SUBDOMAIN_TESTNET = "stream-demo-testnet"
DEMO_SUBDOMAIN_MAINNET = "stream-demo"
DOMAIN_MAIN = "bybit"
DOMAIN_ALT = "bytick"
TLD_MAIN = "com"


WSS_NAME = "WebSocket Trading"
TRADE_WSS = "wss://{SUBDOMAIN}.{DOMAIN}.{TLD}/v5/trade"


class _WebSocketManager:
    def __init__(
        self,
        _callback_function,
        ws_name,
        testnet,
        tld="",
        domain="",
        demo=False,
        rsa_authentication=False,
        api_key=None,
        api_secret=None,
        ping_interval=20,
        ping_timeout=10,
        retries=10,
        restart_on_error=True,
        trace_logging=False,
        private_auth_expire=1,
    ):
        self.testnet = testnet
        self.domain = domain
        self.tld = tld
        self.rsa_authentication = rsa_authentication
        self.demo = demo
        # Set API keys.
        self.api_key = api_key
        self.api_secret = api_secret

        self.callback = _callback_function
        self.ws_name = ws_name
        if api_key:
            self.ws_name += " (Auth)"

        # Delta time for private auth expiration in seconds
        self.private_auth_expire = private_auth_expire

        # Setup the callback directory following the format:
        #   {
        #       "topic_name": function
        #   }
        self.callback_directory = {}

        # Record the subscriptions made so that we can resubscribe if the WSS
        # connection is broken.
        self.subscriptions = []

        # Set ping settings.
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.custom_ping_message = json.dumps({"op": "ping"})
        self.retries = retries

        # Other optional data handling settings.
        self.handle_error = restart_on_error

        # Enable websocket-client's trace logging for extra debug information
        # on the websocket connection, including the raw sent & recv messages
        websocket.enableTrace(trace_logging)

        # Set initial state, initialize dictionary and connect.
        self._reset()
        self.attempting_connection = False

    def stop(self) -> None:
        self.exit()

    def _on_open(self):
        """
        Log WS open.
        """
        logger.debug(f"WebSocket {self.ws_name} opened.")

    def _on_message(self, message):
        """
        Parse incoming messages.
        """
        message = json.loads(message)
        if self._is_custom_pong(message):
            return
        else:
            self.callback(message)

    def is_connected(self):
        try:
            if self.ws.sock.connected:
                return True
            else:
                return False
        except AttributeError:
            return False

    def _connect(self, url):
        """
        Open websocket in a thread.
        """

        def resubscribe_to_topics():
            if not self.subscriptions:
                # There are no subscriptions to resubscribe to, probably
                # because this is a brand new WSS initialisation so there was
                # no previous WSS connection.
                return

            for req_id, subscription_message in self.subscriptions.items():
                self.ws.send(subscription_message)

        self.attempting_connection = True

        # Set endpoint.
        subdomain = SUBDOMAIN_TESTNET if self.testnet else SUBDOMAIN_MAINNET
        domain = DOMAIN_MAIN if not self.domain else self.domain
        tld = TLD_MAIN if not self.tld else self.tld
        if self.demo:
            if self.testnet:
                subdomain = DEMO_SUBDOMAIN_TESTNET
            else:
                subdomain = DEMO_SUBDOMAIN_MAINNET
        url = url.format(SUBDOMAIN=subdomain, DOMAIN=domain, TLD=tld)
        self.endpoint = url

        # Attempt to connect for X seconds.
        retries = self.retries
        if retries == 0:
            infinitely_reconnect = True
        else:
            infinitely_reconnect = False

        while (infinitely_reconnect or retries > 0) and not self.is_connected():
            logger.info(f"WebSocket {self.ws_name} attempting connection...")
            self.ws = websocket.WebSocketApp(
                url=url,
                on_message=lambda ws, msg: self._on_message(msg),
                on_close=lambda ws, *args: self._on_close(),
                on_open=lambda ws, *args: self._on_open(),
                on_error=lambda ws, err: self._on_error(err),
                on_pong=lambda ws, *args: self._on_pong(),
            )

            # Setup the thread running WebSocketApp.
            self.wst = threading.Thread(
                target=lambda: self.ws.run_forever(
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                )
            )

            # Configure as daemon; start.
            self.wst.daemon = True
            self.wst.start()

            retries -= 1
            while self.wst.is_alive():
                if self.ws.sock and self.is_connected():
                    break

            # If connection was not successful, raise error.
            if not infinitely_reconnect and retries <= 0:
                self.exit()
                raise websocket.WebSocketTimeoutException(
                    f"WebSocket {self.ws_name} ({self.endpoint}) connection "
                    f"failed. Too many connection attempts. pybit will no "
                    f"longer try to reconnect."
                )

        logger.info(f"WebSocket {self.ws_name} connected")

        # If given an api_key, authenticate.
        if self.api_key and self.api_secret:
            self._auth()

        resubscribe_to_topics()
        self._send_initial_ping()

        self.attempting_connection = False

    def _auth(self):
        """
        Prepares authentication signature per Bybit API specifications.
        """

        expires = generate_timestamp() + (self.private_auth_expire * 1000)

        param_str = f"GET/realtime{expires}"

        signature = generate_signature(self.rsa_authentication, self.api_secret, param_str)

        # Authenticate with API.
        self.ws.send(json.dumps({"op": "auth", "args": [self.api_key, expires, signature]}))

    def _on_error(self, error):
        """
        Exit on errors and raise exception, or attempt reconnect.
        """
        if type(error).__name__ not in [
            "WebSocketConnectionClosedException",
            "ConnectionResetError",
            "WebSocketTimeoutException",
        ]:
            # Raises errors not related to websocket disconnection.
            self.exit()
            raise error

        if not self.exited:
            logger.error(f"WebSocket {self.ws_name} ({self.endpoint}) encountered error: {error}.")
            self.exit()

        # Reconnect.
        if self.handle_error and not self.attempting_connection:
            self._reset()
            self._connect(self.endpoint)

    def _on_close(self):
        """
        Log WS close.
        """
        logger.debug(f"WebSocket {self.ws_name} closed.")

    def _on_pong(self):
        """
        Sends a custom ping upon the receipt of the pong frame.

        The websocket library will automatically send ping frames. However, to
        ensure the connection to Bybit stays open, we need to send a custom
        ping message separately from this. When we receive the response to the
        ping frame, this method is called, and we will send the custom ping as
        a normal OPCODE_TEXT message and not an OPCODE_PING.
        """
        self._send_custom_ping()

    def _send_custom_ping(self):
        self.ws.send(self.custom_ping_message)

    def _send_initial_ping(self):
        """https://github.com/bybit-exchange/pybit/issues/164"""
        timer = threading.Timer(self.ping_interval, self._send_custom_ping)
        timer.start()

    @staticmethod
    def _is_custom_pong(message):
        """
        Referring to OPCODE_TEXT pongs from Bybit, not OPCODE_PONG.
        """
        if message.get("ret_msg") == "pong" or message.get("op") == "pong":
            return True

    def _reset(self):
        """
        Set state booleans and initialize dictionary.
        """
        self.exited = False
        self.auth = False
        self.data = {}

    def exit(self):
        """
        Closes the websocket connection.
        """

        self.ws.close()
        while self.ws.sock:
            continue
        self.exited = True


class _V5WebSocketManager(_WebSocketManager):
    def __init__(self, ws_name, **kwargs):
        callback_function = (
            kwargs.pop("callback_function") if kwargs.get("callback_function") else self._handle_incoming_message
        )
        super().__init__(callback_function, ws_name, **kwargs)

        self.subscriptions = {}

        self.standard_private_topics = [
            "position",
            "execution",
            "order",
            "wallet",
            "greeks",
            "spread.order",
            "spread.execution",
        ]

        self.other_private_topics = ["execution.fast"]

    def subscribe(self, topic: str, callback, symbol: str | list = False):
        def prepare_subscription_args(list_of_symbols):
            """
            Prepares the topic for subscription by formatting it with the
            desired symbols.
            """

            if topic in self.standard_private_topics:
                # private topics do not support filters
                return [topic]

            topics = []
            for single_symbol in list_of_symbols:
                topics.append(topic.format(symbol=single_symbol))
            return topics

        if isinstance(symbol, str):
            symbol = [symbol]

        subscription_args = prepare_subscription_args(symbol)
        self._check_callback_directory(subscription_args)

        req_id = str(uuid4())

        subscription_message = json.dumps({"op": "subscribe", "req_id": req_id, "args": subscription_args})
        while not self.is_connected():
            # Wait until the connection is open before subscribing.
            time.sleep(0.1)
        self.ws.send(subscription_message)
        self.subscriptions[req_id] = subscription_message
        for topic in subscription_args:
            self._set_callback(topic, callback)

    def _initialise_local_data(self, topic):
        # Create self.data
        try:
            self.data[topic]
        except KeyError:
            self.data[topic] = []

    def _process_delta_orderbook(self, message, topic):
        self._initialise_local_data(topic)

        # Record the initial snapshot.
        if "snapshot" in message["type"]:
            self.data[topic] = message["data"]
            return

        # Make updates according to delta response.
        book_sides = {"b": message["data"]["b"], "a": message["data"]["a"]}
        self.data[topic]["u"] = message["data"]["u"]
        self.data[topic]["seq"] = message["data"]["seq"]

        for side, entries in book_sides.items():
            for entry in entries:
                # Delete.
                if float(entry[1]) == 0:
                    index = find_index(self.data[topic][side], entry, 0)
                    self.data[topic][side].pop(index)
                    continue

                # Insert.
                price_level_exists = entry[0] in [level[0] for level in self.data[topic][side]]
                if not price_level_exists:
                    self.data[topic][side].append(entry)
                    continue

                # Update.
                qty_changed = entry[1] != next(level[1] for level in self.data[topic][side] if level[0] == entry[0])
                if price_level_exists and qty_changed:
                    index = find_index(self.data[topic][side], entry, 0)
                    self.data[topic][side][index] = entry
                    continue

    def _process_delta_ticker(self, message, topic):
        self._initialise_local_data(topic)

        # Record the initial snapshot.
        if "snapshot" in message["type"]:
            self.data[topic] = message["data"]

        # Make updates according to delta response.
        elif "delta" in message["type"]:
            for key, value in message["data"].items():
                self.data[topic][key] = value

    def _process_auth_message(self, message):
        # If we get successful futures auth, notify user
        if message.get("success") is True:
            logger.debug(f"Authorization for {self.ws_name} successful.")
            self.auth = True
        # If we get unsuccessful auth, notify user.
        elif message.get("success") is False or message.get("type") == "error":
            raise Exception(
                f"Authorization for {self.ws_name} failed. Please check your "
                f"API keys and resync your system time. Raw error: {message}"
            )

    def _process_subscription_message(self, message):
        if message.get("req_id"):
            sub = self.subscriptions[message["req_id"]]
        else:
            # if req_id is not supported, guess that the last subscription
            # sent was successful
            sub = json.loads(list(self.subscriptions.items())[0][1])["args"][0]

        # If we get successful futures subscription, notify user
        if message.get("success") is True:
            logger.debug(f"Subscription to {sub} successful.")
        # Futures subscription fail
        elif message.get("success") is False:
            response = message["ret_msg"]
            logger.error(f"Couldn't subscribe to topic.Error: {response}.")
            self._pop_callback(sub[0])

    def _process_normal_message(self, message):
        topic = message["topic"]
        if "orderbook" in topic:
            self._process_delta_orderbook(message, topic)
            callback_data = copy.deepcopy(message)
            callback_data["type"] = "snapshot"
            callback_data["data"] = self.data[topic]
        elif "tickers" in topic:
            self._process_delta_ticker(message, topic)
            callback_data = copy.deepcopy(message)
            callback_data["type"] = "snapshot"
            callback_data["data"] = self.data[topic]
        else:
            callback_data = message
        callback_function = self._get_callback(topic)
        callback_function(callback_data)

    def _handle_incoming_message(self, message):
        def is_auth_message():
            if message.get("op") == "auth" or message.get("type") == "AUTH_RESP":
                return True
            else:
                return False

        def is_subscription_message():
            if message.get("op") == "subscribe" or message.get("type") == "COMMAND_RESP":
                return True
            else:
                return False

        if is_auth_message():
            self._process_auth_message(message)
        elif is_subscription_message():
            self._process_subscription_message(message)
        else:
            self._process_normal_message(message)

    def _check_callback_directory(self, topics):
        for topic in topics:
            if topic in self.callback_directory:
                raise Exception(f"You have already subscribed to this topic: {topic}")

    def _set_callback(self, topic, callback_function):
        self.callback_directory[topic] = callback_function

    def _get_callback(self, topic):
        return self.callback_directory[topic]

    def _pop_callback(self, topic):
        self.callback_directory.pop(topic)


class _V5TradeWebSocketManager(_WebSocketManager):
    def __init__(self, recv_window, referral_id, **kwargs):
        super().__init__(self._handle_incoming_message, WSS_NAME, **kwargs)
        self.recv_window = recv_window
        self.referral_id = referral_id
        self._connect(TRADE_WSS)

    def _process_auth_message(self, message):
        # If we get successful auth, notify user
        if message.get("retCode") == 0:
            logger.debug(f"Authorization for {self.ws_name} successful.")
            self.auth = True
        # If we get unsuccessful auth, notify user.
        else:
            raise Exception(
                f"Authorization for {self.ws_name} failed. Please check your "
                f"API keys and resync your system time. Raw error: {message}"
            )

    def _process_error_message(self, message):
        logger.error(
            f"WebSocket request {message['reqId']} hit an error. Enabling "
            f"traceLogging to reproduce the issue. Raw error: {message}"
        )
        self._pop_callback(message["reqId"])

    def _handle_incoming_message(self, message):
        def is_auth_message():
            if message.get("op") == "auth":
                return True
            else:
                return False

        def is_error_message():
            if message.get("retCode") != 0:
                return True
            else:
                return False

        if is_auth_message():
            self._process_auth_message(message)
        elif is_error_message():
            self._process_error_message(message)
        else:
            callback_function = self._pop_callback(message["reqId"])
            callback_function(message)

    def _set_callback(self, topic, callback_function):
        self.callback_directory[topic] = callback_function

    def _pop_callback(self, topic):
        return self.callback_directory.pop(topic)

    def _send_order_operation(self, operation, callback, request):
        request_id = str(uuid.uuid4())

        message = {
            "reqId": request_id,
            "header": {
                "X-BAPI-TIMESTAMP": generate_timestamp(),
            },
            "op": operation,
            "args": [request],
        }

        if self.recv_window:
            message["header"]["X-BAPI-RECV-WINDOW"] = self.recv_window
        if self.referral_id:
            message["header"]["Referer"] = self.referral_id

        self.ws.send(json.dumps(message))
        self._set_callback(request_id, callback)


class WebSocket(_V5WebSocketManager):
    def _validate_public_topic(self):
        if "/v5/public" not in self.WS_URL:
            raise TopicMismatchError("Requested topic does not match channel_type")

    def _validate_private_topic(self):
        if not self.WS_URL.endswith("/private"):
            raise TopicMismatchError("Requested topic does not match channel_type")

    def __init__(
        self,
        channel_type: str,
        **kwargs,
    ):
        super().__init__(WSS_NAME, **kwargs)
        if channel_type not in AVAILABLE_CHANNEL_TYPES:
            raise InvalidChannelTypeError(f"Channel type is not correct. Available: {AVAILABLE_CHANNEL_TYPES}")

        if channel_type == "private":
            self.WS_URL = PRIVATE_WSS
        else:
            self.WS_URL = PUBLIC_WSS.replace("{CHANNEL_TYPE}", channel_type)
            # Do not pass keys and attempt authentication on a public connection
            self.api_key = None
            self.api_secret = None

        if (self.api_key is None or self.api_secret is None) and channel_type == "private":
            raise UnauthorizedExceptionError(
                "API_KEY or API_SECRET is not set. They both are needed in order to access private topics"
            )

        self._connect(self.WS_URL)

    # Private topics

    def position_stream(self, callback):
        """Subscribe to the position stream to see changes to your position data in real-time.

        Push frequency: real-time

        Additional information:
            https://bybit-exchange.github.io/docs/v5/websocket/private/position
        """
        self._validate_private_topic()
        topic = "position"
        self.subscribe(topic, callback)

    def order_stream(self, callback):
        """Subscribe to the order stream to see changes to your orders in real-time.

        Push frequency: real-time

        Additional information:
            https://bybit-exchange.github.io/docs/v5/websocket/private/order
        """
        self._validate_private_topic()
        topic = "order"
        self.subscribe(topic, callback)

    def execution_stream(self, callback):
        """Subscribe to the execution stream to see your executions in real-time.

        Push frequency: real-time

        Additional information:
            https://bybit-exchange.github.io/docs/v5/websocket/private/execution
        """
        self._validate_private_topic()
        topic = "execution"
        self.subscribe(topic, callback)

    def fast_execution_stream(self, callback, categorised_topic=""):
        """Fast execution stream significantly reduces data latency compared
        original "execution" stream. However, it pushes limited execution type
        of trades, and fewer data fields.
        Use categorised_topic as a filter for a certain `category`. See docs.

        Push frequency: real-time

        Additional information:
            https://bybit-exchange.github.io/docs/v5/websocket/private/fast-execution
        """
        self._validate_private_topic()
        topic = "execution.fast"
        if categorised_topic:
            topic += "." + categorised_topic
        self.subscribe(topic, callback)

    def wallet_stream(self, callback):
        """Subscribe to the wallet stream to see changes to your wallet in real-time.

        Push frequency: real-time

        Additional information:
            https://bybit-exchange.github.io/docs/v5/websocket/private/wallet
        """
        self._validate_private_topic()
        topic = "wallet"
        self.subscribe(topic, callback)

    def greek_stream(self, callback):
        """Subscribe to the greeks stream to see changes to your greeks data in real-time. option only.

        Push frequency: real-time

        Additional information:
            https://bybit-exchange.github.io/docs/v5/websocket/private/greek
        """
        self._validate_private_topic()
        topic = "greeks"
        self.subscribe(topic, callback)

    def spread_order_stream(self, callback):
        """Subscribe to the spread trading order stream to see changes to your orders in real-time.

        Push frequency: real-time

        Additional information:
            https://bybit-exchange.github.io/docs/v5/spread/websocket/private/order
        """
        self._validate_private_topic()
        topic = "spread.order"
        self.subscribe(topic, callback)

    def spread_execution_stream(self, callback):
        """Subscribe to the spread trading execution stream to see your executions in real-time.

        Push frequency: real-time

        Additional information:
            https://bybit-exchange.github.io/docs/v5/spread/websocket/private/execution
        """
        self._validate_private_topic()
        topic = "spread.execution"
        self.subscribe(topic, callback)

    # Public topics

    def orderbook_stream(self, depth: int, symbol: str | list, callback):
        """Subscribe to the orderbook stream. Supports different depths.

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

        Required args:
            symbol (string/list): Symbol name(s)
            depth (int): Orderbook depth

        Additional information:
            https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook
        """
        self._validate_public_topic()
        topic = f"orderbook.{depth}." + "{symbol}"
        self.subscribe(topic, callback, symbol)

    def trade_stream(self, symbol: str | list, callback):
        """
        Subscribe to the recent trades stream.
        After subscription, you will be pushed trade messages in real-time.

        Push frequency: real-time

        Required args:
            symbol (string/list): Symbol name(s)

         Additional information:
            https://bybit-exchange.github.io/docs/v5/websocket/public/trade
        """
        self._validate_public_topic()
        topic = "publicTrade." + "{symbol}"
        self.subscribe(topic, callback, symbol)

    def ticker_stream(self, symbol: str | list, callback):
        """Subscribe to the ticker stream.

        Push frequency: 100ms

        Required args:
            symbol (string/list): Symbol name(s)

         Additional information:
            https://bybit-exchange.github.io/docs/v5/websocket/public/ticker
        """
        self._validate_public_topic()
        topic = "tickers.{symbol}"
        self.subscribe(topic, callback, symbol)

    def kline_stream(self, interval: int, symbol: str | list, callback):
        """Subscribe to the klines stream.

        Push frequency: 1-60s

        Required args:
            symbol (string/list): Symbol name(s)
            interval (int): Kline interval

         Additional information:
            https://bybit-exchange.github.io/docs/v5/websocket/public/kline
        """
        self._validate_public_topic()
        topic = f"kline.{interval}." + "{symbol}"
        self.subscribe(topic, callback, symbol)

    def liquidation_stream(self, symbol: str | list, callback):
        """
        Pushes at most one order per second per symbol.
        As such, this feed does not push all liquidations that occur on Bybit.

        Push frequency: 1s

        Required args:
            symbol (string/list): Symbol name(s)

         Additional information:
            https://bybit-exchange.github.io/docs/v5/websocket/public/liquidation
        """
        logger.warning("liquidation_stream() is deprecated. Please use all_liquidation_stream().")
        self._validate_public_topic()
        topic = "liquidation.{symbol}"
        self.subscribe(topic, callback, symbol)

    def all_liquidation_stream(self, symbol: str | list, callback):
        """Subscribe to the liquidation stream, push all liquidations that
        occur on Bybit.

        Push frequency: 500ms

        Required args:
            symbol (string/list): Symbol name(s)

         Additional information:
            https://bybit-exchange.github.io/docs/v5/websocket/public/all-liquidation
        """
        self._validate_public_topic()
        topic = "allLiquidation.{symbol}"
        self.subscribe(topic, callback, symbol)

    def lt_kline_stream(self, interval: int, symbol: str | list, callback):
        """Subscribe to the leveraged token kline stream.

        Push frequency: 1-60s

        Required args:
            symbol (string/list): Symbol name(s)
            interval (int): Leveraged token Kline stream interval

         Additional information:
            https://bybit-exchange.github.io/docs/v5/websocket/public/etp-kline
        """
        self._validate_public_topic()
        topic = f"kline_lt.{interval}." + "{symbol}"
        self.subscribe(topic, callback, symbol)

    def lt_ticker_stream(self, symbol: str | list, callback):
        """Subscribe to the leveraged token ticker stream.

        Push frequency: 300ms

        Required args:
            symbol (string/list): Symbol name(s)

         Additional information:
            https://bybit-exchange.github.io/docs/v5/websocket/public/etp-ticker
        """
        self._validate_public_topic()
        topic = "tickers_lt.{symbol}"
        self.subscribe(topic, callback, symbol)

    def lt_nav_stream(self, symbol: str | list, callback):
        """Subscribe to the leveraged token nav stream.

        Push frequency: 300ms

        Required args:
            symbol (string/list): Symbol name(s)

         Additional information:
            https://bybit-exchange.github.io/docs/v5/websocket/public/etp-nav
        """
        self._validate_public_topic()
        topic = "lt.{symbol}"
        self.subscribe(topic, callback, symbol)


class WebSocketTrading(_V5TradeWebSocketManager):
    def __init__(self, recv_window=0, referral_id="", **kwargs):
        super().__init__(recv_window, referral_id, **kwargs)

    def place_order(self, callback, **kwargs):
        operation = "order.create"
        self._send_order_operation(operation, callback, kwargs)

    def amend_order(self, callback, **kwargs):
        operation = "order.amend"
        self._send_order_operation(operation, callback, kwargs)

    def cancel_order(self, callback, **kwargs):
        operation = "order.cancel"
        self._send_order_operation(operation, callback, kwargs)

    def place_batch_order(self, callback, **kwargs):
        operation = "order.create-batch"
        self._send_order_operation(operation, callback, kwargs)

    def amend_batch_order(self, callback, **kwargs):
        operation = "order.amend-batch"
        self._send_order_operation(operation, callback, kwargs)

    def cancel_batch_order(self, callback, **kwargs):
        operation = "order.cancel-batch"
        self._send_order_operation(operation, callback, kwargs)
