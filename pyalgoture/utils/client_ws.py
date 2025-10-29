import json
import ssl
import time
from threading import Thread
from typing import Any

import websocket

from .logger import get_logger


class WebsocketClient:
    """
    Websocket API

    After creating the client object, use start() to run worker thread.
    The worker thread connects websocket automatically.

    Use stop to stop threads and disconnect websocket before destroying the client
    object (especially when exiting the programme).

    Default serialization format is json.

    Callbacks to overrides:
    * on_connected
    * on_disconnected
    * on_packet
    * on_error

    If you want to send anything other than JSON, override send_packet.
    """

    def __init__(self, debug: bool = False, logger: Any | None = None, ws_name: str = "WebSocket") -> None:
        """Constructor"""
        self.active: bool = False
        self.host: str = ""

        self.wsapp: websocket.WebSocketApp | None = None
        self.thread: Thread | None = None
        self.wst: Thread | None = None

        self.proxy_host: str = ""
        self.proxy_port: int = 0
        self.header: dict | None = None
        self.ping_interval: int = 0
        self.ping_timeout: int = 0
        self.receive_timeout: int = 0
        self.retries: int = 0
        self.restart_on_error: bool = True

        self.ws_name: str = ws_name
        self.debug: bool = debug
        self.logger = get_logger() if logger is None else logger

        self.attempting_connection = False
        self.health_check_interval: int = 15 * 60  # 15 minutes
        self.health_check_timeout: int = 5
        self.health_check_thread: Thread | None = None
        self.last_message_time: float = time.time()

    def init(
        self,
        host: str,
        proxy_host: str = "",
        proxy_port: int = 0,
        ping_interval: int = 10,
        ping_timeout: int = 10,
        receive_timeout: int = 60,
        retries: int = 10,
        restart_on_error: bool = True,
        header: dict | None = None,
        # debug: bool = False
    ) -> None:
        """
        :param host:
        :param proxy_host:
        :param proxy_port:
        :param header:
        :param ping_interval: unit: seconds, type: int
        """
        self.host = host
        self.ping_interval = ping_interval  # seconds
        self.ping_timeout = ping_timeout
        self.receive_timeout = receive_timeout
        # self.debug = debug
        self.retries = retries
        self.restart_on_error = restart_on_error

        if header:
            self.header = header

        if proxy_host and proxy_port:
            self.proxy_host = proxy_host
            self.proxy_port = proxy_port

        websocket.enableTrace(self.debug)
        websocket.setdefaulttimeout(receive_timeout)

    def start(self) -> None:
        """
        Start the client and on_connected function is called after webscoket
        is connected succesfully.

        Please don't send packet untill on_connected fucntion is called.
        """
        self.active = True
        # self.thread = Thread(target=self.run)
        # self.thread.start()
        self._connect()

    def stop(self) -> None:
        """
        Stop the client.
        """
        self.logger.info(f"{self.ws_name} stopping...")
        if self.wsapp:
            self.wsapp.close()
            while self.wsapp.sock:
                print("stopping.....")
                time.sleep(0.1)

        self.active = False

        if self.health_check_thread and self.health_check_thread.is_alive():
            self.logger.info(f"{self.ws_name} stopping health check thread...")
            self.health_check_thread = None
            self.logger.debug(f"{self.ws_name} health check thread stopped")

        self.wsapp = None
        self.wst = None
        self.logger.info(f"{self.ws_name} stopped...")

    def join(self) -> None:
        """
        Wait till all threads finish.

        This function cannot be called from worker thread or callback function.
        """
        # if self.thread:
        #     self.thread.join()
        pass

    def send_packet(self, packet: dict | str | bytes) -> bool:
        """
        Send a packet (dict data) to server

        override this if you want to send non-json packet
        """
        if not self.wsapp:
            self.logger.warning(f"{self.ws_name} not connected, cannot send packet.")
            return False
        try:
            if isinstance(packet, str | bytes):
                text: str | bytes = packet
            else:
                text = json.dumps(packet)
            self.wsapp.send(text)
            return True
        except Exception as e:
            self.logger.error(f"{self.ws_name} failed to send packet: {e}")
            return False

    def custom_ping(self) -> bytes:
        return b""

    def waitfor_connection(self, timeout: float = 5.0) -> bool:
        """
        Wait for connection to be established

        :param timeout: Maximum time to wait in seconds
        :return: True if connected within timeout, False otherwise
        """
        start_time = time.time()
        while not self.is_connected() and self.active:
            if time.time() - start_time >= timeout:
                return False
            time.sleep(0.1)
        return True
        # time.sleep(1.5)
        # return True

    def is_connected(self) -> bool:
        try:
            if self.wsapp and self.wsapp.sock and self.wsapp.sock.connected:
                return True
            else:
                return False
        except AttributeError:
            return False

    # def run(self) -> None:
    #     """
    #     Keep running till stop is called.
    #     """
    #     def on_open(wsapp: websocket.WebSocket) -> None:
    #         self.logger.info(f"{self.ws_name} connected successfully")
    #         self.on_connected()

    #     def on_close(wsapp: websocket.WebSocket, status_code: int, msg: str) -> None:
    #         self.logger.info(f"{self.ws_name} disconnected - status_code: {status_code} | msg: {msg}")
    #         self.on_disconnected(status_code, msg)

    #     def on_error(wsapp: websocket.WebSocket, e: Exception) -> None:
    #         self.logger.error(f"{self.ws_name} error: {e}")
    #         self.on_error(e)

    #     def on_message(wsapp: websocket.WebSocket, message: str) -> None:
    #         self.on_message(message)

    #     def on_pong(wsapp: websocket.WebSocket, message:str):
    #         self.send_packet(self.custom_ping())

    #     self.wsapp = websocket.WebSocketApp(
    #         url=self.host,
    #         header=self.header,
    #         on_open=on_open,
    #         on_close=on_close,
    #         on_error=on_error,
    #         on_message=on_message,
    #         on_pong=on_pong,
    #     )

    #     proxy_type: str = ""
    #     if self.proxy_host:
    #         proxy_type = "http"

    #     self.wsapp.run_forever(
    #         sslopt={"cert_reqs": ssl.CERT_NONE},
    #         ping_interval=self.ping_interval,
    #         http_proxy_host=self.proxy_host,
    #         http_proxy_port=self.proxy_port,
    #         proxy_type=proxy_type,
    #         reconnect=1
    #     )

    def _connect(self, connection_timeout: int = 10) -> None:
        """
        Open websocket in a thread.
        """

        self.attempting_connection = True

        # Attempt to connect for X seconds.
        retries = self.retries
        if retries == 0:
            infinitely_reconnect = True
        else:
            infinitely_reconnect = False

        while (infinitely_reconnect or retries > 0) and not self.is_connected():
            self.logger.info(f"{self.ws_name} attempting connection...")
            self.wsapp = websocket.WebSocketApp(
                url=self.host,
                on_message=lambda ws, msg: self.on_message(msg),
                on_close=lambda ws, *args: self.on_close(),
                on_open=lambda ws, *args: self.on_open(),
                on_error=lambda ws, err: self.on_error(err),
                on_pong=lambda ws, *args: self.on_pong(),
            )

            proxy_type: str = ""
            if self.proxy_host:
                proxy_type = "http"

            # Setup the thread running WebSocketApp.
            self.wst = Thread(
                target=lambda: self.wsapp.run_forever(
                    sslopt={"cert_reqs": ssl.CERT_NONE},
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                    http_proxy_host=self.proxy_host,
                    http_proxy_port=self.proxy_port,
                    proxy_type=proxy_type,
                )
                if self.wsapp
                else None
            )

            # Configure as daemon; start.
            self.wst.daemon = True
            self.wst.start()

            # Wait for connection to be established with timeout
            start_time = time.time()
            while self.wst.is_alive() and (time.time() - start_time) < connection_timeout:
                if self.wsapp and self.wsapp.sock and self.is_connected():
                    time.sleep(0.5)
                    break
                time.sleep(0.1)

            # If connection was not successful, raise error.
            retries -= 1
            if not infinitely_reconnect and retries <= 0:
                self.stop()
                raise websocket.WebSocketTimeoutException(
                    f"{self.ws_name} ({self.host}) connection "
                    f"failed. Too many connection attempts. WebSocket will no "
                    f"longer try to reconnect."
                )

        self.logger.info(f"{self.ws_name} connected")
        self.attempting_connection = False

        if not self.health_check_thread or not self.health_check_thread.is_alive():
            self.logger.info(f"{self.ws_name} starting health check thread...")
            self.health_check_thread = Thread(target=self._run_health_check, daemon=True)
            self.health_check_thread.start()
            self.logger.debug(f"{self.ws_name} health check thread started")

    def on_open(self) -> None:
        self.logger.info(f"{self.ws_name} connected successfully")
        self.on_connected()

    def on_close(self) -> None:
        self.logger.info(f"{self.ws_name} disconnected")
        self.on_disconnected()

    def on_pong(self) -> None:
        """
        Callback when websocket receives pong.
        """
        if self.debug:
            self.logger.debug(f"{self.ws_name} received pong and send custom ping")
        self.send_packet(self.custom_ping())

    def on_message(self, message: str) -> None:
        """
        Callback when weboscket app receives new message
        """
        try:
            self.last_message_time = time.time()
            if self.debug:
                self.logger.debug(f"{self.ws_name} received message: {message}")
            if message:
                self.on_packet(json.loads(message))
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"{self.ws_name} failed to parse message as JSON: {e}. Message: {message}")

    def on_connected(self) -> None:
        """
        Callback when websocket is connected successfully.
        """
        pass

    def on_disconnected(self) -> None:
        """
        Callback when websocket connection is closed.
        """
        pass

    def on_packet(self, packet: dict) -> None:
        """
        Callback when receiving data from server.
        """
        pass

    def on_error(self, e: Exception) -> None:
        """
        Callback when exception raised.
        """
        try:
            self.logger.error("-" * 6 + f" {self.ws_name} on error " + "-" * 6 + f"\n[{type(e).__name__}]: {str(e)}")

            if type(e).__name__ not in [
                "WebSocketConnectionClosedException",
                "ConnectionResetError",
                "WebSocketTimeoutException",
            ]:
                # Raises errors not related to websocket disconnection.
                self.stop()
                raise e

            # Stop current connection and wait for cleanup
            if self.active:
                self.stop()

            # Add a small delay before reconnecting to avoid rapid reconnection attempts
            time.sleep(2.5)

            try_to_reconnect = self.restart_on_error and not self.attempting_connection
            self.logger.debug(
                f"Try to reconnect:{try_to_reconnect} [Restart on error: {self.restart_on_error} & Not attempting connection: {not self.attempting_connection}]"
            )
            # Reconnect.
            if try_to_reconnect:
                self.logger.info(f"{self.ws_name} attempting to reconnect after error...")
                self.start()

        except Exception as e:
            self.logger.exception(e)

    def _run_health_check(self) -> None:
        """
        Periodically check connection health and data reception
        """
        self.logger.info(
            f"{self.ws_name} health check started - interval: {self.health_check_interval}s, timeout: {self.health_check_timeout}s"
        )

        while self.active and self.health_check_thread:
            try:
                time.sleep(self.health_check_interval)

                if not self.active:
                    self.logger.debug(f"{self.ws_name} health check stopping - client not active")
                    break

                current_time = time.time()
                time_since_last_message = current_time - self.last_message_time

                self.logger.debug(
                    f"{self.ws_name} health check running - last message: {time_since_last_message:.1f}s ago"
                )

                if not self.is_connected():
                    self.logger.warning(f"{self.ws_name} connection lost - reconnecting...")
                    if self.restart_on_error:
                        self._reconnect()
                elif time_since_last_message > (self.health_check_interval + self.health_check_timeout):
                    self.logger.warning(
                        f"{self.ws_name} no data received for {time_since_last_message:.1f}s - checking connection..."
                    )
                    if self.wsapp and hasattr(self.wsapp, "ping"):
                        try:
                            self.logger.debug(f"{self.ws_name} sending ping to check connection...")
                            self.wsapp.ping()
                            self.logger.debug(f"{self.ws_name} ping sent successfully")
                        except Exception as e:
                            self.logger.error(f"{self.ws_name} ping failed: {e} - reconnecting...")
                            if self.restart_on_error:
                                self._reconnect()
                else:
                    self.logger.debug(f"{self.ws_name} health check passed - connection healthy")

            except Exception as e:
                self.logger.error(f"{self.ws_name} health check error: {e}")
                if not self.active:
                    self.logger.debug(f"{self.ws_name} health check stopping due to client inactive")
                    break

        self.logger.info(f"{self.ws_name} health check thread exiting")

    def _reconnect(self) -> None:
        """
        Reconnect the websocket
        """
        try:
            self.logger.info(f"{self.ws_name} attempting reconnection...")
            if self.wsapp:
                self.logger.debug(f"{self.ws_name} closing existing connection...")
                self.wsapp.close()
            time.sleep(1)
            self.logger.debug(f"{self.ws_name} initiating new connection...")
            self._connect()
            self.logger.info(f"{self.ws_name} reconnection completed successfully")
        except Exception as e:
            self.logger.error(f"{self.ws_name} reconnection failed: {e}")
