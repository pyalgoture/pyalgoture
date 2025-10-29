import sys
import time
import traceback
from collections.abc import Callable
from datetime import datetime
from multiprocessing.dummy import Pool
from multiprocessing.pool import ThreadPool
from queue import Empty, Queue
from types import TracebackType
from typing import Any

import requests

from .logger import get_logger

CALLBACK_TYPE = Callable[[dict | None, "Request"], Any]
ON_FAILED_TYPE = Callable[[int, "Request"], Any]
ON_ERROR_TYPE = Callable[[type[BaseException], BaseException, TracebackType, "Request"], Any]

Response = requests.Response


class Request:
    """
    Request object

    method: API request method (GET, POST, PUT, DELETE, QUERY)
    path: API request path (without base URL)
    callback: Callback function on request success
    params: Dictionary of request parameters
    data: Request body data, dictionaries will be automatically converted to JSON
    headers: Dictionary of request headers
    on_failed: Callback function on request failure
    on_error: Callback function on request exception
    extra: Any additional data (for use in callbacks)
    """

    def __init__(
        self,
        method: str,
        path: str,
        params: dict | None,
        data: dict | str | None,
        headers: dict | None,
        callback: CALLBACK_TYPE | None = None,
        on_failed: ON_FAILED_TYPE | None = None,
        on_error: ON_ERROR_TYPE | None = None,
        extra: Any | None = None,
    ) -> None:
        """Initialize a request object"""
        self.method: str = method
        self.path: str = path
        self.callback: CALLBACK_TYPE | None = callback
        self.params: dict | None = params
        self.data: dict | str | None = data
        self.headers: dict | None = headers

        self.on_failed: ON_FAILED_TYPE | None = on_failed
        self.on_error: ON_ERROR_TYPE | None = on_error
        self.extra: Any | None = extra

        self.response: requests.Response | None = None

    def __str__(self) -> str:
        """String representation of the request"""
        if self.response is None:
            status_code = "terminated"
        else:
            status_code = str(self.response.status_code)

        return (
            f"request: {self.method} {self.path}[{status_code}]: \n"
            f"headers: {self.headers}\n"
            f"params: {self.params}\n"
            f"data: {self.data}\n"
            f"response:"
            f"{'' if self.response is None else self.response.text}\n"
        )


class RestClient:
    """
    Client for various REST APIs

    * Override the sign method to implement request signature logic
    * Override the on_failed method to implement standard callback handling for request failures
    * Override the on_error method to implement standard callback handling for request exceptions
    """

    def __init__(
        self,
        url_base: str,
        proxy_host: str = "",
        proxy_port: int = 0,
        debug: bool = False,
        logger: Any | None = None,
    ) -> None:
        """Constructor"""
        self.logger: Any = get_logger() if logger is None else logger
        self._debug = debug

        self._url_base = url_base

        self._proxies: dict | None = None
        if proxy_host and proxy_port:
            proxy: str = f"http://{proxy_host}:{proxy_port}"
            self._proxies = {"http": proxy, "https": proxy}

        self._active: bool = False

        self._queue: Queue = Queue()

    def start(self, n: int = 5) -> None:
        """
        Start the client

        :param n: Number of worker threads
        """
        if self._active:
            return
        self._active = True

        self.pool: ThreadPool = Pool(n)
        self.pool.apply_async(self.run)

    def stop(self) -> None:
        """Stop the client"""
        self._active = False

    def join(self) -> None:
        """Wait for threads to complete"""
        self._queue.join()

    def add_request(
        self,
        method: str,
        path: str,
        callback: CALLBACK_TYPE | None = None,
        params: dict | None = None,
        data: dict | str | None = None,
        headers: dict | None = None,
        on_failed: ON_FAILED_TYPE | None = None,
        on_error: ON_ERROR_TYPE | None = None,
        extra: Any | None = None,
    ) -> Request | Response:
        """
        Add a new request task

        :param method: HTTP method
        :param path: API endpoint path
        :param callback: Callback function for successful responses
        :param params: Query parameters
        :param data: Request body data
        :param headers: HTTP headers
        :param on_failed: Callback for failed requests
        :param on_error: Callback for request exceptions
        :param extra: Additional data to pass to callbacks
        :return: Request object
        """
        if not callback:
            return self.request(method, path, params, data, headers)

        request: Request = Request(
            method,
            path,
            params,
            data,
            headers,
            callback,
            on_failed,
            on_error,
            extra,
        )
        self._queue.put(request)
        return request

    def run(self) -> None:
        """Process tasks in each thread"""
        try:
            session = requests.session()
            while self._active:
                try:
                    request = self._queue.get(timeout=1)
                    try:
                        self.process_request(request, session)
                    finally:
                        self._queue.task_done()
                except Empty:
                    pass
        except Exception:
            exc, value, tb = sys.exc_info()
            if exc and value and tb:
                self.on_error(exc, value, tb, None)

    def sign(self, request: Request) -> Request:
        """
        Signature function (override to implement specific signature logic)

        :param request: Request to sign
        :return: Signed request
        """
        return request

    def on_failed(self, status_code: int, request: Request) -> None:
        """
        Default callback for request failures

        :param status_code: HTTP status code
        :param request: Failed request
        """
        print("RestClient on failed" + "-" * 10)
        print(str(request))

    def on_error(
        self,
        exc: type[BaseException],
        value: BaseException,
        tb: TracebackType,
        request: Request | None,
    ) -> None:
        """
        Default callback for request exceptions

        :param exc: Exception class
        :param value: Exception instance
        :param tb: Traceback object
        :param request: Request that caused the exception
        """
        try:
            print("RestClient on error" + "-" * 10)
            print(self.exception_detail(exc, value, tb, request))
        except Exception:
            traceback.print_exc()

    def exception_detail(
        self,
        exc: type[BaseException],
        value: BaseException,
        tb: TracebackType,
        request: Request | None,
    ) -> str:
        """
        Convert exception information to string

        :param exc: Exception class
        :param value: Exception instance
        :param tb: Traceback object
        :param request: Request that caused the exception
        :return: Formatted exception details
        """
        text = f"[{datetime.now().isoformat()}]: Unhandled RestClient Error:{exc}\n"
        text += f"request:{request}\n"
        text += "Exception trace: \n"
        text += "".join(traceback.format_exception(exc, value, tb))
        return text

    def process_request(self, request: Request, session: requests.Session) -> None:
        """
        Send request to server and process response

        :param request: Request to process
        :param session: Requests session
        """
        try:
            # Sign the request
            request = self.sign(request)

            # Send synchronous request
            response: Response = session.request(
                request.method,
                self._make_full_url(request.path),
                headers=request.headers,
                params=request.params,
                data=request.data,
                proxies=self._proxies,
            )

            # Bind response to request
            request.response = response

            # Parse response data
            status_code = response.status_code

            if status_code // 100 == 2:  # 2xx indicates success
                json_body: dict | None = None

                if status_code != 204:
                    json_body = response.json()

                if request.callback:
                    request.callback(json_body, request)
            else:
                if request.on_failed:
                    request.on_failed(status_code, request)
                else:
                    self.on_failed(status_code, request)
        except Exception:
            # Get exception information
            exc, value, tb = sys.exc_info()

            # Push exception callback
            if exc and value and tb:
                if request.on_error:
                    request.on_error(exc, value, tb, request)
                else:
                    self.on_error(exc, value, tb, request)

    def _make_full_url(self, path: str, cus_url_base: str = "") -> str:
        """
        Combine base URL and path to generate full request URL

        :param path: API endpoint path
        :return: Complete URL
        """
        if cus_url_base:
            return cus_url_base + path
        else:
            return self._url_base + path

    def request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        data: dict | str | None = None,
        headers: dict | None = None,
    ) -> Response:
        """
        Make a synchronous request

        :param method: HTTP method
        :param path: API endpoint path
        :param params: Query parameters
        :param data: Request body data
        :param headers: HTTP headers
        :return: Response object
        """
        # Create request object
        request: Request = Request(method, path, params, data, headers)

        # Sign the request
        request = self.sign(request)

        # Send synchronous request
        response: Response = requests.request(
            request.method,
            self._make_full_url(request.path),
            headers=request.headers,
            params=request.params,
            data=request.data,
            proxies=self._proxies,
        )
        return response
