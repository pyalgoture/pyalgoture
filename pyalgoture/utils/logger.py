import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import BufferingHandler
from typing import Any

import pytz
from cachetools import TTLCache, cached
from loguru import logger
from loguru._logger import Logger

CHINA_TZ = pytz.timezone("Asia/Hong_Kong")

"""
https://betterstack.com/community/guides/logging/loguru/
https://loguru.readthedocs.io/en/stable/api/logger.html#file
"""


class CustomBufferingHandler(BufferingHandler):
    def flush(self) -> None:
        """
        Override Flush behaviour - we keep half of the configured capacity
        otherwise, we have moments with "empty" logs.
        """
        self.acquire()
        try:
            # Keep half of the records in buffer.
            records_to_keep = -int(self.capacity / 2)
            self.buffer = self.buffer[records_to_keep:]
        finally:
            self.release()


# Initialize bufferhandler - will be used for /log endpoints
bufferHandler = CustomBufferingHandler(1000)


@dataclass
class LogData:
    """
    Log data is used for recording log messages on GUI or in log files.
    """

    tick: datetime
    content: str
    level: str = "INFO"

    def __repr__(self) -> str:
        return f"<LogData tick:{self.tick}; level: {self.level}; content:{self.content}>"


class LevelFilter:
    def __init__(self, level: str) -> None:
        self.level = level

    def __call__(self, record: Any) -> bool:
        levelno = logger.level(self.level).no
        return bool(record["level"].no >= levelno)


def get_logger(
    name: str = "",
    log_path: str = "",
    console_log: bool = True,
    use_default_dt: bool = True,
    default_level: str = "DEBUG",
    mode: str = "w",
    custom_sink: Any = None,
    tz: Any = CHINA_TZ,
    rotation: str = "monthly",
    retention: int = 24,
    log_once_ttl: int | None = None,
) -> Any:
    """
    mode: str ['w', 'a']
    default is w, which with overwrite the log file.

    trace color: cyan
    debug color: blue
    info color: green
    notice color: magenta
    warning color: shit yellow
    error color: red
    critical color: red background
    """

    def _set_datetime(record: Any) -> None:
        dt = datetime.now(tz)  # .replace(microsecond=0)
        record["extra"]["datetime"] = dt
        if "tick" not in record["extra"]:
            record["extra"]["tick"] = dt

    level_filter = LevelFilter(default_level.upper())
    logger.remove()  # Remove the default logger configuration
    new_logger = deepcopy(logger)
    new_logger.configure(patcher=_set_datetime)
    if name:
        new_logger = new_logger.bind(name=name)

    # COLOR: magenta, black, white, cyan, blue, green, red, yellow
    time_field = "datetime" if use_default_dt else "tick"
    _format = f"<cyan>{{extra[{time_field}]}}</cyan> <level>[{{level}}]</level> {{message}}"
    if name:
        _format = f"<cyan>{{extra[{time_field}]}}</cyan> <level>[{{level}}]</level> <cyan><{{extra[name]}}></cyan> {{message}}"

    if console_log:
        new_logger.add(
            sink=sys.stdout,
            format=_format,
            level="TRACE",
            backtrace=True,
            filter=level_filter,
        )
    if log_path:
        new_logger.add(
            sink=log_path,
            format=_format,
            level="TRACE",
            backtrace=True,
            mode=mode,
            filter=level_filter,
            rotation=rotation,
            retention=retention,
        )
    if custom_sink:
        new_logger.add(
            sink=custom_sink,
            format=_format,
            level="TRACE",
            backtrace=True,
            filter=level_filter,
        )

    new_logger.add(
        sink=bufferHandler,
        format=_format,
        level="TRACE",
        backtrace=True,
        filter=level_filter,
    )
    new_logger.filter = level_filter  # type: ignore
    new_logger.use_default_dt = use_default_dt  # type: ignore

    new_logger.level("INFO", color="<green><bold>")
    new_logger.level("NOTICE", no=25, color="<magenta><bold>")

    # Create a notice method on the logger
    def notice_method(self: Any, message: str, *args: Any, **kwargs: Any) -> None:
        self.log("NOTICE", message, *args, **kwargs)

    new_logger.notice = notice_method.__get__(new_logger)  # type: ignore # Bind the method to the logger

    # Create log_once method
    if log_once_ttl:
        log_cache: TTLCache[str, None] = TTLCache(
            maxsize=1024, ttl=log_once_ttl
        )  # add TTLCache for log_once function, default 1 hour expiration

        def log_once_method(self: Any, message: str, level: str = "INFO", *args: Any, **kwargs: Any) -> None:
            @cached(cache=log_cache)
            def _log_once(msg: str) -> None:
                getattr(self, level.lower())(msg, *args, **kwargs)  # call the corresponding log method based on level

            self.trace(message)  # always log at DEBUG level
            _log_once(message)  # call the cached log method

        new_logger.log_once = log_once_method.__get__(new_logger)  # type: ignore # Bind the method to the logger

    return new_logger


def change_logger_level(logger: Any, new_level: str) -> Any:
    logger.filter.level = new_level.upper()
    return logger
