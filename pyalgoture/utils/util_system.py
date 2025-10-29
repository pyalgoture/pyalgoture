import gc
import platform
import sys
import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any

from .logger import get_logger

logger = get_logger()


def deprecated(msg: str | None = None, stacklevel: int = 2) -> Callable:
    """
    Used to mark a function as deprecated.
    Parameters
    ----------
    msg : str
        The message to display in the deprecation warning.
    stacklevel : int
        How far up the stack the warning needs to go, before
        showing the relevant calling lines.
    Usage
    -----
    @deprecated(msg='function_a is deprecated! Use function_b instead.')
    def function_a(*args, **kwargs):
    """

    def deprecated_dec(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                msg or f"Function {fn.__name__} is deprecated.",
                category=DeprecationWarning,
                stacklevel=stacklevel,
            )
            return fn(*args, **kwargs)

        return wrapper

    return deprecated_dec


def asyncio_setup() -> None:  # pragma: no cover
    # Set eventloop for win32 setups

    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def gc_set_threshold() -> None:
    """
    Reduce number of GC runs to improve performance (explanation video)
    https://www.youtube.com/watch?v=p4Sn6UcFTOU

    """
    if platform.python_implementation() == "CPython":
        # allocs, g1, g2 = gc.get_threshold()
        gc.set_threshold(50_000, 500, 1000)
        logger.debug("Adjusting python allocations to reduce GC runs")
