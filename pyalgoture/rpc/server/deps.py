from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends

from ..rpc import RPC, RPCException
from .server import ApiServer


def get_rpc_optional() -> RPC | None:
    if ApiServer._has_rpc:
        return ApiServer._rpc
    return None


async def get_rpc() -> AsyncIterator[RPC]:
    _rpc = get_rpc_optional()
    if _rpc:
        yield _rpc
    else:
        raise RPCException("Bot is not in the correct state")


def get_config() -> dict[str, Any]:
    return ApiServer._config


def get_api_config() -> dict[str, Any]:
    return ApiServer._config["api_server"]
