from collections import defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from .objects import (
    AccountData,
    AssetType,
    Exchange,
    ExecutionType,
    MarginMode,
    OptionType,
    OrderData,
    OrderType,
    PositionData,
    PositionMode,
    PositionSide,
    Side,
    Status,
    TimeInForce,
    TradeData,
)


class AttrDict(dict[str, Any]):
    def __getattr__(self, k: str) -> Any:
        # return self[k]
        return self.get(k)

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v


def custom_dict() -> defaultdict[str, dict[str, Any]]:
    return defaultdict(dict)


def set_default(obj: Any) -> Any:
    # print('=====>' ,obj, type(obj),'<====')
    # if isinstance(obj, list):
    #     for i in obj:
    #         i = set_default(i)
    if (
        isinstance(obj, pd.Timestamp)
        or isinstance(obj, date)
        or isinstance(obj, datetime)
        or isinstance(obj, timedelta)
    ):
        return str(obj)
    elif isinstance(obj, Side):
        return obj.name
    elif isinstance(obj, TimeInForce):
        return obj.name
    elif isinstance(obj, PositionSide):
        return obj.name
    elif isinstance(obj, PositionMode):
        return obj.name
    elif isinstance(obj, MarginMode):
        return obj.name
    elif isinstance(obj, OrderType):
        return obj.name
    elif isinstance(obj, ExecutionType):
        return obj.name
    elif isinstance(obj, Status):
        return obj.name
    elif isinstance(obj, Exchange):
        return obj.name
    elif isinstance(obj, AssetType):
        return obj.name
    elif isinstance(obj, OptionType):
        return obj.name
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, OrderData):
        return obj.to_dict()
    elif isinstance(obj, TradeData):
        return obj.to_dict()
    elif isinstance(obj, PositionData):
        return obj.to_dict()
    elif isinstance(obj, AccountData):
        return obj.to_dict()
    elif isinstance(obj, Decimal):
        if obj % 1 > 0:
            return float(obj)
        else:
            return int(obj)
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        # print(f"Unserializable object {obj} of type {type(obj)}")
        # return str(obj)
        raise TypeError(f"Unserializable object {obj} of type {type(obj)}")
