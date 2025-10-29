from decimal import Decimal
from math import ceil, floor
from typing import Any

import numpy as np


def almost_equal(a: Any, b: Any) -> bool:
    try:
        return bool(np.isclose(a, b, rtol=1.0e-8))
    except TypeError:
        return bool(a == b)


def is_integer_num(n: Any) -> bool:
    """
    Check if the float is integer number
    """
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False


def prec_round(value: float, target: float) -> float:
    """
    Round to specified precision
    """
    # from decimal import Decimal, ROUND_UP, getcontext, ROUND_HALF_UP
    # # return Decimal(Decimal('1.45').quantize(Decimal('.1'), rounding=ROUND_HALF_UP))
    # nnum = Decimal(str(num))
    # getcontext().rounding = ROUND_HALF_UP
    # return float(round(nnum, d))
    k = 1 / (10 ** (target + 1))
    return round(value + k, int(target))


def round_to(value: float, target: float, return_decimal: bool = False) -> float | Decimal:
    """
    Round price to price tick value.
    """
    value_decimal: Decimal = Decimal(str(value))
    target_decimal: Decimal = Decimal(str(target))
    if return_decimal:
        rounded: Decimal = int(round(value_decimal / target_decimal)) * target_decimal
    else:
        rounded_float: float = float(int(round(value_decimal / target_decimal)) * target_decimal)
        return rounded_float
    return rounded


def floor_to(value: float, target: float) -> float:
    """
    Similar to math.floor function, but to target float number.
    """
    value_decimal: Decimal = Decimal(str(value))
    target_decimal: Decimal = Decimal(str(target))
    result: float = float(int(floor(value_decimal / target_decimal)) * target_decimal)
    return result


def ceil_to(value: float, target: float) -> float:
    """
    Similar to math.ceil function, but to target float number.
    """
    value_decimal: Decimal = Decimal(str(value))
    target_decimal: Decimal = Decimal(str(target))
    result: float = float(int(ceil(value_decimal / target_decimal)) * target_decimal)
    return result


def get_digits(value: float) -> int:
    """
    Get number of digits after decimal point.
    """
    value_str: str = str(value)

    if "e-" in value_str:
        _, buf = value_str.split("e-")
        return int(buf)
    elif "." in value_str:
        _, buf = value_str.split(".")
        return len(buf)
    else:
        return 0


def to_decimal(val: Any) -> Decimal:
    return Decimal(str(val))


def scale_number(number: float, from_min: float, from_max: float, to_min: float, to_max: float) -> float:
    # Calculate the range differences
    from_range = from_max - from_min
    to_range = to_max - to_min

    # Scale the number
    scaled_number = (((number - from_min) * to_range) / from_range) + to_min

    return scaled_number
