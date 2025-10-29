from datetime import datetime

import pytz
from dateutil.parser import parse
from pytz import BaseTzInfo
from tzlocal import get_localzone

LOCAL_TZ = pytz.timezone(str(get_localzone()))
CHINA_TZ = pytz.timezone("Asia/Hong_Kong")
# UTC_TZ = pytz.utc
UTC_TZ = pytz.timezone("Etc/UTC")


class TimezoneManager:
    tz: BaseTzInfo | None = None

    def __init__(self) -> None:
        self.tz = LOCAL_TZ

    def set_tz(self, tz: BaseTzInfo) -> None:
        self.tz = tz


tz_manager = TimezoneManager()

# add_default_tz = lambda x, tzinfo=CHINA_TZ: x.replace(tzinfo=x.tzinfo or tzinfo)  # allow datetime default timezone
add_default_tz = lambda x, tzinfo=CHINA_TZ: tzinfo.localize(  # noqa
    x.replace(tzinfo=None)
)  #  LOCAL_TZ force to convert datetime timezone


def to_timestamp(dt: datetime) -> int:
    return int(dt.astimezone(UTC_TZ).timestamp() * 1000)


def is_first_bday_of_mth(d: datetime) -> bool:
    weekday = d.weekday()
    day = d.day

    if (weekday == 0 and day in (1, 2, 3)) or (weekday in (1, 2, 3, 4) and day == 1):
        return True
    else:
        return False


def time2sec(minutes: int, hours: int, days: int, weeks: int) -> int:
    s = 0
    if minutes:
        s += minutes * 60
    if hours:
        s += hours * 60 * 60
    if days:
        s += days * 24 * 60 * 60
    if weeks:
        s += weeks * 7 * 24 * 60 * 60
    return s


def parse2datetime(dt: datetime | str) -> datetime:
    if isinstance(dt, str):
        dt = parse(dt)
    return dt


def find_nearest_datetime(dt_list: list[datetime], target_dt: datetime) -> datetime | None:
    """
    Given a list of datetime objects (dt_list) and a target datetime object (target_dt),
    this function finds the datetime object from the list that is closest in time to the target
    datetime object and returns it.

    %%timeit
    find_nearest_datetime(target_datetime, datetime_list)
    >>> 936 ns ± 67.2 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    """
    nearest_dt = None
    min_diff = None

    for dt in dt_list:
        diff = abs(dt - target_dt)
        if min_diff is None or diff < min_diff:
            nearest_dt = dt
            min_diff = diff

    return nearest_dt


def find_closest_datetime(target_datetime: datetime, datetime_list: list[datetime]) -> datetime:
    """
    Given a list of datetime objects (dt_list) and a target datetime object (target_dt),
    this function finds the datetime object from the list that is closest in time to the target
    datetime object and returns it.

    %%timeit
    find_closest_datetime(target_datetime, datetime_list)
    >>> 1.31 µs ± 31.4 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    """

    return min(datetime_list, key=lambda dt: abs(dt - target_datetime))


def convert_timezone(dt: datetime | None = None, old_tz: str | None = None) -> datetime:
    """
    convert timezone to hk timezone\n
    return datetime object\n
    example:\n
    convert_timezone('2021-04-16 09:38:40', 'GMT')\n
    convert_timezone('2021-04-16 09:38:40', 'Asia/Hong_Kong')
    """
    if not dt:
        dt = datetime.now()
    if old_tz:
        return pytz.timezone(old_tz).localize(parse(dt) if isinstance(dt, str) else dt).astimezone(CHINA_TZ)
    else:
        return (parse(dt) if isinstance(dt, str) else dt).astimezone(CHINA_TZ)


DATETIME_PRINT_FORMAT = "%Y-%m-%d %H:%M:%S"


def format_date(date: datetime | None) -> str:
    """
    Return a formatted date string.
    Returns an empty string if date is None.
    :param date: datetime to format
    """
    if date:
        return date.strftime(DATETIME_PRINT_FORMAT)
    return ""


def parse_timeframe(timeframe: str) -> int:
    amount = int(timeframe[0:-1])
    unit = timeframe[-1]
    if "y" == unit:
        scale = 60 * 60 * 24 * 365
    elif "M" == unit:
        scale = 60 * 60 * 24 * 30
    elif "w" == unit:
        scale = 60 * 60 * 24 * 7
    elif "d" == unit:
        scale = 60 * 60 * 24
    elif "h" == unit:
        scale = 60 * 60
    elif "m" == unit:
        scale = 60
    elif "s" == unit:
        scale = 1
    else:
        raise Exception(f"timeframe unit {unit} is not supported")
    return amount * scale


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Translates the timeframe interval value written in the human readable
    form ('1m', '5m', '1h', '1d', '1w', etc.) to the number
    of seconds for one timeframe interval.
    """
    return parse_timeframe(timeframe)


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Same as timeframe_to_seconds, but returns minutes.
    """
    return parse_timeframe(timeframe) // 60


def timeframe_to_msecs(timeframe: str) -> int:
    """
    Same as timeframe_to_seconds, but returns milliseconds.
    """
    return parse_timeframe(timeframe) * 1000
