import math
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta
from importlib import import_module
from statistics import geometric_mean

import numpy as np
import pandas as pd
import pytz
from dateutil.parser import parse

from ..datafeed import DataFeed
from ..utils.objects import BarData, TickData

CHINA_TZ = pytz.timezone("Asia/Hong_Kong")

"""
Why does Tradingview show me different information than crypto-signal?
    There are a number of reasons why the information crypto-signal provides could be different from tradingview and the truth is we have no way to be 100% certain of why the differences exist. Below are some things that affect the indicators that may differ between crypto-signal and tradingview.

    tradingview will have more historical data and for some indicators this can make a big difference.

    tradingview uses a rolling 15 minute timeframe which means that the data they are analyzing can be more recent than ours by a factor of minutes or hours depending on what candlestick timeframe you are using.

    tradingview may collect data in a way that means the timeperiods we have may not line up with theirs, which can have an effect on the analysis. This seems unlikely to us, but stranger things have happened.

    So if it doesn't match Tradingview how do you know your information is accurate?
    Underpinning crypto-signal for most of our technical analysis is TA-Lib which is an open source technical analysis project started in 1999. This project has been used in a rather large number of technical analysis projects over the last two decades and is one of the most trusted open source libraries for analyzing candlestick data.


1. time series container of bar data
2. calculating technical indicator value
"""


class Indicator:
    # MA_Type: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3 (Default=SMA), 9 =RMA (custom)
    # https://github.com/Bitvested/ta.py/blob/main/ta_py/ta.py
    # https://github.com/twopirllc/pandas-ta/issues/519

    def __init__(
        self,
        size: int = 100,
        window: int = 1,
        hour_window: bool = False,
        strict_agg: bool = False,
        datafeed: DataFeed | None = None,
        init_data: dict[str, BarData] | None = None,
    ) -> None:
        """
        Args:
            size (int, optional): store size. Defaults to 100.
            window (int, optional): time window. Defaults to 1.
            hour_window (bool, optional): if True, bar will be hour bar, otherwise, it is minute bar. Defaults to False.
            strict_agg (bool, optional): mainly for 4h & 6h bar data, if true, the aggreagted datetime will be same as exchange. Defaults to False.
            datafeed (_type_, optional): if specified, will automatically download the `size` data so that the indicator will be successfully initialized. Defaults to None.

        Raises:
            ImportError: if ta-lib is not installed
        """
        try:
            import talib  # noqa: F401
        except ImportError:
            raise ImportError("Indicator required package 'talib'. pip install ta-lib")

        self.talib = import_module("talib")

        self.count: int = 0
        self.size: int = size
        self.inited: bool = False

        self.open_array: np.ndarray = np.zeros(size)
        self.high_array: np.ndarray = np.zeros(size)
        self.low_array: np.ndarray = np.zeros(size)
        self.close_array: np.ndarray = np.zeros(size)
        self.volume_array: np.ndarray = np.zeros(size)
        self.turnover_array: np.ndarray = np.zeros(size)
        # self.open_interest_array: np.ndarray = np.zeros(size)
        self.datetime_array: np.ndarray = np.empty(size, dtype="datetime64[s]")
        self.bar = None
        self.bars: dict = {}

        self.bar_generator = BarGenerator(
            on_bar=None,
            window=window,
            on_window_bar=None,
            hour_window=hour_window,
            strict_agg=strict_agg,
        )
        if init_data:
            for _, bar in init_data.items():
                self.update_bar(bar=bar)

        if datafeed:
            end = datetime.now() if datafeed.is_live else parse(datafeed.start_date)
            seconds = size * window * (60 if hour_window else 1) * 60
            start = end - timedelta(seconds=seconds)

            # df_interval = datafeed.interval
            # if df_interval == "1m" and not hour_window:
            #     interval = f"{window}m"
            # elif hour_window:
            #     interval = f"{window}h"
            #     if interval == '24h':
            #         interval = '1d'
            # else:
            #     interval = df_interval

            if hour_window:
                interval = f"{window}h"
                if interval == "24h":
                    interval = "1d"
            else:
                interval = f"{window}m"

            hist_data = datafeed.fetch_hist(start=start, end=end, interval=interval, is_store=False, tz=True)
            # print(
            #     f"Successfully fetched {len(hist_data)} bars for {datafeed.symbol} between {start} and {end} on {interval} to initialize indicator"
            # )
            # print(f"[DEBUG - Indicator - __init__] input hist_data:{hist_data}({len(hist_data)}); start={start}, end={end}, interval={interval}; size:{size}; window:{window}; hour_window:{hour_window}; strict_agg:{strict_agg}; datafeed:{datafeed}; ")

            for _, bar in hist_data.items():
                # bar.datetime = bar.datetime.astimezone(tz=CHINA_TZ)
                self.update_bar(bar=bar)

            # print(f"[DEBUG - Indicator - __init__] self.count:{self.count}")

    def update_bar(self, bar: BarData) -> None:
        """
        Update new bar data into array manager.
        """
        # print(f"[DEBUG - Indicator - update_bar] input bar data:{bar}; current bar data:{self.bar}")
        bar.datetime = bar.datetime.replace(second=0, microsecond=0)
        self.bar_generator.update_bar(bar=bar)
        self.bar = self.bar_generator.bar
        is_finished = self.bar_generator.is_finished
        # print(self.bar, '|||', bar)
        agg_bar_dt = self.bar.datetime
        inp_bar_dt = bar.datetime
        # print(f"[Indicator-update_bar] agg_bar.dt == inp_bar.dt:{agg_bar_dt == inp_bar_dt} | agg_bar.dt:{agg_bar_dt}, inp_bar.dt:{inp_bar_dt}, is_finished:{is_finished}")
        is_new: bool = False
        if agg_bar_dt == inp_bar_dt and agg_bar_dt not in self.bars:
            # if is_finished:
            ### new bar - shift last value in array
            self.count += 1
            if not self.inited and self.count >= self.size:
                self.inited = True
            self.open_array[:-1] = self.open_array[1:]
            self.high_array[:-1] = self.high_array[1:]
            self.low_array[:-1] = self.low_array[1:]
            self.close_array[:-1] = self.close_array[1:]
            self.volume_array[:-1] = self.volume_array[1:]
            self.datetime_array[:-1] = self.datetime_array[1:]
            # self.open_interest_array[:-1] = self.open_interest_array[1:]
            self.turnover_array[:-1] = self.turnover_array[1:]
            is_new = True
            # print(f"[Indicator-update_bar] len:{len(self.bars)} ----- bar:{self.bar}")

        ### update last value
        self.open_array[-1] = self.bar.open
        self.high_array[-1] = self.bar.high
        self.low_array[-1] = self.bar.low
        self.close_array[-1] = self.bar.close
        self.volume_array[-1] = self.bar.volume if "volume" in bar else None
        self.datetime_array[-1] = bar.datetime if "datetime" in bar else None
        # self.open_interest_array[-1] = bar.open_interest
        self.turnover_array[-1] = self.bar.turnover if "turnover" in bar else None
        self.bar.is_new = is_new
        self.bar.is_finished = is_finished
        # print(f"self.inited:{self.inited}; dt:{bar.datetime}")

        self.bars[agg_bar_dt] = self.bar

        # print(f">>>>>>>>>>> open:{self.bar.open};high:{self.bar.high};low:{self.bar.low};close:{self.bar.close};datetime:{self.bar.datetime};turnover:{self.bar.turnover} {'' if not is_new else '||| new bar!!!!!!'}")

    @property
    def cum_dataframe(self):
        # print(f"[cum_dataframe] self.bars:{len(self.bars)}")
        return pd.DataFrame.from_records(list(self.bars.values()))

    @property
    def dataframe(self):
        # df= pd.DataFrame(data={
        #     'datetime': self.datetime_array,
        #     "open": self.open,
        #     "high": self.high,
        #     "low": self.low,
        #     "close": self.close,
        #     "volume": self.volume,
        #     "turnover": self.turnover,
        # })
        # df["datetime"] = pd.to_datetime(df["datetime"])
        # df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Hong_Kong")
        # return df
        return self.cum_dataframe.iloc[-self.size :]

    @property
    def open(self) -> np.ndarray:
        """
        Get open price time series.
        """
        return self.open_array

    @property
    def high(self) -> np.ndarray:
        """
        Get high price time series.
        """
        return self.high_array

    @property
    def low(self) -> np.ndarray:
        """
        Get low price time series.
        """
        return self.low_array

    @property
    def close(self) -> np.ndarray:
        """
        Get close price time series.
        """
        return self.close_array

    @property
    def ticks(self) -> np.ndarray:
        """
        Get datetime time series.
        """
        return self.datetime_array

    @property
    def volume(self) -> np.ndarray:
        """
        Get trading volume time series.
        """
        return self.volume_array

    @property
    def turnover(self) -> np.ndarray:
        """
        Get trading turnover time series.
        """
        return self.turnover_array

    # @property
    # def open_interest(self) -> np.ndarray:
    #     """
    #     Get trading volume time series.
    #     """
    #     return self.open_interest_array

    def _get_input(self, default, input):
        return default if input is None else getattr(self, input) if isinstance(input, str) else input

    def _rolling_window(a, window):
        # pd.Series(a).rolling(window).std()
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1])
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def shift(self, n: int, input: np.ndarray, fill_value=np.nan):
        """
        shift data
        """
        # if n >= 0:
        #     return np.concatenate((np.full(n, fill_value), input[:-n]))
        # else:
        #     return np.concatenate((input[-n:], np.full(-n, fill_value)))

        ### faster
        result: np.ndarray = np.empty_like(input)
        if n > 0:
            result[:n] = fill_value
            result[n:] = input[:-n]
        elif n < 0:
            result[n:] = fill_value
            result[:n] = input[-n:]
        else:
            result[:] = input
        return result

    @property
    def zeros(self) -> np.ndarray:
        return np.zeros(self.size)

    # def custom_dep(self, func: Callable, n: int = None, array: bool = False) -> float | np.ndarray:
    #     """
    #     Custom func
    #     """
    #     result: np.ndarray = func(self.open, self.high, self.low, self.close, self.volume, self.turnover)
    #     if array:
    #         return result
    #     return result[-1]

    def custom(self, func: Callable, array: bool = False, *args, **kwargs) -> float | np.ndarray:
        """
        Custom func
        NOTE:
            all the inputs need to be the same shape
            the sequence of inputs need to follow the func params
        """
        # np_func = np.frompyfunc(partial(func, n=1.1), nin=4, nout=1)

        if args:
            inputs = args
        else:
            inputs = kwargs.values()

        np_func = np.frompyfunc(func, nin=len(inputs), nout=1)
        result: np.ndarray = np_func(*inputs).astype("float")
        if array:
            return result
        return result[-1]

    def where(
        self,
        condition,
        true_result: float | np.ndarray,
        false_result: float | np.ndarray,
        array: bool = False,
    ) -> float | np.ndarray:
        """
        like np.where
        """
        true_result: float | np.ndarray = self._get_input(default=None, input=true_result)
        false_result: float | np.ndarray = self._get_input(default=None, input=false_result)
        result: np.ndarray = np.where(condition, true_result, false_result)
        if array:
            return result
        return result[-1]

    def hlc3(self, array: bool = False) -> float | np.ndarray:
        """
        (high + low + close) / 3
        """
        result: np.ndarray = (self.high + self.low + self.close) / 3
        if array:
            return result
        return result[-1]

    def hl2(self, array: bool = False) -> float | np.ndarray:
        """
        (high + low ) / 2
        """
        result: np.ndarray = (self.high + self.low) / 2
        if array:
            return result
        return result[-1]

    def rma(self, n: int, array: bool = False, input: np.ndarray | None = None) -> float | np.ndarray:
        """
        wildeR's Moving Average (RMA)
        """
        # input: np.ndarray = self._get_input(default=self.close, input=input)
        # print(input, type(input))
        # result = [0.0 for i in range(input.size)]  # crate a set same as input
        # alpha = 1 / n

        # for i in range(input.size):
        #     if i < n-1:
        #         result[i] = 0
        #     else:
        #         if result[i-1]:
        #             result[i] = alpha * input[i] + (1 - alpha) * result[i-1]
        #         else:
        #             ma = 0
        #             for i2 in range(i-n,i):  # cal mean
        #                 ma += input[i2+1]
        #             result[i] = ma / n
        # df = pd.DataFrame(data=np.array([4086.29, 4310.01, 4509.08, 4130.37, 3699.99, 3660.02, 4378.48, 4640.0, 5709.99, 5950.02]), columns=['close'])
        df = pd.DataFrame(data=self._get_input(default=self.close, input=input), columns=["close"])
        df["rma"] = df["close"].copy()
        df["rma"].iloc[:n] = df["rma"].rolling(n).mean().iloc[:n]
        df["rma"] = df["rma"].ewm(alpha=(1.0 / n), adjust=False).mean()
        result = df["rma"].to_numpy()
        # print(f"rma result:{result}")
        if array:
            return result
        return result[-1]

    def smma(self, n: int, array: bool = False, input: np.ndarray | None = None) -> float | np.ndarray:
        """
        Smoothed moving average.
        """
        df = pd.DataFrame(data=self._get_input(default=self.close, input=input), columns=["close"])
        result = df["close"].ewm(alpha=1 / n, adjust=False).mean().to_numpy()

        if array:
            return result
        return result[-1]

    def sma(self, n: int, array: bool = False, input: np.ndarray | None = None) -> float | np.ndarray:
        """
        Simple moving average.
        """
        result: np.ndarray = self.talib.SMA(self._get_input(default=self.close, input=input), n)
        if array:
            return result
        return result[-1]

    # def dev(self, n: int, array: bool = False, input: np.ndarray = None) -> float | np.ndarray:
    #     """
    # https://www.tradingview.com/pine-script-reference/v5/#fun_ta{dot}dev
    #     Measure of difference between the series and it's ta.sma
    #     """
    #     result: np.ndarray = self.talib.SMA(self._get_input(default=self.close, input=input), n)
    #     summ       = 0.0
    #     mean       = column.mean()
    #     length     = len(column)
    #     for i in range(0, length):
    #         summ = summ +  abs( column[i] - mean )

    #     ret_val  = summ / length
    #     return ret_val
    #     dev  = hlc3.rolling(length).apply(pine_dev) # Measure of difference between the series and it's ta.sma

    #     if array:
    #         return result
    #     return result[-1]

    def ema(self, n: int, array: bool = False, input: np.ndarray | None = None) -> float | np.ndarray:
        """
        Exponential moving average.
        """
        # result: np.ndarray = self.talib.EMA(self.close, n)
        result: np.ndarray = self.talib.EMA(self._get_input(default=self.close, input=input), n)
        if array:
            return result
        return result[-1]

    def kama(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        KAMA.
        """
        result: np.ndarray = self.talib.KAMA(self.close, n)
        if array:
            return result
        return result[-1]

    def wma(self, n: int, array: bool = False, input: np.ndarray | None = None) -> float | np.ndarray:
        """
        WMA.
        """
        result: np.ndarray = self.talib.WMA(self._get_input(default=self.close, input=input), n)
        if array:
            return result
        return result[-1]

    def hma(self, n: int, array: bool = False, input: np.ndarray | None = None) -> float | np.ndarray:
        """
        HMA - Hull ma.
        HMA= WMA(2*WMA(n/2) - WMA(n)),sqrt(n))
        """
        input = self._get_input(default=self.close, input=input)
        result: np.ndarray = self.wma(
            n=np.sqrt(n),
            array=True,
            input=2 * self.wma(n / 2, array=True, input=input) - self.wma(n, array=True, input=input),
        )
        if array:
            return result
        return result[-1]

    def pine_sar(self, start: float = 0.02, inc: float = 0.02, maximum: float = 0.2) -> np.ndarray:
        close = self.close
        high = self.high
        low = self.low

        result = np.full_like(close, np.nan)
        max_min = np.full_like(close, np.nan)
        acceleration = np.full_like(close, np.nan)
        is_below = None
        is_first_trend_bar = False
        prev_high = high[0]
        prev_low = low[0]

        for i in range(len(close)):
            if i == 1:
                is_below = np.where(close[i] > close[i - 1], True, False)
                max_min = np.where(is_below, high[i - 1], low[i - 1])
                result = np.where(is_below, low[i - 1], high[i - 1])
                is_first_trend_bar = True
                acceleration.fill(start)

            result = result + acceleration * (max_min - result)

            is_below = np.where(result > low[i], False, True)
            max_min = np.where(is_below, prev_high, prev_low)
            result = np.where(is_below, np.maximum(prev_high, max_min), np.maximum(high[i], max_min))
            prev_high = np.maximum(prev_high, high[i])
            acceleration = np.where(is_below, start, acceleration)
            acceleration = np.where(
                is_below & (high[i] > max_min),
                np.minimum(acceleration + inc, maximum),
                acceleration,
            )
            prev_low = np.where(is_below, low[i], prev_low)

            is_below = np.where(result < high[i], True, False)
            max_min = np.where(is_below, prev_low, prev_high)
            result = np.where(is_below, np.minimum(prev_low, max_min), np.minimum(low[i], max_min))
            prev_low = np.minimum(prev_low, low[i])
            acceleration = np.where(~is_below, start, acceleration)
            acceleration = np.where(
                ~is_below & (low[i] < max_min),
                np.minimum(acceleration + inc, maximum),
                acceleration,
            )
            prev_high = np.where(~is_below, high[i], prev_high)

            is_first_trend_bar = np.where(
                is_first_trend_bar,
                False,
                np.where(is_below, high[i] > max_min, low[i] < max_min),
            )

            result = np.where(is_below, np.minimum(result, prev_low), np.maximum(result, prev_high))
            result = np.where(
                i > 1,
                np.where(
                    is_below,
                    np.minimum(result, low[i - 2]),
                    np.maximum(result, high[i - 2]),
                ),
                result,
            )

        return result[-1]

    def sar(self, acceleration: int = 0, maximum: int = 0, array: bool = False) -> float | np.ndarray:
        """
        SAR.

        start: 0.02
        increament: 0.02
        maximum: 0.2
        """
        result: np.ndarray = self.talib.SAR(self.high, self.low, acceleration, maximum)
        if array:
            return result
        return result[-1]

    def sarext(
        self,
        startvalue: int = 0,
        offsetonreverse: int = 0,
        accelerationinitlong: int = 0,
        accelerationlong: int = 0,
        accelerationmaxlong: int = 0,
        accelerationinitshort: int = 0,
        accelerationshort: int = 0,
        accelerationmaxshort: int = 0,
        array: bool = False,
    ) -> float | np.ndarray:
        """
        SAR Extended.

        start: 0.02
        increament: 0.02
        maximum: 0.2
        """
        result: np.ndarray = self.talib.SAREXT(
            self.high,
            self.low,
            startvalue,
            offsetonreverse,
            accelerationinitlong,
            accelerationlong,
            accelerationmaxlong,
            accelerationinitshort,
            accelerationshort,
            accelerationmaxshort,
        )
        if array:
            return result
        return result[-1]

    def apo(self, fast_period: int, slow_period: int, matype: int = 0, array: bool = False) -> float | np.ndarray:
        """
        APO.
        """
        result: np.ndarray = self.talib.APO(self.close, fast_period, slow_period, matype)
        if array:
            return result
        return result[-1]

    def cmo(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        CMO.
        """
        result: np.ndarray = self.talib.CMO(self.close, n)
        if array:
            return result
        return result[-1]

    def mom(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        MOM.
        """
        result: np.ndarray = self.talib.MOM(self.close, n)
        if array:
            return result
        return result[-1]

    def ppo(self, fast_period: int, slow_period: int, matype: int = 0, array: bool = False) -> float | np.ndarray:
        """
        PPO.
        """
        result: np.ndarray = self.talib.PPO(self.close, fast_period, slow_period, matype)
        if array:
            return result
        return result[-1]

    def roc(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        ROC.
        """
        result: np.ndarray = self.talib.ROC(self.close, n)
        if array:
            return result
        return result[-1]

    def rocr(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        ROCR.
        """
        result: np.ndarray = self.talib.ROCR(self.close, n)
        if array:
            return result
        return result[-1]

    def rocp(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        ROCP.
        """
        result: np.ndarray = self.talib.ROCP(self.close, n)
        if array:
            return result
        return result[-1]

    def rocr_100(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        ROCR100.
        """
        result: np.ndarray = self.talib.ROCR100(self.close, n)
        if array:
            return result
        return result[-1]

    def trix(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        TRIX.
        """
        result: np.ndarray = self.talib.TRIX(self.close, n)
        if array:
            return result
        return result[-1]

    def std(self, n: int, nbdev: int = 1, array: bool = False, input: np.ndarray | None = None) -> float | np.ndarray:
        """
        Standard deviation.
        """
        # result: np.ndarray = self.talib.STDDEV(self.close, n, nbdev)
        result: np.ndarray = self.talib.STDDEV(self._get_input(default=self.close, input=input), n, nbdev)
        if array:
            return result
        return result[-1]

    def sum(self, n: int, array: bool = False, input: np.ndarray | None = None) -> float | np.ndarray:
        """
        Summation.
        """
        # result: np.ndarray = self.talib.SUM(self.close, n)
        # print(self._get_input(default=self.close, input=input),'????????????')
        # for i in self._get_input(default=self.close, input=input):
        #     print(i, type(i),'?????????')
        result: np.ndarray = self.talib.SUM(self._get_input(default=self.close, input=input), n)
        if array:
            return result
        return result[-1]

    def obv(self, array: bool = False) -> float | np.ndarray:
        """
        On Balance Volume
        """

        result: np.ndarray = self.talib.OBV(self.close, self.volume)
        if array:
            return result
        return result[-1]

    def cci(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        Commodity Channel Index (CCI).
        """
        result: np.ndarray = self.talib.CCI(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def atr(self, n: int, multiplier: float = 1, array: bool = False, matype: int = 0) -> float | np.ndarray:
        """
        Average True Range (ATR).
        """
        if matype != 0:
            tr: float | np.ndarray = self.tr(array=True)
            result: float | np.ndarray = self.rma(n, array=True, input=tr)
        else:
            result: np.ndarray = self.talib.ATR(self.high, self.low, self.close, n)
        if array:
            return result * multiplier
        return result[-1] * multiplier

    def tr(self, array: bool = False) -> float | np.ndarray:
        """
        True Range (TR).
        """
        result: np.ndarray = self.talib.TRANGE(self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def natr(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        NATR.
        """
        result: np.ndarray = self.talib.NATR(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def stochf(self, fastk_period: int, fastd_period: int, array: bool = False) -> float | np.ndarray:
        """
        STOCHF.
        """
        result: np.ndarray = self.talib.STOCHF(
            self.high,
            self.low,
            self.close,
            fastk_period=fastk_period,
            fastd_period=fastd_period,
        )  # , fastd_matype=0
        if array:
            return result
        return result[-1]

    def rsi(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        Relative Strenght Index (RSI).
        """
        result: np.ndarray = self.talib.RSI(self.close, n)
        if array:
            return result
        return result[-1]

    def kdj0(
        self, n: int, smooth_k: int, smooth_d: int, array: bool = False
    ) -> tuple[float, float, float] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Stochastic Relative Strenght Index
        """
        # Stochastic. It is calculated by a formula: 100 * (close - lowest(low, length)) / (highest(high, length) - lowest(low, length)).

        hi = self.max(n, input="high", array=True)
        lo = self.min(n, input="low", array=True)
        raw_k = 100 * ((self.close - lo) / (hi - lo))
        # print(f"raw_k:{raw_k} -")
        k = self.rma(smooth_k, input=raw_k, array=True)
        d = self.rma(smooth_d, input=k, array=True)
        j = 3 * k - 2 * d
        print(f"     pk: {k[-1]};pd: {d[-1]}; |j:{j[-1]}| hi: {hi[-1]}; lo: {lo[-1]};")

        if array:
            return k, d, j
        return k[-1], d[-1], j[-1]

    def kdj(
        self, n: int, smooth_k: int, smooth_d: int, array: bool = False
    ) -> tuple[float, float, float] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Stochastic Relative Strenght Index
        """
        # Stochastic. It is calculated by a formula: 100 * (close - lowest(low, length)) / (highest(high, length) - lowest(low, length)).

        result: np.ndarray = self.talib.RSI(self.close, n)
        # print(self.close,'???', result)
        stochastic_rsi = (
            100
            * (result - self.min(n, input=result, array=True))
            / (self.max(n, input=result, array=True) - self.min(n, input=result, array=True))
        )
        k = self.sma(n=smooth_k, input=stochastic_rsi, array=True)
        d = self.sma(n=smooth_d, input=k, array=True)
        j = 3 * k - 2 * d

        if array:
            return k, d, j
        return k[-1], d[-1], j[-1]

    def kdj1(
        self,
        n: int,
        fastk_period: int,
        fastd_period: int,
        fastd_matype: int = 0,
        array: bool = False,
    ) -> tuple[float, float, float] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        KDJ
        """
        # RSV = stoch(close, high, low, RangeLength)
        # var K = 0.0
        # var D = 0.0
        # K[i] = (0.666)*nz(K[1],50)+(0.334)*nz(RSV,50)
        # D[i] = (0.666)*nz(D[1],50)+(0.334)*nz(K,50)
        # J = sma(3 * K - 2 * D,JsmoothLength)

        k, d = self.talib.STOCHRSI(
            self.close,
            timeperiod=n,
            fastk_period=fastk_period,
            fastd_period=fastd_period,
            fastd_matype=fastd_matype,
        )
        j = 3 * k - 2 * d
        if array:
            return k, d, j
        return k[-1], d[-1], j[-1]

    def macd(
        self,
        fast_period: int = 13,
        slow_period: int = 34,
        signal_period: int = 9,
        matype: int = 0,
        array: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[float, float, float]:
        """
        MACD.
            0-SMA; 1-EMA
        """
        # macd, signal, hist = self.talib.MACD(self.close, fast_period, slow_period, signal_period)
        macd, signal, hist = self.talib.MACDEXT(
            self.close,
            fastperiod=fast_period,
            fastmatype=matype,
            slowperiod=slow_period,
            slowmatype=matype,
            signalperiod=signal_period,
            signalmatype=matype,
        )

        def _find_color(hist: float, prev_hist: float) -> str:
            if hist > 0:
                if hist > prev_hist:
                    return "darkgreen"
                else:
                    return "lightgreen"
            else:
                if hist > prev_hist:
                    return "lightred"
                else:
                    return "darkred"

        m = self.close.size
        up_peak: np.ndarray = np.zeros(self.size)
        dn_peak: np.ndarray = np.zeros(self.size)
        color: np.ndarray = np.zeros(self.size, dtype=object)

        for i in range(m):
            if i == 0:
                color[i] = _find_color(hist[i], 0)
            else:
                color[i] = _find_color(hist[i], hist[i - 1])
            if color[i] == "darkgreen":
                if i == 0 or color[i - 1] != "darkgreen":
                    val = hist[i]
                else:
                    if up_peak[i - 1] <= hist[i]:
                        val = hist[i]
                        up_peak[i - 1] = np.nan
                    else:
                        val = np.nan
                up_peak[i] = val
            else:
                up_peak[i] = np.nan

            if color[i] == "darkred":
                if i == 0 or color[i - 1] != "darkred":
                    val = hist[i]
                else:
                    if dn_peak[i - 1] > hist[i]:
                        val = hist[i]
                        dn_peak[i - 1] = np.nan
                    else:
                        val = np.nan
                dn_peak[i] = val
            else:
                dn_peak[i] = np.nan

        if array:
            return macd, signal, hist, color, up_peak, dn_peak
        return macd[-1], signal[-1], hist[-1], color[-1], up_peak[-1], dn_peak[-1]

    def macd_divergence_v1(
        self,
        fast_period: int = 13,
        slow_period: int = 34,
        signal_period: int = 9,
        matype: int = 0,
        min_hist_diff: float = 0.0,
        min_ranges: int = 1,
        array: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[float, float, float]:
        macd_arr, signal_arr, hist_arr, color_arr, up_peak_arr, dn_peak_arr = self.macd(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            matype=1,
            array=True,
        )
        dark_green_ranges = []
        dark_red_ranges = []

        for i in range(len(macd_arr)):
            if i < 2:
                continue
            macd = macd_arr[i]
            hist = hist_arr[i]
            signal = signal_arr[i]
            up_peak, dn_peak = up_peak_arr[i], dn_peak_arr[i]
            color = color_arr[i]
            prev_hist = hist_arr[i - 1]
            prev_color = color_arr[i - 1]
            prev_prev_color = color_arr[i - 2]
            prev_up_peak, prev_dn_peak = up_peak_arr[i - 1], dn_peak_arr[i - 1]

            last_price = self.close[i]
            prev_last_price = self.close[i - 1]
            last_tick = self.ticks[i - 1]

            if prev_color == "darkgreen" and prev_prev_color != "darkgreen":
                # from light green turn into dark green, ignore
                pass
            else:
                if color == "lightgreen" and prev_color == "darkgreen":
                    """
                    min_ranges, min_hist_diff

                    if current is lightgreen bar and prev is darkgreen

                        if has start point
                            if has end point
                                if val >= end point
                                    append start point
                                    ### entry - if list len is 3
                                        mark
                                        clear list
                                else:
                                    append both new start point and end point (new range)
                        else not end point:
                            if val >= start point
                                replace start point
                            else:'
                                mark as end point

                        else start point missed
                            add a new range and set as start point

                        dark_green_ranges = [
                            [{'sp':(20230101, 12), 'ep':(20230102, 10.1)}, {'sp':(20230102, 10.1), 'ep':(20230103, 9)}, {'sp':(20230103, 9), 'ep':(20230104, 8.2)}],
                            [{'sp':(20230101, 12), 'ep':(20230102, 10.1)}, {'sp':(20230102, 10.1), 'ep':(20230103, 9)}],
                            ....
                        ]
                    """
                    if dark_green_ranges:
                        dark_green_lastest_range = dark_green_ranges[-1]
                        lastest_range_sp = dark_green_lastest_range[-1].get("sp") if dark_green_lastest_range else None
                        sp_dt, sp_val = (lastest_range_sp[0], lastest_range_sp[1]) if lastest_range_sp else (None, None)
                        if sp_val:
                            # has start point
                            lastest_range_ep = dark_green_lastest_range[-1].get("ep")
                            if lastest_range_ep:
                                # has end point
                                ep_dt, ep_val = (
                                    lastest_range_ep[0],
                                    lastest_range_ep[1],
                                )
                                if prev_hist >= ep_val:
                                    # append start point
                                    print(
                                        f"has start point, has end point, but new point > end point, append new set of ranges: {(last_tick, prev_hist, prev_last_price)} || old ranges length: {len(dark_green_lastest_range)}"
                                    )
                                    dark_green_new_range = []
                                    if prev_hist >= sp_val:
                                        dark_green_new_range.append(
                                            {
                                                "sp": (
                                                    last_tick,
                                                    prev_hist,
                                                    prev_last_price,
                                                ),
                                                "ep": (),
                                            }
                                        )
                                    else:
                                        # dark_green_new_range.append({'sp': lastest_range_sp, 'ep': (last_tick, prev_hist, prev_last_price)})
                                        if abs((prev_hist - ep_val) / ep_val) > min_hist_diff:
                                            dark_green_new_range.append(
                                                {
                                                    "sp": lastest_range_sp,
                                                    "ep": (
                                                        last_tick,
                                                        prev_hist,
                                                        prev_last_price,
                                                    ),
                                                }
                                            )
                                        else:
                                            dark_green_new_range.append({"sp": lastest_range_sp, "ep": ()})

                                    dark_green_ranges.append(dark_green_new_range)
                                else:
                                    # append both new start point and end point
                                    # dark_green_lastest_range[-1]['ep'] = (last_tick, prev_hist)
                                    ### TODO: at least has over 30% diff sin append, otherwise append empty dict
                                    # dark_green_lastest_range.append({'sp': lastest_range_ep, 'ep': (last_tick, prev_hist, prev_last_price)})
                                    if abs((prev_hist - ep_val) / ep_val) > min_hist_diff:
                                        dark_green_lastest_range.append(
                                            {
                                                "sp": lastest_range_ep,
                                                "ep": (
                                                    last_tick,
                                                    prev_hist,
                                                    prev_last_price,
                                                ),
                                            }
                                        )
                                    else:
                                        dark_green_lastest_range[-1]["ep"] = ()

                                    print(
                                        f"has start point, has end point, but new point < end point, append new range: {(last_tick, prev_hist, prev_last_price)} || new ranges length: {len(dark_green_lastest_range)}"
                                    )
                                    if (
                                        dark_green_lastest_range[-1]["ep"]
                                        and len(dark_green_lastest_range) >= min_ranges
                                    ):
                                        ### NOTE: entry point
                                        # compare the corresponding price bar getting high ==> short
                                        is_fulfill = True
                                        for _range in dark_green_lastest_range:
                                            try:
                                                _x0, _y0, _px0 = _range["sp"]
                                                _x1, _y1, _px1 = _range["ep"]
                                                if _px0 >= _px1:  # or abs((_y1 - _y0)/_y0) >= min_hist_diff:
                                                    is_fulfill = False
                                                    print(
                                                        f"Short Entry failed, the price in this range is not fulfilled - range:{_range}"
                                                    )
                                                    break
                                            except Exception as e:
                                                is_fulfill = False
                                                print(f"_range: {_range} - e:{e}")

                                        if is_fulfill:
                                            print(
                                                f">>>>> Short Entry point - dark_green_lastest_range:{dark_green_lastest_range}"
                                            )
                                            is_short = True

                            else:  # no end point
                                if prev_hist >= sp_val:
                                    # replace start point
                                    print(
                                        f"has start point, no end point, but new point > start point, replace start point: {(last_tick, prev_hist, prev_last_price)}"
                                    )
                                    dark_green_lastest_range[-1]["sp"] = (
                                        last_tick,
                                        prev_hist,
                                        prev_last_price,
                                    )
                                else:
                                    # mark as end point
                                    print(
                                        f"has start point, no end point, but new point < start point, mark as end point: {(last_tick, prev_hist, prev_last_price)}"
                                    )
                                    dark_green_lastest_range[-1]["ep"] = (
                                        last_tick,
                                        prev_hist,
                                        prev_last_price,
                                    )
                        else:
                            # start point missed - add a new range and set as start point
                            print(f"no start point, append new range: {(last_tick, prev_hist, prev_last_price)}")
                            dark_green_ranges[-1].append(
                                {
                                    "sp": (last_tick, prev_hist, prev_last_price),
                                    "ep": (),
                                }
                            )
                    else:
                        dark_green_first_range = []
                        dark_green_first_range.append({"sp": (last_tick, prev_hist, prev_last_price), "ep": ()})
                        print(f"dark_green_ranges is None, append new range: {(last_tick, prev_hist, prev_last_price)}")
                        dark_green_ranges.append(dark_green_first_range)

            if prev_color == "darkred" and prev_prev_color != "darkred":
                pass
            else:
                if color == "lightred" and prev_color == "darkred":
                    if dark_red_ranges:
                        dark_red_lastest_range = dark_red_ranges[-1]
                        lastest_range_sp = dark_red_lastest_range[-1].get("sp") if dark_red_lastest_range else None
                        sp_dt, sp_val = (lastest_range_sp[0], lastest_range_sp[1]) if lastest_range_sp else (None, None)
                        if sp_val:
                            # has start point
                            lastest_range_ep = dark_red_lastest_range[-1].get("ep")
                            if lastest_range_ep:
                                # has end point
                                ep_dt, ep_val = (
                                    lastest_range_ep[0],
                                    lastest_range_ep[1],
                                )
                                if prev_hist <= ep_val:
                                    # append start point
                                    print(
                                        f"has start point, has end point, but new point > end point, append new set of ranges: {(last_tick, prev_hist, prev_last_price)} || old ranges length: {len(dark_red_lastest_range)}"
                                    )
                                    dark_red_new_range = []
                                    if prev_hist <= sp_val:
                                        dark_red_new_range.append(
                                            {
                                                "sp": (
                                                    last_tick,
                                                    prev_hist,
                                                    prev_last_price,
                                                ),
                                                "ep": (),
                                            }
                                        )
                                    else:
                                        # dark_red_new_range.append({'sp': lastest_range_sp, 'ep': (last_tick, prev_hist, prev_last_price)})
                                        if abs((prev_hist - ep_val) / ep_val) > min_hist_diff:
                                            dark_red_new_range.append(
                                                {
                                                    "sp": lastest_range_sp,
                                                    "ep": (
                                                        last_tick,
                                                        prev_hist,
                                                        prev_last_price,
                                                    ),
                                                }
                                            )
                                        else:
                                            dark_red_new_range.append({"sp": lastest_range_sp, "ep": ()})

                                    dark_red_ranges.append(dark_red_new_range)
                                else:
                                    # append both new start point and end point
                                    ### TODO: at least has over 30% diff sin append, otherwise append empty dict
                                    # dark_red_lastest_range.append({'sp': lastest_range_ep, 'ep': (last_tick, prev_hist, prev_last_price)})

                                    if abs((prev_hist - ep_val) / ep_val) > min_hist_diff:
                                        dark_red_lastest_range.append(
                                            {
                                                "sp": lastest_range_ep,
                                                "ep": (
                                                    last_tick,
                                                    prev_hist,
                                                    prev_last_price,
                                                ),
                                            }
                                        )
                                    else:
                                        dark_red_lastest_range[-1]["ep"] = ()

                                    print(
                                        f"has start point, has end point, but new point < end point, append new range: {(last_tick, prev_hist, prev_last_price)} || new ranges length: {len(dark_red_lastest_range)}"
                                    )
                                    if (
                                        dark_red_lastest_range[-1]["ep"] and len(dark_red_lastest_range) >= min_ranges
                                    ):  ### NOTE: in here aldy has at least has 2 ranges
                                        ### NOTE: entry point
                                        # compare the corresponding price bar getting low ==> long
                                        is_fulfill = True
                                        for _range in dark_red_lastest_range:
                                            try:
                                                _x0, _y0, _px0 = _range["sp"]
                                                _x1, _y1, _px1 = _range["ep"]
                                                if _px0 <= _px1:  #  or abs((_y1 - _y0)/_y0) >= min_hist_diff:
                                                    is_fulfill = False
                                                    print(
                                                        f"Long Entry failed, the price in this range is not fulfilled - range:{_range}"
                                                    )
                                                    break
                                            except Exception as e:
                                                is_fulfill = False
                                                print(f"_range: {_range} - e:{e}")
                                        if is_fulfill:
                                            print(
                                                f">>>>> Long Entry point - dark_red_lastest_range:{dark_red_lastest_range}"
                                            )
                                            is_long = True

                            else:  # no end point
                                if prev_hist <= sp_val:
                                    # replace start point
                                    print(
                                        f"has start point, no end point, but new point > start point, replace start point: {(last_tick, prev_hist, prev_last_price)}"
                                    )
                                    dark_red_lastest_range[-1]["sp"] = (
                                        last_tick,
                                        prev_hist,
                                        prev_last_price,
                                    )
                                else:
                                    # mark as end point
                                    print(
                                        f"has start point, no end point, but new point < start point, mark as end point: {(last_tick, prev_hist, prev_last_price)}"
                                    )
                                    dark_red_lastest_range[-1]["ep"] = (
                                        last_tick,
                                        prev_hist,
                                        prev_last_price,
                                    )
                        else:
                            # start point missed
                            print(f"no start point, append new range: {(last_tick, prev_hist, prev_last_price)}")
                            dark_red_ranges[-1].append(
                                {
                                    "sp": (last_tick, prev_hist, prev_last_price),
                                    "ep": (),
                                }
                            )
                    else:
                        dark_red_first_range = []
                        dark_red_first_range.append({"sp": (last_tick, prev_hist, prev_last_price), "ep": ()})
                        print(f"dark_red_ranges is None, append new range: {(last_tick, prev_hist, prev_last_price)}")
                        dark_red_ranges.append(dark_red_first_range)

        if array:
            return dark_green_ranges, dark_red_ranges
        return dark_green_ranges[-1] if dark_green_ranges else [], dark_red_ranges[-1] if dark_red_ranges else []

    def macd_divergence(
        self,
        fast_period: int = 13,
        slow_period: int = 34,
        signal_period: int = 9,
        matype: int = 0,
        debug: bool = False,
        array: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[float, float, float]:
        macd_arr, signal_arr, hist_arr, color_arr, up_peak_arr, dn_peak_arr = self.macd(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            matype=1,
            array=True,
        )
        dark_green_ranges = []
        dark_red_ranges = []

        for i in range(len(macd_arr)):
            if i < 2:
                continue
            macd = macd_arr[i]
            hist = hist_arr[i]
            signal = signal_arr[i]
            up_peak, dn_peak = up_peak_arr[i], dn_peak_arr[i]
            color = color_arr[i]
            prev_hist = hist_arr[i - 1]
            prev_color = color_arr[i - 1]
            prev_prev_color = color_arr[i - 2]
            prev_up_peak, prev_dn_peak = up_peak_arr[i - 1], dn_peak_arr[i - 1]

            last_price = self.close[i]
            prev_last_price = self.close[i - 1]
            last_tick = self.ticks[i - 1]

            if prev_color == "darkreen" and prev_prev_color != "darkgreen":
                # from light green turn into dark green, ignore
                pass
            else:
                if color == "lightgreen" and prev_color == "darkgreen":
                    """

                    if current is lightgreen bar and prev is darkgreen

                        if has start point
                            if has end point
                                if val >= end point
                                    append start point
                                    # ### entry - if list len is 3
                                    #     mark
                                    #     clear list
                                else:
                                    append both new start point and end point (new range)

                            else not end point:
                                if val >= start point
                                    replace start point
                                else:
                                    mark as end point

                        else start point missed
                            add a new range and set as start point

                        dark_green_ranges = [
                            [{'sp':(20230101, 12), 'ep':(20230102, 10.1)}, {'sp':(20230102, 10.1), 'ep':(20230103, 9)}, {'sp':(20230103, 9), 'ep':(20230104, 8.2)}],
                            [{'sp':(20230101, 12), 'ep':(20230102, 10.1)}, {'sp':(20230102, 10.1), 'ep':(20230103, 9)}],
                            ....
                        ]
                    """
                    if dark_green_ranges:
                        dark_green_lastest_range = dark_green_ranges[-1]
                        lastest_range_sp = dark_green_lastest_range[-1].get("sp") if dark_green_lastest_range else None
                        sp_dt, sp_val = (lastest_range_sp[0], lastest_range_sp[1]) if lastest_range_sp else (None, None)
                        if sp_val:
                            # has start point
                            lastest_range_ep = dark_green_lastest_range[-1].get("ep")
                            if lastest_range_ep:
                                # has end point
                                ep_dt, ep_val = (
                                    lastest_range_ep[0],
                                    lastest_range_ep[1],
                                )
                                if prev_hist >= ep_val:
                                    # append start point
                                    if debug:
                                        print(
                                            f"has start point, has end point, but new point > end point, append new set of ranges: {(last_tick, prev_hist, prev_last_price)} || old ranges length: {len(dark_green_lastest_range)}"
                                        )
                                    dark_green_new_range = []
                                    if prev_hist >= sp_val:
                                        dark_green_new_range.append(
                                            {
                                                "sp": (
                                                    last_tick,
                                                    prev_hist,
                                                    prev_last_price,
                                                ),
                                                "ep": (),
                                            }
                                        )
                                    else:
                                        dark_green_new_range.append(
                                            {
                                                "sp": lastest_range_sp,
                                                "ep": (
                                                    last_tick,
                                                    prev_hist,
                                                    prev_last_price,
                                                ),
                                            }
                                        )

                                    dark_green_ranges.append(dark_green_new_range)
                                else:
                                    # append both new start point and end point
                                    if debug:
                                        print(
                                            f"has start point, has end point, but new point < end point, append new range: {(last_tick, prev_hist, prev_last_price)} || new ranges length: {len(dark_green_lastest_range)}"
                                        )
                                    dark_green_lastest_range.append(
                                        {
                                            "sp": lastest_range_ep,
                                            "ep": (
                                                last_tick,
                                                prev_hist,
                                                prev_last_price,
                                            ),
                                        }
                                    )
                            else:  # no end point
                                if prev_hist >= sp_val:
                                    # replace start point
                                    if debug:
                                        print(
                                            f"has start point, no end point, but new point > start point, replace start point: {(last_tick, prev_hist, prev_last_price)}"
                                        )
                                    dark_green_lastest_range[-1]["sp"] = (
                                        last_tick,
                                        prev_hist,
                                        prev_last_price,
                                    )
                                else:
                                    # mark as end point
                                    if debug:
                                        print(
                                            f"has start point, no end point, but new point < start point, mark as end point: {(last_tick, prev_hist, prev_last_price)}"
                                        )
                                    dark_green_lastest_range[-1]["ep"] = (
                                        last_tick,
                                        prev_hist,
                                        prev_last_price,
                                    )
                        else:
                            # start point missed - add a new range and set as start point
                            if debug:
                                print(f"no start point, append new range: {(last_tick, prev_hist, prev_last_price)}")
                            dark_green_ranges[-1].append(
                                {
                                    "sp": (last_tick, prev_hist, prev_last_price),
                                    "ep": (),
                                }
                            )
                    else:
                        dark_green_first_range = []
                        dark_green_first_range.append({"sp": (last_tick, prev_hist, prev_last_price), "ep": ()})
                        if debug:
                            print(
                                f"dark_green_ranges is None, append new range: {(last_tick, prev_hist, prev_last_price)}"
                            )
                        dark_green_ranges.append(dark_green_first_range)

            if prev_color == "darkred" and prev_prev_color != "darkred":
                pass
            else:
                if color == "lightred" and prev_color == "darkred":
                    if dark_red_ranges:
                        dark_red_lastest_range = dark_red_ranges[-1]
                        lastest_range_sp = dark_red_lastest_range[-1].get("sp") if dark_red_lastest_range else None
                        sp_dt, sp_val = (lastest_range_sp[0], lastest_range_sp[1]) if lastest_range_sp else (None, None)
                        if sp_val:
                            # has start point
                            lastest_range_ep = dark_red_lastest_range[-1].get("ep")
                            if lastest_range_ep:
                                # has end point
                                ep_dt, ep_val = (
                                    lastest_range_ep[0],
                                    lastest_range_ep[1],
                                )
                                if prev_hist <= ep_val:
                                    # append start point
                                    if debug:
                                        print(
                                            f"has start point, has end point, but new point > end point, append new set of ranges: {(last_tick, prev_hist, prev_last_price)} || old ranges length: {len(dark_red_lastest_range)}"
                                        )
                                    dark_red_new_range = []
                                    if prev_hist <= sp_val:
                                        dark_red_new_range.append(
                                            {
                                                "sp": (
                                                    last_tick,
                                                    prev_hist,
                                                    prev_last_price,
                                                ),
                                                "ep": (),
                                            }
                                        )
                                    else:
                                        dark_red_new_range.append(
                                            {
                                                "sp": lastest_range_sp,
                                                "ep": (
                                                    last_tick,
                                                    prev_hist,
                                                    prev_last_price,
                                                ),
                                            }
                                        )

                                    dark_red_ranges.append(dark_red_new_range)
                                else:
                                    # append both new start point and end point
                                    if debug:
                                        print(
                                            f"has start point, has end point, but new point < end point, append new range: {(last_tick, prev_hist, prev_last_price)} || new ranges length: {len(dark_red_lastest_range)}"
                                        )
                                    dark_red_lastest_range.append(
                                        {
                                            "sp": lastest_range_ep,
                                            "ep": (
                                                last_tick,
                                                prev_hist,
                                                prev_last_price,
                                            ),
                                        }
                                    )

                            else:  # no end point
                                if prev_hist <= sp_val:
                                    # replace start point
                                    if debug:
                                        print(
                                            f"has start point, no end point, but new point > start point, replace start point: {(last_tick, prev_hist, prev_last_price)}"
                                        )
                                    dark_red_lastest_range[-1]["sp"] = (
                                        last_tick,
                                        prev_hist,
                                        prev_last_price,
                                    )
                                else:
                                    # mark as end point
                                    if debug:
                                        print(
                                            f"has start point, no end point, but new point < start point, mark as end point: {(last_tick, prev_hist, prev_last_price)}"
                                        )
                                    dark_red_lastest_range[-1]["ep"] = (
                                        last_tick,
                                        prev_hist,
                                        prev_last_price,
                                    )
                        else:
                            # start point missed
                            if debug:
                                print(f"no start point, append new range: {(last_tick, prev_hist, prev_last_price)}")
                            dark_red_ranges[-1].append(
                                {
                                    "sp": (last_tick, prev_hist, prev_last_price),
                                    "ep": (),
                                }
                            )
                    else:
                        dark_red_first_range = []
                        dark_red_first_range.append({"sp": (last_tick, prev_hist, prev_last_price), "ep": ()})
                        if debug:
                            print(
                                f"dark_red_ranges is None, append new range: {(last_tick, prev_hist, prev_last_price)}"
                            )
                        dark_red_ranges.append(dark_red_first_range)

        if array:
            return dark_green_ranges, dark_red_ranges
        return dark_green_ranges[-1] if dark_green_ranges else [], dark_red_ranges[-1] if dark_red_ranges else []

    def impulse_macd(
        self, ma_period: int = 34, signal_period: int = 9, array: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[float, float, float]:
        """
        Impulse Macd
        """

        def calc_zlema(length, src):
            ema1 = self.talib.EMA(src, length)
            ema2 = self.talib.EMA(ema1, length)
            d = ema1 - ema2
            return ema1 + d

        close = self.close
        high = self.high
        low = self.low

        src = (high + low + close) / 3
        hi = self.smma(n=ma_period, input=high, array=True)
        lo = self.smma(n=ma_period, input=low, array=True)
        mi = calc_zlema(ma_period, src=src)
        # print(f"\nsrc:{src[-1]}; hi:{hi[-1]}; lo:{lo[-1]}; mi:{mi[-1]}; || srclen:{len(src)}; hilen:{len(hi)}")
        macd = np.where(mi > hi, mi - hi, np.where(mi < lo, mi - lo, 0))  # ImpulseMACD
        signal = self.talib.SMA(macd, signal_period)  # ImpulseMACDCDSignal
        hist = macd - signal  # ImpulseHisto
        # macd_coloe = np.where(close > mi, np.where(close > hi, 'lime', 'green'), np.where(close < lo, 'red', 'orange'))

        if array:
            return macd, signal, hist
        return macd[-1], signal[-1], hist[-1]

    def adx(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        ADX.
        """
        result: np.ndarray = self.talib.ADX(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def adx1(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        ADX.

        ```
        TrueRange = max(max(high-low, abs(high-nz(close[1]))), abs(low-nz(close[1])))
        DirectionalMovementPlus = high-nz(high[1]) > nz(low[1])-low ? max(high-nz(high[1]), 0): 0
        DirectionalMovementMinus = nz(low[1])-low > high-nz(high[1]) ? max(nz(low[1])-low, 0): 0

        SmoothedTrueRange = 0.0
        SmoothedTrueRange[i] = nz(SmoothedTrueRange[1]) - (nz(SmoothedTrueRange[1])/len) + TrueRange

        SmoothedDirectionalMovementPlus = 0.0
        SmoothedDirectionalMovementPlus[i] = nz(SmoothedDirectionalMovementPlus[1]) - (nz(SmoothedDirectionalMovementPlus[1])/len) + DirectionalMovementPlus

        SmoothedDirectionalMovementMinus = 0.0
        SmoothedDirectionalMovementMinus[i] = nz(SmoothedDirectionalMovementMinus[1]) - (nz(SmoothedDirectionalMovementMinus[1])/len) + DirectionalMovementMinus

        DIPlus = SmoothedDirectionalMovementPlus / SmoothedTrueRange * 100
        DIMinus = SmoothedDirectionalMovementMinus / SmoothedTrueRange * 100
        DX = abs(DIPlus-DIMinus) / (DIPlus+DIMinus)*100
        ADX = sma(DX, len)
        ```
        """

        m = self.close.size
        TrueRange: np.ndarray = self.tr(array=True)
        # TrueRange: np.ndarray = np.zeros(self.size)
        DirectionalMovementPlus: np.ndarray = np.zeros(self.size)
        DirectionalMovementMinus: np.ndarray = np.zeros(self.size)
        SmoothedTrueRange: np.ndarray = np.zeros(self.size)
        SmoothedDirectionalMovementPlus: np.ndarray = np.zeros(self.size)
        SmoothedDirectionalMovementMinus: np.ndarray = np.zeros(self.size)
        DIPlus: np.ndarray = np.zeros(self.size)
        DIMinus: np.ndarray = np.zeros(self.size)
        DX: np.ndarray = np.zeros(self.size)
        ADX: np.ndarray = np.zeros(self.size)

        for i in range(1, m):
            DirectionalMovementPlus[i] = (
                max(self.high[i] - self.high[i - 1], 0)
                if self.high[i] - self.high[i - 1] > self.low[i - 1] - self.low[i]
                else 0
            )
            DirectionalMovementMinus[i] = (
                max(self.low[i - 1] - self.low[i], 0)
                if self.low[i - 1] - self.low[i] > self.high[i] - self.high[i - 1]
                else 0
            )
            SmoothedTrueRange[i] = SmoothedTrueRange[i - 1] - (SmoothedTrueRange[i - 1] / n) + TrueRange[i]
            SmoothedDirectionalMovementPlus[i] = (
                SmoothedDirectionalMovementPlus[i - 1]
                - (SmoothedDirectionalMovementPlus[i - 1] / n)
                + DirectionalMovementPlus[i]
            )
            SmoothedDirectionalMovementMinus[i] = (
                SmoothedDirectionalMovementMinus[i - 1]
                - (SmoothedDirectionalMovementMinus[i - 1] / n)
                + DirectionalMovementMinus[i]
            )
            DIPlus[i] = SmoothedDirectionalMovementPlus[i] / SmoothedTrueRange[i] * 100
            DIMinus[i] = SmoothedDirectionalMovementMinus[i] / SmoothedTrueRange[i] * 100
            DX[i] = abs(DIPlus[i] - DIMinus[i]) / (DIPlus[i] + DIMinus[i]) * 100
            DX[i] = 0.0 if np.isnan(DX[i]) else DX[i]
            # print(f"({DIPlus[i]}-{DIMinus[i]}) / ({DIPlus[i]}+{DIMinus[i]} -- i:{i}")

        # ADX = self.sma(n=n, input=DX, array=True)
        ADX = self.talib.SMA(DX, timeperiod=n)

        # np.isnan(myarray).any()
        # print(f"TrueRange:{TrueRange[-1]}; DirectionalMovementPlus:{DirectionalMovementPlus[-1]}; DirectionalMovementMinus:{DirectionalMovementMinus[-1]}; SmoothedTrueRange:{SmoothedTrueRange[-1]}; SmoothedDirectionalMovementPlus:{SmoothedDirectionalMovementPlus[-1]}; SmoothedDirectionalMovementMinus:{SmoothedDirectionalMovementMinus[-1]}; DIPlus:{DIPlus[-1]}; DIMinus:{DIMinus[-1]}; DX:{DX[-1]}; ADX:{ADX[-1]}; ")
        # print(f"DX:{np.isnan(DX).any()}\nADX:{np.isnan(ADX).any()} n:{n}")

        # len = n
        # TrueRange = np.maximum(np.maximum(self.high - self.low, np.abs(self.high - np.roll(self.close, 1))), np.abs(self.low - np.roll(self.close, 1)))
        # DirectionalMovementPlus = np.where(self.high - np.roll(self.high, 1) > np.roll(self.low, 1) - self.low, np.maximum(self.high - np.roll(self.high, 1), 0), 0)
        # DirectionalMovementMinus = np.where(np.roll(self.low, 1) - self.low > self.high - np.roll(self.high, 1), np.maximum(np.roll(self.low, 1) - self.low, 0), 0)

        # SmoothedTrueRange = np.zeros_like(self.close)
        # SmoothedDirectionalMovementPlus = np.zeros_like(self.close)
        # SmoothedDirectionalMovementMinus = np.zeros_like(self.close)

        # for i in range(1, len):
        #     SmoothedTrueRange[i] = SmoothedTrueRange[i-1] - SmoothedTrueRange[i-1] / len + TrueRange[i]
        #     SmoothedDirectionalMovementPlus[i] = SmoothedDirectionalMovementPlus[i-1] - SmoothedDirectionalMovementPlus[i-1] / len + DirectionalMovementPlus[i]
        #     SmoothedDirectionalMovementMinus[i] = SmoothedDirectionalMovementMinus[i-1] - SmoothedDirectionalMovementMinus[i-1] / len + DirectionalMovementMinus[i]

        # DIPlus = SmoothedDirectionalMovementPlus / SmoothedTrueRange * 100
        # DIMinus = SmoothedDirectionalMovementMinus / SmoothedTrueRange * 100
        # DX = np.abs(DIPlus - DIMinus) / (DIPlus + DIMinus) * 100
        # ADX = self.talib.SMA(DX, len)

        # # ADX_TALIB: np.ndarray = self.talib.ADX(self.high, self.low, self.close, n)
        # print(f"TrueRange:{TrueRange[-1]}; DirectionalMovementPlus:{DirectionalMovementPlus[-1]}; DirectionalMovementMinus:{DirectionalMovementMinus[-1]}; SmoothedTrueRange:{SmoothedTrueRange[-1]}; SmoothedDirectionalMovementPlus:{SmoothedDirectionalMovementPlus[-1]}; SmoothedDirectionalMovementMinus:{SmoothedDirectionalMovementMinus[-1]}; DIPlus:{DIPlus[-1]}; DIMinus:{DIMinus[-1]}; DX:{DX[-1]}; ADX:{ADX[-1]}; ADX_TALIB:{ADX_TALIB[-1]}; ")

        if array:
            return ADX
        return ADX[-1]

    def adxr(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        ADXR.
        """
        result: np.ndarray = self.talib.ADXR(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def dx(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        DX.
        """
        result: np.ndarray = self.talib.DX(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def minus_di(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        MINUS_DI.
        """
        result: np.ndarray = self.talib.MINUS_DI(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def plus_di(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        PLUS_DI.
        """
        result: np.ndarray = self.talib.PLUS_DI(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def willr(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        WILLR.
        """
        result: np.ndarray = self.talib.WILLR(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def ultosc(
        self,
        time_period1: int = 7,
        time_period2: int = 14,
        time_period3: int = 28,
        array: bool = False,
    ) -> float | np.ndarray:
        """
        Ultimate Oscillator.
        """
        result: np.ndarray = self.talib.ULTOSC(
            self.high, self.low, self.close, time_period1, time_period2, time_period3
        )
        if array:
            return result
        return result[-1]

    def trange(self, array: bool = False) -> float | np.ndarray:
        """
        TRANGE.
        """
        result: np.ndarray = self.talib.TRANGE(self.high, self.low, self.close)
        if array:
            return result
        return result[-1]

    def bband(
        self,
        n: int,
        dev: float,
        matype: int = 0,
        array: bool = False,
        input: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """
        Bollinger Channel.
        """
        upperband, middleband, lowerband = self.talib.BBANDS(
            self._get_input(default=self.close, input=input),
            timeperiod=n,
            nbdevup=dev,
            nbdevdn=dev,
            matype=matype,
        )
        if array:
            return upperband, lowerband
        return upperband[-1], lowerband[-1]

    def boll(
        self, n: int, dev: float, array: bool = False, input: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """
        Bollinger Channel.
        """
        mid: float | np.ndarray = self.sma(n, array, input=input)
        std: float | np.ndarray = self.std(n, 1, array, input=input)

        up: float | np.ndarray = mid + std * dev
        down: float | np.ndarray = mid - std * dev

        return up, down

    def keltner(
        self,
        n: int,
        dev: float,
        atr_length: int = None,
        matype: int = 0,
        array: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """
        Keltner Channel.
        """
        assert matype in [0, 1]  # 0-SMA; 1-EMA
        if not atr_length:
            atr_length = n
        if not matype:
            mid: float | np.ndarray = self.sma(n, array)
            atr: float | np.ndarray = self.atr(atr_length, array)
        else:
            mid: float | np.ndarray = self.ema(n, array)
            tr: float | np.ndarray = self.tr(array=True)
            atr: float | np.ndarray = self.ema(atr_length, array, input=tr)

        up: float | np.ndarray = mid + atr * dev
        down: float | np.ndarray = mid - atr * dev

        return up, mid, down

    def min(self, n: int, array: bool = False, input: np.ndarray = None) -> float | np.ndarray:
        """
        min.
        """
        result: np.ndarray = self._get_input(default=self.close, input=input)
        if n > 1:
            result: np.ndarray = self.talib.MIN(result, n)
        if array:
            return result
        return result[-1]

    def minimum(self, input1: np.ndarray, input2: np.ndarray, array: bool = False) -> float | np.ndarray:
        """
        minimum.
        """
        result: np.ndarray = np.maximum(
            self._get_input(default=None, input=input1),
            self._get_input(default=None, input=input2),
        )

        if array:
            return result
        return result[-1]

    def lowest(self, n: int, array: bool = False, input: np.ndarray = None) -> float | np.ndarray:
        """
        lowest.
        """
        result: np.ndarray = self._get_input(default=self.close, input=input)
        lowest = np.full_like(result, np.nan)
        for i in range(n - 1, result.size):
            lowest[i] = np.min(result[i - n + 1 : i + 1])
        if array:
            return lowest
        return lowest[-1]

    def maximum(self, input1: np.ndarray, input2: np.ndarray, array: bool = False) -> float | np.ndarray:
        """
        maximum.
        """
        result: np.ndarray = np.maximum(
            self._get_input(default=None, input=input1),
            self._get_input(default=None, input=input2),
        )
        if array:
            return result
        return result[-1]

    def max(self, n: int, array: bool = False, input: np.ndarray = None) -> float | np.ndarray:
        """
        max.
        """
        result: np.ndarray = self._get_input(default=self.close, input=input)
        if n > 1:
            result: np.ndarray = self.talib.MAX(result, n)
        if array:
            return result
        return result[-1]

    def highest(self, n: int, array: bool = False, input: np.ndarray = None) -> float | np.ndarray:
        """
        highest.
        """
        result: np.ndarray = self._get_input(default=self.close, input=input)
        highest = np.full_like(result, np.nan)
        for i in range(n - 1, result.size):
            highest[i] = np.max(result[i - n + 1 : i + 1])
        if array:
            return highest
        return highest[-1]

    def donchian(self, n: int, array: bool = False) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """
        Donchian Channel.
        """
        up: np.ndarray = self.talib.MAX(self.high, n)
        down: np.ndarray = self.talib.MIN(self.low, n)

        if array:
            return up, down
        return up[-1], down[-1]

    def aroon(self, n: int, array: bool = False) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """
        Aroon indicator.
        """
        aroon_down, aroon_up = self.talib.AROON(self.high, self.low, n)

        if array:
            return aroon_up, aroon_down
        return aroon_up[-1], aroon_down[-1]

    def aroonosc(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        Aroon Oscillator.
        """
        result: np.ndarray = self.talib.AROONOSC(self.high, self.low, n)

        if array:
            return result
        return result[-1]

    def minus_dm(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        MINUS_DM.
        """
        result: np.ndarray = self.talib.MINUS_DM(self.high, self.low, n)

        if array:
            return result
        return result[-1]

    def plus_dm(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        PLUS_DM.
        """
        result: np.ndarray = self.talib.PLUS_DM(self.high, self.low, n)

        if array:
            return result
        return result[-1]

    def mfi(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        Money Flow Index.
        """
        result: np.ndarray = self.talib.MFI(self.high, self.low, self.close, self.volume, n)
        if array:
            return result
        return result[-1]

    def ad(self, array: bool = False) -> float | np.ndarray:
        """
        Chaikin A/D Line
        """
        result: np.ndarray = self.talib.AD(self.high, self.low, self.close, self.volume)
        if array:
            return result
        return result[-1]

    def adosc(self, fast_period: int, slow_period: int, array: bool = False) -> float | np.ndarray:
        """
        Chaikin A/D Oscillator
        AKA Chaikin Money Flow
        """
        result: np.ndarray = self.talib.ADOSC(self.high, self.low, self.close, self.volume, fast_period, slow_period)
        if array:
            return result
        return result[-1]

    def bop(self, array: bool = False) -> float | np.ndarray:
        """
        BOP.
        """
        result: np.ndarray = self.talib.BOP(self.open, self.high, self.low, self.close)

        if array:
            return result
        return result[-1]

    def stoch(
        self,
        fastk_period: int,
        slowk_period: int,
        slowk_matype: int,
        slowd_period: int,
        slowd_matype: int,
        array: bool = False,
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """
        Stochastic Indicator
        """
        k, d = self.talib.STOCH(
            self.high,
            self.low,
            self.close,
            fastk_period,
            slowk_period,
            slowk_matype,
            slowd_period,
            slowd_matype,
        )
        if array:
            return k, d
        return k[-1], d[-1]

    def avg(self, *args) -> float:
        """
        avg
        """
        return sum(args) / len(args)

    def ahr999(self) -> float | np.ndarray:
        """
        ahr999 指数 = （比特币价格 /200 日定投成本） * （比特币价格 / 指数增长估值）
        - 200日定投成本 = 过去200日的比特币价格的均值
        - 指数增长估值 = 10^[5.84 * log(币龄) - 17.01]
        - 币龄 = 当前距离2009年1月3日(该日期被认为是btc诞生日)的天数 (timestamp: 1230940800)

        当ahr999<0.45时适合抄底, 0.45<ahr999<1.2时适合定投BTC, ahr999>1.2时不是良好的投资btc的时机

        """

        cost_200d = round(geometric_mean(self.close[-200:]), 2)
        # dt = datetime.utcfromtimestamp(self.ticks[-1].astype(int))
        dt = datetime.fromtimestamp(self.ticks[-1].astype(int))
        timestamp = dt.timestamp() * 1000

        coin_age = (timestamp - 1230940800 * 1000) / 24 / 60 / 60 / 1000
        # print(f"cost_200d:{cost_200d} - last_tick_timestamp:{timestamp} - coin_age:{coin_age}")
        logprice = 10 ** (5.84 * math.log(coin_age, 10) - 17.01)

        try:
            ahr999 = round((self.close[-1] / cost_200d) * (self.close[-1] / logprice), 4)
            # print(f"ahr999: {ahr999} - logprice:{logprice} - cost_200d:{cost_200d}")

        except BaseException as e:
            print(
                f"Something went wrong when calculating ahr999. Error:{str(e)}. cost_200d:{cost_200d}; logprice:{logprice}"
            )

        return ahr999

        # e1: np.ndarray = self.ema(n, array=True)
        # e2: np.ndarray = self.ema(n, input=e1, array=True)
        # dema: np.ndarray = 2 * e1 - e2
        # if array:
        #     return dema
        # return dema[-1]

    def rainbow(self) -> float | np.ndarray:
        """
        https://www.tradingview.com/script/hifY3Gu9-Bitcoin-Rainbow-Logarithmic-Curves/
        """

        # dt = datetime.utcfromtimestamp(self.ticks[-1].astype(int))
        dt = datetime.fromtimestamp(self.ticks[-1].astype(int))
        timestamp = dt.timestamp() * 1000

        TimeIndex = timestamp < 1279670400000 if 3.0 else (timestamp - 1279670400000) / 86400000
        x = TimeIndex
        fairCurveMid = math.pow(10, 1.910 * math.log(x) - 11.3955) + 1

        curve292 = fairCurveMid * 0.2923
        curve236 = fairCurveMid * 0.2361
        curve382 = fairCurveMid * 0.382
        curve500 = fairCurveMid * 0.5
        curve618 = fairCurveMid * 0.618
        curve786 = fairCurveMid * 0.786
        curve1272 = fairCurveMid * 1.272
        curve1414 = fairCurveMid * 1.618
        curve2000 = fairCurveMid * 2
        curve2414 = fairCurveMid * 2.618
        curve3414 = fairCurveMid * 3.414
        curve4272 = fairCurveMid * 4.272

        return (
            curve292,
            curve236,
            curve382,
            curve500,
            curve618,
            curve786,
            curve1272,
            curve1414,
            curve2000,
            curve2414,
            curve3414,
            curve4272,
        )

    def ichimoku(
        self,
        tenkan_sen_period: int = 9,
        kijun_sen_period: int = 26,
        senkou_span_period: int = 52,
        chikou_span_period: int = 26,
        array: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[float, float, float, float, float]:
        """
        Ichimoku Cloud (Ichimoku)
        """

        # Args :
        #     df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        #     ohlc: list defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
        #     param: Periods to be used in computation (default [tenkan_sen_period, kijun_sen_period, senkou_span_period, chikou_span_period] = [9, 26, 52, 26])

        # Returns :
        #     df : Pandas DataFrame with new columns added for ['Tenkan Sen', 'Kijun Sen', 'Senkou Span A', 'Senkou Span B', 'Chikou Span']

        # tenkan_sen_column = 'Tenkan Sen'
        # kijun_sen_column = 'Kijun Sen'
        # senkou_span_a_column = 'Senkou Span A'
        # senkou_span_b_column = 'Senkou Span B'
        # chikou_span_column = 'Chikou Span'

        # Tenkan-sen (Conversion Line)
        tenkan_sen_high = self.max(
            n=tenkan_sen_period, input="high", array=True
        )  # self.high.rolling(window=tenkan_sen_period).max()
        tenkan_sen_low = self.min(
            n=tenkan_sen_period, input="low", array=True
        )  # self.low.rolling(window=tenkan_sen_period).min()
        tenkan_sen = (tenkan_sen_high + tenkan_sen_low) / 2

        # Kijun-sen (Base Line)
        kijun_sen_high = self.max(
            n=kijun_sen_period, input="high", array=True
        )  # self.high.rolling(window=kijun_sen_period).max()
        kijun_sen_low = self.min(
            n=kijun_sen_period, input="low", array=True
        )  # self.low.rolling(window=kijun_sen_period).min()
        kijun_sen = (kijun_sen_high + kijun_sen_low) / 2

        # Senkou Span A (Leading Span A)
        senkou_span_a = self.shift(
            n=chikou_span_period - 1, input=((tenkan_sen + kijun_sen) / 2)
        )  # ((tenkan_sen + kijun_sen) / 2).shift(kijun_sen_period)

        # Senkou Span B (Leading Span B)
        senkou_span_high = self.max(
            n=senkou_span_period, input="high", array=True
        )  # self.high.rolling(window=senkou_span_period).max()
        senkou_span_low = self.min(
            n=senkou_span_period, input="low", array=True
        )  # self.low.rolling(window=senkou_span_period).min()
        senkou_span_b = self.shift(
            n=chikou_span_period - 1, input=((senkou_span_high + senkou_span_low) / 2)
        )  # ((senkou_span_high + senkou_span_low) / 2).shift(senkou_span_period)

        # The most current closing price plotted chikou_span_period time periods behind (Lagging Span)
        chikou_span = self.shift(
            n=-1 * chikou_span_period + 1, input=self.close
        )  # self.close.shift(-1 * chikou_span_period)

        if array:
            return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
        return (
            tenkan_sen[-1],
            kijun_sen[-1],
            senkou_span_a[-1],
            senkou_span_b[-1],
            chikou_span[-1],
        )

    def ichimoku1(
        self,
        tenkan_sen_period: int = 9,
        kijun_sen_period: int = 26,
        senkou_span_period: int = 52,
        chikou_span_period: int = 26,
        array: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[float, float, float, float, float]:
        """
        Ichimoku Cloud (Ichimoku)
        """

        # Args :
        #     df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        #     ohlc: list defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
        #     param: Periods to be used in computation (default [tenkan_sen_period, kijun_sen_period, senkou_span_period, chikou_span_period] = [9, 26, 52, 26])

        # Returns :
        #     df : Pandas DataFrame with new columns added for ['Tenkan Sen', 'Kijun Sen', 'Senkou Span A', 'Senkou Span B', 'Chikou Span']

        def donchian(n):
            return self.avg(
                self.min(n, input="low", array=True),
                self.max(n, input="high", array=True),
            )

        conversionLine = donchian(tenkan_sen_period)
        baseLine = donchian(kijun_sen_period)
        leadLine1 = self.avg(conversionLine, baseLine)
        leadLine1 = self.shift(n=chikou_span_period - 1, input=leadLine1)
        leadLine2 = donchian(senkou_span_period)
        leadLine2 = self.shift(n=chikou_span_period - 1, input=leadLine2)

        laggingLine = self.shift(
            n=-1 * chikou_span_period + 1, input=self.close
        )  # self.close.shift(-1 * chikou_span_period)
        # print(self.close,'!!!!!!!!!', laggingLine)
        if array:
            return conversionLine, baseLine, leadLine1, leadLine2, laggingLine
        return (
            conversionLine[-1],
            baseLine[-1],
            leadLine1[-1],
            leadLine2[-1],
            laggingLine[-1],
        )

    def supertrend(
        self, period, multiplier: int = 1, array: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[float, str]:
        """
        SuperTrend

        SuperTrend Algorithm :

            BASIC UPPERBAND = (HIGH + LOW) / 2 + Multiplier * ATR
            BASIC LOWERBAND = (HIGH + LOW) / 2 - Multiplier * ATR

            FINAL UPPERBAND = IF( (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous Close > Previous FINAL UPPERBAND))
                                THEN (Current BASIC UPPERBAND) ELSE Previous FINALUPPERBAND)
            FINAL LOWERBAND = IF( (Current BASIC LOWERBAND > Previous FINAL LOWERBAND) or (Previous Close < Previous FINAL LOWERBAND))
                                THEN (Current BASIC LOWERBAND) ELSE Previous FINAL LOWERBAND)

            SUPERTREND = IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close <= Current FINAL UPPERBAND)) THEN
                            Current FINAL UPPERBAND
                        ELSE
                            IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close > Current FINAL UPPERBAND)) THEN
                                Current FINAL LOWERBAND
                            ELSE
                                IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close >= Current FINAL LOWERBAND)) THEN
                                    Current FINAL LOWERBAND
                                ELSE
                                    IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close < Current FINAL LOWERBAND)) THEN
                                        Current FINAL UPPERBAND
        """
        # Args :
        #     df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        #     period : Integer indicates the period of computation in terms of number of candles
        #     multiplier : Integer indicates value to multiply the ATR
        #     ohlc: list defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])

        # Returns :
        #     df : Pandas DataFrame with new columns added for
        #         True Range (TR), ATR (ATR_$period)
        #         SuperTrend (ST_$period_$multiplier)
        #         SuperTrend Direction (STX_$period_$multiplier)

        # ### pandas ta:
        # hl2_ = hl2(high, low)
        # matr = multiplier * atr(high, low, close, length)
        # upperband = hl2_ + matr
        # lowerband = hl2_ - matr

        # for i in range(1, m):
        #     if close.iloc[i] > upperband.iloc[i - 1]:
        #         dir_[i] = 1
        #     elif close.iloc[i] < lowerband.iloc[i - 1]:
        #         dir_[i] = -1
        #     else:
        #         dir_[i] = dir_[i - 1]
        #         if dir_[i] > 0 and lowerband.iloc[i] < lowerband.iloc[i - 1]:
        #             lowerband.iloc[i] = lowerband.iloc[i - 1]
        #         if dir_[i] < 0 and upperband.iloc[i] > upperband.iloc[i - 1]:
        #             upperband.iloc[i] = upperband.iloc[i - 1]

        #     if dir_[i] > 0:
        #         trend[i] = long[i] = lowerband.iloc[i]
        #     else:
        #         trend[i] = short[i] = upperband.iloc[i]

        atr = self.atr(period, array=True)

        # Compute basic upper and lower bands
        basic_ub = (self.high + self.low) / 2 + multiplier * atr
        basic_lb = (self.high + self.low) / 2 - multiplier * atr

        # Compute final upper and lower bands
        final_ub = self.zeros
        final_lb = self.zeros
        for i in range(period, self.size):
            final_ub[i] = (
                basic_ub[i] if basic_ub[i] < final_ub[i - 1] or self.close[i - 1] > final_ub[i - 1] else final_ub[i - 1]
            )
            final_lb[i] = (
                basic_lb[i] if basic_lb[i] > final_lb[i - 1] or self.close[i - 1] < final_lb[i - 1] else final_lb[i - 1]
            )

        # Set the Supertrend value
        st = self.zeros
        for i in range(period, self.size):
            st[i] = (
                final_ub[i]
                if st[i - 1] == final_ub[i - 1] and self.close[i] <= final_ub[i]
                else final_lb[i]
                if st[i - 1] == final_ub[i - 1] and self.close[i] > final_ub[i]
                else final_lb[i]
                if st[i - 1] == final_lb[i - 1] and self.close[i] >= final_lb[i]
                else final_ub[i]
                if st[i - 1] == final_lb[i - 1] and self.close[i] < final_lb[i]
                else 0.00
            )

        # Mark the trend direction up/down
        # stx = np.where((st > 0.00), np.where((self.close < st), 'down',  'up'), np.NaN)
        stx = np.where((st > 0.00), np.where((self.close < st), -1, 1), np.NaN)

        if array:
            return st, stx
        return st[-1], stx[-1]

    def ha(
        self,
        is_simple: bool = False,
        array: bool = False,
        input_open: np.ndarray | None = None,
        input_high: np.ndarray | None = None,
        input_low: np.ndarray | None = None,
        input_close: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[float, float, float, float]:
        """
        Heiken Ashi Candles (HA)
        """

        # Args :
        #     df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        #     ohlc: list defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])

        # Returns :
        #     df : Pandas DataFrame with new columns added for
        #         Heiken Ashi Close (HA_$ohlc[3])
        #         Heiken Ashi Open (HA_$ohlc[0])
        #         Heiken Ashi High (HA_$ohlc[1])
        #         Heiken Ashi Low (HA_$ohlc[2])
        input_open = input_open if input_open is not None else self.open
        input_high = input_high if input_high is not None else self.high
        input_low = input_low if input_low is not None else self.low
        input_close = input_close if input_close is not None else self.close
        ha_close = (input_open + input_high + input_low + input_close) / 4

        if is_simple:
            ha_open = self.zeros
            for i in range(0, self.size):
                if np.isnan(input_open[i]):
                    ha_open[i] = np.nan
                elif i == 0 or np.isnan(input_open[i - 1]):
                    ha_open[i] = (input_open[i] + input_close[i]) / 2
                else:
                    ha_open[i] = (input_open[i - 1] + input_close[i - 1]) / 2

            ha_high = self.high
            ha_low = self.low
        else:
            ha_open = self.zeros
            for i in range(0, self.size):
                if np.isnan(input_open[i]):
                    ha_open[i] = np.nan
                elif i == 0 or np.isnan(input_open[i - 1]):
                    ha_open[i] = (input_open[i] + input_close[i]) / 2
                else:
                    ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
                # if self.size -1 == i:
                #     print(input_open[i],'???????',i, ha_open[i] , ha_close[i], input_high[i], max(ha_open[i] , ha_close[i], input_high[i]))

            ha_high = np.maximum.reduce([ha_open, ha_close, input_high])
            ha_low = np.minimum.reduce([ha_open, ha_close, input_low])

        if array:
            return ha_open, ha_high, ha_low, ha_close
        return ha_open[-1], ha_high[-1], ha_low[-1], ha_close[-1]

    def volume_osc(
        self,
        n_short: int = 1,
        n_long: int = 14,
        array: bool = False,
        input: np.ndarray | None = None,
    ) -> float | np.ndarray:
        """
        Volume Oscillator
        """
        if n_short == 1:
            short: np.ndarray = self._get_input(default=self.volume, input=input)
        else:
            short: np.ndarray = self.ema(
                n_short,
                input=self._get_input(default=self.volume, input=input),
                array=True,
            )
        long: np.ndarray = self.ema(n_long, input=self._get_input(default=self.volume, input=input), array=True)
        osc: np.ndarray = 100 * (short - long) / long
        if array:
            return osc
        return osc[-1]

    def dema(self, n: int, array: bool = False, input: np.ndarray = None) -> float | np.ndarray:
        """
        Double EMA
        """
        # length = input.int(9, minval=1)
        # src = input(close, title="Source")
        # e1 = ta.ema(src, length)
        # e2 = ta.ema(e1, length)
        # dema = 2 * e1 - e2

        e1: np.ndarray = self.ema(n, array=True)
        e2: np.ndarray = self.ema(n, input=e1, array=True)
        dema: np.ndarray = 2 * e1 - e2
        if array:
            return dema
        return dema[-1]

    def chop(self, n: int, array: bool = False, input: np.ndarray = None) -> float | np.ndarray:
        """
        choppiness index

        > 61.8 consolidation
        < 38.2 trending
        """

        # cmi = abs((self.close[-1] - self.close[-n])/(max(self.high[-n:]) - min(self.low[-n:]))) * 100

        # ci = 100 * math.log10(math.sum(ta.atr(1), length) / (ta.highest(length) - ta.lowest(length))) / math.log10(length)
        atr_sum = sum(self.atr(1, array=True)[-n:])
        highest = self.highest(n, input="high")
        lowest = self.lowest(n, input="low")
        # print(f"atr_sum:{atr_sum}; highest:{highest}; lowest:{lowest}")
        numerator = atr_sum / (highest - lowest)
        denominator = math.log10(n)

        cmi = 100 * math.log10(numerator) / denominator

        return cmi

    def qqe(
        self,
        rsi_period: int,
        rsi_smoothing: int,
        qqe_factor: int,
        array: bool = False,
        input: np.ndarray | None = None,
    ):
        """
        Quantitative Qualitative Estimation (QQE)

        The Quantitative Qualitative Estimation (QQE) is similar to SuperTrend but uses a Smoothed RSI with an upper and lower bands. The band width is a combination of a one period True Range of the Smoothed RSI which is double smoothed using Wilder's smoothing length (2 * rsiLength - 1) and multiplied by the default factor of 4.236. A Long trend is determined when the Smoothed RSI crosses the previous upperband and a Short trend when the Smoothed RSI crosses the previous lowerband.

        Based on QQE.mq5 by EarnForex Copyright © 2010, based on version by Tim Hyder (2008), based on version by Roman Ignatov (2006)

        Sources:
            https://www.tradingview.com/script/IYfA9R2k-QQE-MT4/
            https://www.tradingpedia.com/forex-trading-indicators/quantitative-qualitative-estimation
            https://www.prorealcode.com/prorealtime-indicators/qqe-quantitative-qualitative-estimation/

        Calculation:
            Default Inputs:
                length=14, smooth=5, factor=4.236, mamode="ema", drift=1

        Args:
            close (pd.Series): Series of 'close's
            length (int): RSI period. Default: 14
            smooth (int): RSI smoothing period. Default: 5
            factor (float): QQE Factor. Default: 4.236
            mamode (str): See ```help(ta.ma)```. Default: 'sma'
            drift (int): The difference period. Default: 1
            offset (int): How many periods to offset the result. Default: 0

        Kwargs:
            fillna (value, optional): pd.DataFrame.fillna(value)
            fill_method (value, optional): Type of fill method

        Returns:
            pd.DataFrame: QQE, RSI_MA (basis), QQEl (long), and QQEs (short) columns.

        qqe_df = cal_qqe(pd.Series(indicator.close), length=self.rsi_length, smooth=self.rsi_smoothing, factor=self.fast_qqe_factor, ).fillna(0.0)
        # qqe_df = qqe_df if qqe_df is None else qqe_df.fillna(0.0)
        # print(qqe_df,'????')
        # qqe_df = cal_qqe(hist.close, )
        qqe_long = qqe_df[f"QQEl_{self.rsi_length}_{self.rsi_smoothing}_{self.fast_qqe_factor}"].iloc[-1]
        qqe_long = qqe_long - 50 if qqe_long else qqe_long
        qqe_short = qqe_df[f"QQEs_{self.rsi_length}_{self.rsi_smoothing}_{self.fast_qqe_factor}"].iloc[-1]
        qqe_short = qqe_short - 50 if qqe_short else qqe_short

        """
        # rsi_period = 6
        # rsi_smoothing = 5
        # qqe_factor = 3
        # thresh_hold = 3

        wilders_period = rsi_period * 2 - 1
        rsi = self.rsi(rsi_period, array=True)
        rsi_ma = self.ema(input=rsi, n=rsi_smoothing, array=True)
        atr_rsi = np.abs(np.diff(rsi_ma, n=1, prepend=rsi_ma[0]))  # prepend in order to make size the same
        ma_atr_rsi = self.ema(input=atr_rsi, n=wilders_period, array=True)
        dar = self.ema(input=ma_atr_rsi, n=wilders_period, array=True) * qqe_factor

        # print(f"111111 rsi:{rsi}; rsi_ma:{rsi_ma};  rsi_ma_tr:{atr_rsi}; smoothed_rsi_tr_ma:{ma_atr_rsi}; dar:{dar}")
        # print(f"111111 rsi:{rsi[-1]}; rsi_ma:{rsi_ma[-1]};  rsi_ma_tr:{atr_rsi[-1]}; smoothed_rsi_tr_ma:{ma_atr_rsi[-1]}; dar:{dar[-1]} -- {len(atr_rsi)}:{len(rsi_ma)}:{len(dar)}")

        # Create the Upper and Lower Bands around RSI MA.
        upperband = rsi_ma + dar
        lowerband = rsi_ma - dar

        m = self.close.size
        long: np.ndarray = np.zeros(self.size)
        short: np.ndarray = np.zeros(self.size)
        trend: np.ndarray = np.zeros(self.size)
        qqe: np.ndarray = np.zeros(self.size)
        qqe_long: np.ndarray = np.zeros(self.size)
        qqe_short: np.ndarray = np.zeros(self.size)

        for i in range(1, m):
            c_rsi, p_rsi = rsi_ma[i], rsi_ma[i - 1]
            c_long, p_long = long[i - 1], long[i - 2]
            c_short, p_short = short[i - 1], short[i - 2]

            # Long Line
            if p_rsi > c_long and c_rsi > c_long:
                long[i] = np.maximum(c_long, lowerband[i])
            else:
                long[i] = lowerband[i]

            # Short Line
            if p_rsi < c_short and c_rsi < c_short:
                short[i] = np.minimum(c_short, upperband[i])
            else:
                short[i] = upperband[i]

            # Trend & QQE Calculation
            # Long: Current RSI_MA value Crosses the Prior Short Line Value
            # Short: Current RSI_MA Crosses the Prior Long Line Value
            if (c_rsi > c_short and p_rsi < p_short) or (c_rsi <= c_short and p_rsi >= p_short):
                trend[i] = 1
                qqe[i] = qqe_long[i] = long[i]
            elif (c_rsi > c_long and p_rsi < p_long) or (c_rsi <= c_long and p_rsi >= p_long):
                trend[i] = -1
                qqe[i] = qqe_short[i] = short[i]
            else:
                trend[i] = trend[i - 1]
                if trend[i] == 1:
                    qqe[i] = qqe_long[i] = long[i]
                else:
                    qqe[i] = qqe_short[i] = short[i]

        if array:
            return rsi_ma, qqe, qqe_long, qqe_short
        return rsi_ma[-1], qqe[-1], qqe_long[-1], qqe_short[-1]

    def gchannel(self, length: int = 100, array: bool = False, input: np.ndarray = None):
        """
        G-Channel
        https://www.tradingview.com/script/fIvlS64B-G-Channels-Efficient-Calculation-Of-Upper-Lower-Extremities/

        """

        m = self.close.size
        a: np.ndarray = np.zeros(self.size)
        b: np.ndarray = np.zeros(self.size)
        avg: np.ndarray = np.zeros(self.size)

        for i in range(0, m):
            if i:
                a[i] = max(self.close[i], a[i - 1]) - (a[i - 1] - b[i - 1]) / length
                b[i] = min(self.close[i], b[i - 1]) + (a[i - 1] - b[i - 1]) / length
            else:
                a[i] = b[i] = self.close[i]
            avg[i] = (a[i] + b[i]) / 2.0

        # print(f">>> a:{a[-1]}; b:{b[-1]}")
        if array:
            return avg, a, b
        return avg[-1], a[-1], b[-1]

    def av2(
        self,
        ma_period: int,
        ma_period_smoothing: float,
        ma_type: int = 0,
        array: bool = False,
        input: np.ndarray | None = None,
    ):
        """Smoothed Heikin Ashi Cloud
        A-V2: https://www.tradingview.com/script/DTDQ3y76/
        B-V2: https://www.tradingview.com/script/acmdQs2A/

        an overlay showing “Smoothed Heikin Ashi” and measure the momentum of the trend and identify potential trend rejections.

        """

        def f_ma_type(ma_type, input_source, input_ma_period):
            if ma_type == 0:
                return self.ema(n=input_ma_period, input=input_source, array=True)
            elif ma_type == 1:
                return self.sma(n=input_ma_period, input=input_source, array=True)
            elif ma_type == 2:
                return self.wma(n=input_ma_period, input=input_source, array=True)

        ### cal ma
        o = f_ma_type(ma_type, self.open, ma_period)
        c = f_ma_type(ma_type, self.close, ma_period)
        h = f_ma_type(ma_type, self.high, ma_period)
        l = f_ma_type(ma_type, self.low, ma_period)

        # print('--->',o,  c) # h, l,
        # print('--->',o[-1], c[-1], h[-1], l[-1], )
        ### cal Heikin Ashi
        ha_open, ha_high, ha_low, ha_close = self.ha(array=True, input_open=o, input_high=h, input_low=l, input_close=c)
        # print('===>',ha_open[-1], ha_high[-1], ha_low[-1], ha_close[-1])
        ### cal Heikin Ashi ma smoothing
        ha_o_smooth = f_ma_type(ma_type, ha_open, ma_period_smoothing)
        ha_c_smooth = f_ma_type(ma_type, ha_close, ma_period_smoothing)
        ha_h_smooth = f_ma_type(ma_type, ha_high, ma_period_smoothing)
        ha_l_smooth = f_ma_type(ma_type, ha_low, ma_period_smoothing)

        if array:
            return ha_o_smooth, ha_h_smooth, ha_l_smooth, ha_c_smooth

        return ha_o_smooth[-1], ha_h_smooth[-1], ha_l_smooth[-1], ha_c_smooth[-1]

    def alligator(
        self,
        jaw_length: int = 13,
        teeth_length: int = 8,
        lips_length: int = 5,
        jaw_offset: int = 8,
        teeth_offset: int = 5,
        lips_offset: int = 3,
        array: bool = False,
        input: np.ndarray | None = None,
    ):
        jaw = self.smma(jaw_length, array=True, input=input)
        teeth = self.smma(teeth_length, array=True, input=input)
        lips = self.smma(lips_length, array=True, input=input)

        jaw = self.shift(n=jaw_offset, input=jaw)
        teeth = self.shift(n=teeth_offset, input=teeth)
        lips = self.shift(n=lips_offset, input=lips)

        if array:
            return jaw, teeth, lips
        return jaw[-1], teeth[-1], lips[-1]

    def ssl_channel(
        self,
        ma_high_period: int,
        ma_low_period: int,
        array: bool = False,
        input: np.ndarray | None = None,
    ):
        """
        SSL Channel

        """
        sma_high = self.sma(ma_high_period, input="high", array=True)
        sma_low = self.sma(ma_low_period, input="low", array=True)

        m = self.close.size
        Hlv: np.ndarray = np.zeros(self.size)  # high low value
        ssl_down: np.ndarray = np.zeros(self.size)
        ssl_up: np.ndarray = np.zeros(self.size)
        buySignals: np.ndarray = np.zeros(self.size, dtype=bool)
        sellSignals: np.ndarray = np.zeros(self.size, dtype=bool)

        for i in range(1, m):
            hlv = 1 if self.close[i] > sma_high[i] else -1 if self.close[i] < sma_low[i] else Hlv[i - 1]
            Hlv[i] = hlv
            ssl_down[i] = sma_high[i] if hlv < 0 else sma_low[i]
            ssl_up[i] = sma_low[i] if hlv < 0 else sma_high[i]
            buySignals[i] = Hlv[i] == 1 and Hlv[i - 1] == -1 if i else False
            sellSignals[i] = Hlv[i] == -1 and Hlv[i - 1] == 1 if i else False

        if array:
            return ssl_up, ssl_down, Hlv, buySignals, sellSignals
        return ssl_up[-1], ssl_down[-1], Hlv[-1], buySignals[-1], sellSignals[-1]

    def hama(
        self,
        ma_open_period: int = 25,
        ma_high_period: int = 20,
        ma_low_period: int = 20,
        ma_close_period: int = 20,
        ma_type: int = 0,
        array: bool = False,
        input: np.ndarray | None = None,
    ):
        """Heiken-Ashi Moving Average

        https://www.tradingview.com/script/k7nrF2oI-NSDT-HAMA-Candles/

        """

        def f_ma_type(ma_type, input_source, input_ma_period):
            if ma_type == 0:
                return self.ema(n=input_ma_period, input=input_source, array=True)
            elif ma_type == 1:
                return self.sma(n=input_ma_period, input=input_source, array=True)
            elif ma_type == 2:
                return self.wma(n=input_ma_period, input=input_source, array=True)

        ### cal Heikin Ashi
        ha_open, ha_high, ha_low, ha_close = self.ha(array=True, is_simple=True)
        # print('===>',ha_open[-1], ha_high[-1], ha_low[-1], ha_close[-1])

        ### cal Heikin Ashi ma smoothing
        ha_o_smooth = f_ma_type(ma_type, ha_open, ma_open_period)
        ha_h_smooth = f_ma_type(ma_type, ha_high, ma_high_period)
        ha_l_smooth = f_ma_type(ma_type, ha_low, ma_low_period)
        ha_c_smooth = f_ma_type(ma_type, ha_close, ma_close_period)

        # wma_array = self.wma(n=55, array=True)
        # ema_array = self.ema(n=3, input=wma_array, array=True)
        # _source = wma_array
        # _center = ema_array
        # _xUp     = self.crossover(_source, _center)
        # _xDn     = self.crossunder(_source, _center)
        # _chg     = self.change(_source)
        # _up      = _chg > 0
        # _dn      = _chg < 0
        # _srcBull = _source > _center
        # _srcBear = _source < _center
        # res = None
        # if _srcBull:
        #     if _xUp:
        #         res = 1
        #     else:
        #         if _up:
        #             s = min(5, res+1)
        #         else:
        #             if _dn:
        #                 res = max(1, res- 1)
        #             else:
        #                 pass

        # else:
        #     if _srcBear:
        #         if _xDn:
        #             res = 1
        #         else:
        #             if _dn:
        #                 res = min(5, res +1)
        #             else:
        #                 if _up:
        #                     res = max(1, res -1)
        #                 else:
        #                     res = res
        #     else:
        #         pass
        # print(res)

        if array:
            return ha_o_smooth, ha_h_smooth, ha_l_smooth, ha_c_smooth

        return ha_o_smooth[-1], ha_h_smooth[-1], ha_l_smooth[-1], ha_c_smooth[-1]

    def chandelier_exit(
        self,
        atr_length: int,
        atr_multiplier: float,
        array: bool = False,
        input: np.ndarray | None = None,
    ):
        """
        Chandelier Exit

        longStop = (useClose ? highest(close, length) : highest(length)) - atr
        longStopPrev = nz(longStop[1], longStop)
        longStop[i] = close[1] > longStopPrev ? max(longStop, longStopPrev) : longStop

        shortStop = (useClose ? lowest(close, length) : lowest(length)) + atr
        shortStopPrev = nz(shortStop[1], shortStop)
        shortStop[i] = close[1] < shortStopPrev ? min(shortStop, shortStopPrev) : shortStop

        var int dir = 1
        dir[i] = close > shortStopPrev ? 1 : close < longStopPrev ? -1 : dir

        buySignal = dir == 1 and dir[1] == -1
        sellSignal = dir == -1 and dir[1] == 1

        buypx = buySignal ? longStop : na
        sellpx = sellSignal ? shortStop : na

        """
        atr = atr_multiplier * self.atr(n=atr_length, array=True)
        # print(f"atr: {atr[-1]}")

        longStop = self.max(n=atr_length, array=True) - atr
        # longStopPrev = self.shift(n=1, input=longStop)
        # print(f"longStop: {longStop[-1]}; longStopPrev: {longStopPrev[-1]}")

        # longStop = max(longStop[-1], longStopPrev[-1]) if self.close[-1] > longStopPrev[-1] else longStop[-1]
        # longStop_ = np.where(self.close > longStopPrev, max(longStop, longStopPrev), longStop)

        shortStop = self.min(n=atr_length, array=True) + atr
        # shortStopPrev = self.shift(n=1, input=shortStop)
        # shortStop =  min(shortStop[-1], shortStopPrev[-1]) if self.close[-1] < shortStopPrev[-1] else shortStop[-1]
        # shortStop_ = np.where(self.close < shortStopPrev, min(shortStop, shortStopPrev), shortStop)

        # dir = 1 if self.close[-1] > shortStopPrev[-1] else -1 if self.close[-1] < longStopPrev[-1] else 1
        # dir = np.where(self.close > shortStopPrev, 1, np.where(self.close < longStopPrev, -1, dir))

        m = self.close.size
        dirs: np.ndarray = np.zeros(self.size)
        longStops: np.ndarray = np.zeros(self.size)
        shortStops: np.ndarray = np.zeros(self.size)
        buySignals: np.ndarray = np.zeros(self.size, dtype=bool)
        sellSignals: np.ndarray = np.zeros(self.size, dtype=bool)

        for i in range(0, m):
            longStopPrev = longStops[i - 1] if i else longStops[i]
            shortStopPrev = shortStops[i - 1] if i else shortStops[i]

            longStops[i] = max(longStop[i], longStopPrev) if self.close[i] > longStopPrev else longStop[i]
            shortStops[i] = min(shortStop[i], shortStopPrev) if self.close[i] < shortStopPrev else shortStop[i]
            dirs[i] = 1 if self.close[i] > shortStopPrev else -1 if self.close[i] < longStopPrev else dirs[i - 1]
            buySignals[i] = dirs[i] == 1 and dirs[i - 1] == -1 if i else False
            sellSignals[i] = dirs[i] == -1 and dirs[i - 1] == 1 if i else False
        # print(f"longStop: {longStop[-1]}; longStopPrev: {longStopPrev}")

        # buySignal = dirs[-1] == 1 and dirs[-2] == -1
        # sellSignal = dirs[-1] == -1 and dirs[-2] == 1

        # print(f"dir:{dir}; longStop:{longStop}; shortStop:{shortStop}")
        # print(f"buySignal:{buySignal}; sellSignal:{sellSignal}; ")

        if array:
            return longStops, shortStops, buySignals, sellSignals, dirs
        return longStops[-1], shortStops[-1], buySignals[-1], sellSignals[-1], dirs[-1]

    def zlsma(self, n: int, offset: int = 0, array: bool = False, input: np.ndarray = None):
        """
        Zero Lag Least Squared Moving Average
        """
        src: np.ndarray = self._get_input(default=self.close, input=input)

        lsma: np.ndarray = self.talib.LINEARREG(src, n)
        if offset:
            lsma: np.ndarray = self.shift(n=offset, input=lsma)
        lsma2: np.ndarray = self.talib.LINEARREG(lsma, n)
        if offset:
            lsma2: np.ndarray = self.shift(n=offset, input=lsma2)
        eq: np.ndarray = lsma - lsma2
        zlsma: np.ndarray = lsma + eq
        if array:
            return zlsma
        return zlsma[-1]

    def crossover(self, source1: np.ndarray, source2: np.ndarray) -> bool:
        return True if source1[-1] > source2[-1] and source1[-2] < source2[-2] else False

    def crossunder(self, source1: np.ndarray, source2: np.ndarray) -> bool:
        return True if source1[-1] < source2[-1] and source1[-2] > source2[-2] else False

    def change(self, source: np.ndarray) -> float:
        return (source[-1] - source[-2]) / source[-2]

    def nz(self, source1: float, source2: float) -> float:
        return source1 if source1 == source1 else source2

    def na(self, source: float) -> bool:
        return source != source

    def ut_bot(
        self,
        dev: int,
        atr_period: int = 0,
        array: bool = False,
        input: np.ndarray | None = None,
    ):
        """
        https://cn.tradingview.com/script/n8ss8BID-UT-Bot-Alerts/
        """
        xATR = self.atr(n=atr_period, array=True)
        nLoss = dev * xATR

        m = self.close.size
        xATRTrailingStop: np.ndarray = np.zeros(self.size)
        buySignals: np.ndarray = np.zeros(self.size, dtype=bool)
        sellSignals: np.ndarray = np.zeros(self.size, dtype=bool)
        close = self.close

        for i in range(0, m):
            if close[i] > xATRTrailingStop[i - 1] and close[i - 1] > xATRTrailingStop[i - 1]:
                xATRTrailingStop[i] = max(xATRTrailingStop[i - 1], close[i] - nLoss[i])
            elif close[i] < xATRTrailingStop[i - 1] and close[i - 1] < xATRTrailingStop[i - 1]:
                xATRTrailingStop[i] = min(xATRTrailingStop[i - 1], close[i] + nLoss[i])
            else:
                if close[i] > xATRTrailingStop[i - 1]:
                    xATRTrailingStop[i] = close[i] - nLoss[i]
                else:
                    xATRTrailingStop[i] = close[i] + nLoss[i]

            above = self.crossover(close, xATRTrailingStop)
            below = self.crossover(xATRTrailingStop, close)

            buySignals[i] = 1 if close[i] > xATRTrailingStop[i] and above else 0
            sellSignals[i] = 1 if close[i] < xATRTrailingStop[i] and below else 0

        if array:
            return buySignals, sellSignals, xATRTrailingStop
        return buySignals[-1], sellSignals[-1], xATRTrailingStop[-1]

    def ut_bot_chatgpt(
        self,
        dev: int = 1,
        atr_period: int = 10,
        array: bool = False,
        input: np.ndarray | None = None,
    ):
        a = dev
        c = atr_period
        h = False

        df = pd.DataFrame({"Close": self.close, "High": self.high, "Low": self.low, "Open": self.open})

        df["xATR"] = self.talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=c)
        df["nLoss"] = a * df["xATR"]

        src = df["Close"] if not h else self.talib.HeikinAshi(df["Open"], df["High"], df["Low"], df["Close"])

        df["xATRTrailingStop"] = 0.0
        for i in range(1, len(df)):
            if (
                src.iloc[i] > df["xATRTrailingStop"].iloc[i - 1]
                and src.iloc[i - 1] > df["xATRTrailingStop"].iloc[i - 1]
            ):
                df["xATRTrailingStop"].iloc[i] = max(
                    df["xATRTrailingStop"].iloc[i - 1],
                    src.iloc[i] - df["nLoss"].iloc[i],
                )
            elif (
                src.iloc[i] < df["xATRTrailingStop"].iloc[i - 1]
                and src.iloc[i - 1] < df["xATRTrailingStop"].iloc[i - 1]
            ):
                df["xATRTrailingStop"].iloc[i] = min(
                    df["xATRTrailingStop"].iloc[i - 1],
                    src.iloc[i] + df["nLoss"].iloc[i],
                )
            elif src.iloc[i] > df["xATRTrailingStop"].iloc[i - 1]:
                df["xATRTrailingStop"].iloc[i] = src.iloc[i] - df["nLoss"].iloc[i]
            else:
                df["xATRTrailingStop"].iloc[i] = src.iloc[i] + df["nLoss"].iloc[i]

        pos = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if (
                src.iloc[i - 1] < df["xATRTrailingStop"].iloc[i - 1]
                and src.iloc[i] > df["xATRTrailingStop"].iloc[i - 1]
            ):
                pos.iloc[i] = 1
            elif (
                src.iloc[i - 1] > df["xATRTrailingStop"].iloc[i - 1]
                and src.iloc[i] < df["xATRTrailingStop"].iloc[i - 1]
            ):
                pos.iloc[i] = -1
            else:
                pos.iloc[i] = pos.iloc[i - 1]
        df["xcolor"] = np.where(pos == -1, "red", np.where(pos == 1, "green", "blue"))
        # print(src,'????src')
        # df["ema"] = self.talib.EMA(src, timeperiod=1)
        df["ema"] = src
        df["above"] = (df["ema"] > df["xATRTrailingStop"]).shift(1)
        df["below"] = (df["xATRTrailingStop"] > df["ema"]).shift(1)

        df["buy"] = (src > df["xATRTrailingStop"]) & df["above"]
        df["sell"] = (src < df["xATRTrailingStop"]) & df["below"]

        df["barbuy"] = src > df["xATRTrailingStop"]
        df["barsell"] = src < df["xATRTrailingStop"]
        return df
        # if array:
        #     return buySignals, sellSignals, xATRTrailingStop
        # return buySignals[-1], sellSignals[-1], xATRTrailingStop[-1]

    def stc_chatgpt(
        self,
        period: int = 12,
        fast_period: int = 26,
        slow_period: int = 50,
        multiplier: float = 0.5,
        array: bool = False,
        input: np.ndarray | None = None,
    ):
        EEEEEE = period
        BBBB = fast_period
        BBBBB = slow_period
        AAA = multiplier

        close = self.close

        # def AAAA(BBB, BBBB, BBBBB):
        #     fastMA = self.talib.EMA(BBB, timeperiod=BBBB)
        #     slowMA = self.talib.EMA(BBB, timeperiod=BBBBB)
        #     AAAA = fastMA - slowMA
        #     return AAAA

        # def AAAAA(EEEEEE, BBBB, BBBBB):
        #     CCCCC = 0.0
        #     DDD = 0.0
        #     DDDDDD = 0.0
        #     EEEEE = 0.0
        #     # close = np.random.rand(100) # replace with actual close prices
        #     BBBBBB = AAAA(close, BBBB, BBBBB)
        #     CCC = self.talib.MIN(BBBBBB, timeperiod=EEEEEE)
        #     CCCC = self.talib.MAX(BBBBBB, timeperiod=EEEEEE) - CCC
        #     CCCCC = np.where(CCCC > 0, (BBBBBB - CCC) / CCCC * 100, np.roll(CCCCC, 1))
        #     DDD = np.where(np.isnan(np.roll(DDD, 1)), CCCCC, DDD[1] + AAA * (CCCCC - DDD[1]))
        #     DDDD = self.talib.MIN(DDD, timeperiod=EEEEEE)
        #     DDDDD = self.talib.MAX(DDD, timeperiod=EEEEEE) - DDDD
        #     DDDDDD = np.where(DDDDDD > 0, (DDD - DDDD) / DDDDD * 100, np.roll(DDDDDD, 1))
        #     EEEEE = np.where(np.isnan(np.roll(EEEEE, 1)), DDDDDD, EEEEE[1] + AAA * (DDDDDD - EEEEE[1]))
        #     return EEEEE

        # def AAAA(BBB, BBBB, BBBBB):
        #     fastMA = self.talib.EMA(BBB, BBBB)
        #     slowMA = self.talib.EMA(BBB, BBBBB)
        #     return fastMA - slowMA

        # # define function AAAAA
        # def AAAAA(EEEEEE, BBBB, BBBBB):
        #     CCCCC = np.zeros_like(close)
        #     CCCC = np.zeros_like(close)
        #     DDD = np.zeros_like(close)
        #     DDDDD = np.zeros_like(close)
        #     DDDDDD = np.zeros_like(close)
        #     EEEEE = np.zeros_like(close)
        #     BBBBBB = AAAA(close, BBBB, BBBBB)
        #     CCC = self.talib.MIN(BBBBBB, EEEEEE)
        #     CCCC[EEEEEE-1:] = np.maximum(BBBBBB[EEEEEE-1:], self.talib.MAX(BBBBBB, EEEEEE) - CCC)
        #     CCCCC[EEEEEE-1:] = np.where(CCCC[EEEEEE-1:] > 0,
        #                                 (BBBBBB[EEEEEE-1:] - CCC[EEEEEE-1:]) / CCCC[EEEEEE-1:] * 100,
        #                                 np.nan)
        #     DDD[EEEEEE-1] = CCCCC[EEEEEE-1]
        #     for i in range(EEEEEE, len(close)):
        #         DDD[i] = DDD[i-1] + AAA * (CCCCC[i] - DDD[i-1])
        #     DDDD = self.talib.MIN(DDD, EEEEEE)
        #     DDDDD[EEEEEE-1:] = np.maximum(DDD[EEEEEE-1:], self.talib.MAX(DDD, EEEEEE) - DDDD)
        #     DDDDDD[EEEEEE-1:] = np.where(DDDDDD[EEEEEE-1:] > 0,
        #                                 (DDD[EEEEEE-1:] - DDDD[EEEEEE-1:]) / DDDDD[EEEEEE-1:] * 100,
        #                                 np.nan)
        #     EEEEE[EEEEEE-1] = DDDDDD[EEEEEE-1]
        #     for i in range(EEEEEE, len(close)):
        #         EEEEE[i] = EEEEE[i-1] + AAA * (DDDDDD[i] - EEEEE[i-1])
        #     return EEEEE
        # mAAAAA = AAAAA(EEEEEE, BBBB, BBBBB)

        # # variables to be updated in the script
        # CCCCC = 0.0
        # DDD = 0.0
        # DDDDDD = 0.0
        # mAAAAA = 0.0

        # # BBBBBB = AAAA(close, BBBB, BBBBB)
        # fastMA = np.convolve(close, np.ones(BBBB)/BBBBB, mode='valid')
        # slowMA = np.convolve(close, np.ones(BBBBB)/BBBBB, mode='valid')
        # BBBBBB = fastMA - slowMA

        # CCC = np.min(BBBBBB[-EEEEEE:])
        # CCCC = np.max(BBBBBB[-EEEEEE:]) - CCC
        # CCCCC = np.where(CCCC > 0, (BBBBBB - CCC) / CCCC * 100, np.roll(CCCCC, 1))
        # DDD = np.where(np.isnan(DDD[1]), CCCCC, DDD[1] + AAA * (CCCCC - DDD[1]))
        # DDDD = np.min(DDD[-EEEEEE:])
        # DDDDD = np.max(DDD[-EEEEEE:]) - DDDD
        # DDDDDD = np.where(DDDDD > 0, (DDD - DDDD) / DDDDD * 100, np.roll(DDDDDD, 1))
        # mAAAAA = np.where(np.isnan(mAAAAA[1]), DDDDDD, mAAAAA[1] + AAA * (DDDDDD - mAAAAA[1]))

        fastMA = self.talib.EMA(close, BBBB)
        slowMA = self.talib.EMA(close, BBBBB)
        BBBBBB = fastMA - slowMA

        # CCC = self.highest(n=EEEEEE, input=BBBBBB, array=True)
        # CCCC = self.lowest(n=EEEEEE, input=BBBBBB, array=True) - CCC
        CCC = self.talib.MIN(BBBBBB, EEEEEE)
        CCCC = self.talib.MAX(BBBBBB, EEEEEE) - CCC
        CCCCC = np.where(CCCC > 0, (BBBBBB - CCC) / CCCC * 100, np.nan)
        DDD = np.empty_like(close)
        DDD[0] = CCCCC[0]
        for i in range(1, len(close)):
            if np.isnan(DDD[i - 1]):
                DDD[i] = CCCCC[i]
            else:
                DDD[i] = DDD[i - 1] + AAA * (CCCCC[i] - DDD[i - 1])
        # print(f"CCCmin:{CCC}; CCCCmax:{CCCC}; \nCCCCC:{CCCCC}\nDDD: {DDD}")
        DDDD = self.talib.MIN(DDD, EEEEEE)
        DDDDD = self.talib.MAX(DDD, EEEEEE) - DDDD
        DDDDDD = np.where(DDDDD > 0, (DDD - DDDD) / DDDDD * 100, np.nan)
        mAAAAA = np.empty_like(close)
        direction = np.empty_like(close)
        mAAAAA[0] = DDDDDD[0]
        for i in range(1, len(close)):
            if np.isnan(mAAAAA[i - 1]):
                mAAAAA[i] = DDDDDD[i]
            else:
                mAAAAA[i] = mAAAAA[i - 1] + AAA * (DDDDDD[i] - mAAAAA[i - 1])
            direction[i] = 1 if mAAAAA[i] > mAAAAA[i - 1] else 0

        # print(f"\nBBBBBB(ma_diff):{BBBBBB[-1]}; CCC(min_ma_diff):{CCC[-1]}; CCCC(minmax_diff_ma_diff):{CCCC[-1]}; CCCCC:{CCCCC[-1]}; DDD:{DDD[-1]}; DDDD:{DDDD[-1]}; DDDDD:{DDDDD[-1]}; DDDDDD:{DDDDDD[-1]}")

        if array:
            return mAAAAA, direction
        return mAAAAA[-1], direction[-1]

    def stc(
        self,
        period: int = 12,
        fast_period: int = 26,
        slow_period: int = 50,
        multiplier: float = 0.5,
        array: bool = False,
        input: np.ndarray | None = None,
    ):
        """
        https://cn.tradingview.com/script/WhRRThMI-STC-Indicator-A-Better-MACD-SHK/

        AAA = input(0.5)
        var CCCCC = 0.0
        var DDD = 0.0
        var DDDDDD = 0.0
        var mAAAAA = 0.0

        fastMA = ta.ema(close, BBBB)
        slowMA = ta.ema(close, BBBBB)
        BBBBBB = fastMA - slowMA

        CCC = ta.lowest(BBBBBB, EEEEEE)
        CCCC = ta.highest(BBBBBB, EEEEEE) - CCC
        CCCCC := CCCC > 0 ? (BBBBBB - CCC) / CCCC * 100 : nz(CCCCC[1])
        DDD := na(DDD[1]) ? CCCCC : DDD[1] + AAA * (CCCCC - DDD[1])
        DDDD = ta.lowest(DDD, EEEEEE)
        DDDDD = ta.highest(DDD, EEEEEE) - DDDD
        DDDDDD := DDDDD > 0 ? (DDD - DDDD) / DDDDD * 100 : nz(DDDDDD[1])
        mAAAAA := na(mAAAAA[1]) ? DDDDDD : mAAAAA[1] + AAA * (DDDDDD - mAAAAA[1])


        mColor = mAAAAA > mAAAAA[1] ? color.new(color.green, 20) : color.new(color.red, 20)

        """

        fastMA = self.ema(fast_period, array=True)
        slowMA = self.ema(slow_period, array=True)
        ma_diff = fastMA - slowMA
        min_ma_diff = self.min(period, input=ma_diff, array=True)
        minmax_diff_ma_diff = self.max(period, input=ma_diff, array=True) - min_ma_diff

        m = self.close.size
        CCCCC: np.ndarray = np.zeros(self.size)
        DDD: np.ndarray = np.zeros(self.size)
        DDD[:] = np.nan
        DDDDDD: np.ndarray = np.zeros(self.size)
        stc: np.ndarray = np.zeros(self.size)
        direction: np.ndarray = np.zeros(self.size)

        for i in range(1, m):
            CCCCC[i] = (
                (ma_diff[i] - min_ma_diff[i]) / minmax_diff_ma_diff[i] * 100
                if minmax_diff_ma_diff[i] > 0
                else CCCCC[i - 1]
            )
            if not (i - 1):
                DDD[i - 1] = CCCCC[i]
            DDD[i] = CCCCC[i] if self.na(DDD[i - 1]) else DDD[i - 1] + multiplier * (CCCCC[i] - DDD[i - 1])
            # DDD[DDD == 0] = np.nan
            # allnan = np.isnan(DDD).all()
            DDDD = self.min(
                period, input=DDD
            )  # NOTE: since the first few items of return from min, max function are nan, so this require larger initial size of indicator
            DDDDD = self.max(period, input=DDD) - DDDD
            DDDDDD[i] = (DDD[i] - DDDD) / DDDDD * 100 if DDDDD > 0 else DDDDDD[i - 1]
            stc[i] = DDDDDD[i] if stc[i - 1] else stc[i - 1] + multiplier * (DDDDDD[i] - stc[i - 1])
            direction[i] = 1 if stc[i] > stc[i - 1] else 0
        # print(f"\nBBBBBB(ma_diff):{ma_diff[-1]}; CCC(min_ma_diff):{min_ma_diff[-1]}; CCCC(minmax_diff_ma_diff):{minmax_diff_ma_diff[-1]}; CCCCC:{CCCCC[-1]}; DDD:{DDD[-1]}; DDDD:{DDDD}; DDDDD:{DDDDD}; DDDDDD:{DDDDDD[-1]}")

        if array:
            return stc, direction
        return stc[-1], direction[-1]

    def rh_overlay_set(self, array: bool = False, input: np.ndarray = None):
        """
        https://cn.tradingview.com/script/al4ABhwU-Rob-Hoffman-Overlay-Set/

        plot(a, title = "Fast Speed Line", linewidth = 2, color = #0000FF)
        plot(b, title = "Slow Speed Line", linewidth = 2, color = fuchsia)
        plot(c, title = "Fast Primary Trend Line", linewidth = 3, color = #00FF00)
        plot(d, title = "Slow Primary Trend Line", linewidth = 3, color = #000000)
        plot(e, title = "Trend Line - 1", linewidth = 3, color = #0000FF, style = circles)
        plot(f, title = "Trend Line - 2", linewidth = 3, color = #20B2AA)
        plot(g, title = "Trend Line - 3", linewidth = 3, color = #FF4500)
        plot(h, title = "Trend Line - 4", linewidth = 3, color = fuchsia)

        plot(k, title = "No Trend Zone - Midline", linewidth = 2, color = #3CB371)
        plot(ku, title = "No Trend Zone - Upperline", linewidth = 2, color = #3CB371)
        plot(kl, title = "No Trend Zone - Lowerline", linewidth = 2, color = #3CB371)

        """

        a = self.sma(n=3, array=True)  # Fast Speed Line
        b = self.sma(n=5, array=True)  # Slow Speed Line

        c = self.ema(n=18, array=True)  # Fast Primary Trend Line
        d = self.ema(n=20, array=True)  # Slow Primary Trend Line

        e = self.sma(n=50, array=True)  # Trend Line - 1
        f = self.sma(n=89, array=True)  # Trend Line - 2
        g = self.ema(n=144, array=True)  # Trend Line - 3
        h = self.sma(n=200, array=True)  # Trend Line - 4

        k = self.ema(n=35, array=True)  # No Trend Zone - Midline
        r = self.rma(n=35, input=self.tr(array=True), array=True)
        ku = k + r * 0.5  # No Trend Zone - Upperline
        kl = k - r * 0.5  # No Trend Zone - Lowerline
        if array:
            return a, b, c, d, e, f, g, h, k, r, ku, kl
        return (
            a[-1],
            b[-1],
            c[-1],
            d[-1],
            e[-1],
            f[-1],
            g[-1],
            h[-1],
            k[-1],
            r[-1],
            ku[-1],
            kl[-1],
        )

    def rh_inventory_retracement_bar(self, z: int = 45, array: bool = False, input: np.ndarray = None):
        """
        https://cn.tradingview.com/script/VNByUGpE-Rob-Hoffman-s-Inventory-Retracement-Bar-by-UCSgears/

        z: Inventory Retracement Percentage %
        """

        # Candle Range
        a = abs(self.high - self.low)
        # Candle Body
        b = abs(self.close - self.open)
        # Percent to Decimal
        c = z / 100

        m = self.close.size
        rv: np.ndarray = np.zeros(self.size)
        x: np.ndarray = np.zeros(self.size)
        y: np.ndarray = np.zeros(self.size)
        sl: np.ndarray = np.zeros(self.size, dtype=bool)
        ss: np.ndarray = np.zeros(self.size, dtype=bool)
        li: np.ndarray = np.zeros(self.size)

        for i in range(1, m):
            # Range Verification
            rv[i] = b[i] < c * a[i]

            # Price Level for Retracement
            x[i] = self.low[i] + (c * a[i])
            y[i] = self.high[i] - (c * a[i])

            sl[i] = rv[i] == 1 and self.high[i] > y[i] and self.close[i] < y[i] and self.open[i] < y[i]
            ss[i] = rv[i] == 1 and self.low[i] < x[i] and self.close[i] > x[i] and self.open[i] > x[i]

            # Line Definition - Inventory Bar Retracement Price Line
            li[i] = y[i] if sl[i] else x[i] if ss[i] else (x[i] + y[i]) / 2

        if array:
            return sl, ss, li
        return sl[-1], ss[-1], li[-1]

    def pivothigh(self, left, right=0):
        """
        https://blog.csdn.net/sumubaiblog/article/details/122095719
        通道裡面的最低點跟最高點
        在一段震盪行情的高點與低點, RSI 背離
        """
        right = right if right else left
        pivot: np.ndarray = np.zeros(self.size)
        for i in range(self.size):
            if i >= left + right:
                rolling = self.high[i - right - left : i + 1]
                m = max(rolling)
                if self.high[i - right] == m:
                    pivot[i] = m
        return pivot

    def pivotlow(self, left, right=0):
        right = right if right else left
        pivot: np.ndarray = np.zeros(self.size)
        for i in range(self.size):
            if i >= left + right:
                rolling = self.low[i - right - left : i + 1]
                m = min(rolling)
                if self.low[i - right] == m:
                    pivot[i] = m
        return pivot

    def trendlines_with_breaks(
        self,
        length: int = 14,
        slope: float = 1,
        method: str = "atr",
        array: bool = False,
        input: np.ndarray | None = None,
    ):
        """
        indicator("Trendlines with Breaks [LuxAlgo]",overlay=true)
        length = input.int(14)
        k      = input.float(1.,'Slope',minval=0,step=.1)
        method = input.string('Atr','Slope Calculation Method',
        options=['Atr','Stdev','Linreg'])
        show   = input(false,'Show Only Confirmed Breakouts')
        //----
        upper = 0.,lower = 0.
        slope_ph = 0.,slope_pl = 0.
        src = close
        n = bar_index
        //----
        ph = ta.pivothigh(length,length)
        pl = ta.pivotlow(length,length)
        plot(ph,'pivothigh',color =color.white)
        plot(pl,'pivotlow',color =color.white)

        slope = switch method
            'Atr'      => ta.atr(length)/length*k
            'Stdev'    => ta.stdev(src,length)/length*k
            'Linreg'   => math.abs(ta.sma(src*bar_index,length)-ta.sma(src,length)*ta.sma(bar_index,length))/ta.variance(n,length)/2*k

        slope_ph := ph ? slope : slope_ph[1]
        slope_pl := pl ? slope : slope_pl[1]

        upper := ph ? ph : upper[1] - slope_ph
        lower := pl ? pl : lower[1] + slope_pl
        //----
        single_upper = 0
        single_lower = 0
        single_upper := src[length] > upper ? 0 : ph ? 1 : single_upper[1]
        single_lower := src[length] < lower ? 0 : pl ? 1 : single_lower[1]
        upper_breakout = single_upper[1] and src[length] > upper and (show ? src > src[length] : 1)
        lower_breakout = single_lower[1] and src[length] < lower and (show ? src < src[length] : 1)
        plotshape(upper_breakout ? low[length] : na,"Upper Break",shape.labelup,location.absolute,#26a69a,-length,text="B",textcolor=color.white,size=size.tiny)
        plotshape(lower_breakout ? high[length] : na,"Lower Break",shape.labeldown,location.absolute,#ef5350,-length,text="B",textcolor=color.white,size=size.tiny)
        //----
        var line up_l = na
        var line dn_l = na
        var label recent_up_break = na
        var label recent_dn_break = na

        if ph[1]
            line.delete(up_l[1])
            label.delete(recent_up_break[1])

            up_l := line.new(n-length-1,ph[1],n-length,upper,color=#26a69a,
            extend=extend.right,style=line.style_dashed)
        if pl[1]
            line.delete(dn_l[1])
            label.delete(recent_dn_break[1])

            dn_l := line.new(n-length-1,pl[1],n-length,lower,color=#ef5350,
            extend=extend.right,style=line.style_dashed)

        if ta.crossover(src,upper-slope_ph*length)
            label.delete(recent_up_break[1])
            recent_up_break := label.new(n,low,'B',color=#26a69a,
            textcolor=color.white,style=label.style_label_up,size=size.small)

        if ta.crossunder(src,lower+slope_pl*length)
            label.delete(recent_dn_break[1])
            recent_dn_break := label.new(n,high,'B',color=#ef5350,
            textcolor=color.white,style=label.style_label_down,size=size.small)

        //----
        plot(upper,'Upper',color = ph ? na : #26a69a,offset=-length)
        plot(lower,'Lower',color = pl ? na : #ef5350,offset=-length)

        alertcondition(ta.crossover(src,upper-slope_ph*length),'Upper Breakout','Price broke upper trendline')
        alertcondition(ta.crossunder(src,lower+slope_pl*length),'Lower Breakout','Price broke lower trendline')

        """
        ph = self.pivothigh(length, length)
        pl = self.pivotlow(length, length)
        # slope = switch method
        #     'Atr'      => ta.atr(length)/length*k
        #     'Stdev'    => ta.stdev(src,length)/length*k
        #     'Linreg'   => math.abs(ta.sma(src*bar_index,length)-ta.sma(src,length)*ta.sma(bar_index,length))/ta.variance(n,length)/2*k
        assert method in ["atr", "stdev", "linreg"]
        if method == "atr":
            slopes = self.atr(n=length, array=True) / length * slope
        elif method == "stdev":
            slopes = self.std(n=length, array=True) / length * slope
        else:
            # 'Linreg'   => math.abs(ta.sma(src*bar_index,length)-ta.sma(src,length)*ta.sma(bar_index,length))/ta.variance(n,length)/2*k
            slopes = np.zeros(self.size)

        m = self.close.size
        upper: np.ndarray = np.zeros(self.size)
        lower: np.ndarray = np.zeros(self.size)
        slope_ph: np.ndarray = np.zeros(self.size)
        slope_pl: np.ndarray = np.zeros(self.size)
        # single_upper: np.ndarray = np.zeros(self.size)
        # single_lower: np.ndarray = np.zeros(self.size)

        for i in range(1, m):
            slope_ph[i] = slopes[i] if ph[i] else slope_ph[i - 1]
            slope_pl[i] = slopes[i] if pl[i] else slope_pl[i - 1]
            upper[i] = ph[i] if ph[i] else upper[i - 1] - slope_ph[i]
            lower[i] = pl[i] if pl[i] else lower[i - 1] + slope_pl[i]

        # up_break = self.crossover(self.close,upper-slope_ph*length)
        # dn_break = self.crossunder(self.close,lower+slope_pl*length)

        if array:
            return ph, pl, upper, lower
        return ph[-1], pl[-1], upper[-1], lower[-1]

    def heatmap_volume(
        self,
        ma_length: int = 610,
        std_length: int = 610,
        extra_high_threshold: float = 4,
        high_threshold: float = 2.5,
        medium_threshold: float = 1,
        normal_threshold: float = -0.5,
        array: bool = False,
        input: np.ndarray | None = None,
    ):
        """
        extra_high
        high
        medium
        normal
        low


        length := length > bar_index + 1 ? bar_index + 1 : length
        slength := slength > bar_index + 1 ? bar_index + 1 : slength


        pstdev(Series, Period) =>
            mean = sum(Series, Period) / Period
            summation = 0.0
            for i=0 to Period-1
                sampleMinusMean = nz(Series[i]) - mean
                summation := summation + sampleMinusMean * sampleMinusMean
            return = sqrt(summation / Period)


        mean    = sma(volume, length)
        std     = pstdev(volume, slength)
        stdbar  = (volume - mean) / std
        dir     = close > open
        v       = osc ? volume - mean : volume
        mosc    = osc ? 0 : mean


        bcolor = stdbar > thresholdExtraHigh ? dir ? cthresholdExtraHighUp : cthresholdExtraHighDn :
        stdbar > thresholdHigh  ? dir ? cthresholdHighUp : cthresholdHighDn :
        stdbar > thresholdMedium ? dir ? cthresholdMediumUp : cthresholdMediumDn :
        stdbar > thresholdNormal ? dir ? cthresholdNormalUp : cthresholdNormalDn :
        dir ? cthresholdLowUp : cthresholdLowDn

        """
        # def pstdev(series, period):
        #     mean = self.talib.SMA(series, period)
        #     summation = 0.0
        #     for i in range(period):
        #         sampleMinusMean = series[i] - mean[i]
        #         summation = summation + sampleMinusMean * sampleMinusMean
        #     return np.sqrt(summation / period)

        # mean = self.sma(n=ma_length, input=self.volume, array=True)
        # # std =self.std(n=std_length, input=self.volume, array=True)#
        # std = pstdev(self.volume, std_length)#
        # print(f"std:{std}")

        # m = self.close.size
        # stdbar: np.ndarray = np.zeros(self.size)
        # bcolor: np.ndarray = np.zeros(self.size, dtype=str)

        # for i in range(1, m):
        #     stdbar[i]  = (self.volume[i] - mean[i]) / std[i]
        #     # dir = self.close[i] > self.open[i]
        #     bcolor[i] = 'extra_high' if stdbar[i] > extra_high_threshold else 'high' if stdbar[i] > high_threshold else 'medium' if stdbar[i] > medium_threshold else 'normal' if stdbar[i] > normal_threshold else 'low'
        # # if array:
        # #     return sl, ss, li
        # # return sl[-1], ss[-1], li[-1]
        # print(f"mean:{mean[-1]}; std:{std[-1]}; stdbar:{stdbar[-1]}; bcolor:{bcolor[-1]}")

        # def pstdev(series, period):
        #     mean = self.talib.SMA(series, period)
        #     deviation = np.square(series - mean)
        #     stdev = np.sqrt(self.talib.SMA(deviation, period))
        #     return stdev

        # Calculate mean, standard deviation, and standardized bars
        mean = self.talib.SMA(self.volume, ma_length)
        # std = pstdev(self.volume, std_length)
        # std = self.talib.STDDEV(self.volume, timeperiod=std_length, nbdev=1)
        std = self.talib.STDDEV(
            self.volume,
            timeperiod=std_length,
            nbdev=np.sqrt(std_length / (std_length - 1)),
        )
        stdbar = (self.volume - mean) / std
        bcolor = np.where(
            stdbar > extra_high_threshold,
            "extra_high",
            np.where(
                stdbar > high_threshold,
                "high",
                np.where(
                    stdbar > medium_threshold,
                    "medium",
                    np.where(stdbar > normal_threshold, "normal", "low"),
                ),
            ),
        )
        # print(f"mean:{mean[-1]}; std:{std[-1]}; stdbar:{stdbar[-1]}; bcolor:{bcolor[-1]}")
        if array:
            return stdbar, bcolor
        return stdbar[-1], bcolor[-1]

    def wr(self, period: int = 14, array: bool = False, input: np.ndarray = None):
        """
        0-20: overbought -> can short
        80-100: oversold -> can long

        Williams Percent Range

        //@version=5
        indicator("Williams Percent Range", shorttitle="Williams %R", format=format.price, precision=2, timeframe="", timeframe_gaps=true)
        length = input(title="Length", defval=14)
        src = input(close, "Source")
        _pr(length) =>
            max = ta.highest(length)
            min = ta.lowest(length)
            100 * (src - max) / (max - min)
        percentR = _pr(length)
        obPlot = hline(-20, title="Upper Band", color=#787B86)
        hline(-50, title="Middle Level", linestyle=hline.style_dotted, color=#787B86)
        osPlot = hline(-80, title="Lower Band", color=#787B86)
        fill(obPlot, osPlot, title="Background", color=color.rgb(126, 87, 194, 90))
        plot(percentR, title="%R", color=#7E57C2)
        """
        n_high = self.max(n=period, input="high", array=True)
        n_low = self.min(n=period, input="low", array=True)

        wr = 100 * (n_high - self.close) / (n_high - n_low)

        if array:
            return wr
        return wr[-1]

    def halftrend(
        self,
        amplitude: int = 4,
        deviation: int = 2,
        array: bool = False,
        input: np.ndarray | None = None,
    ):
        """
        https://tw.tradingview.com/script/U1SJ8ubc-HalfTrend/

        """
        atr2 = self.atr(n=100, array=True) / 2
        dev = deviation * atr2

        high_price = self.max(n=amplitude, input="high", array=True)
        low_price = self.min(n=amplitude, input="low", array=True)
        highma = self.sma(amplitude, input="high", array=True)
        lowma = self.sma(amplitude, input="low", array=True)

        m = self.close.size
        trend: np.ndarray = np.zeros(self.size)
        # trend[:] = np.nan
        next_trend: np.ndarray = np.zeros(self.size)
        # next_trend[:] = np.nan

        # max_low_px: np.ndarray = np.zeros(self.size)
        # min_high_px: np.ndarray = np.zeros(self.size)
        max_low_px = self.low[0]
        min_high_px = self.high[0]

        up: np.ndarray = np.zeros(self.size)
        up[:] = np.nan
        down: np.ndarray = np.zeros(self.size)
        down[:] = np.nan
        atr_high: np.ndarray = np.zeros(self.size)
        atr_low: np.ndarray = np.zeros(self.size)
        arrow_up: np.ndarray = np.zeros(self.size)
        arrow_down: np.ndarray = np.zeros(self.size)
        ht: np.ndarray = np.zeros(self.size)

        for i in range(1, m):
            next_trend[i] = next_trend[i - 1]
            trend[i] = trend[i - 1]

            if next_trend[i] == 1:
                max_low_px = max(low_price[i], max_low_px)
                # print(f"max_low_px:{max_low_px}")

                if highma[i] < max_low_px and self.close[i] < self.low[i - 1]:
                    trend[i] = 1
                    next_trend[i] = 0
                    min_high_px = high_price[i]
                    # print(f"min_high_px:{min_high_px}; next_trend:{next_trend[i]}; trend[i]:{trend[i]}")
            else:
                min_high_px = min(high_price[i], min_high_px)
                # print(f"min_high_px:{min_high_px}")
                if lowma[i] > min_high_px and self.close[i] > self.high[i - 1]:
                    trend[i] = 0
                    next_trend[i] = 1
                    max_low_px = low_price[i]
                    # print(f"max_low_px:{max_low_px}; next_trend:{next_trend[i]}; trend[i]:{trend[i]}")

            if trend[i] == 0:
                if trend[i - 1] != 0:
                    up[i] = down[i] if self.na(down[i - 1]) else down[i - 1]
                    arrow_up[i] = up[i] - atr2[i]
                else:
                    up[i] = max_low_px if self.na(up[i - 1]) else max(max_low_px, up[i - 1])
                atr_high[i] = up[i] + dev[i]
                atr_low[i] = up[i] - dev[i]
            else:
                if trend[i - 1] != 1:
                    down[i] = up[i] if self.na(up[i - 1]) else up[i - 1]
                    arrow_down[i] = down[i] + atr2[i]
                else:
                    down[i] = min_high_px if self.na(down[i - 1]) else min(min_high_px, down[i - 1])
                atr_high[i] = down[i] + dev[i]
                atr_low[i] = down[i] - dev[i]
            ht[i] = up[i] if trend[i] == 0 else down[i]
        #     print(f"down[i]:{down[i]} - up[i]:{up[i]}")
        # print(f"----"*5)

        # var color buyColor = color.blue
        # var color sellColor = color.red
        # htColor = trend == 0 ? buyColor : sellColor

        if array:
            return ht, trend

        # print(f"high_price:{high_price[-1]}; low_price:{low_price[-1]}; trend:{trend[-1]}; next_trend:{next_trend[-1]}; min_high_px:{min_high_px}; max_low_px:{max_low_px}; up:{up[-1]}; down:{down[-1]}; atr_high:{atr_high[-1]}; atr_low:{atr_low[-1]}; arrow_up:{arrow_up[-1]}; arrow_down:{arrow_down[-1]}")
        return ht[-1], trend[-1]

    def sushi_roll(self, data):
        """detect revesal signal

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Calculate the size of the dataset
        size = len(data)

        # Check if the dataset is large enough to form the pattern
        if size < 5:
            return False

        # Define the conditions for a bullish and bearish sushi roll
        bullish_conditions = [
            data["Open"][size - 5] > data["Close"][size - 5],
            data["Close"][size - 4] > data["Open"][size - 4],
            data["Close"][size - 4] > data["Close"][size - 5],
            data["Open"][size - 3] > data["Close"][size - 4],
            data["Close"][size - 3] > data["Open"][size - 3],
            data["Close"][size - 3] > data["Close"][size - 4],
            data["Open"][size - 2] > data["Close"][size - 3],
            data["Close"][size - 2] > data["Open"][size - 2],
            data["Close"][size - 2] > data["Close"][size - 3],
            data["Open"][size - 1] > data["Close"][size - 2],
            data["Close"][size - 1] > data["Open"][size - 1],
            data["Close"][size - 1] > data["Close"][size - 2],
        ]

        bearish_conditions = [
            data["Open"][size - 5] < data["Close"][size - 5],
            data["Close"][size - 4] < data["Open"][size - 4],
            data["Close"][size - 4] < data["Close"][size - 5],
            data["Open"][size - 3] < data["Close"][size - 4],
            data["Close"][size - 3] < data["Open"][size - 3],
            data["Close"][size - 3] < data["Close"][size - 4],
            data["Open"][size - 2] < data["Close"][size - 3],
            data["Close"][size - 2] < data["Open"][size - 2],
            data["Close"][size - 2] < data["Close"][size - 3],
            data["Open"][size - 1] < data["Close"][size - 2],
            data["Close"][size - 1] < data["Open"][size - 1],
            data["Close"][size - 1] < data["Close"][size - 2],
        ]

        # Check if the pattern is bullish or bearish
        if all(bullish_conditions):
            return "Bullish Sushi Roll"
        elif all(bearish_conditions):
            return "Bearish Sushi Roll"
        else:
            return False

    # def smartmoney_concept(self, amplitude: int=4, deviation: int=2, array: bool = False, input: np.ndarray = None):
    #     """
    #     https://www.tradingview.com/script/CnB3fSph-Smart-Money-Concepts-LuxAlgo/
    #         Mode: Allows the user to select Historical (default) or Present, which displays only recent data on the chart.
    #         Style: Allows the user to select different styling for the entire indicator between Colored (default) and Monochrome.
    #         Color Candles: Plots candles based on the internal & swing structures from within the indicator on the chart.
    #         Internal Structure: Displays the internal structure labels & dashed lines to represent them. (BOS & CHoCH).
    #         Confluence Filter: Filter non-significant internal structure breakouts.
    #         Swing Structure: Displays the swing structure labels & solid lines on the chart (larger BOS & CHoCH labels).
    #         Swing Points: Displays swing points labels on chart such as HH, HL, LH, LL.
    #         Internal Order Blocks: Enables Internal Order Blocks & allows the user to select how many most recent Internal Order Blocks appear on the chart.
    #         Swing Order Blocks: Enables Swing Order Blocks & allows the user to select how many most recent Swing Order Blocks appear on the chart.
    #         Equal Highs & Lows: Displays EQH / EQL labels on chart for detecting equal highs & lows.
    #         Bars Confirmation: Allows the user to select how many bars are needed to confirm an EQH / EQL symbol on chart.
    #         Fair Value Gaps: Displays boxes to highlight imbalance areas on the chart.
    #         Auto Threshold: Filter out non-significant fair value gaps.
    #         Timeframe: Allows the user to select the timeframe for the Fair Value Gap detection.
    #         Extend FVG: Allows the user to choose how many bars to extend the Fair Value Gap boxes on the chart.
    #         Highs & Lows MTF: Allows the user to display previous highs & lows from daily, weekly, & monthly timeframes as significant levels.
    #         Premium/ Discount Zones: Allows the user to display Premium, Discount , and Equilibrium zones on the chart
    #     """
    #     atr = self.atr(n=200, array=True)

    #     high_price = self.max(n=amplitude, input="high", array=True)
    #     low_price = self.min(n=amplitude, input="low", array=True)
    #     highma = self.sma(amplitude, input="high", array=True)
    #     lowma = self.sma(amplitude, input="low", array=True)

    #     m = self.close.size
    #     trend: np.ndarray = np.zeros(self.size)
    #     # trend[:] = np.nan
    #     next_trend: np.ndarray = np.zeros(self.size)
    #     # next_trend[:] = np.nan

    #     # max_low_px: np.ndarray = np.zeros(self.size)
    #     # min_high_px: np.ndarray = np.zeros(self.size)
    #     max_low_px = self.low[0]
    #     min_high_px = self.high[0]

    #     up: np.ndarray = np.zeros(self.size)
    #     up[:] = np.nan
    #     down: np.ndarray = np.zeros(self.size)
    #     down[:] = np.nan
    #     atr_high: np.ndarray = np.zeros(self.size)
    #     atr_low: np.ndarray = np.zeros(self.size)
    #     arrow_up: np.ndarray = np.zeros(self.size)
    #     arrow_down: np.ndarray = np.zeros(self.size)
    #     ht: np.ndarray = np.zeros(self.size)

    #     for i in range(1, m):
    #         next_trend[i] = next_trend[i-1]
    #         trend[i] = trend[i-1]

    #         if next_trend[i] == 1:
    #             max_low_px = max(low_price[i], max_low_px)
    #             # print(f"max_low_px:{max_low_px}")

    #             if highma[i] < max_low_px and self.close[i] < self.low[i-1]:
    #                 trend[i] = 1
    #                 next_trend[i] = 0
    #                 min_high_px = high_price[i]
    #                 # print(f"min_high_px:{min_high_px}; next_trend:{next_trend[i]}; trend[i]:{trend[i]}")
    #         else:
    #             min_high_px = min(high_price[i], min_high_px)
    #             # print(f"min_high_px:{min_high_px}")
    #             if lowma[i] > min_high_px and self.close[i] > self.high[i-1]:
    #                 trend[i] = 0
    #                 next_trend[i] = 1
    #                 max_low_px = low_price[i]
    #                 # print(f"max_low_px:{max_low_px}; next_trend:{next_trend[i]}; trend[i]:{trend[i]}")

    #         if trend[i] == 0:
    #             if trend[i-1] != 0:
    #                 up[i] = down[i] if self.na(down[i-1]) else down[i-1]
    #                 arrow_up[i] = up[i] - atr2[i]
    #             else:
    #                 up[i] = max_low_px if self.na(up[i-1]) else max(max_low_px, up[i-1])
    #             atr_high[i] = up[i] + dev[i]
    #             atr_low[i] = up[i] - dev[i]
    #         else:
    #             if trend[i-1] != 1:
    #                 down[i] = up[i] if self.na(up[i-1]) else up[i-1]
    #                 arrow_down[i] = down[i] + atr2[i]
    #             else:
    #                 down[i] = min_high_px if self.na(down[i-1]) else min(min_high_px, down[i-1])
    #             atr_high[i] = down[i] + dev[i]
    #             atr_low[i] = down[i] - dev[i]
    #         ht[i] = up[i] if trend[i] == 0 else down[i]
    #     #     print(f"down[i]:{down[i]} - up[i]:{up[i]}")
    #     # print(f"----"*5)

    #     # var color buyColor = color.blue
    #     # var color sellColor = color.red
    #     # htColor = trend == 0 ? buyColor : sellColor

    #     if array:
    #         return ht, trend

    #     # print(f"high_price:{high_price[-1]}; low_price:{low_price[-1]}; trend:{trend[-1]}; next_trend:{next_trend[-1]}; min_high_px:{min_high_px}; max_low_px:{max_low_px}; up:{up[-1]}; down:{down[-1]}; atr_high:{atr_high[-1]}; atr_low:{atr_low[-1]}; arrow_up:{arrow_up[-1]}; arrow_down:{arrow_down[-1]}")
    #     return ht[-1], trend[-1]


class Patterns(Indicator):
    def __init__(
        self,
        size: int = 100,
        window: int = 1,
        hour_window: bool = False,
        strict_agg: bool = False,
        datafeed: DataFeed | None = None,
        init_data: dict[str, BarData] | None = None,
    ) -> None:
        super().__init__(size, window, hour_window, strict_agg, datafeed, init_data)

    # @property
    @classmethod
    def patterns(self) -> list[str]:
        return [i for i in dir(self) if i.startswith("CDL")]

    def CDL2CROWS(self, array: bool = False, series: bool = False) -> float | np.ndarray:
        out = self.talib.CDL2CROWS(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        if series:
            return pd.Series(data=out, index=self.datetime_array)
        return out[-1]

    def CDL3BLACKCROWS(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDL3BLACKCROWS(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDL3INSIDE(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDL3INSIDE(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDL3STARSINSOUTH(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDL3STARSINSOUTH(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDL3WHITESOLDIERS(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDL3WHITESOLDIERS(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLABANDONEDBABY(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLABANDONEDBABY(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLADVANCEBLOCK(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLADVANCEBLOCK(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLBELTHOLD(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLBELTHOLD(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLBREAKAWAY(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLBREAKAWAY(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLCONCEALBABYSWALL(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLCONCEALBABYSWALL(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLCOUNTERATTACK(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLCOUNTERATTACK(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLDARKCLOUDCOVER(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLDARKCLOUDCOVER(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLDOJISTAR(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLDOJISTAR(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLENGULFING(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLENGULFING(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLEVENINGDOJISTAR(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLEVENINGDOJISTAR(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLEVENINGSTAR(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLEVENINGSTAR(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLHAMMER(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLHAMMER(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLHANGINGMAN(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLHANGINGMAN(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLHARAMI(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLHARAMI(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLHARAMICROSS(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLHARAMICROSS(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLHOMINGPIGEON(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLHOMINGPIGEON(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLIDENTICAL3CROWS(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLIDENTICAL3CROWS(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLINVERTEDHAMMER(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLINVERTEDHAMMER(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLKICKING(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLKICKING(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLKICKINGBYLENGTH(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLKICKINGBYLENGTH(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLLADDERBOTTOM(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLLADDERBOTTOM(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLMATCHINGLOW(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLMATCHINGLOW(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLMORNINGDOJISTAR(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLMORNINGDOJISTAR(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLMORNINGSTAR(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLMORNINGSTAR(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLPIERCING(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLPIERCING(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLSHOOTINGSTAR(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLSHOOTINGSTAR(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLSTALLEDPATTERN(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLSTALLEDPATTERN(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLSTICKSANDWICH(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLSTICKSANDWICH(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLTRISTAR(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLTRISTAR(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLUNIQUE3RIVER(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLUNIQUE3RIVER(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLUPSIDEGAP2CROWS(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLUPSIDEGAP2CROWS(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLXSIDEGAP3METHODS(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLXSIDEGAP3METHODS(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDL3LINESTRIKE(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDL3LINESTRIKE(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDL3OUTSIDE(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDL3OUTSIDE(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLCLOSINGMARUBOZU(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLCLOSINGMARUBOZU(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLDOJI(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLDOJI(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLDRAGONFLYDOJI(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLDRAGONFLYDOJI(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLGAPSIDESIDEWHITE(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLGAPSIDESIDEWHITE(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLGRAVESTONEDOJI(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLGRAVESTONEDOJI(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLHIGHWAVE(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLHIGHWAVE(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLHIKKAKE(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLHIKKAKE(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLHIKKAKEMOD(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLHIKKAKEMOD(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLINNECK(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLINNECK(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLLONGLEGGEDDOJI(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLLONGLEGGEDDOJI(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLLONGLINE(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLLONGLINE(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLMARUBOZU(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLMARUBOZU(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLMATHOLD(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLMATHOLD(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLONNECK(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLONNECK(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLRICKSHAWMAN(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLRICKSHAWMAN(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLRISEFALL3METHODS(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLRISEFALL3METHODS(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLSEPARATINGLINES(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLSEPARATINGLINES(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLSHORTLINE(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLSHORTLINE(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLSPINNINGTOP(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLSPINNINGTOP(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLTAKURI(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLTAKURI(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLTASUKIGAP(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLTASUKIGAP(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]

    def CDLTHRUSTING(self, array: bool = False) -> float | np.ndarray:
        out = self.talib.CDLTHRUSTING(open=self.open, high=self.high, low=self.low, close=self.close)
        if array:
            return out
        return out[-1]


class BarGenerator:
    """ """

    hour_window_map = {
        2: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        4: [0, 4, 8, 12, 16, 20],
        6: [2, 8, 14, 20],
        8: [0, 8, 16],
        12: [0, 12],
    }

    def __init__(
        self,
        on_bar: Callable | None = None,
        window: int = 0,
        on_window_bar: Callable | None = None,
        hour_window: bool = False,
        strict_agg: bool = False,  # if true, the agg datetime will same as exchange
    ) -> None:
        """
        1. generating 1 minute bar data from tick data
        2. generating x minute bar/x hour bar data from 1 minute data


        Args:
            on_bar (Callable, optional): callback function when new bar generated. Defaults to None.
            window (int, optional): window. Defaults to 0.
                if hour_window is False, this is for x minute bar, x must be able to divide 60: 2, 3, 5, 6, 10, 15, 20, 30
                if hour_window is True, this is for x hour bar, x can be any number
            on_window_bar (Callable, optional): callback function when new window bar generated. Defaults to None.
            hour_window (bool, optional): if is generator is for genrating hour window abr. Defaults to False.
        """
        if hour_window:
            assert window <= 24
        else:
            assert window <= 60

        self.minute_bar = None
        self.bar = None
        self.on_bar: Callable = on_bar

        self.interval_count: int = 0

        self.hour_bar = None

        self.hour_window: bool = hour_window
        self.strict_agg: bool = strict_agg

        self.window: int = window
        self.window_bar = None
        self.on_window_bar: Callable = on_window_bar

        self.last_tick = None
        self.last_bar = None
        self.force_finish_once: bool = None
        self.is_finished = False

    def update_tick(self, tick: TickData) -> None:
        """
        Update new tick data into generator.
        """
        new_minute: bool = False

        # Filter tick data with 0 last price
        last_price = tick.last_price
        if not last_price:
            return

        # Filter tick data with older timestamp
        if self.last_tick and tick.datetime < self.last_tick.datetime:
            return

        if self.minute_bar is None:
            new_minute = True
        elif (self.minute_bar.datetime.minute != tick.datetime.minute) or (
            self.minute_bar.datetime.hour != tick.datetime.hour
        ):
            self.minute_bar.datetime = self.minute_bar.datetime.replace(second=0, microsecond=0)
            if self.on_bar:
                self.on_bar(self.minute_bar)

            new_minute = True
        # print(f"=============> self.minute_bar:{self.minute_bar}")
        if new_minute:
            self.minute_bar = BarData(
                symbol=tick.symbol,
                exchange=tick.exchange,
                code=tick.code,
                asset_type=tick.asset_type,
                interval="1m",
                datetime=tick.datetime,
                open=last_price,
                high=last_price,
                low=last_price,
                close=last_price,
                volume=0.0,
                turnover=0.0,
            )
        else:
            self.minute_bar.high = max(self.minute_bar.high, tick.last_price)
            # if tick.high > self.last_tick.high:
            #     self.minute_bar.high = max(self.minute_bar.high, tick.high)

            self.minute_bar.low = min(self.minute_bar.low, tick.last_price)
            # if tick.low < self.last_tick.low:
            #     self.minute_bar.low = min(self.minute_bar.low, tick.low)

            self.minute_bar.close = tick.last_price
            # self.minute_bar.open_interest = tick.open_interest
            self.minute_bar.datetime = tick.datetime

        if self.last_tick:
            volume_change: float = tick.prev_volume_24h - self.last_tick.prev_volume_24h
            self.minute_bar.volume += max(volume_change, 0)

            turnover_change: float = tick.prev_turnover_24h - self.last_tick.prev_turnover_24h
            self.minute_bar.turnover += max(turnover_change, 0)

        self.last_tick = tick

    def update_bar(self, bar: dict) -> None:
        """
        Update new bar data into array manager.
        """
        self.is_finished = False
        # print(f'[BarGenerator-update_bar] bar.interval :{bar.interval }; self.window:{self.window}; self.hour_window:{self.hour_window} | bar:{bar} | self.window_bar:{self.window_bar}')
        if "m" in bar.interval and not self.hour_window:
            self.update_bar_minute_window(bar)
        elif self.hour_window:
            self.update_bar_hour_window(bar)
        else:
            self.is_finished = True
            self.bar = bar

    def update_bar_minute_window(self, bar) -> None:
        """"""
        # If not inited, create window bar object
        if self.window_bar is None:
            dt: datetime = bar.datetime.replace(
                second=0,
                microsecond=0,
                minute=int(bar.datetime.minute / self.window) * self.window,
            )
            # dt: datetime = bar.datetime.replace(second=0, microsecond=0,)
            self.bar = self.window_bar = BarData(
                symbol=bar.symbol,
                code=bar.code,
                exchange=bar.exchange,
                asset_type=bar.asset_type,
                interval=f"{self.window}m",
                datetime=dt,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=0.0,
                turnover=0.0,
            )
        # Otherwise, update high/low price into window bar
        else:
            self.window_bar.high = max(self.window_bar.high, bar.high)
            self.window_bar.low = min(self.window_bar.low, bar.low)

        # Update close price/volume/turnover into window bar
        self.window_bar.close = bar.close
        self.window_bar.volume += bar.volume
        self.window_bar.turnover += bar.turnover
        # self.window_bar.open_interest = bar.open_interest

        # Check if window bar completed
        self.bar = self.window_bar
        window = int(bar["interval"].replace("m", ""))
        # print(f'[BarGenerator-update_minute_window] {bar} [bar]')
        # print(f'[BarGenerator-update_minute_window] window:{window} | bar.datetime.minute+window:{bar.datetime.minute+window} | (bar.datetime.minute + window) % self.window:{(bar.datetime.minute + window) % self.window}')
        # print(f'[BarGenerator-update_minute_window] {self.last_bar} [self.last_bar]')
        # print(f'[BarGenerator-update_minute_window] {self.window_bar} [self.window_bar]')
        if (
            not (bar.datetime.minute + window) % self.window
        ):  # and (self.last_bar and self.last_bar.datetime != bar.datetime)
            self.is_finished = True
            if self.on_window_bar:
                self.on_window_bar(self.window_bar)
            self.last_bar = self.window_bar
            self.window_bar = None

    def update_bar_hour_window(self, bar) -> None:
        """
        - 1 minute bar to 1 hour bar
        - 1 minute bar to 4 hour bar
        - 15 minute bar to 1 hour bar
        - 15 minute bar to 4 hour bar
        - 1 hour bar to 1 hour bar
        - 1 hour bar to 4 hour bar
        - 2 hour bar to 4 hour bar

        1: 0, 1, 2, 3, 4, 5, 6, 7, .... 22, 23
        2: 0, 2, 4, 6, 8, .... 20, 22
        4: 0, 4, 8, 12, 16, 20
        6: 2, 8, 14, 20
        8: 0, 8, 16
        12: 0, 12
        """
        interval = bar.interval
        # If not inited, create window bar object
        if self.window_bar is None:
            if interval != "1m":
                self.interval_count += 1

            dt: datetime = bar.datetime.replace(minute=0, second=0, microsecond=0)
            self.bar = self.window_bar = BarData(
                symbol=bar.symbol,
                code=bar.code,
                exchange=bar.exchange,
                asset_type=bar.asset_type,
                interval=bar.interval,
                datetime=dt,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                turnover=bar.turnover,
                # 'open_interest':bar.open_interest
            )
            # print('........', self.window_bar,'????')
            return

        finished_bar = None
        if interval == "1m":
            self._agg_bar(bar)
            if bar.datetime.minute == 59:
                ### FIXME: how to handle if it happened to miss the 59 min bar
                self.interval_count += 1
                _window = self.window
                # if not self.interval_count % _window:
                #     self.interval_count = 0
                #     if self.on_window_bar:
                #         self.on_window_bar(self.window_bar)
                #     finished_bar = self.window_bar
                #     # print(f">>>>>>> self.window: {self.window }; self.window_bar dt:{self.window_bar.datetime if self.window_bar is not None else None} ||| finished_bar dt:{finished_bar.datetime if finished_bar is not None else None} ")
                #     # print(f"1m1m >>>> finished_bar:{finished_bar}")
                #     self.window_bar = None

                ### TODO:
                force_finish = False
                if (
                    self.strict_agg
                    and self.force_finish_once is None
                    and bar.datetime.hour + 1 in self.hour_window_map.get(self.window, [])
                ):
                    # print(f'----->',self.window_bar)
                    self.force_finish_once = True
                    force_finish = True
                finished_bar = self._gen_bar(_window, force_finish)

        elif "m" in interval:
            self._agg_bar(bar)
            self.interval_count += 1
            interval_num = int(interval.replace("m", ""))
            _window = self.window * 60 // interval_num
            if not _window:
                self.is_finished = True
                self.window_bar = BarData(
                    symbol=bar.symbol,
                    code=bar.code,
                    exchange=bar.exchange,
                    asset_type=bar.asset_type,
                    interval=bar.interval,
                    datetime=bar.datetime,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    turnover=bar.turnover,
                    # 'open_interest':bar.open_interest
                )
            else:
                # # print(f"_window:{_window}; interval_count:{self.interval_count}")
                # if not self.interval_count % _window:
                #     self.interval_count = 0
                #     if self.on_window_bar:
                #         self.on_window_bar(self.window_bar)
                #     finished_bar = self.window_bar
                #     # print(f">>>>>>> self.window: {self.window }; self.window_bar dt:{self.window_bar.datetime if self.window_bar is not None else None} ||| finished_bar dt:{finished_bar.datetime if finished_bar is not None else None} ")
                #     # print(f"mmmm >>>> finished_bar:{finished_bar}")
                #     self.window_bar = None

                ### TODO:
                force_finish = False
                # print(f"self.strict_agg:{self.strict_agg}|| self.force_finish_once is None:{self.force_finish_once is None} || bar.datetime.hour:{bar.datetime.hour} || ||bar.datetime.hour + 1 in self.hour_window_map.get(self.window, []):{bar.datetime.hour + 1 in self.hour_window_map.get(self.window, [])} ||  not bar.datetime.minute+interval_num % interval_num:{not bar.datetime.minute+interval_num % interval_num} || bar.datetime.minute+interval_num:{bar.datetime.minute+interval_num} || interval_num:{interval_num}")
                if (
                    self.strict_agg
                    and self.force_finish_once is None
                    and bar.datetime.hour + 1 in self.hour_window_map.get(self.window, [])
                    and bar.datetime.minute + interval_num == 60
                ):
                    # print(f'----->',self.window_bar)
                    self.force_finish_once = True
                    force_finish = True
                finished_bar = self._gen_bar(_window, force_finish)

        elif "h" in interval:
            interval_num = int(interval.replace("h", ""))
            if self.window == 1 or self.window == interval_num:
                self.is_finished = True
                self.window_bar = BarData(
                    symbol=bar.symbol,
                    code=bar.code,
                    exchange=bar.exchange,
                    asset_type=bar.asset_type,
                    interval=bar.interval,
                    datetime=bar.datetime,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    turnover=bar.turnover,
                    # 'open_interest':bar.open_interest
                )
            else:
                self._agg_bar(bar)
                self.interval_count += 1

                _window = self.window // interval_num
                if not _window:
                    self.is_finished = True
                    self.window_bar = BarData(
                        symbol=bar.symbol,
                        code=bar.code,
                        exchange=bar.exchange,
                        asset_type=bar.asset_type,
                        interval=bar.interval,
                        datetime=bar.datetime,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                        turnover=bar.turnover,
                        # 'open_interest':bar.open_interest
                    )
                else:
                    # # print(f"_window:{_window}; interval_count:{self.interval_count}")
                    # if not self.interval_count % _window:
                    #     self.interval_count = 0
                    #     if self.on_window_bar:
                    #         self.on_window_bar(self.window_bar)
                    #     finished_bar = self.window_bar
                    #     # print(f">>>>>>> self.window: {self.window }; self.window_bar dt:{self.window_bar.datetime if self.window_bar is not None else None} ||| finished_bar dt:{finished_bar.datetime if finished_bar is not None else None} ")
                    #     # print(f"hhhh >>>> finished_bar:{finished_bar}")
                    #     self.window_bar = None

                    force_finish = False
                    # print(f"self.strict_agg:{self.strict_agg}|| self.force_finish_once is None:{self.force_finish_once is None} || bar.datetime.hour:{bar.datetime.hour} || ||bar.datetime.hour + interval_num in self.hour_window_map.get(self.window, []):{bar.datetime.hour + interval_num in self.hour_window_map.get(self.window, [])} ||  not self.interval_count+1 % _window:{ not self.interval_count+1 % _window}")
                    if (
                        self.strict_agg
                        and self.force_finish_once is None
                        and bar.datetime.hour + interval_num in self.hour_window_map.get(self.window, [])
                    ):  # and not self.interval_count+1 % _window:
                        # print(f'----->',self.window_bar)
                        self.force_finish_once = True
                        force_finish = True

                    finished_bar = self._gen_bar(_window, force_finish)

        else:
            ### unknown interval -> directly assign to bar
            self.is_finished = True
            self.window_bar = BarData(
                symbol=bar.symbol,
                code=bar.code,
                exchange=bar.exchange,
                asset_type=bar.asset_type,
                interval=bar.interval,
                datetime=bar.datetime,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                turnover=bar.turnover,
                # 'open_interest':bar.open_interest
            )

        self.bar = finished_bar if finished_bar is not None else self.window_bar

    def _agg_bar(self, bar: BarData) -> None:
        # if self.window_bar is None or bar is None:
        #     print(f"???? windowsbar:{self.window_bar}; bar: {bar}")
        self.window_bar.high = max(self.window_bar.high, bar.high)
        self.window_bar.low = min(self.window_bar.low, bar.low)

        self.window_bar.close = bar.close
        self.window_bar.volume += bar.volume
        self.window_bar.turnover += bar.turnover

    def _gen_bar(self, _window: int, force_finish: bool = False) -> BarData | None:
        finished_bar = None
        # if self.strict_agg
        if not self.interval_count % _window or force_finish:
            self.interval_count = 0
            self.is_finished = True
            if self.on_window_bar:
                self.on_window_bar(self.window_bar)
            finished_bar = self.window_bar
            # print(f">>>>>>> self.window: {self.window }; self.window_bar dt:{self.window_bar.datetime if self.window_bar is not None else None} ||| finished_bar dt:{finished_bar.datetime if finished_bar is not None else None} ")
            # print(f"mmmm >>>> finished_bar:{finished_bar}")
            self.window_bar = None
        return finished_bar


class BarManager:
    """
    for managing tick data ==> bar data
    """

    def __init__(self, on_bar: Callable | None = None) -> None:
        self.bars = {}
        self.on_bar: Callable = on_bar

        self.last_ticks = defaultdict(dict)
        self.last_dt: datetime = None

    def update_tick(self, tick: TickData) -> None:
        # print(f">>> {tick}")
        last_price = tick.get("last_price")
        if not last_price:
            aio_symbol = tick["aio_symbol"]
            last_tick = self.last_ticks.get(aio_symbol, None)
            if last_tick:
                self.last_ticks[aio_symbol].update(tick)
        else:
            tick_datetime = tick["datetime"]
            symbol = tick["symbol"]
            exchange = tick["exchange"]
            code = tick["code"]
            asset_type = tick["asset_type"]
            prev_volume = tick.get("prev_volume_24h")
            prev_turnover = tick.get("prev_turnover_24h")
            curr_volume = tick.get("volume")
            curr_turnover = tick.get("turnover")
            aio_symbol = tick["aio_symbol"]

            # print(tick,'.........', type(tick))
            ### new tick for new bars
            if self.last_dt and self.last_dt.minute != tick_datetime.minute:
                if self.on_bar:
                    # print(f'update_tick -> abt to trigger callback !!!self.bars: {self.bars}')
                    ## TODO: thread??
                    self.on_bar(tick=tick_datetime, bars=self.bars)

                self.bars = {}

            # print(f"---> DEBUG[{tick_datetime}] - aio_symbol:{aio_symbol}; last_price:{last_price}; volume:{volume}")
            # print(f"---> DEBUG[{tick_datetime}] - tick:{tick}")

            bar = self.bars.get(aio_symbol, None)
            if bar is None:
                # print(f">>>> bar open:{last_price}; tick_datetime:{tick_datetime}")
                bar = BarData(
                    symbol=symbol,
                    exchange=exchange,
                    code=code,
                    asset_type=asset_type,
                    interval="1m",
                    datetime=tick_datetime.replace(second=0, microsecond=0),
                    open=last_price,
                    high=last_price,
                    low=last_price,
                    close=last_price,
                    volume=0,
                    turnover=0,
                )
                self.bars[aio_symbol] = bar
            else:
                bar.high = max(bar.high, last_price)
                bar.low = min(bar.low, last_price)
                bar.close = last_price
                # bar.open_interest = tick.open_interest
                bar.datetime = tick_datetime

            last_tick = self.last_ticks.get(aio_symbol, None)
            if last_tick:
                if prev_volume:
                    bar.volume += max(prev_volume - last_tick["prev_volume_24h"], 0)
                if prev_turnover:
                    bar.turnover += max(prev_turnover - last_tick["prev_turnover_24h"], 0)

                if curr_volume:
                    bar.volume += max(curr_volume + last_tick["volume"], 0)
                if curr_turnover:
                    bar.volume += max(curr_turnover + last_tick["turnover"], 0)

            # self.last_ticks[aio_symbol] = tick
            self.last_ticks[aio_symbol].update(tick)
            self.last_dt = tick_datetime

        return self.last_ticks[aio_symbol]

    def trigger_callback(self) -> None:
        bar = None
        for bar in self.bars.values():
            bar.datetime = bar.datetime.replace(second=0, microsecond=0)
        if self.on_bar and bar is not None:
            # print(f'trigger_callback !!!self.bars: {self.bars}')
            self.on_bar(tick=bar.datetime, bars=self.bars)

    def get_tick_data(
        self, symbol: str | None = None, exchange: str | None = None, aio_symbol: str | None = None
    ) -> TickData | None:
        if symbol and exchange:
            if not isinstance(exchange, str):
                exchange = exchange.value
            aio_symbol = f"{symbol.upper()}|{exchange.upper()}"

        return self.last_ticks.get(aio_symbol) if aio_symbol else None


if __name__ == "__main__":
    print(Patterns.patterns(), len(Patterns.patterns()))
