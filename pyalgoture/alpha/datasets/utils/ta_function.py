"""
Technical Analysis Operators
"""

from typing import Any

import pandas as pd
import polars as pl
import talib

from .utility import DataProxy


def to_pd_series(feature: DataProxy) -> pd.Series:
    """Convert to pandas.Series data structure"""
    result = feature.df.to_pandas().set_index(["datetime", feature.key])["data"]
    return pd.Series(result, dtype=object)


def to_pl_dataframe(series: pd.Series) -> pl.DataFrame:
    """Convert to polars.DataFrame data structure"""
    return pl.from_pandas(series.reset_index().rename(columns={0: "data"}))


def ta_rsi(close: DataProxy, window: int) -> DataProxy:
    """Calculate RSI indicator by contract"""
    close_: pd.Series = to_pd_series(close)

    result: pd.Series = talib.RSI(close_, timeperiod=window)

    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)


def ta_atr(high: DataProxy, low: DataProxy, close: DataProxy, window: int) -> DataProxy:
    """Calculate ATR indicator by contract"""
    high_: pd.Series = to_pd_series(high)
    low_: pd.Series = to_pd_series(low)
    close_: pd.Series = to_pd_series(close)

    result: pd.Series = talib.ATR(high_, low_, close_, timeperiod=window)

    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)


def ta_sma(close: DataProxy, window: int) -> DataProxy:
    """
    Simple moving average.
    """
    close_: pd.Series = to_pd_series(close)
    result: pd.Series = talib.SMA(close_, timeperiod=window)
    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)


def ta_ema(close: DataProxy, window: int) -> DataProxy:
    """
    Exponential moving average.
    """
    close_: pd.Series = to_pd_series(close)
    result: pd.Series = talib.EMA(close_, timeperiod=window)
    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)


def ta_wma(close: DataProxy, window: int) -> DataProxy:
    """
    Weighted moving average.
    """
    close_: pd.Series = to_pd_series(close)
    result: pd.Series = talib.WMA(close_, timeperiod=window)
    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)


def ta_sar(high: DataProxy, low: DataProxy, acceleration: int = 0, maximum: int = 0) -> DataProxy:
    """
    SAR.

    start: 0.02
    increament: 0.02
    maximum: 0.2
    """
    high_: pd.Series = to_pd_series(high)
    low_: pd.Series = to_pd_series(low)
    result: pd.Series = talib.SAR(high_, low_, acceleration, maximum)
    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)


def ta_adx(high: DataProxy, low: DataProxy, close: DataProxy, window: int) -> DataProxy:
    """
    ADX.
    """
    high_: pd.Series = to_pd_series(high)
    low_: pd.Series = to_pd_series(low)
    close_: pd.Series = to_pd_series(close)
    result: pd.Series = talib.ADX(high_, low_, close_, timeperiod=window)
    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)


def ta_willr(high: DataProxy, low: DataProxy, close: DataProxy, window: int) -> DataProxy:
    """
    WILLR.
    """
    high_: pd.Series = to_pd_series(high)
    low_: pd.Series = to_pd_series(low)
    close_: pd.Series = to_pd_series(close)
    result: pd.Series = talib.WILLR(high_, low_, close_, timeperiod=window)
    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)


def ta_bband(high: DataProxy, low: DataProxy, close: DataProxy, n: int, dev: float, matype: int = 0) -> DataProxy:
    """
    Bollinger Channel.
    """
    high_: pd.Series = to_pd_series(high)
    low_: pd.Series = to_pd_series(low)
    close_: pd.Series = to_pd_series(close)
    result: pd.Series = talib.BBANDS(high_, low_, close_, timeperiod=n, nbdevup=dev, nbdevdn=dev, matype=matype)
    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)


def ta_stoch(
    high: DataProxy,
    low: DataProxy,
    close: DataProxy,
    fastk_period: int,
    slowk_period: int,
    slowk_matype: int,
    slowd_period: int,
    slowd_matype: int,
) -> DataProxy:
    """
    Stochastic Indicator
    """
    high_: pd.Series = to_pd_series(high)
    low_: pd.Series = to_pd_series(low)
    close_: pd.Series = to_pd_series(close)
    result: pd.Series = talib.STOCH(
        high_, low_, close_, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype
    )
    df: pl.DataFrame = to_pl_dataframe(result)
    return DataProxy(df)
