"""
# Introduction

The basics of running this platform:

- Create a `Strategy`
    - Decide on potential adjustable parameters
    - Instantiate the Indicators you need in the `Strategy`
    - Write down the logic for entering/exiting the market

- Create `DataFeeds`
    - Specify the symbol, start/end date, interval, asset_type

- Create `Broker`
    - Add exchange connections (if realtime)
    - Add account details (if backtest)
        - slippage
        - initial balance
        - ...

- Create a `Orchestrator` (Backtest / Realtime)
    - Inject the `Strategy`
    - Inject the `Broker`
    - Load and Inject `DataFeeds`
    - And execute `Orchestrator.start()`
    - For visual feedback use: `contest_output()`

There is a `context` \ `ctx` object that store all the shared variables and common functions.

The platform is highly configurable

## Flowchart
![xkcd.com/1570](https://xkcd.com/1570/)

## Installation

Create a virtual environment
```
virtualenv venv --python=python3
```
Activate the virtual environment
```python
# Macbook / Linux
source venv/bin/activate

# Windows
venv/Scripts/activate
```
Deactivate
```
deactivate
```
Install PyAlgoture package
```
pip install pyalgoture
```


### Install dependency (optional)
**Use `Indicator` requires to install `ta-lib`**

Mac:
```
1.1 Install TA-LIB
brew install ta-lib

1.2 Install TA-LIB Python Wrapper
Install TA-Lib Python Wrapper via pip (or pip3):
pip install ta-lib
```

Linux:
```
1.1 Download
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/

1.2 Install TA-LIB
If the next command fails, then gcc is missing, install it by doing "apt-get install build-essential")
sudo ./configure
sudo make
sudo make install

1.3 Install TA-LIB Python Wrapper
Install TA-Lib Python Wrapper via pip (or pip3):
pip install ta-lib
```

Windows
```
1.1 Download
Download [ta-lib-0.4.0-msvc.zip](http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip)

1.2 Unzip
unzip to C:/\ ta-lib

1.3 Install TA-LIB Python Wrapper
Install TA-Lib Python Wrapper via pip (or pip3):
pip install ta-lib
```

**Use `Plotter` requires to install `bokeh`**
```
pip install -U bokeh
```

**Use `optimize` requires to install `tqdm`**
```
pip install -U tqdm
```

**Use `optimize(skopt='skopt')` requires to install `scikit-optimize`**

```
pip install -U scikit-optimize
```

**Use `SignalMQ` requires to install `redis`**
```
pip install -U redis
```


## QuickStart

* [Download Data and Backtesting](./examples/sample_data.html)
* [Strategy Backtesting & Optimization](./examples/sample_strategy.html)
* [Plot Custom Graph](./examples/sample_plot.html)
* [Backtesting with Trade Records](./examples/sample_history.html)
* [Sample Strategy & Realtime - Supertrend](./examples/sample_ta_supertrend.html)



# API Reference Documentation
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any

from pyalgoture.base import Base
from pyalgoture.broker import Broker
from pyalgoture.brokers.backtest_broker import BacktestBroker
from pyalgoture.brokers.realtime_broker import RealtimeBroker
from pyalgoture.context import Context

# from pyalgoture.utils.simple_webserver import SignalServer
# from pyalgoture.utils.util_tmux import TmuxManager
from pyalgoture.databases.filesystem_db import FilesystemDatabase
from pyalgoture.datafeed import DataFeed, DataFeedDownloader
from pyalgoture.datafeeds.crypto_datafeed import CryptoDataFeed
from pyalgoture.datafeeds.custom_datafeed import CustomDataFeed
from pyalgoture.datafeeds.stock_datafeed import StockDataFeed

# from pyalgoture.exchanges.binance import (
#     BinanceFutureClient,
#     BinanceFutureDataWebsocket,
#     BinanceFutureTradeWebsocket,
#     BinanceInverseFutureClient,
#     BinanceInverseFutureDataWebsocket,
#     BinanceInverseFutureTradeWebsocket,
#     BinanceOptionClient,
#     BinanceOptionDataWebsocket,
#     BinanceOptionTradeWebsocket,
#     BinanceSpotClient,
#     BinanceSpotDataWebsocket,
#     BinanceSpotTradeWebsocket,
# )
from pyalgoture.exchanges.binance_v2 import (
    BinanceClient,
    BinanceDataWebsocket,
    BinanceTradeWebsocket,
)

# from pyalgoture.exchanges.bingx import BingxClient, BingxDataWebsocket, BingxTradeWebsocket
from pyalgoture.exchanges.bitget_v2 import (
    BitgetClient,
    BitgetDataWebsocket,
    BitgetTradeWebsocket,
)

# from pyalgoture.exchanges.bybit import (
#     BybitFutureClient,
#     BybitFutureDataWebsocket,
#     BybitFutureTradeWebsocket,
#     BybitInverseFutureClient,
#     BybitInverseFutureDataWebsocket,
#     BybitInverseFutureTradeWebsocket,
#     BybitOptionClient,
#     BybitSpotClient,
#     BybitSpotDataWebsocket,
#     BybitSpotTradeWebsocket,
# )
from pyalgoture.exchanges.bybit_v2 import (
    BybitClient,
    BybitDataWebsocket,
    BybitTradeWebsocket,
)
from pyalgoture.exchanges.dummy import DummyClient, DummyTradeWebsocket

# from pyalgoture.exchanges.eqonex import (
#     EqonexClient,
#     EqonexDataWebsocket,
#     EqonexTradeWebsocket,
# )
# from pyalgoture.exchanges.ftx import FTXClient, FTXSpotDataWebsocket, FTXSpotTradeWebsocket
from pyalgoture.exchanges.okx import OKXClient, OKXDataWebsocket, OKXTradeWebsocket
from pyalgoture.orchestra import Orchestrator
from pyalgoture.orchestrators.backtest import BackTest
from pyalgoture.orchestrators.realtime import RealTime
from pyalgoture.statistic import Statistic
from pyalgoture.strategy import Strategy

# from pyalgoture.utils.cli import get_args
from pyalgoture.utils.client_rest import Request, Response, RestClient
from pyalgoture.utils.client_ws import WebsocketClient
from pyalgoture.utils.event_engine import Event, EventEngine
from pyalgoture.utils.indicator import BarGenerator, BarManager, Indicator, Patterns
from pyalgoture.utils.logger import get_logger
from pyalgoture.utils.models import normalize_name

# from pyalgoture.utils.pnl import PositionPnL
from pyalgoture.utils.nav import FundTracker
from pyalgoture.utils.notification import NotificationSender
from pyalgoture.utils.objects import (
    AccountData,
    AssetType,
    BarData,
    Exchange,
    ExecutionType,
    InstrumentData,
    MarginMode,
    OptionType,
    OrderData,
    OrderType,
    PositionData,
    PositionMode,
    PositionSide,
    Side,
    Status,
    TickData,
    TimeInForce,
    TradeData,
)
from pyalgoture.utils.portfolio_manager import (
    PortfolioManager,  # , PortfolioManagerConnector
)
from pyalgoture.utils.simple_mq import SignalMQ

# from pyalgoture.utils.util_auth import authentication, set_user_info
from pyalgoture.utils.util_connection import (
    _check_permission_error_msg,
    fetch_account,
    fetch_history,
    fetch_open_orders,
    fetch_permission,
    fetch_position,
    fetch_price,
    fetch_symbols,
    fetch_trades,
    get_exchange_connection,
)
from pyalgoture.utils.util_dt import (
    CHINA_TZ,
    LOCAL_TZ,
    UTC_TZ,
    convert_timezone,
    find_closest_datetime,
    find_nearest_datetime,
    is_first_bday_of_mth,
    parse2datetime,
    time2sec,
    tz_manager,
)
from pyalgoture.utils.util_io import (
    create_report_folder,
    dump_conf,
    get_engine_config,
    read_conf,
    read_csv,
    retrieve_strategy_class,
    retrieve_strategy_name,
)
from pyalgoture.utils.util_math import (
    almost_equal,
    ceil_to,
    floor_to,
    prec_round,
    round_to,
    scale_number,
)
from pyalgoture.utils.util_objects import AttrDict, set_default
from pyalgoture.utils.util_schedule import Scheduler
from pyalgoture.utils.util_suuid import ShortUUID, int_to_string, string_to_int
from pyalgoture.utils.util_system import deprecated


# Lazy imports for database classes
class LazyMysqlDatabase:
    _instance = None

    def __new__(cls, *args, **kwargs) -> Any:
        try:
            import peewee  # noqa
            import pymysql  # noqa
            from pyalgoture.databases.mysql_db import MysqlDatabase

            return MysqlDatabase(*args, **kwargs)
        except ImportError:
            raise ImportError("MysqlDatabase required package 'peewee' & 'pymysql'. pip install peewee pymysql")


class LazyDuckdbDatabase:
    _instance = None

    def __new__(cls, *args, **kwargs) -> Any:
        try:
            import duckdb  # noqa
            from pyalgoture.databases.duckdb_db import DuckdbDatabase

            return DuckdbDatabase(*args, **kwargs)
        except ImportError:
            raise ImportError("DuckdbDatabase required package 'duckdb'. pip install duckdb")


class LazySqliteDatabase:
    _instance = None

    def __new__(cls, *args, **kwargs) -> Any:
        from pyalgoture.databases.sqlite_db import SqliteDatabase

        return SqliteDatabase(*args, **kwargs)


# Expose lazy versions
MysqlDatabase = LazyMysqlDatabase
DuckdbDatabase = LazyDuckdbDatabase
SqliteDatabase = LazySqliteDatabase


### Needa install extra library
# from pyalgoture.utils.behavioural import Trader
# from pyalgoture.utils.plotter import Plotter
# from pyalgoture.utils.preprocessor import CointAnalysis, KalmanFilter, GaussianHMM, PCA, FirstValueScaler, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, ADF

__version__ = "0.5.1"


### pdoc ignore modules/functions/classes
__pdoc__ = {
    "quick_start": False,
    "pyalgoture.datafeeds.binance_datafeed": False,
    "pyalgoture.datafeeds.bybit_datafeed": False,
    "pyalgoture.datafeeds.eqonex_datafeed": False,
    "pyalgoture.datafeeds.okx_datafeed": False,
    "pyalgoture.exchanges": False,
    "pyalgoture.utils.deploy": False,
    "pyalgoture.utils.docker_utils": False,
    "pyalgoture.utils.tmux_util": False,
    "pyalgoture.utils.hot_reloader": False,
    "pyalgoture.utils.auth": False,
    "pyalgoture.utils.cli": False,
    "pyalgoture.utils.logger": False,
    "pyalgoture.utils.pnl": False,
    "pyalgoture.utils.rest_client": False,
    "pyalgoture.utils.ws_client": False,
    "pyalgoture.utils.schedule": False,
    "pyalgoture.utils.models": False,
    "pyalgoture.utils.liqudation": False,
    "pyalgoture.utils.event_engine": False,
    "pyalgoture.utils.portopt": False,
    "pyalgoture.utils.indicator.BarManager": False,
    "pyalgoture.utils.indicator.Patterns": False,
    "pyalgoture.utils.utility.retrieve_strategy_name": False,
    "pyalgoture.utils.utility.retrieve_strategy_class": False,
    "pyalgoture.utils.utility.AttrDict": False,
    "pyalgoture.utils.utility.add_default_tz": False,
    "pyalgoture.utils.utility.time2sec": False,
    "pyalgoture.utils.utility.set_default": False,
    "pyalgoture.utils.utility.parse2datetime": False,
    "pyalgoture.utils.utility.is_first_bday_of_mth": False,
    "pyalgoture.utils.utility.deprecated": False,
    "pyalgoture.utils.utility.custom_dict": False,
    "pyalgoture.utils.utility.timeout_input": False,
    # "pyalgoture.utils.cex_connection.fetch_account": False,
    # "pyalgoture.utils.cex_connection.fetch_history": False,
    # "pyalgoture.utils.cex_connection.fetch_open_orders": False,
    # "pyalgoture.utils.cex_connection.fetch_permission": False,
    # "pyalgoture.utils.cex_connection.fetch_position": False,
    # "pyalgoture.utils.cex_connection.fetch_price": False,
    # "pyalgoture.utils.cex_connection.fetch_symbols": False,
    # "pyalgoture.utils.cex_connection.fetch_trades": False,
    # "pyalgoture.utils.cex_connection.get_exchange_connection": False,
    "pyalgoture.utils.auth_util": False,
    "pyalgoture.utils.preprocessor": False,
    "pyalgoture.utils.simple_webserver": False,
    "pyalgoture.utils.simple_mq": False,
    "pyalgoture.statistic": False,
    "pyalgoture.plotter": False,
    "pyalgoture.broker": False,
    "pyalgoture.datafeed": False,
    "pyalgoture.brokers.realtime_broker.RealtimeBroker": False,
    "pyalgoture.brokers.backtest_broker.BacktestBroker.add_connection": False,
    "pyalgoture.brokers.backtest_broker.BacktestBroker.close": False,
    "pyalgoture.brokers.backtest_broker.BacktestBroker.reset": False,
    "pyalgoture.brokers.backtest_broker.BacktestBroker.connect": False,
    "pyalgoture.brokers.backtest_broker.BacktestBroker.subscribe": False,
    # "pyalgoture.datafeed.DataFeed": False,
    # "pyalgoture.datafeed.CryptoDataFeed": False,
    # "pyalgoture.datafeed.StockDataFeed": False,
    "pyalgoture.base.Base.log": False,
    "pyalgoture.context.Context.set_bar_data": False,
    "pyalgoture.context.Context.info": False,
    "pyalgoture.context.Context.debug": False,
    "pyalgoture.context.Context.warning": False,
    "pyalgoture.context.Context.critical": False,
    "pyalgoture.context.Context.error": False,
    "pyalgoture.strategy.Strategy.__init__": False,
    "pyalgoture.strategy.Strategy.run": False,
    # "pyalgoture.strategy.Strategy.order": False,
    "pyalgoture.strategy.Strategy.on_order": False,
    "pyalgoture.strategy.Strategy.set_bar_data": False,
    "pyalgoture.strategy.Strategy.process_timer_event": False,
    "pyalgoture.strategy.Strategy.process_account_event": False,
    "pyalgoture.strategy.Strategy.process_bar_event": False,
    "pyalgoture.strategy.Strategy.process_tick_event": False,
    "pyalgoture.strategy.Strategy.process_log_event": False,
    "pyalgoture.strategy.Strategy.process_position_event": False,
    "pyalgoture.strategy.Strategy.process_order_event": False,
    "pyalgoture.strategy.Strategy.process_trade_event": False,
    "pyalgoture.strategy.Strategy.register_event": False,
    "pyalgoture.orchestra": False,
    "pyalgoture.orchestrators.realtime.RealTime": False,
    "pyalgoture.orchestrators.realtime_multi": False,
    # "pyalgoture.orchestrators.realtime.add_tickdata": False,
    # "pyalgoture.orchestrators.realtime.add_ticker": False,
    # "pyalgoture.orchestrators.realtime.start": False,
    # "pyalgoture.orchestra.Orchestrator.export_csv": False,
    # "pyalgoture.orchestra.Orchestrator.export_json": False,
    # "pyalgoture.orchestra.Orchestrator.timeout_input": False,
    # "pyalgoture.orchestra.Orchestrator.update_strategytegy": False,
    # "pyalgoture.orchestra.Orchestrator.reset": False,
    # "pyalgoture.orchestra.Orchestrator.start": False
    "pyalgoture.rpc": False,
    "pyalgoture.database": False,
    "pyalgoture.databases.arcticdb_db": False,
    "pyalgoture.databases.influxdb_db": False,
    "pyalgoture.databases.postgresql_db": False,
    "pyalgoture.databases.sqlite_db": False,
    "pyalgoture.databases.timescaledb_db": False,
    "pyalgoture.databases.mongodb_db": False,
    "pyalgoture.databases.mysqldb_db.ReconnectMySQLDatabase": False,
    "pyalgoture.databases.mysqldb_db.DbBarData": False,
    "pyalgoture.databases.mysqldb_db.DbBarOverview": False,
    "pyalgoture.databases.mysqldb_db.DbTickData": False,
    "pyalgoture.databases.mysqldb_db.DbTickOverview": False,
    "pyalgoture.databases.mysqldb_db.DateTimeMillisecondField": False,
}


# def quick_start(
#     strategyClass: Strategy,
#     broker: str,
#     balance: float,
#     pairs: list,
#     interval: str,
#     start_date: datetime,
#     end_date: datetime,
#     base_path: str = None,
#     exchange: str = "bybit",
#     asset_type: str = "future",
#     benchmark_code: str = "BTCUSDT",
#     is_realtime: bool = False,
#     trade_history: list = [],
#     hist_trade_skip_datafeed: bool = False,
#     hist_trade_skip_checking: bool = False,
#     leverage: float = None,
#     **kwargs,
# ) -> Orchestrator:
#     """
#     --strategy: path of strategy py file
#     --balance: balance
#     --pairs: list of symbols
#     --interval:
#     --start: start date
#     --end
#     --path
#     --basecurr
#     --benchmark
#     --broker

#     --commission_rate
#     --min_commission
#     """
#     # print("pyalgoture: Hello, world!!")
#     balance = float(balance)
#     if base_path:
#         base_path = create_report_folder(base_path)

#     if leverage:
#         # TODO:
#         pass

#     if broker == "local":
#         broker = BacktestBroker(
#             trade_at="close",
#             balance=balance,
#             short_cash=False,
#             slippage=0.0,
#         )

#     else:
#         return None

#     if is_realtime:
#         pass
#         # mytest = RealTime(
#         #     symbols=[symbol],
#         #     StrategyClass=strategyClass,
#         #     benchmark_symbol=benchmark_code,
#         #     store_path=base,
#         #     broker= broker
#         # )
#         # # mytest.start(rt=True, include_bt=False, is_console=False, minutes=5)
#         # # mytest.start(is_console=True, minutes=1, start_date='2021-10-22 15:00:00', end_date='2021-12-14 08:00:00')
#         # mytest.start(is_console=True, interval='1h', hours=1, start_date='2021-10-10 23:47:00', end_date='2021-12-14 08:00:00')

#     else:
#         benchmark_df = None
#         if benchmark_code:
#             benchmark_df = CryptoDataFeed(
#                 code=benchmark_code,
#                 exchange=exchange,
#                 asset_type=asset_type,
#                 start_date=start_date,
#                 end_date=end_date,
#                 interval=interval,
#                 store_path=base_path,
#                 **kwargs,
#             )

#         dfs = []
#         for data in pairs:
#             if isinstance(data, str):
#                 symbol = data
#                 # exchange = exchange
#                 # asset_type = asset_type
#             else:
#                 symbol = data["symbol"]
#                 exchange = data["exchange"].lower()
#                 asset_type = data["asset_type"]
#                 start_date = data.get("start_date", start_date)
#                 end_date = data.get("end_date", end_date)

#             asset_type = asset_type.upper()

#             df = CryptoDataFeed(
#                 code=symbol,
#                 exchange=exchange,
#                 asset_type=asset_type,
#                 start_date=start_date,
#                 end_date=end_date,
#                 interval=interval,
#                 order_ascending=True,
#                 store_path=base_path,
#                 is_live=True if hist_trade_skip_datafeed else False,
#                 **kwargs,
#             )

#             time.sleep(0.1)
#             dfs.append(df)

#         if not dfs:
#             dfs.append(benchmark_df)
#         # return
#         statistic = Statistic()
#         mytest = BackTest(
#             feeds=dfs,
#             StrategyClass=strategyClass,
#             benchmark=benchmark_df,
#             statistic=statistic,
#             store_path=base_path,
#             broker=broker,
#             trade_history=trade_history,
#             hist_trade_skip_datafeed=hist_trade_skip_datafeed,
#             hist_trade_skip_checking=hist_trade_skip_checking,
#             **kwargs,
#         )
#         # if leverage:
#         #     mytest.ctx.set_leverage(leverage=leverage, symbol=symbol, exchange=exchange)

#         mytest.start()

#     # prefix = ''
#     # stats = mytest.ctx.statistic.stats(interval=interval)  # annualization=365, risk_free=0.0442
#     # print(stats)

#     # mytest.start(rt=True, include_bt=False, is_console=True, minutes=1)

#     # mytest.ctx.statistic.contest_output(interval=interval, is_plot=False)

#     return mytest


# def submission(user_code: str, strategy_path: str, codes: list, **kwargs):
#     """strategy submission

#     The backtest will conduct in the same environment for all the submitted script

#     Environment:
#         1. Past 3 months backtest with 1 minute data
#         2. balance $100,000
#         3. 5x leverage
#         4. On bybit future

#     Criterias:
#         1. win_ratio > 0.51
#         2. sqn > 1.1
#         3. return_on_investment > 0.22
#         4. risk_score < 5
#         5. sharpe_ratio > 0.95
#         6. annualized_volatility < 0.18
#         7. max_drawdown < 0.19
#         8. ending_balance > beginning_balance
#         9. standardized_expected_value > 1

#     Args:
#         user_code (str): unique user code for pyalgoture user
#         strategy_path (str): strategy file path
#         codes (list): exchange symbols to conduct backtesting
#         **kwargs (dict): custom parameter specified for strategy

#     Returns:
#         None
#     """

#     # Beta is a measure of volatility relative to a benchmark ( >1.0 means return is more volatile than the benchmark; <1.0 indicates the return with lower volatility.)
#     # Alpha shows how well (or badly) the return has performed in comparison to a benchmark (A high alpha is always good.)
#     if not os.path.exists(strategy_path):
#         print(f"{strategy_path} is not existed.")
#         return
#     tmp_path = os.path.join(r"/tmp", "pyalgoture_tmp")

#     print("backtest is conducting, this may take some time...")
#     # TODO: download data concurrently
#     failed_metric = None
#     tdy = datetime.now()
#     past_90days = tdy - timedelta(days=30)
#     past_90days = tdy - timedelta(days=90)
#     bt = quick_start(
#         strategyClass=retrieve_strategy_class(strategy_path),
#         broker="local",
#         balance=100_000,
#         pairs=codes,
#         interval="1h",
#         start_date=past_90days,
#         end_date=tdy,
#         base_path=tmp_path,
#         benchmark_code=None,
#         base_currency="USDT",
#         **kwargs,
#     )
#     stats = bt.ctx.statistic.stats()
#     # print(stats)

#     # criterias = {
#     #     'win_ratio': 0.4,
#     #     'return_on_investment': 0.22,
#     #     'sqn': 1.01,
#     #     'max_drawdown': 0.05,
#     #     'sharpe_ratio': 0.95,
#     #     'risk_score': 5,
#     #     'annualized_volatility': 0.1
#     # }

#     for metric in [
#         "win_ratio",
#         "return_on_investment",
#         "sqn",
#         "max_drawdown",
#         "sharpe_ratio",
#         "risk_score",
#         "annualized_volatility",
#     ]:
#         if metric == "win_ratio" and stats["win_ratio"] < 0.4:
#             failed_metric = metric
#             break
#         elif metric == "return_on_investment" and stats["return_on_investment"] < 0.2:
#             failed_metric = metric
#             break
#         elif metric == "sqn" and stats["sqn"] < 1.02:
#             failed_metric = metric
#             break
#         elif metric == "max_drawdown" and stats["max_drawdown"] > 0.19:
#             failed_metric = metric
#             break
#         elif metric == "sharpe_ratio" and stats["sharpe_ratio"] < 0.95:
#             failed_metric = metric
#             break
#         elif metric == "risk_score" and stats["risk_score"] > 5:
#             failed_metric = metric
#             break
#         elif (
#             metric == "annualized_volatility" and stats["annualized_volatility"] > 0.18
#         ):
#             failed_metric = metric
#             break

#     if (
#         metric == "ending_balance"
#         and stats["ending_balance"] > stats["beginning_balance"]
#     ):
#         failed_metric = metric

#     if failed_metric:
#         print(
#             f"The strategy failed in {failed_metric}, please tune your strategy and submited later."
#         )
#     else:
#         # send to ????
#         import requests

#         files = {"file": open(strategy_path, "rb")}
#         res = requests.post(
#             "https://api-dev.pyalgoture.com/file",
#             params={"bucket": "pyalgoture", "folder": f"strategies/{user_code}"},
#             files=files,
#         )
