from typing import Any

from ..brokers.backtest_broker import BacktestBroker
from ..datafeeds.crypto_datafeed import CryptoDataFeed
from ..orchestrators.backtest import BackTest
from ..statistic import Statistic
from ..utils.objects import BarData, TradeData
from ..utils.util_io import create_report_folder
from ..utils.util_math import round_to
from .datasets.utils.utility import Segment
from .lab import AlphaLab
from .strategy import AlphaStrategy


def run_backtest(
    lab_path: str,
    name: str,
    index: str,
    StrategyClass: type[AlphaStrategy],
    StrategyParams: dict[str, Any],
    segment: Segment = Segment.TEST,
    db: Any = None,
    capital: float = 100_000,
    benchmark_code: str = "BTCUSDT",
) -> BackTest:
    lab = AlphaLab(lab_path=lab_path)

    components = lab.load_component_symbols(index)
    component_symbols = [d["symbol"] for d in components]
    print(f"components:{components}; component_symbols: {component_symbols}")

    signal = lab.load_signal(name)
    dataset = lab.load_dataset(name)
    print(f"signal: {signal}")

    if dataset is None:
        raise ValueError("Dataset is None")

    start_date = dataset.data_periods[segment][0]
    end_date = dataset.data_periods[segment][1]
    print(f"start_date: {start_date}, end_date: {end_date}")
    base = create_report_folder(str(lab.report_path))
    feeds = []
    for component in components:
        code: str = component["code"]
        exchange: str = component["exchange"]
        asset_type: str = component["asset_type"]
        interval: str = component["interval"]
        feed = CryptoDataFeed(
            code=code,
            exchange=exchange,
            asset_type=asset_type,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            db=db,
        )
        feeds.append(feed)

    benchmark = CryptoDataFeed(
        code=benchmark_code,
        exchange=exchange,
        asset_type=asset_type,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        order_ascending=True,
        db=db,
    )

    statistic = Statistic()

    broker = BacktestBroker(
        balance=capital,
    )
    engine = BackTest(
        feeds,
        StrategyClass,
        statistic=statistic,
        benchmark=benchmark,
        store_path=base,
        broker=broker,
        # do_print =False,
        lab=lab,
        signal_df=signal,
        exchange=exchange,
        **StrategyParams,
    )

    engine.start()

    stats = engine.ctx.statistic.stats(interval=interval)
    print(stats)
    engine.ctx.statistic.plot(path=base, interval=interval, is_plot=True, open_browser=False)
    return engine
