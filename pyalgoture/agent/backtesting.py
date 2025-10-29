from collections import defaultdict
from datetime import timedelta
from typing import Any

from ..brokers.backtest_broker import BacktestBroker
from ..datafeeds.crypto_datafeed import CryptoDataFeed
from ..orchestrators.backtest import BackTest
from ..statistic import Statistic
from ..strategy import Strategy
from ..utils.objects import BarData, Exchange, OrderData, TradeData
from ..utils.util_io import create_report_folder
from ..utils.util_math import round_to
from .agent import Agent


class AgentStrategy(Strategy):
    agent: Agent

    def initialize(self) -> None:
        if not self.agent:
            raise ValueError("agent is not initialized")
        self.symbol, self.exchange = self.ctx.symbols[0]
        self.feed = self.ctx.get_datafeed(symbol=self.symbol, exchange=self.exchange)
        # self.ctx.set_leverage(symbol=self.symbol, exchange=self.exchange, leverage=self.leverage)
        # self.ctx.set_margin_mode(symbol=self.symbol, exchange=self.exchange, is_isolated=False)

        # Get commission rates and step size
        self.commission_rate = (
            self.feed.commission_rate if self.feed.commission_rate else self.feed.taker_commission_rate
        )
        self.step_size = self.feed.step_size

    def finish(self) -> None:
        pass

    def on_order_fail(self, order: Any) -> None:
        pass

    def on_order_cancel(self, order: Any) -> None:
        pass

    def on_order_complete(self, order: Any) -> None:
        pass

    def on_trade(self, trade: Any) -> None:
        pass

    def on_bar(self, tick: Any) -> None:
        bar: BarData = self.ctx.get_bar_data(symbol=self.symbol, exchange=self.exchange)
        position = self.ctx.get_position(symbol=self.symbol, exchange=self.exchange, force_return=True)
        end_date = tick
        td_payload = {"hours": 130} if "h" in self.agent.interval else {"days": 130}
        start_date = end_date - timedelta(**td_payload)
        self.debug(f">>>> start_date: {start_date}, end_date: {end_date}")
        stats = self.ctx.statistic.stats(lite=True, return_dict=True)
        self.debug(f"Stats: {stats}")

        result = self.agent.run(
            start_date=start_date, end_date=end_date, position=position, account=self.ctx.default_account
        )
        decision = result.get("decision", {})
        analysis = result.get("analysis", {})
        self.debug(f">>>> Decision: {decision}")
        # self.debug(f"Market Data: {analysis['market_data']['ohlcv'].iloc[0]} & {analysis['market_data']['ohlcv'].iloc[-1]}")
        self.debug(f"Pattern Analysis: {analysis['pattern_analysis']}")
        self.debug(f"Quant Analysis: {analysis['quant_analysis']}")
        self.debug(f"Risk Analysis: {analysis['risk']}")
        if not decision:
            return

        self.execute_trading(
            symbol=self.symbol,
            exchange=self.exchange,
            quantity=decision["quantity"],
            action=decision["action"],
            current_price=bar.close,
        )

    def execute_trading(
        self, symbol: str, exchange: Exchange, quantity: float, action: str, current_price: float
    ) -> float:
        """Execute position adjustment based on targets with portfolio management"""
        if quantity <= 0.0:
            return 0.0

        # position = self.ctx.get_position(symbol=symbol, exchange=exchange, force_return=True)

        if action in ["buy", "cover"]:
            order = self.buy(symbol=symbol, exchange=exchange, quantity=quantity, tag=action.upper())
            self.info(f"Buy executed: {symbol}, quantity: {quantity}, price: {current_price} | Action: {action}")
            return float(order.quantity)

        elif action in ["sell", "short"]:
            order = self.sell(symbol=symbol, exchange=exchange, quantity=quantity, tag=action.upper())
            self.info(f"Sell executed: {symbol}, quantity: {quantity}, price: {current_price} | Action: {action}")
            return float(order.quantity)

        return 0.0


def run_backtest(
    settings: dict,
    start_date: str,
    end_date: str,
    path: str,
    db: Any = None,
    capital: float = 100_000,
    benchmark_code: str = "BTCUSDT",
) -> BackTest:
    agent = Agent(**settings)
    symbol, exchange = settings["symbol"], settings["exchange"]
    asset_type: str = settings["asset_type"]
    interval: str = settings["interval"]

    base = create_report_folder(path)
    feeds = []

    feed = CryptoDataFeed(
        code=symbol,
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
        AgentStrategy,
        statistic=statistic,
        benchmark=benchmark,
        store_path=base,
        broker=broker,
        # do_print =False,
        agent=agent,
    )

    engine.start()

    stats = engine.ctx.statistic.stats(interval=interval)
    print(stats)
    engine.ctx.statistic.plot(path=base, interval=interval, is_plot=True, open_browser=False)
    return engine
