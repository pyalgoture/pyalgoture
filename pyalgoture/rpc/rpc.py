"""
This module contains class to define a RPC communications
"""

from abc import abstractmethod
from datetime import date, datetime, timedelta, timezone
from typing import Any

import psutil
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzlocal

from ..orchestrators.realtime import RealTime
from ..utils.logger import bufferHandler, get_logger
from ..utils.objects import PositionSide, Side, State, TradeData
from ..utils.util_dt import format_date, timeframe_to_minutes, timeframe_to_msecs
from .rpc_db import RpcDatabase
from .rpc_types import RPCSendMsg

logger = get_logger()


class PricingError(Exception):
    """
    Subclass of DependencyException.
    Indicates that the price could not be determined.
    Implicitly a buy / sell operation.
    """


class ExchangeError(Exception):
    """
    Error raised out of the exchange.
    Has multiple Errors to determine the appropriate error.
    """


class RPCException(Exception):
    """
    Should be raised with a rpc-formatted message in an _rpc_* method
    if the required state is wrong, i.e.:

    raise RPCException('*Status:* `no active trade`')
    """

    def __init__(self, message: str) -> None:
        super().__init__(self)
        self.message = message

    def __str__(self):
        return self.message

    def __json__(self):
        return {"msg": self.message}


class RPCHandler:
    def __init__(self, rpc: "RPC", config: dict) -> None:
        """
        Initializes RPCHandlers
        :param rpc: instance of RPC Helper class
        :param config: Configuration object
        :return: None
        """
        self._rpc = rpc
        self._config: dict = config

    @property
    def name(self) -> str:
        """Returns the lowercase name of the implementation"""
        return self.__class__.__name__.lower()

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup pending module resources"""

    @abstractmethod
    def send_msg(self, msg: RPCSendMsg) -> None:
        """Sends a message to all registered rpc modules"""


class RPC:
    """
    RPC class can be used to have extra feature, like bot data, and access to DB data
    """

    def __init__(self, engine: RealTime, config, db: RpcDatabase) -> None:
        """
        Initializes all enabled rpc modules
        :param engine: Instance of a Orchestrator
        :param config: Configuration object
        :param db: RpcDatabase instance
        :return: None
        """
        self._engine = engine
        self._config: dict = config
        self._db: RpcDatabase = db
        self._strategy_name: str = self._engine.StrategyClass.__name__

    def _rpc_start(self) -> dict[str, str]:
        """Handler for start"""
        if self._engine.state == State.RUNNING:
            return {"status": "already running"}

        self._engine.state = State.RUNNING
        # NOTE: start the engine & set_startup_time
        self._engine.start(keep_alive=False)
        self._engine.startup_time = self._engine.ctx.now

        if hasattr(self._engine, "rpc_manager"):
            self._engine.rpc_manager.startup_messages(
                {
                    "strategy": self._strategy_name,
                    "bot_startup": self._engine.startup_time,
                    "state": self._engine.state,
                    "symbols": self._engine.ctx.symbols,
                }
            )

        stra = self._db.load_strategy(strategy=self._strategy_name)
        if not stra:
            self._db.save_strategy(
                stra_data={
                    "strategy": self._strategy_name,
                    "bot_start": None,
                    "bot_startup": self._engine.startup_time,
                    "last_recon": None,
                }
            )
        print(f">>>> [_rpc_start] self._engine.state:{self._engine.state}; db stra:{stra}")
        return {"status": "starting trader ..."}

    def _rpc_stop(self, exit_all: bool = False, reports: bool = True) -> dict[str, str]:
        """Handler for stop"""
        if self._engine.state == State.RUNNING:
            self._engine.state = State.STOPPED
            if exit_all:
                for pos in self._engine.ctx.get_positions():
                    if pos.size or pos.avg_open_price or pos.total_pnl:
                        o = self._engine.ctx.strategy.close(
                            symbol=pos.symbol,
                            exchange=pos.exchange,
                            position_side=pos.position_side,
                            tag="One click exit",
                        )
                        print(f"[One click exit] pos: {pos} is closed. \n\t\tRef Order: {o}")

            for runner in self._engine._full_runner_list:
                # print(f"[{runner}] finish func running...")
                runner.finish()

            self._engine.broker.close()
            self._engine.event_engine.stop()
            if reports:
                stats = self._engine.ctx.statistic.stats()  # lite=True
                print(stats)

                self._engine.ctx.statistic.contest_output()

            return {"status": "stopping trader ..."}

        return {"status": "already stopped"}

    def _rpc_reload_config(self) -> dict[str, str]:
        """Handler for reload_config."""
        self._engine.state = State.RELOAD_CONFIG
        return {"status": "Reloading config ..."}

    def health(self) -> dict[str, str | int | None]:
        """
        bot_start: first trade time
        bot_startup: engine start running time
        last_process: last process time
        """
        last_p = self._engine.last_process
        res: dict[str, None | str | int] = {
            "last_process": None,
            "last_process_loc": None,
            "last_process_ts": None,
            "bot_start": None,
            "bot_start_loc": None,
            "bot_start_ts": None,
            "bot_startup": None,
            "bot_startup_loc": None,
            "bot_startup_ts": None,
        }

        if last_p is not None:
            res.update(
                {
                    "last_process": str(last_p),
                    "last_process_loc": format_date(last_p.astimezone(tzlocal())),
                    "last_process_ts": int(last_p.timestamp()),
                }
            )
        trades = self._engine.ctx.get_trades() if hasattr(self._engine, "ctx") else []
        bot_start = trades[0].traded_at if trades else None
        if bot_start:  # NOTE: first trade time
            res.update(
                {
                    "bot_start": str(bot_start),
                    "bot_start_loc": format_date(bot_start.astimezone(tzlocal())),
                    "bot_start_ts": int(bot_start.timestamp()),
                }
            )
        if bot_startup := self._engine.startup_time:  # NOTE: engine start running time
            res.update(
                {
                    "bot_startup": str(bot_startup),
                    "bot_startup_loc": format_date(bot_startup.astimezone(tzlocal())),
                    "bot_startup_ts": int(bot_startup.timestamp()),
                }
            )

        # if bot_start := KeyValueStore.get_datetime_value(KeyStoreKeys.BOT_START_TIME):
        #     res.update(
        #         {
        #             "bot_start": str(bot_start),
        #             "bot_start_loc": format_date(bot_start.astimezone(tzlocal())),
        #             "bot_start_ts": int(bot_start.timestamp()),
        #         }
        #     )
        # if bot_startup := KeyValueStore.get_datetime_value(KeyStoreKeys.STARTUP_TIME):
        #     res.update(
        #         {
        #             "bot_startup": str(bot_startup),
        #             "bot_startup_loc": format_date(bot_startup.astimezone(tzlocal())),
        #             "bot_startup_ts": int(bot_startup.timestamp()),
        #         }
        #     )
        print(f">>>> [health api] res: {res}")
        return res

    @staticmethod
    def _rpc_sysinfo() -> dict[str, Any]:
        return {
            "cpu_pct": psutil.cpu_percent(interval=1, percpu=True),
            "ram_pct": psutil.virtual_memory().percent,
        }

    @staticmethod
    def _rpc_get_logs(limit: int | None) -> dict[str, Any]:
        """Returns the last X logs"""
        if limit:
            buffer = bufferHandler.buffer[-limit:]
        else:
            buffer = bufferHandler.buffer
        records = [
            [
                format_date(datetime.fromtimestamp(r.created)),
                r.created * 1000,
                r.name,
                r.levelname,
                r.msg + ("\n" + r.exc_text if r.exc_text else ""),
            ]
            for r in buffer
        ]

        return {"log_count": len(records), "logs": records}

    def __get_last_price(self, position):
        last_price = None
        tick_data = self._engine.ctx.get_tick_data(position.symbol, position.exchange)
        if not tick_data:
            print(f">>>> [__get_last_price] tick_data is None; position:{position}")
            return None
        if position.position_side == PositionSide.NET:
            last_price = tick_data.get("last_price")
        else:
            if position.position_side == PositionSide.LONG:
                last_price = tick_data.get("bid_price_1")
            else:
                last_price = tick_data.get("ask_price_1")
        return last_price

    def _rpc_positions(self, symbols: list[str] | None = None, open_only: bool = False) -> list[dict[str, Any]]:
        """
        Below follows the RPC backend it is prefixed with rpc_ to raise awareness that it is
        a remotely exposed function
        """
        if self._engine.state != State.RUNNING:
            raise RPCException("bot is not running")
        # if symbol:
        #     positions = [self._engine.ctx.get_position(symbol)]
        # else:
        positions = self._engine.ctx.get_positions()

        if not positions:
            raise RPCException("no active positions")
        else:
            results = []
            """
            NOTE: calculate realtime pnl (mode: tick, ws, bar)
            if mode == bar, use last price to calculate pnl (get_bar_data)
            if mode == tick/ws, use ask/bid price to calculate pnl (get_tick_data)
            if is long position, use bid price to calculate pnl(sell at bid price)
            if is short position, use ask price to calculate pnl(buy at ask price)
            """
            # print(f">>>> [_rpc_positions] positions:{positions}")
            for position in positions:
                if position.size:
                    last_price = self.__get_last_price(position=position)
                    if last_price:
                        position.update_by_marketdata(
                            last_price=last_price,
                        )
                    if open_only and not position.is_open:
                        continue
                    if symbols and position.symbol not in symbols:
                        continue
                    results.append(position)
            return results

    def _rpc_tradepnls(
        self,
        limit: int,
        offset: int = 0,
        order_by_id: bool = False,
        symbols: list[str] | None = None,
        open_only: bool = False,
    ) -> dict:
        """Returns the X last trades"""
        if self._engine.state != State.RUNNING:
            raise RPCException("bot is not running")
        all_tradepnls = []

        # if symbol:
        #     positions = [self._engine.ctx.get_position(symbol)]
        # else:
        positions = self._engine.ctx.get_positions()

        if not positions:
            raise RPCException("no active positions")
        for position in positions:
            for tradepnl in position.tradepnls:
                if open_only and not tradepnl.is_open:
                    continue
                if symbols and tradepnl.symbol not in symbols:
                    continue
                all_tradepnls.append(tradepnl)

        total_tradepnls = len(all_tradepnls)
        if limit:
            output = all_tradepnls[offset : offset + limit]
        else:
            output = all_tradepnls[offset:]

        # order_by: Any = TradeData.id if order_by_id else TradeData.close_date.desc()
        # if limit:
        #     trades = TradeData.session.scalars(
        #         TradeData.get_trades_query([TradeData.is_open.is_(False)])
        #         .order_by(order_by)
        #         .limit(limit)
        #         .offset(offset)
        #     )
        # else:
        #     trades = TradeData.session.scalars(
        #         TradeData.get_trades_query([TradeData.is_open.is_(False)]).order_by(TradeData.close_date.desc())
        #     )

        # output = [trade.to_json() for trade in trades]
        # total_trades = TradeData.session.scalar(
        #     select(func.count(TradeData.id)).filter(TradeData.is_open.is_(False))
        # )
        return {
            "tradepnls": output,
            "tradepnls_count": len(output),
            "offset": offset,
            "total_tradepnls": total_tradepnls,
        }

    def _rpc_trades(
        self,
        limit: int,
        offset: int = 0,
        order_by_id: bool = False,
        symbols: list[str] | None = None,
    ) -> dict:
        """Returns the X last trades"""
        if self._engine.state != State.RUNNING:
            raise RPCException("bot is not running")
        all_trades = []

        trades = self._engine.ctx.get_trades()

        # if not trades:
        #     raise RPCException("no active trades")
        for trade in trades:
            if symbols and trade.symbol not in symbols:
                continue
            all_trades.append(trade)

        total_trades = len(all_trades)
        if limit:
            output = all_trades[offset : offset + limit]
        else:
            output = all_trades[offset:]

        return {
            "trades": output,
            "trades_count": len(output),
            "offset": offset,
            "total_trades": total_trades,
        }

    def _rpc_open_orders(
        self,
        limit: int,
        offset: int = 0,
        order_by_id: bool = False,
        symbols: list[str] | None = None,
    ) -> dict:
        """Returns the X last trades"""
        if self._engine.state != State.RUNNING:
            raise RPCException("bot is not running")
        all_orders = []

        orders = self._engine.ctx.get_open_orders()

        # if not orders:
        #     raise RPCException("no active open orders")
        for order in orders:
            if symbols and order.symbol not in symbols:
                continue
            all_orders.append(order)

        total_orders = len(all_orders)
        if limit:
            output = all_orders[offset : offset + limit]
        else:
            output = all_orders[offset:]

        return {
            "orders": output,
            "orders_count": len(output),
            "offset": offset,
            "total_orders": total_orders,
        }

    def _rpc_entry_count(self) -> dict[str, float]:
        """Returns the number of trades running"""
        if self._engine.state != State.RUNNING:
            raise RPCException("bot is not running")

        # trades = self._engine.ctx.get_trades() if hasattr(self._engine, 'ctx') else []
        positions = self._engine.ctx.get_positions()

        tradepnls = []
        # if not positions:
        #     raise RPCException("no active positions")
        # else:
        for position in positions:
            for tradepnl in position.tradepnls:
                if tradepnl.is_open:
                    tradepnls.append(tradepnl)

        return {
            "current": len(tradepnls),
            "max": (
                int(self._config["max_open_trades"])
                if self._config.get("max_open_trades", float("inf")) != float("inf")
                else -1
            ),
            "total_trade_amount": sum((tradepnl.open_price * tradepnl.size) for tradepnl in tradepnls),
        }

    def _rpc_symbol_performance(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """
        Handler for performance.
        Shows a performance statistic from finished trades
        """
        if self._engine.state != State.RUNNING:
            raise RPCException("bot is not running")
        pair_metrics = {}
        if symbol:
            positions = [self._engine.ctx.get_position(symbol)]
        else:
            positions = self._engine.ctx.get_positions()

        if not positions:
            raise RPCException("no active positions")
        else:
            for position in positions:
                for tradepnl in position.tradepnls:
                    if tradepnl.is_open:
                        continue
                    symbol = tradepnl.symbol
                    if symbol not in pair_metrics:
                        pair_metrics[symbol] = {
                            "symbol": symbol,
                            "close_roi": 0.0,
                            "close_pnl": 0.0,
                            "count": 0,
                        }
                    pair_metrics[symbol]["close_roi"] += tradepnl.close_roi
                    pair_metrics[symbol]["close_pnl"] += tradepnl.close_pnl
                    pair_metrics[symbol]["count"] += 1
        return list(pair_metrics.values())

    def _rpc_balance(self, base_currency: str, fiat_display_currency: str) -> dict:
        """Returns current account balance per crypto"""
        if self._engine.state != State.RUNNING:
            raise RPCException("bot is not running")
        accounts = self._engine.ctx.get_accounts()
        print(f">>>> [_rpc_balance] accounts:{accounts}; default account:{self._engine.ctx.default_account}")
        """
        >>>> [_rpc_balance] accounts:[
        AccountData(asset='USDT', exchange=<Exchange.LOCAL: 'LOCAL'>, balance=100000.0, frozen=0, available=100000.0, initial_capital=100000.0, capital=100000.0, additional_capitals=OrderedDict()),
        AccountData(asset='USDC', exchange=<Exchange.BYBIT: 'BYBIT'>, balance=81867.12577398, frozen=0.0, available=81867.12577398, initial_capital=81867.12577398, capital=81867.12577398, additional_capitals=OrderedDict()),
        AccountData(asset='BTC', exchange=<Exchange.BYBIT: 'BYBIT'>, balance=1.0, frozen=0.0, available=1.0,  initial_capital=1.0, capital=1.0, additional_capitals=OrderedDict()),
        AccountData(asset='ETH', exchange=<Exchange.BYBIT: 'BYBIT'>, balance=1.0, frozen=0.0, available=1.0, initial_capital=1.0, capital=1.0, additional_capitals=OrderedDict()),
        AccountData(asset='USDT', exchange=<Exchange.BYBIT: 'BYBIT'>, balance=30466.57209605, frozen=0.0, available=30466.57209605, initial_capital=30466.57209605, capital=30466.57209605, additional_capitals=OrderedDict())]
        """
        if not accounts:
            raise RPCException("no active account")

        # currencies: list[dict] = []
        # total = 0.0
        # total_bot = 0.0

        # open_trades: list[TradeData] = TradeData.get_open_trades()
        # open_assets: dict[str, TradeData] = {t.safe_base_currency: t for t in open_trades}
        # self._engine.wallets.update(require_update=False)
        # starting_capital = self._engine.wallets.get_starting_balance()
        # # starting_cap_fiat = (
        # #     self._fiat_converter.convert_amount(
        # #         starting_capital, base_currency, fiat_display_currency
        # #     )
        # #     if self._fiat_converter
        # #     else 0
        # # )
        # coin: str
        # balance: Wallet
        # for coin, balance in self._engine.wallets.get_all_balances().items():
        #     if not balance.total and not balance.free:
        #         continue

        #     trade = open_assets.get(coin, None)
        #     is_bot_managed = coin == base_currency or trade is not None
        #     trade_amount = trade.amount if trade else 0
        #     if coin == base_currency:
        #         trade_amount = self._engine.wallets.get_available_stake_amount()

        #     try:
        # def __balance_get_est_stake(
        #     self, coin: str, base_currency: str, amount: float, balance: Wallet
        # ) -> tuple[float, float]:
        #     est_stake = 0.0
        #     est_bot_stake = 0.0
        #     if coin == base_currency:
        #         est_stake = balance.total
        #         if self._config.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT:
        #             # in Futures, "total" includes the locked stake, and therefore all positions
        #             est_stake = balance.free
        #         est_bot_stake = amount
        #     else:
        #         try:
        #             rate: float | None = self._engine.exchange.get_conversion_rate(
        #                 coin, base_currency
        #             )
        #             if rate:
        #                 est_stake = rate * balance.total
        #                 est_bot_stake = rate * amount

        #             return est_stake, est_bot_stake
        #         except (ExchangeError, PricingError) as e:
        #             logger.warning(f"Error {e} getting rate for {coin}")
        #             pass
        #     return est_stake, est_bot_stake

        #         est_stake, est_stake_bot = self.__balance_get_est_stake(
        #             coin, base_currency, trade_amount, balance
        #         )
        #     except ValueError:
        #         continue

        #     total += est_stake

        #     if is_bot_managed:
        #         total_bot += est_stake_bot
        #     currencies.append(
        #         {
        #             "currency": coin,
        #             "free": balance.free,
        #             "balance": balance.total,
        #             "used": balance.used,
        #             "bot_owned": trade_amount,
        #             "est_stake": est_stake or 0,
        #             "est_stake_bot": est_stake_bot if is_bot_managed else 0,
        #             "stake": base_currency,
        #             "side": "long",
        #             "position": 0,
        #             "is_bot_managed": is_bot_managed,
        #             "is_position": False,
        #         }
        #     )
        # symbol: str
        # position: PositionWallet
        # for symbol, position in self._engine.wallets.get_all_positions().items():
        #     total += position.collateral
        #     total_bot += position.collateral

        #     currencies.append(
        #         {
        #             "currency": symbol,
        #             "free": 0,
        #             "balance": 0,
        #             "used": 0,
        #             "position": position.position,
        #             "est_stake": position.collateral,
        #             "est_stake_bot": position.collateral,
        #             "stake": base_currency,
        #             "side": position.side,
        #             "is_bot_managed": True,
        #             "is_position": True,
        #         }
        #     )

        # # value = (
        # #     self._fiat_converter.convert_amount(total, base_currency, fiat_display_currency)
        # #     if self._fiat_converter
        # #     else 0
        # # )
        # # value_bot = (
        # #     self._fiat_converter.convert_amount(total_bot, base_currency, fiat_display_currency)
        # #     if self._fiat_converter
        # #     else 0
        # # )

        # starting_capital_ratio = (total_bot / starting_capital) - 1 if starting_capital else 0.0
        # # starting_cap_fiat_ratio = (value_bot / starting_cap_fiat) - 1 if starting_cap_fiat else 0.0

        # return {
        #     "currencies": currencies,
        #     "total": total,
        #     "total_bot": total_bot,
        #     "symbol": fiat_display_currency,
        #     # "value": value,
        #     # "value_bot": value_bot,
        #     "stake": base_currency,
        #     "starting_capital": starting_capital,
        #     "starting_capital_ratio": starting_capital_ratio,
        #     "starting_capital_pct": round(starting_capital_ratio * 100, 2),
        #     # "starting_capital_fiat": starting_cap_fiat,
        #     # "starting_capital_fiat_ratio": starting_cap_fiat_ratio,
        #     # "starting_capital_fiat_pct": round(starting_cap_fiat_ratio * 100, 2),
        #     "note": "Simulated balances" if self._config["dry_run"] else "",
        # }

        total = self._engine.ctx.default_account.balance
        starting_capital = self._engine.ctx.default_account.initial_capital
        starting_capital_ratio = (total / starting_capital) - 1 if starting_capital else 0.0
        return {
            "currencies": accounts,
            "total": total,
            # "total_bot": total_bot,
            "symbol": fiat_display_currency,
            # "value": value,
            # "value_bot": value_bot,
            "stake": base_currency,
            "starting_capital": starting_capital,
            "starting_capital_ratio": starting_capital_ratio,
            "starting_capital_pct": round(starting_capital_ratio * 100, 2),
            # "starting_capital_fiat": starting_cap_fiat,
            # "starting_capital_fiat_ratio": starting_cap_fiat_ratio,
            # "starting_capital_fiat_pct": round(starting_cap_fiat_ratio * 100, 2),
            # "note": "Simulated balances" if self._config["dry_run"] else "",
        }

    def _rpc_trade_status(self, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        """
        Below follows the RPC backend it is prefixed with rpc_ to raise awareness that it is
        a remotely exposed function
        """
        if self._engine.state != State.RUNNING:
            raise RPCException("bot is not running")
        # Fetch open trades
        tradepnls = []
        positions = self._engine.ctx.get_positions()
        for position in positions:
            for tradepnl in position.tradepnls:
                if tradepnl.is_open:
                    if symbols and tradepnl.symbol not in symbols:
                        continue
                    tradepnls.append(tradepnl)

        if not tradepnls:
            raise RPCException("no active trade")
        else:
            return tradepnls

    def _rpc_period_performance(
        self,
        timescale: int,
        base_currency: str = "USDT",
        fiat_display_currency: str = "USD",
        timeunit: str = "days",
        symbol: str | None = None,
    ) -> dict[str, Any]:
        """
        :param timeunit: Valid entries are 'days', 'weeks', 'months'
        """
        if self._engine.state != State.RUNNING:
            raise RPCException("bot is not running")
        start_date = datetime.now(timezone.utc).date()
        if timeunit == "weeks":
            # weekly
            start_date = start_date - timedelta(days=start_date.weekday())  # Monday
        if timeunit == "months":
            start_date = start_date.replace(day=1)

        def time_offset(step: int):
            if timeunit == "months":
                return relativedelta(months=step)
            return timedelta(**{timeunit: step})

        if not (isinstance(timescale, int) and timescale > 0):
            raise RPCException("timescale must be an integer greater than 0")

        profit_units: dict[date, dict] = {}
        # daily_stake = self._engine.wallets.get_total_stake_amount()
        daily_stake = self._engine.ctx.default_account.balance
        all_tradepnls = []

        print(
            f">>>>[_rpc_timeunit_profit] start_date:{start_date}; timescale:{timescale}; daily_stake:{daily_stake}(Current)"
        )
        if symbol:
            positions = [self._engine.ctx.get_position(symbol)]
        else:
            positions = self._engine.ctx.get_positions()

        if not positions:
            raise RPCException("no active positions")
        for position in positions:
            for tradepnl in position.tradepnls:
                all_tradepnls.append(tradepnl)

        for day in range(0, timescale):
            profitday: date = start_date - time_offset(day)

            tradepnls = [
                tradepnl
                for tradepnl in all_tradepnls
                if not tradepnl.is_open
                and tradepnl.close_date.date() >= profitday
                and tradepnl.close_date.date() < (profitday + time_offset(1))
            ]

            curdayprofit = sum(tradepnl.close_pnl for tradepnl in tradepnls if tradepnl.close_pnl is not None)
            # Calculate this periods starting balance
            daily_stake = daily_stake - curdayprofit
            print(
                f">>>>[_rpc_timeunit_profit] day: {day}, profitday:{profitday}; curdayprofit: {curdayprofit}; daily_stake:{daily_stake}"
            )
            profit_units[profitday] = {
                "amount": curdayprofit,
                "daily_stake": daily_stake,
                "rel_profit": round(curdayprofit / daily_stake, 8) if daily_stake > 0 else 0,
                "trades": len(tradepnls),
            }

        data = [
            {
                "date": key,
                "abs_profit": value["amount"],
                "starting_balance": value["daily_stake"],
                "rel_profit": value["rel_profit"],
                "fiat_value": 0.0,
                # "fiat_value": (
                #     self._fiat_converter.convert_amount(
                #         value["amount"], base_currency, fiat_display_currency
                #     )
                #     if self._fiat_converter
                #     else 0
                # ),
                "trade_count": value["trades"],
            }
            for key, value in profit_units.items()
        ]
        return {
            "base_currency": base_currency,
            "fiat_display_currency": fiat_display_currency,
            "data": data,
        }

    def _rpc_generic_performance(self, symbol: str | None = None) -> dict[str, Any]:
        """
        Generate generic stats for tradepnls
        """
        if self._engine.state != State.RUNNING:
            raise RPCException("bot is not running")

        def trade_win_loss(tradepnl):
            if tradepnl.close_roi > 0:
                return "wins"
            elif tradepnl.close_roi < 0:
                return "losses"
            else:
                return "draws"

        # Duration
        dur: dict[str, list[float]] = {"wins": [], "draws": [], "losses": []}
        # Exit reason
        exit_reasons = {}
        if symbol:
            positions = [self._engine.ctx.get_position(symbol)]
        else:
            positions = self._engine.ctx.get_positions()

        if not positions:
            raise RPCException("no active positions")
        else:
            for position in positions:
                for tradepnl in position.tradepnls:
                    if tradepnl.is_open:
                        continue
                    exit_reason = tradepnl.exit_reason if tradepnl.exit_reason else "Other"
                    if exit_reason not in exit_reasons:
                        exit_reasons[exit_reason] = {"wins": 0, "losses": 0, "draws": 0}
                    exit_reasons[exit_reason][trade_win_loss(tradepnl)] += 1

                    if tradepnl.close_date is not None and tradepnl.open_date is not None:
                        trade_dur = (tradepnl.close_date - tradepnl.open_date).total_seconds()
                        dur[trade_win_loss(tradepnl)].append(trade_dur)

        wins_dur = sum(dur["wins"]) / len(dur["wins"]) if len(dur["wins"]) > 0 else None
        draws_dur = sum(dur["draws"]) / len(dur["draws"]) if len(dur["draws"]) > 0 else None
        losses_dur = sum(dur["losses"]) / len(dur["losses"]) if len(dur["losses"]) > 0 else None

        durations = {"wins": wins_dur, "draws": draws_dur, "losses": losses_dur}
        return {"exit_reasons": exit_reasons, "durations": durations}

    def _rpc_performance(
        self,  # base_currency: str, fiat_display_currency: str, start_date: datetime | None = None
    ) -> dict[str, Any]:
        """Returns cumulative profit statistics"""
        if self._engine.state != State.RUNNING:
            raise RPCException("bot is not running")
        stats = self._engine.ctx.statistic.stats(
            # interval= self._engine.benchmark.interval if self._engine.benchmark else '1h',
            # benchmark=self._engine.benchmark,
            lite=True,
            return_dict=True,
        )
        for k in [
            "rolling_sharpe",
            "monthly_sharpe",
            "daily_changes",
            "asset_values",
            "drawdown_series",
            "positions",
            "trades",
            "pnl",
        ]:
            stats.pop(k, None)
        # print('>>>>[_rpc_trade_statistics - stats]',stats)
        return stats

        # start_date = datetime.fromtimestamp(0) if start_date is None else start_date

        # trade_filter = (
        #     TradeData.is_open.is_(False) & (TradeData.close_date >= start_date)
        # ) | TradeData.is_open.is_(True)
        # trades: Sequence[TradeData] = TradeData.session.scalars(
        #     TradeData.get_trades_query(trade_filter, include_orders=False).order_by(TradeData.id)
        # ).all()

        # profit_all_coin = []
        # profit_all_ratio = []
        # profit_closed_coin = []
        # profit_closed_ratio = []
        # durations = []
        # winning_trades = 0
        # losing_trades = 0
        # winning_profit = 0.0
        # losing_profit = 0.0

        # for trade in trades:
        #     current_rate: float = 0.0

        #     if trade.close_date:
        #         durations.append((trade.close_date - trade.open_date).total_seconds())

        #     if not trade.is_open:
        #         profit_ratio = tradepnl.close_roi or 0.0
        #         profit_abs = tradepnl.close_pnl or 0.0
        #         profit_closed_coin.append(profit_abs)
        #         profit_closed_ratio.append(profit_ratio)
        #         if profit_ratio >= 0:
        #             winning_trades += 1
        #             winning_profit += profit_abs
        #         else:
        #             losing_trades += 1
        #             losing_profit += profit_abs
        #     else:
        #         # Get current rate
        #         if len(trade.select_filled_orders(trade.entry_side)) == 0:
        #             # Skip trades with no filled orders
        #             continue
        #         try:
        #             current_rate = self._engine.exchange.get_rate(
        #                 trade.pair, side="exit", is_short=trade.is_short, refresh=False
        #             )
        #         except (PricingError, ExchangeError):
        #             current_rate = nan
        #             profit_ratio = nan
        #             profit_abs = nan
        #         else:
        #             _profit = trade.calculate_profit(trade.close_rate or current_rate)

        #             profit_ratio = _profit.profit_ratio
        #             profit_abs = _profit.total_profit

        #     profit_all_coin.append(profit_abs)
        #     profit_all_ratio.append(profit_ratio)

        # closed_trade_count = len([t for t in trades if not t.is_open])

        # best_pair = TradeData.get_best_pair(start_date)
        # trading_volume = TradeData.get_trading_volume(start_date)

        # # Prepare data to display
        # profit_closed_coin_sum = round(sum(profit_closed_coin), 8)
        # profit_closed_ratio_mean = float(mean(profit_closed_ratio) if profit_closed_ratio else 0.0)
        # profit_closed_ratio_sum = sum(profit_closed_ratio) if profit_closed_ratio else 0.0

        # # profit_closed_fiat = (
        # #     self._fiat_converter.convert_amount(
        # #         profit_closed_coin_sum, base_currency, fiat_display_currency
        # #     )
        # #     if self._fiat_converter
        # #     else 0
        # # )

        # profit_all_coin_sum = round(sum(profit_all_coin), 8)
        # profit_all_ratio_mean = float(mean(profit_all_ratio) if profit_all_ratio else 0.0)
        # # Doing the sum is not right - overall profit needs to be based on initial capital
        # profit_all_ratio_sum = sum(profit_all_ratio) if profit_all_ratio else 0.0
        # starting_balance = self._engine.wallets.get_starting_balance()
        # profit_closed_ratio_fromstart = 0
        # profit_all_ratio_fromstart = 0
        # if starting_balance:
        #     profit_closed_ratio_fromstart = profit_closed_coin_sum / starting_balance
        #     profit_all_ratio_fromstart = profit_all_coin_sum / starting_balance

        # profit_factor = winning_profit / abs(losing_profit) if losing_profit else float("inf")

        # winrate = (winning_trades / closed_trade_count) if closed_trade_count > 0 else 0

        # trades_df = DataFrame(
        #     [
        #         {
        #             "close_date": format_date(trade.close_date),
        #             "close_date_dt": trade.close_date,
        #             "profit_abs": tradepnl.close_pnl,
        #         }
        #         for trade in trades
        #         if not trade.is_open and trade.close_date
        #     ]
        # )

        # expectancy, expectancy_ratio = calculate_expectancy(trades_df)

        # drawdown = DrawDownResult()
        # if len(trades_df) > 0:
        #     try:
        #         drawdown = calculate_max_drawdown(
        #             trades_df,
        #             value_col="profit_abs",
        #             date_col="close_date_dt",
        #             starting_balance=starting_balance,
        #         )
        #     except ValueError:
        #         # ValueError if no losing trade.
        #         pass

        # # profit_all_fiat = (
        # #     self._fiat_converter.convert_amount(
        # #         profit_all_coin_sum, base_currency, fiat_display_currency
        # #     )
        # #     if self._fiat_converter
        # #     else 0
        # # )

        # first_date = trades[0].open_date_utc if trades else None
        # last_date = trades[-1].open_date_utc if trades else None
        # num = float(len(durations) or 1)
        # bot_start = KeyValueStore.get_datetime_value(KeyStoreKeys.BOT_START_TIME)
        # return {
        #     "profit_closed_coin": profit_closed_coin_sum,
        #     "profit_closed_percent_mean": round(profit_closed_ratio_mean * 100, 2),
        #     "profit_closed_ratio_mean": profit_closed_ratio_mean,
        #     "profit_closed_percent_sum": round(profit_closed_ratio_sum * 100, 2),
        #     "profit_closed_ratio_sum": profit_closed_ratio_sum,
        #     "profit_closed_ratio": profit_closed_ratio_fromstart,
        #     "profit_closed_percent": round(profit_closed_ratio_fromstart * 100, 2),
        #     # "profit_closed_fiat": profit_closed_fiat,
        #     "profit_all_coin": profit_all_coin_sum,
        #     "profit_all_percent_mean": round(profit_all_ratio_mean * 100, 2),
        #     "profit_all_ratio_mean": profit_all_ratio_mean,
        #     "profit_all_percent_sum": round(profit_all_ratio_sum * 100, 2),
        #     "profit_all_ratio_sum": profit_all_ratio_sum,
        #     "profit_all_ratio": profit_all_ratio_fromstart,
        #     "profit_all_percent": round(profit_all_ratio_fromstart * 100, 2),
        #     # "profit_all_fiat": profit_all_fiat,
        #     "trade_count": len(trades),
        #     "closed_trade_count": closed_trade_count,
        #     "first_trade_date": format_date(first_date),
        #     "first_trade_humanized": dt_humanize_delta(first_date) if first_date else "",
        #     "first_trade_timestamp": dt_ts_def(first_date, 0),
        #     "latest_trade_date": format_date(last_date),
        #     "latest_trade_humanized": dt_humanize_delta(last_date) if last_date else "",
        #     "latest_trade_timestamp": dt_ts_def(last_date, 0),
        #     "avg_duration": str(timedelta(seconds=sum(durations) / num)).split(".")[0],
        #     "best_pair": best_pair[0] if best_pair else "",
        #     "best_rate": round(best_pair[1] * 100, 2) if best_pair else 0,  # Deprecated
        #     "best_pair_profit_ratio": best_pair[1] if best_pair else 0,
        #     "winning_trades": winning_trades,
        #     "losing_trades": losing_trades,
        #     "profit_factor": profit_factor,
        #     "winrate": winrate,
        #     "expectancy": expectancy,
        #     "expectancy_ratio": expectancy_ratio,
        #     "max_drawdown": drawdown.relative_account_drawdown,
        #     "max_drawdown_abs": drawdown.drawdown_abs,
        #     "max_drawdown_start": format_date(drawdown.high_date),
        #     "max_drawdown_start_timestamp": dt_ts_def(drawdown.high_date),
        #     "max_drawdown_end": format_date(drawdown.low_date),
        #     "max_drawdown_end_timestamp": dt_ts_def(drawdown.low_date),
        #     "drawdown_high": drawdown.high_value,
        #     "drawdown_low": drawdown.low_value,
        #     "trading_volume": trading_volume,
        #     "bot_start_timestamp": dt_ts_def(bot_start, 0),
        #     "bot_start_date": format_date(bot_start),
        # }

    def _rpc_stopentry(self) -> dict[str, str]:
        """
        Handler to stop buying, but handle open trades gracefully.
        """
        if self._engine.state == State.RUNNING:
            # Set 'max_open_trades' to 0
            self._config["max_open_trades"] = 0
            self._engine.strategy.max_open_trades = 0

        return {"status": "No more entries will occur from now. Run /reload_config to reset."}

    def _rpc_reload_trade_from_exchange(self, trade_id: int) -> dict[str, str]:
        """
        Handler for reload_trade_from_exchange.
        Reloads a trade from it's orders, should manual interaction have happened.
        """
        trade = TradeData.get_trades(trade_filter=[TradeData.id == trade_id]).first()
        if not trade:
            raise RPCException(f"Could not find trade with id {trade_id}.")

        self._engine.handle_onexchange_order(trade)
        return {"status": "Reloaded from orders from exchange"}

    def _rpc_force_exit(
        self,
        symbol: str,
        price: float | None = None,
        # ordertype: str | None = None,
        # *,
        # amount: float | None = None,
        # quantity: float | None = None,
        # cost: float | None = None
    ) -> dict[str, str]:
        """
        Handler for forceexit.
        Sells the given trade at current price
        """
        # def __exec_force_exit(
        #     self, trade: TradeData, ordertype: str | None, amount: float | None = None
        # ) -> bool:
        #     # Check if there is there are open orders
        #     trade_entry_cancelation_registry = []
        #     for oo in trade.open_orders:
        #         trade_entry_cancelation_res = {"order_id": oo.order_id, "cancel_state": False}
        #         order = self._engine.exchange.fetch_order(oo.order_id, trade.pair)

        #         if order["side"] == trade.entry_side:
        #             fully_canceled = self._engine.handle_cancel_enter(
        #                 trade, order, oo, CANCEL_REASON["FORCE_EXIT"]
        #             )
        #             trade_entry_cancelation_res["cancel_state"] = fully_canceled
        #             trade_entry_cancelation_registry.append(trade_entry_cancelation_res)

        #         if order["side"] == trade.exit_side:
        #             # Cancel order - so it is placed anew with a fresh price.
        #             self._engine.handle_cancel_exit(trade, order, oo, CANCEL_REASON["FORCE_EXIT"])

        #     if all(tocr["cancel_state"] is False for tocr in trade_entry_cancelation_registry):
        #         if trade.has_open_orders:
        #             # Order cancellation failed, so we can't exit.
        #             return False
        #         # Get current rate and execute sell
        #         current_rate = self._engine.exchange.get_rate(
        #             trade.pair, side="exit", is_short=trade.is_short, refresh=True
        #         )
        #         exit_check = ExitCheckTuple(exit_type=ExitType.FORCE_EXIT)
        #         order_type = ordertype or self._engine.strategy.order_types.get(
        #             "force_exit", self._engine.strategy.order_types["exit"]
        #         )
        #         sub_amount: float | None = None
        #         if amount and amount < trade.amount:
        #             # Partial exit ...
        #             min_exit_stake = self._engine.exchange.get_min_pair_stake_amount(
        #                 trade.pair, current_rate, trade.stop_loss_pct
        #             )
        #             remaining = (trade.amount - amount) * current_rate
        #             if remaining < min_exit_stake:
        #                 raise RPCException(f"Remaining amount of {remaining} would be too small.")
        #             sub_amount = amount

        #         self._engine.execute_trade_exit(
        #             trade, current_rate, exit_check, ordertype=order_type, sub_trade_amt=sub_amount
        #         )

        #         return True
        #     return False

        # if self._engine.state != State.RUNNING:
        #     raise RPCException("bot is not running")

        # with self._engine._exit_lock:
        #     if trade_id == "all":
        #         # Execute exit for all open orders
        #         for trade in TradeData.get_open_trades():
        #             self.__exec_force_exit(trade, ordertype)
        #         TradeData.commit()
        #         self._engine.wallets.update()
        #         return {"result": "Created exit orders for all open trades."}

        #     # Query for trade
        #     trade = TradeData.get_trades(
        #         trade_filter=[
        #             TradeData.id == trade_id,
        #             TradeData.is_open.is_(True),
        #         ]
        #     ).first()
        #     if not trade:
        #         logger.warning("force_exit: Invalid argument received")
        #         raise RPCException("invalid argument")

        #     result = self.__exec_force_exit(trade, ordertype, amount)
        #     TradeData.commit()
        #     self._engine.wallets.update()
        #     if not result:
        #         raise RPCException("Failed to exit trade.")
        #     return {"result": f"Created exit order for trade {trade_id}."}
        pos = self._engine.ctx.get_position(symbol)
        if not pos:
            raise RPCException(f"Position for {symbol} not found.")
        if pos.is_open:
            raise RPCException(f"Position for {symbol} is already open.")

        # last_price = self.__get_last_price(position=pos)
        # if amount:
        #     quantity = amount / last_price / pos.leverage
        # elif cost:
        #     quantity = cost / last_price

        # if not quantity:
        #     raise RPCException("Quantity is not correct.")

        o = self._engine.ctx.strategy.close(
            symbol=pos.symbol,
            exchange=pos.exchange,
            position_side=pos.position_side,
            price=price,
            tag="force_exit",
        )
        print(f"[Force Exit] pos: {pos} is closed. \n\t\tRef Order: {o}")
        return o

    def _rpc_force_entry(
        self,
        symbol: str,
        side: Side,
        price: float | None,
        *,
        amount: float | None = None,
        quantity: float | None = None,
        cost: float | None = None,
        enter_reason: str | None = "force_entry",
        leverage: float | None = None,
    ) -> TradeData | None:
        """
        Handler for forcebuy <asset> <price>
        Buys a pair trade at the given or current price
        """
        # def _force_entry_validations(self, pair: str, order_side: Side):
        #     if not self._config.get("force_entry_enable", False):
        #         raise RPCException("Force_entry not enabled.")

        #     if self._engine.state != State.RUNNING:
        #         raise RPCException("bot is not running")

        #     if order_side == Side.SHORT and self._engine.trading_mode == TradingMode.SPOT:
        #         raise RPCException("Can't go short on Spot markets.")

        #     if pair not in self._engine.exchange.get_markets(tradable_only=True):
        #         raise RPCException("Symbol does not exist or market is not active.")
        #     # Check if pair quote currency equals to the stake currency.
        #     base_currency = self._config.get("base_currency")
        #     if not self._engine.exchange.get_pair_quote_currency(pair) == base_currency:
        #         raise RPCException(
        #             f"Wrong pair selected. Only pairs with stake-currency {base_currency} allowed."
        #         )
        # self._force_entry_validations(pair, order_side)

        # # check if valid pair

        # # check if pair already has an open pair
        # trade: TradeData | None = TradeData.get_trades(
        #     [TradeData.is_open.is_(True), TradeData.pair == pair]
        # ).first()
        # is_short = order_side == Side.SHORT
        # if trade:
        #     is_short = trade.is_short
        #     if not self._engine.strategy.position_adjustment_enable:
        #         raise RPCException(f"position for {pair} already open - id: {trade.id}")
        #     if trade.has_open_orders:
        #         raise RPCException(
        #             f"position for {pair} already open - id: {trade.id} "
        #             f"and has open order {','.join(trade.open_orders_ids)}"
        #         )
        # else:
        #     if TradeData.get_open_trade_count() >= self._config["max_open_trades"]:
        #         raise RPCException("Maximum number of trades is reached.")

        # if not stake_amount:
        #     # gen stake amount
        #     stake_amount = self._engine.wallets.get_trade_stake_amount(
        #         pair, self._config["max_open_trades"]
        #     )

        # # execute buy
        # if not order_type:
        #     order_type = self._engine.strategy.order_types.get(
        #         "force_entry", self._engine.strategy.order_types["entry"]
        #     )
        # with self._engine._exit_lock:
        #     if self._engine.execute_entry(
        #         pair,
        #         stake_amount,
        #         price,
        #         ordertype=order_type,
        #         trade=trade,
        #         is_short=is_short,
        #         enter_reason=enter_reason,
        #         leverage_=leverage,
        #         mode="pos_adjust" if trade else "initial",
        #     ):
        #         TradeData.commit()
        #         trade = TradeData.get_trades([TradeData.is_open.is_(True), TradeData.pair == pair]).first()
        #         return trade
        #     else:
        #         raise RPCException(f"Failed to enter position for {pair}.")
        pos = self._engine.ctx.get_position(symbol)
        if not pos:
            raise RPCException(f"Position for {symbol} not found.")
        if pos.is_open:
            raise RPCException(f"Position for {symbol} is already open.")

        last_price = self.__get_last_price(position=pos)
        if amount:
            quantity = amount / last_price / pos.leverage
        elif cost:
            quantity = cost / last_price

        if not quantity:
            raise RPCException("Quantity is not correct.")

        o = self._engine.ctx.strategy.order(
            symbol=pos.symbol,
            exchange=pos.exchange,
            position_side=pos.position_side,
            side=side,
            price=price,
            quantity=quantity,
            tag=enter_reason,
        )
        print(f"[Force Entry] pos: {pos} is closed. \n\t\tRef Order: {o}")
        return o

    def _rpc_cancel_open_order(self, order_id: str, symbol: str):
        if self._engine.state != State.RUNNING:
            raise RPCException("bot is not running")
        # with self._engine._exit_lock:
        #     # Query for trade
        #     trade = TradeData.get_trades(
        #         trade_filter=[
        #             TradeData.id == trade_id,
        #             TradeData.is_open.is_(True),
        #         ]
        #     ).first()
        #     if not trade:
        #         logger.warning("cancel_open_order: Invalid trade_id received.")
        #         raise RPCException("Invalid trade_id.")
        #     if not trade.has_open_orders:
        #         logger.warning("cancel_open_order: No open order for trade_id.")
        #         raise RPCException("No open order for trade_id.")

        #     for open_order in trade.open_orders:
        #         try:
        #             order = self._engine.exchange.fetch_order(open_order.order_id, trade.pair)
        #         except ExchangeError as e:
        #             logger.info(f"Cannot query order for {trade} due to {e}.", exc_info=True)
        #             raise RPCException("Order not found.")
        #         self._engine.handle_cancel_order(
        #             order, open_order, trade, CANCEL_REASON["USER_CANCEL"]
        #         )
        #     TradeData.commit()
        pos = self._engine.ctx.get_position(symbol)
        if not pos:
            raise RPCException(f"Position for {symbol} not found.")
        if not pos.is_open:
            raise RPCException(f"Position for {symbol} is not open.")

        o = self._engine.ctx.strategy.cancel_order(symbol=pos.symbol, exchange=pos.exchange, order_id=order_id)
        print(f"[Cancel Open Order] order_id: {order_id} is canceled. \n\t\tRef Order: {o}")
        return o

    def _rpc_enter_reason_performance(self, symbol: str | None) -> list[dict[str, Any]]:
        """
        Handler for buy tag performance.
        Shows a performance statistic from finished trades
        """
        if self._engine.state != State.RUNNING:
            raise RPCException("bot is not running")

        if symbol:
            positions = [self._engine.ctx.get_position(symbol)]
        else:
            positions = self._engine.ctx.get_positions()

        if not positions:
            raise RPCException("no active positions")
        else:
            enter_reason_map = {}

            for position in positions:
                for tradepnl in position.tradepnls:
                    enter_reason = tradepnl.enter_reason if tradepnl.enter_reason else "Other"
                    if enter_reason not in enter_reason_map:
                        enter_reason_map[enter_reason] = {
                            "enter_reason": enter_reason,
                            "close_roi": 0,
                            "close_pnl": 0,
                            "count": 0,
                        }
                    enter_reason_map[enter_reason]["close_roi"] += tradepnl.close_roi or 0
                    enter_reason_map[enter_reason]["close_pnl"] += tradepnl.close_pnl or 0
                    enter_reason_map[enter_reason]["count"] += 1

        return list(enter_reason_map.values())

    def _rpc_exit_reason_performance(self, symbol: str | None) -> list[dict[str, Any]]:
        """
        Handler for exit reason performance.
        Shows a performance statistic from finished trades
        """
        if self._engine.state != State.RUNNING:
            raise RPCException("bot is not running")
        if symbol:
            positions = [self._engine.ctx.get_position(symbol)]
        else:
            positions = self._engine.ctx.get_positions()

        if not positions:
            raise RPCException("no active positions")
        else:
            exit_reason_map = {}

            for position in positions:
                for tradepnl in position.tradepnls:
                    exit_reason = tradepnl.exit_reason if tradepnl.exit_reason else "Other"
                    if exit_reason not in exit_reason_map:
                        exit_reason_map[exit_reason] = {
                            "exit_reason": exit_reason,
                            "close_roi": 0,
                            "close_pnl": 0,
                            "count": 0,
                        }
                    exit_reason_map[exit_reason]["close_roi"] += tradepnl.close_roi or 0
                    exit_reason_map[exit_reason]["close_pnl"] += tradepnl.close_pnl or 0
                    exit_reason_map[exit_reason]["count"] += 1
        return list(exit_reason_map.values())

    def _rpc_mix_tag_performance(self, symbol: str | None) -> list[dict[str, Any]]:
        """
        Handler for mix tag (enter_reason + exit_reason) performance.
        Shows a performance statistic from finished trades
        """
        if self._engine.state != State.RUNNING:
            raise RPCException("bot is not running")
        if symbol:
            positions = [self._engine.ctx.get_position(symbol)]
        else:
            positions = self._engine.ctx.get_positions()

        if not positions:
            raise RPCException("no active positions")
        else:
            mix_tag_map = {}

            for position in positions:
                for tradepnl in position.tradepnls:
                    enter_reason = tradepnl.enter_reason if tradepnl.enter_reason else "Other"
                    exit_reason = tradepnl.exit_reason if tradepnl.exit_reason else "Other"
                    mix_tag = enter_reason + " | " + exit_reason

                    if mix_tag not in mix_tag_map:
                        mix_tag_map[mix_tag] = {
                            "mix_tag": mix_tag,
                            "close_roi": 0,
                            "close_pnl": 0,
                            "count": 0,
                        }
                    mix_tag_map[mix_tag]["close_roi"] += tradepnl.close_roi or 0
                    mix_tag_map[mix_tag]["close_pnl"] += tradepnl.close_pnl or 0
                    mix_tag_map[mix_tag]["count"] += 1
        return list(mix_tag_map.values())

    def _rpc_status_table(self, base_currency: str, fiat_display_currency: str) -> tuple[list, list, float]:
        trades_list = []
        columns = []
        fiat_profit_sum = 0.0

        # trades = [
        #     trade
        #     for order_id, trade in self._engine.broker.trade_records.items()
        #     if trade.is_open
        # ]
        # nonspot = self._config.get("trading_mode", TradingMode.SPOT) != TradingMode.SPOT
        # if not trades:
        #     raise RPCException("no active trade")
        # else:
        #     trades_list = []
        #     fiat_profit_sum = nan
        #     for trade in trades:
        #         # calculate profit and send message to user
        #         try:
        #             current_rate = self._engine.exchange.get_rate(
        #                 trade.pair, side="exit", is_short=trade.is_short, refresh=False
        #             )
        #         except (PricingError, ExchangeError):
        #             current_rate = nan
        #             trade_profit = nan
        #             profit_str = f"{nan:.2%}"
        #         else:
        #             if trade.nr_of_successful_entries > 0:
        #                 profit = trade.calculate_profit(current_rate)
        #                 trade_profit = profit.profit_abs
        #                 profit_str = f"{profit.profit_ratio:.2%}"
        #             else:
        #                 trade_profit = 0.0
        #                 profit_str = f"{0.0:.2f}"
        #         leverage = f"{trade.leverage:.3g}"
        #         direction_str = (
        #             (f"S {leverage}x" if trade.is_short else f"L {leverage}x")
        #             if nonspot
        #             else ""
        #         )
        #         if self._fiat_converter:
        #             fiat_profit = self._fiat_converter.convert_amount(
        #                 trade_profit, base_currency, fiat_display_currency
        #             )
        #             if not isnan(fiat_profit):
        #                 profit_str += f" ({fiat_profit:.2f})"
        #                 fiat_profit_sum = (
        #                     fiat_profit
        #                     if isnan(fiat_profit_sum)
        #                     else fiat_profit_sum + fiat_profit
        #                 )
        #         else:
        #             profit_str += f" ({trade_profit:.2f})"
        #             fiat_profit_sum = (
        #                 trade_profit
        #                 if isnan(fiat_profit_sum)
        #                 else fiat_profit_sum + trade_profit
        #             )

        #         active_attempt_side_symbols = [
        #             "*" if (oo and oo.ft_order_side == trade.entry_side) else "**"
        #             for oo in trade.open_orders
        #         ]

        #         # example: '*.**.**' trying to enter, exit and exit with 3 different orders
        #         active_attempt_side_symbols_str = ".".join(active_attempt_side_symbols)

        #         detail_trade = [
        #             f"{trade.id} {direction_str}",
        #             trade.pair + active_attempt_side_symbols_str,
        #             shorten_date(dt_humanize_delta(trade.open_date_utc)),
        #             profit_str,
        #         ]

        #         if self._config.get("position_adjustment_enable", False):
        #             max_entry_str = ""
        #             if self._config.get("max_entry_position_adjustment", -1) > 0:
        #                 max_entry_str = (
        #                     f"/{self._config['max_entry_position_adjustment'] + 1}"
        #                 )
        #             filled_entries = trade.nr_of_successful_entries
        #             detail_trade.append(f"{filled_entries}{max_entry_str}")
        #         trades_list.append(detail_trade)
        #     profitcol = "Profit"
        #     if self._fiat_converter:
        #         profitcol += " (" + fiat_display_currency + ")"
        #     else:
        #         profitcol += " (" + base_currency + ")"

        #     columns = ["ID L/S" if nonspot else "ID", "Pair", "Since", profitcol]
        #     if self._config.get("position_adjustment_enable", False):
        #         columns.append("# Entries")
        return trades_list, columns, fiat_profit_sum

    def _rpc_reload_trade_from_exchange(self, trade_id: int) -> dict[str, str]:
        """
        Handler for reload_trade_from_exchange.
        Reloads a trade from it's orders, should manual interaction have happened.
        """
        # trade = Trade.get_trades(trade_filter=[Trade.id == trade_id]).first()
        # if not trade:
        #     raise RPCException(f"Could not find trade with id {trade_id}.")

        # self._engine.handle_onexchange_order(trade)
        return {"status": "Reloaded from orders from exchange"}
