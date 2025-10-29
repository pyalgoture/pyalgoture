from collections import OrderedDict, defaultdict
from datetime import datetime
from typing import Any

from ..broker import Broker
from ..utils.objects import (
    AccountData,
    AssetType,
    BarData,
    Exchange,
    ExecutionType,
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
from ..utils.util_math import prec_round


class BacktestBroker(Broker):
    """Broker implementation for backtesting purposes."""

    def __init__(
        self,
        base_currency: str = "USDT",
        balance: float = 10000,
        commission_rate: float = 0.0,
        slippage: float = 0.0,
        trade_at: str = "",
        short_cash: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the backtest broker.

        Args:
            base_currency: Base currency for the account
            balance: Initial balance for the account
            commission_rate: Commission rate for trades
            slippage: Slippage rate for orders
            trade_at: Price type to trade at ('open', 'close', 'bid', 'ask')
            short_cash: Whether short selling is allowed
            **kwargs: Additional keyword arguments

        Raises:
            ValueError: If balance is <= 0
            TypeError: If balance is not a number or trade_at is invalid
        """
        super().__init__(base_currency=base_currency)

        if balance <= 0:
            raise ValueError("Balance cannot be less than or equal to 0")
        if isinstance(balance, str):
            raise TypeError("`balance` must be a number")
        if trade_at and trade_at not in {"open", "close", "bid", "ask"}:
            raise TypeError("`trade_at` must be one of open, close, bid, ask")

        self.balance = balance
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.trade_at = trade_at
        self.short_cash = short_cash

        self.default_account = AccountData(
            asset=base_currency,
            balance=balance,
            exchange=Exchange.LOCAL,
        )
        self.accounts[self.default_account.aio_account_id] = self.default_account

    def initialize(self) -> None:
        """Initialize the broker and set up context."""
        super().initialize()
        try:
            self.ctx.broker.default_account = self.default_account
        except Exception as e:
            tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
            tb_filename = e.__traceback__.tb_frame.f_code.co_filename if e.__traceback__ else "unknown"
            self.execption(
                f"Something went wrong in initialize - Error: {e} in line {tb_lineno} for file {tb_filename}"
            )

    def reset(self) -> None:
        """Reset broker state for new backtest run."""
        self.order_count = 0
        self.default_account = AccountData(asset=self.base_currency, exchange=Exchange.LOCAL, balance=self.balance)

        self.open_order_records: dict[str, OrderData] = OrderedDict()
        self.order_records: dict[str, OrderData] = OrderedDict()
        self.trade_records: dict[str, TradeData] = OrderedDict()

        self.positions: dict[str, PositionData] = defaultdict(PositionData)
        self.accounts: dict[str, TradeData] = defaultdict(AccountData)
        self.accounts[self.default_account.aio_account_id] = self.default_account

    def on_tick(self, tick_data: TickData) -> None:
        """Process tick data and handle open orders.

        Args:
            tick_data: Market tick data
        """
        try:
            self.trace(
                f">>>>>> [on backtest_broker] open_order_records: {list(self.open_order_records.keys())}({len(self.open_order_records)})"
            )
            for order_id, order in self.open_order_records.copy().items():
                if order.aio_symbol == tick_data.aio_symbol:
                    active_order = self._process_order(order=order, data=tick_data)
                    if active_order:
                        self.open_order_records[order_id] = active_order

        except Exception as e:
            tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
            tb_filename = e.__traceback__.tb_frame.f_code.co_filename if e.__traceback__ else "unknown"
            self.execption(
                f"Something went wrong in broker on_tick - Error: {str(e)} in line {tb_lineno} for file {tb_filename}."
            )

    def on_bar(self, tick: datetime) -> None:
        """Process bar data and handle open orders and positions.

        Args:
            tick: Market bar data
        """
        if not self.ctx.is_live or hasattr(self.ctx, "backtest_mode"):
            if self.ctx.backtest_mode == "tick":
                return

            self.trace(
                f"[backtest_broker - on_bar] >> open_order_records: {list(self.open_order_records.keys())} | len: {len(self.open_order_records)}"
            )

            for order_id, order in self.open_order_records.copy().items():
                if order_id not in self.open_order_records:
                    continue

                bar_data = self.ctx.get_bar_data(symbol=order.symbol, exchange=order.exchange)

                self.trace(
                    f"on_bar -> working on order_id: {order_id} | aio_symbol: {order.aio_symbol} | bar_data: {bar_data}"
                )

                if bar_data is None:
                    self.error(
                        f"bar_data is None - order: {order} is being put into next run - aio_symbol: {order.aio_symbol}"
                    )
                    continue

                active_order = self._process_order(order=order, data=bar_data)

            # Check liquidation for positions
            if self.ctx.trade_history and not self.ctx.is_live:
                return

            for position in self.get_positions():
                if position.size:
                    bar_data = self.ctx.get_bar_data(symbol=position.symbol, exchange=position.exchange)
                    if bar_data is None:
                        continue
                    self._check_position_liquidation(position, data=bar_data)
                    position.update_by_marketdata(
                        last_price=bar_data["close"],
                        available_balance=self.ctx.strategy.available_balance,
                    )

    def cancel_order(self, order: OrderData) -> OrderData:
        """Cancel an existing order.

        Args:
            order: Order to cancel

        Returns:
            Updated order with cancellation status
        """
        if order.order_id in self.open_order_records:
            msg = f"Order {order['order_id']} has been canceled."
            order["status"] = Status.CANCELED
        else:
            msg = f"Cancel order {order['order_id']} failed. It is not existing in your order pending list."
            order["status"] = Status.FAILED

        order["msg"] += msg
        order["updated_at"] = self.ctx.now if self.ctx.is_live else self.ctx.tick

        self.on_order_callback(order)
        return order

    def send_order(self, order: OrderData, **kwargs: Any) -> OrderData:
        """Send an order for execution.

        Args:
            order: Order to send
            **kwargs: Additional keyword arguments

        Returns:
            Updated order with execution status
        """
        datafeed = self.ctx.get_datafeed(symbol=order.symbol, exchange=order.exchange)
        asset_type = datafeed.asset_type
        code = datafeed.code
        position = self.ctx.get_position(
            symbol=order.symbol, exchange=order.exchange, position_side=order.position_side, force_return=True
        )
        old_quantity = position.size if position else 0.0
        bar_data = self.ctx.get_bar_data(symbol=order.symbol, exchange=order.exchange)

        # Handle trailing orders
        order._limit_offset = 0.0
        if order.trail_percent or order.trail_price:
            if order.price:
                order._limit_offset = bar_data.close - order.price
            order = self._adjust_trailing_order(price=bar_data.close, order=order)

        # Determine order type and price
        if not order.price:
            if order.stop_price:
                order_type = OrderType.STOP
                order.price = order.stop_price
            else:
                order_type = OrderType.MARKET
                trade_at = self.trade_at if hasattr(self, "trade_at") and self.trade_at else None
                order.price = bar_data[trade_at if trade_at else "close"]
        else:
            order_type = OrderType.STOPLIMIT if order.stop_price else OrderType.LIMIT

        order.code = code
        order.order_type = order_type
        order.asset_type = asset_type
        order.created_at = self.ctx.now if self.ctx.is_live else self.ctx.tick

        # Adjust side for short selling
        if order.side == Side.SELL and old_quantity <= 0 and not position.is_contract:
            order.side = Side.SHORTSELL

        quantity_with_direction = order.quantity if order.side in {Side.BUY, Side.BUYCOVER} else (-1) * order.quantity
        order.is_open = (old_quantity * quantity_with_direction) >= 0

        order.status = Status.NEW
        self.sent_orders[order.order_id] = order

        # Pre-check order validity
        is_executable, order = self._validate_order(order, position)
        if not is_executable:
            order["status"] = Status.FAILED
            order["updated_at"] = self.ctx.now if self.ctx.is_live else self.ctx.tick
            self.on_order_callback(order)
            return order

        self.on_order_callback(order)  # status => NEW

        if order.price:
            order.price = prec_round(order.price, 6)
        order.commission = 0.0
        order.commission_asset = self.ctx.default_account.asset

        order.status = Status.FILLING
        self.on_order_callback(order)  # status => FILLING

        if kwargs.get("fill_now"):
            from copy import deepcopy

            fake_next_bar = deepcopy(bar_data)
            fake_next_bar.open = fake_next_bar.close
            self.debug(f"fill_now -> order: {order}; fake_next_bar: {fake_next_bar}")
            self._process_order(order=order, data=fake_next_bar)

        return order

    def add_balance(self, amount: float) -> None:
        """Add or deduct balance from the default account.

        Args:
            amount: Amount to add (positive) or deduct (negative)
        """
        self.default_account.add_balance(dt=self.ctx.tick, amount=amount)

        if amount > 0:
            self.warning(f"The balance is added $ {amount}")
        else:
            self.warning(f"The balance is deducted $ {amount}")

    def update_trade(self, trade: TradeData) -> None:
        """Update trade and account information.

        Args:
            trade: Trade data to update
        """
        super().update_trade(trade)

        position = self.ctx.get_position(
            symbol=trade.symbol, exchange=trade.exchange, position_side=trade.position_side, force_return=True
        )

        # Update account based on trade
        trade_value = position.get_trade_value(price=trade.price, quantity=trade.quantity)
        pnl = position.trade_pnl
        is_contract = position.is_contract
        commission = trade.commission

        amount = trade_value
        if is_contract:
            # Contract trades only affect balance through commission & pnl
            if trade.is_open:
                amount = commission * -1  # Deduct commission
            else:
                amount = pnl - commission  # Add PnL minus commission
        else:
            if trade.side in {Side.BUY, Side.BUYCOVER}:
                amount = (amount + commission) * -1  # Deduct trade value + commission
            else:
                amount = (amount - commission) * 1  # Add trade value minus commission

        self.default_account.available += amount
        self.default_account.balance += amount

    def _check_position_liquidation(self, position: PositionData, data: BarData) -> None:
        """Check if a position should be liquidated.

        Args:
            position: Position to check
            data: Market data for liquidation check
        """
        liquidation_price = position.liquidation_price
        is_live = False if hasattr(self.ctx, "backtest_mode") else self.ctx.is_live
        backtest_mode = self.ctx.backtest_mode if hasattr(self.ctx, "backtest_mode") else "tick" if is_live else "bar"
        position_side = position.side

        if liquidation_price <= 0:
            return

        if is_live:
            long_cross_price = short_cross_price = data.last_price
        elif backtest_mode == "bar":
            long_cross_price = data.low
            short_cross_price = data.high
        else:
            long_cross_price = short_cross_price = data.close

        is_liquidation_hit = (
            (long_cross_price <= liquidation_price)
            if position_side == PositionSide.LONG
            else (short_cross_price >= liquidation_price)
        )

        self.trace(
            f"[_check_position_liquidation] is_liquidation_hit: {is_liquidation_hit}; "
            f"position_side: {position_side}; short_cross_price: {short_cross_price}; "
            f"long_cross_price: {long_cross_price} | liquidation_price: {liquidation_price}"
        )

        if is_liquidation_hit:
            # Generate liquidation order
            order_id = self.ctx.broker._new_order_id(prefix="liq")
            liquidation_order = OrderData(
                order_id=order_id,
                ori_order_id=order_id,
                symbol=position.symbol,
                code=position.code,
                asset_type=position.asset_type,
                exchange=position.exchange,
                price=liquidation_price,
                quantity=abs(position.size),
                executed_quantity=0.0,
                side=Side.SELL if position_side == PositionSide.LONG else Side.BUY,
                position_side=position.position_side,
                created_at=self.ctx.tick,
                updated_at=self.ctx.tick,
                tag="liquidation",
                msg="Liquidation hit.",
            )
            self.warning(
                f"===> liquidation_hit - position_side: {position_side}; "
                f"avg_open_price: {position.avg_open_price}; liquidation_price: {liquidation_price}; "
                f"long_cross_price: {long_cross_price}; short_cross_price: {short_cross_price}. "
                f"liquidation_position: {position}; liquidation_order: {liquidation_order}"
            )
            self.send_order(order=liquidation_order)

    def _check_position_tpsl(self, position: PositionData, data: BarData) -> None:
        """Check take profit and stop loss for a position.

        Args:
            position: Position to check
            data: Market data for TP/SL check
        """
        take_profit = position.take_profit
        stop_loss = position.stop_loss
        is_live = self.ctx.is_live
        backtest_mode = self.ctx.backtest_mode if hasattr(self.ctx, "backtest_mode") else "tick" if is_live else "bar"
        position_side = position.position_side

        if not (take_profit or stop_loss):
            return

        # Set cross prices based on mode
        if is_live:
            long_tp_cross_price = long_sl_cross_price = data.last_price
            short_tp_cross_price = short_sl_cross_price = data.last_price
        elif backtest_mode == "bar":
            long_tp_cross_price = data.low
            long_sl_cross_price = data.high
            short_tp_cross_price = data.high
            short_sl_cross_price = data.low
        else:
            long_tp_cross_price = long_sl_cross_price = data.close
            short_tp_cross_price = short_sl_cross_price = data.close

        # Check take profit
        if take_profit:
            is_tp_hit = (
                (long_tp_cross_price <= take_profit)
                if position_side == PositionSide.LONG
                else (long_sl_cross_price >= take_profit)
            )

            if is_tp_hit:
                tp_order = OrderData(
                    order_id=self.ctx.broker._new_order_id(),
                    symbol=position.symbol,
                    quantity=abs(position.size),
                    side=Side.SELL if position_side == PositionSide.LONG else Side.BUY,
                    position_side=position.position_side,
                    exchange=position.exchange,
                )
                self.debug(
                    f"===> tp_hit - position_side: {position_side}; "
                    f"avg_open_price: {position.avg_open_price}; take_profit: {take_profit}; "
                    f"long_tp_cross_price: {long_tp_cross_price}; short_tp_cross_price: {short_tp_cross_price}. "
                    f"tp_position: {position}; tp_order: {tp_order}"
                )
                self.send_order(order=tp_order)

        # Check stop loss
        if stop_loss:
            is_sl_hit = (
                (long_sl_cross_price >= stop_loss)
                if position_side == PositionSide.LONG
                else (short_sl_cross_price <= stop_loss)
            )

            if is_sl_hit:
                sl_order = OrderData(
                    order_id=self.ctx.broker._new_order_id(),
                    symbol=position.symbol,
                    quantity=abs(position.size),
                    side=Side.SELL if position_side == PositionSide.LONG else Side.BUY,
                    position_side=position.position_side,
                    exchange=position.exchange,
                )
                self.debug(
                    f"===> sl_hit - position_side: {position_side}; "
                    f"avg_open_price: {position.avg_open_price}; stop_loss: {stop_loss}; "
                    f"long_sl_cross_price: {long_sl_cross_price}; short_sl_cross_price: {short_sl_cross_price}. "
                    f"sl_position: {position}; sl_order: {sl_order}"
                )
                self.send_order(order=sl_order)

    def _adjust_trailing_order(self, price: float, order: OrderData) -> OrderData:
        """Adjust trailing stop order price.

        Args:
            price: Current market price
            order: Order to adjust

        Returns:
            Adjusted order
        """
        if order.trail_price:
            adjustment_amount = order.trail_price
        elif order.trail_percent:
            adjustment_amount = price * order.trail_percent
        else:
            adjustment_amount = 0.0

        # Stop sell is below (-), stop buy is above, move only if needed
        if order.side in {Side.BUY, Side.BUYCOVER}:  # Close short position
            price += adjustment_amount
            if not order.stop_price:
                order.stop_price = price
            elif price < order.stop_price:
                order.stop_price = price
                order.status = Status.AMENDED
                order.msg = f"stop price has been adjusted to ${price}."
                # Adjust limit price for stop limit orders
                if order.price:
                    order.price = price - order._limit_offset
                    order.msg += f"limit price has been adjusted to ${price}."
                self.ctx.strategy.on_order_amend(order)
        else:  # Close long position
            price -= adjustment_amount
            if not order.stop_price:
                order.stop_price = price
            elif price > order.stop_price:
                order.stop_price = price
                order.status = Status.AMENDED
                order.msg = f"stop price has been adjusted to ${price}."
                if order.price:
                    order.price = price - order._limit_offset
                    order.msg += f"limit price has been adjusted to ${price}."
                self.ctx.strategy.on_order_amend(order)

        return order

    def __get_market_data(self, data: BarData) -> tuple[bool, str, float, float, float, float]:
        """Get market data based on live/backtest mode.

        Returns:
            tuple: (is_live, backtest_mode, long_cross_price, short_cross_price, long_best_price, short_best_price)
        """
        is_live = False if hasattr(self.ctx, "backtest_mode") else self.ctx.is_live
        backtest_mode = self.ctx.backtest_mode if hasattr(self.ctx, "backtest_mode") else "tick" if is_live else "bar"

        if is_live:
            long_cross_price = short_cross_price = data.last_price
            long_best_price = short_best_price = data.last_price
        elif backtest_mode == "bar":
            long_cross_price, short_cross_price = data.high, data.low
            long_best_price = short_best_price = data.open
        elif backtest_mode == "tick":
            long_cross_price = short_cross_price = data.close
            long_best_price = short_best_price = data.close
        else:  # backtest_mode == "bar_tick" or other tick modes
            long_cross_price, short_cross_price = data.ask_price_1, data.bid_price_1
            long_best_price = short_best_price = data.close

        return is_live, backtest_mode, long_cross_price, short_cross_price, long_best_price, short_best_price

    def __process_market_order(self, order: OrderData, data: BarData, is_live: bool, backtest_mode: str) -> float:
        """Process market order and return execution price."""
        price: float = 0.0
        if is_live:
            price = data.last_price
        elif backtest_mode == "tick":
            price = data.ask_price_1 if order.side == Side.BUY else data.bid_price_1
        else:
            price = data.open

        if hasattr(self, "trade_at") and self.trade_at:
            price = data[self.trade_at]

        return price

    def __process_stop_order(
        self,
        order: OrderData,
        data: BarData,
        long_cross_price: float,
        short_cross_price: float,
        long_best_price: float,
        short_best_price: float,
    ) -> tuple[OrderData | None, float | None]:
        """Process stop/stop-limit order and return updated order and price."""
        stop_price = order.stop_price
        is_stop_hit = long_cross_price >= stop_price if order.side == Side.BUY else short_cross_price <= stop_price

        if not is_stop_hit:
            # Check trail_price & trail_percent, update stop price if needed
            if order.trail_price or order.trail_percent:
                order = self._adjust_trailing_order(price=data.close, order=order)
            return order, None

        # Stop hit - convert order type
        if order.order_type == OrderType.STOPLIMIT:
            order.order_type = OrderType.LIMIT
            order.msg += "StopLimit -> Limit."
            return order, None
        else:
            # Market-if-touched / market order
            order.order_type = OrderType.MARKET
            order.msg += "Stop -> Market."
            price = max(long_best_price, stop_price) if order.side == Side.BUY else min(short_best_price, stop_price)
            return order, price

    def __process_limit_order(
        self,
        order: OrderData,
        long_cross_price: float,
        short_cross_price: float,
        long_best_price: float,
        short_best_price: float,
    ) -> tuple[OrderData, float | None]:
        """Process limit order and return updated order and price."""
        if "liq-" in order.order_id:
            # Liquidation order - process immediately
            price = min(order.price, long_best_price) if order.side == Side.BUY else max(order.price, short_best_price)
            return order, price

        is_limit_hit = long_cross_price <= order.price if order.side == Side.BUY else short_cross_price >= order.price

        # self.debug(
        #     f"limit order-{order.order_id} == {is_limit_hit}; "
        #     f"order_px:{order.price}; side:{order.side} "
        #     f"long_cross_price:{long_cross_price}; short_cross_price:{short_cross_price}; "
        #     f"long_best_price:{long_best_price}; short_best_price:{short_best_price};"
        # )

        if not is_limit_hit:
            if order.time_in_force == TimeInForce.GTC:
                return order, None  # GTC order - keep waiting
            else:  # FOK, IOC orders
                order.updated_at = self.ctx.now if self.ctx.is_live else self.ctx.tick
                order.status = Status.CANCELED
                order.msg += f"{order.time_in_force.value} order canceled."
                return order, None
        else:
            price = min(order.price, long_best_price) if order.side == Side.BUY else max(order.price, short_best_price)
            return order, price

    def __apply_slippage(self, order: OrderData, price: float) -> float:
        """Apply slippage to the execution price."""
        if not self.slippage:
            return price

        return price * (1 + self.slippage) if order.side == Side.BUY else price * (1 - self.slippage)

    def _process_order(
        self,
        order: OrderData,
        data: BarData,
        skip_checking: bool = False,
    ) -> OrderData | None:
        """
        Process orders with different execution logic based on order type and market conditions.

        Order processing rules:
        - Market orders: Execute at next bar's open (backtest) or current price (live)
        - Limit orders: Execute when price crosses limit threshold
        - Stop orders: Convert to market/limit when stop price is hit
        - Stop-limit orders: Convert to limit order when stop price is hit

        Args:
            order: Order to process
            data: Market data for current bar/tick
            skip_checking: Skip order validation if True

        Returns:
            None if order is executed/canceled, order if still pending
        """
        position = self.ctx.get_position(
            symbol=order.symbol, exchange=order.exchange, position_side=order.position_side, force_return=True
        )

        try:
            is_live, backtest_mode, long_cross_price, short_cross_price, long_best_price, short_best_price = (
                self.__get_market_data(data)
            )

            price = None

            if order.order_type:
                if order.order_type == OrderType.MARKET:
                    price = self.__process_market_order(order, data, is_live, backtest_mode)

                elif order.order_type in (OrderType.STOP, OrderType.STOPLIMIT):
                    order, price = self.__process_stop_order(
                        order, data, long_cross_price, short_cross_price, long_best_price, short_best_price
                    )
                    if order and price is None:  # Order not hit or converted to limit
                        return order

                elif order.order_type == OrderType.LIMIT:
                    order, price = self.__process_limit_order(
                        order, long_cross_price, short_cross_price, long_best_price, short_best_price
                    )
                    if price is None:  # Order not hit or canceled
                        return None if order.status == Status.CANCELED else order

                # Apply slippage if price is determined
                if price is not None:
                    price = self.__apply_slippage(order, price)
                    order.price = price

            # Validate and execute order
            if skip_checking:
                is_executable = True
            else:
                if order.status == Status.CANCELED:
                    is_executable = False
                else:
                    is_executable, order = self._validate_order(order, position)

            # Update order status and timestamp
            order.status = (
                Status.FILLED if is_executable else (order.status if order.status == Status.CANCELED else Status.FAILED)
            )
            order.updated_at = self.ctx.now if self.ctx.is_live else self.ctx.tick

            self.trace(f"[_process_order]>>>>>>> is_executable:{is_executable} || order:{order}")

            # Trigger callbacks
            self.on_order_callback(order)

            if order.status == Status.FILLED:
                trade = TradeData(
                    ori_order_id=order.order_id,
                    order_id=order.order_id,
                    quantity=order.quantity,
                    side=order.side,
                    price=order.price,
                    code=order.code,
                    symbol=order.symbol,
                    exchange=order.exchange,
                    asset_type=order.asset_type,
                    position_side=order.position_side,
                    commission=order.commission,
                    commission_asset=order.commission_asset,
                    traded_at=order.updated_at,
                    tag=order.tag,
                    msg=order.msg,
                    is_open=order.is_open,
                    execution_type=ExecutionType.LIQUIDATION if "liq-" in order.order_id else ExecutionType.TRADE,
                )

                self.on_trade_callback(trade)

            return None

        except Exception as e:
            tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
            tb_filename = e.__traceback__.tb_frame.f_code.co_filename if e.__traceback__ else "unknown"
            self.execption(
                f"Something went wrong _process_order - Error: {str(e)} in line {tb_lineno} for file {tb_filename}."
            )
            return None

    def _validate_order(self, order: OrderData, position: PositionData) -> tuple[bool, OrderData]:
        """Validate order before execution.

        Args:
            order: Order to validate
            position: Current position

        Returns:
            Tuple of (is_executable, updated_order)
        """
        old_quantity = position.size if position else 0.0
        asset_type = order.asset_type
        side = order.side
        quantity = order.quantity
        quantity_with_direction = quantity if side in {Side.BUY, Side.BUYCOVER} else (-1) * quantity
        order.is_open = is_open = (old_quantity * quantity_with_direction) >= 0

        datafeed = self.ctx.get_datafeed(symbol=order.symbol, exchange=order.exchange)
        step_size = datafeed.step_size if datafeed else 0.0001
        step_size = step_size if step_size else 0.0

        order.commission = position.cal_commission(order.price, quantity)

        available_balance = self.ctx.strategy.available_balance

        self.trace(
            f"[_validate_order] - order_id: {order.order_id}; available_balance: {available_balance}; "
            f"balance: {self.ctx.default_account.balance} - frozen: {self.ctx.default_account.frozen} "
            f"- account available: {self.ctx.default_account.available}; "
            f"- order type: {order.order_type} || account: {self.ctx.default_account}"
        )

        # Basic validation checks
        if order.status in {Status.CANCELED, Status.CANCELING}:
            return False, order

        if not quantity or quantity != quantity:  # Check for 0 or NaN
            order.msg += "The order quantity cannot be 0 or invalid."
            return False, order

        if asset_type == AssetType.SPOT and quantity < step_size:
            order.msg += f"The order quantity({quantity}) must be larger than minimum volume {step_size}."
            return False, order
        elif asset_type == AssetType.EQUITY and quantity % step_size != 0:
            order.msg += f"The order quantity({quantity}) must be multiples of lot size {step_size}."
            return False, order

        if side not in {Side.BUY, Side.SELL, Side.SHORTSELL}:
            order.msg += "The order side must be one of [Side.BUY, Side.SELL, Side.SHORTSELL]."
            return False, order

        # Position-specific validation
        if position.is_contract:
            if is_open:
                return self.__check_available_balance(order, old_quantity, position, available_balance, step_size)
            else:
                return self.__check_available_quantity(order, old_quantity, position, available_balance, step_size)
        else:
            if side in {Side.BUY, Side.BUYCOVER}:
                return self.__check_available_balance(order, old_quantity, position, available_balance, step_size)
            elif side == Side.SHORTSELL:
                if not self.short_cash:
                    order.msg += "The broker is not allowed to perform short selling."
                    return False, order
                else:
                    if old_quantity > 0:
                        order.msg += "Please close your long position before performing short selling."
                        return False, order
            elif side == Side.SELL:
                return self.__check_available_quantity(order, old_quantity, position, available_balance, step_size)

        return True, order

    def __check_available_balance(
        self, order: OrderData, old_quantity: float, position: PositionData, available_balance: float, step_size: float
    ) -> tuple[bool, OrderData]:
        """Check if sufficient balance is available for the order.

        Args:
            order: Order to check
            old_quantity: Current position quantity
            position: Position data
            available_balance: Available balance
            step_size: Minimum step size

        Returns:
            Tuple of (is_valid, updated_order)
        """
        self.trace(f"[__check_available_balance] order: {order}")
        max_order_quantity = position.cal_max_order_quantity(
            available_balance=available_balance,
            price=order.price,
            step_size=step_size,
            order_quantity=order.quantity,
            is_open=order.is_open,
        )

        self.trace(
            f"[__check_available_balance] - max_order_quantity: {max_order_quantity} "
            f"VS order.quantity: {order.quantity} | is_open: {order.is_open} | "
            f"old_quantity: {old_quantity}({position.size}) * last_price: {position.last_price} | "
            f"order.price: {order.price} | step_size: {step_size} | available_balance: {available_balance}"
        )

        if order.quantity > max_order_quantity:
            required_additional = (
                (order.quantity - max_order_quantity)
                if position.is_inverse
                else (order.quantity * position.get_margin(order.price) - available_balance)
            )
            order.msg += (
                f"You don't have enough balance(${available_balance}) to {order.side.value.lower()} "
                f"{order.symbol}({order.asset_type}) {order.quantity} at price ${order.price}. "
                f"Require more ${required_additional} OR Max order qty {max_order_quantity}. "
                f"\n[DEBUG] pos: {position}"
            )
            return False, order
        return True, order

    def __check_available_quantity(
        self, order: OrderData, old_quantity: float, position: PositionData, available_balance: float, step_size: float
    ) -> tuple[bool, OrderData]:
        """Check if sufficient quantity is available for the order.

        Args:
            order: Order to check
            old_quantity: Current position quantity
            position: Position data
            available_balance: Available balance
            step_size: Minimum step size

        Returns:
            Tuple of (is_valid, updated_order)
        """
        side = order.side
        inverse_direction = -1 if side in {Side.BUY, Side.BUYCOVER} else 1
        old_quantity *= inverse_direction

        if old_quantity < order.quantity:
            # For inverse future contracts, check if balance is sufficient for new position
            if position.is_contract and order.position_side == PositionSide.NET:
                return self.__check_available_balance(order, old_quantity, position, available_balance, step_size)
            else:
                order.msg += (
                    f"You don't have enough position({old_quantity}) to {order.side.value.lower()} "
                    f"{order.quantity}({order.asset_type}) on {order.symbol} at price ${order.price}. "
                    f"\n[DEBUG] pos: {position}"
                )
                return False, order

        return True, order
