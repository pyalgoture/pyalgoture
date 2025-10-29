import logging
from abc import abstractmethod
from datetime import datetime
from math import floor
from threading import Thread
from typing import Any

from .base import Base
from .utils.event_engine import (
    EVENT_ACCOUNT,
    EVENT_BAR,
    EVENT_LOG,
    EVENT_ORDER,
    EVENT_POSITION,
    EVENT_TICK,
    EVENT_TIMER,
    EVENT_TRADE,
    Event,
)
from .utils.logger import LogData
from .utils.objects import (
    AccountData,
    BarData,
    Exchange,
    OrderData,
    OrderType,
    PositionSide,
    Side,
    Status,
    TickData,
    TimeInForce,
    TradeData,
)
from .utils.util_math import prec_round


class Strategy(Base):
    """
    Base strategy class for trading algorithms.

    This class provides the foundation for implementing trading strategies,
    handling events, and managing orders and positions.
    """

    def __init__(self) -> None:
        """Initialize the strategy."""
        super().__init__()

    def initialize(self) -> None:
        """Initialize strategy before backtesting starts."""
        pass

    def register_event(self) -> None:
        """Register event handlers for various market events."""
        event_handlers = [
            (EVENT_LOG, self.process_log_event),
            (EVENT_TICK, self.process_tick_event),
            (EVENT_BAR, self.process_bar_event),
            (EVENT_TIMER, self.process_timer_event),
            (EVENT_ORDER, self.process_order_event),
            (EVENT_TRADE, self.process_trade_event),
            (EVENT_POSITION, self.process_position_event),
            (EVENT_ACCOUNT, self.process_account_event),
        ]

        for event_type, handler in event_handlers:
            self.ctx.event_engine.register(event_type, handler)

    def process_log_event(self, event: Event) -> None:
        """Process log events and handle logging output."""
        log: LogData = event.data
        try:
            if not self.ctx.do_print:
                return

            self._handle_log_output(log)
        except Exception as e:
            self._handle_log_error(e, log)

    def _handle_log_output(self, log: LogData) -> None:
        """Handle the actual log output based on logger configuration."""
        use_default_dt = getattr(self.logger, "use_default_dt", True)

        if not use_default_dt:
            if log.level == "EXCEPTION":
                self.logger.bind(tick=log.tick).exception(log.content)
            else:
                self.logger.bind(tick=log.tick).log(log.level, log.content)
        else:
            if log.level == "EXCEPTION":
                self.logger.exception(log.content)
            elif hasattr(self.logger, "use_default_dt"):
                self.logger.log(log.level, log.content)
            else:
                if log.level == "TRACE":
                    return
                level = getattr(logging, log.level, logging.INFO) if log.level != "NOTICE" else logging.INFO
                self.logger.log(level, log.content)

    def _handle_log_error(self, error: Exception, log: LogData) -> None:
        """Handle errors that occur during log processing."""
        error_msg = f"Something went wrong when processing log event. Error: {str(error)}. log:{log}"
        print(error_msg)
        print(f"{log.tick} [{log.level}]: {log.content}")

    def process_tick_event(self, event: Event) -> None:
        """Process tick events and update market data."""
        tick_data: TickData = event.data
        try:
            if not tick_data or not tick_data.get("last_price"):
                return

            symbol = tick_data["aio_symbol"]
            self.ctx.latest_price[symbol] = tick_data["last_price"]
            self.ctx.tick_data[symbol] = tick_data

            for runner in self.ctx.runners:
                runner.on_tick(tick_data)
        except Exception as e:
            error_msg = f"Something went wrong when processing tick event. Error: {str(e)}. tick_data:{tick_data}"
            print(error_msg)
            self.exception(error_msg)

    def process_bar_event(self, event: Event) -> None:
        """Process bar events and update OHLCV data."""
        data: dict[str, Any] = event.data
        try:
            tick: datetime = data["tick"]
            bars: dict[str, BarData] = data["bars"]

            self.trace(f"[process_bar_event] bars:{bars}")

            self.ctx.tick = tick
            self.ctx.bar_data = bars
            self.ctx.latest_price = {symbol: bar.get("close") if bar else None for symbol, bar in bars.items()}

            for runner in self.ctx.runners:
                runner.on_bar(tick)

        except Exception as e:
            self.exception(f"Error processing bar event: {str(e)}\nData: {data}")

    def process_timer_event(self, event: Event) -> None:
        """Process timer events."""
        pass

    def process_order_event(self, event: Event) -> None:
        """Process order events and update order status."""
        order: OrderData = event.data
        try:
            self.trace(f"[process_order_event] order:{order}")

            if isinstance(order, OrderData):
                self.ctx.broker.update_order(order)
                self.on_order(order)
            else:
                self.error(f"Order data not correctly parsed in process order event. order:{order}")

        except Exception as e:
            self.exception(f"Error processing order event: {str(e)}\nData: {order}")

    def process_trade_event(self, event: Event) -> None:
        """Process trade events and handle trade execution."""
        trade: TradeData = event.data
        try:
            self.trace(f"[process_trade_event] trade:{trade}")

            if not isinstance(trade, TradeData):
                self.error(f"Trade data not correctly parsed in process trade event. trade:{trade}")
                return

            self.ctx.broker.update_trade(trade)

            # Upload trade to statistics if available
            if hasattr(self.ctx, "statistic") and hasattr(self.ctx.statistic, "upload_trade"):
                self.ctx.statistic.upload_trade(trade)

            self.ctx.strategy.on_trade(trade)

            # Handle custom trade callback for live trading
            self._handle_trade_callback(trade)

        except Exception as e:
            self.exception(f"Error processing trade event: {str(e)}\nData: {trade}")

    def _handle_trade_callback(self, trade: TradeData) -> None:
        """Handle custom trade callback in a separate thread."""
        if not (hasattr(self.ctx, "on_trade_callback_custom") and self.ctx.is_live):
            return

        try:
            thread = Thread(
                target=self.ctx.on_trade_callback_custom,
                kwargs={"ctx": self.ctx, "trade": trade},
                daemon=True,
            )
            thread.start()
        except Exception as e:
            self.exception(f"Error processing trade callback event: {str(e)}\nData: {trade}")

    def process_position_event(self, event: Event) -> None:
        """Process position events and update position data."""
        position: dict[str, Any] = event.data
        try:
            self.trace(f"[process_position_event] position:{position} (before update position)")
            self.ctx.broker.check_position(position)
        except Exception as e:
            self.exception(f"Error processing position event: {str(e)}\nData: {position}")

    def process_account_event(self, event: Event) -> None:
        """Process account events and update account data."""
        account: AccountData = event.data
        try:
            self.trace(f"[process_account_event] account:{account}")

            if isinstance(account, AccountData):
                self.ctx.broker.update_account(account)
            else:
                self.error(f"Account data not correctly parsed in process account event. account:{account}")

        except Exception as e:
            self.exception(f"Error processing account event: {str(e)}\nData: {account}")

    def on_order(self, order: OrderData) -> None:
        """Handle order status changes by dispatching to appropriate handlers."""
        status_handlers = {
            Status.NEW: self.on_order_place,
            Status.CANCELED: self.on_order_cancel,
            Status.AMENDED: self.on_order_amend,
            Status.FAILED: self.on_order_fail,
            Status.LIQUIDATED: self.on_order_liquidate,
            Status.FILLED: self.on_order_complete,
        }

        handler = status_handlers.get(order.status)
        if handler:
            handler(order)

    def on_trade(self, trade: TradeData) -> None:
        """
        Handle trade execution events.

        Args:
            trade: Trade data containing execution details
        """
        pass

    def on_order_liquidate(self, order: OrderData) -> None:
        """
        Handle liquidated order events.

        Args:
            order: Order that was liquidated
        """
        pass

    def on_order_complete(self, order: OrderData) -> None:
        """
        Handle completely filled order events.

        Args:
            order: Order that was completely filled
        """
        pass

    def on_order_fail(self, trade: TradeData) -> None:
        """
        Handle failed order events.

        Args:
            trade: Trade data for the failed order
        """
        pass

    def on_order_cancel(self, order: OrderData) -> None:
        """
        Handle canceled order events.

        Args:
            order: Order that was canceled
        """
        pass

    def on_order_amend(self, order: OrderData) -> None:
        """
        Handle amended order events.

        Args:
            order: Order that was amended
        """
        pass

    def on_order_place(self, order: OrderData) -> None:
        """
        Handle placed order events.

        Args:
            order: Order that was placed
        """
        pass

    def finish(self) -> None:
        """Handle strategy completion after backtesting."""
        pass

    def on_tick(self, tick: datetime) -> None:
        """
        Handle tick events for strategy evaluation.

        Args:
            tick: Current datetime of the tick
        """
        pass

    @abstractmethod
    def on_bar(self, tick: datetime) -> None:
        """
        Handle bar events for strategy evaluation. Must be implemented by subclasses.

        Args:
            tick: Current datetime of the bar
        """
        pass

    @property
    def available_balance(self) -> float:
        """
        Calculate available balance considering positions and unrealized PnL.

        Returns:
            Available balance for trading
        """
        available: float = 0.0
        if self.ctx.is_live:
            available = self.ctx.default_account.available
            return available

        # Calculate position margin and unrealized PnL
        pos_margin = 0.0
        unrealized_pnl = 0.0
        for pos in self.ctx.get_positions():
            if pos:
                pos_margin += pos.position_margin
                unrealized_pnl += pos.unrealized_pnl

        available = max(0, self.ctx.default_account.available - pos_margin + unrealized_pnl)

        self.trace(
            f"[Strategy - available_balance] > all_pos_margin: {pos_margin} + unrealized_pnl: {unrealized_pnl} || "
            f"available_balance:{available}; balance:{self.ctx.default_account.balance}; "
            f"frozen:{self.ctx.default_account.frozen} || available:{self.ctx.default_account.available} || "
            f"self.ctx.default_account:{self.ctx.default_account} - [{id(self.ctx.default_account)}]"
        )
        return available

    def order(
        self,
        symbol: str,
        exchange: Exchange,
        quantity: float,
        side: str | Side | None,
        tag: str,
        msg: str = "",
        price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
        trail_percent: float | None = None,
        position_side: str | PositionSide | None = PositionSide.NET,
        time_in_force: TimeInForce | None = TimeInForce.GTC,
        **kwargs: Any,
    ) -> OrderData:
        """
        Create and send an order to the broker.

        Args:
            symbol: Trading symbol that must exist in self.ctx.feeds
            exchange: Exchange for the order
            quantity: Order quantity (must be positive)
            side: Order side (BUY, SELL, SHORTSELL)
            tag: Order tag for identification
            msg: Optional message
            price: Limit price (None for market order)
            stop_price: Stop price for stop orders
            trail_price: Trailing stop price
            trail_percent: Trailing stop percentage
            position_side: Position side (NET by default)
            time_in_force: Time in force (GTC by default)
            **kwargs: Additional order parameters

        Returns:
            Created order data
        """
        datafeed = self.ctx.get_datafeed(symbol=symbol, exchange=exchange)
        order_id = kwargs.pop("order_id", self.ctx.broker._new_order_id())

        order = OrderData(
            order_id=order_id,
            ori_order_id=None,
            exchange=exchange,
            code=datafeed.code,
            symbol=symbol,
            asset_type=datafeed.asset_type,
            price=price,
            quantity=prec_round(abs(quantity), 6),
            side=side,
            position_side=position_side,
            executed_quantity=0.0,
            order_type=OrderType.LIMIT if price else OrderType.MARKET,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=Status.NEW,
            time_in_force=time_in_force,
            requested_price=price,
            stop_price=stop_price,
            trail_price=trail_price,
            trail_percent=trail_percent,
            msg=msg,
            tag=tag,
        )

        # Set is_open if provided in kwargs
        if "is_open" in kwargs:
            order.is_open = kwargs.pop("is_open")
        return self.ctx.broker.send_order(order, **kwargs)

    def buy(
        self,
        symbol: str,
        exchange: Exchange,
        quantity: float,
        tag: str,
        msg: str = "",
        price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
        trail_percent: float | None = None,
        position_side: str | PositionSide | None = PositionSide.NET,
        time_in_force: TimeInForce | None = TimeInForce.GTC,
        **kwargs: Any,
    ) -> OrderData:
        """
        Create a buy (long) order.

        Args:
            See order() method for parameter descriptions

        Returns:
            Created buy order
        """
        return self.order(
            symbol=symbol,
            quantity=quantity,
            side=Side.BUY,
            price=price,
            stop_price=stop_price,
            trail_price=trail_price,
            trail_percent=trail_percent,
            position_side=position_side,
            exchange=exchange,
            tag=tag,
            msg=msg,
            time_in_force=time_in_force,
            **kwargs,
        )

    def sell(
        self,
        symbol: str,
        exchange: Exchange,
        quantity: float,
        tag: str,
        msg: str = "",
        price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
        trail_percent: float | None = None,
        position_side: str | PositionSide | None = PositionSide.NET,
        time_in_force: TimeInForce | None = TimeInForce.GTC,
        **kwargs: Any,
    ) -> OrderData:
        """
        Create a sell (short) order.

        Args:
            See order() method for parameter descriptions

        Returns:
            Created sell order
        """
        return self.order(
            symbol=symbol,
            quantity=quantity,
            side=Side.SELL,
            price=price,
            stop_price=stop_price,
            trail_price=trail_price,
            trail_percent=trail_percent,
            position_side=position_side,
            exchange=exchange,
            tag=tag,
            msg=msg,
            time_in_force=time_in_force,
            **kwargs,
        )

    def close(
        self,
        symbol: str,
        exchange: Exchange,
        tag: str,
        msg: str = "",
        price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
        trail_percent: float | None = None,
        position_side: str | PositionSide | None = PositionSide.NET,
        time_in_force: TimeInForce | None = TimeInForce.GTC,
        **kwargs: Any,
    ) -> OrderData:
        """
        Close the position by setting target quantity to zero.

        Args:
            See order() method for parameter descriptions

        Returns:
            Created close order
        """
        return self.order_target_quantity(
            symbol=symbol,
            overall_quantity=0,
            price=price,
            stop_price=stop_price,
            trail_price=trail_price,
            trail_percent=trail_percent,
            position_side=position_side,
            exchange=exchange,
            tag=tag,
            msg=msg,
            time_in_force=time_in_force,
            **kwargs,
        )

    def order_by_lotsize(
        self,
        symbol: str,
        exchange: Exchange,
        lots: float,
        side: str | Side | None,
        tag: str,
        msg: str = "",
        price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
        trail_percent: float | None = None,
        position_side: str | PositionSide | None = PositionSide.NET,
        time_in_force: TimeInForce | None = TimeInForce.GTC,
        **kwargs: Any,
    ) -> OrderData:
        """
        Create an order based on lot size.

        Args:
            lots: Number of lots to trade (quantity = lots * step_size)
            See order() method for other parameter descriptions

        Returns:
            Created order
        """
        step_size = self.ctx.get_datafeed(symbol=symbol, exchange=exchange).step_size
        quantity = lots * step_size

        return self.order(
            symbol=symbol,
            quantity=quantity,
            side=side,
            price=price,
            stop_price=stop_price,
            trail_price=trail_price,
            trail_percent=trail_percent,
            position_side=position_side,
            exchange=exchange,
            tag=tag,
            msg=msg,
            time_in_force=time_in_force,
            **kwargs,
        )

    def order_by_qty(
        self,
        symbol: str,
        exchange: Exchange,
        quantity: float,
        side: str | Side | None,
        tag: str,
        msg: str = "",
        price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
        trail_percent: float | None = None,
        position_side: str | PositionSide | None = PositionSide.NET,
        time_in_force: TimeInForce | None = TimeInForce.GTC,
        **kwargs: Any,
    ) -> OrderData:
        """
        Create an order based on designated quantity (alias for order method).

        Args:
            See order() method for parameter descriptions

        Returns:
            Created order
        """
        return self.order(
            symbol=symbol,
            quantity=quantity,
            side=side,
            price=price,
            stop_price=stop_price,
            trail_price=trail_price,
            trail_percent=trail_percent,
            position_side=position_side,
            exchange=exchange,
            tag=tag,
            msg=msg,
            time_in_force=time_in_force,
            **kwargs,
        )

    def _get_current_price(self, symbol: str, exchange: Exchange, price: float | None = None) -> float:
        """Get current price for calculations."""
        if price:
            return price

        current_price: float = 0.0
        trade_at = getattr(self.ctx.broker, "trade_at", None)
        if trade_at:
            bar_data = self.ctx.get_bar_data(symbol=symbol, exchange=exchange)
            current_price = bar_data[trade_at]

        current_price = self.ctx.get_latest_price(symbol=symbol, exchange=exchange)
        return current_price

    def order_by_cost(
        self,
        symbol: str,
        exchange: Exchange,
        cost: float,
        side: str | Side | None,
        tag: str,
        msg: str = "",
        price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
        trail_percent: float | None = None,
        position_side: str | PositionSide | None = PositionSide.NET,
        time_in_force: TimeInForce | None = TimeInForce.GTC,
        **kwargs: Any,
    ) -> OrderData:
        """
        Create an order based on designated cost (without leverage).

        Args:
            cost: Order cost amount
            See order() method for other parameter descriptions

        Returns:
            Created order

        Raises:
            ValueError: If cost is less than or equal to 0
        """
        if cost <= 0:
            raise ValueError("The order amount cannot be less than 0")

        current_price = self._get_current_price(symbol, exchange, price)
        datafeed = self.ctx.get_datafeed(symbol=symbol, exchange=exchange)
        step_size = datafeed.step_size or 1

        if datafeed.is_inverse:
            quantity = cost
        else:
            position = self.ctx.get_position(
                symbol=symbol,
                exchange=exchange,
                position_side=position_side,
                force_return=True,
            )
            commission_rate = self.ctx.get_commission_rate(symbol=symbol, exchange=exchange)
            margin_price = price or current_price
            quantity = (
                floor(cost / position.get_margin(price=margin_price) / (commission_rate + 1) / step_size) * step_size
            )

        self.trace(
            f"[order_by_cost] [{symbol}] cost:{cost}| available_balance:{self.available_balance} | "
            f"step_size:{step_size}; quantity:{quantity}; current_price:{current_price};"
        )

        return self.order(
            symbol=symbol,
            quantity=quantity,
            side=side,
            price=price,
            stop_price=stop_price,
            trail_price=trail_price,
            trail_percent=trail_percent,
            position_side=position_side,
            exchange=exchange,
            tag=tag,
            msg=msg,
            time_in_force=time_in_force,
            **kwargs,
        )

    def order_by_value(
        self,
        symbol: str,
        exchange: Exchange,
        amount: float,
        side: str | Side | None,
        tag: str,
        msg: str = "",
        price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
        trail_percent: float | None = None,
        position_side: str | PositionSide | None = PositionSide.NET,
        time_in_force: TimeInForce | None = TimeInForce.GTC,
        **kwargs: Any,
    ) -> OrderData:
        """
        Create an order based on designated value (after leverage).

        Args:
            amount: Order amount after leverage
            See order() method for other parameter descriptions

        Returns:
            Created order

        Raises:
            ValueError: If amount is less than or equal to 0
        """
        if amount <= 0:
            raise ValueError("The order amount cannot be less than 0")

        current_price = self._get_current_price(symbol, exchange, price)
        datafeed = self.ctx.get_datafeed(symbol=symbol, exchange=exchange)
        step_size = datafeed.step_size or 1

        if datafeed.is_inverse:
            position = self.ctx.get_position(
                symbol=symbol,
                exchange=exchange,
                position_side=position_side,
                force_return=True,
            )
            quantity = amount / position.leverage
        else:
            commission_rate = self.ctx.get_commission_rate(symbol=symbol, exchange=exchange)
            order_price = price or current_price
            quantity = floor(amount / order_price / (commission_rate + 1) / step_size) * step_size

        self.trace(
            f"[order_by_value] [{symbol}] amount:{amount}| available_balance:{self.available_balance} | "
            f"step_size:{step_size}; quantity:{quantity}; current_price:{current_price};"
        )

        return self.order(
            symbol=symbol,
            quantity=quantity,
            side=side,
            price=price,
            stop_price=stop_price,
            trail_price=trail_price,
            trail_percent=trail_percent,
            position_side=position_side,
            exchange=exchange,
            tag=tag,
            msg=msg,
            time_in_force=time_in_force,
            **kwargs,
        )

    def order_target_quantity(
        self,
        symbol: str,
        exchange: Exchange,
        overall_quantity: float,
        tag: str,
        msg: str = "",
        price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
        trail_percent: float | None = None,
        position_side: str | PositionSide | None = PositionSide.NET,
        time_in_force: TimeInForce | None = TimeInForce.GTC,
        **kwargs: Any,
    ) -> OrderData:
        """
        Place an order to rebalance position to target quantity.

        Args:
            overall_quantity: Target total position quantity
            See order() method for other parameter descriptions

        Returns:
            Created order
        """
        current_quantity = self.ctx.get_position_size(symbol=symbol, exchange=exchange, position_side=position_side)
        delta_quantity = overall_quantity - current_quantity

        if kwargs.get("print_delta"):
            self.info(f"Delta quantity: {delta_quantity}")

        quantity = abs(delta_quantity)
        side = Side.BUY if delta_quantity > 0 else Side.SELL

        return self.order(
            symbol=symbol,
            quantity=quantity,
            side=side,
            price=price,
            stop_price=stop_price,
            trail_price=trail_price,
            trail_percent=trail_percent,
            position_side=position_side,
            exchange=exchange,
            tag=tag,
            msg=msg,
            time_in_force=time_in_force,
            **kwargs,
        )

    def order_target_value(
        self,
        symbol: str,
        exchange: Exchange,
        overall_amount: float,
        tag: str,
        msg: str = "",
        price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
        trail_percent: float | None = None,
        position_side: str | PositionSide | None = PositionSide.NET,
        time_in_force: TimeInForce | None = TimeInForce.GTC,
        **kwargs: Any,
    ) -> OrderData:
        """
        Place an order to rebalance position to target value.

        Args:
            overall_amount: Target total position value
            See order() method for other parameter descriptions

        Returns:
            Created order

        Raises:
            ValueError: If overall_amount is negative
        """
        if overall_amount < 0:
            raise ValueError("The order amount cannot be less than 0")

        bar_data = self.ctx.get_bar_data(symbol=symbol, exchange=exchange)
        close_price = bar_data["close"]
        step_size = self.ctx.get_datafeed(symbol=symbol, exchange=exchange).step_size
        current_value = self.ctx.get_market_value(symbol=symbol, exchange=exchange, position_side=position_side)
        delta_amount = overall_amount - current_value
        commission_rate = self.ctx.get_commission_rate(symbol=symbol, exchange=exchange)

        self.trace(
            f"[order_target_value] overall_amount: {overall_amount}; current_value:{current_value}; "
            f"delta_amount: {delta_amount}; price: {price}; close_price: {close_price}; "
            f"commission_rate: {commission_rate}; step_size: {step_size}"
        )

        order_price = price or close_price
        delta_quantity = floor(delta_amount / order_price / (commission_rate + 1) / step_size) * step_size

        if kwargs.get("print_delta"):
            self.info(
                f"Current hold amount: ${current_value}, Delta amount: ${delta_amount}, "
                f"Delta quantity: {delta_quantity}"
            )

        quantity = abs(delta_quantity)
        side = Side.BUY if delta_quantity > 0 else Side.SELL

        return self.order(
            symbol=symbol,
            quantity=quantity,
            side=side,
            price=price,
            stop_price=stop_price,
            trail_price=trail_price,
            trail_percent=trail_percent,
            position_side=position_side,
            exchange=exchange,
            tag=tag,
            msg=msg,
            time_in_force=time_in_force,
            **kwargs,
        )

    def order_target_percent(
        self,
        symbol: str,
        exchange: Exchange,
        target_percent: float,
        tag: str,
        msg: str = "",
        price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
        trail_percent: float | None = None,
        position_side: str | PositionSide | None = PositionSide.NET,
        time_in_force: TimeInForce | None = TimeInForce.GTC,
        **kwargs: Any,
    ) -> OrderData:
        """
        Place an order to rebalance position to target percentage of portfolio.

        Args:
            target_percent: Target percentage of portfolio value (as decimal)
            See order() method for other parameter descriptions

        Returns:
            Created order
        """
        overall_amount = self.ctx.assets_value * target_percent

        if not overall_amount:
            return self.order_target_quantity(
                symbol=symbol,
                overall_quantity=0,
                price=price,
                stop_price=stop_price,
                trail_price=trail_price,
                trail_percent=trail_percent,
                position_side=position_side,
                exchange=exchange,
                tag=tag,
                msg=msg,
                time_in_force=time_in_force,
                **kwargs,
            )

        return self.order_target_value(
            symbol=symbol,
            overall_amount=overall_amount,
            price=price,
            stop_price=stop_price,
            trail_price=trail_price,
            trail_percent=trail_percent,
            position_side=position_side,
            exchange=exchange,
            tag=tag,
            msg=msg,
            time_in_force=time_in_force,
            **kwargs,
        )

    def cancel_order(
        self,
        order: OrderData | None = None,
        symbol: str | None = None,
        exchange: Exchange | None = None,
        order_id: str | None = None,
    ) -> OrderData:
        """
        Cancel an order in the broker.

        Args:
            order: Order object to cancel
            symbol: Symbol of the order to cancel
            exchange: Exchange of the order to cancel
            order_id: ID of the order to cancel

        Returns:
            Canceled order data
        """
        self.trace(
            f">>>>> [strategy - cancel_order] order:{order}; order_id:{order_id}; exchange:{exchange}; symbol:{symbol}"
        )

        if order:
            return self.ctx.broker.cancel_order(order=order)
        else:
            return self.ctx.broker.cancel_order(symbol=symbol, exchange=exchange, order_id=order_id)

    def cancel_all(
        self,
        symbol: str | None = None,
        exchange: Exchange | None = None,
        position_side: PositionSide = PositionSide.NET,
    ) -> None:
        """
        Cancel all open orders, optionally filtered by symbol.

        Args:
            symbol: Optional symbol filter
            exchange: Optional exchange filter
            position_side: Position side filter
        """
        open_orders = self.ctx.broker.get_open_orders()

        for order in open_orders:
            if symbol is None or order.symbol == symbol:
                self.cancel_order(order)

    def draw_figure(self) -> None:
        """Plot custom bokeh figure into report HTML."""
        pass

    def add_balance(self, amount: float) -> None:
        """
        Add or remove balance from the account.

        Args:
            amount: Amount to add (positive) or remove (negative)
        """
        if hasattr(self.ctx.broker, "add_balance"):
            self.ctx.broker.add_balance(amount)

        if amount > 0:
            self.ctx.statistic.fund_tracker.subscribe(amount, dt=self.ctx.tick)
        else:
            self.ctx.statistic.fund_tracker.redeem(amount, dt=self.ctx.tick)
