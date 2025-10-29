from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import polars as pl

from ..strategy import Strategy
from ..utils.objects import BarData, Exchange, OrderData, PositionSide, TradeData
from .lab import AlphaLab

if TYPE_CHECKING:
    from ..orchestrators.backtest import BackTest


class AlphaStrategy(Strategy):
    lab: AlphaLab | None = None
    signal_df: pl.DataFrame | None = None
    exchange: Exchange | None = None

    def initialize(self) -> None:
        self.holding_days: defaultdict[str, int] = defaultdict(int)
        if not self.lab:
            raise ValueError("lab is not initialized")
        if self.signal_df is None:
            raise ValueError("signal_df is not initialized")

        # Add a column named 'symbol' which is the first part of aio_symbol (before the '|')
        self.signal_df = self.signal_df.with_columns(pl.col("aio_symbol").str.split("|").list.get(0).alias("symbol"))
        self.debug(f"signal_df: {self.signal_df}")

        self.pos_data: dict[str, float] = defaultdict(float)  # Actual positions
        self.target_data: dict[str, float] = defaultdict(float)  # Target positions

        self.exchange: Exchange = Exchange(self.exchange.upper()) if isinstance(self.exchange, str) else self.exchange

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
        pass

    def get_target(self, symbol: str, exchange: Exchange) -> float:
        """Query target position"""
        return self.target_data[f"{symbol}|{exchange.value}"]

    def set_target(self, symbol: str, exchange: Exchange, target: float) -> None:
        """Set target position"""
        self.target_data[f"{symbol}|{exchange.value}"] = target

    def get_signal(self, dt: Any) -> pl.DataFrame:
        """Get model prediction signal for current time"""
        if self.signal_df is None:
            raise ValueError("signal_df is not initialized")

        signal: pl.DataFrame = self.signal_df.filter(pl.col("datetime") == dt.replace(tzinfo=None))

        if signal.is_empty():
            self.error(f"Cannot find signal for {dt}")

        # self.debug(f"Signal with symbol column: {signal.head(4)}")
        """
        shape: (4, 4)
        ┌─────────────────────┬───────────────┬───────────┬─────────┐
        │ datetime            ┆ aio_symbol    ┆ signal    ┆ symbol  │
        │ ---                 ┆ ---           ┆ ---       ┆ ---     │
        │ datetime[μs]        ┆ str           ┆ f64       ┆ str     │
        ╞═════════════════════╪═══════════════╪═══════════╪═════════╡
        │ 2024-05-22 08:00:00 ┆ SOLUSDT|bybit ┆ 0.024129  ┆ SOLUSDT │
        │ 2024-05-22 08:00:00 ┆ BTCUSDT|bybit ┆ 0.009035  ┆ BTCUSDT │
        │ 2024-05-22 08:00:00 ┆ ETHUSDT|bybit ┆ 0.009035  ┆ ETHUSDT │
        │ 2024-05-22 08:00:00 ┆ XRPUSDT|bybit ┆ -0.012684 ┆ XRPUSDT │
        └─────────────────────┴───────────────┴───────────┴─────────┘;
        """
        return signal

    def execute_trading(self, price_add: float) -> None:
        """Execute position adjustment based on targets"""
        self.cancel_all()

        # Only send orders for feed_infos with current bar data
        for symbol, exchange in self.ctx.symbols:
            bar: BarData | None = self.ctx.get_bar_data(symbol=symbol, exchange=exchange)
            if bar is None:
                continue

            # Calculate position difference
            target: float = self.get_target(symbol=symbol, exchange=exchange)
            pos: float = self.ctx.get_position_size(symbol=symbol, exchange=exchange)
            diff: float = target - pos

            self.notice(f"symbol:{symbol}, exchange:{exchange}, target:{target}, pos:{pos}, diff:{diff}")

            # Long position
            if diff > 0:
                # Calculate long order price
                order_price: float = bar.close * (1 + price_add)

                # Calculate cover and buy volumes
                cover_volume: float = 0
                buy_volume: float = 0

                if pos < 0:
                    cover_volume = min(diff, abs(pos))
                    buy_volume = diff - cover_volume
                else:
                    buy_volume = diff

                self.notice(
                    f"symbol:{symbol}, exchange:{exchange}, order_price:{order_price}, cover_volume:{cover_volume}, buy_volume:{buy_volume}"
                )
                # Send corresponding orders
                if cover_volume:
                    o = self.buy(
                        symbol=symbol, exchange=exchange, price=order_price, quantity=cover_volume, tag="close long"
                    )
                    self.info(f"Close long order: {o}")

                if buy_volume:
                    o = self.buy(
                        symbol=symbol, exchange=exchange, price=order_price, quantity=buy_volume, tag="open long"
                    )
                    self.info(f"Open long order: {o}")
            # Short position
            elif diff < 0:
                # Calculate short order price
                order_price = bar.close * (1 - price_add)

                # Calculate sell and short volumes
                sell_volume: float = 0
                short_volume: float = 0

                if pos > 0:
                    sell_volume = min(abs(diff), pos)
                    short_volume = abs(diff) - sell_volume
                else:
                    short_volume = abs(diff)
                self.notice(
                    f"symbol:{symbol}, exchange:{exchange}, order_price:{order_price}, sell_volume:{sell_volume}, short_volume:{short_volume}"
                )
                # Send corresponding orders
                if sell_volume:
                    o = self.sell(
                        symbol=symbol, exchange=exchange, price=order_price, quantity=sell_volume, tag="close short"
                    )
                    self.info(f"Close short order: {o}")

                if short_volume:
                    o = self.sell(
                        symbol=symbol, exchange=exchange, price=order_price, quantity=short_volume, tag="open short"
                    )
                    self.info(f"Open short order: {o}")
