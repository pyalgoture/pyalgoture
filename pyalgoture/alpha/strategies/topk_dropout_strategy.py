from collections import defaultdict
from typing import Any

import polars as pl

from ...utils.objects import BarData, TradeData
from ...utils.util_math import round_to
from ..strategy import AlphaStrategy


class TopkDropoutStrategy(AlphaStrategy):
    top_k: int = 50  # Maximum number of assets to hold
    n_drop: int = 5  # Number of assets to sell each time
    min_days: int = 3  # Minimum holding period in days
    cash_ratio: float = 0.95  # Cash utilization ratio
    price_add: float = 0.05  # Order price adjustment ratio

    def initialize(self) -> None:
        super().initialize()

        self.holding_days: defaultdict[str, int] = defaultdict(int)
        self.info(
            f"Strategy initialized.\n\ttopk:{self.top_k}\n\tn_drop:{self.n_drop}\n\tmin_days:{self.min_days}\n\tcash_ratio:{self.cash_ratio}\n\tprice_add:{self.price_add}"
        )

    def finish(self) -> None:
        pass

    def on_order_fail(self, order: Any) -> None:
        self.error(f"Order [number {order['order_id']}] [{order['status']}]. Msg: {order['msg']}")

    def on_order_cancel(self, order: Any) -> None:
        self.warning(f"Order [number {order['order_id']}] [{order['status']}]. Msg: {order['msg']}")

    def on_order_complete(self, order: Any) -> None:
        self.info(f"Order completed [number {order['order_id']}] [{order['status']}].")

    def on_trade(self, trade: Any) -> None:
        is_open = trade.is_open
        self.info(
            f"({trade.aio_position_id}) - {'OPEN' if is_open else 'CLOSE'}{'' if is_open else '(TP)' if trade.pnl > 0 else '(SL)'} {trade['side']} order [number {trade['order_id']}] executed [quantity {trade['quantity']}] [price ${trade['price']:2f}] [Cost ${trade['gross_amount']:2f}] [Commission: ${trade['commission']}] [Available balance: ${self.available_balance}] [Position: #{self.ctx.get_position_size(symbol=trade['symbol'], exchange=trade['exchange'], position_side=trade['position_side'])}] [Gross P&L: ${trade['pnl']}] "
        )  # [Msg: {trade['msg']}]

        if not is_open:
            self.info(f"========> Trade closed, pnl: {trade['pnl']}")
            self.holding_days.pop(trade.symbol, None)

    def on_bar(self, tick: Any) -> None:
        # Get the latest signals and sort them
        last_signal: pl.DataFrame = self.get_signal(tick)
        if last_signal.is_empty():
            return

        last_signal = last_signal.sort("signal", descending=True)

        # Get position symbols and update holding days
        pos_symbols: list[str] = [pos.symbol for pos in self.ctx.get_open_positions() if pos]

        for symbol in pos_symbols:
            self.holding_days[symbol] += 1

        # Generate sell list
        active_symbols: set[str] = set(last_signal["symbol"][: self.top_k])  # Extract symbols with highest signals
        active_symbols.update(pos_symbols)  # Merge with currently held symbols
        active_df: pl.DataFrame = last_signal.filter(
            pl.col("symbol").is_in(active_symbols)
        )  # Filter signals for these symbols

        component_symbols: set[str] = set(last_signal["symbol"])  # Extract current index component symbols
        sell_symbols: set[str] = set(pos_symbols).difference(component_symbols)  # Sell positions not in components

        for symbol in active_df["symbol"][-self.n_drop :]:  # Iterate through lowest signal portion
            if symbol in pos_symbols:  # If the contract is in current positions
                sell_symbols.add(symbol)  # Add it to sell list

        # Generate buy list
        buyable_df: pl.DataFrame = last_signal.filter(
            ~pl.col("symbol").is_in(pos_symbols)
        )  # Filter contracts available for purchase
        buy_quantity: int = len(sell_symbols) + self.top_k - len(pos_symbols)  # Calculate number of contracts to buy
        buy_symbols: list[str] = list(buyable_df[:buy_quantity]["symbol"])  # Select buy contract code list

        self.debug(f"last_signal:{last_signal};")
        self.debug(
            f"pos_symbols:{pos_symbols}; active_symbols:{active_symbols}; sell_symbols:{sell_symbols}; buy_symbols: {buy_symbols}"
        )
        """
        last_signal:shape: (2, 3)
        ┌─────────────────────┬───────────────┬───────────┐
        │ datetime            ┆ aio_symbol    ┆ signal    │
        │ ---                 ┆ ---           ┆ ---       │
        │ datetime[μs]        ┆ str           ┆ f64       │
        ╞═════════════════════╪═══════════════╪═══════════╡
        │ 2024-05-01 08:00:00 ┆ XRPUSDT|bybit ┆ -0.021213 │
        │ 2024-05-01 08:00:00 ┆ SOLUSDT|bybit ┆ -0.044998 │
        └─────────────────────┴───────────────┴───────────┘;
        pos_symbols:[]; active_symbols:{'SOLUSDT|bybit', 'XRPUSDT|bybit'}; sell_symbols:set(); buy_symbols: ['XRPUSDT|bybit', 'SOLUSDT|bybit']

        """

        # Sell rebalancing
        for symbol in sell_symbols:
            if self.holding_days[symbol] < self.min_days:  # Check if holding period exceeds threshold
                continue

            bar: BarData | None = self.ctx.get_bar_data(
                symbol=symbol, exchange=self.exchange
            )  # Get current price of the contract
            if not bar:
                continue
            # sell_price: float = bar.close
            # sell_volume: float = self.ctx.get_position_size(symbol=symbol, exchange=self.exchange)            # Get current holding volume

            self.set_target(symbol=symbol, exchange=self.exchange, target=0)  # Set target volume to 0

        # Buy rebalancing
        if buy_symbols:
            buy_value: float = (
                self.available_balance * self.cash_ratio / len(buy_symbols)
            )  # Calculate investment amount per contract

            for symbol in buy_symbols:
                buy_bar: BarData | None = self.ctx.get_bar_data(
                    symbol=symbol, exchange=self.exchange
                )  # Get current price of the contract
                if not buy_bar:
                    continue

                buy_price: float = buy_bar.close  # Get current price of the contract
                if not buy_price:
                    continue

                buy_volume: float = round(buy_value / buy_price, 6)  # Calculate volume to buy

                self.set_target(symbol=symbol, exchange=self.exchange, target=buy_volume)  # Set target holding volume

        # Execute trading
        self.execute_trading(price_add=self.price_add)
