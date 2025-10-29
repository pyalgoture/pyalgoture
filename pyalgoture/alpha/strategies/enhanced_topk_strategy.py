from collections import defaultdict
from typing import Any

import numpy as np
import polars as pl

from ...utils.objects import BarData, TradeData
from ...utils.util_math import round_to
from ..strategy import AlphaStrategy


class EnhancedTopkStrategy(AlphaStrategy):
    """
    Enhanced TopK investment strategy with improved risk management, signal processing,
    and position sizing logic based on volatility and correlation.
    """

    # Core strategy parameters
    top_k: int = 50  # Maximum number of assets to hold
    n_drop: int = 5  # Number of assets to sell each time
    min_days: int = 3  # Minimum holding period in days
    cash_ratio: float = 0.95  # Cash utilization ratio
    price_add: float = 0.05  # Order price adjustment ratio

    # Enhanced risk management parameters
    stop_loss_pct: float = 0.05  # Stop loss percentage (5%)
    take_profit_pct: float = 0.10  # Take profit percentage (10%)
    max_risk_per_trade: float = 0.02  # Maximum risk per trade (2% of portfolio)

    # Signal processing parameters
    signal_smoothing_period: int = 3  # Period for signal moving average
    correlation_threshold: float = 0.7  # Threshold for high correlation

    # Position sizing parameters
    vol_lookback_period: int = 20  # Period for volatility calculation
    vol_scaling_factor: float = 1.0  # Scaling factor for volatility-based sizing

    def initialize(self) -> None:
        super().initialize()

        # Initialize dictionaries to track various metrics
        self.holding_days: defaultdict[str, int] = defaultdict(int)
        self.entry_prices: dict[str, float] = {}  # To track entry prices for stop-loss/take-profit
        self.historical_prices: defaultdict[str, list[float]] = defaultdict(list)  # For volatility calculation
        self.smoothed_signals: dict[str, list[float]] = {}  # For storing smoothed signals
        self.position_sizes: dict[str, float] = {}  # For storing calculated position sizes

        # Log configuration
        self.info(
            f"Enhanced Strategy initialized:\n"
            f"\ttopk: {self.top_k}\n"
            f"\tn_drop: {self.n_drop}\n"
            f"\tmin_days: {self.min_days}\n"
            f"\tcash_ratio: {self.cash_ratio}\n"
            f"\tprice_add: {self.price_add}\n"
            f"\tstop_loss_pct: {self.stop_loss_pct}\n"
            f"\ttake_profit_pct: {self.take_profit_pct}\n"
            f"\tmax_risk_per_trade: {self.max_risk_per_trade}\n"
            f"\tsignal_smoothing_period: {self.signal_smoothing_period}\n"
            f"\tcorrelation_threshold: {self.correlation_threshold}\n"
        )

    def finish(self) -> None:
        self.info("Strategy execution finished.")

    def on_order_fail(self, order: Any) -> None:
        self.error(f"Order [number {order['order_id']}] [{order['status']}]. Msg: {order['msg']}")

        # If order fails, we should adjust our tracking data
        symbol = order["symbol"]
        if "open" in order.get("tag", "").lower() and symbol in self.entry_prices:
            # If opening position failed, remove entry price tracking
            del self.entry_prices[symbol]

    def on_order_cancel(self, order: Any) -> None:
        self.warning(f"Order [number {order['order_id']}] [{order['status']}]. Msg: {order['msg']}")

    def on_order_complete(self, order: Any) -> None:
        self.info(f"Order completed [number {order['order_id']}] [{order['status']}].")

        # Track entry price for positions being opened
        symbol = order["symbol"]
        if "open" in order.get("tag", "").lower():
            # For opening positions, store the entry price
            self.entry_prices[symbol] = order["price"]

    def on_trade(self, trade: Any) -> None:
        is_open = trade.is_open
        self.info(
            f"({trade.aio_position_id}) - {'OPEN' if is_open else 'CLOSE'}"
            f"{'' if is_open else '(TP)' if trade.pnl > 0 else '(SL)'} {trade['side']} "
            f"order [number {trade['order_id']}] executed "
            f"[quantity {trade['quantity']}] [price ${trade['price']:.2f}] "
            f"[Cost ${trade['gross_amount']:.2f}] [Commission: ${trade['commission']}] "
            f"[Available balance: ${self.available_balance}] "
            f"[Position: #{self.ctx.get_position_size(symbol=trade['symbol'], exchange=trade['exchange'], position_side=trade['position_side'])}] "
            f"[Gross P&L: ${trade['pnl']}] "
        )

        if not is_open:
            self.info(f"========> Trade closed, pnl: {trade['pnl']}")
            symbol = trade["symbol"]
            # Clear tracking data for closed positions
            self.holding_days.pop(symbol, None)
            if symbol in self.entry_prices:
                del self.entry_prices[symbol]

    def calculate_volatility(self, symbol: str) -> float:
        """Calculate historical volatility for a symbol"""
        prices = self.historical_prices.get(symbol, [])
        if len(prices) < self.vol_lookback_period:
            return 1.0  # Default volatility if not enough data

        # Calculate returns
        returns = np.diff(np.log(prices[-self.vol_lookback_period :]))
        # Calculate annualized volatility (assuming daily data)
        volatility = float(np.std(returns)) * np.sqrt(252)
        return float(max(0.01, volatility))  # Ensure minimum volatility

    def update_historical_prices(self, tick: Any) -> None:
        """Update historical price data for volatility calculation"""
        for symbol, exchange in self.ctx.symbols:
            bar = self.ctx.get_bar_data(symbol=symbol, exchange=exchange)
            if bar and bar.close > 0:
                self.historical_prices[symbol].append(bar.close)
                # Keep only necessary history
                if len(self.historical_prices[symbol]) > self.vol_lookback_period * 2:
                    self.historical_prices[symbol] = self.historical_prices[symbol][-self.vol_lookback_period * 2 :]

    def smooth_signals(self, signal_df: pl.DataFrame) -> pl.DataFrame:
        """Apply smoothing to raw signals"""
        # If we don't have enough history, just return the original signals
        if len(self.smoothed_signals) < self.signal_smoothing_period - 1:
            for row in signal_df.rows(named=True):
                symbol = row["symbol"]
                self.smoothed_signals.setdefault(symbol, []).append(row["signal"])
            return signal_df

        # Update signal history
        for row in signal_df.rows(named=True):
            symbol = row["symbol"]
            signals = self.smoothed_signals.setdefault(symbol, [])
            signals.append(row["signal"])
            # Keep only necessary history
            if len(signals) > self.signal_smoothing_period:
                self.smoothed_signals[symbol] = signals[-self.signal_smoothing_period :]

        # Create smoothed signals dataframe
        smoothed_rows = []
        for row in signal_df.rows(named=True):
            symbol = row["symbol"]
            if symbol in self.smoothed_signals and len(self.smoothed_signals[symbol]) > 0:
                smoothed_signal = sum(self.smoothed_signals[symbol]) / len(self.smoothed_signals[symbol])
                new_row = row.copy()
                new_row["signal"] = smoothed_signal
                smoothed_rows.append(new_row)
            else:
                smoothed_rows.append(row)

        return pl.DataFrame(smoothed_rows)

    def calculate_correlation_matrix(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        """Calculate correlation matrix between symbols"""
        if len(symbols) <= 1:
            return {}

        # Create price history matrix
        price_matrix = []
        valid_symbols = []

        for symbol in symbols:
            prices = self.historical_prices.get(symbol, [])
            if len(prices) >= self.vol_lookback_period:
                price_matrix.append(prices[-self.vol_lookback_period :])
                valid_symbols.append(symbol)

        if len(valid_symbols) <= 1:
            return {}

        # Convert to numpy array and calculate returns
        price_matrix = np.array(price_matrix)
        returns_matrix = np.diff(np.log(price_matrix), axis=1)

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(returns_matrix)

        # Convert to dictionary format
        correlation_dict: dict[str, dict[str, float]] = {}
        for i, symbol1 in enumerate(valid_symbols):
            correlation_dict[symbol1] = {}
            for j, symbol2 in enumerate(valid_symbols):
                correlation_dict[symbol1][symbol2] = corr_matrix[i, j]

        return correlation_dict

    def check_stop_loss_take_profit(self, tick: Any) -> None:
        """Check for stop loss and take profit conditions"""
        for symbol, entry_price in list(self.entry_prices.items()):
            bar = self.ctx.get_bar_data(symbol=symbol, exchange=self.exchange)
            if not bar:
                continue

            current_price = bar.close
            position_size = self.ctx.get_position_size(symbol=symbol, exchange=self.exchange)

            # Skip if no position
            if position_size == 0:
                del self.entry_prices[symbol]
                continue

            # Calculate price change percentage
            pct_change = (current_price - entry_price) / entry_price

            # For long positions
            if position_size > 0:
                # Check stop loss
                if pct_change <= -self.stop_loss_pct:
                    self.info(
                        f"Stop loss triggered for {symbol} at ${current_price:.2f} (Entry: ${entry_price:.2f}, Change: {pct_change:.2%})"
                    )
                    self.set_target(symbol=symbol, exchange=self.exchange, target=0)
                # Check take profit
                elif pct_change >= self.take_profit_pct:
                    self.info(
                        f"Take profit triggered for {symbol} at ${current_price:.2f} (Entry: ${entry_price:.2f}, Change: {pct_change:.2%})"
                    )
                    self.set_target(symbol=symbol, exchange=self.exchange, target=0)

            # For short positions
            elif position_size < 0:
                # Check stop loss (loss for short position is when price increases)
                if pct_change >= self.stop_loss_pct:
                    self.info(
                        f"Stop loss triggered for short {symbol} at ${current_price:.2f} (Entry: ${entry_price:.2f}, Change: {pct_change:.2%})"
                    )
                    self.set_target(symbol=symbol, exchange=self.exchange, target=0)
                # Check take profit (profit for short position is when price decreases)
                elif pct_change <= -self.take_profit_pct:
                    self.info(
                        f"Take profit triggered for short {symbol} at ${current_price:.2f} (Entry: ${entry_price:.2f}, Change: {pct_change:.2%})"
                    )
                    self.set_target(symbol=symbol, exchange=self.exchange, target=0)

    def calculate_position_sizes(self, buy_symbols: list[str], available_cash: float) -> dict[str, float]:
        """Calculate position sizes based on volatility"""
        if not buy_symbols:
            return {}

        # Calculate inverse volatility for each symbol
        inv_volatilities = {}
        total_inv_vol = 0.0

        for symbol in buy_symbols:
            vol = self.calculate_volatility(symbol)
            inv_vol = 1.0 / vol
            inv_volatilities[symbol] = inv_vol
            total_inv_vol += inv_vol

        # Calculate position sizes proportional to inverse volatility
        position_sizes = {}
        for symbol in buy_symbols:
            if total_inv_vol > 0:
                weight = inv_volatilities[symbol] / total_inv_vol
                position_sizes[symbol] = available_cash * weight
            else:
                # Equal weighting as fallback
                position_sizes[symbol] = available_cash / len(buy_symbols)

        return position_sizes

    def filter_by_correlation(self, symbols: list[str], signal_df: pl.DataFrame) -> list[str]:
        """Filter out highly correlated assets while preserving highest signal assets"""
        if len(symbols) <= 1:
            return symbols

        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(symbols)
        if not corr_matrix:
            return symbols

        # Create a lookup for signal values
        signal_lookup = {row["symbol"]: row["signal"] for row in signal_df.rows(named=True) if row["symbol"] in symbols}

        # Sort symbols by signal value (descending)
        sorted_symbols = sorted(symbols, key=lambda s: signal_lookup.get(s, float("-inf")), reverse=True)

        # Filter out highly correlated assets
        filtered_symbols = []
        for symbol in sorted_symbols:
            # If symbol not in correlation matrix, include it
            if symbol not in corr_matrix:
                filtered_symbols.append(symbol)
                continue

            # Check if this symbol is highly correlated with any already selected symbol
            should_include = True
            for selected in filtered_symbols:
                if selected in corr_matrix.get(symbol, {}):
                    correlation = abs(corr_matrix[symbol][selected])
                    if correlation > self.correlation_threshold:
                        should_include = False
                        self.debug(f"Excluding {symbol} due to high correlation ({correlation:.2f}) with {selected}")
                        break

            if should_include:
                filtered_symbols.append(symbol)

            # Stop once we have enough symbols
            if len(filtered_symbols) >= self.top_k:
                break

        return filtered_symbols

    def on_bar(self, tick: Any) -> None:
        # Update historical prices for volatility calculation
        self.update_historical_prices(tick)

        # Check stop-loss and take-profit conditions
        self.check_stop_loss_take_profit(tick)

        # Get the latest signals and apply smoothing
        last_signal: pl.DataFrame = self.get_signal(tick)
        if last_signal.is_empty():
            return

        # Apply signal smoothing
        smoothed_signal = self.smooth_signals(last_signal)

        # Sort by smoothed signal
        sorted_signal = smoothed_signal.sort("signal", descending=True)

        # Get position symbols and update holding days
        pos_symbols: list[str] = [pos.symbol for pos in self.ctx.get_open_positions() if pos]

        for symbol in pos_symbols:
            self.holding_days[symbol] += 1

        # Generate sell list
        active_symbols: set[str] = set(sorted_signal["symbol"][: self.top_k])  # Extract symbols with highest signals
        active_symbols.update(pos_symbols)  # Merge with currently held symbols
        active_df: pl.DataFrame = sorted_signal.filter(
            pl.col("symbol").is_in(active_symbols)
        )  # Filter signals for these symbols

        component_symbols: set[str] = set(sorted_signal["symbol"])  # Extract current index component symbols
        sell_symbols: set[str] = set(pos_symbols).difference(component_symbols)  # Sell positions not in components

        for symbol in active_df["symbol"][-self.n_drop :]:  # Iterate through lowest signal portion
            if symbol in pos_symbols:  # If the contract is in current positions
                # Only add to sell list if minimum holding period has passed
                if self.holding_days[symbol] >= self.min_days:
                    sell_symbols.add(symbol)  # Add it to sell list

        # Generate buy list (initial candidates)
        buyable_df: pl.DataFrame = sorted_signal.filter(
            ~pl.col("symbol").is_in(pos_symbols)
        )  # Filter contracts available for purchase

        buy_quantity: int = len(sell_symbols) + self.top_k - len(pos_symbols)  # Calculate number of contracts to buy
        initial_buy_candidates: list[str] = list(buyable_df[:buy_quantity]["symbol"])  # Select initial buy candidates

        # Apply correlation filtering to buy candidates
        buy_symbols = self.filter_by_correlation(initial_buy_candidates, sorted_signal)

        self.debug(f"sorted_signal: {sorted_signal}")
        self.debug(
            f"pos_symbols: {pos_symbols}; active_symbols: {active_symbols}; "
            f"sell_symbols: {sell_symbols}; buy_symbols: {buy_symbols}"
        )

        # Sell rebalancing
        for symbol in sell_symbols:
            # Apply minimum holding period check for all sell conditions
            if self.holding_days[symbol] < self.min_days:
                continue

            sell_bar: BarData | None = self.ctx.get_bar_data(symbol=symbol, exchange=self.exchange)
            if not sell_bar:
                continue

            self.set_target(symbol=symbol, exchange=self.exchange, target=0)  # Set target volume to 0

        # Buy rebalancing with volatility-based position sizing
        if buy_symbols:
            # Calculate total buy value with cash ratio
            total_buy_value: float = self.available_balance * self.cash_ratio

            # Calculate position sizes based on volatility
            position_sizes = self.calculate_position_sizes(buy_symbols, total_buy_value)

            for symbol in buy_symbols:
                bar: BarData | None = self.ctx.get_bar_data(symbol=symbol, exchange=self.exchange)
                if not bar or bar.close <= 0:
                    continue

                buy_price: float = bar.close
                buy_value = position_sizes.get(symbol, total_buy_value / len(buy_symbols))
                buy_volume: float = round(buy_value / buy_price, 6)

                # Ensure we don't exceed max risk per trade
                max_position_value = self.available_balance * self.max_risk_per_trade / self.stop_loss_pct
                if buy_value > max_position_value:
                    buy_volume = round(max_position_value / buy_price, 6)
                    self.debug(f"Limiting position size for {symbol} due to risk management")

                self.set_target(symbol=symbol, exchange=self.exchange, target=buy_volume)

        # Execute trading with dynamic price adjustment based on volatility
        price_adj_factor = self.price_add
        self.execute_trading(price_add=price_adj_factor)
