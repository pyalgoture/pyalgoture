# PyAlgoture

A comprehensive algorithmic trading platform for cryptocurrency and stock markets with backtesting and real-time trading capabilities.

**Developed by Algoture**

## Features

- **Backtesting**: Test your trading strategies on historical data
- **Real-time Trading**: Execute strategies in live markets
- **Multiple Exchanges**: Support for Binance, Bybit, Bitget, OKX, and more
- **Data Feeds**: Crypto and stock data feed support
- **Portfolio Management**: Advanced portfolio management tools
- **Risk Management**: Built-in risk management agents
- **AI Agents**: LLM-powered trading agents for sentiment analysis, valuation, and more
- **Technical Indicators**: Comprehensive technical analysis tools

## Installation

```bash
pip install pyalgoture
```

## Quick Start

### Create a Strategy

```python
from pyalgoture import Strategy, BackTest, BacktestBroker, CryptoDataFeed
from datetime import datetime

class MyStrategy(Strategy):
    def on_init(self):
        # Initialize indicators and parameters
        pass
    
    def on_bar(self, bar):
        # Your trading logic here
        pass

# Create a data feed
datafeed = CryptoDataFeed(
    code="BTCUSDT",
    exchange="binance",
    asset_type="SPOT",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    interval="1d"
)

# Create a broker
broker = BacktestBroker(
    balance=10000,
    trade_at="close",
    slippage=0.001
)

# Run backtest
backtest = BackTest(
    feeds=[datafeed],
    StrategyClass=MyStrategy,
    broker=broker
)

backtest.start()
```

## Main Classes

### Core Classes

- **`Strategy`** - Base class for implementing trading strategies
- **`Orchestrator`** - Main orchestrator for coordinating data feeds, strategies, and brokers
- **`Context`** - Stores shared data and common functions for trading operations
- **`Broker`** - Base broker class for handling trading operations
- **`DataFeed`** - Base class for market data feeds

### Orchestrators

- **`BackTest`** - Backtesting orchestrator for testing strategies on historical data
- **`RealTime`** - Real-time trading orchestrator for live market execution
- **`BackTestRapid`** - Fast backtesting orchestrator

### Brokers

- **`BacktestBroker`** - Broker for backtesting scenarios
- **`RealtimeBroker`** - Broker for live trading

### Data Feeds

- **`CryptoDataFeed`** - Cryptocurrency market data feed
- **`StockDataFeed`** - Stock market data feed
- **`CustomDataFeed`** - Custom data feed implementation

### Exchange Clients

**Binance:**
- `BinanceClient` - REST client for Binance
- `BinanceDataWebsocket` - WebSocket client for market data
- `BinanceTradeWebsocket` - WebSocket client for trading

**Bybit:**
- `BybitClient` - REST client for Bybit
- `BybitDataWebsocket` - WebSocket client for market data
- `BybitTradeWebsocket` - WebSocket client for trading

**Bitget:**
- `BitgetClient` - REST client for Bitget
- `BitgetDataWebsocket` - WebSocket client for market data
- `BitgetTradeWebsocket` - WebSocket client for trading

**OKX:**
- `OKXClient` - REST client for OKX
- `OKXDataWebsocket` - WebSocket client for market data
- `OKXTradeWebsocket` - WebSocket client for trading

### Utilities

- **`Indicator`** - Technical indicator calculations
- **`BarGenerator`** - Generate OHLCV bars from tick data
- **`BarManager`** - Manage and aggregate bar data
- **`Patterns`** - Pattern recognition indicators
- **`EventEngine`** - Event-driven architecture engine
- **`FundTracker`** - Track fund performance and NAV
- **`PortfolioManager`** - Manage portfolio allocations
- **`SignalMQ`** - Message queue for trading signals
- **`NotificationSender`** - Send notifications for trading events

### Agent Classes (AI Trading Agents)

- **`Agent`** - Main agent orchestrator
- **`BaseAgent`** - Base class for all trading agents
- **`MarketDataAgent`** - Market data analysis agent
- **`SentimentAgent`** - Sentiment analysis agent
- **`ValuationAgent`** - Asset valuation agent
- **`RiskManagementAgent`** - Risk management agent
- **`PortfolioManagementAgent`** - Portfolio management agent
- **`PricePatternAgent`** - Price pattern recognition agent
- **`QuantAgent`** - Quantitative analysis agent
- **`StatArbAgent`** - Statistical arbitrage agent
- **`LiquidityManagementAgent`** - Liquidity management agent

### Alpha Research (Machine Learning)

- **`AlphaDataset`** - Dataset for alpha research
- **`AlphaModel`** - Base class for alpha models
- **`AlphaStrategy`** - Strategy class for alpha research
- **`AlphaLab`** - Lab environment for alpha research
- **`LgbModel`** - LightGBM model implementation
- **`MlpModel`** - Multi-layer perceptron model
- **`LassoModel`** - Lasso regression model
- **`DoubleEnsembleModel`** - Double ensemble model

### Data Objects

- **`OrderData`** - Order information
- **`TradeData`** - Trade execution information
- **`PositionData`** - Position information
- **`AccountData`** - Account information
- **`BarData`** - OHLCV bar data
- **`TickData`** - Tick data
- **`InstrumentData`** - Instrument information

### Enums

- **`Side`** - Buy/Sell side
- **`OrderType`** - Order types (Limit, Market, etc.)
- **`Status`** - Order/trade status
- **`Exchange`** - Supported exchanges
- **`AssetType`** - Asset types (Spot, Future, Option, etc.)
- **`PositionSide`** - Position sides (Long, Short)
- **`TimeInForce`** - Order time in force

### Database Classes

- **`FilesystemDatabase`** - File-based database
- **`MysqlDatabase`** - MySQL database (lazy import)
- **`DuckdbDatabase`** - DuckDB database (lazy import)
- **`SqliteDatabase`** - SQLite database (lazy import)

### Statistics

- **`Statistic`** - Statistical analysis and performance metrics

## Documentation

For detailed documentation and API reference, please visit the [project repository](https://github.com/yourusername/pyalgoture).

## Requirements

- Python >= 3.8
- See `pyproject.toml` for full list of dependencies

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on [GitHub](https://github.com/yourusername/pyalgoture/issues).

---

**Algoture** - Advanced Algorithmic Trading Solutions
