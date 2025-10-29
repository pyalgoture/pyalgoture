# PyAlgoture

A comprehensive algorithmic trading platform for cryptocurrency and stock markets with backtesting and real-time trading capabilities.

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

## Documentation

For detailed documentation, please visit the [project repository](https://github.com/yourusername/pyalgoture).

## Requirements

- Python >= 3.8
- See `pyproject.toml` for full list of dependencies

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on [GitHub](https://github.com/yourusername/pyalgoture/issues).

