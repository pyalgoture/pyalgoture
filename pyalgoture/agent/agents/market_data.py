import os

import pandas as pd
from pycoingecko import CoinGeckoAPI  # type: ignore

from ...datafeed import DataFeed
from ...utils.models import CommonResponseSchema
from ...utils.objects import TickData
from ...utils.util_dt import tz_manager
from . import AgentState, BaseAgent, logger, show_agent_reasoning


class MarketDataAgent(BaseAgent):
    def __init__(self) -> None:
        self.cg: CoinGeckoAPI = CoinGeckoAPI()

    def __call__(self, state: AgentState) -> CommonResponseSchema:
        if state["metadata"].get("coingecko_api_key"):
            self.cg = CoinGeckoAPI(
                **{
                    "demo_api_key" if state["metadata"]["coingecko_is_demo"] else "api_key": state["metadata"][
                        "coingecko_api_key"
                    ]
                }
            )
        symbol = state["data"]["symbol"]
        datafeed = state["data"]["datafeed"]
        start_date = state["data"]["start_date"]
        end_date = state["data"]["end_date"]

        market_data = state["analysis"]["market_data"]

        mode = state["metadata"]["mode"]
        try:
            if mode == "live":
                result = self.fetch_order_book(symbol, datafeed)
                if result:
                    market_data["order_book"] = result
                else:
                    market_data["order_book"] = {}

                result = self.scan_topk_coins(topk=5)
                if result:
                    market_data["topk_coins"] = result
                else:
                    market_data["topk_coins"] = {}

                result = self.scan_meme_coins()
                if result:
                    market_data["meme_coins"] = result
                else:
                    market_data["meme_coins"] = {}

            result = self.fetch_ohlcv(symbol, start_date, end_date, datafeed)
            if not result.empty:
                market_data["ohlcv"] = result
            else:
                market_data["ohlcv"] = {}

            # if state["metadata"]["show_reasoning"]:
            #     show_agent_reasoning(market_data, "Market Data")

            return CommonResponseSchema(success=True, error=False, data=market_data, msg="Market data fetched")
        except Exception as e:
            logger.exception(f"Error fetching data for {symbol}: {str(e)}")
            return CommonResponseSchema(
                success=False, error=True, data={}, msg=f"Error fetching data for {symbol}: {str(e)}"
            )

    def fetch_order_book(self, symbol: str, datafeed: DataFeed) -> TickData | None:
        """Fetch order book for a symbol."""
        return datafeed.fetch_tick(code=symbol)

    def fetch_ohlcv(self, symbol: str, start_date: str, end_date: str, datafeed: DataFeed) -> pd.DataFrame:
        """Fetch OHLCV and order book for a symbol."""
        try:
            ohlcv = datafeed.fetch_hist(
                start=start_date,
                end=end_date,
                is_store=True,
            )
            df = pd.DataFrame.from_records([v.to_dict(return_obj=False, return_aio=False) for v in ohlcv.values()])
            df["datetime"] = pd.to_datetime(df["datetime"].astype(str))
            df.set_index("datetime", inplace=True)
            df.index = pd.to_datetime(df.index).tz_convert(tz_manager.tz)

            # logger.debug(f"Fetched {len(df)} OHLCV candles for {symbol} from {start_date} to {end_date}")
            return df
        except Exception as e:
            logger.exception(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def scan_meme_coins(self) -> list[dict]:
        """
        Scan for newly listed meme coins.
        Using CoinGecko
        """
        try:
            coins = self.cg.get_coins_markets(vs_currency="usd", category="meme-token", per_page=50)
            new_coins = []
            for coin in coins:
                categories = coin.get("categories", []) or []
                # Relaxed filtering to include any coin in meme-token category
                if coin.get("market_cap", 0) > 0:
                    new_coins.append(
                        {
                            "symbol": str(coin["symbol"]).upper() + "/USDT",
                            "name": coin["name"],
                            "price": coin["current_price"],
                            "volume": coin["total_volume"],
                            "market_cap": coin["market_cap"],
                            "listed_date": coin.get("last_updated", ""),
                        }
                    )

            logger.info(f"Found {len(new_coins)} meme coins")
            return sorted(new_coins, key=lambda x: x["market_cap"], reverse=True)[:5]
        except Exception as e:
            logger.error(f"Error retrieving meme coins: {str(e)}")
            return []

    def scan_topk_coins(self, topk: int = 5) -> list[dict]:
        """
        Scan for newly listed meme coins.
        Using CoinGecko
        """
        try:
            coins = self.cg.get_coins_markets(vs_currency="usd", per_page=topk)
            new_coins = []
            for coin in coins:
                if coin.get("market_cap", 0) > 0:
                    new_coins.append(
                        {
                            "symbol": str(coin["symbol"]).upper() + "/USDT",
                            "name": coin["name"],
                            "price": coin["current_price"],
                            "volume": coin["total_volume"],
                            "market_cap": coin["market_cap"],
                            "listed_date": coin.get("last_updated", ""),
                        }
                    )

            logger.info(f"Found {len(new_coins)} meme coins")
            return sorted(new_coins, key=lambda x: x["market_cap"], reverse=True)[:5]
        except Exception as e:
            logger.error(f"Error retrieving meme coins: {str(e)}")
            return []
