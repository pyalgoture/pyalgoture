import numpy as np

from ...utils.models import CommonResponseSchema
from ..agents import AgentState, BaseAgent, logger, show_agent_reasoning


class StatArbAgent(BaseAgent):
    live_only: bool = True

    def __call__(self, state: AgentState) -> CommonResponseSchema:
        market_data = state["analysis"]["market_data"]
        mode = state["metadata"]["mode"]
        try:
            if self.live_only and mode == "backtest":
                return CommonResponseSchema(
                    success=False,
                    error=True,
                    data={},
                    msg="Stat arb is not available in backtest mode",
                )
            results = {}
            # Require at least two tickers for arbitrage
            tickers = [symbol for symbol in market_data if market_data[symbol].get("status") == "success"]
            if len(tickers) < 2:
                logger.warning(f"Insufficient tickers for arbitrage: {tickers}")
                return {"status": "error", "analysis": "Need at least two tickers"}

            # Simple pair trading example
            for i, ticker1 in enumerate(tickers):
                for ticker2 in tickers[i + 1 :]:
                    data1 = market_data[ticker1]
                    data2 = market_data[ticker2]
                    if not (data1.get("ohlcv") and data2.get("ohlcv")):
                        logger.warning(f"Missing OHLCV for {ticker1} or {ticker2}")
                        continue
                    closes1 = [candle[4] for candle in data1["ohlcv"]]
                    closes2 = [candle[4] for candle in data2["ohlcv"]]
                    if len(closes1) < 20 or len(closes2) < 20:
                        logger.warning(f"Insufficient OHLCV data for {ticker1}/{ticker2}")
                        continue
                    spread = np.array(closes1[-20:]) - np.array(closes2[-20:])
                    mean_spread = np.mean(spread)
                    std_spread = np.std(spread)
                    current_spread = closes1[-1] - closes2[-1]
                    z_score = (current_spread - mean_spread) / (std_spread + 1e-6)
                    results[f"{ticker1}-{ticker2}"] = {
                        "z_score": z_score,
                        "signal": "buy" if z_score < -2 else "sell" if z_score > 2 else "hold",
                    }
            if not results:
                logger.warning("No valid arbitrage pairs found")
                return {"status": "error", "analysis": "No valid pairs"}
            logger.info(f"Arbitrage results: {results}")
            return {"status": "success", "analysis": results}
        except Exception as e:
            logger.error(f"Stat arb error: {str(e)}")
            return {"status": "error", "analysis": str(e)}
