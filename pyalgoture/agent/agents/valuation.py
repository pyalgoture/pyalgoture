from ...utils.models import CommonResponseSchema
from ..agents import AgentState, BaseAgent, logger, show_agent_reasoning


class ValuationAgent(BaseAgent):
    live_only: bool = True

    def __call__(self, state: AgentState) -> CommonResponseSchema:
        meme_coins = state["analysis"]["market_data"]["meme_coins"]
        mode = state["metadata"]["mode"]
        try:
            if self.live_only and mode == "backtest":
                return CommonResponseSchema(
                    success=False,
                    error=True,
                    data={},
                    msg="Valuation is not available in backtest mode",
                )
            results = {}
            for coin in meme_coins:
                symbol = coin["symbol"]
                volume = coin["volume"]
                market_cap = coin["market_cap"]

                # Volume-to-market-cap ratio
                vol_mcap_ratio = volume / market_cap if market_cap > 0 else 0
                valuation = "undervalued" if vol_mcap_ratio > 0.1 else "overvalued" if vol_mcap_ratio < 0.05 else "fair"

                results[symbol] = {"valuation": valuation, "vol_mcap_ratio": vol_mcap_ratio}
                logger.info(f"Valuation {symbol}: {valuation}, Ratio={vol_mcap_ratio:.4f}")

            state["analysis"]["valuation"] = results
            if state["metadata"]["show_reasoning"]:
                show_agent_reasoning(results, "Valuation")

            return CommonResponseSchema(
                success=True,
                error=False,
                data=results,
                msg="Valuation completed",
            )
        except Exception as e:
            logger.exception(f"Valuation error: {str(e)}")
            return CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg=f"Valuation error: {str(e)}",
            )
