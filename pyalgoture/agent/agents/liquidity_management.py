from ...utils.models import CommonResponseSchema
from ...utils.objects import TickData
from ..agents import AgentState, BaseAgent, logger, show_agent_reasoning


class LiquidityManagementAgent(BaseAgent):
    live_only: bool = True

    def __call__(self, state: AgentState) -> CommonResponseSchema:
        symbol = state["data"]["symbol"]
        tick: TickData = state["analysis"]["market_data"]["order_book"]
        mode = state["metadata"]["mode"]
        try:
            if self.live_only and mode == "backtest":
                return CommonResponseSchema(
                    success=False,
                    error=True,
                    data={},
                    msg="Liquidity management is not available in backtest mode",
                )

            bid_price = tick["bid_price_1"]
            ask_price = tick["ask_price_1"]
            spread = (ask_price - bid_price) / bid_price
            # depth = sum(bid[1] for bid in bids) + sum(ask[1] for ask in asks)
            depth = tick["bid_volume_1"] + tick["ask_volume_1"]

            # Adjust spread if too wide
            target_spread = 0.01  # 1%
            if spread > target_spread:
                new_bid = bid_price * 1.005
                new_ask = ask_price * 0.995
            else:
                new_bid, new_ask = bid_price, ask_price

            results = {"bid": new_bid, "ask": new_ask, "spread": spread, "depth": depth}
            state["analysis"]["liquidity_management"] = results
            if state["metadata"]["show_reasoning"]:
                show_agent_reasoning(state["analysis"]["liquidity_management"], "Liquidity Management")

            logger.info(f"Liquidity {symbol}: Spread={spread:.4f}, Depth={depth:.2f}")

            return CommonResponseSchema(
                success=True,
                error=False,
                data=results,
                msg="Liquidity management completed",
            )
        except Exception as e:
            logger.exception(f"Liquidity error: {str(e)}")
            return CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg=f"Liquidity error: {str(e)}",
            )
