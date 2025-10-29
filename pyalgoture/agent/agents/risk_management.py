from typing import Any, Optional

import numpy as np

from ...utils.models import CommonResponseSchema
from ..agents import AgentState, BaseAgent, logger, show_agent_reasoning


class RiskManagementAgent(BaseAgent):
    def __call__(self, state: AgentState) -> CommonResponseSchema:
        """Controls position sizing based on real-world risk factors for multiple tickers."""
        # TODO: add stats of trading performance

        symbol = state["data"]["symbol"]
        price_df = state["analysis"]["market_data"]["ohlcv"]
        position = state["data"]["position"]
        account = state["data"]["account"]

        closes = price_df["close"].iloc[-20:]
        volatility = np.std(closes) / np.mean(closes)

        # Calculate position value
        current_price = price_df["close"].iloc[-1]

        # Calculate current position value for this symbol
        current_position_value = position["notional_value"]

        # Calculate total position value using stored prices
        total_portfolio_value = position["unrealized_pnl"] + account.available

        # Base limit is 90% of position for any single position
        position_limit = total_portfolio_value * position.leverage * 0.90

        # For existing positions, subtract current position value from limit
        remaining_position_limit = position_limit - current_position_value

        # Ensure we don't exceed available cash
        max_position_size = min(remaining_position_limit, account.available)

        risk_analysis = {
            "remaining_position_limit": float(max_position_size),
            "volatility": float(volatility),
            "leverage": float(position.leverage),
            "stop_price": float(current_price * 0.95) if position["size"] > 0.0 else float(current_price * 1.05),
            "current_price": float(current_price),
            "reasoning": {
                "portfolio_value": float(total_portfolio_value),
                "current_position": float(current_position_value),
                "position_limit": float(position_limit),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(account.available),
            },
        }
        # logger.debug(f"Risk analysis: {risk_analysis}")

        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(risk_analysis, "Risk Management Agent")

        # Add the signal to the analyst_signals list
        state["analysis"]["risk"] = risk_analysis

        return CommonResponseSchema(
            success=True,
            error=False,
            data=risk_analysis,
            msg="Risk management completed",
        )

        # return state
