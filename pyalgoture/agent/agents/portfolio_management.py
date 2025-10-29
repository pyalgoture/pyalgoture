import json
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate

from ...utils.models import CommonResponseSchema
from ..agents import AgentState, BaseAgent, logger, show_agent_reasoning
from ..llm import call_llm


class PortfolioManagementAgent(BaseAgent):
    def __call__(self, state: AgentState) -> CommonResponseSchema:
        """Makes final trading decisions and generates orders for multiple tickers"""

        # Get the portfolio and analyst signals
        symbol = state["data"]["symbol"]
        position = state["data"]["position"]
        account = state["data"]["account"]
        signals = state["analysis"]
        risk_data = state["analysis"]["risk"]

        # Get position limits and current prices for the symbol
        remaining_position_limit = risk_data.get("remaining_position_limit", 0.0)
        stop_price = risk_data.get("stop_price", 0.0)
        volatility = risk_data.get("volatility", 0.0)
        leverage = risk_data.get("leverage", 0.0)
        current_price = risk_data.get("current_price", 0.0)

        # Calculate maximum shares allowed based on position limit and price
        if current_price > 0.0:
            max_shares = float(remaining_position_limit / current_price)
        else:
            max_shares = 0.0

        signals.pop("market_data")
        signals = {k: v for k, v in signals.items() if v}

        # Generate the trading decision
        result = self.generate_trading_decision(
            symbol=symbol,
            signals=signals,
            current_price=current_price,
            max_shares=max_shares,
            stop_price=stop_price,
            leverage=leverage,
            volatility=volatility,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
            model_api_key=state["metadata"]["model_api_key"],
            model_base_url=state["metadata"]["model_base_url"],
        )

        # Print the decision if the flag is set
        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(result.get("decision", {}), "Portfolio Management Agent")

        state["decision"] = result.get("decision", {})

        # return CommonResponseSchema(
        #     success=True,
        #     error=False,
        #     data=result.get("decision", {}),
        #     msg="Portfolio management completed",
        # )

        # The final stage must return the state, otherwise the agent will not be able to access the state["decision"]
        return state

    @staticmethod
    def generate_trading_decision(
        symbol: str,
        signals: dict[str, dict[str, Any]],
        current_price: float,
        max_shares: float,
        stop_price: float,
        leverage: float,
        volatility: float,
        model_name: str,
        model_provider: str,
        model_api_key: str,
        model_base_url: str,
    ) -> dict:
        """Attempts to get a decision from the LLM with retry logic"""
        # Create the prompt template
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a portfolio manager making final trading decisions based on a single symbol.

                    Trading Rules:
                    - For long positions:
                    * Only buy if you have available cash
                    * Only sell if you currently hold long shares of that symbol
                    * Sell quantity must be ≤ current long position shares
                    * Buy quantity must be ≤ max_shares for that symbol

                    - For short positions:
                    * Only short if you have available margin (position value × margin requirement)
                    * Only cover if you currently have short shares of that symbol
                    * Cover quantity must be ≤ current short position shares
                    * Short quantity must respect margin requirements

                    - The max_shares values are pre-calculated to respect position limits
                    - Consider both long and short opportunities based on signals
                    - Maintain appropriate risk management with both long and short exposure

                    Available Actions:
                    - "buy": Open or add to long position
                    - "sell": Close or reduce long position
                    - "short": Open or add to short position
                    - "cover": Close or reduce short position
                    - "hold": No action

                    Inputs:
                    - symbol: The trading symbol being analyzed
                    - signals: dictionary of trading signals for the symbol
                    - current_price: current price of the symbol
                    - max_shares: maximum shares allowed for the symbol
                    - stop_price: suggested stop price for risk management
                    - leverage: current leverage being used (e.g., 2.0 means 2x)
                    - volatility: current volatility measure for the symbol
                    """,
                ),
                (
                    "human",
                    """Based on the team's analysis, make your trading decision for {symbol} using our long-short strategy.

                    Here are the signals by symbol:
                    {signals}

                    Current Prices:
                    {current_price}

                    Maximum Shares Allowed For Purchases:
                    {max_shares}

                    Stop Price: {stop_price}
                    Current Leverage: {leverage}x
                    Current Volatility: {volatility}

                    Consider the stop price for risk management and use volatility to inform position sizing.
                    The leverage parameter indicates your available leverage multiplier.

                    Output strictly in JSON with the following structure:
                    {{
                        "decision": {{
                            "action": "buy/sell/short/cover/hold",
                            "quantity": float,
                            "confidence": float between 0 and 100,
                            "reasoning": "string"
                        }}
                    }}
                    """,
                ),
            ]
        )

        prompt = template.invoke(
            {
                "symbol": symbol,
                "signals": json.dumps(signals, indent=2),
                "current_price": current_price,
                "max_shares": max_shares,
                "stop_price": stop_price,
                "leverage": leverage,
                "volatility": volatility,
            }
        )
        for message in prompt.messages:
            logger.debug(f"Role: {message.type} - Content: {message.content}")

        return call_llm(
            prompt=prompt,
            model_name=model_name,
            model_provider=model_provider,
            api_key=model_api_key,
            base_url=model_base_url,
        )
