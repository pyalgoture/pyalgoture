from collections import OrderedDict
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from ..datafeed import DataFeed
from ..utils.logger import get_logger
from ..utils.objects import AccountData, PositionData
from .agents import AgentState, BaseAgent
from .agents.liquidity_management import LiquidityManagementAgent
from .agents.market_data import MarketDataAgent
from .agents.portfolio_management import PortfolioManagementAgent
from .agents.price_pattern import PricePatternAgent, PricePatternLLMAgent
from .agents.quant import QuantAgent
from .agents.risk_management import RiskManagementAgent
from .agents.sentiment import SentimentAgent
from .agents.stat_arb import StatArbAgent
from .agents.valuation import ValuationAgent
from .utils.utility import save_graph_as_png

logger = get_logger(__name__)


class Agent:
    def __init__(
        self,
        mode: str,
        symbol: str,
        exchange: str,
        asset_type: str,
        datafeed: DataFeed,
        interval: str,
        sub_intervals: list[str],
        show_reasoning: bool = False,
        agent_graph_path: str | None = None,
        model_name: str = "deepseek-chat",
        model_provider: str = "DeepSeek",
        model_api_key: str | None = None,
        model_base_url: str | None = None,
        coingecko_api_key: str | None = None,
        coingecko_is_demo: bool | None = None,
        twitter_bearer_token: str | None = None,
    ):
        if model_provider not in ["OpenAI", "DeepSeek"]:
            raise ValueError("Invalid model provider. Please choose from 'OpenAI' or 'DeepSeek'.")

        self.mode = mode
        self.symbol = symbol
        self.exchange = exchange
        self.asset_type = asset_type
        self.datafeed = datafeed
        self.interval = interval
        self.sub_intervals = sub_intervals

        self.show_reasoning = show_reasoning
        self.agent_graph_path = agent_graph_path
        self.model_name = model_name
        self.model_provider = model_provider
        self.model_api_key = model_api_key
        self.model_base_url = model_base_url
        self.coingecko_api_key = coingecko_api_key
        self.coingecko_is_demo = coingecko_is_demo
        self.twitter_bearer_token = twitter_bearer_token

        self.agents: OrderedDict[str, dict[str, Any]] = self.get_agents(self.mode)

    def run(self, start_date: str, end_date: str, position: PositionData, account: AccountData) -> dict[str, Any]:
        """
        Executes the trading workflow using the specified configuration.
        Parameters:
            start_date (str): The start date for historical data used in the workflow.
            end_date (str): The end date for historical data used in the workflow.
            position (PositionData): The initial state of the position.
            account (AccountData): The initial state of the account.
        Returns:
            dict: A dictionary containing the decision and analysis.
        """
        ### TODO: add pass memeory of trading decision

        # Create a new workflow if analysts are customized
        workflow = self.create_workflow()
        agent = workflow.compile()

        if self.agent_graph_path:
            if not self.agent_graph_path.endswith(".png") and not self.agent_graph_path.endswith(".jpg"):
                self.agent_graph_path += ".png"
            save_graph_as_png(agent, self.agent_graph_path)

        final_state = agent.invoke(
            {
                "data": {
                    "symbol": self.symbol,
                    "exchange": self.exchange,
                    "asset_type": self.asset_type,
                    "datafeed": self.datafeed,
                    "interval": self.interval,
                    "sub_intervals": self.sub_intervals,
                    "start_date": start_date,
                    "end_date": end_date,
                    "position": position,
                    "account": account,
                },
                "analysis": {
                    "market_data": {},
                    "market_scan": [],
                    "pattern_analysis": {},
                    "quant_analysis": {},
                    "sentiment_analysis": {},
                    "arb_analysis": {},
                    "valuation": {},
                    "liquidity": {},
                    "risk": {},
                },
                "decision": {},
                "metadata": {
                    "mode": self.mode,
                    "show_reasoning": self.show_reasoning,
                    "model_name": self.model_name,
                    "model_provider": self.model_provider,
                    "model_api_key": self.model_api_key,
                    "model_base_url": self.model_base_url,
                    "coingecko_api_key": self.coingecko_api_key,
                    "coingecko_is_demo": self.coingecko_is_demo,
                    "twitter_bearer_token": self.twitter_bearer_token,
                },
            },
        )
        # logger.debug(f"the final state: {final_state}")
        return {
            "decision": final_state["decision"],
            "analysis": final_state["analysis"],
        }

    def register_agent(self, agent_name: str, agent_class: BaseAgent, order: int = -1, **kwargs) -> None:
        """
        Register a new agent to the agent collection.

        Args:
            agent_name: Name of the agent to register
            agent_class: The agent class to register
            order: Position to insert the agent (-1 for append at end)
            **kwargs: Additional parameters to pass to the agent when instantiating
        """
        agent_info = {"class": agent_class, "params": kwargs}

        if order == -1:
            # Append at the end
            self.agents[agent_name] = agent_info
        else:
            # Insert at specific position
            items = list(self.agents.items())
            items.insert(order, (agent_name, agent_info))
            self.agents.clear()
            self.agents.update(items)

    def unregister_agent(self, agent_name: str) -> bool:
        """
        Unregister an agent from the agent collection.

        Args:
            agent_name: Name of the agent to unregister

        Returns:
            bool: True if agent was successfully removed, False if agent was not found
        """
        if agent_name in self.agents:
            del self.agents[agent_name]
            return True
        return False

    def unregister_all_agents(self) -> None:
        """
        Unregister all agents from the agent collection.
        """
        self.agents.clear()

    def get_agents(self, mode: str) -> OrderedDict[str, dict[str, Any]]:
        agents_dict = {
            # "market_scan": MarketDataAgent,
            "price_pattern": PricePatternAgent,
            "quant": QuantAgent,
            "sentiment": SentimentAgent,
            "stat_arb": StatArbAgent,
            "valuation": ValuationAgent,
            "liquidity": LiquidityManagementAgent,
            # "risk": RiskManagementAgent,
            # "portfolio": PortfolioManagementAgent,
        }

        # Filter out live_only agents when in backtest mode
        filtered_agents = {}
        for agent_name, agent_class in agents_dict.items():
            if not (agent_class.live_only and mode == "backtest"):
                filtered_agents[agent_name] = {"class": agent_class, "params": {}}

        # Convert to OrderedDict
        agents = OrderedDict(filtered_agents)

        print(agents)
        return agents

    def create_workflow(self) -> StateGraph:
        """Create the workflow with Strategy."""
        workflow = StateGraph(AgentState)

        workflow.add_node("market_scan", MarketDataAgent())
        for agent_name, agent_info in self.agents.items():
            agent_class = agent_info["class"]
            agent_kwargs = agent_info["params"]
            workflow.add_node(agent_name, agent_class(**agent_kwargs))
            workflow.add_edge("market_scan", agent_name)

        workflow.add_node("risk", RiskManagementAgent())
        workflow.add_node("portfolio", PortfolioManagementAgent())

        # Sequential workflow
        for agent_name in self.agents.keys():
            workflow.add_edge(agent_name, "risk")

        workflow.add_edge("risk", "portfolio")
        workflow.add_edge("portfolio", END)

        workflow.set_entry_point("market_scan")
        return workflow
