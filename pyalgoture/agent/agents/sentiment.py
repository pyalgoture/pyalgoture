from langchain.prompts import PromptTemplate

from ...utils.models import CommonResponseSchema
from ..agents import AgentState, BaseAgent, logger, show_agent_reasoning

try:
    from ..utils.sentiment_tools import scrape_twitter
except ImportError:
    scrape_twitter = None  # type: ignore


# messages=[
#     {"role": "system",
#         "content": "You are a sentiment analyst for financial markets."},
#     {"role": "user", "content": prompt}
# ],
class SentimentAgent(BaseAgent):
    live_only: bool = True

    def __init__(self) -> None:
        self.prompt = PromptTemplate(
            input_variables=["symbol", "tweets"],
            template="Analyze sentiment for {symbol} based on tweets: {tweets}. Provide a sentiment score (0-100, 100=very bullish).",
        )
        self.llm = None

    def __call__(self, state: AgentState) -> CommonResponseSchema:
        symbol = state["data"]["symbol"]
        mode = state["metadata"]["mode"]
        twitter_bearer_token = state["metadata"]["twitter_bearer_token"]
        try:
            if self.live_only and mode == "backtest":
                return CommonResponseSchema(
                    success=False,
                    error=True,
                    data={},
                    msg="Sentiment analysis is not available in backtest mode",
                )
            if scrape_twitter is None:
                return {
                    "symbol": symbol,
                    "sentiment_score": 50.0,
                    "analysis": "Twitter scraping not available",
                    "status": "error",
                }

            tweets = scrape_twitter(symbol, twitter_bearer_token, max_tweets=10)
            if self.llm:
                analysis = self.llm.generate(self.prompt.format(symbol=symbol, tweets=tweets))
                try:
                    score = float(analysis.split("Score: ")[1].split()[0])
                except:
                    score = 50.0
            else:
                analysis = f"Simulated sentiment for {symbol}: {tweets}"
                score = 50.0

            logger.info(f"Sentiment score for {symbol}: {score}")
            state["analysis"]["sentiment_analysis"] = {"sentiment_score": score, "analysis": analysis, "tweets": tweets}
            if state["metadata"]["show_reasoning"]:
                show_agent_reasoning(state["analysis"]["sentiment_analysis"], "Sentiment Analysis")

            return CommonResponseSchema(
                success=True,
                error=False,
                data={"sentiment_score": score, "analysis": analysis, "tweets": tweets},
                msg=f"Sentiment analysis completed for {symbol}",
            )
        except Exception as e:
            logger.error(f"Sentiment analysis error for {symbol}: {str(e)}")
            return CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg=f"Sentiment analysis error for {symbol}: {str(e)}",
            )
