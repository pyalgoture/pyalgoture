import tweepy  # type: ignore[import-untyped]

from ...utils.logger import get_logger

logger = get_logger(__name__)


def scrape_twitter(ticker: str, twitter_bearer_token: str, max_tweets: int = 10) -> list[str]:
    try:
        if not twitter_bearer_token:
            logger.warning(f"No Twitter API credentials for {ticker}, using placeholder")
            return [f"Sample tweet about {ticker} #{ticker}" for _ in range(max_tweets)]

        client = tweepy.Client(bearer_token=twitter_bearer_token)
        query = f"#{ticker.replace('/', '')} -is:retweet lang:en"
        tweets = client.search_recent_tweets(query=query, max_results=max_tweets)

        if not tweets.data:
            logger.info(f"No tweets found for {ticker}")
            return ["No recent tweets found"]

        result = [tweet.text for tweet in tweets.data]
        logger.info(f"Fetched {len(result)} tweets for {ticker}")
        return result
    except Exception as e:
        logger.error(f"Twitter scraping error for {ticker}: {str(e)}")
        return [f"Error fetching tweets: {str(e)}"]
