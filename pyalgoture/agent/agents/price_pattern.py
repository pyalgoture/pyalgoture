import json
from typing import Any, Literal

import numpy as np
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from ...utils.models import CommonResponseSchema
from ..agents import AgentState, BaseAgent, logger, show_agent_reasoning
from ..llm import call_llm
from ..utils.technical_indicators import (
    calculate_mean_reversion_signals,
    calculate_momentum_signals,
    calculate_stat_arb_signals,
    calculate_technical_indicators,
    calculate_trend_signals,
    calculate_volatility_signals,
    normalize_pandas,
    weighted_signal_combination,
)


class PricePatternLLMAgent(BaseAgent):
    def __init__(self) -> None:
        # self.prompt = PromptTemplate(
        #     input_variables=["symbol", "rsi", "sma", "bb_upper", "bb_lower", "bb_mid", "price_data"],
        #     template="Analyze price patterns for {symbol}. RSI: {rsi}. SMA: {sma}. "
        #     "Bollinger Bands (Upper: {bb_upper}, Lower: {bb_lower}, Middle: {bb_mid}). "
        #     "Recent prices: {price_data}. Suggest trading implications.",
        # )
        self.template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a Professional Trader with 10 years of experience using technical analysis to make investment decisions using your principles:

                1. Confluence Trading: Look for at least 3 technical indicators aligning before entering positions
                2. Trend Following: Trade with the primary trend, not against it
                3. Support and Resistance: Identify key levels and wait for clear breaks or bounces
                4. Volume Confirmation: High volume should confirm price movements for validity
                5. Risk-Reward Ratio: Maintain minimum 1:2 risk-reward ratio on all trades
                6. Market Structure: Analyze higher highs/lows and market phases (accumulation, markup, distribution, decline)
                7. Momentum Analysis: Use RSI, MACD, and Stochastic to gauge momentum shifts
                8. Price Action Priority: Candlestick patterns and price action take precedence over lagging indicators

                Your trading methodology combines classical technical analysis with modern risk management. You focus on probability-based decisions rather than predictions, always considering multiple scenarios and maintaining strict discipline in execution. You understand that markets are dynamic and adapt your analysis based on current market conditions, volatility, and sector rotation patterns.

                When analyzing any security, you systematically evaluate trend direction, momentum, support/resistance levels, volume patterns, and overall market sentiment to generate high-probability trading opportunities.

                """,
                ),
                (
                    "human",
                    """Based on the following analysis, create a professional trader investment signal.

                Analysis Data for {symbol}:
                {analysis_data}

                Return the trading signal in this JSON format:
                {{
                "signal": "bullish/bearish/neutral",
                "confidence": float (0-100),
                "reasoning": "string"
                }}
                """,
                ),
            ]
        )

    def __call__(self, state: AgentState) -> CommonResponseSchema:
        symbol = state["data"]["symbol"]
        market_data = state["analysis"]["market_data"]
        try:
            price_df = state["analysis"]["market_data"]["ohlcv"]
            if price_df.empty or len(price_df) < 15:
                logger.warning(f"Insufficient OHLCV data for {symbol}: {len(price_df)} candles")
                return CommonResponseSchema(
                    success=False,
                    error=True,
                    data={
                        "indicators": {
                            "rsi": "unavailable",
                            "sma": "unavailable",
                            "bb_upper": "unavailable",
                            "bb_lower": "unavailable",
                            "bb_mid": "unavailable",
                        }
                    },
                    msg="Insufficient data for technical analysis",
                )

            indicators = calculate_technical_indicators(price_df)

            # Format indicator values
            formatted_indicators = {
                "rsi": f"{indicators['rsi']:.2f}" if not np.isnan(indicators["rsi"]) else "unavailable",
                "sma": f"{indicators['sma']:.2f}" if not np.isnan(indicators["sma"]) else "unavailable",
                "bb_upper": f"{indicators['bb_upper']:.2f}" if not np.isnan(indicators["bb_upper"]) else "unavailable",
                "bb_lower": f"{indicators['bb_lower']:.2f}" if not np.isnan(indicators["bb_lower"]) else "unavailable",
                "bb_mid": f"{indicators['bb_mid']:.2f}" if not np.isnan(indicators["bb_mid"]) else "unavailable",
            }

            prompt = self.template.invoke(
                {"analysis_data": json.dumps(formatted_indicators, indent=2), "symbol": symbol}
            )

            analysis = call_llm(
                prompt=prompt,
                model_name=state["metadata"]["model_name"],
                model_provider=state["metadata"]["model_provider"],
                api_key=state["metadata"]["model_api_key"],
                base_url=state["metadata"]["model_base_url"],
            )

            if state["metadata"]["show_reasoning"]:
                show_agent_reasoning(analysis, "Price Pattern Analysis")

            state["analysis"]["pattern_analysis"] = analysis
            return CommonResponseSchema(
                success=True,
                error=False,
                data={"indicators": formatted_indicators, "analysis": analysis},
                msg="Price pattern analysis completed",
            )
        except Exception as e:
            logger.exception(f"Price pattern analysis error for {symbol}: {str(e)}")
            return CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg=f"Price pattern analysis error for {symbol}: {str(e)}",
            )


class PricePatternAgent(BaseAgent):
    def __call__(self, state: AgentState) -> CommonResponseSchema:
        """
        Sophisticated technical analysis system that combines multiple trading strategies for multiple tickers:
        1. Trend Following
        2. Mean Reversion
        3. Momentum
        4. Volatility Analysis
        5. Statistical Arbitrage Signals
        """

        symbol = state["data"]["symbol"]
        interval = state["data"]["interval"]
        sub_intervals = state["data"]["sub_intervals"]

        # Initialize analysis for each symbol
        technical_analysis = {}

        # Combine all signals using a weighted ensemble approach
        strategy_weights = {
            "trend": 0.25,
            "mean_reversion": 0.20,
            "momentum": 0.25,
            "volatility": 0.15,
            "stat_arb": 0.15,
        }

        # for interval in sub_intervals:
        price_df = state["analysis"]["market_data"]["ohlcv"]
        if price_df.empty:
            return CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg="No OHLCV data available",
            )

        trend_signals = calculate_trend_signals(price_df)
        mean_reversion_signals = calculate_mean_reversion_signals(price_df)
        momentum_signals = calculate_momentum_signals(price_df)

        volatility_signals = calculate_volatility_signals(price_df, interval)
        stat_arb_signals = calculate_stat_arb_signals(price_df)

        combined_signal = weighted_signal_combination(
            {
                "trend": trend_signals,
                "mean_reversion": mean_reversion_signals,
                "momentum": momentum_signals,
                "volatility": volatility_signals,
                "stat_arb": stat_arb_signals,
            },
            strategy_weights,
        )

        # Generate detailed analysis report for this symbol
        technical_analysis = {
            "signal": combined_signal["signal"],
            "confidence": round(combined_signal["confidence"] * 100),
            "reasoning": {
                "trend_following": {
                    "signal": trend_signals["signal"],
                    "confidence": round(trend_signals["confidence"] * 100),
                    "metrics": normalize_pandas(trend_signals["metrics"]),
                },
                "mean_reversion": {
                    "signal": mean_reversion_signals["signal"],
                    "confidence": round(mean_reversion_signals["confidence"] * 100),
                    "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                },
                "momentum": {
                    "signal": momentum_signals["signal"],
                    "confidence": round(momentum_signals["confidence"] * 100),
                    "metrics": normalize_pandas(momentum_signals["metrics"]),
                },
                "volatility": {
                    "signal": volatility_signals["signal"],
                    "confidence": round(volatility_signals["confidence"] * 100),
                    "metrics": normalize_pandas(volatility_signals["metrics"]),
                },
                "statistical_arbitrage": {
                    "signal": stat_arb_signals["signal"],
                    "confidence": round(stat_arb_signals["confidence"] * 100),
                    "metrics": normalize_pandas(stat_arb_signals["metrics"]),
                },
            },
        }

        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(technical_analysis, "Technical Analyst")

        # Add the signal to the analyst_signals list
        state["analysis"]["pattern_analysis"] = technical_analysis

        return CommonResponseSchema(
            success=True,
            error=False,
            data=technical_analysis,
            msg="Technical analysis completed",
        )
