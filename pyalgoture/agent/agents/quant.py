from ...utils.models import CommonResponseSchema
from ..agents import AgentState, BaseAgent, logger, show_agent_reasoning
from ..utils.technical_indicators import calculate_macd, calculate_sma_crossover, weighted_signal_combination


class QuantAgent(BaseAgent):
    def __call__(self, state: AgentState) -> CommonResponseSchema:
        symbol = state["data"]["symbol"]
        price_df = state["analysis"]["market_data"]["ohlcv"]
        try:
            if price_df.empty or len(price_df) < 15:
                logger.warning(f"Insufficient OHLCV data for {symbol}: {len(price_df)} candles")
                return CommonResponseSchema(
                    success=False,
                    error=True,
                    data={
                        "indicators": {
                            "macd_signal": "unavailable",
                            "volume": "unavailable",
                        }
                    },
                    msg="Insufficient data for quant analysis",
                )

            price_df["sma_fast"] = price_df["close"].rolling(window=10).mean()
            price_df["sma_slow"] = price_df["close"].rolling(window=30).mean()

            # Generate signal based on your strategy logic
            sma_crossover_signal = calculate_sma_crossover(price_df)

            # MACD
            macd_signal = calculate_macd(price_df, fastperiod=12, slowperiod=26, signalperiod=9)

            combined_signal = weighted_signal_combination(
                {
                    "sma_crossover": sma_crossover_signal,
                    "macd": macd_signal,
                },
                {"sma_crossover": 0.5, "macd": 0.5},
            )

            quant_results = {
                "signal": combined_signal["signal"],
                "confidence": round(combined_signal["confidence"] * 100),
                "strategy_signals": {
                    "macd_strength": macd_signal,
                    "simple_ma_crossover": sma_crossover_signal,
                },
            }
            # logger.info(f"MACD={macd_signal}, Volume={vol_spike}")
            state["analysis"]["quant_analysis"] = quant_results
            if state["metadata"]["show_reasoning"]:
                show_agent_reasoning(quant_results, "Quant Analysis")

            return CommonResponseSchema(
                success=True,
                error=False,
                data=quant_results,
                msg="Quant analysis completed",
            )
        except Exception as e:
            logger.exception(f"Quant error: {str(e)}")
            return CommonResponseSchema(
                success=False,
                error=True,
                data={},
                msg=f"Quant error: {str(e)}",
            )
