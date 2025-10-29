"""
Orchestrators module for pyalgoture trading framework.

This module provides different orchestrator implementations for running backtests
and live trading strategies.
"""

from .backtest import BackTest
from .realtime import RealTime

try:
    from .backtest_rapid import BackTestRapid

    __all__ = ["BackTest", "BackTestRapid", "RealTime"]
except ImportError:
    # Rust engine not available
    __all__ = ["BackTest", "RealTime"]
