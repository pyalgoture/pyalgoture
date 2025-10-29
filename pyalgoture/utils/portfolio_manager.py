import json
import os
import pickle
import warnings
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta
from itertools import groupby
from math import erf, sqrt
from typing import Any

import numpy as np
import pandas as pd
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from six import iteritems

from ..datafeed import DataFeed
from .logger import get_logger
from .nav import FundTracker
from .objects import AccountData, ExecutionType, TradeData
from .util_dt import tz_manager
from .util_math import prec_round
from .util_objects import set_default

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


OHLC_RESAMPLE_RULE = {
    "open": "first",
    "close": "last",
    "high": "max",
    "low": "min",
    "volume": "sum",
    "turnover": "last",
}
RESAMPLE_MAP = {
    "1m": "1T",
    "3m": "3T",
    "5m": "5T",
    "15m": "15T",
    "30m": "30T",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "8h": "8H",
    "12h": "12H",
    "1d": "1D",
    "3d": "3D",
    "1w": "1W",
    "1M": "1M",
    "1Q": "1Q",
    "1Y": "1A",
}


def resample_agg_gen(columns: list[str]) -> dict[str, str]:
    return {col: OHLC_RESAMPLE_RULE.get(col, "last") for col in columns}


pd.set_option("display.max_rows", 150)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", "{:.6f}".format)


ANNUALIZATION_MAP: dict[str, float] = {
    "1m": 365 * 24 * 60 / 1,
    "3m": 365 * 24 * 60 / 3,
    "5m": 365 * 24 * 60 / 5,
    "15m": 365 * 24 * 60 / 15,
    "30m": 365 * 24 * 60 / 30,
    "1h": 365 * 24 / 1,
    "2h": 365 * 24 / 2,
    "4h": 365 * 24 / 4,
    "6h": 365 * 24 / 6,
    "8h": 365 * 24 / 8,
    "12h": 365 * 24 / 12,
    "1d": 365,
    "3d": 121,
    "1w": 52,
    "1M": 12,
}
ANNUALIZATION_MAP_REV: dict[float, str] = {v: k for k, v in ANNUALIZATION_MAP.items()}
DURANTION_MAP: dict[str, int] = {
    "1m": 60,
    "3m": 60 * 3,
    "5m": 60 * 5,
    "15m": 60 * 15,
    "30m": 60 * 30,
    "1h": 60 * 60,
    "2h": 60 * 60 * 2,
    "4h": 60 * 60 * 4,
    "6h": 60 * 60 * 6,
    "8h": 60 * 60 * 8,
    "12h": 60 * 60 * 12,
    "1d": 60 * 60 * 24,
    "3d": 60 * 60 * 24 * 3,
    "1w": 60 * 60 * 24 * 7,
    "1M": 60 * 60 * 24 * 30,
}


def np_linregress(x, y):
    """
    Calculates the slope, intercept, and R-squared value of a simple linear regression model.

    Arguments:
    x -- A list or array of independent variable values.
    y -- A list or array of dependent variable values.

    Returns:
    A tuple containing the slope, intercept, and R-squared value of the linear regression model.
    """
    # n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate the sums of squares and cross-products
    ss_xx: float = np.sum((x - x_mean) ** 2)
    ss_yy: float = np.sum((y - y_mean) ** 2)
    ss_xy: float = np.sum((x - x_mean) * (y - y_mean))

    # Calculate the slope and intercept
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # Calculate the R-squared value
    r_squared = ss_xy**2 / (ss_xx * ss_yy)
    # print(f"slope:{slope}, intercept:{intercept}, r_squared:{r_squared}")
    return slope, intercept, r_squared


def _aligned_series(*many_series):
    """
    Return a new list of series containing the data in the input series, but with their indices aligned. NaNs will be filled in for missing values.

    *many_series: The series to align.
    aligned_series : iterable[array-like]: A new list of series containing the data in the input series, but with their indices aligned. NaNs will be filled in for missing values.
    """

    def _to_pandas(ob):
        """Convert an array-like to a pandas object."""
        if isinstance(ob, pd.Series | pd.DataFrame):
            return ob

        if ob.ndim == 1:
            return pd.Series(ob)
        elif ob.ndim == 2:
            return pd.DataFrame(ob)
        else:
            raise ValueError("cannot convert array of dim > 2 to a pandas structure")

    head = many_series[0]
    tail = many_series[1:]
    n = len(head)
    if isinstance(head, np.ndarray) and all(len(s) == n and isinstance(s, np.ndarray) for s in tail):
        # optimization: ndarrays of the same length are already aligned
        return many_series

    return (v for _, v in iteritems(pd.concat(map(_to_pandas, many_series), axis=1)))


class PortfolioManager:
    """
    Comprehensive portfolio management system for tracking performance and analytics.

    This class provides functionality for:
    - Portfolio performance tracking and analysis
    - Risk metrics calculation (VaR, Sharpe ratio, Sortino ratio, etc.)
    - Trade execution monitoring and position management
    - Statistical analysis and reporting
    - Data visualization and export capabilities

    Attributes:
        aum (float): Assets under management
        logger: Logger instance for debugging and monitoring
        stats_result (pd.Series): Cached statistics results
        data (pd.DataFrame): Main portfolio data with returns and metrics
        trades_df (pd.DataFrame): DataFrame containing all trade records
        fund_tracker (FundTracker): NAV tracking instance
        trade_records (dict): Dictionary of trade data by order ID
        positions (dict): Current portfolio positions
        account (AccountData): Account information and balances
        symbols (set): Set of traded symbols
        last_updated_at (datetime): Timestamp of last portfolio update
    """

    def __init__(self, aum: float, logger=None):
        """
        Initialize the PortfolioManager.

        Args:
            aum (float): Initial assets under management
            logger: Optional logger instance (creates default if None)
        """
        self.logger = logger or get_logger()
        self.aum = aum

        # Initialize data containers
        self.stats_result = pd.Series(dtype="float64")
        self.data = pd.DataFrame()
        self.data_resample_daily = pd.DataFrame()
        self.data_resample_weekly = pd.DataFrame()
        self.data_resample_monthly = pd.DataFrame()
        self.data_resample_yearly = pd.DataFrame()
        self.trades_df = pd.DataFrame()

        # Initialize fund tracking
        self.fund_tracker = FundTracker(nav_per_share=100, aum=aum)

        # Initialize portfolio state
        self.trade_records: dict[str, TradeData] = {}
        self.positions: dict[str, Any] = {}
        self.account: AccountData | None = None
        self.symbols: set[str] = set()
        self.last_updated_at: datetime | None = None

        # Initialize historical data lists
        self._initialize_history_lists()

    def _initialize_history_lists(self) -> None:
        """Initialize all historical data tracking lists."""
        self._date_hist: list[datetime] = []
        self._available_balance_hist: list[float] = []
        self._hld_val_hist: list[float] = []
        self._ast_val_hist: list[float] = []
        self._returns_hist: list[float] = []
        self._pos_count_hist: list[int] = []
        self._total_commission: list[float] = []
        self._total_turnover: list[float] = []
        self._realized_pnl: list[float] = []
        self._unrealized_pnl: list[float] = []
        self._net_investment: list[float] = []
        self._total_pnl: list[float] = []
        self._positions: list[list[dict]] = []
        self._navs: list[float] = []

    def __getstate__(self):
        """Prepare object state for pickling by removing non-serializable objects."""
        state = self.__dict__.copy()
        # Remove logger as it's not serializable
        state.pop("logger", None)
        return state

    def __setstate__(self, state_dict):
        """Restore object state from pickle."""
        self.__dict__.update(state_dict)

    def dump(self, path: str | None = None) -> None:
        """
        Serialize and save the PortfolioManager instance to disk.

        Args:
            path (str | None): File path to save the snapshot. If None, no action taken.
        """
        if path is None:
            return

        try:
            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=5)
        except Exception as e:
            self.logger.exception(f"Failed to dump snapshot to {path}: {e}")

    def load(self, path: str):
        """
        Load a PortfolioManager instance from disk.

        Args:
            path (str): File path to load the snapshot from

        Returns:
            PortfolioManager | None: Loaded instance or None if failed
        """
        if not os.path.exists(path):
            self.logger.error(f"Snapshot file does not exist: {path}")
            return None

        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
                self.__setstate__(obj.__dict__)
            return self
        except Exception as e:
            self.logger.exception(f"Failed to load snapshot from {path}: {e}")
            return None

    @property
    def now(self) -> datetime:
        """Get current datetime with timezone and no microseconds."""
        return datetime.now(tz_manager.tz).replace(microsecond=0)

    def init(self, positions: dict, account: AccountData, symbols: set) -> None:
        """
        Initialize portfolio with positions, account data, and symbols.

        Args:
            positions (dict): Current portfolio positions
            account (AccountData): Account information and balances
            symbols (set): Set of trading symbols
        """
        self.positions = positions
        self.account = account
        self.symbols = symbols

    def upload_trade(self, trade: TradeData | dict) -> None:
        """
        Upload and store a trade record.

        Args:
            trade (TradeData | dict): Trade data as TradeData object or dictionary
        """
        if isinstance(trade, dict):
            order_id = trade["order_id"]
            traded_at = trade["traded_at"]
            self.trade_records[order_id] = TradeData(**trade)
        else:
            order_id = trade.order_id
            traded_at = trade.traded_at
            self.trade_records[order_id] = trade

        self.last_updated_at = traded_at

    def add_snapshot(self, tick: datetime) -> dict:
        """
        Add a portfolio snapshot at the given timestamp.

        Args:
            tick (datetime): Timestamp for the snapshot

        Returns:
            dict: Snapshot data including balances, positions, and metrics
        """
        if isinstance(tick, str):
            tick = parse(tick)

        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics()
        current_positions = self._get_current_positions_data(portfolio_metrics)

        # Get account balance and calculate total assets
        cash = self.account.balance if self.account else 0.0
        assets_value = cash + portfolio_metrics["holding_value"]
        nav = self.fund_tracker.get_nav_per_share()

        # Update fund tracker with PnL change
        if self._total_pnl:
            pnl_change = portfolio_metrics["total_pnl"] - self._total_pnl[-1]
            self.fund_tracker.update_pnl(pnl_change)
            nav = self.fund_tracker.get_nav_per_share()

        # Update position weights
        self._update_position_weights(current_positions, portfolio_metrics["holding_notional"], cash)

        # Store or update snapshot data
        snapshot_data = {
            "tick": tick,
            "available_balance": cash,
            "holding_value": portfolio_metrics["holding_value"],
            "assets_value": assets_value,
            "holding_number": portfolio_metrics["holding_number"],
            "total_commission": portfolio_metrics["total_commission"],
            "total_turnover": portfolio_metrics["total_turnover"],
            "realized_pnl": portfolio_metrics["realized_pnl"],
            "unrealized_pnl": portfolio_metrics["unrealized_pnl"],
            "net_investment": portfolio_metrics["net_investment"],
            "total_pnl": portfolio_metrics["total_pnl"],
            "nav": nav,
            "positions": current_positions,
        }

        self._store_snapshot_data(tick, snapshot_data)
        self.last_updated_at = tick

        return snapshot_data

    def _calculate_portfolio_metrics(self) -> dict:
        """Calculate aggregated portfolio metrics from all positions."""
        metrics = {
            "holding_number": 0,
            "total_commission": 0,
            "total_turnover": 0,
            "realized_pnl": 0,
            "unrealized_pnl": 0,
            "net_investment": 0,
            "total_pnl": 0,
            "holding_notional": 0,
            "holding_value": 0,
        }

        for position in self.positions.values():
            metrics["total_commission"] += abs(position.total_commission)
            metrics["total_turnover"] += abs(position.total_turnover)
            metrics["net_investment"] += position.net_investment
            metrics["realized_pnl"] += position.realized_pnl
            metrics["unrealized_pnl"] += position.unrealized_pnl
            metrics["total_pnl"] += position.total_pnl

            if position.get("size"):
                metrics["holding_number"] += 1
                metrics["holding_notional"] += position.get("notional_value", 0)
                metrics["holding_value"] += position.get("market_value", 0)

        return metrics

    def _get_current_positions_data(self, metrics: dict) -> list[dict]:
        """Get current positions data as list of dictionaries."""
        current_positions = []
        for position in self.positions.values():
            if position.get("size"):
                current_positions.append(position.to_dict())
        return current_positions

    def _update_position_weights(self, positions: list[dict], holding_notional: float, cash: float) -> None:
        """Update weight calculations for all positions."""
        total_with_cash = holding_notional + cash

        for position in positions:
            notional_value = position.get("notional_value", 0) if position.get("size") else 0

            # Weight including cash
            position["weight"] = prec_round(notional_value / total_with_cash, 4) if total_with_cash else 0.0

            # Weight excluding cash
            position["weight_wo_cash"] = prec_round(notional_value / holding_notional, 4) if holding_notional else 0.0

    def _store_snapshot_data(self, tick: datetime, data: dict) -> None:
        """Store snapshot data in historical lists, updating if timestamp exists."""
        if tick in self._date_hist:
            # Update existing snapshot
            idx = self._date_hist.index(tick)
            self._update_existing_snapshot(idx, data)
        else:
            # Add new snapshot
            self._add_new_snapshot(tick, data)

    def _update_existing_snapshot(self, idx: int, data: dict) -> None:
        """Update existing snapshot at given index."""
        self._date_hist[idx] = data["tick"]
        self._available_balance_hist[idx] = data["available_balance"]
        self._hld_val_hist[idx] = data["holding_value"]
        self._ast_val_hist[idx] = data["assets_value"]
        self._pos_count_hist[idx] = data["holding_number"]
        self._total_commission[idx] = data["total_commission"]
        self._total_turnover[idx] = data["total_turnover"]
        self._realized_pnl[idx] = data["realized_pnl"]
        self._unrealized_pnl[idx] = data["unrealized_pnl"]
        self._net_investment[idx] = data["net_investment"]
        self._total_pnl[idx] = data["total_pnl"]
        self._navs[idx] = data["nav"]
        self._positions[idx] = data["positions"]

    def _add_new_snapshot(self, tick: datetime, data: dict) -> None:
        """Add new snapshot to historical lists."""
        self._date_hist.append(tick)
        self._available_balance_hist.append(data["available_balance"])
        self._hld_val_hist.append(data["holding_value"])
        self._ast_val_hist.append(data["assets_value"])
        self._pos_count_hist.append(data["holding_number"])
        self._total_commission.append(data["total_commission"])
        self._total_turnover.append(data["total_turnover"])
        self._realized_pnl.append(data["realized_pnl"])
        self._unrealized_pnl.append(data["unrealized_pnl"])
        self._net_investment.append(data["net_investment"])
        self._total_pnl.append(data["total_pnl"])
        self._navs.append(data["nav"])
        self._positions.append(data["positions"])

    @property
    def capital(self) -> float:
        """Get current account capital."""
        return self.account.capital if self.account else 0.0

    @property
    def initial_capital(self) -> float:
        """Get initial account capital."""
        return self.account.initial_capital if self.account else 0.0

    @staticmethod
    def currency(amount: float) -> str:
        """
        Format amount as currency string.

        Args:
            amount (float): Amount to format

        Returns:
            str: Formatted currency string
        """
        return f"${amount:,.2f}" if amount >= 0 else f"-${-amount:,.2f}"

    @property
    def start(self) -> datetime | None:
        """Get the start datetime of the portfolio tracking period."""
        if self.trades_df.empty:
            if self.trade_records:
                first_trade_key = next(iter(self.trade_records))
                first_trade = self.trade_records[first_trade_key]
                return first_trade["traded_at"]
            return self.data.index[0] if not self.data.empty else None

        if not self.data.empty:
            return min(self.trades_df["traded_at"].iloc[0], self.data.index[0])
        return self.trades_df["traded_at"].iloc[0] if not self.trades_df.empty else None

    @property
    def end(self) -> datetime:
        """Get the end datetime of the portfolio tracking period."""
        return self.last_updated_at or self.now

    @property
    def duration(self) -> timedelta | None:
        """Get the total duration of the portfolio tracking period."""
        return (self.end - self.start) if self.start and self.end else None

    @property
    def diff(self) -> relativedelta | None:
        """Get the relative difference between start and end times."""
        return relativedelta(self.end, self.start) if self.start and self.end else None

    @property
    def diff_in_years(self) -> float | None:
        """Get the duration in years as a float."""
        if not (self.start and self.end):
            return None
        diff = self.end - self.start
        return (diff.days + diff.seconds / 86400) / 365.2425

    @property
    def max_runup(self) -> tuple[float, tuple]:
        """
        Calculate maximum runup (peak-to-trough profit) and its period.

        Returns:
            tuple: (max_runup_value, (start_date, end_date))
        """
        if self.data.empty:
            return 0.0, ()

        runup_list, _, runup_date_list = self._calculate_runups(self.data)

        if runup_list:
            max_value = max(runup_list)
            max_index = runup_list.index(max_value)
            max_date = runup_date_list[max_index]
            return prec_round(max_value, 6), max_date

        return 0.0, ()

    def _calculate_runups(self, data: pd.DataFrame) -> tuple[list, list, list]:
        """
        Calculate all runup periods from asset value data.

        Args:
            data (pd.DataFrame): Portfolio data with assets_value column

        Returns:
            tuple: (runup_values, runup_indices, runup_dates)
        """
        high_val = None
        low_val = np.inf
        high_index = 0
        low_index = 0
        runup_list = []
        runup_index_list = []
        runup_date_list = []

        for idx, val in enumerate(data["assets_value"]):
            # When current value drops below low, calculate runup if we had a peak
            if val <= low_val:
                if high_val == low_val or high_index >= low_index:
                    high_val = low_val = val
                    high_index = low_index = idx
                    continue

                runup = (high_val - low_val) / low_val if low_val else 0.0
                runup_list.append(runup)
                runup_index_list.append((high_index, low_index))
                runup_date_list.append((data.index[low_index], data.index[high_index]))

                high_val = low_val = val
                high_index = low_index = idx

            # Update high value and index if current value is higher
            if not high_val or val > high_val:
                high_val = val
                high_index = idx

        # Handle final runup if we end on a peak
        if low_index < high_index:
            runup = (high_val - low_val) / low_val if low_val else 0.0
            runup_list.append(runup)
            runup_index_list.append((low_index, high_index))
            runup_date_list.append((data.index[low_index], data.index[high_index]))

        return runup_list, runup_index_list, runup_date_list

    @property
    def max_drawdown(self) -> tuple[float, tuple]:
        """
        Calculate maximum drawdown (peak-to-trough loss) and its period.

        A maximum drawdown (MDD) is the maximum observed loss from a peak to a trough
        of a portfolio before a new peak is attained. It's an indicator of downside
        risk over a specified time period.

        Returns:
            tuple: (max_drawdown_value, (start_date, end_date))
        """
        if self.data.empty:
            return 0.0, ()

        drawdown_list, _, drawdown_date_list = self._calculate_drawdowns(self.data)

        if drawdown_list:
            max_value = max(drawdown_list)
            max_index = drawdown_list.index(max_value)
            max_date = drawdown_date_list[max_index]
            return prec_round(max_value, 6), max_date

        return 0.0, ()

    def _calculate_drawdowns(self, data: pd.DataFrame) -> tuple[list, list, list]:
        """
        Calculate all drawdown periods from asset value data.

        Args:
            data (pd.DataFrame): Portfolio data with assets_value column

        Returns:
            tuple: (drawdown_values, drawdown_indices, drawdown_dates)
        """
        high_val = -1
        low_val = None
        high_index = 0
        low_index = 0
        drawdown_list = []
        drawdown_index_list = []
        drawdown_date_list = []

        for idx, val in enumerate(data["assets_value"]):
            # When current value rises above high, calculate drawdown if we had a trough
            if val >= high_val:
                if high_val == low_val or high_index >= low_index:
                    high_val = low_val = val
                    high_index = low_index = idx
                    continue

                drawdown = (high_val - low_val) / high_val if high_val else 0.0
                drawdown_list.append(drawdown)
                drawdown_index_list.append((high_index, low_index))
                drawdown_date_list.append((data.index[high_index], data.index[low_index]))

                high_val = low_val = val
                high_index = low_index = idx

            # Update low value and index if current value is lower
            if not low_val or val < low_val:
                low_val = val
                low_index = idx

        # Handle final drawdown if we end on a trough
        if low_index > high_index:
            drawdown = (high_val - low_val) / high_val if high_val else 0.0
            drawdown_list.append(drawdown)
            drawdown_index_list.append((high_index, low_index))
            drawdown_date_list.append((data.index[high_index], data.index[low_index]))

        return drawdown_list, drawdown_index_list, drawdown_date_list

    def cum_returns(self, returns, starting_value=0, out=None):
        """
        Compute cumulative returns from simple returns.

        Parameters
        ----------
        returns : pd.Series, np.ndarray, or pd.DataFrame
            Returns of the strategy as a percentage, noncumulative.
            - Time series with decimal returns.
            - Example::

                2015-07-16   -0.012143
                2015-07-17    0.045350
                2015-07-20    0.030957
                2015-07-21    0.004902

            - Also accepts two dimensional data. In this case, each column is
            cumulated.

        starting_value : float, optional
        The starting returns.
        out : array-like, optional
            Array to use as output buffer.
            If not passed, a new array will be created.

        Returns
        -------
        cumulative_returns : array-like
            Series of cumulative returns.
        """
        if returns is None:
            if self.data.empty:
                return np.nan
            returns = self.data["pct_returns"]

        if len(returns) < 1:
            return returns.copy()

        nanmask = np.isnan(returns)
        if np.any(nanmask):
            returns = returns.copy()
            returns[nanmask] = 0

        allocated_output = out is None
        if allocated_output:
            out = np.empty_like(returns)

        np.add(returns, 1, out=out)
        out.cumprod(axis=0, out=out)

        if starting_value == 0:
            np.subtract(out, 1, out=out)
        else:
            np.multiply(out, starting_value, out=out)

        if allocated_output:
            if returns.ndim == 1 and isinstance(returns, pd.Series):
                out = pd.Series(out, index=returns.index)
            elif isinstance(returns, pd.DataFrame):
                out = pd.DataFrame(
                    out,
                    index=returns.index,
                    columns=returns.columns,
                )

        return out

    def drawdown_series(
        self,
        returns=None,
    ):
        """Convert returns series to drawdown series"""
        if returns is None:
            if self.data.empty:
                return np.nan
            returns = self.data["pct_returns"]

        returns = self.cum_returns(returns, starting_value=1)
        dd = returns / np.maximum.accumulate(returns) - 1.0
        dd = dd.replace([np.inf, -np.inf, -0], 0)
        dd.index = dd.index.strftime("%Y-%m-%d %H:%M:%S")
        return dd.to_dict()

    @property
    def information_ratio(self):
        """
        Determines the Information ratio of a strategy.
        See https://en.wikipedia.org/wiki/information_ratio for more details.
        """
        if self.data.empty:
            return np.nan
        returns = self.data["pct_returns"]
        factor_returns = self.data["benchmark_pct_returns"]
        returns, factor_returns = _aligned_series(returns, factor_returns)

        if len(returns) < 2:
            return np.nan

        active_return = returns - factor_returns
        tracking_error = np.std(active_return, ddof=1)
        # print(tracking_error, type(tracking_error))
        if np.isnan(tracking_error):
            return 0.0
        if tracking_error == 0:
            return np.nan
        return prec_round(np.mean(active_return) / tracking_error, 6)

    @property
    def cumulative_returns(self):
        """Get cumulative returns from percentage returns."""
        return self.data["pct_returns"].cumsum()

    @property
    def total_return(self) -> float:
        """
        Calculate total return as percentage.

        Total Returns = (Pe - Ps) / Ps * 100%
        Where Pe = ending value, Ps = starting value
        """
        if self.data.empty:
            return 0.0
        return prec_round(self.data["cum_returns"].iloc[-1], 6)

    @property
    def max_return(self) -> float:
        """Get maximum cumulative return achieved."""
        if self.data.empty:
            return 0.0
        return prec_round(max(self.data["cum_returns"]), 6)

    @property
    def min_return(self) -> float:
        """Get minimum cumulative return (maximum loss)."""
        if self.data.empty:
            return 0.0
        return prec_round(min(self.data["cum_returns"]), 6)

    def annualized_return(self, annualization=None) -> float:
        """
        Calculate compound annual growth rate (CAGR).

        Uses two possible formulas:
        1. y = (v/c)^(D/T) - 1 where v=final value, c=initial value, D=period
        2. y = (1+P)**(252/n) - 1 where P=return rate, n=number of periods

        Args:
            annualization (int, optional): Annualization factor. Uses self.annualization if None.

        Returns:
            float: Annualized return as decimal (e.g., 0.15 for 15%)
        """
        if self.data.empty:
            return np.nan

        annualization = annualization or self.annualization
        returns = self.data["pct_returns"]

        if len(returns) < 1:
            return np.nan

        # Calculate duration in terms of the interval
        interval = self.resample_interval or self.interval
        duration_seconds = (self.data.index[-1] - self.data.index[0]).total_seconds()
        duration = duration_seconds / DURANTION_MAP[interval]

        if not duration:
            return np.nan

        # Calculate annualized return
        num_years = float(duration) / annualization
        cum_return = float(self.total_return)
        annual_return = (1 + cum_return) ** (1.0 / num_years) - 1

        # Handle complex numbers (shouldn't happen but safety check)
        if isinstance(annual_return, complex):
            annual_return = annual_return.real

        # Sanity check for unrealistic values
        if np.isinf(annual_return) or annual_return > 10_000:
            return np.nan

        return prec_round(annual_return, 6)

    def sortino_ratio(self, returns=None, annualization=None, required_return=0, _downside_risk=None):
        """
        Determines the Sortino ratio of a strategy.
        See `<https://www.sunrisecapital.com/wp-content/uploads/2014/06/Futures_Mag_Sortino_0213.pdf>`__ for more details.
        """
        annualization = annualization if annualization else self.annualization
        if returns is None:
            if self.data.empty:
                return np.nan
            returns = self.data["pct_returns"]
        out = np.empty(())

        if len(returns) < 2:
            out = out.item()
            return out
        tmp = returns - required_return
        adj_returns = np.asanyarray(tmp)

        average_annual_return = np.nanmean(adj_returns, axis=0) * annualization
        annualized_downside_risk = (
            _downside_risk if _downside_risk is not None else self.downside_risk(annualization, required_return)
        )
        np.divide(average_annual_return, annualized_downside_risk, out=out)

        return prec_round(out.item(), 6)

    def rolling_sortino(self, returns=None, risk_free=0.0, rolling_period=183, annualization=None):  # 6month
        annualization = annualization if annualization else self.annualization
        if returns is None:
            if self.data_resample_daily.empty:
                return {}
            returns = self.data_resample_daily["pct_returns"]
        annualization = self._get_annualization(interval="1d")
        res = returns.rolling(rolling_period).apply(
            lambda x: self.sortino_ratio(x, annualization=annualization, required_return=risk_free)
        )
        res = res.replace([np.inf, -np.inf, np.nan], [None, None, None])

        return {k.strftime("%Y-%m-%d %H:%M:%S"): v for k, v in res.to_dict().items()}

    def annualized_volatility(self, returns=None, annualization=None, alpha=2.0):
        """
        Determines the annual volatility of a strategy.
        """
        annualization = annualization if annualization else self.annualization
        if returns is None:
            if self.data.empty:
                return np.nan
            returns = self.data["pct_returns"]
        out = np.empty(())

        if len(returns) < 2:
            return out.item()

        np.nanstd(returns, ddof=1, axis=0, out=out)
        out = np.multiply(out, annualization ** (1.0 / alpha), out=out)

        return prec_round(out.item(), 6)

    def rolling_volatility(self, returns=None, rolling_period=126, annualization=None):
        annualization = annualization if annualization else self.annualization
        if returns is None:
            if self.data_resample_daily.empty:
                return {}
            returns = self.data_resample_daily["pct_returns"]

        res = returns.rolling(rolling_period).std()
        if annualization:
            res = res * np.sqrt(annualization)
        res = res.fillna(0)
        return res

    def sharpe_ratio_breakdown(self, returns=None, risk_free=0.0, annualization=None):
        annualization = annualization if annualization else self.annualization
        sr = self.sharpe_ratio(returns=returns, risk_free=risk_free, annualization=annualization)
        sr_std = sqrt((1 + 0.5 * sr**2) / self.duration.days) if self.duration else 0.0
        sr_max = sr + 3 * sr_std
        sr_min = sr - 3 * sr_std
        return sr_std, sr_max, sr_min

    def sharpe_ratio(self, returns=None, risk_free=0.0, annualization=None):
        """
        summary Returns the daily Sharpe ratio of the returns.
        param rets: 1d numpy array or fund list of daily returns (centered on 0)
        param risk_free: risk free returns, default is 0%
        return Sharpe Ratio, computed off daily returns
        """
        # p = returns is None
        # print(f">> risk_free:{risk_free}; annualization:{annualization}")
        annualization = annualization if annualization else self.annualization
        if returns is None:
            if self.data.empty:
                return np.nan
            returns = self.data["pct_returns"]
        std = np.nanstd(returns, axis=0)
        mean = np.nanmean(returns, axis=0)
        sharpe = (mean * annualization - risk_free) / (std * np.sqrt(annualization))
        # sharpe1 = ((mean - risk_free) / std) * np.sqrt(annualization)
        # print(f">>>>>> self.annualization:{self.annualization} | annualization:{annualization} | std:{std} | mean:{mean} | sharpe:{sharpe} | sharpe1:{sharpe1}")
        return prec_round(sharpe, 6)

    def probabilistic_sharpe_ratio(self, returns=None, target_sharpe=2, risk_free=0.0, annualization=None):
        """
        Calculate the Probabilistic Sharpe Ratio. 用于评估实际夏普比率相对于目标夏普比率的显著性，特别是在小样本情况下更为有效

        :param returns: Array-like, the returns of the strategy.
        :param target_sharpe: The target Sharpe ratio to compare against.
        :param risk_free: The risk-free rate (default is 0).
        :param annualization: The number of periods per year (default is 252 for daily returns).
        :return: The Probabilistic Sharpe Ratio.
        """
        annualization = annualization if annualization else self.annualization
        if returns is None:
            if self.data.empty:
                return np.nan
            returns = self.data["pct_returns"]

        # Calculate annualized Sharpe ratio
        std = np.nanstd(returns, axis=0)
        mean = np.nanmean(returns, axis=0)
        sharpe = (mean * annualization - risk_free) / (std * np.sqrt(annualization))

        # Calculate the Probabilistic Sharpe Ratio
        sqrt_n = np.sqrt(len(returns))
        numerator = (sharpe - target_sharpe) * sqrt_n
        denominator = np.sqrt(1 - (sharpe**2 / (2 * sqrt_n)))

        def custom_norm_cdf(x):
            """
            Custom implementation of the CDF for the standard normal distribution.

            :param x: The value at which to evaluate the CDF.
            :return: The CDF value.
            """
            return 0.5 * (1 + erf(x / sqrt(2)))

        psr = custom_norm_cdf(numerator / denominator)
        # from scipy.stats import norm
        # psr = norm.cdf(numerator / denominator)
        # print(f">>>>>>>>> psr:{psr} ||| psr_cus:{psr_cus}")
        return psr

    def rolling_sharpe(self, returns=None, risk_free=0.0, rolling_period=30, annualization=None):  # 6month
        annualization = annualization if annualization else self.annualization
        if returns is None:
            if self.data_resample_daily.empty:
                return {}, {}
            returns = self.data_resample_daily["pct_returns"]
        annualization = self._get_annualization(interval="1d")
        res = returns.rolling(rolling_period).apply(
            lambda x: self.sharpe_ratio(x, risk_free=risk_free, annualization=annualization)
        )
        res = res.replace([np.inf, -np.inf, np.nan], [None, None, None])
        month_res = res.resample("ME", label="right", closed="right").ffill().fillna(0).copy()  # .ffill().bfill()
        return {k.strftime("%Y-%m-%d %H:%M:%S"): v for k, v in res.to_dict().items()}, {
            k.strftime("%Y-%m-%d %H:%M:%S"): v for k, v in month_res.to_dict().items()
        }

    def calmar_ratio(self, annualization=None):
        """
        Determines the Calmar ratio, or drawdown ratio, of a strategy.

        See https://en.wikipedia.org/wiki/Calmar_ratio for more details.
        """
        annualization = annualization if annualization else self.annualization
        # returns = self.data['pct_returns']
        max_dd, _ = self.max_drawdown
        res = np.nan
        if max_dd:
            # if max_dd < 0:
            res = self.annualized_return(annualization) / abs(max_dd)
        # else:
        #     return np.nan
        # print(f"annualized_return:{self.annualized_return(annualization)}; max_dd:{max_dd}; res:{res}")
        if np.isinf(res) or res > 10_000:
            return np.nan

        return prec_round(res, 6)

    def downside_risk(self, annualization=None, required_return=0):
        """
        Determines the downside deviation below a threshold

        See `<https://www.sunrisecapital.com/wp-content/uploads/2014/06/Futures_Mag_Sortino_0213.pdf>`__ for more details, specifically why using the
        standard deviation of the negative returns is not correct.
        """
        annualization = annualization if annualization else self.annualization
        returns = self.data["pct_returns"]

        out = np.empty(())

        if len(returns) < 1:
            out[()] = np.nan
            out = out.item()
            return out

        tmp = np.asanyarray(returns) - np.asanyarray(required_return)
        downside_diff = np.clip(tmp, -np.inf, 0)

        np.square(downside_diff, out=downside_diff)
        np.nanmean(downside_diff, axis=0, out=out)
        np.sqrt(out, out=out)
        np.multiply(out, np.sqrt(annualization), out=out)

        return prec_round(out.item(), 6)

    @property
    def tail_ratio(self):
        """
        Determines the ratio between the right (95%) and left tail (5%).
        For example, a ratio of 0.25 means that losses are four times
        as bad as profits.
        """
        returns = self.data["pct_returns"]
        if len(returns) < 1:
            return np.nan

        returns = np.asanyarray(returns)
        # Be tolerant of nan's
        returns = returns[~np.isnan(returns)]
        if len(returns) < 1:
            return np.nan
        # print(f"95%:{np.abs(np.percentile(returns, 95))}; 5%:{np.abs(np.percentile(returns, 5))}")
        return prec_round(np.abs(np.percentile(returns, 95)) / np.abs(np.percentile(returns, 5)), 6)

    @property
    def common_sense_ratio(self):
        """Measures the common sense ratio (profit factor * tail ratio)"""
        return self.profit_factor * self.tail_ratio
        # return self.payoff_ratio * self.tail_ratio

    @property
    def skew(self):
        """
        Calculates returns' skewness
        (the degree of asymmetry of a distribution around its mean)
        """
        returns = self.data["pct_returns"]
        if len(returns) < 1:
            return np.nan
        return returns.skew()

    @property
    def kurtosis(self):
        """
        Calculates returns' kurtosis
        (the degree to which a distribution peak compared to a normal distribution)
        """
        returns = self.data["pct_returns"]
        if len(returns) < 1:
            return np.nan
        return returns.kurtosis()

    @property
    def stability_of_timeseries(self):
        """
        Determines R-squared of a linear fit to the cumulative
        log returns. Computes an ordinary least squares linear fit,
        and returns R-squared.
        """
        returns = self.data["pct_returns"]
        if len(returns) < 2:
            return np.nan

        returns = np.asanyarray(returns)
        returns = returns[~np.isnan(returns)]

        cum_log_returns = np.log1p(returns).cumsum()
        # rhat = linregress(np.arange(len(cum_log_returns)), cum_log_returns)[2]
        # return prec_round(rhat**2, 6)
        rhat = np_linregress(np.arange(len(cum_log_returns)), cum_log_returns)[2]
        return prec_round(rhat, 6)

    def omega_ratio(self, returns=None, risk_free=0.0, annualization=None, required_return=0.0):
        """
        Determines the Omega ratio of a strategy.
        See https://en.wikipedia.org/wiki/Omega_ratio for more details.
        """
        annualization = annualization if annualization else self.annualization
        if returns is None:
            if self.data.empty:
                return np.nan
            returns = self.data["pct_returns"]
        if len(returns) < 2:
            return np.nan

        if annualization == 1:
            return_threshold = required_return
        elif required_return <= -1:
            return np.nan
        else:
            return_threshold = (1 + required_return) ** (1.0 / annualization) - 1

        returns_less_thresh = returns - risk_free - return_threshold

        numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
        denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])

        if denom > 0.0:
            return prec_round(numer / denom, 6)
        else:
            return np.nan

    def alpha(
        self,
        returns=None,
        factor_returns=None,
        risk_free=0.0,
        annualization=None,
        _beta=None,
    ):
        """
        Calculates annualized alpha.
        """
        # Alpha: The range of alpha values for a stock is typically between -1 and +1. A positive alpha value indicates that a stock has outperformed its expected return, while a negative alpha value indicates that a stock has underperformed its expected return. An alpha value of zero indicates that a stock has performed in line with its expected return.
        annualization = annualization if annualization else self.annualization
        if returns is None:
            if self.data.empty:
                return np.nan
            returns = self.data["pct_returns"]
        if factor_returns is None:
            factor_returns = self.data["benchmark_pct_returns"]
        returns, factor_returns = _aligned_series(returns, factor_returns)
        # print(f"[alpha] returns:{returns}; factor_returns:{factor_returns}")
        # returns.to_csv(r'ret.csv')
        # self.data['benchmark'].to_csv(r'bt.csv')

        out = np.empty(())

        if len(returns) < 2:
            return out.item()

        if _beta is None:
            _beta = self.beta(returns=returns, risk_free=risk_free)

        adj_returns = returns - risk_free
        adj_factor_returns = factor_returns - risk_free
        alpha_series = adj_returns - (_beta * adj_factor_returns)

        out = np.subtract(
            np.power(
                np.add(np.nanmean(alpha_series, axis=0), 1, out=out),
                annualization,
                out=out,
            ),
            1,
            out=out,
        )
        # print(f"[alpha] out:{out} ({type(out)}) | annualization:{annualization}")

        # if -20 < out.item() < 20:
        #     return prec_round(out.item(), 6)
        # else:
        #     return np.nan
        return prec_round(out.item(), 6)

    def beta(self, returns=None, factor_returns=None, risk_free=0.0):
        """
        Calculates beta.
        """
        # Beta: The range of beta values for a stock is typically between 0 and 2. A beta value of 1 indicates that a stock has the same level of volatility as the market. A beta value greater than 1 indicates that a stock is more volatile than the market, while a beta value less than 1 indicates that a stock is less volatile than the market.

        if returns is None:
            if self.data.empty:
                return np.nan
            returns = self.data["pct_returns"]
        if factor_returns is None:
            factor_returns = self.data["benchmark_pct_returns"]
        returns, factor_returns = _aligned_series(returns, factor_returns)

        nan = np.nan
        isnan = np.isnan

        returns = np.asanyarray(returns)[:, np.newaxis]

        if factor_returns.ndim == 1:
            factor_returns = np.asanyarray(factor_returns)[:, np.newaxis]

        N, M = returns.shape

        out = np.full(M, nan)
        # print(out, type(out),'-------------------------beta') # nan

        if len(returns) < 1 or len(factor_returns) < 2:
            return prec_round(out.item(), 6)

        # shape: (N, M)
        independent = np.where(isnan(returns), nan, factor_returns)

        # Calculate beta as Cov(X, Y) / Cov(X, X).
        # https://en.wikipedia.org/wiki/Simple_linear_regression#Fitting_the_regression_line  # noqa

        ind_residual = independent - np.nanmean(independent, axis=0)

        covariances = np.nanmean(ind_residual * returns, axis=0)

        # We end up with different variances in each column here because each
        # column may have a different subset of the data dropped due to missing
        # data in the corresponding dependent column.
        # shape: (M)

        np.square(ind_residual, out=ind_residual)
        independent_variances = np.nanmean(ind_residual, axis=0)
        independent_variances[independent_variances < 1.0e-30] = np.nan

        np.divide(covariances, independent_variances, out=out)
        # print(out, type(out),'-------------------------beta') # nan
        if -4 < out.item() < 4:
            return prec_round(out.item(), 6)
        else:
            return np.nan

    def rolling_greeks(self, returns=None, factor_returns=None, rolling_period=183, annualization=None):
        """Calculates rolling alpha and beta of the portfolio"""
        if returns is None:
            if self.data_resample_daily.empty:
                return {}
            returns = self.data_resample_daily["pct_returns"]
        if factor_returns is None:
            factor_returns = self.data_resample_daily["benchmark_pct_returns"]

        df = pd.DataFrame(data={"returns": returns, "benchmark": factor_returns})
        df = df.fillna(0)
        corr = df.rolling(int(rolling_period)).corr().unstack()["returns"]["benchmark"]
        std = df.rolling(int(rolling_period)).std()
        beta = corr * std["returns"] / std["benchmark"]

        alpha = df["returns"].mean() - beta * df["benchmark"].mean()

        # alpha = alpha * periods
        return pd.DataFrame(index=returns.index, data={"beta": beta, "alpha": alpha})

    @property
    def win_ratio(self):
        # if self.number_trades:
        #     return self.num_winning_trades / self.number_trades
        # else:
        #     return 0.0
        if self.number_closed_trades:
            return prec_round(self.num_winning_trades / self.number_closed_trades, 6)
        else:
            return 0.0

    @property
    def win_ratio_long(self):
        if self.number_closed_trades:
            return prec_round(self.num_winning_long_trades / self.number_long_closed_trades, 6)
        else:
            return 0.0

    @property
    def win_ratio_short(self):
        if self.number_closed_trades:
            return prec_round(self.num_winning_short_trades / self.number_short_closed_trades, 6)
        else:
            return 0.0

    @property
    def loss_ratio(self):
        # if self.number_trades:
        #     return self.num_losing_trades / self.number_trades
        # else:
        #     return 0.0
        if self.number_closed_trades:
            return prec_round(self.num_losing_trades / self.number_closed_trades, 6)
        else:
            return 0.0

    @property
    def loss_ratio_long(self):
        if self.number_closed_trades:
            return prec_round(self.num_losing_long_trades / self.number_long_closed_trades, 6)
        else:
            return 0.0

    @property
    def loss_ratio_short(self):
        if self.number_closed_trades:
            return prec_round(self.num_losing_short_trades / self.number_short_closed_trades, 6)
        else:
            return 0.0

    @property
    def win_days(self):
        # if len(self.pnls) > 1:
        #     nums_win = (self.pnls["pnl_daily"] > 0).sum()
        if len(self.data_resample_daily) > 1:
            nums_win = (self.data_resample_daily["pnl_net"] > 0).sum()
            if nums_win:
                return nums_win
            else:
                return 0
        else:
            return 0

    @property
    def loss_days(self):
        # if len(self.pnls) > 1:
        #     nums_loss = (self.pnls["pnl_daily"] < 0).sum()
        if len(self.data_resample_daily) > 1:
            nums_loss = (self.data_resample_daily["pnl_net"] < 0).sum()
            if nums_loss:
                return nums_loss
            else:
                return 0
        else:
            return 0

    @property
    def max_win_in_day(self):
        try:
            # if len(self.pnls) > 1:
            #     return max(self.pnls[self.pnls["pnl_daily"] > 0]["pnl_daily"])
            if len(self.data_resample_daily) > 1:
                return prec_round(
                    max(self.data_resample_daily[self.data_resample_daily["pnl_net"] > 0]["pnl_net"]),
                    4,
                )
            else:
                return 0.0
        except:  # noqa: E722
            return 0.0

    @property
    def max_loss_in_day(self):
        try:
            # if len(self.pnls) > 1:
            #     return min(self.pnls[self.pnls["pnl_daily"] < 0]["pnl_daily"])
            if len(self.data_resample_daily) > 1:
                return prec_round(
                    min(self.data_resample_daily[self.data_resample_daily["pnl_net"] < 0]["pnl_net"]),
                    4,
                )
            else:
                return 0.0
        except:  # noqa: E722
            return 0.0

    def max_consecutive_days(self):
        # if len(self.pnls) > 1:
        if len(self.data_resample_daily) > 1:
            counter = defaultdict(list)
            for key, val in groupby(
                # self.pnls["pnl_daily"].to_list(),
                self.data_resample_daily["pnl_net"].to_list(),
                lambda ele: "plus" if ele > 0 else "minus" if ele < 0 else "-",
            ):
                counter[key].append(len(list(val)))

            return (
                max(counter["plus"]) if counter["plus"] else 0,
                max(counter["minus"]) if counter["minus"] else 0,
            )
        else:
            return 0, 0

    @property
    def ending_balance(self):
        if self.data.empty:
            return 0.0
        return self.data.iloc[-1]["assets_value"]

    @property
    def holding_values(self):
        if self.data.empty:
            return 0.0
        return self.data.iloc[-1]["holding_value"]

    @property
    def available_balance(self):
        if self.data.empty:
            return 0.0
        return self.data.iloc[-1]["available_balance"]

    @property
    def net_investment(self):
        if self.data.empty:
            return 0.0
        return self.data.iloc[-1]["total_net_investment"]

    @property
    def total_net_pnl(self):
        if self.data.empty:
            return 0.0
        return self.data.iloc[-1]["pnl_cumsum"]

    @property
    def total_commission(self):
        if self.data.empty:
            return 0.0
        return self.data.iloc[-1]["total_commission"]

    @property
    def total_turnover(self):
        if self.data.empty:
            return 0.0
        return self.data.iloc[-1]["total_turnover"]

    def avg_pnl_by_freq(self, freq):
        if self.data.empty:
            return 0.0, 0.0
        # print(self.data,'-----------avg_pnl_by_freq')
        freqly = self.data.resample(freq).last().ffill()
        # print(freqly,'===========')
        freqly["freqlypnl"] = freqly["assets_value"] - freqly["assets_value"].shift(1)
        return prec_round(freqly["freqlypnl"].mean(), 4), prec_round(freqly["assets_value"].pct_change(1).mean(), 6)

    def trade_freq(self, freq):
        if self.trades_df.empty:
            return 0.0
        return prec_round(self.trades_df.order_id.resample(freq).count().mean(), 2)
        # return self.closed_trades.id.resample(freq).count().mean()

    @property
    def past_24hr_pnl(self):
        try:
            if self.last_updated_at is None:
                return 0.0
            lastest_tick = self.last_updated_at
            past24hr = lastest_tick - relativedelta(days=1, minutes=1)
            latest_snapshot = self.data[self.data.index >= past24hr]
            if latest_snapshot.empty:
                return 0.0
            return prec_round(
                latest_snapshot.iloc[-1]["assets_value"] - latest_snapshot.iloc[0]["assets_value"],
                4,
            )
        except Exception:
            return 0.0

    @property
    def past_24hr_roi(self):
        try:
            if self.last_updated_at is None:
                return 0.0
            lastest_tick = self.last_updated_at
            past24hr = lastest_tick - relativedelta(days=1, minutes=1)
            latest_snapshot = self.data[self.data.index >= past24hr]
            if latest_snapshot.empty:
                return 0.0
            return prec_round(
                (latest_snapshot.iloc[-1]["assets_value"] - latest_snapshot.iloc[0]["assets_value"])
                / latest_snapshot.iloc[0]["assets_value"],
                6,
            )

        except Exception:
            return 0.0

    @property
    def past_24hr_apr(self):
        return prec_round(self.past_24hr_roi * 365, 6)

    def past_ndays_roi(self, days: int):
        try:
            if self.last_updated_at is None:
                return 0.0
            lastest_tick = self.last_updated_at
            pastnd = lastest_tick - relativedelta(days=days, minutes=1)
            latest_snapshot = self.data[self.data.index >= pastnd]
            if latest_snapshot.empty:
                return 0.0
            return prec_round(
                (latest_snapshot.iloc[-1]["assets_value"] - latest_snapshot.iloc[0]["assets_value"])
                / latest_snapshot.iloc[0]["assets_value"],
                6,
            )

        except Exception:
            return 0.0

    @property
    def gross_profit(self):
        try:
            # if self.data[self.data['pnl_net'] > 0].empty:
            #     return 0
            # print('gross_profit....',self.data[self.data["pnl_net"] > 0].sum())
            # portfolio_gross_profit = prec_round(float(self.data[self.data["pnl_net"] > 0]["pnl_net"].sum()), 4)
            # trade_gross_profit = prec_round(float(self.trades_df[self.trades_df["pnl"] > 0]["pnl"].sum()), 4)
            # print(f"====>  portfolio_gross_profit: {portfolio_gross_profit}; trade_gross_profit: {trade_gross_profit}")
            return prec_round(float(self.data[self.data["pnl_net"] >= 0]["pnl_net"].sum()), 4)
            # return prec_round(float(self.data[self.data["pct_returns"] >= 0]["pct_returns"].sum()), 4)
        except Exception:
            # print(self.data['pnl_net'], str(e),'error-------')
            return 0.0

    @property
    def gross_loss(self):
        try:
            # if self.data[self.data['pnl_net'] < 0].empty:
            #     return 0
            # portfolio_gross_loss = prec_round(float(self.data[self.data["pnl_net"] < 0]["pnl_net"].sum()), 4)
            # trade_gross_loss = prec_round(float(self.trades_df[self.trades_df["pnl"] < 0]["pnl"].sum()), 4)
            # print(f"====> portfolio_gross_loss: {portfolio_gross_loss}; trade_gross_loss: {trade_gross_loss}")
            return prec_round(float(self.data[self.data["pnl_net"] < 0]["pnl_net"].sum()), 4)
            # return prec_round(float(self.data[self.data["pct_returns"] < 0]["pct_returns"].sum()), 4)
        except Exception:
            # print(self.data['pnl_net'], str(e),'error-------')
            return 0.0

    @property
    def profit_factor(self):
        # ret_p = self.data[self.data["pct_returns"] > 0]["pct_returns"].sum()
        # ret_n = self.data[self.data["pct_returns"] < 0]["pct_returns"].sum()
        # if ret_p == 0:
        #     return 0.0
        # if ret_n == 0:
        #     return 1000
        # return prec_round(ret_p / ret_n * -1, 6)
        if self.gross_profit == 0:
            return 0.0
        if self.gross_loss == 0:
            return 1000
        return prec_round(self.gross_profit / self.gross_loss * -1, 6)

    @property
    def trades_profit_factor(self):
        if self.gross_trades_profit == 0:
            return 0.0
        if self.gross_trades_loss == 0:
            return 1000
        return prec_round(self.gross_trades_profit / self.gross_trades_loss * -1, 6)

    def payoff_ratio(self, avg_winning_trades_pnl, avg_losing_trades_pnl):
        """
        The payoff ratio is a financial ratio used to evaluate the potential profit of a trading strategy or investment.
        """
        return prec_round(avg_winning_trades_pnl / abs(avg_losing_trades_pnl), 4) if avg_losing_trades_pnl else 0.0

    @property
    def return_on_initial_capital(self):
        return prec_round(
            self.total_net_pnl / self.initial_capital if self.initial_capital else 0.0,
            6,
        )

    @property
    def return_on_capital(self):
        return prec_round(self.total_net_pnl / self.capital if self.capital else 0.0, 6)

    @property
    def return_on_investment(self):
        return prec_round(
            self.total_net_pnl / self.net_investment if self.net_investment else 0.0,
            6,
        )

    @property
    def trading_period(self):
        if self.diff:
            return f"{self.diff.years} years {self.diff.months} months {self.diff.days} days {self.diff.hours} hours"
        return "N/A"

    @property
    def avg_holding_period(self):
        if self.trades_df.empty:
            return 0.0

        trades_dt: dict[str, dict] = {}
        ttl_secs = []
        ttl_hrs = []
        ttl_days = []
        for _, trade in self.trades_df.iterrows():
            symbol = trade["symbol"]
            position_side = trade["position_side"]
            action = trade["action"]
            if not action:
                break
            qty = trade["quantity"]
            # is_open = trade['is_open']
            # traded_at = parse(trade['traded_at'])
            traded_at = trade["traded_at"]
            key = f"{symbol}|{position_side}|{action}"
            if "OPEN" in action:
                if trades_dt.get(key) and trades_dt[key].get("qty"):
                    trades_dt[key]["qty"] += qty
                    trades_dt[key]["qty"] = round(trades_dt[key]["qty"], 6)
                    trades_dt[key]["traded_at"] = traded_at
                else:
                    #             print(f'=====> item:{trades_dt[key] if key in trades_dt else None}; qty:{qty}; traded_at:{traded_at}')
                    trades_dt[key] = {
                        "qty": qty,
                        "traded_at": traded_at,
                    }
            else:
                if key.replace("CLOSE", "OPEN") not in trades_dt:
                    # print(f"     [ERROR - avg_holding_period] trades_dt:{trades_dt}; key:{key}; trade:{trade}")
                    # import sys
                    # sys.exit()
                    continue
                item = trades_dt[key.replace("CLOSE", "OPEN")]
                item["qty"] -= qty
                item["qty"] = round(item["qty"], 6)
                prev_traded_at = item["traded_at"]
                t = (traded_at - prev_traded_at).total_seconds()
                ttl_secs.append(t)
                ttl_hrs.append(t / 60 / 60)
                ttl_days.append(t / 60 / 60 / 24)
                # print(f"key:{key} - qty:{qty} closed after {t}hrs | {f'{prev_traded_at} - {traded_at}' if t > 100 else None}")

        hld_hrs = np.mean(ttl_hrs)
        # hld_days = np.mean(ttl_days)

        return prec_round(hld_hrs, 1)  # , prec_round(hld_days, 1)

    @property
    def pct_time_in_market(self):
        """
        The percentage of days in which the strategy is not completely holding cash.
        """
        # position_count
        if self.data.empty:
            return np.nan
        position_count = self.data["position_count"]
        return position_count[position_count != 0].count() / position_count.count()

    @property
    def number_trades(self):
        return len(self.trades_df)

    @property
    def number_closed_trades(self):
        # only count closed_trades
        # return len(self.trades_df)
        # return len(self.closed_trades)
        if self.trades_df.empty:
            return 0.0
        # return len(self.trades_df[self.trades_df['pnl'] != 0.0])
        return len(self.trades_df[~self.trades_df["is_open"]])

    @property
    def number_long_closed_trades(self):
        if self.trades_df.empty:
            return 0.0
        return len(
            self.trades_df[(~self.trades_df["is_open"]) & (self.trades_df["action"].str.contains("LONG", na=False))]
        )

    @property
    def number_short_closed_trades(self):
        if self.trades_df.empty:
            return 0.0
        return len(
            self.trades_df[(~self.trades_df["is_open"]) & (self.trades_df["action"].str.contains("SHORT", na=False))]
        )

    @property
    def num_liquidation_trades(self):
        if self.trades_df.empty:
            return 0.0
        # print(self.trades_df["execution_type"] =='ExecutionType.LIQUIDATION','????')
        return (self.trades_df["execution_type"] == ExecutionType.LIQUIDATION).sum()

    @property
    def num_winning_trades(self):
        if self.trades_df.empty:
            return 0.0
        return (self.trades_df["pnl"] > 0).sum()
        # if self.closed_trades.empty:
        #     return 0
        # return (self.closed_trades["pnl"] > 0).sum()

    @property
    def num_winning_long_trades(self):
        if self.trades_df.empty:
            return 0.0
        trades = self.trades_df[self.trades_df["action"].str.contains("LONG", na=False)]
        return (trades["pnl"] > 0).sum()

    @property
    def num_winning_short_trades(self):
        if self.trades_df.empty:
            return 0.0
        trades = self.trades_df[self.trades_df["action"].str.contains("SHORT", na=False)]
        return (trades["pnl"] > 0).sum()

    @property
    def num_losing_trades(self):
        if self.trades_df.empty:
            return 0
        return (self.trades_df["pnl"] < 0).sum()

    @property
    def num_losing_long_trades(self):
        if self.trades_df.empty:
            return 0.0
        trades = self.trades_df[self.trades_df["action"].str.contains("LONG", na=False)]
        return (trades["pnl"] < 0).sum()

    @property
    def num_losing_short_trades(self):
        if self.trades_df.empty:
            return 0.0
        trades = self.trades_df[self.trades_df["action"].str.contains("SHORT", na=False)]
        return (trades["pnl"] < 0).sum()

    @property
    def gross_trades_profit(self):
        try:
            if self.trades_df.empty:
                return 0.0
            return prec_round(float(self.trades_df[self.trades_df["pnl"] > 0]["pnl"].sum()), 4)
        except Exception:
            # print(self.data['pnl_net'], str(e),'error-------')
            return 0.0

    @property
    def gross_trades_loss(self):
        try:
            if self.trades_df.empty:
                return 0.0
            return prec_round(float(self.trades_df[self.trades_df["pnl"] < 0]["pnl"].sum()), 4)
        except Exception:
            return 0.0

    @property
    def gross_winning_trades_amount(self):
        try:
            if self.trades_df.empty:
                return 0.0
            return prec_round(
                float(self.trades_df[self.trades_df["pnl"] > 0]["gross_amount"].sum()),
                4,
            )
        except Exception:
            # print(self.data['pnl_net'], str(e),'error-------')
            return 0.0

    @property
    def gross_losing_trades_amount(self):
        try:
            if self.trades_df.empty:
                return 0.0
            return prec_round(
                float(self.trades_df[self.trades_df["pnl"] < 0]["gross_amount"].sum()),
                4,
            )
        except Exception:
            return 0.0

    @property
    def largest_profit_winning_trade(self):
        try:
            if self.trades_df.empty:
                return 0.0
            return prec_round(float(self.trades_df[self.trades_df["pnl"] > 0]["pnl"].max()), 4)
        except Exception:
            return 0.0

    @property
    def largest_loss_losing_trade(self):
        try:
            if self.trades_df.empty:
                return 0.0
            return prec_round(float(self.trades_df[self.trades_df["pnl"] < 0]["pnl"].min()), 4)
        except Exception:
            return 0.0

    @property
    def avg_pnl_per_trade(self):
        # if len(self.closed_trades) > 1:
        #     # trades = self.closed_trades[self.closed_trades['pnl'] != 0]
        #     trades = self.closed_trades[self.closed_trades['exited_at'].notnull()]
        #     if len(trades) > 1:
        #         return trades["pnl"].sum() / len(trades)

        # return 0.0
        if self.trades_df.empty:
            return 0.0
        # trades = self.trades_df[self.trades_df['pnl'] != 0.0]
        trades = self.trades_df[~self.trades_df["is_open"]]
        if len(trades) >= 1:
            return prec_round(trades["pnl"].sum() / len(trades), 4)

        return 0.0

    @property
    def avg_amount_per_open_trade(self):
        if self.trades_df.empty:
            return 0.0
        trades = self.trades_df[self.trades_df["is_open"]]
        if len(trades) >= 1:
            return prec_round(trades["gross_amount"].sum() / len(trades), 4)

        return 0.0

    @property
    def avg_amount_per_closed_trade(self):
        if self.trades_df.empty:
            return 0.0
        # trades = self.trades_df[self.trades_df['pnl'] != 0.0]
        trades = self.trades_df[~self.trades_df["is_open"]]
        if len(trades) >= 1:
            return prec_round(trades["gross_amount"].sum() / len(trades), 4)

        return 0.0

    @property
    def avg_amount_per_trade(self):
        if self.trades_df.empty:
            return 0.0
        trades = self.trades_df
        if len(trades) >= 1:
            return prec_round(trades["gross_amount"].sum() / len(trades), 4)
        return 0.0

    @property
    def avg_amount_per_long_trade(self):
        if self.trades_df.empty:
            return 0.0
        trades = self.trades_df[self.trades_df["action"].str.contains("LONG", na=False)]
        if len(trades) >= 1:
            return prec_round(trades["gross_amount"].sum() / len(trades), 4)
        return 0.0

    @property
    def avg_amount_per_short_trade(self):
        if self.trades_df.empty:
            return 0.0
        trades = self.trades_df[self.trades_df["action"].str.contains("SHORT", na=False)]
        if len(trades) >= 1:
            return prec_round(trades["gross_amount"].sum() / len(trades), 4)
        return 0.0

    @property
    def avg_pnl_per_trade_in_percentile(self):
        return prec_round(
            self.avg_pnl_per_trade / self.capital if self.capital else 0.0, 4
        )  # FIXME: capital OR net_investment

    def max_consecutive_trades(self):
        if self.trades_df.empty:
            return 0.0, 0.0
        trades = self.trades_df[~self.trades_df["is_open"]]
        if len(trades) >= 1:
            counter = defaultdict(list)
            for key, val in groupby(
                trades["pnl"].to_list(),
                lambda ele: "plus" if ele > 0 else "minus" if ele < 0 else "-",
            ):
                counter[key].append(len(list(val)))

            return (
                max(counter["plus"]) if counter["plus"] else 0,
                max(counter["minus"]) if counter["minus"] else 0,
            )

        return 0.0, 0.0

    @property
    def sqn(self):
        """
        SQN or SystemQualityNumber. Defined by Van K. Tharp to categorize trading systems.

            - 1.6 - 1.9 Below average
            - 2.0 - 2.4 Average
            - 2.5 - 2.9 Good
            - 3.0 - 5.0 Excellent
            - 5.1 - 6.9 Superb
            - 7.0 -     Holy Grail?

            The formula:

            - SquareRoot(NumberTrades) * Average(TradesProfit) / StdDev(TradesProfit)

            The sqn value should be deemed reliable when the number of trades >= 30

            Methods:

            - get_analysis

                Returns a dictionary with keys "sqn" and "trades" (number of
                considered trades)

            Total Trades = TradeID count
            Wins % = % of wins using Total Trades (Win: PNL % >0)
            Avg Win = Avg of PNL% of the winning trades
            Losses % = % of losses using Total Trades (Loss: PNL % <=0)
            Avg Loss = Avg of PNL% of the Losing trades
            Expectancy = (Wins % * Avg Win) - (Losses % * Avg Loss)

            PNL % in R = PNL % / 25
            SQN = Squareroot (Total Trades) * Avg (PNL % in R) / Std Dev (PNL % in R)

            Assuming a set of N trades (N>30 for being statistically significant), SQN is defined as follow:

            ----
            SQN= Squareroot(N) * Average (of the N Profit&Loss) / Std dev (of the N Profit&Loss).

            The large the N, the more trading opportunities you have.
            The large the average P&L, the better you are obviously.
            The smaller the Std dev (P&L), the more regular are your results and the smaller are the drawdowns.

            Note here that if you optimize for the largest SQN, you maximize in fact the product N*average P&L and you minimize the Std dev (P&L) and the drawdowns at the same time.
        """

        # if len(self.pnls) > 1:
        #     pnl_av = self.pnls["pnl_daily"].mean()
        #     pnl_stddev = self.pnls["pnl_daily"].std()
        #     try:
        #         sqn = sqrt(len(self.pnls)) * pnl_av / pnl_stddev
        #     except ZeroDivisionError:
        #         sqn = None
        # else:
        #     sqn = 0
        # return sqn

        # TODO: cases that has buy trade but hvnt close
        # if len(self.closed_trades) > 1:

        #     pnl_av = self.closed_trades["pnl"].mean()
        #     pnl_stddev = self.closed_trades["pnl"].std()
        #     try:
        #         sqn = sqrt(len(self.closed_trades)) * pnl_av / pnl_stddev
        #     except ZeroDivisionError:
        #         sqn = None
        # else:
        #     sqn = 0
        # return sqn

        # if len(self.closed_trades) > 1:
        #     # trades = self.closed_trades[self.closed_trades['pnl'] != 0]
        #     trades = self.closed_trades[self.closed_trades['exited_at'].notnull()]

        if self.trades_df.empty:
            return 0.0

        # trades = self.trades_df[self.trades_df['pnl'] != 0.0]
        trades = self.trades_df[~self.trades_df["is_open"]]
        if len(trades) > 1:
            pnl_avg = trades["pnl"].mean()
            pnl_stddev = trades["pnl"].std()
            # print(f"[DEBUG] pnl_avg: {pnl_avg}; pnl_stddev: {pnl_stddev}; # of trades:{len(trades)}")
            try:
                sqn = sqrt(len(trades)) * pnl_avg / pnl_stddev if pnl_stddev else 0
            except ZeroDivisionError:
                sqn = None
            # print(f'sqn ============> std:{pnl_stddev}; mean:{pnl_avg}; sqn:{sqn}; # of trades:{len(trades)}')
        else:
            sqn = 0.0

        return prec_round(sqn, 6)

    def kelly(self):
        win_ratio = self.win_ratio
        profit_factor = self.profit_factor

        k = 0.0
        if profit_factor:
            k = win_ratio - (1 - win_ratio) / profit_factor
        # print(f"---------> kelly:{k}")
        return k

    def var(self, returns=None, conf_level=0.99, return_var: bool = False):
        """
        Value at Risk (VaR) is a measurement showing a normal distribution of past losses
        VaR(Value at Risk), quantify the risk of potential losses
        calculation methods includes the historical, variance-covariance, and Monte Carlo methods
        E.g. an asset has a 3% one-month VaR of 2%, representing a 3% chance of the asset declining in value by 2% during the one-month time frame.

        https://www.investopedia.com/terms/v/var.asp
        In order to calculate the VaR of a portfolio, you can follow the steps below:

        1. Calculate periodic returns of the stocks in the portfolio
        2. Create a covariance matrix based on the returns
        3. Calculate the portfolio mean and standard deviation
            (weighted based on investment levels of each stock in portfolio)
        4. Calculate the inverse of the normal cumulative distribution (PPF -  percent-point function - Quantile function) with a specified confidence interval, standard deviation, and mean
        5. Estimate the value at risk (VaR) for the portfolio by subtracting the initial investment from the calculation in step (4)

        !!!!!!!!! “VaR measures the worst expected loss over a given horizon under normal market conditions at a given level of confidence”.
        Example:
        Suppose, an analyst says that the 1-day VaR of a portfolio is 1 million dollar (var value is -0.82%), with a 95% confidence level.

        It implies there is 95% chance that the maximum losses will not exceed 1 million dollar( or 0.82% of your portfolio) in a single day.
        In other words, there is only 5% chance that the portfolio losses on a particular day will be greater than 1 million dollar( or 0.82% of your portfolio).

        def value_at_risk(returns, sigma=1, confidence=0.95, prepare_returns=True):
            # Calculats the daily value-at-risk
            # (variance-covariance calculation with confidence n)
            if prepare_returns:
                returns = _utils._prepare_returns(returns)
            mu = returns.mean()
            sigma *= returns.std()

            if confidence > 1:
                confidence = confidence/100

            return _norm.ppf(1-confidence, mu, sigma)

        !!!!!!
        If a stock has a VaR of -5% with a confidence level of 90%,
        it means its losses will not exceed -5% with a 90% probability in a given day based on historical values.
        !!!!!!

            Est. Daily Maximum Loss
        1 - 2%
        2 - 5%
        3 - 10%
        4 - 17%
        5 - 27%
        6 - 40%
        7 - 60%
        8 - 90%
        9 - >90%
        """
        if self.trades_df.empty:
            if return_var:
                return 0.0, 0.0
            return 0
        if returns is None:
            returns = self.data["pct_returns"]

        var_range_dict = {
            1: (None, 0.02),
            2: (0.02, 0.05),
            3: (0.05, 0.1),
            4: (0.1, 0.17),
            5: (0.17, 0.27),
            6: (0.27, 0.4),
            7: (0.4, 0.6),
            8: (0.6, 0.9),
            9: (0.9, None),
        }

        stdev = np.nanstd(returns, axis=0)
        mean = np.nanmean(returns, axis=0)

        # conf_level1 = 1 - conf_level  # 0.05

        # vvv1 = returns.quantile(conf_level1)

        # z_value = norm.ppf(conf_level1)
        z_score_lookup = {
            0.90: 1.282,
            0.91: 1.341,
            0.92: 1.405,
            0.93: 1.476,
            0.94: 1.555,
            0.95: 1.645,
            0.96: 1.751,
            0.97: 1.881,
            0.98: 2.054,
            0.99: 2.326,
        }
        z_value = z_score_lookup.get(conf_level, None)
        if z_value is None:
            print("confidence level is not in the calculation range")
            if return_var:
                return np.nan, np.nan
            return np.nan
        else:
            z_value *= -1

        var_p = z_value * stdev - mean
        # var_p1 = norm.ppf(conf_level1, loc=mean, scale=stdev)
        # print(f"var_p = {var_p}; var_p1 = {var_p1}; mean={mean}; stdev:{stdev}")
        # risk_score = prec_round(var_p * -1 * 100, 2)
        var_prob = var_p * -1
        risk_score = None
        for risk_score, var_range in var_range_dict.items():
            left, right = var_range
            if left and right and left < var_prob <= right:
                break
            elif left and not right and var_prob > left:
                break
            elif not left and right and var_prob <= right:
                break

        if return_var:
            # Calculate mean of investment
            mean_investment = (1 + mean) * self.capital
            # Calculate standard deviation of investmnet
            stdev_investment = self.capital * stdev

            # cutoff1 = norm.ppf(conf_level1, loc=mean_investment, scale=stdev_investment)
            cutoff1 = z_value * stdev_investment + mean_investment

            var_1d1 = prec_round(self.capital - cutoff1, 4)

            # vvv = -self.capital * (z_value * stdev + mean)  #### result same as var_1d1

            # print(f"var_1d1: {var_1d1} ({round(var_1d1/self.capital, 5)}); risk_score: {risk_score}; cutoff1: {cutoff1}; var_p:{var_p}; mean_investment:{mean_investment}; stdev_investment:{stdev_investment} | mean:{mean}; std:{stdev}")

            # var_array = []
            # num_days = int(15)
            # for x in range(1, num_days+1):
            #     var_d = prec_round(var_1d1 * np.sqrt(x),2)
            #     var_array.append(var_d)
            #     print(f"{x} day VaR @ 95% confidence: {var_d}")
            return var_1d1, risk_score
        else:
            return risk_score

    def cvar(self, returns=None, conf_level=0.99):
        """
        conditional_value_at_risk. Aka. cvar, expected_shortfall
        Calculats the conditional daily value-at-risk (aka expected shortfall)
        quantifies the return of tail risk an investment
        """
        if self.trades_df.empty:
            return 0
        if returns is None:
            returns = self.data["pct_returns"]

        var = self.var(returns=returns, conf_level=conf_level)
        c_var = returns[returns < var].values.mean()  ## TODO: so many 0.0 will lower the
        return c_var if ~np.isnan(c_var) else var

    @property
    def avg_daily_risk_score(self):
        return prec_round(self.data["risk_scores"].mean(), 2)

    @property
    def avg_risk_score_past_7days(self):
        return prec_round(self.data["risk_scores"].tail(7).mean(), 2)

    @property
    def monthly_avg_risk_score(self):
        """
        return {
                '2021-09-30': 4.34,
                '2021-10-31': 2.543,
                '2021-11-30': 5.543,
                '2021-12-31': 2.342,
                '2022-01-31': 1.443,
                '2022-02-28': 3.432,
                '2022-03-31': 8.325,
                '2022-04-30': 4.543}
        """
        # returns = self.data_resample_monthly[['risk_scores', 'datetime']].copy()
        # returns['datetime_s'] = returns['datetime'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        # return returns.set_index('datetime_s').to_dict()['risk_scores']

        # monthly average
        rs = self.data[["risk_scores", "datetime"]].copy()
        returns = rs.groupby(pd.PeriodIndex(rs.index, freq="M")).agg(
            {"risk_scores": "mean", "datetime": "last"}
        )  # ['risk_scores'].mean()
        returns["datetime_s"] = returns["datetime"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        # print(returns,'<====')
        return returns.round(2).set_index("datetime_s").to_dict()["risk_scores"]

    @property
    def daily_changes(self):
        # TODO: self.data_resample_daily
        # returns = self.data[['pct_returns']].copy()
        if self.data_resample_daily.empty:
            return {}
        returns = self.data_resample_daily[["pct_returns"]].copy()
        returns["datetime_s"] = returns.index.to_series().apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        # print(returns.set_index('datetime_s').to_dict()['pct_returns'])
        return returns.set_index("datetime_s").to_dict()["pct_returns"]

    @property
    def asset_values(self):
        if self.data_resample_daily.empty:
            return {}
        returns = self.data_resample_daily[["assets_value"]].copy()
        returns["datetime_s"] = returns.index.to_series().apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        return returns.set_index("datetime_s").to_dict()["assets_value"]

    @property
    def monthly_changes(self):
        """
        return {
            '2021-09-30': 4.543,
            '2021-10-31': -2.543,
            '2021-11-30': 5.543,
            '2021-12-31': 2.342,
            '2022-01-31': 1.443,
            '2022-02-28': -3.432,
            '2022-03-31': -0.325,
            '2022-04-30': 4.543}
        """
        # print(self.data_resample_monthly,'!!!')
        if self.data_resample_monthly.empty:
            return {}
        returns = self.data_resample_monthly[["pct_returns", "datetime"]].copy()
        # print(returns,'????')
        returns["datetime_s"] = returns["datetime"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        # print(returns.set_index('datetime_s').to_dict()['pct_returns'])
        return returns.set_index("datetime_s").to_dict()["pct_returns"]

    @property
    def frequently_traded(self):
        """
        [
            {
                'symbol': 'BTC/USDT',
                'asset_type': 'SPOT',
                'number_trades': 12,
                'avg_profit': 0.25,
                'avg_loss': 0.15,
                'profitable': 0.75, # Profitable: count all trade per symbol and count all trade of positive return
            },
            {
                'symbol': 'ETH/USDT',
                'asset_type': 'SPOT',
                'number_trades': 63,
                'avg_profit': 0.195,
                'avg_loss': 0.13,
                'profitable': 0.761}]
        # avg_profit  = sum(pnl+) / sum(closed_trade_amount)
        # avg_loss  = sum(pnl-) / sum(closed_trade_amount)
        # profitable = num of pnl+ trades / num of closed trades

        'profitable': 0.75, # Profitable: count all trade per symbol and count all trade of positive return

        """
        trades = self.trades_df
        ret = []
        if not trades.empty:
            trades["asset_type"] = trades["asset_type"].apply(lambda x: x if isinstance(x, str) else x.value)
            # print(trades.groupby(['symbol','asset_type']).agg({'quantity':'sum'}),'????')
            gts = (
                trades.groupby(["symbol", "asset_type"]).agg({"side": "count"}).sort_values("side", ascending=False)[:3]
            )

            for i, cnt_row in gts.iterrows():
                count = int(cnt_row["side"])
                symbol, asset_type = i
                fts = trades[(~trades["is_open"]) & (trades["symbol"] == symbol) & (trades["asset_type"] == asset_type)]
                sum_pnl_p = 0
                sum_pnl_n = 0
                sum_close_trade_amt = 0
                cnt_close_trade = 0
                cnt_pnl_p_trade = 0

                for j, row in fts.iterrows():
                    pnl = row["pnl"]
                    px = row["price"]
                    qty = row["quantity"]
                    sum_close_trade_amt += px * qty
                    cnt_close_trade += 1
                    if pnl > 0:
                        cnt_pnl_p_trade += 1
                        sum_pnl_p += pnl
                    elif pnl < 0:
                        sum_pnl_n += pnl
                avg_profit = sum_pnl_p / sum_close_trade_amt if sum_close_trade_amt else sum_close_trade_amt
                avg_loss = sum_pnl_n / sum_close_trade_amt if sum_close_trade_amt else sum_close_trade_amt
                profitable = cnt_pnl_p_trade / cnt_close_trade if cnt_close_trade else cnt_close_trade
                # print(f"symbol: {symbol}, asset_type: {asset_type}; count:{count} sum_pnl_n:{sum_pnl_n}, sum_pnl_p:{sum_pnl_p}, sum_close_trade_amt: {sum_close_trade_amt}; cnt_close_trade:{cnt_close_trade}; cnt_pnl_p_trade: {cnt_pnl_p_trade}")
                ret.append(
                    {
                        "symbol": symbol,
                        "asset_type": asset_type,
                        "number_trades": count,
                        "avg_profit": prec_round(avg_profit, 4),
                        "avg_loss": prec_round(avg_loss, 4),
                        "profitable": prec_round(profitable, 6),
                    }
                )

        return ret

    def _generate_quantstats(
        self,
        risk_free=0.0,
        annualization=None,
        sigma_value=1.0,
        confidence_value=0.95,
        path: str | None = None,
    ):
        try:
            import quantstats as qs

            # from empyrical import (alpha_beta, cagr,  # information_ratio
            #                        cum_returns, max_drawdown, sortino_ratio,
            #                        stability_of_timeseries, tail_ratio)
        except ImportError:
            print(
                "Reminder: This test required package 'quantstats' & 'empyrical'. "
                "pip install quantstats==0.0.62 empyrical"
            )
            return
        if self.data.empty:
            return pd.Series(dtype="object", name="value")
        returns = self.data["pct_returns"]
        returns, benchmark = _aligned_series(returns, self.data["benchmark_pct_returns"])

        # returns.index = returns.index.to_pydatetime()
        returns.index = returns.index.tz_convert(None)
        benchmark.index = benchmark.index.tz_convert(None)

        if path:
            qs.reports.html(
                returns=returns,
                benchmark=benchmark,
                periods_per_year=annualization,
                output=path,
                compounded=True,
            )
            return
        # print(f">>>> annualization:{annualization}")
        stats = qs.reports.metrics(
            returns,
            benchmark,
            periods_per_year=annualization,
            compounded=True,
            display=False,
            mode="full",
        )

        return stats

    def _generate_portfolio_df_lite(self):
        df = pd.DataFrame(
            {
                "available_balance": self._available_balance_hist,
                "holding_value": self._hld_val_hist,
                "assets_value": self._ast_val_hist,
                "position_count": self._pos_count_hist,
                "total_commission": self._total_commission,
                "total_turnover": self._total_turnover,
                "total_realized_pnl": self._realized_pnl,
                "total_unrealized_pnl": self._unrealized_pnl,
                "total_net_investment": self._net_investment,
                "total_pnl": self._total_pnl,
                "navs": self._navs,
                "positions": self._positions,
            },
            index=self._date_hist,
        )
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.astype(str)
            df.index = df.index.str.split("+").str[0].str.split(".").str[0]
            df.index = pd.to_datetime(df.index)
        if df.index.tz:
            df.index = df.index.tz_convert(tz_manager.tz)
        else:
            df.index = df.index.tz_localize(tz_manager.tz)

        df["datetime"] = df.index
        if not df.empty:
            if self.resample_rule:
                df = df[~df.index.duplicated(keep="last")]
                last_index = df.index[-1]
                df = (
                    df.resample(self.resample_rule, label="right", closed="right")
                    .agg(resample_agg_gen(df.columns))
                    .ffill()
                    .fillna(0)
                    .round(5)
                )
                df = df.rename(index={df.index[-1]: last_index})

            df["datetime"] = df.index
            df.index.name = "date"

            correct_asset_value = pd.concat([pd.Series(self.initial_capital), df["assets_value"]])
            df["pct_returns"] = correct_asset_value.pct_change()[1:]
            df["pnl_net"] = (correct_asset_value - correct_asset_value.shift(1))[1:]  # NOTE: T0 - T-1   (daily pnl)
        else:
            df["pct_returns"] = np.nan
            df["pnl_net"] = np.nan

        df["cum_returns"] = self.cum_returns(df["pct_returns"])
        df["pnl_cumsum"] = df["pnl_net"].cumsum()

        if not df.empty:
            df = df.fillna(0)
            df = df[~df.index.duplicated(keep="last")]
            df = df.round(5)

        self.data = df

        self.trades_df = pd.DataFrame(list(self.trade_records.values()))
        if not self.trades_df.empty:
            self.trades_df.traded_at = pd.to_datetime(self.trades_df.traded_at).dt.tz_convert(tz_manager.tz)
            self.trades_df.index = self.trades_df.traded_at

    def _generate_portfolio_df(self, benchmark: pd.DataFrame | None = None):
        df = pd.DataFrame(
            {
                "available_balance": self._available_balance_hist,
                "holding_value": self._hld_val_hist,
                "assets_value": self._ast_val_hist,
                "position_count": self._pos_count_hist,
                "total_commission": self._total_commission,
                "total_turnover": self._total_turnover,
                "total_realized_pnl": self._realized_pnl,
                "total_unrealized_pnl": self._unrealized_pnl,
                "total_net_investment": self._net_investment,
                "total_pnl": self._total_pnl,
                "navs": self._navs,
                "positions": self._positions,
            },
            index=self._date_hist,
        )
        # print(f"[_generate_portfolio_df] df:{df} | df.index:{df.index}")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.astype(str)
            df.index = df.index.str.split("+").str[0].str.split(".").str[0]
            df.index = pd.to_datetime(df.index)
        # print(df.index, df.index.tz)
        if df.index.tz:
            df.index = df.index.tz_convert(tz_manager.tz)
        else:
            df.index = df.index.tz_localize(tz_manager.tz)

        # print(f"1111>>>>> df:\n{df}")

        # self.data = self._construct_data(df_raw=df, benchmark_raw=benchmark)
        """
        pct_returns: total asset percentage change on ticks
        pnl_net: total asset change on ticks
        total_pnl: = total_realized_pnl + total_unrealized_pnl
        pnl_cumsum: cumulative pnl on every ticks = total_pnl - total_commission
        # INP DF COLUMNS: available_balance  holding_value   assets_value  position_count  total_commission  total_realized_pnl  total_unrealized_pnl  total_net_investment     total_pnl  positions
        # OUT COLUMNS: available_balance  holding_value   assets_value  position_count  total_commission  total_realized_pnl  total_unrealized_pnl  total_net_investment     total_pnl  positions  pct_returns     pnl_net    pnl_cumsum  benchmark_pct_returns  benchmark   datetime
        """

        df["datetime"] = df.index

        has_benchmark = benchmark is not None and not benchmark.empty
        if has_benchmark:
            bm_close = benchmark["close"]
            # print(f"has_benchmark:{has_benchmark} !!! bm_close:{bm_close}")
            # df = df.merge(bm_close, how="left", left_index=True, right_index=True)
            # print(df,'<<<<<<<<<<<<,')
            # print(bm_close,'<<<<<<<<<<')
            df = df.sort_index()
            df = pd.merge_asof(
                df,
                bm_close,
                left_index=True,
                right_index=True,
                allow_exact_matches=True,
                direction="nearest",
            )
            df.rename(columns={"close": "benchmark_close"}, inplace=True)
            # print(df,'---1111---',df.index)

        # print(df,'---1111---',df.index)
        if not df.empty:
            if self.resample_rule:
                df = df[~df.index.duplicated(keep="last")]
                last_index = df.index[-1]
                df = (
                    df.resample(self.resample_rule, label="right", closed="right")
                    .agg(resample_agg_gen(df.columns))
                    .ffill()
                    .fillna(0)
                    .round(5)
                )
                df = df.rename(index={df.index[-1]: last_index})

                # print(f'!!!!!!!!!\n{df.index}\n{df.datetime}')
            df["datetime"] = df.index
            df.index.name = "date"

            # beginning_balance = beginning_balance if beginning_balance else self.initial_capital
            # correct_asset_value = pd.concat([pd.Series(beginning_balance), df["assets_value"]])
            correct_asset_value = pd.concat([pd.Series(self.initial_capital), df["assets_value"]])
            # print(f"---> correct_asset_value:{correct_asset_value}")
            df["pct_returns"] = correct_asset_value.pct_change()[1:]
            df["log_returns"] = np.log(correct_asset_value / correct_asset_value.shift(1))[1:]
            df["pnl_net"] = (correct_asset_value - correct_asset_value.shift(1))[1:]  # NOTE: T0 - T-1   (daily pnl)
            # df["pct_returns"] =  df["assets_value"].pct_change().fillna(0)
            # df["pnl_net"] = df["assets_value"] - df["assets_value"].shift(1) # T0 - T-1
            # df["pnl_net_pct_returns"] = df["pnl_net"].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
        else:
            df["pct_returns"] = np.nan
            df["log_returns"] = np.nan
            df["pnl_net"] = np.nan

        # df["cum_returns"] = self.cum_returns(df["pct_returns"])
        df["cum_returns"] = np.exp(self.cum_returns(df["log_returns"])) - 1
        df["pnl_cumsum"] = df["pnl_net"].cumsum()
        df["risk_scores"] = (
            df["pct_returns"].expanding().apply(self.var)
        )  # expanding is similar to rolling, only diff is the window size is growing
        # df["rolling_sharp"] = df["pct_returns"].rolling(window=60).apply(lambda x: self.sharpe_ratio(returns=x, annualization=annualization))  if has_benchmark else 0.0
        # df["rolling_volatility"] = df["pct_returns"].rolling(window=60).apply(lambda x: self.annualized_volatility(returns=x, annualization=annualization)) if has_benchmark else 0.0
        # df["rolling_alpha"] = df["pct_returns"].rolling(window=60).apply(lambda x: self.alpha(returns=x, factor_returns=, annualization=annualization)) if has_benchmark else 0.0
        # df["rolling_beta"] = df["pct_returns"].rolling(window=60).apply(lambda x: self.beta(returns=x, factor_returns=, annualization=annualization)) if has_benchmark else 0.0

        if has_benchmark:
            df["benchmark_pct_returns"] = df["benchmark_close"].pct_change()
            df["benchmark_log_returns"] = np.log(df["benchmark_close"] / df["benchmark_close"].shift(1))[1:]
            # df["benchmark"] = df["benchmark_pct_returns"].cumsum()
            df["benchmark"] = np.exp(df["benchmark_log_returns"].cumsum()) - 1

        if not df.empty:
            # print(df,'!!!!1111!!',len(df))
            df = df.fillna(0)
            df = df[~df.index.duplicated(keep="last")]
            last_index = df.index[-1]
            # print(df,'!!!22222!!!',len(df))
            if self.resample_rule == "1D":
                self.data_resample_daily = df
            else:
                self.data_resample_daily = (
                    df.resample("D", label="right", closed="right")
                    .agg(resample_agg_gen(df.columns))
                    .ffill()
                    .fillna(0)
                    .round(5)
                )
                self.data_resample_daily = self.data_resample_daily.rename(
                    index={self.data_resample_daily.index[-1]: last_index}
                )

            # self.data_resample_weekly = df_raw.resample('W', closed='right').ffill().fillna(0)
            # self.data_resample_weekly['pct_change'] = self.data_resample_weekly['assets_value'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)

            self.data_resample_monthly = (
                df.resample("ME", label="right", closed="right")
                .agg(resample_agg_gen(df.columns))
                .ffill()
                .bfill()
                .fillna(0)
                .copy()
            )
            self.data_resample_monthly = self.data_resample_monthly.rename(
                index={self.data_resample_monthly.index[-1]: last_index}
            )
            # self.data_resample_monthly["pct_returns"] = self.data_resample_monthly["assets_value"].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
            # # self.data_resample_monthly['risk_scores'] = self.data_resample_monthly['pct_returns'].expanding().apply(self.var)
            # self.data_resample_monthly["pnl_net"] = self.data_resample_monthly["assets_value"] - self.data_resample_monthly["assets_value"].shift(1)
            correct_asset_value = pd.concat(
                [
                    pd.Series(self.initial_capital),
                    self.data_resample_monthly["assets_value"],
                ]
            )
            self.data_resample_monthly["pct_returns"] = correct_asset_value.pct_change()[1:]
            self.data_resample_monthly["pnl_net"] = (correct_asset_value - correct_asset_value.shift(1))[
                1:
            ]  # NOTE: T0 - T-1   (daily pnl)
            # print(correct_asset_value,'||||', self.data_resample_monthly["pct_returns"], )

            # self.data_resample_monthly["pnl_net_pct_returns"] = self.data_resample_monthly["pnl_net"].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
            self.data_resample_monthly["dt_index"] = self.data_resample_monthly.index
            self.data_resample_monthly = self.data_resample_monthly.fillna(0).round(5)
            # print(f"!!! self.data_resample_monthly:{self.data_resample_monthly}")

            # self.data_resample_yearly = df_raw.resample('Y', closed='right').ffill().fillna(0)
            # self.data_resample_yearly['pct_change'] = self.data_resample_yearly['assets_value'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)

            # print(self.data_resample_daily,'------')
            df = df.round(5)
        # print(f">>>>> resample_rule:{self.resample_rule}; df:\n{df}")
        self.data = df

        self.trades_df = pd.DataFrame(list(self.trade_records.values()))
        if not self.trades_df.empty:
            # self.trades_df = self.trades_df.set_index(["traded_at"], drop=False)
            # self.trades_df.index = pd.to_datetime(self.trades_df.index, utc=True)
            # self.trades_df.index = pd.to_datetime(self.trades_df.index, utc=True, format="mixed")

            # print(f">>> trades_df.traded_at:{self.trades_df.traded_at}")
            # print(f">>> trades_df:{self.trades_df}")
            # self.trades_df.traded_at = pd.to_datetime(self.trades_df.traded_at)
            # self.trades_df.index = pd.to_datetime(self.trades_df.index)
            self.trades_df.traded_at = pd.to_datetime(self.trades_df.traded_at).dt.tz_convert(tz_manager.tz)
            self.trades_df.index = self.trades_df.traded_at

    def _get_annualization(self, interval: str | None = None):
        if interval is None:
            interval = self.interval

        annualization = ANNUALIZATION_MAP.get(interval, 365)
        return annualization

    def genereate_stats_lite(
        self,
        interval: str | None = None,
        resample_interval: str | None = None,
        annualization: int | None = None,
        risk_free: float = 0.0,
        required_return: float = 0.0,
        return_dict: bool = False,
    ):
        if not (interval or annualization):
            raise ValueError("interval or annualization is required")
        if annualization:
            if annualization in ANNUALIZATION_MAP_REV:
                self.interval = ANNUALIZATION_MAP_REV[annualization]
            else:
                raise ValueError(f"annualization: {annualization} is not supported")
        self.resample_interval = resample_interval
        if interval:
            self.interval = interval

        self.resample_rule = RESAMPLE_MAP.get(resample_interval) if resample_interval else None
        self.annualization = annualization = (
            annualization
            if annualization
            else self._get_annualization(resample_interval if resample_interval else self.interval)
        )
        self.risk_free = risk_free

        self._generate_portfolio_df_lite()

        stats = pd.Series(dtype="object", name="value")

        ### OVERALL RESULTS
        stats["start"] = self.start
        stats["end"] = self.end
        stats["interval"] = resample_interval if resample_interval else self.interval
        stats["duration"] = self.duration
        stats["trading_instruments"] = list(self.symbols)

        stats["annualized_return"] = self.annualized_return(annualization)
        stats["total_return"] = self.total_return
        stats["total_net_pnl"] = self.total_net_pnl

        stats["avg_holding_period(hrs)"] = self.avg_holding_period

        stats["number_trades"] = self.number_trades
        stats["number_closed_trades"] = self.number_closed_trades
        stats["win_ratio"] = self.win_ratio
        stats["loss_ratio"] = self.loss_ratio
        stats["trades_profit_factor"] = self.trades_profit_factor
        stats["avg_pnl_per_trade($)"] = self.avg_pnl_per_trade

        stats["sharpe_ratio"] = self.sharpe_ratio(risk_free=risk_free, annualization=annualization)
        stats["sortino_ratio"] = self.sortino_ratio(annualization=annualization, required_return=required_return)
        stats["calmar_ratio"] = self.calmar_ratio(annualization=annualization)
        stats["annualized_volatility"] = self.annualized_volatility(annualization=annualization)
        stats["max_drawdown"], _ = self.max_drawdown
        stats["sqn"] = self.sqn

        stats = stats.replace([np.inf, -np.inf, np.nan], [None, None, None])

        if return_dict:
            return json.loads(json.dumps(stats.to_dict(), indent=4, default=set_default))

        return stats

    def genereate_stats(
        self,
        interval: str | None = None,
        benchmark: DataFeed | None = None,
        resample_interval: str | None = None,
        annualization: int | None = None,
        risk_free: float = 0.0,
        required_return: float = 0.0,
        return_dict: bool = False,
        use_qs: bool = False,
    ):
        """
        in realtime:
            returns related - update in daily basis
            trade related - realtime update
                - positions, trades, pnl, end, sqn, frequently_traded
                    start, avg_daily_trades, avg_weekly_trades, avg_monthly_trades, win_ratio, loss_ratio
        TODO:
            duration - number of bars / extact times
            long/short pnl/trades
            number of margin call / liquidation
        """
        if not (interval or annualization):
            raise ValueError("interval or annualization is required")
        if annualization:
            if annualization in ANNUALIZATION_MAP_REV:
                self.interval = ANNUALIZATION_MAP_REV[annualization]
            else:
                raise ValueError(f"annualization: {annualization} is not supported")
        self.resample_interval = resample_interval
        if interval:
            self.interval = interval

        self.resample_rule = RESAMPLE_MAP.get(resample_interval) if resample_interval else None
        # print(f">>>>> PM - genereate_stats, interval:{self.interval}, resample_rule:{self.resample_rule}")
        # print(f"annualization: {annualization} | self._get_annualization():{self._get_annualization()} | interval:{self.interval}; resample_interval:{self.resample_interval}")
        self.annualization = annualization = (
            annualization
            if annualization
            else self._get_annualization(resample_interval if resample_interval else self.interval)
        )
        self.risk_free = risk_free

        has_benchmark = benchmark is not None
        benchmark_df = None
        benchmark_symbol = None
        if has_benchmark and benchmark is not None:
            if hasattr(benchmark, "is_live") and benchmark.is_live:
                start = (
                    self.start.strftime("%Y-%m-%d %H:%M:%S")
                    if self.start
                    else (self.end - timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S")
                )
                end = self.end.strftime("%Y-%m-%d %H:%M:%S")
                benchmark.fetch_data(interval=self.interval, start_date=start, end_date=end)
            benchmark_df = benchmark.dataframe
            benchmark_symbol = benchmark.symbol
            if benchmark_df.empty:
                has_benchmark = False
        # print(f"benchmark_df >>> {benchmark_df} | has_benchmark:{has_benchmark} || self.start:{self.start}; self.end:{self.end}")
        self._generate_portfolio_df(benchmark_df)

        if use_qs:
            stats = self._generate_quantstats(risk_free=risk_free, annualization=annualization)
        else:
            stats = pd.Series(dtype="object", name="value")

            ### OVERALL RESULTS
            stats["start"] = self.start
            stats["end"] = self.end
            stats["interval"] = resample_interval if resample_interval else self.interval
            stats["duration"] = self.duration
            stats["trading_instruments"] = list(self.symbols)
            stats["base_currency"] = self.account.asset if self.account else "N/A"
            stats["benchmark"] = benchmark_symbol if benchmark_symbol else "N/A"

            stats["beginning_balance"] = self.initial_capital
            stats["ending_balance"] = self.ending_balance
            stats["available_balance"] = self.available_balance
            stats["holding_values"] = self.holding_values
            stats["capital"] = self.capital
            stats["additional_capitals"] = self.account.additional_capitals if self.account else []
            stats["net_investment"] = self.net_investment
            stats["total_net_pnl"] = self.total_net_pnl
            stats["total_commission"] = self.total_commission
            stats["total_turnover"] = self.total_turnover
            # stats["gross_profit"] = self.gross_profit
            # stats["gross_loss"] = self.gross_loss
            stats["profit_factor"] = self.profit_factor
            stats["return_on_capital"] = self.return_on_capital
            stats["return_on_initial_capital"] = self.return_on_initial_capital
            stats["return_on_investment"] = self.return_on_investment
            stats["annualized_return"] = self.annualized_return(annualization)
            stats["total_return"] = self.total_return
            stats["max_return"] = self.max_return
            stats["min_return"] = self.min_return
            stats["past_24hr_pnl"] = self.past_24hr_pnl
            stats["past_24hr_apr"] = self.past_24hr_apr
            stats["past_24hr_roi"] = self.past_ndays_roi(days=1)
            stats["past_7d_roi"] = self.past_ndays_roi(days=7)
            stats["past_14d_roi"] = self.past_ndays_roi(days=14)
            stats["past_30d_roi"] = self.past_ndays_roi(days=30)
            stats["past_90d_roi"] = self.past_ndays_roi(days=90)
            stats["past_180d_roi"] = self.past_ndays_roi(days=180)
            stats["past_1yr_roi"] = self.past_ndays_roi(days=365)
            stats["trading_period"] = self.trading_period
            stats["avg_holding_period(hrs)"] = self.avg_holding_period
            stats["pct_time_in_market"] = self.pct_time_in_market

            ### TRADES
            stats["number_trades"] = self.number_trades
            stats["number_closed_trades"] = self.number_closed_trades
            stats["number_winning_trades"] = self.num_winning_trades
            stats["number_losing_trades"] = self.num_losing_trades
            stats["number_winning_long_trades"] = self.num_winning_long_trades
            stats["number_losing_long_trades"] = self.num_losing_long_trades
            stats["number_winning_short_trades"] = self.num_winning_short_trades
            stats["number_losing_short_trades"] = self.num_losing_short_trades
            stats["number_liquidation_trades"] = self.num_liquidation_trades
            stats["avg_daily_trades"] = self.trade_freq("B")
            stats["avg_weekly_trades"] = self.trade_freq("W")
            stats["avg_monthly_trades"] = self.trade_freq("BME")
            stats["win_ratio"] = self.win_ratio
            stats["loss_ratio"] = self.loss_ratio
            stats["win_ratio_long"] = self.win_ratio_long
            stats["loss_ratio_long"] = self.loss_ratio_long
            stats["win_ratio_short"] = self.win_ratio_short
            stats["loss_ratio_short"] = self.loss_ratio_short
            stats["gross_trades_profit"] = self.gross_trades_profit
            stats["gross_trades_loss"] = self.gross_trades_loss
            stats["trades_profit_factor"] = self.trades_profit_factor
            stats["gross_winning_trades_amount"] = self.gross_winning_trades_amount
            stats["gross_losing_trades_amount"] = self.gross_losing_trades_amount
            stats["avg_winning_trades_pnl"] = (
                stats["gross_trades_profit"] / stats["number_winning_trades"] if stats["number_winning_trades"] else 0.0
            )
            stats["avg_losing_trades_pnl"] = (
                stats["gross_trades_loss"] / stats["number_losing_trades"] if stats["number_losing_trades"] else 0.0
            )
            stats["avg_winning_trades_amount"] = (
                stats["gross_winning_trades_amount"] / stats["number_winning_trades"]
                if stats["number_winning_trades"]
                else 0.0
            )
            stats["avg_losing_trades_amount"] = (
                stats["gross_winning_trades_amount"] / stats["number_losing_trades"]
                if stats["number_losing_trades"]
                else 0.0
            )
            stats["largest_profit_winning_trade"] = self.largest_profit_winning_trade
            stats["largest_loss_losing_trade"] = self.largest_loss_losing_trade
            stats["avg_amount_per_trade"] = self.avg_amount_per_trade
            stats["avg_amount_per_open_trade"] = self.avg_amount_per_open_trade
            stats["avg_amount_per_closed_trade"] = self.avg_amount_per_closed_trade
            stats["avg_amount_per_long_trade"] = self.avg_amount_per_long_trade
            stats["avg_amount_per_short_trade"] = (
                self.avg_amount_per_short_trade
            )  ## TODO: open long, open short, close long, close short
            stats["avg_pnl_per_trade($)"] = self.avg_pnl_per_trade
            stats["avg_pnl_per_trade"] = self.avg_pnl_per_trade_in_percentile
            (
                stats["max_consecutive_win_trades"],
                stats["max_consecutive_loss_trades"],
            ) = self.max_consecutive_trades()
            stats["payoff_ratio"] = self.payoff_ratio(stats["avg_winning_trades_pnl"], stats["avg_losing_trades_pnl"])

            ### PNL
            stats["expected_value"] = prec_round(
                stats["win_ratio"] * stats["avg_winning_trades_pnl"]
                + stats["loss_ratio"] * stats["avg_losing_trades_pnl"],
                2,
            )
            # stats["standardized_expected_value"] = stats["expected_value"] / stats["avg_amount_per_closed_trade"] * 100
            stats["standardized_expected_value"] = prec_round(
                (
                    stats["win_ratio"]
                    * stats["avg_winning_trades_pnl"]
                    / (stats["avg_winning_trades_amount"] if stats["avg_winning_trades_amount"] else 1.0)
                    + stats["loss_ratio"]
                    * stats["avg_losing_trades_pnl"]
                    / (stats["avg_winning_trades_amount"] if stats["avg_losing_trades_amount"] else 1.0)
                )
                * 100,
                2,
            )
            stats["win_days"] = self.win_days
            stats["loss_days"] = self.loss_days
            stats["max_win_in_day"] = self.max_win_in_day
            stats["max_loss_in_day"] = self.max_loss_in_day
            stats["max_consecutive_win_days"], stats["max_consecutive_loss_days"] = self.max_consecutive_days()
            stats["avg_daily_pnl($)"], stats["avg_daily_pnl"] = self.avg_pnl_by_freq("D")
            stats["avg_weekly_pnl($)"], stats["avg_weekly_pnl"] = self.avg_pnl_by_freq("W")
            stats["avg_monthly_pnl($)"], stats["avg_monthly_pnl"] = self.avg_pnl_by_freq("BME")
            stats["avg_quarterly_pnl($)"], stats["avg_quarterly_pnl"] = self.avg_pnl_by_freq("BQE")
            stats["avg_annualy_pnl($)"], stats["avg_annualy_pnl"] = self.avg_pnl_by_freq("YE")

            ### RATIOS
            stats["var"], stats["risk_score"] = self.var(conf_level=0.95, return_var=True)
            # stats["cvar"] = self.cvar(conf_level=0.95)
            stats["avg_daily_risk_score"] = self.avg_daily_risk_score
            stats["avg_risk_score_past_7days"] = self.avg_risk_score_past_7days
            stats["monthly_avg_risk_score"] = self.monthly_avg_risk_score
            stats["frequently_traded"] = self.frequently_traded
            stats["sharpe_ratio"] = self.sharpe_ratio(risk_free=risk_free, annualization=annualization)
            stats["sortino_ratio"] = self.sortino_ratio(annualization=annualization, required_return=required_return)
            stats["calmar_ratio"] = self.calmar_ratio(annualization=annualization)
            stats["probabilistic_sharpe_ratio"] = self.probabilistic_sharpe_ratio(
                risk_free=risk_free, annualization=annualization
            )
            # sr_std, sr_max, sr_min = self.sharpe_ratio_breakdown(risk_free=risk_free, annualization=annualization)
            stats["annualized_volatility"] = self.annualized_volatility(annualization=annualization)
            stats["omega_ratio"] = self.omega_ratio(
                risk_free=risk_free,
                annualization=annualization,
                required_return=required_return,
            )
            stats["downside_risk"] = self.downside_risk(annualization=annualization, required_return=required_return)
            stats["information_ratio"] = self.information_ratio if has_benchmark else None
            stats["beta"] = self.beta(risk_free=risk_free) if has_benchmark else None
            stats["alpha"] = self.alpha(risk_free=risk_free, annualization=annualization) if has_benchmark else None
            stats["tail_ratio"] = self.tail_ratio
            stats["common_sense_ratio"] = self.common_sense_ratio
            stats["skew"] = self.skew
            stats["kurtosis"] = self.kurtosis
            stats["stability_of_timeseries"] = self.stability_of_timeseries
            stats["max_drawdown"], stats["max_drawdown_period"] = self.max_drawdown

            if stats["max_drawdown_period"]:
                stats["max_drawdown_duration"] = stats["max_drawdown_period"][1] - stats["max_drawdown_period"][0]
            else:
                stats["max_drawdown_duration"] = None  # "N/A"
            stats["max_runup"], stats["max_runup_period"] = self.max_runup
            if stats["max_runup_period"]:
                stats["max_runup_duration"] = stats["max_runup_period"][1] - stats["max_runup_period"][0]
            else:
                stats["max_runup_duration"] = None  # "N/A"
            stats["sqn"] = self.sqn

            ### ROLLING STATS
            # print(self.rolling_sharpe(rolling_period=30, risk_free=risk_free, annualization=annualization))
            stats["rolling_sharpe"], stats["monthly_sharpe"] = self.rolling_sharpe(
                rolling_period=30, risk_free=risk_free, annualization=annualization
            )
            stats["monthly_changes"] = self.monthly_changes
            stats["daily_changes"] = self.daily_changes
            stats["asset_values"] = self.asset_values
            stats["drawdown_series"] = self.drawdown_series()

            # ### DOD
            # print(f">>>>>>> self.positions:{self.positions}")
            stats["positions"] = [position.to_dict() for position in self.positions.values()]
            stats["trades"] = list(self.trade_records.values())
            stats["pnl"] = self.data_resample_daily.to_dict("records")  # resample to daily

            stats = stats.replace([np.inf, -np.inf, np.nan], [None, None, None])

        self.stats_result = stats
        """
        test
        """
        # self.test_pyfolio(risk_free=risk_free,annualization=annualization)
        # self.test_empyrical(risk_free=risk_free,annualization=annualization)
        # self.test_roll(risk_free=risk_free,annualization=annualization)

        # print(f">>>>>stats:\n{stats}")

        if return_dict:
            return json.loads(json.dumps(stats.to_dict(), indent=4, default=set_default))

        return stats

    def genereate_stats_custom(
        self,
        returns: pd.Series | None = None,
        assets_value: pd.Series | None = None,
        benchmark: pd.DataFrame | pd.Series | None = None,
        trades_df: pd.DataFrame | None = None,
        interval: str = "1d",
        resample_interval: str | None = None,
        annualization: int | None = None,
        risk_free: float = 0.0,
        required_return: float = 0.0,
        return_dict: bool = False,
    ):
        """
        in realtime:
            returns related - update in daily basis
            trade related - realtime update
                - positions, trades, pnl, end, sqn, frequently_traded
                    start, avg_daily_trades, avg_weekly_trades, avg_monthly_trades, win_ratio, loss_ratio
        TODO:
            duration - number of bars / extact times
            long/short pnl/trades
            number of margin call / liquidation
        """
        if returns is None and assets_value is None:
            raise Exception("Either `returns` or `assets_value` needs to be specified.")

        self.resample_rule = RESAMPLE_MAP.get(resample_interval) if resample_interval else None
        self.interval = interval
        self.annualization = annualization if annualization else self._get_annualization()
        self.trades_df = pd.DataFrame() if trades_df is None else trades_df

        has_benchmark = benchmark is not None

        if returns is not None:
            df = returns.to_frame(name="pct_returns")
            self.last_updated_at = returns.index[-1]
        else:
            df = assets_value.to_frame(name="assets_value")
            self.last_updated_at = assets_value.index[-1]

        df["datetime"] = df.index
        df["pnl_net"] = np.nan

        if has_benchmark:
            df = df.sort_index()
            df = pd.merge_asof(
                df,
                benchmark,
                left_index=True,
                right_index=True,
                allow_exact_matches=True,
                direction="nearest",
            )
            df.rename(columns={"close": "benchmark_close"}, inplace=True)

        if returns is not None:
            df["cum_returns"] = self.cum_returns(df["pct_returns"])
            df["assets_value"] = df["cum_returns"] * self.initial_capital
        else:
            df["pct_returns"] = df["assets_value"].pct_change().fillna(0.0)
            df["cum_returns"] = self.cum_returns(df["pct_returns"])

        df["pnl_cumsum"] = df["pnl_net"].cumsum()
        df["risk_scores"] = (
            df["pct_returns"].expanding().apply(self.var)
        )  # expanding is similar to rolling, only diff is the window size is growing

        if has_benchmark:
            df["benchmark_pct_returns"] = df["benchmark_close"].pct_change()
            df["benchmark"] = df["benchmark_pct_returns"].cumsum()

        if not df.empty:
            df = df.fillna(0)
            df = df[~df.index.duplicated(keep="last")]
            last_index = df.index[-1]
            if self.resample_rule == "1D":
                self.data_resample_daily = df
            else:
                self.data_resample_daily = (
                    df.resample("D", label="right", closed="right")
                    .agg(resample_agg_gen(df.columns))
                    .ffill()
                    .fillna(0)
                    .round(5)
                )
                self.data_resample_daily = self.data_resample_daily.rename(
                    index={self.data_resample_daily.index[-1]: last_index}
                )

            self.data_resample_monthly = (
                df.resample("M", label="right", closed="right")
                .agg(resample_agg_gen(df.columns))
                .ffill()
                .bfill()
                .fillna(0)
                .copy()
            )
            self.data_resample_monthly = self.data_resample_monthly.rename(
                index={self.data_resample_monthly.index[-1]: last_index}
            )
            self.data_resample_monthly = self.data_resample_monthly.fillna(0).round(5)

            df = df.round(5)

        self.data = df
        # print(self.data,'?????')

        stats = pd.Series(dtype="object", name="value")

        ### OVERALL RESULTS
        stats["start"] = self.start
        stats["end"] = self.end
        stats["interval"] = None  # ''
        stats["duration"] = self.duration
        stats["trading_instruments"] = list(self.symbols)
        stats["base_currency"] = self.account.asset if self.account else "N/A"
        stats["benchmark"] = None  # ""

        stats["beginning_balance"] = None  # self.initial_capital
        stats["ending_balance"] = None  # self.ending_balance
        stats["available_balance"] = None  # self.available_balance
        stats["holding_values"] = None  # self.holding_values
        stats["capital"] = None  # self.capital
        stats["additional_capitals"] = None  # self.account.additional_capitals
        stats["net_investment"] = None  # self.net_investment
        stats["total_net_pnl"] = None  # self.total_net_pnl
        stats["total_commission"] = None  # self.total_commission
        stats["total_turnover"] = None  # self.total_turnover
        ## stats["gross_profit"] = self.gross_profit
        ## stats["gross_loss"] = self.gross_loss
        stats["profit_factor"] = self.profit_factor
        stats["return_on_capital"] = self.return_on_capital
        stats["return_on_initial_capital"] = self.return_on_initial_capital
        stats["return_on_investment"] = None  # self.return_on_investment
        stats["annualized_return"] = self.annualized_return(annualization)
        stats["total_return"] = self.total_return
        stats["max_return"] = self.max_return
        stats["min_return"] = self.min_return
        stats["past_24hr_pnl"] = None  # self.past_24hr_pnl
        stats["past_24hr_apr"] = self.past_24hr_apr
        stats["past_24hr_roi"] = self.past_ndays_roi(days=1)
        stats["past_7d_roi"] = self.past_ndays_roi(days=7)
        stats["past_14d_roi"] = self.past_ndays_roi(days=14)
        stats["past_30d_roi"] = self.past_ndays_roi(days=30)
        stats["past_90d_roi"] = self.past_ndays_roi(days=90)
        stats["past_180d_roi"] = self.past_ndays_roi(days=180)
        stats["past_1yr_roi"] = self.past_ndays_roi(days=365)
        stats["trading_period"] = self.trading_period
        stats["avg_holding_period(hrs)"] = self.avg_holding_period
        stats["pct_time_in_market"] = None  # self.pct_time_in_market

        ### TRADES
        stats["number_trades"] = self.number_trades
        stats["number_closed_trades"] = self.number_closed_trades
        stats["number_winning_trades"] = self.num_winning_trades
        stats["number_losing_trades"] = self.num_losing_trades
        stats["number_winning_long_trades"] = self.num_winning_long_trades
        stats["number_losing_long_trades"] = self.num_losing_long_trades
        stats["number_winning_short_trades"] = self.num_winning_short_trades
        stats["number_losing_short_trades"] = self.num_losing_short_trades
        stats["number_liquidation_trades"] = self.num_liquidation_trades
        stats["avg_daily_trades"] = self.trade_freq("B")
        stats["avg_weekly_trades"] = self.trade_freq("W")
        stats["avg_monthly_trades"] = self.trade_freq("BME")
        stats["win_ratio"] = self.win_ratio
        stats["loss_ratio"] = self.loss_ratio
        stats["win_ratio_long"] = self.win_ratio_long
        stats["loss_ratio_long"] = self.loss_ratio_long
        stats["win_ratio_short"] = self.win_ratio_short
        stats["loss_ratio_short"] = self.loss_ratio_short
        stats["gross_trades_profit"] = self.gross_trades_profit
        stats["gross_trades_loss"] = self.gross_trades_loss
        stats["gross_winning_trades_amount"] = self.gross_winning_trades_amount
        stats["gross_losing_trades_amount"] = self.gross_losing_trades_amount
        stats["avg_winning_trades_pnl"] = (
            stats["gross_trades_profit"] / stats["number_winning_trades"] if stats["number_winning_trades"] else 0.0
        )
        stats["avg_losing_trades_pnl"] = (
            stats["gross_trades_loss"] / stats["number_losing_trades"] if stats["number_losing_trades"] else 0.0
        )
        stats["avg_winning_trades_amount"] = (
            stats["gross_winning_trades_amount"] / stats["number_winning_trades"]
            if stats["number_winning_trades"]
            else 0.0
        )
        stats["avg_losing_trades_amount"] = (
            stats["gross_winning_trades_amount"] / stats["number_losing_trades"]
            if stats["number_losing_trades"]
            else 0.0
        )
        stats["largest_profit_winning_trade"] = self.largest_profit_winning_trade
        stats["largest_loss_losing_trade"] = self.largest_loss_losing_trade
        stats["avg_amount_per_trade"] = self.avg_amount_per_trade
        stats["avg_amount_per_open_trade"] = self.avg_amount_per_open_trade
        stats["avg_amount_per_closed_trade"] = self.avg_amount_per_closed_trade
        stats["avg_amount_per_long_trade"] = self.avg_amount_per_long_trade
        stats["avg_amount_per_short_trade"] = (
            self.avg_amount_per_short_trade
        )  ## TODO: open long, open short, close long, close short
        stats["avg_pnl_per_trade($)"] = self.avg_pnl_per_trade
        stats["avg_pnl_per_trade"] = self.avg_pnl_per_trade_in_percentile
        stats["max_consecutive_win_trades"], stats["max_consecutive_loss_trades"] = self.max_consecutive_trades()
        stats["payoff_ratio"] = (
            prec_round(stats["avg_winning_trades_pnl"] / stats["avg_losing_trades_pnl"], 4)
            if stats["avg_losing_trades_pnl"]
            else 0.0
        )  # The payoff ratio is a financial ratio used to evaluate the potential profit of a trading strategy or investment.

        ### PNL
        stats["expected_value"] = prec_round(
            stats["win_ratio"] * stats["avg_winning_trades_pnl"] + stats["loss_ratio"] * stats["avg_losing_trades_pnl"],
            2,
        )
        stats["standardized_expected_value"] = (
            None  # stats["expected_value"] / stats["avg_amount_per_closed_trade"] * 100
        )
        stats["standardized_expected_value"] = prec_round(
            (
                stats["win_ratio"]
                * stats["avg_winning_trades_pnl"]
                / (stats["avg_winning_trades_amount"] if stats["avg_winning_trades_amount"] else 1.0)
                + stats["loss_ratio"]
                * stats["avg_losing_trades_pnl"]
                / (stats["avg_winning_trades_amount"] if stats["avg_losing_trades_amount"] else 1.0)
            )
            * 100,
            2,
        )
        stats["win_days"] = self.win_days
        stats["loss_days"] = self.loss_days
        stats["max_win_in_day"] = self.max_win_in_day
        stats["max_loss_in_day"] = self.max_loss_in_day
        stats["max_consecutive_win_days"], stats["max_consecutive_loss_days"] = self.max_consecutive_days()
        stats["avg_daily_pnl($)"], stats["avg_daily_pnl"] = (
            None,
            None,
        )  # self.avg_pnl_by_freq("D")
        stats["avg_weekly_pnl($)"], stats["avg_weekly_pnl"] = (
            None,
            None,
        )  # self.avg_pnl_by_freq("W")
        stats["avg_monthly_pnl($)"], stats["avg_monthly_pnl"] = (
            None,
            None,
        )  # self.avg_pnl_by_freq("BM")
        stats["avg_quarterly_pnl($)"], stats["avg_quarterly_pnl"] = (
            None,
            None,
        )  # self.avg_pnl_by_freq("BQ")
        stats["avg_annualy_pnl($)"], stats["avg_annualy_pnl"] = (
            None,
            None,
        )  # self.avg_pnl_by_freq("Y")

        ### RATIOS
        stats["var"], stats["risk_score"] = self.var(conf_level=0.95, return_var=True)
        ## stats["cvar"] = self.cvar(conf_level=0.95)
        stats["avg_daily_risk_score"] = self.avg_daily_risk_score
        stats["avg_risk_score_past_7days"] = self.avg_risk_score_past_7days
        stats["monthly_avg_risk_score"] = self.monthly_avg_risk_score
        stats["frequently_traded"] = self.frequently_traded
        stats["sharpe_ratio"] = self.sharpe_ratio(risk_free=risk_free, annualization=annualization)
        stats["sortino_ratio"] = self.sortino_ratio(annualization=annualization, required_return=required_return)
        stats["calmar_ratio"] = self.calmar_ratio(annualization=annualization)
        stats["probabilistic_sharpe_ratio"] = self.probabilistic_sharpe_ratio(
            risk_free=risk_free, annualization=annualization
        )
        ## sr_std, sr_max, sr_min = self.sharpe_ratio_breakdown(risk_free=risk_free, annualization=annualization)
        stats["annualized_volatility"] = self.annualized_volatility(annualization=annualization)
        stats["omega_ratio"] = self.omega_ratio(
            risk_free=risk_free,
            annualization=annualization,
            required_return=required_return,
        )
        stats["downside_risk"] = self.downside_risk(annualization=annualization, required_return=required_return)
        stats["information_ratio"] = self.information_ratio if has_benchmark else None
        stats["beta"] = self.beta(risk_free=risk_free) if has_benchmark else None
        stats["alpha"] = self.alpha(risk_free=risk_free, annualization=annualization) if has_benchmark else None
        stats["tail_ratio"] = self.tail_ratio
        stats["common_sense_ratio"] = self.common_sense_ratio
        stats["skew"] = self.skew
        stats["kurtosis"] = self.kurtosis
        stats["stability_of_timeseries"] = self.stability_of_timeseries
        stats["max_drawdown"], stats["max_drawdown_period"] = self.max_drawdown
        if stats["max_drawdown_period"]:
            stats["max_drawdown_duration"] = stats["max_drawdown_period"][1] - stats["max_drawdown_period"][0]
        else:
            stats["max_drawdown_duration"] = None  # "N/A"
        stats["max_runup"], stats["max_runup_period"] = self.max_runup
        if stats["max_runup_period"]:
            stats["max_runup_duration"] = stats["max_runup_period"][1] - stats["max_runup_period"][0]
        else:
            stats["max_runup_duration"] = None  # "N/A"
        stats["sqn"] = self.sqn

        ### ROLLING STATS
        stats["rolling_sharpe"], stats["monthly_sharpe"] = self.rolling_sharpe(
            rolling_period=30, risk_free=risk_free, annualization=annualization
        )
        stats["monthly_changes"] = self.monthly_changes
        stats["daily_changes"] = self.daily_changes
        stats["asset_values"] = self.asset_values
        stats["drawdown_series"] = self.drawdown_series()

        # ### DOD
        stats["positions"] = self._positions  # list(self.positions.values())
        stats["trades"] = list(self.trade_records.values())
        stats["pnl"] = self.data_resample_daily.to_dict("records")  # resample to daily

        stats = stats.replace([np.inf, -np.inf, np.nan], [None, None, None])
        self.stats_result = stats
        """
        test
        """
        # self._generate_quantstats(risk_free=risk_free,annualization=annualization)
        # self.test_pyfolio(risk_free=risk_free,annualization=annualization)
        # self.test_empyrical(risk_free=risk_free,annualization=annualization)
        # self.test_roll(risk_free=risk_free,annualization=annualization)

        # print(f">>>>>stats:\n{stats}")

        if return_dict:
            return json.loads(json.dumps(stats.to_dict(), indent=4, default=set_default))

        return stats

    def to_dict(self):
        if self.stats_result.empty:
            return self.genereate_stats(return_dict=True)
        return json.loads(json.dumps(self.stats_result.to_dict(), indent=4, default=set_default))

    def plot(
        self,
        path,
        feeds: list = [],
        interactive=False,
        open_browser: bool | None = None,
        debug: bool = False,
        extra_ohlc: bool = False,
        only_ohlc: bool = False,
        in_nav: bool = False,
        custom_plot_data: dict = {},
        draw_figure: Callable | None = None,
        use_qs: bool = False,
        **kwargs,
    ):
        """
        The function to plot the diagram.
        Args:
            path (str): The path that you would like to store the generated chart.
            interactive (bool, optional): Decide whether generated chart is interactive or not. Defaults to False.
        """
        try:
            if use_qs:
                self._generate_quantstats(
                    risk_free=self.risk_free,
                    annualization=self.annualization,
                    path=path,
                )
                return
        except ImportError as e:
            print(f"Something went wront when plotting the quantstats graph. Error:{str(e)}")
            return

        try:
            from .plotter import Plotter
        except ImportError:
            print(
                "Plotter required package 'bokeh', if you would like to plot the graph, please install the package 'bokeh'. "
                "pip install -U bokeh"
            )
            return

        try:
            stats = self.stats_result
            if in_nav:
                self.genereate_stats()
                portfolio_data = self.data.copy()
                nav_stats = self.stats_result.copy()

                self.genereate_stats()
                stats = self.stats_result
            else:
                portfolio_data = self.data.copy()
                nav_stats = None

            # print(f">>>>> PM - plot, interval:{self.interval}, resample_rule:{self.resample_rule}, in_nav:{in_nav}, nav_stats:{nav_stats}")
            for feed in feeds:
                if hasattr(feed, "is_live") and feed.is_live:
                    feed.fetch_data(
                        interval=self.interval,
                        start_date=self.start.strftime("%Y-%m-%d %H:%M:%S"),
                        end_date=self.end.strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    # print(f'''>>>>>>>>>>>>>>>>feed:{feed.code} is fetching data, interval={self.interval}, start_date={self.start.strftime("%Y-%m-%d %H:%M:%S")}, end_date={self.end.strftime("%Y-%m-%d %H:%M:%S")}''')
                    # print(f">>>>>>>>>>>>>>>>{feed.dataframe} ")#| data:{data}")
            # print(f'''>>>>>>>>>>>>>>>>feeds:{feeds}, interval={self.interval}, start_date={self.start.strftime("%Y-%m-%d %H:%M:%S")}, end_date={self.end.strftime("%Y-%m-%d %H:%M:%S")}''')

            plotter = Plotter(
                feeds=feeds,
                portfolio_data=portfolio_data,
                stats=stats,
                in_nav=in_nav,
                nav_stats=nav_stats,
                # all_trades=list(self.trade_records.values()),
                all_trades=[trade.to_dict() for trade in self.trade_records.values()],
                resample_rule=self.resample_rule if hasattr(self, "resample_rule") else None,
                custom_plot_data=custom_plot_data,
                draw_figure=draw_figure,
            )
            plotter.plot(
                path=path,
                interactive=interactive,
                open_browser=open_browser,
                extra_ohlc=extra_ohlc,
                only_ohlc=only_ohlc,
                **kwargs,
            )
            if path and debug:
                print(f"The chart is stored in {path}.")

            return stats
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Something went wront when plotting the graph. Error:{str(e)}")
            return

    def export_json(self, name: str, content: dict | list) -> None:
        if not self.path:
            return
        exp_path = os.path.join(self.path, name)
        # if not os.path.exists(exp_path):
        #     return
        if isinstance(content, pd.DataFrame):
            # content.to_json(exp_path, orient='index', indent=4, default_handler=set_default)
            # content.index = pd.to_datetime(content.index)
            # content = content.copy()
            # content.index = content.index.strftime('%Y-%m-%d %H:%M:%S')
            # content = content.to_dict('index')
            content = content.to_dict("records")
        # if 'trade' in name:
        #     print(content,'!!!!!!')
        with open(exp_path, "w") as wf:
            # json.encoder.FLOAT_REPR = lambda x: f"{x:f}"  # By default :f has a precision of 6
            json.dump(content, wf, indent=4, default=set_default)

    def export_csv(self, name: str, content: dict | list, **kwargs) -> None:
        if not self.path:
            return
        exp_path = os.path.join(self.path, name)
        # if not os.path.exists(exp_path):
        #     return
        df = pd.DataFrame(content)
        if "positions" in df.columns:
            df.drop(["positions"], axis=1, inplace=True)
        try:
            df.to_csv(exp_path, **kwargs)
        except Exception as e:
            print(f"{str(e)}")

    def export_dict(self, content: dict) -> dict:
        ### method 2
        # if isinstance(content, pd.Series) or isinstance(content, pd.DataFrame):
        #     pass
        # else:
        #     content = pd.DataFrame(content)
        # return content.to_json(indent=None, default_handler=set_default)

        ### method 3
        # try:
        #     import simplejson # will handle nan -> None
        # except ImportError:
        #     raise ImportError("Need package 'simplejson' for exporting a normal dict. " "pip install simplejson")

        # return simplejson.loads(simplejson.dumps(content, indent=4, default=set_default, ignore_nan=True))

        ### method 1
        if isinstance(content, pd.DataFrame):
            content = content.to_dict()
        elif isinstance(content, pd.Series):
            content = content.to_dict()
        # json.encoder.FLOAT_REPR = lambda x: f"{x:f}"  # By default :f has a precision of 6
        return json.loads(json.dumps(content, indent=None, default=set_default))

    def contest_output(
        self,
        path: str,
        interval: str,
        benchmark: DataFeed | None = None,
        resample_interval: str | None = None,
        prefix: str = "",
        suffix: str = "",
        is_live: bool = False,
        ## below parameters are for plot only
        is_plot: bool = True,
        feeds: list = [],
        extra_ohlc: bool = False,
        only_ohlc: bool = False,
        in_nav: bool = False,
        custom_plot_data: dict = {},
        draw_figure=None,
        open_browser: bool = False,
        debug: bool = False,
        use_qs: bool = False,
        **kwargs,
    ):
        """
        Generate all the contest required output at once
        Args:
            prefix (str, optional): prefix of file name. Defaults to ''.
        """
        # print(locals(),'!!!!!!!!!!!!!???!!!!')
        self.path = path

        trades = list(self.trade_records.values())
        # print(trades,'???????', len(trades))
        tradepnls = []
        for position in self.positions.values():
            for tradepnl in position.tradepnls:
                tradepnls.append(tradepnl.to_dict())
        self.export_json(f"{prefix}tradepnls{suffix}.json", tradepnls)

        self.export_json(f"{prefix}records_trade{suffix}.json", trades)
        self.export_csv(f"{prefix}records_trade{suffix}.csv", trades)

        stats = None
        if self.data.empty or is_live:
            stats = self.genereate_stats(
                interval=interval,
                benchmark=benchmark,
                resample_interval=resample_interval,
                use_qs=use_qs,
            )
            # print('----\n',stats,'\n----')

        self.export_csv(f"{prefix}pnl_detail{suffix}.csv", self.data, index=True)
        self.export_json(f"{prefix}pnl_detail{suffix}.json", self.data)
        stats = self.stats_result if stats is None else stats
        self.export_csv(f"{prefix}analysis{suffix}.csv", stats, index=True)
        self.export_json(f"{prefix}analysis{suffix}.json", stats.to_dict())
        # print(stats,'........')
        if not use_qs:
            if not stats.empty:
                stats_short = stats.to_frame().drop(["trades", "pnl"], axis=0)
            else:
                stats_short = stats
            self.export_csv(f"{prefix}statistic{suffix}.csv", stats_short, index=True)
            self.export_json(
                f"{prefix}statistic{suffix}.json",
                stats_short.to_dict().get("value", {}),
            )

        if is_plot:
            if self.path:
                try:
                    self.plot(
                        feeds=feeds,
                        path=os.path.join(
                            self.path,
                            f"{prefix}{'qs_' if use_qs else ''}report{suffix}.html",
                        ),
                        interactive=True,
                        open_browser=open_browser,
                        debug=debug,
                        extra_ohlc=extra_ohlc,
                        only_ohlc=only_ohlc,
                        in_nav=in_nav,
                        custom_plot_data=custom_plot_data,
                        draw_figure=draw_figure,
                        use_qs=use_qs,
                        **kwargs,
                    )
                except Exception as e:
                    tb_lineno = e.__traceback__.tb_lineno if e.__traceback__ else "unknown"
                    tb_filename = (
                        e.__traceback__.tb_frame.f_code.co_filename
                        if e.__traceback__ and e.__traceback__.tb_frame
                        else "unknown"
                    )
                    print(
                        f"[ERROR] Something went wrong when plot the graph - Error:{e} in line {tb_lineno} for file {tb_filename}."
                    )
                    return

        print(
            f"""The output files are stored in {os.path.join(self.path, f"{prefix}{'qs_' if use_qs else ''}report{suffix}.html") if is_plot else self.path}."""
        )
        return stats
