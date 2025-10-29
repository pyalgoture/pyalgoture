from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from ...utils.objects import OrderData, OrderType, Side, TradeData, TradePnL
from .webserver_bgwork import ProgressTask

# ============================


class Ping(BaseModel):
    status: str


class AccessToken(BaseModel):
    access_token: str


class AccessAndRefreshToken(AccessToken):
    refresh_token: str


class Version(BaseModel):
    version: str


class StatusMsg(BaseModel):
    status: str


class BgJobStarted(StatusMsg):
    job_id: str


class BackgroundTaskStatus(BaseModel):
    job_id: str
    job_category: str
    status: str
    running: bool
    progress: float | None = None
    progress_tasks: dict[str, ProgressTask] | None = None
    error: str | None = None


class BackgroundTaskResult(BaseModel):
    error: str | None = None
    status: str


class ResultMsg(BaseModel):
    result: str


class Balance(BaseModel):
    currency: str
    free: float
    balance: float
    used: float
    bot_owned: float | None = None
    est_stake: float
    est_stake_bot: float | None = None
    stake: str
    # Starting with 2.x
    side: str
    is_position: bool
    position: float
    is_bot_managed: bool


class Balances(BaseModel):
    currencies: list[Balance]
    total: float
    total_bot: float
    symbol: str
    value: float
    value_bot: float
    stake: str
    note: str
    starting_capital: float
    starting_capital_ratio: float
    starting_capital_pct: float
    starting_capital_fiat: float
    starting_capital_fiat_ratio: float
    starting_capital_fiat_pct: float


class Count(BaseModel):
    current: int
    max: int
    total_trade_amount: float


class __BaseStatsModel(BaseModel):
    close_pnl: float
    close_roi: float
    count: int


class Entry(__BaseStatsModel):
    enter_reason: str


class Exit(__BaseStatsModel):
    exit_reason: str


class MixTag(__BaseStatsModel):
    mix_tag: str


class PerformanceEntry(__BaseStatsModel):
    symbol: str


class Profit(BaseModel):
    profit_closed_coin: float
    profit_closed_percent_mean: float
    profit_closed_ratio_mean: float
    profit_closed_percent_sum: float
    profit_closed_ratio_sum: float
    profit_closed_percent: float
    profit_closed_ratio: float
    profit_closed_fiat: float
    profit_all_coin: float
    profit_all_percent_mean: float
    profit_all_ratio_mean: float
    profit_all_percent_sum: float
    profit_all_ratio_sum: float
    profit_all_percent: float
    profit_all_ratio: float
    profit_all_fiat: float
    trade_count: int
    closed_trade_count: int
    first_trade_date: str
    first_trade_humanized: str
    first_trade_timestamp: int
    latest_trade_date: str
    latest_trade_humanized: str
    latest_trade_timestamp: int
    avg_duration: str
    best_pair: str
    best_rate: float
    best_pair_profit_ratio: float
    winning_trades: int
    losing_trades: int
    profit_factor: float
    winrate: float
    expectancy: float
    expectancy_ratio: float
    max_drawdown: float
    max_drawdown_abs: float
    max_drawdown_start: str
    max_drawdown_start_timestamp: int
    max_drawdown_end: str
    max_drawdown_end_timestamp: int
    trading_volume: float | None = None
    bot_start_timestamp: int
    bot_start_date: str


class SellReason(BaseModel):
    wins: int
    losses: int
    draws: int


class Stats(BaseModel):
    exit_reasons: dict[str, SellReason]
    durations: dict[str, float | None]


class DailyWeeklyMonthlyRecord(BaseModel):
    date: date
    abs_profit: float
    rel_profit: float
    starting_balance: float
    fiat_value: float
    trade_count: int


class DailyWeeklyMonthly(BaseModel):
    data: list[DailyWeeklyMonthlyRecord]
    fiat_display_currency: str
    base_currency: str


class UnfilledTimeout(BaseModel):
    entry: int | None = None
    exit: int | None = None
    unit: str | None = None
    exit_timeout_count: int | None = None


class OrderTypes(BaseModel):
    entry: OrderType
    exit: OrderType
    emergency_exit: OrderType | None = None
    force_exit: OrderType | None = None
    force_entry: OrderType | None = None
    stoploss: OrderType
    stoploss_on_exchange: bool
    stoploss_on_exchange_interval: int | None = None


class ShowConfig(BaseModel):
    version: str
    strategy: str
    api_version: float
    state: str
    strategy_version: str | None = None


class TradePnLResponse(BaseModel):
    tradepnls: list[TradePnL]
    tradepnls_count: int
    offset: int
    total_tradepnls: int


class TradeResponse(BaseModel):
    trades: list[TradeData]
    trades_count: int
    offset: int
    total_trades: int


class OrderResponse(BaseModel):
    orders: list[OrderData]
    orders_count: int
    offset: int
    total_orders: int


class DeleteLockRequest(BaseModel):
    symbol: str | None = None
    lockid: int | None = None


class Logs(BaseModel):
    log_count: int
    logs: list[list]


class StopPayload(BaseModel):
    exit_all: bool
    reports: bool


class ForceEnterPayload(BaseModel):
    symbol: str
    side: Side
    price: float | None = None
    ordertype: OrderType | None = None
    amount: float | None = None
    quantity: float | None = None
    cost: float | None = None
    msg: str | None = None
    leverage: float | None = None


class ForceExitPayload(BaseModel):
    symbol: str | int
    ordertype: OrderType | None = None
    quantity: float | None = None


class DeleteTrade(BaseModel):
    cancel_order_count: int
    result: str
    result_msg: str
    trade_id: int


class StrategyListResponse(BaseModel):
    strategies: list[str]


class StrategyResponse(BaseModel):
    strategy: str
    code: str
    timeframe: str | None


class AvailablePairs(BaseModel):
    length: int
    pairs: list[str]
    pair_interval: list[list[str]]


class SysInfo(BaseModel):
    cpu_pct: list[float]
    ram_pct: float


class Health(BaseModel):
    last_process: datetime | None = None
    last_process_ts: int | None = None
    bot_start: datetime | None = None
    bot_start_ts: int | None = None
    bot_startup: datetime | None = None
    bot_startup_ts: int | None = None
