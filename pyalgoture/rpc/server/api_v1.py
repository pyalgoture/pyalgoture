from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.exceptions import HTTPException

from ... import __version__
from ...utils.logger import get_logger
from ...utils.objects import OrderData, PositionData, TradePnL
from ..rpc import RPC, RPCException
from .api_schemas import (
    Count,
    DailyWeeklyMonthly,
    Entry,
    Exit,
    ForceEnterPayload,
    ForceExitPayload,
    Health,
    Logs,
    MixTag,
    OrderResponse,
    PerformanceEntry,
    Ping,
    ShowConfig,
    Stats,
    StatusMsg,
    StopPayload,
    StrategyListResponse,
    StrategyResponse,
    SysInfo,
    TradePnLResponse,
    TradeResponse,
    Version,
)
from .deps import get_config, get_rpc, get_rpc_optional

logger = get_logger()


class OperationalException(Exception):
    """
    Requires manual intervention and will stop the bot.
    Most of the time, this is caused by an invalid Configuration.
    """


# API version
# Pre-1.1, no version was provided
# Version increments should happen in "small" steps (1.1, 1.12, ...) unless big changes happen.
# 1.11: basic apis like positions, trades and balance etc
#
API_VERSION = 1.12

# Public API, requires no auth.
router_public = APIRouter()
# Private API, protected by authentication
router = APIRouter()


"""
##############
Info & System
##############
"""


@router_public.get("/ping", response_model=Ping)
def ping() -> dict[str, str]:
    """simple ping"""
    return {"status": "pong"}


@router_public.get("/version", response_model=Version, tags=["info"])
def version() -> dict[str, str]:
    """Bot Version info"""
    return {"version": __version__}


@router.get("/show_config", response_model=ShowConfig, tags=["info"])
def show_config(
    rpc: RPC | None = Depends(get_rpc_optional), config: dict[str, Any] = Depends(get_config)
) -> dict[str, Any]:
    if rpc is None:
        raise ValueError("RPC is not available")

    resp = {
        "version": __version__,
        "strategy": rpc._strategy_name,
        "api_version": API_VERSION,
        "state": rpc._engine.state,
        "strategy_version": None,  # rpc._engine.strategy.version() if rpc else None
    }

    return resp


@router.get("/logs", response_model=Logs, tags=["info"])
def logs(limit: int | None = None) -> dict[str, Any]:
    return RPC._rpc_get_logs(limit)


@router.post("/start", response_model=StatusMsg, tags=["botcontrol"])
def start(rpc: RPC = Depends(get_rpc)) -> dict[str, Any]:
    return rpc._rpc_start()


@router.post("/stop", response_model=StatusMsg, tags=["botcontrol"])
def stop(payload: StopPayload, rpc: RPC = Depends(get_rpc)) -> dict[str, Any]:
    exit_all = payload.exit_all
    reports = payload.reports
    print(f">>>>[stop] exit_all: {exit_all}({type(exit_all)}), reports: {reports}({type(reports)})")
    return rpc._rpc_stop(exit_all=exit_all, reports=reports)


@router.post("/stopentry", response_model=StatusMsg, tags=["botcontrol"])
def stop_entry(rpc: RPC = Depends(get_rpc)) -> dict[str, Any]:
    return rpc._rpc_stopentry()


@router.post("/reload_config", response_model=StatusMsg, tags=["botcontrol"])
def reload_config(rpc: RPC = Depends(get_rpc)) -> dict[str, Any]:
    return rpc._rpc_reload_config()


@router.get("/sysinfo", response_model=SysInfo, tags=["info"])
def sysinfo() -> dict[str, Any]:
    return RPC._rpc_sysinfo()


@router.get("/health", response_model=Health, tags=["info"])
def health(rpc: RPC = Depends(get_rpc)) -> dict[str, Any]:
    return rpc.health()


"""
##############
Performance
##############
"""


@router.get("/performance/entries", response_model=list[Entry], tags=["info"])
def performance_entries(pair: str | None = None, rpc: RPC = Depends(get_rpc)) -> list[dict[str, Any]]:
    return rpc._rpc_enter_reason_performance(pair)


@router.get("/performance/exits", response_model=list[Exit], tags=["info"])
def performance_exits(pair: str | None = None, rpc: RPC = Depends(get_rpc)) -> list[dict[str, Any]]:
    return rpc._rpc_exit_reason_performance(pair)


@router.get("/performance/mix_tags", response_model=list[MixTag], tags=["info"])
def performance_mix_tags(pair: str | None = None, rpc: RPC = Depends(get_rpc)) -> list[dict[str, Any]]:
    return rpc._rpc_mix_tag_performance(pair)


@router.get("/performance/symbol", response_model=list[PerformanceEntry], tags=["info"])
def performance_symbol(rpc: RPC = Depends(get_rpc)) -> list[dict[str, Any]]:
    return rpc._rpc_symbol_performance()


@router.get("/performance/stats", tags=["info"])  # , response_model=Profit
def performance(rpc: RPC = Depends(get_rpc), config: dict[str, Any] = Depends(get_config)) -> dict[str, Any]:
    return rpc._rpc_performance()


@router.get("/performance/generic", response_model=Stats, tags=["info"])
def performance_generic(rpc: RPC = Depends(get_rpc)) -> dict[str, Any]:
    return rpc._rpc_generic_performance()


@router.get("/performance/daily", response_model=DailyWeeklyMonthly, tags=["info"])
def performance_daily(
    timescale: int = 7, rpc: RPC = Depends(get_rpc), config: dict[str, Any] = Depends(get_config)
) -> dict[str, Any]:
    return rpc._rpc_period_performance(timescale=timescale, timeunit="days")


@router.get("/performance/weekly", response_model=DailyWeeklyMonthly, tags=["info"])
def performance_weekly(
    timescale: int = 4, rpc: RPC = Depends(get_rpc), config: dict[str, Any] = Depends(get_config)
) -> dict[str, Any]:
    return rpc._rpc_period_performance(timescale=timescale, timeunit="weeks")


@router.get("/performance/monthly", response_model=DailyWeeklyMonthly, tags=["info"])
def performance_monthly(
    timescale: int = 3, rpc: RPC = Depends(get_rpc), config: dict[str, Any] = Depends(get_config)
) -> dict[str, Any]:
    return rpc._rpc_period_performance(timescale=timescale, timeunit="months")


"""
##############
Account
    positions & open position
    tradepnls & tradepnl
    trades & tradepnl
    account balance
    open_orders
##############
"""


@router.get("/balance", tags=["info"])  # response_model=Balances,
def balance(rpc: RPC = Depends(get_rpc), config: dict[str, Any] = Depends(get_config)) -> dict[str, Any]:
    """Account Balances"""
    return rpc._rpc_balance(
        config.get("base_currency", ""),
        config.get("fiat_display_currency", ""),
    )


@router.get("/entrycount", response_model=Count, tags=["info"])
def entrycount(rpc: RPC = Depends(get_rpc)) -> dict[str, Any]:
    return rpc._rpc_entry_count()


# Using the responsemodel here will cause a ~100% increase in response time (from 1s to 2s)
# on big databases. Correct response model: response_model=TradeResponse
@router.get(
    "/tradepnls",
    tags=["info", "trading"],
    response_model=TradePnLResponse,
    response_model_exclude={"trades"},
)
def tradepnls(
    symbol: str | None = None,
    limit: int = 500,
    offset: int = 0,
    open_only: bool = False,
    rpc: RPC = Depends(get_rpc),
) -> dict[str, Any]:
    """
    TradePnLs
    """
    return rpc._rpc_tradepnls(
        symbols=[symbol] if symbol else None,
        limit=limit,
        offset=offset,
        open_only=open_only,
        order_by_id=True,
    )


@router.get("/trades", tags=["info", "trading"], response_model=TradeResponse)
def trades(
    symbol: str | None = None,
    limit: int = 500,
    offset: int = 0,
    open_only: bool = False,
    rpc: RPC = Depends(get_rpc),
) -> dict[str, Any]:
    """
    Trades
    """
    return rpc._rpc_trades(
        symbols=[symbol] if symbol else None,
        limit=limit,
        offset=offset,
        order_by_id=True,
    )


@router.get("/orders", tags=["info", "trading"], response_model=OrderResponse)
def open_order(
    symbol: str | None = None,
    limit: int = 500,
    offset: int = 0,
    open_only: bool = False,
    rpc: RPC = Depends(get_rpc),
) -> dict[str, Any]:
    """
    Open Orders
    """
    return rpc._rpc_open_orders(
        symbols=[symbol] if symbol else None,
        limit=limit,
        offset=offset,
        order_by_id=True,
    )


@router.get(
    "/positions",
    response_model=list[PositionData],
    response_model_exclude={"trades", "tradepnls"},
    tags=["info"],
)
def positions(symbol: str | None = None, open_only: bool = False, rpc: RPC = Depends(get_rpc)) -> list[PositionData]:
    return rpc._rpc_positions(symbols=[symbol] if symbol else None, open_only=open_only)


# @router.get("/trade/{symbol}", response_model=TradePnL, tags=["info", "trading"])
# def trade(symbol: str, rpc: RPC = Depends(get_rpc)):
#     '''
#     Open TradePnL
#     '''
#     try:
#         return rpc._rpc_trade_status([symbol])[0]
#     except (RPCException, KeyError):
#         raise HTTPException(status_code=404, detail="Trade not found.")


@router.delete("/trades/{order_id}/{symbol}/open-order", response_model=TradePnL, tags=["trading"])
def cancel_open_order(order_id: str, symbol: str, rpc: RPC = Depends(get_rpc)) -> TradePnL:
    rpc._rpc_cancel_open_order(order_id, symbol)
    return rpc._rpc_trade_status([symbol])[0]


@router.post("/forceenter", response_model=OrderData, tags=["trading"])
def force_entry(payload: ForceEnterPayload, rpc: RPC = Depends(get_rpc)) -> OrderData:
    # ordertype = payload.ordertype.value if payload.ordertype else None

    # trade = rpc._rpc_force_entry(
    #     payload.symbol,
    #     side=payload.side,
    #     price=payload.price,
    #     order_type=ordertype,
    #     amount=payload.amount,
    #     quantity=payload.quantity,
    #     cost=payload.cost,
    #     enter_reason=payload.enter_reason or "force_entry",
    #     leverage=payload.leverage,
    # )

    # if trade:
    #     return ForceEnterResponse.model_validate(trade.to_json())
    # else:
    #     return ForceEnterResponse.model_validate(
    #         {"status": f"Error entering {payload.side} trade for pair {payload.pair}."}
    #     )
    return rpc._rpc_force_entry(
        payload.symbol,
        side=payload.side,
        price=payload.price,
        # order_type=ordertype,
        amount=payload.amount,
        quantity=payload.quantity,
        cost=payload.cost,
        enter_reason=payload.enter_reason or "force_entry",
        leverage=payload.leverage,
    )


@router.post("/forceexit", response_model=OrderData, tags=["trading"])
def forceexit(payload: ForceExitPayload, rpc: RPC = Depends(get_rpc)) -> OrderData:
    # ordertype = payload.ordertype.value if payload.ordertype else None
    return rpc._rpc_force_exit(symbol=str(payload.symbol), price=payload.price)


@router.post("/trades/{symbol}/reload", response_model=TradePnL, tags=["trading"])
def trade_reload(symbol: str, rpc: RPC = Depends(get_rpc)) -> TradePnL:
    rpc._rpc_reload_trade_from_exchange(symbol)
    return rpc._rpc_trade_status([symbol])[0]


@router.get("/strategies", response_model=StrategyListResponse, tags=["strategy"])
def list_strategies(config: dict[str, Any] = Depends(get_config), rpc: RPC = Depends(get_rpc)) -> dict[str, Any]:
    strategies = rpc._db.load_strategies()
    print(f">>> [list_strategies] strategies:{strategies}")

    # strategies = sorted(strategies, key=lambda x: x["name"])

    return {"strategies": strategies}


@router.get("/strategy/{strategy}", response_model=StrategyResponse, tags=["strategy"])
def get_strategy(
    strategy: str, config: dict[str, Any] = Depends(get_config), rpc: RPC = Depends(get_rpc)
) -> dict[str, Any]:
    if ":" in strategy:
        raise HTTPException(status_code=500, detail="base64 encoded strategies are not allowed.")
    try:
        strategies = rpc._db.load_strategy(strategy=strategy)
        print(f">>> [get_strategy] strategies:{strategies}")

    except OperationalException:
        raise HTTPException(status_code=404, detail="Strategy not found")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {
        "strategy": strategies.get("strategy"),
        "code": strategies.get("code"),
        "timeframe": strategies.get("timeframe"),
    }
