"""
This module contains class to manage RPC communications (Telegram, API, ...)
"""

import signal
import time
from collections import deque
from typing import Any

from ..context import Context
from ..orchestrators.realtime import RealTime
from ..utils.logger import get_logger
from ..utils.objects import NO_ECHO_MESSAGES, PositionData, RPCMessageType, TradeData
from .rpc import RPC, RPCHandler
from .rpc_db import RpcDatabase
from .rpc_types import RPCSendMsg

logger = get_logger()


class RPCManager:
    """
    Class to manage RPC objects (Telegram, API, ...)
    """

    def on_trade_callback_custom(self, ctx: Context, trade: TradeData) -> None:
        """
        Store data in db
        """
        pos: PositionData = ctx.get_position(
            symbol=trade.symbol,
            exchange=trade.exchange,
            position_side=trade.position_side,
        )
        trade_data: dict = trade.to_dict(return_obj=False)
        trade_data["strategy"] = self.strategy_name
        print(f">>>>> [rpc_manager - on_trade_callback_custom] pos:{pos}; trade_data:{trade_data}")
        """
        >>>>> [rpc_manager - on_trade_callback_custom]
        pos:<PositionData symbol:BTCUSDT, asset_type:AssetType.PERPETUAL, exchange:Exchange.BYBIT, position_side:PositionSide.NET, leverage: 5, multiplier:1, consider_price:True, is_contract: True, is_inverse: None, is_isolated: False, commission_rate: None, maker_commission_rate: 0.0002, taker_commission_rate: 0.00055; avg_open_price:94701.2, size:0.006; market_value:0.0; notional_value:568.2072, net_investment:454.66092, total_pnl:-0.3444(rlze:-0.3444 & unrlze:0.0), total_commission:4.68915678, total_turnover:1705.14792, liquidation_price:-8019562.035128, is_open:True, last_price:94701.2>
        trade_data:{'symbol': 'BTCUSDT', 'code': 'BTCUSDT', 'asset_type': 'PERPETUAL', 'exchange': 'BYBIT', 'ori_order_id': 'b4a3cdd8-3378-494f-8791-36e00bc80509', 'order_id': 'aio-1745638680-2gufo4qch8-00012', 'price': 94701.2, 'quantity': 0.006, 'side': 'BUY', 'position_side': 'NET', 'traded_at': datetime.datetime(2025, 4, 26, 3, 38, 0, 264000, tzinfo=<UTC>), 'commission': 0.31251396, 'commission_rate': 0.00055, 'commission_asset': None, 'is_open': True, 'is_maker': False, 'traded_at_ts': 1745638680264, 'action': 'OPEN LONG', 'execution_type': 'TRADE', 'gross_amount': 568.2072, 'pnl': 0, 'leverage': 5, 'msg': 'long', 'strategy': 'TestStrategy'}

        """
        self.db.save_trade(trade_data)
        stra = self.db.load_strategy(strategy=self.strategy_name)
        print(f">>>>>> [rpc_manager - on_trade_callback_custom] stra:{stra}")
        if stra.get("bot_start") is None:
            stra["bot_start"] = trade_data["traded_at"]
            self.db.update_strategy(strategy=self.strategy_name, stra_data=stra)

    def __init__(self, engine: RealTime, config: dict[str, Any]) -> None:
        """Initializes all enabled rpc modules"""
        self.registered_modules: list[RPCHandler] = []
        self.strategy_name = engine.StrategyClass.__name__
        engine._setup_custom_trade_callback(self.on_trade_callback_custom)

        self.db: RpcDatabase = RpcDatabase(config.get("db", {}).get("path"), logger=engine.logger)

        self._rpc = RPC(engine, config, self.db)

        # Enable telegram
        if config.get("telegram", {}).get("enabled", False):
            logger.info("Enabling rpc.telegram ...")
            from .third_party.telegram import Telegram

            print(f">>>>>> [rpc_manager - __init__ - register telegram] config:{config}")
            self.registered_modules.append(Telegram(self._rpc, config))

        # Enable discord
        if config.get("discord", {}).get("enabled", False):
            logger.info("Enabling rpc.discord ...")
            from .third_party.discord import Discord

            self.registered_modules.append(Discord(self._rpc, config))

        # Enable Webhook
        if config.get("webhook", {}).get("enabled", False):
            logger.info("Enabling rpc.webhook ...")
            from .third_party.webhook import Webhook

            self.registered_modules.append(Webhook(self._rpc, config))

        # Enable local rest api server for cmd line control
        if config.get("api_server", {}).get("enabled", False):
            logger.info("Enabling rpc.api_server")
            from .server.server import ApiServer

            apiserver = ApiServer(config)
            apiserver.add_rpc_handler(self._rpc)
            self.registered_modules.append(apiserver)

        # Don't send startup messages immediately - wait for proper initialization
        logger.info(f"RPCManager initialized with {len(self.registered_modules)} modules")

        # Store startup config for delayed sending
        self._startup_config = {
            "strategy": self.strategy_name,
            "bot_startup": "2025-09-27 00:00:00",
            "state": "RUNNING",
            "symbols": ["BTCUSDT", "ETHUSDT"],
        }

    def send_startup_messages_when_ready(self):
        """Send startup messages after all RPC modules are ready"""

        # Wait a short time for all modules to initialize
        time.sleep(2)

        # Check if Telegram is ready
        telegram_ready = True
        for mod in self.registered_modules:
            if hasattr(mod, "name") and mod.name == "telegram":
                # Check if Telegram bot has the is_ready method and is ready
                if hasattr(mod, "is_ready"):
                    telegram_ready = mod.is_ready()
                    if not telegram_ready:
                        logger.warning("Telegram bot is not ready yet, delaying startup messages")
                        return
                break

        if telegram_ready:
            logger.info("All RPC modules ready, sending startup messages")
            self.startup_messages(self._startup_config)

    # def start(self) -> int:
    #     """
    #     Main entry point for trading mode
    #     """

    #     def term_handler(signum: int, frame: Any) -> None:
    #         # Raise KeyboardInterrupt - so we can handle it in the same way as Ctrl-C
    #         raise KeyboardInterrupt()

    #     try:
    #         signal.signal(signal.SIGTERM, term_handler)
    #         while True:
    #             time.sleep(1)
    #     finally:
    #         pass

    #     return 0

    def cleanup(self) -> None:
        """Stops all enabled rpc modules"""
        logger.info("Cleaning up rpc modules ...")
        while self.registered_modules:
            mod = self.registered_modules.pop()
            logger.info(f"Cleaning up rpc.{mod.name} ...")
            mod.cleanup()
            del mod

    def send_msg(self, msg: RPCSendMsg) -> None:
        """
        Send given message to all registered rpc modules.
        A message consists of one or more key value pairs of strings.
        e.g.:
        {
            'status': 'stopping bot'
        }
        """
        print(f">>>>>> [rpc_manager - send_msg] msg:{msg}")
        if msg.get("type") not in NO_ECHO_MESSAGES:
            logger.info(f"Sending rpc message: {msg}")
        for mod in self.registered_modules:
            logger.debug(f"Forwarding message to rpc.{mod.name}")
            try:
                mod.send_msg(msg)
            except NotImplementedError:
                logger.error(f"Message type '{msg['type']}' not implemented by handler {mod.name}.")
            except Exception:
                logger.exception(f"Exception occurred within RPC module {mod.name}")

    # def process_msg_queue(self, queue: deque) -> None:
    #     """
    #     Process all messages in the queue.
    #     """
    #     while queue:
    #         msg = queue.popleft()
    #         logger.info(f"Sending rpc strategy_msg: {msg}")
    #         for mod in self.registered_modules:
    #             if mod._config.get(mod.name, {}).get("allow_custom_messages", False):
    #                 mod.send_msg(
    #                     {
    #                         "type": RPCMessageType.STRATEGY_MSG,
    #                         "msg": msg,
    #                     }
    #                 )

    def startup_messages(
        self,
        config: dict[str, Any],
    ) -> None:
        if config.get("dry_run", False):
            self.send_msg(
                {
                    "type": RPCMessageType.WARNING,
                    "status": "Dry run is enabled. All trades are simulated.",
                }
            )

        strategy_name = config["strategy"]
        bot_startup = config["bot_startup"]
        state = config["state"]
        symbols = config["symbols"]
        print(">>>>>> [rpc_manager - startup_messages]")
        self.send_msg(
            {
                "type": RPCMessageType.STARTUP,
                "status": f"*Strategy:* `{strategy_name}`\n"
                f"*State:* `{state}`\n"
                f"*Symbols:* `{symbols}`\n"
                f"*Bot startup:* `{bot_startup}`",
            }
        )
