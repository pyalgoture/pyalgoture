# import argparse
# import os
# from datetime import datetime

# from dateutil.parser import parse

# from .util_io import retrieve_strategy_class


# def _valid_datetime(s: str) -> datetime:
#     try:
#         return parse(s)
#     except ValueError:
#         msg = f"Not a valid datetime: '{s}'."
#         raise argparse.ArgumentTypeError(msg)


# def get_args() -> dict:
#     """
#     --strategy: path of strategy py file
#     --balance: balance
#     --codes: list of symbols
#     --interval:
#     --start: start date
#     --end
#     --path
#     --basecurr
#     --benchmark
#     --broker

#     --commission_rate
#     --min_commission

#     put the quick start into init

#     Action: it defines how to handle command-line arguments: store it as a constant,
#     append into a list, store a boolean value etc. There are 6 built-in actions available

#     Metavar: it provides a different name for optional argument in help messages.

#     args list:
#         parser.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True)
#         # Use like:
#         # python arg.py -l 1234 2345 3456 4567

#         parser.add_argument('-l','--list', action='append', help='<Required> Set flag', required=True)
#         # Use like:
#         # python arg.py -l 1234 -l 2345 -l 3456 -l 4567

#     def quick_start(strategyClass:Strategy, broker:str, balance:float, pairs:list, interval:str, start_date:datetime, end_date:datetime,
#         base_path:str=None, benchmark_code:str = 'BTC/USDT', base_currency:str ='USDT', commission_rate:float =0.001, min_commission=None, is_realtime=False) -> TradingManagementSystem:

#     """

#     parser = argparse.ArgumentParser(description="Areix IO qucik start")
#     parser.add_argument(
#         "--version", action="store_true", default=False, help="check current version"
#     )
#     parser.add_argument(
#         "--strategy",
#         "-s",
#         required=True,
#         help="strategy file. E.g. -s <path to your strategy>/strategy.py",
#     )
#     parser.add_argument(
#         "--pairs",
#         "-p",
#         type=str,
#         metavar="OPTION=VALUE",
#         action="append",
#         default=[],
#         required=True,
#         help="specify the trading pairs. This option can be specified multiple times. E.g. -p BTC/USDT -p ETH/USDC",
#     )
#     parser.add_argument(
#         "--broker",
#         "-b",
#         choices=["local", "binance", "bybit", "eqonex"],
#         help="indicate the broker to conduct backtesting, possible values: [local | binance | bybit | eqonex]",
#     )
#     parser.add_argument(
#         "--realtime",
#         "-r",
#         default=False,
#         type=bool,
#         help="running backtesting or realtime E.g. --realtime True",
#     )
#     parser.add_argument(
#         "--balance",
#         "-c",
#         required=True,
#         type=float,
#         help="amount for trading E.g. --balance 1000",
#     )
#     parser.add_argument(
#         "--start",
#         "-sd",
#         type=_valid_datetime,
#         help="start date of backtesting. E.g. --start '20210101 101010'",
#     )
#     parser.add_argument(
#         "--end",
#         "-ed",
#         default=datetime.now(),
#         type=_valid_datetime,
#         help="end date of backtesting. E.g. --end '20210101 101010'",
#     )
#     parser.add_argument(
#         "--interval",
#         "-i",
#         type=str,
#         required=True,
#         help="datafeed interval of backtesting. E.g. --interval 1h",
#     )
#     parser.add_argument(
#         "--path",
#         help="data/output store path. E.g. --path /<path to store all outputs>",
#     )
#     parser.add_argument(
#         "--benchmark",
#         "-be",
#         default="BTC/USDT",
#         type=str,
#         help="benchmark for statistics calculation and graph E.g. --benchmark BTC/USDT",
#     )
#     parser.add_argument(
#         "--basecurr",
#         "-bc",
#         default="BTC/USDT",
#         type=str,
#         help="base currency for trading E.g. --basecurr USDT",
#     )
#     parser.add_argument(
#         "--commissionrate",
#         "-cr",
#         default=0.001,
#         type=float,
#         help="commission rate for trading E.g. --commissionrate 0.0001",
#     )
#     parser.add_argument(
#         "--mincommission",
#         "-mc",
#         type=float,
#         help="minium commission for trading E.g. --mincommission 10",
#     )

#     out_args = {}
#     args = parser.parse_args()
#     # print(args)

#     if not os.path.exists(args.strategy):
#         raise TypeError("Please specify a valid strategy file.")
#     else:
#         out_args["strategyClass"] = retrieve_strategy_class(args.strategy)
#     out_args["broker"] = args.broker
#     if not args.realtime and args.broker != "local":
#         raise TypeError("Real broker cannot conduct backtesting.")
#     else:
#         out_args["is_realtime"] = args.realtime

#     out_args["balance"] = args.balance
#     out_args["pairs"] = args.pairs
#     out_args["interval"] = args.interval
#     out_args["start_date"] = args.start
#     out_args["end_date"] = args.end
#     out_args["base_path"] = args.path
#     out_args["benchmark_code"] = args.benchmark
#     out_args["base_currency"] = args.basecurr
#     out_args["commission_rate"] = args.commissionrate
#     out_args["min_commission"] = args.mincommission

#     return out_args
