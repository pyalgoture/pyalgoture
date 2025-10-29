# from memory_profiler import profile
import gc
import multiprocessing as mp
import os
import sys
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache, partial
from itertools import chain, compress, product, repeat
from random import choice, getstate, random, setstate
from typing import Any

import numpy as np
import pandas as pd

from ..broker import Broker
from ..context import Context
from ..datafeed import DataFeed
from ..orchestra import Orchestrator
from ..statistic import Statistic
from ..strategy import Strategy
from ..utils.event_engine import EVENT_TICK, Event
from ..utils.objects import AssetType, Exchange, TickData
from ..utils.util_objects import AttrDict


def set_mp_start_method() -> None:
    """Set multiprocessing start method based on platform."""
    if sys.platform == "win32":
        mp.set_start_method("spawn", force=True)
    else:
        mp.set_start_method("fork", force=True)


class BackTest(Orchestrator):
    """Orchestrator for conducting backtesting operations."""

    def __init__(
        self,
        feeds: list[DataFeed] | DataFeed,
        StrategyClass: Strategy,
        trade_history: list[Any] | None = None,
        store_path: str = "",
        broker: Broker | None = None,
        statistic: Statistic | None = None,
        backtest_mode: str = "bar",
        do_print: bool = True,
        debug: bool = False,
        logger: Any | None = None,
        benchmark: DataFeed | None = None,
        prefix: str = "",
        hist_trade_skip_datafeed: bool = False,
        hist_trade_skip_checking: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize BackTest orchestrator for conducting backtesting.

        Args:
            feeds: Data feeds for backtesting
            StrategyClass: Strategy class to be tested
            trade_history: Historical trade data. Example:
                [
                    {
                        'symbol': 'BTC/USDT',
                        'datetime': datetime(2021, 3, 30, 20, 58, 8),
                        'quantity': 0.003,
                        'price': 30304,
                        'side': 'BUY',
                        'asset_type': 'SPOT',
                        'leverage': 10
                    },
                    ...
                ]
            store_path: Path to store results
            broker: Broker instance
            statistic: Statistic instance
            backtest_mode: Mode for backtesting ('bar', 'tick', 'bar_tick')
            do_print: Whether to print output
            debug: Debug mode flag
            logger: Logger instance
            benchmark: Benchmark data feed
            prefix: Prefix for output files
            hist_trade_skip_datafeed: Skip datafeed for historical trades
            hist_trade_skip_checking: Skip checking for historical trades
            **kwargs: Additional keyword arguments
        """
        if trade_history is None:
            trade_history = []

        super().__init__(
            feeds=feeds,
            StrategyClass=StrategyClass,
            logger=logger,
            store_path=store_path,
            broker=broker,
            statistic=statistic,
            trade_history=trade_history,
            prefix=prefix,
            hist_trade_skip_datafeed=hist_trade_skip_datafeed,
            hist_trade_skip_checking=hist_trade_skip_checking,
            debug=debug,
            benchmark=benchmark,
            **kwargs,
        )

        self.do_print = do_print
        if backtest_mode not in ["bar", "tick", "bar_tick"]:
            raise ValueError(f"backtest_mode must be one of ['bar', 'tick', 'bar_tick'], got {backtest_mode}")
        self.backtest_mode = backtest_mode

    def start(self) -> None:
        """Start the backtesting process."""
        self.event_engine.start()

        self.add_runner(self.broker)
        self.add_runner(self.strategy)
        if self.statistic:
            self.add_runner(self.statistic)

        self.ctx = self.context = Context(
            event_engine=self.event_engine,
            feeds=self.feeds,
            broker=self.broker,
            strategy=self.strategy,
            statistic=self.statistic,
            benchmark=self.benchmark,
            do_print=self.do_print,
            path=self.path,
            logger=self.logger,
        )
        self.ctx.tick = self.tradedays[0] if self.tradedays else None
        self.ctx.trade_history = self.trade_history
        self.ctx.backtest_mode = self.backtest_mode

        # runner refers to those object with initialize, finish, run functions (inherent from Base Class)
        # default sequence: broker, strategy, statistic
        self._full_runner_list = list(chain(self._pre_hook_list, self._runner_list, self._post_hook_list))
        self.ctx.runners = self._full_runner_list

        # before loop, invoke all the runner's initialize() and bind the ctx obj
        for runner in self._full_runner_list:
            runner.ctx = runner.context = self.ctx
            runner.logger = self.logger

        for runner in self._full_runner_list:
            runner.initialize()

        # Backtest Main Entry
        for tick in self.tradedays:
            # Execute trade history
            if self.ctx.trade_history and tick in self.ctx.trade_history:
                for trade in self.ctx.trade_history[tick]:
                    self.execute_history_trade(trade)

            # Set bar/tick data
            if self.ctx.backtest_mode == "tick":
                self._process_tick_mode(tick)
            else:
                self._process_bar_mode(tick)

        for runner in self._full_runner_list:
            runner.finish()

        while not self.event_engine.empty:
            pass

        self.event_engine.stop()

    def _process_tick_mode(self, tick) -> None:
        """Process tick mode data."""
        tick_datas = []

        for aio_symbol, feed in self.feeds.items():
            hist = feed.data
            if tick not in hist:
                continue
            tick_data = hist.get(tick)
            if not tick_data:
                continue
            tick_datas.append(tick_data)

        for tick_data in tick_datas:
            tick_data["aio_symbol"] = f"{tick_data['symbol']}|{tick_data['exchange']}"
            tick_data["exchange"] = Exchange(tick_data["exchange"])
            tick_data["asset_type"] = AssetType(tick_data["asset_type"])
            tick_obj = TickData(**tick_data)
            event = Event(type=EVENT_TICK, data=tick_obj, priority=1)
            self.process_tick_event(event=event)
            self.ctx.strategy.process_tick_event(event=event)

    def _process_bar_mode(self, tick) -> None:
        """Process bar mode data."""
        bar_data = {}

        for aio_symbol, feed in self.feeds.items():
            hist = feed.data
            if tick not in hist:
                continue
            bar = hist.get(tick)
            if not bar:
                continue
            bar_data[aio_symbol] = bar

        self.set_bar_data(bars=bar_data, tick=tick)

    def optimize(
        self,
        maximize: str | Callable[[pd.Series], float] = "sharpe_ratio",
        constraint: Callable[[dict[Any, Any]], bool] | None = None,
        return_heatmap: bool = False,
        return_mean_value: bool = False,
        return_optimization: bool = False,
        display_attributes: list[str] | None = None,
        method: str = "grid",
        max_tries: int | float | None = None,  # for skopt
        random_state: int | None = None,  # for skopt
        max_workers: int | None = None,  # for ga
        population_size: int | None = None,  # for ga
        ngen_size: int | None = None,  # for ga
        mutation_rate: float | None = None,  # for ga
        settings: list[dict[str, Any]] | None = None,
        cache_path: str | None = None,
        **kwargs: Any,
    ) -> pd.Series | tuple[pd.Series, pd.Series]:
        """Optimize strategy parameters using various optimization methods.

        Args:
            maximize: Metric to maximize (str) or custom function
            constraint: Function to validate parameter combinations
            return_heatmap: Whether to return optimization heatmap
            return_mean_value: Whether to return mean values
            return_optimization: Whether to return optimization result
            display_attributes: Additional attributes to display
            method: Optimization method ('grid', 'skopt', 'optuna', 'ga')
            max_tries: Maximum number of trials
            random_state: Random state for reproducibility
            max_workers: Number of parallel workers
            population_size: Population size for genetic algorithm
            ngen_size: Number of generations for genetic algorithm
            mutation_rate: Mutation rate for genetic algorithm
            settings: Predefined parameter settings
            cache_path: Path to cache results
            **kwargs: Strategy parameters to optimize

        Returns:
            Optimization results based on return flags
        """
        if display_attributes is None:
            display_attributes = []
        if settings is None:
            settings = []

        try:
            from tqdm.auto import tqdm
        except ImportError:
            raise ImportError("Need package 'tqdm' for optimize function. pip install tqdm")

        if not kwargs and not settings:
            raise ValueError("The optimization requires some strategy parameters.")

        maximize_key = None
        if isinstance(maximize, str):
            maximize_key = str(maximize)
            valid_stats = {
                "ending_balance",
                "total_net_pnl",
                "gross_profit",
                "gross_loss",
                "profit_factor",
                "return_on_initial_capital",
                "annualized_return",
                "total_return",
                "max_return",
                "min_return",
                "number_trades",
                "number_winning_trades",
                "number_losing_trades",
                "avg_daily_trades",
                "avg_weekly_trades",
                "avg_monthly_trades",
                "win_ratio",
                "loss_ratio",
                "win_days",
                "loss_days",
                "max_win_in_day",
                "max_loss_in_day",
                "max_consecutive_win_days",
                "max_consecutive_loss_days",
                "avg_profit_per_trade",
                "trading_period",
                "avg_daily_pnl($)",
                "avg_daily_pnl",
                "avg_weekly_pnl($)",
                "avg_weekly_pnl",
                "avg_monthly_pnl($)",
                "avg_monthly_pnl",
                "avg_quarterly_pnl($)",
                "avg_quarterly_pnl",
                "avg_annualy_pnl($)",
                "avg_annualy_pnl",
                "sharpe_ratio",
                "sortino_ratio",
                "annualized_volatility",
                "omega_ratio",
                "downside_risk",
                "information_ratio",
                "beta",
                "alpha",
                "calmar_ratio",
                "tail_ratio",
                "stability_of_timeseries",
                "max_drawdown",
                "max_drawdown_duration",
                "sqn",
            }

            if maximize not in valid_stats:
                raise ValueError("`maximize`, if str, must match a key in pd.Series result of backtest.start()")

            def maximize(stats: pd.Series, _key=maximize):
                return stats[_key]

        elif not callable(maximize):
            raise TypeError(
                "`maximize` must be str (a field of backtest.start() result "
                "Series) or a function that accepts result Series "
                "and returns a number; the higher the better"
            )

        have_constraint = bool(constraint)
        if constraint is None:
            # def constraint(_):
            #     return True
            constraint = lambda _: True
        elif not callable(constraint):
            raise TypeError(
                "`constraint` must be a function that accepts a dict "
                "of strategy parameters and returns a bool whether "
                "the combination of parameters is admissible or not"
            )

        if return_optimization and method not in ["skopt", "optuna"]:
            raise ValueError("return_optimization=True only valid if method='skopt' or method='optuna'")

        def _tuple(x: Any) -> tuple[Any, ...]:
            """Convert value to tuple if not already a sequence."""
            return x if isinstance(x, Sequence) and not isinstance(x, str) else (x,)

        for k, v in kwargs.items():
            if len(_tuple(v)) == 0:
                raise ValueError(f"Optimization variable '{k}' is passed no optimization values: {k}={v}")

        _hist_cache = {}
        _param_keys = list(settings[0].keys()) if settings else list(kwargs.keys())

        # Read cache from file if path is provided
        cache_file_exists = cache_path and (
            os.path.exists(cache_path) or os.path.exists(os.path.splitext(cache_path)[0] + ".parquet")
        )

        if cache_file_exists:
            try:
                import pyarrow  # noqa: F401
            except ImportError:
                raise ImportError("Need package 'pyarrow' for parquet file. pip install pyarrow")

            parquet_path = os.path.splitext(cache_path)[0] + ".parquet"

            if os.path.exists(parquet_path):
                hist_df = pd.read_parquet(parquet_path)
            elif cache_path.endswith(".csv"):
                hist_df = pd.read_csv(cache_path, low_memory=False)
            elif cache_path.endswith((".xls", ".xlsx", ".xlsm", ".xlsb")):
                hist_df = pd.read_excel(cache_path)
            else:
                raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

            if len(hist_df) < 300:
                print(f"Found existing heatmap: {hist_df}")
            else:
                print("Found existing heatmap, but too many rows to display")

            # Convert DataFrame to dictionary and update cache
            for index, row in hist_df.iterrows():
                key = tuple(
                    str(int(row[k]) if isinstance(row[k], float) and row[k] == int(row[k]) else row[k])
                    for k in _param_keys
                )
                _hist_cache[key] = {k: row[k] for k in hist_df.columns if k not in _param_keys}
            print(f"Loaded {len(_hist_cache)} cached results from {cache_path}")

        _save_cache_func = partial(
            BackTest._save_cache,
            hist_cache=_hist_cache,
            param_keys=_param_keys,
            cache_path=cache_path,
            maximize_key=maximize_key,
            display_attributes=display_attributes,
        )

        # @profile
        def _optimize_grid():
            """Optimize using grid search method."""
            if settings:
                param_combos = settings
            else:
                param_combos = tuple(
                    map(
                        dict,
                        filter(
                            constraint,
                            map(
                                AttrDict,
                                product(*(zip(repeat(k), _tuple(v)) for k, v in kwargs.items())),
                            ),
                        ),
                    )
                )

            print(f"Total parameter combinations: {len(param_combos)}")
            # Filter out already cached combinations
            param_combos = [
                params for params in param_combos if tuple(str(pv) for pv in params.values()) not in _hist_cache
            ]
            print(f"New combinations to evaluate: {len(param_combos)}")

            if not param_combos:
                raise ValueError("No admissible parameter combinations to test")

            if len(param_combos) > 300:
                warnings.warn(
                    f"Searching for best of {len(param_combos)} configurations.",
                    stacklevel=2,
                )
            else:
                print(f"Parameters Combos({len(param_combos)}):\n", param_combos)

            heatmap = pd.Series(
                np.nan,
                name=maximize_key,
                index=pd.MultiIndex.from_tuples(
                    [list(str(pv) for pv in p.values()) for p in param_combos],
                    names=list(next(iter(param_combos)).keys()),
                ),
            )
            print("Empty Heatmap:\n", heatmap)

            def _batch(seq):
                """Create batches of parameter combinations for parallel processing."""
                n = np.clip(len(seq) // (os.cpu_count() or 1), 5, 300)
                for i in range(0, len(seq), n):
                    yield seq[i : i + n]

            backtest_uuid = np.random.random()
            param_batches = list(_batch(param_combos))
            stats_map = {}

            """
            1. fork{默认值, 适用于 Unix 平台): 子进程通过 fork 系统调用从父进程复制.它继承了父进程的资源状态, 因此速度较快, 但可能导致资源竞争和安全性问题.
            2. spawn{适用于跨平台, 尤其是 Windows): 创建一个全新的 Python 解释器进程, 执行 main 函数.它不继承父进程的资源, 比较安全但启动速度相对较慢.
            3. forkserver{适用于 Unix 平台): 启动一个单独的服务器进程, fork 所有的子进程.这种方法既解决了 fork 的安全问题, 又比 spawn 更快.
            """
            # If multiprocessing start method is 'fork' (i.e. on POSIX), use a pool of processes to compute results in parallel.
            # Otherwise (i.e. on Windows), sequential computation will be "faster".
            set_mp_start_method()

            # Create a manager and a shared dictionary
            manager = mp.Manager()
            cache = manager.dict()
            eval_counter = manager.Value("i", 0)
            save_lock = manager.Lock()
            cache.update(_hist_cache)

            BackTest._mp_backtests[backtest_uuid] = (
                self,
                param_batches,
                maximize,
                cache,
                eval_counter,
                save_lock,
                _save_cache_func if cache_path else None,
            )

            try:
                if mp.get_start_method(allow_none=False) == "fork":
                    with ProcessPoolExecutor() as executor:
                        futures = [
                            executor.submit(BackTest._mp_task, backtest_uuid, i) for i in range(len(param_batches))
                        ]
                        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
                            batch_index, values, stats = future.result()
                            for value, params in zip(values, param_batches[batch_index]):
                                heatmap[tuple(str(pv) for pv in params.values())] = value
                            for stat, params in zip(stats, param_batches[batch_index]):
                                stats_map[tuple(str(pv) for pv in params.values())] = stat
                            gc.collect()
                else:
                    if os.name == "posix":
                        warnings.warn(
                            "For multiprocessing support in `BackTest.optimize()`, "
                            "set multiprocessing start method to 'fork'."
                        )

                    for batch_index in tqdm(range(len(param_batches))):
                        _, values, stats = BackTest._mp_task(backtest_uuid, batch_index)
                        for value, params in zip(values, param_batches[batch_index]):
                            heatmap[tuple(str(pv) for pv in params.values())] = value
                        for stat, params in zip(stats, param_batches[batch_index]):
                            stats_map[tuple(str(pv) for pv in params.values())] = stat
                        gc.collect()
            finally:
                del BackTest._mp_backtests[backtest_uuid]

            return heatmap, stats_map

        def _optimize_optuna():
            """Optimize using Optuna for Bayesian optimization.

            Optuna provides efficient hyperparameter optimization with state-of-the-art
            samplers and pruners.
            """
            try:
                import optuna
                from optuna.pruners import MedianPruner
                from optuna.samplers import TPESampler
            except ImportError:
                raise ImportError("Need package 'optuna' for method='optuna'. pip install optuna")

            # Create a stats map to track results for display attributes
            stats_map = {}

            # For tracking progress
            progress_bar = tqdm(total=max_tries, desc="Backtest.optimize with Optuna")

            # Cache for already evaluated parameter sets to avoid recomputation
            param_evaluated: dict[tuple[Any, ...], float] = {}

            def objective(trial):
                """Define the objective function for Optuna."""
                # Get list of parameters to optimize
                params = {}
                for key, values in kwargs.items():
                    values = np.asarray(values)

                    # Handle different types of parameters
                    if values.dtype.kind in "iumM":  # Integer parameters
                        # For datetime/timedelta, convert to int for processing
                        if values.dtype.kind in "mM":
                            values = values.astype(int)
                        params[key] = trial.suggest_int(key, values.min(), values.max())

                    elif values.dtype.kind == "f":  # Float parameters
                        params[key] = trial.suggest_float(key, values.min(), values.max())

                    else:  # Categorical parameters
                        params[key] = trial.suggest_categorical(key, values.tolist())

                # Skip if parameters don't meet constraints
                if not constraint(AttrDict(params)):
                    return float("-inf")  # Return worst possible value

                # Check if we already evaluated this parameter set
                param_key = tuple(params.values())
                if param_key in param_evaluated:
                    return param_evaluated[param_key]

                # Run backtest with these parameters
                stats = BackTest._run_bt(self, **params)

                # Extract target metric and handle invalid values
                value = maximize(stats)
                if np.isnan(value):
                    value = float("-inf")

                # Store results
                param_key_str = tuple(str(pv) for pv in params.values())
                stats_map[param_key_str] = stats
                param_evaluated[param_key] = value

                # Update progress
                progress_bar.update(1)

                return value

            # Create Optuna study with TPE sampler (good for Bayesian optimization)
            study = optuna.create_study(
                sampler=TPESampler(seed=random_state),
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
                direction="maximize",
            )

            # Optimize with the given number of trials
            try:
                study.optimize(objective, n_trials=max_tries)
            except KeyboardInterrupt:
                print("Optimization interrupted by user.")
            finally:
                progress_bar.close()

            # Get best parameters and their values
            best_params = study.best_params
            best_value = study.best_value

            # Create heatmap compatible with other optimization methods
            trials_df = study.trials_dataframe()

            # Format heatmap series with all trials
            param_names = list(kwargs.keys())
            param_values = []
            score_values = []

            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    param_tuple = tuple(str(trial.params[k]) for k in param_names)
                    param_values.append(param_tuple)
                    score_values.append(trial.value)

            heatmap = pd.Series(dict(zip(param_values, score_values)), name=maximize_key)
            heatmap.index.names = param_names

            # Print optimization results
            print("\nOptuna optimization result:")
            print(f"  Best parameters: {best_params}")
            print(f"  Best value: {best_value}")
            print(f"  Number of finished trials: {len(study.trials)}")

            # Return the optimization result
            optuna_result = {
                "best_params": best_params,
                "best_value": best_value,
                "study": study,
                "trials_df": trials_df,
            }

            return heatmap, stats_map, optuna_result

        def _optimize_skopt():
            """Optimize using scikit-optimize for Bayesian optimization.

            Reference: https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html
            """
            try:
                from skopt import forest_minimize
                from skopt.callbacks import DeltaXStopper
                from skopt.learning import ExtraTreesRegressor
                from skopt.space import Categorical, Integer, Real
                from skopt.utils import use_named_args
            except ImportError:
                raise ImportError("Need package 'scikit-optimize' for method='skopt'. pip install scikit-optimize")

            dimensions = []
            for key, values in kwargs.items():
                values = np.asarray(values)
                if values.dtype.kind in "mM":  # timedelta, datetime64
                    # these dtypes are unsupported in skopt, so convert to raw int
                    values = values.astype(int)

                if values.dtype.kind in "iumM":
                    dimensions.append(Integer(low=values.min(), high=values.max(), name=key))
                elif values.dtype.kind == "f":
                    dimensions.append(Real(low=values.min(), high=values.max(), name=key))
                else:
                    dimensions.append(Categorical(values.tolist(), name=key, transform="onehot"))

            # Avoid recomputing re-evaluations:
            # "The objective has been evaluated at this point before."
            # https://github.com/scikit-optimize/scikit-optimize/issues/302
            stats_map = {}

            def skopt_task(**kwargs: Any) -> Any:
                """Run backtest for given parameters."""
                stats = BackTest._run_bt(self, **kwargs)
                stats_map[tuple(kwargs.values())] = stats
                return stats

            memoized_run = lru_cache()(lambda tup: skopt_task(**dict(tup)))

            # np.inf/np.nan breaks sklearn, np.finfo(float).max breaks skopt.plots.plot_objective
            INVALID = 1e300
            progress = iter(tqdm(repeat(None), total=max_tries, desc="Backtest.optimize"))

            @use_named_args(dimensions=dimensions)
            def objective_function(**params):
                """Objective function for scikit-optimize."""
                next(progress)
                # Check constraints
                if not constraint(AttrDict(params)):
                    return INVALID

                res = memoized_run(tuple(params.items()))
                value = -maximize(res)
                if np.isnan(value):
                    return INVALID
                return value

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "The objective has been evaluated at this point before.")

                res = forest_minimize(
                    func=objective_function,
                    dimensions=dimensions,
                    n_calls=max_tries,
                    base_estimator=ExtraTreesRegressor(n_estimators=20, min_samples_leaf=2),
                    acq_func="LCB",
                    kappa=3,
                    n_initial_points=min(max_tries, 20 + 3 * len(kwargs)),
                    initial_point_generator="lhs",
                    callback=DeltaXStopper(9e-7),
                    random_state=random_state,
                )

            skopt_task(**dict(zip(kwargs.keys(), res.x)))

            heatmap = pd.Series(dict(zip(map(tuple, res.x_iters), -res.func_vals)), name=maximize_key)
            heatmap.index.names = list(kwargs.keys())
            heatmap = heatmap[heatmap != -INVALID]
            heatmap.sort_index(inplace=True)

            valid = res.func_vals != INVALID
            res.x_iters = list(compress(res.x_iters, valid))
            res.func_vals = res.func_vals[valid]

            print("Optimization result:")
            print(f"  x: {res.x}")
            print(f"  fun: {res.fun}")
            print(f"  x_iters: {res.x_iters}")
            print(f"  func_vals: {res.func_vals}")
            print(f"  models: {res.models}")

            return heatmap, stats_map, res

        def _optimize_ga(settings: list[dict[str, Any]] | None = None) -> tuple[pd.Series, dict[Any, Any]]:
            """Optimize using genetic algorithm."""
            if settings is None:
                settings = []

            try:
                from deap import algorithms, base, creator, tools
            except ImportError:
                raise ImportError("Need package 'deap' for method='ga'. pip install deap")

            if settings:
                param_combos = settings
            else:
                param_combos = list(
                    map(
                        dict,
                        filter(
                            constraint,
                            map(
                                AttrDict,
                                product(*(zip(repeat(k), _tuple(v)) for k, v in kwargs.items())),
                            ),
                        ),
                    )
                )

            if len(param_combos) > 300:
                warnings.warn(
                    f"Searching for best of {len(param_combos)} configurations.",
                    stacklevel=2,
                )
            else:
                print(f"Parameters Combos({len(param_combos)}):\n", param_combos)

            heatmap = pd.Series(
                np.nan,
                name=maximize_key,
                index=pd.MultiIndex.from_tuples(
                    [list(str(pv) for pv in p.values()) for p in param_combos],
                    names=list(next(iter(param_combos)).keys()),
                ),
            )

            settings = [list(d.items()) for d in param_combos]

            # Create individual class used in genetic algorithm optimization
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

            stats_map = {}
            BackTest._mp_backtests["ga"] = self
            evaluate_func = BackTest._ga_task
            maximize_func = partial(BackTest._ga_maximize, maximize_key=maximize_key)

            def generate_parameter() -> list:
                """Generate random parameter combination."""
                return choice(settings)

            def mutate_individual(individual: list, indpb: float) -> tuple:
                """Mutate individual by randomly changing parameters."""
                size: int = len(individual)
                paramlist: list = generate_parameter()
                for i in range(size):
                    if random() < indpb:
                        individual[i] = paramlist[i]
                return (individual,)

            # Set up multiprocessing Pool and Manager
            set_mp_start_method()
            ctx = mp.get_context()

            with ctx.Manager() as manager, ctx.Pool(max_workers) as pool:
                # Create shared dict for result cache
                cache: dict[tuple, tuple] = manager.dict()

                # Add history cache
                if _hist_cache:
                    for key, value in _hist_cache.items():
                        cache[key] = value

                # Create a shared counter and lock
                eval_counter = manager.Value("i", 0)
                save_lock = manager.Lock()

                # Set up toolbox (sequence doesn't matter, unless has dependency)
                toolbox = base.Toolbox()
                toolbox.register("individual", tools.initIterate, creator.Individual, generate_parameter)
                toolbox.register("population", tools.initRepeat, list, toolbox.individual)
                toolbox.register("mate", tools.cxTwoPoint)  # crossover operator
                toolbox.register("mutate", mutate_individual, indpb=1)  # mutation operator
                toolbox.register("select", tools.selNSGA2)  # selection operator
                toolbox.register("map", pool.map)  # map operator
                toolbox.register(
                    "evaluate",
                    self._ga_evaluate,
                    cache=cache,
                    evaluate_func=evaluate_func,
                    maximize_func=maximize_func,
                    save_lock=save_lock,
                    eval_counter=eval_counter,
                    save_cache_func=_save_cache_func if cache_path else None,
                )

                total_size: int = len(settings)
                pop_size: int = (
                    population_size
                    if population_size
                    else min(max(int((len(settings) * 2 + len(settings) * 4) / 2), 30), 300)
                )
                lambda_: int = pop_size
                mu: int = int(pop_size * 0.8)

                if mutation_rate is not None:
                    mutpb: float = mutation_rate
                    cxpb: float = 1 - mutpb
                else:
                    cxpb: float = 0.95
                    mutpb: float = 1 - cxpb

                ngen: int = ngen_size if ngen_size else min(max(int(len(settings) * 1.5), 20), 100)

                pop: list = toolbox.population(pop_size)

                # Run ga optimization
                print("Start executing genetic algorithm optimization")
                print(f"Using {max_workers} CPU cores")
                print(f"Total number of individuals in the population: {total_size}")
                print(f"Number of individuals in each generation(Population Size): {pop_size}")
                print(f"Number of individuals selected for the next generation(Elite Size): {mu}")
                print(f"Number of offspring generated per generation(Lambda Size): {lambda_}")
                print(f"Number of generations: {ngen}")
                print(f"Crossover probability: {cxpb:.0%}")
                print(f"Mutation probability: {mutpb:.0%}")

                try:
                    algorithms.eaMuPlusLambda(
                        pop,
                        toolbox,
                        mu,
                        lambda_,
                        cxpb,
                        mutpb,
                        ngen,
                        verbose=True,
                    )
                finally:
                    BackTest._mp_backtests.pop("ga", None)

                # Store results in stats_map and heatmap
                for params, stats in cache.items():
                    stats_map[params] = stats
                    heatmap[params] = maximize_func(stats)

            return heatmap, stats_map

        # Execute optimization based on method
        if method == "grid":
            heatmap, stats_map = _optimize_grid()
            optimize_result = None
        elif method == "skopt":
            heatmap, stats_map, optimize_result = _optimize_skopt()
        elif method == "optuna":
            heatmap, stats_map, optimize_result = _optimize_optuna()
        elif method == "ga":
            heatmap, stats_map = _optimize_ga(settings=settings)
            optimize_result = None
        else:
            raise ValueError(f"Method should be 'grid', 'skopt', 'ga', or 'optuna', not {method!r}")

        # Prepare output
        output = []

        heatmap = heatmap.to_frame()
        if display_attributes:
            for attr in display_attributes:
                heatmap[attr] = heatmap.index.to_series().apply(lambda x: stats_map[x][attr])

        best_params = heatmap[maximize_key].idxmax()
        output.append(best_params)

        # Save cache if path provided
        if cache_path:
            if cache_path.endswith(".csv"):
                heatmap.to_csv(cache_path, index=True)
            elif cache_path.endswith((".xls", ".xlsx", ".xlsm", ".xlsb")):
                heatmap.to_excel(cache_path, index=True)
            else:
                print(f"Warning: Unsupported file format for {cache_path}. Defaulting to CSV.")
                csv_path = cache_path + ".csv"
                heatmap.to_csv(csv_path, index=True)

        # Add optional outputs
        if return_heatmap:
            output.append(heatmap)

        if return_mean_value:
            numeric_cols = heatmap.select_dtypes(include=["float", "int"]).columns
            mean_value = {}
            for k in _param_keys:
                mean_value[k] = heatmap[numeric_cols].groupby([k]).mean()[maximize_key]
            output.append(mean_value)

        if return_optimization:
            output.append(optimize_result)

        return tuple(output)

    @staticmethod
    def _evaluate_portfolio(
        individual: list[int],
        cache: dict[tuple[int, ...], dict[str, Any]],
        strategies_list: list[dict[str, Any]],
        maximize: str,
        equal_weight: bool,
        combine_strategy_param_name: str | None,
        constraint: Callable[[dict[str, Any]], bool] | None,
    ) -> tuple[float]:
        """Evaluate portfolio individual in genetic algorithm."""
        selected_strategies = [s for i, s in zip(individual, strategies_list) if i]

        if not selected_strategies:
            return (0.0,)

        tp: tuple = tuple(individual)
        in_cache = False

        if tp in cache:
            combined_performance = cache[tp]
            in_cache = True
        else:
            if equal_weight:
                weights = [1 / len(selected_strategies)] * len(selected_strategies)
            else:
                weights = np.random.dirichlet(np.ones(len(selected_strategies)))

            combined_performance = BackTest._run_combined_backtest(
                selected_strategies, weights, combine_strategy_param_name
            )

            if maximize not in combined_performance:
                print(
                    f"[ERROR] Cannot find {maximize} in combined_performance. "
                    f"Selected strategies: {selected_strategies}"
                )
                return (0.0,)

            combined_performance["number_strategies"] = len(selected_strategies)
            cache[tp] = combined_performance

        # Check the constraint
        if constraint and not constraint(combined_performance):
            return (0.0,)

        value: float = combined_performance[maximize]
        if value is None:
            if not in_cache:
                print(f"Warning: combined_performance returned None for {maximize}. Individual: {individual}")
            return (0.0,)

        return (value,)

    @staticmethod
    def _run_combined_backtest(
        strategies: list[dict[str, Any]], weights: list[float], combine_strategy_param_name: str | None
    ) -> dict[str, Any]:
        """Run a combined backtest for multiple strategies with given weights.

        Args:
            strategies: List of strategy configurations
            weights: List of weights for each strategy
            combine_strategy_param_name: Parameter name for combining strategies

        Returns:
            Combined performance metrics
        """
        bt = BackTest._mp_backtests["ga"]

        if combine_strategy_param_name:
            bt.reset()
            setattr(bt.StrategyClass, combine_strategy_param_name, strategies)
            bt.strategy = bt.StrategyClass()
            bt.start()
            stats = bt.ctx.statistic.stats(interval="1d", resample_interval="1d")
            return stats
        else:
            daily_returns = []
            for strategy, weight in zip(strategies, weights):
                bt.reset()
                for k, v in strategy.items():
                    setattr(bt.StrategyClass, k, v)
                bt.strategy = bt.StrategyClass()
                bt.start()
                stats = bt.ctx.statistic.stats(interval="1d", resample_interval="1d")
                daily_returns.append(pd.Series(stats["daily_changes"]))

            # Ensure all strategies have the same date range
            start_date = max(dr.index[0] for dr in daily_returns)
            end_date = min(dr.index[-1] for dr in daily_returns)

            # Adjust date range and apply weights
            weighted_returns = [dr.loc[start_date:end_date] * w for dr, w in zip(daily_returns, weights)]

            # Calculate combined daily return
            combined_daily_return = pd.concat(weighted_returns, axis=1).sum(axis=1)

            # Calculate combined cumulative return
            cumulative_return = (1 + combined_daily_return).cumprod() - 1

            # Calculate combined performance metrics
            total_return = cumulative_return.iloc[-1]
            sharpe_ratio = combined_daily_return.mean() / combined_daily_return.std() * np.sqrt(365)
            max_drawdown = (cumulative_return / cumulative_return.cummax() - 1).min()
            sortino_ratio = (
                combined_daily_return.mean() / combined_daily_return[combined_daily_return < 0].std()
            ) * np.sqrt(365)
            calmar_ratio = combined_daily_return.mean() / max_drawdown * np.sqrt(365)

            combined_results = {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "max_drawdown": max_drawdown,
                "daily_returns": combined_daily_return,
                "cumulative_return": cumulative_return,
            }

        return combined_results

    @staticmethod
    def _save_cache(
        cache,
        hist_cache,
        param_keys,
        cache_path,
        maximize_key,
        display_attributes,
        save_interval: int | None = None,
    ):
        """Save optimization cache to file."""
        cond = (
            (len(cache) > len(hist_cache) and len(cache) % save_interval == 0)
            if save_interval
            else (len(cache) > len(hist_cache))
        )

        if cond:
            out = []
            for key, stats in cache.items():
                d = {k: v for k, v in zip(param_keys, key)}
                d[maximize_key] = stats[maximize_key]

                if display_attributes:
                    for attr in display_attributes:
                        d[attr] = stats[attr]

                if "number_strategies" in stats:
                    d["number_strategies"] = stats["number_strategies"]
                out.append(d)

            out = pd.DataFrame(out)

            parquet_path = os.path.splitext(cache_path)[0] + ".parquet"
            lock_path = parquet_path + ".lock"

            try:
                # Create lock file
                with open(lock_path, "w") as f:
                    f.write("1")

                if "start" in out.columns or "end" in out.columns:
                    out["start"] = out["start"].astype(str)
                    out["end"] = out["end"].astype(str)
                out.to_parquet(parquet_path)
            except Exception as e:
                print(f"Some error occurred when saving cache: {e}")
            else:
                print(f"Cache saved. Current cache size: {len(cache)}. Datetime: {datetime.now()}")
            finally:
                # Always remove the lock file
                if os.path.exists(lock_path):
                    os.remove(lock_path)

    @staticmethod
    def _run_bt(bt: "BackTest", **kwargs: Any) -> Any:
        """Run backtest with given parameters."""
        bt.reset()
        for k, v in kwargs.items():
            setattr(bt.StrategyClass, k, v)
        bt.strategy = bt.StrategyClass()

        bt.start()
        stats = bt.ctx.statistic.stats(interval="1d", resample_interval="1d")
        return stats

    @staticmethod
    def _ga_evaluate(
        parameters: list[Any],
        *,
        cache: dict[Any, Any],
        evaluate_func: Callable[..., Any],
        maximize_func: Callable[..., Any],
        save_lock: Any,  # mp.Lock type
        eval_counter: Any,  # mp.Value type
        save_cache_func: Callable[..., Any] | None,
    ) -> tuple[float]:
        """Evaluate individual in genetic algorithm optimization."""
        tp: tuple = tuple(str(pv) for pv in dict(parameters).values())
        in_cache = False

        if tp in cache:
            result: pd.Series = cache[tp]
            in_cache = True
        else:
            setting: dict = dict(parameters)
            result: pd.Series = evaluate_func(**setting)
            cache[tp] = result

            if save_cache_func:
                save_interval = 66
                eval_counter.value += 1
                current_count = eval_counter.value
                if current_count % save_interval == 0:
                    with save_lock:
                        save_cache_func(cache)

        value: float = maximize_func(result)
        if value is None:
            if not in_cache:
                print(f"Warning: maximize_func returned value None for parameters: {dict(parameters)}")
            return (0.0,)

        return (value,)

    @staticmethod
    def _ga_task(**kwargs: Any) -> Any:
        """Run backtest task for genetic algorithm."""
        bt = BackTest._mp_backtests["ga"]
        return BackTest._run_bt(bt, **kwargs)

    @staticmethod
    def _ga_maximize(stats: Any, maximize_key: str) -> Any:
        """Extract maximize value from stats."""
        return stats[maximize_key]

    @staticmethod
    def _mp_task(backtest_uuid: float, batch_index: int) -> tuple[int, list[Any], list[Any]]:
        """Multiprocessing task for optimization."""
        (
            bt,
            param_batches,
            maximize_func,
            cache,
            eval_counter,
            save_lock,
            save_cache_func,
        ) = BackTest._mp_backtests[backtest_uuid]

        tmp_res = []
        tmp_stats = []

        for params in param_batches[batch_index]:
            key = tuple(str(pv) for pv in params.values())
            if key in cache:
                stats = cache[key]
            else:
                stats = BackTest._run_bt(bt, **params)
                cache[key] = stats

            if save_cache_func:
                save_interval = 66
                eval_counter.value += 1
                current_count = eval_counter.value
                if current_count % save_interval == 0:
                    with save_lock:
                        save_cache_func(cache)

            tmp_stats.append(stats)
            m = maximize_func(stats)
            tmp_res.append(m)

        return batch_index, tmp_res, tmp_stats

    _mp_backtests: dict[float, tuple["BackTest", list[Any], Callable[..., Any]]] = {}

    def portfolio_optimize(
        self,
        strategies: list[dict[str, Any]],
        maximize: str = "sharpe_ratio",
        population_size: int | None = None,
        ngen_size: int | None = None,
        max_workers: int | None = None,
        equal_weight: bool = True,
        combine_strategy_param_name: str | None = None,
        display_attributes: list[str] | None = None,
        cache_path: str | None = None,
        checkpoint_path: str | None = None,
        checkpoint_freq: int = 5,
        resume: bool = False,
        constraint: Callable[[dict[str, Any]], bool] | None = None,
        min_strategies: int = 2,
    ) -> tuple[list[dict[str, Any]], float]:
        """Optimize a portfolio of strategies using genetic algorithm.

        Args:
            strategies: List of strategy configurations
            maximize: The metric to maximize (default: "sharpe_ratio")
            population_size: Number of individuals in each generation
            ngen_size: Number of generations
            max_workers: Number of parallel workers (default: None, uses all available cores)
            equal_weight: If True, all selected strategies have equal weight (default: True)
            combine_strategy_param_name: If the trading strategy can combine multiple strategies,
                set this param to tell the backtest framework which param is used to pass the strategies list
            display_attributes: Additional attributes to display in results
            cache_path: Cache file path
            checkpoint_path: Checkpoint file path
            checkpoint_freq: Checkpoint frequency (in generations)
            resume: Indicate if resuming from checkpoint
            constraint: Function to validate strategy combinations
            min_strategies: Minimum number of strategies

        Returns:
            tuple[list[dict], float]: Best combination of strategies and its performance metric
        """
        if display_attributes is None:
            display_attributes = []

        try:
            import pickle

            from deap import algorithms, base, creator, tools
        except ImportError:
            raise ImportError("Need packages 'deap'. pip install deap")

        if not combine_strategy_param_name:
            valid_metrics = {"sharpe_ratio", "total_return", "sortino_ratio", "calmar_ratio"}
            if maximize not in valid_metrics:
                raise ValueError(f"maximize must be one of {valid_metrics}")
            if display_attributes:
                raise ValueError("display_attributes cannot be set if `combine_strategy_param_name` is not provided")

        _param_keys = [str(s) for s in strategies]
        _hist_cache = {}
        # Read cache from file if cache_path is provided
        cache_file_exists = cache_path and (
            os.path.exists(cache_path) or os.path.exists(os.path.splitext(cache_path)[0] + ".parquet")
        )

        if cache_file_exists:
            try:
                import pyarrow
            except ImportError:
                raise ImportError("Need package 'pyarrow' for parquet file. pip install pyarrow")

            parquet_path = os.path.splitext(cache_path)[0] + ".parquet"

            if os.path.exists(parquet_path):
                hist_df = pd.read_parquet(parquet_path)
            elif cache_path.endswith(".csv"):
                hist_df = pd.read_csv(cache_path, low_memory=False)
            elif cache_path.endswith((".xls", ".xlsx", ".xlsm", ".xlsb")):
                hist_df = pd.read_excel(cache_path)
            else:
                raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

            if len(hist_df) < 300:
                print(f"Found existing heatmap: {hist_df}")
            else:
                print("Found existing heatmap, but too many rows to display")

            columns = hist_df.columns
            _param_keys = [
                k for k in columns if k not in display_attributes and k not in [maximize, "number_strategies"]
            ]

            # Convert DataFrame to dictionary and update cache
            for index, row in hist_df.iterrows():
                key = tuple(int(row[k]) for k in _param_keys)
                _hist_cache[key] = {k: row[k] for k in hist_df.columns if k not in _param_keys}
            print(f"Loaded {len(_hist_cache)} cached results from {cache_path}")

        save_cache_func = partial(
            BackTest._save_cache,
            hist_cache=_hist_cache,
            param_keys=_param_keys,
            cache_path=cache_path,
            maximize_key=maximize,
            display_attributes=display_attributes,
        )

        # Create individual class used in genetic algorithm optimization
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        def generate_individual() -> list[int]:
            """Generate an individual with at least `min_strategies` active strategies."""
            individual = [0] * len(strategies)
            active_indices = np.random.choice(len(strategies), min(len(strategies), min_strategies) - 1, replace=False)
            for index in active_indices:
                individual[index] = 1

            # Randomly decide on the activation of the remaining strategies
            for i in range(len(strategies)):
                if individual[i] == 0:
                    individual[i] = np.random.choice([0, 1])
            return individual

        BackTest._mp_backtests["ga"] = self

        # Set up multiprocessing Pool and Manager
        set_mp_start_method()
        ctx = mp.get_context()

        with ctx.Manager() as manager, ctx.Pool(max_workers) as pool:
            # Create shared dict for result cache
            cache: dict[tuple[int, ...], dict[str, Any]] = manager.dict()

            # Add history cache
            if _hist_cache:
                for key, value in _hist_cache.items():
                    cache[key] = value

            # Set up toolbox (sequence doesn't matter, unless has dependency)
            toolbox = base.Toolbox()
            toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register(
                "evaluate",
                BackTest._evaluate_portfolio,
                cache=cache,
                strategies_list=strategies,
                maximize=maximize,
                equal_weight=equal_weight,
                combine_strategy_param_name=combine_strategy_param_name,
                constraint=constraint,
            )
            toolbox.register("mate", tools.cxTwoPoint)  # crossover operator
            toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # mutation operator
            toolbox.register("select", tools.selTournament, tournsize=3)  # selection operator
            toolbox.register("map", pool.map)  # map operator

            total_size: int = len(strategies)
            pop_size: int = population_size if population_size else int((len(strategies) * 2 + len(strategies) * 4) / 2)
            mu: int = int(pop_size * 0.8)
            cxpb: float = 0.7
            mutpb: float = 1 - cxpb
            ngen: int = ngen_size if ngen_size else int(len(strategies) * 1.5)
            lambda_: int = pop_size

            # Run ga optimization
            print("Start executing genetic algorithm optimization for portfolio")
            print(f"Using {max_workers} CPU cores")
            print(f"Total number of individuals in the population: {total_size}")
            print(f"Number of individuals in each generation(Population Size): {pop_size}")
            print(f"Number of individuals selected for the next generation(Elite Size): {mu}")
            print(f"Number of generations: {ngen}")
            print(f"Crossover probability: {cxpb:.0%}")
            print(f"Mutation probability: {mutpb:.0%}")

            halloffame = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

            def save_checkpoint(
                population: list[Any], generation: int, halloffame: Any, rndstate: Any, checkpoint_path: str
            ) -> None:
                """Save the current state of the evolution"""
                cp = dict(
                    population=population,
                    generation=generation,
                    halloffame=halloffame,
                    rndstate=rndstate,
                )
                with open(checkpoint_path, "wb") as cp_file:
                    pickle.dump(cp, cp_file)

            def load_checkpoint(checkpoint_path: str) -> dict[str, Any]:
                """Load a saved checkpoint"""
                with open(checkpoint_path, "rb") as cp_file:
                    cp = pickle.load(cp_file)
                return cp

            start_gen = 0
            population = toolbox.population(pop_size)

            # Load checkpoint if resuming
            if resume and checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_path}")
                cp = load_checkpoint(checkpoint_path)
                population = cp["population"]
                start_gen = cp["generation"]
                halloffame = cp["halloffame"]
                setstate(cp["rndstate"])
                print(f"Resuming from generation {start_gen}")
            else:
                halloffame = tools.HallOfFame(1)

            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

            try:
                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in population if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                if halloffame is not None:
                    halloffame.update(population)

                record = stats.compile(population) if stats is not None else {}

                # Begin the generational process
                for gen in range(start_gen, ngen):
                    start_time = time.time()

                    # Vary the population
                    offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

                    # Evaluate the individuals with an invalid fitness
                    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit

                    # Update the hall of fame with the generated individuals
                    if halloffame is not None:
                        halloffame.update(offspring)

                    # Select the next generation population
                    population[:] = toolbox.select(population + offspring, mu)

                    # Update the statistics with the new population
                    record = stats.compile(population) if stats is not None else {}

                    end_time = time.time()
                    print(
                        f"Generation {gen + 1}/{ngen}: "
                        f"Max={record['max']:.4f}, Avg={record['avg']:.4f}, Std={record['std']:.4f}, "
                        f"Time={end_time - start_time:.2f}s"
                    )
                    # Checkpoint if necessary
                    if checkpoint_path and (gen + 1) % checkpoint_freq == 0:
                        save_cache_func(cache)
                        print(f"Saved {len(cache)} cached results at generation {gen + 1}")
                        save_checkpoint(
                            population,
                            gen + 1,
                            halloffame,
                            getstate(),
                            checkpoint_path,
                        )
                        print(f"Saved checkpoint at generation {gen + 1}")

                    ### TODO: early stopping

            except KeyboardInterrupt:
                print("\nEvolution interrupted by user.")
                if checkpoint_path:
                    save_checkpoint(population, start_gen, halloffame, getstate(), checkpoint_path)
                    print("Saved checkpoint at interruption")
            finally:
                BackTest._mp_backtests.pop("ga", None)

            heatmap: defaultdict[tuple[int, ...], dict[str, Any]] = defaultdict(dict)
            for params, stats in dict(cache).items():
                # print(f"===params:{params} | stats:{stats}")
                heatmap[params][maximize] = stats[maximize]
                heatmap[params]["number_strategies"] = sum(params)
                for attr in display_attributes:
                    heatmap[params][attr] = stats[attr]

        heatmap = pd.DataFrame.from_dict(heatmap, orient="index")

        heatmap.index = pd.MultiIndex.from_tuples(heatmap.index, names=[str(i) for i in strategies])
        if cache_path:
            if cache_path.endswith(".csv"):
                heatmap.to_csv(cache_path, index=True)
            elif cache_path.endswith((".xls", ".xlsx", ".xlsm", ".xlsb")):
                heatmap.to_excel(cache_path, index=True)
            else:
                print(f"Warning: Unsupported file format for {cache_path}. Defaulting to CSV.")
                csv_path = cache_path + ".csv"
                heatmap.to_csv(csv_path, index=True)

        print(f"[portfolio_optimization] ===> halloffame: {halloffame} ")
        best_individual = halloffame[0]
        best_strategies = [s for i, s in zip(best_individual, strategies) if i]
        best_performance = best_individual.fitness.values[0]

        return best_strategies, best_performance
