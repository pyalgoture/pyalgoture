"""Utility functions for I/O operations, configuration management, and strategy handling."""

import csv
import json
import os
import traceback
from ast import literal_eval
from collections import defaultdict
from collections.abc import Callable, Generator, Iterator
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from re import findall
from typing import Any

import pandas as pd

# ============================= CSV Operations ============================


def to_csv(data: dict[str, dict[str, Any]], filepath: str | Path) -> bool:
    """Write dictionary data to CSV file with datetime column first.

    Args:
        data: Dictionary where values are dictionaries containing row data
        filepath: Path to output CSV file

    Returns:
        True if successful, False if data is empty
    """
    if not data:
        return False

    filepath = Path(filepath)
    first_row = next(iter(data.values()))
    fieldnames = list(first_row.keys())

    # Move datetime to first column if present
    if "datetime" in fieldnames:
        fieldnames.remove("datetime")
        fieldnames.insert(0, "datetime")

    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data.values())

    return True


def read_csv(filepath: str | Path, **col_conversions: Callable[[str], Any]) -> Generator[dict[str, Any], None, None]:
    """Read CSV file and yield rows with optional column type conversions.

    Args:
        filepath: Path to CSV file
        **col_conversions: Column name to conversion function mapping

    Yields:
        Dictionary representing each row with converted values
    """

    def parse_value(key: str, value: str) -> Any:
        """Parse string value with optional type conversion."""
        if key in col_conversions:
            return col_conversions[key](value)
        try:
            return literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    filepath = Path(filepath)
    with open(filepath, encoding="utf-8", buffering=4096) as f:
        reader = csv.DictReader(f, quoting=csv.QUOTE_NONE)
        for row in reader:
            yield {k: parse_value(k, v) for k, v in row.items()}


# ============================= Strategy Operations ============================


def retrieve_strategy_name(path: str | Path, name: str | None = None) -> str:
    """Extract strategy class name from Python file.

    Args:
        path: Path to Python file containing strategy class
        name: Specific strategy name to validate (optional)

    Returns:
        Strategy class name if found, empty string otherwise
    """
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
            matches = findall(r"class\s+(.*?)\s*\(\s*Strategy\s*\)\s*:", content)

            if not matches:
                return ""

            if name is not None:
                return name if name in matches else ""
            return str(matches[0])
    except (OSError, FileNotFoundError) as e:
        print(f"Error reading strategy file {path}: {e}")
        return ""


def retrieve_strategy_class(path: str | Path, name: str | None = None) -> type | None:
    """Load strategy class from Python file.

    Args:
        path: Path to Python file containing strategy class
        name: Specific strategy name to load (optional)

    Returns:
        Strategy class if found, None otherwise
    """
    strategy_name = retrieve_strategy_name(path, name)
    if not strategy_name:
        return None

    try:
        path = Path(path)
        spec = spec_from_file_location(strategy_name, path)
        if spec is None or spec.loader is None:
            print(f"Failed to create module spec for {path}")
            return None

        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        obj = getattr(module, strategy_name)
        if isinstance(obj, type):
            return obj
        else:
            return None
    except Exception as e:
        print(f"Error loading strategy class from {path}: {e}")
        return None


# def create_report_folder(base: str | Path | None = None, prefix: str = "", suffix: str = "") -> str:
#     """Create report folder with optional prefix/suffix.

#     Args:
#         base: Base path for folder creation. If None, uses caller's filename
#         prefix: Prefix to add to folder name
#         suffix: Suffix to add to folder name

#     Returns:
#         Absolute path of created folder
#     """
#     if base is None:
#         # Get caller's filename from stack trace
#         filename, *_ = traceback.extract_stack()[-2]
#         base = Path(filename).stem
#         if "ipykernel" in str(base):
#             base = "ipython"

#     folder_name = f"{prefix}{base}{suffix}"
#     folder_path = Path(folder_name)
#     data_path = folder_path / ".data"

#     # Create main folder
#     if not folder_path.exists():
#         folder_path.mkdir(parents=True, exist_ok=True)
#         print(f"Created folder: {folder_path}")

#     # Create data subfolder
#     data_path.mkdir(exist_ok=True)

#     return str(folder_path.absolute())


def create_report_folder(base: str = "", prefix: str = "", suffix: str = "") -> str:
    """Create folder with name same as your strategy script

    Args:
        base (str, optional): The base path. Defaults to None.

    Returns:
        srt: The abosulte path of the folder
    """
    # if not base:
    #     caller_frame = inspect.stack()[1]
    #     caller_filename = caller_frame.filename
    #     # print(caller_frame)
    #     # print(caller_filename)
    #     base = os.path.splitext(caller_filename)[0]
    #     # print(base)
    #     if 'ipython' in base:
    #         base = 'ipython'
    if not base:
        # print(len(traceback.extract_stack()),'???',type(traceback.extract_stack()[-2]),'??',traceback.extract_stack()[-2])
        (filename, line_number, function_name, text) = traceback.extract_stack()[-2]
        # print(filename, '\n',line_number, '\n',function_name, type(function_name), '\n',text)
        base = os.path.splitext(filename)[0]
        # print(base)
        if "ipykernel" in base:
            base = "ipython"
    base = f"{prefix}{base}{suffix}"
    if not os.path.exists(base):
        os.makedirs(base, exist_ok=True)
        print(f"{base} is successfully created.")

    return os.path.abspath(base)


# ============================= Configuration Management ============================

MODULE_NAME = "pyalgoture"


def read_conf(config_file: str | Path) -> dict[str, Any]:
    """Read YAML configuration file.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        ImportError: If pyyaml is not installed
        Exception: If file cannot be read or parsed
    """
    try:
        import yaml
        from yaml.representer import Representer

        yaml.add_representer(defaultdict, Representer.represent_dict)
    except ImportError as e:
        raise ImportError("Package 'pyyaml' required. Install with: pip install pyyaml") from e

    config_file = Path(config_file)
    print(f"Reading config file: '{config_file}'")

    try:
        with open(config_file, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Failed to read config file {config_file}: {e}")
        raise


def _has_path(config: dict[str, Any], items: str | list[str]) -> bool:
    """Check if configuration path exists."""
    current_config = config
    path_items = _parse_config_path(items)

    for key in path_items:
        if not isinstance(current_config, dict) or key not in current_config:
            return False
        current_config = current_config[key]
    return True


def _get_path(config: dict[str, Any], items: str | list[str], default: Any = None, warn: bool = False) -> Any:
    """Get value from configuration path."""
    current_config = config
    path_items = _parse_config_path(items)

    for key in path_items:
        if not isinstance(current_config, dict) or key not in current_config:
            path_str = "/" + "/".join(path_items)
            message = f"{path_str} not specified in config, using default: {default}"
            if warn:
                print(f"WARNING: {message}")
            else:
                print(message)
            return default
        current_config = current_config[key]
    return current_config


def _parse_config_path(items: str | list[str]) -> list[str]:
    """Parse configuration path into list of keys."""
    if isinstance(items, str):
        if items.startswith("/"):
            return items.split("/")[1:]
        return [items]
    return items


def has(config: dict[str, Any], item: str) -> bool:
    """Check if configuration contains specified item.

    Args:
        config: Configuration dictionary
        item: Configuration item path (e.g., "broker/class" or "/broker/class")

    Returns:
        True if item exists, False otherwise
    """
    return _has_path(config, item)


def get(config: dict[str, Any], item: str = "", default: Any = None, warn: bool = False) -> Any:
    """Get configuration value with optional default.

    Args:
        config: Configuration dictionary
        item: Configuration item path. Empty string returns entire config
        default: Default value if item not found
        warn: Whether to show warning for missing items

    Returns:
        Configuration value or default
    """
    if not item:
        return config

    if item.startswith("/"):
        return _get_path(config, item, default, warn)

    if item in config:
        return config[item]

    message = f"'{item}' not specified in config, using default: {default}"
    if warn:
        print(f"WARNING: {message}")
    else:
        print(message)
    return default


def dump_conf(config: dict[str, Any], path: str | Path) -> None:
    """Write configuration dictionary to YAML file.

    Args:
        config: Configuration dictionary to write
        path: Output file path

    Raises:
        ImportError: If pyyaml is not installed
    """
    try:
        import yaml
        from yaml.representer import Representer

        yaml.add_representer(defaultdict, Representer.represent_dict)
    except ImportError as e:
        raise ImportError("Package 'pyyaml' required. Install with: pip install pyyaml") from e

    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)


# ============================= Component Factory Functions ============================


def get_broker(config: dict[str, Any]) -> Any:
    """Create broker instance from configuration.

    Args:
        config: Configuration dictionary containing broker settings

    Returns:
        Configured broker instance

    Raises:
        Exception: If broker configuration is invalid
    """
    try:
        class_name = get(config, "/broker/class")
        broker_config = get(config, "broker", {}).copy()
        is_debug = broker_config.pop("debug", False)
        connections = broker_config.pop("connections", [])

        module = import_module(MODULE_NAME)
        broker_class = getattr(module, class_name)
        broker = broker_class(**broker_config)

        # Add connections if specified
        for connection in connections:
            broker.add_connection(**connection)

        if is_debug:
            broker._print_user_trade_data = True

        return broker

    except Exception as e:
        print(f"Failed to create broker: {e}")
        raise


def get_analyzer(config: dict[str, Any]) -> Any:
    """Create analyzer instance from configuration.

    Args:
        config: Configuration dictionary containing analyzer settings

    Returns:
        Configured analyzer instance

    Raises:
        Exception: If analyzer configuration is invalid
    """
    try:
        analyzer_config = get(config, "analyzer", {})
        module = import_module(MODULE_NAME)
        analyzer_class = getattr(module, "Statistic")
        return analyzer_class(**analyzer_config)

    except Exception as e:
        print(f"Failed to create analyzer: {e}")
        raise


def get_datafeeds(config: dict[str, Any]) -> tuple[list[Any], Any]:
    """Create datafeed and benchmark instances from configuration.

    Args:
        config: Configuration dictionary containing datafeed settings

    Returns:
        Tuple of (datafeeds list, benchmark instance)

    Raises:
        ImportError: If required modules cannot be imported
        AttributeError: If required classes cannot be found
    """
    try:
        module = import_module(MODULE_NAME)

        # Create benchmark
        benchmark_class_name = get(config, "/benchmark/class")
        benchmark_config = get(config, "benchmark", {}).copy()
        benchmark_config["bypass"] = True

        benchmark_class = getattr(module, benchmark_class_name)
        benchmark = benchmark_class(**benchmark_config)

        # Create datafeeds
        feeds = []
        datafeeds_config = get(config, "datafeeds", [])

        for feed_config in datafeeds_config:
            feed_config = feed_config.copy()
            class_name = feed_config.pop("class")
            feed_config["bypass"] = True

            feed_class = getattr(module, class_name)
            feed = feed_class(**feed_config)
            feeds.append(feed)

        return feeds, benchmark

    except ImportError as e:
        print(f"Module import failed: {e}")
        raise
    except AttributeError as e:
        print(f"Class not found: {e}")
        raise


def get_engine_config(config: dict[str, Any], strategy_folder_path: str | Path | None = None) -> dict[str, Any]:
    """Create engine configuration from config file.

    Args:
        config: Configuration dictionary
        strategy_folder_path: Optional base path for strategy files

    Returns:
        Engine configuration dictionary

    Raises:
        Exception: If engine configuration is invalid
    """
    try:
        engine_config: dict = get(config, "strategy", {}).copy()

        # Extract strategy-specific settings
        trade_history_path = engine_config.pop("trade_history_path", None)
        strategy_params = engine_config.pop("strategy_parameters", {})
        strategy_class_path = engine_config.pop("strategy_class_path")
        strategy_class_name = engine_config.pop("strategy_class_name", None)

        # Resolve strategy path
        if strategy_folder_path:
            strategy_class_path = Path(strategy_folder_path) / strategy_class_path

        # Load strategy class
        strategy_class = retrieve_strategy_class(strategy_class_path, strategy_class_name)
        if strategy_class is None:
            raise ValueError(f"Failed to load strategy class from {strategy_class_path}")

        # Create components
        broker = get_broker(config)
        analyzer = get_analyzer(config)
        feeds, benchmark = get_datafeeds(config)

        # Build engine configuration
        engine_config.update(
            {
                "StrategyClass": strategy_class,
                "broker": broker,
                "statistic": analyzer,
                "feeds": feeds,
                "benchmark": benchmark,
                **strategy_params,
            }
        )

        # Load trade history if specified
        if trade_history_path:
            trade_history_path = Path(trade_history_path)
            if trade_history_path.suffix == ".json":
                with open(trade_history_path, encoding="utf-8") as f:
                    engine_config["trade_history"] = json.load(f)
            elif trade_history_path.suffix == ".csv":
                trades = pd.read_csv(trade_history_path)
                engine_config["trade_history"] = trades.to_dict("records")

        return engine_config

    except Exception as e:
        print(f"Failed to create engine config: {e}")
        raise
