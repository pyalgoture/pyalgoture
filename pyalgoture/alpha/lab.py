import json
import pickle
import shelve
from collections import defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import polars as pl

from ..utils.logger import get_logger
from ..utils.objects import AssetType, BarData, Exchange
from ..utils.util_dt import tz_manager
from .dataset import AlphaDataset
from .datasets.utils.utility import to_datetime
from .model import AlphaModel


class AlphaLab:
    """Alpha Research Laboratory"""

    def __init__(self, lab_path: str) -> None:
        """Constructor"""
        self.logger = get_logger()

        # Set data paths
        self.lab_path: Path = Path(lab_path)

        self.data_path: Path = self.lab_path.joinpath("data")
        self.component_path: Path = self.lab_path.joinpath("component")

        self.dataset_path: Path = self.lab_path.joinpath("dataset")
        self.model_path: Path = self.lab_path.joinpath("model")
        self.signal_path: Path = self.lab_path.joinpath("signal")
        self.report_path: Path = self.lab_path.joinpath("report")

        self.feed_info_path: Path = self.lab_path.joinpath("feed_info.json")

        # Create folders
        for path in [
            self.lab_path,
            self.data_path,
            self.component_path,
            self.dataset_path,
            self.model_path,
            self.signal_path,
            self.report_path,
        ]:
            if not path.exists():
                path.mkdir(parents=True)

    def save_bar_data(self, bars: dict[datetime, BarData], index: str) -> None:
        """Save bar data"""
        if not bars:
            return

        # Get file path
        bar: BarData = bars[next(iter(bars))]
        code: str = bar.code
        symbol: str = bar.symbol
        asset_type: str = bar.asset_type.value
        exchange: str = bar.exchange.value
        interval: str = bar.interval
        self.save_component_symbols(
            index,
            component={
                "code": code,
                "symbol": symbol,
                "asset_type": asset_type,
                "exchange": exchange,
                "interval": interval,
            },
        )

        key = f"{symbol.replace('/', '-')}_{exchange}_{interval}".lower()
        file_path = self.data_path.joinpath(f"{key}.parquet")

        # data: list = []
        # for bar in bars:
        #     bar_data: dict = {
        #         "datetime": bar.datetime.replace(tzinfo=None),
        #         "open": bar.open_price,
        #         "high": bar.high_price,
        #         "low": bar.low_price,
        #         "close": bar.close_price,
        #         "volume": bar.volume,
        #         "turnover": bar.turnover,
        #         "open_interest": bar.open_interest
        #     }
        #     data.append(bar_data)

        new_df: pl.DataFrame = pl.DataFrame([data.to_dict(return_obj=False) for data in bars.values()])

        # If file exists, read and merge
        if file_path.exists():
            old_df: pl.DataFrame = pl.read_parquet(file_path)

            new_df = pl.concat([old_df, new_df])

            new_df = new_df.unique(subset=["datetime"])

            new_df = new_df.sort("datetime")

        # Save to file
        new_df.write_parquet(file_path)

    def load_bar_data(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start: datetime | str,
        end: datetime | str,
    ) -> dict[datetime, BarData]:
        """Load bar data"""
        start = to_datetime(start, tz_manager.tz)
        end = to_datetime(end, tz_manager.tz)
        start_ts = int(start.timestamp() * 1000)
        end_ts = int(end.timestamp() * 1000)
        print(f"start: {start}, end: {end}; start_ts: {start_ts}, end_ts: {end_ts}")

        folder_path = self.data_path
        key = f"{symbol.replace('/', '-')}_{exchange}_{interval}".lower()

        # Check if file exists
        file_path: Path = folder_path.joinpath(f"{key}.parquet")
        if not file_path.exists():
            self.logger.error(f"File {file_path} does not exist")
            return {}

        # Open file
        df: pl.DataFrame = pl.read_parquet(file_path)
        # df = df.with_columns(pl.col("datetime").dt.convert_time_zone(tz_manager.tz))
        # df = df.with_columns(pl.col("datetime").dt.replace_time_zone(tz_manager.tz))

        # data_tz = df["datetime"].dt.tz()
        # print(f"[load_bar_data] data_tz: {data_tz}")
        # start = start.astimezone(data_tz)
        # end = end.astimezone(data_tz)

        ## TODO: temp fix
        df = df.with_columns(pl.col("datetime").dt.replace_time_zone(None))
        start = start.replace(tzinfo=None)
        end = end.replace(tzinfo=None)

        print(f"[load_bar_data] df: {df}; start: {start}; end: {end}")

        # Filter by date range - ensure timezone consistency
        df = df.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))
        # df = df.filter((pl.col("ts") >= start_ts) & (pl.col("ts") <= end_ts))

        # Convert to BarData objects
        bars: dict[datetime, BarData] = {}

        # for row in df.iter_rows(named=True):
        #     bar = BarData(
        #         symbol=symbol,
        #         exchange=exchange,
        #         datetime=row["datetime"],
        #         interval=interval,
        #         open_price=row["open"],
        #         high_price=row["high"],
        #         low_price=row["low"],
        #         close_price=row["close"],
        #         volume=row["volume"],
        #         turnover=row["turnover"],
        #         open_interest=row["open_interest"],
        #         gateway_name="DB"
        #     )
        #     bars[bar.datetime] = bar
        # df["datetime"] = df.index
        for v in df.to_dicts():
            v["exchange"] = Exchange(v["exchange"])
            v["asset_type"] = AssetType(v["asset_type"])
            bars[v["datetime"]] = BarData(**v)

        return bars

    def load_bar_df(
        self,
        symbols: list[str],
        exchange: str,
        interval: str,
        start: datetime | str,
        end: datetime | str,
        extended_days: int,
    ) -> pl.DataFrame | None:
        """Load bar data as DataFrame
        extended_days: extend the data range by this number of days
        """
        if not symbols:
            return None

        start = to_datetime(start) - timedelta(days=extended_days)
        end = to_datetime(end) + timedelta(days=extended_days // 10)
        start_ts = int(start.timestamp() * 1000)
        end_ts = int(end.timestamp() * 1000)

        # Get folder path
        folder_path = self.data_path

        # Read data for each symbol
        dfs: list = []
        # print(f"start: {start}, end: {end}; start_ts: {start_ts}, end_ts: {end_ts}")
        for symbol in symbols:
            key = f"{symbol.replace('/', '-')}_{exchange}_{interval}".lower()

            # Check if file exists
            file_path: Path = folder_path.joinpath(f"{key}.parquet")
            # print(f"file_path: {file_path}")
            if not file_path.exists():
                self.logger.error(f"File {file_path} does not exist")
                continue

            # Open file
            df: pl.DataFrame = pl.read_parquet(file_path)
            ## TODO: temp fix
            df = df.with_columns(pl.col("datetime").dt.replace_time_zone(None))
            start = start.replace(tzinfo=None)
            end = end.replace(tzinfo=None)

            # print(f"[load_bar_df] df1: {df}")

            # Filter by date range
            # df = df.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))
            df = df.filter((pl.col("ts") >= start_ts) & (pl.col("ts") <= end_ts))

            # Specify data types
            df = df.with_columns(
                pl.col("open").cast(pl.Float32),
                pl.col("high").cast(pl.Float32),
                pl.col("low").cast(pl.Float32),
                pl.col("close").cast(pl.Float32),
                pl.col("volume").cast(pl.Float32),
                pl.col("turnover").cast(pl.Float32),
                pl.col("open_interest").cast(pl.Float32),
                (pl.col("turnover") / pl.col("volume")).cast(pl.Float32).alias("vwap"),
            )
            # print(f"[load_bar_df] df2: {df}")

            # Check for empty data
            if df.is_empty():
                continue

            # Normalize prices
            close_0: float = df.select(pl.col("close")).item(0, 0)

            df = df.with_columns(
                (pl.col("open") / close_0).alias("open"),
                (pl.col("high") / close_0).alias("high"),
                (pl.col("low") / close_0).alias("low"),
                (pl.col("close") / close_0).alias("close"),
            )
            # print(f"[load_bar_df] df3: {df}")

            # Convert zeros to NaN for suspended trading days
            numeric_columns: list = df.columns[6:]  # Extract numeric columns
            # print(f"[load_bar_df] numeric_columns: {numeric_columns}")
            mask: pl.Series = df[numeric_columns].sum_horizontal() == 0  # Sum by row, if 0 then suspended

            df = df.with_columns(  # Convert suspended day values to NaN
                [pl.when(mask).then(float("nan")).otherwise(pl.col(col)).alias(col) for col in numeric_columns]
            )
            # print(f"[load_bar_df] df4: {df} {df.columns}")

            # Add symbol column
            # df = df.with_columns(pl.lit(symbol).alias("aio_symbol"))
            # Add aio_symbol column with format "{symbol}|{exchange}"
            df = df.with_columns(pl.lit(f"{symbol}|{exchange.upper()}").alias("aio_symbol"))

            # Keep only the specified columns in the required order
            df = df.select(
                [
                    "aio_symbol",
                    "datetime",
                    "volume",
                    "turnover",
                    "open_interest",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vwap",
                ]
            )

            # Cache in list
            dfs.append(df)

        # Concatenate results
        result_df: pl.DataFrame = pl.concat(dfs)
        return result_df

    def save_component_symbols(
        self, index: str, components: list[dict] = [], component: dict = {}, overwrite: bool = False
    ) -> None:
        """Save index component data
        component: dict = {'symbol':symbol,'exchange':exchange,'interval':interval}
        """
        file_path: Path = self.component_path.joinpath(f"{index}")

        with shelve.open(str(file_path)) as db:
            if overwrite:
                db.clear()
            if components:
                db.update({d["symbol"]: d for d in components})
            if component:
                db.update({component["symbol"]: component})

    @lru_cache  # noqa
    def load_component_symbols(
        self,
        index: str,
    ) -> list[dict]:
        """Collect index component symbols"""
        file_path: Path = self.component_path.joinpath(f"{index}")
        components: list = []
        with shelve.open(str(file_path)) as db:
            components = list(db.values())

        return components

    def add_feed_info(
        self,
        aio_symbol: str,
        taker_commission_rate: float,
        maker_commission_rate: float,
        step_size: float,
        tick_size: float,
    ) -> None:
        """Add feed_info information"""
        feed_infos: dict = {}

        if self.feed_info_path.exists():
            with open(self.feed_info_path, encoding="UTF-8") as f:
                feed_infos = json.load(f)

        feed_infos[aio_symbol] = {
            "taker_commission_rate": taker_commission_rate,  # Commission rate for long
            "maker_commission_rate": maker_commission_rate,  # Commission rate for short
            "step_size": step_size,  # step_size
            "tick_size": tick_size,  # Price tick
        }

        with open(self.feed_info_path, mode="w+", encoding="UTF-8") as f:
            json.dump(feed_infos, f, indent=4, ensure_ascii=False)

    def load_feed_info(self) -> dict:
        """Load feed_info settings"""
        feed_infos: dict = {}

        if self.feed_info_path.exists():
            with open(self.feed_info_path, encoding="UTF-8") as f:
                feed_infos = json.load(f)

        return feed_infos

    def save_dataset(self, name: str, dataset: AlphaDataset) -> None:
        """Save dataset"""
        file_path: Path = self.dataset_path.joinpath(f"{name}.pkl")

        with open(file_path, mode="wb") as f:
            pickle.dump(dataset, f)

    def load_dataset(self, name: str) -> AlphaDataset | None:
        """Load dataset"""
        file_path: Path = self.dataset_path.joinpath(f"{name}.pkl")
        if not file_path.exists():
            self.logger.error(f"Dataset file {name} does not exist")
            return None

        with open(file_path, mode="rb") as f:
            dataset: AlphaDataset = pickle.load(f)
            return dataset

    def remove_dataset(self, name: str) -> bool:
        """Remove dataset"""
        file_path: Path = self.dataset_path.joinpath(f"{name}.pkl")
        if not file_path.exists():
            self.logger.error(f"Dataset file {name} does not exist")
            return False

        file_path.unlink()
        return True

    def list_all_datasets(self) -> list[str]:
        """List all datasets"""
        return [file.stem for file in self.dataset_path.glob("*.pkl")]

    def save_model(self, name: str, model: AlphaModel) -> None:
        """Save model"""
        file_path: Path = self.model_path.joinpath(f"{name}.pkl")

        with open(file_path, mode="wb") as f:
            pickle.dump(model, f)

    def load_model(self, name: str) -> AlphaModel | None:
        """Load model"""
        file_path: Path = self.model_path.joinpath(f"{name}.pkl")
        if not file_path.exists():
            self.logger.error(f"Model file {name} does not exist")
            return None

        with open(file_path, mode="rb") as f:
            model: AlphaModel = pickle.load(f)
            return model

    def remove_model(self, name: str) -> bool:
        """Remove model"""
        file_path: Path = self.model_path.joinpath(f"{name}.pkl")
        if not file_path.exists():
            self.logger.error(f"Model file {name} does not exist")
            return False

        file_path.unlink()
        return True

    def list_all_models(self) -> list[str]:
        """List all models"""
        return [file.stem for file in self.model_path.glob("*.pkl")]

    def save_signal(self, name: str, signal: pl.DataFrame) -> None:
        """Save signal"""
        file_path: Path = self.signal_path.joinpath(f"{name}.parquet")

        signal.write_parquet(file_path)

    def load_signal(self, name: str) -> pl.DataFrame | None:
        """Load signal"""
        file_path: Path = self.signal_path.joinpath(f"{name}.parquet")
        if not file_path.exists():
            self.logger.error(f"Signal file {name} does not exist")
            return None

        return pl.read_parquet(file_path)

    def remove_signal(self, name: str) -> bool:
        """Remove signal"""
        file_path: Path = self.signal_path.joinpath(f"{name}.parquet")
        if not file_path.exists():
            self.logger.error(f"Signal file {name} does not exist")
            return False

        file_path.unlink()
        return True

    def list_all_signals(self) -> list[str]:
        """List all signals"""
        return [file.stem for file in self.model_path.glob("*.parquet")]
