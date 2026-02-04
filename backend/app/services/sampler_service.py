import random
from typing import Optional, Iterator
import numpy as np
import pandas as pd
from pathlib import Path

from app.models.schemas import SamplingMethod, SamplingConfig
from app.services.parser_service import ExcelParser
from app.config import EXCEL_CHUNK_SIZE


class ReservoirSampler:
    def __init__(self, sample_size: int, seed: Optional[int] = None):
        self.sample_size = sample_size
        self.reservoir: list[tuple[int, list]] = []
        self.count = 0
        self.rng = random.Random(seed)

    def add(self, index: int, row: list):
        self.count += 1
        if len(self.reservoir) < self.sample_size:
            self.reservoir.append((index, row))
        else:
            j = self.rng.randint(0, self.count - 1)
            if j < self.sample_size:
                self.reservoir[j] = (index, row)

    def get_sample(self) -> list[tuple[int, list]]:
        return sorted(self.reservoir, key=lambda x: x[0])


class StratifiedReservoirSampler:
    def __init__(self, sample_size: int, seed: Optional[int] = None):
        self.sample_size = sample_size
        self.seed = seed
        self.strata_counts: dict[str, int] = {}
        self.strata_reservoirs: dict[str, ReservoirSampler] = {}
        self.total_count = 0

    def count_stratum(self, stratum: str):
        self.strata_counts[stratum] = self.strata_counts.get(stratum, 0) + 1
        self.total_count += 1

    def initialize_reservoirs(self):
        for stratum, count in self.strata_counts.items():
            proportion = count / self.total_count
            stratum_sample_size = max(1, int(self.sample_size * proportion))
            self.strata_reservoirs[stratum] = ReservoirSampler(stratum_sample_size, self.seed)

    def add(self, stratum: str, index: int, row: list):
        if stratum in self.strata_reservoirs:
            self.strata_reservoirs[stratum].add(index, row)

    def get_sample(self) -> list[tuple[int, list]]:
        result = []
        for reservoir in self.strata_reservoirs.values():
            result.extend(reservoir.get_sample())
        return sorted(result, key=lambda x: x[0])


class SamplerService:
    def __init__(self):
        self._sample_store: dict[str, pd.DataFrame] = {}
        self._sample_metadata: dict[str, dict] = {}

    def sample_random(
        self,
        parser: ExcelParser,
        sample_size: int,
        seed: Optional[int] = None,
        with_replacement: bool = False,
        sheet_name: Optional[str] = None,
    ) -> pd.DataFrame:
        sampler = ReservoirSampler(sample_size, seed)
        columns = None

        for chunk in parser.iter_rows(sheet_name):
            if columns is None:
                columns = list(chunk.columns)
            for idx, row in chunk.iterrows():
                sampler.add(idx, row.tolist())

        sample_data = sampler.get_sample()
        if not sample_data:
            return pd.DataFrame(columns=columns or [])

        indices, rows = zip(*sample_data)
        return pd.DataFrame(rows, columns=columns)

    def sample_stratified(
        self,
        parser: ExcelParser,
        sample_size: int,
        strata_column: str,
        seed: Optional[int] = None,
        sheet_name: Optional[str] = None,
    ) -> pd.DataFrame:
        sampler = StratifiedReservoirSampler(sample_size, seed)
        columns = None
        strata_col_idx = None

        for chunk in parser.iter_rows(sheet_name):
            if columns is None:
                columns = list(chunk.columns)
                if strata_column not in columns:
                    raise ValueError(f"Strata column '{strata_column}' not found in data")
                strata_col_idx = columns.index(strata_column)

            for _, row in chunk.iterrows():
                stratum = str(row.iloc[strata_col_idx])
                sampler.count_stratum(stratum)

        sampler.initialize_reservoirs()

        row_idx = 0
        for chunk in parser.iter_rows(sheet_name):
            for _, row in chunk.iterrows():
                stratum = str(row.iloc[strata_col_idx])
                sampler.add(stratum, row_idx, row.tolist())
                row_idx += 1

        sample_data = sampler.get_sample()
        if not sample_data:
            return pd.DataFrame(columns=columns or [])

        indices, rows = zip(*sample_data)
        return pd.DataFrame(rows, columns=columns)

    def sample_systematic(
        self,
        parser: ExcelParser,
        sample_size: int,
        seed: Optional[int] = None,
        sheet_name: Optional[str] = None,
    ) -> pd.DataFrame:
        total_rows = parser.get_row_count(sheet_name)
        if total_rows == 0:
            return pd.DataFrame()

        interval = max(1, total_rows // sample_size)
        rng = random.Random(seed)
        start = rng.randint(0, interval - 1)

        selected_indices = set(range(start, total_rows, interval))
        if len(selected_indices) > sample_size:
            selected_indices = set(list(selected_indices)[:sample_size])

        columns = None
        sampled_rows = []
        row_idx = 0

        for chunk in parser.iter_rows(sheet_name):
            if columns is None:
                columns = list(chunk.columns)

            for _, row in chunk.iterrows():
                if row_idx in selected_indices:
                    sampled_rows.append(row.tolist())
                row_idx += 1

        return pd.DataFrame(sampled_rows, columns=columns)

    def sample_cluster(
        self,
        parser: ExcelParser,
        sample_size: int,
        cluster_column: str,
        seed: Optional[int] = None,
        sheet_name: Optional[str] = None,
    ) -> pd.DataFrame:
        columns = None
        cluster_col_idx = None
        clusters: dict[str, list[list]] = {}

        for chunk in parser.iter_rows(sheet_name):
            if columns is None:
                columns = list(chunk.columns)
                if cluster_column not in columns:
                    raise ValueError(f"Cluster column '{cluster_column}' not found in data")
                cluster_col_idx = columns.index(cluster_column)

            for _, row in chunk.iterrows():
                cluster = str(row.iloc[cluster_col_idx])
                if cluster not in clusters:
                    clusters[cluster] = []
                clusters[cluster].append(row.tolist())

        rng = random.Random(seed)
        cluster_names = list(clusters.keys())
        rng.shuffle(cluster_names)

        sampled_rows = []
        for cluster_name in cluster_names:
            sampled_rows.extend(clusters[cluster_name])
            if len(sampled_rows) >= sample_size:
                break

        return pd.DataFrame(sampled_rows[:sample_size], columns=columns)

    def sample_weighted(
        self,
        parser: ExcelParser,
        sample_size: int,
        weight_column: str,
        seed: Optional[int] = None,
        with_replacement: bool = False,
        sheet_name: Optional[str] = None,
    ) -> pd.DataFrame:
        columns = None
        weight_col_idx = None
        all_rows: list[list] = []
        weights: list[float] = []

        for chunk in parser.iter_rows(sheet_name):
            if columns is None:
                columns = list(chunk.columns)
                if weight_column not in columns:
                    raise ValueError(f"Weight column '{weight_column}' not found in data")
                weight_col_idx = columns.index(weight_column)

            for _, row in chunk.iterrows():
                all_rows.append(row.tolist())
                try:
                    weights.append(float(row.iloc[weight_col_idx]))
                except (ValueError, TypeError):
                    weights.append(0.0)

        if not all_rows:
            return pd.DataFrame(columns=columns or [])

        weights = np.array(weights)
        weights = np.maximum(weights, 0)
        total_weight = weights.sum()
        if total_weight == 0:
            weights = np.ones(len(weights))
            total_weight = len(weights)
        probabilities = weights / total_weight

        rng = np.random.default_rng(seed)
        actual_sample_size = min(sample_size, len(all_rows))

        if with_replacement:
            indices = rng.choice(len(all_rows), size=actual_sample_size, replace=True, p=probabilities)
        else:
            indices = rng.choice(len(all_rows), size=actual_sample_size, replace=False, p=probabilities)

        sampled_rows = [all_rows[i] for i in sorted(indices)]
        return pd.DataFrame(sampled_rows, columns=columns)

    def sample(self, file_path: Path, config: SamplingConfig) -> tuple[pd.DataFrame, dict]:
        parser = ExcelParser(file_path)
        file_info = parser.get_file_info(config.sheet_name)

        sample_size = config.sample_size
        if config.sample_percentage is not None:
            sample_size = max(1, int(file_info["row_count"] * config.sample_percentage / 100))

        sample_size = min(sample_size, file_info["row_count"])

        if config.method == SamplingMethod.RANDOM:
            df = self.sample_random(
                parser, sample_size, config.random_seed, config.with_replacement, config.sheet_name
            )
        elif config.method == SamplingMethod.STRATIFIED:
            if not config.strata_column:
                raise ValueError("Strata column required for stratified sampling")
            df = self.sample_stratified(
                parser, sample_size, config.strata_column, config.random_seed, config.sheet_name
            )
        elif config.method == SamplingMethod.SYSTEMATIC:
            df = self.sample_systematic(parser, sample_size, config.random_seed, config.sheet_name)
        elif config.method == SamplingMethod.CLUSTER:
            if not config.cluster_column:
                raise ValueError("Cluster column required for cluster sampling")
            df = self.sample_cluster(
                parser, sample_size, config.cluster_column, config.random_seed, config.sheet_name
            )
        elif config.method == SamplingMethod.WEIGHTED:
            if not config.weight_column:
                raise ValueError("Weight column required for weighted sampling")
            df = self.sample_weighted(
                parser, sample_size, config.weight_column, config.random_seed, config.with_replacement, config.sheet_name
            )
        else:
            raise ValueError(f"Unknown sampling method: {config.method}")

        statistics = {
            "original_rows": file_info["row_count"],
            "sampled_rows": len(df),
            "sampling_rate": len(df) / file_info["row_count"] if file_info["row_count"] > 0 else 0,
            "method": config.method.value,
            "columns": file_info["columns"],
        }

        return df, statistics

    def store_sample(self, sample_id: str, df: pd.DataFrame, metadata: dict):
        self._sample_store[sample_id] = df
        self._sample_metadata[sample_id] = metadata

    def get_sample(self, sample_id: str) -> Optional[pd.DataFrame]:
        return self._sample_store.get(sample_id)

    def get_sample_metadata(self, sample_id: str) -> Optional[dict]:
        return self._sample_metadata.get(sample_id)

    def iter_sample_chunks(self, sample_id: str, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
        df = self._sample_store.get(sample_id)
        if df is None:
            return

        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size]


sampler_service = SamplerService()
