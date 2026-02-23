"""
Tests for SamplerService and sampling algorithms.

These tests verify the sampling algorithms (random, stratified, systematic,
cluster, weighted) and the service class that manages sampling operations.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.sampler_service import (
    ReservoirSampler,
    StratifiedReservoirSampler,
    SamplerService,
)
from app.models.schemas import SamplingMethod, SamplingConfig


class TestReservoirSampler:
    """Tests for ReservoirSampler class."""
    
    def test_sample_size_less_than_population(self):
        """Test sampling when sample size is less than population."""
        sampler = ReservoirSampler(sample_size=3, seed=42)
        
        for i in range(10):
            sampler.add(i, [f"row_{i}"])
        
        sample = sampler.get_sample()
        
        assert len(sample) == 3
        assert sampler.count == 10
    
    def test_sample_size_equals_population(self):
        """Test sampling when sample size equals population."""
        sampler = ReservoirSampler(sample_size=5, seed=42)
        
        for i in range(5):
            sampler.add(i, [f"row_{i}"])
        
        sample = sampler.get_sample()
        
        assert len(sample) == 5
    
    def test_sample_size_greater_than_population(self):
        """Test sampling when sample size is greater than population."""
        sampler = ReservoirSampler(sample_size=10, seed=42)
        
        for i in range(3):
            sampler.add(i, [f"row_{i}"])
        
        sample = sampler.get_sample()
        
        assert len(sample) == 3
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        sampler1 = ReservoirSampler(sample_size=3, seed=42)
        sampler2 = ReservoirSampler(sample_size=3, seed=42)
        
        for i in range(10):
            sampler1.add(i, [f"row_{i}"])
            sampler2.add(i, [f"row_{i}"])
        
        sample1 = sampler1.get_sample()
        sample2 = sampler2.get_sample()
        
        assert sample1 == sample2
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        sampler1 = ReservoirSampler(sample_size=3, seed=42)
        sampler2 = ReservoirSampler(sample_size=3, seed=123)
        
        for i in range(100):
            sampler1.add(i, [f"row_{i}"])
            sampler2.add(i, [f"row_{i}"])
        
        sample1 = sampler1.get_sample()
        sample2 = sampler2.get_sample()
        
        assert sample1 != sample2
    
    def test_sample_sorted_by_index(self):
        """Test that sample is sorted by original index."""
        sampler = ReservoirSampler(sample_size=5, seed=42)
        
        for i in range(5):
            sampler.add(i, [f"row_{i}"])
        
        sample = sampler.get_sample()
        indices = [item[0] for item in sample]
        
        assert indices == sorted(indices)


class TestStratifiedReservoirSampler:
    """Tests for StratifiedReservoirSampler class."""
    
    def test_proportional_sampling(self):
        """Test that sampling is proportional to stratum sizes."""
        sampler = StratifiedReservoirSampler(sample_size=10, seed=42)
        
        for i in range(60):
            sampler.count_stratum('A')
        for i in range(40):
            sampler.count_stratum('B')
        
        sampler.initialize_reservoirs()
        
        for i in range(60):
            sampler.add('A', i, [f"A_{i}"])
        for i in range(40):
            sampler.add('B', 60 + i, [f"B_{i}"])
        
        sample = sampler.get_sample()
        
        a_count = sum(1 for _, row in sample if row[0].startswith('A_'))
        b_count = sum(1 for _, row in sample if row[0].startswith('B_'))
        
        assert a_count == 6
        assert b_count == 4
    
    def test_single_stratum(self):
        """Test sampling with a single stratum."""
        sampler = StratifiedReservoirSampler(sample_size=5, seed=42)
        
        for i in range(10):
            sampler.count_stratum('A')
        
        sampler.initialize_reservoirs()
        
        for i in range(10):
            sampler.add('A', i, [f"row_{i}"])
        
        sample = sampler.get_sample()
        
        assert len(sample) == 5
    
    def test_many_strata(self):
        """Test sampling with many strata."""
        sampler = StratifiedReservoirSampler(sample_size=10, seed=42)
        
        for stratum in ['A', 'B', 'C', 'D', 'E']:
            for i in range(20):
                sampler.count_stratum(stratum)
        
        sampler.initialize_reservoirs()
        
        idx = 0
        for stratum in ['A', 'B', 'C', 'D', 'E']:
            for i in range(20):
                sampler.add(stratum, idx, [f"{stratum}_{i}"])
                idx += 1
        
        sample = sampler.get_sample()
        
        assert len(sample) == 10


class TestSamplerService:
    """Tests for SamplerService class."""
    
    @pytest.fixture
    def service(self):
        """Create a fresh service instance for each test."""
        return SamplerService()
    
    @pytest.fixture
    def sample_excel_file(self, tmp_path):
        """Create a sample Excel file for testing."""
        df = pd.DataFrame({
            'id': range(1, 101),
            'category': ['A'] * 30 + ['B'] * 40 + ['C'] * 30,
            'value': np.random.randint(1, 100, 100),
            'weight': np.random.uniform(0.1, 1.0, 100)
        })
        file_path = tmp_path / "test_data.xlsx"
        df.to_excel(file_path, index=False)
        return file_path
    
    def test_store_and_get_sample(self, service):
        """Test storing and retrieving a sample."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        metadata = {'method': 'random', 'size': 3}
        
        service.store_sample('sample123', df, metadata)
        
        retrieved_df = service.get_sample('sample123')
        retrieved_metadata = service.get_sample_metadata('sample123')
        
        assert retrieved_df is not None
        assert len(retrieved_df) == 3
        assert retrieved_metadata['method'] == 'random'
    
    def test_get_nonexistent_sample(self, service):
        """Test getting a sample that doesn't exist."""
        result = service.get_sample('nonexistent')
        
        assert result is None
    
    def test_get_nonexistent_metadata(self, service):
        """Test getting metadata for a sample that doesn't exist."""
        result = service.get_sample_metadata('nonexistent')
        
        assert result is None
    
    def test_iter_sample_chunks(self, service):
        """Test iterating over sample chunks."""
        df = pd.DataFrame({'a': range(100)})
        service.store_sample('sample123', df, {})
        
        chunks = list(service.iter_sample_chunks('sample123', chunk_size=30))
        
        assert len(chunks) == 4
        assert len(chunks[0]) == 30
        assert len(chunks[1]) == 30
        assert len(chunks[2]) == 30
        assert len(chunks[3]) == 10
    
    def test_iter_sample_chunks_nonexistent(self, service):
        """Test iterating over chunks for nonexistent sample."""
        chunks = list(service.iter_sample_chunks('nonexistent'))
        
        assert chunks == []


class TestSamplerServiceWithMockedParser:
    """Tests for SamplerService sampling methods with mocked parser."""
    
    @pytest.fixture
    def service(self):
        """Create a fresh service instance for each test."""
        return SamplerService()
    
    @pytest.fixture
    def mock_parser(self):
        """Create a mock parser that yields DataFrame chunks."""
        parser = MagicMock()
        
        df = pd.DataFrame({
            'id': range(1, 101),
            'category': ['A'] * 30 + ['B'] * 40 + ['C'] * 30,
            'value': list(range(1, 101)),
            'weight': [1.0] * 100
        })
        
        def iter_rows(sheet_name=None):
            for i in range(0, len(df), 10):
                yield df.iloc[i:i+10]
        
        parser.iter_rows = iter_rows
        parser.get_row_count = MagicMock(return_value=100)
        parser.get_file_info = MagicMock(return_value={
            'row_count': 100,
            'columns': ['id', 'category', 'value', 'weight']
        })
        
        return parser
    
    def test_sample_random(self, service, mock_parser):
        """Test random sampling."""
        result = service.sample_random(mock_parser, sample_size=10, seed=42)
        
        assert len(result) == 10
        assert list(result.columns) == ['id', 'category', 'value', 'weight']
    
    def test_sample_random_reproducibility(self, service, mock_parser):
        """Test that random sampling is reproducible with same seed."""
        result1 = service.sample_random(mock_parser, sample_size=10, seed=42)
        result2 = service.sample_random(mock_parser, sample_size=10, seed=42)
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_sample_stratified(self, service, mock_parser):
        """Test stratified sampling."""
        result = service.sample_stratified(
            mock_parser, sample_size=10, strata_column='category', seed=42
        )
        
        assert len(result) == 10
        assert 'category' in result.columns
    
    def test_sample_stratified_invalid_column(self, service, mock_parser):
        """Test stratified sampling with invalid column."""
        with pytest.raises(ValueError, match="not found"):
            service.sample_stratified(
                mock_parser, sample_size=10, strata_column='nonexistent', seed=42
            )
    
    def test_sample_systematic(self, service, mock_parser):
        """Test systematic sampling."""
        result = service.sample_systematic(mock_parser, sample_size=10, seed=42)
        
        assert len(result) == 10
    
    def test_sample_cluster(self, service, mock_parser):
        """Test cluster sampling."""
        result = service.sample_cluster(
            mock_parser, sample_size=30, cluster_column='category', seed=42
        )
        
        assert len(result) <= 30
    
    def test_sample_cluster_invalid_column(self, service, mock_parser):
        """Test cluster sampling with invalid column."""
        with pytest.raises(ValueError, match="not found"):
            service.sample_cluster(
                mock_parser, sample_size=10, cluster_column='nonexistent', seed=42
            )
    
    def test_sample_weighted(self, service, mock_parser):
        """Test weighted sampling."""
        result = service.sample_weighted(
            mock_parser, sample_size=10, weight_column='weight', seed=42
        )
        
        assert len(result) == 10
    
    def test_sample_weighted_invalid_column(self, service, mock_parser):
        """Test weighted sampling with invalid column."""
        with pytest.raises(ValueError, match="not found"):
            service.sample_weighted(
                mock_parser, sample_size=10, weight_column='nonexistent', seed=42
            )


class TestSamplerServiceIntegration:
    """Integration tests for SamplerService with real Excel files."""
    
    @pytest.fixture
    def service(self):
        """Create a fresh service instance for each test."""
        return SamplerService()
    
    @pytest.fixture
    def sample_excel_file(self, tmp_path):
        """Create a sample Excel file for testing."""
        df = pd.DataFrame({
            'id': range(1, 101),
            'category': ['A'] * 30 + ['B'] * 40 + ['C'] * 30,
            'value': list(range(1, 101)),
            'weight': [1.0] * 50 + [2.0] * 50
        })
        file_path = tmp_path / "test_data.xlsx"
        df.to_excel(file_path, index=False)
        return file_path
    
    def test_sample_method_random(self, service, sample_excel_file):
        """Test sample method with random sampling."""
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.RANDOM,
            sample_size=10,
            random_seed=42
        )
        
        df, stats = service.sample(sample_excel_file, config)
        
        assert len(df) == 10
        assert stats['original_rows'] == 100
        assert stats['sampled_rows'] == 10
        assert stats['method'] == 'random'
    
    def test_sample_method_stratified(self, service, sample_excel_file):
        """Test sample method with stratified sampling."""
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.STRATIFIED,
            sample_size=10,
            strata_column='category',
            random_seed=42
        )
        
        df, stats = service.sample(sample_excel_file, config)
        
        assert len(df) == 10
        assert stats['method'] == 'stratified'
    
    def test_sample_method_systematic(self, service, sample_excel_file):
        """Test sample method with systematic sampling."""
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.SYSTEMATIC,
            sample_size=10,
            random_seed=42
        )
        
        df, stats = service.sample(sample_excel_file, config)
        
        assert len(df) == 10
        assert stats['method'] == 'systematic'
    
    def test_sample_method_cluster(self, service, sample_excel_file):
        """Test sample method with cluster sampling."""
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.CLUSTER,
            sample_size=30,
            cluster_column='category',
            random_seed=42
        )
        
        df, stats = service.sample(sample_excel_file, config)
        
        assert len(df) <= 30
        assert stats['method'] == 'cluster'
    
    def test_sample_method_weighted(self, service, sample_excel_file):
        """Test sample method with weighted sampling."""
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.WEIGHTED,
            sample_size=10,
            weight_column='weight',
            random_seed=42
        )
        
        df, stats = service.sample(sample_excel_file, config)
        
        assert len(df) == 10
        assert stats['method'] == 'weighted'
    
    def test_sample_with_percentage(self, service, sample_excel_file):
        """Test sampling with percentage instead of fixed size."""
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.RANDOM,
            sample_size=1,
            sample_percentage=10.0,
            random_seed=42
        )
        
        df, stats = service.sample(sample_excel_file, config)
        
        assert len(df) == 10
        assert stats['sampling_rate'] == 0.1
    
    def test_sample_stratified_missing_column(self, service, sample_excel_file):
        """Test stratified sampling without required column."""
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.STRATIFIED,
            sample_size=10,
            random_seed=42
        )
        
        with pytest.raises(ValueError, match="strata_column or strata_columns required"):
            service.sample(sample_excel_file, config)
    
    def test_sample_cluster_missing_column(self, service, sample_excel_file):
        """Test cluster sampling without required column."""
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.CLUSTER,
            sample_size=10,
            random_seed=42
        )
        
        with pytest.raises(ValueError, match="Cluster column required"):
            service.sample(sample_excel_file, config)
    
    def test_sample_weighted_missing_column(self, service, sample_excel_file):
        """Test weighted sampling without required column."""
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.WEIGHTED,
            sample_size=10,
            random_seed=42
        )
        
        with pytest.raises(ValueError, match="Weight column required"):
            service.sample(sample_excel_file, config)


class TestMultiColumnStratifiedSampling:
    """Tests for multi-column stratified sampling."""
    
    @pytest.fixture
    def service(self):
        """Create a fresh service instance for each test."""
        return SamplerService()
    
    @pytest.fixture
    def multi_column_excel_file(self, tmp_path):
        """Create an Excel file with multiple categorical columns for testing."""
        # Create data with 2 categorical columns: region and category
        # Region: North (40), South (60)
        # Category: A (50), B (50)
        # Combined strata: North-A (20), North-B (20), South-A (30), South-B (30)
        data = []
        for i in range(20):
            data.append({'id': len(data) + 1, 'region': 'North', 'category': 'A', 'value': i})
        for i in range(20):
            data.append({'id': len(data) + 1, 'region': 'North', 'category': 'B', 'value': i})
        for i in range(30):
            data.append({'id': len(data) + 1, 'region': 'South', 'category': 'A', 'value': i})
        for i in range(30):
            data.append({'id': len(data) + 1, 'region': 'South', 'category': 'B', 'value': i})
        
        df = pd.DataFrame(data)
        file_path = tmp_path / "multi_column_test.xlsx"
        df.to_excel(file_path, index=False)
        return file_path
    
    def test_multi_column_stratified_sampling(self, service, multi_column_excel_file):
        """Test stratified sampling with multiple columns."""
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.STRATIFIED,
            sample_size=20,
            strata_columns=['region', 'category'],
            random_seed=42
        )
        
        df, stats = service.sample(multi_column_excel_file, config)
        
        assert len(df) == 20
        assert stats['method'] == 'stratified'
        
        # Check that all strata are represented
        strata_counts = df.groupby(['region', 'category']).size()
        assert len(strata_counts) == 4  # All 4 combinations should be present
    
    def test_multi_column_proportional_representation(self, service, multi_column_excel_file):
        """Test that multi-column stratified sampling maintains proportions."""
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.STRATIFIED,
            sample_size=20,
            strata_columns=['region', 'category'],
            random_seed=42
        )
        
        df, stats = service.sample(multi_column_excel_file, config)
        
        # Original proportions: North-A: 20%, North-B: 20%, South-A: 30%, South-B: 30%
        # With sample_size=20, expected: North-A: 4, North-B: 4, South-A: 6, South-B: 6
        strata_counts = df.groupby(['region', 'category']).size().to_dict()
        
        # Allow some variance due to rounding
        assert strata_counts.get(('North', 'A'), 0) >= 3
        assert strata_counts.get(('North', 'B'), 0) >= 3
        assert strata_counts.get(('South', 'A'), 0) >= 5
        assert strata_counts.get(('South', 'B'), 0) >= 5
    
    def test_single_column_backward_compatibility(self, service, multi_column_excel_file):
        """Test that single strata_column still works (backward compatibility)."""
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.STRATIFIED,
            sample_size=10,
            strata_column='region',
            random_seed=42
        )
        
        df, stats = service.sample(multi_column_excel_file, config)
        
        assert len(df) == 10
        assert 'region' in df.columns
    
    def test_strata_columns_takes_precedence(self, service, multi_column_excel_file):
        """Test that strata_columns takes precedence over strata_column."""
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.STRATIFIED,
            sample_size=20,
            strata_column='region',  # This should be ignored
            strata_columns=['region', 'category'],  # This should be used
            random_seed=42
        )
        
        df, stats = service.sample(multi_column_excel_file, config)
        
        # Should have 4 strata (region x category), not 2 (just region)
        strata_counts = df.groupby(['region', 'category']).size()
        assert len(strata_counts) == 4
    
    def test_invalid_strata_column_in_list(self, service, multi_column_excel_file):
        """Test error when one of the strata columns doesn't exist."""
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.STRATIFIED,
            sample_size=10,
            strata_columns=['region', 'nonexistent'],
            random_seed=42
        )
        
        with pytest.raises(ValueError, match="not found"):
            service.sample(multi_column_excel_file, config)
    
    def test_three_column_stratification(self, service, tmp_path):
        """Test stratified sampling with three columns."""
        # Create data with 3 categorical columns
        data = []
        for region in ['North', 'South']:
            for category in ['A', 'B']:
                for status in ['Active', 'Inactive']:
                    for i in range(10):
                        data.append({
                            'id': len(data) + 1,
                            'region': region,
                            'category': category,
                            'status': status,
                            'value': i
                        })
        
        df = pd.DataFrame(data)
        file_path = tmp_path / "three_column_test.xlsx"
        df.to_excel(file_path, index=False)
        
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.STRATIFIED,
            sample_size=16,  # 2 per stratum (8 strata total)
            strata_columns=['region', 'category', 'status'],
            random_seed=42
        )
        
        result_df, stats = service.sample(file_path, config)
        
        assert len(result_df) == 16
        
        # All 8 strata should be represented
        strata_counts = result_df.groupby(['region', 'category', 'status']).size()
        assert len(strata_counts) == 8
    
    def test_dict_format_value_filtering(self, service, multi_column_excel_file):
        """Test stratified sampling with dict format to filter specific values."""
        # multi_column_excel_file has: North-A (20), North-B (20), South-A (30), South-B (30)
        # Filter to only North region and category A
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.STRATIFIED,
            sample_size=10,
            strata_columns={"region": ["North"], "category": ["A"]},
            random_seed=42
        )
        
        df, stats = service.sample(multi_column_excel_file, config)
        
        # Should only have rows from North-A stratum (20 rows total, sample 10)
        assert len(df) == 10
        assert all(df['region'] == 'North')
        assert all(df['category'] == 'A')
    
    def test_dict_format_multiple_values_per_column(self, service, multi_column_excel_file):
        """Test dict format with multiple allowed values per column."""
        # Filter to North/South regions but only category A
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.STRATIFIED,
            sample_size=10,
            strata_columns={"region": ["North", "South"], "category": ["A"]},
            random_seed=42
        )
        
        df, stats = service.sample(multi_column_excel_file, config)
        
        # Should have rows from North-A (20) and South-A (30) = 50 total, sample 10
        assert len(df) == 10
        assert all(df['category'] == 'A')
        # Both regions should be represented proportionally
        region_counts = df['region'].value_counts()
        assert 'North' in region_counts.index
        assert 'South' in region_counts.index
    
    def test_dict_format_excludes_non_matching_rows(self, service, tmp_path):
        """Test that dict format excludes rows not matching the filter."""
        # Create data with 3 categories
        data = []
        for cat in ['A', 'B', 'C']:
            for i in range(30):
                data.append({'id': len(data) + 1, 'category': cat, 'value': i})
        
        df = pd.DataFrame(data)
        file_path = tmp_path / "three_cat_test.xlsx"
        df.to_excel(file_path, index=False)
        
        # Only sample from categories A and B, exclude C
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.STRATIFIED,
            sample_size=10,
            strata_columns={"category": ["A", "B"]},
            random_seed=42
        )
        
        result_df, stats = service.sample(file_path, config)
        
        assert len(result_df) == 10
        # No rows should have category C
        assert 'C' not in result_df['category'].values
        # Both A and B should be present
        assert 'A' in result_df['category'].values
        assert 'B' in result_df['category'].values
    
    def test_dict_format_with_nonexistent_value(self, service, multi_column_excel_file):
        """Test dict format with a value that doesn't exist in data."""
        # Filter to a region that doesn't exist
        config = SamplingConfig(
            file_id="test123",
            method=SamplingMethod.STRATIFIED,
            sample_size=10,
            strata_columns={"region": ["East"]},  # East doesn't exist
            random_seed=42
        )
        
        df, stats = service.sample(multi_column_excel_file, config)
        
        # Should return empty DataFrame since no rows match
        assert len(df) == 0
