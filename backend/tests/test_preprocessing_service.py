"""
Tests for PreprocessingService class.

These tests verify the service class that orchestrates preprocessing operations
and manages state (storing processed data, reports, metadata).
"""

import pytest
import pandas as pd
import numpy as np

from app.services.preprocessing_service import (
    PreprocessingService,
    PreprocessingConfig,
    QualityReport,
)


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PreprocessingConfig()
        
        assert config.remove_duplicates is True
        assert config.handle_missing == "keep"
        assert config.fill_value is None
        assert config.trim_whitespace is True
        assert config.remove_empty_rows is True
        assert config.remove_empty_columns is True
        assert config.convert_types is False
        assert config.normalize_columns == []
        assert config.filter_conditions == {}
        assert config.columns_to_keep is None
        assert config.columns_to_drop == []
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PreprocessingConfig(
            remove_duplicates=False,
            handle_missing="fill_mean",
            convert_types=True,
            normalize_columns=["value"],
            columns_to_drop=["temp"]
        )
        
        assert config.remove_duplicates is False
        assert config.handle_missing == "fill_mean"
        assert config.convert_types is True
        assert config.normalize_columns == ["value"]
        assert config.columns_to_drop == ["temp"]


class TestQualityReport:
    """Tests for QualityReport class."""
    
    def test_default_report(self):
        """Test default report values."""
        report = QualityReport()
        
        assert report.original_rows == 0
        assert report.original_columns == 0
        assert report.processed_rows == 0
        assert report.processed_columns == 0
        assert report.duplicates_removed == 0
        assert report.missing_values_handled == 0
        assert report.empty_rows_removed == 0
        assert report.empty_columns_removed == 0
        assert report.columns_dropped == []
        assert report.type_conversions == {}
        assert report.column_stats == {}
        assert report.warnings == []
        assert report.processing_time_ms == 0
    
    def test_to_dict(self):
        """Test converting report to dictionary."""
        report = QualityReport()
        report.original_rows = 100
        report.processed_rows = 90
        report.duplicates_removed = 10
        
        result = report.to_dict()
        
        assert isinstance(result, dict)
        assert result['original_rows'] == 100
        assert result['processed_rows'] == 90
        assert result['duplicates_removed'] == 10


class TestPreprocessingService:
    """Tests for PreprocessingService class."""
    
    @pytest.fixture
    def service(self):
        """Create a fresh service instance for each test."""
        return PreprocessingService()
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'id': [1, 2, 2, 3, 4],
            'name': ['  Alice  ', 'Bob', 'Bob', 'Charlie', None],
            'value': [10.0, 20.0, 20.0, None, 40.0],
            'category': ['A', 'B', 'B', 'A', 'C']
        })
    
    def test_analyze_data(self, service, sample_df):
        """Test data analysis."""
        result = service.analyze_data(sample_df)
        
        assert result['row_count'] == 5
        assert result['column_count'] == 4
        assert 'id' in result['columns']
        assert result['duplicate_rows'] == 1
    
    def test_preprocess_default_config(self, service, sample_df):
        """Test preprocessing with default configuration."""
        result_df, report = service.preprocess(sample_df)
        
        assert report.original_rows == 5
        assert report.duplicates_removed == 1
        assert len(result_df) < 5
    
    def test_preprocess_remove_duplicates(self, service):
        """Test duplicate removal."""
        df = pd.DataFrame({
            'a': [1, 1, 2],
            'b': ['x', 'x', 'y']
        })
        config = PreprocessingConfig(
            remove_duplicates=True,
            remove_empty_rows=False,
            remove_empty_columns=False,
            trim_whitespace=False,
            handle_missing="keep"
        )
        
        result_df, report = service.preprocess(df, config)
        
        assert len(result_df) == 2
        assert report.duplicates_removed == 1
    
    def test_preprocess_handle_missing_drop(self, service):
        """Test dropping rows with missing values."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': [4, 5, 6]
        })
        config = PreprocessingConfig(
            remove_duplicates=False,
            remove_empty_rows=False,
            remove_empty_columns=False,
            trim_whitespace=False,
            handle_missing="drop"
        )
        
        result_df, report = service.preprocess(df, config)
        
        assert len(result_df) == 2
        assert report.missing_values_handled == 1
    
    def test_preprocess_handle_missing_fill_mean(self, service):
        """Test filling missing values with mean."""
        df = pd.DataFrame({
            'value': [10.0, None, 30.0]
        })
        config = PreprocessingConfig(
            remove_duplicates=False,
            remove_empty_rows=False,
            remove_empty_columns=False,
            trim_whitespace=False,
            handle_missing="fill_mean"
        )
        
        result_df, report = service.preprocess(df, config)
        
        assert result_df['value'].isnull().sum() == 0
        assert result_df['value'].iloc[1] == 20.0
    
    def test_preprocess_trim_whitespace(self, service):
        """Test whitespace trimming."""
        df = pd.DataFrame({
            'name': ['  Alice  ', '  Bob  ']
        })
        config = PreprocessingConfig(
            remove_duplicates=False,
            remove_empty_rows=False,
            remove_empty_columns=False,
            trim_whitespace=True,
            handle_missing="keep"
        )
        
        result_df, report = service.preprocess(df, config)
        
        assert result_df['name'].tolist() == ['Alice', 'Bob']
    
    def test_preprocess_remove_empty_rows(self, service):
        """Test removing empty rows."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': [4, None, 6]
        })
        config = PreprocessingConfig(
            remove_duplicates=False,
            remove_empty_rows=True,
            remove_empty_columns=False,
            trim_whitespace=False,
            handle_missing="keep"
        )
        
        result_df, report = service.preprocess(df, config)
        
        assert len(result_df) == 2
        assert report.empty_rows_removed == 1
    
    def test_preprocess_remove_empty_columns(self, service):
        """Test removing empty columns."""
        df = pd.DataFrame({
            'a': [1, 2],
            'b': [None, None],
            'c': [3, 4]
        })
        config = PreprocessingConfig(
            remove_duplicates=False,
            remove_empty_rows=False,
            remove_empty_columns=True,
            trim_whitespace=False,
            handle_missing="keep"
        )
        
        result_df, report = service.preprocess(df, config)
        
        assert 'b' not in result_df.columns
        assert report.empty_columns_removed == 1
    
    def test_preprocess_convert_types(self, service):
        """Test type conversion."""
        df = pd.DataFrame({
            'value': ['1', '2', '3']
        })
        config = PreprocessingConfig(
            remove_duplicates=False,
            remove_empty_rows=False,
            remove_empty_columns=False,
            trim_whitespace=False,
            handle_missing="keep",
            convert_types=True
        )
        
        result_df, report = service.preprocess(df, config)
        
        assert result_df['value'].dtype in [np.float64, np.int64]
        assert 'value' in report.type_conversions
    
    def test_preprocess_normalize_columns(self, service):
        """Test column normalization."""
        df = pd.DataFrame({
            'value': [0, 50, 100]
        })
        config = PreprocessingConfig(
            remove_duplicates=False,
            remove_empty_rows=False,
            remove_empty_columns=False,
            trim_whitespace=False,
            handle_missing="keep",
            normalize_columns=['value']
        )
        
        result_df, report = service.preprocess(df, config)
        
        assert result_df['value'].min() == 0.0
        assert result_df['value'].max() == 1.0
        assert result_df['value'].iloc[1] == 0.5
    
    def test_preprocess_filter_data(self, service):
        """Test data filtering."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C'],
            'value': [10, 20, 30, 40]
        })
        config = PreprocessingConfig(
            remove_duplicates=False,
            remove_empty_rows=False,
            remove_empty_columns=False,
            trim_whitespace=False,
            handle_missing="keep",
            filter_conditions={'category': {'op': 'eq', 'value': 'A'}}
        )
        
        result_df, report = service.preprocess(df, config)
        
        assert len(result_df) == 2
        assert all(result_df['category'] == 'A')
    
    def test_preprocess_select_columns(self, service):
        """Test column selection."""
        df = pd.DataFrame({
            'a': [1, 2],
            'b': [3, 4],
            'c': [5, 6]
        })
        config = PreprocessingConfig(
            remove_duplicates=False,
            remove_empty_rows=False,
            remove_empty_columns=False,
            trim_whitespace=False,
            handle_missing="keep",
            columns_to_keep=['a', 'c']
        )
        
        result_df, report = service.preprocess(df, config)
        
        assert list(result_df.columns) == ['a', 'c']
    
    def test_preprocess_drop_columns(self, service):
        """Test dropping columns."""
        df = pd.DataFrame({
            'a': [1, 2],
            'b': [3, 4],
            'c': [5, 6]
        })
        config = PreprocessingConfig(
            remove_duplicates=False,
            remove_empty_rows=False,
            remove_empty_columns=False,
            trim_whitespace=False,
            handle_missing="keep",
            columns_to_drop=['b']
        )
        
        result_df, report = service.preprocess(df, config)
        
        assert 'b' not in result_df.columns
        assert 'b' in report.columns_dropped
    
    def test_preprocess_and_store(self, service, sample_df):
        """Test preprocessing and storing data."""
        processed_id, report = service.preprocess_and_store('file123', sample_df)
        
        assert processed_id is not None
        assert report.original_rows == 5
        
        stored_df = service.get_processed(processed_id)
        assert stored_df is not None
        assert len(stored_df) < 5
    
    def test_get_processed_not_found(self, service):
        """Test getting non-existent processed data."""
        result = service.get_processed('nonexistent')
        
        assert result is None
    
    def test_get_report(self, service, sample_df):
        """Test getting quality report."""
        processed_id, _ = service.preprocess_and_store('file123', sample_df)
        
        report = service.get_report(processed_id)
        
        assert report is not None
        assert isinstance(report, QualityReport)
    
    def test_get_report_not_found(self, service):
        """Test getting non-existent report."""
        result = service.get_report('nonexistent')
        
        assert result is None
    
    def test_get_metadata(self, service, sample_df):
        """Test getting metadata."""
        processed_id, _ = service.preprocess_and_store('file123', sample_df)
        
        metadata = service.get_metadata(processed_id)
        
        assert metadata is not None
        assert metadata['original_file_id'] == 'file123'
        assert 'processed_at' in metadata
        assert 'config' in metadata
    
    def test_get_metadata_not_found(self, service):
        """Test getting non-existent metadata."""
        result = service.get_metadata('nonexistent')
        
        assert result is None
    
    def test_processing_time_recorded(self, service, sample_df):
        """Test that processing time is recorded."""
        _, report = service.preprocess(sample_df)
        
        assert report.processing_time_ms >= 0
    
    def test_column_stats_generated(self, service):
        """Test that column statistics are generated."""
        df = pd.DataFrame({
            'value': [10, 20, 30],
            'name': ['A', 'B', 'C']
        })
        config = PreprocessingConfig(
            remove_duplicates=False,
            remove_empty_rows=False,
            remove_empty_columns=False,
            trim_whitespace=False,
            handle_missing="keep"
        )
        
        _, report = service.preprocess(df, config)
        
        assert 'value' in report.column_stats
        assert 'name' in report.column_stats
        assert report.column_stats['value']['min'] == 10
        assert report.column_stats['value']['max'] == 30
