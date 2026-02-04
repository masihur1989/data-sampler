"""
Tests for preprocessing utility functions.

These tests verify the pure transformation functions in preprocessing_utils.py.
Each function is tested independently with various inputs and edge cases.
"""

import pytest
import pandas as pd
import numpy as np

from app.utils.preprocessing_utils import (
    analyze_data,
    remove_duplicates,
    handle_missing_values,
    trim_whitespace,
    remove_empty_rows,
    remove_empty_columns,
    convert_types,
    normalize_columns,
    filter_data,
    select_columns,
    generate_column_stats,
)


class TestAnalyzeData:
    """Tests for analyze_data function."""
    
    def test_basic_analysis(self):
        """Test basic data analysis."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10.5, 20.0, 30.5]
        })
        
        result = analyze_data(df)
        
        assert result['row_count'] == 3
        assert result['column_count'] == 3
        assert result['columns'] == ['id', 'name', 'value']
        assert result['duplicate_rows'] == 0
    
    def test_analysis_with_missing_values(self):
        """Test analysis with missing values."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': ['x', 'y', None]
        })
        
        result = analyze_data(df)
        
        assert result['missing_values']['a'] == 1
        assert result['missing_values']['b'] == 1
    
    def test_analysis_with_duplicates(self):
        """Test analysis with duplicate rows."""
        df = pd.DataFrame({
            'a': [1, 1, 2],
            'b': ['x', 'x', 'y']
        })
        
        result = analyze_data(df)
        
        assert result['duplicate_rows'] == 1
    
    def test_numeric_stats(self):
        """Test numeric column statistics."""
        df = pd.DataFrame({
            'value': [10, 20, 30, 40, 50]
        })
        
        result = analyze_data(df)
        
        assert 'numeric_stats' in result
        assert 'value' in result['numeric_stats']
    
    def test_categorical_stats(self):
        """Test categorical column statistics."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'A']
        })
        
        result = analyze_data(df)
        
        assert 'categorical_stats' in result
        assert 'category' in result['categorical_stats']
        assert result['categorical_stats']['category']['unique_values'] == 3


class TestRemoveDuplicates:
    """Tests for remove_duplicates function."""
    
    def test_no_duplicates(self):
        """Test with no duplicates."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        result_df, count = remove_duplicates(df)
        
        assert len(result_df) == 3
        assert count == 0
    
    def test_with_duplicates(self):
        """Test with duplicate rows."""
        df = pd.DataFrame({
            'a': [1, 1, 2, 2, 3],
            'b': ['x', 'x', 'y', 'y', 'z']
        })
        
        result_df, count = remove_duplicates(df)
        
        assert len(result_df) == 3
        assert count == 2
    
    def test_all_duplicates(self):
        """Test when all rows are duplicates."""
        df = pd.DataFrame({
            'a': [1, 1, 1],
            'b': ['x', 'x', 'x']
        })
        
        result_df, count = remove_duplicates(df)
        
        assert len(result_df) == 1
        assert count == 2


class TestHandleMissingValues:
    """Tests for handle_missing_values function."""
    
    def test_keep_strategy(self):
        """Test keep strategy (no changes)."""
        df = pd.DataFrame({'a': [1, None, 3]})
        
        result_df, count = handle_missing_values(df, 'keep')
        
        assert len(result_df) == 3
        assert result_df['a'].isnull().sum() == 1
        assert count == 0
    
    def test_drop_strategy(self):
        """Test drop strategy."""
        df = pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, None]})
        
        result_df, count = handle_missing_values(df, 'drop')
        
        assert len(result_df) == 1
        assert count == 2
    
    def test_fill_mean_strategy(self):
        """Test fill_mean strategy."""
        df = pd.DataFrame({'a': [1.0, None, 3.0]})
        
        result_df, count = handle_missing_values(df, 'fill_mean')
        
        assert result_df['a'].isnull().sum() == 0
        assert result_df['a'].iloc[1] == 2.0  # mean of 1 and 3
        assert count == 1
    
    def test_fill_median_strategy(self):
        """Test fill_median strategy."""
        df = pd.DataFrame({'a': [1.0, None, 5.0]})
        
        result_df, count = handle_missing_values(df, 'fill_median')
        
        assert result_df['a'].isnull().sum() == 0
        assert result_df['a'].iloc[1] == 3.0  # median of 1 and 5
        assert count == 1
    
    def test_fill_value_strategy(self):
        """Test fill_value strategy."""
        df = pd.DataFrame({'a': [1, None, 3]})
        
        result_df, count = handle_missing_values(df, 'fill_value', '0')
        
        assert result_df['a'].isnull().sum() == 0
        assert count == 1


class TestTrimWhitespace:
    """Tests for trim_whitespace function."""
    
    def test_trim_leading_trailing(self):
        """Test trimming leading and trailing whitespace."""
        df = pd.DataFrame({'name': ['  Alice  ', 'Bob  ', '  Charlie']})
        
        result_df = trim_whitespace(df)
        
        assert result_df['name'].tolist() == ['Alice', 'Bob', 'Charlie']
    
    def test_no_whitespace(self):
        """Test with no whitespace to trim."""
        df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie']})
        
        result_df = trim_whitespace(df)
        
        assert result_df['name'].tolist() == ['Alice', 'Bob', 'Charlie']
    
    def test_numeric_columns_unchanged(self):
        """Test that numeric columns are unchanged."""
        df = pd.DataFrame({
            'name': ['  Alice  '],
            'value': [10]
        })
        
        result_df = trim_whitespace(df)
        
        assert result_df['value'].iloc[0] == 10


class TestRemoveEmptyRows:
    """Tests for remove_empty_rows function."""
    
    def test_no_empty_rows(self):
        """Test with no empty rows."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        result_df, count = remove_empty_rows(df)
        
        assert len(result_df) == 3
        assert count == 0
    
    def test_with_empty_rows(self):
        """Test with empty rows."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': [4, None, 6]
        })
        
        result_df, count = remove_empty_rows(df)
        
        assert len(result_df) == 2
        assert count == 1
    
    def test_partial_empty_rows_kept(self):
        """Test that partially empty rows are kept."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': [4, 5, 6]
        })
        
        result_df, count = remove_empty_rows(df)
        
        assert len(result_df) == 3
        assert count == 0


class TestRemoveEmptyColumns:
    """Tests for remove_empty_columns function."""
    
    def test_no_empty_columns(self):
        """Test with no empty columns."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        
        result_df, removed = remove_empty_columns(df)
        
        assert len(result_df.columns) == 2
        assert removed == []
    
    def test_with_empty_columns(self):
        """Test with empty columns."""
        df = pd.DataFrame({
            'a': [1, 2],
            'b': [None, None],
            'c': [3, 4]
        })
        
        result_df, removed = remove_empty_columns(df)
        
        assert len(result_df.columns) == 2
        assert 'b' in removed


class TestConvertTypes:
    """Tests for convert_types function."""
    
    def test_convert_numeric_strings(self):
        """Test converting numeric strings to numbers."""
        df = pd.DataFrame({'a': ['1', '2', '3']})
        
        result_df, conversions = convert_types(df)
        
        assert result_df['a'].dtype in [np.float64, np.int64]
        assert 'a' in conversions
    
    def test_keep_non_numeric_strings(self):
        """Test that non-numeric strings are kept as strings."""
        df = pd.DataFrame({'a': ['Alice', 'Bob', 'Charlie']})
        
        result_df, conversions = convert_types(df)
        
        assert 'a' not in conversions
    
    def test_mixed_column_threshold(self):
        """Test that columns with <50% valid numbers are not converted."""
        df = pd.DataFrame({'a': ['1', 'two', 'three', 'four']})
        
        result_df, conversions = convert_types(df)
        
        assert 'a' not in conversions


class TestNormalizeColumns:
    """Tests for normalize_columns function."""
    
    def test_normalize_single_column(self):
        """Test normalizing a single column."""
        df = pd.DataFrame({'value': [0, 50, 100]})
        
        result_df, normalizations = normalize_columns(df, ['value'])
        
        assert result_df['value'].min() == 0.0
        assert result_df['value'].max() == 1.0
        assert result_df['value'].iloc[1] == 0.5
        assert 'value' in normalizations
    
    def test_normalize_nonexistent_column(self):
        """Test normalizing a column that doesn't exist."""
        df = pd.DataFrame({'value': [0, 50, 100]})
        
        result_df, normalizations = normalize_columns(df, ['nonexistent'])
        
        assert 'nonexistent' not in normalizations
    
    def test_normalize_non_numeric_column(self):
        """Test that non-numeric columns are not normalized."""
        df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie']})
        
        result_df, normalizations = normalize_columns(df, ['name'])
        
        assert 'name' not in normalizations


class TestFilterData:
    """Tests for filter_data function."""
    
    def test_filter_eq(self):
        """Test equals filter."""
        df = pd.DataFrame({'category': ['A', 'B', 'A', 'C']})
        
        result_df, warnings = filter_data(df, {'category': {'op': 'eq', 'value': 'A'}})
        
        assert len(result_df) == 2
    
    def test_filter_gt(self):
        """Test greater than filter."""
        df = pd.DataFrame({'value': [10, 20, 30, 40]})
        
        result_df, warnings = filter_data(df, {'value': {'op': 'gt', 'value': 25}})
        
        assert len(result_df) == 2
    
    def test_filter_in(self):
        """Test in list filter."""
        df = pd.DataFrame({'category': ['A', 'B', 'C', 'D']})
        
        result_df, warnings = filter_data(df, {'category': {'op': 'in', 'value': ['A', 'C']}})
        
        assert len(result_df) == 2
    
    def test_filter_contains(self):
        """Test contains filter."""
        df = pd.DataFrame({'name': ['Alice Corp', 'Bob Inc', 'Charlie Corp']})
        
        result_df, warnings = filter_data(df, {'name': {'op': 'contains', 'value': 'Corp'}})
        
        assert len(result_df) == 2
    
    def test_filter_nonexistent_column(self):
        """Test filter on nonexistent column."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        result_df, warnings = filter_data(df, {'nonexistent': {'op': 'eq', 'value': 1}})
        
        assert len(result_df) == 3
        assert any('not found' in w for w in warnings)


class TestSelectColumns:
    """Tests for select_columns function."""
    
    def test_keep_columns(self):
        """Test keeping specific columns."""
        df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        
        result_df, dropped, warnings = select_columns(df, columns_to_keep=['a', 'c'])
        
        assert list(result_df.columns) == ['a', 'c']
    
    def test_drop_columns(self):
        """Test dropping specific columns."""
        df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        
        result_df, dropped, warnings = select_columns(df, columns_to_drop=['b'])
        
        assert 'b' not in result_df.columns
        assert 'b' in dropped
    
    def test_keep_nonexistent_column(self):
        """Test keeping a column that doesn't exist."""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        
        result_df, dropped, warnings = select_columns(df, columns_to_keep=['a', 'nonexistent'])
        
        assert list(result_df.columns) == ['a']
        assert any('not found' in w for w in warnings)


class TestGenerateColumnStats:
    """Tests for generate_column_stats function."""
    
    def test_numeric_column_stats(self):
        """Test statistics for numeric columns."""
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})
        
        stats = generate_column_stats(df)
        
        assert 'value' in stats
        assert stats['value']['min'] == 10
        assert stats['value']['max'] == 50
        assert stats['value']['mean'] == 30
    
    def test_string_column_stats(self):
        """Test statistics for string columns."""
        df = pd.DataFrame({'name': ['Alice', 'Bob', None]})
        
        stats = generate_column_stats(df)
        
        assert 'name' in stats
        assert stats['name']['non_null_count'] == 2
        assert stats['name']['null_count'] == 1
        assert stats['name']['unique_count'] == 2
