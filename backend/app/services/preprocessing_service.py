"""
Pre-processing Service Module

This module provides data pre-processing capabilities for raw Excel files before sampling.
It manages the pre-processing pipeline and stores processed data for subsequent sampling.

The actual transformation logic is delegated to pure functions in preprocessing_utils.py,
making the transformations easier to test and reuse independently.

Pre-processing Steps:
1. Data Cleaning: Remove duplicates, handle missing values, trim whitespace
2. Data Validation: Check data types, validate ranges, detect anomalies
3. Data Transformation: Type conversion, normalization, filtering
4. Quality Report: Generate statistics about data quality
"""

import uuid
import time
from pathlib import Path
from typing import Optional
from datetime import datetime
import pandas as pd

from app.config import UPLOAD_DIR
from app.utils.preprocessing_utils import (
    analyze_data as _analyze_data,
    remove_duplicates as _remove_duplicates,
    handle_missing_values as _handle_missing_values,
    trim_whitespace as _trim_whitespace,
    remove_empty_rows as _remove_empty_rows,
    remove_empty_columns as _remove_empty_columns,
    convert_types as _convert_types,
    normalize_columns as _normalize_columns,
    filter_data as _filter_data,
    select_columns as _select_columns,
    generate_column_stats as _generate_column_stats,
)


class PreprocessingConfig:
    """Configuration for pre-processing operations."""
    
    def __init__(
        self,
        remove_duplicates: bool = True,
        handle_missing: str = "keep",  # "keep", "drop", "fill_mean", "fill_median", "fill_mode", "fill_value"
        fill_value: Optional[str] = None,
        trim_whitespace: bool = True,
        remove_empty_rows: bool = True,
        remove_empty_columns: bool = True,
        convert_types: bool = False,
        normalize_columns: Optional[list[str]] = None,
        filter_conditions: Optional[dict] = None,
        columns_to_keep: Optional[list[str]] = None,
        columns_to_drop: Optional[list[str]] = None,
    ):
        """
        Initialize pre-processing configuration.
        
        Args:
            remove_duplicates: Remove duplicate rows
            handle_missing: Strategy for missing values
            fill_value: Value to use when handle_missing="fill_value"
            trim_whitespace: Trim whitespace from string columns
            remove_empty_rows: Remove rows where all values are empty
            remove_empty_columns: Remove columns where all values are empty
            convert_types: Attempt to convert string columns to numeric
            normalize_columns: List of columns to normalize (0-1 scale)
            filter_conditions: Dict of column -> condition for filtering
            columns_to_keep: Only keep these columns (if specified)
            columns_to_drop: Drop these columns
        """
        self.remove_duplicates = remove_duplicates
        self.handle_missing = handle_missing
        self.fill_value = fill_value
        self.trim_whitespace = trim_whitespace
        self.remove_empty_rows = remove_empty_rows
        self.remove_empty_columns = remove_empty_columns
        self.convert_types = convert_types
        self.normalize_columns = normalize_columns or []
        self.filter_conditions = filter_conditions or {}
        self.columns_to_keep = columns_to_keep
        self.columns_to_drop = columns_to_drop or []


class QualityReport:
    """Data quality report generated during pre-processing."""
    
    def __init__(self):
        self.original_rows = 0
        self.original_columns = 0
        self.processed_rows = 0
        self.processed_columns = 0
        self.duplicates_removed = 0
        self.missing_values_handled = 0
        self.empty_rows_removed = 0
        self.empty_columns_removed = 0
        self.columns_dropped = []
        self.type_conversions = {}
        self.column_stats = {}
        self.warnings = []
        self.processing_time_ms = 0
    
    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "original_rows": self.original_rows,
            "original_columns": self.original_columns,
            "processed_rows": self.processed_rows,
            "processed_columns": self.processed_columns,
            "duplicates_removed": self.duplicates_removed,
            "missing_values_handled": self.missing_values_handled,
            "empty_rows_removed": self.empty_rows_removed,
            "empty_columns_removed": self.empty_columns_removed,
            "columns_dropped": self.columns_dropped,
            "type_conversions": self.type_conversions,
            "column_stats": self.column_stats,
            "warnings": self.warnings,
            "processing_time_ms": self.processing_time_ms,
        }


class PreprocessingService:
    """
    Service for pre-processing raw Excel files before sampling.
    
    This service provides a configurable pipeline for cleaning, validating,
    and transforming data. The processed data is stored for subsequent
    sampling operations.
    
    Workflow:
    1. Upload raw file -> file_service
    2. Pre-process file -> preprocessing_service (this)
    3. Sample from processed file -> sampler_service
    """
    
    def __init__(self):
        """Initialize the pre-processing service."""
        self._processed_store: dict[str, pd.DataFrame] = {}
        self._reports: dict[str, QualityReport] = {}
        self._metadata: dict[str, dict] = {}
    
    def analyze_data(self, df: pd.DataFrame) -> dict:
        """
        Analyze data and generate statistics without modifying it.
        
        Delegates to the pure analyze_data function from preprocessing_utils.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with analysis results
        """
        return _analyze_data(df)
    
    def preprocess(
        self,
        df: pd.DataFrame,
        config: Optional[PreprocessingConfig] = None,
    ) -> tuple[pd.DataFrame, QualityReport]:
        """
        Pre-process a DataFrame according to the configuration.
        
        Uses pure transformation functions from preprocessing_utils.py
        and aggregates results into a QualityReport.
        
        Args:
            df: Input DataFrame
            config: Pre-processing configuration (uses defaults if not provided)
            
        Returns:
            Tuple of (processed DataFrame, quality report)
        """
        start_time = time.time()
        
        if config is None:
            config = PreprocessingConfig()
        
        report = QualityReport()
        report.original_rows = len(df)
        report.original_columns = len(df.columns)
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Step 1: Remove empty rows and columns
        if config.remove_empty_rows:
            df, rows_removed = _remove_empty_rows(df)
            report.empty_rows_removed = rows_removed
        
        if config.remove_empty_columns:
            df, empty_cols = _remove_empty_columns(df)
            report.empty_columns_removed = len(empty_cols)
            report.columns_dropped.extend(empty_cols)
        
        # Step 2: Trim whitespace
        if config.trim_whitespace:
            df = _trim_whitespace(df)
        
        # Step 3: Remove duplicates
        if config.remove_duplicates:
            df, dups_removed = _remove_duplicates(df)
            report.duplicates_removed = dups_removed
        
        # Step 4: Handle missing values
        df, values_handled = _handle_missing_values(
            df, config.handle_missing, config.fill_value
        )
        report.missing_values_handled = values_handled
        
        # Step 5: Convert types
        if config.convert_types:
            df, conversions = _convert_types(df)
            report.type_conversions.update(conversions)
        
        # Step 6: Normalize columns
        if config.normalize_columns:
            df, normalizations = _normalize_columns(df, config.normalize_columns)
            report.type_conversions.update(normalizations)
        
        # Step 7: Filter data
        if config.filter_conditions:
            df, filter_warnings = _filter_data(df, config.filter_conditions)
            report.warnings.extend(filter_warnings)
        
        # Step 8: Select/drop columns
        df, dropped_cols, select_warnings = _select_columns(
            df, config.columns_to_keep, config.columns_to_drop
        )
        report.columns_dropped.extend(dropped_cols)
        report.warnings.extend(select_warnings)
        
        # Generate final statistics
        report.processed_rows = len(df)
        report.processed_columns = len(df.columns)
        report.column_stats = _generate_column_stats(df)
        
        report.processing_time_ms = int((time.time() - start_time) * 1000)
        
        return df, report
    
    def preprocess_and_store(
        self,
        file_id: str,
        df: pd.DataFrame,
        config: Optional[PreprocessingConfig] = None,
    ) -> tuple[str, QualityReport]:
        """
        Pre-process a DataFrame and store the result.
        
        Args:
            file_id: Original file ID
            df: Input DataFrame
            config: Pre-processing configuration
            
        Returns:
            Tuple of (processed_id, quality report)
        """
        processed_df, report = self.preprocess(df, config)
        
        processed_id = str(uuid.uuid4())
        self._processed_store[processed_id] = processed_df
        self._reports[processed_id] = report
        self._metadata[processed_id] = {
            "original_file_id": file_id,
            "processed_at": datetime.utcnow().isoformat(),
            "config": config.__dict__ if config else PreprocessingConfig().__dict__,
        }
        
        return processed_id, report
    
    def get_processed(self, processed_id: str) -> Optional[pd.DataFrame]:
        """Get a processed DataFrame by ID."""
        return self._processed_store.get(processed_id)
    
    def get_report(self, processed_id: str) -> Optional[QualityReport]:
        """Get the quality report for a processed file."""
        return self._reports.get(processed_id)
    
    def get_metadata(self, processed_id: str) -> Optional[dict]:
        """Get metadata for a processed file."""
        return self._metadata.get(processed_id)
    
    def save_processed_to_file(self, processed_id: str, format: str = "xlsx") -> Optional[Path]:
        """
        Save processed data to a file.
        
        Args:
            processed_id: ID of the processed data
            format: Output format ("xlsx", "csv", "json")
            
        Returns:
            Path to the saved file, or None if not found
        """
        df = self._processed_store.get(processed_id)
        if df is None:
            return None
        
        file_path = UPLOAD_DIR / f"processed_{processed_id}.{format}"
        
        if format == "xlsx":
            df.to_excel(file_path, index=False, engine="openpyxl")
        elif format == "csv":
            df.to_csv(file_path, index=False)
        elif format == "json":
            df.to_json(file_path, orient="records", indent=2)
        else:
            return None
        
        return file_path


# Singleton instance
preprocessing_service = PreprocessingService()
