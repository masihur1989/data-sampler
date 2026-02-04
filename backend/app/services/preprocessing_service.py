"""
Pre-processing Service Module

This module provides data pre-processing capabilities for raw Excel files before sampling.
It includes data cleaning, validation, transformation, and quality reporting features.

Pre-processing Steps:
1. Data Cleaning: Remove duplicates, handle missing values, trim whitespace
2. Data Validation: Check data types, validate ranges, detect anomalies
3. Data Transformation: Type conversion, normalization, filtering
4. Quality Report: Generate statistics about data quality

The pre-processing pipeline is configurable and can be customized per file.
"""

import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime
import pandas as pd
import numpy as np

from app.config import UPLOAD_DIR


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
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        }
        
        # Add numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis["numeric_stats"] = df[numeric_cols].describe().to_dict()
        
        # Add categorical column statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            analysis["categorical_stats"] = {}
            for col in categorical_cols:
                analysis["categorical_stats"][col] = {
                    "unique_values": df[col].nunique(),
                    "top_values": df[col].value_counts().head(5).to_dict(),
                }
        
        return analysis
    
    def _remove_duplicates(self, df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
        """Remove duplicate rows."""
        original_count = len(df)
        df = df.drop_duplicates()
        report.duplicates_removed = original_count - len(df)
        return df
    
    def _handle_missing_values(
        self, df: pd.DataFrame, strategy: str, fill_value: Optional[str], report: QualityReport
    ) -> pd.DataFrame:
        """Handle missing values based on strategy."""
        missing_count = df.isnull().sum().sum()
        
        if strategy == "keep":
            pass  # Do nothing
        elif strategy == "drop":
            df = df.dropna()
        elif strategy == "fill_mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == "fill_median":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == "fill_mode":
            for col in df.columns:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
        elif strategy == "fill_value" and fill_value is not None:
            df = df.fillna(fill_value)
        
        report.missing_values_handled = missing_count - df.isnull().sum().sum()
        return df
    
    def _trim_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trim whitespace from string columns."""
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        return df
    
    def _remove_empty_rows(self, df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
        """Remove rows where all values are empty/null."""
        original_count = len(df)
        df = df.dropna(how='all')
        report.empty_rows_removed = original_count - len(df)
        return df
    
    def _remove_empty_columns(self, df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
        """Remove columns where all values are empty/null."""
        empty_cols = df.columns[df.isnull().all()].tolist()
        df = df.dropna(axis=1, how='all')
        report.empty_columns_removed = len(empty_cols)
        report.columns_dropped.extend(empty_cols)
        return df
    
    def _convert_types(self, df: pd.DataFrame, report: QualityReport) -> pd.DataFrame:
        """Attempt to convert string columns to numeric types."""
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Try to convert to numeric
                converted = pd.to_numeric(df[col], errors='coerce')
                # Only convert if most values are valid numbers
                if converted.notna().sum() / len(converted) > 0.5:
                    df[col] = converted
                    report.type_conversions[col] = "numeric"
            except (ValueError, TypeError):
                pass
        return df
    
    def _normalize_columns(self, df: pd.DataFrame, columns: list[str], report: QualityReport) -> pd.DataFrame:
        """Normalize specified columns to 0-1 scale."""
        for col in columns:
            if col in df.columns and df[col].dtype in [np.float64, np.int64, float, int]:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                    report.type_conversions[col] = f"normalized (min={min_val}, max={max_val})"
        return df
    
    def _filter_data(self, df: pd.DataFrame, conditions: dict, report: QualityReport) -> pd.DataFrame:
        """
        Filter data based on conditions.
        
        Conditions format:
        {
            "column_name": {"op": "gt", "value": 10},  # greater than
            "column_name": {"op": "eq", "value": "A"},  # equals
            "column_name": {"op": "in", "value": ["A", "B"]},  # in list
        }
        """
        original_count = len(df)
        
        for col, condition in conditions.items():
            if col not in df.columns:
                report.warnings.append(f"Filter column '{col}' not found")
                continue
            
            op = condition.get("op", "eq")
            value = condition.get("value")
            
            if op == "eq":
                df = df[df[col] == value]
            elif op == "ne":
                df = df[df[col] != value]
            elif op == "gt":
                df = df[df[col] > value]
            elif op == "gte":
                df = df[df[col] >= value]
            elif op == "lt":
                df = df[df[col] < value]
            elif op == "lte":
                df = df[df[col] <= value]
            elif op == "in":
                df = df[df[col].isin(value)]
            elif op == "not_in":
                df = df[~df[col].isin(value)]
            elif op == "contains":
                df = df[df[col].astype(str).str.contains(str(value), na=False)]
        
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            report.warnings.append(f"Filtered out {filtered_count} rows based on conditions")
        
        return df
    
    def _select_columns(
        self, df: pd.DataFrame, keep: Optional[list[str]], drop: list[str], report: QualityReport
    ) -> pd.DataFrame:
        """Select or drop columns."""
        if keep:
            missing_cols = [c for c in keep if c not in df.columns]
            if missing_cols:
                report.warnings.append(f"Columns not found: {missing_cols}")
            keep = [c for c in keep if c in df.columns]
            df = df[keep]
        
        for col in drop:
            if col in df.columns:
                df = df.drop(columns=[col])
                report.columns_dropped.append(col)
        
        return df
    
    def _generate_column_stats(self, df: pd.DataFrame, report: QualityReport) -> None:
        """Generate statistics for each column."""
        for col in df.columns:
            stats = {
                "dtype": str(df[col].dtype),
                "non_null_count": df[col].notna().sum(),
                "null_count": df[col].isnull().sum(),
                "unique_count": df[col].nunique(),
            }
            
            if df[col].dtype in [np.float64, np.int64, float, int]:
                stats.update({
                    "min": float(df[col].min()) if pd.notna(df[col].min()) else None,
                    "max": float(df[col].max()) if pd.notna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                    "std": float(df[col].std()) if pd.notna(df[col].std()) else None,
                })
            
            report.column_stats[col] = stats
    
    def preprocess(
        self,
        df: pd.DataFrame,
        config: Optional[PreprocessingConfig] = None,
    ) -> tuple[pd.DataFrame, QualityReport]:
        """
        Pre-process a DataFrame according to the configuration.
        
        Args:
            df: Input DataFrame
            config: Pre-processing configuration (uses defaults if not provided)
            
        Returns:
            Tuple of (processed DataFrame, quality report)
        """
        import time
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
            df = self._remove_empty_rows(df, report)
        
        if config.remove_empty_columns:
            df = self._remove_empty_columns(df, report)
        
        # Step 2: Trim whitespace
        if config.trim_whitespace:
            df = self._trim_whitespace(df)
        
        # Step 3: Remove duplicates
        if config.remove_duplicates:
            df = self._remove_duplicates(df, report)
        
        # Step 4: Handle missing values
        df = self._handle_missing_values(df, config.handle_missing, config.fill_value, report)
        
        # Step 5: Convert types
        if config.convert_types:
            df = self._convert_types(df, report)
        
        # Step 6: Normalize columns
        if config.normalize_columns:
            df = self._normalize_columns(df, config.normalize_columns, report)
        
        # Step 7: Filter data
        if config.filter_conditions:
            df = self._filter_data(df, config.filter_conditions, report)
        
        # Step 8: Select/drop columns
        df = self._select_columns(df, config.columns_to_keep, config.columns_to_drop, report)
        
        # Generate final statistics
        report.processed_rows = len(df)
        report.processed_columns = len(df.columns)
        self._generate_column_stats(df, report)
        
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
