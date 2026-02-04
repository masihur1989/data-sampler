"""
Pre-processing Utility Functions

This module contains pure transformation functions for data pre-processing.
These functions are stateless and can be used independently or composed together.

Each function takes a DataFrame and returns a transformed DataFrame along with
metadata about the transformation (counts, warnings, etc.).
"""

from typing import Optional
import pandas as pd
import numpy as np


def analyze_data(df: pd.DataFrame) -> dict:
    """
    Analyze a DataFrame and generate statistics without modifying it.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with analysis results including row/column counts,
        data types, missing values, duplicates, and column statistics.
    """
    analysis = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict() if len(df) > 0 else {},
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        analysis["numeric_stats"] = df[numeric_cols].describe().to_dict()
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        analysis["categorical_stats"] = {}
        for col in categorical_cols:
            analysis["categorical_stats"][col] = {
                "unique_values": int(df[col].nunique()),
                "top_values": df[col].value_counts().head(5).to_dict(),
            }
    
    return analysis


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (DataFrame with duplicates removed, count of duplicates removed)
    """
    original_count = len(df)
    df = df.drop_duplicates()
    duplicates_removed = original_count - len(df)
    return df, duplicates_removed


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "keep",
    fill_value: Optional[str] = None
) -> tuple[pd.DataFrame, int]:
    """
    Handle missing values based on the specified strategy.
    
    Args:
        df: Input DataFrame
        strategy: One of "keep", "drop", "fill_mean", "fill_median", "fill_mode", "fill_value"
        fill_value: Value to use when strategy is "fill_value"
        
    Returns:
        Tuple of (DataFrame with missing values handled, count of values handled)
    """
    missing_before = df.isnull().sum().sum()
    
    if strategy == "keep":
        pass
    elif strategy == "drop":
        df = df.dropna()
    elif strategy == "fill_mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df.copy()
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "fill_median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df.copy()
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == "fill_mode":
        df = df.copy()
        for col in df.columns:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
    elif strategy == "fill_value" and fill_value is not None:
        df = df.fillna(fill_value)
    
    missing_after = df.isnull().sum().sum()
    values_handled = missing_before - missing_after
    
    return df, int(values_handled)


def trim_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim leading and trailing whitespace from string columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with whitespace trimmed from string columns
    """
    df = df.copy()
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def remove_empty_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove rows where all values are empty/null.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (DataFrame with empty rows removed, count of rows removed)
    """
    original_count = len(df)
    df = df.dropna(how='all')
    rows_removed = original_count - len(df)
    return df, rows_removed


def remove_empty_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove columns where all values are empty/null.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (DataFrame with empty columns removed, list of removed column names)
    """
    empty_cols = df.columns[df.isnull().all()].tolist()
    df = df.dropna(axis=1, how='all')
    return df, empty_cols


def convert_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Attempt to convert string columns to numeric types.
    
    Only converts columns where more than 50% of values are valid numbers.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (DataFrame with converted types, dict of column -> new type)
    """
    df = df.copy()
    conversions = {}
    
    for col in df.select_dtypes(include=['object']).columns:
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() / len(converted) > 0.5:
                df[col] = converted
                conversions[col] = "numeric"
        except (ValueError, TypeError):
            pass
    
    return df, conversions


def normalize_columns(
    df: pd.DataFrame,
    columns: list[str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Normalize specified columns to 0-1 scale using min-max normalization.
    
    Args:
        df: Input DataFrame
        columns: List of column names to normalize
        
    Returns:
        Tuple of (DataFrame with normalized columns, dict of column -> normalization info)
    """
    df = df.copy()
    normalizations = {}
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64, float, int]:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
                normalizations[col] = f"normalized (min={min_val}, max={max_val})"
    
    return df, normalizations


def filter_data(
    df: pd.DataFrame,
    conditions: dict
) -> tuple[pd.DataFrame, list[str]]:
    """
    Filter data based on conditions.
    
    Conditions format:
    {
        "column_name": {"op": "gt", "value": 10},  # greater than
        "column_name": {"op": "eq", "value": "A"},  # equals
        "column_name": {"op": "in", "value": ["A", "B"]},  # in list
    }
    
    Supported operators: eq, ne, gt, gte, lt, lte, in, not_in, contains
    
    Args:
        df: Input DataFrame
        conditions: Dictionary of column -> condition
        
    Returns:
        Tuple of (filtered DataFrame, list of warnings)
    """
    warnings = []
    original_count = len(df)
    
    for col, condition in conditions.items():
        if col not in df.columns:
            warnings.append(f"Filter column '{col}' not found")
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
        warnings.append(f"Filtered out {filtered_count} rows based on conditions")
    
    return df, warnings


def select_columns(
    df: pd.DataFrame,
    columns_to_keep: Optional[list[str]] = None,
    columns_to_drop: Optional[list[str]] = None
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Select or drop columns from a DataFrame.
    
    Args:
        df: Input DataFrame
        columns_to_keep: If specified, only keep these columns
        columns_to_drop: Columns to drop
        
    Returns:
        Tuple of (DataFrame with selected columns, list of dropped columns, list of warnings)
    """
    dropped_cols = []
    warnings = []
    
    if columns_to_keep:
        missing_cols = [c for c in columns_to_keep if c not in df.columns]
        if missing_cols:
            warnings.append(f"Columns not found: {missing_cols}")
        keep = [c for c in columns_to_keep if c in df.columns]
        dropped_cols = [c for c in df.columns if c not in keep]
        df = df[keep]
    
    if columns_to_drop:
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
                dropped_cols.append(col)
    
    return df, dropped_cols, warnings


def generate_column_stats(df: pd.DataFrame) -> dict[str, dict]:
    """
    Generate statistics for each column in the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping column names to their statistics
    """
    column_stats = {}
    
    for col in df.columns:
        stats = {
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].notna().sum()),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique()),
        }
        
        if df[col].dtype in [np.float64, np.int64, float, int]:
            stats.update({
                "min": float(df[col].min()) if pd.notna(df[col].min()) else None,
                "max": float(df[col].max()) if pd.notna(df[col].max()) else None,
                "mean": float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                "std": float(df[col].std()) if pd.notna(df[col].std()) else None,
            })
        
        column_stats[col] = stats
    
    return column_stats
