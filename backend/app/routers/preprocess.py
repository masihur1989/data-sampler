"""
Pre-processing API Router

Provides endpoints for pre-processing raw Excel files before sampling.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path

from app.services.file_service import file_service
from app.services.parser_service import ExcelParser
from app.services.preprocessing_service import (
    preprocessing_service,
    PreprocessingConfig,
)

router = APIRouter(prefix="/api/preprocess", tags=["preprocess"])


class FilterCondition(BaseModel):
    """Filter condition for a column."""
    op: str = Field(..., description="Operator: eq, ne, gt, gte, lt, lte, in, not_in, contains")
    value: str | int | float | list = Field(..., description="Value to compare against")


class PreprocessRequest(BaseModel):
    """Request body for pre-processing."""
    file_id: str = Field(..., description="ID of the uploaded file")
    sheet_name: Optional[str] = Field(None, description="Sheet to process")
    remove_duplicates: bool = Field(True, description="Remove duplicate rows")
    handle_missing: str = Field(
        "keep",
        description="Missing value strategy: keep, drop, fill_mean, fill_median, fill_mode, fill_value"
    )
    fill_value: Optional[str] = Field(None, description="Value for fill_value strategy")
    trim_whitespace: bool = Field(True, description="Trim whitespace from strings")
    remove_empty_rows: bool = Field(True, description="Remove rows with all empty values")
    remove_empty_columns: bool = Field(True, description="Remove columns with all empty values")
    convert_types: bool = Field(False, description="Auto-convert string columns to numeric")
    normalize_columns: Optional[list[str]] = Field(None, description="Columns to normalize (0-1)")
    filter_conditions: Optional[dict[str, FilterCondition]] = Field(None, description="Filter conditions")
    columns_to_keep: Optional[list[str]] = Field(None, description="Only keep these columns")
    columns_to_drop: Optional[list[str]] = Field(None, description="Drop these columns")


class AnalyzeRequest(BaseModel):
    """Request body for data analysis."""
    file_id: str = Field(..., description="ID of the uploaded file")
    sheet_name: Optional[str] = Field(None, description="Sheet to analyze")


@router.post("/analyze")
async def analyze_file(request: AnalyzeRequest):
    """
    Analyze a file and return data quality statistics without modifying it.
    
    This endpoint provides insights into the data including:
    - Row and column counts
    - Data types for each column
    - Missing value statistics
    - Duplicate row count
    - Numeric column statistics (min, max, mean, std)
    - Categorical column statistics (unique values, top values)
    """
    file_path = file_service.get_file_path(request.file_id)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        parser = ExcelParser(file_path)
        df = parser.read_all(request.sheet_name)
        analysis = preprocessing_service.analyze_data(df)
        
        return {
            "file_id": request.file_id,
            "sheet_name": request.sheet_name,
            "analysis": analysis,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("")
async def preprocess_file(request: PreprocessRequest):
    """
    Pre-process a file and store the result for sampling.
    
    The pre-processing pipeline includes:
    1. Remove empty rows and columns
    2. Trim whitespace from string values
    3. Remove duplicate rows
    4. Handle missing values (keep, drop, or fill)
    5. Convert data types (optional)
    6. Normalize numeric columns (optional)
    7. Filter data based on conditions (optional)
    8. Select or drop columns (optional)
    
    Returns a processed_id that can be used for sampling.
    """
    file_path = file_service.get_file_path(request.file_id)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Read the file
        parser = ExcelParser(file_path)
        df = parser.read_all(request.sheet_name)
        
        # Convert filter conditions to the expected format
        filter_conditions = None
        if request.filter_conditions:
            filter_conditions = {
                col: {"op": cond.op, "value": cond.value}
                for col, cond in request.filter_conditions.items()
            }
        
        # Create config
        config = PreprocessingConfig(
            remove_duplicates=request.remove_duplicates,
            handle_missing=request.handle_missing,
            fill_value=request.fill_value,
            trim_whitespace=request.trim_whitespace,
            remove_empty_rows=request.remove_empty_rows,
            remove_empty_columns=request.remove_empty_columns,
            convert_types=request.convert_types,
            normalize_columns=request.normalize_columns,
            filter_conditions=filter_conditions,
            columns_to_keep=request.columns_to_keep,
            columns_to_drop=request.columns_to_drop,
        )
        
        # Pre-process and store
        processed_id, report = preprocessing_service.preprocess_and_store(
            request.file_id, df, config
        )
        
        return {
            "processed_id": processed_id,
            "original_file_id": request.file_id,
            "report": report.to_dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pre-processing failed: {str(e)}")


@router.get("/{processed_id}")
async def get_processed_info(processed_id: str):
    """Get information about a processed file."""
    df = preprocessing_service.get_processed(processed_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Processed data not found")
    
    report = preprocessing_service.get_report(processed_id)
    metadata = preprocessing_service.get_metadata(processed_id)
    
    return {
        "processed_id": processed_id,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "report": report.to_dict() if report else None,
        "metadata": metadata,
        "preview": df.head(10).to_dict(orient="records"),
    }


@router.get("/{processed_id}/preview")
async def preview_processed(processed_id: str, rows: int = 100):
    """Preview the processed data."""
    df = preprocessing_service.get_processed(processed_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Processed data not found")
    
    preview_rows = min(rows, len(df))
    return {
        "processed_id": processed_id,
        "total_rows": len(df),
        "preview_rows": preview_rows,
        "columns": list(df.columns),
        "data": df.head(preview_rows).to_dict(orient="records"),
    }


@router.post("/{processed_id}/sample")
async def sample_from_processed(processed_id: str):
    """
    Get the processed DataFrame for sampling.
    
    This endpoint returns the processed_id which can be used with
    the /api/sample endpoint by setting use_processed=true.
    """
    df = preprocessing_service.get_processed(processed_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Processed data not found")
    
    return {
        "processed_id": processed_id,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "message": "Use this processed_id with /api/sample endpoint",
    }
