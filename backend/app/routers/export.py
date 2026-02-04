"""
Export Router

This module provides API endpoints for exporting sampled data to various formats.
Supports both file download and streaming export for large datasets.

Supported Formats:
    - XLSX: Excel spreadsheet (default)
    - CSV: Comma-separated values
    - JSON: JSON array of records

Endpoints:
    GET /api/export/{sample_id} - Download sample as file
    GET /api/export/{sample_id}/stream/csv - Stream as CSV
    GET /api/export/{sample_id}/stream/json - Stream as NDJSON

Streaming exports are useful for large samples where generating the entire
file at once would consume too much memory.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path

from app.services.sampler_service import sampler_service
from app.services.export_service import export_service
from app.models.schemas import ExportFormat, ErrorResponse

router = APIRouter(prefix="/api/export", tags=["export"])


@router.get(
    "/{sample_id}",
    responses={404: {"model": ErrorResponse}},
)
async def export_sample(sample_id: str, format: str = "xlsx"):
    """
    Export a sample as a downloadable file.
    
    Args:
        sample_id: Unique identifier of the sample
        format: Output format - xlsx, csv, or json (default: xlsx)
        
    Returns:
        FileResponse with the exported file for download.
        
    Raises:
        HTTPException 400: Invalid format specified
        HTTPException 404: Sample not found
    """
    df = sampler_service.get_sample(sample_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    try:
        export_format = ExportFormat(format.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format. Supported: {[f.value for f in ExportFormat]}",
        )

    file_path = export_service.export_to_file(df, export_format, f"sample_{sample_id[:8]}")

    media_types = {
        ExportFormat.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ExportFormat.CSV: "text/csv",
        ExportFormat.JSON: "application/json",
    }

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type=media_types[export_format],
    )


@router.get("/{sample_id}/stream/csv")
async def stream_csv(sample_id: str, chunk_size: int = 1000):
    """
    Stream sample data as CSV.
    
    Generates CSV output in chunks for memory-efficient export of large samples.
    
    Args:
        sample_id: Unique identifier of the sample
        chunk_size: Number of rows per chunk (default: 1000)
        
    Returns:
        StreamingResponse with CSV content and download headers.
        
    Raises:
        HTTPException 404: Sample not found
    """
    df = sampler_service.get_sample(sample_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    return StreamingResponse(
        export_service.stream_as_csv(df, chunk_size),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=sample_{sample_id[:8]}.csv",
        },
    )


@router.get("/{sample_id}/stream/json")
async def stream_json(sample_id: str, chunk_size: int = 1000):
    """
    Stream sample data as newline-delimited JSON (NDJSON).
    
    Each line contains one JSON object representing a row. This format is
    ideal for streaming large datasets as it can be parsed incrementally.
    
    Args:
        sample_id: Unique identifier of the sample
        chunk_size: Number of rows per chunk (default: 1000)
        
    Returns:
        StreamingResponse with NDJSON content and download headers.
        
    Raises:
        HTTPException 404: Sample not found
    """
    df = sampler_service.get_sample(sample_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    return StreamingResponse(
        export_service.stream_as_json(df, chunk_size),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f"attachment; filename=sample_{sample_id[:8]}.ndjson",
        },
    )
