"""
File Upload Router

This module provides API endpoints for uploading and managing Excel files.
It handles chunked file uploads for large files (up to 150MB) and extracts
metadata including row counts, column information, and sheet names.

Endpoints:
    POST /api/upload - Upload an Excel file
    GET /api/upload/{file_id} - Get file metadata
    DELETE /api/upload/{file_id} - Delete an uploaded file

The upload process:
1. Receives the file via multipart form data
2. Saves the file to temporary storage using FileService
3. Parses the Excel file to extract metadata using ExcelParser
4. Returns file information including row/column counts and sheet names
"""

import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path

from app.services.file_service import file_service
from app.services.parser_service import ExcelParser
from app.models.schemas import FileUploadResponse, ErrorResponse

router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post(
    "",
    response_model=FileUploadResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload an Excel file for processing.
    
    Accepts .xlsx, .xls, and .csv files up to 150MB. The file is saved to
    temporary storage and parsed to extract metadata.
    
    Args:
        file: The uploaded file (multipart form data)
        
    Returns:
        FileUploadResponse with file_id, filename, size, row/column counts,
        column names, and available sheet names.
        
    Raises:
        HTTPException 400: Invalid file format or corrupted file
        HTTPException 413: File too large (>150MB)
        HTTPException 500: Server error during processing
    """
    try:
        metadata = await file_service.save_upload(file)

        parser = ExcelParser(Path(metadata.file_path))
        file_info = parser.get_file_info()

        file_service.update_metadata(
            metadata.file_id,
            row_count=file_info["row_count"],
            column_count=file_info["column_count"],
            columns=file_info["columns"],
            sheet_names=file_info["sheet_names"],
        )

        return FileUploadResponse(
            file_id=metadata.file_id,
            filename=metadata.original_filename,
            size=metadata.file_size,
            row_count=file_info["row_count"],
            column_count=file_info["column_count"],
            columns=file_info["columns"],
            sheet_names=file_info["sheet_names"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@router.get("/{file_id}")
async def get_file_info(file_id: str):
    """
    Get metadata for an uploaded file.
    
    Args:
        file_id: Unique identifier returned from upload
        
    Returns:
        File metadata including filename, size, row/column counts,
        column names, sheet names, and upload timestamp.
        
    Raises:
        HTTPException 404: File not found
    """
    metadata = file_service.get_metadata(file_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    return {
        "file_id": metadata.file_id,
        "filename": metadata.original_filename,
        "size": metadata.file_size,
        "row_count": metadata.row_count,
        "column_count": metadata.column_count,
        "columns": metadata.columns,
        "sheet_names": metadata.sheet_names,
        "upload_time": metadata.upload_time.isoformat(),
    }


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """
    Delete an uploaded file.
    
    Removes the file from temporary storage and clears its metadata.
    
    Args:
        file_id: Unique identifier of the file to delete
        
    Returns:
        Confirmation with status and file_id
        
    Raises:
        HTTPException 404: File not found
    """
    success = await file_service.delete_file(file_id)
    if not success:
        raise HTTPException(status_code=404, detail="File not found")
    return {"status": "deleted", "file_id": file_id}
