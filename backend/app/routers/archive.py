"""
Archive Router

This module provides API endpoints for archiving and managing uploaded files.
Archives are compressed using gzip and stored with configurable retention policies.

Features:
    - Gzip compression for storage efficiency
    - Configurable retention periods (default: 30 days)
    - Restore functionality to recover archived files
    - Automatic cleanup of expired archives

Endpoints:
    POST /api/archive/{file_id} - Archive an uploaded file
    GET /api/archive - List all archives
    GET /api/archive/{archive_id} - Get archive metadata
    POST /api/archive/{archive_id}/restore - Restore an archived file
    GET /api/archive/{archive_id}/download - Download restored file
    DELETE /api/archive/{archive_id} - Delete an archive
    POST /api/archive/cleanup - Clean up expired archives
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from app.services.file_service import file_service
from app.services.archive_service import archive_service
from app.models.schemas import ErrorResponse

router = APIRouter(prefix="/api/archive", tags=["archive"])


@router.post("/{file_id}")
async def archive_file(file_id: str):
    """
    Archive an uploaded file.
    
    Compresses the file using gzip and stores it in the archive directory.
    The original file remains accessible until explicitly deleted.
    
    Args:
        file_id: Unique identifier of the file to archive
        
    Returns:
        Archive metadata including archive_id, compression ratio, and expiration date.
        
    Raises:
        HTTPException 404: File not found
    """
    metadata = file_service.get_metadata(file_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = Path(metadata.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    archive_metadata = await archive_service.archive_file(
        file_id, file_path, metadata.original_filename
    )

    return {
        "archive_id": archive_metadata.archive_id,
        "original_file_id": archive_metadata.original_file_id,
        "original_filename": archive_metadata.original_filename,
        "file_size": archive_metadata.file_size,
        "compressed_size": archive_metadata.compressed_size,
        "compression_ratio": round(archive_metadata.compressed_size / archive_metadata.file_size, 3),
        "archived_at": archive_metadata.archived_at.isoformat(),
        "expires_at": archive_metadata.expires_at.isoformat(),
    }


@router.get("")
async def list_archives():
    """
    List all archived files.
    
    Returns:
        List of archive metadata objects with count.
    """
    archives = archive_service.list_archives()
    return {"archives": archives, "count": len(archives)}


@router.get("/{archive_id}")
async def get_archive_info(archive_id: str):
    """
    Get metadata for an archived file.
    
    Args:
        archive_id: Unique identifier of the archive
        
    Returns:
        Archive metadata including original filename, sizes, and dates.
        
    Raises:
        HTTPException 404: Archive not found
    """
    metadata = archive_service.get_archive_metadata(archive_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Archive not found")

    return metadata.to_dict()


@router.post("/{archive_id}/restore")
async def restore_archive(archive_id: str):
    """
    Restore an archived file.
    
    Decompresses the archive and makes the file available for download.
    The restored file is stored temporarily.
    
    Args:
        archive_id: Unique identifier of the archive
        
    Returns:
        Restoration status with restored file path.
        
    Raises:
        HTTPException 404: Archive not found
        HTTPException 500: Restoration failed
    """
    metadata = archive_service.get_archive_metadata(archive_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Archive not found")

    restored_path = await archive_service.restore_file(archive_id)
    if not restored_path:
        raise HTTPException(status_code=500, detail="Failed to restore archive")

    return {
        "status": "restored",
        "archive_id": archive_id,
        "restored_path": str(restored_path),
        "original_filename": metadata.original_filename,
    }


@router.get("/{archive_id}/download")
async def download_restored_archive(archive_id: str):
    """
    Download a restored archive file.
    
    Restores the archive if not already restored, then returns the file.
    
    Args:
        archive_id: Unique identifier of the archive
        
    Returns:
        FileResponse with the restored file for download.
        
    Raises:
        HTTPException 404: Archive not found
        HTTPException 500: Restoration failed
    """
    metadata = archive_service.get_archive_metadata(archive_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Archive not found")

    restored_path = await archive_service.restore_file(archive_id)
    if not restored_path:
        raise HTTPException(status_code=500, detail="Failed to restore archive")

    return FileResponse(
        path=restored_path,
        filename=metadata.original_filename,
        media_type="application/octet-stream",
    )


@router.delete("/{archive_id}")
async def delete_archive(archive_id: str):
    """
    Delete an archived file.
    
    Permanently removes the archive from storage.
    
    Args:
        archive_id: Unique identifier of the archive to delete
        
    Returns:
        Confirmation with status and archive_id.
        
    Raises:
        HTTPException 404: Archive not found
    """
    success = await archive_service.delete_archive(archive_id)
    if not success:
        raise HTTPException(status_code=404, detail="Archive not found")

    return {"status": "deleted", "archive_id": archive_id}


@router.post("/cleanup")
async def cleanup_expired_archives():
    """
    Clean up expired archives.
    
    Removes all archives that have passed their retention period.
    Should be called periodically (e.g., via cron job).
    
    Returns:
        Status and count of deleted archives.
    """
    deleted_count = await archive_service.cleanup_expired()
    return {"status": "completed", "deleted_count": deleted_count}
