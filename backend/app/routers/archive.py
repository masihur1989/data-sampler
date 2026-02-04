from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

from app.services.file_service import file_service
from app.services.archive_service import archive_service
from app.models.schemas import ErrorResponse

router = APIRouter(prefix="/api/archive", tags=["archive"])


@router.post("/{file_id}")
async def archive_file(file_id: str):
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
    archives = archive_service.list_archives()
    return {"archives": archives, "count": len(archives)}


@router.get("/{archive_id}")
async def get_archive_info(archive_id: str):
    metadata = archive_service.get_archive_metadata(archive_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Archive not found")

    return metadata.to_dict()


@router.post("/{archive_id}/restore")
async def restore_archive(archive_id: str):
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
    success = await archive_service.delete_archive(archive_id)
    if not success:
        raise HTTPException(status_code=404, detail="Archive not found")

    return {"status": "deleted", "archive_id": archive_id}


@router.post("/cleanup")
async def cleanup_expired_archives():
    deleted_count = await archive_service.cleanup_expired()
    return {"status": "completed", "deleted_count": deleted_count}
