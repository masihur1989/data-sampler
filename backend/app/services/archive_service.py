"""
Archive Service Module

This module provides file archival capabilities for uploaded Excel files.
Archives are compressed using gzip and stored with configurable retention policies.

Features:
    - Gzip compression (level 6) for storage efficiency
    - Configurable retention periods (default: 30 days)
    - Restore functionality to recover archived files
    - Automatic cleanup of expired archives
    - Metadata tracking for all archived files

The archival process:
1. Compress the original file using gzip
2. Store in the archive directory
3. Track metadata (original file info, compression ratio, expiration)
4. Original file can be deleted after archival

Restoration:
1. Decompress the archive to a temporary location
2. File is available for download or re-processing
"""

import uuid
import gzip
import shutil
import aiofiles
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import json

from app.config import ARCHIVE_DIR, UPLOAD_DIR, ARCHIVE_RETENTION_DAYS


class ArchiveMetadata:
    """
    Metadata for an archived file.
    
    Tracks information about the original file, compression results,
    and retention policy.
    
    Attributes:
        archive_id: Unique identifier for the archive
        original_file_id: ID of the original uploaded file
        original_filename: Original filename
        archived_at: Timestamp when file was archived
        file_size: Original file size in bytes
        compressed_size: Compressed file size in bytes
        archive_path: Path to the compressed archive
        retention_days: Number of days to retain the archive
        expires_at: Timestamp when archive will expire
    """
    def __init__(
        self,
        archive_id: str,
        original_file_id: str,
        original_filename: str,
        archived_at: datetime,
        file_size: int,
        compressed_size: int,
        archive_path: str,
    ):
        """
        Initialize archive metadata.
        
        Args:
            archive_id: Unique identifier for the archive
            original_file_id: ID of the original uploaded file
            original_filename: Original filename
            archived_at: Timestamp when file was archived
            file_size: Original file size in bytes
            compressed_size: Compressed file size in bytes
            archive_path: Path to the compressed archive
        """
        self.archive_id = archive_id
        self.original_file_id = original_file_id
        self.original_filename = original_filename
        self.archived_at = archived_at
        self.file_size = file_size
        self.compressed_size = compressed_size
        self.archive_path = archive_path
        self.retention_days = ARCHIVE_RETENTION_DAYS
        self.expires_at = archived_at + timedelta(days=ARCHIVE_RETENTION_DAYS)

    def to_dict(self) -> dict:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            "archive_id": self.archive_id,
            "original_file_id": self.original_file_id,
            "original_filename": self.original_filename,
            "archived_at": self.archived_at.isoformat(),
            "file_size": self.file_size,
            "compressed_size": self.compressed_size,
            "archive_path": self.archive_path,
            "retention_days": self.retention_days,
            "expires_at": self.expires_at.isoformat(),
        }


class ArchiveService:
    """
    Service for archiving and managing uploaded files.
    
    Provides compression, storage, restoration, and cleanup of archived files.
    Uses gzip compression for storage efficiency.
    
    Attributes:
        _archive_store: Mapping of archive_id to ArchiveMetadata
        _file_to_archive: Mapping of original file_id to archive_id
    """
    
    def __init__(self):
        """Initialize the archive service with empty stores."""
        self._archive_store: dict[str, ArchiveMetadata] = {}
        self._file_to_archive: dict[str, str] = {}

    async def archive_file(
        self,
        file_id: str,
        file_path: Path,
        original_filename: str,
    ) -> ArchiveMetadata:
        """
        Archive a file by compressing it with gzip.
        
        Args:
            file_id: Original file ID
            file_path: Path to the file to archive
            original_filename: Original filename for restoration
            
        Returns:
            ArchiveMetadata with archive details and compression ratio
        """
        archive_id = str(uuid.uuid4())
        archive_path = ARCHIVE_DIR / f"{archive_id}.gz"

        original_size = file_path.stat().st_size

        with open(file_path, "rb") as f_in:
            with gzip.open(archive_path, "wb", compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out)

        compressed_size = archive_path.stat().st_size

        metadata = ArchiveMetadata(
            archive_id=archive_id,
            original_file_id=file_id,
            original_filename=original_filename,
            archived_at=datetime.utcnow(),
            file_size=original_size,
            compressed_size=compressed_size,
            archive_path=str(archive_path),
        )

        self._archive_store[archive_id] = metadata
        self._file_to_archive[file_id] = archive_id

        return metadata

    def get_archive_metadata(self, archive_id: str) -> Optional[ArchiveMetadata]:
        """Get metadata for an archive by ID."""
        return self._archive_store.get(archive_id)

    def get_archive_by_file_id(self, file_id: str) -> Optional[ArchiveMetadata]:
        """Get archive metadata by original file ID."""
        archive_id = self._file_to_archive.get(file_id)
        if archive_id:
            return self._archive_store.get(archive_id)
        return None

    async def restore_file(self, archive_id: str, restore_path: Optional[Path] = None) -> Optional[Path]:
        """
        Restore an archived file by decompressing it.
        
        Args:
            archive_id: ID of the archive to restore
            restore_path: Optional path for the restored file
            
        Returns:
            Path to the restored file, or None if archive not found
        """
        metadata = self._archive_store.get(archive_id)
        if not metadata:
            return None

        archive_path = Path(metadata.archive_path)
        if not archive_path.exists():
            return None

        if restore_path is None:
            restore_path = UPLOAD_DIR / f"restored_{metadata.original_file_id}_{Path(metadata.original_filename).suffix}"

        with gzip.open(archive_path, "rb") as f_in:
            with open(restore_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        return restore_path

    def list_archives(self) -> list[dict]:
        """List all archives as dictionaries."""
        return [metadata.to_dict() for metadata in self._archive_store.values()]

    async def delete_archive(self, archive_id: str) -> bool:
        """
        Delete an archive and its compressed file.
        
        Args:
            archive_id: ID of the archive to delete
            
        Returns:
            True if deleted, False if not found
        """
        metadata = self._archive_store.pop(archive_id, None)
        if metadata:
            archive_path = Path(metadata.archive_path)
            if archive_path.exists():
                archive_path.unlink()
            self._file_to_archive.pop(metadata.original_file_id, None)
            return True
        return False

    async def cleanup_expired(self) -> int:
        """
        Clean up expired archives.
        
        Removes all archives that have passed their retention period.
        
        Returns:
            Number of archives deleted
        """
        now = datetime.utcnow()
        expired = [
            archive_id
            for archive_id, metadata in self._archive_store.items()
            if metadata.expires_at < now
        ]

        for archive_id in expired:
            await self.delete_archive(archive_id)

        return len(expired)


archive_service = ArchiveService()
