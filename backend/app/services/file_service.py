"""
File Service Module

This module provides file management capabilities for uploaded Excel files.
It handles file storage, metadata tracking, and cleanup operations.

Features:
    - Chunked file uploads for memory-efficient handling of large files
    - File size validation (configurable maximum, default 200MB)
    - Extension validation (.xlsx, .xls, .csv)
    - Metadata storage and retrieval
    - Async file operations for non-blocking I/O

The service uses an in-memory metadata store for simplicity. In production,
this could be replaced with a database-backed store.
"""

import uuid
import aiofiles
from pathlib import Path
from datetime import datetime
from fastapi import UploadFile
from typing import Optional

from app.config import UPLOAD_DIR, CHUNK_SIZE, MAX_FILE_SIZE, ALLOWED_EXTENSIONS
from app.models.schemas import FileMetadata


class FileService:
    """
    Service for managing uploaded files and their metadata.
    
    This service handles the complete lifecycle of uploaded files:
    1. Validation (extension, size)
    2. Storage (chunked writing for large files)
    3. Metadata tracking (file info, upload time)
    4. Cleanup (deletion)
    
    Attributes:
        _metadata_store: In-memory dictionary mapping file_id to FileMetadata
    """
    def __init__(self):
        """Initialize the file service with an empty metadata store."""
        self._metadata_store: dict[str, FileMetadata] = {}

    def _validate_extension(self, filename: str) -> bool:
        """
        Check if the file extension is allowed.
        
        Args:
            filename: Name of the file to validate
            
        Returns:
            True if extension is allowed, False otherwise
        """
        ext = Path(filename).suffix.lower()
        return ext in ALLOWED_EXTENSIONS

    async def save_upload(self, file: UploadFile) -> FileMetadata:
        """
        Save an uploaded file to disk with chunked writing.
        
        Reads the file in chunks (default 64KB) to handle large files
        without consuming excessive memory. Validates file extension
        and size during upload.
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            FileMetadata with file_id, path, size, and upload time
            
        Raises:
            ValueError: If file extension is not allowed or file is too large
        """
        if not self._validate_extension(file.filename or ""):
            raise ValueError(f"Invalid file extension. Allowed: {ALLOWED_EXTENSIONS}")

        file_id = str(uuid.uuid4())
        ext = Path(file.filename or "file.xlsx").suffix.lower()
        file_path = UPLOAD_DIR / f"{file_id}{ext}"

        total_size = 0
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    await aiofiles.os.remove(file_path)
                    raise ValueError(f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB")
                await f.write(chunk)

        metadata = FileMetadata(
            file_id=file_id,
            original_filename=file.filename or "unknown",
            file_size=total_size,
            file_path=str(file_path),
            upload_time=datetime.utcnow(),
        )
        self._metadata_store[file_id] = metadata
        return metadata

    def get_metadata(self, file_id: str) -> Optional[FileMetadata]:
        """
        Get metadata for an uploaded file.
        
        Args:
            file_id: Unique identifier of the file
            
        Returns:
            FileMetadata if found, None otherwise
        """
        return self._metadata_store.get(file_id)

    def update_metadata(self, file_id: str, **kwargs) -> Optional[FileMetadata]:
        """
        Update metadata for an uploaded file.
        
        Args:
            file_id: Unique identifier of the file
            **kwargs: Metadata fields to update (row_count, column_count, etc.)
            
        Returns:
            Updated FileMetadata if found, None otherwise
        """
        if file_id not in self._metadata_store:
            return None
        metadata = self._metadata_store[file_id]
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        return metadata

    def get_file_path(self, file_id: str) -> Optional[Path]:
        """
        Get the file path for an uploaded file.
        
        Args:
            file_id: Unique identifier of the file
            
        Returns:
            Path object if found, None otherwise
        """
        metadata = self.get_metadata(file_id)
        if metadata:
            return Path(metadata.file_path)
        return None

    async def delete_file(self, file_id: str) -> bool:
        """
        Delete an uploaded file and its metadata.
        
        Args:
            file_id: Unique identifier of the file to delete
            
        Returns:
            True if file was deleted, False if not found
        """
        metadata = self._metadata_store.pop(file_id, None)
        if metadata:
            file_path = Path(metadata.file_path)
            if file_path.exists():
                await aiofiles.os.remove(file_path)
            return True
        return False


file_service = FileService()
