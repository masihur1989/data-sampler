import uuid
import aiofiles
from pathlib import Path
from datetime import datetime
from fastapi import UploadFile
from typing import Optional

from app.config import UPLOAD_DIR, CHUNK_SIZE, MAX_FILE_SIZE, ALLOWED_EXTENSIONS
from app.models.schemas import FileMetadata


class FileService:
    def __init__(self):
        self._metadata_store: dict[str, FileMetadata] = {}

    def _validate_extension(self, filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in ALLOWED_EXTENSIONS

    async def save_upload(self, file: UploadFile) -> FileMetadata:
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
        return self._metadata_store.get(file_id)

    def update_metadata(self, file_id: str, **kwargs) -> Optional[FileMetadata]:
        if file_id not in self._metadata_store:
            return None
        metadata = self._metadata_store[file_id]
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        return metadata

    def get_file_path(self, file_id: str) -> Optional[Path]:
        metadata = self.get_metadata(file_id)
        if metadata:
            return Path(metadata.file_path)
        return None

    async def delete_file(self, file_id: str) -> bool:
        metadata = self._metadata_store.pop(file_id, None)
        if metadata:
            file_path = Path(metadata.file_path)
            if file_path.exists():
                await aiofiles.os.remove(file_path)
            return True
        return False


file_service = FileService()
