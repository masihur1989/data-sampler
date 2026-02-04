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
    def __init__(self):
        self._archive_store: dict[str, ArchiveMetadata] = {}
        self._file_to_archive: dict[str, str] = {}

    async def archive_file(
        self,
        file_id: str,
        file_path: Path,
        original_filename: str,
    ) -> ArchiveMetadata:
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
        return self._archive_store.get(archive_id)

    def get_archive_by_file_id(self, file_id: str) -> Optional[ArchiveMetadata]:
        archive_id = self._file_to_archive.get(file_id)
        if archive_id:
            return self._archive_store.get(archive_id)
        return None

    async def restore_file(self, archive_id: str, restore_path: Optional[Path] = None) -> Optional[Path]:
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
        return [metadata.to_dict() for metadata in self._archive_store.values()]

    async def delete_archive(self, archive_id: str) -> bool:
        metadata = self._archive_store.pop(archive_id, None)
        if metadata:
            archive_path = Path(metadata.archive_path)
            if archive_path.exists():
                archive_path.unlink()
            self._file_to_archive.pop(metadata.original_file_id, None)
            return True
        return False

    async def cleanup_expired(self) -> int:
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
