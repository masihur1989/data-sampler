import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/data-sampler/uploads"))
ARCHIVE_DIR = Path(os.getenv("ARCHIVE_DIR", "/tmp/data-sampler/archives"))
EXPORT_DIR = Path(os.getenv("EXPORT_DIR", "/tmp/data-sampler/exports"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 200 * 1024 * 1024
CHUNK_SIZE = 64 * 1024
EXCEL_CHUNK_SIZE = 10000

TEMP_FILE_RETENTION_HOURS = 24
ARCHIVE_RETENTION_DAYS = 30
RESULT_RETENTION_DAYS = 7

ALLOWED_EXTENSIONS = {".xlsx", ".xls"}
