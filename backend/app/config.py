"""
Application Configuration Module

This module defines all configuration constants for the Data Sampler application.
Settings can be overridden via environment variables where noted.

Directory Configuration:
    UPLOAD_DIR: Temporary storage for uploaded files (env: UPLOAD_DIR)
    ARCHIVE_DIR: Storage for compressed archives (env: ARCHIVE_DIR)
    EXPORT_DIR: Storage for exported files (env: EXPORT_DIR)

File Processing:
    MAX_FILE_SIZE: Maximum allowed upload size (200MB)
    CHUNK_SIZE: Size of chunks for file upload streaming (64KB)
    EXCEL_CHUNK_SIZE: Number of rows per chunk when parsing Excel (10,000)

Retention Policies:
    TEMP_FILE_RETENTION_HOURS: How long to keep temp files (24 hours)
    ARCHIVE_RETENTION_DAYS: How long to keep archives (30 days)
    RESULT_RETENTION_DAYS: How long to keep sampling results (7 days)

File Validation:
    ALLOWED_EXTENSIONS: Set of allowed file extensions (.xlsx, .xls)
"""

import os
from pathlib import Path

# Base directory of the application
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory paths (can be overridden via environment variables)
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/data-sampler/uploads"))
ARCHIVE_DIR = Path(os.getenv("ARCHIVE_DIR", "/tmp/data-sampler/archives"))
EXPORT_DIR = Path(os.getenv("EXPORT_DIR", "/tmp/data-sampler/exports"))

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# File size limits
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB maximum file size
CHUNK_SIZE = 64 * 1024  # 64KB chunks for file upload streaming
EXCEL_CHUNK_SIZE = 10000  # 10,000 rows per chunk when parsing Excel

# Retention policies
TEMP_FILE_RETENTION_HOURS = 24  # Keep temp files for 24 hours
ARCHIVE_RETENTION_DAYS = 30  # Keep archives for 30 days
RESULT_RETENTION_DAYS = 7  # Keep sampling results for 7 days

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {".xlsx", ".xls"}
