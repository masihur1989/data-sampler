"""
Export Service Module

This module provides data export capabilities for sampled data.
It supports multiple output formats and both file-based and streaming exports.

Supported Formats:
    - XLSX: Excel spreadsheet (using openpyxl)
    - CSV: Comma-separated values
    - JSON: JSON array of records

Streaming Exports:
    - CSV streaming: Outputs CSV in chunks for large datasets
    - JSON streaming: Outputs newline-delimited JSON (NDJSON)

Streaming exports are memory-efficient and suitable for large samples
that would be impractical to generate as a single file.
"""

import uuid
from pathlib import Path
from typing import Iterator, Optional
import pandas as pd
import json

from app.config import EXPORT_DIR
from app.models.schemas import ExportFormat


class ExportService:
    """
    Service for exporting sampled data to various formats.
    
    Provides both file-based exports (for download) and streaming exports
    (for memory-efficient transfer of large datasets).
    
    Attributes:
        _export_store: Mapping of export_id to file path for tracking exports
    """
    
    def __init__(self):
        """Initialize the export service with an empty store."""
        self._export_store: dict[str, Path] = {}

    def export_to_file(
        self,
        df: pd.DataFrame,
        format: ExportFormat,
        filename_prefix: str = "sample",
    ) -> Path:
        """
        Export a DataFrame to a file in the specified format.
        
        Args:
            df: DataFrame to export
            format: Output format (XLSX, CSV, or JSON)
            filename_prefix: Prefix for the output filename
            
        Returns:
            Path to the exported file
            
        Raises:
            ValueError: If format is not supported
        """
        export_id = str(uuid.uuid4())

        if format == ExportFormat.XLSX:
            file_path = EXPORT_DIR / f"{filename_prefix}_{export_id}.xlsx"
            df.to_excel(file_path, index=False, engine="openpyxl")
        elif format == ExportFormat.CSV:
            file_path = EXPORT_DIR / f"{filename_prefix}_{export_id}.csv"
            df.to_csv(file_path, index=False)
        elif format == ExportFormat.JSON:
            file_path = EXPORT_DIR / f"{filename_prefix}_{export_id}.json"
            df.to_json(file_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        self._export_store[export_id] = file_path
        return file_path

    def get_export_path(self, export_id: str) -> Optional[Path]:
        """Get the file path for an export by ID."""
        return self._export_store.get(export_id)

    def stream_as_json(self, df: pd.DataFrame, chunk_size: int = 1000) -> Iterator[str]:
        """
        Stream a DataFrame as newline-delimited JSON (NDJSON).
        
        Each yielded string contains metadata about the chunk and the row data.
        This format is ideal for streaming as it can be parsed incrementally.
        
        Args:
            df: DataFrame to stream
            chunk_size: Number of rows per chunk
            
        Yields:
            JSON strings, one per chunk, each ending with newline
        """
        total_rows = len(df)
        total_chunks = (total_rows + chunk_size - 1) // chunk_size

        for i in range(0, total_rows, chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            chunk_data = {
                "chunk_index": i // chunk_size,
                "total_chunks": total_chunks,
                "rows": chunk.to_dict(orient="records"),
                "is_last": (i + chunk_size) >= total_rows,
            }
            yield json.dumps(chunk_data) + "\n"

    def stream_as_csv(self, df: pd.DataFrame, chunk_size: int = 1000) -> Iterator[str]:
        """
        Stream a DataFrame as CSV.
        
        First yields the header row, then yields data rows in chunks.
        
        Args:
            df: DataFrame to stream
            chunk_size: Number of rows per chunk
            
        Yields:
            CSV strings, starting with header, then data chunks
        """
        yield df.columns.to_frame().T.to_csv(index=False, header=False)

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            yield chunk.to_csv(index=False, header=False)


export_service = ExportService()
