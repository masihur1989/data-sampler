import uuid
from pathlib import Path
from typing import Iterator, Optional
import pandas as pd
import json

from app.config import EXPORT_DIR
from app.models.schemas import ExportFormat


class ExportService:
    def __init__(self):
        self._export_store: dict[str, Path] = {}

    def export_to_file(
        self,
        df: pd.DataFrame,
        format: ExportFormat,
        filename_prefix: str = "sample",
    ) -> Path:
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
        return self._export_store.get(export_id)

    def stream_as_json(self, df: pd.DataFrame, chunk_size: int = 1000) -> Iterator[str]:
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
        yield df.columns.to_frame().T.to_csv(index=False, header=False)

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            yield chunk.to_csv(index=False, header=False)


export_service = ExportService()
