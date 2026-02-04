from pathlib import Path
from typing import Iterator, Optional
import pandas as pd
from openpyxl import load_workbook

from app.config import EXCEL_CHUNK_SIZE


class ExcelParser:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.extension = file_path.suffix.lower()
        self._workbook = None
        self._sheet_names: list[str] = []
        self._columns: list[str] = []
        self._row_count: int = 0

    def get_sheet_names(self) -> list[str]:
        if self._sheet_names:
            return self._sheet_names

        if self.extension == ".xlsx":
            wb = load_workbook(self.file_path, read_only=True, data_only=True)
            self._sheet_names = wb.sheetnames
            wb.close()
        else:
            df = pd.read_excel(self.file_path, sheet_name=None, nrows=0, engine="xlrd")
            self._sheet_names = list(df.keys())

        return self._sheet_names

    def get_columns(self, sheet_name: Optional[str] = None) -> list[str]:
        if self.extension == ".xlsx":
            wb = load_workbook(self.file_path, read_only=True, data_only=True)
            ws = wb[sheet_name] if sheet_name else wb.active
            first_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None)
            wb.close()
            if first_row:
                self._columns = [str(c) if c is not None else f"Column_{i}" for i, c in enumerate(first_row)]
        else:
            df = pd.read_excel(self.file_path, sheet_name=sheet_name or 0, nrows=0, engine="xlrd")
            self._columns = list(df.columns)

        return self._columns

    def get_row_count(self, sheet_name: Optional[str] = None) -> int:
        if self.extension == ".xlsx":
            wb = load_workbook(self.file_path, read_only=True, data_only=True)
            ws = wb[sheet_name] if sheet_name else wb.active
            self._row_count = ws.max_row - 1 if ws.max_row else 0
            wb.close()
        else:
            df = pd.read_excel(self.file_path, sheet_name=sheet_name or 0, engine="xlrd")
            self._row_count = len(df)

        return self._row_count

    def iter_rows(self, sheet_name: Optional[str] = None, chunk_size: int = EXCEL_CHUNK_SIZE) -> Iterator[pd.DataFrame]:
        if self.extension == ".xlsx":
            yield from self._iter_xlsx_rows(sheet_name, chunk_size)
        else:
            yield from self._iter_xls_rows(sheet_name, chunk_size)

    def _iter_xlsx_rows(self, sheet_name: Optional[str], chunk_size: int) -> Iterator[pd.DataFrame]:
        wb = load_workbook(self.file_path, read_only=True, data_only=True)
        ws = wb[sheet_name] if sheet_name else wb.active

        rows = []
        columns = None
        row_num = 0

        for row in ws.iter_rows(values_only=True):
            if row_num == 0:
                columns = [str(c) if c is not None else f"Column_{i}" for i, c in enumerate(row)]
                row_num += 1
                continue

            rows.append(row)
            row_num += 1

            if len(rows) >= chunk_size:
                yield pd.DataFrame(rows, columns=columns)
                rows = []

        if rows:
            yield pd.DataFrame(rows, columns=columns)

        wb.close()

    def _iter_xls_rows(self, sheet_name: Optional[str], chunk_size: int) -> Iterator[pd.DataFrame]:
        for chunk in pd.read_excel(
            self.file_path,
            sheet_name=sheet_name or 0,
            engine="xlrd",
            chunksize=chunk_size,
        ):
            yield chunk

    def read_all(self, sheet_name: Optional[str] = None) -> pd.DataFrame:
        if self.extension == ".xlsx":
            return pd.read_excel(self.file_path, sheet_name=sheet_name or 0, engine="openpyxl")
        else:
            return pd.read_excel(self.file_path, sheet_name=sheet_name or 0, engine="xlrd")

    def get_file_info(self, sheet_name: Optional[str] = None) -> dict:
        sheets = self.get_sheet_names()
        columns = self.get_columns(sheet_name)
        row_count = self.get_row_count(sheet_name)

        return {
            "sheet_names": sheets,
            "columns": columns,
            "row_count": row_count,
            "column_count": len(columns),
        }
