from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum
from datetime import datetime


class SamplingMethod(str, Enum):
    RANDOM = "random"
    STRATIFIED = "stratified"
    SYSTEMATIC = "systematic"
    CLUSTER = "cluster"
    WEIGHTED = "weighted"


class FileMetadata(BaseModel):
    file_id: str
    original_filename: str
    file_size: int
    file_path: str
    upload_time: datetime
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns: Optional[list[str]] = None
    sheet_names: Optional[list[str]] = None


class SamplingConfig(BaseModel):
    file_id: str
    method: SamplingMethod = SamplingMethod.RANDOM
    sample_size: int = Field(..., gt=0, description="Number of rows to sample")
    sample_percentage: Optional[float] = Field(None, ge=0.0, le=100.0, description="Percentage of rows to sample")
    random_seed: Optional[int] = Field(None, description="Seed for reproducibility")
    strata_column: Optional[str] = Field(None, description="Column for stratified sampling")
    interval: Optional[int] = Field(None, gt=0, description="Interval for systematic sampling")
    weight_column: Optional[str] = Field(None, description="Column for weighted sampling")
    cluster_column: Optional[str] = Field(None, description="Column for cluster sampling")
    with_replacement: bool = Field(False, description="Sample with replacement")
    sheet_name: Optional[str] = Field(None, description="Sheet to sample from")


class SamplingResult(BaseModel):
    sample_id: str
    file_id: str
    method: SamplingMethod
    original_row_count: int
    sample_row_count: int
    columns: list[str]
    created_at: datetime
    statistics: Optional[dict] = None


class ExportFormat(str, Enum):
    XLSX = "xlsx"
    CSV = "csv"
    JSON = "json"


class ExportConfig(BaseModel):
    sample_id: str
    format: ExportFormat = ExportFormat.XLSX


class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    size: int
    row_count: int
    column_count: int
    columns: list[str]
    sheet_names: list[str]


class SampleResponse(BaseModel):
    sample_id: str
    method: str
    original_rows: int
    sampled_rows: int
    columns: list[str]
    preview: list[dict]
    statistics: dict


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class StreamChunk(BaseModel):
    chunk_index: int
    total_chunks: int
    rows: list[dict]
    is_last: bool
