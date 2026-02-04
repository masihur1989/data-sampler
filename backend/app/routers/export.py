from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path

from app.services.sampler_service import sampler_service
from app.services.export_service import export_service
from app.models.schemas import ExportFormat, ErrorResponse

router = APIRouter(prefix="/api/export", tags=["export"])


@router.get(
    "/{sample_id}",
    responses={404: {"model": ErrorResponse}},
)
async def export_sample(sample_id: str, format: str = "xlsx"):
    df = sampler_service.get_sample(sample_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    try:
        export_format = ExportFormat(format.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format. Supported: {[f.value for f in ExportFormat]}",
        )

    file_path = export_service.export_to_file(df, export_format, f"sample_{sample_id[:8]}")

    media_types = {
        ExportFormat.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ExportFormat.CSV: "text/csv",
        ExportFormat.JSON: "application/json",
    }

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type=media_types[export_format],
    )


@router.get("/{sample_id}/stream/csv")
async def stream_csv(sample_id: str, chunk_size: int = 1000):
    df = sampler_service.get_sample(sample_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    return StreamingResponse(
        export_service.stream_as_csv(df, chunk_size),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=sample_{sample_id[:8]}.csv",
        },
    )


@router.get("/{sample_id}/stream/json")
async def stream_json(sample_id: str, chunk_size: int = 1000):
    df = sampler_service.get_sample(sample_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    return StreamingResponse(
        export_service.stream_as_json(df, chunk_size),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f"attachment; filename=sample_{sample_id[:8]}.ndjson",
        },
    )
