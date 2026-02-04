import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pathlib import Path
import json

from app.services.file_service import file_service
from app.services.sampler_service import sampler_service
from app.models.schemas import SamplingConfig, SampleResponse, ErrorResponse

router = APIRouter(prefix="/api/sample", tags=["sample"])


@router.post(
    "",
    response_model=SampleResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def create_sample(config: SamplingConfig):
    file_path = file_service.get_file_path(config.file_id)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        df, statistics = sampler_service.sample(file_path, config)

        sample_id = str(uuid.uuid4())
        sampler_service.store_sample(sample_id, df, {
            "config": config.model_dump(),
            "statistics": statistics,
            "created_at": datetime.utcnow().isoformat(),
        })

        preview_rows = min(100, len(df))
        preview = df.head(preview_rows).to_dict(orient="records")

        return SampleResponse(
            sample_id=sample_id,
            method=config.method.value,
            original_rows=statistics["original_rows"],
            sampled_rows=statistics["sampled_rows"],
            columns=statistics["columns"],
            preview=preview,
            statistics=statistics,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sampling failed: {str(e)}")


@router.get("/{sample_id}")
async def get_sample(sample_id: str):
    df = sampler_service.get_sample(sample_id)
    metadata = sampler_service.get_sample_metadata(sample_id)

    if df is None or metadata is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    preview_rows = min(100, len(df))
    preview = df.head(preview_rows).to_dict(orient="records")

    return {
        "sample_id": sample_id,
        "row_count": len(df),
        "columns": list(df.columns),
        "preview": preview,
        "metadata": metadata,
    }


@router.get("/{sample_id}/stream")
async def stream_sample(sample_id: str, chunk_size: int = 1000):
    df = sampler_service.get_sample(sample_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    def generate():
        total_rows = len(df)
        total_chunks = (total_rows + chunk_size - 1) // chunk_size

        for i, chunk_df in enumerate(sampler_service.iter_sample_chunks(sample_id, chunk_size)):
            chunk_data = {
                "chunk_index": i,
                "total_chunks": total_chunks,
                "rows": chunk_df.to_dict(orient="records"),
                "is_last": (i + 1) >= total_chunks,
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/{sample_id}/all")
async def get_all_sample_data(sample_id: str):
    df = sampler_service.get_sample(sample_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    return {
        "sample_id": sample_id,
        "row_count": len(df),
        "columns": list(df.columns),
        "data": df.to_dict(orient="records"),
    }
