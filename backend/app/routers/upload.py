import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path

from app.services.file_service import file_service
from app.services.parser_service import ExcelParser
from app.models.schemas import FileUploadResponse, ErrorResponse

router = APIRouter(prefix="/api/upload", tags=["upload"])


@router.post(
    "",
    response_model=FileUploadResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def upload_file(file: UploadFile = File(...)):
    try:
        metadata = await file_service.save_upload(file)

        parser = ExcelParser(Path(metadata.file_path))
        file_info = parser.get_file_info()

        file_service.update_metadata(
            metadata.file_id,
            row_count=file_info["row_count"],
            column_count=file_info["column_count"],
            columns=file_info["columns"],
            sheet_names=file_info["sheet_names"],
        )

        return FileUploadResponse(
            file_id=metadata.file_id,
            filename=metadata.original_filename,
            size=metadata.file_size,
            row_count=file_info["row_count"],
            column_count=file_info["column_count"],
            columns=file_info["columns"],
            sheet_names=file_info["sheet_names"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@router.get("/{file_id}")
async def get_file_info(file_id: str):
    metadata = file_service.get_metadata(file_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="File not found")

    return {
        "file_id": metadata.file_id,
        "filename": metadata.original_filename,
        "size": metadata.file_size,
        "row_count": metadata.row_count,
        "column_count": metadata.column_count,
        "columns": metadata.columns,
        "sheet_names": metadata.sheet_names,
        "upload_time": metadata.upload_time.isoformat(),
    }


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    success = await file_service.delete_file(file_id)
    if not success:
        raise HTTPException(status_code=404, detail="File not found")
    return {"status": "deleted", "file_id": file_id}
