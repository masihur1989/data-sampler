from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import upload, sample, export, archive

app = FastAPI(
    title="Data Sampler API",
    description="API for sampling data from Excel files",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(sample.router)
app.include_router(export.router)
app.include_router(archive.router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {
        "name": "Data Sampler API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "upload": "/api/upload",
            "sample": "/api/sample",
            "export": "/api/export",
            "archive": "/api/archive",
        },
    }
