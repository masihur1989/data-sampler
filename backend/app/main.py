"""
Data Sampler API - Main Application Module

This is the entry point for the Data Sampler FastAPI application.
It configures the application, sets up CORS middleware, and registers all routers.

The API provides endpoints for:
    - File Upload: Upload Excel files for processing
    - Pre-processing: Clean, transform, and validate data
    - Sampling: Apply various sampling methods to data
    - Export: Download sampled data in multiple formats
    - Archive: Compress and store files for long-term retention

API Documentation:
    - Swagger UI: /docs
    - ReDoc: /redoc
    - OpenAPI JSON: /openapi.json

Usage:
    Run with: poetry run fastapi dev app/main.py
    Or: uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import upload, sample, export, archive, preprocess

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
app.include_router(preprocess.router)
app.include_router(sample.router)
app.include_router(export.router)
app.include_router(archive.router)


@app.get("/healthz")
async def healthz():
    """Health check endpoint for monitoring and load balancers."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint returning API information and available endpoints."""
    return {
        "name": "Data Sampler API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "upload": "/api/upload",
            "preprocess": "/api/preprocess",
            "sample": "/api/sample",
            "export": "/api/export",
            "archive": "/api/archive",
        },
    }
