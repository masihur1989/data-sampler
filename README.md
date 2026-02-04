# Data Sampler

A web application for reading Excel files and creating representative data samples using various statistical sampling methods.

## Overview

Data Sampler is a full-stack application that allows users to upload Excel files, configure sampling parameters, and generate statistically valid samples. The application supports multiple sampling methods, file archival, and both file-based and streaming output options.

## Features

- Upload Excel files (.xlsx, .xls) via drag-and-drop or file browser
- Multiple sampling methods (Random, Stratified, Systematic, Cluster, Weighted)
- Configurable sampling parameters (sample size, random seed, strata column, etc.)
- Data preview before sampling
- Export to Excel, CSV, or JSON formats
- Real-time streaming output via WebSocket/SSE
- File archival with configurable retention policies
- Sampling statistics and distribution metrics

## Architecture

The application follows a layered architecture with clear separation of concerns:

### Layers

1. **Presentation Layer** - React frontend with TypeScript and Tailwind CSS
2. **API Layer** - FastAPI backend with REST endpoints and WebSocket support
3. **Data Processing Layer** - pandas-based data parsing and validation
4. **Sampling Engine** - Core sampling algorithms powered by NumPy and scikit-learn
5. **Output Layer** - File export and streaming capabilities
6. **Archival & Storage Layer** - File lifecycle management and retention policies

### Architecture Diagrams

The detailed architecture diagrams are available in the `docs/architecture/` directory:

- `data-sampler-architecture.drawio` - Original architecture diagram
- `data-sampler-architecture-v2.drawio` - Enhanced architecture with detailed sampling engine, archival mechanism, and streaming support

To view the diagrams:
1. Go to [draw.io](https://app.diagrams.net/)
2. File -> Open from -> Device
3. Select the .drawio file

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | React + TypeScript + Tailwind CSS |
| Backend | FastAPI (Python) |
| Data Processing | pandas + openpyxl |
| Sampling | NumPy + scikit-learn |
| Streaming | WebSocket + Server-Sent Events (SSE) |
| Metadata DB | SQLite |
| Build Tool | Vite |
| Compression | gzip / zlib |

## Sampling Methods

### Random Sampling
Selects records randomly with equal probability. Best for homogeneous datasets where every record has equal importance.

### Stratified Sampling
Divides data into groups (strata) based on a column, then samples proportionally from each group. Ensures representation of all categories.

### Systematic Sampling
Selects every nth record after a random starting point. Useful for ordered data where you want evenly distributed samples.

### Cluster Sampling
Divides the population into clusters, randomly selects some clusters, and includes all members from selected clusters.

### Weighted Sampling
Assigns probability weights to each record based on a weight column. Records with higher weights have a greater chance of being selected.

## Sampling Process Flow

1. **Input Validation** - Verify DataFrame, check sample size, validate parameters
2. **Data Analysis** - Calculate statistics, identify strata/groups, determine weights
3. **Sample Selection** - Apply algorithm, generate indices, handle edge cases
4. **Data Extraction** - Extract selected rows, preserve data types, maintain order
5. **Result Validation** - Verify sample size, check distribution, generate stats
6. **Output Preparation** - Format for export, prepare metadata, ready for stream

## Configurable Parameters

| Parameter | Description |
|-----------|-------------|
| Sample Size | Number of records or percentage to sample |
| Random Seed | For reproducible sampling results |
| Strata Column | Column to use for stratified sampling |
| Interval (k) | Step size for systematic sampling |
| Weight Column | Column containing probability weights |
| Replacement | Sample with or without replacement |

## Archival Mechanism

### Storage Types

- **Temp Storage** - Processing cache with auto-cleanup (24 hours)
- **Archive Storage** - Long-term retention with compression
- **Metadata Store** - File info and history (SQLite/JSON)

### Archival Process

1. **Upload** - Receive file, generate UUID
2. **Process** - Store in temp, extract metadata
3. **Sample** - Run sampling, store results
4. **Archive** - Compress and move to archive, update metadata
5. **Cleanup** - Remove temp files, apply retention policies

### Retention Policies

| Type | Retention Period |
|------|------------------|
| Temp Files | 24 hours |
| Archives | 30/90/365 days (configurable) |
| Metadata | Permanent |
| Results | 7 days (default) |

### Archive Features

- **Compression** - gzip/zip for storage efficiency
- **Versioning** - History tracking for uploaded files
- **Search** - Find files by metadata
- **Restore** - Re-download archived files
- **Audit Log** - Access history tracking

## Output Options

### File Export
- Excel (.xlsx, .xls)
- CSV (.csv)
- JSON (.json)

### Streaming Output
- WebSocket - Real-time bidirectional communication
- Server-Sent Events (SSE) - One-way server-to-client streaming
- Chunk Streaming - Paginated results for large datasets
- REST API Response - Standard JSON responses

## Development Plan

### Phase 1: Backend Development
- [ ] Set up FastAPI project structure
- [ ] Implement file upload endpoint
- [ ] Create Excel parser module
- [ ] Implement sampling algorithms
- [ ] Add export functionality
- [ ] Set up archival system

### Phase 2: Frontend Development
- [ ] Create React project with Vite
- [ ] Build file upload component
- [ ] Implement sampling configuration UI
- [ ] Create results viewer
- [ ] Add export/download functionality

### Phase 3: Integration & Testing
- [ ] Connect frontend to backend API
- [ ] Implement WebSocket streaming
- [ ] Add error handling
- [ ] Write unit tests
- [ ] Perform integration testing

### Phase 4: Deployment
- [ ] Deploy backend to cloud
- [ ] Deploy frontend
- [ ] Configure production settings
- [ ] Set up monitoring

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd backend
poetry install
poetry run fastapi dev app/main.py
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload Excel file |
| GET | `/api/files/{file_id}` | Get file metadata |
| POST | `/api/sample` | Generate sample |
| GET | `/api/sample/{sample_id}` | Get sample results |
| GET | `/api/export/{sample_id}` | Download sampled data |
| WS | `/ws/stream/{sample_id}` | Stream sample results |
| GET | `/api/archive` | List archived files |
| GET | `/api/archive/{file_id}` | Restore archived file |

## License

MIT License

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting a pull request.
