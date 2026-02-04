# Data Sampler Backend

A high-performance FastAPI backend for sampling data from large Excel files (up to 150MB).

## Features

- **Memory-Efficient Processing**: Handles large Excel files without loading entire file into memory
- **Pre-processing Pipeline**: Data cleaning, validation, and transformation before sampling
- **Multiple Sampling Methods**: Random, Stratified, Systematic, Cluster, and Weighted sampling
- **Streaming Support**: Server-Sent Events (SSE) for real-time data streaming
- **File Archival**: Automatic compression and retention policies
- **Export Options**: Excel (.xlsx), CSV, and JSON formats

## Architecture

```
app/
├── main.py              # FastAPI application entry point
├── config.py            # Configuration settings
├── models/
│   └── schemas.py       # Pydantic models for request/response
├── routers/
│   ├── upload.py        # File upload endpoints
│   ├── preprocess.py    # Pre-processing endpoints
│   ├── sample.py        # Sampling endpoints
│   ├── export.py        # Export/download endpoints
│   └── archive.py       # Archival endpoints
├── services/
│   ├── file_service.py         # File upload and storage
│   ├── parser_service.py       # Excel parsing with streaming
│   ├── preprocessing_service.py # Data cleaning and transformation
│   ├── sampler_service.py      # Sampling algorithms
│   ├── export_service.py       # Export functionality
│   └── archive_service.py      # File archival
└── utils/
```

## Performance Optimizations

### Chunked File Upload
Files are uploaded in 64KB chunks to prevent memory issues with large files:
```python
CHUNK_SIZE = 64 * 1024  # 64KB chunks
```

### Streaming Excel Parsing
Uses openpyxl's `read_only=True` mode for .xlsx files, which streams data instead of loading the entire file:
```python
wb = load_workbook(file_path, read_only=True, data_only=True)
for row in ws.iter_rows(values_only=True):
    # Process row by row
```

### Reservoir Sampling Algorithm
For random sampling, we use the Reservoir Sampling algorithm (Algorithm R by Vitter):
- **Time Complexity**: O(n) - single pass through data
- **Space Complexity**: O(k) - only stores k samples in memory
- **Guarantee**: Each item has exactly k/n probability of being selected

```python
# Reservoir Sampling pseudocode
for i, item in enumerate(stream):
    if i < k:
        reservoir[i] = item
    else:
        j = random.randint(0, i)
        if j < k:
            reservoir[j] = item
```

### Two-Pass Stratified Sampling
For stratified sampling, we use a memory-efficient two-pass approach:
1. **Pass 1**: Count items in each stratum to calculate proportions
2. **Pass 2**: Apply reservoir sampling to each stratum with proportional sample sizes

## Pre-processing Pipeline

The pre-processing layer cleans and transforms raw data before sampling. This ensures data quality and allows filtering out irrelevant records.

### Workflow

```
Upload -> Pre-process -> Sample -> Export
```

### Pre-processing Steps

1. **Analyze**: Generate data quality report (missing values, duplicates, statistics)
2. **Clean**: Remove duplicates, trim whitespace, remove empty rows/columns
3. **Handle Missing**: Keep, drop, or fill missing values (mean, median, mode, custom value)
4. **Transform**: Type conversion, column normalization (0-1 scale)
5. **Filter**: Apply conditions to filter rows, select/drop columns
6. **Validate**: Generate final quality report

### Pre-processing Configuration

| Option | Type | Description |
|--------|------|-------------|
| `remove_duplicates` | bool | Remove duplicate rows (default: true) |
| `handle_missing` | string | Strategy: keep, drop, fill_mean, fill_median, fill_mode, fill_value |
| `fill_value` | string | Value to use when handle_missing=fill_value |
| `trim_whitespace` | bool | Trim whitespace from strings (default: true) |
| `remove_empty_rows` | bool | Remove rows with all empty values (default: true) |
| `remove_empty_columns` | bool | Remove columns with all empty values (default: true) |
| `convert_types` | bool | Auto-convert string columns to numeric (default: false) |
| `normalize_columns` | list | Columns to normalize to 0-1 scale |
| `filter_conditions` | dict | Filter conditions: {column: {op, value}} |
| `columns_to_keep` | list | Only keep these columns |
| `columns_to_drop` | list | Drop these columns |

### Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `eq` | Equals | `{"category": {"op": "eq", "value": "A"}}` |
| `ne` | Not equals | `{"status": {"op": "ne", "value": "inactive"}}` |
| `gt` | Greater than | `{"value": {"op": "gt", "value": 100}}` |
| `gte` | Greater than or equal | `{"age": {"op": "gte", "value": 18}}` |
| `lt` | Less than | `{"price": {"op": "lt", "value": 50}}` |
| `lte` | Less than or equal | `{"score": {"op": "lte", "value": 100}}` |
| `in` | In list | `{"region": {"op": "in", "value": ["North", "South"]}}` |
| `not_in` | Not in list | `{"type": {"op": "not_in", "value": ["test", "demo"]}}` |
| `contains` | String contains | `{"name": {"op": "contains", "value": "Corp"}}` |

## Sampling Methods

### Random Sampling
Selects records randomly with equal probability using reservoir sampling.
- Best for: Homogeneous datasets
- Memory: O(k) where k is sample size

### Stratified Sampling
Divides data into groups (strata) and samples proportionally from each.
- Best for: Ensuring representation of all categories
- Memory: O(k) where k is sample size
- Requires: `strata_column` parameter

### Systematic Sampling
Selects every nth record after a random starting point.
- Best for: Ordered data, evenly distributed samples
- Memory: O(k) where k is sample size

### Cluster Sampling
Randomly selects clusters and includes all members from selected clusters.
- Best for: Naturally grouped data (regions, departments)
- Memory: O(n) - stores all rows grouped by cluster
- Requires: `cluster_column` parameter

### Weighted Sampling
Assigns selection probability based on weight column values.
- Best for: Importance-based sampling
- Memory: O(n) - stores all rows and weights
- Requires: `weight_column` parameter

## API Endpoints

### Upload

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload Excel file |
| GET | `/api/upload/{file_id}` | Get file metadata |
| DELETE | `/api/upload/{file_id}` | Delete uploaded file |

### Pre-processing

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/preprocess/analyze` | Analyze file and get quality report |
| POST | `/api/preprocess` | Pre-process file and store result |
| GET | `/api/preprocess/{processed_id}` | Get processed file info |
| GET | `/api/preprocess/{processed_id}/preview` | Preview processed data |

### Sampling

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sample` | Create a sample |
| GET | `/api/sample/{sample_id}` | Get sample preview |
| GET | `/api/sample/{sample_id}/stream` | Stream sample via SSE |
| GET | `/api/sample/{sample_id}/all` | Get all sample data |

### Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/export/{sample_id}?format=xlsx` | Download as Excel |
| GET | `/api/export/{sample_id}?format=csv` | Download as CSV |
| GET | `/api/export/{sample_id}?format=json` | Download as JSON |
| GET | `/api/export/{sample_id}/stream/csv` | Stream as CSV |
| GET | `/api/export/{sample_id}/stream/json` | Stream as NDJSON |

### Archive

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/archive/{file_id}` | Archive a file |
| GET | `/api/archive` | List all archives |
| GET | `/api/archive/{archive_id}` | Get archive info |
| POST | `/api/archive/{archive_id}/restore` | Restore archived file |
| GET | `/api/archive/{archive_id}/download` | Download archived file |
| DELETE | `/api/archive/{archive_id}` | Delete archive |
| POST | `/api/archive/cleanup` | Clean up expired archives |

## Usage Examples

### Upload a File

```bash
curl -X POST -F "file=@data.xlsx" http://localhost:8000/api/upload
```

Response:
```json
{
  "file_id": "abc123",
  "filename": "data.xlsx",
  "size": 1048576,
  "row_count": 10000,
  "column_count": 5,
  "columns": ["id", "name", "category", "value", "weight"],
  "sheet_names": ["Sheet1"]
}
```

### Create a Random Sample

```bash
curl -X POST http://localhost:8000/api/sample \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "abc123",
    "method": "random",
    "sample_size": 100,
    "random_seed": 42
  }'
```

### Create a Stratified Sample

```bash
curl -X POST http://localhost:8000/api/sample \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "abc123",
    "method": "stratified",
    "sample_size": 100,
    "strata_column": "category"
  }'
```

### Export Sample as CSV

```bash
curl -X GET "http://localhost:8000/api/export/{sample_id}?format=csv" -o sample.csv
```

## Configuration

Environment variables can be set in `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| UPLOAD_DIR | /tmp/data-sampler/uploads | Directory for uploaded files |
| ARCHIVE_DIR | /tmp/data-sampler/archives | Directory for archived files |
| EXPORT_DIR | /tmp/data-sampler/exports | Directory for exported files |
| MAX_FILE_SIZE | 200MB | Maximum upload file size |
| CHUNK_SIZE | 64KB | Upload chunk size |
| EXCEL_CHUNK_SIZE | 10000 | Rows per chunk when parsing |
| TEMP_FILE_RETENTION_HOURS | 24 | Temp file retention |
| ARCHIVE_RETENTION_DAYS | 30 | Archive retention |

## Development

### Prerequisites

- Python 3.12+
- Poetry

### Setup

```bash
cd backend
poetry install
```

### Run Development Server

```bash
poetry run fastapi dev app/main.py
```

The API will be available at http://localhost:8000

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Testing

```bash
poetry run pytest
```

## Dependencies

- **FastAPI**: Modern, fast web framework
- **pandas**: Data manipulation and analysis
- **openpyxl**: Excel .xlsx file support (with read_only streaming)
- **xlrd**: Excel .xls file support
- **numpy**: Numerical operations for weighted sampling
- **scikit-learn**: Additional sampling utilities
- **aiofiles**: Async file operations
- **python-multipart**: File upload support

## License

MIT License
