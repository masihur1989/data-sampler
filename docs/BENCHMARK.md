# Sampling Performance Benchmark Results

This document contains benchmark results for the Data Sampler's sampling algorithms, with a focus on the optimized stratified sampling implementation.

## Test Environment

- Python 3.12
- Polars 1.38.1 (Rust-based Excel reading)
- python-calamine 0.6.2 (fallback Excel engine)
- pandas 2.x with openpyxl (legacy fallback)

## Standard Benchmark (100,000 rows, 2.76 MB)

| Method | Time (s) | Rows/sec | Sampled | Peak Mem (MB) |
|--------|----------|----------|---------|---------------|
| Random | 9.73 | 10,276 | 1,000 | 100.9 |
| **Stratified (1 col)** | **1.04** | **95,876** | 1,000 | 195.7 |
| **Stratified (2 cols)** | **1.07** | **93,082** | 1,000 | 197.3 |
| **Stratified (3 cols)** | **1.16** | **86,232** | 996 | 168.2 |
| **Stratified (filtered)** | **1.06** | **93,915** | 1,000 | 196.2 |
| Systematic | 9.32 | 10,725 | 1,000 | 167.8 |
| Cluster | 10.95 | 9,136 | 1,000 | 187.9 |
| Weighted | 10.83 | 9,234 | 1,000 | 189.3 |

## Large File Benchmark (1,000,000 rows, 33.13 MB)

| Metric | Value |
|--------|-------|
| File Size | 33.13 MB |
| Total Rows | 1,000,000 |
| Sampled Rows | 10,000 |
| Time | 12.50 seconds |
| Throughput | 79,980 rows/sec |
| Memory Used | 572.5 MB |
| Peak Memory | 2,593.2 MB |

## Performance Improvements

The stratified sampling algorithm was optimized from the original two-pass streaming approach to a vectorized single-pass approach using Polars.

| Method | Before Optimization | After Optimization | Improvement |
|--------|---------------------|-------------------|-------------|
| Stratified (1 col) | 4,634 rows/sec | 95,876 rows/sec | **20.7x** |
| Stratified (2 cols) | 4,401 rows/sec | 93,082 rows/sec | **21.2x** |
| Stratified (3 cols) | 4,206 rows/sec | 86,232 rows/sec | **20.5x** |
| Stratified (filtered) | 4,065 rows/sec | 93,915 rows/sec | **23.1x** |

## Optimization Techniques

The following optimizations were implemented:

1. **Polars for Excel Reading**: Uses Polars' Rust-based Excel reader which is significantly faster than openpyxl
2. **Native Polars Operations**: Stratum key creation and sampling use native Polars operations instead of pandas iterrows()
3. **Vectorized String Operations**: Composite stratum keys are created using vectorized concat_str operations
4. **Single-Pass Algorithm**: Replaced two-pass counting/sampling with efficient groupby-based sampling

## 150MB File Projections

Based on the large file benchmark:

| Metric | Projected Value |
|--------|-----------------|
| Estimated Rows | ~4,500,000 |
| Projected Time | ~56 seconds |
| Bottleneck | Excel file format (compressed XML) |

## Recommendations for Large Files

For files larger than 50MB, consider the following alternatives for faster processing:

1. **CSV Format**: 5-10x faster to read than Excel
2. **Parquet Format**: Columnar storage, very fast for analytical workloads
3. **Pre-conversion**: Convert Excel to CSV/Parquet before sampling

Example conversion:
```python
import polars as pl

# Convert Excel to Parquet (one-time operation)
df = pl.read_excel("large_file.xlsx")
df.write_parquet("large_file.parquet")

# Subsequent reads are much faster
df = pl.read_parquet("large_file.parquet")
```

## Running Benchmarks

To run the benchmarks yourself:

```bash
cd backend

# Standard benchmark (100K rows)
poetry run python benchmark_sampling.py

# Large file benchmark (1M rows)
poetry run python benchmark_large.py
```

## Hardware Specifications

Benchmarks were run on:
- CPU: 8 cores
- Memory: Available for processing
- Storage: SSD

Results may vary based on hardware configuration.
